from __future__ import annotations

"""Download and audit the frozen MET Norway 2026 temporal factor package."""

import hashlib
import json
import os
import shutil
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "raw" / os.environ.get(
    "TMQC_MET_RAW_SUBDIR", "met_norway_frost_2026_factor"
)
SECRET = ROOT / ".secrets" / "met_norway_frost.json"
PROTOCOL_PREFIX = os.environ.get(
    "TMQC_MET_PROTOCOL_PREFIX", "stage384_met_norway_2026_factor_protocol"
)
CONTRACT_PREFIX = os.environ.get(
    "TMQC_MET_CONTRACT_PREFIX", "stage385_met_norway_2026_factor_contract"
)
PREFIX = os.environ.get(
    "TMQC_MET_DOWNLOAD_PREFIX", "stage386_met_norway_2026_factor_download"
)
STATUS_PREFIX = os.environ.get("TMQC_MET_STATUS_PREFIX", "MET_NORWAY_2026")
STATIONS_SOURCE = TABLES / "stage304_met_norway_protocol_preregistration_stations.csv"
LEGACY_RAW = ROOT / "raw" / "met_norway_frost"
LEGACY_DETAIL = TABLES / "stage306_met_norway_official_download_detail.csv"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage306_met_norway_official_download as frost  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_bounds(protocol: dict, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split == "test":
        return (
            pd.Timestamp(protocol["test_period_start_utc"]),
            pd.Timestamp(protocol["test_period_end_utc"]),
        )
    years = protocol["split"][split]
    return (
        pd.Timestamp(f"{int(years[0])}-01-01T00:00:00Z"),
        pd.Timestamp(f"{int(years[-1])}-12-31T23:59:59Z"),
    )


def expected_hours(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start).total_seconds() // 3600) + 1


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    protocol = json.loads((TABLES / f"{PROTOCOL_PREFIX}.json").read_text(encoding="utf-8"))
    contract = json.loads((TABLES / f"{CONTRACT_PREFIX}.json").read_text(encoding="utf-8"))
    if protocol["status"] not in {
        "PROTOCOL_FROZEN_BEFORE_2026_OBSERVATION_ACCESS",
        "PROTOCOL_FROZEN_BEFORE_SEPTEMBER_2026_OBSERVATION_ACCESS",
        "PROTOCOL_FROZEN_BEFORE_SEPTEMBER_21_2026_OBSERVATION_ACCESS",
    }:
        raise RuntimeError("Stage384 protocol is not frozen")
    if contract["status"] not in {
        "EVALUATION_CONTRACT_FROZEN_BEFORE_PERFORMANCE",
        "EVALUATION_CONTRACT_FROZEN_BEFORE_SEP21_2026_PERFORMANCE",
    }:
        raise RuntimeError("Stage385 contract is not frozen")
    if protocol["observation_access"] or protocol["performance_access"]:
        raise RuntimeError("2026 observation was accessed before freeze")
    if protocol["download_status"] != "NOT_DOWNLOADED":
        raise RuntimeError("Stage384 download state changed")
    stations = pd.read_csv(STATIONS_SOURCE, dtype={"station_id": str})
    frozen_ids = [str(value) for value in protocol["station_ids"]]
    if stations["station_id"].astype(str).tolist() != frozen_ids:
        raise RuntimeError("frozen station table changed")
    station_contract = {str(row["station_id"]): row for row in stations.to_dict("records")}
    legacy_detail = pd.read_csv(LEGACY_DETAIL, dtype={"station_id": str, "year": int})
    legacy_detail = legacy_detail.set_index(["station_id", "year"])
    credentials = frost.load_credentials()
    headers = frost.authorization_header(credentials)
    years = [int(value) for value in protocol["years"]]
    test_year = int(protocol["split"]["test"][0])
    RAW.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.now(timezone.utc).isoformat()
    detail_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    for station_id in frozen_ids:
        source_id = str(station_contract[station_id]["source_id"])
        exposure = str(station_contract[station_id]["exposure_category"])
        for year in years:
            start = f"{year}-01-01T00:00:00Z"
            end = protocol["test_period_end_utc"] if year == test_year else f"{year}-12-31T23:59:59Z"
            query = frost.build_query(source_id, exposure, start, end)
            url = frost.OBSERVATIONS_ENDPOINT + "?" + urllib.parse.urlencode(query)
            raw_path = RAW / f"{station_id}_{year}.json"
            try:
                if raw_path.is_file():
                    raw_bytes = raw_path.read_bytes()
                    payload = json.loads(raw_bytes)
                    status = "cached"
                    response_headers = {"source": "cache"}
                elif year != test_year and (LEGACY_RAW / f"{station_id}_{year}.json").is_file():
                    legacy_path = LEGACY_RAW / f"{station_id}_{year}.json"
                    expected = legacy_detail.loc[(station_id, year)]
                    raw_bytes = legacy_path.read_bytes()
                    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
                    if actual_sha != str(expected["sha256"]):
                        raise RuntimeError(
                            f"prior official cache SHA256 mismatch for {station_id}/{year}"
                        )
                    shutil.copyfile(legacy_path, raw_path)
                    payload = json.loads(raw_bytes)
                    status = "cached_prior_official_valid"
                    response_headers = {
                        "source": "stage306_official_download",
                        "prior_sha256": actual_sha,
                    }
                else:
                    payload, raw_bytes, response_headers = frost.download_json(query, headers)
                    raw_path.write_bytes(raw_bytes)
                    status = "downloaded"
                parsed = frost.normalize_observations(
                    payload, station_id, frost.ACCEPTED_QUALITY_CODE
                )
                frames.append(parsed)
                detail_rows.append(
                    {
                        "station_id": station_id,
                        "source_id": source_id,
                        "exposure_category": exposure,
                        "year": year,
                        "url": url,
                        "status": status,
                        "downloaded_at_utc": downloaded_at,
                        "bytes": len(raw_bytes),
                        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        "rows": len(parsed),
                        "nonmissing_rows": int(parsed["observation_present"].sum()),
                        "content_type": response_headers.get("content-type", ""),
                        "cache_source": response_headers.get("source", ""),
                        "error": "",
                    }
                )
            except Exception as error:
                raw_bytes = b""
                detail_rows.append(
                    {
                        "station_id": station_id,
                        "source_id": source_id,
                        "exposure_category": exposure,
                        "year": year,
                        "url": url,
                        "status": "failed",
                        "downloaded_at_utc": downloaded_at,
                        "bytes": 0,
                        "sha256": "",
                        "rows": 0,
                        "nonmissing_rows": 0,
                        "content_type": "",
                        "cache_source": "",
                        "error": str(error),
                    }
                )
            print(
                json.dumps(
                    {
                        "station_id": station_id,
                        "year": year,
                        "status": detail_rows[-1]["status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    long["station_id"] = long["station_id"].astype(str)
    long = long.sort_values(["station_id", "timestamp_utc"], kind="stable").drop_duplicates(
        ["station_id", "timestamp_utc"], keep="last"
    )
    # Restrict the partial test window on both sides.  The previous version
    # only applied the end bound, which admitted January--August 2026 rows
    # into the September holdout and inflated test coverage.
    test_start = pd.Timestamp(protocol["test_period_start_utc"])
    test_end = pd.Timestamp(protocol["test_period_end_utc"])
    test_mask = long["year"].eq(test_year)
    long = long[
        (~test_mask)
        | long["timestamp_utc"].between(test_start, test_end, inclusive="both")
    ].copy()
    split_map: dict[int, str] = {}
    for split, split_years in protocol["split"].items():
        for year in split_years:
            split_map[int(year)] = str(split)
    long["split"] = long["year"].map(split_map).fillna("outside_protocol")
    long = long[long["split"].ne("outside_protocol")].sort_values(
        ["station_id", "timestamp_utc"], kind="stable"
    ).reset_index(drop=True)
    normalized_path = TABLES / f"{PREFIX}_normalized.parquet"
    detail_path = TABLES / f"{PREFIX}_detail.csv"
    long.to_parquet(normalized_path, index=False)
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    audit_rows: list[dict[str, object]] = []
    retained: list[str] = []
    for station_id in frozen_ids:
        part = long[long["station_id"].eq(station_id)]
        row: dict[str, object] = {"station_id": station_id}
        keep = True
        for split in ("train", "valid", "test"):
            start, end = split_bounds(protocol, split)
            expected = expected_hours(start, end)
            observed = int(
                part.loc[
                    part["split"].eq(split) & part["observation_present"],
                    "timestamp_utc",
                ].nunique()
            )
            coverage = observed / expected if expected else 0.0
            row[f"{split}_expected_hours"] = expected
            row[f"{split}_observed_hours"] = observed
            row[f"{split}_coverage"] = coverage
            if observed == 0 or coverage < 0.70:
                keep = False
        row["retained"] = keep
        audit_rows.append(row)
        if keep:
            retained.append(station_id)
    station_audit = pd.DataFrame(audit_rows)
    station_audit_path = TABLES / f"{PREFIX}_station_audit.csv"
    station_audit.to_csv(station_audit_path, index=False, encoding="utf-8-sig")
    failures = int((detail["status"] == "failed").sum())
    payload = {
        "status": (
            f"PASS_{STATUS_PREFIX}_DOWNLOAD_AND_COVERAGE_AUDIT"
            if failures == 0 and len(retained) == len(frozen_ids)
            else f"FAIL_{STATUS_PREFIX}_DOWNLOAD_AND_COVERAGE_AUDIT"
        ),
        "protocol_id": protocol["protocol_id"],
        "contract_id": contract["contract_id"],
        "expected_requests": len(frozen_ids) * len(years),
        "downloaded_count": int((detail["status"] == "downloaded").sum()),
        "cached_count": int((detail["status"] == "cached").sum()),
        "failed_count": failures,
        "normalized_rows": int(len(long)),
        "station_count": len(frozen_ids),
        "retained_station_count": len(retained),
        "retained_station_ids": retained,
        "station_audit": station_audit.to_dict("records"),
        "normalized_sha256": sha256_file(normalized_path),
        "detail_sha256": sha256_file(detail_path),
        "station_audit_sha256": sha256_file(station_audit_path),
        "download_status": "DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "coverage_rule": "At least 0.70 of expected UTC hourly slots in every split.",
        "quality_boundary": "Only official Frost quality code 0 is retained.",
        "claim_boundary": [
            "This is a new MET Norway 2026 temporal window, not a new provider.",
            "No performance was evaluated in this stage.",
            "All later attacks are controlled simulations, not production incidents.",
        ],
    }
    output = TABLES / f"{PREFIX}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(
            [
                "# Stage386 MET Norway 2026 官方下载与覆盖率审计",
                "",
                f"- 状态：`{payload['status']}`",
                f"- 请求：`{payload['expected_requests']}`；失败：`{payload['failed_count']}`。",
                f"- 规范化行数：`{payload['normalized_rows']}`；保留站点：`{payload['retained_station_count']}`。",
                "- 2026 test 仅用于后续锁定评价；本阶段不训练、不校阈值、不注入攻击。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "failed": failures, "retained_station_count": len(retained)}, ensure_ascii=False), flush=True)
    return 0 if payload["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
