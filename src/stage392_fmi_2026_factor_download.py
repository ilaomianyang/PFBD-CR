from __future__ import annotations

"""Download and audit the frozen FMI 2026 temporal factor window."""

import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "raw" / os.environ.get(
    "TMQC_FMI_RAW_SUBDIR", "fmi_2026_factor"
)
PROTOCOL_PREFIX = os.environ.get(
    "TMQC_FMI_PROTOCOL_PREFIX", "stage390_fmi_2026_factor_protocol"
)
CONTRACT_PREFIX = os.environ.get(
    "TMQC_FMI_CONTRACT_PREFIX", "stage391_fmi_2026_factor_contract"
)
LEGACY_NORMALIZED = TABLES / "stage258_fmi_official_download_normalized.parquet"
OUT_PREFIX = os.environ.get(
    "TMQC_FMI_DOWNLOAD_PREFIX", "stage392_fmi_2026_factor_download"
)
STATUS_PREFIX = os.environ.get("TMQC_FMI_STATUS_PREFIX", "FMI_2026")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import stage258_fmi_official_download as fmi  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_2026(payload: bytes, station_id: str, station_name: str) -> pd.DataFrame:
    wide = fmi.parse_observations(payload, station_id, station_name)
    if wide.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "timestamp_utc",
                "year",
                "temperature_c",
                "humidity_pct",
                "wind_speed_ms",
                "pressure_hpa",
                "observation_present",
                "split",
            ]
        )
    wide = wide.rename(
        columns={
            fmi.PARAMETER_BY_VARIABLE[target]: target
            for target in fmi.PARAMETER_BY_VARIABLE
        }
    )
    for target in fmi.PARAMETER_BY_VARIABLE:
        if target not in wide.columns:
            wide[target] = np.nan
        wide[target] = pd.to_numeric(wide[target], errors="coerce")
    wide["year"] = 2026
    wide["observation_present"] = wide["temperature_c"].notna()
    wide["split"] = "test"
    return wide[
        [
            "station_id",
            "station_name",
            "timestamp_utc",
            "year",
            "temperature_c",
            "humidity_pct",
            "wind_speed_ms",
            "pressure_hpa",
            "observation_present",
            "split",
        ]
    ]


def month_windows() -> list[tuple[int, str, str]]:
    custom_start = os.environ.get("TMQC_FMI_WINDOW_START")
    custom_end = os.environ.get("TMQC_FMI_WINDOW_END")
    if custom_start and custom_end:
        month = int(custom_start[5:7])
        return [(month, custom_start, custom_end)]
    return [
        (
            month,
            f"2026-{month:02d}-01T00:00:00Z",
            f"2026-{month + 1:02d}-01T00:00:00Z",
        )
        for month in range(1, 9)
    ]


def fetch_one(
    station_id: str,
    station_name: str,
    month: int,
    start: str,
    end: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    url = fmi.observation_url(station_id, start, end)
    raw_path = RAW / f"{station_id}_2026_{month:02d}.xml"
    try:
        if raw_path.is_file():
            payload = raw_path.read_bytes()
            status = "cached"
        else:
            payload, _ = fmi.download(url)
            raw_path.write_bytes(payload)
            status = "downloaded"
        parsed = normalize_2026(payload, station_id, station_name)
        detail = {
            "station_id": station_id,
            "month": month,
            "url": url,
            "status": status,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": len(parsed),
            "nonmissing_rows": int(parsed["observation_present"].sum()),
            "error": "",
        }
        return detail, parsed
    except Exception as error:
        return (
            {
                "station_id": station_id,
                "month": month,
                "url": url,
                "status": "failed",
                "bytes": 0,
                "sha256": "",
                "rows": 0,
                "nonmissing_rows": 0,
                "error": str(error),
            },
            pd.DataFrame(),
        )


def expected_hours(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start).total_seconds() // 3600) + 1


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    protocol = json.loads(
        (TABLES / f"{PROTOCOL_PREFIX}.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (TABLES / f"{CONTRACT_PREFIX}.json").read_text(encoding="utf-8")
    )
    if protocol["status"] not in {
        "PROTOCOL_FROZEN_BEFORE_2026_OBSERVATION_ACCESS",
        "PROTOCOL_FROZEN_BEFORE_SEPTEMBER_2026_OBSERVATION_ACCESS",
        "PROTOCOL_FROZEN_BEFORE_SEPTEMBER_21_2026_OBSERVATION_ACCESS",
    }:
        raise RuntimeError("Stage390 protocol is not frozen")
    if contract["status"] not in {
        "EVALUATION_CONTRACT_FROZEN_BEFORE_PERFORMANCE",
        "EVALUATION_CONTRACT_FROZEN_BEFORE_SEP21_2026_PERFORMANCE",
    }:
        raise RuntimeError("Stage391 contract is not frozen")
    if protocol["observation_access"] or protocol["performance_access"]:
        raise RuntimeError("FMI 2026 observation was accessed before freeze")
    RAW.mkdir(parents=True, exist_ok=True)
    source = json.loads(
        (TABLES / "stage257_fmi_protocol_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    names = {
        str(station_id): str(name)
        for station_id, name in zip(source["station_ids"], source["station_names"])
    }
    station_ids = [str(value) for value in protocol["station_ids"]]
    legacy = pd.read_parquet(LEGACY_NORMALIZED)
    legacy["station_id"] = legacy["station_id"].astype(str)
    legacy = legacy[
        legacy["station_id"].isin(station_ids)
        & legacy["split"].isin(["train", "valid"])
    ].copy()
    details: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = [legacy]
    futures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for station_id in station_ids:
            for month, start, end in month_windows():
                futures.append(
                    pool.submit(
                        fetch_one,
                        station_id,
                        names.get(station_id, station_id),
                        month,
                        start,
                        end,
                    )
                )
        for future in as_completed(futures):
            detail, parsed = future.result()
            details.append(detail)
            if not parsed.empty:
                frames.append(parsed)
            print(
                json.dumps(
                    {
                        "station_id": detail["station_id"],
                        "month": detail["month"],
                        "status": detail["status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    long = pd.concat(frames, ignore_index=True)
    long["station_id"] = long["station_id"].astype(str)
    long = (
        long.sort_values(["station_id", "timestamp_utc"], kind="stable")
        .drop_duplicates(["station_id", "timestamp_utc"], keep="last")
        .reset_index(drop=True)
    )
    # Restrict the partial test window on both sides.  The previous version
    # only applied the end bound, which admitted observations before the
    # frozen test start into the partial-month holdout.
    start = pd.Timestamp(protocol["test_period_start_utc"])
    end = pd.Timestamp(protocol["test_period_end_utc"])
    test_mask = long["year"].eq(int(protocol["split"]["test"][0]))
    long = long[
        (~test_mask)
        | long["timestamp_utc"].between(start, end, inclusive="both")
    ].reset_index(drop=True)
    normalized_path = TABLES / f"{OUT_PREFIX}_normalized.parquet"
    detail_path = TABLES / f"{OUT_PREFIX}_detail.csv"
    long.to_parquet(normalized_path, index=False)
    detail_frame = pd.DataFrame(details).sort_values(["station_id", "month"])
    detail_frame.to_csv(detail_path, index=False, encoding="utf-8-sig")

    station_rows: list[dict[str, object]] = []
    retained: list[str] = []
    split_bounds = {
        "train": (
            pd.Timestamp("2021-01-01T00:00:00Z"),
            pd.Timestamp("2023-12-31T23:59:59Z"),
        ),
        "valid": (
            pd.Timestamp("2024-01-01T00:00:00Z"),
            pd.Timestamp("2024-12-31T23:59:59Z"),
        ),
        "test": (
            pd.Timestamp(protocol["test_period_start_utc"]),
            end,
        ),
    }
    for station_id in station_ids:
        part = long[long["station_id"].eq(station_id)]
        row: dict[str, object] = {"station_id": station_id}
        keep = True
        for split, (start, stop) in split_bounds.items():
            expected = expected_hours(start, stop)
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
        station_rows.append(row)
        if keep:
            retained.append(station_id)
    station_audit = pd.DataFrame(station_rows)
    station_audit_path = TABLES / f"{OUT_PREFIX}_station_audit.csv"
    station_audit.to_csv(station_audit_path, index=False, encoding="utf-8-sig")
    failed = int((detail_frame["status"] == "failed").sum())
    payload = {
        "status": (
            f"PASS_{STATUS_PREFIX}_DOWNLOAD_AND_COVERAGE_AUDIT"
            if failed == 0 and len(retained) == len(station_ids)
            else f"FAIL_{STATUS_PREFIX}_DOWNLOAD_AND_COVERAGE_AUDIT"
        ),
        "protocol_id": protocol["protocol_id"],
        "contract_id": contract["contract_id"],
        "expected_requests": len(station_ids) * len(month_windows()),
        "downloaded_count": int((detail_frame["status"] == "downloaded").sum()),
        "cached_count": int((detail_frame["status"] == "cached").sum()),
        "failed_count": failed,
        "normalized_rows": int(len(long)),
        "station_count": len(station_ids),
        "retained_station_count": len(retained),
        "retained_station_ids": retained,
        "station_audit": station_audit.to_dict("records"),
        "normalized_sha256": sha256_file(normalized_path),
        "detail_sha256": sha256_file(detail_path),
        "station_audit_sha256": sha256_file(station_audit_path),
        "download_status": "DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "coverage_rule": "At least 0.70 of expected UTC hourly slots in every split.",
        "claim_boundary": [
            "This is a new FMI temporal window, not a new provider.",
            "No performance was evaluated in this stage.",
            "All later attacks are controlled simulations, not production incidents.",
        ],
    }
    output = TABLES / f"{OUT_PREFIX}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPORTS / f"{OUT_PREFIX}.md").write_text(
        "\n".join(
            [
                f"# {OUT_PREFIX} 官方下载与覆盖率审计",
                "",
                f"- 状态：`{payload['status']}`",
                f"- 请求：`{payload['expected_requests']}`；失败：`{payload['failed_count']}`；保留站点：`{payload['retained_station_count']}`。",
                f"- 规范化行数：`{payload['normalized_rows']}`；性能状态：`{payload['performance_status']}`。",
                "- 本阶段只下载和审计官方数据，不训练、不调阈值、不评价攻击性能。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "failed": failed,
                "retained_station_count": len(retained),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
