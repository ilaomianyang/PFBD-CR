from __future__ import annotations

"""Stage 311: download the frozen disjoint MET Norway station network.

This stage consumes only the Stage310 protocol/data/analysis freezes. It
downloads official Frost observations for the frozen disjoint station IDs,
normalizes them, and audits coverage. It does not fit a detector, calibrate a
threshold, inject an attack, or compute performance.
"""

import hashlib
import json
import os
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
PACKAGE_ROOT = ROOT / "confirmatory_package"
RAW = ROOT / "raw" / "met_norway_frost_disjoint_v1"
PREFIX = "stage311_met_norway_disjoint_download"
PACKAGE_PREFIX = "stage310_disjoint_met_norway_confirmatory_package"
YEARS = [2021, 2022, 2023, 2024, 2025]
COVERAGE_MIN = 0.70
ACCEPTED_QUALITY_CODE = "0"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage304_met_norway_protocol_preregistration as s304  # noqa: E402
import stage306_met_norway_official_download as s306  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_proxy_environment() -> None:
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(name, None)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_year_periods(protocol: dict) -> dict[int, pd.DatetimeIndex]:
    periods: dict[int, pd.DatetimeIndex] = {}
    for raw_year in YEARS:
        year = int(raw_year)
        start = pd.Timestamp(f"{year}-01-01T00:00:00Z")
        end = (
            pd.Timestamp(protocol["test_period_end_utc"])
            if year == 2025
            else pd.Timestamp(f"{year}-12-31T23:00:00Z")
        )
        periods[year] = pd.date_range(
            start=start,
            end=end,
            freq="h",
            inclusive="both",
        )
    return periods


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    clear_proxy_environment()
    protocol = load_json(PACKAGE_ROOT / "protocol_freeze.json")
    data_freeze = load_json(PACKAGE_ROOT / "data_freeze.json")
    analysis_freeze = load_json(PACKAGE_ROOT / "analysis_freeze.json")
    access_gate = load_json(PACKAGE_ROOT / "access_gate.json")
    package_audit = load_json(TABLES / f"{PACKAGE_PREFIX}.json")

    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_OBSERVATION_DOWNLOAD":
        raise RuntimeError("Stage310 protocol is not frozen")
    if data_freeze["status"] != (
        "DATA_FREEZE_DECLARED_BEFORE_OBSERVATION_DOWNLOAD"
    ):
        raise RuntimeError("Stage310 data freeze is missing")
    if analysis_freeze["status"] != (
        "ANALYSIS_FROZEN_BEFORE_OBSERVATION_DOWNLOAD"
    ):
        raise RuntimeError("Stage310 analysis freeze is missing")
    if access_gate["status"] != "ACCESS_GATE_OPENED_AFTER_FREEZE":
        raise RuntimeError("Stage310 access gate was not opened")
    if access_gate["protocol_frozen"] is not True:
        raise RuntimeError("Stage310 protocol was not frozen")
    if access_gate["data_frozen"] is not True:
        raise RuntimeError("Stage310 data was not frozen")
    if access_gate["analysis_frozen"] is not True:
        raise RuntimeError("Stage310 analysis was not frozen")
    if access_gate["labels_inspected"] is not False:
        raise RuntimeError("Stage310 labels were already inspected")
    if access_gate["outcome_summaries_inspected"] is not False:
        raise RuntimeError("Stage310 outcomes were already inspected")
    if package_audit["provider_is_new"] is not False:
        raise RuntimeError("Stage310 provider boundary changed")
    if package_audit["network_is_disjoint_from_stage304"] is not True:
        raise RuntimeError("Stage310 network is not disjoint")
    if protocol["provider_is_new"] is not False:
        raise RuntimeError("Stage310 provider claim changed")
    if protocol["network_is_disjoint_from_stage304"] is not True:
        raise RuntimeError("Stage310 disjoint claim changed")
    if protocol["primary_far"] != 0.01:
        raise RuntimeError("Stage310 primary FAR changed")
    if protocol["test_period_end_utc"] != "2025-08-24T23:59:59Z":
        raise RuntimeError("Stage310 test boundary changed")

    station_table = pd.read_csv(
        PACKAGE_ROOT / "met_norway_disjoint_stations.csv",
        dtype={"station_id": str, "source_id": str},
    )
    if station_table["station_id"].astype(str).tolist() != [
        str(value) for value in protocol["station_ids"]
    ]:
        raise RuntimeError("Stage310 station table changed")
    if sha256_file(
        PACKAGE_ROOT / "met_norway_disjoint_stations.csv"
    ) != package_audit["package_files"]["met_norway_disjoint_stations.csv"]:
        raise RuntimeError("Stage310 station manifest hash mismatch")

    credentials = s304.load_credentials()
    headers = s304.authorization_header(credentials)
    RAW.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.now(timezone.utc).isoformat()
    detail_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for station in station_table.to_dict("records"):
        station_id = str(station["station_id"])
        source_id = str(station["source_id"])
        exposure_category = str(station["exposure_category"])
        for year in YEARS:
            start_utc = f"{int(year)}-01-01T00:00:00Z"
            end_utc = (
                protocol["test_period_end_utc"]
                if int(year) == 2025
                else f"{int(year)}-12-31T23:59:59Z"
            )
            query = s306.build_query(
                source_id,
                exposure_category,
                start_utc,
                end_utc,
            )
            url = (
                s306.OBSERVATIONS_ENDPOINT
                + "?"
                + urllib.parse.urlencode(query)
            )
            raw_path = RAW / f"{station_id}_{int(year)}.json"
            if raw_path.is_file():
                payload_raw = raw_path.read_bytes()
                payload = json.loads(payload_raw)
                status = "cached"
                response_headers: dict[str, str] = {"source": "cache"}
            else:
                try:
                    payload, payload_raw, response_headers = (
                        s306.download_json(query, headers)
                    )
                except Exception as error:
                    detail_rows.append(
                        {
                            "station_id": station_id,
                            "source_id": source_id,
                            "exposure_category": exposure_category,
                            "year": int(year),
                            "url": url,
                            "status": "failed",
                            "downloaded_at_utc": downloaded_at,
                            "content_type": "",
                            "bytes": 0,
                            "sha256": "",
                            "rows": 0,
                            "nonmissing_rows": 0,
                            "error": str(error),
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "station_id": station_id,
                                "year": int(year),
                                "status": "failed",
                                "error": str(error)[:500],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
                raw_path.write_bytes(payload_raw)
                status = "downloaded"
            parsed = s306.normalize_observations(
                payload,
                station_id,
                ACCEPTED_QUALITY_CODE,
            )
            frames.append(parsed)
            row = {
                "station_id": station_id,
                "source_id": source_id,
                "exposure_category": exposure_category,
                "year": int(year),
                "url": url,
                "status": status,
                "downloaded_at_utc": downloaded_at,
                "content_type": response_headers.get("content-type", ""),
                "bytes": int(len(payload_raw)),
                "sha256": hashlib.sha256(payload_raw).hexdigest(),
                "rows": int(len(parsed)),
                "nonmissing_rows": int(
                    parsed["observation_present"].sum()
                ),
            }
            detail_rows.append(row)
            manifest_rows.append(
                {
                    "station_id": station_id,
                    "year": int(year),
                    "url": url,
                    "downloaded_at_utc": downloaded_at,
                    "bytes": int(len(payload_raw)),
                    "sha256": row["sha256"],
                    "raw_path": str(
                        raw_path.relative_to(ROOT)
                    ).replace("\\", "/"),
                }
            )
            print(
                json.dumps(
                    {
                        "station_id": station_id,
                        "year": int(year),
                        "status": status,
                        "rows": int(len(parsed)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    detail = pd.DataFrame(detail_rows)
    if frames:
        long = pd.concat(frames, ignore_index=True)
    else:
        long = pd.DataFrame(
            columns=[
                "station_id",
                "timestamp_utc",
                "temperature_c",
                "quality_code",
                "observation_present",
                "year",
            ]
        )
    long = long.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).drop_duplicates(
        ["station_id", "timestamp_utc"],
        keep="last",
    )
    long = long[
        long["timestamp_utc"].le(
            pd.Timestamp(protocol["test_period_end_utc"])
        )
    ].copy()
    long["split"] = np.select(
        [
            long["year"].between(2021, 2023),
            long["year"].eq(2024),
            long["year"].eq(2025),
        ],
        ["train", "valid", "test"],
        default="outside_protocol",
    )
    long = long[long["split"].ne("outside_protocol")].copy()
    long = long.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    normalized_path = TABLES / f"{PREFIX}_normalized.parquet"
    long.to_parquet(normalized_path, index=False)
    detail.to_csv(
        TABLES / f"{PREFIX}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_by_year = {
        year: "train" if 2021 <= year <= 2023 else (
            "valid" if year == 2024 else "test"
        )
        for year in YEARS
    }
    year_periods = expected_year_periods(protocol)
    detail_status = {
        (str(row["station_id"]), int(row["year"])): str(row["status"])
        for row in detail.to_dict("records")
    }

    year_rows: list[dict[str, object]] = []
    for station_id in protocol["station_ids"]:
        station_id = str(station_id)
        station_frame = long[
            long["station_id"].astype(str).eq(station_id)
        ]
        for year in YEARS:
            year = int(year)
            expected_index = year_periods[year]
            year_frame = station_frame[station_frame["year"].eq(year)]
            present = int(year_frame["observation_present"].sum())
            observed = int(len(year_frame))
            expected = int(len(expected_index))
            coverage = present / expected if expected else 0.0
            year_rows.append(
                {
                    "station_id": station_id,
                    "year": year,
                    "split": split_by_year[year],
                    "download_status": detail_status.get(
                        (station_id, year),
                        "missing",
                    ),
                    "expected_rows": expected,
                    "observed_rows": observed,
                    "present_rows": present,
                    "coverage": float(coverage),
                    "coverage_pass": bool(
                        expected > 0
                        and present > 0
                        and coverage >= COVERAGE_MIN
                    ),
                    "expected_start_utc": expected_index[0].isoformat(),
                    "expected_end_utc": expected_index[-1].isoformat(),
                }
            )
    year_audit = pd.DataFrame(year_rows)
    year_audit.to_csv(
        TABLES / f"{PREFIX}_year_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_rows: list[dict[str, object]] = []
    retained_ids: list[str] = []
    for station_id in protocol["station_ids"]:
        station_id = str(station_id)
        station_years = year_audit[
            year_audit["station_id"].astype(str).eq(station_id)
        ].copy()
        row: dict[str, object] = {"station_id": station_id}
        retained = bool(
            len(station_years) == len(YEARS)
            and station_years["coverage_pass"].all()
        )
        for split in ("train", "valid", "test"):
            split_years = station_years[
                station_years["split"].eq(split)
            ]
            expected = int(split_years["expected_rows"].sum())
            observed = int(split_years["observed_rows"].sum())
            present = int(split_years["present_rows"].sum())
            coverage = present / expected if expected else 0.0
            row[f"{split}_rows"] = expected
            row[f"{split}_observed_rows"] = observed
            row[f"{split}_present"] = present
            row[f"{split}_coverage"] = float(coverage)
            if expected == 0 or present == 0 or coverage < COVERAGE_MIN:
                retained = False
        min_index = station_years["coverage"].idxmin()
        row["min_coverage_year"] = int(
            station_years.loc[min_index, "year"]
        )
        row["min_coverage"] = float(
            station_years.loc[min_index, "coverage"]
        )
        failed_years = station_years.loc[
            ~station_years["coverage_pass"],
            "year",
        ].astype(int).tolist()
        row["failed_years"] = ",".join(str(value) for value in failed_years)
        row["retained"] = bool(retained)
        audit_rows.append(row)
        if retained:
            retained_ids.append(station_id)
    station_audit = pd.DataFrame(audit_rows)
    station_audit.to_csv(
        TABLES / f"{PREFIX}_station_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_manifest = {
        "status": "TEST_MANIFEST_HASH_BOUND_AFTER_FROZEN_DOWNLOAD",
        "package_id": protocol["package_id"],
        "rows": manifest_rows,
        "observation_values_read": int(long["observation_present"].sum()),
        "labels_inspected": False,
        "note": (
            "This manifest records the frozen observation download. It "
            "does not contain attack labels or performance outcomes."
        ),
        "downloaded_at_utc": downloaded_at,
    }
    (PACKAGE_ROOT / "test_manifest.json").write_text(
        json.dumps(
            test_manifest,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = {
        "status": "PASS_MET_NORWAY_DISJOINT_DOWNLOAD_AND_COVERAGE_AUDIT",
        "package_id": protocol["package_id"],
        "provider_id": protocol["provider_id"],
        "provider_is_new": False,
        "network_id": protocol["network_id"],
        "network_is_disjoint_from_stage304": True,
        "station_count": int(len(station_table)),
        "expected_requests": int(len(station_table) * len(YEARS)),
        "downloaded_count": int(detail["status"].eq("downloaded").sum()),
        "cached_count": int(detail["status"].eq("cached").sum()),
        "failed_count": int(detail["status"].eq("failed").sum()),
        "normalized_rows": int(len(long)),
        "observation_present_rows": int(
            long["observation_present"].sum()
        ),
        "coverage_min": float(COVERAGE_MIN),
        "coverage_denominator": (
            "Expected hourly UTC timestamps for each station-year: "
            "2021-2024 full years and 2025-01-01 through "
            "2025-08-24T23:59:59Z."
        ),
        "retained_station_count": int(len(retained_ids)),
        "retained_station_ids": retained_ids,
        "excluded_station_ids": [
            str(value)
            for value in station_audit.loc[
                ~station_audit["retained"],
                "station_id",
            ].tolist()
        ],
        "year_audit_sha256": sha256_file(
            TABLES / f"{PREFIX}_year_audit.csv"
        ),
        "normalized_sha256": sha256_file(normalized_path),
        "detail_sha256": sha256_file(
            TABLES / f"{PREFIX}_detail.csv"
        ),
        "test_manifest_sha256": sha256_file(
            PACKAGE_ROOT / "test_manifest.json"
        ),
        "performance_status": "NOT_EVALUATED",
        "labels_inspected": False,
        "limitations": [
            "MET Norway is not a new independent provider.",
            "This network is disjoint from Stage304 but remains in the "
            "same provider.",
            "No detector was fit and no performance was computed.",
            "All future attacks are transparent controlled injections.",
            "Coverage uses the frozen expected hourly grid, not the "
            "number of rows returned by the API.",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = [
        "# 阶段311 不相交MET Norway官方下载与覆盖率审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 站点数：`{payload['station_count']}`",
        f"- 预期请求：`{payload['expected_requests']}`",
        f"- 下载成功：`{payload['downloaded_count']}`",
        f"- 缓存命中：`{payload['cached_count']}`",
        f"- 规范化行数：`{payload['normalized_rows']}`",
        f"- 保留站点：`{payload['retained_station_count']}`",
        f"- 排除站点：`{', '.join(payload['excluded_station_ids']) or '无'}`",
        f"- 覆盖率下限：`{payload['coverage_min']:.2f}`",
        f"- 性能状态：`{payload['performance_status']}`",
        "",
        "## 边界",
        "",
        "- 只下载Stage310冻结的12个站点和官方质量码0。",
        "- 覆盖率按每个站点每年的预期逐小时时间网格计算，失败年份计0。",
        "- 不训练模型、不选择阈值、不注入攻击、不计算性能。",
        "- 同一provider内的不相交站点网络不能证明机构无关。",
        "- 后续攻击仍为透明受控注入，不是真实攻击日志。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "station_count": payload["station_count"],
                "retained_station_count": payload[
                    "retained_station_count"
                ],
                "normalized_rows": payload["normalized_rows"],
                "failed_count": payload["failed_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
