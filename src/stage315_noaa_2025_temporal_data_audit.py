from __future__ import annotations

"""Stage 315 data audit: freeze the NOAA 2025 temporal holdout.

This stage only materializes and audits the data boundary. It does not fit
models, calibrate thresholds, inject attacks, inspect labels, or compute
detection performance. The NOAA provider and station network are reused
from Stage222/248; only the evaluation period is new.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PACKAGE_ROOT = ROOT / "confirmatory_package" / "noaa_2025_temporal"
PREFIX = "stage315_noaa_2025_temporal_data_audit"
PROTOCOL = PACKAGE_ROOT / "protocol_freeze.json"
SOURCE = TABLES / "stage82_noaa_extended_normalized.parquet"
QUALITY = TABLES / "stage83_noaa_extended_quality_audit.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    protocol = load_json(PROTOCOL)
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_2025_PERFORMANCE_EVALUATION":
        raise RuntimeError("NOAA 2025 protocol is not frozen")
    if protocol["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("NOAA 2025 protocol was already evaluated")
    if protocol["labels_inspected"] is not False:
        raise RuntimeError("NOAA 2025 protocol label boundary changed")

    source_hash = sha256_file(SOURCE)
    quality_hash = sha256_file(QUALITY)
    if source_hash != protocol["data_source_sha256"]:
        raise RuntimeError("NOAA normalized source hash changed")
    if quality_hash != protocol["quality_audit_sha256"]:
        raise RuntimeError("NOAA quality audit hash changed")

    station_ids = [str(value) for value in protocol["station_ids"]]
    years = [int(value) for value in protocol["years"]]
    frame = pd.read_parquet(SOURCE)
    frame["station_id"] = frame["station_id"].astype(str)
    frame = frame[
        frame["station_id"].isin(station_ids)
        & frame["year"].astype(int).isin(years)
    ].copy()
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)

    split_by_year: dict[int, str] = {}
    for split_name in ("train", "valid", "test"):
        for year in protocol["split"][split_name]:
            split_by_year[int(year)] = split_name
    if set(split_by_year) != set(years):
        raise RuntimeError("NOAA temporal split and years differ")
    frame["split"] = frame["year"].astype(int).map(split_by_year)
    frame["observation_present"] = pd.to_numeric(
        frame["temperature_c"],
        errors="coerce",
    ).notna()

    expected_pairs = {
        (station_id, year)
        for station_id in station_ids
        for year in years
    }
    actual_pairs = {
        (str(station_id), int(year))
        for station_id, year in frame[["station_id", "year"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    missing_pairs = sorted(expected_pairs - actual_pairs)
    extra_pairs = sorted(actual_pairs - expected_pairs)
    duplicate_rows = int(
        frame.duplicated(["station_id", "timestamp_utc"]).sum()
    )
    if missing_pairs or extra_pairs:
        raise RuntimeError(
            f"NOAA station-year boundary mismatch: missing={missing_pairs}, "
            f"extra={extra_pairs}"
        )
    if duplicate_rows:
        raise RuntimeError("NOAA station-timestamp duplicates found")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("NOAA temporal split labels are incomplete")

    station_year_rows: list[dict[str, object]] = []
    for (station_id, year), part in frame.groupby(
        ["station_id", "year"],
        sort=True,
    ):
        present = int(part["observation_present"].sum())
        rows = int(len(part))
        station_year_rows.append(
            {
                "station_id": str(station_id),
                "year": int(year),
                "split": str(part["split"].iloc[0]),
                "rows": rows,
                "observation_present_rows": present,
                "temperature_missing_rate": (
                    float(1.0 - present / rows) if rows else 1.0
                ),
                "date_min": str(part["timestamp_utc"].min()),
                "date_max": str(part["timestamp_utc"].max()),
                "unique_timestamps": int(part["timestamp_utc"].nunique()),
                "duplicate_station_timestamps": int(
                    part.duplicated(["station_id", "timestamp_utc"]).sum()
                ),
            }
        )

    station_year = pd.DataFrame(station_year_rows)
    test_rows = station_year[station_year["year"].eq(2025)].copy()
    if len(test_rows) != len(station_ids):
        raise RuntimeError("NOAA 2025 test station coverage is incomplete")
    if test_rows["observation_present_rows"].min() <= 0:
        raise RuntimeError("NOAA 2025 contains an empty test station")
    if test_rows["duplicate_station_timestamps"].sum() != 0:
        raise RuntimeError("NOAA 2025 test duplicates were not removed")

    normalized_out = TABLES / f"{PREFIX}_normalized.parquet"
    frame.to_parquet(normalized_out, index=False)
    station_year.to_csv(
        TABLES / f"{PREFIX}_station_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "status": "PASS_NOAA_2025_TEMPORAL_DATA_AUDIT",
        "protocol_id": protocol["protocol_id"],
        "package_id": protocol["package_id"],
        "provider_id": protocol["provider_id"],
        "provider_is_new": False,
        "network_id": protocol["network_id"],
        "station_count": len(station_ids),
        "retained_station_ids": station_ids,
        "retained_station_count": len(station_ids),
        "years": years,
        "split": protocol["split"],
        "excluded_year": 2024,
        "test_period_start_utc": protocol["test_period_start_utc"],
        "test_period_end_utc": protocol["test_period_end_utc"],
        "rows": int(len(frame)),
        "test_rows": int(frame["split"].eq("test").sum()),
        "train_rows": int(frame["split"].eq("train").sum()),
        "valid_rows": int(frame["split"].eq("valid").sum()),
        "station_year_rows": station_year_rows,
        "test_station_count": int(len(test_rows)),
        "test_min_observation_present_rows": int(
            test_rows["observation_present_rows"].min()
        ),
        "test_max_temperature_missing_rate": float(
            test_rows["temperature_missing_rate"].max()
        ),
        "duplicate_station_timestamp_rows": duplicate_rows,
        "performance_status": "NOT_EVALUATED",
        "labels_inspected": False,
        "outcome_summaries_inspected": False,
        "source_sha256": source_hash,
        "quality_audit_sha256": quality_hash,
        "protocol_sha256": sha256_file(PROTOCOL),
        "normalized_sha256": sha256_file(normalized_out),
        "limitations": [
            "NOAA remains the same provider and the same 11-station network.",
            "The 2025 test period is partial and ends on 2025-08-24.",
            "This audit does not fit models, inject attacks, or compute performance.",
            "The result cannot support provider- or station-independent claims.",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage315 NOAA 2025 时间锁定数据审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- provider：`{payload['provider_id']}`，是否为新机构：`False`",
        f"- 站点数：`{payload['station_count']}`",
        f"- 年份：`{payload['years']}`；排除 2024 旧 test 年份",
        f"- train/valid/test 行数：`{payload['train_rows']}/"
        f"{payload['valid_rows']}/{payload['test_rows']}`",
        f"- 2025 站点数：`{payload['test_station_count']}`",
        f"- 2025 最大温度缺失率："
        f"`{payload['test_max_temperature_missing_rate']}`",
        "",
        "## 边界",
        "",
        "- 这是同一 NOAA provider、同一站点网络上的新时间窗。",
        "- 2025 只用于新的锁定评价，不用于选择特征、阈值或基线。",
        "- 本阶段不计算任何检测性能。",
        "- 结果不能关闭 P0-1、P0-5 或 P0-6。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["rows"],
                "test_rows": payload["test_rows"],
                "station_count": payload["station_count"],
                "test_max_temperature_missing_rate": payload[
                    "test_max_temperature_missing_rate"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
