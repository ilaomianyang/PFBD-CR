from __future__ import annotations

"""Stage 236: normalize ECCC hourly data and audit coverage.

The stage reads only the 16 Stage-234 stations and the raw Stage-235
CSV pages.  It produces one hourly UTC table and a pre-declared
data-availability audit.  A station is retained only when TEMP and
RELATIVE_HUMIDITY coverage is at least 70% in every train, valid, and
test split.  No detector is fitted, no threshold is selected, and no
performance is computed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
WORKSPACE_ROOT = ROOT.parents[1]
DATA_DIR = WORKSPACE_ROOT / "dataset" / "external" / "eccc"
PROTOCOL = TABLES / "stage234_eccc_protocol_preregistration.json"
STATIONS = TABLES / "stage234_eccc_protocol_preregistration_stations.csv"
PREFIX = "stage236_eccc_normalized"

YEAR_START = 2021
YEAR_END = 2025
EXPECTED_ROWS_PER_STATION = 5 * 365 * 24 + 24


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_station(
    station: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    station_id = int(station["station_id"])
    raw_path = DATA_DIR / str(station_id) / "raw_combined.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    raw = pd.read_csv(raw_path, dtype=str)
    raw.columns = [str(column).strip() for column in raw.columns]
    required = {
        "STN_ID",
        "UTC_DATE",
        "TEMP",
        "RELATIVE_HUMIDITY",
        "DEW_POINT_TEMP",
        "STATION_PRESSURE",
    }
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(
            f"ECCC station {station_id} missing columns: {sorted(missing)}"
        )
    station_values = pd.to_numeric(raw["STN_ID"], errors="coerce")
    if set(station_values.dropna().astype(int)) != {station_id}:
        raise RuntimeError(
            f"ECCC station {station_id} archive contains other stations"
        )

    timestamp = pd.to_datetime(raw["UTC_DATE"], errors="coerce", utc=True)
    temperature = pd.to_numeric(raw["TEMP"], errors="coerce")
    humidity = pd.to_numeric(raw["RELATIVE_HUMIDITY"], errors="coerce")
    dew_point = pd.to_numeric(raw["DEW_POINT_TEMP"], errors="coerce")
    pressure_kpa = pd.to_numeric(raw["STATION_PRESSURE"], errors="coerce")

    frame = pd.DataFrame(
        {
            "source": "ECCC_hourly",
            "station_id": str(station_id),
            "station_name": station["station_name"],
            "timestamp_utc": timestamp,
            "latitude": float(station["latitude"]),
            "longitude": float(station["longitude"]),
            "elevation_m": float(station["elevation_m"]),
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "dew_point_c": dew_point,
            "pressure_kpa": pressure_kpa,
        }
    )
    raw_rows = int(len(frame))
    invalid_timestamp = int(frame["timestamp_utc"].isna().sum())
    frame = frame.dropna(subset=["timestamp_utc"]).copy()
    duplicate_station_timestamp = int(
        frame.duplicated(["station_id", "timestamp_utc"]).sum()
    )
    frame = frame.drop_duplicates(
        ["station_id", "timestamp_utc"], keep="first"
    ).copy()
    frame = frame[
        frame["timestamp_utc"].between(
            f"{YEAR_START}-01-01 00:00:00+00:00",
            f"{YEAR_END}-12-31 23:00:00+00:00",
        )
    ].copy()

    expected_index = pd.date_range(
        f"{YEAR_START}-01-01 00:00:00+00:00",
        f"{YEAR_END}-12-31 23:00:00+00:00",
        freq="h",
    )
    observed_index = pd.DatetimeIndex(frame["timestamp_utc"])
    raw_missing_timestamps = int(len(expected_index.difference(observed_index)))
    raw_extra_timestamps = int(len(observed_index.difference(expected_index)))
    frame = (
        frame.set_index("timestamp_utc")
        .reindex(expected_index)
        .rename_axis("timestamp_utc")
        .reset_index()
    )
    frame["observation_present"] = (
        frame["temperature_c"].notna()
        | frame["humidity_pct"].notna()
        | frame["dew_point_c"].notna()
        | frame["pressure_kpa"].notna()
    )
    frame["source"] = frame["source"].fillna("ECCC_hourly")
    frame["station_id"] = frame["station_id"].fillna(str(station_id))
    frame["station_name"] = frame["station_name"].fillna(
        station["station_name"]
    )
    for column in ["latitude", "longitude", "elevation_m"]:
        frame[column] = frame[column].fillna(float(station[column]))
    frame["year"] = frame["timestamp_utc"].dt.year
    frame["split"] = np.select(
        [
            frame["year"].between(2021, 2023),
            frame["year"].eq(2024),
            frame["year"].eq(2025),
        ],
        ["train", "valid", "test"],
        default="outside_protocol",
    )
    audit = {
        "station_id": station_id,
        "raw_rows": raw_rows,
        "raw_invalid_timestamp": invalid_timestamp,
        "raw_duplicate_station_timestamp": duplicate_station_timestamp,
        "raw_missing_timestamps": raw_missing_timestamps,
        "raw_extra_timestamps": raw_extra_timestamps,
        "protocol_rows": int(len(frame)),
        "duplicate_station_timestamp": int(
            frame.duplicated(["station_id", "timestamp_utc"]).sum()
        ),
        "invalid_timestamp": int(frame["timestamp_utc"].isna().sum()),
        "raw_combined_sha256": sha256_file(raw_path),
        "timestamp_min": str(frame["timestamp_utc"].min()),
        "timestamp_max": str(frame["timestamp_utc"].max()),
    }
    return frame, audit


def coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (station_id, split), group in frame.groupby(
        ["station_id", "split"], sort=True
    ):
        rows.append(
            {
                "station_id": station_id,
                "split": split,
                "rows": int(len(group)),
                "observation_present": int(
                    group["observation_present"].sum()
                ),
                "observation_present_pct": float(
                    group["observation_present"].mean()
                ),
                "temperature_present": int(
                    group["temperature_c"].notna().sum()
                ),
                "temperature_present_pct": float(
                    group["temperature_c"].notna().mean()
                ),
                "humidity_present": int(
                    group["humidity_pct"].notna().sum()
                ),
                "humidity_present_pct": float(
                    group["humidity_pct"].notna().mean()
                ),
                "dew_point_present": int(
                    group["dew_point_c"].notna().sum()
                ),
                "pressure_present": int(
                    group["pressure_kpa"].notna().sum()
                ),
                "timestamp_min": str(group["timestamp_utc"].min()),
                "timestamp_max": str(group["timestamp_utc"].max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_ECCC_PERFORMANCE":
        raise RuntimeError("Stage234 protocol is not frozen before download")
    station_table = pd.read_csv(STATIONS, dtype={"station_id": int})
    station_table["station_id"] = station_table["station_id"].astype(int)
    expected_ids = [int(value) for value in protocol["station_ids"]]
    if station_table["station_id"].tolist() != expected_ids:
        raise RuntimeError("Stage234 station order changed")

    frames: list[pd.DataFrame] = []
    archive_rows: list[dict[str, object]] = []
    for station in station_table.to_dict(orient="records"):
        frame, audit = normalize_station(station)
        frames.append(frame)
        archive_rows.append(audit)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["station_id", "timestamp_utc"], kind="stable"
    ).reset_index(drop=True)
    if combined.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("duplicate station-timestamp rows after normalization")
    if combined["timestamp_utc"].isna().any():
        raise RuntimeError("invalid timestamps remain after normalization")

    coverage = coverage_summary(combined)
    minimum_coverage = float(protocol["minimum_primary_coverage"])
    decision_rows: list[dict[str, object]] = []
    retained: list[int] = []
    excluded: list[int] = []
    for station_id, group in coverage.groupby("station_id", sort=True):
        split_rows = group.set_index("split")
        split_ok = True
        reasons: list[str] = []
        for split in ["train", "valid", "test"]:
            if split not in split_rows.index:
                split_ok = False
                reasons.append(f"{split}_missing")
                continue
            row = split_rows.loc[split]
            if float(row["temperature_present_pct"]) < minimum_coverage:
                split_ok = False
                reasons.append(
                    f"{split}_temperature_below_{minimum_coverage:.2f}"
                )
            if float(row["humidity_present_pct"]) < minimum_coverage:
                split_ok = False
                reasons.append(
                    f"{split}_humidity_below_{minimum_coverage:.2f}"
                )
        decision_rows.append(
            {
                "station_id": station_id,
                "decision": "retain" if split_ok else "exclude",
                "reasons": ";".join(reasons),
            }
        )
        if split_ok:
            retained.append(int(station_id))
        else:
            excluded.append(int(station_id))

    decisions = pd.DataFrame(decision_rows)
    station_summary = (
        coverage.pivot(
            index="station_id",
            columns="split",
            values=[
                "temperature_present_pct",
                "humidity_present_pct",
            ],
        )
        .reset_index()
    )
    station_summary.columns = [
        "_".join(
            [str(value) for value in column if str(value) != ""]
        )
        if isinstance(column, tuple)
        else str(column)
        for column in station_summary.columns
    ]
    station_summary = station_summary.merge(
        decisions, on="station_id", how="left", validate="one_to_one"
    )

    combined.to_parquet(TABLES / f"{PREFIX}.parquet", index=False)
    coverage.to_csv(
        TABLES / f"{PREFIX}_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(archive_rows).to_csv(
        TABLES / f"{PREFIX}_archive_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_summary.to_csv(
        TABLES / f"{PREFIX}_station_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    passed = (
        len(retained) >= int(protocol["min_retained_stations"])
        and combined["timestamp_utc"].isna().sum() == 0
    )
    payload = {
        "status": (
            "PASS_PIPELINE_REVIEW_REQUIRED"
            if passed
            else "REVIEW_REQUIRED"
        ),
        "protocol": (
            "阶段236只标准化阶段235下载的16个ECCC官方站点小时观测；"
            "UTC_DATE用于逐小时UTC网格，2021-2023为train，2024为"
            "valid，2025为test；TEMP/RELATIVE_HUMIDITY/DEW_POINT_TEMP/"
            "STATION_PRESSURE空或非数值字段转缺失并保留，不插补。"
            "站点仅在train、valid、test三段温度和湿度覆盖率均不低于"
            f"{minimum_coverage:.2f}时保留；该规则在阶段234预注册，"
            "不是性能筛选。不训练模型、不选择阈值、不计算性能。"
        ),
        "provider": protocol["provider"],
        "dataset": protocol["dataset"],
        "source": "ECCC_hourly",
        "primary_variables": ["TEMP", "RELATIVE_HUMIDITY"],
        "auxiliary_variables": ["DEW_POINT_TEMP", "STATION_PRESSURE"],
        "minimum_primary_coverage": minimum_coverage,
        "station_count": int(combined["station_id"].nunique()),
        "rows": int(len(combined)),
        "expected_rows": int(
            combined["station_id"].nunique() * EXPECTED_ROWS_PER_STATION
        ),
        "retained_station_ids": retained,
        "excluded_station_ids": excluded,
        "retained_station_count": len(retained),
        "excluded_station_count": len(excluded),
        "coverage_rows": int(len(coverage)),
        "archive_audit": archive_rows,
        "station_summary": station_summary.to_dict(orient="records"),
        "limitations": [
            "本阶段只标准化数据，不评价检测器性能。",
            "站点保留/排除只依据预注册的主变量分段覆盖率，不依据test性能。",
            "ECCC小时序列可能因站址迁移、仪器变化和报告规则变化而不均质。",
            "空字段按缺失保留，不插补；缺失不等于攻击标签。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段236 ECCC标准化与覆盖率审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 来源：`{payload['source']}`；站点：`{payload['station_count']}`",
        f"- 行数：`{payload['rows']}`；预期：`{payload['expected_rows']}`",
        f"- 保留站点：`{retained}`",
        f"- 排除站点：`{excluded}`",
        f"- 主变量分段覆盖率阈值：`{minimum_coverage:.2f}`",
        "",
        "## 站点覆盖率",
        "",
        "| 站点 | train温度 | train湿度 | valid温度 | valid湿度 | test温度 | test湿度 | 决定 | 原因 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in station_summary.to_dict(orient="records"):
        lines.append(
            f"| {int(row['station_id'])} | "
            f"{row.get('temperature_present_pct_train', 0.0):.4f} | "
            f"{row.get('humidity_present_pct_train', 0.0):.4f} | "
            f"{row.get('temperature_present_pct_valid', 0.0):.4f} | "
            f"{row.get('humidity_present_pct_valid', 0.0):.4f} | "
            f"{row.get('temperature_present_pct_test', 0.0):.4f} | "
            f"{row.get('humidity_present_pct_test', 0.0):.4f} | "
            f"{row['decision']} | {row['reasons']} |"
        )
    lines += [
        "",
        "本阶段不插补、不训练模型、不选择阈值、不计算性能；",
        "站点保留/排除是预注册的数据可用性规则，不是性能筛选。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "stations": payload["station_count"],
                "rows": payload["rows"],
                "retained_station_count": payload["retained_station_count"],
                "excluded_station_count": payload["excluded_station_count"],
                "retained_station_ids": retained,
                "excluded_station_ids": excluded,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if payload["status"] == "PASS_PIPELINE_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
