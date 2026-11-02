from __future__ import annotations

"""Stage 216: normalize the pre-registered MeteoSwiss hourly CSVs.

This stage reads only the 12 Stage-214 stations and the official hourly
2020-2029 CSVs downloaded in Stage 215.  It produces one hourly table
with UTC timestamps, temperature, relative humidity, dew point,
pressure, station metadata, and the frozen train/valid/test split.

No detector is fitted, no threshold is selected, and no performance is
computed.  Empty CSV fields are converted to missing values and retained;
they are not imputed.  The split follows the Stage-214 contract:
2021-2023 train, 2024 valid, 2025 test.
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
DATA_DIR = WORKSPACE_ROOT / "dataset" / "external" / "meteoswiss"
PROTOCOL = TABLES / "stage214_swiss_protocol_preregistration.json"
STATIONS = TABLES / "stage214_swiss_protocol_preregistration_stations.csv"
DOWNLOAD_DETAIL = TABLES / "stage215_swiss_official_download_detail.csv"
PREFIX = "stage216_swiss_normalized"

YEAR_START = 2021
YEAR_END = 2025
EXPECTED_ROWS_PER_STATION = 5 * 365 * 24 + 24
REQUIRED_COLUMNS = {
    "station_abbr",
    "reference_timestamp",
    "tre200h0",
    "ure200h0",
    "tde200h0",
    "prestah0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_station_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        encoding="utf-8",
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{path.name} missing columns: {sorted(missing)}"
        )
    return frame


def normalized_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_DOWNLOAD":
        raise RuntimeError("Stage214 protocol is not frozen before download")

    station_table = pd.read_csv(STATIONS, dtype={"station_id": str})
    station_table["station_id"] = station_table["station_id"].str.lower()
    download = pd.read_csv(
        DOWNLOAD_DETAIL,
        dtype={"station_id": str},
    )
    download["station_id"] = download["station_id"].str.lower()
    successful = download[
        download["status"].isin(["downloaded", "already_present"])
    ].copy()
    expected_stations = set(station_table["station_id"])
    if set(successful["station_id"]) != expected_stations:
        raise RuntimeError(
            "Stage215 successful download station set differs from Stage214"
        )
    if not successful["audit_required_columns_ok"].fillna(False).astype(
        bool
    ).all():
        raise RuntimeError("Stage215 CSV column audit did not pass")

    station_lookup = station_table.set_index("station_id").to_dict("index")
    frames: list[pd.DataFrame] = []
    archive_rows: list[dict[str, object]] = []
    for row in successful.sort_values("station_id").to_dict(
        orient="records"
    ):
        station_id = str(row["station_id"]).lower()
        archive_path = Path(str(row["target"]))
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        raw = read_station_csv(archive_path)
        station_values = raw["station_abbr"].astype(str).str.strip().str.lower()
        if set(station_values) != {station_id}:
            raise RuntimeError(
                f"archive {archive_path.name} contains station IDs "
                f"{sorted(station_values.unique().tolist())}"
            )
        timestamp = pd.to_datetime(
            raw["reference_timestamp"].astype(str).str.strip(),
            format="%d.%m.%Y %H:%M",
            errors="coerce",
            utc=True,
        )
        temperature = pd.to_numeric(raw["tre200h0"], errors="coerce")
        humidity = pd.to_numeric(raw["ure200h0"], errors="coerce")
        dew_point = pd.to_numeric(raw["tde200h0"], errors="coerce")
        pressure = pd.to_numeric(raw["prestah0"], errors="coerce")
        station = station_lookup[station_id]
        normalized = pd.DataFrame(
            {
                "source": "MeteoSwiss_SwissMetNet",
                "station_id": station_id,
                "station_name": station["station_name"],
                "timestamp_utc": timestamp,
                "latitude": float(station["latitude"]),
                "longitude": float(station["longitude"]),
                "temperature_c": temperature,
                "humidity_pct": humidity,
                "dew_point_c": dew_point,
                "pressure_hpa": pressure,
            }
        )
        raw_missing = int(normalized["timestamp_utc"].isna().sum())
        raw_timestamp_min = normalized["timestamp_utc"].min()
        raw_timestamp_max = normalized["timestamp_utc"].max()
        expected_index = pd.date_range(
            f"{YEAR_START}-01-01 00:00:00+00:00",
            f"{YEAR_END}-12-31 23:00:00+00:00",
            freq="h",
        )
        raw_observed = pd.DatetimeIndex(
            normalized["timestamp_utc"].dropna()
        )
        raw_missing_timestamps = int(
            len(expected_index.difference(raw_observed))
        )
        raw_extra_timestamps = int(
            len(raw_observed.difference(expected_index))
        )
        normalized = (
            normalized.set_index("timestamp_utc")
            .reindex(expected_index)
            .rename_axis("timestamp_utc")
            .reset_index()
        )
        normalized["observation_present"] = (
            normalized["temperature_c"].notna()
            | normalized["humidity_pct"].notna()
            | normalized["dew_point_c"].notna()
            | normalized["pressure_hpa"].notna()
        )
        normalized["source"] = normalized["source"].fillna(
            "MeteoSwiss_SwissMetNet"
        )
        normalized["station_id"] = normalized["station_id"].fillna(
            station_id
        )
        normalized["station_name"] = normalized["station_name"].fillna(
            station["station_name"]
        )
        for column in ["latitude", "longitude"]:
            normalized[column] = normalized[column].fillna(
                float(station[column])
            )
        normalized["year"] = normalized["timestamp_utc"].dt.year
        normalized["split"] = np.select(
            [
                normalized["year"].between(2021, 2023),
                normalized["year"].eq(2024),
                normalized["year"].eq(2025),
            ],
            ["train", "valid", "test"],
            default="outside_protocol",
        )
        frames.append(normalized)
        archive_rows.append(
            {
                "station_id": station_id,
                "archive": archive_path.name,
                "archive_sha256": sha256_file(archive_path),
                "raw_rows": int(len(raw)),
                "raw_missing_timestamps": raw_missing_timestamps,
                "raw_extra_timestamps": raw_extra_timestamps,
                "raw_invalid_timestamp": raw_missing,
                "raw_timestamp_min": (
                    None
                    if pd.isna(raw_timestamp_min)
                    else str(raw_timestamp_min)
                ),
                "raw_timestamp_max": (
                    None
                    if pd.isna(raw_timestamp_max)
                    else str(raw_timestamp_max)
                ),
                "protocol_rows": int(
                    normalized["year"].between(YEAR_START, YEAR_END).sum()
                ),
                "timestamp_min": str(normalized["timestamp_utc"].min()),
                "timestamp_max": str(normalized["timestamp_utc"].max()),
                "duplicate_station_timestamp": int(
                    normalized.duplicated(
                        ["station_id", "timestamp_utc"]
                    ).sum()
                ),
                "invalid_timestamp": int(
                    normalized["timestamp_utc"].isna().sum()
                ),
                "grid_filled_missing": raw_missing_timestamps,
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    combined = combined[
        combined["year"].between(YEAR_START, YEAR_END)
    ].copy()
    if combined.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError(
            "duplicate station-timestamp rows after normalization"
        )
    if combined["timestamp_utc"].isna().any():
        raise RuntimeError("invalid timestamps remain after normalization")
    if set(combined["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("unexpected split labels")
    return combined, pd.DataFrame(archive_rows)


def audit(
    frame: pd.DataFrame,
    archives: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    station_rows: list[dict[str, object]] = []
    for station_id, group in frame.groupby("station_id", sort=True):
        expected_index = pd.date_range(
            f"{YEAR_START}-01-01 00:00:00+00:00",
            f"{YEAR_END}-12-31 23:00:00+00:00",
            freq="h",
        )
        observed_index = pd.DatetimeIndex(group["timestamp_utc"])
        missing_timestamps = expected_index.difference(observed_index)
        extra_timestamps = observed_index.difference(expected_index)
        station_rows.append(
            {
                "station_id": station_id,
                "rows": int(len(group)),
                "expected_rows": int(len(expected_index)),
                "missing_timestamps": int(len(missing_timestamps)),
                "extra_timestamps": int(len(extra_timestamps)),
                "duplicate_rows": int(
                    group.duplicated(
                        ["station_id", "timestamp_utc"]
                    ).sum()
                ),
                "missing_observation_rows": int(
                    (~group["observation_present"]).sum()
                ),
                "temperature_missing": int(
                    group["temperature_c"].isna().sum()
                ),
                "humidity_missing": int(
                    group["humidity_pct"].isna().sum()
                ),
                "dew_point_missing": int(
                    group["dew_point_c"].isna().sum()
                ),
                "pressure_missing": int(
                    group["pressure_hpa"].isna().sum()
                ),
                "train_rows": int(group["split"].eq("train").sum()),
                "valid_rows": int(group["split"].eq("valid").sum()),
                "test_rows": int(group["split"].eq("test").sum()),
                "timestamp_min": str(group["timestamp_utc"].min()),
                "timestamp_max": str(group["timestamp_utc"].max()),
            }
        )
    station_audit = pd.DataFrame(station_rows)
    payload = {
        "status": (
            "PASS_PIPELINE_REVIEW_REQUIRED"
            if station_audit["duplicate_rows"].eq(0).all()
            and station_audit["missing_timestamps"].eq(0).all()
            and station_audit["extra_timestamps"].eq(0).all()
            and station_audit["rows"].eq(EXPECTED_ROWS_PER_STATION).all()
            and archives["duplicate_station_timestamp"].eq(0).all()
            and archives["invalid_timestamp"].eq(0).all()
            else "REVIEW_REQUIRED"
        ),
        "protocol": (
            "阶段216只标准化阶段214冻结的12个MeteoSwiss站点和阶段215"
            "下载的官方小时历史CSV；2021-2023为train，2024为valid，"
            "2025为test；reference_timestamp按官方Open Data Docs明确"
            "声明的UTC解析；空字段转为缺失并保留，不插补、不训练模型、"
            "不选择阈值、不计算性能；所有攻击仍须由后续阶段按冻结协议"
            "透明注入。"
        ),
        "provider": "MeteoSwiss (Federal Office of Meteorology and Climatology)",
        "dataset_id": "ch.meteoschweiz.ogd-smn",
        "source": "MeteoSwiss_SwissMetNet",
        "primary_variables": ["tre200h0", "ure200h0"],
        "auxiliary_variables": ["tde200h0", "prestah0"],
        "timezone": "UTC",
        "timezone_evidence": (
            "Official MeteoSwiss Open Data documentation states: "
            "'All reference time stamps at MeteoSwiss are in UTC'."
        ),
        "missing_rule": (
            "Official documentation states missing values are represented "
            "as empty fields; no sentinel substitution and no imputation."
        ),
        "year_start": YEAR_START,
        "year_end": YEAR_END,
        "train_years": [2021, 2022, 2023],
        "valid_year": 2024,
        "test_year": 2025,
        "station_count": int(frame["station_id"].nunique()),
        "rows": int(len(frame)),
        "expected_rows": int(
            len(station_audit) * EXPECTED_ROWS_PER_STATION
        ),
        "temperature_missing_rows": int(
            frame["temperature_c"].isna().sum()
        ),
        "humidity_missing_rows": int(
            frame["humidity_pct"].isna().sum()
        ),
        "dew_point_missing_rows": int(
            frame["dew_point_c"].isna().sum()
        ),
        "pressure_missing_rows": int(
            frame["pressure_hpa"].isna().sum()
        ),
        "duplicate_station_timestamp_rows": int(
            frame.duplicated(["station_id", "timestamp_utc"]).sum()
        ),
        "missing_observation_rows": int(
            (~frame["observation_present"]).sum()
        ),
        "raw_missing_timestamps": int(
            archives["raw_missing_timestamps"].sum()
        ),
        "station_coverage_pass": bool(
            station_audit["missing_timestamps"].eq(0).all()
            and station_audit["extra_timestamps"].eq(0).all()
        ),
        "performance_status": "NOT_EVALUATED",
        "archive_audit": archives.to_dict(orient="records"),
        "limitations": [
            "本阶段只标准化数据，不评价检测器性能。",
            "MeteoSwiss小时CSV没有与DWD QN_9等价的逐行数值质量码；不得伪造质量码。",
            "空字段按缺失保留，不插补；缺失不等于攻击标签。",
            "所有后续攻击必须是透明受控注入，不是真实攻击日志。",
        ],
    }
    return station_audit, payload


def main() -> int:
    frame, archives = normalized_frame()
    station_audit, payload = audit(frame, archives)
    frame.to_parquet(TABLES / f"{PREFIX}.parquet", index=False)
    station_audit.to_csv(
        TABLES / f"{PREFIX}_station_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    archives.to_csv(
        TABLES / f"{PREFIX}_archive_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段216 MeteoSwiss标准化数据审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 来源：`{payload['source']}`",
        f"- 站点：`{payload['station_count']}`；年份：`{YEAR_START}-{YEAR_END}`",
        f"- 时区：`{payload['timezone']}`；依据：{payload['timezone_evidence']}",
        f"- 行数：`{payload['rows']}`；预期逐小时行数：`{payload['expected_rows']}`",
        f"- train/valid/test：`{payload['train_years']}` / `{payload['valid_year']}` / `{payload['test_year']}`",
        f"- 温度缺失：`{payload['temperature_missing_rows']}`；湿度缺失：`{payload['humidity_missing_rows']}`；露点缺失：`{payload['dew_point_missing_rows']}`；气压缺失：`{payload['pressure_missing_rows']}`",
        f"- 站点-时间重复：`{payload['duplicate_station_timestamp_rows']}`",
        f"- 显式缺测行：`{payload['missing_observation_rows']}`；原始归档缺口：`{payload['raw_missing_timestamps']}`",
        f"- 性能状态：`{payload['performance_status']}`",
        "",
        "| 站点 | 行数 | 缺失时刻 | 缺测行 | 额外时刻 | 重复 | 温度缺失 | 湿度缺失 | 露点缺失 | 气压缺失 | train | valid | test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in station_audit.to_dict(orient="records"):
        lines.append(
            f"| {row['station_id']} | {row['rows']} | "
            f"{row['missing_timestamps']} | "
            f"{row['missing_observation_rows']} | "
            f"{row['extra_timestamps']} | {row['duplicate_rows']} | "
            f"{row['temperature_missing']} | {row['humidity_missing']} | "
            f"{row['dew_point_missing']} | {row['pressure_missing']} | "
            f"{row['train_rows']} | {row['valid_rows']} | "
            f"{row['test_rows']} |"
        )
    lines += [
        "",
        "本阶段不插补、不训练模型、不选择阈值、不计算性能；空字段不是攻击标签。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["rows"],
                "expected_rows": payload["expected_rows"],
                "stations": payload["station_count"],
                "duplicates": payload[
                    "duplicate_station_timestamp_rows"
                ],
                "temperature_missing_rows": payload[
                    "temperature_missing_rows"
                ],
                "humidity_missing_rows": payload[
                    "humidity_missing_rows"
                ],
                "missing_observation_rows": payload[
                    "missing_observation_rows"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if payload["status"] == "PASS_PIPELINE_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
