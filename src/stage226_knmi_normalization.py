from __future__ import annotations

"""Stage 226: normalize the pre-registered KNMI hourly archives.

The stage reads only the 12 Stage-225 stations and their official
2021-2030 hourly ZIP archives.  It produces one hourly table with UTC
timestamps, temperature, relative humidity, dew point, pressure, station
metadata, and the frozen train/valid/test split.

KNMI documents ``HH`` as the hourly division (1--24), with division 5
running from 04:00 to 05:00 UT.  The normalized timestamp is therefore
the start of that division: ``YYYYMMDD + (HH - 1) hours``.

Empty fields are converted to missing values and retained; they are not
imputed.  No detector is fitted, no threshold is selected, and no
performance is computed.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
WORKSPACE_ROOT = ROOT.parents[1]
DATA_DIR = WORKSPACE_ROOT / "dataset" / "external" / "knmi"
PROTOCOL = TABLES / "stage224_knmi_protocol_preregistration.json"
STATIONS = TABLES / "stage225_knmi_stations.csv"
DOWNLOAD_DETAIL = TABLES / "stage225_knmi_official_download_detail.csv"
PREFIX = "stage226_knmi_normalized"

YEAR_START = 2021
YEAR_END = 2025
EXPECTED_ROWS_PER_STATION = 5 * 365 * 24 + 24
REQUIRED_COLUMNS = {"STN", "YYYYMMDD", "HH", "T", "U", "TD", "P"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_station_archive(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [
            name for name in archive.namelist() if name.lower().endswith(".txt")
        ]
        if len(members) != 1:
            raise RuntimeError(
                f"expected one text member in {path.name}, found {members}"
            )
        text = archive.read(members[0]).decode("utf-8", errors="replace")
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip("# ").startswith("STN,")
        ),
        None,
    )
    if header_index is None:
        raise RuntimeError(f"{path.name} has no KNMI data header")
    csv_text = "\n".join(
        [lines[header_index].lstrip("# ").strip()] + lines[header_index + 1 :]
    )
    frame = pd.read_csv(io.StringIO(csv_text), dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{path.name} missing columns: {sorted(missing)}"
        )
    return frame


def normalized_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_KNMI_PERFORMANCE":
        raise RuntimeError("Stage224 protocol is not frozen before download")

    station_table = pd.read_csv(STATIONS, dtype={"station_id": str})
    station_table["station_id"] = station_table["station_id"].str.strip()
    download = pd.read_csv(
        DOWNLOAD_DETAIL,
        dtype={"station_id": str},
    )
    download["station_id"] = download["station_id"].str.strip()
    successful = download[
        download["status"].isin(["downloaded", "already_present"])
    ].copy()
    expected_stations = set(station_table["station_id"])
    if set(successful["station_id"]) != expected_stations:
        raise RuntimeError(
            "Stage225 successful download station set differs from Stage225"
        )
    if not successful["audit_required_columns_ok"].fillna(False).astype(
        bool
    ).all():
        raise RuntimeError("Stage225 archive column audit did not pass")

    station_lookup = station_table.set_index("station_id").to_dict("index")
    frames: list[pd.DataFrame] = []
    archive_rows: list[dict[str, object]] = []
    for row in successful.sort_values("station_id").to_dict(
        orient="records"
    ):
        station_id = str(row["station_id"]).strip()
        archive_path = Path(str(row["target"]))
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        raw = read_station_archive(archive_path)
        raw_station = raw["STN"].astype(str).str.strip()
        if set(raw_station) != {station_id}:
            raise RuntimeError(
                f"archive {archive_path.name} contains station IDs "
                f"{sorted(raw_station.unique().tolist())}"
            )
        date = pd.to_datetime(
            raw["YYYYMMDD"].astype(str).str.strip(),
            format="%Y%m%d",
            errors="coerce",
            utc=True,
        )
        hour = pd.to_numeric(
            raw["HH"].astype(str).str.strip(), errors="coerce"
        )
        timestamp = date + pd.to_timedelta(hour - 1, unit="h")
        temperature = (
            pd.to_numeric(raw["T"], errors="coerce") / 10.0
        )
        humidity = pd.to_numeric(raw["U"], errors="coerce")
        dew_point = (
            pd.to_numeric(raw["TD"], errors="coerce") / 10.0
        )
        pressure = (
            pd.to_numeric(raw["P"], errors="coerce") / 10.0
        )
        station = station_lookup[station_id]
        normalized = pd.DataFrame(
            {
                "source": "KNMI_hourly",
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
        raw_invalid_timestamp = int(
            normalized["timestamp_utc"].isna().sum()
        )
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
        normalized["source"] = normalized["source"].fillna("KNMI_hourly")
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
                "raw_invalid_timestamp": raw_invalid_timestamp,
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
        ["station_id", "timestamp_utc"], kind="stable"
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
    frame: pd.DataFrame, archives: pd.DataFrame
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
            "阶段226只标准化阶段225下载的12个KNMI官方站点小时归档；"
            "2021-2023为train，2024为valid，2025为test；"
            "HH按官方文件头解释为小时区间，时间戳取区间起点；"
            "T/TD/P除以10，U保持百分比；空字段转缺失并保留；"
            "不插补、不训练模型、不选择阈值、不计算性能；"
            "所有攻击仍须由后续阶段按冻结协议透明注入。"
        ),
        "provider": "Royal Netherlands Meteorological Institute (KNMI)",
        "dataset": "KNMI hourly meteorological observations",
        "source": "KNMI_hourly",
        "primary_variables": ["T", "U"],
        "auxiliary_variables": ["TD", "P"],
        "timezone": "UTC",
        "timezone_evidence": (
            "KNMI hourly files define HH as UT (universal time); the "
            "hourly division convention is documented in the official "
            "hourly-data file header."
        ),
        "missing_rule": (
            "Empty KNMI fields are converted to missing values; no "
            "sentinel substitution and no imputation."
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
            "KNMI小时序列可能因站址迁移和观测方式变化而不均质，不能用于趋势分析。",
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
        "# 阶段226 KNMI标准化数据审计",
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
        "\n".join(lines) + "\n", encoding="utf-8"
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
