from __future__ import annotations

"""Stage 206: normalize the pre-registered DWD archives.

This stage reads only the 12 Stage-204 stations and the primary ``TU``
archives downloaded in Stage 205.  It produces one hourly table with
UTC timestamps, temperature, relative humidity, the original DWD
quality code, station metadata, and the frozen train/valid/test split.

No detector is fitted, no threshold is selected, and no performance is
computed.  Missing codes are converted to missing values and retained;
they are not imputed.  The split follows the Stage-204 contract:
2021-2023 train, 2024 valid, 2025 test.
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
DATA_DIR = WORKSPACE_ROOT / "dataset" / "external" / "dwd" / "TU"
PROTOCOL = TABLES / "stage204_dwd_protocol_preregistration.json"
STATIONS = TABLES / "stage204_dwd_protocol_preregistration_stations.csv"
DOWNLOAD_DETAIL = TABLES / "stage205_dwd_official_download_detail.csv"
PREFIX = "stage206_dwd_normalized"

MISSING_CODE = 9990.0
ACCEPTED_QUALITY_MIN = 3
YEAR_START = 2021
YEAR_END = 2025
EXPECTED_ROWS_PER_STATION = 5 * 365 * 24 + 24


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_product(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().startswith("produkt_")
            and name.lower().endswith(".txt")
        ]
        if len(members) != 1:
            raise RuntimeError(
                f"expected one product member in {path.name}, found {members}"
            )
        frame = pd.read_csv(
            archive.open(members[0]),
            sep=";",
            dtype=str,
            encoding="latin1",
        )
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {
        "STATIONS_ID",
        "MESS_DATUM",
        "QN_9",
        "TT_TU",
        "RF_TU",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(
            f"{path.name} missing columns: {sorted(required - set(frame.columns))}"
        )
    return frame


def normalized_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    station_table = pd.read_csv(STATIONS, dtype={"station_id": str})
    station_table["station_id"] = station_table["station_id"].str.zfill(5)
    download = pd.read_csv(DOWNLOAD_DETAIL, dtype={"station_id": str})
    download["station_id"] = download["station_id"].str.zfill(5)
    primary = download[
        download["role"].eq("primary")
        & download["product"].eq("TU")
        & download["status"].isin(["downloaded", "already_present"])
    ].copy()
    if len(primary) != len(station_table):
        raise RuntimeError(
            f"expected {len(station_table)} primary archives, found {len(primary)}"
        )
    expected_stations = set(station_table["station_id"])
    if set(primary["station_id"]) != expected_stations:
        raise RuntimeError("primary archive station set differs from Stage204")

    station_lookup = station_table.set_index("station_id").to_dict("index")
    frames: list[pd.DataFrame] = []
    archive_rows: list[dict[str, object]] = []
    for row in primary.sort_values("station_id").to_dict(orient="records"):
        station_id = str(row["station_id"]).zfill(5)
        archive_path = Path(str(row["target"]))
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        raw = read_product(archive_path)
        raw["station_id"] = (
            raw["STATIONS_ID"].astype(str).str.strip().str.zfill(5)
        )
        if set(raw["station_id"]) != {station_id}:
            raise RuntimeError(
                f"archive {archive_path.name} contains station IDs "
                f"{sorted(raw['station_id'].unique().tolist())}"
            )
        timestamp = pd.to_datetime(
            raw["MESS_DATUM"].astype(str).str.strip(),
            format="%Y%m%d%H",
            errors="coerce",
            utc=True,
        )
        temperature = pd.to_numeric(raw["TT_TU"], errors="coerce")
        humidity = pd.to_numeric(raw["RF_TU"], errors="coerce")
        quality = pd.to_numeric(raw["QN_9"], errors="coerce")
        temperature = temperature.mask(temperature.eq(MISSING_CODE))
        humidity = humidity.mask(humidity.eq(MISSING_CODE))
        station = station_lookup[station_id]
        normalized = pd.DataFrame(
            {
                "source": "DWD_CDC",
                "station_id": station_id,
                "station_name": station["station_name"],
                "timestamp_utc": timestamp,
                "latitude": float(station["latitude"]),
                "longitude": float(station["longitude"]),
                "elevation_m": float(station["elevation_m"]),
                "temperature_c": temperature,
                "humidity_pct": humidity,
                "temperature_quality": quality.astype("Int64"),
                "qn_9": quality.astype("Int64"),
                "quality_accepted": quality.ge(ACCEPTED_QUALITY_MIN),
            }
        )
        normalized["quality_accepted"] = (
            normalized["quality_accepted"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        raw_missing = int(
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
            | normalized["qn_9"].notna()
        )
        normalized["source"] = normalized["source"].fillna("DWD_CDC")
        normalized["station_id"] = normalized["station_id"].fillna(
            station_id
        )
        normalized["station_name"] = normalized["station_name"].fillna(
            station["station_name"]
        )
        for column in ["latitude", "longitude", "elevation_m"]:
            normalized[column] = normalized[column].fillna(
                float(station[column])
            )
        normalized["quality_accepted"] = (
            normalized["quality_accepted"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
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
                    None if pd.isna(raw_timestamp_min) else str(raw_timestamp_min)
                ),
                "raw_timestamp_max": (
                    None if pd.isna(raw_timestamp_max) else str(raw_timestamp_max)
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
        raise RuntimeError("duplicate station-timestamp rows after normalization")
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
                    group.duplicated(["station_id", "timestamp_utc"]).sum()
                ),
                "missing_observation_rows": int(
                    (~group["observation_present"]).sum()
                ),
                "temperature_missing": int(group["temperature_c"].isna().sum()),
                "humidity_missing": int(group["humidity_pct"].isna().sum()),
                "quality_accepted": int(group["quality_accepted"].sum()),
                "quality_rejected_observed": int(
                    (
                        group["observation_present"]
                        & ~group["quality_accepted"]
                    ).sum()
                ),
                "quality_missing": int(group["qn_9"].isna().sum()),
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
            "阶段206只标准化阶段204冻结的12个DWD站点和阶段205下载的"
            "TU主产品；2021-2023为train，2024为valid，2025为test；"
            "9990.0转为缺失，保留QN_9和质量接受标记；不插补、不训练模型、"
            "不选择阈值、不计算性能；所有攻击仍须由后续阶段按冻结协议透明注入。"
        ),
        "provider": "Deutscher Wetterdienst (DWD)",
        "dataset_version": "v24.03",
        "source": "DWD_CDC",
        "primary_product": "TU",
        "primary_variables": ["TT_TU", "RF_TU"],
        "quality_field": "QN_9",
        "quality_accepted_minimum": ACCEPTED_QUALITY_MIN,
        "missing_code": MISSING_CODE,
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
        "temperature_missing_rows": int(frame["temperature_c"].isna().sum()),
        "humidity_missing_rows": int(frame["humidity_pct"].isna().sum()),
        "quality_accepted_rows": int(frame["quality_accepted"].sum()),
        "quality_rejected_observed_rows": int(
            (
                frame["observation_present"]
                & ~frame["quality_accepted"]
            ).sum()
        ),
        "quality_missing_rows": int(frame["qn_9"].isna().sum()),
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
            "DWD质量码不是攻击标签；质量拒绝行不等于攻击行。",
            "缺失值和低质量行被保留为缺失/拒绝标记，没有插补。",
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
        "# 阶段206 DWD标准化数据审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 来源：`{payload['source']}`；产品：`{payload['primary_product']}`",
        f"- 站点：`{payload['station_count']}`；年份：`{YEAR_START}-{YEAR_END}`",
        f"- 行数：`{payload['rows']}`；预期逐小时行数：`{payload['expected_rows']}`",
        f"- train/valid/test：`{payload['train_years']}` / `{payload['valid_year']}` / `{payload['test_year']}`",
        f"- 质量接受：`{payload['quality_accepted_rows']}`；观测行质量拒绝：`{payload['quality_rejected_observed_rows']}`；质量缺失：`{payload['quality_missing_rows']}`",
        f"- 温度缺失：`{payload['temperature_missing_rows']}`；湿度缺失：`{payload['humidity_missing_rows']}`",
        f"- 站点-时间重复：`{payload['duplicate_station_timestamp_rows']}`",
        f"- 显式缺测行：`{payload['missing_observation_rows']}`；原始归档缺口：`{payload['raw_missing_timestamps']}`",
        f"- 性能状态：`{payload['performance_status']}`",
        "",
        "| 站点 | 行数 | 缺失时刻 | 缺测行 | 额外时刻 | 重复 | 温度缺失 | 湿度缺失 | 质量接受 | 质量拒绝(观测) | 质量缺失 | train | valid | test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in station_audit.to_dict(orient="records"):
        lines.append(
            f"| {row['station_id']} | {row['rows']} | "
            f"{row['missing_timestamps']} | {row['missing_observation_rows']} | "
            f"{row['extra_timestamps']} | "
            f"{row['duplicate_rows']} | {row['temperature_missing']} | "
            f"{row['humidity_missing']} | {row['quality_accepted']} | "
            f"{row['quality_rejected_observed']} | "
            f"{row['quality_missing']} | "
            f"{row['train_rows']} | {row['valid_rows']} | "
            f"{row['test_rows']} |"
        )
    lines += [
        "",
        "本阶段不插补、不训练模型、不选择阈值、不计算性能；质量码不是攻击标签。",
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
                "duplicates": payload["duplicate_station_timestamp_rows"],
                "quality_accepted_rows": payload["quality_accepted_rows"],
                "quality_rejected_observed_rows": payload[
                    "quality_rejected_observed_rows"
                ],
                "quality_missing_rows": payload["quality_missing_rows"],
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
