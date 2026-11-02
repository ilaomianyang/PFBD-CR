from __future__ import annotations

"""Stage 204: pre-register the first cross-provider external validation.

The Stage-203 audit found no source in the repository that is independent
at the provider/sensor-network level.  This stage freezes the protocol
before any DWD station observations are downloaded or inspected.  It only
uses official DWD metadata (station inventory and dataset description) to
define the eligible station pool and the station-selection rule.

No detector is trained, no threshold is selected, and no performance
result is computed here.  DWD observations and all transparent controlled
attacks are handled by later stages under this frozen contract.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage204_dwd_protocol_preregistration"

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
STATION_METADATA_CACHE = (
    WORKSPACE_ROOT
    / "dataset"
    / "external"
    / "dwd"
    / "TU_Stundenwerte_Beschreibung_Stationen.txt"
)
STATION_METADATA_FALLBACK = Path(
    r"C:\Users\ilaom\AppData\Local\Temp\dwd_tu_stations.txt"
)
STATION_METADATA = (
    STATION_METADATA_CACHE
    if STATION_METADATA_CACHE.is_file()
    else STATION_METADATA_FALLBACK
)
STATION_COLUMNS = [
    "station_id",
    "start",
    "end",
    "elevation_m",
    "latitude",
    "longitude",
    "station_name",
    "state",
    "status",
]

PROVIDER = "Deutscher Wetterdienst (DWD)"
DATASET = (
    "Hourly station observations of 2 m air temperature and humidity "
    "for Germany"
)
DATASET_VERSION = "v24.03"
DATASET_ID = (
    "urn:x-wmo:md:de.dwd.cdc::obsgermany-climate-hourly-air_temperature"
)
OFFICIAL_ROOT = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/hourly/air_temperature"
)
HISTORICAL_ROOT = f"{OFFICIAL_ROOT}/historical"
RECENT_ROOT = f"{OFFICIAL_ROOT}/recent"
STATION_METADATA_URL = f"{RECENT_ROOT}/TU_Stundenwerte_Beschreibung_Stationen.txt"
DESCRIPTION_URL = (
    f"{OFFICIAL_ROOT}/"
    "DESCRIPTION_obsgermany_climate_hourly_air_temperature_en.pdf"
)
LICENSE_NOTE = (
    "DWD CDC OpenData; the dataset description must be cited; "
    "raw files, URLs, download timestamps, and SHA256 values are retained."
)

YEARS = [2021, 2022, 2023, 2024, 2025]
VALID_YEAR = 2024
TEST_YEAR = 2025
STATION_STATE = "Bayern"
STATION_GRID_DEGREES = 0.75
STATION_COUNT = 12
QUALITY_LEVEL_MIN = 3

VARIABLES = {
    "TT_TU": {
        "description": "air temperature at 2 m",
        "unit": "degC",
        "missing_code": 9990.0,
        "product_code": "TU",
        "historical_product": "stundenwerte_TU",
    },
    "TF_TU": {
        "description": "relative humidity",
        "unit": "%",
        "missing_code": 9990.0,
        "product_code": "TF",
        "historical_product": "stundenwerte_TF",
    },
}
QUALITY_CODE = {
    "field": "QN_9",
    "accepted_minimum": QUALITY_LEVEL_MIN,
    "rule": (
        "QN_9 >= 3 is required; QN=1/2 and missing values are excluded "
        "from model fitting, threshold calibration, and evaluation."
    ),
}

ATTACK_TYPES = [
    "spike",
    "offset",
    "drift",
    "time_shift",
    "stuck",
    "gaussian_noise",
]
ATTACK_LENGTH = {
    "spike": 8,
    "offset": 8,
    "drift": 12,
    "time_shift": 8,
    "stuck": 8,
    "gaussian_noise": 8,
}
ATTACK_BLOCKS = 6
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FAR_GRID = [0.005, 0.01, 0.02, 0.05]
PRIMARY_FAR = 0.01
PRIMARY_ENDPOINT = "event_coverage"
SECONDARY_ENDPOINT = "row_recall"
CLUSTER_UNIT = "station"
MULTIPLICITY = "Benjamini-Hochberg over attack types within the primary FAR"


def load_metadata() -> pd.DataFrame:
    if not STATION_METADATA.is_file():
        raise FileNotFoundError(
            f"DWD station metadata not found: {STATION_METADATA}"
        )
    frame = pd.read_fwf(
        STATION_METADATA,
        skiprows=2,
        names=STATION_COLUMNS,
        dtype=str,
        encoding="latin1",
    ).dropna(subset=["station_id"])
    for column in ["start", "end", "state", "status"]:
        frame[column] = frame[column].astype(str).str.strip()
    for column in ["elevation_m", "latitude", "longitude"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def eligible_station_pool(metadata: pd.DataFrame) -> pd.DataFrame:
    eligible = metadata[
        metadata["state"].eq(STATION_STATE)
        & metadata["start"].le("20210101")
        & metadata["end"].ge("20251231")
    ].copy()
    if eligible.empty:
        raise RuntimeError("no eligible DWD stations")
    eligible["latitude_bin"] = (
        (eligible["latitude"] - 47.0) // STATION_GRID_DEGREES
    ).astype(int)
    eligible["longitude_bin"] = (
        (eligible["longitude"] - 9.0) // STATION_GRID_DEGREES
    ).astype(int)
    eligible["spatial_cell"] = (
        eligible["latitude_bin"].astype(str)
        + "_"
        + eligible["longitude_bin"].astype(str)
    )
    return eligible


def select_stations(eligible: pd.DataFrame) -> pd.DataFrame:
    first_per_cell = (
        eligible.sort_values(["spatial_cell", "station_id"])
        .drop_duplicates("spatial_cell", keep="first")
        .sort_values("station_id")
    )
    selected = first_per_cell.head(STATION_COUNT).copy()
    if len(selected) < STATION_COUNT:
        remaining = eligible.loc[
            ~eligible["station_id"].isin(selected["station_id"])
        ].sort_values("station_id")
        selected = pd.concat([selected, remaining], ignore_index=True)
        selected = selected.drop_duplicates("station_id").head(STATION_COUNT)
    return selected.reset_index(drop=True)


def build_outputs() -> int:
    metadata = load_metadata()
    eligible = eligible_station_pool(metadata)
    selected = select_stations(eligible)
    station_ids = selected["station_id"].astype(str).tolist()

    station_table = selected[
        [
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "elevation_m",
            "state",
            "start",
            "end",
            "spatial_cell",
        ]
    ].copy()
    station_table["selection_rank"] = range(1, len(station_table) + 1)
    station_table.to_csv(
        TABLES / f"{PREFIX}_stations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "status": "PROTOCOL_FROZEN_BEFORE_DOWNLOAD",
        "protocol": (
            "阶段204在下载前冻结跨机构独立外部验证协议，不下载任何DWD观测："
            "DWD与NOAA ISD、六盘山本地观测属于不同机构和传感器网络；"
            "固定站点选择、年份、变量、质量码、透明受控攻击、种子、"
            "FAR、主终点、聚类单位和多重校正。本阶段不训练模型、不选择"
            "阈值、不下载观测、不计算性能结果；test只用于后续评价，"
            "不根据test选择任何组件；攻击不是真实攻击日志。"
        ),
        "provider": PROVIDER,
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "dataset_id": DATASET_ID,
        "official_root": OFFICIAL_ROOT,
        "historical_root": HISTORICAL_ROOT,
        "recent_root": RECENT_ROOT,
        "station_metadata_url": STATION_METADATA_URL,
        "description_url": DESCRIPTION_URL,
        "license_note": LICENSE_NOTE,
        "independent_source": True,
        "source_independence": (
            "DWD is a different provider and sensor network from NOAA ISD "
            "and the Liupanshan local observations."
        ),
        "years": YEARS,
        "valid_year": VALID_YEAR,
        "test_year": TEST_YEAR,
        "split": (
            "Train on 2021-2023 normal observations; calibrate on 2024 "
            "valid normal observations and transparent injected events; "
            "evaluate on 2025 test observations and transparent injected "
            "events."
        ),
        "station_state": STATION_STATE,
        "station_grid_degrees": STATION_GRID_DEGREES,
        "eligible_station_count": int(len(eligible)),
        "station_count": STATION_COUNT,
        "station_selection_rule": (
            "Eligible stations have state=Bayern, start<=20210101, "
            "end>=20251231. Sort by spatial_cell and station_id, keep the "
            "first station per spatial cell, then take the first 12 station "
            "IDs. If fewer than 12 cells exist, fill by station_id order."
        ),
        "station_ids": station_ids,
        "variables": VARIABLES,
        "quality_code": QUALITY_CODE,
        "attack_types": ATTACK_TYPES,
        "attack_length": ATTACK_LENGTH,
        "attack_blocks": ATTACK_BLOCKS,
        "seeds": SEEDS,
        "far_grid": FAR_GRID,
        "primary_far": PRIMARY_FAR,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoint": SECONDARY_ENDPOINT,
        "cluster_unit": CLUSTER_UNIT,
        "multiplicity": MULTIPLICITY,
        "selection_boundary": (
            "The detector, features, fusion weights, protection rule, "
            "threshold, attack type, station, and year are frozen before "
            "the test evaluation. Test labels are used only for evaluation. "
            "No target-domain test tuning is permitted."
        ),
        "attack_boundary": (
            "All attacks are transparent controlled injections generated "
            "by the frozen protocol. They are not real attack logs."
        ),
        "quality_boundary": (
            "QN_9 is a DWD quality level, not an attack label. Low-quality "
            "rows are excluded rather than treated as positive events."
        ),
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段204 DWD 跨机构外部验证预注册",
        "",
        f"- 状态：`{protocol['status']}`",
        f"- 来源：{PROVIDER}",
        f"- 数据集：{DATASET}，版本 `{DATASET_VERSION}`",
        f"- 数据集标识：`{DATASET_ID}`",
        f"- 官方入口：{OFFICIAL_ROOT}",
        f"- 站点：{STATION_STATE} 区域，`{len(station_ids)}` 站，规则在下载前冻结",
        f"- 年份：`{YEARS}`；valid `{VALID_YEAR}`；test `{TEST_YEAR}`",
        f"- 主运行点：FAR `{PRIMARY_FAR}`；主终点：`{PRIMARY_ENDPOINT}`；聚类单位：`{CLUSTER_UNIT}`",
        f"- 攻击：`{', '.join(ATTACK_TYPES)}`，透明受控注入，不是真实攻击日志",
        f"- 下载状态：`{protocol['download_status']}`；性能状态：`{protocol['performance_status']}`",
        "",
        "## 站点选择规则",
        "",
        protocol["station_selection_rule"],
        "",
        "## 站点清单",
        "",
        "| 排名 | 站号 | 站名 | 纬度 | 经度 | 海拔(m) | 空间单元 |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for row in station_table.to_dict(orient="records"):
        lines.append(
            f"| {int(row['selection_rank'])} | {row['station_id']} | "
            f"{row['station_name']} | {float(row['latitude']):.4f} | "
            f"{float(row['longitude']):.4f} | "
            f"{float(row['elevation_m']):.0f} | {row['spatial_cell']} |"
        )
    lines += [
        "",
        "## 边界",
        "",
        protocol["selection_boundary"],
        "",
        protocol["attack_boundary"],
        "",
        protocol["quality_boundary"],
        "",
        "本阶段只冻结协议和站点，不下载观测数据、不训练模型、不计算性能结果。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "eligible_stations": int(len(eligible)),
                "selected_stations": int(len(station_ids)),
                "station_ids": station_ids,
                "download_status": protocol["download_status"],
                "performance_status": protocol["performance_status"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-existing", action="store_true")
    parser.parse_args(argv)
    return build_outputs()


if __name__ == "__main__":
    raise SystemExit(main())
