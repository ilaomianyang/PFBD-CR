"""Stage 234: pre-register the fourth cross-provider validation source.

The DWD evidence chain and the MeteoSwiss, NOAA ISD, and KNMI transfer
targets are already frozen.  This stage freezes a fourth independent
national surface-station source, Environment and Climate Change Canada
(ECCC), before any hourly observation values are downloaded or used for
model selection or performance.

Only the official ECCC GeoMet station catalogue is read here: station
identifiers, coordinates, names, and hourly coverage metadata.  No hourly
observation values are downloaded, no detector is trained, no threshold
is selected, and no performance result is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage234_eccc_protocol_preregistration"

PROVIDER = (
    "Environment and Climate Change Canada (ECCC) / "
    "Meteorological Service of Canada (MSC)"
)
DATASET = "ECCC/MSC Climate - Hourly Observations"
COLLECTION_URL = "https://api.weather.gc.ca/collections/climate-hourly"
STATION_COLLECTION_URL = (
    "https://api.weather.gc.ca/collections/climate-stations"
)
ITEMS_URL = f"{COLLECTION_URL}/items"
STATION_ITEMS_URL = f"{STATION_COLLECTION_URL}/items"
LICENSE = "Open Government Licence - Canada"
LICENSE_URL = (
    "https://open.canada.ca/en/open-government-licence-canada"
)
USER_AGENT = "TrustMeteo-QC/0.1 (reproducible research; ECCC Open Data)"

STATION_COUNT = 16
MIN_RETAINED_STATIONS = 8
STATION_GRID_DEGREES = 0.75
GRID_ORIGIN_LATITUDE = 42.0
GRID_ORIGIN_LONGITUDE = -142.0
MIN_PRIMARY_COVERAGE = 0.70

YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_YEARS = [2021, 2022, 2023]
VALID_YEAR = 2024
TEST_YEAR = 2025

PRIMARY_VARIABLES = {
    "temperature_c": {
        "source_field": "TEMP",
        "description": "air temperature",
        "unit": "degC",
    },
    "humidity_pct": {
        "source_field": "RELATIVE_HUMIDITY",
        "description": "relative humidity",
        "unit": "%",
    },
}
AUXILIARY_VARIABLES = {
    "dew_point_c": {
        "source_field": "DEW_POINT_TEMP",
        "unit": "degC",
    },
    "pressure_hpa": {
        "source_field": "STATION_PRESSURE",
        "unit": "kPa",
    },
}
MISSING_RULE = (
    "Empty or non-numeric ECCC fields are converted to missing values; "
    "no sentinel substitution and no imputation are allowed."
)
QUALITY_BOUNDARY = (
    "ECCC flag fields are preserved in the download audit when present; "
    "they are metadata, not attack labels. No undocumented quality-code "
    "exclusion is applied."
)

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


def fetch_json(url: str, timeout: int = 120) -> dict[str, object]:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def station_features() -> list[dict[str, object]]:
    payload = fetch_json(f"{STATION_ITEMS_URL}?f=json&limit=10000")
    features = payload.get("features", [])
    if not isinstance(features, list) or not features:
        raise RuntimeError("ECCC station catalogue returned no features")
    return features


def parse_station_frame(features: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [None, None])
        if not isinstance(properties, dict):
            continue
        rows.append(
            {
                "station_id": properties.get("STN_ID"),
                "climate_identifier": properties.get("CLIMATE_IDENTIFIER"),
                "station_name": properties.get("STATION_NAME"),
                "province_code": properties.get("PROV_STATE_TERR_CODE"),
                "province_name": properties.get("ENG_PROV_NAME"),
                "country": properties.get("COUNTRY"),
                "latitude": (
                    float(coordinates[1])
                    if coordinates and coordinates[1] is not None
                    else float("nan")
                ),
                "longitude": (
                    float(coordinates[0])
                    if coordinates and coordinates[0] is not None
                    else float("nan")
                ),
                "elevation_m": properties.get("ELEVATION"),
                "timezone": properties.get("TIMEZONE"),
                "hly_first_date": properties.get("HLY_FIRST_DATE"),
                "hly_last_date": properties.get("HLY_LAST_DATE"),
                "has_hourly_data": properties.get("HAS_HOURLY_DATA"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["station_id"] = pd.to_numeric(
        frame["station_id"], errors="coerce"
    )
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["hly_first_date"] = pd.to_datetime(
        frame["hly_first_date"], errors="coerce"
    )
    frame["hly_last_date"] = pd.to_datetime(
        frame["hly_last_date"], errors="coerce"
    )
    frame = frame.dropna(
        subset=[
            "station_id",
            "climate_identifier",
            "station_name",
            "province_code",
            "latitude",
            "longitude",
            "hly_first_date",
            "hly_last_date",
        ]
    ).copy()
    frame["station_id"] = frame["station_id"].astype(int)
    frame = frame.drop_duplicates("station_id").sort_values(
        "station_id"
    ).reset_index(drop=True)
    return frame


def eligible_station_pool(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[
        frame["country"].eq("CAN")
        & frame["hly_first_date"].le(pd.Timestamp("2021-01-01"))
        & frame["hly_last_date"].ge(pd.Timestamp("2025-12-31 23:59:59"))
    ].copy()
    if eligible.empty:
        raise RuntimeError("no eligible ECCC hourly stations")
    eligible["latitude_bin"] = (
        (eligible["latitude"] - GRID_ORIGIN_LATITUDE)
        // STATION_GRID_DEGREES
    ).map(math.floor)
    eligible["longitude_bin"] = (
        (eligible["longitude"] - GRID_ORIGIN_LONGITUDE)
        // STATION_GRID_DEGREES
    ).map(math.floor)
    eligible["spatial_cell"] = (
        eligible["latitude_bin"].astype(int).astype(str)
        + "_"
        + eligible["longitude_bin"].astype(int).astype(str)
    )
    return eligible


def select_stations(eligible: pd.DataFrame) -> pd.DataFrame:
    anchors = (
        eligible.sort_values(
            ["province_code", "station_id", "climate_identifier"]
        )
        .drop_duplicates("province_code", keep="first")
        .sort_values(["province_code", "station_id"])
        .copy()
    )
    anchors["selection_stage"] = "province_or_territory_anchor"

    if len(anchors) > STATION_COUNT:
        anchors = anchors.head(STATION_COUNT).copy()

    remaining = eligible.loc[
        ~eligible["station_id"].isin(anchors["station_id"])
    ].copy()
    first_per_cell = (
        remaining.sort_values(["spatial_cell", "station_id"])
        .drop_duplicates("spatial_cell", keep="first")
        .sort_values(["spatial_cell", "station_id"])
    )
    fill_count = max(0, STATION_COUNT - len(anchors))
    fill = first_per_cell.head(fill_count).copy()
    fill["selection_stage"] = "spatial_cell_fill"
    selected = pd.concat([anchors, fill], ignore_index=True)
    if len(selected) < STATION_COUNT:
        extra = (
            remaining.loc[
                ~remaining["station_id"].isin(selected["station_id"])
            ]
            .sort_values(["spatial_cell", "station_id"])
            .head(STATION_COUNT - len(selected))
            .copy()
        )
        extra["selection_stage"] = "station_id_fallback"
        selected = pd.concat([selected, extra], ignore_index=True)
    selected = (
        selected.drop_duplicates("station_id")
        .sort_values(["selection_stage", "station_id"])
        .reset_index(drop=True)
    )
    selected["selection_rank"] = range(1, len(selected) + 1)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-limit", type=int, default=0)
    args = parser.parse_args(argv)

    features = station_features()
    all_stations = parse_station_frame(features)
    if args.station_limit:
        all_stations = all_stations.head(args.station_limit).copy()
    eligible = eligible_station_pool(all_stations)
    selected = select_stations(eligible)
    station_ids = selected["station_id"].astype(int).tolist()
    if len(station_ids) != STATION_COUNT:
        raise RuntimeError(
            f"expected {STATION_COUNT} ECCC stations, found {len(station_ids)}"
        )

    station_output = selected[
        [
            "station_id",
            "climate_identifier",
            "station_name",
            "province_code",
            "province_name",
            "latitude",
            "longitude",
            "elevation_m",
            "timezone",
            "hly_first_date",
            "hly_last_date",
            "spatial_cell",
            "selection_stage",
            "selection_rank",
        ]
    ].copy()
    station_output.to_csv(
        TABLES / f"{PREFIX}_stations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "status": "PROTOCOL_FROZEN_BEFORE_ECCC_PERFORMANCE",
        "protocol": (
            "阶段234在任何ECCC小时观测值用于模型选择或性能评价之前冻结"
            "第四跨机构现场站点协议。来源为ECCC/MSC官方GeoMet"
            "Climate - Hourly Observations；年份固定为2021-2025，"
            "2021-2023为train、2024为valid、2025为test。站点只从"
            "官方站点集合中按HLY_FIRST_DATE<=2021-01-01且"
            "HLY_LAST_DATE>=2025-12-31筛选，先为每个省/地区取一个"
            "站号最小的锚点站，再按0.75度空间单元和站号补齐至16站；"
            "主变量为TEMP和RELATIVE_HUMIDITY，辅助变量为"
            "DEW_POINT_TEMP和STATION_PRESSURE；空或非数值字段为缺失，"
            "不插补。主FAR=0.01，主终点为event_coverage，次终点为"
            "row_recall，聚类单位为station。攻击类型、块数、长度和"
            "种子与DWD/MeteoSwiss/NOAA/KNMI协议一致。test只评价，"
            "不根据test选择特征、窗口、权重、阈值、站点或攻击位置。"
            "所有攻击都是透明受控注入，不是真实攻击日志。"
        ),
        "provider": PROVIDER,
        "dataset": DATASET,
        "collection_url": COLLECTION_URL,
        "station_collection_url": STATION_COLLECTION_URL,
        "items_url": ITEMS_URL,
        "station_items_url": STATION_ITEMS_URL,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "station_count": STATION_COUNT,
        "min_retained_stations": MIN_RETAINED_STATIONS,
        "station_grid_degrees": STATION_GRID_DEGREES,
        "grid_origin_latitude": GRID_ORIGIN_LATITUDE,
        "grid_origin_longitude": GRID_ORIGIN_LONGITUDE,
        "station_selection_rule": (
            "Eligible stations require COUNTRY=CAN, HLY_FIRST_DATE <= "
            "2021-01-01, and HLY_LAST_DATE >= 2025-12-31 23:59:59. "
            "First select the lowest STN_ID in each province/territory. "
            "Then fill remaining slots from 0.75-degree spatial cells, "
            "sorted by spatial_cell and STN_ID, until 16 stations are "
            "selected. Selection never uses observation values or "
            "performance."
        ),
        "station_ids": station_ids,
        "station_selection_stages": (
            selected["selection_stage"].value_counts().to_dict()
        ),
        "eligible_station_count": int(len(eligible)),
        "all_station_count": int(len(all_stations)),
        "years": YEARS,
        "split": {
            "train": TRAIN_YEARS,
            "valid": [VALID_YEAR],
            "test": [TEST_YEAR],
        },
        "primary_variables": PRIMARY_VARIABLES,
        "auxiliary_variables": AUXILIARY_VARIABLES,
        "missing_rule": MISSING_RULE,
        "quality_boundary": QUALITY_BOUNDARY,
        "timezone": "UTC",
        "timezone_evidence": (
            "The official ECCC GeoMet climate-hourly schema exposes "
            "UTC_DATE, UTC_YEAR, UTC_MONTH, and UTC_DAY alongside "
            "LOCAL_DATE. Normalization uses UTC_DATE."
        ),
        "minimum_primary_coverage": MIN_PRIMARY_COVERAGE,
        "coverage_rule": (
            "After download, a station is retained only if both TEMP and "
            "RELATIVE_HUMIDITY are present for at least 70% of the hourly "
            "grid in each train, valid, and test split. This is a frozen "
            "data-availability rule, not a performance rule."
        ),
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
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "这是下载前预注册，不是性能结果。",
            "站点选择只依据官方站点元数据和预声明规则，不依据test性能。",
            "ECCC小时序列可能因站址迁移、仪器变化和报告规则变化而不均质。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "provider": PROVIDER,
                "dataset": DATASET,
                "collection_url": COLLECTION_URL,
                "license": LICENSE,
                "license_url": LICENSE_URL,
                "station_count": STATION_COUNT,
                "eligible_station_count": int(len(eligible)),
                "years": ",".join(str(value) for value in YEARS),
                "primary_far": PRIMARY_FAR,
                "primary_endpoint": PRIMARY_ENDPOINT,
                "minimum_primary_coverage": MIN_PRIMARY_COVERAGE,
            }
        ]
    ).to_csv(
        TABLES / f"{PREFIX}_contract.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段234 ECCC第四独立来源预注册",
        "",
        f"- 状态：**{protocol['status']}**",
        f"- 来源：`{PROVIDER}`",
        f"- 数据集：`{DATASET}`",
        f"- 官方集合：{COLLECTION_URL}",
        f"- 许可：`{LICENSE}`；{LICENSE_URL}",
        f"- 官方站点总数：`{len(all_stations)}`；冻结站点：`{len(selected)}`",
        f"- 年份：`{YEARS}`；切分：`2021-2023/2024/2025`",
        f"- 主FAR：`{PRIMARY_FAR}`；主终点：`{PRIMARY_ENDPOINT}`；聚类：`{CLUSTER_UNIT}`",
        f"- 最小主变量分段覆盖率：`{MIN_PRIMARY_COVERAGE}`",
        "",
        "## 冻结边界",
        "",
        "- 站点只按官方站点元数据和预声明规则选择，不按任何test性能选择。",
        "- 主变量TEMP/RELATIVE_HUMIDITY，辅助变量DEW_POINT_TEMP/STATION_PRESSURE；空或非数值字段为缺失，不插补。",
        "- test只评价，不选择特征、窗口、权重、阈值、站点或攻击位置。",
        "- 所有攻击都是透明受控注入，不是真实攻击日志。",
        "",
        "## 站点清单",
        "",
        "| 排名 | 站号 | 气候标识 | 站名 | 省/地区 | 纬度 | 经度 | 选择阶段 |",
        "|---:|---:|---|---|---|---:|---:|---|",
    ]
    for row in station_output.to_dict(orient="records"):
        lines.append(
            f"| {int(row['selection_rank'])} | {int(row['station_id'])} | "
            f"{row['climate_identifier']} | {row['station_name']} | "
            f"{row['province_code']} | {float(row['latitude']):.4f} | "
            f"{float(row['longitude']):.4f} | {row['selection_stage']} |"
        )
    lines += [
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in protocol["limitations"]],
        "",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "provider": protocol["provider"],
                "all_station_count": len(all_stations),
                "eligible_station_count": int(len(eligible)),
                "station_count": len(selected),
                "station_ids": station_ids,
                "download_status": protocol["download_status"],
                "performance_status": protocol["performance_status"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
