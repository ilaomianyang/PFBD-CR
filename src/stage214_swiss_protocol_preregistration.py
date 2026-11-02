from __future__ import annotations

"""Stage 214: pre-register the second cross-provider external validation.

Stage 204--213 established a complete single-provider DWD external
evidence chain.  This stage freezes a second cross-provider source before
any SwissMetNet observation values are downloaded or inspected.

Only official MeteoSwiss metadata exposed through the Swiss federal STAC
API is read: station identifiers, coordinates, names, and asset URLs.
No observation CSV is downloaded, no detector is trained, no threshold is
selected, and no performance result is computed.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage214_swiss_protocol_preregistration"

PROVIDER = "MeteoSwiss (Federal Office of Meteorology and Climatology)"
DATASET = "SwissMetNet automatic weather station measurements"
DATASET_ID = "ch.meteoschweiz.ogd-smn"
STAC_COLLECTION = (
    "https://data.geo.admin.ch/api/stac/v0.9/collections/"
    "ch.meteoschweiz.ogd-smn"
)
STAC_ITEMS = f"{STAC_COLLECTION}/items"
LICENSE = "CC-BY"
USER_AGENT = "TrustMeteo-QC/0.1 (reproducible research; MeteoSwiss OpenData)"

STATION_COUNT = 12
STATION_GRID_DEGREES = 0.75
GRID_ORIGIN_LATITUDE = 45.5
GRID_ORIGIN_LONGITUDE = 5.5

YEARS = [2021, 2022, 2023, 2024, 2025]
VALID_YEAR = 2024
TEST_YEAR = 2025

PRIMARY_VARIABLES = {
    "temperature_c": {
        "source_field": "tre200h0",
        "description": "air temperature at 2 m",
        "unit": "degC",
    },
    "humidity_pct": {
        "source_field": "ure200h0",
        "description": "relative humidity at 2 m",
        "unit": "%",
    },
}
AUXILIARY_VARIABLES = {
    "dew_point_c": {"source_field": "tde200h0", "unit": "degC"},
    "pressure_hpa": {"source_field": "prestah0", "unit": "hPa"},
}
MISSING_RULE = (
    "Empty CSV fields are converted to missing values; no sentinel "
    "substitution and no imputation are allowed."
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


def fetch_json(url: str) -> dict[str, object]:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def fetch_station_items() -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    url: str | None = STAC_ITEMS
    while url:
        payload = fetch_json(url)
        items.extend(payload.get("features", []))
        next_link = next(
            (
                link
                for link in payload.get("links", [])
                if link.get("rel") == "next"
            ),
            None,
        )
        url = None if next_link is None else str(next_link["href"])
    if not items:
        raise RuntimeError("MeteoSwiss STAC returned no station items")
    return items


def station_table(items: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in items:
        station_id = str(item["id"]).lower()
        coordinates = item["geometry"]["coordinates"]
        title = str(item["properties"].get("title", station_id))
        station_name = title.rsplit("(", 1)[0].strip()
        rows.append(
            {
                "station_id": station_id,
                "station_name": station_name,
                "latitude": float(coordinates[1]),
                "longitude": float(coordinates[0]),
                "title": title,
            }
        )
    frame = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    if frame["station_id"].duplicated().any():
        raise RuntimeError("duplicate MeteoSwiss station identifiers")
    return frame


def select_stations(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.copy()
    selected["latitude_bin"] = (
        (selected["latitude"] - GRID_ORIGIN_LATITUDE)
        // STATION_GRID_DEGREES
    ).map(math.floor)
    selected["longitude_bin"] = (
        (selected["longitude"] - GRID_ORIGIN_LONGITUDE)
        // STATION_GRID_DEGREES
    ).map(math.floor)
    selected["spatial_cell"] = (
        selected["latitude_bin"].astype(str)
        + "_"
        + selected["longitude_bin"].astype(str)
    )
    first_per_cell = (
        selected.sort_values(["spatial_cell", "station_id"])
        .drop_duplicates("spatial_cell", keep="first")
        .sort_values("station_id")
    )
    chosen = first_per_cell.head(STATION_COUNT).copy()
    if len(chosen) < STATION_COUNT:
        remaining = selected.loc[
            ~selected["station_id"].isin(chosen["station_id"])
        ].sort_values("station_id")
        chosen = pd.concat([chosen, remaining], ignore_index=True)
        chosen = chosen.drop_duplicates("station_id").head(STATION_COUNT)
    chosen = chosen.reset_index(drop=True)
    chosen["selection_rank"] = range(1, len(chosen) + 1)
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-limit", type=int, default=0)
    args = parser.parse_args(argv)

    items = fetch_station_items()
    all_stations = station_table(items)
    if args.station_limit:
        all_stations = all_stations.head(args.station_limit).copy()
    selected = select_stations(all_stations)
    station_ids = selected["station_id"].tolist()

    station_output = selected[
        [
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "spatial_cell",
            "selection_rank",
        ]
    ].copy()
    station_output.to_csv(
        TABLES / f"{PREFIX}_stations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "status": "PROTOCOL_FROZEN_BEFORE_DOWNLOAD",
        "protocol": (
            "阶段214在下载任何MeteoSwiss观测前冻结第二个跨机构外部验证"
            "协议，只读取瑞士联邦官方STAC中的站点元数据和资产URL；"
            "MeteoSwiss SwissMetNet与DWD、NOAA ISD和六盘山本地观测属于"
            "不同机构和传感器网络；固定站点选择、年份、变量、缺失规则、"
            "透明受控攻击、种子、FAR、主终点、聚类单位和多重校正。"
            "本阶段不训练模型、不选择阈值、不下载观测CSV、不计算性能；"
            "test只用于后续评价，不根据test选择任何组件；"
            "攻击不是真实攻击日志。"
        ),
        "provider": PROVIDER,
        "dataset": DATASET,
        "dataset_id": DATASET_ID,
        "stac_collection": STAC_COLLECTION,
        "license": LICENSE,
        "independent_source": True,
        "source_independence": (
            "MeteoSwiss is a different provider and sensor network from "
            "DWD, NOAA ISD, and the Liupanshan local observations."
        ),
        "all_station_count": int(len(all_stations)),
        "station_count": int(len(selected)),
        "station_grid_degrees": STATION_GRID_DEGREES,
        "grid_origin_latitude": GRID_ORIGIN_LATITUDE,
        "grid_origin_longitude": GRID_ORIGIN_LONGITUDE,
        "station_selection_rule": (
            "Use the complete official STAC station list. Compute a 0.75 "
            "degree spatial cell from latitude-45.5 and longitude-5.5, "
            "sort by spatial_cell and station_id, keep the first station "
            "per cell, then take the first 12 station IDs. If fewer than "
            "12 cells exist, fill by station_id order."
        ),
        "station_ids": station_ids,
        "years": YEARS,
        "valid_year": VALID_YEAR,
        "test_year": TEST_YEAR,
        "split": (
            "Train on 2021-2023 normal observations; calibrate on 2024 "
            "valid normal observations and transparent injected events; "
            "evaluate on 2025 test observations and transparent injected "
            "events."
        ),
        "primary_variables": PRIMARY_VARIABLES,
        "auxiliary_variables": AUXILIARY_VARIABLES,
        "missing_rule": MISSING_RULE,
        "quality_boundary": (
            "MeteoSwiss has no per-row numeric quality code equivalent to "
            "DWD QN_9 in the frozen hourly CSV; missingness and documented "
            "station metadata are retained, and no undocumented quality "
            "code is invented."
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
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段214 MeteoSwiss第二跨机构外部验证预注册",
        "",
        f"- 状态：`{protocol['status']}`",
        f"- 来源：{PROVIDER}",
        f"- 数据集：{DATASET}",
        f"- 数据集标识：`{DATASET_ID}`",
        f"- STAC 集合：{STAC_COLLECTION}",
        f"- 许可：`{LICENSE}`",
        f"- 官方站点总数：`{len(all_stations)}`；冻结站点：`{len(selected)}`",
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
        "| 排名 | 站号 | 站名 | 纬度 | 经度 | 空间单元 |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in station_output.to_dict(orient="records"):
        lines.append(
            f"| {int(row['selection_rank'])} | {row['station_id']} | "
            f"{row['station_name']} | {float(row['latitude']):.4f} | "
            f"{float(row['longitude']):.4f} | {row['spatial_cell']} |"
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
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "all_stations": int(len(all_stations)),
                "selected_stations": int(len(selected)),
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
