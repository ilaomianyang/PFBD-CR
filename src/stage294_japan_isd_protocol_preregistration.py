from __future__ import annotations

"""Stage 294: freeze an independent NOAA ISD Japan station-network protocol.

This stage uses only the locally cached official NOAA ISD station history
metadata.  It does not request, read, or inspect any observation values.  The
station network is deliberately disjoint from the earlier Liupanshan NOAA ISD
network and is frozen before any Japanese hourly observations are downloaded.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
HISTORY = (
    WORKSPACE_ROOT
    / "dataset"
    / "external"
    / "noaa_isd"
    / "isd-history.csv"
)
PREFIX = "stage294_japan_isd_protocol_preregistration"

PROTOCOL_ID = "TMQC-NOAA-JAPAN-NETWORK-20260921-v1.0"
PROVIDER = "NOAA National Centers for Environmental Information (NCEI)"
DATASET = "Integrated Surface Database (ISD), hourly station observations"
COUNTRY_CODE = "JA"
LATITUDE_BANDS = [
    {"name": "south", "latitude_min": 24.0, "latitude_max_exclusive": 30.0},
    {"name": "mid_south", "latitude_min": 30.0, "latitude_max_exclusive": 35.0},
    {"name": "mid_north", "latitude_min": 35.0, "latitude_max_exclusive": 40.0},
    {"name": "north", "latitude_min": 40.0, "latitude_max_exclusive": 46.0},
]
STATIONS_PER_BAND = 3
STATION_COUNT = STATIONS_PER_BAND * len(LATITUDE_BANDS)

YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_YEARS = [2021, 2022, 2023]
VALID_YEAR = 2024
TEST_YEAR = 2025
TEST_PERIOD_START_UTC = "2025-01-01T00:00:00Z"
TEST_PERIOD_END_UTC = "2025-08-24T23:59:59Z"
MIN_METADATA_END_DATE = pd.Timestamp("2025-08-24")

VARIABLES = {
    "temperature_c": {
        "isd_field": "tmp",
        "unit": "degC",
        "scale": 10.0,
    },
    "dew_point_c": {
        "isd_field": "dew",
        "unit": "degC",
        "scale": 10.0,
    },
    "wind_speed_ms": {
        "isd_field": "wspd",
        "unit": "m/s",
        "scale": 10.0,
    },
    "pressure_hpa": {
        "isd_field": "slp",
        "unit": "hPa",
        "scale": 10.0,
    },
}
PRIMARY_VARIABLE = "temperature_c"
AUXILIARY_VARIABLES = [
    "dew_point_c",
    "wind_speed_ms",
    "pressure_hpa",
]

MISSING_RULE = (
    "ISD missing sentinels and empty fields are retained as missing; no "
    "sentinel substitution, interpolation, or imputation is permitted."
)
QUALITY_ACCEPTED_CODE = "1"
QUALITY_RULE = (
    "For each pre-registered ISD field, retain a non-missing value only "
    "when the official per-field quality flag equals 1; all other flags "
    "are treated as ineligible/missing. Quality flags are data-quality "
    "metadata, not attack labels."
)
COVERAGE_MIN = 0.70

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
ATTACK_PARAMETERS = {
    "spike": {
        "kind": "additive",
        "magnitude": 8.0,
        "unit": "degC",
    },
    "offset": {
        "kind": "additive",
        "magnitude": 4.0,
        "unit": "degC",
    },
    "drift": {
        "kind": "linear_additive",
        "start": 0.0,
        "end": 6.0,
        "unit": "degC",
    },
    "time_shift": {
        "kind": "lag_copy",
        "lag_rows": 4,
        "direction": "past",
        "source_series": "clean_temperature_c",
    },
    "stuck": {
        "kind": "constant_hold",
        "source": "clean_value_at_first_attack_row",
    },
    "gaussian_noise": {
        "kind": "additive_noise",
        "sigma": 2.0,
        "unit": "degC",
    },
}
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FAR_GRID = [0.005, 0.01, 0.02, 0.05]
PRIMARY_FAR = 0.01
PRIMARY_ENDPOINT = "event_coverage"
SECONDARY_ENDPOINTS = [
    "f1",
    "precision",
    "recall",
    "row_recall",
    "false_alarm_rate",
    "average_precision",
]
CLUSTER_UNIT = "station"
MULTIPLICITY = (
    "Benjamini-Hochberg across pre-registered primary comparisons; "
    "station-clustered bootstrap intervals."
)

OFFICIAL_ROOT = "https://www.ncei.noaa.gov/data/global-hourly/access"
DOWNLOAD_CONTRACT = (
    "Use only https://www.ncei.noaa.gov/data/global-hourly/access/<year>/"
    "<station_id>.csv for the frozen station IDs and years 2021-2025. "
    "Record URL, download time, bytes, SHA256, station id, year, and "
    "response headers. Do not use any other mirror or derived observation "
    "source as the locked historical source."
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distance_km(
    latitude: pd.Series,
    longitude: pd.Series,
    reference_latitude: float,
    reference_longitude: float,
) -> pd.Series:
    p1 = np.radians(reference_latitude)
    p2 = np.radians(latitude.astype(float))
    dl = np.radians(longitude.astype(float) - reference_longitude)
    dp = p2 - p1
    a = (
        np.sin(dp / 2.0) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_metadata() -> pd.DataFrame:
    if not HISTORY.is_file():
        raise FileNotFoundError(
            f"NOAA ISD station history metadata not found: {HISTORY}"
        )
    frame = pd.read_csv(
        HISTORY,
        dtype={
            "USAF": str,
            "WBAN": str,
            "BEGIN": str,
            "END": str,
        },
    )
    frame["begin_date"] = pd.to_datetime(
        frame["BEGIN"],
        format="%Y%m%d",
        errors="coerce",
    )
    frame["end_date"] = pd.to_datetime(
        frame["END"],
        format="%Y%m%d",
        errors="coerce",
    )
    frame["station_id"] = (
        frame["USAF"].str.zfill(6)
        + frame["WBAN"].str.zfill(5)
    )
    frame["latitude"] = pd.to_numeric(frame["LAT"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["LON"], errors="coerce")
    frame["elevation_m"] = pd.to_numeric(
        frame["ELEV(M)"],
        errors="coerce",
    )
    return frame


def eligible_station_pool(metadata: pd.DataFrame) -> pd.DataFrame:
    eligible = metadata[
        metadata["CTRY"].eq(COUNTRY_CODE)
        & metadata["begin_date"].le(pd.Timestamp("2021-01-01"))
        & metadata["end_date"].ge(MIN_METADATA_END_DATE)
        & metadata["latitude"].between(24.0, 46.0)
        & metadata["longitude"].between(122.0, 154.0)
        & metadata["latitude"].notna()
        & metadata["longitude"].notna()
    ].copy()
    eligible = eligible[
        ~eligible["STATION NAME"].astype(str).str.upper().str.contains(
            r"\b(?:NAVAL|AIR STATION|AIR BASE|AB|AAF|HELIPAD)\b",
            regex=True,
            na=False,
        )
    ].copy()
    if len(eligible) < STATION_COUNT:
        raise RuntimeError("NOAA ISD Japan metadata produced too few stations")
    return eligible


def select_stations(eligible: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for band in LATITUDE_BANDS:
        band_rows = eligible[
            eligible["latitude"].ge(band["latitude_min"])
            & eligible["latitude"].lt(band["latitude_max_exclusive"])
        ].copy()
        band_rows = band_rows.sort_values(
            ["longitude", "station_id"],
            kind="stable",
        ).reset_index(drop=True)
        if len(band_rows) < STATIONS_PER_BAND:
            raise RuntimeError(
                f"NOAA ISD Japan latitude band {band['name']} "
                "has too few stations"
            )
        positions = sorted(
            {
                round(
                    index
                    * (len(band_rows) - 1)
                    / (STATIONS_PER_BAND - 1)
                )
                for index in range(STATIONS_PER_BAND)
            }
        )
        chosen = band_rows.iloc[positions].copy()
        chosen["latitude_band"] = band["name"]
        chosen["band_pool_size"] = int(len(band_rows))
        chosen["rank_in_band"] = [
            positions.index(position) + 1 for position in positions
        ]
        selected.append(chosen)
    output = pd.concat(selected, ignore_index=True)
    if len(output) != STATION_COUNT:
        raise RuntimeError("NOAA ISD Japan station selection failed")
    output["selection_rank"] = range(1, len(output) + 1)
    return output


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    metadata = load_metadata()
    eligible = eligible_station_pool(metadata)
    selected = select_stations(eligible)
    station_table = selected[
        [
            "selection_rank",
            "station_id",
            "USAF",
            "WBAN",
            "STATION NAME",
            "ICAO",
            "latitude",
            "longitude",
            "elevation_m",
            "BEGIN",
            "END",
            "latitude_band",
            "band_pool_size",
            "rank_in_band",
        ]
    ].copy()
    station_table.columns = [
        "selection_rank",
        "station_id",
        "usaf",
        "wban",
        "station_name",
        "icao",
        "latitude",
        "longitude",
        "elevation_m",
        "begin_date",
        "end_date",
        "latitude_band",
        "band_pool_size",
        "rank_in_band",
    ]
    station_table["distance_km_from_liupanshan_reference"] = distance_km(
        station_table["latitude"],
        station_table["longitude"],
        35.25,
        106.25,
    )

    station_path = TABLES / f"{PREFIX}_stations.csv"
    station_table.to_csv(
        station_path,
        index=False,
        encoding="utf-8-sig",
    )
    metadata_path = TABLES / f"{PREFIX}_metadata_source.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_path": str(HISTORY),
                "source_sha256": sha256_file(HISTORY),
                "source_bytes": int(HISTORY.stat().st_size),
                "country_code": COUNTRY_CODE,
                "station_count_eligible": int(len(eligible)),
                "selection_rule": (
                    "Retain CTRY=JA, begin<=2021-01-01, "
                    "end>=2025-08-24, latitude 24-46, longitude 122-154; "
                    "exclude military/air-station/air-base/helipad names; "
                    "partition into latitude bands 24-30, 30-35, 35-40, "
                    "and 40-46; sort each band by longitude then station_id; "
                    "select three positions by round(i*(n-1)/2), i=0..2; "
                    "freeze the resulting 12 station IDs."
                ),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    protocol = {
        "status": "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "阶段294仅使用本地缓存的NOAA ISD官方站点历史元数据，在下载"
            "任何日本站点小时观测前冻结独立站点网络协议。该网络与六盘山"
            "附近NOAA ISD站点网络完全不相交；固定站点、年份、变量、缺失"
            "规则、透明受控攻击、种子、FAR、主终点、聚类单位和多重校正。"
            "本阶段不请求观测值、不训练模型、不选择阈值、不计算性能。"
        ),
        "provider": PROVIDER,
        "dataset": DATASET,
        "dataset_root": OFFICIAL_ROOT,
        "country_code": COUNTRY_CODE,
        "license_note": (
            "NOAA NCEI Integrated Surface Database; official NCEI access "
            "endpoint only. The final paper must record the source URL, "
            "retrieval date, and applicable NCEI terms."
        ),
        "independent_source": False,
        "source_independence": (
            "This is a new, spatially disjoint NOAA ISD station network, "
            "not a new provider. It reduces station-network dependence but "
            "does not by itself establish provider independence."
        ),
        "network_id": "NOAA_ISD_JAPAN_MAINLAND_V1",
        "network_disjoint_from_prior_noaa": True,
        "prior_noaa_network_reference": (
            "The earlier NOAA ISD network was selected around the "
            "Liupanshan reference point (35.25N, 106.25E). The frozen "
            "Japanese stations are separated by the distance column in the "
            "station table and are not among the earlier downloaded IDs."
        ),
        "station_count": int(len(station_table)),
        "station_selection_rule": (
            "Retain CTRY=JA, begin<=2021-01-01, end>=2025-08-24, "
            "latitude 24-46, longitude 122-154; exclude military and "
            "air-station/air-base/helipad names; partition into latitude "
            "bands 24-30, 30-35, 35-40, and 40-46; sort each band by "
            "longitude then station_id; select three positions by "
            "round(i*(n-1)/2), i=0..2; freeze the resulting 12 station IDs."
        ),
        "station_ids": station_table["station_id"].astype(str).tolist(),
        "station_names": station_table["station_name"].astype(str).tolist(),
        "years": YEARS,
        "train_years": TRAIN_YEARS,
        "valid_year": VALID_YEAR,
        "test_year": TEST_YEAR,
        "test_period_start_utc": TEST_PERIOD_START_UTC,
        "test_period_end_utc": TEST_PERIOD_END_UTC,
        "test_period_is_partial_year": True,
        "split": {
            "train": TRAIN_YEARS,
            "valid": [VALID_YEAR],
            "test": [TEST_YEAR],
        },
        "timezone": "UTC",
        "timestamp_normalization": (
            "Parse the ISD DATE field as UTC. No local-time conversion is "
            "permitted."
        ),
        "variables": VARIABLES,
        "primary_variable": PRIMARY_VARIABLE,
        "auxiliary_variables": AUXILIARY_VARIABLES,
        "missing_rule": MISSING_RULE,
        "quality_accepted_code": QUALITY_ACCEPTED_CODE,
        "quality_rule": QUALITY_RULE,
        "coverage_rule": (
            "A station is retained only if primary temperature coverage is "
            "at least 0.70 in every split year and at least one non-missing "
            "value is present in each split. The 2025 test split is bounded "
            "by the frozen metadata snapshot and ends at "
            "2025-08-24T23:59:59Z; no claim of full-year 2025 coverage is "
            "permitted."
        ),
        "download_contract": DOWNLOAD_CONTRACT,
        "attack_types": ATTACK_TYPES,
        "attack_length": ATTACK_LENGTH,
        "attack_blocks": ATTACK_BLOCKS,
        "attack_parameters": ATTACK_PARAMETERS,
        "seeds": SEEDS,
        "far_grid": FAR_GRID,
        "primary_far": PRIMARY_FAR,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoints": SECONDARY_ENDPOINTS,
        "cluster_unit": CLUSTER_UNIT,
        "multiplicity": MULTIPLICITY,
        "selection_boundary": (
            "Detector, features, fusion weights, protection rule, "
            "threshold, attack location, station, and year are frozen "
            "before the Japanese test evaluation. Test labels are used "
            "only for evaluation. No target-domain test tuning is "
            "permitted."
        ),
        "attack_boundary": (
            "All attacks are transparent controlled injections generated "
            "by this frozen protocol. They are not real attack logs."
        ),
        "quality_boundary": (
            QUALITY_RULE
        ),
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "metadata_source_sha256": sha256_file(metadata_path),
        "stations_sha256": sha256_file(station_path),
        "strict_no_observation_access_before_freeze": True,
        "pre_freeze_reconnaissance": {
            "occurred": False,
            "scope": "No Japanese ISD observation CSV was requested or read.",
            "used_for_station_selection": False,
            "used_for_threshold_or_performance": False,
        },
        "promotion_policy": {
            "current": "BOUNDED",
            "after_locked_transfer": "PARTIALLY_CLOSED_FOR_STATION_NETWORK",
            "resolved_requires": (
                "A new independent provider and independent production "
                "incident or fault logs with auditable labels."
            ),
        },
        "forbidden_claims": [
            "A new NOAA station network proves provider independence.",
            "The Japanese network alone proves station-independent performance.",
            "Controlled injections are real production attacks.",
            "Test labels may be used to tune any model or threshold.",
            "This transfer may overwrite Stage246 or the FMI/SMHI results.",
        ],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol_path = TABLES / f"{PREFIX}.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段294 NOAA ISD日本独立站点网络预注册",
        "",
        f"- 状态：**{protocol['status']}**",
        f"- 协议ID：`{PROTOCOL_ID}`",
        f"- 来源：`{PROVIDER}`",
        f"- 数据集：`{DATASET}`",
        f"- 网络：`{protocol['network_id']}`，与六盘山NOAA网络空间不相交",
        f"- 站点数：`{protocol['station_count']}`",
        f"- 年份：`{', '.join(str(value) for value in YEARS)}`",
        "- 切分：`train=2021-2023, valid=2024, "
        "test=2025-01-01至2025-08-24`",
        f"- 主FAR：`{PRIMARY_FAR}`；主终点：`{PRIMARY_ENDPOINT}`",
        "- 下载状态：`NOT_DOWNLOADED`；性能状态：`NOT_EVALUATED`",
        "",
        "## 冻结边界",
        "",
        "- 只读取本地缓存的NOAA ISD官方站点历史元数据。",
        "- 本阶段不请求、不读取任何日本站点小时观测值。",
        "- 站点网络与先前六盘山附近NOAA网络完全不相交。",
        "- 这是一个新站点网络，不是新机构；不能单独证明机构无关。",
        "- test只用于后续锁定评价，不用于任何调参。",
        "- 所有攻击均为透明受控注入，不是真实攻击日志。",
        "",
        "## 站点",
        "",
        "| 排名 | 纬度带 | 站号 | 站名 | 纬度 | 经度 | "
        "海拔(m) | 距六盘山参考点km |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in station_table.to_dict(orient="records"):
        lines.append(
            f"| {int(row['selection_rank'])} | {row['latitude_band']} | "
            f"{row['station_id']} | {row['station_name']} | "
            f"{float(row['latitude']):.4f} | "
            f"{float(row['longitude']):.4f} | "
            f"{float(row['elevation_m']):.0f} | "
            f"{float(row['distance_km_from_liupanshan_reference']):.0f} |"
        )
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "protocol_id": PROTOCOL_ID,
                "station_count": protocol["station_count"],
                "station_ids": protocol["station_ids"],
                "metadata_source_sha256": protocol["metadata_source_sha256"],
                "stations_sha256": protocol["stations_sha256"],
                "protocol_sha256": sha256_file(protocol_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
