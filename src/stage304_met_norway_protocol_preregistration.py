from __future__ import annotations

"""Stage 304: freeze a MET Norway Frost independent-provider protocol.

This stage queries only official Frost metadata: source geometry, element
metadata, and hourly time-series availability.  It does not request any
observation value.  The selected Norwegian station network, years, split,
variables, missing/quality rules, transparent attacks, seeds, primary FAR,
primary endpoint, clustering unit, and multiplicity rule are frozen before
the first observation download.

MET Norway is an independent national provider and is not derived from the
NOAA, MeteoSwiss, DWD, KNMI, ECCC, FMI, or SMHI sources used earlier.
"""

import base64
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SECRET = ROOT / ".secrets" / "met_norway_frost.json"
PREFIX = "stage304_met_norway_protocol_preregistration"

PROTOCOL_ID = "TMQC-MET-NORWAY-FROST-20260921-v1.0"
PROTOCOL_REVISION = "v1.1"
PROVIDER = "Norwegian Meteorological Institute (MET Norway)"
DATASET = "Frost API hourly station observations"
OFFICIAL_ROOT = "https://frost.met.no"
SOURCE_ENDPOINT = (
    "https://frost.met.no/sources/v0.jsonld"
    "?types=SensorSystem&country=NO&fields="
    "id,name,geometry,country,county,municipality,masl,stationholders"
)
TIMESERIES_ENDPOINT = (
    "https://frost.met.no/observations/availableTimeSeries/v0.jsonld"
    "?elements=air_temperature&timeresolutions=PT1H&timeoffsets=PT0H"
)
ELEMENT_ENDPOINT = (
    "https://frost.met.no/elements/v0.jsonld?fields=id,name,unit"
)
QUALITY_ENDPOINT = (
    "https://frost.met.no/observations/availableQualityCodes/v0.jsonld"
    "?lang=en"
)

NETWORK_ID = "MET_NORWAY_FROST_MAINLAND_V1"
COUNTRY_CODE = "NO"
LATITUDE_BANDS = [
    {"name": "south", "latitude_min": 57.0, "latitude_max_exclusive": 61.0},
    {"name": "mid", "latitude_min": 61.0, "latitude_max_exclusive": 64.0},
    {"name": "north", "latitude_min": 64.0, "latitude_max_exclusive": 67.0},
    {"name": "arctic", "latitude_min": 67.0, "latitude_max_exclusive": 72.0},
]
STATIONS_PER_BAND = 3
STATION_COUNT = STATIONS_PER_BAND * len(LATITUDE_BANDS)

YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_YEARS = [2021, 2022, 2023]
VALID_YEAR = 2024
TEST_YEAR = 2025
TEST_PERIOD_START_UTC = "2025-01-01T00:00:00Z"
TEST_PERIOD_END_UTC = "2025-08-24T23:59:59Z"
MIN_SERIES_START_UTC = "2021-01-01T00:00:00Z"
MIN_SERIES_END_UTC = "2025-08-24T00:00:00Z"

PRIMARY_ELEMENT = "air_temperature"
PRIMARY_VARIABLE = "temperature_c"
PRIMARY_LEVEL_M = 2.0
PRIMARY_TIME_OFFSET = "PT0H"
PRIMARY_TIME_RESOLUTION = "PT1H"
PRIMARY_PERFORMANCE_CATEGORY = "C"
PRIMARY_STATUS = "Authoritative"

QUALITY_ACCEPTED_CODES = [0]
QUALITY_RULE = (
    "Retain a value only when the official Frost qualityCode is 0 "
    "(controlled and OK). Frost qualityCode 1 can mean corrected or "
    "missing and interpolated, so it is not accepted. Quality codes are "
    "data-quality metadata, not attack labels. Values with codes 1, 2, "
    "4, 5, 6, or 7 are retained as missing."
)
MISSING_RULE = (
    "Missing values and disallowed quality codes remain missing; no "
    "interpolation, sentinel substitution, or imputation is permitted."
)
COVERAGE_MIN = 0.70
COVERAGE_RULE = (
    "A station is retained only if the primary air_temperature coverage is "
    "at least 0.70 in every split year and at least one accepted value is "
    "present in each split. The 2025 test split ends at "
    "2025-08-24T23:59:59Z and is not a full-year test period."
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
ATTACK_PARAMETERS = {
    "spike": {"kind": "additive", "magnitude": 8.0, "unit": "degC"},
    "offset": {"kind": "additive", "magnitude": 4.0, "unit": "degC"},
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_credentials() -> dict[str, str]:
    if not SECRET.is_file():
        raise FileNotFoundError(
            "MET Norway Frost credentials are not available in .secrets"
        )
    payload = json.loads(SECRET.read_text(encoding="utf-8"))
    client_id = str(payload.get("client_id", "")).strip()
    client_secret = str(payload.get("client_secret", "")).strip()
    if not client_id or not client_secret:
        raise RuntimeError("MET Norway Frost credentials are incomplete")
    return {"client_id": client_id, "client_secret": client_secret}


def authorization_header(credentials: dict[str, str]) -> dict[str, str]:
    token = base64.b64encode(
        (
            credentials["client_id"] + ":" + credentials["client_secret"]
        ).encode("ascii")
    ).decode("ascii")
    return {
        "Authorization": f"Basic {token}",
        "User-Agent": "TrustMeteo-QC reproducible research MET Norway audit",
    }


def get_json(
    url: str,
    headers: dict[str, str],
    timeout: int = 120,
) -> tuple[dict, bytes]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Frost metadata request failed: {url} "
            f"HTTP {error.code}: {body[:500]}"
        ) from error
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"unexpected Frost response type: {url}")
    return parsed, payload


def normalize_source_rows(payload: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in payload.get("data", []):
        geometry = item.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) != 2:
            continue
        rows.append(
            {
                "station_id": str(item.get("id", "")),
                "station_name": str(item.get("name", "")),
                "country": str(item.get("country", "")),
                "county": str(item.get("county", "")),
                "municipality": str(item.get("municipality", "")),
                "longitude": float(coordinates[0]),
                "latitude": float(coordinates[1]),
                "elevation_m": item.get("masl"),
                "stationholders": "|".join(
                    str(value)
                    for value in (item.get("stationholders") or [])
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Frost source metadata returned no usable rows")
    if frame["station_id"].duplicated().any():
        raise RuntimeError("Frost source metadata contains duplicate ids")
    return frame


def normalize_timeseries_rows(payload: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in payload.get("data", []):
        level = item.get("level") or {}
        rows.append(
            {
                "source_id": str(item.get("sourceId", "")),
                "station_id": str(item.get("sourceId", "")).split(":")[0],
                "element_id": str(item.get("elementId", "")),
                "unit": str(item.get("unit", "")),
                "time_offset": str(item.get("timeOffset", "")),
                "time_resolution": str(item.get("timeResolution", "")),
                "time_series_id": item.get("timeSeriesId"),
                "performance_category": str(
                    item.get("performanceCategory", "")
                ),
                "exposure_category": str(item.get("exposureCategory", "")),
                "status": str(item.get("status", "")),
                "level_type": str(level.get("levelType", "")),
                "level_unit": str(level.get("unit", "")),
                "level_value": level.get("value"),
                "valid_from": item.get("validFrom"),
                "valid_to": item.get("validTo"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Frost time-series metadata returned no rows")
    exact_duplicate_count = int(frame.duplicated().sum())
    if exact_duplicate_count:
        frame = frame.drop_duplicates().reset_index(drop=True)
    return frame


def select_stations(
    sources: pd.DataFrame,
    timeseries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_series = timeseries[
        timeseries["element_id"].eq(PRIMARY_ELEMENT)
        & timeseries["time_resolution"].eq(PRIMARY_TIME_RESOLUTION)
        & timeseries["time_offset"].eq(PRIMARY_TIME_OFFSET)
        & timeseries["status"].eq(PRIMARY_STATUS)
        & timeseries["performance_category"].eq(
            PRIMARY_PERFORMANCE_CATEGORY
        )
        & timeseries["time_series_id"].eq(0)
        & timeseries["level_value"].eq(PRIMARY_LEVEL_M)
        & pd.to_datetime(
            timeseries["valid_from"],
            errors="coerce",
            utc=True,
        ).le(pd.Timestamp(MIN_SERIES_START_UTC))
        & (
            timeseries["valid_to"].isna()
            | pd.to_datetime(
                timeseries["valid_to"],
                errors="coerce",
                utc=True,
            ).ge(pd.Timestamp(MIN_SERIES_END_UTC))
        )
    ].copy()
    eligible = sources.merge(
        eligible_series[
            [
                "station_id",
                "source_id",
                "time_series_id",
                "valid_from",
                "valid_to",
                "performance_category",
                "exposure_category",
            ]
        ],
        on="station_id",
        how="inner",
        validate="one_to_one",
    )
    eligible = eligible[
        eligible["country"].eq("Norge")
        & eligible["latitude"].between(57.0, 72.0)
        & eligible["longitude"].between(4.0, 32.0)
    ].copy()
    if len(eligible) < STATION_COUNT:
        raise RuntimeError("Frost produced too few eligible stations")

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
                f"Frost latitude band {band['name']} has too few stations"
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
        raise RuntimeError("Frost station selection failed")
    output["selection_rank"] = range(1, len(output) + 1)
    return output, eligible


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    credentials = load_credentials()
    headers = authorization_header(credentials)

    source_payload, source_raw = get_json(SOURCE_ENDPOINT, headers)
    timeseries_payload, timeseries_raw = get_json(
        TIMESERIES_ENDPOINT,
        headers,
    )
    element_payload, element_raw = get_json(ELEMENT_ENDPOINT, headers)
    quality_payload, quality_raw = get_json(QUALITY_ENDPOINT, headers)

    sources = normalize_source_rows(source_payload)
    timeseries = normalize_timeseries_rows(timeseries_payload)
    selected, eligible = select_stations(sources, timeseries)

    station_table = selected[
        [
            "selection_rank",
            "station_id",
            "source_id",
            "station_name",
            "county",
            "municipality",
            "longitude",
            "latitude",
            "elevation_m",
            "stationholders",
            "latitude_band",
            "band_pool_size",
            "rank_in_band",
            "valid_from",
            "valid_to",
            "performance_category",
            "exposure_category",
        ]
    ].copy()
    station_contract = station_table[
        [
            "selection_rank",
            "station_id",
            "source_id",
            "exposure_category",
        ]
    ].copy()
    station_contract["element_id"] = PRIMARY_ELEMENT
    station_contract["time_offset"] = PRIMARY_TIME_OFFSET
    station_contract["time_resolution"] = PRIMARY_TIME_RESOLUTION
    station_contract["level_m"] = PRIMARY_LEVEL_M
    station_contract["time_series_id"] = 0
    station_contract["performance_category"] = (
        PRIMARY_PERFORMANCE_CATEGORY
    )
    station_contract["quality_accepted_codes"] = ",".join(
        str(value) for value in QUALITY_ACCEPTED_CODES
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
                "provider": PROVIDER,
                "dataset": DATASET,
                "official_root": OFFICIAL_ROOT,
                "source_endpoint": SOURCE_ENDPOINT,
                "timeseries_endpoint": TIMESERIES_ENDPOINT,
                "element_endpoint": ELEMENT_ENDPOINT,
                "quality_endpoint": QUALITY_ENDPOINT,
                "source_response_sha256": sha256_bytes(source_raw),
                "timeseries_response_sha256": sha256_bytes(
                    timeseries_raw
                ),
                "element_response_sha256": sha256_bytes(element_raw),
                "quality_response_sha256": sha256_bytes(quality_raw),
                "source_item_count": int(
                    source_payload.get("totalItemCount", 0)
                ),
                "timeseries_item_count": int(
                    timeseries_payload.get("totalItemCount", 0)
                ),
                "eligible_station_count": int(len(eligible)),
                "selection_rule": (
                    "Retain official Frost SensorSystem stations in Norway "
                    "with an authoritative PT1H air_temperature series, "
                    "PT0H offset, 2 m level, series id 0, validFrom <= "
                    "2021-01-01, and validTo absent or >= 2025-08-24. "
                    "Partition latitude into 57-61, 61-64, 64-67, and "
                    "67-72; sort each band by longitude then station_id; "
                    "select three positions by round(i*(n-1)/2), i=0..2; "
                    "freeze the resulting 12 station IDs."
                ),
                "observation_values_accessed": False,
                "observation_downloads": 0,
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
        "protocol_revision": PROTOCOL_REVISION,
        "protocol": (
            "阶段304仅使用MET Norway Frost官方元数据端点，在下载任何"
            "挪威站点小时观测前冻结独立机构协议。该机构与先前NOAA、"
            "MeteoSwiss、DWD、KNMI、ECCC、FMI和SMHI来源独立；固定"
            "站点、年份、变量、质量规则、透明受控攻击、种子、FAR、"
            "主终点、聚类单位和多重校正。本阶段不请求观测值、不训练"
            "模型、不选择阈值、不计算性能。"
        ),
        "protocol_revision_note": (
            "v1.1修正下载契约：按Stage304已冻结元数据逐站记录并查询"
            "source_id和exposure_category。v1.0曾把曝光类别固定为2，"
            "导致曝光类别为1的冻结站点无法下载。修订只补全元数据契约，"
            "不改变站点集合、年份、变量、质量规则、攻击、种子、FAR、"
            "主终点或任何评价选择；修订发生在任何test评价之前。"
        ),
        "provider": PROVIDER,
        "dataset": DATASET,
        "dataset_root": OFFICIAL_ROOT,
        "country_code": COUNTRY_CODE,
        "license": "Norwegian Licence for Open Government Data (NLOD) 2.0",
        "license_note": (
            "Frost API documentation reports NLOD 2.0. The final paper "
            "must record the official source URL, retrieval date, "
            "attribution, and applicable NLOD terms."
        ),
        "independent_source": True,
        "source_independence": (
            "MET Norway is an independent national meteorological "
            "institute. It is not NOAA and is not a mirror of the earlier "
            "providers. Independence is a source property; it does not by "
            "itself prove universal or production superiority."
        ),
        "network_id": NETWORK_ID,
        "station_count": int(len(station_table)),
        "station_selection_rule": (
            "Retain official Frost SensorSystem stations in Norway with an "
            "authoritative PT1H air_temperature series, PT0H offset, 2 m "
            "level, series id 0, validFrom <= 2021-01-01, and validTo "
            "absent or >= 2025-08-24. Partition latitude into 57-61, "
            "61-64, 64-67, and 67-72; sort each band by longitude then "
            "station_id; select three positions by round(i*(n-1)/2), "
            "i=0..2; freeze the resulting 12 station IDs."
        ),
        "station_ids": station_table["station_id"].astype(str).tolist(),
        "station_names": station_table["station_name"].astype(str).tolist(),
        "station_download_contract": station_contract.to_dict("records"),
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
            "Use the official Frost referenceTime as UTC and add the "
            "official timeOffset. No local-time conversion is permitted."
        ),
        "primary_element": PRIMARY_ELEMENT,
        "primary_variable": PRIMARY_VARIABLE,
        "primary_level_m": PRIMARY_LEVEL_M,
        "primary_time_offset": PRIMARY_TIME_OFFSET,
        "primary_time_resolution": PRIMARY_TIME_RESOLUTION,
        "quality_accepted_codes": QUALITY_ACCEPTED_CODES,
        "quality_rule": QUALITY_RULE,
        "missing_rule": MISSING_RULE,
        "coverage_rule": COVERAGE_RULE,
        "download_contract": (
            "Use only the official Frost /observations/v0.jsonld endpoint "
            "with the per-station frozen source_id, exposure_category, "
            "element air_temperature, PT0H offset, PT1H resolution, "
            "level 2 m, series id 0, performance category C, and "
            "explicit reference-time intervals. Record URL, query "
            "parameters, retrieval time, response bytes, SHA256, "
            "station id, and pagination metadata. Do not use a mirror "
            "or derived source."
        ),
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
            "before the MET Norway test evaluation. Test labels are used "
            "only for evaluation. No target-domain test tuning is "
            "permitted."
        ),
        "attack_boundary": (
            "All attacks are transparent controlled injections generated "
            "by this frozen protocol. They are not real attack logs."
        ),
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "metadata_source_sha256": sha256_file(metadata_path),
        "stations_sha256": sha256_file(station_path),
        "strict_no_observation_access_before_freeze": True,
        "pre_freeze_reconnaissance": {
            "occurred": True,
            "scope": (
                "Only official Frost source, element, quality-code, and "
                "hourly time-series availability metadata were queried. "
                "No observation value was requested or read."
            ),
            "used_for_station_selection": True,
            "used_for_threshold_or_performance": False,
            "observation_value_count_read": 0,
        },
        "promotion_policy": {
            "current": "BOUNDED",
            "after_locked_transfer": (
                "PARTIALLY_CLOSED_FOR_INDEPENDENT_PROVIDER"
            ),
            "resolved_requires": (
                "Locked evaluation on the frozen MET Norway network, "
                "retention of all baseline wins and losses, and separate "
                "production-incident evidence before any real-production "
                "claim."
            ),
        },
        "forbidden_claims": [
            "One independent provider alone proves universal superiority.",
            "MET Norway controlled injections are real production attacks.",
            "Test labels may be used to tune any model or threshold.",
            "This transfer may overwrite Stage246, FMI, SMHI, or Japan.",
            "Provider independence alone proves station independence.",
        ],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol_path = TABLES / f"{PREFIX}.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段304 MET Norway Frost独立机构协议预注册",
        "",
        f"- 状态：**{protocol['status']}**",
        f"- 协议ID：`{PROTOCOL_ID}`",
        f"- 来源：`{PROVIDER}`",
        f"- 数据集：`{DATASET}`",
        f"- 网络：`{NETWORK_ID}`",
        f"- 站点数：`{protocol['station_count']}`",
        f"- 年份：`{', '.join(str(value) for value in YEARS)}`",
        "- 切分：`train=2021-2023, valid=2024, "
        "test=2025-01-01至2025-08-24`",
        f"- 主FAR：`{PRIMARY_FAR}`；主终点：`{PRIMARY_ENDPOINT}`",
        "- 下载状态：`NOT_DOWNLOADED`；性能状态：`NOT_EVALUATED`",
        "- 观测值访问：`0`；只读取官方元数据。",
        "",
        "## 冻结边界",
        "",
        "- 只读取Frost官方机构、元素、质量码和小时可用性元数据。",
        "- 本阶段不请求、不读取任何MET Norway站点小时观测值。",
        "- 站点选择在观测下载前冻结，test只用于后续锁定评价。",
        "- 所有攻击均为透明受控注入，不是真实生产攻击日志。",
        "- 独立机构证据不等于全面领先，也不证明真实生产召回率。",
        "",
        "## 站点",
        "",
        "| 排名 | 纬度带 | 站号 | 站名 | 纬度 | 经度 | 海拔(m) |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in station_table.to_dict(orient="records"):
        elevation = (
            ""
            if pd.isna(row["elevation_m"])
            else f"{float(row['elevation_m']):.0f}"
        )
        lines.append(
            f"| {int(row['selection_rank'])} | {row['latitude_band']} | "
            f"{row['station_id']} | {row['station_name']} | "
            f"{float(row['latitude']):.4f} | "
            f"{float(row['longitude']):.4f} | {elevation} |"
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
                "provider": PROVIDER,
                "station_count": protocol["station_count"],
                "station_ids": protocol["station_ids"],
                "metadata_source_sha256": protocol[
                    "metadata_source_sha256"
                ],
                "stations_sha256": protocol["stations_sha256"],
                "protocol_sha256": sha256_file(protocol_path),
                "observation_values_accessed": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
