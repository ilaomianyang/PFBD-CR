from __future__ import annotations

"""Stage 264: freeze the SMHI external-validation protocol before download.

The script reads official SMHI catalogue and station metadata for four
hourly parameters.  It does not download full observation archives, train
or score a detector, select a threshold, or compute performance.  The
artifact separately and explicitly discloses the reconnaissance range
request performed before the freeze.  Station selection uses the metadata
intersection of all four parameters, coverage dates, coordinates, and a
deterministic latitude-band rule.
"""

import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage264_smhi_protocol_preregistration"

API_ROOT = "https://opendata-download-metobs.smhi.se/api"
API_VERSION_URL = f"{API_ROOT}/version/1.0.json"
PARAMETER_URL_TEMPLATE = (
    f"{API_ROOT}/version/1.0/parameter/{{parameter_id}}.json"
)
USER_AGENT = "TrustMeteo-QC SMHI protocol preregistration"

PROTOCOL_ID = "TMQC-SMHI-EXTERNAL-20260920-v1"
PROVIDER = "Swedish Meteorological and Hydrological Institute (SMHI)"
DATASET = "SMHI Open Data Meteorological Observations, hourly station data"
PARAMETERS = {
    "temperature_c": {
        "parameter_id": 1,
        "source_field": "Lufttemperatur",
        "detail": "momentanvärde, 1 gång/tim",
        "unit": "degC",
    },
    "humidity_pct": {
        "parameter_id": 6,
        "source_field": "Relativ Luftfuktighet",
        "detail": "momentanvärde, 1 gång/tim",
        "unit": "%",
    },
    "wind_speed_ms": {
        "parameter_id": 4,
        "source_field": "Vindhastighet",
        "detail": "medelvärde 10 min, 1 gång/tim",
        "unit": "m/s",
    },
    "pressure_hpa": {
        "parameter_id": 9,
        "source_field": "Lufttryck reducerat havsytans nivå",
        "detail": "vid havsytans nivå, momentanvärde, 1 gång/tim",
        "unit": "hPa",
    },
}

YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_YEARS = [2021, 2022, 2023]
VALID_YEAR = 2024
TEST_YEAR = 2025
LOCK_START_UTC = "2021-01-01T00:00:00Z"
LOCK_END_UTC = "2026-01-01T00:00:00Z"
LOCK_START_MS = 1577836800000
LOCK_END_MS = 1767225600000

COORDINATE_BOUNDS = {
    "latitude_min": 55.0,
    "latitude_max": 69.5,
    "longitude_min": 10.0,
    "longitude_max": 25.0,
}
LATITUDE_BANDS = [
    {"name": "south", "latitude_min": 55.0, "latitude_max_exclusive": 59.5},
    {"name": "middle", "latitude_min": 59.5, "latitude_max_exclusive": 64.0},
    {"name": "north", "latitude_min": 64.0, "latitude_max_exclusive": 69.5},
]
STATIONS_PER_BAND = 6
STATION_COUNT = STATIONS_PER_BAND * len(LATITUDE_BANDS)

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
    "auprc",
]
CLUSTER_UNIT = "station"
MULTIPLICITY = (
    "Benjamini-Hochberg across pre-registered primary comparisons; "
    "provider, station, and seed clustered bootstrap intervals."
)
COVERAGE_MIN = 0.70
MISSING_RULE = (
    "SMHI blank values and all quality codes other than G are retained as "
    "missing/excluded; no sentinel substitution and no imputation. G is an "
    "official quality-control code, not an attack label."
)
DOWNLOAD_CONTRACT = (
    "Use only the official SMHI corrected-archive CSV endpoint for the "
    "selected station and each pre-registered parameter. Record URL, "
    "download time, bytes, SHA256, station id, parameter id, and response "
    "headers. Do not use latest-hour, latest-day, or latest-months as the "
    "locked historical source."
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_bytes(url: str, retries: int = 6) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
                headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                }
            if not payload:
                raise RuntimeError(f"empty response: {url}")
            return payload, headers
        except Exception as error:  # repeated SMHI TLS EOFs were observed
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2.0 + 2.0 * attempt)
    raise RuntimeError(f"SMHI metadata request failed: {url}") from last_error


def load_parameter_catalogue(
    parameter_id: int,
) -> tuple[bytes, dict[str, object]]:
    url = PARAMETER_URL_TEMPLATE.format(parameter_id=parameter_id)
    payload, _ = fetch_bytes(url)
    catalogue = json.loads(payload.decode("utf-8"))
    stations = catalogue.get("station")
    if not isinstance(stations, list) or not stations:
        raise RuntimeError(f"SMHI parameter {parameter_id} has no stations")
    return payload, catalogue


def eligible_station_table(
    catalogues: dict[str, dict[str, object]],
) -> pd.DataFrame:
    station_maps: dict[str, dict[str, dict[str, object]]] = {}
    for variable, catalogue in catalogues.items():
        rows = catalogue.get("station")
        if not isinstance(rows, list):
            raise RuntimeError(f"invalid SMHI station list for {variable}")
        station_maps[variable] = {
            str(row["id"]): row for row in rows if "id" in row
        }

    station_sets = [set(rows) for rows in station_maps.values()]
    common_ids = set.intersection(*station_sets)
    rows: list[dict[str, object]] = []
    for station_id in sorted(common_ids, key=int):
        records = {
            variable: station_maps[variable][station_id]
            for variable in station_maps
        }
        if not all(bool(record.get("active")) for record in records.values()):
            continue
        if max(int(record["from"]) for record in records.values()) > LOCK_START_MS:
            continue
        if min(int(record["to"]) for record in records.values()) < LOCK_END_MS:
            continue
        primary = records["temperature_c"]
        latitude = float(primary["latitude"])
        longitude = float(primary["longitude"])
        if not (
            COORDINATE_BOUNDS["latitude_min"]
            <= latitude
            <= COORDINATE_BOUNDS["latitude_max"]
            and COORDINATE_BOUNDS["longitude_min"]
            <= longitude
            <= COORDINATE_BOUNDS["longitude_max"]
        ):
            continue
        rows.append(
            {
                "station_id": station_id,
                "station_name": str(primary.get("name", "")),
                "latitude": latitude,
                "longitude": longitude,
                "owner": str(primary.get("owner", "")),
                "measuring_stations": str(
                    primary.get("measuringStations", "")
                ),
                "active_all_parameters": True,
                "metadata_from_max_epoch_ms": max(
                    int(record["from"]) for record in records.values()
                ),
                "metadata_to_min_epoch_ms": min(
                    int(record["to"]) for record in records.values()
                ),
            }
        )
    if len(rows) < STATION_COUNT:
        raise RuntimeError("SMHI metadata produced too few eligible stations")
    return pd.DataFrame(rows)


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
                f"SMHI latitude band {band['name']} has too few stations"
            )
        positions = sorted(
            {
                round(index * (len(band_rows) - 1) / (STATIONS_PER_BAND - 1))
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
    output["selection_rank"] = range(1, len(output) + 1)
    return output


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    catalogue_payloads: dict[str, bytes] = {}
    catalogues: dict[str, dict[str, object]] = {}
    parameter_metadata: dict[str, object] = {}
    for variable, definition in PARAMETERS.items():
        payload, catalogue = load_parameter_catalogue(
            int(definition["parameter_id"])
        )
        catalogue_payloads[variable] = payload
        catalogues[variable] = catalogue
        parameter_metadata[variable] = {
            "parameter_id": int(definition["parameter_id"]),
            "title": str(catalogue.get("title", "")),
            "summary": str(catalogue.get("summary", "")),
            "unit": str(catalogue.get("unit", "")),
            "station_count": int(len(catalogue.get("station", []))),
            "sha256": sha256_bytes(payload),
            "url": PARAMETER_URL_TEMPLATE.format(
                parameter_id=int(definition["parameter_id"])
            ),
        }

    version_payload, _ = fetch_bytes(API_VERSION_URL)
    eligible = eligible_station_table(catalogues)
    selected = select_stations(eligible)
    station_table = selected[
        [
            "selection_rank",
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "owner",
            "measuring_stations",
            "latitude_band",
            "band_pool_size",
            "rank_in_band",
            "metadata_from_max_epoch_ms",
            "metadata_to_min_epoch_ms",
        ]
    ].copy()

    metadata_path = TABLES / f"{PREFIX}_parameter_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "api_version_url": API_VERSION_URL,
                "api_version_sha256": sha256_bytes(version_payload),
                "parameters": parameter_metadata,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    station_path = TABLES / f"{PREFIX}_stations.csv"
    station_table.to_csv(
        station_path,
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "status": "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "阶段264在下载任何SMHI观测值之前冻结第六个独立机构外部"
            "验证协议。脚本只读取SMHI官方API版本元数据和四个小时参数"
            "目录，不完整下载任何历史观测序列；冻结前曾读取单个站点"
            "CSV前20KB以核实字段和质量码定义，该事实单列披露且未用于"
            "选站、调参或性能。固定站点选择、"
            "年份、变量、质量码、缺失规则、透明受控攻击、种子、主FAR"
            "和主终点；test只用于后续锁定评价，不用于选择站点、特征、"
            "阈值、权重或攻击参数。所有攻击不是真实攻击日志。"
        ),
        "provider": PROVIDER,
        "dataset": DATASET,
        "api_version_url": API_VERSION_URL,
        "api_version_sha256": sha256_bytes(version_payload),
        "parameter_url_template": PARAMETER_URL_TEMPLATE,
        "license_note": (
            "SMHI Open Data meteorological observations; the official API "
            "is used under its published open-data terms. The final paper "
            "must record the SMHI source URL, retrieval date, and any "
            "license statement shown at access time."
        ),
        "independent_source": True,
        "source_independence": (
            "SMHI is a national Swedish provider and surface-station "
            "network distinct from DWD, MeteoSwiss, KNMI, ECCC, NOAA ISD, "
            "FMI, and the Liupanshan local observations."
        ),
        "parameters": parameter_metadata,
        "primary_variable": "temperature_c",
        "auxiliary_variables": [
            "humidity_pct",
            "wind_speed_ms",
            "pressure_hpa",
        ],
        "parameter_ids": {
            variable: int(definition["parameter_id"])
            for variable, definition in PARAMETERS.items()
        },
        "eligible_station_count": int(len(eligible)),
        "station_count": int(len(station_table)),
        "station_selection_rule": (
            "Use the intersection of station metadata for parameter IDs "
            "1, 6, 4, and 9 in version 1.0. Retain stations that are active "
            "for all four parameters, have metadata availability covering "
            "2021-01-01 through 2026-01-01, and lie within latitude "
            "55.0-69.5 and longitude 10.0-25.0. Partition into south "
            "55.0<=lat<59.5, middle 59.5<=lat<64.0, north "
            "64.0<=lat<69.5. In each band sort by longitude then numeric "
            "station id, choose six positions by round(i*(n-1)/5), i=0..5. "
            "No observation values or performance results participate."
        ),
        "station_ids": station_table["station_id"].astype(str).tolist(),
        "station_names": station_table["station_name"].astype(str).tolist(),
        "years": YEARS,
        "valid_year": VALID_YEAR,
        "test_year": TEST_YEAR,
        "split": {
            "train": TRAIN_YEARS,
            "valid": [VALID_YEAR],
            "test": [TEST_YEAR],
        },
        "timezone": "UTC",
        "timestamp_normalization": (
            "Parse the official CSV Date and Tid (UTC) fields as UTC; no "
            "local-time conversion is permitted."
        ),
        "missing_rule": MISSING_RULE,
        "coverage_rule": (
            "Station retention is evaluated per split after download. A "
            "station is retained only if the primary temperature coverage "
            "is at least 0.70 in every split year and at least one G-quality "
            "non-missing value is present in each split."
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
            "before the SMHI test evaluation. Test labels are used only "
            "for evaluation. No target-domain test tuning is permitted."
        ),
        "attack_boundary": (
            "All attacks are transparent controlled injections generated "
            "by this frozen protocol. They are not real attack logs."
        ),
        "quality_boundary": (
            "SMHI quality codes are official data-quality metadata, not "
            "attack labels. Only G rows are eligible normal observations."
        ),
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "strict_no_observation_access_before_freeze": False,
        "pre_freeze_reconnaissance": {
            "occurred": True,
            "scope": (
                "A single corrected-archive CSV range request (first 20,000 "
                "bytes) for parameter 1 and station 81540 was used only to "
                "confirm the official CSV field layout, UTC labels, and "
                "quality-code definitions."
            ),
            "used_for_station_selection": False,
            "used_for_threshold_or_performance": False,
            "full_observation_download": False,
            "integrity_consequence": (
                "This protocol cannot be called a strict preregistration "
                "with zero observation-value access. It is frozen before "
                "the full locked download and before any model or endpoint "
                "inspection."
            ),
        },
        "metadata_sha256": sha256_file(metadata_path),
        "stations_sha256": sha256_file(station_path),
        "promotion_policy": {
            "current": "BOUNDED",
            "after_locked_transfer": "PARTIALLY_CLOSED",
            "resolved_requires": (
                "Independent production incident or fault logs with "
                "auditable labels and governance."
            ),
        },
        "forbidden_claims": [
            "SMHI transfer alone proves provider-independent performance.",
            "SMHI controlled injections are real production attacks.",
            "SMHI test labels may be used to tune any model or threshold.",
            "SMHI results may overwrite the frozen Stage246 endpoint.",
            "A single new provider proves cross-provider generalisation.",
            "This SMHI protocol had zero observation-value access before freeze.",
        ],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not protocol["metadata_sha256"]:
        raise RuntimeError("SMHI parameter metadata hash is missing")
    if not protocol["stations_sha256"]:
        raise RuntimeError("SMHI station table hash is missing")
    protocol_path = TABLES / f"{PREFIX}.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段264 SMHI第六机构下载前预注册",
        "",
        f"- 状态：**{protocol['status']}**",
        f"- 协议ID：`{PROTOCOL_ID}`",
        f"- 机构：`{PROVIDER}`",
        f"- 数据集：`{DATASET}`",
        f"- 站点数：`{protocol['station_count']}`，三个纬度带各 `{STATIONS_PER_BAND}`",
        f"- 年份：`{', '.join(str(value) for value in YEARS)}`",
        f"- 切分：`train=2021-2023, valid=2024, test=2025`",
        f"- 主FAR：`{PRIMARY_FAR}`；主终点：`{PRIMARY_ENDPOINT}`",
        f"- 下载状态：`{protocol['download_status']}`；性能状态：`{protocol['performance_status']}`",
        "",
        "## 冻结边界",
        "",
        "- 本阶段只读取SMHI官方参数目录和站点元数据，不请求观测值。",
        "- 冻结前曾对单个站点的CSV头部做20 KB字段/质量码核验；未用于选站、调参或性能。",
        "- 因此该协议是“完整下载前冻结”，不是“零观测值访问的严格预注册”。",
        "- 站点只由元数据交集、覆盖年份、坐标和固定纬度带规则选择。",
        "- 只接受官方质量码 G；质量码不是攻击标签。",
        "- test只用于后续锁定评价，不用于任何调参。",
        "- 所有攻击均为透明受控注入，不是真实攻击日志。",
        "- SMHI结果最多降低机构依赖风险，不能单独证明机构无关。",
        "",
        "## 站点",
        "",
        "| 排名 | 纬度带 | 站号 | 站名 | 纬度 | 经度 | 带内候选数 |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in station_table.to_dict(orient="records"):
        lines.append(
            f"| {int(row['selection_rank'])} | {row['latitude_band']} | "
            f"{row['station_id']} | {row['station_name']} | "
            f"{float(row['latitude']):.4f} | "
            f"{float(row['longitude']):.4f} | "
            f"{int(row['band_pool_size'])} |"
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
                "eligible_station_count": protocol["eligible_station_count"],
                "station_ids": protocol["station_ids"],
                "parameter_metadata_sha256": sha256_file(metadata_path),
                "stations_sha256": sha256_file(station_path),
                "protocol_sha256": sha256_file(protocol_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
