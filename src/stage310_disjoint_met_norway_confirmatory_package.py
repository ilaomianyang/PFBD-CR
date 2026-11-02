from __future__ import annotations

"""Stage 310: freeze a disjoint MET Norway confirmatory package.

This stage creates the protocol, data, and analysis freezes required by
Stage278/293 before any observation value from the new station network is
downloaded. It queries only official Frost metadata to select stations and
does not request observation values, fit a model, calibrate a threshold,
inject an attack, or compute performance.

The package addresses test isolation for an unused station network within
MET Norway. It does not turn MET Norway into a second independent provider
and cannot close P0-1 by itself.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PACKAGE_ROOT = ROOT / "confirmatory_package"
PREFIX = "stage310_disjoint_met_norway_confirmatory_package"
OLD_PROTOCOL_PREFIX = "stage304_met_norway_protocol_preregistration"
OLD_CONTRACT_PREFIX = "stage305_met_norway_locked_evaluation_contract"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage304_met_norway_protocol_preregistration as s304  # noqa: E402

PROTOCOL_ID = "TMQC-MET-NORWAY-DISJOINT-STATIONS-20260921-v1.0"
PACKAGE_ID = "TMQC-CONFIRMATORY-DISJOINT-MET-NORWAY-20260921-v1.0"
NETWORK_ID = "MET_NORWAY_FROST_DISJOINT_V1"
STATION_COUNT = 12
STATIONS_PER_BAND = 3
YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_YEARS = [2021, 2022, 2023]
VALID_YEAR = 2024
TEST_YEAR = 2025
TEST_PERIOD_START_UTC = "2025-01-01T00:00:00Z"
TEST_PERIOD_END_UTC = "2025-08-24T23:59:59Z"
PRIMARY_FAR = 0.01
PRIMARY_ENDPOINT = "event_coverage"
NON_INFERIORITY_MARGIN = -0.02
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260921


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def clear_proxy_environment() -> None:
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(name, None)


def select_disjoint_stations() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    bytes,
    bytes,
]:
    old_protocol = json.loads(
        (TABLES / f"{OLD_PROTOCOL_PREFIX}.json").read_text(
            encoding="utf-8"
        )
    )
    old_station_ids = set(
        str(value) for value in old_protocol["station_ids"]
    )
    credentials = s304.load_credentials()
    headers = s304.authorization_header(credentials)
    source_payload, source_raw = s304.get_json(
        s304.SOURCE_ENDPOINT,
        headers,
    )
    timeseries_payload, timeseries_raw = s304.get_json(
        s304.TIMESERIES_ENDPOINT,
        headers,
    )
    sources = s304.normalize_source_rows(source_payload)
    timeseries = s304.normalize_timeseries_rows(timeseries_payload)
    eligible_series = timeseries[
        timeseries["element_id"].eq(s304.PRIMARY_ELEMENT)
        & timeseries["time_resolution"].eq(s304.PRIMARY_TIME_RESOLUTION)
        & timeseries["time_offset"].eq(s304.PRIMARY_TIME_OFFSET)
        & timeseries["status"].eq(s304.PRIMARY_STATUS)
        & timeseries["performance_category"].eq(
            s304.PRIMARY_PERFORMANCE_CATEGORY
        )
        & timeseries["time_series_id"].eq(0)
        & timeseries["level_value"].eq(s304.PRIMARY_LEVEL_M)
        & pd.to_datetime(
            timeseries["valid_from"],
            errors="coerce",
            utc=True,
        ).le(pd.Timestamp(s304.MIN_SERIES_START_UTC))
        & (
            timeseries["valid_to"].isna()
            | pd.to_datetime(
                timeseries["valid_to"],
                errors="coerce",
                utc=True,
            ).ge(pd.Timestamp(s304.MIN_SERIES_END_UTC))
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
        & ~eligible["station_id"].astype(str).isin(old_station_ids)
    ].copy()
    if len(eligible) < STATION_COUNT:
        raise RuntimeError("too few disjoint MET Norway stations")

    selected_rows: list[pd.DataFrame] = []
    for band in s304.LATITUDE_BANDS:
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
                f"too few disjoint stations in band {band['name']}"
            )
        positions = [
            round(index * (len(band_rows) - 1) / (STATIONS_PER_BAND - 1))
            for index in range(STATIONS_PER_BAND)
        ]
        if len(set(positions)) != STATIONS_PER_BAND:
            raise RuntimeError("station selection positions collapsed")
        chosen = band_rows.iloc[positions].copy()
        chosen["latitude_band"] = band["name"]
        chosen["band_pool_size"] = int(len(band_rows))
        chosen["rank_in_band"] = [
            positions.index(position) + 1 for position in positions
        ]
        selected_rows.append(chosen)
    selected = pd.concat(selected_rows, ignore_index=True)
    if len(selected) != STATION_COUNT:
        raise RuntimeError("disjoint station selection failed")
    if set(selected["station_id"].astype(str)).intersection(old_station_ids):
        raise RuntimeError("disjoint selection overlaps Stage304 stations")
    selected["selection_rank"] = range(1, len(selected) + 1)
    return selected, eligible, source_raw, timeseries_raw


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    clear_proxy_environment()
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    selected, eligible, source_raw, timeseries_raw = (
        select_disjoint_stations()
    )
    frozen_at = datetime.now(timezone.utc).isoformat()
    old_protocol = json.loads(
        (TABLES / f"{OLD_PROTOCOL_PREFIX}.json").read_text(
            encoding="utf-8"
        )
    )
    old_contract = json.loads(
        (TABLES / f"{OLD_CONTRACT_PREFIX}.json").read_text(
            encoding="utf-8"
        )
    )

    station_table = selected[
        [
            "selection_rank",
            "station_id",
            "source_id",
            "station_name",
            "country",
            "county",
            "municipality",
            "latitude_band",
            "longitude",
            "latitude",
            "elevation_m",
            "stationholders",
            "band_pool_size",
            "rank_in_band",
            "time_series_id",
            "performance_category",
            "exposure_category",
        ]
    ].copy()
    station_table["element_id"] = s304.PRIMARY_ELEMENT
    station_table["time_offset"] = s304.PRIMARY_TIME_OFFSET
    station_table["time_resolution"] = s304.PRIMARY_TIME_RESOLUTION
    station_table["level_m"] = s304.PRIMARY_LEVEL_M
    station_table["quality_accepted_codes"] = ",".join(
        str(value) for value in old_protocol["quality_accepted_codes"]
    )
    station_path = (
        PACKAGE_ROOT / "met_norway_disjoint_stations.csv"
    )
    station_table.to_csv(station_path, index=False, encoding="utf-8-sig")

    protocol_freeze = {
        "status": "PROTOCOL_FROZEN_BEFORE_OBSERVATION_DOWNLOAD",
        "package_id": PACKAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "provider_id": "met_norway_frost",
        "provider": "Norwegian Meteorological Institute (MET Norway)",
        "provider_is_new": False,
        "provider_reuse_boundary": (
            "MET Norway was already used by Stage304-308. This package "
            "reuses the provider but selects a disjoint station network. "
            "It can support test isolation, not provider independence."
        ),
        "network_id": NETWORK_ID,
        "network_is_disjoint_from_stage304": True,
        "stage304_station_ids": old_protocol["station_ids"],
        "station_count": int(len(station_table)),
        "station_ids": station_table["station_id"].astype(str).tolist(),
        "station_selection_rule": (
            "Use the same official Frost eligibility rule as Stage304, "
            "remove every Stage304 station ID, partition latitude into "
            "57-61, 61-64, 64-67, and 67-72, sort each band by longitude "
            "then station_id, and select three equally spaced positions "
            "by round(i*(n-1)/2), i=0..2."
        ),
        "selection_observation_access": (
            "Only official source and availableTimeSeries metadata were "
            "read. No observation value was requested or inspected."
        ),
        "years": YEARS,
        "split": {
            "train": TRAIN_YEARS,
            "valid": [VALID_YEAR],
            "test": [TEST_YEAR],
        },
        "test_period_start_utc": TEST_PERIOD_START_UTC,
        "test_period_end_utc": TEST_PERIOD_END_UTC,
        "primary_element": s304.PRIMARY_ELEMENT,
        "primary_variable": "temperature_c",
        "primary_level_m": s304.PRIMARY_LEVEL_M,
        "primary_time_offset": s304.PRIMARY_TIME_OFFSET,
        "primary_time_resolution": s304.PRIMARY_TIME_RESOLUTION,
        "quality_accepted_codes": old_protocol["quality_accepted_codes"],
        "quality_rule": old_protocol["quality_rule"],
        "missing_rule": old_protocol["missing_rule"],
        "coverage_rule": old_protocol["coverage_rule"],
        "attack_types": old_protocol["attack_types"],
        "attack_length": old_protocol["attack_length"],
        "attack_blocks": old_protocol["attack_blocks"],
        "attack_parameters": old_protocol["attack_parameters"],
        "seeds": old_protocol["seeds"],
        "far_grid": old_protocol["far_grid"],
        "primary_far": PRIMARY_FAR,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoints": old_protocol["secondary_endpoints"],
        "cluster_unit": "station",
        "multiplicity": old_protocol["multiplicity"],
        "selection_boundary": (
            "Stations, years, split, features, detector components, "
            "candidate rules, thresholds, attack parameters, seeds, "
            "baselines, primary endpoint, and statistics are frozen "
            "before any observation value is downloaded."
        ),
        "attack_boundary": (
            "All attacks are transparent controlled injections, not real "
            "attack logs."
        ),
        "frozen_at_utc": frozen_at,
        "source_metadata_sha256": {
            "sources": hashlib.sha256(source_raw).hexdigest(),
            "available_time_series": hashlib.sha256(
                timeseries_raw
            ).hexdigest(),
        },
        "eligible_disjoint_station_count": int(len(eligible)),
        "forbidden_claims": [
            "The disjoint MET Norway network is a new independent provider.",
            "Disjoint stations prove station or provider independence.",
            "Controlled injections are real production attacks.",
            "The new network may overwrite Stage304-308 results.",
        ],
    }
    protocol_path = PACKAGE_ROOT / "protocol_freeze.json"
    write_json(protocol_path, protocol_freeze)

    data_freeze = {
        "status": "DATA_FREEZE_DECLARED_BEFORE_OBSERVATION_DOWNLOAD",
        "package_id": PACKAGE_ID,
        "provider_id": "met_norway_frost",
        "network_id": NETWORK_ID,
        "station_manifest_path": (
            "confirmatory_package/met_norway_disjoint_stations.csv"
        ),
        "station_manifest_sha256": sha256_file(station_path),
        "raw_files_downloaded": False,
        "observation_values_read": 0,
        "download_contract": (
            "After this freeze, download only the official Frost "
            "/observations/v0.jsonld endpoint for the frozen station IDs "
            "and years using the frozen per-station source_id, "
            "exposure_category, quality code, and test-period boundary."
        ),
        "raw_output_root": "raw/met_norway_frost_disjoint_v1",
        "normalized_output": (
            "tables/stage311_met_norway_disjoint_download_normalized.parquet"
        ),
        "test_manifest_requirement": (
            "Every downloaded station-year response must be recorded with "
            "URL, retrieval time, bytes, and SHA256 in test_manifest.json."
        ),
        "frozen_at_utc": frozen_at,
    }
    data_path = PACKAGE_ROOT / "data_freeze.json"
    write_json(data_path, data_freeze)

    analysis_freeze = {
        "status": "ANALYSIS_FROZEN_BEFORE_OBSERVATION_DOWNLOAD",
        "package_id": PACKAGE_ID,
        "analysis_frozen": True,
        "analysis_script": (
            "src/stage312_met_norway_disjoint_locked_transfer.py"
        ),
        "analysis_script_present_at_freeze": False,
        "analysis_script_note": (
            "The analysis contract is frozen here; the implementation "
            "must match it exactly before the access gate is opened."
        ),
        "detector_components": old_contract["detector_components"],
        "methods": old_contract["methods"],
        "method_contract": old_contract["method_contract"],
        "trust_selection_rule": old_contract["trust_selection_rule"],
        "feature_contract": old_contract["feature_contract"],
        "train_rule": old_contract["train_rule"],
        "threshold_rule": old_contract["threshold_rule"],
        "primary_far": PRIMARY_FAR,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoints": old_contract["secondary_endpoints"],
        "cluster_unit": "station",
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "hierarchy": "station within the disjoint MET Norway network",
            "resampling_unit": "station",
        },
        "multiplicity": old_contract["multiplicity"],
        "non_inferiority_margin": NON_INFERIORITY_MARGIN,
        "superiority_rule": old_contract["superiority_rule"],
        "non_inferiority_rule": old_contract["non_inferiority_rule"],
        "claim_ceiling": (
            "A new isolated station-network test under the same provider. "
            "It may support station-network robustness and test isolation, "
            "but not provider independence, production superiority, or "
            "universal superiority."
        ),
        "frozen_at_utc": frozen_at,
    }
    analysis_path = PACKAGE_ROOT / "analysis_freeze.json"
    write_json(analysis_path, analysis_freeze)

    access_gate = {
        "status": "ACCESS_GATE_CLOSED_BEFORE_OBSERVATION_DOWNLOAD",
        "package_id": PACKAGE_ID,
        "protocol_frozen": True,
        "data_frozen": True,
        "analysis_frozen": True,
        "labels_inspected": False,
        "outcome_summaries_inspected": False,
        "observation_values_read": 0,
        "gate_opened_utc": None,
        "gate_rule": (
            "Do not download observation values or inspect labels until "
            "the protocol, data, and analysis files are hash-recorded and "
            "the Stage310 audit passes."
        ),
        "frozen_at_utc": frozen_at,
    }
    access_path = PACKAGE_ROOT / "access_gate.json"
    write_json(access_path, access_gate)

    test_manifest = {
        "status": "TEST_MANIFEST_NOT_YET_POPULATED",
        "package_id": PACKAGE_ID,
        "rows": [],
        "observation_values_read": 0,
        "labels_inspected": False,
        "note": (
            "Populate only after the frozen observation download. Every "
            "row must include station_id, year, URL, retrieval time, "
            "bytes, and SHA256."
        ),
        "frozen_at_utc": frozen_at,
    }
    test_manifest_path = PACKAGE_ROOT / "test_manifest.json"
    write_json(test_manifest_path, test_manifest)

    payload = {
        "status": "PASS_DISJOINT_MET_NORWAY_PACKAGE_FROZEN",
        "package_id": PACKAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "provider_id": "met_norway_frost",
        "provider_is_new": False,
        "network_id": NETWORK_ID,
        "network_is_disjoint_from_stage304": True,
        "station_count": int(len(station_table)),
        "eligible_disjoint_station_count": int(len(eligible)),
        "observation_values_read": 0,
        "labels_inspected": False,
        "test_manifest_rows": 0,
        "package_files": {
            "protocol_freeze.json": sha256_file(protocol_path),
            "data_freeze.json": sha256_file(data_path),
            "analysis_freeze.json": sha256_file(analysis_path),
            "access_gate.json": sha256_file(access_path),
            "test_manifest.json": sha256_file(test_manifest_path),
            "met_norway_disjoint_stations.csv": sha256_file(
                station_path
            ),
        },
        "claim_ceiling": (
            "TEST_ISOLATION_ONLY; no new provider and no production or "
            "universal claim."
        ),
        "limitations": [
            "MET Norway is not a new independent provider.",
            "The station network is disjoint, but provider-level "
            "independence remains unresolved.",
            "No observation value has been downloaded or inspected.",
            "The analysis implementation must match the frozen contract "
            "before the access gate is opened.",
        ],
        "created_utc": frozen_at,
    }
    output_path = TABLES / f"{PREFIX}.json"
    write_json(output_path, payload)
    pd.DataFrame(
        [
            {"package_file": name, "sha256": digest}
            for name, digest in payload["package_files"].items()
        ]
    ).to_csv(
        TABLES / f"{PREFIX}_files.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_table.to_csv(
        TABLES / f"{PREFIX}_stations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = [
        "# 阶段310 不相交MET Norway确认性包冻结",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 包ID：`{PACKAGE_ID}`",
        f"- provider：`MET Norway Frost`",
        f"- provider是否为新机构：`False`",
        f"- 与Stage304站点是否不相交：`True`",
        f"- 站点数：`{payload['station_count']}`",
        f"- 读取观测值：`0`",
        f"- 查看标签：`False`",
        f"- 声称上限：`{payload['claim_ceiling']}`",
        "",
        "## 边界",
        "",
        "- 本阶段只读取官方元数据，不下载观测、不训练、不调阈值。",
        "- 该包用于隔离新的未使用站点网络，不是新的独立机构。",
        "- 不能据此关闭P0-1，也不能声称真实生产攻击或全面领先。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
