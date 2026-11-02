from __future__ import annotations

"""Stage 369: freeze a second, unused SMHI station set for factor stress tests.

The selection is metadata-only and deterministic. It excludes both the
Stage267 transfer stations and the Stage364/367 holdout stations. Because this
protocol is initiated after Stage367 results were observed, its results are
explicitly exploratory and cannot replace the frozen V6 or Stage367 claims.
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
PREFIX = "stage369_smhi_factor_holdout_protocol"
API_ROOT = "https://opendata-download-metobs.smhi.se/api"
PARAMETER_URL = f"{API_ROOT}/version/1.0/parameter/{{}}.json"
VERSION_URL = f"{API_ROOT}/version/1.0.json"
PARAMETERS = {
    "temperature_c": 1,
    "humidity_pct": 6,
    "wind_speed_ms": 4,
    "pressure_hpa": 9,
}
LOCK_START_MS = 1577836800000
LOCK_END_MS = 1767225600000
COORDINATE_BOUNDS = {
    "latitude_min": 55.0,
    "latitude_max": 69.5,
    "longitude_min": 10.0,
    "longitude_max": 25.0,
}
BANDS = [
    {"name": "south", "latitude_min": 55.0, "latitude_max_exclusive": 59.5},
    {"name": "middle", "latitude_min": 59.5, "latitude_max_exclusive": 64.0},
    {"name": "north", "latitude_min": 64.0, "latitude_max_exclusive": 69.5},
]
STATIONS_PER_BAND = 6


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str) -> tuple[bytes, dict]:
    last: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "TrustMeteo-QC Stage369 protocol freeze"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            return payload, json.loads(payload.decode("utf-8"))
        except Exception as error:
            last = error
            if attempt < 4:
                time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"metadata request failed: {url}") from last


def eligible(catalogues: dict[str, dict]) -> pd.DataFrame:
    maps: dict[str, dict[str, dict]] = {}
    for variable, catalogue in catalogues.items():
        rows = catalogue.get("station")
        if not isinstance(rows, list):
            raise RuntimeError(f"invalid catalogue for {variable}")
        maps[variable] = {str(row["id"]): row for row in rows if "id" in row}
    common = set.intersection(*(set(value) for value in maps.values()))
    output: list[dict] = []
    for station_id in sorted(common, key=int):
        records = {variable: maps[variable][station_id] for variable in maps}
        if not all(bool(record.get("active")) for record in records.values()):
            continue
        if max(int(record["from"]) for record in records.values()) > LOCK_START_MS:
            continue
        if min(int(record["to"]) for record in records.values()) < LOCK_END_MS:
            continue
        primary = records["temperature_c"]
        lat = float(primary["latitude"])
        lon = float(primary["longitude"])
        if not (
            COORDINATE_BOUNDS["latitude_min"] <= lat <= COORDINATE_BOUNDS["latitude_max"]
            and COORDINATE_BOUNDS["longitude_min"] <= lon <= COORDINATE_BOUNDS["longitude_max"]
        ):
            continue
        output.append(
            {
                "station_id": station_id,
                "station_name": str(primary.get("name", "")),
                "latitude": lat,
                "longitude": lon,
                "owner": str(primary.get("owner", "")),
                "measuring_stations": str(primary.get("measuringStations", "")),
            }
        )
    return pd.DataFrame(output)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    old = json.loads(
        (TABLES / "stage264_smhi_protocol_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    first = json.loads(
        (TABLES / "stage364_smhi_station_holdout_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    excluded = {
        *[str(value) for value in old["station_ids"]],
        *[str(value) for value in first["station_ids"]],
    }
    payload_hashes: dict[str, str] = {}
    catalogues: dict[str, dict] = {}
    for variable, parameter_id in PARAMETERS.items():
        raw, catalogue = fetch(PARAMETER_URL.format(parameter_id))
        payload_hashes[variable] = sha256_bytes(raw)
        catalogues[variable] = catalogue
    version_raw, _ = fetch(VERSION_URL)
    all_eligible = eligible(catalogues)
    remaining = all_eligible[
        ~all_eligible["station_id"].astype(str).isin(excluded)
    ].copy()
    selected: list[pd.DataFrame] = []
    for band in BANDS:
        pool = remaining[
            remaining["latitude"].ge(band["latitude_min"])
            & remaining["latitude"].lt(band["latitude_max_exclusive"])
        ].sort_values(["longitude", "station_id"], kind="stable")
        if len(pool) < STATIONS_PER_BAND:
            raise RuntimeError(f"insufficient stations in {band['name']}")
        positions = sorted(
            {
                round((index + 0.5) * (len(pool) - 1) / STATIONS_PER_BAND)
                for index in range(STATIONS_PER_BAND)
            }
        )
        chosen = pool.iloc[positions].copy()
        chosen["latitude_band"] = band["name"]
        chosen["band_pool_size_after_exclusion"] = int(len(pool))
        chosen["rank_in_band"] = [
            positions.index(position) + 1 for position in positions
        ]
        selected.append(chosen)
    stations = pd.concat(selected, ignore_index=True)
    stations["selection_rank"] = range(1, len(stations) + 1)
    station_path = TABLES / f"{PREFIX}_stations.csv"
    stations[
        [
            "selection_rank",
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "owner",
            "measuring_stations",
            "latitude_band",
            "band_pool_size_after_exclusion",
            "rank_in_band",
        ]
    ].to_csv(station_path, index=False, encoding="utf-8")
    protocol = {
        "status": "PROTOCOL_FROZEN_BEFORE_OBSERVATION_ACCESS",
        "protocol_id": "TMQC-SMHI-FACTOR-HOLDOUT-20260924-v1",
        "provider": "Swedish Meteorological and Hydrological Institute (SMHI)",
        "scope": "second unused SMHI station holdout for factor stress testing",
        "exploratory_boundary": (
            "This protocol was initiated after Stage367 results were observed. "
            "It is an independent station set with frozen methods, but its "
            "performance is exploratory and cannot replace Stage246/V6."
        ),
        "observation_access": False,
        "performance_access": False,
        "excluded_station_ids": sorted(excluded, key=int),
        "eligible_station_count_before_exclusion": int(len(all_eligible)),
        "eligible_station_count_after_exclusion": int(len(remaining)),
        "station_count": int(len(stations)),
        "station_ids": stations["station_id"].astype(str).tolist(),
        "station_names": stations["station_name"].astype(str).tolist(),
        "parameter_ids": PARAMETERS,
        "selection_rule": (
            "Use only four-parameter metadata intersection, active status, "
            "2021-01-01 through 2026-01-01 metadata coverage and fixed "
            "coordinates. Exclude Stage267 and Stage364 IDs. Within each "
            "latitude band sort by longitude and numeric station ID and select "
            "six positions using round((i+0.5)*(n-1)/6), i=0..5."
        ),
        "attack_factor_protocol": {
            "factorial": {
                "intensity": ["weak", "strong"],
                "active_duration": ["short", "long"],
                "event_window": ["short", "long"],
            },
            "unseen_combination": "coordinated_drift_unseen",
            "random_streams": [1070001, 1080001],
            "primary_far": 0.01,
            "primary_endpoint": "event_coverage",
            "seeds": [3407, 6257, 7919, 104729, 20260715],
        },
        "api_version_url": VERSION_URL,
        "api_version_sha256": sha256_bytes(version_raw),
        "parameter_catalogue_sha256": payload_hashes,
        "stations_sha256": sha256_file(station_path),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Same-provider station boundary; controlled simulated attacks; "
            "no institution-independent or production-attack claim."
        ),
    }
    output = TABLES / f"{PREFIX}.json"
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = REPORTS / f"{PREFIX}.md"
    report.write_text(
        "\n".join(
            [
                "# Stage369 SMHI 第二批因子压力测试站点协议冻结",
                "",
                f"- 协议：`{protocol['protocol_id']}`",
                f"- 排除旧站点：`{len(excluded)}`",
                f"- 元数据合格站点：`{protocol['eligible_station_count_after_exclusion']}`",
                f"- 新站点：`{protocol['station_count']}`",
                "- 观测值访问：否；性能结果访问：否。",
                "- 因为协议启动时间晚于 Stage367 结果查看，本批结果仅为探索性。",
                "",
                "## 攻击因子",
                "",
                "- 强度：weak/strong",
                "- 有效持续时间：short/long",
                "- 事件窗口：short/long",
                "- 未见组合：coordinated_drift_unseen",
                "- 受控模拟，不是真实生产攻击。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol_id": protocol["protocol_id"],
                "status": protocol["status"],
                "eligible_before": protocol["eligible_station_count_before_exclusion"],
                "eligible_after": protocol["eligible_station_count_after_exclusion"],
                "station_count": protocol["station_count"],
                "observation_access": False,
                "exploratory": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
