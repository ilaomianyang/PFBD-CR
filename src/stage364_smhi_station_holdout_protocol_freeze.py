from __future__ import annotations

"""Stage 364: freeze an untouched SMHI station-holdout protocol.

This stage reads only the public SMHI parameter catalogues and station
metadata. It deliberately excludes the 18 stations used in Stage267 and
selects a deterministic, latitude-stratified holdout from the remaining
metadata-only eligible pool. No observation CSV or performance result is
requested or inspected here.
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
PREFIX = "stage364_smhi_station_holdout_protocol"
OLD_PROTOCOL = TABLES / "stage264_smhi_protocol_preregistration.json"

API_ROOT = "https://opendata-download-metobs.smhi.se/api"
API_VERSION_URL = f"{API_ROOT}/version/1.0.json"
PARAMETER_URL = f"{API_ROOT}/version/1.0/parameter/{{}}.json"
USER_AGENT = "TrustMeteo-QC Stage364 station holdout protocol"

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
LATITUDE_BANDS = [
    {"name": "south", "latitude_min": 55.0, "latitude_max_exclusive": 59.5},
    {"name": "middle", "latitude_min": 59.5, "latitude_max_exclusive": 64.0},
    {"name": "north", "latitude_min": 64.0, "latitude_max_exclusive": 69.5},
]
STATIONS_PER_BAND = 6


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str, retries: int = 5) -> tuple[bytes, dict[str, object]]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError(f"empty response: {url}")
            return payload, json.loads(payload.decode("utf-8"))
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2.0 + 2.0 * attempt)
    raise RuntimeError(f"SMHI metadata request failed: {url}") from last_error


def eligible_table(catalogues: dict[str, dict[str, object]]) -> pd.DataFrame:
    maps: dict[str, dict[str, dict[str, object]]] = {}
    for variable, catalogue in catalogues.items():
        rows = catalogue.get("station")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"invalid station catalogue: {variable}")
        maps[variable] = {
            str(row["id"]): row for row in rows if "id" in row
        }
    common = set.intersection(*(set(value) for value in maps.values()))
    rows: list[dict[str, object]] = []
    for station_id in sorted(common, key=int):
        records = {
            variable: maps[variable][station_id] for variable in maps
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
            COORDINATE_BOUNDS["latitude_min"] <= latitude
            <= COORDINATE_BOUNDS["latitude_max"]
            and COORDINATE_BOUNDS["longitude_min"] <= longitude
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
                "metadata_from_max_epoch_ms": max(
                    int(record["from"]) for record in records.values()
                ),
                "metadata_to_min_epoch_ms": min(
                    int(record["to"]) for record in records.values()
                ),
            }
        )
    return pd.DataFrame(rows)


def select_holdout(eligible: pd.DataFrame, old_ids: set[str]) -> pd.DataFrame:
    remaining = eligible[
        ~eligible["station_id"].astype(str).isin(old_ids)
    ].copy()
    if len(remaining) < STATIONS_PER_BAND * len(LATITUDE_BANDS):
        raise RuntimeError("too few untouched SMHI stations for holdout")
    chosen: list[pd.DataFrame] = []
    for band in LATITUDE_BANDS:
        pool = remaining[
            remaining["latitude"].ge(band["latitude_min"])
            & remaining["latitude"].lt(band["latitude_max_exclusive"])
        ].sort_values(["longitude", "station_id"], kind="stable")
        if len(pool) < STATIONS_PER_BAND:
            raise RuntimeError(f"too few holdout stations in {band['name']}")
        positions = sorted(
            {
                round(i * (len(pool) - 1) / (STATIONS_PER_BAND - 1))
                for i in range(STATIONS_PER_BAND)
            }
        )
        selected = pool.iloc[positions].copy()
        selected["latitude_band"] = band["name"]
        selected["band_pool_size_after_exclusion"] = int(len(pool))
        selected["rank_in_band"] = [
            positions.index(position) + 1 for position in positions
        ]
        chosen.append(selected)
    output = pd.concat(chosen, ignore_index=True)
    output["selection_rank"] = range(1, len(output) + 1)
    return output


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    old = json.loads(OLD_PROTOCOL.read_text(encoding="utf-8"))
    old_ids = {str(value) for value in old["station_ids"]}
    payload_hashes: dict[str, str] = {}
    catalogues: dict[str, dict[str, object]] = {}
    for variable, parameter_id in PARAMETERS.items():
        payload, catalogue = fetch_json(PARAMETER_URL.format(parameter_id))
        payload_hashes[variable] = sha256_bytes(payload)
        catalogues[variable] = catalogue
    version_payload, _ = fetch_json(API_VERSION_URL)
    eligible = eligible_table(catalogues)
    selected = select_holdout(eligible, old_ids)
    station_path = TABLES / f"{PREFIX}_stations.csv"
    selected[
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
            "metadata_from_max_epoch_ms",
            "metadata_to_min_epoch_ms",
        ]
    ].to_csv(station_path, index=False, encoding="utf-8")
    protocol = {
        "status": "PROTOCOL_FROZEN_BEFORE_OBSERVATION_ACCESS",
        "protocol_id": "TMQC-SMHI-STATION-HOLDOUT-20260924-v1",
        "provider": "Swedish Meteorological and Hydrological Institute (SMHI)",
        "scope": "new station holdout within the same SMHI provider",
        "observation_access": False,
        "performance_access": False,
        "selection_rule": (
            "Use only the official four-parameter metadata intersection, "
            "active status, 2021-01-01 through 2026-01-01 metadata coverage, "
            "and fixed coordinates. Exclude the 18 Stage267 station IDs, "
            "then sort remaining stations by longitude and numeric ID within "
            "three fixed latitude bands and choose six evenly spaced positions "
            "per band using round(i*(n-1)/5), i=0..5."
        ),
        "excluded_stage267_station_ids": sorted(old_ids, key=int),
        "eligible_station_count_before_exclusion": int(len(eligible)),
        "eligible_station_count_after_exclusion": int(
            len(eligible) - len(old_ids & set(eligible["station_id"].astype(str)))
        ),
        "station_count": int(len(selected)),
        "station_ids": selected["station_id"].astype(str).tolist(),
        "station_names": selected["station_name"].astype(str).tolist(),
        "latitude_bands": selected["latitude_band"].value_counts().to_dict(),
        "parameter_ids": PARAMETERS,
        "metadata_urls": {
            variable: PARAMETER_URL.format(parameter_id)
            for variable, parameter_id in PARAMETERS.items()
        },
        "api_version_url": API_VERSION_URL,
        "api_version_sha256": sha256_bytes(version_payload),
        "parameter_catalogue_sha256": payload_hashes,
        "stations_sha256": sha256_file(station_path),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "This is a new station boundary within one provider. It cannot "
            "establish a new independent institution and all attacks remain "
            "controlled simulations, not real incidents."
        ),
        "promotion_gate": (
            "Only a later locked evaluation may assess station-holdout "
            "performance. No threshold, feature, attack parameter, station, "
            "or baseline may be changed after this freeze."
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
                "# Stage364 SMHI 未使用站点保留集协议冻结",
                "",
                f"- 协议：`{protocol['protocol_id']}`",
                f"- 元数据合格站点：`{protocol['eligible_station_count_before_exclusion']}`",
                f"- 排除 Stage267 旧站点：`{len(old_ids)}`",
                f"- 新保留站点：`{protocol['station_count']}`",
                "- 观测值访问：否；性能结果访问：否。",
                "",
                "## 证据边界",
                "",
                "- 这是同一 SMHI 机构内的新站点边界，不是新增独立机构。",
                "- 站点由元数据和固定规则确定，未使用观测值或旧性能结果。",
                "- 后续必须先按本协议下载，再使用既有评价契约；不得在 test 上调参。",
                "- 攻击仍是透明受控模拟，不能写成真实事故召回率。",
                "",
                "## 产物",
                "",
                f"- `{output}`",
                f"- `{station_path}`",
                f"- `{report}`",
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
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
