from __future__ import annotations

"""Freeze a third unused SMHI station holdout from metadata only."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage377_smhi_third_station_holdout_protocol"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from stage369_smhi_factor_holdout_protocol_freeze import (  # noqa: E402
    BANDS,
    PARAMETERS,
    PARAMETER_URL,
    VERSION_URL,
    fetch,
    eligible,
    sha256_bytes,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
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
    second = json.loads(
        (TABLES / "stage369_smhi_factor_holdout_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    excluded = {
        *map(str, old["station_ids"]),
        *map(str, first["station_ids"]),
        *map(str, second["station_ids"]),
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
        if len(pool) < 6:
            raise RuntimeError(f"insufficient third-batch stations in {band['name']}")
        positions = sorted(
            {round((i + 0.5) * (len(pool) - 1) / 6) for i in range(6)}
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
        "protocol_id": "TMQC-SMHI-THIRD-STATION-HOLDOUT-20260924-v1",
        "provider": "Swedish Meteorological and Hydrological Institute (SMHI)",
        "scope": "third unused SMHI station holdout",
        "exploratory_boundary": (
            "The package is frozen before this batch's observations are read, "
            "but it was initiated after earlier holdout outcomes were observed. "
            "It is therefore exploratory and cannot replace V6."
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
            "Use official four-parameter metadata intersection, active status, "
            "fixed 2021-01-01 through 2026-01-01 metadata coverage and fixed "
            "coordinates. Exclude Stage267, Stage364 and Stage369 station IDs. "
            "Within each latitude band sort by longitude and numeric station ID "
            "and select six positions using round((i+0.5)*(n-1)/6), i=0..5."
        ),
        "attack_protocol": {
            "attack_types": [
                "spike",
                "offset",
                "drift",
                "time_shift",
                "stuck",
                "gaussian_noise",
            ],
            "intensity": ["weak", "strong"],
            "duration": ["short", "long"],
            "event_length": ["short", "long"],
            "unseen_combination": "coordinated_drift_unseen",
            "seeds": [3407, 6257, 7919, 104729, 20260715],
            "primary_far": 0.01,
            "primary_endpoint": "event_coverage",
        },
        "api_version_url": VERSION_URL,
        "api_version_sha256": sha256_bytes(version_raw),
        "parameter_catalogue_sha256": payload_hashes,
        "stations_sha256": sha256_file(station_path),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Same-provider station boundary; all attacks are controlled "
            "simulations and not real production incidents."
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
                "# Stage377 SMHI 第三批未使用站点协议冻结",
                "",
                f"- 状态：`{protocol['status']}`",
                f"- 元数据合格站点：`{protocol['eligible_station_count_before_exclusion']}`",
                f"- 排除既有站点：`{len(excluded)}`",
                f"- 剩余合格站点：`{protocol['eligible_station_count_after_exclusion']}`",
                f"- 冻结第三批站点：`{protocol['station_count']}`",
                "- 观测值访问：否；性能结果访问：否。",
                "- 该批属于同一 SMHI provider 的站点边界，且因启动时点属于探索性。",
                "- 后续不得在 test 结果上修改方法、阈值、攻击参数或评价脚本。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "protocol_id": protocol["protocol_id"],
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
