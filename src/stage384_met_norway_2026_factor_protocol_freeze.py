from __future__ import annotations

"""Freeze an unseen 2026 MET Norway temporal factor protocol."""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE_PROTOCOL = TABLES / "stage304_met_norway_protocol_preregistration.json"
SOURCE_STATIONS = TABLES / "stage304_met_norway_protocol_preregistration_stations.csv"
OUT_PREFIX = "stage384_met_norway_2026_factor_protocol"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenario_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for intensity, magnitude in (("weak", 0.8), ("strong", 4.0)):
        for duration, active_length in (("short", 3), ("long", 12)):
            for length_name, length in (("short", 8), ("long", 24)):
                name = f"offset_{intensity}_{duration}_{length_name}"
                rows.append(
                    {
                        "name": name,
                        "base_type": "offset",
                        "magnitude": magnitude,
                        "length": length,
                        "active_length": active_length,
                        "intensity": intensity,
                        "duration": duration,
                        "event_length": length_name,
                    }
                )
    rows.append(
        {
            "name": "coordinated_drift_unseen",
            "base_type": "drift",
            "start": 0.0,
            "end": 4.0,
            "length": 24,
            "active_length": 12,
            "station_count": 2,
            "coordinated": True,
            "unseen_combination": True,
        }
    )
    return rows


def main() -> int:
    source = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    stations = pd.read_csv(SOURCE_STATIONS, dtype={"station_id": str})
    station_ids = [str(value) for value in source["station_ids"]]
    if stations["station_id"].astype(str).tolist() != station_ids:
        raise RuntimeError("source station table does not match Stage304")
    scenarios = scenario_grid()
    payload = {
        "status": "PROTOCOL_FROZEN_BEFORE_2026_OBSERVATION_ACCESS",
        "protocol_id": "TMQC-MET-NORWAY-2026-TEMPORAL-FACTOR-20260924-v1",
        "provider": source["provider"],
        "dataset": source["dataset"],
        "network_id": source["network_id"],
        "scope": "unseen 2026 temporal factor holdout on the frozen MET Norway network",
        "station_ids": station_ids,
        "station_count": len(station_ids),
        "years": [2021, 2022, 2023, 2024, 2026],
        "split": {"train": [2021, 2022, 2023], "valid": [2024], "test": [2026]},
        "excluded_year": 2025,
        "test_period_start_utc": "2026-01-01T00:00:00Z",
        "test_period_end_utc": "2026-08-31T23:59:59Z",
        "test_period_is_partial_year": True,
        "primary_variable": source["primary_variable"],
        "primary_level_m": source["primary_level_m"],
        "quality_accepted_codes": source["quality_accepted_codes"],
        "coverage_rule": source["coverage_rule"],
        "quality_rule": source["quality_rule"],
        "attack_factor_protocol": {
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
            "scenarios": scenarios,
            "seeds": [3407, 6257, 7919, 104729, 20260715],
            "primary_far": 0.01,
            "primary_endpoint": "event_coverage",
        },
        "selection_boundary": (
            "Station IDs, years, split, variables, quality rule, attack factor grid, "
            "methods, thresholds, seeds and statistics are frozen before any 2026 "
            "observation value is accessed. Test labels are evaluation-only."
        ),
        "provider_boundary": (
            "This is a new time window on the same MET Norway provider and network, "
            "not a new provider or proof of station independence."
        ),
        "attack_boundary": "All attacks are controlled simulations, not production incidents.",
        "source_stage304_protocol_sha256": sha256(SOURCE_PROTOCOL),
        "source_stage304_stations_sha256": sha256(SOURCE_STATIONS),
        "observation_access": False,
        "performance_access": False,
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "exploratory": False,
        "frozen_at_utc": "2026-09-24T00:00:00Z",
    }
    out_json = TABLES / f"{OUT_PREFIX}.json"
    out_csv = TABLES / f"{OUT_PREFIX}_stations.csv"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(SOURCE_STATIONS, out_csv)
    report = [
        "# Stage384 MET Norway 2026 因子协议冻结",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 站点：`{payload['station_count']}`；训练：`2021-2023`；验证：`2024`；测试：`2026-01-01` 至 `2026-08-31`。",
        "- 2025 明确排除，避免复用已查看的 MET Norway 2025 test。",
        "- 因子：强度、持续时间、事件长度和 coordinated drift 未见组合。",
        "- 冻结时未访问 2026 观测值、标签或性能结果。",
        "- 所有攻击均为透明受控模拟，不是真实生产攻击。",
    ]
    (REPORTS / f"{OUT_PREFIX}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "station_count": len(station_ids), "scenario_count": len(scenarios)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
