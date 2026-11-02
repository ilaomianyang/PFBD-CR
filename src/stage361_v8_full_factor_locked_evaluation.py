from __future__ import annotations

"""Evaluate the V8 full-factor stress-test package without changing V6."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage340_v4_route_locked_evaluation as base  # noqa: E402
import stage320_simulated_attack_confirmatory_evaluation as simbase  # noqa: E402


def _internal_type(name: str) -> str:
    if str(name) == "coordinated_drift_unseen":
        return "drift"
    return "offset"


_ORIGINAL_INJECT = simbase.inject_simulated_attack


def _inject_v8(frame: pd.DataFrame, blocks: pd.DataFrame, scenario_by_name: dict) -> pd.DataFrame:
    """Inject the attack only on active rows, leaving event-window context intact."""
    attacked = _ORIGINAL_INJECT(frame, blocks, scenario_by_name)
    clean = pd.to_numeric(frame["temperature_c"], errors="coerce").to_numpy(dtype=float)
    for (station_id, scenario_name), part in blocks.groupby(["station_id", "scenario"], sort=True):
        spec = scenario_by_name[str(scenario_name)]
        active_length = min(int(spec.get("active_length", spec["length"])), len(part))
        positions = part["row_position"].to_numpy(dtype=int)
        inactive = positions[active_length:]
        if len(inactive):
            attacked.loc[inactive, "temperature_c"] = clean[inactive]
            attacked.loc[inactive, "observed_value"] = clean[inactive]
            attacked.loc[inactive, "is_attack"] = False
            attacked.loc[inactive, "attack_type"] = "normal"
            attacked.loc[inactive, "attack_scenario"] = "normal"
            attacked.loc[inactive, "attack_event_id"] = ""
    return attacked


base.PROTOCOL_PATH = ROOT / "confirmatory_package" / "simulated_attack_v8" / "protocol_freeze.json"
base.OUT_JSON = ROOT / "tables" / "stage361_v8_full_factor_locked_evaluation.json"
base.OUT_DETAIL = ROOT / "tables" / "stage361_v8_full_factor_locked_evaluation_detail.csv"
base.OUT_SCENARIO = ROOT / "tables" / "stage361_v8_full_factor_locked_evaluation_scenario.csv"
base.OUT_STATION = ROOT / "tables" / "stage361_v8_full_factor_locked_evaluation_station.csv"
base.OUT_REPORT = ROOT / "reports" / "stage361_v8_full_factor_locked_evaluation.md"
base.VALID_OFFSET = 1050001
base.TEST_OFFSET = 1060001
base.PRIMARY_FAR = 0.0065
base.simbase._scenario_base_type = _internal_type
base.simbase.inject_simulated_attack = _inject_v8


if __name__ == "__main__":
    raise SystemExit(base.main())
