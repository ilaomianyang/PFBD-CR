from __future__ import annotations

"""Run the frozen V2 simulation package with the common Stage320 contract.

The wrapper changes only the protocol, seed offsets and output paths.  It
does not inspect test labels to choose a method or threshold.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage320_simulated_attack_confirmatory_evaluation as base  # noqa: E402

V2_PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v2"
V2_PROTOCOL = V2_PACKAGE / "protocol_freeze.json"
_ORIGINAL_INJECT = base.inject_simulated_attack


def _internal_type(name: str) -> str:
    mapping = {
        "low_spike": "spike",
        "low_offset": "offset",
        "slow_drift": "drift",
        "short_lag": "time_shift",
        "delay": "time_shift",
        "long_stuck": "stuck",
        "drop_replay": "stuck",
        "low_noise": "gaussian_noise",
        "coordinated_offset": "offset",
    }
    return mapping[str(name)]


def _inject_v2(frame, blocks, scenario_by_name):
    attacked = _ORIGINAL_INJECT(frame, blocks, scenario_by_name)
    attacked.loc[attacked["attack_scenario"].eq("delay"), "attack_type"] = "delay"
    attacked.loc[attacked["attack_scenario"].eq("drop_replay"), "attack_type"] = "drop"
    return attacked


def main() -> int:
    base.PROTOCOL_PATH = V2_PROTOCOL
    base.OUT_JSON = base.TABLES / "stage326_simulated_attack_v2_evaluation.json"
    base.OUT_EVAL = base.TABLES / "stage326_simulated_attack_v2_evaluation.csv"
    base.OUT_STATION = base.TABLES / "stage326_simulated_attack_v2_station.csv"
    base.OUT_TYPE = base.TABLES / "stage326_simulated_attack_v2_type.csv"
    base.OUT_SCENARIO = base.TABLES / "stage326_simulated_attack_v2_scenario.csv"
    base.OUT_SELECTION = base.TABLES / "stage326_simulated_attack_v2_selection.csv"
    base.OUT_FIT = base.TABLES / "stage326_simulated_attack_v2_fit_resources.csv"
    base.OUT_COMPARE = base.TABLES / "stage326_simulated_attack_v2_win_loss_matrix.csv"
    base.OUT_REPORT = base.REPORTS / "stage326_simulated_attack_v2_evaluation.md"
    base.VALID_OFFSET = 930001
    base.TEST_OFFSET = 940001
    base.BOOTSTRAP_SEED = 20260923
    base._scenario_base_type = _internal_type
    base.inject_simulated_attack = _inject_v2
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
