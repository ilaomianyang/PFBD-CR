from __future__ import annotations

"""Evaluate the frozen V7 attack-factor stress-test package.

This wrapper deliberately reuses the frozen V6 detector/baseline implementation
and changes only the protocol, random streams, and output paths.  It never
overwrites V6 artifacts.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage340_v4_route_locked_evaluation as base  # noqa: E402


def _internal_type(name: str) -> str:
    mapping = {
        "weak_spike_short": "spike",
        "strong_spike_long": "spike",
        "weak_offset_short": "offset",
        "strong_offset_long": "offset",
        "short_drift": "drift",
        "long_drift": "drift",
        "short_lag": "time_shift",
        "long_delay": "time_shift",
        "short_stuck": "stuck",
        "long_stuck": "stuck",
        "weak_noise_short": "gaussian_noise",
        "strong_noise_long": "gaussian_noise",
        "coordinated_drift": "drift",
    }
    return mapping[str(name)]


base.PROTOCOL_PATH = ROOT / "confirmatory_package" / "simulated_attack_v7" / "protocol_freeze.json"
base.OUT_JSON = ROOT / "tables" / "stage358_v7_factor_locked_evaluation.json"
base.OUT_DETAIL = ROOT / "tables" / "stage358_v7_factor_locked_evaluation_detail.csv"
base.OUT_SCENARIO = ROOT / "tables" / "stage358_v7_factor_locked_evaluation_scenario.csv"
base.OUT_STATION = ROOT / "tables" / "stage358_v7_factor_locked_evaluation_station.csv"
base.OUT_REPORT = ROOT / "reports" / "stage358_v7_factor_locked_evaluation.md"
base.VALID_OFFSET = 1030001
base.TEST_OFFSET = 1040001
base.PRIMARY_FAR = 0.0065
base.simbase._scenario_base_type = _internal_type


if __name__ == "__main__":
    raise SystemExit(base.main())
