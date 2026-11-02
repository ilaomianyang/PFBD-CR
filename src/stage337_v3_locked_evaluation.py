from __future__ import annotations
"""Run the pre-frozen V3 attack stream with the fixed candidate family."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import stage334_exploratory_budgeted_ensemble_search as base  # noqa: E402
from stage326_simulated_attack_v2_evaluation import _internal_type  # noqa: E402

base.PROTOCOL_PATH = ROOT / "confirmatory_package" / "simulated_attack_v3" / "protocol_freeze.json"
base.OUT_JSON = ROOT / "tables" / "stage337_v3_locked_evaluation.json"
base.OUT_DETAIL = ROOT / "tables" / "stage337_v3_locked_evaluation_detail.csv"
base.OUT_SCENARIO = ROOT / "tables" / "stage337_v3_locked_evaluation_scenario.csv"
base.OUT_REPORT = ROOT / "reports" / "stage337_v3_locked_evaluation.md"
base.VALID_OFFSET = 950001
base.TEST_OFFSET = 960001
base.VIEWED_TEST = False
base.simbase._scenario_base_type = _internal_type

if __name__ == "__main__":
    raise SystemExit(base.main())
