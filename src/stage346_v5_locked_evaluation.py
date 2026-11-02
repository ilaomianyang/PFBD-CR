from __future__ import annotations
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
import stage340_v4_route_locked_evaluation as base  # noqa: E402
from stage326_simulated_attack_v2_evaluation import _internal_type  # noqa: E402
base.PROTOCOL_PATH=ROOT/"confirmatory_package"/"simulated_attack_v5"/"protocol_freeze.json"; base.OUT_JSON=ROOT/"tables"/"stage346_v5_locked_evaluation.json"; base.OUT_DETAIL=ROOT/"tables"/"stage346_v5_locked_evaluation_detail.csv"; base.OUT_SCENARIO=ROOT/"tables"/"stage346_v5_locked_evaluation_scenario.csv"; base.OUT_REPORT=ROOT/"reports"/"stage346_v5_locked_evaluation.md"; base.VALID_OFFSET=990001; base.TEST_OFFSET=1000001; base.PRIMARY_FAR=0.0075; base.simbase._scenario_base_type=_internal_type
if __name__=="__main__": raise SystemExit(base.main())
