from __future__ import annotations

"""Audit the exploratory simulated-attack optimization without re-running data."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    stage208 = json.loads((TABLES / "stage208_dwd_persistent_repeat_protection.json").read_text(encoding="utf-8"))
    stage317 = json.loads((TABLES / "stage317_simulated_attack_optimization.json").read_text(encoding="utf-8"))
    rows = [r for r in stage208.get("aggregate", []) if r.get("selection_rule") == "valid_f1" and abs(float(r.get("max_far", -1)) - 0.01) < 1e-12]
    if len(rows) != 1:
        raise RuntimeError("expected one Stage208 valid_f1/FAR=0.01 row")
    ref = rows[0]
    cur = stage317["mean_test"]
    comparisons = {
        "f1_difference_stage317_minus_stage208": float(cur["f1"] - ref["test_f1"]),
        "event_coverage_difference_stage317_minus_stage208": float(cur["event_coverage"] - ref["test_event_coverage"]),
        "auprc_difference_stage317_minus_stage208": float(cur["auprc"] - ref["test_average_precision"]),
        "far_difference_stage317_minus_stage208": float(cur["far"] - ref["test_false_alarm_rate"]),
    }
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "stage": "stage318_simulated_attack_optimization_audit",
        "decision": "REJECT_STAGE317_AS_PRIMARY_CANDIDATE",
        "reason": "The pre-declared expert-fusion candidate is worse than the existing valid-selected repeat-protection candidate on DWD exploratory test at FAR=0.01.",
        "stage208_reference": {"selection_rule": ref["selection_rule"], "max_far": ref["max_far"], "test_f1": ref["test_f1"], "test_event_coverage": ref["test_event_coverage"], "test_average_precision": ref["test_average_precision"], "test_false_alarm_rate": ref["test_false_alarm_rate"]},
        "stage317_candidate": {"f1": cur["f1"], "event_coverage": cur["event_coverage"], "auprc": cur["auprc"], "far": cur["far"]},
        "comparisons": comparisons,
        "scientific_boundary": [
            "Stage317 remains exploratory because its historical DWD test was previously explored.",
            "The rejection is retained and does not alter Stage246 or any frozen primary result.",
            "Stage208 is not upgraded to universal superiority; it remains an existing bounded candidate with mixed cross-provider evidence.",
            "All attacks are transparent simulated injections on real observations, not production incident logs.",
        ],
        "input_sha256": {
            "stage208": sha256_file(TABLES / "stage208_dwd_persistent_repeat_protection.json"),
            "stage317": sha256_file(TABLES / "stage317_simulated_attack_optimization.json"),
        },
    }
    (TABLES / "stage318_simulated_attack_optimization_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Stage318 模拟攻击优化审计", "",
        "- 决策：**拒绝 Stage317 作为主候选**。",
        "- Stage317 五种子 test 均值：F1=`{:.6f}`，event coverage=`{:.6f}`，AUPRC=`{:.6f}`，FAR=`{:.6f}`。".format(cur["f1"], cur["event_coverage"], cur["auprc"], cur["far"]),
        "- Stage208 valid_f1/FAR=0.01 候选：F1=`{:.6f}`，event coverage=`{:.6f}`，AUPRC=`{:.6f}`，FAR=`{:.6f}`。".format(ref["test_f1"], ref["test_event_coverage"], ref["test_average_precision"], ref["test_false_alarm_rate"]),
        "- 结论：Stage317 的 F1 和事件覆盖均下降，不能用于升级全面领先声称。",
        "- 证据边界：两者均属于真实观测背景上的透明模拟攻击；该审计不证明真实生产攻击性能。",
    ]
    (REPORTS / "stage318_simulated_attack_optimization_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "decision": payload["decision"], "comparisons": comparisons}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
