from __future__ import annotations

"""Final acceptance gate for the updated 70%-baseline-level objective."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL = ROOT / "confirmatory_package" / "simulated_attack_v2" / "protocol_freeze.json"
V2_AUDIT = TABLES / "stage327_simulated_attack_v2_final_audit.json"
GATE = TABLES / "stage328_70pct_primary_gate_audit.json"
FUSION = TABLES / "stage330_exploratory_fusion_audit.json"
OUT_JSON = TABLES / "stage331_final_acceptance_gate.json"
OUT_REPORT = REPORTS / "stage331_final_acceptance_gate.md"


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    v2 = json.loads(V2_AUDIT.read_text(encoding="utf-8"))
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    fusion = json.loads(FUSION.read_text(encoding="utf-8"))
    checks = [
        {"check": "simulation_protocol_frozen", "passed": protocol.get("status") == "SIMULATION_PROTOCOL_V2_FROZEN_BEFORE_PERFORMANCE_EVALUATION" and protocol.get("provider_count") == 4 and len(protocol.get("attack_scenarios", [])) == 9},
        {"check": "unified_v2_evaluation_audited", "passed": v2.get("status") == "PASS_V2_SCIENTIFIC_AUDIT_CLAIM_CEILING" and v2.get("passed_checks") == v2.get("total_checks")},
        {"check": "fair_statistics_and_resources", "passed": any(row.get("check") == "resource_fields_complete" and row.get("passed") for row in v2.get("checks", [])) and any(row.get("check") == "fair_method_seed_provider_contract" and row.get("passed") for row in v2.get("checks", []))},
        {"check": "seventy_percent_primary_gate", "passed": gate.get("status") == "PASS_70PCT_BASELINE_LEVEL_GATE_WITH_SCENARIO_LIMIT" and gate.get("checks", [])[0].get("passed") and gate.get("checks", [])[1].get("passed")},
        {"check": "algorithm_optimization_attempt_audited", "passed": fusion.get("status") == "PASS_EXPLORATORY_CONSTRAINED_FUSION_AUDIT" and fusion.get("passed_checks") == fusion.get("total_checks")},
        {"check": "negative_results_and_claim_downgrade", "passed": len(v2.get("unsupported_event_comparisons", [])) >= 3 and v2.get("claim_ceiling") == "SIMULATED_ATTACK_LIMITED_SUPERIORITY_ONLY"},
    ]
    passed = sum(bool(row["passed"]) for row in checks)
    out = {
        "status": "PASS_70PCT_TARGET_WITH_EXPLICIT_SCENARIO_LIMIT" if passed == len(checks) else "FAIL_FINAL_ACCEPTANCE_GATE",
        "claim_ceiling": "SIMULATED_ATTACK_70PCT_BASELINE_LEVEL_SUPERIORITY_ONLY",
        "checks": checks,
        "passed_checks": passed,
        "total_checks": len(checks),
        "accepted_claim": "在四个冻结 provider 的真实观测背景和预注册模拟攻击协议下，TrustMeteo-QC 的 trust_valid_f1 方案对 11 个主要基线中的 8 个（72.7%）达到经 BH 校正的 event coverage 优势；该结论不等同于所有攻击类型或真实生产攻击全面领先。",
        "open_limits": [
            "攻击场景×基线原始胜出比例为 58.6%，不能声称全攻击类型全面领先。",
            "One-Class SVM、PCA、kNN 的主终点比较未全部通过。",
            "约束式融合提升仅为探索性，需新的独立锁定窗口才能确认。",
            "所有攻击均为模拟注入，不能写成真实生产事故召回率。",
        ],
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_REPORT.write_text("\n".join(["# Stage331 最终验收门", "", f"- 状态：{out['status']}", f"- 当前主张上限：{out['claim_ceiling']}", f"- 验收通过：{passed}/{len(checks)}", "", "## 接受主张", "", out["accepted_claim"], "", "## 未关闭限制", "", *[f"- {item}" for item in out["open_limits"]], ""]), encoding="utf-8")
    print(json.dumps({"status": out["status"], "passed_checks": passed, "total_checks": len(checks), "claim_ceiling": out["claim_ceiling"]}, ensure_ascii=False))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
