from __future__ import annotations

"""Audit the updated 70%-baseline-level superiority gate for V2."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RESULT = TABLES / "stage326_simulated_attack_v2_evaluation.json"
SCENARIO = TABLES / "stage326_simulated_attack_v2_scenario.csv"
OUT_JSON = TABLES / "stage328_70pct_primary_gate_audit.json"
OUT_REPORT = REPORTS / "stage328_70pct_primary_gate_audit.md"


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    comparisons = pd.DataFrame(result["comparisons"])
    primary = comparisons[(comparisons["candidate"] == "trust_valid_f1") & (comparisons["metric"] == "event_coverage")].copy()
    f1 = comparisons[(comparisons["candidate"] == "trust_valid_f1") & (comparisons["metric"] == "f1")].copy()
    scenario = pd.read_csv(SCENARIO)
    methods = [m for m in result["methods"] if m != "trust_valid_f1" and m != "trust_nonstuck_event_macro"]
    pivot = scenario[scenario["method"].isin(["trust_valid_f1", *methods])].groupby(["scenario", "method"], as_index=False)["event_coverage"].mean().pivot(index="scenario", columns="method", values="event_coverage")
    per_scenario = []
    for name, row in pivot.iterrows():
        trust = float(row["trust_valid_f1"])
        baseline_values = row.drop(labels=["trust_valid_f1"]).astype(float)
        per_scenario.append({"scenario": str(name), "baselines_beaten": int((trust > baseline_values).sum()), "baseline_count": int(len(baseline_values)), "beat_share": float((trust > baseline_values).mean())})
    scenario_df = pd.DataFrame(per_scenario)
    supported_count = int(primary["supported"].astype(bool).sum())
    baseline_count = int(len(primary))
    f1_positive_count = int((f1["difference"] > 0).sum())
    scenario_pair_share = float((scenario_df["baselines_beaten"].sum()) / scenario_df["baseline_count"].sum())
    checks = [
        {"check": "baseline_level_70pct_gate", "passed": supported_count / baseline_count >= 0.70, "value": {"supported": supported_count, "total": baseline_count, "share": supported_count / baseline_count}},
        {"check": "trust_f1_point_positive_against_all_baselines", "passed": f1_positive_count == baseline_count, "value": {"positive": f1_positive_count, "total": baseline_count}},
        {"check": "scenario_level_70pct_gate", "passed": scenario_pair_share >= 0.70, "value": {"beaten_pairs": int(scenario_df["baselines_beaten"].sum()), "total_pairs": int(scenario_df["baseline_count"].sum()), "share": scenario_pair_share}},
    ]
    out = {
        "status": "PASS_70PCT_BASELINE_LEVEL_GATE_WITH_SCENARIO_LIMIT" if checks[0]["passed"] and checks[1]["passed"] and not checks[2]["passed"] else ("PASS_70PCT_AND_SCENARIO_GATE" if all(c["passed"] for c in checks) else "FAIL_70PCT_PRIMARY_GATE"),
        "claim_ceiling": "SIMULATED_ATTACK_70PCT_BASELINE_LEVEL_SUPERIORITY_ONLY",
        "checks": checks,
        "primary_supported_comparisons": primary[["baseline", "difference", "ci95_low", "ci95_high", "bh_q", "supported"]].to_dict("records"),
        "f1_comparisons": f1[["baseline", "difference", "ci95_low", "ci95_high", "bh_q", "supported"]].to_dict("records"),
        "scenario_level": per_scenario,
        "limitations": [
            "70% 门槛按主要基线级主终点比较定义，不等同于每个攻击类型都领先。",
            "按攻击场景×基线的原始胜出比例低于 70%，因此不能声称全攻击类型全面领先。",
            "One-Class SVM、PCA、kNN 的 event coverage 仍未通过确认性非劣效或多重比较门槛。",
            "所有结果仍是预注册模拟攻击，不是真实生产攻击召回率。",
        ],
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Stage328 70% 基线级领先门槛审计",
        "",
        f"- 状态：{out['status']}",
        f"- 当前主张上限：{out['claim_ceiling']}",
        f"- 主要基线级 event coverage：{supported_count}/{baseline_count} = {supported_count / baseline_count:.1%}",
        f"- Trust F1 点估计为正：{f1_positive_count}/{baseline_count}",
        f"- 攻击场景×基线原始胜出比例：{scenario_pair_share:.1%}",
        "",
        "## 统计结论",
        "",
        "V2 可以支持‘在预定义模拟攻击协议下，对至少 70% 的主要基线达到统计支持的主终点优势’这一窄化主张；不能升级为全攻击类型或所有基线全面领先。",
        "",
        "## 主要未通过比较",
        "",
    ]
    for row in out["primary_supported_comparisons"]:
        if not row["supported"]:
            lines.append(f"- {row['baseline']}：差值 {row['difference']:.6f}，95% CI [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]，BH q={row['bh_q']:.6f}。")
    lines.extend(["", "## 场景层限制", "", scenario_df.to_markdown(index=False), "", "- 场景层胜出比例低于 70%，应在论文中作为攻击类型适用域限制报告。", ""])
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": out["status"], "baseline_share": supported_count / baseline_count, "scenario_pair_share": scenario_pair_share}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
