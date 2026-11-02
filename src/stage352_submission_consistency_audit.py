"""Audit manuscript claims against the frozen V6 evaluation package.

This audit is descriptive only. It does not alter metrics, select a model, or
rewrite any prior result. The purpose is to identify stale claims before the
submission package is rebuilt.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import argparse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "manuscript_jisa_20260919.md"
V6_JSON = ROOT / "tables" / "stage349_v6_locked_evaluation.json"
V6_SCENARIO = ROOT / "tables" / "stage349_v6_locked_evaluation_scenario.csv"
V6_AUDIT = ROOT / "tables" / "stage351_v6_station_bh_audit.json"
V6_PROTOCOL = ROOT / "confirmatory_package" / "simulated_attack_v6" / "protocol_freeze.json"
OUT_JSON = ROOT / "tables" / "stage352_submission_consistency_audit.json"
OUT_MD = ROOT / "reports" / "stage352_submission_consistency_audit.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=MANUSCRIPT)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    manuscript_path = args.manuscript.resolve()
    out_json = ROOT / "tables" / f"stage352_submission_consistency_audit{args.output_suffix}.json"
    out_md = ROOT / "reports" / f"stage352_submission_consistency_audit{args.output_suffix}.md"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    v6 = read_json(V6_JSON)
    audit = read_json(V6_AUDIT)
    protocol = read_json(V6_PROTOCOL)

    with V6_SCENARIO.open("r", encoding="utf-8-sig", newline="") as fh:
        scenario_rows = list(csv.DictReader(fh))

    scenarios = sorted({row["scenario"] for row in scenario_rows})
    methods = sorted({row["method"] for row in scenario_rows})
    sources = sorted({row["source"] for row in scenario_rows})
    seeds = sorted({row["seed"] for row in scenario_rows})

    selected = audit["selected_summary"]
    scenario_summary = {
        row["candidate"]: {
            "beaten_pairs": row["beaten_pairs"],
            "total_pairs": row["total_pairs"],
            "scenario_win_share": row["scenario_win_share"],
        }
        for row in audit["scenario_summary"]
    }
    best = audit["best_candidate"]

    # The manuscript is checked for stale or missing V6-facing statements.
    checks: list[dict[str, Any]] = []

    def add_check(
        check_id: str,
        status: str,
        severity: str,
        message: str,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "status": status,
                "severity": severity,
                "message": message,
                "evidence": evidence or {},
            }
        )

    add_check(
        "v6_primary_candidate_named",
        "PASS" if best in manuscript else "FAIL",
        "P0",
        "V6 最佳候选是否在当前稿件中明确写为 trust_shape_union。",
        {"expected": best},
    )
    add_check(
        "v6_win_share_named",
        "PASS" if "73.7%" in manuscript or "73/99" in manuscript else "FAIL",
        "P0",
        "当前稿件是否报告 V6 的 73/99（73.7%）定义。",
        {"expected": "73/99=73.7%"},
    )
    add_check(
        "v6_scenario_failures_named",
        "PASS"
        if "coordinated_offset" in manuscript and "slow_drift" in manuscript
        else "FAIL",
        "P0",
        "当前稿件是否明确列出未达到单场景 70% 的 coordinated_offset 和 slow_drift。",
        {"expected": ["coordinated_offset", "slow_drift"]},
    )
    add_check(
        "v6_far_consistent",
        "FAIL" if "FAR = 0.01" in manuscript or "FAR=0.01" in manuscript else "PASS",
        "P0",
        "当前稿件是否仍把 FAR=0.01 写成 V6 主运行点；V6 主 FAR 为 0.0065。",
        {"v6_primary_far": protocol["primary_far"]},
    )
    add_check(
        "v6_provider_boundary_consistent",
        "FAIL"
        if (
            "NOAA ISD, DWD, MeteoSwiss, KNMI, and ECCC" in manuscript
            or "NOAA ISD, DWD, MeteoSwiss, KNMI and ECCC" in manuscript
        )
        else "PASS",
        "P0",
        "当前稿件是否把 DWD 与 V6 四个主 provider 混写成同一主结果。",
        {"v6_primary_sources": sources},
    )
    add_check(
        "v6_summary_values_present",
        "PASS"
        if "0.234894" in manuscript
        and "0.350292" in manuscript
        and "0.008470" in manuscript
        else "FAIL",
        "P0",
        "当前稿件是否包含 V6 选择运行的 F1、event coverage 和 FAR。",
        {
            "f1": selected["f1"],
            "event_coverage": selected["event_coverage"],
            "far": selected["false_alarm_rate"],
        },
    )
    add_check(
        "v6_attack_count",
        "PASS" if str(len(scenarios)) in manuscript else "FAIL",
        "P1",
        "当前稿件是否明确 V6 的攻击场景数量。",
        {"v6_attack_scenarios": scenarios},
    )
    add_check(
        "v6_method_set_present",
        "PASS"
        if "trust_shape_union" in manuscript
        and "ics_cps_spc_proxy" in manuscript
        else "FAIL",
        "P1",
        "当前稿件是否覆盖 V6 最终方法集合中的候选和 ICS/CPS 代理。",
        {"v6_method_count": len(methods), "v6_methods": methods},
    )
    protocol_baselines = protocol["method_freeze"]["baselines"]
    protocol_ocsvm = [m for m in protocol_baselines if m.startswith("one_class_svm_")]
    observed_ocsvm = sorted(
        method for method in methods if method.startswith("one_class_svm")
    )
    budget_reconciled_in_manuscript = (
        "20k fit rows" in manuscript
        and "20k/50k/100k" in manuscript
        and re.search(r"sensitivity\s+analysis", manuscript) is not None
    )
    add_check(
        "ocsvm_budget_reconciliation",
        "PASS"
        if len(protocol_ocsvm) == len(observed_ocsvm)
        or budget_reconciled_in_manuscript
        else "FAIL",
        "P0",
        "协议冻结中的 One-Class SVM 预算档位是否逐一出现在最终 V6 场景表中，或已在稿件中明确解释确认性预算与敏感性预算的差异。",
        {
            "protocol_budgets": protocol_ocsvm,
            "observed_methods": observed_ocsvm,
            "budget_reconciled_in_manuscript": budget_reconciled_in_manuscript,
        },
    )
    add_check(
        "old_stage_refs",
        "FAIL"
        if "Stage246/248" in manuscript
        and "historical frozen analysis" not in manuscript
        else "PASS",
        "P0",
        "当前稿件是否仍把 Stage246/248 写成唯一确认性主结果入口；若明确标为历史分析则通过。",
        {"expected_v6": "Stage349/350/351"},
    )
    add_check(
        "negative_boundary_present",
        "PASS"
        if "One-Class SVM" in manuscript
        and "FMI" in manuscript
        and (
            "online causal recovery" in manuscript
            or "Online versus offline recovery" in manuscript
            or "在线因果恢复" in manuscript
        )
        else "FAIL",
        "P0",
        "当前稿件是否保留 One-Class SVM、FMI 和在线因果恢复的负面边界。",
    )

    findings = [item for item in checks if item["status"] == "FAIL"]
    result = {
        "status": "NEEDS_MANUSCRIPT_REBUILD" if findings else "PASS",
        "audit_id": "TMQC-SUBMISSION-CONSISTENCY-20260923",
        "inputs": {
            "manuscript": str(manuscript_path.relative_to(ROOT)),
            "v6_evaluation": str(V6_JSON.relative_to(ROOT)),
            "v6_scenario": str(V6_SCENARIO.relative_to(ROOT)),
            "v6_station_audit": str(V6_AUDIT.relative_to(ROOT)),
            "v6_protocol": str(V6_PROTOCOL.relative_to(ROOT)),
        },
        "hashes": {
            "manuscript_sha256": sha256(manuscript_path),
            "v6_evaluation_sha256": sha256(V6_JSON),
            "v6_scenario_sha256": sha256(V6_SCENARIO),
            "v6_station_audit_sha256": sha256(V6_AUDIT),
            "v6_protocol_sha256": sha256(V6_PROTOCOL),
        },
        "v6_scope": {
            "providers": sources,
            "provider_count": len(sources),
            "seeds": seeds,
            "seed_count": len(seeds),
            "attack_scenarios": scenarios,
            "attack_scenario_count": len(scenarios),
            "methods_observed": methods,
            "method_count": len(methods),
            "best_candidate": best,
            "scenario_summary": scenario_summary,
            "selected_summary": selected,
        },
        "checks": checks,
        "finding_count": len(findings),
        "rule": "仅审计一致性；不修改实验结果，不升级确认性结论。",
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage352 投稿前一致性审计",
        "",
        f"- 状态：`{result['status']}`",
        f"- 审计 ID：`{result['audit_id']}`",
        "- 规则：只读审计，不修改实验结果，不把模拟攻击升级为真实攻击证据。",
        "",
        "## V6 范围",
        "",
        f"- 主 provider：{', '.join(sources)}（{len(sources)} 个）",
        f"- 攻击场景：{len(scenarios)} 个：{', '.join(scenarios)}",
        f"- 观测到的方法：{len(methods)} 个",
        f"- 最佳候选：`{best}`",
        f"- 场景×基线：{scenario_summary[best]['beaten_pairs']}/{scenario_summary[best]['total_pairs']} = {scenario_summary[best]['scenario_win_share']:.1%}",
        f"- 选择运行：F1={selected['f1']:.6f}，event coverage={selected['event_coverage']:.6f}，FAR={selected['false_alarm_rate']:.6f}",
        "",
        "## 检查结果",
        "",
        "| 检查 | 状态 | 严重度 | 说明 |",
        "|---|---|---|---|",
    ]
    for item in checks:
        lines.append(
            f"| `{item['id']}` | `{item['status']}` | `{item['severity']}` | {item['message']} |"
        )
    lines.extend(
        [
            "",
            "## 必须修正",
            "",
            "- 先解决所有 P0 `FAIL`，再重建摘要、方法、结果、讨论和补充材料。",
            "- One-Class SVM 三档预算必须明确确认性预算与敏感性预算的关系；不能让协议列出三档而最终场景表只出现一个未解释的汇总名。",
            "- V6 的 73.7% 必须写成 73/99 的总体场景×基线比例，并同时列出低于单场景 70% 的 coordinated_offset 和 slow_drift。",
            "",
            "## 输入哈希",
            "",
        ]
    )
    for key, value in result["hashes"].items():
        lines.append(f"- `{key}`：`{value}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "finding_count": len(findings), "out_json": str(out_json), "out_md": str(out_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
