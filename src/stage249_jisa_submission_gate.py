from __future__ import annotations

"""Stage 249: final JISA submission gate for TrustMeteo-QC.

This stage performs no training, threshold selection, feature selection, or
test-set re-evaluation.  It reads the frozen evidence package and evaluates
the submission objective requirement by requirement.  The gate is deliberately
claim-proportional: a requirement passes only when the frozen evidence supports
the stated interpretation and the required negative boundaries are retained.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage249_jisa_submission_gate"

INPUTS = {
    "stage246": TABLES / "stage246_frozen_submission_package.json",
    "stage248": TABLES / "stage248_cross_provider_fair_baseline_audit.json",
    "stage250": TABLES / "stage250_jisa_risk_closure_audit.json",
    "stage29": TABLES / "stage29_evidence_matrix.json",
    "stage10": TABLES / "stage10_paper_readiness_audit.json",
    "stage112": TABLES / "stage112_full_artifact_manifest.json",
    "stage114": TABLES / "stage114_recent_semantic_sanity_audit.json",
    "stage115": TABLES / "stage115_protocol_boundary_audit.json",
    "README": ROOT / "README.md",
}

REQUIRED_TABLE8 = {
    "stage109",
    "stage110",
    "stage116",
    "stage142",
    "stage153",
    "stage157",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    missing = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing submission gate inputs: {missing}")

    s246 = load_json(INPUTS["stage246"])
    s248 = load_json(INPUTS["stage248"])
    s250 = load_json(INPUTS["stage250"])
    s29 = load_json(INPUTS["stage29"])
    s10 = load_json(INPUTS["stage10"])
    s112 = load_json(INPUTS["stage112"])
    s114 = load_json(INPUTS["stage114"])
    s115 = load_json(INPUTS["stage115"])
    s230 = load_json(TABLES / "stage230_cross_provider_evidence_synthesis.json")
    s232 = load_json(TABLES / "stage232_attack_type_robustness.json")
    s233 = load_json(TABLES / "stage233_valid_attack_correction_audit.json")
    s242 = load_json(TABLES / "stage242_multisource_transfer_matrix.json")
    s244 = load_json(TABLES / "stage244_cross_provider_nonstuck_candidate_features.json")
    s245 = load_json(TABLES / "stage245_cross_provider_nonstuck_seed_replication.json")

    table8 = pd.read_csv(
        TABLES / "stage246_frozen_submission_package_table8_natural_validation.csv"
    )
    table7 = pd.read_csv(
        TABLES / "stage246_frozen_submission_package_table7_fair_baseline.csv"
    )
    evidence_rows = s29.get("rows", [])
    evidence_ids = {str(row.get("evidence_id", "")) for row in evidence_rows}

    fair_valid_f1 = [
        row
        for row in s248.get("comparison_rows", [])
        if row.get("candidate") == "trust_valid_f1"
        and row.get("baseline")
        in {"robust_max_deviation", "isolation_forest"}
    ]
    fair_nonstuck_event = [
        row
        for row in s248.get("comparison_rows", [])
        if row.get("candidate") == "trust_nonstuck_event_macro"
        and row.get("baseline")
        in {"robust_max_deviation", "isolation_forest"}
    ]
    fair_svm = [
        row
        for row in s248.get("comparison_rows", [])
        if row.get("baseline") == "one_class_svm"
    ]

    checks = [
        {
            "requirement": "风险闭合与边界分层",
            "status": "PASS"
            if s250.get("status") == "PASS_RISK_CLOSURE_AUDIT"
            and int(s250.get("risk_count", 0)) == 7
            and int(s250.get("resolved_count", 0)) == 0
            and int(s250.get("bounded_count", 0)) == 7
            and all(
                row.get("disposition") == "BOUNDED"
                for row in s250.get("risks", [])
            )
            else "FAIL",
            "evidence": [
                "stage250_jisa_risk_closure_audit.json",
                "stage250_jisa_risk_closure_audit_risks.csv",
            ],
            "interpretation": (
                "所有剩余风险均记录为边界结果；"
                "任何未被独立验证的结论不得升级为主优势。"
            ),
        },
        {
            "requirement": "检测—风险评分—可信恢复闭环",
            "status": "PASS"
            if s246.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED"
            and s10.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED"
            and len(s10.get("review_items", [1])) == 0
            else "FAIL",
            "evidence": [
                "stage10_paper_readiness_audit.json",
                "stage246_frozen_submission_package.json",
                "stage246_frozen_submission_package_table5_recovery_safety.csv",
            ],
            "interpretation": (
                "冻结主表同时覆盖检测/风险运行点与因果恢复安全边界；"
                "恢复弱于离线双向插值仍被保留。"
            ),
        },
        {
            "requirement": "透明受控注入与低误报主终点FAR=0.01",
            "status": "PASS"
            if float(s246.get("primary_far", 0.0)) == 0.01
            and "透明受控注入" in str(s246.get("protocol", ""))
            and "不是真实攻击日志"
            in " ".join(s246.get("limitations", []))
            else "FAIL",
            "evidence": [
                "stage246_frozen_submission_package.json",
                "stage246_frozen_submission_package_table1_primary_summary.csv",
            ],
            "interpretation": (
                "主终点固定为FAR=0.01；实验攻击是透明受控注入，"
                "不是真实入侵日志。"
            ),
        },
        {
            "requirement": "相对公平基线稳定领先或互补",
            "status": "PASS"
            if len(fair_valid_f1) == 2
            and all(
                float(row["f1_difference"]) > 0.0
                and float(row["f1_ci95_low"]) > 0.0
                and float(row["f1_bootstrap_bh_q"]) <= 0.05
                and bool(row["f1_primary_supported"])
                for row in fair_valid_f1
            )
            and len(fair_nonstuck_event) == 2
            and all(
                float(row["event_coverage_difference"]) > 0.0
                and float(row["event_ci95_low"]) > 0.0
                and float(row["event_coverage_bootstrap_bh_q"]) <= 0.05
                and bool(row["event_coverage_primary_supported"])
                for row in fair_nonstuck_event
            )
            and len(fair_svm) == 2
            and any(
                float(row["event_coverage_difference"]) < 0.0
                for row in fair_svm
            )
            and any(
                "不能宣称对所有公平基线全面领先" in item
                for item in s248.get("limitations", [])
            )
            else "FAIL",
            "evidence": [
                "stage248_cross_provider_fair_baseline_audit.json",
                "stage246_frozen_submission_package_table7_fair_baseline.csv",
            ],
            "interpretation": (
                "Trust(valid_f1)相对Robust与Isolation Forest的F1差异"
                "经BH校正后仍为正；非stuck事件覆盖同样稳定为正；"
                "One-Class SVM保留为互补边界。"
            ),
        },
        {
            "requirement": "优势不由单一机构、站点、种子或攻击类型驱动",
            "status": "PASS"
            if int(s248.get("provider_count", 0)) == 4
            and int(s248.get("seed_count", 0)) == 5
            and len(s248.get("leave_one_provider_rows", [])) == 24
            and all(
                bool(row.get("f1_direction_positive"))
                for row in s248.get("leave_one_provider_rows", [])
                if row.get("candidate") == "trust_valid_f1"
                and row.get("baseline")
                in {"robust_max_deviation", "isolation_forest"}
            )
            and all(
                bool(row.get("event_direction_positive"))
                for row in s248.get("leave_one_provider_rows", [])
                if row.get("candidate") == "trust_nonstuck_event_macro"
                and row.get("baseline")
                in {"robust_max_deviation", "isolation_forest"}
            )
            and any(
                "F1在移除ECCC后转负" in item
                for item in s246.get("limitations", [])
            )
            and any(
                "优势主要集中在stuck" in item
                for item in s246.get("limitations", [])
            )
            else "FAIL",
            "evidence": [
                "stage248_cross_provider_fair_baseline_audit_leave_one_provider_out.csv",
                "stage247_provider_balanced_nonstuck_audit.json",
                "stage232_attack_type_robustness.json",
            ],
            "interpretation": (
                "公平基线的F1与非stuck事件覆盖在四个机构逐机构留一后"
                "保持正方向；同时如实保留移除ECCC后非stuck F1转负、"
                "以及优势集中在stuck的边界。"
            ),
        },
        {
            "requirement": "特征、窗口、权重可迁移而绝对阈值不可迁移",
            "status": "PASS"
            if int(s230.get("direct_threshold_transfer_failure_count", 0)) == 4
            and int(s230.get("target_valid_selected_f1_advantage_count", 0)) == 4
            and int(s230.get("target_valid_selected_event_advantage_count", 0)) == 4
            and str(s233.get("old_claim_status")) == "INVALIDATED"
            and int(s242.get("pairwise_count", 0)) == 20
            and int(s242.get("pairwise_local_positive", 0)) == 18
            and int(s242.get("negative_pair_count", 0)) == 2
            and any("绝对" in str(item) for item in s230.get("limitations", []))
            else "FAIL",
            "evidence": [
                "stage230_cross_provider_evidence_synthesis.json",
                "stage233_valid_attack_correction_audit.json",
                "stage242_multisource_transfer_matrix.json",
            ],
            "interpretation": (
                "冻结证据区分可迁移的特征/窗口/权重与不可迁移的绝对分数阈值；"
                "目标域必须只用valid正常样本重校准。"
            ),
        },
        {
            "requirement": "非stuck攻击与负反例保留",
            "status": "PASS"
            if any(
                float(row.get("row_recall_difference", 0.0)) < 0.0
                for row in s244.get("nonstuck_type_equal_rows", [])
                if row.get("attack_type") == "drift"
            )
            and any(
                float(row.get("row_recall_difference", 0.0)) < 0.0
                for row in s245.get("nonstuck_type_equal_rows", [])
                if row.get("attack_type") == "drift"
            )
            and any(
                float(row.get("row_recall_difference", 0.0)) < 0.0
                and float(row.get("event_coverage_difference", 0.0)) < 0.0
                for row in s232.get("leave_one_attack_type_out_rows", [])
                if row.get("removed_attack_type") == "stuck"
            )
            and any(
                "区间仍跨零" in item for item in s246.get("limitations", [])
            )
            and any(
                "优势主要集中在stuck" in item
                for item in s246.get("limitations", [])
            )
            else "FAIL",
            "evidence": [
                "stage244_cross_provider_nonstuck_candidate_features.json",
                "stage245_cross_provider_nonstuck_seed_replication.json",
                "stage232_attack_type_robustness.json",
            ],
            "interpretation": (
                "非stuck扩展与独立种子复现保留方向性和区间跨零的边界；"
                "drift等反例不被删除，不能写成所有攻击类型全面领先。"
            ),
        },
        {
            "requirement": "真实/半真实异常与弱标签边界",
            "status": "PASS"
            if len(table8) == 6
            and set(table8["source_stage"]) == REQUIRED_TABLE8
            and int(
                table8["supported_interpretation"]
                .astype(str)
                .str.contains(
                    "not attack recall|not attack ground truth",
                    regex=True,
                )
                .sum()
            )
            >= 2
            else "FAIL",
            "evidence": [
                "stage246_frozen_submission_package_table8_natural_validation.csv",
                "stage107_noaa_natural_svm_validation.json",
                "stage109_noaa_quality_weak_label_bootstrap.json",
                "stage110_noaa_quality_hierarchical_bootstrap.json",
                "stage157_noaa_natural_two_way_cluster_bootstrap.json",
            ],
            "interpretation": (
                "NOAA自然观测用于自然运行与弱标签关联验证；"
                "官方质量标记不是攻击真值，自然报警率不能解释为攻击召回率。"
            ),
        },
        {
            "requirement": "离线与因果恢复安全边界",
            "status": "PASS"
            if any(
                "严格因果恢复弱于离线双向插值" in item
                for item in s246.get("limitations", [])
            )
            and any(
                row.get("check") == "causal_recovery_boundary_present"
                and row.get("status") == "PASS"
                for row in s246.get("checks", [])
            )
            else "FAIL",
            "evidence": [
                "stage133_causal_temporal_recovery.json",
                "stage134_causal_recovery_optimization.json",
                "stage246_frozen_submission_package_table5_recovery_safety.csv",
            ],
            "interpretation": (
                "严格因果恢复明确弱于离线双向插值；"
                "不能把离线恢复结果写成在线能力。"
            ),
        },
        {
            "requirement": "四机构、多指标与可复现材料",
            "status": "PASS"
            if int(s246.get("provider_count", 0)) >= 4
            and int(s246.get("frozen_input_count", 0)) == 31
            and int(s246.get("generated_deliverable_count", 0)) == 28
            and s112.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED"
            and len(s112.get("failures", [1])) == 0
            and (s115.get("failures") in (None, [], ()))
            and (s114.get("failures") in (None, [], ()))
            and int(s29.get("counts", {}).get("evidence_items", 0))
            == len(s29.get("rows", []))
            and len(evidence_ids) == len(s29.get("rows", []))
            else "FAIL",
            "evidence": [
                "stage246_frozen_submission_package.json",
                "stage112_full_artifact_manifest.json",
                "stage115_protocol_boundary_audit.json",
                "stage114_recent_semantic_sanity_audit.json",
                "stage29_evidence_matrix.json",
            ],
            "interpretation": (
                "冻结包覆盖四个目标机构、31个冻结输入、28个交付物，"
                "证据矩阵条目与自报计数一致且ID唯一，"
                "文件清单、语义审计和协议边界均通过。"
            ),
        },
    ]

    failed = [row for row in checks if row["status"] != "PASS"]
    payload = {
        "status": "PASS_JISA_SUBMISSION_GATE" if not failed else "REVIEW_REQUIRED",
        "protocol": (
            "阶段249只读取阶段246冻结投稿包、阶段248公平基线审计、"
            "阶段29证据矩阵和阶段10/112/114/115审计结果，逐条检查目标要求；"
            "不训练、不重新选择、不重新评价任何test集。"
        ),
        "target_journal": "Journal of Information Security and Applications (JISA)",
        "primary_far": 0.01,
        "requirements_total": len(checks),
        "requirements_passed": len(checks) - len(failed),
        "requirements_failed": len(failed),
        "checks": checks,
        "failed_requirements": failed,
        "claim_boundaries": [
            "所有攻击均为透明受控注入，不是真实攻击日志。",
            "One-Class SVM是互补边界，不宣称对所有公平基线全面领先。",
            "非stuck的F1在移除ECCC后转负，事件覆盖留一方向较稳。",
            "优势主要集中在stuck，不能写成所有攻击类型普遍领先。",
            "NOAA质量标记是弱标签，不是攻击真值。",
            "严格因果恢复弱于离线双向插值，不能写成在线等同离线。",
            "只有四个跨机构目标，机构级精确检验功效有限。",
        ],
        "input_hashes": {
            name: sha256_file(path) for name, path in INPUTS.items()
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(checks).to_csv(
        TABLES / f"{PREFIX}_requirements.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rows = [
        "# 阶段249 JISA投稿门审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 目标期刊：`{payload['target_journal']}`",
        f"- 主终点：FAR=`{payload['primary_far']}`",
        f"- 要求通过：`{payload['requirements_passed']}` / `{payload['requirements_total']}`",
        "",
        "## 逐条要求",
        "",
        "| 要求 | 状态 | 证据 |",
        "|---|---|---|",
    ]
    for row in checks:
        rows.append(
            f"| {row['requirement']} | {row['status']} | "
            + "; ".join(row["evidence"])
            + " |"
        )
    rows += ["", "## 解释边界", ""]
    rows.extend(f"- {item}" for item in payload["claim_boundaries"])
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "requirements_passed": payload["requirements_passed"],
                "requirements_total": payload["requirements_total"],
                "failed": len(failed),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
