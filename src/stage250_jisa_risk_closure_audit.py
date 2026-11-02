from __future__ import annotations

"""Stage 250: JISA risk-closure and claim-boundary audit.

Stage249 checks whether the frozen JISA evidence package satisfies the
predeclared submission requirements.  It does not answer a second question:
which remaining risks are accepted only as boundaries, which risks are
resolved by an independent confirmatory result, and which claims must never
enter the main contribution.

This stage freezes that distinction.  It performs no model fitting,
threshold selection, feature selection, or test-set re-evaluation.  Every
risk is mapped to existing evidence, a disposition, and an explicit allowed
claim boundary.  A risk can be marked ``RESOLVED`` only when the existing
evidence supports it under independent validation.  If the evidence is
mixed, the risk remains ``BOUNDED`` and the corresponding result cannot be
promoted to a main claim.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage250_jisa_risk_closure_audit"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def risk_rows() -> list[dict[str, object]]:
    s246 = load_json(TABLES / "stage246_frozen_submission_package.json")
    s248 = load_json(TABLES / "stage248_cross_provider_fair_baseline_audit.json")
    s249 = load_json(TABLES / "stage249_jisa_submission_gate.json")
    s244 = load_json(
        TABLES / "stage244_cross_provider_nonstuck_candidate_features.json"
    )
    s245 = load_json(
        TABLES / "stage245_cross_provider_nonstuck_seed_replication.json"
    )

    return [
        {
            "risk_id": "R1_natural_to_controlled_gap",
            "risk": "所有攻击结果均来自透明受控注入，不能替代真实攻击日志。",
            "existing_evidence": [
                "stage246_frozen_submission_package.json",
                "stage246_frozen_submission_package_table8_natural_validation.csv",
            ],
            "observed_status": (
                "受控攻击证据完整；自然观测证据只支持关联和弱标签边界。"
            ),
            "disposition": "BOUNDED",
            "allowed_main_claim": (
                "可以声称在预定义透明攻击模型下有效；不能声称真实入侵普遍领先。"
            ),
            "forbidden_claim": (
                "不能把自然观测异常率、官方质量标记或弱标签写成攻击召回率。"
            ),
            "next_action": (
                "若后续获得真实故障或攻击日志，单独开新阶段；当前不阻塞投稿。"
            ),
        },
        {
            "risk_id": "R2_nonstuck_instability",
            "risk": "非stuck优势方向一致但区间跨零，移除ECCC后F1转负。",
            "existing_evidence": [
                "stage244_cross_provider_nonstuck_candidate_features.json",
                "stage245_cross_provider_nonstuck_seed_replication.json",
                "stage247_provider_balanced_nonstuck_audit.json",
            ],
            "observed_status": (
                f"阶段244 F1差值={s244['hierarchical_bootstrap']['f1_difference']:.6f}，"
                f"95%CI=[{s244['hierarchical_bootstrap']['f1_ci95_low']:.6f},"
                f"{s244['hierarchical_bootstrap']['f1_ci95_high']:.6f}]；"
                f"阶段245 F1差值={s245['hierarchical_bootstrap']['f1_difference']:.6f}，"
                f"95%CI=[{s245['hierarchical_bootstrap']['f1_ci95_low']:.6f},"
                f"{s245['hierarchical_bootstrap']['f1_ci95_high']:.6f}]；"
                "移除ECCC后F1方向转负。"
            ),
            "disposition": "BOUNDED",
            "allowed_main_claim": (
                "可以写成扩展分析中非stuck事件覆盖方向一致的证据；"
                "主优势仍以stuck为主，非stuck为边界扩展。"
            ),
            "forbidden_claim": (
                "不能宣称非stuck F1稳定显著优于基线，或优势不由单一机构驱动。"
            ),
            "next_action": (
                "当前停止继续在test上调参；等待新的独立机构或数据后再复核。"
            ),
        },
        {
            "risk_id": "R3_one_class_svm_complementarity",
            "risk": (
                "One-Class SVM训练子采样是计算预算约束，"
                "Trust相对其F1与事件覆盖呈互补而非全面领先。"
            ),
            "existing_evidence": [
                "stage248_cross_provider_fair_baseline_audit.json",
                "stage248_cross_provider_fair_baseline_audit_comparisons.csv",
                "stage248_cross_provider_fair_baseline_audit_fit_rows.csv",
                "stage254_ocsvm_training_scale_sensitivity.json",
                "stage254_ocsvm_training_scale_sensitivity_provider_equal.csv",
            ],
            "observed_status": (
                "相对Robust和Isolation Forest的F1优势区间为正；"
                "阶段254将One-Class SVM训练规模扩展到20k、50k、100k，"
                "四机构事件覆盖均值仅从0.413690变化到0.410498，"
                "NOAA仍接近失效，互补结论未改变。"
            ),
            "disposition": "BOUNDED",
            "allowed_main_claim": (
                "可以声称相对Robust max deviation和Isolation Forest"
                "在冻结低FAR终点上稳定；相对One-Class SVM为互补。"
            ),
            "forbidden_claim": (
                "不能宣称对所有公平基线全面领先，"
                "也不能省略One-Class SVM的训练子采样边界。"
            ),
            "next_action": (
                "保留当前冻结结论，并在论文中报告阶段254的训练规模、"
                "实际可用行数和全部档位；不修改当前主结果。"
            ),
        },
        {
            "risk_id": "R4_provider_and_station_dependence",
            "risk": (
                "只有四个外部目标机构；非stuck F1在移除ECCC后转负，"
                "存在机构依赖风险。"
            ),
            "existing_evidence": [
                "stage248_cross_provider_fair_baseline_audit_leave_one_provider_out.csv",
                "stage247_provider_balanced_nonstuck_audit_leave_one_provider_out.csv",
            ],
            "observed_status": (
                "Robust和Isolation Forest方向的留一机构检查保留，"
                "但非stuck F1仍不允许写成机构无关。"
            ),
            "disposition": "BOUNDED",
            "allowed_main_claim": (
                "可以报告四机构跨域结果和留一机构稳定性；"
                "不能把结论外推到未观测机构。"
            ),
            "forbidden_claim": "不能宣称机构无关或站点无关的普适性能。",
            "next_action": (
                "增加新的独立机构或站点网络后再升级为确认性结论。"
            ),
        },
        {
            "risk_id": "R5_causal_recovery_non_equivalence",
            "risk": (
                "严格因果恢复弱于离线双向插值，不能写成在线等同离线。"
            ),
            "existing_evidence": [
                "stage133_causal_temporal_recovery.json",
                "stage134_causal_recovery_optimization.json",
                "stage246_frozen_submission_package_table5_recovery_safety.csv",
            ],
            "observed_status": (
                "安全约束恢复在评价数据上达到零清洁伤害和正恢复收益，"
                "但因果恢复边界必须保留。"
            ),
            "disposition": "BOUNDED",
            "allowed_main_claim": (
                "可以声称安全约束恢复的风险权衡和现场安全边界；"
                "不能声明在线恢复等同离线插值。"
            ),
            "forbidden_claim": "不能把离线恢复指标写成在线因果恢复性能。",
            "next_action": (
                "若需要更高在线性能，必须新增低延迟因果实验并单独预注册。"
            ),
        },
        {
            "risk_id": "R6_control_attack_external_validity",
            "risk": (
                "攻击参数、位置和强度均为透明控制变量，"
                "外部效度不能被声称已完全解决。"
            ),
            "existing_evidence": [
                "stage246_frozen_submission_package.json",
                "stage248_cross_provider_fair_baseline_audit.json",
                "stage245_cross_provider_nonstuck_seed_replication.json",
            ],
            "observed_status": (
                "种子、机构和攻击类型的边界已明确，"
                "但真实攻击日志仍缺失。"
            ),
            "disposition": "BOUNDED",
            "allowed_main_claim": (
                "可以声称方法在预定义攻击模型、冻结阈值和低误报运行点下有效。"
            ),
            "forbidden_claim": "不能把受控注入结果写成真实攻击普遍性证明。",
            "next_action": (
                "把受控实验与自然弱标签证据分开呈现；"
                "获取真实日志前不提升结论。"
            ),
        },
        {
            "risk_id": "R7_test_set_reuse",
            "risk": "多次阶段探索后存在测试集复用和选择性报告风险。",
            "existing_evidence": [
                "stage249_jisa_submission_gate.json",
                "stage10_paper_readiness_audit.json",
                "stage115_protocol_boundary_audit.json",
            ],
            "observed_status": (
                "当前冻结门通过；主结果、负结果和限制均被保留。"
                "但探索历史意味着论文必须只把冻结主终点作为确认性结论。"
            ),
            "disposition": "BOUNDED",
            "allowed_main_claim": (
                "可以用冻结主终点、预定义阈值和独立种子复现作为确认性证据。"
            ),
            "forbidden_claim": (
                "不能把探索阶段中事后挑选的最好结果写成确认性结论。"
            ),
            "next_action": (
                "投稿包保持当前冻结哈希；任何新增分析都标注为探索性，"
                "除非另建完全锁定测试。"
            ),
        },
    ]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = risk_rows()
    input_paths = [
        TABLES / "stage246_frozen_submission_package.json",
        TABLES / "stage248_cross_provider_fair_baseline_audit.json",
        TABLES / "stage249_jisa_submission_gate.json",
        TABLES / "stage244_cross_provider_nonstuck_candidate_features.json",
        TABLES / "stage245_cross_provider_nonstuck_seed_replication.json",
    ]
    statuses = {row["disposition"] for row in rows}
    if statuses - {"RESOLVED", "BOUNDED"}:
        raise RuntimeError(f"unexpected risk dispositions: {statuses}")
    payload = {
        "status": "PASS_RISK_CLOSURE_AUDIT",
        "protocol": (
            "阶段250只读取冻结证据，不训练、不选阈值、不选特征、"
            "不重新评价测试集。它把每个剩余风险映射到现有证据、"
            "处置状态、允许的主声称和禁止声称；"
            "任何未通过独立验证的风险保持为边界结果，"
            "不得升级为论文主优势。"
        ),
        "risk_count": len(rows),
        "resolved_count": sum(
            row["disposition"] == "RESOLVED" for row in rows
        ),
        "bounded_count": sum(row["disposition"] == "BOUNDED" for row in rows),
        "risks": rows,
        "frozen_input_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in input_paths
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(
        TABLES / f"{PREFIX}_risks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段250 JISA风险闭合与声称边界审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 风险数：`{payload['risk_count']}`",
        f"- 已解决：`{payload['resolved_count']}`",
        f"- 保留边界：`{payload['bounded_count']}`",
        "",
        "## 风险台账",
        "",
        "| 风险 | 处置 | 允许的主声称 | 禁止声称 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['risk_id']} | {row['disposition']} | "
            f"{row['allowed_main_claim']} | {row['forbidden_claim']} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "- 冻结门通过不等于风险消失；当前仍有多项结论只能作为边界证据保留。",
        "- 论文主声称应优先使用稳定、可复现、跨机构方向一致的冻结结果。",
        "- 非stuck、One-Class SVM和真实攻击外推不得被包装为全面领先。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "risk_count": payload["risk_count"],
                "resolved_count": payload["resolved_count"],
                "bounded_count": payload["bounded_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
