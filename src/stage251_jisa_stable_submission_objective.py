from __future__ import annotations

"""Stage 251: stable-submission objective contract for JISA.

Stage250 records which scientific risks remain bounded.  This stage turns the
user's "maximise the chance of a stable submission" objective into an
auditable contract.  It does not train, select thresholds, select features,
or re-evaluate any test set.  For each open risk it states the best defensible
status currently reachable, why that ceiling exists, and the evidence that
would be required to promote it.  It also prioritises the proposed workstreams
by expected claim value, cost, and explicit stop rules.

The contract is deliberately structured so that "stable" means claim
proportionality, not an assertion that every risk is resolved.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage251_jisa_stable_submission_objective"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def objective_rows(s250: dict, s249: dict) -> list[dict[str, object]]:
    ceilings = {
        "R1_natural_to_controlled_gap": {
            "best_reachable_status": "BOUNDED",
            "why_ceiling": (
                "No real attack log is present in the frozen data; controlled "
                "injection cannot by itself establish external validity."
            ),
            "promotion_evidence": (
                "An independently governed real fault or incident log with "
                "auditable labels, split before any tuning."
            ),
            "paper_action": (
                "Keep the primary claim to the predefined attack model and "
                "report natural observations only as weak-label association."
            ),
        },
        "R2_nonstuck_instability": {
            "best_reachable_status": "BOUNDED",
            "why_ceiling": (
                "Direction is repeatable on new attack seeds, but the "
                "provider-equal interval crosses zero and the ECCC-excluded "
                "F1 direction reverses."
            ),
            "promotion_evidence": (
                "A new independent provider or a fully locked external test "
                "that preserves the same positive direction."
            ),
            "paper_action": (
                "Report non-stuck as an exploratory extension and keep the "
                "main advantage on the frozen stuck evidence."
            ),
        },
        "R3_one_class_svm_complementarity": {
            "best_reachable_status": "BOUNDED",
            "why_ceiling": (
                "The pre-registered 20k/50k/100k sensitivity is complete, "
                "but larger training budgets do not produce a consistent "
                "advantage over Trust across all providers."
            ),
            "promotion_evidence": (
                "A new independent provider or locked test on which a larger "
                "One-Class SVM budget changes the complementary direction."
            ),
            "paper_action": (
                "Claim stable advantage over Robust and Isolation Forest and "
                "describe One-Class SVM as complementary, with the budget "
                "boundary disclosed."
            ),
        },
        "R4_provider_and_station_dependence": {
            "best_reachable_status": "BOUNDED",
            "why_ceiling": (
                "Only four external target providers are frozen; provider-level "
                "exact tests have limited power."
            ),
            "promotion_evidence": (
                "Additional independent providers or station networks with a "
                "pre-registered transfer protocol."
            ),
            "paper_action": (
                "Present provider-level results and leave-one-provider checks "
                "without claiming provider-independent universality."
            ),
        },
        "R5_causal_recovery_non_equivalence": {
            "best_reachable_status": "BOUNDED",
            "why_ceiling": (
                "The frozen causal recovery is intentionally weaker than "
                "offline bidirectional interpolation."
            ),
            "promotion_evidence": (
                "A new, separately pre-registered low-latency causal study "
                "evaluated on an independent test period."
            ),
            "paper_action": (
                "Frame recovery as a safety-constrained trade-off and do not "
                "equate it with offline interpolation."
            ),
        },
        "R6_control_attack_external_validity": {
            "best_reachable_status": "BOUNDED",
            "why_ceiling": (
                "Attack parameters, locations, and strengths are transparent "
                "control variables, not observed attack behaviour."
            ),
            "promotion_evidence": (
                "Real attack or fault logs collected under an independent "
                "governance and labelling process."
            ),
            "paper_action": (
                "Separate controlled experiments from natural weak-label "
                "evidence and do not generalise the attack population."
            ),
        },
        "R7_test_set_reuse": {
            "best_reachable_status": "BOUNDED",
            "why_ceiling": (
                "The frozen endpoint is clean, but the project has a long "
                "exploration history; no untouched locked test remains for "
                "newly proposed variants."
            ),
            "promotion_evidence": (
                "A genuinely new locked test set, or a confirmatory analysis "
                "using only the already frozen endpoint without new selection."
            ),
            "paper_action": (
                "Restrict confirmatory claims to the frozen endpoint and mark "
                "all new analyses as exploratory."
            ),
        },
    }

    source_by_id = {row["risk_id"]: row for row in s250["risks"]}
    rows: list[dict[str, object]] = []
    for risk_id, ceiling in ceilings.items():
        source = source_by_id[risk_id]
        rows.append(
            {
                "risk_id": risk_id,
                "source_disposition": source["disposition"],
                "best_reachable_status": ceiling["best_reachable_status"],
                "why_ceiling": ceiling["why_ceiling"],
                "promotion_evidence": ceiling["promotion_evidence"],
                "paper_action": ceiling["paper_action"],
                "allowed_main_claim": source["allowed_main_claim"],
                "forbidden_claim": source["forbidden_claim"],
            }
        )
    return rows


def workstream_rows() -> list[dict[str, object]]:
    return [
        {
            "priority": 1,
            "workstream": "冻结主终点与主声称收敛",
            "action": (
                "仅使用FAR=0.01、冻结候选、Provider等权、站点簇统计作为"
                "确认性证据；非stuck、One-Class SVM和真实攻击外推全部降级。"
            ),
            "expected_effect": (
                "消除选择性报告和事后挑选风险，使主表、主图和限制逐项一致。"
            ),
            "resource": "无需训练；只读现有冻结产物和论文稿。",
            "stop_rule": (
                "任何需要重新选特征、窗口、权重、阈值或攻击位置的想法都"
                "不得用于确认性主结果。"
            ),
        },
        {
            "priority": 2,
            "workstream": "检测与恢复结果的可复现性固化",
            "action": (
                "把Stage249/250/251的输入哈希、证据矩阵、协议边界、语义"
                "审计和单元测试串成单条复现入口。"
            ),
            "expected_effect": (
                "让审稿人能从冻结包重放到报告，避免哈希、计数和作者结论"
                "脱节。"
            ),
            "resource": "无需模型训练；CPU级审计。",
            "stop_rule": (
                "若审计失败，先修复可追溯性，不新增性能实验。"
            ),
        },
        {
            "priority": 3,
            "workstream": "One-Class SVM互补边界与全量预算说明",
            "action": (
                "主文保留冻结子采样结果，补一张计算预算与训练规模说明表；"
                "全量敏感性只作为未来工作，不替换主表。"
            ),
            "expected_effect": (
                "把“未能全面领先”转化为透明的方法学边界，降低审稿人质疑"
                "基线不公平的风险。"
            ),
            "resource": "无需新算力即可完成说明；全量训练需要额外算力。",
            "stop_rule": (
                "当前不把One-Class SVM写成落后或领先；阶段254的全部"
                "预注册档位必须同时报告，未来新结果不得修改冻结主表。"
            ),
        },
        {
            "priority": 4,
            "workstream": "非stuck攻击的边界复现",
            "action": (
                "保留Stage244/245方向一致、区间跨零、移除ECCC后转负的"
                "全部结果，作为扩展分析和失败类型分解。"
            ),
            "expected_effect": (
                "展示负结果保留和机制分析，提升方法可信度，但不升级为"
                "普适优势。"
            ),
            "resource": "无新增训练；若换新机构需重新预注册数据协议。",
            "stop_rule": (
                "不再在现有test上尝试提高非stuck F1；节点方向不稳定就停止。"
            ),
        },
        {
            "priority": 5,
            "workstream": "跨机构与站点依赖的边界写作",
            "action": (
                "四机构逐一报告，保留留一机构、站点簇区间和ECCC依赖，"
                "明确不声称机构无关。"
            ),
            "expected_effect": (
                "把有限外部效度写成清晰的适用域，降低过度外推风险。"
            ),
            "resource": "无需新增算力；新增机构属于后续独立研究。",
            "stop_rule": (
                "没有新独立机构时，不把四个目标机构写成普适证明。"
            ),
        },
        {
            "priority": 6,
            "workstream": "真实/半真实异常与在线恢复边界",
            "action": (
                "自然观测只做弱标签关联；严格因果恢复单独报告，明确在线"
                "不等同离线双向插值。"
            ),
            "expected_effect": (
                "封住“受控攻击冒充真实日志”和“离线结果冒充在线能力”两类"
                "致命审稿质疑。"
            ),
            "resource": "无需新增算力；真实日志和低延迟在线实验需外部数据或"
            "独立预注册。",
            "stop_rule": (
                "没有真实标签或因果在线实验时，不提升对应结论。"
            ),
        },
        {
            "priority": 7,
            "workstream": "授权真实测试床攻击执行",
            "action": (
                "在隔离loopback TCP链路上执行真实运行时重放、篡改、冻结、"
                "延迟、丢包和协同偏移，保存原始接收/发送日志与哈希；"
                "结果只作为独立外部效度证据。"
            ),
            "expected_effect": (
                "把攻击外部效度风险从纯数值注入提升为授权协议测试床证据；"
                "最多降为PARTIALLY_CLOSED，不覆盖冻结主终点。"
            ),
            "resource": "无需新依赖；本机运行，已完成35组。",
            "stop_rule": (
                "任何运行若失败或日志不完整，保留失败结果且不得进入主表；"
                "没有独立生产日志时不得写成RESOLVED。"
            ),
        },
    ]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    input_paths = [
        TABLES / "stage249_jisa_submission_gate.json",
        TABLES / "stage250_jisa_risk_closure_audit.json",
    ]
    s249 = load_json(input_paths[0])
    s250 = load_json(input_paths[1])
    if s249.get("status") != "PASS_JISA_SUBMISSION_GATE":
        raise RuntimeError("JISA submission gate must pass before objective freeze")
    if s250.get("status") != "PASS_RISK_CLOSURE_AUDIT":
        raise RuntimeError("risk-closure audit must pass before objective freeze")

    risks = objective_rows(s250, s249)
    workstreams = workstream_rows()
    resolved_now = sum(
        row["best_reachable_status"] == "RESOLVED" for row in risks
    )
    bounded_now = sum(
        row["best_reachable_status"] == "BOUNDED" for row in risks
    )
    external_gates = {
        "real_attack_or_fault_labels_required": True,
        "new_independent_provider_or_network_required": True,
        "full_scale_one_class_svm_compute_required": False,
        "new_locked_test_required_for_new_variants": True,
    }
    payload = {
        "status": "PASS_STABLE_SUBMISSION_OBJECTIVE",
        "objective": (
            "最大化TrustMeteo-QC达到JISA/安全二区稳定投稿水平的概率："
            "以冻结主终点、可复现证据和清晰的声称边界为核心；"
            "不以牺牲边界为代价追求全面领先。"
        ),
        "primary_far": s249.get("primary_far"),
        "primary_claim_policy": (
            "只有冻结主终点、预定义阈值、跨机构方向一致且区间支持的结果"
            "可以进入确认性主声称；所有其他结果必须标注为边界、探索性或"
            "负结果。"
        ),
        "stable_submission_definition": [
            "主声称可由冻结产物、代码哈希、统计区间和负结果共同支持。",
            "受控攻击与真实攻击外部效度明确分开。",
            "One-Class SVM和因果恢复只作为互补/安全边界陈述。",
            "任何新方向都不能通过重新选择现有test来升级结论。",
        ],
        "risk_ceiling_count": len(risks),
        "resolved_count": resolved_now,
        "bounded_count": bounded_now,
        "risk_ceilings": risks,
        "workstream_count": len(workstreams),
        "workstreams": workstreams,
        "external_gates": external_gates,
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
    pd.DataFrame(risks).to_csv(
        TABLES / f"{PREFIX}_risk_ceilings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(workstreams).to_csv(
        TABLES / f"{PREFIX}_workstreams.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段251 JISA稳投稿目标契约",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 主FAR：`{payload['primary_far']}`",
        f"- 风险上限项：`{len(risks)}`；当前可升级为RESOLVED：`{resolved_now}`；"
        f"必须保留为BOUNDED：`{bounded_now}`",
        "",
        "## 稳定投稿的定义",
        "",
        *[f"- {item}" for item in payload["stable_submission_definition"]],
        "",
        "## 风险上限",
        "",
        "| 风险 | 当前处置 | 当前证据下最高状态 | 上限原因 | 升级所需证据 | 论文动作 |",
        "|---|---|---|---|---|---|",
    ]
    for row in risks:
        lines.append(
            f"| {row['risk_id']} | {row['source_disposition']} | "
            f"{row['best_reachable_status']} | {row['why_ceiling']} | "
            f"{row['promotion_evidence']} | {row['paper_action']} |"
        )
    lines += [
        "",
        "## 投入顺序",
        "",
        "| 优先级 | 工作流 | 动作 | 预期效果 | 资源 | 停止规则 |",
        "|---:|---|---|---|---|---|",
    ]
    for row in workstreams:
        lines.append(
            f"| {row['priority']} | {row['workstream']} | {row['action']} | "
            f"{row['expected_effect']} | {row['resource']} | {row['stop_rule']} |"
        )
    lines += [
        "",
        "## 外部门槛",
        "",
        "- 真实攻击或故障日志：当前没有，因此不能把受控注入写成真实攻击普遍性。",
        "- 新的独立机构或站点网络：当前没有，因此不能把四机构结果写成机构无关。",
        "- One-Class SVM训练规模敏感性：阶段254已完成20k/50k/100k四机构审计，但结果仍为互补而非全面领先。",
        "- 新变体的全新锁定测试：当前没有，因此不能把探索结果升级为确认性结果。",
        "",
        "## 结论",
        "",
        "- 当前版本可以按“受控攻击、低误报、安全恢复、边界清晰”路线稳投。",
        "- 当前版本不能按“所有攻击类型全面领先、真实攻击普适、在线等同离线”路线投稿。",
        "- 最值得投入的顺序是先收敛主声称和冻结链，再处理边界说明；未满足外部门槛的方向不进入主表。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "risk_ceiling_count": payload["risk_ceiling_count"],
                "resolved_count": payload["resolved_count"],
                "bounded_count": payload["bounded_count"],
                "workstream_count": payload["workstream_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
