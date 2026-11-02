from __future__ import annotations

"""Stage 324: reframe and audit the simulation-first acceptance target.

This stage does not alter any performance table.  It converts the current
evidence into an explicit two-level target: (1) a complete, reproducible
simulated-attack package and (2) a comprehensive-superiority claim, which is
allowed only when every preregistered comparator and attack family passes.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
AUDIT = TABLES / "stage321_simulated_attack_final_audit.json"
HYBRID = TABLES / "stage323_exploratory_hybrid_audit.json"
OUT_JSON = TABLES / "stage324_simulated_attack_goal_and_gap_audit.json"
OUT_REPORT = REPORTS / "stage324_simulated_attack_goal_and_gap_audit.md"


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    hybrid = json.loads(HYBRID.read_text(encoding="utf-8")) if HYBRID.exists() else {}
    checks = audit.get("checks", [])
    failed = [row for row in checks if not bool(row.get("passed"))]
    negative = audit.get("key_negative_results", [])
    blockers = [
        {
            "id": "B1",
            "name": "主终点全面领先未通过",
            "status": "OPEN",
            "evidence": [
                "ARIMA innovation 在 event coverage 上稳定领先 trust_valid_f1",
                "One-Class SVM、PCA、kNN、轻量 AE 的非劣效未通过",
            ],
            "required_action": "只能在新的未使用锁定窗口上预注册候选并重新评价；旧 test 结果不得升级",
        },
        {
            "id": "B2",
            "name": "攻击类型覆盖不完整",
            "status": "OPEN",
            "evidence": audit.get("checks", [])[-1].get("value", {}),
            "required_action": "补充 delay 与 drop 的冻结协议，并在可用的新锁定边界上评价",
        },
        {
            "id": "B3",
            "name": "历史 test 复用限制",
            "status": "RANGE_BLOCKED",
            "evidence": "Stage320 test 已查看；Stage322/323 只能探索性",
            "required_action": "如无未使用时间窗或新授权数据，保持探索性，不强行关闭",
        },
        {
            "id": "B4",
            "name": "真实生产攻击外推",
            "status": "OUT_OF_SCOPE",
            "evidence": "当前证据均为真实观测背景上的透明模拟攻击",
            "required_action": "论文只写 simulated/controlled attack，不写真实事故召回率",
        },
    ]
    target = {
        "primary_goal": "COMPREHENSIVE_SIMULATED_ATTACK_PACKAGE",
        "definition": "在四个冻结 provider 的真实观测背景上，完成可复核的攻击协议、统一基线、分层统计、资源审计和诚实的适用域结论",
        "not_allowed": [
            "不得把模拟攻击写成真实生产攻击",
            "不得删除或降级稳定胜出的基线",
            "不得把探索性融合升级为确认性主结果",
        ],
        "superiority_claim_gate": "仅当所有主要基线和攻击类型在 pooled、站点等权、provider 等权且 BH 校正后满足预注册门槛时开放",
    }
    next_actions = [
        {"priority": 1, "action": "保留 Stage320 主结果与所有不利结果；把 Stage322/323 明确标为探索性候选", "acceptance": "文档和哈希审计通过"},
        {"priority": 2, "action": "冻结含 delay/drop 的模拟攻击协议，但不在旧 test 上宣称确认性", "acceptance": "protocol/data/analysis 三哈希存在"},
        {"priority": 3, "action": "若取得未使用锁定窗口，再评价固定的 Trust+ARIMA 守门融合与全部基线", "acceptance": "test 未接触前冻结；主终点与 FAR 规则不变"},
        {"priority": 4, "action": "若没有新窗口，停止追求全面领先，把结论收敛为有限模拟攻击优势与互补边界", "acceptance": "不再事后调参"},
    ]
    out = {
        "status": "PASS_GOAL_REFRAMING_SCIENTIFIC_BOUNDARY",
        "target": target,
        "current_claim_ceiling": audit.get("claim_ceiling"),
        "stage321_failed_checks": failed,
        "negative_results_count": len(negative),
        "blockers": blockers,
        "next_actions": next_actions,
        "exploratory_hybrid_audit_status": hybrid.get("status"),
        "exploratory_hybrid_claim_ceiling": hybrid.get("claim_ceiling"),
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Stage324 模拟攻击主线目标与缺口审计",
        "",
        f"- 状态：`{out['status']}`",
        f"- 当前主张上限：`{out['current_claim_ceiling']}`",
        f"- Stage321 未通过检查：`{len(failed)}`；保留不利比较：`{len(negative)}` 条",
        "",
        "## 目标调整",
        "",
        "本项目的主目标改为“全面、可复核的模拟攻击评价包”，而不是无条件宣称真实环境全面领先。",
        "只有在全部预注册基线和攻击类型通过统一统计门槛后，才开放“模拟攻击范围内全面领先”措辞。",
        "",
        "## 当前阻断",
        "",
    ]
    for item in blockers:
        lines.extend([f"- **{item['id']} {item['name']}**：`{item['status']}`；{item['required_action']}"])
    lines.extend(["", "## 执行顺序", ""])
    for item in next_actions:
        lines.append(f"{item['priority']}. {item['action']}；验收：{item['acceptance']}")
    lines.extend(["", "## 结论", "", "当前可以完成高质量的模拟攻击主线和适用域论文版本，但不能把 Stage320/322/323 解释为已经全面领先。若没有新的未使用锁定窗口，最稳妥的停止点是有限模拟攻击优势 + 对 ARIMA/One-Class SVM/重建类方法的互补结论。", ""])
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": out["status"], "failed_checks": len(failed), "blockers": len(blockers), "claim_ceiling": out["current_claim_ceiling"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
