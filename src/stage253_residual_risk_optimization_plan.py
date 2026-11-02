from __future__ import annotations

"""Stage 253: actionable optimization plan for the remaining risk classes.

This is a planning and claim-boundary artifact.  It does not train, select
thresholds, choose features, or re-evaluate the frozen project endpoint.
Each row states whether the risk can move within the current resources,
what evidence is required, what the elevated claim ceiling is, and which
actions are forbidden.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage253_residual_risk_optimization_plan"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_rows() -> list[dict[str, object]]:
    return [
        {
            "risk_id": "R1_natural_to_controlled_gap",
            "current": "BOUNDED",
            "can_optimize_now": "PARTIAL",
            "best_reachable_now": "PARTIALLY_CLOSED",
            "optimization": (
                "授权真实测试床已执行35组真实loopback TCP运行；"
                "攻击在运行时发生，报文、时间戳、发送日志和接收日志均保留；"
                "阶段255进一步完成严格因果在线检测、延迟和恢复审计。"
            ),
            "required_evidence": (
                "测试床协议、原始发送/接收日志、哈希清单、失败运行保留；"
                "另需独立生产事故/故障日志才能进一步升级。"
            ),
            "forbidden": (
                "不得把授权测试床描述为第三方或生产环境攻击；"
                "不得把测试床结果合并进Stage246冻结主终点。"
            ),
        },
        {
            "risk_id": "R2_nonstuck_instability",
            "current": "BOUNDED",
            "can_optimize_now": "LIMITED",
            "best_reachable_now": "PARTIALLY_CLOSED",
            "optimization": (
                "可补齐按攻击类型、机构、留一机构的边界分解；"
                "方向可复现，但区间跨零和移除ECCC后转负已被冻结，"
                "不能在现有test上继续追赶F1。"
            ),
            "required_evidence": (
                "新的独立机构、站点网络或全新锁定测试；"
                "并在评价前预先冻结非stuck终点和选择规则。"
            ),
            "forbidden": (
                "不得在现有test上重新选择特征、窗口、权重、阈值或种子；"
                "不得删除drift等反例。"
            ),
        },
        {
            "risk_id": "R3_one_class_svm_complementarity",
            "current": "BOUNDED",
            "can_optimize_now": "YES",
            "best_reachable_now": "PARTIALLY_CLOSED",
            "optimization": (
                "可把20000行确定性子采样扩展为多个预先冻结的训练规模档位，"
                "报告时间、内存、F1、事件覆盖和FAR曲线，区分预算效应与算法效应；"
                "阶段254已在四个机构上完成20k、50k、100k预注册敏感性；"
                "训练规模变化未形成全面领先，互补结论保持不变。"
            ),
            "required_evidence": (
                "预先声明规模档位、固定拟合种子、相同test和FAR=0.01；"
                "若全量仍无优势则保留互补结论。"
            ),
            "forbidden": (
                "不得用test选择规模档位；不得只报告对Trust有利的档位；"
                "不得修改冻结主表。"
            ),
        },
        {
            "risk_id": "R4_provider_and_station_dependence",
            "current": "BOUNDED",
            "can_optimize_now": "LIMITED",
            "best_reachable_now": "PARTIALLY_CLOSED",
            "optimization": (
                "可增加第五个独立机构、站点网络的预注册与迁移矩阵，"
                "也可补站点层级影响诊断；当前四个机构的功效上限已经固定。"
            ),
            "required_evidence": (
                "新增机构/网络的官方数据、完整覆盖率审计、下载前冻结协议、"
                "同一FAR和同一攻击生成规则。"
            ),
            "forbidden": (
                "不得按结果挑选机构或删除弱站点；"
                "没有新机构时不得宣称机构无关。"
            ),
        },
        {
            "risk_id": "R5_causal_recovery_non_equivalence",
            "current": "BOUNDED",
            "can_optimize_now": "YES",
            "best_reachable_now": "PARTIALLY_CLOSED",
            "optimization": (
                "可在授权测试床上增加逐帧在线延迟、吞吐和因果恢复实验，"
                "报告检测延迟、恢复延迟、清洁伤害和恢复收益；"
                "阶段255已完成逐帧与事件级在线审计；仍不把在线结果"
                "等同于离线双向插值。"
            ),
            "required_evidence": (
                "测试床原始时间戳、队列/处理日志、延迟分位数、"
                "同一安全约束下的恢复前后误差。"
            ),
            "forbidden": (
                "不得用离线双向插值冒充在线因果性能；"
                "不得隐藏安全约束造成的召回损失。"
            ),
        },
        {
            "risk_id": "R6_control_attack_external_validity",
            "current": "BOUNDED",
            "can_optimize_now": "PARTIAL",
            "best_reachable_now": "PARTIALLY_CLOSED",
            "optimization": (
                "测试床把数值注入提升为真实运行时协议交换；"
                "阶段255已加入到达间隔、序号和重复事件审计；"
                "可继续增加Modbus TCP/MQTT适配器，但需先安装依赖并保持隔离。"
            ),
            "required_evidence": (
                "真实协议适配器、原始报文、攻击执行日志、"
                "独立生产日志才能升级为RESOLVED。"
            ),
            "forbidden": (
                "不得把模拟节点或合成PCAP称为生产攻击；"
                "不得把测试床外推为所有协议和所有部署。"
            ),
        },
        {
            "risk_id": "R7_test_set_reuse",
            "current": "BOUNDED",
            "can_optimize_now": "LIMITED",
            "best_reachable_now": "PARTIALLY_CLOSED",
            "optimization": (
                "可建立冻结哈希链和单条复现入口，把所有新增分析标记为"
                "探索性；真正解决需要全新锁定测试。"
            ),
            "required_evidence": (
                "新锁定测试或严格限定为冻结主终点的确认性分析；"
                "所有新变体必须在新数据上预注册。"
            ),
            "forbidden": (
                "不得把探索性最好结果写成确认性结论；"
                "不得在新变体上复用旧test选择参数。"
            ),
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed-analysis", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    testbed_path = TABLES / "stage252_authorized_testbed_analysis.json"
    testbed = (
        json.loads(testbed_path.read_text(encoding="utf-8"))
        if testbed_path.is_file()
        else {}
    )
    rows = plan_rows()
    payload = {
        "status": "PASS_RESIDUAL_RISK_OPTIMIZATION_PLAN",
        "protocol": (
            "阶段253只制定剩余风险的可执行优化上限；不训练、不选阈值、"
            "不选择特征，不重新评价冻结test。"
        ),
        "testbed_status": testbed.get("status", "NOT_RUN"),
        "testbed_run_count": int(testbed.get("run_count", 0)),
        "testbed_attack_type_count": int(
            testbed.get("attack_type_count", 0)
        ),
        "testbed_seed_count": int(testbed.get("seed_count", 0)),
        "rows": rows,
        "row_count": len(rows),
        "can_optimize_now_count": sum(
            row["can_optimize_now"] in {"YES", "PARTIAL"} for row in rows
        ),
        "in_progress_count": sum(
            row["can_optimize_now"] == "IN_PROGRESS" for row in rows
        ),
        "limited_count": sum(
            row["can_optimize_now"] == "LIMITED" for row in rows
        ),
        "resolved_now_count": sum(
            row["best_reachable_now"] == "RESOLVED" for row in rows
        ),
        "input_hashes": (
            {str(testbed_path.relative_to(ROOT)): sha256_file(testbed_path)}
            if testbed_path.is_file()
            else {}
        ),
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(
        TABLES / f"{PREFIX}_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段253 剩余风险优化上限",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 风险项：`{payload['row_count']}`",
        f"- 当前可实质推进：`{payload['can_optimize_now_count']}`",
        f"- 已预注册但未完成：`{payload['in_progress_count']}`",
        f"- 仅有限推进：`{payload['limited_count']}`",
        f"- 当前可升级为RESOLVED：`{payload['resolved_now_count']}`",
        "",
        f"授权测试床状态：`{payload['testbed_status']}`；运行`{payload['testbed_run_count']}`组；"
        f"攻击类型`{payload['testbed_attack_type_count']}`；种子`{payload['testbed_seed_count']}`。",
        "",
        "## 逐项优化",
        "",
        "| 风险 | 当前 | 可优化 | 当前最高状态 | 优化内容 | 所需证据 | 禁止 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['risk_id']} | {row['current']} | {row['can_optimize_now']} | "
            f"{row['best_reachable_now']} | {row['optimization']} | "
            f"{row['required_evidence']} | {row['forbidden']} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "- 真实测试床可以让R1/R6从纯数值注入提升为授权运行时攻击证据，但仍不能称为生产攻击。",
        "- One-Class SVM训练规模敏感性已完成，R3仍因全量档位未全面领先而保持部分闭合。",
        "- 因果恢复延迟已做在线审计；后续同类分析仍必须预先冻结档位和终点。",
        "- non-stuck、机构和test复用问题需要新数据或全新锁定测试，不能在旧test上追赶结果。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "row_count": payload["row_count"],
                "can_optimize_now_count": payload["can_optimize_now_count"],
                "in_progress_count": payload["in_progress_count"],
                "limited_count": payload["limited_count"],
                "resolved_now_count": payload["resolved_now_count"],
                "testbed_run_count": payload["testbed_run_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
