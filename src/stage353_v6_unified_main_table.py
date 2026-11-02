"""Build traceable V6 main and scenario tables from frozen artifacts.

This script only aggregates existing V6 rows. It does not fit models, select
thresholds, alter labels, or create new experimental results.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DETAIL = ROOT / "tables" / "stage349_v6_locked_evaluation_detail.csv"
SCENARIO = ROOT / "tables" / "stage349_v6_locked_evaluation_scenario.csv"
EVAL_JSON = ROOT / "tables" / "stage349_v6_locked_evaluation.json"
AUDIT_JSON = ROOT / "tables" / "stage351_v6_station_bh_audit.json"
PROTOCOL = ROOT / "confirmatory_package" / "simulated_attack_v6" / "protocol_freeze.json"

MAIN_CSV = ROOT / "tables" / "stage353_v6_unified_main_table.csv"
SCENARIO_CSV = ROOT / "tables" / "stage353_v6_scenario_gate.csv"
OUT_JSON = ROOT / "tables" / "stage353_v6_unified_main_table.json"
OUT_MD = ROOT / "reports" / "stage353_v6_unified_main_table.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def f(value: str) -> float:
    return float(value)


def main() -> int:
    protocol = load_json(PROTOCOL)
    evaluation = load_json(EVAL_JSON)
    audit = load_json(AUDIT_JSON)

    with DETAIL.open("r", encoding="utf-8-sig", newline="") as fh:
        detail_rows = list(csv.DictReader(fh))
    with SCENARIO.open("r", encoding="utf-8-sig", newline="") as fh:
        scenario_rows = list(csv.DictReader(fh))

    metrics = [
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "roc_auc",
        "average_precision",
        "event_coverage",
        "row_recall",
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in detail_rows:
        grouped[row["method"]].append(row)

    method_types = {
        "trust_only": "Trust候选",
        "route_knn_arima": "Trust路由候选",
        "trust_shape_union": "Trust形态融合候选",
        "robust_max_deviation": "经典稳健统计",
        "isolation_forest": "经典无监督",
        "one_class_svm": "经典无监督",
        "pca_reconstruction": "重建",
        "mlp_autoencoder_reconstruction": "重建",
        "knn_distance": "经典无监督",
        "ewma_deviation": "时间统计",
        "arima_innovation": "预测/创新",
        "gru_predictor": "轻量预测",
        "lightweight_autoencoder": "轻量重建",
        "ics_cps_spc_proxy": "ICS/CPS透明代理",
    }

    main_rows: list[dict[str, Any]] = []
    for method, rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "method": method,
            "method_type": method_types.get(method, "未分类"),
            "runs": len(rows),
            "provider_count": len({row["source"] for row in rows}),
            "seed_count": len({row["seed"] for row in rows}),
            "attack_scenario_count": len(
                {row["scenario"] for row in scenario_rows}
            ),
            "primary_far_budget_reference": protocol["primary_far"],
            "evaluation_far_gate": 0.01,
            "selected_in_v6": method in evaluation["selection_counts"],
        }
        for metric in metrics:
            out[metric] = mean(f(row[metric]) for row in rows)
        out["measured_far_le_gate"] = (
            out["false_alarm_rate"] <= 0.01 + 1e-12
        )
        main_rows.append(out)

    # Reconcile the generated aggregation with the frozen JSON summary.
    expected = {row["method"]: row for row in evaluation["summary"]}
    for row in main_rows:
        source = expected[row["method"]]
        for metric in ["f1", "event_coverage", "row_recall", "average_precision", "false_alarm_rate"]:
            if abs(row[metric] - float(source[metric])) > 1e-10:
                raise ValueError(
                    f"V6 aggregation mismatch for {row['method']} {metric}: "
                    f"{row[metric]} != {source[metric]}"
                )

    baselines = [
        method
        for method in sorted(grouped)
        if method not in {"trust_only", "route_knn_arima", "trust_shape_union"}
    ]
    scenario_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in scenario_rows:
        scenario_group[(row["scenario"], row["method"])].append(
            f(row["event_coverage"])
        )

    scenario_out: list[dict[str, Any]] = []
    for scenario in sorted({row["scenario"] for row in scenario_rows}):
        candidate_value = mean(scenario_group[(scenario, "trust_shape_union")])
        wins = 0
        baseline_values: dict[str, float] = {}
        for baseline in baselines:
            baseline_value = mean(scenario_group[(scenario, baseline)])
            baseline_values[baseline] = baseline_value
            wins += int(candidate_value > baseline_value)
        scenario_out.append(
            {
                "scenario": scenario,
                "candidate": "trust_shape_union",
                "candidate_event_coverage": candidate_value,
                "wins": wins,
                "total_baselines": len(baselines),
                "win_share": wins / len(baselines),
                "passes_single_scenario_70pct": wins / len(baselines) >= 0.70,
                **{
                    f"baseline_{name}_event_coverage": value
                    for name, value in baseline_values.items()
                },
            }
        )

    MAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MAIN_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        fields = list(main_rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(main_rows)

    with SCENARIO_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        fields = list(scenario_out[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scenario_out)

    result = {
        "status": "PASS_V6_TABLE_REBUILT",
        "protocol_id": protocol["protocol_id"],
        "scope": {
            "providers": sorted({row["source"] for row in detail_rows}),
            "provider_count": len({row["source"] for row in detail_rows}),
            "seeds": sorted({row["seed"] for row in detail_rows}),
            "attack_scenarios": sorted({row["scenario"] for row in scenario_rows}),
            "method_count": len(main_rows),
            "baseline_count": len(baselines),
        },
        "primary_candidate": audit["best_candidate"],
        "selection_counts": evaluation["selection_counts"],
        "selected_run_summary": audit["selected_summary"],
        "main_rows": main_rows,
        "scenario_rows": scenario_out,
        "inputs": {
            "detail": str(DETAIL.relative_to(ROOT)),
            "scenario": str(SCENARIO.relative_to(ROOT)),
            "evaluation": str(EVAL_JSON.relative_to(ROOT)),
            "station_audit": str(AUDIT_JSON.relative_to(ROOT)),
            "protocol": str(PROTOCOL.relative_to(ROOT)),
        },
        "hashes": {
            "detail_sha256": sha256(DETAIL),
            "scenario_sha256": sha256(SCENARIO),
            "evaluation_sha256": sha256(EVAL_JSON),
            "station_audit_sha256": sha256(AUDIT_JSON),
            "protocol_sha256": sha256(PROTOCOL),
        },
        "rule": "仅聚合冻结 V6 产物；不重新拟合、不调参、不改变攻击标签或统计口径。",
    }
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage353 V6 统一主表",
        "",
        "- 状态：`PASS_V6_TABLE_REBUILT`",
        "- 说明：本产物只聚合冻结 V6 明细，不产生新实验结果。",
        f"- 主候选：`{audit['best_candidate']}`",
        f"- V6 选择计数：{evaluation['selection_counts']}",
        f"- provider：{', '.join(result['scope']['providers'])}",
        f"- 基线数：{len(baselines)}；攻击场景数：{len(result['scope']['attack_scenarios'])}",
        "",
        "## 选择运行汇总（混合候选）",
        "",
        f"- F1：`{audit['selected_summary']['f1']:.6f}`",
        f"- Event coverage：`{audit['selected_summary']['event_coverage']:.6f}`",
        f"- FAR：`{audit['selected_summary']['false_alarm_rate']:.6f}`",
        f"- 运行数：`{audit['selected_summary']['runs']}`",
        "- 该汇总不能与单一候选 `trust_shape_union` 的方法行直接互换。",
        "",
        "## 主表",
        "",
        "| 方法 | 类型 | 运行数 | F1 | Precision | Row Recall | Event Coverage | AUPRC | FAR | FAR≤1%验收门 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['method']} | {row['method_type']} | {row['runs']} | "
            f"{row['f1']:.6f} | {row['precision']:.6f} | {row['row_recall']:.6f} | "
            f"{row['event_coverage']:.6f} | {row['average_precision']:.6f} | "
            f"{row['false_alarm_rate']:.6f} | {row['measured_far_le_gate']} |"
        )
    lines.extend(
        [
            "",
            "## 场景层 70% 核查",
            "",
            "| 场景 | Trust event coverage | 胜出基线数 | 总基线数 | 胜出比例 | 单场景≥70% |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in scenario_out:
        lines.append(
            f"| {row['scenario']} | {row['candidate_event_coverage']:.6f} | "
            f"{row['wins']} | {row['total_baselines']} | {row['win_share']:.1%} | "
            f"{row['passes_single_scenario_70pct']} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `73/99=73.7%` 是总体场景×基线比例，不是每个攻击场景均达到70%。",
            "- `coordinated_offset` 和 `slow_drift` 若低于70%，必须作为失败边界保留。",
            "- 本表不把模拟攻击写成真实生产事故，也不覆盖 FMI/SMHI 的外部负结果。",
            "",
            "## 输入哈希",
            "",
        ]
    )
    for key, value in result["hashes"].items():
        lines.append(f"- `{key}`：`{value}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "main_csv": str(MAIN_CSV),
                "scenario_csv": str(SCENARIO_CSV),
                "out_json": str(OUT_JSON),
                "out_md": str(OUT_MD),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
