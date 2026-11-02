from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

THRESHOLDS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]


def select_dataset(path: Path, dataset: str, group_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.read_csv(path)
    selected_rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        for _, group in detail.groupby(group_columns, sort=True):
            feasible = group.loc[group["valid_clean_harm_rate"] <= threshold]
            if feasible.empty:
                feasible = group.loc[group["strategy"] == "no_recovery"]
            chosen = feasible.sort_values(["valid_utility", "valid_gated_alert_rows", "strategy", "gate_quantile"], ascending=[False, True, True, False]).iloc[0]
            held_out = chosen.get("held_out_station", "")
            held_out = "" if pd.isna(held_out) else str(held_out)
            fields = [
                "seed", "strategy", "gate_quantile", "risk_threshold", "valid_utility", "test_utility",
                "valid_attack_mae_after", "test_attack_mae_after", "valid_attack_improvement_rate",
                "test_attack_improvement_rate", "valid_clean_harm_rate", "test_clean_harm_rate",
                "valid_gated_alert_rows", "test_gated_alert_rows",
            ]
            record = {field: chosen[field] for field in fields}
            record.update({"dataset": dataset, "held_out_station": held_out, "max_valid_clean_harm_rate": float(threshold)})
            selected_rows.append(record)
    selected = pd.DataFrame(selected_rows)
    aggregate = selected.groupby(["dataset", "max_valid_clean_harm_rate"], as_index=False).agg(
        combinations=(group_columns[0], "size"),
        selected_no_recovery=("strategy", lambda s: int((s == "no_recovery").sum())),
        test_utility_mean=("test_utility", "mean"),
        test_attack_mae_after_mean=("test_attack_mae_after", "mean"),
        test_attack_improvement_rate_mean=("test_attack_improvement_rate", "mean"),
        test_clean_harm_rate_mean=("test_clean_harm_rate", "mean"),
    )
    return selected, aggregate


def main() -> int:
    local_path = TABLES / "stage129_valid_recovery_strategy_selection_detail.csv"
    noaa_path = TABLES / "stage130_noaa_safety_constrained_recovery_detail.csv"
    if not local_path.is_file() or not noaa_path.is_file():
        raise FileNotFoundError("stage129 and stage130 detail artifacts are required")
    local, local_agg = select_dataset(local_path, "local_leave_one_station", ["seed", "held_out_station"])
    noaa, noaa_agg = select_dataset(noaa_path, "NOAA_external", ["seed"])
    selected = pd.concat([local, noaa], ignore_index=True)
    aggregate = pd.concat([local_agg, noaa_agg], ignore_index=True)
    counts = selected.groupby(["dataset", "max_valid_clean_harm_rate", "strategy"], as_index=False).size().rename(columns={"size": "selected_combinations"})
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段131读取阶段129本地五个固定种子×留出站点和阶段130 NOAA五个固定种子的全候选valid/test明细，对预先固定的正常误报警改写恶化率上限0%、5%、10%、20%、30%、50%、100%分别执行valid-only选择；每个组合按valid utility选择，test只评价，不根据test重新选择，不重新训练；该阶段不增加新的种子不确定性。stage130中的NOAA攻击仍为透明受控注入。",
        "thresholds": THRESHOLDS,
        "detail_rows": len(selected),
        "aggregate_rows": len(aggregate),
        "count_rows": len(counts),
        "selected": selected.to_dict(orient="records"),
        "aggregate": aggregate.to_dict(orient="records"),
        "selected_counts": counts.to_dict(orient="records"),
        "limitations": [
            "敏感性分析复用阶段129/130已生成的候选明细，不增加独立攻击样本。",
            "阈值网格是安全性权衡分析，不证明任何一个阈值具有普适最优性。",
            "本地与NOAA的攻击均包含受控注入；NOAA质量标记不能替代真实攻击真值。",
        ],
    }
    selected.to_csv(TABLES / "stage131_safety_constraint_sensitivity_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage131_safety_constraint_sensitivity.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(TABLES / "stage131_safety_constraint_sensitivity_counts.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage131_safety_constraint_sensitivity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段131安全约束敏感性", "", "阶段129本地留一站点和阶段130 NOAA外部候选明细，分别在多个valid正常误报警改写恶化率上限下重新选择；test只评价。", "", "| 数据集 | valid恶化率上限 | 组合数 | 选择no_recovery | test utility | test攻击MAE后 | test攻击改善率 | test正常改写恶化率 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['dataset']} | {row['max_valid_clean_harm_rate']:.2f} | {row['combinations']} | {row['selected_no_recovery']} | {row['test_utility_mean']:.4f} | {row['test_attack_mae_after_mean']:.4f} | {row['test_attack_improvement_rate_mean']:.4f} | {row['test_clean_harm_rate_mean']:.4f} |")
    lines += ["", "该阶段只检验安全约束阈值敏感性，不根据locked test选阈值，也不替换阶段124主结果。"]
    (REPORTS / "stage131_safety_constraint_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(selected), "aggregate_rows": len(aggregate), "count_rows": len(counts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
