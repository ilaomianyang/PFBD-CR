"""Analyse the Stage503 deep-baseline run against the frozen V6 table.

Exploratory only: V6 test was already inspected in Stage349, so these numbers
must never be presented as confirmatory evidence.  The script only aggregates
existing CSVs; it does not fit models or change labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
DEEP_MODELS = ["lstm_ae", "usad", "timesnet", "anomaly_transformer"]
TRUST = ["trust_only", "route_knn_arima", "trust_shape_union"]
LABELS = {
    "lstm_ae": "LSTM-AE",
    "usad": "USAD",
    "timesnet": "TimesNet",
    "anomaly_transformer": "Anomaly Transformer",
    "trust_only": "Trust only",
    "route_knn_arima": "Route kNN+ARIMA",
    "trust_shape_union": "Trust+shape union",
    "mlp_autoencoder_reconstruction": "MLP AE (64-16-64)",
    "pca_reconstruction": "PCA AE",
}


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return pd.read_csv(path)


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    deep_detail = load(TABLES / "stage503_deep_baseline_detail.csv")
    deep_summary = load(TABLES / "stage503_deep_baseline_summary.csv")
    deep_scen = load(TABLES / "stage503_deep_baseline_scenario.csv")
    frozen_detail = load(TABLES / "stage349_v6_locked_evaluation_detail.csv")
    frozen_scen = load(TABLES / "stage349_v6_locked_evaluation_scenario.csv")

    # Validate the run shape before analysing it.
    test = deep_detail[deep_detail["split"].eq("test")]
    assert len(test) == 4 * 4 * 5, len(test)
    assert len(deep_scen) == 4 * 4 * 5 * 9, len(deep_scen)
    assert len(deep_summary) == 4
    # Reconcile provider-seed macro aggregation.
    for _, row in deep_summary.iterrows():
        sub = test[test["method"].eq(row["method"])]
        for metric in ["precision", "recall", "f1", "false_alarm_rate", "roc_auc",
                       "average_precision", "event_coverage", "row_recall"]:
            if abs(float(sub[metric].mean()) - float(row[metric])) > 1e-9:
                raise SystemExit(f"aggregate mismatch {row['method']} {metric}")

    baselines = [m for m in frozen_detail["method"].unique() if m not in TRUST]
    baseline_means = (
        frozen_scen[frozen_scen["method"].isin(baselines)]
        .groupby(["scenario", "method"])["event_coverage"].mean()
    )
    trust_means = (
        frozen_scen[frozen_scen["method"].isin(TRUST)]
        .groupby(["scenario", "method"])["event_coverage"].mean()
    )
    deep_means = deep_scen.groupby(["scenario", "method"])[
        ["event_coverage", "f1", "precision", "recall", "average_precision",
         "false_alarm_rate", "row_recall"]
    ].mean()

    rows = []
    totals = []
    for model in DEEP_MODELS:
        wins_total = 0
        pass_scen = 0
        for scenario in sorted(deep_scen["scenario"].unique()):
            cov = float(deep_means.loc[(scenario, model), "event_coverage"])
            wins = sum(
                cov > float(baseline_means.loc[(scenario, baseline)])
                for baseline in baselines
            )
            wins_total += wins
            pass_scen += int(wins / len(baselines) >= 0.70)
            ranked = {
                **{b: float(baseline_means.loc[(scenario, b)]) for b in baselines},
                **{t: float(trust_means.loc[(scenario, t)]) for t in TRUST},
                model: cov,
            }
            order = sorted(ranked, key=ranked.get, reverse=True)
            rank = order.index(model) + 1
            best = max(baselines, key=lambda b: float(baseline_means.loc[(scenario, b)]))
            union_cov = float(trust_means.loc[(scenario, "trust_shape_union")])
            rows.append({
                "model": model,
                "scenario": scenario,
                "event_coverage": cov,
                "f1": float(deep_means.loc[(scenario, model), "f1"]),
                "precision": float(deep_means.loc[(scenario, model), "precision"]),
                "recall": float(deep_means.loc[(scenario, model), "recall"]),
                "average_precision": float(deep_means.loc[(scenario, model), "average_precision"]),
                "false_alarm_rate": float(deep_means.loc[(scenario, model), "false_alarm_rate"]),
                "wins_vs_11": wins,
                "win_share": wins / len(baselines),
                "passes_70pct": wins / len(baselines) >= 0.70,
                "rank_among_15": rank,
                "best_baseline": best,
                "best_baseline_coverage": float(baseline_means.loc[(scenario, best)]),
                "delta_vs_best_baseline": cov - float(baseline_means.loc[(scenario, best)]),
                "union_coverage": union_cov,
                "union_minus_model": union_cov - cov,
            })
        totals.append({
            "model": model,
            "wins_total": wins_total,
            "total_comparisons": len(baselines) * 9,
            "win_share_total": wins_total / (len(baselines) * 9),
            "scenarios_ge_70pct": pass_scen,
            "mean_event_coverage": float(deep_summary.loc[deep_summary["method"].eq(model), "event_coverage"].iloc[0]),
            "mean_f1": float(deep_summary.loc[deep_summary["method"].eq(model), "f1"].iloc[0]),
            "mean_precision": float(deep_summary.loc[deep_summary["method"].eq(model), "precision"].iloc[0]),
            "mean_recall": float(deep_summary.loc[deep_summary["method"].eq(model), "recall"].iloc[0]),
            "mean_average_precision": float(deep_summary.loc[deep_summary["method"].eq(model), "average_precision"].iloc[0]),
            "mean_false_alarm_rate": float(deep_summary.loc[deep_summary["method"].eq(model), "false_alarm_rate"].iloc[0]),
            "mean_roc_auc": float(deep_summary.loc[deep_summary["method"].eq(model), "roc_auc"].iloc[0]),
        })

    scen_df = pd.DataFrame(rows)
    total_df = pd.DataFrame(totals)
    scen_df.to_csv(TABLES / "stage505_deep_scenario_wins.csv", index=False, encoding="utf-8-sig")
    total_df.to_csv(TABLES / "stage505_deep_totals.csv", index=False, encoding="utf-8-sig")

    # Head-to-head: union vs each deep model, per scenario (9 comparisons).
    h2h = []
    for model in DEEP_MODELS:
        sub = scen_df[scen_df["model"].eq(model)]
        h2h.append({
            "model": model,
            "union_wins": int((sub["union_coverage"] > sub["event_coverage"]).sum()),
            "deep_wins": int((sub["event_coverage"] > sub["union_coverage"]).sum()),
            "ties": int((sub["union_coverage"] == sub["event_coverage"]).sum()),
            "mean_union_minus_model": float(sub["union_minus_model"].mean()),
            "max_union_minus_model": float(sub["union_minus_model"].max()),
            "min_union_minus_model": float(sub["union_minus_model"].min()),
        })
    h2h_df = pd.DataFrame(h2h)
    h2h_df.to_csv(TABLES / "stage505_union_vs_deep.csv", index=False, encoding="utf-8-sig")

    # Comparison with the two reconstruction baselines used in Table 4.
    comparison = []
    for model in DEEP_MODELS:
        row = {"model": model}
        for baseline in ["mlp_autoencoder_reconstruction", "pca_reconstruction"]:
            b = frozen_detail[frozen_detail["method"].eq(baseline)]
            row[f"coverage_delta_vs_{baseline}"] = (
                float(deep_summary.loc[deep_summary["method"].eq(model), "event_coverage"].iloc[0])
                - float(b["event_coverage"].mean())
            )
            row[f"f1_delta_vs_{baseline}"] = (
                float(deep_summary.loc[deep_summary["method"].eq(model), "f1"].iloc[0])
                - float(b["f1"].mean())
            )
        comparison.append(row)
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(TABLES / "stage505_deep_vs_reconstruction.csv", index=False, encoding="utf-8-sig")

    result = {
        "status": "EXPLORATORY_POST_V6",
        "n_detail_test": int(len(test)),
        "n_scenario": int(len(deep_scen)),
        "baselines": baselines,

        "totals": total_df.to_dict(orient="records"),
        "union_vs_deep": h2h_df.to_dict(orient="records"),
        "comparison": comp_df.to_dict(orient="records"),
    }
    (TABLES / "stage505_deep_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def fmt(value: float) -> str:
        return f"{value:.4f}"

    lines = [
        "# Stage503 深度异常检测基线：探索性补充分析",
        "",
        "> 状态：**EXPLORATORY_POST_V6**。V6 测试集已在 Stage349 检视；本补充只用于回应“缺少主流深度基线”的批评，不能替代 V6 主证据，也不能标为确认性。",
        "",
        "## 1. 运行范围",
        "",
        f"- 4 个提供方 × 5 seeds × 4 模型 = {len(test)} 个 test provider-seed 运行（另含同数 valid 运行）。",
        "- 同一 16 维特征空间；窗口 24；train+observed 20,000 窗；α=0.0065；仅在 valid normal 行定阈值；无 point adjustment。",
        "- 模型：LSTM-AE（h64/z16, denoise 0.1, 30 ep）、USAD（latent 8, min-max, 10 ep）、TimesNet（d32, 20 ep）、Anomaly Transformer（d64, 4 heads, 20 ep）。",
        "",
        "## 2. 主指标（provider-seed 等权）",
        "",
        "| 方法 | 精确率 | 行召回 | F1 | 事件覆盖 | AP | FAR | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in deep_summary.iterrows():
        lines.append(
            f"| {LABELS.get(row['method'], row['method'])} | {fmt(row['precision'])} | "
            f"{fmt(row['row_recall'])} | {fmt(row['f1'])} | {fmt(row['event_coverage'])} | "
            f"{fmt(row['average_precision'])} | {100*row['false_alarm_rate']:.3f}% | {fmt(row['roc_auc'])} |"
        )
    lines += [
        "",
        "## 3. 场景×基线胜率（按事件覆盖，11 个非 Trust 基线）",
        "",
        "| 模型 | 胜/99 | 总胜率 | ≥70% 场景数 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in total_df.iterrows():
        lines.append(
            f"| {LABELS.get(row['model'], row['model'])} | {int(row['wins_total'])}/99 | "
            f"{100*row['win_share_total']:.1f}% | {int(row['scenarios_ge_70pct'])}/9 |"
        )
    lines += [
        "",
        "对照：V6 冻结的 `trust_shape_union` 为 73/99 = 73.7%。",
        "",
        "## 4. union 对深度模型的头对头（9 场景）",
        "",
        "| 深度模型 | union 胜 | 深度胜 | 平 | 平均覆盖差（union−模型） |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in h2h_df.iterrows():
        lines.append(
            f"| {LABELS.get(row['model'], row['model'])} | {int(row['union_wins'])} | "
            f"{int(row['deep_wins'])} | {int(row['ties'])} | {row['mean_union_minus_model']:+.4f} |"
        )
    lines += [
        "",
        "## 5. 与 Table 4 两个重建基线的差值",
        "",
        "| 模型 | Δ覆盖 vs MLP AE | ΔF1 vs MLP AE | Δ覆盖 vs PCA AE | ΔF1 vs PCA AE |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in comp_df.iterrows():
        lines.append(
            f"| {LABELS.get(row['model'], row['model'])} | "
            f"{row['coverage_delta_vs_mlp_autoencoder_reconstruction']:+.4f} | "
            f"{row['f1_delta_vs_mlp_autoencoder_reconstruction']:+.4f} | "
            f"{row['coverage_delta_vs_pca_reconstruction']:+.4f} | "
            f"{row['f1_delta_vs_pca_reconstruction']:+.4f} |"
        )
    lines += [
        "",
        "## 6. 解释边界",
        "",
        "- 这 4 个模型是**同一特征空间、同一阈值协议**下的实现，不是各自论文的官方全流程；不能据此宣称“复现了原论文最优性能”。",
        "- 所有深度模型都在 V6 的 9 个模拟攻击场景上评估；`coordinated_offset`、`short_lag` 等场景对所有方法都困难，不能外推为生产攻击召回。",
        "- V6 测试已检视，因此本补充只能作为 exploratory robustness check；若写入论文，应放在补充材料或明确标注“post-hoc exploratory”。",
        "- 时延/内存见 `stage505_resource_claims.*`；必须区分参数字节与 PyTorch 进程峰值，且实测硬件为 i5-14600KF，不是论文的 i7-12700H。",
        "",
    ]
    (REPORTS / "stage505_deep_baseline_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(total_df.to_string(index=False))
    print()
    print(h2h_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
