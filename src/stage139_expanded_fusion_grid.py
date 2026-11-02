from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SEEDS = [271828, 314159, 161803, 8675309, 424242]
ALPHAS = [0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 1.00]
WINDOWS = [1, 3, 6, 12]
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402


def bootstrap_diff(values: np.ndarray, seed: int, repeats: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_candidate(frame: pd.DataFrame, alpha: float, window: int) -> dict[str, object]:
    valid = frame["split"].eq("valid")
    test = frame["split"].eq("test")
    y_valid = frame.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    y_test = frame.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    base = pd.to_numeric(frame["base_fused_risk"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    structural = pd.to_numeric(frame["structural_risk"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw = alpha * base + (1.0 - alpha) * structural
    valid_risk = smooth_risk(frame.loc[valid], raw[valid.to_numpy()], window)
    test_risk = smooth_risk(frame.loc[test], raw[test.to_numpy()], window)
    threshold, valid_f1 = _best_threshold(y_valid, valid_risk)
    valid_pred = valid_risk >= threshold
    test_pred = test_risk >= threshold
    vm = _metrics(y_valid, valid_pred, valid_risk)
    tm = _metrics(y_test, test_pred, test_risk)
    return {
        "alpha_base": alpha,
        "alpha_structural": 1.0 - alpha,
        "window": window,
        "threshold": float(threshold),
        "valid_f1": float(valid_f1),
        "valid_false_alarm_rate": float(np.mean(valid_pred[~y_valid])) if (~y_valid).any() else float("nan"),
        "test_f1": float(tm["f1"]),
        "test_recall": float(tm["recall"]),
        "test_precision": float(tm["precision"]),
        "test_auprc": float(tm["average_precision"]),
        "test_false_alarm_rate": float(np.mean(test_pred[~y_test])) if (~y_test).any() else float("nan"),
        "valid_rows": int(len(y_valid)),
        "test_rows": int(len(y_test)),
    }


def main() -> int:
    all_rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    default_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        path = TABLES / f"stage09b_local_robust_scores_stage137_default_seed{seed}.parquet"
        result_path = TABLES / f"stage09b_robust_fusion_stage137_default_seed{seed}.json"
        frame = pd.read_parquet(path)
        candidates = []
        for alpha in ALPHAS:
            for window in WINDOWS:
                row = evaluate_candidate(frame, alpha, window)
                row["seed"] = seed
                all_rows.append(row)
                candidates.append(row)
        best = max(candidates, key=lambda row: (row["valid_f1"], -row["valid_false_alarm_rate"], -row["threshold"], row["window"], row["alpha_base"]))
        selected.append({**best, "selection_policy": "expanded_grid_max_valid_f1"})
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        default_rows.append({"seed": seed, "test_f1": float(payload["local_test"]["f1"]), "test_recall": float(payload["local_test"]["recall"]), "test_precision": float(payload["local_test"]["precision"]), "test_auprc": float(payload["local_test"]["average_precision"]), "test_false_alarm_rate": float(payload["test_false_alarm_rate_on_clean_rows"])})
    detail = pd.DataFrame(all_rows)
    selected_df = pd.DataFrame(selected)
    default_df = pd.DataFrame(default_rows)
    merged = selected_df.merge(default_df, on="seed", suffixes=("_expanded", "_default"))
    for metric in METRICS:
        merged[f"diff_{metric}"] = merged[f"{metric}_expanded"] - merged[f"{metric}_default"]
    aggregate = pd.DataFrame([{
        "variant": "expanded_valid_selected",
        "seed_count": len(selected_df),
        **{f"{metric}_mean": float(selected_df[metric].mean()) for metric in METRICS},
        **{f"{metric}_std": float(selected_df[metric].std(ddof=1)) for metric in METRICS},
    }, {
        "variant": "stage137_default_reference",
        "seed_count": len(default_df),
        **{f"{metric}_mean": float(default_df[metric].mean()) for metric in METRICS},
        **{f"{metric}_std": float(default_df[metric].std(ddof=1)) for metric in METRICS},
    }])
    paired_rows = []
    for index, metric in enumerate(METRICS):
        differences = merged[f"diff_{metric}"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_diff(differences, seed=139139 + index)
        paired_rows.append({"metric": metric, "expanded_minus_default_mean": float(differences.mean()), "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high, "expanded_selected_count_better": int((differences > 0).sum()), "expanded_selected_count_worse": int((differences < 0).sum())})
    paired = pd.DataFrame(paired_rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段139只扩展预先固定的alpha_base={0.25,0.35,0.50,0.65,0.80,0.90,1.00}与window={1,3,6,12}网格；每个新种子仅在valid最大化F1并按误报/阈值/窗口/alpha固定顺序打破平局，test只评价，不根据test重新选择。default作为阶段137冻结参考。",
        "seeds": SEEDS,
        "alpha_grid": ALPHAS,
        "window_grid": WINDOWS,
        "candidate_count": len(detail),
        "selected_count": len(selected_df),
        "aggregate_count": len(aggregate),
        "paired_count": len(paired),
        "limitations": [
            "该阶段是新种子上的探索性网格扩展，不证明全局最优。",
            "候选仍使用同一数据源、同一受控攻击生成器和同一特征实现。",
            "若扩展网格未形成稳定提升，后续不应继续在locked test上扩大搜索。",
        ],
        "aggregate": aggregate.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
    }
    detail.to_csv(TABLES / "stage139_expanded_fusion_grid_detail.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage139_expanded_fusion_grid_selected.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(TABLES / "stage139_expanded_fusion_grid_paired.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage139_expanded_fusion_grid_aggregate.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / "stage139_expanded_fusion_grid_metric_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage139_expanded_fusion_grid.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段139扩展融合网格", "", "每个新种子只在valid选择，test只评价；default为阶段137冻结参考。", "", "| 方案 | F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['variant']} | {row['test_f1_mean']:.4f} | {row['test_recall_mean']:.4f} | {row['test_precision_mean']:.4f} | {row['test_auprc_mean']:.4f} | {row['test_false_alarm_rate_mean']:.4f} |")
    lines += ["", "指标级成对差异见 `stage139_expanded_fusion_grid_metric_paired.csv`；未形成稳定提升时不替换default。"]
    (REPORTS / "stage139_expanded_fusion_grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "candidate_count": len(detail), "selected_count": len(selected_df), "aggregate_count": len(aggregate), "paired": paired.to_dict(orient="records")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
