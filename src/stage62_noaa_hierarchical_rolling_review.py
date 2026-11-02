from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = {
    "spring_to_summer": ("2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01"),
    "summer_to_winter": ("2022-01-01", "2022-07-01", "2022-10-01", "2023-01-01"),
    "short_recent_train": ("2022-04-01", "2022-07-01", "2022-10-01", "2023-01-01"),
}
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage50_noaa_rolling_seasonal_review import build_fold  # noqa: E402
from stage61_noaa_hierarchical_score_calibration import calibrated_score, fit_hierarchical  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402


def evaluate_fold(source: pd.DataFrame, fold: str, bounds: tuple[str, str, str, str], seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    frame = build_fold(source, seed, bounds)
    scored, reference = score(frame)
    scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
    train = scored["split"].eq("train")
    valid = scored["split"].eq("valid")
    test = scored["split"].eq("test")
    yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    candidates: list[dict[str, object]] = []
    for shrink_k in [20.0, 50.0, 100.0, 200.0]:
        calibration = fit_hierarchical(scored.loc[train], shrink_k)
        for hour_weight in [0.0, 0.5, 1.0]:
            calibrated = calibrated_score(scored, calibration, hour_weight)
            train_values = calibrated[train.to_numpy()]
            for q in [0.90, 0.95, 0.975, 0.99]:
                threshold = float(np.nanquantile(train_values, q))
                pv = calibrated[valid.to_numpy()] >= threshold
                pt = calibrated[test.to_numpy()] >= threshold
                mv = _metrics(yv, pv, calibrated[valid.to_numpy()])
                mt = _metrics(yt, pt, calibrated[test.to_numpy()])
                candidates.append({
                    "fold": fold,
                    "seed": seed,
                    "shrink_k": shrink_k,
                    "hour_weight": hour_weight,
                    "q": q,
                    "threshold": threshold,
                    "valid_f1": mv["f1"],
                    "valid_far": float(pv[~yv].mean()),
                    "test_f1": mt["f1"],
                    "test_recall": mt["recall"],
                    "test_precision": mt["precision"],
                    "test_auprc": mt["average_precision"],
                    "test_far": float(pt[~yt].mean()),
                    "fixed_test_f1": reference["test"]["f1"],
                    "fixed_test_recall": reference["test"]["recall"],
                    "fixed_test_auprc": reference["test"]["average_precision"],
                    "fixed_test_far": reference["test_false_alarm_rate"],
                })
    best = max(candidates, key=lambda x: (x["valid_f1"], -x["valid_far"]))
    return candidates, best


def main() -> int:
    source = pd.read_parquet(NOAA)
    rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for fold, bounds in FOLDS.items():
        for seed in SEEDS:
            candidates, best = evaluate_fold(source, fold, bounds, seed)
            rows.extend(candidates)
            selected.append(best)
            print(json.dumps({"fold": fold, "seed": seed, "selected": best, "candidate_count": len(candidates)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected)
    selected_df["f1_gain_vs_fixed"] = selected_df["test_f1"] - selected_df["fixed_test_f1"]
    selected_df["auprc_gain_vs_fixed"] = selected_df["test_auprc"] - selected_df["fixed_test_auprc"]
    aggregate = selected_df.groupby("fold", sort=False)[["test_f1", "fixed_test_f1", "f1_gain_vs_fixed", "test_recall", "test_auprc", "auprc_gain_vs_fixed", "test_far", "fixed_test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["fold"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    metrics = ["valid_f1", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far", "fixed_test_f1", "fixed_test_recall", "fixed_test_auprc", "fixed_test_far"]
    summary = {metric: {"mean": float(selected_df[metric].mean()), "std": float(selected_df[metric].std(ddof=1)), "min": float(selected_df[metric].min()), "max": float(selected_df[metric].max())} for metric in metrics}
    f1_gain = float(selected_df["f1_gain_vs_fixed"].mean())
    auprc_gain = float(selected_df["auprc_gain_vs_fixed"].mean())
    fold_gain = selected_df.groupby("fold")["f1_gain_vs_fixed"].mean()
    positive_folds = int((fold_gain > 0).sum())
    status = "CANDIDATE_GAIN_REQUIRES_MORE_SEASONS" if f1_gain > 0 and positive_folds >= 2 else "NEGATIVE_RESULT_RETAINED"
    payload = {
        "status": status,
        "protocol": "NOAA三个预先固定滚动季节窗口、五个攻击种子；层级中位数/MAD仅由每折train正常样本估计；收缩系数、小时权重和分位阈值只在每折valid选择；test只评价；攻击为透明受控注入。",
        "folds": FOLDS,
        "seeds": SEEDS,
        "grid": {"shrink_k": [20.0, 50.0, 100.0, 200.0], "hour_weight": [0.0, 0.5, 1.0], "q": [0.90, 0.95, 0.975, 0.99]},
        "summary": summary,
        "aggregate_by_fold": aggregate.to_dict(orient="records"),
        "selected_by_valid": selected,
        "mean_f1_gain_vs_fixed": f1_gain,
        "mean_auprc_gain_vs_fixed": auprc_gain,
        "folds_with_positive_mean_f1_gain": positive_folds,
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage62_noaa_hierarchical_rolling_review_detail.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage62_noaa_hierarchical_rolling_review_selected.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage62_noaa_hierarchical_rolling_review_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage62_noaa_hierarchical_rolling_review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段62 NOAA层级校准滚动季节复核", "", "每个窗口只用train正常样本估计层级风险分数校准，valid选择参数，test只评价；受控注入不代表真实攻击。", "", "| 窗口 | 层级F1 | 固定F1 | F1差值 | Recall | AUPRC | AUPRC差值 | FAR | 固定FAR |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['fold']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['fixed_test_f1_mean']:.4f}±{row['fixed_test_f1_std']:.4f} | {row['f1_gain_vs_fixed_mean']:.4f} | {row['test_recall_mean']:.4f} | {row['test_auprc_mean']:.4f} | {row['auprc_gain_vs_fixed_mean']:.4f} | {row['test_far_mean']:.4f} | {row['fixed_test_far_mean']:.4f} |")
    lines += ["", f"- 状态：**{status}**", f"- 全部折/种子平均 F1 差值：`{f1_gain:.4f}`；AUPRC差值：`{auprc_gain:.4f}`；F1正向窗口：`{positive_folds}/{len(FOLDS)}`。", "- 该候选不能自动替换9B主模型；若F1未稳定提升，只作为低误报/AUPRC候选和负结果保存。"]
    (REPORTS / "stage62_noaa_hierarchical_rolling_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "mean_f1_gain_vs_fixed": f1_gain, "mean_auprc_gain_vs_fixed": auprc_gain, "positive_folds": positive_folds}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
