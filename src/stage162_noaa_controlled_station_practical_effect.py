from __future__ import annotations

"""Stage 162: station-level practical-effect sensitivity for stage 161.

This is a frozen post-hoc heterogeneity audit.  It reuses the station-level
test rows from stage 161, does not retrain or select a model, and keeps weak
stations and non-estimable metrics visible instead of replacing them with 0.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = TABLES / "stage161_noaa_controlled_station_regime_performance_detail.csv"
METHODS = ["TrustMeteo-QC default", "TrustMeteo-QC frozen expanded"]
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]
MARGINS = {
    "test_f1": 0.02,
    "test_recall": 0.05,
    "test_precision": 0.02,
    "test_auprc": 0.02,
    "test_false_alarm_rate": 0.01,
}
REPEATS = 10000


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def station_bootstrap(values: np.ndarray, seed: int) -> tuple[float | None, float | None]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(REPEATS, len(values)))]
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def classify(diffs: np.ndarray, margin: float) -> str:
    positive = int((diffs >= margin).sum())
    negative = int((diffs <= -margin).sum())
    if positive == len(diffs):
        return "all_stations_practical_gain"
    if negative == len(diffs):
        return "all_stations_practical_loss"
    if int((np.abs(diffs) < margin).sum()) == len(diffs):
        return "all_stations_within_margin"
    if positive > len(diffs) / 2:
        return "mixed_majority_gain"
    if negative > len(diffs) / 2:
        return "mixed_majority_loss"
    return "mixed_or_small"


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    detail = pd.read_csv(SOURCE)
    required = {"method", "context_variable", "context", "station_key", *METRICS}
    missing = sorted(required.difference(detail.columns))
    if missing:
        raise ValueError(f"stage161 detail missing columns: {missing}")

    grouped = (
        detail[detail["method"].isin(METHODS)]
        .groupby(["method", "context_variable", "context", "station_key"], sort=True)[METRICS]
        .mean()
        .reset_index()
    )
    station_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    not_estimable: list[dict[str, object]] = []

    contexts = detail[["context_variable", "context"]].drop_duplicates().sort_values(["context_variable", "context"])
    for context_row in contexts.itertuples(index=False):
        variable = str(context_row.context_variable)
        context = str(context_row.context)
        left = grouped[(grouped["method"] == METHODS[1]) & grouped["context_variable"].eq(variable) & grouped["context"].eq(context)].set_index("station_key")
        right = grouped[(grouped["method"] == METHODS[0]) & grouped["context_variable"].eq(variable) & grouped["context"].eq(context)].set_index("station_key")
        common = left[METRICS].add_suffix("_expanded").join(right[METRICS].add_suffix("_default"), how="inner")
        for metric in METRICS:
            expanded_col = f"{metric}_expanded"
            default_col = f"{metric}_default"
            pairs = common[[expanded_col, default_col]].dropna()
            if len(pairs) < 2:
                not_estimable.append({"metric": metric, "context_variable": variable, "context": context, "station_count": int(len(pairs)), "practical_margin": MARGINS[metric], "reason": "fewer than two stations with estimable paired metric"})
                continue
            diffs = pairs[expanded_col].to_numpy(dtype=float) - pairs[default_col].to_numpy(dtype=float)
            margin = MARGINS[metric]
            low, high = station_bootstrap(diffs, stable_seed("stage162", metric, variable, context))
            for station_key, row in pairs.iterrows():
                diff = float(row[expanded_col] - row[default_col])
                station_rows.append({
                    "metric": metric,
                    "context_variable": variable,
                    "context": context,
                    "station_key": str(station_key),
                    "expanded_mean": float(row[expanded_col]),
                    "default_mean": float(row[default_col]),
                    "diff_expanded_minus_default": diff,
                    "practical_margin": margin,
                    "positive_practical_effect": bool(diff >= margin),
                    "negative_practical_effect": bool(diff <= -margin),
                    "within_margin": bool(abs(diff) < margin),
                })
            weak = pairs.index[(diffs <= -margin)].astype(str).tolist()
            small_or_negative = pairs.index[(diffs < margin)].astype(str).tolist()
            summary_rows.append({
                "metric": metric,
                "context_variable": variable,
                "context": context,
                "station_count": int(len(diffs)),
                "practical_margin": margin,
                "mean_station_diff": float(diffs.mean()),
                "median_station_diff": float(np.median(diffs)),
                "station_bootstrap_ci95_low": low,
                "station_bootstrap_ci95_high": high,
                "positive_effect_station_count": int((diffs >= margin).sum()),
                "negative_effect_station_count": int((diffs <= -margin).sum()),
                "within_margin_station_count": int((np.abs(diffs) < margin).sum()),
                "weak_station_count": len(weak),
                "weak_station_keys": ";".join(weak),
                "small_or_negative_station_keys": ";".join(small_or_negative),
                "conclusion": classify(diffs, margin),
            })

    station_frame = pd.DataFrame(station_rows)
    summary_frame = pd.DataFrame(summary_rows)
    not_estimable_frame = pd.DataFrame(not_estimable)
    prefix = "stage162_noaa_controlled_station_practical_effect"
    station_frame.to_csv(TABLES / f"{prefix}_station_detail.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_csv(TABLES / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    not_estimable_frame.to_csv(TABLES / f"{prefix}_not_estimable.csv", index=False, encoding="utf-8-sig")
    weak_frame = summary_frame[["metric", "context_variable", "context", "weak_station_count", "weak_station_keys", "small_or_negative_station_keys"]].copy() if not summary_frame.empty else pd.DataFrame()
    weak_frame.to_csv(TABLES / f"{prefix}_weak_stations.csv", index=False, encoding="utf-8-sig")

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段161站点×状态test明细；不重新训练、不重新选择候选、不根据站点test结果调参。按站点先聚合四折五种子，再比较冻结扩展与default的F1、Recall、Precision、AUPRC和FAR；预先固定F1/Precision/AUPRC=0.02、Recall=0.05、FAR=0.01实际效应边界，执行站点bootstrap并完整列出弱站点。NOAA攻击为透明受控注入，不是真实攻击日志；不可估计指标保留为空。",
        "source_artifact": SOURCE.name,
        "methods": METHODS,
        "margins": MARGINS,
        "repeats": REPEATS,
        "station_detail_rows": int(len(station_frame)),
        "summary_rows": int(len(summary_frame)),
        "not_estimable_count": int(len(not_estimable_frame)),
        "summary": summary_frame.to_dict(orient="records"),
        "not_estimable": not_estimable_frame.to_dict(orient="records"),
        "limitations": [
            "站点数量和站点覆盖来自阶段161的冻结外部协议，站点bootstrap不创造新的独立样本。",
            "弱站点不能删除或隐藏；站点层实际效应不替代多重比较校正，也不证明跨地区普适性。",
            "受控攻击标签由透明注入生成，不能写成真实攻击日志验证。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    lines = ["# 阶段162 NOAA受控攻击站点层实际效应", "", "复用阶段161冻结站点层test结果；按站点汇总并计算实际效应边界和bootstrap区间。", "", "| 指标 | 状态变量 | 状态 | 站点数 | 边界 | 平均差 | CI95 | 正向站点 | 负向站点 | 弱站点 | 结论 |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary_rows:
        lines.append(f"| {row['metric']} | {row['context_variable']} | {row['context']} | {int(row['station_count'])} | {row['practical_margin']:.3f} | {row['mean_station_diff']:+.6f} | [{row['station_bootstrap_ci95_low']:+.6f},{row['station_bootstrap_ci95_high']:+.6f}] | {int(row['positive_effect_station_count'])} | {int(row['negative_effect_station_count'])} | {int(row['weak_station_count'])} | {row['conclusion']} |" )
    lines += ["", f"可估计单元：`{len(summary_rows)}`；不可估计单元：`{len(not_estimable_frame)}`。", "弱站点列表已单独写入 `stage162_noaa_controlled_station_practical_effect_weak_stations.csv`；不能据此删除站点或宣称普遍领先。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "station_detail_rows": len(station_frame), "summary_rows": len(summary_frame), "not_estimable": len(not_estimable_frame)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
