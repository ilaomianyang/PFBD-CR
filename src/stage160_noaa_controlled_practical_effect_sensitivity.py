from __future__ import annotations

"""Stage 160: predefined practical-effect margins for stage-158 strata."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = TABLES / "stage158_noaa_controlled_regime_gap_performance_detail.csv"
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]
MARGINS = {"test_f1": 0.02, "test_recall": 0.05, "test_precision": 0.02, "test_auprc": 0.02, "test_false_alarm_rate": 0.01}
REPEATS = 10000


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(REPEATS, len(values)))]
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def conclusion(diffs: np.ndarray, margin: float) -> str:
    positive = int((diffs >= margin).sum())
    negative = int((diffs <= -margin).sum())
    if positive == len(diffs):
        return "all_folds_practical_gain"
    if negative == len(diffs):
        return "all_folds_practical_loss"
    if int((np.abs(diffs) < margin).sum()) == len(diffs):
        return "all_folds_within_margin"
    if positive > len(diffs) / 2:
        return "mixed_majority_gain"
    if negative > len(diffs) / 2:
        return "mixed_majority_loss"
    return "mixed_or_small"


def main() -> int:
    detail = pd.read_csv(SOURCE)
    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    default = detail[detail["method"].eq("TrustMeteo-QC default")]
    contexts = detail[["context_variable", "context"]].drop_duplicates().sort_values(["context_variable", "context"]).to_records(index=False)
    rows: list[dict[str, object]] = []
    not_estimable: list[dict[str, object]] = []
    for metric in METRICS:
        left = expanded.groupby(["fold", "context_variable", "context"], sort=True)[metric].mean().rename("expanded")
        right = default.groupby(["fold", "context_variable", "context"], sort=True)[metric].mean().rename("default")
        common = pd.concat([left, right], axis=1)
        for variable, context in contexts:
            key = (variable, context)
            try:
                group = common.xs(key, level=["context_variable", "context"]).dropna()
            except KeyError:
                group = pd.DataFrame(columns=["expanded", "default"])
            diffs = (group["expanded"] - group["default"]).to_numpy(dtype=float) if not group.empty else np.array([], dtype=float)
            margin = MARGINS[metric]
            if len(diffs) < 2:
                not_estimable.append({"metric": metric, "context_variable": variable, "context": context, "fold_count": int(len(diffs)), "margin": margin, "reason": "fewer than two folds with estimable metric"})
                continue
            low, high = bootstrap_mean(diffs, stable_seed("stage160", metric, variable, context))
            rows.append({
                "comparison": "expanded - default",
                "metric": metric,
                "context_variable": variable,
                "context": context,
                "fold_count": int(len(diffs)),
                "practical_margin": margin,
                "mean_fold_diff": float(diffs.mean()),
                "median_fold_diff": float(np.median(diffs)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "positive_margin_folds": int((diffs >= margin).sum()),
                "negative_margin_folds": int((diffs <= -margin).sum()),
                "within_margin_folds": int((np.abs(diffs) < margin).sum()),
                "conclusion": conclusion(diffs, margin),
            })

    out = pd.DataFrame(rows)
    prefix = "stage160_noaa_controlled_practical_effect_sensitivity"
    out.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(not_estimable).to_csv(TABLES / f"{prefix}_not_estimable.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段158四折滚动、五种子和valid阈值选择；test只用于分层评价。按fold先汇总种子差异，对F1/Precision/AUPRC预先固定0.02、Recall预先固定0.05、FAR预先固定0.01的实际效应边界，计算fold bootstrap区间、越过边界的fold数和全部fold结论。不重新训练、不重新选择候选、不依据实际效应结果调参；受控攻击不是真实攻击日志。",
        "source_artifact": SOURCE.name,
        "margins": MARGINS,
        "hypothesis_count": int(len(out)),
        "not_estimable_count": int(len(not_estimable)),
        "detail": out.to_dict(orient="records"),
        "not_estimable": not_estimable,
        "conclusion_counts": out["conclusion"].value_counts().to_dict() if not out.empty else {},
        "limitations": ["实际效应边界是预先固定的分析约定，不是临床或安全标准；应与原始指标和CI同时报告。", "四个fold不足以证明长期稳定性；不可估计状态不填0。", "实际效应分析不改变模型选择。"],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 阶段160 NOAA受控攻击实际效应边界敏感性", "", "预先固定指标边界：F1/Precision/AUPRC=0.02，Recall=0.05，FAR=0.01；按fold计算种子汇总差异和bootstrap区间。", "", "| 指标 | 状态变量 | 状态 | 边界 | fold数 | 平均差 | CI95 | 越过正边界 | 越过负边界 | 结论 |", "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['context_variable']} | {row['context']} | {row['practical_margin']:.3f} | {row['fold_count']} | {row['mean_fold_diff']:+.6f} | [{row['bootstrap_ci95_low']:+.6f},{row['bootstrap_ci95_high']:+.6f}] | {row['positive_margin_folds']} | {row['negative_margin_folds']} | {row['conclusion']} |")
    lines += ["", f"可估计单元：`{len(rows)}`；不可估计单元：`{len(not_estimable)}`。", "实际效应边界用于透明区分小幅波动和可能有意义的变化，不替代统计显著性或真实攻击验证。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "hypotheses": len(rows), "not_estimable": len(not_estimable), "conclusions": payload["conclusion_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
