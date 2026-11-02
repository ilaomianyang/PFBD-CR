from __future__ import annotations

"""Stage 168: practical-effect margins for stage 166 interactions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = TABLES / "stage166_noaa_controlled_attack_environment_interaction_detail.csv"
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]
BASELINES = ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
MARGINS = {"test_f1": 0.02, "test_recall": 0.05, "test_precision": 0.02, "test_auprc": 0.02, "test_false_alarm_rate": 0.01}
REPEATS = 10000


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float | None, float | None]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(REPEATS, len(values)))]
    means = sample.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def classify(diffs: np.ndarray, margin: float) -> str:
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
    contexts = detail[["context_variable", "context", "attack_type"]].drop_duplicates().sort_values(["context_variable", "context", "attack_type"])
    rows: list[dict[str, object]] = []
    not_estimable: list[dict[str, object]] = []
    keys = ["fold", "seed", "test_year", "context_variable", "context", "attack_type"]
    for baseline in BASELINES:
        other = detail[detail["method"].eq(baseline)]
        common = expanded.set_index(keys).join(other.set_index(keys), lsuffix="_expanded", rsuffix="_baseline", how="inner").reset_index()
        for metric in METRICS:
            common["diff"] = pd.to_numeric(common[f"{metric}_expanded"], errors="coerce") - pd.to_numeric(common[f"{metric}_baseline"], errors="coerce")
            for context_row in contexts.itertuples(index=False):
                variable, context, attack_type = str(context_row.context_variable), str(context_row.context), str(context_row.attack_type)
                group = common[common["context_variable"].eq(variable) & common["context"].eq(context) & common["attack_type"].eq(attack_type)].dropna(subset=["diff"])
                diffs = group.groupby("fold", sort=True)["diff"].mean().dropna().to_numpy(dtype=float)
                margin = MARGINS[metric]
                if len(diffs) < 2:
                    not_estimable.append({"comparison": f"expanded - {baseline}", "metric": metric, "context_variable": variable, "context": context, "attack_type": attack_type, "fold_count": int(len(diffs)), "practical_margin": margin, "reason": "fewer than two folds with estimable metric"})
                    continue
                low, high = bootstrap_mean(diffs, stable_seed("stage168", baseline, metric, variable, context, attack_type))
                rows.append({
                    "comparison": f"expanded - {baseline}", "metric": metric, "context_variable": variable, "context": context, "attack_type": attack_type,
                    "fold_count": int(len(diffs)), "practical_margin": margin, "mean_fold_diff": float(diffs.mean()), "median_fold_diff": float(np.median(diffs)),
                    "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                    "positive_margin_folds": int((diffs >= margin).sum()), "negative_margin_folds": int((diffs <= -margin).sum()), "within_margin_folds": int((np.abs(diffs) < margin).sum()),
                    "conclusion": classify(diffs, margin),
                })
    out = pd.DataFrame(rows)
    not_estimable_out = pd.DataFrame(not_estimable)
    prefix = "stage168_noaa_controlled_attack_environment_practical_effect"
    out.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    not_estimable_out.to_csv(TABLES / f"{prefix}_not_estimable.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段166攻击类型×环境状态交互的四折滚动、五种子和valid阈值选择结果；按fold先汇总五种子差异，预先固定F1/Precision/AUPRC=0.02、Recall=0.05、FAR=0.01实际效应边界，计算交互单元的bootstrap区间、越界fold数和稳定性结论；test只评价，不用于选择或回写。不重新训练、不重新选择候选、不根据实际效应结果调参；攻击为透明受控注入，不是真实攻击日志；不可估计单元保留为空。",
        "source_artifact": SOURCE.name,
        "comparisons": [f"expanded - {baseline}" for baseline in BASELINES],
        "metrics": METRICS,
        "margins": MARGINS,
        "repeats": REPEATS,
        "hypothesis_count": int(len(out)),
        "not_estimable_count": int(len(not_estimable_out)),
        "detail": out.to_dict(orient="records"),
        "not_estimable": not_estimable_out.to_dict(orient="records"),
        "conclusion_counts": out["conclusion"].value_counts().to_dict() if not out.empty else {},
        "limitations": ["实际效应边界是预先固定的分析约定，不替代多重比较或真实攻击验证。", "四个fold限制交互稳定性判断；不可估计单元不能填0。", "弱攻击类型和负结果必须与正向结果并列报告。"],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 阶段168 NOAA攻击类型×环境交互实际效应", "", "预先固定F1/Precision/AUPRC=0.02、Recall=0.05、FAR=0.01实际效应边界，按fold判断稳定增益或损失。", "", "| 比较 | 指标 | 状态变量 | 状态 | 攻击类型 | 边界 | fold数 | 平均差 | CI95 | 正向越界 | 负向越界 | 结论 |", "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['context_variable']} | {row['context']} | {row['attack_type']} | {row['practical_margin']:.3f} | {row['fold_count']} | {row['mean_fold_diff']:+.6f} | [{row['bootstrap_ci95_low']:+.6f},{row['bootstrap_ci95_high']:+.6f}] | {row['positive_margin_folds']} | {row['negative_margin_folds']} | {row['conclusion']} |")
    lines += ["", f"可估计单元：{len(rows)}；不可估计单元：{len(not_estimable_out)}。", "实际效应边界用于透明区分工程意义，不替代统计显著性；受控攻击不是真实攻击日志。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "hypotheses": len(rows), "not_estimable": len(not_estimable_out), "conclusions": payload["conclusion_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
