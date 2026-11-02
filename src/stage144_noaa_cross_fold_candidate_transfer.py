from __future__ import annotations

"""Stage 144: cross-fold transfer of the frozen Trust/SVM candidate choice.

Candidate selection is performed from valid metrics in the other three rolling
folds only.  The held-out fold's test metrics are then evaluated descriptively.
No test metric participates in candidate selection; 不根据test重新选择、不重新训练。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = TABLES / "stage143_noaa_frozen_trust_svm_safety_combination_detail.csv"
SELECTION_SOURCE = TABLES / "stage143_noaa_frozen_trust_svm_safety_combination_selected.csv"
FAR_CAP = 0.10
CANDIDATES = ["Trust frozen expanded", "One-Class SVM", "Trust-SVM mean alpha 0.25", "Trust-SVM mean alpha 0.50", "Trust-SVM mean alpha 0.75", "Trust-SVM max"]
METRICS = ["f1", "recall", "precision", "auprc", "far"]


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_cluster(values: pd.DataFrame, seed: int, repeats: int = 10000) -> tuple[float, float]:
    folds = values["held_out_fold"].drop_duplicates().tolist()
    grouped = {fold: values.loc[values["held_out_fold"].eq(fold), "diff"].to_numpy(dtype=float) for fold in folds}
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for i in range(repeats):
        sampled = rng.choice(folds, size=len(folds), replace=True)
        means[i] = np.concatenate([grouped[fold] for fold in sampled]).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    detail = pd.read_csv(SOURCE)
    selected = pd.read_csv(SELECTION_SOURCE)
    expected_folds = sorted(detail["fold"].astype(str).unique().tolist())
    if len(expected_folds) != 4 or detail["method"].nunique() != 6 or len(detail) != 120:
        raise ValueError("stage143 detail does not match the frozen protocol")
    transfer_rows: list[dict[str, object]] = []
    transfer_detail: list[dict[str, object]] = []
    for held_out in expected_folds:
        source = detail.loc[detail["fold"].ne(held_out)].copy()
        feasible = source.loc[source["valid_far"] <= FAR_CAP]
        if feasible.empty:
            feasible = source
        summary = feasible.groupby("method", sort=False)[["valid_f1", "valid_auprc", "valid_far"]].mean().reset_index()
        chosen = summary.sort_values(["valid_f1", "valid_auprc", "valid_far"], ascending=[False, False, True]).iloc[0]
        method = str(chosen["method"])
        test = detail.loc[(detail["fold"] == held_out) & (detail["method"] == method)].copy()
        svm = detail.loc[(detail["fold"] == held_out) & (detail["method"] == "One-Class SVM")].copy()
        for _, row in test.iterrows():
            transfer_detail.append({"held_out_fold": held_out, "seed": int(row["seed"]), "selected_method": method, **{f"test_{metric}": float(row[f"test_{metric}"]) for metric in METRICS}, "source_valid_f1": float(chosen["valid_f1"]), "source_valid_auprc": float(chosen["valid_auprc"]), "source_valid_far": float(chosen["valid_far"]), "svm_test_f1": float(svm.loc[svm["seed"] == row["seed"], "test_f1"].iloc[0])})
        aggregate = {"held_out_fold": held_out, "selected_method": method, "source_folds": "|".join(fold for fold in expected_folds if fold != held_out), "source_valid_f1": float(chosen["valid_f1"]), "source_valid_auprc": float(chosen["valid_auprc"]), "source_valid_far": float(chosen["valid_far"]), "test_f1_mean": float(test["test_f1"].mean()), "test_f1_std": float(test["test_f1"].std(ddof=1)), "test_recall_mean": float(test["test_recall"].mean()), "test_precision_mean": float(test["test_precision"].mean()), "test_auprc_mean": float(test["test_auprc"].mean()), "test_far_mean": float(test["test_far"].mean()), "svm_test_f1_mean": float(svm["test_f1"].mean())}
        aggregate["test_f1_diff_vs_svm"] = aggregate["test_f1_mean"] - aggregate["svm_test_f1_mean"]
        transfer_rows.append(aggregate)

    transfer = pd.DataFrame(transfer_rows)
    held_detail = pd.DataFrame(transfer_detail)
    paired_rows: list[dict[str, object]] = []
    for metric in METRICS:
        values = []
        for _, row in held_detail.iterrows():
            method_row = detail.loc[(detail["fold"] == row["held_out_fold"]) & (detail["seed"] == row["seed"]) & (detail["method"] == row["selected_method"])].iloc[0]
            svm_row = detail.loc[(detail["fold"] == row["held_out_fold"]) & (detail["seed"] == row["seed"]) & (detail["method"] == "One-Class SVM")].iloc[0]
            values.append({"held_out_fold": row["held_out_fold"], "diff": float(method_row[f"test_{metric}"] - svm_row[f"test_{metric}"])})
        diff_frame = pd.DataFrame(values)
        low, high = bootstrap_cluster(diff_frame, stable_seed("stage144", metric))
        paired_rows.append({"comparison": "cross-fold valid-selected - One-Class SVM", "metric": metric, "n": len(diff_frame), "mean_diff": float(diff_frame["diff"].mean()), "bootstrap_fold_ci95_low": low, "bootstrap_fold_ci95_high": high})
    paired = pd.DataFrame(paired_rows)
    selection_counts = transfer["selected_method"].value_counts().rename_axis("method").reset_index(name="fold_count")
    selection_counts["fold_fraction"] = selection_counts["fold_count"] / len(transfer)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段143冻结六候选的逐折逐种子valid指标；每次留出一个滚动折，只用其余三折的valid F1/AUPRC/FAR选择一个固定候选，随后评价留出折test；不使用留出折test做选择，不根据test重新选择、不重新训练。攻击为透明受控注入，不是真实攻击日志。",
        "source_artifact": SOURCE.name,
        "folds": expected_folds,
        "candidate_methods": CANDIDATES,
        "valid_far_cap": FAR_CAP,
        "transfer": transfer.to_dict(orient="records"),
        "transfer_detail": held_detail.to_dict(orient="records"),
        "selection_counts": selection_counts.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "limitations": ["本阶段是基于阶段143明细的跨折选择审计，不增加新的原始观测或攻击样本。", "候选选择只使用源折valid指标；留出折test仅评价。", "受控注入仍不等同于真实攻击日志；若区间跨零，不得宣称组合普适优于SVM。"],
    }
    prefix = "stage144_noaa_cross_fold_candidate_transfer"
    transfer.to_csv(TABLES / f"{prefix}.csv", index=False, encoding="utf-8-sig")
    held_detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    selection_counts.to_csv(TABLES / f"{prefix}_selection_counts.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段144 NOAA跨折候选迁移审计", "", "每次留出一个滚动折，仅用其余三折valid选择固定候选，再评价留出折test。", "", "| 留出折 | 选择候选 | 源valid F1 | test F1 | Recall | Precision | AUPRC | FAR | SVM F1 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in transfer.to_dict(orient="records"):
        lines.append(f"| {row['held_out_fold']} | {row['selected_method']} | {row['source_valid_f1']:.4f} | {row['test_f1_mean']:.4f} | {row['test_recall_mean']:.4f} | {row['test_precision_mean']:.4f} | {row['test_auprc_mean']:.4f} | {row['test_far_mean']:.4f} | {row['svm_test_f1_mean']:.4f} |")
    lines += ["", "## 相对SVM的折聚类成对差异", "", "| 指标 | 平均差 | CI95 |", "|---|---:|---:|"]
    for row in paired.to_dict(orient="records"):
        lines.append(f"| {row['metric']} | {row['mean_diff']:.4f} | [{row['bootstrap_fold_ci95_low']:.4f}, {row['bootstrap_fold_ci95_high']:.4f}] |")
    lines += ["", "跨折选择仅使用源折valid指标；test不参与候选选择。该结果用于迁移稳定性审计，不支持全面领先声明。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "folds": len(transfer), "detail": len(held_detail), "paired": len(paired)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
