from __future__ import annotations

"""Stage 146: target-year heterogeneity audit for frozen NOAA results."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = TABLES / "stage141_noaa_frozen_external_validation_by_fold.csv"
METHOD_A = "TrustMeteo-QC frozen expanded"
METHOD_B = "TrustMeteo-QC default"
METHOD_C = "One-Class SVM"
METRICS = ["f1", "recall", "precision", "auprc", "far"]


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_fold(values: pd.DataFrame, column: str, seed: int, repeats: int = 10000) -> tuple[float, float]:
    folds = values["target_year"].to_numpy(dtype=int)
    lookup = values.set_index("target_year")[column].to_dict()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(folds, size=(repeats, len(folds)), replace=True)
    means = np.asarray([[lookup[int(year)] for year in row] for row in sampled], dtype=float).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    source = pd.read_csv(SOURCE)
    expected = {METHOD_A, METHOD_B, METHOD_C}
    if not expected.issubset(set(source["method"].unique())):
        raise ValueError("stage141 fold detail is missing a required method")
    source["target_year"] = source["fold"].astype(str).str.extract(r"(\d{4})$")[0].astype(int)
    source = source[source["method"].isin(expected)].copy()
    wide = source.pivot_table(index="target_year", columns="method", values=["test_f1_mean", "test_recall_mean", "test_precision_mean", "test_auprc_mean", "test_false_alarm_rate_mean"], aggfunc="first").reset_index()
    wide.columns = ["_".join(str(part) for part in column if str(part) != "") if isinstance(column, tuple) else str(column) for column in wide.columns]
    rename = {"f1": "test_f1_mean", "recall": "test_recall_mean", "precision": "test_precision_mean", "auprc": "test_auprc_mean", "far": "test_false_alarm_rate_mean"}
    year_rows: list[dict[str, object]] = []
    for _, row in wide.iterrows():
        item: dict[str, object] = {"target_year": int(row["target_year"])}
        for metric, source_name in rename.items():
            for method, tag in [(METHOD_A, "expanded"), (METHOD_B, "default"), (METHOD_C, "svm")]:
                item[f"{metric}_{tag}"] = float(row[f"{source_name}_{method}"])
            item[f"{metric}_expanded_minus_default"] = item[f"{metric}_expanded"] - item[f"{metric}_default"]
            item[f"{metric}_expanded_minus_svm"] = item[f"{metric}_expanded"] - item[f"{metric}_svm"]
        year_rows.append(item)
    year_summary = pd.DataFrame(year_rows).sort_values("target_year").reset_index(drop=True)
    paired_rows: list[dict[str, object]] = []
    for comparison, suffix in [("expanded - default", "expanded_minus_default"), ("expanded - One-Class SVM", "expanded_minus_svm")]:
        for metric in METRICS:
            column = f"{metric}_{suffix}"
            low, high = bootstrap_fold(year_summary, column, stable_seed("stage146", comparison, metric))
            paired_rows.append({"comparison": comparison, "metric": metric, "fold_count": len(year_summary), "mean_diff": float(year_summary[column].mean()), "fold_bootstrap_ci95_low": low, "fold_bootstrap_ci95_high": high, "positive_years": int((year_summary[column] > 0).sum()), "negative_years": int((year_summary[column] < 0).sum())})
    paired = pd.DataFrame(paired_rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段141冻结NOAA四个滚动目标年份的by-fold结果做年度异质性审计；不重新训练、不重新选择候选、不根据目标年test调参；按目标年做成对fold bootstrap。攻击为透明受控注入，不是真实攻击日志。",
        "source_artifact": SOURCE.name,
        "target_years": year_summary["target_year"].astype(int).tolist(),
        "folds": source["fold"].drop_duplicates().tolist(),
        "methods": [METHOD_A, METHOD_B, METHOD_C],
        "year_summary": year_summary.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "limitations": ["只有四个滚动目标年份，fold bootstrap只能量化当前年度集合的不确定性。", "该审计复用阶段141冻结结果，不增加独立年份或新的攻击样本。", "若年度差异显示Recall与FAR方向相反，必须按年度漂移和安全—召回权衡解释。"],
    }
    prefix = "stage146_noaa_year_heterogeneity_audit"
    year_summary.to_csv(TABLES / f"{prefix}_year_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段146 NOAA目标年度异质性审计", "", "复用阶段141冻结的四个滚动目标年份结果，不重新训练或根据目标年test选择候选。", "", "| 比较 | 指标 | 平均差 | CI95 | 正向年份 | 负向年份 |", "|---|---|---:|---:|---:|---:|"]
    for row in paired.to_dict(orient="records"):
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_diff']:.4f} | [{row['fold_bootstrap_ci95_low']:.4f}, {row['fold_bootstrap_ci95_high']:.4f}] | {row['positive_years']} | {row['negative_years']} |")
    lines += ["", "该分析用于识别跨年度漂移和运行点权衡，不支持跨年份普适性或真实攻击能力声明。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "target_years": len(year_summary), "paired": len(paired)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
