from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def main() -> int:
    baseline = pd.read_csv(TABLES / "stage41_noaa_baseline_comparison_detail.csv")
    candidate = pd.read_csv(TABLES / "stage64_noaa_empirical_percentile_calibration_selected.csv")
    candidate = candidate.rename(columns={"test_recall": "recall", "test_precision": "precision", "test_auprc": "auprc", "test_far": "false_alarm_rate", "reference_test_f1": "reference_f1"})
    candidate["method"] = "Empirical percentile calibration"
    candidate = candidate[["seed", "method", "test_f1", "recall", "precision", "auprc", "false_alarm_rate"]]
    baseline = baseline.rename(columns={"test_recall": "recall", "test_precision": "precision", "test_auprc": "auprc", "test_false_alarm_rate": "false_alarm_rate"})
    baseline = baseline[["seed", "method", "test_f1", "recall", "precision", "auprc", "false_alarm_rate"]]
    all_rows = pd.concat([baseline, candidate], ignore_index=True)
    metrics = ["test_f1", "recall", "precision", "auprc", "false_alarm_rate"]
    summary = all_rows.groupby("method", sort=False)[metrics].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = ["method"] + [f"{a}_{b}" for a, b in summary.columns.tolist()[1:]]
    pairs = []
    for method in ["Robust max deviation", "Isolation Forest", "One-Class SVM", "TrustMeteo-QC"]:
        left = candidate.set_index("seed")
        right = baseline[baseline["method"].eq(method)].set_index("seed")
        merged = left.join(right, lsuffix="_candidate", rsuffix="_baseline", how="inner")
        pairs.append({"candidate": "Empirical percentile calibration", "baseline": method, "f1_diff_mean": float((merged["test_f1_candidate"] - merged["test_f1_baseline"]).mean()), "recall_diff_mean": float((merged["recall_candidate"] - merged["recall_baseline"]).mean()), "auprc_diff_mean": float((merged["auprc_candidate"] - merged["auprc_baseline"]).mean()), "far_diff_mean": float((merged["false_alarm_rate_candidate"] - merged["false_alarm_rate_baseline"]).mean()), "seeds": int(len(merged))})
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "同一NOAA 12站点、同一时间切分、同一五种子；候选来自阶段64 valid选择结果；不重新调参；仅做外部公平汇总。", "methods": summary["method"].tolist(), "summary": summary.to_dict(orient="records"), "paired_differences": pairs, "interpretation": "候选的FAR/AUPRC改善必须和F1、Recall及无监督基线共同解释，不能宣称全面领先。"}
    all_rows.to_csv(TABLES / "stage66_noaa_candidate_baseline_comparison_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "stage66_noaa_candidate_baseline_comparison_summary.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage66_noaa_candidate_baseline_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段66 NOAA候选与基线公平比较", "", "同一外部数据、时间切分和五个种子；候选不重新调参，仅汇总阶段64和阶段41结果。", "", "| 方法 | F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['recall_mean']:.4f}±{row['recall_std']:.4f} | {row['precision_mean']:.4f}±{row['precision_std']:.4f} | {row['auprc_mean']:.4f}±{row['auprc_std']:.4f} | {row['false_alarm_rate_mean']:.4f}±{row['false_alarm_rate_std']:.4f} |")
    lines += ["", "## 候选相对基线的配对均值差", "", "| 基线 | F1差值 | Recall差值 | AUPRC差值 | FAR差值 |", "|---|---:|---:|---:|---:|"]
    for row in pairs:
        lines.append(f"| {row['baseline']} | {row['f1_diff_mean']:.4f} | {row['recall_diff_mean']:.4f} | {row['auprc_diff_mean']:.4f} | {row['far_diff_mean']:.4f} |")
    lines += ["", "该比较用于定位候选的适用运行点，不支持‘全面优于所有基线’的表述。"]
    (REPORTS / "stage66_noaa_candidate_baseline_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "methods": payload["methods"], "paired_comparisons": len(pairs)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
