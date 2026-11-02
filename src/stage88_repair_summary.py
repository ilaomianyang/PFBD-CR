from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def main() -> int:
    detail = pd.read_csv(TABLES / "stage88_noaa_physical_rule_cap_rolling_detail.csv")
    selected = pd.read_csv(TABLES / "stage88_noaa_physical_rule_cap_rolling_selected.csv")
    aggregate = detail.groupby("physical_rule_cap", dropna=False, sort=False)[["valid_f1", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["physical_rule_cap"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    reference = detail[detail["physical_rule_cap"].isna()][["fold", "seed", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].rename(columns={"test_f1": "reference_test_f1", "test_recall": "reference_test_recall", "test_precision": "reference_test_precision", "test_auprc": "reference_test_auprc", "test_far": "reference_test_far"})
    paired = selected.merge(reference, on=["fold", "seed"], how="left", validate="one_to_one")
    selected_vs_reference = {"selected_test_f1": float(paired["test_f1"].mean()), "selected_test_recall": float(paired["test_recall"].mean()), "selected_test_auprc": float(paired["test_auprc"].mean()), "selected_test_far": float(paired["test_far"].mean()), "reference_test_f1": float(paired["reference_test_f1"].mean()), "reference_test_recall": float(paired["reference_test_recall"].mean()), "reference_test_auprc": float(paired["reference_test_auprc"].mean()), "reference_test_far": float(paired["reference_test_far"].mean()), "f1_gain": float((paired["test_f1"] - paired["reference_test_f1"]).mean()), "auprc_gain": float((paired["test_auprc"] - paired["reference_test_auprc"]).mean()), "far_gain": float((paired["test_far"] - paired["reference_test_far"]).mean())}
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_BOOTSTRAP" if selected_vs_reference["f1_gain"] > 0 and selected_vs_reference["auprc_gain"] >= 0 and selected_vs_reference["far_gain"] <= 0 else "NEGATIVE_RESULT_RETAINED", "protocol": "NOAA官方2020-2025扩展表；阶段88已完成的四个滚动折、11站点和5种子；从全量候选明细中恢复同一折/种子下valid选中上限与无上限参考的配对比较。", "selected_vs_uncapped": selected_vs_reference, "aggregate_all_caps": aggregate.to_dict(orient="records"), "selected_by_valid": selected.to_dict(orient="records")}
    aggregate.to_csv(TABLES / "stage88_noaa_physical_rule_cap_rolling_summary.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage88_noaa_physical_rule_cap_rolling.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    s = selected_vs_reference
    lines = ["# 阶段88 NOAA滚动物理规则分量上限", "", "Physical Rules分量上限只由每折valid选择；以下比较严格按同一折/种子配对的valid选中候选与无上限参考。", "", f"- 状态：**{payload['status']}**", f"- 选中候选test F1：`{s['selected_test_f1']:.4f}`；无上限参考：`{s['reference_test_f1']:.4f}`；差值：`{s['f1_gain']:.4f}`", f"- AUPRC差值：`{s['auprc_gain']:.4f}`；FAR差值：`{s['far_gain']:.4f}`；Recall差值：`{s['selected_test_recall'] - s['reference_test_recall']:.4f}`", "", "| cap | valid F1 | test F1 | Recall | Precision | AUPRC | FAR |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        cap = "∞" if pd.isna(row["physical_rule_cap"]) else f"{row['physical_rule_cap']:.1f}"
        lines.append(f"| {cap} | {row['valid_f1_mean']:.4f}±{row['valid_f1_std']:.4f} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f} | {row['test_precision_mean']:.4f} | {row['test_auprc_mean']:.4f} | {row['test_far_mean']:.4f} |")
    lines += ["", "该结果只决定是否值得进一步bootstrap；未通过统计稳定性前不进入主模型。"]
    (REPORTS / "stage88_noaa_physical_rule_cap_rolling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected_vs_uncapped": selected_vs_reference}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
