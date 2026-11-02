from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def main() -> int:
    detail = pd.read_csv(TABLES / "stage91_noaa_physical_component_repair_detail.csv")
    by_type = pd.read_csv(TABLES / "stage91_noaa_physical_component_repair_by_type.csv")
    diagnostics = pd.read_csv(TABLES / "stage91_noaa_physical_component_repair_diagnostics.csv")
    selected = pd.read_csv(TABLES / "stage91_noaa_physical_component_repair_selected.csv")
    aggregate = detail.groupby("variant", sort=False)[["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far", "train_natural_alarm_rate"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["variant"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    selected_summary = {key: {"mean": float(selected[key].mean()), "std": float(selected[key].std(ddof=1)), "min": float(selected[key].min()), "max": float(selected[key].max())} for key in ["selected_valid_f1", "selected_test_f1", "selected_test_recall", "selected_test_auprc", "selected_test_far"]}
    legacy = aggregate.loc[aggregate["variant"].eq("legacy")].iloc[0]
    candidate = aggregate.loc[aggregate["variant"].eq("component_zmax")].iloc[0]
    candidate_gain = {"f1": float(candidate["test_f1_mean"] - legacy["test_f1_mean"]), "recall": float(candidate["test_recall_mean"] - legacy["test_recall_mean"]), "auprc": float(candidate["test_auprc_mean"] - legacy["test_auprc_mean"]), "far": float(candidate["test_far_mean"] - legacy["test_far_mean"])}
    selected_gain = float(selected_summary["selected_test_f1"]["mean"] - legacy["test_f1_mean"])
    old = json.loads((TABLES / "stage91_noaa_physical_component_repair.json").read_text(encoding="utf-8"))
    payload = {**old, "status": "CANDIDATE_LOW_FAR_REQUIRES_BOOTSTRAP" if candidate_gain["far"] < 0 and candidate_gain["auprc"] > 0 else "NEGATIVE_RESULT_RETAINED", "candidate_variant": "component_zmax", "detail": detail.to_dict(orient="records"), "aggregate": aggregate.to_dict(orient="records"), "by_type": by_type.to_dict(orient="records"), "diagnostics": diagnostics.to_dict(orient="records"), "selected_by_valid": selected.to_dict(orient="records"), "selected_summary": selected_summary, "selected_test_f1_gain_vs_legacy": selected_gain, "candidate_gain_vs_legacy": candidate_gain, "limitations": ["component_zmax尚未经过站点簇bootstrap；不得替换9B主模型。", "组件方案明显牺牲Recall，只能作为低误报运行点候选。", "所有攻击均为透明受控注入，不是自然攻击日志。", "自然观测报警率需结合阶段87/90解释，不能当作攻击召回率。"]}
    (TABLES / "stage91_noaa_physical_component_repair.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段91 NOAA物理规则组件级修复基准", "", "将范围、变化率和重复段拆开进行训练期稳健归一化；四折滚动、五种子、六类受控攻击；每折运行设置只由valid选择。", "", f"- 状态：**{payload['status']}**", f"- 预先固定低误报候选 `component_zmax` 相对 legacy：F1 `{candidate_gain['f1']:+.4f}`；Recall `{candidate_gain['recall']:+.4f}`；AUPRC `{candidate_gain['auprc']:+.4f}`；FAR `{candidate_gain['far']:+.4f}`", f"- 按valid选择方案后 test F1 相对 legacy：`{selected_gain:+.4f}`（验证集无法区分组件方案）", "", "| 方案 | test F1 | Recall | Precision | AUPRC | FAR | train自然报警率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['variant']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} | {row['train_natural_alarm_rate_mean']:.4f} |")
    lines += ["", "component_zmax只作为低误报运行点候选；它牺牲了大量Recall，必须经过站点簇配对bootstrap和自然观测复核后，才可讨论是否进入主模型。"]
    (REPORTS / "stage91_noaa_physical_component_repair.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "candidate_gain_vs_legacy": candidate_gain, "selected_test_f1_gain_vs_legacy": selected_gain}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
