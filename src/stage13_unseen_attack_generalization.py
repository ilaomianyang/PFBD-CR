from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402


def main() -> int:
    df = pd.read_parquet(ROOT / "tables" / "stage09b_local_robust_scores.parquet")
    valid = df[df["split"].eq("valid")].copy(); test = df[df["split"].eq("test")].copy()
    types = sorted(valid.loc[valid["combined_is_attack"], "combined_attack_type"].unique().tolist())
    rows = []
    for attack_type in types:
        train_valid = valid[valid["combined_attack_type"].ne(attack_type) | ~valid["combined_is_attack"]]
        yv = train_valid["combined_is_attack"].to_numpy(dtype=bool)
        threshold, valid_f1 = _best_threshold(yv, train_valid["robust_fused_risk"].to_numpy(dtype=float))
        y_type = test["combined_is_attack"].to_numpy(dtype=bool) & test["combined_attack_type"].eq(attack_type).to_numpy(dtype=bool)
        pred = test["robust_fused_risk"].to_numpy(dtype=float) >= threshold
        rows.append({"held_out_attack_type": attack_type, "valid_rows_used": len(train_valid), "threshold": threshold, "valid_f1_other_types": valid_f1, "test_recall_held_out_type": float(pred[y_type].mean()) if y_type.any() else float("nan"), "test_precision_one_vs_rest": _metrics(y_type, pred, test["robust_fused_risk"])["precision"], "test_f1_one_vs_rest": _metrics(y_type, pred, test["robust_fused_risk"])["f1"], "test_false_alarm_rate": float(pred[~test["combined_is_attack"].to_numpy(dtype=bool)].mean())})
    result = pd.DataFrame(rows)
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    result.to_csv(out_dir / "stage13_unseen_attack_generalization.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "每次从valid中留出一种攻击类型，不用其标签选阈值；test只评价留出类型和正常样本", "results": result.to_dict(orient="records"), "mean_held_out_recall": float(result["test_recall_held_out_type"].mean())}
    (out_dir / "stage13_unseen_attack_generalization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段13未知攻击类型留出验证", "", "每次留出一种攻击类型，不使用该类型的 valid 攻击标签选择阈值，再在 test 评价该类型。", "", "| 留出攻击类型 | test Recall | test F1 | test Precision | 正常误伤率 |", "|---|---:|---:|---:|---:|"]
    for _, r in result.iterrows():
        lines.append(f"| {r['held_out_attack_type']} | {r['test_recall_held_out_type']:.4f} | {r['test_f1_one_vs_rest']:.4f} | {r['test_precision_one_vs_rest']:.4f} | {r['test_false_alarm_rate']:.4f} |")
    lines += ["", f"平均留出类型 Recall：`{payload['mean_held_out_recall']:.4f}`", "该实验用于评估未知攻击类型泛化，不得将同一类型的验证集调参结果替代它。"]
    report = report_dir / "stage13_unseen_attack_generalization.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report), "mean_held_out_recall": payload["mean_held_out_recall"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
