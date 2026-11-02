from __future__ import annotations

import json
import sys
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402


def main() -> int:
    df = pd.read_parquet(ROOT / "tables" / "stage09b_local_robust_scores.parquet")
    cols = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency", "ERA5 Background", "structural_risk"]
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = df["combined_is_attack"].to_numpy(dtype=int)
    train = df["split"].eq("train").to_numpy(); valid = df["split"].eq("valid").to_numpy(); test = df["split"].eq("test").to_numpy()
    pos_weight = max((y[train] == 0).sum() / max((y[train] == 1).sum(), 1), 1.0)
    weights = np.where(y[train] == 1, pos_weight, 1.0)
    models = {
        "Logistic Regression supervised": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260911)),
        "HistGradientBoosting supervised": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=20260911),
    }
    rows, type_rows = [], []
    for name, model in models.items():
        if name.startswith("Hist"):
            model.fit(X[train], y[train], sample_weight=weights)
        else:
            model.fit(X[train], y[train])
        score = model.predict_proba(X)[:, 1]
        threshold, valid_f1 = _best_threshold(y[valid], score[valid])
        pred = score >= threshold
        m = _metrics(y[test], pred[test], score[test])
        rows.append({"method": name, "training_labels": "train controlled-attack labels", "threshold": threshold, "valid_f1": valid_f1, "test_f1": m["f1"], "test_recall": m["recall"], "test_precision": m["precision"], "test_auprc": m["average_precision"], "test_false_alarm_rate": float(pred[test][y[test] == 0].mean())})
        test_frame = df.loc[test].copy(); test_frame["_pred"] = pred[test]
        for attack_type, group in test_frame[test_frame["combined_is_attack"]].groupby("combined_attack_type", sort=True):
            type_rows.append({"method": name, "attack_type": attack_type, "rows": len(group), "recall": float(group["_pred"].mean()), "seen_in_train": bool((df.loc[train, "combined_attack_type"] == attack_type).any())})
    summary = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    per_type = pd.DataFrame(type_rows)
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    summary.to_csv(out_dir / "stage14_supervised_upper_bounds.csv", index=False, encoding="utf-8-sig")
    per_type.to_csv(out_dir / "stage14_supervised_by_attack_type.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "监督基线使用train受控攻击标签拟合、valid选阈值、test评价；不可与仅正常样本拟合方法混为同类", "results": summary.to_dict(orient="records"), "by_type": per_type.to_dict(orient="records")}
    (out_dir / "stage14_supervised_upper_bounds.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段14监督上界基线", "", "监督模型使用 train 中的受控攻击标签，因此只能作为有标签上界，不能与无监督方法作完全同类比较。", "", "| 方法 | test F1 | Recall | Precision | AUPRC | 正常误伤率 |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['method']} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_precision']:.4f} | {r['test_auprc']:.4f} | {r['test_false_alarm_rate']:.4f} |")
    lines += ["", "扩展攻击 replay、coordinated_offset 和 mixed_missing_attack 在 train 中未出现，其分类型结果用于检查监督模型对未知类型的迁移能力。"]
    report = report_dir / "stage14_supervised_upper_bounds.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
