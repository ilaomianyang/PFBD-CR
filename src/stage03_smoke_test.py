from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from detectors import (
    CUSUMDetector,
    HampelMADDetector,
    KalmanInnovationDetector,
    PhysicalRulesDetector,
    SpatialConsistencyDetector,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(r"F:\数据集\Analysis\experiments\pc_mstf_ap_v4\parquet\v4_anomaly_injection_table.parquet")


def best_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(score)
    y, score = y[finite].astype(int), score[finite]
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 193)))
    values = [(f1_score(y, score >= t, zero_division=0), float(t)) for t in candidates]
    return max(values, key=lambda x: x[0])[1], max(values, key=lambda x: x[0])[0]


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "PR_AUC": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "threshold": float(threshold),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TrustMeteo-QC 阶段3轻量检测器 smoke test")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    df = pd.read_parquet(args.data).sort_values(["split", "timestamp_utc", "station_key"]).reset_index(drop=True)
    y = df["is_anomaly"].astype(int).to_numpy()
    train = df["split"].eq("train")
    valid = df["split"].eq("valid")
    test = df["split"].eq("test")
    train_normal = train & ~df["is_anomaly"].astype(bool)
    detectors = [
        PhysicalRulesDetector(),
        HampelMADDetector(),
        KalmanInnovationDetector(),
        CUSUMDetector(),
        SpatialConsistencyDetector(),
    ]
    rows = []
    for detector in detectors:
        detector.fit(df.loc[train], train_normal.loc[train])
        # Calculate temporal/spatial context within each split only.
        score = np.full(len(df), np.nan, dtype=float)
        for split_name in ["train", "valid", "test"]:
            idx = np.flatnonzero(df["split"].eq(split_name).to_numpy())
            score[idx] = detector.score(df.iloc[idx])
        threshold, valid_f1 = best_threshold(y[valid.to_numpy()], score[valid.to_numpy()])
        metrics = evaluate(y[test.to_numpy()], score[test.to_numpy()], threshold)
        rows.append({"detector": detector.name, "valid_F1": valid_f1, **metrics, "finite_scores": int(np.isfinite(score).sum())})
    result = {
        "status": "PASS" if all(r["finite_scores"] == len(df) for r in rows) else "FAIL",
        "data": str(args.data),
        "rows": int(len(df)),
        "detectors": rows,
        "protocol": "train正常样本拟合，valid选择阈值，test只评价；本脚本不修改任何基线结果",
    }
    out_json = ROOT / "tables" / "stage03_smoke_test.json"
    out_md = ROOT / "reports" / "stage03_smoke_test.md"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段3轻量检测器 smoke test", "", f"- 状态：**{result['status']}**", f"- 数据行数：`{len(df)}`", "", "| 检测器 | valid F1 | test F1 | test AUROC | test AUPRC | 阈值 |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['detector']} | {r['valid_F1']:.4f} | {r['F1']:.4f} | {r['ROC_AUC']:.4f} | {r['PR_AUC']:.4f} | {r['threshold']:.6g} |")
    lines += ["", "所有检测器均使用统一 fit/score/predict 接口；阈值只从 valid 选择，test 未参与调参。"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output_json": str(out_json), "output_report": str(out_md)}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
