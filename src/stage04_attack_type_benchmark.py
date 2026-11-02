from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from detectors import CUSUMDetector, HampelMADDetector, KalmanInnovationDetector, PhysicalRulesDetector, SpatialConsistencyDetector

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(r"F:\数据集\Analysis\experiments\pc_mstf_ap_v4\parquet\v4_anomaly_injection_table.parquet")


def best_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(score)
    y, score = y[finite].astype(int), score[finite]
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 193)))
    values = [(f1_score(y, score >= t, zero_division=0), float(t)) for t in candidates]
    best = max(values, key=lambda x: x[0])
    return best[1], best[0]


def split_scores(detector, df: pd.DataFrame) -> np.ndarray:
    score = np.full(len(df), np.nan, dtype=float)
    for split in ["train", "valid", "test"]:
        idx = np.flatnonzero(df["split"].eq(split).to_numpy())
        score[idx] = detector.score(df.iloc[idx])
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description="TrustMeteo-QC 阶段4：按攻击类型评估轻量检测器")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    df = pd.read_parquet(args.data).sort_values(["split", "timestamp_utc", "station_key"]).reset_index(drop=True)
    train = df["split"].eq("train")
    valid = df["split"].eq("valid")
    test = df["split"].eq("test")
    y = df["is_anomaly"].astype(int).to_numpy()
    train_normal = train & ~df["is_anomaly"].astype(bool)
    detectors = [PhysicalRulesDetector(), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    rows = []
    for detector in detectors:
        detector.fit(df.loc[train], train_normal.loc[train])
        score = split_scores(detector, df)
        threshold, valid_f1 = best_threshold(y[valid.to_numpy()], score[valid.to_numpy()])
        test_idx = test.to_numpy()
        pred = score >= threshold
        for anomaly_type, group in df.loc[test].groupby("anomaly_type", sort=True):
            idx = group.index.to_numpy()
            yt = y[idx]
            st = score[idx]
            pt = pred[idx]
            rows.append({
                "detector": detector.name,
                "anomaly_type": str(anomaly_type),
                "rows": int(len(idx)),
                "is_anomaly": bool(group["is_anomaly"].iloc[0]),
                "valid_F1": float(valid_f1),
                "threshold": float(threshold),
                "precision": float(precision_score(yt, pt, zero_division=0)) if yt.sum() else float("nan"),
                "recall": float(recall_score(yt, pt, zero_division=0)) if yt.sum() else float("nan"),
                "F1": float(f1_score(yt, pt, zero_division=0)) if yt.sum() else float("nan"),
                "mean_score": float(np.nanmean(st)) if np.isfinite(st).any() else float("nan"),
                "detected_rows": int(pt.sum()),
            })
        overall = {"detector": detector.name, "anomaly_type": "ALL", "rows": int(test.sum()), "is_anomaly": True, "valid_F1": float(valid_f1), "threshold": float(threshold), "precision": float(precision_score(y[test_idx], pred[test_idx], zero_division=0)), "recall": float(recall_score(y[test_idx], pred[test_idx], zero_division=0)), "F1": float(f1_score(y[test_idx], pred[test_idx], zero_division=0)), "mean_score": float(np.nanmean(score[test_idx])), "detected_rows": int(pred[test_idx].sum())}
        rows.append(overall)
    out = pd.DataFrame(rows)
    out_csv = ROOT / "tables" / "stage04_attack_type_benchmark.csv"
    out_json = ROOT / "tables" / "stage04_attack_type_benchmark.json"
    out_md = ROOT / "reports" / "stage04_attack_type_benchmark.md"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    payload = {"status": "PASS", "data": str(args.data), "rows": int(len(df)), "protocol": "train正常样本拟合，split内计算上下文，valid选阈值，test只评价", "results": rows}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段4攻击类型基准评估", "", "- 状态：**PASS**", "- 说明：所有检测器均在 split 内计算上下文，避免跨 train/valid/test 边界泄漏。", "", "| 检测器 | 攻击类型 | Recall | F1 | 检出行数 |", "|---|---|---:|---:|---:|"]
    for _, r in out.iterrows():
        lines.append(f"| {r['detector']} | {r['anomaly_type']} | {r['recall']:.4f} | {r['F1']:.4f} | {int(r['detected_rows'])} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_csv": str(out_csv), "output_json": str(out_json), "output_report": str(out_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
