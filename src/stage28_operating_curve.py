from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage09_trust_pipeline import _best_threshold, _metrics

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段28：阈值运行曲线")
    parser.add_argument("--source", type=Path, default=TABLES / "stage09b_local_robust_scores.parquet")
    args = parser.parse_args()
    df = pd.read_parquet(args.source)
    valid = df["split"].eq("valid")
    test = df["split"].eq("test")
    yv = df.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    sv = df.loc[valid, "robust_fused_risk"].to_numpy(dtype=float)
    yt = df.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    st = df.loc[test, "robust_fused_risk"].to_numpy(dtype=float)
    normal_valid = sv[~yv]
    targets = [0.005, 0.01, 0.02, 0.05, 0.10]
    rows = []
    for target in targets:
        threshold = float(np.quantile(normal_valid, 1.0 - target))
        pred_v = sv >= threshold
        pred_t = st >= threshold
        mv = _metrics(yv, pred_v, sv)
        mt = _metrics(yt, pred_t, st)
        rows.append({"operating_point": f"valid_FAR_{target:.1%}", "target_valid_false_alarm_rate": target, "threshold": threshold, "valid_false_alarm_rate": float(np.mean(pred_v[~yv])), "valid_f1": mv["f1"], "test_false_alarm_rate": float(np.mean(pred_t[~yt])), "test_precision": mt["precision"], "test_recall": mt["recall"], "test_f1": mt["f1"], "test_auprc": mt["average_precision"]})
    f1_threshold, valid_f1 = _best_threshold(yv, sv)
    pred_t = st >= f1_threshold
    mt = _metrics(yt, pred_t, st)
    rows.append({"operating_point": "valid_F1_optimal", "target_valid_false_alarm_rate": float(np.mean(sv[~yv] >= f1_threshold)), "threshold": float(f1_threshold), "valid_false_alarm_rate": float(np.mean(sv[~yv] >= f1_threshold)), "valid_f1": valid_f1, "test_false_alarm_rate": float(np.mean(pred_t[~yt])), "test_precision": mt["precision"], "test_recall": mt["recall"], "test_f1": mt["f1"], "test_auprc": mt["average_precision"]})
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "stage28_operating_curve.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "阈值只由valid正常样本分位数或valid F1选择；所有运行点在锁定test上一次评价，不根据test反选。", "results": rows, "source": str(args.source)}
    (TABLES / "stage28_operating_curve.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段28阈值运行曲线", "", "阈值由 valid 正常样本分位数或 valid F1 选择，test 只评价。", "", "| 运行点 | 阈值 | valid误伤率 | test误伤率 | test Precision | test Recall | test F1 | test AUPRC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in result.iterrows():
        lines.append(f"| {r['operating_point']} | {r['threshold']:.4f} | {r['valid_false_alarm_rate']:.4f} | {r['test_false_alarm_rate']:.4f} | {r['test_precision']:.4f} | {r['test_recall']:.4f} | {r['test_f1']:.4f} | {r['test_auprc']:.4f} |")
    lines += ["", "该表用于展示安全运行点与召回/F1之间的权衡，不允许只报告其中一个运行点。"]
    (REPORTS / "stage28_operating_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "points": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
