from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from stage09_trust_pipeline import _best_threshold, _event_metrics, _metrics, _recovery_metrics  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402
from secure_recovery import add_recovery_candidates, choose_weights, combine_candidates  # noqa: E402
from trust_fusion import robust_location_scale  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段12：鲁棒融合与Isolation Forest混合验证")
    parser.add_argument("--local", type=Path, default=ROOT / "tables" / "stage09b_local_robust_scores.parquet")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    path = args.local
    df = pd.read_parquet(path)
    cols = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency", "ERA5 Background", "structural_risk"]
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    train = df["split"].eq("train").to_numpy(); valid = df["split"].eq("valid").to_numpy(); test = df["split"].eq("test").to_numpy(); normal_train = train & ~df["combined_is_attack"].to_numpy(dtype=bool)
    iso = IsolationForest(n_estimators=300, random_state=20260911, n_jobs=1, contamination="auto").fit(X[normal_train])
    if_raw = -iso.score_samples(X)
    med, scale = robust_location_scale(if_raw[normal_train], floor=0.25)
    if_z = np.maximum(if_raw - med, 0.0) / scale
    base = pd.to_numeric(df["robust_fused_risk"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    yv = df.loc[valid, "combined_is_attack"].to_numpy(dtype=bool)
    best = None
    for beta in [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0]:
        raw = beta * base + (1.0 - beta) * if_z
        threshold, f1 = _best_threshold(yv, raw[valid])
        pred = raw[valid] >= threshold
        far = float(pred[~yv].mean()) if (~yv).any() else 1.0
        candidate = (f1, -far, -threshold)
        if best is None or candidate > best[0]:
            best = (candidate, beta, threshold, f1, far)
    _, beta, threshold, valid_f1, valid_far = best
    hybrid = beta * base + (1.0 - beta) * if_z
    pred = hybrid >= threshold
    valid_pred = pred[valid]; test_pred = pred[test]
    valid_metrics = _metrics(df.loc[valid, "combined_is_attack"], valid_pred, hybrid[valid])
    test_metrics = _metrics(df.loc[test, "combined_is_attack"], test_pred, hybrid[test])
    test_y = df.loc[test, "combined_is_attack"].to_numpy(dtype=bool)
    test_far = float(test_pred[~test_y].mean())
    event = _event_metrics(df.loc[test], test_pred)
    valid_input = df.loc[valid].copy(); valid_input["observed_value"] = valid_input["effective_observed_value"]; valid_input["is_anomaly"] = valid_input["combined_is_attack"]
    test_input = df.loc[test].copy(); test_input["observed_value"] = test_input["effective_observed_value"]; test_input["is_anomaly"] = test_input["combined_is_attack"]
    weights = choose_weights(add_recovery_candidates(valid_input, pd.Series(valid_pred, index=valid_input.index), valid_input["era5_corrected_t2m_c"]))
    rec = combine_candidates(add_recovery_candidates(test_input, pd.Series(test_pred, index=test_input.index), test_input["era5_corrected_t2m_c"]), weights)
    recovery = _recovery_metrics(df.loc[test].assign(recovered_value=rec), rec)
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "Isolation Forest仅在train正常样本拟合；valid选择有限混合权重和阈值；test只评价", "beta_stage09b": beta, "beta_isolation_forest": 1.0 - beta, "threshold": threshold, "valid": valid_metrics, "test": test_metrics, "test_false_alarm_rate": test_far, "event_metrics": event, "recovery": recovery, "recovery_weights": weights, "interpretation": "混合结果必须与独立Isolation Forest和Stage09B同时报告；不因混合结果不利而删除基线。"}
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    suffix = f"_{args.tag}" if args.tag else ""
    (out_dir / f"stage12_hybrid_fusion{suffix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段12混合证据验证", "", f"- Stage09B / Isolation Forest 权重：`{beta:.2f}` / `{1.0-beta:.2f}`", f"- valid F1/误伤率：`{valid_metrics['f1']:.4f}` / `{valid_far:.4f}`", f"- test F1/Recall/Precision/AUPRC：`{test_metrics['f1']:.4f}` / `{test_metrics['recall']:.4f}` / `{test_metrics['precision']:.4f}` / `{test_metrics['average_precision']:.4f}`", f"- test正常样本误伤率：`{test_far:.4f}`", f"- test事件级召回：`{event['event_recall']:.4f}`", f"- 恢复后 MAE/RMSE：`{recovery['mae_after']:.4f}` / `{recovery['rmse_after']:.4f}`", "", "该结果用于判断独立无监督基线是否提供互补证据，不得只展示混合模型而省略单模型结果。"]
    report = report_dir / f"stage12_hybrid_fusion{suffix}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report), "test": test_metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
