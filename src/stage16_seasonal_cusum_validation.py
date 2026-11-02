from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from detectors import SeasonalResidualCUSUMDetector  # noqa: E402
from stage09_trust_pipeline import _best_threshold, _event_metrics, _metrics, _prepare_local, _recovery_metrics  # noqa: E402
from secure_recovery import add_recovery_candidates, choose_weights, combine_candidates  # noqa: E402
from trust_fusion import robust_location_scale  # noqa: E402


def main() -> int:
    score_path = ROOT / "tables" / "stage09b_local_robust_scores.parquet"
    scores = pd.read_parquet(score_path)
    local = _prepare_local(ROOT / "tables" / "stage05_controlled_attack_extensions.parquet")
    if len(local) != len(scores):
        raise ValueError("stage09b scores and stage05 input row counts differ")
    frame = local.copy()
    frame["observed_value"] = frame["effective_observed_value"]
    train = frame["split"].eq("train")
    normal_train = train & ~frame["combined_is_attack"]
    detector = SeasonalResidualCUSUMDetector().fit(frame.loc[train], normal_train.loc[train])
    new_score = np.full(len(frame), np.nan)
    for split in ["train", "valid", "test"]:
        mask = frame["split"].eq(split).to_numpy()
        new_score[mask] = detector.score(frame.loc[mask])
    med, scale = robust_location_scale(new_score[normal_train.to_numpy()], floor=0.25)
    new_z = np.maximum(new_score - med, 0.0) / scale
    base = scores["robust_fused_risk"].to_numpy(dtype=float)
    valid = frame["split"].eq("valid").to_numpy(); test = frame["split"].eq("test").to_numpy()
    yv = frame.loc[valid, "combined_is_attack"].to_numpy(dtype=bool)
    best = None
    for beta in [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0]:
        hybrid = beta * base + (1.0 - beta) * new_z
        threshold, f1 = _best_threshold(yv, hybrid[valid])
        pred = hybrid[valid] >= threshold
        far = float(pred[~yv].mean())
        cand = (f1, -far, -threshold)
        if best is None or cand > best[0]:
            best = (cand, beta, threshold, f1, far)
    _, beta, threshold, valid_f1, valid_far = best
    hybrid = beta * base + (1.0 - beta) * new_z
    pred = hybrid >= threshold
    ytest = frame.loc[test, "combined_is_attack"].to_numpy(dtype=bool)
    test_metrics = _metrics(ytest, pred[test], hybrid[test])
    valid_metrics = _metrics(yv, pred[valid], hybrid[valid])
    event = _event_metrics(frame.loc[test], pred[test])
    far_test = float(pred[test][~ytest].mean())
    valid_input = frame.loc[valid].copy(); valid_input["observed_value"] = valid_input["effective_observed_value"]; valid_input["is_anomaly"] = valid_input["combined_is_attack"]
    test_input = frame.loc[test].copy(); test_input["observed_value"] = test_input["effective_observed_value"]; test_input["is_anomaly"] = test_input["combined_is_attack"]
    weights = choose_weights(add_recovery_candidates(valid_input, pd.Series(pred[valid], index=valid_input.index), valid_input.get("era5_corrected_t2m_c")))
    rec = combine_candidates(add_recovery_candidates(test_input, pd.Series(pred[test], index=test_input.index), test_input.get("era5_corrected_t2m_c")), weights)
    recovery = _recovery_metrics(frame.loc[test].assign(recovered_value=rec), rec)
    type_rows = []
    tframe = frame.loc[test].copy(); tframe["_pred"] = pred[test]
    for attack_type, group in tframe[tframe["combined_is_attack"]].groupby("combined_attack_type", sort=True):
        type_rows.append({"attack_type": attack_type, "rows": len(group), "recall": float(group["_pred"].mean())})
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    pd.DataFrame(type_rows).to_csv(out_dir / "stage16_seasonal_cusum_by_type.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "季节基线残差CUSUM仅用train正常样本拟合；valid选择与9B风险的有限融合权重和阈值；test只评价", "beta_stage09b": beta, "beta_seasonal_cusum": 1.0 - beta, "threshold": threshold, "valid": valid_metrics, "test": test_metrics, "test_false_alarm_rate": far_test, "event_metrics": event, "recovery": recovery, "recovery_weights": weights, "by_type": type_rows, "interpretation": "该检测器用于验证站点日变化校正能否改善结构化攻击；若无稳定增益，保留为负结果。"}
    (out_dir / "stage16_seasonal_cusum_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段16季节残差 CUSUM 验证", "", f"- 9B / 季节残差 CUSUM 权重：`{beta:.2f}` / `{1.0-beta:.2f}`", f"- valid F1/误伤率：`{valid_metrics['f1']:.4f}` / `{valid_far:.4f}`", f"- test F1/Recall/Precision/AUPRC：`{test_metrics['f1']:.4f}` / `{test_metrics['recall']:.4f}` / `{test_metrics['precision']:.4f}` / `{test_metrics['average_precision']:.4f}`", f"- test正常样本误伤率：`{far_test:.4f}`", f"- test事件级召回：`{event['event_recall']:.4f}`", f"- 恢复后 MAE/RMSE：`{recovery['mae_after']:.4f}` / `{recovery['rmse_after']:.4f}`", "", "该结果必须与 9B、Isolation Forest 和监督上界同时报告。"]
    report = report_dir / "stage16_seasonal_cusum_validation.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report), "test": test_metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
