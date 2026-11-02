from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from stage09_trust_pipeline import _event_metrics, _metrics, _recovery_metrics  # noqa: E402
from secure_recovery import add_recovery_candidates, combine_candidates, choose_weights  # noqa: E402


def station_thresholds(train: pd.DataFrame, q: float) -> tuple[pd.Series, float]:
    clean = train[~train["combined_is_attack"] & np.isfinite(train["robust_fused_risk"])]
    return clean.groupby("station_key")["robust_fused_risk"].quantile(q), float(clean["robust_fused_risk"].quantile(q))


def apply_threshold(frame: pd.DataFrame, thresholds: pd.Series, global_threshold: float) -> np.ndarray:
    threshold = frame["station_key"].map(thresholds).fillna(global_threshold).to_numpy(dtype=float)
    return frame["robust_fused_risk"].to_numpy(dtype=float) >= threshold


def bootstrap_ci(frame: pd.DataFrame, pred: np.ndarray, n: int = 500, seed: int = 20260911) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    y = frame["combined_is_attack"].to_numpy(dtype=bool)
    score = frame["robust_fused_risk"].to_numpy(dtype=float)
    values = {k: [] for k in ["f1", "recall", "precision", "false_alarm_rate"]}
    for _ in range(n):
        idx = rng.integers(0, len(frame), len(frame))
        m = _metrics(y[idx], pred[idx], score[idx])
        clean = ~y[idx]
        values["f1"].append(m["f1"]); values["recall"].append(m["recall"]); values["precision"].append(m["precision"])
        values["false_alarm_rate"].append(float(pred[idx][clean].mean()) if clean.any() else 0.0)
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in values.items()}


def main() -> int:
    local_path = ROOT / "tables" / "stage09b_local_robust_scores.parquet"
    test_path = ROOT / "tables" / "stage09b_test_recovery.parquet"
    df = pd.read_parquet(local_path)
    train, valid, test = [df[df["split"].eq(s)].copy() for s in ["train", "valid", "test"]]
    candidates = []
    for q in [0.95, 0.97, 0.98, 0.985, 0.99, 0.995, 0.997, 0.999]:
        thresholds, global_threshold = station_thresholds(train, q)
        pred = apply_threshold(valid, thresholds, global_threshold)
        y = valid["combined_is_attack"].to_numpy(dtype=bool)
        far = float(pred[~y].mean()) if (~y).any() else 1.0
        recall = float(pred[y].mean()) if y.any() else 0.0
        f1 = _metrics(y, pred)["f1"]
        candidates.append({"q": q, "thresholds": thresholds, "global_threshold": global_threshold, "valid_far": far, "valid_recall": recall, "valid_f1": f1})
    feasible = [x for x in candidates if x["valid_far"] <= 0.01]
    selected = max(feasible or candidates, key=lambda x: (x["valid_recall"], x["valid_f1"], -x["valid_far"]))
    test_pred = apply_threshold(test, selected["thresholds"], selected["global_threshold"])
    valid_pred = apply_threshold(valid, selected["thresholds"], selected["global_threshold"])
    valid_metrics = _metrics(valid["combined_is_attack"], valid_pred, valid["robust_fused_risk"])
    test_metrics = _metrics(test["combined_is_attack"], test_pred, test["robust_fused_risk"])
    test_clean = ~test["combined_is_attack"].to_numpy(dtype=bool)
    far_test = float(test_pred[test_clean].mean()) if test_clean.any() else float("nan")
    event = _event_metrics(test, test_pred)
    recovery = pd.read_parquet(test_path)
    recovery_input = test.copy()
    recovery_input["observed_value"] = recovery_input["effective_observed_value"]
    recovery_input["is_anomaly"] = recovery_input["combined_is_attack"]
    valid_input = valid.copy(); valid_input["observed_value"] = valid_input["effective_observed_value"]; valid_input["is_anomaly"] = valid_input["combined_is_attack"]
    weights = choose_weights(add_recovery_candidates(valid_input, pd.Series(valid_pred, index=valid.index), valid_input["era5_corrected_t2m_c"]))
    rec_candidates = add_recovery_candidates(recovery_input, pd.Series(test_pred, index=test.index), recovery_input["era5_corrected_t2m_c"])
    recovered = combine_candidates(rec_candidates, weights)
    recovery_metrics = _recovery_metrics(test.assign(recovered_value=recovered), recovered)
    recovery_detail = test.copy()
    recovery_detail["recovered_value"] = recovered
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "operating_point": "站点自适应分位阈值，valid正常误伤率不超过1%，在可行点中最大化Recall",
        "selected_quantile": selected["q"],
        "selected_global_threshold": selected["global_threshold"],
        "valid": valid_metrics,
        "test": test_metrics,
        "test_false_alarm_rate": far_test,
        "test_event_metrics": event,
        "test_recovery": recovery_metrics,
        "recovery_weights": weights,
        "test_bootstrap_95ci": bootstrap_ci(test, test_pred),
        "candidate_points": [{k: v for k, v in x.items() if k not in {"thresholds"}} for x in candidates],
        "interpretation": "该运行点用于安全型低误报部署；F1最优点与低误报点应同时报告，不得只选择有利运行点。",
    }
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    recovery_detail.to_parquet(out_dir / "stage09d_test_recovery.parquet", index=False)
    (out_dir / "stage09d_operating_point.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(payload["candidate_points"]).to_csv(out_dir / "stage09d_operating_points.csv", index=False, encoding="utf-8-sig")
    lines = ["# 阶段9D低误报运行点", "", "- 仅用训练集正常样本估计站点分位阈值。", "- valid 选择误伤率不超过1%的可行运行点；test 只评价。", "", f"- 选定分位数：`{selected['q']}`；valid误伤率：`{selected['valid_far']:.4f}`；valid Recall：`{selected['valid_recall']:.4f}`", f"- test F1/Recall/Precision/AUPRC：`{test_metrics['f1']:.4f}` / `{test_metrics['recall']:.4f}` / `{test_metrics['precision']:.4f}` / `{test_metrics['average_precision']:.4f}`", f"- test正常样本误伤率：`{far_test:.4f}`", f"- test事件级召回：`{event['event_recall']:.4f}`", f"- 恢复后 MAE/RMSE：`{recovery_metrics['mae_after']:.4f}` / `{recovery_metrics['rmse_after']:.4f}`", f"- test bootstrap 95% CI：`{json.dumps(payload['test_bootstrap_95ci'], ensure_ascii=False)}`", "", "论文中应同时呈现 F1 优化点和低误报安全运行点，避免将运行点选择误写成测试集调参。"]
    report = report_dir / "stage09d_operating_point.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report), "test": test_metrics, "test_false_alarm_rate": far_test}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
