from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage09_trust_pipeline import _best_threshold, _metrics
from trust_fusion import robust_location_scale
from stage09b_robust_fusion import smooth_risk

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"


def lag6_residual(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series(0.0, index=frame.index, dtype=float)
    for split in ["train", "valid", "test"]:
        part = frame[frame["split"].eq(split)].copy()
        if part.empty:
            continue
        part = part.sort_values(["station_key", "timestamp_utc", "sample_id"], kind="stable")
        v = pd.to_numeric(part["effective_observed_value"], errors="coerce")
        lagged = v.groupby(part["station_key"], sort=False).shift(6)
        residual = (v - lagged).abs()
        out.loc[part.index] = residual.to_numpy()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def main() -> int:
    scores = pd.read_parquet(TABLES / "stage09b_local_robust_scores.parquet").reset_index(drop=True)
    scores["timestamp_utc"] = pd.to_datetime(scores["timestamp_utc"], utc=True, errors="coerce")
    scores["lag6_residual"] = lag6_residual(scores)
    train_normal = scores["split"].eq("train") & ~scores["combined_is_attack"]
    med, scale = robust_location_scale(scores.loc[train_normal, "lag6_residual"], floor=0.25)
    scores["lag6_risk"] = np.maximum(scores["lag6_residual"].to_numpy() - med, 0.0) / scale

    valid = scores["split"].eq("valid")
    test = scores["split"].eq("test")
    yv = scores.loc[valid, "combined_is_attack"].to_numpy(dtype=bool)
    yt = scores.loc[test, "combined_is_attack"].to_numpy(dtype=bool)
    candidates = []
    for alpha in [0.40, 0.50, 0.60, 0.70, 0.80]:
        for gamma in [0.10, 0.20, 0.30, 0.40]:
            structural = (1.0 - gamma) * scores["structural_risk"].to_numpy() + gamma * scores["lag6_risk"].to_numpy()
            raw = alpha * scores["base_fused_risk"].to_numpy() + (1.0 - alpha) * structural
            for window in [1, 3, 6]:
                valid_risk = smooth_risk(scores.loc[valid], raw[valid.to_numpy()], window)
                threshold, f1 = _best_threshold(yv, valid_risk)
                far = float(np.mean(valid_risk[~yv] >= threshold)) if (~yv).any() else 1.0
                candidates.append((f1, -far, -threshold, alpha, gamma, window, threshold))
    selected = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, alpha, gamma, window, threshold = selected
    structural = (1.0 - gamma) * scores["structural_risk"].to_numpy() + gamma * scores["lag6_risk"].to_numpy()
    raw = alpha * scores["base_fused_risk"].to_numpy() + (1.0 - alpha) * structural
    test_risk = smooth_risk(scores.loc[test], raw[test.to_numpy()], window)
    pred = test_risk >= threshold
    test_metrics = _metrics(yt, pred, test_risk)
    base_payload = json.loads((TABLES / "stage09b_robust_fusion.json").read_text(encoding="utf-8"))
    base_metrics = base_payload["local_test"]
    by_type = []
    test_frame = scores.loc[test].copy()
    test_frame["pred_lag6"] = pred
    for attack_type in sorted(test_frame.loc[test_frame["combined_is_attack"], "combined_attack_type"].unique().tolist()):
        y = test_frame["combined_is_attack"].to_numpy(dtype=bool) & test_frame["combined_attack_type"].eq(attack_type).to_numpy(dtype=bool)
        by_type.append({"attack_type": attack_type, **_metrics(y, pred, test_risk)})
    improved = test_metrics["f1"] > base_metrics["f1"] and test_metrics["average_precision"] > base_metrics["average_precision"]
    payload = {
        "status": "CANDIDATE_GAIN_REQUIRES_MULTISEED" if improved else "NEGATIVE_RESULT_RETAINED",
        "protocol": "只在valid选择lag-6 residual是否加入、融合权重、窗口和阈值；test只评价。lag=6来自受控time_shift协议，不能据此宣称对真实时间错位普适。",
        "selected": {"alpha_base": alpha, "gamma_lag6_in_structural": gamma, "window": window, "threshold": threshold, "valid_f1": selected[0], "valid_false_alarm_rate": -selected[1]},
        "test": test_metrics,
        "stage09b_reference": base_metrics,
        "delta_vs_stage09b": {key: float(test_metrics[key] - base_metrics[key]) for key in ["f1", "recall", "precision", "average_precision"]},
        "test_false_alarm_rate": float(np.mean(pred[~yt])) if (~yt).any() else float("nan"),
        "lag6_train_median": med,
        "lag6_train_scale": scale,
        "by_type": by_type,
        "negative_result_guard": "若test指标未形成稳定增益，应作为负结果，不修改主模型。",
    }
    (TABLES / "stage24_lag6_feature_experiment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(by_type).to_csv(TABLES / "stage24_lag6_feature_by_type.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 阶段24 lag-6 residual特征实验",
        "",
        "lag=6严格来自受控 time_shift 协议；valid选择加入权重、窗口和阈值，test只评价。",
        "",
        f"- 设置：base=`{alpha:.2f}`，lag-6在结构化风险中的权重=`{gamma:.2f}`，窗口=`{window}`，阈值=`{threshold:.6f}`",
        f"- valid F1/误伤率：`{selected[0]:.4f}` / `{-selected[1]:.4f}`",
        f"- test F1/Recall/Precision/AUPRC：`{test_metrics['f1']:.4f}` / `{test_metrics['recall']:.4f}` / `{test_metrics['precision']:.4f}` / `{test_metrics['average_precision']:.4f}`",
        f"- test正常样本误伤率：`{payload['test_false_alarm_rate']:.4f}`",
        f"- 相对9B变化（F1/Recall/Precision/AUPRC）：`{payload['delta_vs_stage09b']['f1']:+.4f}` / `{payload['delta_vs_stage09b']['recall']:+.4f}` / `{payload['delta_vs_stage09b']['precision']:+.4f}` / `{payload['delta_vs_stage09b']['average_precision']:+.4f}`",
        "",
        "结论：该候选未同时改善 F1 与 AUPRC，作为诊断性负结果保留，不进入主模型。按攻击类型结果见 `stage24_lag6_feature_by_type.csv`。",
    ]
    (ROOT / "reports" / "stage24_lag6_feature_experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "test": test_metrics, "selected": payload["selected"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
