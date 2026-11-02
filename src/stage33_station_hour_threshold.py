from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage09_trust_pipeline import _best_threshold, _metrics

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def threshold_tables(train: pd.DataFrame, q: float) -> tuple[pd.Series, pd.Series, float]:
    normal = train[(~train["combined_is_attack"].astype(bool)) & np.isfinite(train["robust_fused_risk"])]
    station = normal.groupby("station_key")["robust_fused_risk"].quantile(q)
    hour = normal.assign(hour_utc=normal["timestamp_utc"].dt.hour).groupby(["station_key", "hour_utc"])["robust_fused_risk"].quantile(q)
    global_threshold = float(normal["robust_fused_risk"].quantile(q))
    return station, hour, global_threshold


def apply(frame: pd.DataFrame, station: pd.Series, hour: pd.Series, global_threshold: float, lam: float) -> np.ndarray:
    hours = frame["timestamp_utc"].dt.hour
    keys = pd.MultiIndex.from_arrays([frame["station_key"].astype(str), hours], names=["station_key", "hour_utc"])
    hour_t = hour.reindex(keys).to_numpy(dtype=float)
    station_t = frame["station_key"].astype(str).map(station).to_numpy(dtype=float)
    hour_t = np.where(np.isfinite(hour_t), hour_t, station_t)
    hour_t = np.where(np.isfinite(hour_t), hour_t, float(global_threshold))
    station_t = np.where(np.isfinite(station_t), station_t, float(global_threshold))
    threshold = lam * hour_t + (1.0 - lam) * station_t
    return frame["robust_fused_risk"].to_numpy(dtype=float) >= threshold


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段33：站点-小时自适应阈值候选")
    parser.add_argument("--source", type=Path, default=TABLES / "stage09b_local_robust_scores.parquet")
    args = parser.parse_args()
    df = pd.read_parquet(args.source)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    train = df[df["split"].eq("train")].copy()
    valid = df[df["split"].eq("valid")].copy()
    test = df[df["split"].eq("test")].copy()
    yv = valid["combined_is_attack"].astype(bool).to_numpy()
    yt = test["combined_is_attack"].astype(bool).to_numpy()
    candidates = []
    for q in [0.98, 0.985, 0.99, 0.995, 0.997, 0.999]:
        station, hour, global_threshold = threshold_tables(train, q)
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pv = apply(valid, station, hour, global_threshold, lam)
            mv = _metrics(yv, pv, valid["robust_fused_risk"].to_numpy(dtype=float))
            far = float(pv[~yv].mean()) if (~yv).any() else 1.0
            candidates.append({"q": q, "hour_weight": lam, "valid_f1": mv["f1"], "valid_recall": mv["recall"], "valid_precision": mv["precision"], "valid_false_alarm_rate": far, "station_thresholds": station, "hour_thresholds": hour, "global_threshold": global_threshold})
    selected = max(candidates, key=lambda x: (x["valid_f1"], -x["valid_false_alarm_rate"], x["valid_recall"]))
    pt = apply(test, selected["station_thresholds"], selected["hour_thresholds"], selected["global_threshold"], selected["hour_weight"])
    mt = _metrics(yt, pt, test["robust_fused_risk"].to_numpy(dtype=float))
    far_test = float(pt[~yt].mean()) if (~yt).any() else float("nan")
    baseline = json.loads((TABLES / "stage09b_robust_fusion.json").read_text(encoding="utf-8"))["local_test"]
    delta = {k: float(mt[k] - baseline[k]) for k in ["f1", "recall", "precision", "average_precision"]}
    result = pd.DataFrame([{k: v for k, v in x.items() if k not in {"station_thresholds", "hour_thresholds"}} for x in candidates])
    result.to_csv(TABLES / "stage33_station_hour_threshold_candidates.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_MULTISEED" if delta["f1"] > 0 and delta["average_precision"] >= 0 else "NEGATIVE_RESULT_RETAINED", "protocol": "阈值只由train正常样本按站点×UTC小时分位数估计；valid选择分位数与小时阈值收缩比例；test只评价。", "selected": {k: v for k, v in selected.items() if k not in {"station_thresholds", "hour_thresholds"}}, "test": mt, "test_false_alarm_rate": far_test, "stage09b_reference": baseline, "delta_vs_stage09b": delta, "candidate_count": len(candidates)}
    (TABLES / "stage33_station_hour_threshold.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段33站点-小时自适应阈值候选", "", "阈值由 train 正常样本按站点×UTC小时分位数估计，valid 选择分位数与收缩比例，test 只评价。", "", f"- 选定 q/hour_weight：`{selected['q']}` / `{selected['hour_weight']:.2f}`", f"- valid F1/Recall/误伤率：`{selected['valid_f1']:.4f}` / `{selected['valid_recall']:.4f}` / `{selected['valid_false_alarm_rate']:.4f}`", f"- test F1/Recall/Precision/AUPRC：`{mt['f1']:.4f}` / `{mt['recall']:.4f}` / `{mt['precision']:.4f}` / `{mt['average_precision']:.4f}`", f"- test正常样本误伤率：`{far_test:.4f}`", f"- 相对9B变化（F1/Recall/Precision/AUPRC）：`{delta['f1']:+.4f}` / `{delta['recall']:+.4f}` / `{delta['precision']:+.4f}` / `{delta['average_precision']:+.4f}`", "", "该候选必须在多种子和滚动时间窗口验证后才能考虑进入主流程；当前不修改9B主模型。"]
    (REPORTS / "stage33_station_hour_threshold.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "test": mt, "test_false_alarm_rate": far_test, "delta_vs_stage09b": delta}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
