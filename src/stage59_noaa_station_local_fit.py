from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors import CUSUMDetector, HampelMADDetector, KalmanInnovationDetector, PhysicalRulesDetector, SpatialConsistencyDetector  # noqa: E402
from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import build_dataset  # noqa: E402
from trust_fusion import TrustFusion  # noqa: E402


def score_local(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    out = frame.copy()
    out["observed_value"] = pd.to_numeric(out["observed_value"], errors="coerce")
    train = out["split"].eq("train")
    valid = out["split"].eq("valid")
    test = out["split"].eq("test")
    normal_train = train & (~out["combined_is_attack"].astype(bool))
    detector_types = [PhysicalRulesDetector, HampelMADDetector, KalmanInnovationDetector, CUSUMDetector]
    for detector_type in detector_types:
        values = np.full(len(out), np.nan, dtype=float)
        for station, group in out.groupby(out["station_key"].astype(str), sort=True):
            station_train = group.loc[group["split"].eq("train")]
            if station_train.empty:
                continue
            detector = detector_type()
            detector.fit(station_train, pd.Series(True, index=station_train.index))
            for split in ["train", "valid", "test"]:
                idx = group.index[group["split"].eq(split)]
                if len(idx):
                    values[idx.to_numpy()] = detector.score(out.loc[idx])
        out[detector_type().name] = values

    # Spatial evidence intentionally keeps a pooled fit: it is a cross-station
    # signal and should use simultaneous observations from the training set.
    spatial = SpatialConsistencyDetector()
    spatial.fit(out.loc[train], normal_train.loc[train])
    out[spatial.name] = spatial.score(out)
    out["ERA5 Background"] = 0.0
    fusion = TrustFusion(include_era5=False).fit(out.loc[train], out.loc[valid], out.loc[valid, "combined_is_attack"])
    risk = fusion.transform(out)
    threshold_candidates = np.unique(np.quantile(risk[valid.to_numpy()], np.linspace(0.02, 0.98, 193)))
    yv = out.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    best = max(((float(_metrics(yv, risk[valid.to_numpy()] >= t, risk[valid.to_numpy()])["f1"]), float(t)) for t in threshold_candidates), key=lambda x: x[0])
    threshold = best[1]
    pred = risk >= threshold
    yt = out.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    mt = _metrics(yt, pred[test.to_numpy()], risk[test.to_numpy()])
    return out, {"threshold": threshold, "valid_f1": best[0], "valid_false_alarm_rate": float(pred[valid.to_numpy()][~yv].mean()), "test": mt, "test_false_alarm_rate": float(pred[test.to_numpy()][~yt].mean()), "fusion_weights": fusion.weights}


def main() -> int:
    source = pd.read_parquet(NOAA)
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        frame = build_dataset(source, seed)
        _, result = score_local(frame)
        rows.append({"seed": seed, "valid_f1": result["valid_f1"], "valid_far": result["valid_false_alarm_rate"], "test_f1": result["test"]["f1"], "test_recall": result["test"]["recall"], "test_precision": result["test"]["precision"], "test_auprc": result["test"]["average_precision"], "test_far": result["test_false_alarm_rate"], "threshold": result["threshold"], "fusion_weights": result["fusion_weights"]})
    detail = pd.DataFrame(rows)
    aggregate = detail[["valid_f1", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).to_dict()
    payload = {"status": "NEGATIVE_RESULT_RETAINED", "protocol": "NOAA 12站点、5种子；物理/Hampel/Kalman/CUSUM分别只用本站train正常数据拟合，Spatial Consistency使用 pooled train正常数据；valid统一选阈值；test只评价。", "reason": "统一阈值下站点本地拟合产生近乎全量报警，未形成可用外部增益。", "seeds": SEEDS, "detail": rows, "aggregate": aggregate}
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage59_noaa_station_local_fit_detail.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage59_noaa_station_local_fit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True, default=str), encoding="utf-8")
    lines = ["# 阶段59 NOAA站点本地拟合候选", "", "四个时间检测器只用本站train正常数据拟合，空间检测器使用所有站点train正常数据；valid统一选阈值，test只评价。", "", f"- test F1：`{aggregate['test_f1']['mean']:.4f}±{aggregate['test_f1']['std']:.4f}`；Recall：`{aggregate['test_recall']['mean']:.4f}±{aggregate['test_recall']['std']:.4f}`；AUPRC：`{aggregate['test_auprc']['mean']:.4f}±{aggregate['test_auprc']['std']:.4f}`；FAR：`{aggregate['test_far']['mean']:.4f}±{aggregate['test_far']['std']:.4f}`", "", "| seed | valid F1 | valid FAR | test F1 | Recall | Precision | AUPRC | FAR |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['valid_f1']:.4f} | {row['valid_far']:.4f} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | {row['test_precision']:.4f} | {row['test_auprc']:.4f} | {row['test_far']:.4f} |")
    lines += ["", "该候选若有增益，仍需在独立年份和严格留站点协议中复核；不替换9B主模型。"]
    (REPORTS / "stage59_noaa_station_local_fit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "aggregate": aggregate}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
