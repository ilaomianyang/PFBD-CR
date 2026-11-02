from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors import CUSUMDetector, HampelMADDetector, KalmanInnovationDetector, PhysicalRulesDetector, SpatialConsistencyDetector  # noqa: E402
from stage09_trust_pipeline import _metrics  # noqa: E402
from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, best_robust_setting, smooth_risk, structural_features, structural_risk_from_norm  # noqa: E402
from stage38_noaa_controlled_benchmark import build_dataset  # noqa: E402
from trust_fusion import TrustFusion  # noqa: E402


def score_by_split(detector, frame: pd.DataFrame, fit_mask: pd.Series) -> np.ndarray:
    out = np.full(len(frame), np.nan, dtype=float)
    for split in ["train", "valid", "test"]:
        mask = frame["split"].eq(split)
        if split != "test":
            mask &= fit_mask
        if mask.any():
            out[mask.to_numpy()] = detector.score(frame.loc[mask])
    return out


def structural_by_split(frame: pd.DataFrame, fit_mask: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index, columns=STRUCTURAL_BASE, dtype=float)
    for split in ["train", "valid", "test"]:
        mask = frame["split"].eq(split)
        if split != "test":
            mask &= fit_mask
        if mask.any():
            out.loc[mask, STRUCTURAL_BASE] = structural_features(frame.loc[mask], STRUCTURAL_BASE).to_numpy()
    return out.fillna(0.0)


def evaluate(source: pd.DataFrame, seed: int, held_out: str) -> dict[str, object]:
    frame = build_dataset(source, seed)
    frame["observed_value"] = frame["observed_value"].astype(float)
    fit_station = frame["station_key"].astype(str).ne(str(held_out))
    train = frame["split"].eq("train") & fit_station
    valid = frame["split"].eq("valid") & fit_station
    test = frame["split"].eq("test")
    normal_train = train & (~frame["combined_is_attack"].astype(bool))
    detectors = [PhysicalRulesDetector(), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    for detector in detectors:
        detector.fit(frame.loc[train], normal_train.loc[train])
        frame[detector.name] = score_by_split(detector, frame, fit_station)
    frame["ERA5 Background"] = 0.0
    features = structural_by_split(frame, fit_station)
    features["station_key"] = frame["station_key"].astype(str)
    scaler = AdaptiveScaler(STRUCTURAL_BASE).fit(features.loc[train], normal_train.loc[train])
    structural = structural_risk_from_norm(scaler.transform(features), enhanced=False)
    fusion = TrustFusion(include_era5=False).fit(frame.loc[train], frame.loc[valid], frame.loc[valid, "combined_is_attack"])
    base = fusion.transform(frame)
    setting = best_robust_setting(frame.loc[valid], base[valid.to_numpy()], structural[valid.to_numpy()])
    raw = setting["alpha_base"] * base + setting["alpha_structural"] * structural
    risk = np.full(len(frame), np.nan, dtype=float)
    pred = np.zeros(len(frame), dtype=bool)
    for split, mask in [("train", train), ("valid", valid), ("test", test)]:
        if mask.any():
            values = smooth_risk(frame.loc[mask], raw[mask.to_numpy()], int(setting["window"]))
            risk[mask.to_numpy()] = values
            pred[mask.to_numpy()] = values >= float(setting["threshold"])
    held = test & frame["station_key"].astype(str).eq(str(held_out))
    y = frame.loc[held, "combined_is_attack"].astype(bool).to_numpy()
    p = pred[held.to_numpy()]
    s = risk[held.to_numpy()]
    m = _metrics(y, p, s)
    normal = ~y
    return {
        "seed": seed,
        "held_out_station": str(held_out),
        "fit_station_count": int(fit_station.sum() and frame.loc[fit_station, "station_key"].nunique()),
        "held_out_rows": int(held.sum()),
        "test_f1": m["f1"],
        "test_recall": m["recall"],
        "test_precision": m["precision"],
        "test_auprc": m["average_precision"],
        "test_far": float(p[normal].mean()) if normal.any() else float("nan"),
        "threshold": float(setting["threshold"]),
        "valid_f1": float(setting["valid_f1"]),
    }


def main() -> int:
    source = pd.read_parquet(NOAA)
    stations = sorted(source["station_id"].astype(str).unique().tolist())
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for station in stations:
            rows.append(evaluate(source, seed, station))
            print(json.dumps({"seed": seed, "completed_station": station, "rows": len(rows)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("held_out_station", sort=True)[["test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["held_out_station"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    summary = {metric: {"mean": float(detail[metric].mean()), "std": float(detail[metric].std(ddof=1)), "min": float(detail[metric].min()), "max": float(detail[metric].max())} for metric in ["test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]}
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA 12站点；每次留出一个站点，留出站点不参与train/valid拟合和阈值选择；其它站点选择融合和阈值；test仅评价留出站点；3种子。", "seeds": SEEDS, "stations": stations, "summary": summary, "detail": rows, "aggregate_by_station": aggregate.to_dict(orient="records")}
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage57_noaa_leave_one_station_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage57_noaa_leave_one_station.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage57_noaa_leave_one_station.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段57 NOAA严格留一站点验证", "", "留出站点不参与train/valid拟合和阈值选择；仅使用其它站点选择设置；test只评价留出站点。", "", f"- 站点数：`{len(stations)}`；种子数：`{len(SEEDS)}`", f"- 总体 F1：`{summary['test_f1']['mean']:.4f}±{summary['test_f1']['std']:.4f}`；AUPRC：`{summary['test_auprc']['mean']:.4f}±{summary['test_auprc']['std']:.4f}`；FAR：`{summary['test_far']['mean']:.4f}±{summary['test_far']['std']:.4f}`", "", "| 留出站点 | F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['held_out_station']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} |")
    lines += ["", "该结果用于外部冷启动泛化审计，不能通过删除弱站点或重新使用留出站点标签来改善。"]
    (REPORTS / "stage57_noaa_leave_one_station.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(stations), "seeds": len(SEEDS), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
