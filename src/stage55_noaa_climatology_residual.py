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

from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402


def residualize(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = frame.copy()
    if mode == "raw":
        return out
    timestamp = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    station = out["station_key"].astype(str)
    hour = timestamp.dt.hour.fillna(-1).astype(int)
    train = out["split"].eq("train").to_numpy()
    normal_train = train & (~out["combined_is_attack"].astype(bool).to_numpy())
    values = pd.to_numeric(out["observed_value"], errors="coerce")
    train_frame = pd.DataFrame({"station": station[normal_train].to_numpy(), "hour": hour[normal_train].to_numpy(), "value": values[normal_train].to_numpy()})
    cell_median = train_frame.groupby(["station", "hour"], sort=True)["value"].median()
    station_median = train_frame.groupby("station", sort=True)["value"].median()
    global_median = float(train_frame["value"].median())
    cell_key = pd.MultiIndex.from_arrays([station.to_numpy(), hour.to_numpy()])
    baseline = cell_median.reindex(cell_key).to_numpy(dtype=float)
    baseline = np.where(np.isfinite(baseline), baseline, station_median.reindex(station).to_numpy(dtype=float))
    baseline = np.where(np.isfinite(baseline), baseline, global_median)
    out["observed_value"] = values.to_numpy(dtype=float) - baseline
    out["effective_observed_value"] = out["observed_value"]
    return out


def main() -> int:
    source = pd.read_parquet(NOAA)
    rows: list[dict[str, object]] = []
    modes = ["raw", "station_hour_residual"]
    for seed in SEEDS:
        base = build_dataset(source, seed)
        for mode in modes:
            frame = residualize(base, mode)
            _, result = score(frame)
            rows.append({
                "seed": seed,
                "mode": mode,
                "valid_f1": result["valid"]["f1"],
                "test_f1": result["test"]["f1"],
                "test_recall": result["test"]["recall"],
                "test_precision": result["test"]["precision"],
                "test_auprc": result["test"]["average_precision"],
                "test_far": result["test_false_alarm_rate"],
                "threshold": result["setting"]["threshold"],
            })
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("mode", sort=False)[["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["mode"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA 12站点、5种子；站点—小时气候基线只由train正常观测估计；valid选择阈值；test只评价。", "seeds": SEEDS, "modes": modes, "detail": rows, "aggregate": aggregate.to_dict(orient="records")}
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage55_noaa_climatology_residual_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage55_noaa_climatology_residual.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage55_noaa_climatology_residual.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段55 NOAA站点—小时气候残差", "", "基线仅由train正常观测估计；valid选择阈值；test只评价。", "", "| 版本 | test F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['mode']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} |")
    lines += ["", "该实验用于判断季节/站点域偏移是否可由训练期气候基线削弱；若有效，需再做跨季节和留一站点复核。"]
    (REPORTS / "stage55_noaa_climatology_residual.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "aggregate": payload["aggregate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
