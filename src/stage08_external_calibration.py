from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detectors import CUSUMDetector, HampelMADDetector, KalmanInnovationDetector, PhysicalRulesDetector, SpatialConsistencyDetector

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ATTACK = Path(r"F:\数据集\Analysis\experiments\pc_mstf_ap_v4\parquet\v4_anomaly_injection_table.parquet")
NOAA = ROOT / "tables" / "stage06_noaa_isd_normalized.parquet"
ERA5 = ROOT / "tables" / "stage06_era5_station_reference.parquet"


def split_scores(detector, df: pd.DataFrame) -> np.ndarray:
    score = np.full(len(df), np.nan, dtype=float)
    if "split" in df:
        splits = [x for x in ["train", "valid", "test"] if (df["split"] == x).any()]
    else:
        splits = ["external"]
    for split in splits:
        idx = np.flatnonzero(df["split"].eq(split).to_numpy())
        if len(idx):
            score[idx] = detector.score(df.iloc[idx])
    return score


def prepare_noaa(noaa: pd.DataFrame) -> pd.DataFrame:
    out = noaa.rename(columns={"station_id": "station_key", "temperature_c": "observed_value"}).copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp_utc", "observed_value"])
    out["split"] = np.where(out["timestamp_utc"] < pd.Timestamp("2022-07-01", tz="UTC"), "calibration", "holdout")
    out["is_anomaly"] = False
    return out[["station_key", "timestamp_utc", "observed_value", "split", "is_anomaly"]]


def noaa_calibration(local: pd.DataFrame, noaa: pd.DataFrame) -> pd.DataFrame:
    local = local.copy()
    local["timestamp_utc"] = pd.to_datetime(local["timestamp_utc"], utc=True, errors="coerce")
    noaa = prepare_noaa(noaa)
    train = local["split"].eq("train")
    valid = local["split"].eq("valid")
    normal = train & ~local["is_anomaly"].astype(bool)
    y = local["is_anomaly"].astype(int).to_numpy()
    detectors = [PhysicalRulesDetector(), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    rows = []
    for detector in detectors:
        detector.fit(local.loc[train], normal.loc[train])
        local_score = split_scores(detector, local)
        from sklearn.metrics import f1_score
        candidates = np.unique(np.quantile(local_score[valid.to_numpy()][np.isfinite(local_score[valid.to_numpy()])], np.linspace(0.02, 0.98, 193)))
        local_threshold = max(((f1_score(y[valid.to_numpy()], local_score[valid.to_numpy()] >= t, zero_division=0), float(t)) for t in candidates), key=lambda x: x[0])[1]
        noaa_score = detector.score(noaa)
        cal = noaa["split"].eq("calibration").to_numpy()
        hold = noaa["split"].eq("holdout").to_numpy()
        external_threshold = float(np.nanquantile(noaa_score[cal], 0.995))
        rows.append({"detector": detector.name, "local_valid_threshold": float(local_threshold), "external_calibration_quantile": 0.995, "external_threshold": external_threshold, "calibration_rows": int(cal.sum()), "holdout_rows": int(hold.sum()), "holdout_alarm_rate_local_threshold": float(np.mean(noaa_score[hold] >= local_threshold)), "holdout_alarm_rate_external_threshold": float(np.mean(noaa_score[hold] >= external_threshold)), "calibration_alarm_rate_external_threshold": float(np.mean(noaa_score[cal] >= external_threshold))})
    return pd.DataFrame(rows)


def era5_bias_calibration(local: pd.DataFrame, era5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = local.copy()
    local["timestamp_utc"] = pd.to_datetime(local["timestamp_utc"], utc=True, errors="coerce")
    local["hour_utc"] = local["timestamp_utc"].dt.floor("h")
    era = era5.copy()
    era["timestamp_utc"] = pd.to_datetime(era["timestamp_utc"], utc=True, errors="coerce")
    era = era.rename(columns={"timestamp_utc": "hour_utc"})

    def aggregate(split: str) -> pd.DataFrame:
        part = local[local["split"].eq(split) & ~local["is_anomaly"].astype(bool)]
        return part.groupby(["station_key", "hour_utc"], as_index=False).agg(local_original_c=("original_value", "mean"))

    valid = aggregate("valid").merge(era, on=["station_key", "hour_utc"], how="inner")
    test = local[local["split"].eq("test")].groupby(["station_key", "hour_utc"], as_index=False).agg(local_original_c=("original_value", "mean"), local_observed_c=("observed_value", "mean"), anomaly_rows=("is_anomaly", "sum")).merge(era, on=["station_key", "hour_utc"], how="inner")
    bias = valid.assign(bias_c=valid["local_original_c"] - valid["era5_t2m_c"]).groupby("station_key", as_index=False)["bias_c"].median().rename(columns={"bias_c": "median_bias_c"})
    test = test.merge(bias, on="station_key", how="left")
    test["era5_corrected_t2m_c"] = test["era5_t2m_c"] + test["median_bias_c"].fillna(0.0)
    test["error_raw_c"] = test["local_original_c"] - test["era5_t2m_c"]
    test["error_corrected_c"] = test["local_original_c"] - test["era5_corrected_t2m_c"]
    test["regional_temp_change_c"] = test.groupby("station_key", sort=False)["era5_t2m_c"].diff().abs()
    summary = pd.DataFrame([{"valid_calibration_rows": int(len(valid)), "test_rows": int(len(test)), "stations": int(test["station_key"].nunique()), "raw_mae_c": float(test["error_raw_c"].abs().mean()), "corrected_mae_c": float(test["error_corrected_c"].abs().mean()), "raw_rmse_c": float(np.sqrt(np.mean(test["error_raw_c"] ** 2))), "corrected_rmse_c": float(np.sqrt(np.mean(test["error_corrected_c"] ** 2))), "extreme_background_hours": int((test["regional_temp_change_c"] >= 2.0).sum())}])
    return summary, test


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段8：NOAA外部阈值校准与ERA5站点偏差订正")
    parser.add_argument("--local", type=Path, default=LOCAL_ATTACK)
    parser.add_argument("--noaa", type=Path, default=NOAA)
    parser.add_argument("--era5", type=Path, default=ERA5)
    args = parser.parse_args()
    local, noaa, era5 = pd.read_parquet(args.local), pd.read_parquet(args.noaa), pd.read_parquet(args.era5)
    noaa_result = noaa_calibration(local, noaa)
    era_summary, era_detail = era5_bias_calibration(local, era5)
    out = ROOT / "tables"
    noaa_result.to_csv(out / "stage08_noaa_calibration.csv", index=False, encoding="utf-8-sig")
    era_summary.to_csv(out / "stage08_era5_bias_summary.csv", index=False, encoding="utf-8-sig")
    era_detail.to_parquet(out / "stage08_era5_bias_detail.parquet", index=False)
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "interpretation": "外部阈值校准和ERA5偏差订正流程通过；校准后性能需进入正式多种子攻击实验验证，不得将无攻击外部报警率等同于攻击召回率", "noaa": noaa_result.to_dict(orient="records"), "era5": era_summary.to_dict(orient="records")}
    (out / "stage08_external_calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ROOT / "reports" / "stage08_external_calibration.md"
    lines = ["# 阶段8外部校准", "", "- 状态：**PASS_PIPELINE_REVIEW_REQUIRED**", "- NOAA：前半年作为无攻击校准期，7月作为留出期；外部报警率不是攻击召回率。", "- ERA5：用验证期干净观测估计站点偏差，在测试期评估订正效果。", "", "## NOAA", "", "| 检测器 | 本地阈值下留出报警率 | 外部校准阈值下留出报警率 | 外部校准期报警率 |", "|---|---:|---:|---:|"]
    for _, r in noaa_result.iterrows(): lines.append(f"| {r['detector']} | {r['holdout_alarm_rate_local_threshold']:.4f} | {r['holdout_alarm_rate_external_threshold']:.4f} | {r['calibration_alarm_rate_external_threshold']:.4f} |")
    r = era_summary.iloc[0]
    lines += ["", "## ERA5", "", f"- 验证校准记录：`{int(r['valid_calibration_rows'])}`；测试记录：`{int(r['test_rows'])}`", f"- 原始MAE：`{r['raw_mae_c']:.4f} °C`；订正后MAE：`{r['corrected_mae_c']:.4f} °C`", f"- 原始RMSE：`{r['raw_rmse_c']:.4f} °C`；订正后RMSE：`{r['corrected_rmse_c']:.4f} °C`", f"- ERA5背景温度小时变化不低于2°C：`{int(r['extreme_background_hours'])}`", "", "阶段8只完成跨域校准流程。下一阶段需要在校准后的设置下进行多种子、未见攻击和可信处置实验。"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
