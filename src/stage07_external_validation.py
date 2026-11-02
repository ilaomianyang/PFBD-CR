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


def best_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import f1_score
    finite = np.isfinite(score)
    y, score = y[finite].astype(int), score[finite]
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 193)))
    values = [(f1_score(y, score >= t, zero_division=0), float(t)) for t in candidates]
    return max(values, key=lambda x: x[0])[1], max(values, key=lambda x: x[0])[0]


def split_scores(detector, df: pd.DataFrame) -> np.ndarray:
    score = np.full(len(df), np.nan, dtype=float)
    for split in ["train", "valid", "test"]:
        idx = np.flatnonzero(df["split"].eq(split).to_numpy())
        if len(idx):
            score[idx] = detector.score(df.iloc[idx])
    return score


def prepare_external_noaa(noaa: pd.DataFrame) -> pd.DataFrame:
    out = noaa.rename(columns={"station_id": "station_key", "temperature_c": "observed_value"}).copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp_utc", "observed_value"])
    out["split"] = "external"
    out["is_anomaly"] = False
    return out[["station_key", "timestamp_utc", "observed_value", "split", "is_anomaly"]]


def run_noaa_validation(local: pd.DataFrame, noaa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = local.copy()
    local["timestamp_utc"] = pd.to_datetime(local["timestamp_utc"], utc=True, errors="coerce")
    noaa = prepare_external_noaa(noaa)
    train, valid = local["split"].eq("train"), local["split"].eq("valid")
    normal = train & ~local["is_anomaly"].astype(bool)
    y = local["is_anomaly"].astype(int).to_numpy()
    detectors = [PhysicalRulesDetector(), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    summary, detail = [], []
    for detector in detectors:
        detector.fit(local.loc[train], normal.loc[train])
        local_score = split_scores(detector, local)
        threshold, valid_f1 = best_threshold(y[valid.to_numpy()], local_score[valid.to_numpy()])
        ext_score = detector.score(noaa)
        ext_pred = ext_score >= threshold
        summary.append({"detector": detector.name, "local_valid_F1": float(valid_f1), "threshold": float(threshold), "noaa_rows": int(len(noaa)), "noaa_predicted_anomaly_rate": float(np.mean(ext_pred)), "noaa_mean_score": float(np.nanmean(ext_score)), "noaa_p95_score": float(np.nanpercentile(ext_score, 95))})
        for station, group in noaa.assign(_score=ext_score, _pred=ext_pred).groupby("station_key", sort=True):
            detail.append({"detector": detector.name, "station_key": station, "rows": int(len(group)), "predicted_anomaly_rate": float(group["_pred"].mean()), "mean_score": float(group["_score"].mean()), "p95_score": float(group["_score"].quantile(0.95))})
    return pd.DataFrame(summary), pd.DataFrame(detail)


def run_era5_alignment(local: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    local = local.copy()
    local["timestamp_utc"] = pd.to_datetime(local["timestamp_utc"], utc=True, errors="coerce")
    local["hour_utc"] = local["timestamp_utc"].dt.floor("h")
    clean = local[local["split"].eq("test")].groupby(["station_key", "hour_utc"], as_index=False).agg(local_original_c=("original_value", "mean"), local_observed_c=("observed_value", "mean"), anomaly_rows=("is_anomaly", "sum"))
    era = era5.copy()
    era["timestamp_utc"] = pd.to_datetime(era["timestamp_utc"], utc=True, errors="coerce")
    era = era.rename(columns={"timestamp_utc": "hour_utc"})
    merged = clean.merge(era, on=["station_key", "hour_utc"], how="inner")
    merged["original_minus_era5_c"] = merged["local_original_c"] - merged["era5_t2m_c"]
    merged["observed_minus_era5_c"] = merged["local_observed_c"] - merged["era5_t2m_c"]
    merged["regional_temp_change_c"] = merged.groupby("station_key", sort=False)["era5_t2m_c"].diff().abs()
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段7：NOAA ISD与ERA5外部验证")
    parser.add_argument("--local", type=Path, default=LOCAL_ATTACK)
    parser.add_argument("--noaa", type=Path, default=NOAA)
    parser.add_argument("--era5", type=Path, default=ERA5)
    args = parser.parse_args()
    local, noaa, era5 = pd.read_parquet(args.local), pd.read_parquet(args.noaa), pd.read_parquet(args.era5)
    noaa_summary, noaa_detail = run_noaa_validation(local, noaa)
    alignment = run_era5_alignment(local, era5)
    era_summary = pd.DataFrame([{"rows": int(len(alignment)), "stations": int(alignment["station_key"].nunique()), "mae_original_vs_era5_c": float(alignment["original_minus_era5_c"].abs().mean()), "rmse_original_vs_era5_c": float(np.sqrt(np.mean(alignment["original_minus_era5_c"] ** 2))), "mean_bias_original_minus_era5_c": float(alignment["original_minus_era5_c"].mean()), "mae_observed_vs_era5_c": float(alignment["observed_minus_era5_c"].abs().mean()), "extreme_background_hours": int((alignment["regional_temp_change_c"] >= 2.0).sum())}])
    out_dir = ROOT / "tables"
    noaa_summary.to_csv(out_dir / "stage07_noaa_external_summary.csv", index=False, encoding="utf-8-sig")
    noaa_detail.to_csv(out_dir / "stage07_noaa_external_by_station.csv", index=False, encoding="utf-8-sig")
    alignment.to_parquet(out_dir / "stage07_era5_alignment.parquet", index=False)
    era_summary.to_csv(out_dir / "stage07_era5_alignment_summary.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "本地train拟合、local valid选阈值；NOAA无攻击外部稳健性；ERA5区域背景对齐；不把外部数据当攻击真值", "noaa_summary": noaa_summary.to_dict(orient="records"), "era5_summary": era_summary.to_dict(orient="records"), "interpretation": "数据接入和指标计算通过；外部域偏移与ERA5代表性偏差仍需在后续校准阶段处理，不得直接宣称外部检测性能通过"}
    (out_dir / "stage07_external_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ROOT / "reports" / "stage07_external_validation.md"
    lines = ["# 阶段7外部数据验证", "", "- 状态：**PASS_PIPELINE_REVIEW_REQUIRED**", "- NOAA ISD：作为无攻击外部地面观测，报告自然数据上的预测异常率，不赋予攻击标签。", "- ERA5：作为区域背景场，报告与六盘山原始观测的对齐误差和极端背景小时数。", "- 解释：流程和文件通过，但外部域偏移、站点尺度校准及 ERA5 高程代表性偏差尚未处理。", "", "## NOAA汇总", "", "| 检测器 | 本地valid F1 | NOAA预测异常率 | NOAA平均分 | NOAA P95分数 |", "|---|---:|---:|---:|---:|"]
    for _, r in noaa_summary.iterrows(): lines.append(f"| {r['detector']} | {r['local_valid_F1']:.4f} | {r['noaa_predicted_anomaly_rate']:.4f} | {r['noaa_mean_score']:.4f} | {r['noaa_p95_score']:.4f} |")
    r = era_summary.iloc[0]
    lines += ["", "## ERA5汇总", "", f"- 对齐行数：`{int(r['rows'])}`；站点数：`{int(r['stations'])}`", f"- 原始观测与 ERA5 MAE：`{r['mae_original_vs_era5_c']:.4f} °C`", f"- 原始观测与 ERA5 RMSE：`{r['rmse_original_vs_era5_c']:.4f} °C`", f"- 平均偏差：`{r['mean_bias_original_minus_era5_c']:.4f} °C`", f"- 观测值与 ERA5 MAE：`{r['mae_observed_vs_era5_c']:.4f} °C`", f"- 区域背景温度小时变化不低于 2°C 的记录数：`{int(r['extreme_background_hours'])}`", "", "外部验证不替代六盘山主实验；ERA5不作为现场真值，NOAA无攻击数据不用于计算攻击召回率。"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
