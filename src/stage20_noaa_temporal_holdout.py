from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from detectors import CUSUMDetector, HampelMADDetector, KalmanInnovationDetector, PhysicalRulesDetector, SpatialConsistencyDetector  # noqa: E402
from stage09_trust_pipeline import _prepare_local, _score_splits  # noqa: E402
from trust_fusion import TrustFusion  # noqa: E402


def main() -> int:
    local = _prepare_local(ROOT / "tables" / "stage05_controlled_attack_extensions.parquet")
    frame = local.copy(); frame["observed_value"] = frame["effective_observed_value"]
    train = frame["split"].eq("train"); normal = train & ~frame["combined_is_attack"]
    detectors = [PhysicalRulesDetector(), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    for detector in detectors:
        detector.fit(frame.loc[train], normal.loc[train])
        local[detector.name] = _score_splits(detector, frame)
    local["ERA5 Background"] = 0.0
    valid = local["split"].eq("valid")
    fusion = TrustFusion(include_era5=False).fit(local.loc[train], local.loc[valid], local.loc[valid, "combined_is_attack"])
    noaa = pd.read_parquet(ROOT / "tables" / "stage06_noaa_isd_normalized.parquet").rename(columns={"station_id": "station_key", "temperature_c": "observed_value"}).copy()
    noaa["timestamp_utc"] = pd.to_datetime(noaa["timestamp_utc"], utc=True, errors="coerce")
    noaa = noaa.dropna(subset=["timestamp_utc", "station_key"]).copy()
    for detector in detectors:
        noaa[detector.name] = detector.score(noaa)
    noaa["ERA5 Background"] = 0.0; noaa["missing_observation"] = noaa["observed_value"].isna()
    noaa["fused_risk"] = fusion.transform(noaa, extreme_background=pd.Series(False, index=noaa.index))
    calibration = noaa["timestamp_utc"] < pd.Timestamp("2022-07-01", tz="UTC")
    holdout = ~calibration
    threshold = float(np.nanquantile(noaa.loc[calibration, "fused_risk"], 0.995))
    noaa["alarm"] = noaa["fused_risk"] >= threshold
    noaa["month"] = noaa["timestamp_utc"].dt.month
    rows = []
    for month, group in noaa[holdout].groupby("month", sort=True):
        rows.append({"period": f"2022-{int(month):02d}", "station_key": "ALL", "rows": len(group), "mean_risk": float(group["fused_risk"].mean()), "p95_risk": float(group["fused_risk"].quantile(0.95)), "alarm_rate": float(group["alarm"].mean())})
        for station, station_group in group.groupby("station_key", sort=True):
            rows.append({"period": f"2022-{int(month):02d}", "station_key": station, "rows": len(station_group), "mean_risk": float(station_group["fused_risk"].mean()), "p95_risk": float(station_group["fused_risk"].quantile(0.95)), "alarm_rate": float(station_group["alarm"].mean())})
    detail = pd.DataFrame(rows)
    overall = {"calibration_rows": int(calibration.sum()), "holdout_rows": int(holdout.sum()), "threshold": threshold, "holdout_alarm_rate": float(noaa.loc[holdout, "alarm"].mean()), "monthly_all_max_alarm_rate": float(detail.loc[detail.station_key.eq("ALL"), "alarm_rate"].max())}
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    detail.to_csv(out_dir / "stage20_noaa_temporal_holdout.csv", index=False, encoding="utf-8-sig")
    (out_dir / "stage20_noaa_temporal_holdout.json").write_text(json.dumps({"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA 2022年1-6月校准，7-12月按月和站点留出；无攻击外部数据不计算攻击召回率", "overall": overall}, ensure_ascii=False, indent=2), encoding="utf-8")
    monthly = detail[detail.station_key.eq("ALL")]
    lines = ["# 阶段20 NOAA跨时间留出", "", "使用 2022 年 1–6 月 NOAA 自然观测估计 0.995 分位外部阈值，在 7–12 月按月和站点检查报警率。", "", f"- 校准行数：`{overall['calibration_rows']}`；留出行数：`{overall['holdout_rows']}`；阈值：`{threshold:.6f}`", f"- 留出期总体报警率：`{overall['holdout_alarm_rate']:.6f}`；月度总体最高报警率：`{overall['monthly_all_max_alarm_rate']:.6f}`", "", "| 月份 | 行数 | 平均风险 | P95风险 | 报警率 |", "|---|---:|---:|---:|---:|"]
    for _, r in monthly.iterrows():
        lines.append(f"| {r['period']} | {int(r['rows'])} | {r['mean_risk']:.4f} | {r['p95_risk']:.4f} | {r['alarm_rate']:.4f} |")
    lines += ["", "该结果描述自然观测上的时间稳健性，不是攻击召回率；站点和月份差异应作为外部域偏移讨论。"]
    report = report_dir / "stage20_noaa_temporal_holdout.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_PIPELINE_REVIEW_REQUIRED", "report": str(report), "overall": overall}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
