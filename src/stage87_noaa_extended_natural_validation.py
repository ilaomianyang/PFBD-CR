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
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"
AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
TARGET_YEARS = [2021, 2022, 2023, 2024, 2025]
Q_GRID = [0.99, 0.995, 0.999]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors import CUSUMDetector, HampelMADDetector, KalmanInnovationDetector, PhysicalRulesDetector, SpatialConsistencyDetector  # noqa: E402
from trust_fusion import robust_location_scale  # noqa: E402


def make_frame(source: pd.DataFrame, target_year: int, train_years: list[int], stations: set[str]) -> pd.DataFrame:
    frame = source[source["station_id"].astype(str).isin(stations) & source["year"].isin([*train_years, target_year])].copy()
    frame["station_key"] = frame["station_id"].astype(str)
    frame["observed_value"] = pd.to_numeric(frame["temperature_c"], errors="coerce")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame["missing_observation"] = frame["observed_value"].isna()
    frame["split"] = frame["year"].map({**{year: "train" for year in train_years}, target_year: "test"})
    return frame.sort_values(["station_key", "timestamp_utc"], kind="stable").reset_index(drop=True)


def score_natural(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = frame.copy()
    train = out["split"].eq("train")
    detectors = [PhysicalRulesDetector(), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    for detector in detectors:
        detector.fit(out.loc[train], ~out.loc[train, "missing_observation"])
        values = np.full(len(out), np.nan, dtype=float)
        for split in ["train", "test"]:
            mask = out["split"].eq(split)
            if mask.any():
                values[mask.to_numpy()] = detector.score(out.loc[mask])
        out[detector.name] = values
    locations: dict[str, dict[str, float]] = {}
    normalized = []
    for detector in detectors:
        name = detector.name
        loc, scale = robust_location_scale(out.loc[train & ~out["missing_observation"], name], floor=0.25)
        locations[name] = {"location": float(loc), "scale": float(scale)}
        values = pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=float)
        normalized.append(np.nan_to_num(np.maximum(values - loc, 0.0) / scale, nan=0.0, posinf=1e6))
    out["equal_weight_risk"] = np.mean(np.vstack(normalized), axis=0)
    return out, locations


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    detail_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    locations: dict[str, object] = {}
    for target_year in TARGET_YEARS:
        train_years = [year for year in years if year < target_year]
        if not train_years:
            continue
        frame = make_frame(source, target_year, train_years, set(stations))
        scored, loc = score_natural(frame)
        locations[str(target_year)] = loc
        train = scored["split"].eq("train")
        test = scored["split"].eq("test")
        quality_flagged = pd.to_numeric(scored["temperature_quality"], errors="coerce").fillna(9).ne(1).to_numpy()
        train_risk = scored.loc[train, "equal_weight_risk"].to_numpy(dtype=float)
        test_risk = scored.loc[test, "equal_weight_risk"].to_numpy(dtype=float)
        for q in Q_GRID:
            threshold = float(np.nanquantile(train_risk[np.isfinite(train_risk)], q))
            pred = np.isfinite(scored["equal_weight_risk"].to_numpy(dtype=float)) & (scored["equal_weight_risk"].to_numpy(dtype=float) >= threshold)
            target = test.to_numpy()
            flagged = target & quality_flagged
            clean = target & ~quality_flagged
            detail_rows.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "q": q, "threshold": threshold, "train_rows": int(train.sum()), "target_rows": int(test.sum()), "target_mean_risk": float(np.nanmean(test_risk)) if len(test_risk) else float("nan"), "target_p95_risk": float(np.nanquantile(test_risk, 0.95)) if len(test_risk) else float("nan"), "target_alarm_rate": float(pred[target].mean()) if target.any() else float("nan"), "quality_flagged_rows": int(flagged.sum()), "quality_flagged_alarm_rate": float(pred[flagged].mean()) if flagged.any() else float("nan"), "quality_clean_rows": int(clean.sum()), "quality_clean_alarm_rate": float(pred[clean].mean()) if clean.any() else float("nan")})
            for station in stations:
                mask = target & scored["station_key"].eq(station).to_numpy()
                flag = mask & quality_flagged
                cln = mask & ~quality_flagged
                station_rows.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "q": q, "station_key": station, "rows": int(mask.sum()), "alarm_rate": float(pred[mask].mean()) if mask.any() else float("nan"), "quality_flagged_rows": int(flag.sum()), "quality_flagged_alarm_rate": float(pred[flag].mean()) if flag.any() else float("nan"), "quality_clean_rows": int(cln.sum()), "quality_clean_alarm_rate": float(pred[cln].mean()) if cln.any() else float("nan")})
        print(json.dumps({"target_year": target_year, "train_years": train_years, "rows": int(test.sum()), "stations": len(stations)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(detail_rows)
    station_detail = pd.DataFrame(station_rows)
    aggregate = detail.groupby(["target_year", "q"], sort=False).agg(train_years=("train_years", "first"), target_rows=("target_rows", "first"), target_alarm_rate=("target_alarm_rate", "first"), quality_flagged_rows=("quality_flagged_rows", "first"), quality_flagged_alarm_rate=("quality_flagged_alarm_rate", "first"), quality_clean_rows=("quality_clean_rows", "first"), quality_clean_alarm_rate=("quality_clean_alarm_rate", "first")).reset_index()
    station_summary = station_detail.groupby(["target_year", "q"], sort=False).agg(stations=("station_key", "nunique"), alarm_rate_mean=("alarm_rate", "mean"), alarm_rate_median=("alarm_rate", "median"), alarm_rate_p95=("alarm_rate", lambda x: float(x.quantile(0.95))), positive_quality_contrast_stations=("quality_flagged_alarm_rate", lambda x: int((x > station_detail.loc[x.index, "quality_clean_alarm_rate"]).sum()))).reset_index()
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020-2025扩展表；仅使用所有年度质量合格的11个站点；每个目标年使用此前所有年度训练正常观测拟合五个检测器和等权风险，训练风险分位数设阈值；目标年为自然观测，不注入攻击、不使用目标年标签。质量标记只作弱标签。", "stations": stations, "target_years": TARGET_YEARS, "q_grid": Q_GRID, "detail": detail_rows, "station_detail": station_rows, "aggregate": aggregate.to_dict(orient="records"), "station_summary": station_summary.to_dict(orient="records"), "component_locations": locations, "excluded_station_years": audit["excluded_station_years"]}
    detail.to_csv(TABLES / "stage87_noaa_extended_natural_validation_detail.csv", index=False, encoding="utf-8-sig")
    station_detail.to_csv(TABLES / "stage87_noaa_extended_natural_validation_station_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage87_noaa_extended_natural_validation.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(TABLES / "stage87_noaa_extended_natural_validation_station_summary.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage87_noaa_extended_natural_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段87 NOAA扩展多年度自然观测验证", "", "历史年份训练、目标年份自然观测测试；不注入攻击，不使用目标年标签。质量标记仅作为弱标签。", "", f"- 站点：`{len(stations)}`；目标年：`{TARGET_YEARS}`；分位数：`{Q_GRID}`", "", "| 目标年 | 训练年份 | q | 报警率 | 质量可疑报警率 | 质量正常报警率 |", "|---:|---|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['target_year']} | {row['train_years']} | {row['q']:.3f} | {row['target_alarm_rate']:.4f} | {row['quality_flagged_alarm_rate']:.4f} ({row['quality_flagged_rows']}) | {row['quality_clean_alarm_rate']:.4f} ({row['quality_clean_rows']}) |")
    lines += ["", "## 站点异质性", "", "| 目标年 | q | 站点数 | 平均报警率 | 报警率中位数 | 报警率P95 | 可疑组报警率更高站点 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in station_summary.to_dict(orient="records"):
        lines.append(f"| {row['target_year']} | {row['q']:.3f} | {row['stations']} | {row['alarm_rate_mean']:.4f} | {row['alarm_rate_median']:.4f} | {row['alarm_rate_p95']:.4f} | {row['positive_quality_contrast_stations']} |")
    lines += ["", "自然报警率不是攻击召回率；质量标记不是攻击真值；该阶段用于时间稳定性、误报和站点异质性审计。"]
    (REPORTS / "stage87_noaa_extended_natural_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(stations), "target_years": TARGET_YEARS, "rows": len(detail)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
