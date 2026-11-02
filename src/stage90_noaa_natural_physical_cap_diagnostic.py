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
CAPS = [float("inf"), 10.0, 5.0, 3.0, 2.0, 1.0]
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


def component_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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
    return out, [detector.name for detector in detectors]


def risk_with_cap(scored: pd.DataFrame, names: list[str], cap: float) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    train = scored["split"].eq("train")
    normalized = []
    locations = {}
    for name in names:
        values = pd.to_numeric(scored[name], errors="coerce").to_numpy(dtype=float)
        if name == "Physical Rules" and np.isfinite(cap):
            values = np.minimum(values, cap)
        series = pd.Series(values, index=scored.index)
        loc, scale = robust_location_scale(series.loc[train & ~scored["missing_observation"]], floor=0.25)
        locations[name] = (float(loc), float(scale))
        normalized.append(np.nan_to_num(np.maximum(values - loc, 0.0) / scale, nan=0.0, posinf=1e6))
    return np.mean(np.vstack(normalized), axis=0), locations


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for target_year in TARGET_YEARS:
        train_years = [year for year in years if year < target_year]
        if not train_years:
            continue
        frame = make_frame(source, target_year, train_years, set(stations))
        scored, names = component_scores(frame)
        train = scored["split"].eq("train").to_numpy()
        test = scored["split"].eq("test").to_numpy()
        quality = pd.to_numeric(scored["temperature_quality"], errors="coerce").fillna(9).ne(1).to_numpy()
        for cap in CAPS:
            risk, locations = risk_with_cap(scored, names, cap)
            train_risk = risk[train]
            test_risk = risk[test]
            for q in Q_GRID:
                threshold = float(np.nanquantile(train_risk[np.isfinite(train_risk)], q))
                pred = np.isfinite(risk) & (risk >= threshold)
                flagged = test & quality
                clean = test & ~quality
                rows.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "physical_rule_cap": None if not np.isfinite(cap) else cap, "q": q, "threshold": threshold, "target_rows": int(test.sum()), "target_mean_risk": float(np.nanmean(test_risk)), "target_p95_risk": float(np.nanquantile(test_risk, 0.95)), "target_alarm_rate": float(pred[test].mean()), "quality_flagged_rows": int(flagged.sum()), "quality_flagged_alarm_rate": float(pred[flagged].mean()) if flagged.any() else float("nan"), "quality_clean_rows": int(clean.sum()), "quality_clean_alarm_rate": float(pred[clean].mean()) if clean.any() else float("nan"), "physical_location": locations["Physical Rules"][0], "physical_scale": locations["Physical Rules"][1]})
                for station in stations:
                    mask = test & scored["station_key"].eq(station).to_numpy()
                    station_rows.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "physical_rule_cap": None if not np.isfinite(cap) else cap, "q": q, "station_key": station, "rows": int(mask.sum()), "alarm_rate": float(pred[mask].mean()) if mask.any() else float("nan")})
        print(json.dumps({"target_year": target_year, "train_years": train_years, "rows": int(test.sum()), "caps": len(CAPS)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    station_detail = pd.DataFrame(station_rows)
    aggregate = detail.groupby(["target_year", "physical_rule_cap", "q"], dropna=False, sort=False).first().reset_index()
    station_summary = station_detail.groupby(["target_year", "physical_rule_cap", "q"], dropna=False, sort=False).agg(stations=("station_key", "nunique"), alarm_rate_mean=("alarm_rate", "mean"), alarm_rate_median=("alarm_rate", "median"), alarm_rate_p95=("alarm_rate", lambda x: float(x.quantile(0.95)))).reset_index()
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020-2025扩展表；11个全年度合格站点；历史年份自然观测训练、目标年份自然观测测试；Physical Rules上限仅作预先列出的诊断网格，不使用目标年标签选择；不注入攻击。", "stations": stations, "target_years": TARGET_YEARS, "caps": [None if not np.isfinite(x) else x for x in CAPS], "q_grid": Q_GRID, "detail": rows, "station_detail": station_rows, "aggregate": aggregate.to_dict(orient="records"), "station_summary": station_summary.to_dict(orient="records"), "excluded_station_years": audit["excluded_station_years"]}
    detail.to_csv(TABLES / "stage90_noaa_natural_physical_cap_detail.csv", index=False, encoding="utf-8-sig")
    station_detail.to_csv(TABLES / "stage90_noaa_natural_physical_cap_station_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage90_noaa_natural_physical_cap.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(TABLES / "stage90_noaa_natural_physical_cap_station_summary.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage90_noaa_natural_physical_cap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段90 NOAA自然观测物理规则上限诊断", "", "在自然观测上比较Physical Rules分量上限对训练阈值、目标年报警率和站点异质性的影响；不使用目标年标签选择。", "", "| 目标年 | cap | q | 阈值 | 目标报警率 | 质量可疑报警率 | 质量正常报警率 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        cap = "∞" if pd.isna(row["physical_rule_cap"]) else f"{row['physical_rule_cap']:.1f}"
        lines.append(f"| {row['target_year']} | {cap} | {row['q']:.3f} | {row['threshold']:.4f} | {row['target_alarm_rate']:.4f} | {row['quality_flagged_alarm_rate']:.4f} ({row['quality_flagged_rows']}) | {row['quality_clean_alarm_rate']:.4f} ({row['quality_clean_rows']}) |")
    lines += ["", "自然报警率不是攻击召回率；该阶段只判断分数尺度和自然误报的诊断方向，不自动替换主模型。"]
    (REPORTS / "stage90_noaa_natural_physical_cap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(stations), "rows": len(detail)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
