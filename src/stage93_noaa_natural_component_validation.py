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
METHODS = ["legacy", "component_zmax"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09b_robust_fusion import robust_location_scale  # noqa: E402
from stage90_noaa_natural_physical_cap_diagnostic import component_scores, make_frame  # noqa: E402
from stage91_noaa_physical_component_repair import component_physical_scores  # noqa: E402
from trust_fusion import TrustFusion  # noqa: E402


def natural_scores(frame: pd.DataFrame, physical: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    out = frame.copy()
    out["Physical Rules"] = physical
    train = out["split"].eq("train")
    fusion = TrustFusion(include_era5=False).fit(out.loc[train], out.loc[~train], pd.Series(False, index=out.index[~train]))
    risk = fusion.transform(out)
    locations = {name: fusion.location_scale[name] for name in fusion.components}
    return risk, locations


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for target_year in TARGET_YEARS:
        train_years = [year for year in years if year < target_year]
        if not train_years:
            continue
        frame = make_frame(source, target_year, train_years, set(stations))
        frame["combined_is_attack"] = False
        frame["combined_attack_type"] = "normal"
        scored, _ = component_scores(frame)
        physical, component_diag = component_physical_scores(frame)
        train = scored["split"].eq("train").to_numpy()
        test = scored["split"].eq("test").to_numpy()
        quality = pd.to_numeric(scored["temperature_quality"], errors="coerce").fillna(9).ne(1).to_numpy()
        for method in METHODS:
            risk, locations = natural_scores(scored, physical[method], method)
            train_finite = train & np.isfinite(risk)
            target_finite = test & np.isfinite(risk)
            for q in Q_GRID:
                threshold = float(np.nanquantile(risk[train_finite], q))
                pred = np.isfinite(risk) & (risk >= threshold)
                flagged = target_finite & quality
                clean = target_finite & ~quality
                rows.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "method": method, "q": q, "threshold": threshold, "train_rows": int(train.sum()), "target_rows": int(test.sum()), "target_alarm_rate": float(pred[target_finite].mean()) if target_finite.any() else float("nan"), "target_risk_mean": float(np.nanmean(risk[target_finite])) if target_finite.any() else float("nan"), "target_risk_p95": float(np.nanquantile(risk[target_finite], 0.95)) if target_finite.any() else float("nan"), "quality_flagged_rows": int(flagged.sum()), "quality_flagged_alarm_rate": float(pred[flagged].mean()) if flagged.any() else float("nan"), "quality_clean_rows": int(clean.sum()), "quality_clean_alarm_rate": float(pred[clean].mean()) if clean.any() else float("nan"), "physical_location": float(locations["Physical Rules"][0]), "physical_scale": float(locations["Physical Rules"][1])})
                for station in stations:
                    mask = target_finite & scored["station_key"].eq(station).to_numpy()
                    station_flagged = mask & quality
                    station_clean = mask & ~quality
                    station_rows.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "method": method, "q": q, "station_key": station, "rows": int(mask.sum()), "alarm_rate": float(pred[mask].mean()) if mask.any() else float("nan"), "quality_flagged_rows": int(station_flagged.sum()), "quality_flagged_alarm_rate": float(pred[station_flagged].mean()) if station_flagged.any() else float("nan"), "quality_clean_rows": int(station_clean.sum()), "quality_clean_alarm_rate": float(pred[station_clean].mean()) if station_clean.any() else float("nan")})
            diagnostics.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "method": method, "component_rate_scale": component_diag["rate_scale"], "component_stats": json.dumps(component_diag["component_stats"], ensure_ascii=False), "train_quantiles": json.dumps(component_diag["train_quantiles"], ensure_ascii=False)})
        print(json.dumps({"target_year": target_year, "train_years": train_years, "rows": int(test.sum()), "methods": len(METHODS)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    station_detail = pd.DataFrame(station_rows)
    diag_df = pd.DataFrame(diagnostics)
    aggregate = detail.groupby(["target_year", "method", "q"], sort=False).first().reset_index()
    station_summary = station_detail.groupby(["target_year", "method", "q"], sort=False).agg(stations=("station_key", "nunique"), alarm_rate_mean=("alarm_rate", "mean"), alarm_rate_median=("alarm_rate", "median"), alarm_rate_p95=("alarm_rate", lambda x: float(x.quantile(0.95)))).reset_index()
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020–2025扩展表；11个全年度质量合格站点；历史年份训练、目标年份自然观测测试；q网格预先固定；不注入攻击、不使用目标年标签；温度质量标记仅作弱标签。", "stations": stations, "target_years": TARGET_YEARS, "q_grid": Q_GRID, "methods": METHODS, "detail": rows, "station_detail": station_rows, "aggregate": aggregate.to_dict(orient="records"), "station_summary": station_summary.to_dict(orient="records"), "diagnostics": diagnostics}
    detail.to_csv(TABLES / "stage93_noaa_natural_component_validation_detail.csv", index=False, encoding="utf-8-sig")
    station_detail.to_csv(TABLES / "stage93_noaa_natural_component_validation_station_detail.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(TABLES / "stage93_noaa_natural_component_validation_station_summary.csv", index=False, encoding="utf-8-sig")
    diag_df.to_csv(TABLES / "stage93_noaa_natural_component_validation_diagnostics.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage93_noaa_natural_component_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段93 NOAA自然观测组件方案验证", "", "在无攻击自然观测上比较legacy与component_zmax；目标年不参与阈值选择，质量标记只作弱标签。", "", "| 目标年 | 方法 | q | 阈值 | 自然报警率 | 质量可疑报警率 | 质量正常报警率 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['target_year']} | {row['method']} | {row['q']:.3f} | {row['threshold']:.4f} | {row['target_alarm_rate']:.4f} | {row['quality_flagged_alarm_rate']:.4f} ({row['quality_flagged_rows']}) | {row['quality_clean_alarm_rate']:.4f} ({row['quality_clean_rows']}) |")
    lines += ["", "自然报警率不是攻击召回率；该阶段用于判断component_zmax是否仅通过压低所有自然风险而获得低误报。"]
    (REPORTS / "stage93_noaa_natural_component_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(stations), "rows": len(detail)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
