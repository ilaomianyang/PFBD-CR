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
METHODS = ["Trust legacy", "Trust component_zmax", "One-Class SVM"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09b_robust_fusion import robust_location_scale  # noqa: E402
from stage90_noaa_natural_physical_cap_diagnostic import component_scores, make_frame  # noqa: E402
from stage91_noaa_physical_component_repair import component_physical_scores  # noqa: E402
from stage92_noaa_physical_component_bootstrap import output_for_method  # noqa: E402
from stage98_noaa_low_far_baseline_comparison import add_structural, baseline_risks  # noqa: E402
from stage93_noaa_natural_component_validation import natural_scores  # noqa: E402


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
        frame["combined_event_id"] = frame.index.astype(str)
        scored, detector_names = component_scores(frame)
        physical, physical_diag = component_physical_scores(scored)
        scored = add_structural(scored)
        trust_legacy, loc_legacy = natural_scores(scored, physical["legacy"], "legacy")
        trust_component, loc_component = natural_scores(scored, physical["component_zmax"], "component_zmax")
        baseline = baseline_risks(scored)
        risk_map = {"Trust legacy": trust_legacy, "Trust component_zmax": trust_component, "One-Class SVM": baseline["One-Class SVM"]}
        train_mask = scored["split"].eq("train").to_numpy()
        test_mask = scored["split"].eq("test").to_numpy()
        quality_bad = pd.to_numeric(scored["temperature_quality"], errors="coerce").fillna(9).ne(1).to_numpy()
        for method, risk in risk_map.items():
            risk = np.asarray(risk, dtype=float)
            finite_train = train_mask & np.isfinite(risk)
            finite_test = test_mask & np.isfinite(risk)
            for q in Q_GRID:
                threshold = float(np.nanquantile(risk[finite_train], q))
                pred = np.isfinite(risk) & (risk >= threshold)
                bad = finite_test & quality_bad
                clean = finite_test & ~quality_bad
                rows.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "method": method, "q": q, "threshold": threshold, "train_rows": int(train_mask.sum()), "target_rows": int(finite_test.sum()), "target_alarm_rate": float(pred[finite_test].mean()) if finite_test.any() else float("nan"), "target_risk_mean": float(np.nanmean(risk[finite_test])) if finite_test.any() else float("nan"), "target_risk_p95": float(np.nanquantile(risk[finite_test], 0.95)) if finite_test.any() else float("nan"), "quality_flagged_rows": int(bad.sum()), "quality_flagged_alarm_rate": float(pred[bad].mean()) if bad.any() else float("nan"), "quality_clean_rows": int(clean.sum()), "quality_clean_alarm_rate": float(pred[clean].mean()) if clean.any() else float("nan")})
                for station in stations:
                    mask = finite_test & scored["station_key"].astype(str).eq(station).to_numpy()
                    station_bad = mask & quality_bad
                    station_clean = mask & ~quality_bad
                    station_rows.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "method": method, "q": q, "station_key": station, "rows": int(mask.sum()), "alarm_rate": float(pred[mask].mean()) if mask.any() else float("nan"), "quality_flagged_rows": int(station_bad.sum()), "quality_flagged_alarm_rate": float(pred[station_bad].mean()) if station_bad.any() else float("nan"), "quality_clean_rows": int(station_clean.sum()), "quality_clean_alarm_rate": float(pred[station_clean].mean()) if station_clean.any() else float("nan")})
        diagnostics.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "detectors": detector_names, "physical_diagnostics": physical_diag, "legacy_physical_location": loc_legacy, "component_physical_location": loc_component})
        print(json.dumps({"target_year": target_year, "train_years": train_years, "rows": int(test_mask.sum()), "methods": len(METHODS)}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(rows)
    station_detail = pd.DataFrame(station_rows)
    aggregate = detail.copy()
    station_summary = station_detail.groupby(["target_year", "method", "q"], sort=False).agg(stations=("station_key", "nunique"), alarm_rate_mean=("alarm_rate", "mean"), alarm_rate_median=("alarm_rate", "median"), alarm_rate_p95=("alarm_rate", lambda x: float(x.quantile(0.95))), quality_contrast_positive_stations=("quality_flagged_alarm_rate", lambda x: int((x > station_detail.loc[x.index, "quality_clean_alarm_rate"]).sum()))).reset_index()
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020–2025扩展表；11个全年度质量合格站点；历史年份训练、目标年份自然观测测试；Trust legacy、Trust component_zmax与One-Class SVM；q=0.99/0.995/0.999预先固定；不注入攻击、不使用目标年标签；温度质量标记仅作弱标签。", "stations": stations, "target_years": TARGET_YEARS, "q_grid": Q_GRID, "methods": METHODS, "detail": rows, "station_detail": station_rows, "station_summary": station_summary.to_dict(orient="records"), "diagnostics": diagnostics, "limitations": ["自然报警率不是攻击召回率。", "NOAA质量标记是弱标签，不是真实攻击真值。", "不能根据自然观测报警率直接宣称检测性能普适。"]}
    detail.to_csv(TABLES / "stage107_noaa_natural_svm_validation_detail.csv", index=False, encoding="utf-8-sig")
    station_detail.to_csv(TABLES / "stage107_noaa_natural_svm_validation_station_detail.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(TABLES / "stage107_noaa_natural_svm_validation_station_summary.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage107_noaa_natural_svm_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段107 NOAA自然观测 Trust–SVM验证", "", "在无攻击自然观测上比较 Trust legacy、Trust component_zmax 和 One-Class SVM；质量标记只作弱标签。", "", "| 目标年 | 方法 | q | 报警率 | 质量可疑报警率 | 质量正常报警率 |", "|---:|---|---:|---:|---:|---:|"]
    for row in detail.to_dict(orient="records"):
        lines.append(f"| {row['target_year']} | {row['method']} | {row['q']:.3f} | {row['target_alarm_rate']:.4f} | {row['quality_flagged_alarm_rate']:.4f} ({row['quality_flagged_rows']}) | {row['quality_clean_alarm_rate']:.4f} ({row['quality_clean_rows']}) |")
    lines += ["", "自然报警率只能支持误报和弱标签审计，不能解释成攻击召回率。"]
    (REPORTS / "stage107_noaa_natural_svm_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(detail), "station_rows": len(station_detail)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
