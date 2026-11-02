from __future__ import annotations

"""Stage 142: label-free natural NOAA validation of the frozen candidate.

No controlled attacks are injected here.  Temperature quality fields are used
only as weak labels for association, never as attack truth.  The frozen
alpha/window rule from stage 141 is applied to a label-free operational score;
detector locations/scales are fitted on historical normal observations for each
target year.  This is intentionally a natural-alarm audit, not a new F1 test.
"""

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"
AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
TARGET_YEARS = [2021, 2022, 2023, 2024, 2025]
Q_GRID = [0.99, 0.995, 0.999]
FROZEN_ALPHA = 0.25
FROZEN_WINDOW = 1
METHODS = ["TrustMeteo-QC default", "TrustMeteo-QC frozen expanded", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
DETECTOR_COLS = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency"]
SCORE_COLS = DETECTOR_COLS + ["structural_risk"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors import CUSUMDetector, HampelMADDetector, KalmanInnovationDetector, PhysicalRulesDetector, SpatialConsistencyDetector  # noqa: E402
from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, structural_features, structural_risk_from_norm  # noqa: E402
from stage09b_robust_fusion import robust_location_scale  # noqa: E402


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def make_frame(source: pd.DataFrame, target_year: int, train_years: list[int], stations: set[str]) -> pd.DataFrame:
    frame = source[source["station_id"].astype(str).isin(stations) & source["year"].isin([*train_years, target_year])].copy()
    frame["station_key"] = frame["station_id"].astype(str)
    frame["observed_value"] = pd.to_numeric(frame["temperature_c"], errors="coerce")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame["missing_observation"] = frame["observed_value"].isna()
    frame["combined_is_attack"] = False
    frame["combined_attack_type"] = "normal"
    frame["split"] = frame["year"].map({**{year: "train" for year in train_years}, target_year: "test"})
    return frame.dropna(subset=["timestamp_utc"]).sort_values(["station_key", "timestamp_utc"], kind="stable").reset_index(drop=True)


def score_components(frame: pd.DataFrame) -> pd.DataFrame:
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
    parts = []
    for split in ["train", "test"]:
        mask = out["split"].eq(split)
        part = structural_features(out.loc[mask], STRUCTURAL_BASE).set_axis(out.index[mask])
        parts.append(part)
    features = pd.concat(parts).reindex(out.index).fillna(0.0)
    features["station_key"] = out["station_key"].astype(str)
    normal_train = train & ~out["missing_observation"]
    scaler = AdaptiveScaler(STRUCTURAL_BASE).fit(features, normal_train)
    out["structural_risk"] = structural_risk_from_norm(scaler.transform(features), enhanced=False)
    return out


def equal_weight_base_risk(scored: pd.DataFrame) -> np.ndarray:
    train = scored["split"].eq("train") & ~scored["missing_observation"]
    normalized = []
    for name in DETECTOR_COLS:
        location, scale = robust_location_scale(scored.loc[train, name], floor=0.25)
        values = pd.to_numeric(scored[name], errors="coerce").to_numpy(dtype=float)
        normalized.append(np.nan_to_num(np.maximum(values - location, 0.0) / scale, nan=0.0, posinf=1e6))
    return np.mean(np.vstack(normalized), axis=0)


def baseline_risks(scored: pd.DataFrame) -> dict[str, np.ndarray]:
    train = scored["split"].eq("train") & ~scored["missing_observation"]
    x = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    med = np.nanmedian(x[train], axis=0)
    scale = np.maximum(1.4826 * np.nanmedian(np.abs(x[train] - med), axis=0), 0.25)
    robust = np.maximum((x - med) / scale, 0.0).max(axis=1)
    iso = IsolationForest(n_estimators=300, random_state=stable_seed("stage142", "if"), n_jobs=1, contamination="auto").fit(x[train])
    svm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)).fit(x[train])
    return {"Robust max deviation": robust, "Isolation Forest": -iso.score_samples(x), "One-Class SVM": -svm.decision_function(x).ravel()}


def station_bootstrap(rows: pd.DataFrame, value_col: str, seed: int, repeats: int = 10000) -> tuple[float, float]:
    rows = rows.loc[np.isfinite(pd.to_numeric(rows[value_col], errors="coerce"))].copy()
    if rows.empty:
        return float("nan"), float("nan")
    stations = rows["station_key"].drop_duplicates().astype(str).tolist()
    by_station = {s: rows.loc[rows["station_key"].astype(str).eq(s), value_col].to_numpy(dtype=float) for s in stations}
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for i in range(repeats):
        sample = rng.choice(stations, size=len(stations), replace=True)
        means[i] = np.concatenate([by_station[s] for s in sample]).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def finalize_existing() -> int:
    """Rebuild weak-label summaries after a serialization-only interruption."""
    detail_df = pd.read_csv(TABLES / "stage142_noaa_frozen_natural_operational_validation_detail.csv")
    station_df = pd.read_csv(TABLES / "stage142_noaa_frozen_natural_operational_validation_station_detail.csv")
    aggregate = detail_df.groupby(["target_year", "method", "q"], sort=False).first().reset_index()
    station_summary = station_df.groupby(["target_year", "method", "q"], sort=False).agg(stations=("station_key", "nunique"), alarm_rate_mean=("alarm_rate", "mean"), alarm_rate_median=("alarm_rate", "median"), alarm_rate_p95=("alarm_rate", lambda x: float(x.quantile(0.95))), quality_contrast_positive_stations=("quality_flagged_alarm_rate", lambda x: int((x > station_df.loc[x.index, "quality_clean_alarm_rate"]).sum()))).reset_index()
    contrast_rows = []
    q_rows = station_df.loc[station_df["q"].eq(0.995)].copy()
    a = q_rows.loc[q_rows["method"].eq("TrustMeteo-QC frozen expanded")].set_index(["target_year", "station_key"])
    for other in ["TrustMeteo-QC default", "One-Class SVM"]:
        b = q_rows.loc[q_rows["method"].eq(other)].set_index(["target_year", "station_key"])
        common = a.join(b, lsuffix="_expanded", rsuffix="_other", how="inner")
        for metric in ["alarm_rate", "quality_flagged_alarm_rate", "quality_clean_alarm_rate"]:
            diff = pd.to_numeric(common[f"{metric}_expanded"], errors="coerce") - pd.to_numeric(common[f"{metric}_other"], errors="coerce")
            finite = np.isfinite(diff.to_numpy(dtype=float))
            tmp = pd.DataFrame({"station_key": common.index.get_level_values("station_key")[finite], "value": diff.to_numpy(dtype=float)[finite]})
            low, high = station_bootstrap(tmp, "value", stable_seed("stage142", other, metric))
            contrast_rows.append({"comparison": f"expanded - {other}", "metric": metric, "q": 0.995, "mean_diff": float(diff.mean()) if finite.any() else float("nan"), "bootstrap_station_ci95_low": low, "bootstrap_station_ci95_high": high, "cells": int(finite.sum())})
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(TABLES / "stage142_noaa_frozen_natural_operational_validation_weak_label_paired.csv", index=False, encoding="utf-8-sig")
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(r["year"]), str(r["station_id"])) for r in audit["detail"] if r.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({s for s in source["station_id"].astype(str).unique() if all((year, s) in eligible for year in years)})
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020-2025扩展表；11个全年度质量合格站点；历史年份自然观测训练→目标年份自然观测测试；不注入攻击、不使用攻击真值。阶段141冻结alpha=0.25/window=1规则用于无标签自然报警运行投影；每个目标年仅用历史训练正常观测估计尺度，q=0.99/0.995/0.999预先固定。温度质量标记仅作弱标签，不能计算攻击召回。", "stations": stations, "target_years": TARGET_YEARS, "methods": METHODS, "q_grid": Q_GRID, "frozen_alpha": FROZEN_ALPHA, "frozen_window": FROZEN_WINDOW, "detail": detail_df.to_dict(orient="records"), "station_detail": station_df.to_dict(orient="records"), "aggregate": aggregate.to_dict(orient="records"), "station_summary": station_summary.to_dict(orient="records"), "weak_label_paired": contrast_df.to_dict(orient="records"), "limitations": ["自然报警率不是攻击召回率，也不是安全事件真值。", "NOAA温度质量标记只作为弱标签，不能解释因果或真实攻击能力。", "冻结候选的自然投影重估历史尺度，不等于在受控攻击valid上重新训练或调参。", "部分站点-年份没有质量可疑行，因此对应弱标签对比不可估计，保留为空值。"]}
    (TABLES / "stage142_noaa_frozen_natural_operational_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段142 NOAA冻结候选自然观测运行验证", "", "不注入攻击、不使用攻击真值；目标年报警率和温度质量弱标签关联仅用于自然误报与时间稳定性审计。", "", "| 目标年 | 方法 | q | 报警率 | 质量可疑报警率 | 质量正常报警率 |", "|---:|---|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['target_year']} | {row['method']} | {row['q']:.3f} | {row['target_alarm_rate']:.4f} | {row['quality_flagged_alarm_rate']:.4f} ({row['quality_flagged_rows']}) | {row['quality_clean_alarm_rate']:.4f} ({row['quality_clean_rows']}) |")
    lines += ["", "## 弱标签成对差异（q=0.995）", "", "| 比较 | 指标 | 平均差 | 站点bootstrap CI95 | 有效单元格数 |", "|---|---|---:|---:|---:|"]
    for row in contrast_df.to_dict(orient="records"):
        low = "NA" if pd.isna(row["bootstrap_station_ci95_low"]) else f"[{row['bootstrap_station_ci95_low']:.4f}, {row['bootstrap_station_ci95_high']:.4f}]"
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_diff']:.4f} | {low} | {row['cells']} |")
    lines += ["", "自然报警率不等于攻击召回率；质量字段是弱标签；该阶段不支持把任何方法写成真实攻击检测的全面最优。"]
    (REPORTS / "stage142_noaa_frozen_natural_operational_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(detail_df), "station_rows": len(station_df), "contrast_rows": len(contrast_df)}, ensure_ascii=False))
    return 0


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(r["year"]), str(r["station_id"])) for r in audit["detail"] if r.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({s for s in source["station_id"].astype(str).unique() if all((year, s) in eligible for year in years)})
    detail: list[dict[str, object]] = []
    station_detail: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    for target_year in TARGET_YEARS:
        train_years = [year for year in years if year < target_year]
        if not train_years:
            continue
        frame = make_frame(source, target_year, train_years, set(stations))
        scored = score_components(frame)
        base = equal_weight_base_risk(scored)
        structural = scored["structural_risk"].to_numpy(dtype=float)
        risks = {"TrustMeteo-QC default": base, "TrustMeteo-QC frozen expanded": FROZEN_ALPHA * base + (1.0 - FROZEN_ALPHA) * structural, **baseline_risks(scored)}
        test = scored["split"].eq("test").to_numpy()
        quality_bad = pd.to_numeric(scored["temperature_quality"], errors="coerce").fillna(9).ne(1).to_numpy()
        for method in METHODS:
            risk = np.asarray(risks[method], dtype=float)
            train_mask = scored["split"].eq("train").to_numpy() & np.isfinite(risk)
            test_mask = test & np.isfinite(risk)
            for q in Q_GRID:
                threshold = float(np.nanquantile(risk[train_mask], q))
                pred = np.isfinite(risk) & (risk >= threshold)
                bad = test_mask & quality_bad
                clean = test_mask & ~quality_bad
                detail.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "method": method, "q": q, "threshold": threshold, "target_rows": int(test_mask.sum()), "target_mean_risk": float(np.nanmean(risk[test_mask])), "target_p95_risk": float(np.nanquantile(risk[test_mask], 0.95)), "target_alarm_rate": float(pred[test_mask].mean()), "quality_flagged_rows": int(bad.sum()), "quality_flagged_alarm_rate": float(pred[bad].mean()) if bad.any() else float("nan"), "quality_clean_rows": int(clean.sum()), "quality_clean_alarm_rate": float(pred[clean].mean()) if clean.any() else float("nan"), "quality_alarm_contrast": float(pred[bad].mean() - pred[clean].mean()) if bad.any() and clean.any() else float("nan")})
                for station in stations:
                    mask = test_mask & scored["station_key"].astype(str).eq(station).to_numpy()
                    bad_s = mask & quality_bad
                    clean_s = mask & ~quality_bad
                    station_detail.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "method": method, "q": q, "station_key": station, "rows": int(mask.sum()), "alarm_rate": float(pred[mask].mean()) if mask.any() else float("nan"), "quality_flagged_rows": int(bad_s.sum()), "quality_flagged_alarm_rate": float(pred[bad_s].mean()) if bad_s.any() else float("nan"), "quality_clean_rows": int(clean_s.sum()), "quality_clean_alarm_rate": float(pred[clean_s].mean()) if clean_s.any() else float("nan")})
        print(json.dumps({"target_year": target_year, "train_years": train_years, "rows": int(test.sum()), "methods": len(METHODS)}, ensure_ascii=False), flush=True)

    detail_df = pd.DataFrame(detail)
    station_df = pd.DataFrame(station_detail)
    aggregate = detail_df.groupby(["target_year", "method", "q"], sort=False).first().reset_index()
    station_summary = station_df.groupby(["target_year", "method", "q"], sort=False).agg(stations=("station_key", "nunique"), alarm_rate_mean=("alarm_rate", "mean"), alarm_rate_median=("alarm_rate", "median"), alarm_rate_p95=("alarm_rate", lambda x: float(x.quantile(0.95))), quality_contrast_positive_stations=("quality_flagged_alarm_rate", lambda x: int((x > station_df.loc[x.index, "quality_clean_alarm_rate"]).sum()))).reset_index()

    # At q=.995, compare the frozen candidate with the default and SVM across
    # station-year cells; this is a weak-label association audit, not a test of recall.
    q_rows = station_df.loc[station_df["q"].eq(0.995)].copy()
    for other in ["TrustMeteo-QC default", "One-Class SVM"]:
        a = q_rows.loc[q_rows["method"].eq("TrustMeteo-QC frozen expanded")].set_index(["target_year", "station_key"])
        b = q_rows.loc[q_rows["method"].eq(other)].set_index(["target_year", "station_key"])
        common = a.join(b, lsuffix="_expanded", rsuffix="_other", how="inner")
        for metric in ["alarm_rate", "quality_flagged_alarm_rate", "quality_clean_alarm_rate"]:
            diff = common[f"{metric}_expanded"] - common[f"{metric}_other"]
            tmp = pd.DataFrame({"station_key": common.index.get_level_values("station_key"), "value": diff.to_numpy(dtype=float)})
            low, high = station_bootstrap(tmp, "value", stable_seed("stage142", other, metric))
            contrast_rows.append({"comparison": f"expanded - {other}", "metric": metric, "q": 0.995, "mean_diff": float(diff.mean()), "bootstrap_station_ci95_low": low, "bootstrap_station_ci95_high": high, "cells": int(len(diff))})
    contrast_df = pd.DataFrame(contrast_rows)
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020-2025扩展表；11个全年度质量合格站点；历史年份自然观测训练→目标年份自然观测测试；不注入攻击、不使用攻击真值。阶段141冻结alpha=0.25/window=1规则用于无标签自然报警运行投影；每个目标年仅用历史训练正常观测估计尺度，q=0.99/0.995/0.999预先固定。温度质量标记仅作弱标签，不能计算攻击召回。", "stations": stations, "target_years": TARGET_YEARS, "methods": METHODS, "q_grid": Q_GRID, "frozen_alpha": FROZEN_ALPHA, "frozen_window": FROZEN_WINDOW, "detail": detail, "station_detail": station_detail, "aggregate": aggregate.to_dict(orient="records"), "station_summary": station_summary.to_dict(orient="records"), "weak_label_paired": contrast_df.to_dict(orient="records"), "limitations": ["自然报警率不是攻击召回率，也不是安全事件真值。", "NOAA温度质量标记只作为弱标签，不能解释因果或真实攻击能力。", "冻结候选的自然投影重估历史尺度，不等于在受控攻击valid上重新训练或调参。"]}
    detail_df.to_csv(TABLES / "stage142_noaa_frozen_natural_operational_validation_detail.csv", index=False, encoding="utf-8-sig")
    station_df.to_csv(TABLES / "stage142_noaa_frozen_natural_operational_validation_station_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage142_noaa_frozen_natural_operational_validation.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(TABLES / "stage142_noaa_frozen_natural_operational_validation_station_summary.csv", index=False, encoding="utf-8-sig")
    contrast_df.to_csv(TABLES / "stage142_noaa_frozen_natural_operational_validation_weak_label_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage142_noaa_frozen_natural_operational_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段142 NOAA冻结候选自然观测运行验证", "", "不注入攻击、不使用攻击真值；目标年报警率和温度质量弱标签关联仅用于自然误报与时间稳定性审计。", "", "| 目标年 | 方法 | q | 报警率 | 质量可疑报警率 | 质量正常报警率 |", "|---:|---|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['target_year']} | {row['method']} | {row['q']:.3f} | {row['target_alarm_rate']:.4f} | {row['quality_flagged_alarm_rate']:.4f} ({row['quality_flagged_rows']}) | {row['quality_clean_alarm_rate']:.4f} ({row['quality_clean_rows']}) |")
    lines += ["", "## 弱标签成对差异（q=0.995）", "", "| 比较 | 指标 | 平均差 | 站点bootstrap CI95 | 单元格数 |", "|---|---|---:|---:|---:|"]
    for row in contrast_df.to_dict(orient="records"):
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_diff']:.4f} | [{row['bootstrap_station_ci95_low']:.4f}, {row['bootstrap_station_ci95_high']:.4f}] | {row['cells']} |")
    lines += ["", "自然报警率不等于攻击召回率；质量字段是弱标签；该阶段不支持把任何方法写成真实攻击检测的全面最优。"]
    (REPORTS / "stage142_noaa_frozen_natural_operational_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(detail_df), "station_rows": len(station_df), "contrast_rows": len(contrast_df)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(finalize_existing() if "--finalize-existing" in sys.argv else main())
