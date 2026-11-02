from __future__ import annotations

"""Stage 154: temporal drift of natural NOAA alarm rates."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = TABLES / "stage142_noaa_frozen_natural_operational_validation_station_detail.csv"
META = TABLES / "stage142_noaa_frozen_natural_operational_validation.json"
METHODS = ["TrustMeteo-QC default", "TrustMeteo-QC frozen expanded", "Robust max deviation", "Isolation Forest", "One-Class SVM"]


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def slope(years: np.ndarray, values: np.ndarray) -> float | None:
    mask = np.isfinite(years) & np.isfinite(values)
    if int(mask.sum()) < 2 or np.ptp(years[mask]) == 0:
        return None
    return float(np.polyfit(years[mask], values[mask], 1)[0])


def station_slope_bootstrap(frame: pd.DataFrame, metric: str, repeats: int, seed: int) -> tuple[float | None, float | None, int]:
    pivot = frame.pivot_table(index="station_key", columns="target_year", values=metric, aggfunc="mean").sort_index().sort_index(axis=1)
    pivot = pivot.dropna(axis=0, how="all")
    if len(pivot) < 2 or pivot.isna().any().any():
        return None, None, len(pivot)
    stations = pivot.index.astype(str).tolist()
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(stations), size=(repeats, len(stations)))
    yearly_means = pivot.to_numpy(dtype=float)[sampled].mean(axis=1)
    slopes = np.polyfit(pivot.columns.to_numpy(dtype=float), yearly_means.T, 1)[0]
    return float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975)), len(stations)


def station_value_bootstrap(frame: pd.DataFrame, metric: str, repeats: int, seed: int) -> tuple[float | None, float | None, int]:
    values = frame.groupby("station_key", sort=True)[metric].mean().dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return None, None, len(values)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(repeats, len(values)))]
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), len(values)


def main() -> int:
    detail = pd.read_csv(SOURCE)
    meta = json.loads(META.read_text(encoding="utf-8"))
    detail["station_key"] = detail["station_key"].astype(str)
    detail["q"] = detail["q"].astype(float)
    if sorted(detail["method"].unique().tolist()) != sorted(METHODS):
        raise ValueError("stage142 methods are incomplete")
    years = sorted(int(x) for x in detail["target_year"].unique())
    rows = []
    for method in METHODS:
        for q in sorted(detail["q"].unique()):
            part = detail[(detail["method"].eq(method)) & (detail["q"].eq(q))]
            yearly = part.groupby("target_year", sort=True)[["alarm_rate", "quality_clean_alarm_rate"]].mean().reset_index()
            alarm_slope = slope(yearly["target_year"].to_numpy(dtype=float), yearly["alarm_rate"].to_numpy(dtype=float))
            clean_slope = slope(yearly["target_year"].to_numpy(dtype=float), yearly["quality_clean_alarm_rate"].to_numpy(dtype=float))
            low, high, station_count = station_slope_bootstrap(part, "alarm_rate", 20000, stable_seed("stage154", method, q, "alarm"))
            clean_low, clean_high, _ = station_slope_bootstrap(part, "quality_clean_alarm_rate", 20000, stable_seed("stage154", method, q, "clean"))
            rows.append({"method": method, "q": float(q), "year_count": len(yearly), "station_count": station_count, "alarm_rate_year_slope": alarm_slope, "alarm_rate_slope_ci95_low": low, "alarm_rate_slope_ci95_high": high, "clean_alarm_rate_year_slope": clean_slope, "clean_alarm_rate_slope_ci95_low": clean_low, "clean_alarm_rate_slope_ci95_high": clean_high, "alarm_rate_first_year": float(yearly.iloc[0]["alarm_rate"]), "alarm_rate_last_year": float(yearly.iloc[-1]["alarm_rate"])})

    paired = []
    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    for other in ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]:
        baseline = detail[detail["method"].eq(other)]
        for metric in ["alarm_rate", "quality_clean_alarm_rate"]:
            slopes = []
            for q in sorted(detail["q"].unique()):
                for station in sorted(detail["station_key"].unique()):
                    left = expanded[(expanded["q"].eq(q)) & (expanded["station_key"].eq(station))].set_index("target_year")[metric]
                    right = baseline[(baseline["q"].eq(q)) & (baseline["station_key"].eq(station))].set_index("target_year")[metric]
                    common = left.index.intersection(right.index)
                    if len(common) >= 2:
                        slopes.append({"q": float(q), "station_key": station, "diff_slope": slope(common.to_numpy(dtype=float), (left.loc[common] - right.loc[common]).to_numpy(dtype=float))})
            sf = pd.DataFrame(slopes).dropna(subset=["diff_slope"])
            low, high, station_count = station_value_bootstrap(sf, "diff_slope", 20000, stable_seed("stage154-paired", other, metric)) if not sf.empty else (None, None, 0)
            mean_diff = float(sf["diff_slope"].mean()) if not sf.empty else None
            paired.append({"comparison": f"expanded - {other}", "metric": metric, "q_count": int(sf["q"].nunique()) if not sf.empty else 0, "station_q_count": int(len(sf)), "mean_station_q_slope_difference": mean_diff, "bootstrap_ci95_low": low, "bootstrap_ci95_high": high, "positive_cells": int((sf["diff_slope"] > 0).sum()) if not sf.empty else 0, "negative_cells": int((sf["diff_slope"] < 0).sum()) if not sf.empty else 0})

    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "复用阶段142冻结的NOAA 2021–2025自然观测站点明细；不注入攻击、不使用目标年攻击标签、不根据目标年标签调参，NOAA质量字段仅作弱标签。按5个目标年份、11个站点和3个预先固定q运行点估计自然报警率年度斜率，并对站点重采样描述斜率不确定性；不把自然报警率解释为攻击召回。", "source_artifact": SOURCE.name, "methods": METHODS, "target_years": meta["target_years"], "q_grid": meta["q_grid"], "stations": sorted(detail["station_key"].unique().tolist()), "summary": rows, "paired": paired, "limitations": ["仅5个目标年份，年度斜率用于诊断时间漂移，不证明长期稳定性。", "自然报警率不是攻击召回率；质量字段仅作弱标签。", "q和候选设置沿用阶段142冻结结果，本阶段不重新选择运行点。"]}
    prefix = "stage154_noaa_natural_temporal_drift"
    pd.DataFrame(rows).to_csv(TABLES / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired).to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段154 NOAA自然报警率跨年度漂移", "", "复用阶段142冻结自然观测结果；不注入攻击、不使用目标年攻击标签、不根据目标年调参。", "", "| 方法 | q | 年度报警率斜率 | CI95 | 首年报警率 | 末年报警率 |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        slope_text = "NA" if row["alarm_rate_year_slope"] is None else f"{row['alarm_rate_year_slope']:+.8f}"
        ci_text = "NA" if row["alarm_rate_slope_ci95_low"] is None else f"[{row['alarm_rate_slope_ci95_low']:+.8f},{row['alarm_rate_slope_ci95_high']:+.8f}]"
        lines.append(f"| {row['method']} | {row['q']:.3f} | {slope_text} | {ci_text} | {row['alarm_rate_first_year']:.6f} | {row['alarm_rate_last_year']:.6f} |")
    lines += ["", "自然报警率的年度斜率只用于运行漂移诊断，不能转译为攻击检测召回或真实攻击能力。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary_rows": len(rows), "paired_rows": len(paired)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
