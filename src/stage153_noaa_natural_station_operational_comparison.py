from __future__ import annotations

"""Stage 153: natural NOAA station-level operational comparison.

This stage reuses the frozen, label-free stage-142 natural-observation output.
It compares alarm-rate stability across methods and stations without treating
NOAA quality flags as attack truth and without selecting q from target-year
labels.
"""

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


def station_bootstrap(values: pd.DataFrame, metric: str, repeats: int = 20000, seed: int = 153) -> tuple[float | None, float | None]:
    grouped = values.groupby("station_key", sort=True)[metric].mean().dropna()
    if grouped.empty:
        return None, None
    stations = grouped.index.astype(str).tolist()
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(stations), size=(repeats, len(stations)))
    matrix = grouped.to_numpy(dtype=float)[sample_idx]
    means = matrix.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def finite_stats(values: pd.Series) -> tuple[float | None, int, int]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None, 0, 0
    return float(clean.mean()), int((clean > 0).sum()), int((clean < 0).sum())


def main() -> int:
    station = pd.read_csv(SOURCE)
    meta = json.loads(META.read_text(encoding="utf-8"))
    station["station_key"] = station["station_key"].astype(str)
    station["q"] = station["q"].astype(float)
    required = {"target_year", "q", "method", "station_key", "alarm_rate", "quality_flagged_alarm_rate", "quality_clean_alarm_rate"}
    missing = sorted(required.difference(station.columns))
    if missing:
        raise ValueError(f"missing required natural station columns: {missing}")
    if sorted(station["method"].unique().tolist()) != sorted(METHODS):
        raise ValueError("stage142 natural station methods are incomplete")
    if len(station) != 5 * 3 * 5 * 11:
        raise ValueError(f"unexpected station row count: {len(station)}")

    detail = station.copy()
    # Per-method operational summaries retain every target-year × q cell.
    summary = detail.groupby(["method", "target_year", "q"], sort=True).agg(
        stations=("station_key", "nunique"),
        alarm_rate_mean=("alarm_rate", "mean"),
        alarm_rate_median=("alarm_rate", "median"),
        alarm_rate_p95=("alarm_rate", lambda x: float(np.quantile(x, 0.95))),
        alarm_rate_max=("alarm_rate", "max"),
        clean_alarm_rate_mean=("quality_clean_alarm_rate", "mean"),
        flagged_alarm_rate_mean=("quality_flagged_alarm_rate", "mean"),
    ).reset_index()
    summary["alarm_rate_iqr"] = detail.groupby(["method", "target_year", "q"], sort=True)["alarm_rate"].quantile(0.75).to_numpy() - detail.groupby(["method", "target_year", "q"], sort=True)["alarm_rate"].quantile(0.25).to_numpy()

    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    paired_rows: list[dict[str, object]] = []
    for other in ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]:
        baseline = detail[detail["method"].eq(other)]
        keys = ["target_year", "q", "station_key"]
        left = expanded.set_index(keys)
        right = baseline.set_index(keys)
        common = left.join(right, lsuffix="_expanded", rsuffix="_other", how="inner")
        for metric in ["alarm_rate", "quality_clean_alarm_rate", "quality_flagged_alarm_rate"]:
            diff = (common[f"{metric}_expanded"] - common[f"{metric}_other"]).rename("diff").reset_index()
            cell_rows = []
            for (year, q), group in diff.groupby(["target_year", "q"], sort=True):
                low, high = station_bootstrap(group, "diff", seed=stable_seed("stage153", other, metric, year, q))
                mean_diff, positive, negative = finite_stats(group["diff"])
                cell_rows.append({"target_year": int(year), "q": float(q), "station_count": int(group["diff"].notna().sum()), "mean_station_diff": mean_diff, "station_bootstrap_ci95_low": low, "station_bootstrap_ci95_high": high, "stations_positive": positive, "stations_negative": negative})
            overall = diff.groupby("station_key", sort=True)["diff"].mean().rename("diff").reset_index()
            low, high = station_bootstrap(overall, "diff", seed=stable_seed("stage153-overall", other, metric))
            overall_mean, overall_positive, overall_negative = finite_stats(overall["diff"])
            paired_rows.append({"comparison": f"expanded - {other}", "metric": metric, "cell_count": len(cell_rows), "cell_detail": cell_rows, "overall_mean_station_diff": overall_mean, "overall_station_bootstrap_ci95_low": low, "overall_station_bootstrap_ci95_high": high, "overall_stations_positive": overall_positive, "overall_stations_negative": overall_negative})

    contrast = detail.copy()
    contrast["quality_contrast"] = contrast["quality_flagged_alarm_rate"] - contrast["quality_clean_alarm_rate"]
    contrast_summary = contrast.groupby(["method", "target_year", "q"], sort=True).agg(
        contrast_mean=("quality_contrast", "mean"),
        contrast_median=("quality_contrast", "median"),
        contrast_positive_stations=("quality_contrast", lambda x: int((x > 0).sum())),
        contrast_valid_stations=("quality_contrast", lambda x: int(np.isfinite(x).sum())),
    ).reset_index()
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段142冻结的NOAA 2021–2025自然观测站点明细；不注入攻击、不使用目标年攻击标签、不根据目标年标签选择q或方法。按11站点和5个目标年份×3个预先固定q运行点比较自然报警率、站点离散度及质量字段弱标签关联；站点bootstrap只用于不确定性描述。",
        "source_artifact": SOURCE.name,
        "methods": METHODS,
        "target_years": meta["target_years"],
        "q_grid": meta["q_grid"],
        "stations": sorted(detail["station_key"].unique().tolist()),
        "detail_rows": int(len(detail)),
        "summary": summary.to_dict(orient="records"),
        "contrast_summary": contrast_summary.to_dict(orient="records"),
        "paired": paired_rows,
        "limitations": ["自然报警率不是攻击召回率；NOAA质量标记只是弱标签。", "站点数量为11个，bootstrap用于描述当前站点集合的不确定性，不能替代独立地区验证。", "阶段142已冻结q和候选设置，本阶段不根据自然观测结果调参。"],
    }
    prefix = "stage153_noaa_natural_station_operational_comparison"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    contrast_summary.to_csv(TABLES / f"{prefix}_contrast_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in row.items() if k != "cell_detail"} for row in paired_rows]).to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段153 NOAA自然观测站点级运行比较", "", "复用阶段142冻结的自然观测结果；不注入攻击、不使用目标年攻击标签、不根据目标年标签调参。", "", "| 方法 | 目标年 | q | 报警率均值 | 报警率中位数 | 报警率P95 | 正常质量记录报警率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {int(row['target_year'])} | {row['q']:.3f} | {row['alarm_rate_mean']:.6f} | {row['alarm_rate_median']:.6f} | {row['alarm_rate_p95']:.6f} | {row['clean_alarm_rate_mean']:.6f} |")
    lines += ["", "## 扩展候选相对基线的站点簇差异", "", "| 比较 | 指标 | 总体站点差异 | CI95 | 正向站点 | 负向站点 |", "|---|---|---:|---:|---:|---:|"]
    for row in paired_rows:
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['overall_mean_station_diff']:+.6f} | [{row['overall_station_bootstrap_ci95_low']:+.6f},{row['overall_station_bootstrap_ci95_high']:+.6f}] | {row['overall_stations_positive']} | {row['overall_stations_negative']} |")
    lines += ["", "质量字段关联仅作弱标签描述，不能解释为攻击检测真值；自然报警率也不能解释为攻击召回。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "summary_rows": len(summary), "paired": len(paired_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
