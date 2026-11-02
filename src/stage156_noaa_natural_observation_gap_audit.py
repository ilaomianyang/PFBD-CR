from __future__ import annotations

"""Stage 156: natural NOAA alarm rates after irregular observation gaps.

The NOAA archive is predominantly three-hourly but contains a long tail of
larger inter-observation gaps.  This stage audits frozen detectors by fixed,
pre-declared gap buckets.  Gap buckets are descriptive operational strata,
not tuned hyperparameters, and no target-year labels are used.
"""

import hashlib
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
METHODS = ["TrustMeteo-QC default", "TrustMeteo-QC frozen expanded", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
GAP_LABELS = {"first": "first_observation", "regular": "regular_le_3h", "moderate": "moderate_3_6h", "long": "long_6_24h", "very_long": "very_long_gt_24h"}
FROZEN_ALPHA = 0.25

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage142_noaa_frozen_natural_operational_validation import (  # noqa: E402
    baseline_risks,
    equal_weight_base_risk,
    make_frame,
    score_components,
)


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def gap_bucket(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values(["station_key", "timestamp_utc"], kind="stable")
    gap = ordered.groupby("station_key", sort=False)["timestamp_utc"].diff().dt.total_seconds().div(3600.0)
    labels = pd.Series("first_observation", index=ordered.index, dtype="object")
    finite = gap.notna()
    numeric = gap.to_numpy(dtype=float)
    labels.loc[finite & (numeric <= 3.0)] = GAP_LABELS["regular"]
    labels.loc[finite & (numeric > 3.0) & (numeric <= 6.0)] = GAP_LABELS["moderate"]
    labels.loc[finite & (numeric > 6.0) & (numeric <= 24.0)] = GAP_LABELS["long"]
    labels.loc[finite & (numeric > 24.0)] = GAP_LABELS["very_long"]
    return labels.reindex(frame.index)


def station_bootstrap(values: pd.DataFrame, seed: int, repeats: int = 10000) -> tuple[float | None, float | None, int]:
    grouped = values.groupby("station_key", sort=True)["diff"].mean().dropna()
    if len(grouped) < 2:
        return None, None, int(len(grouped))
    rng = np.random.default_rng(seed)
    sampled = grouped.to_numpy(dtype=float)[rng.integers(0, len(grouped), size=(repeats, len(grouped)))]
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), int(len(grouped))


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(value) for value in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    detail_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for target_year in TARGET_YEARS:
        train_years = [year for year in years if year < target_year]
        frame = make_frame(source, target_year, train_years, set(stations))
        frame["gap_bucket"] = gap_bucket(frame)
        scored = score_components(frame)
        train = scored["split"].eq("train") & ~scored["missing_observation"]
        base = equal_weight_base_risk(scored)
        structural = scored["structural_risk"].to_numpy(dtype=float)
        risks = {"TrustMeteo-QC default": base, "TrustMeteo-QC frozen expanded": FROZEN_ALPHA * base + (1 - FROZEN_ALPHA) * structural, **baseline_risks(scored)}
        train_array = train.to_numpy()
        test_array = scored["split"].eq("test").to_numpy()
        for method in METHODS:
            risk = np.asarray(risks[method], dtype=float)
            for q in Q_GRID:
                threshold = float(np.nanquantile(risk[train_array & np.isfinite(risk)], q))
                pred = np.isfinite(risk) & (risk >= threshold)
                for bucket in GAP_LABELS.values():
                    mask = test_array & scored["gap_bucket"].eq(bucket).to_numpy()
                    if not mask.any():
                        continue
                    detail_rows.append({"target_year": target_year, "train_years": "-".join(map(str, train_years)), "method": method, "q": q, "gap_bucket": bucket, "threshold": threshold, "rows": int(mask.sum()), "alarm_rate": float(pred[mask].mean())})
                    for station in stations:
                        station_mask = mask & scored["station_key"].astype(str).eq(station).to_numpy()
                        if station_mask.any():
                            station_rows.append({"target_year": target_year, "method": method, "q": q, "gap_bucket": bucket, "station_key": station, "rows": int(station_mask.sum()), "alarm_rate": float(pred[station_mask].mean())})
        print(json.dumps({"target_year": target_year, "rows": int(test_array.sum()), "detail_rows": len(detail_rows)}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(detail_rows)
    station = pd.DataFrame(station_rows)
    if detail.empty or station.empty:
        raise ValueError("stage156 produced no gap rows")
    paired_rows: list[dict[str, object]] = []
    expanded = station[station["method"].eq("TrustMeteo-QC frozen expanded")].set_index(["target_year", "q", "gap_bucket", "station_key"])
    for other in ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]:
        baseline = station[station["method"].eq(other)].set_index(["target_year", "q", "gap_bucket", "station_key"])
        common = expanded[["alarm_rate"]].join(baseline[["alarm_rate"]], lsuffix="_expanded", rsuffix="_other", how="inner").reset_index()
        common["diff"] = common["alarm_rate_expanded"] - common["alarm_rate_other"]
        for (q, bucket), group in common.groupby(["q", "gap_bucket"], sort=True):
            low, high, count = station_bootstrap(group[["station_key", "diff"]], stable_seed("stage156", other, q, bucket))
            paired_rows.append({"comparison": f"expanded - {other}", "q": float(q), "gap_bucket": bucket, "cell_count": int(len(group)), "station_count": count, "mean_station_diff": float(group["diff"].mean()), "station_bootstrap_ci95_low": low, "station_bootstrap_ci95_high": high, "positive_cells": int((group["diff"] > 0).sum()), "negative_cells": int((group["diff"] < 0).sum())})

    prefix = "stage156_noaa_natural_observation_gap_audit"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    station.to_csv(TABLES / f"{prefix}_station_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired_rows).to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段142冻结的NOAA自然观测运行设置；按站点和时间戳计算固定观测间隔桶（首条观测、≤3小时、3–6小时、6–24小时、>24小时），不使用目标年攻击标签、不根据目标年报警结果调参，NOAA质量字段仅作弱标签且不参与选择。每个目标年只用历史训练年估计风险分位数阈值，目标年只评价自然报警率；不注入攻击，结果用于审计观测间隔变化下的运行安全，不能解释为攻击召回率。",
        "source_data": NOAA.name,
        "target_years": TARGET_YEARS,
        "stations": stations,
        "methods": METHODS,
        "q_grid": Q_GRID,
        "gap_buckets": list(GAP_LABELS.values()),
        "detail_rows": int(len(detail)),
        "station_detail_rows": int(len(station)),
        "paired": paired_rows,
        "limitations": ["自然报警率不是攻击召回率；本阶段不注入攻击。", "间隔桶是运行审计分层，不是调参网格，也不证明通信中断的因果影响。", "11个全年度合格站点仍不足以证明跨地区长期普适性。"],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    aggregate = detail.groupby(["gap_bucket", "method", "q"], sort=True).agg(cells=("target_year", "size"), alarm_rate_mean=("alarm_rate", "mean"), alarm_rate_max=("alarm_rate", "max")).reset_index()
    lines = ["# 阶段156 NOAA自然观测观测间隔分层审计", "", "固定按观测间隔分层；目标年只评价自然报警率，不注入攻击或用标签调参。", "", "| 间隔桶 | 方法 | q | 单元数 | 报警率均值 |", "|---|---|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['gap_bucket']} | {row['method']} | {row['q']:.3f} | {int(row['cells'])} | {row['alarm_rate_mean']:.6f} |")
    lines += ["", "| 比较 | 间隔桶 | q | 平均站点差异 | CI95 | 正向单元 | 负向单元 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in paired_rows:
        ci = "NA" if row["station_bootstrap_ci95_low"] is None else f"[{row['station_bootstrap_ci95_low']:+.6f},{row['station_bootstrap_ci95_high']:+.6f}]"
        lines.append(f"| {row['comparison']} | {row['gap_bucket']} | {row['q']:.3f} | {row['mean_station_diff']:+.6f} | {ci} | {row['positive_cells']} | {row['negative_cells']} |")
    lines += ["", "观测间隔分层只描述自然运行安全，不能转译为攻击检测召回率。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "station_detail_rows": len(station), "paired_rows": len(paired_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
