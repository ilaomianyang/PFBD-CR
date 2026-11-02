from __future__ import annotations

"""Stage 155: weather-regime stratified natural NOAA alarm audit.

This stage reuses the frozen stage-142 protocol and asks a narrower operational
question: are natural alarms concentrated in particular temperature or wind
regimes?  Regime boundaries and detector thresholds are estimated from prior
training years only.  Target-year NOAA quality fields are retained only for
descriptive counts and are never used for selection or attack evaluation.
"""

import hashlib
import json
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
METHODS = [
    "TrustMeteo-QC default",
    "TrustMeteo-QC frozen expanded",
    "Robust max deviation",
    "Isolation Forest",
    "One-Class SVM",
]
REGIME_SPECS = {
    "temperature": ("temperature_c", {"low": "cold", "mid": "mild", "high": "hot"}),
    "wind_speed": ("wind_speed_ms", {"low": "calm", "mid": "normal", "high": "strong"}),
}
FROZEN_ALPHA = 0.25
STAGE142_SOURCE = TABLES / "stage142_noaa_frozen_natural_operational_validation.json"

if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from stage142_noaa_frozen_natural_operational_validation import (  # noqa: E402
    baseline_risks,
    equal_weight_base_risk,
    make_frame,
    score_components,
    stable_seed as stage142_seed,
)


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def local_regime(frame: pd.DataFrame, column: str, train_mask: pd.Series, labels: dict[str, str]) -> pd.Series:
    """Assign target rows to station-local train-derived low/mid/high bins."""
    values = pd.to_numeric(frame[column], errors="coerce")
    train_values = values.loc[train_mask]
    global_values = train_values[np.isfinite(train_values)]
    global_q = np.quantile(global_values, [1 / 3, 2 / 3]) if len(global_values) >= 2 else np.array([0.0, 0.0])
    output = pd.Series("missing", index=frame.index, dtype="object")
    for station, idx in frame.groupby("station_key", sort=False).groups.items():
        train_station = train_mask.loc[idx]
        local = values.loc[idx[train_station.to_numpy()]]
        local = local[np.isfinite(local)]
        cuts = np.quantile(local, [1 / 3, 2 / 3]) if len(local) >= 20 else global_q
        current = values.loc[idx]
        finite = np.isfinite(current.to_numpy(dtype=float))
        codes = np.digitize(current.to_numpy(dtype=float), cuts, right=False)
        assigned = np.full(len(idx), "missing", dtype=object)
        assigned[finite & (codes == 0)] = labels["low"]
        assigned[finite & (codes == 1)] = labels["mid"]
        assigned[finite & (codes >= 2)] = labels["high"]
        output.loc[idx] = assigned
    return output


def station_bootstrap(values: pd.DataFrame, metric: str, seed: int, repeats: int = 10000) -> tuple[float | None, float | None, int]:
    grouped = values.groupby("station_key", sort=True)[metric].mean().dropna()
    if len(grouped) < 2:
        return None, None, int(len(grouped))
    rng = np.random.default_rng(seed)
    sampled = grouped.to_numpy(dtype=float)[rng.integers(0, len(grouped), size=(repeats, len(grouped)))]
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), int(len(grouped))


def finite_mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if not clean.empty else None


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(value) for value in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    if not stations:
        raise ValueError("no all-year eligible NOAA stations")

    detail_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for target_year in TARGET_YEARS:
        train_years = [year for year in years if year < target_year]
        if not train_years:
            continue
        frame = make_frame(source, target_year, train_years, set(stations))
        scored = score_components(frame)
        train_mask = scored["split"].eq("train") & ~scored["missing_observation"]
        regimes: dict[str, pd.Series] = {}
        for regime_name, (column, labels) in REGIME_SPECS.items():
            regimes[regime_name] = local_regime(scored, column, scored["split"].eq("train"), labels)
            scored[f"{regime_name}_regime"] = regimes[regime_name]

        base = equal_weight_base_risk(scored)
        structural = scored["structural_risk"].to_numpy(dtype=float)
        risks = {
            "TrustMeteo-QC default": base,
            "TrustMeteo-QC frozen expanded": FROZEN_ALPHA * base + (1.0 - FROZEN_ALPHA) * structural,
            **baseline_risks(scored),
        }
        train_array = train_mask.to_numpy()
        test_array = scored["split"].eq("test").to_numpy()
        quality_bad = pd.to_numeric(scored["temperature_quality"], errors="coerce").fillna(9).ne(1).to_numpy()
        for method in METHODS:
            risk = np.asarray(risks[method], dtype=float)
            for q in Q_GRID:
                threshold = float(np.nanquantile(risk[train_array & np.isfinite(risk)], q))
                pred = np.isfinite(risk) & (risk >= threshold)
                for regime_name in REGIME_SPECS:
                    regime_values = scored[f"{regime_name}_regime"]
                    for regime in sorted(regime_values.dropna().unique().tolist()):
                        mask = test_array & regime_values.eq(regime).to_numpy()
                        bad = mask & quality_bad
                        clean = mask & ~quality_bad
                        if not mask.any():
                            continue
                        detail_rows.append({
                            "target_year": int(target_year),
                            "train_years": "-".join(map(str, train_years)),
                            "method": method,
                            "q": float(q),
                            "regime_variable": regime_name,
                            "regime": str(regime),
                            "threshold": threshold,
                            "rows": int(mask.sum()),
                            "alarm_rate": float(pred[mask].mean()),
                            "quality_flagged_rows": int(bad.sum()),
                            "quality_clean_rows": int(clean.sum()),
                            "quality_flagged_alarm_rate": float(pred[bad].mean()) if bad.any() else None,
                            "quality_clean_alarm_rate": float(pred[clean].mean()) if clean.any() else None,
                        })
                        for station in stations:
                            station_mask = mask & scored["station_key"].astype(str).eq(station).to_numpy()
                            if not station_mask.any():
                                continue
                            station_bad = station_mask & quality_bad
                            station_clean = station_mask & ~quality_bad
                            station_rows.append({
                                "target_year": int(target_year),
                                "method": method,
                                "q": float(q),
                                "regime_variable": regime_name,
                                "regime": str(regime),
                                "station_key": str(station),
                                "rows": int(station_mask.sum()),
                                "alarm_rate": float(pred[station_mask].mean()),
                                "quality_flagged_rows": int(station_bad.sum()),
                                "quality_clean_rows": int(station_clean.sum()),
                                "quality_flagged_alarm_rate": float(pred[station_bad].mean()) if station_bad.any() else None,
                                "quality_clean_alarm_rate": float(pred[station_clean].mean()) if station_clean.any() else None,
                            })
        print(json.dumps({"target_year": target_year, "train_years": train_years, "rows": int(test_array.sum()), "detail_rows": len(detail_rows)}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(detail_rows)
    station = pd.DataFrame(station_rows)
    if detail.empty or station.empty:
        raise ValueError("stage155 produced no regime rows")

    paired_rows: list[dict[str, object]] = []
    expanded = station[station["method"].eq("TrustMeteo-QC frozen expanded")]
    for other in ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]:
        baseline = station[station["method"].eq(other)]
        keys = ["target_year", "q", "regime_variable", "regime", "station_key"]
        left = expanded.set_index(keys)[["alarm_rate"]]
        right = baseline.set_index(keys)[["alarm_rate"]]
        common = left.join(right, lsuffix="_expanded", rsuffix="_other", how="inner").reset_index()
        common["diff"] = common["alarm_rate_expanded"] - common["alarm_rate_other"]
        for (q, variable, regime), group in common.groupby(["q", "regime_variable", "regime"], sort=True):
            low, high, station_count = station_bootstrap(group[["station_key", "diff"]], "diff", stable_seed("stage155", other, q, variable, regime))
            paired_rows.append({
                "comparison": f"expanded - {other}",
                "q": float(q),
                "regime_variable": str(variable),
                "regime": str(regime),
                "cell_count": int(len(group)),
                "station_count": station_count,
                "mean_station_diff": finite_mean(group["diff"]),
                "station_bootstrap_ci95_low": low,
                "station_bootstrap_ci95_high": high,
                "positive_cells": int((group["diff"] > 0).sum()),
                "negative_cells": int((group["diff"] < 0).sum()),
            })

    prefix = "stage155_noaa_natural_weather_regime_audit"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    station.to_csv(TABLES / f"{prefix}_station_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired_rows).to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段142冻结的NOAA自然观测运行设置；每个目标年只用历史训练年估计站点本地温度和风速三分位状态边界，并用训练正常风险分位数确定q=0.99/0.995/0.999阈值；目标年只评价自然报警率，不注入攻击、不使用目标年攻击标签、不根据目标年结果调参，NOAA质量字段仅作弱标签。状态分层用于运行安全审计，不能解释为攻击召回率。",
        "source_artifact": STAGE142_SOURCE.name,
        "source_data": NOAA.name,
        "target_years": TARGET_YEARS,
        "train_year_policy": "all prior years",
        "stations": stations,
        "methods": METHODS,
        "q_grid": Q_GRID,
        "regime_variables": {name: {"column": spec[0], "labels": spec[1]} for name, spec in REGIME_SPECS.items()},
        "detail_rows": int(len(detail)),
        "station_detail_rows": int(len(station)),
        "paired": paired_rows,
        "limitations": [
            "自然报警率不是攻击召回率；NOAA质量字段只作弱标签描述。",
            "状态边界是站点本地训练年三分位，不能解释为物理异常阈值或因果关系。",
            "11个全年度合格站点和5个目标年份仍不足以证明跨地区长期普适性。",
            "本阶段只审计冻结设置，不根据状态分层结果回写主模型或阈值。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# 阶段155 NOAA自然观测气象状态分层审计",
        "",
        "本阶段复用冻结自然观测协议；状态边界和阈值只由历史训练年确定，目标年不注入攻击、不调参。",
        "",
        "| 状态变量 | 状态 | 方法 | q | 目标年单元数 | 报警率均值 |",
        "|---|---|---|---:|---:|---:|",
    ]
    aggregate = detail.groupby(["regime_variable", "regime", "method", "q"], sort=True).agg(cells=("target_year", "size"), alarm_rate_mean=("alarm_rate", "mean"), alarm_rate_median=("alarm_rate", "median"), alarm_rate_max=("alarm_rate", "max")).reset_index()
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['regime_variable']} | {row['regime']} | {row['method']} | {row['q']:.3f} | {int(row['cells'])} | {row['alarm_rate_mean']:.6f} |")
    lines += ["", "## 扩展候选相对基线的站点—状态差异", "", "| 比较 | 状态变量 | 状态 | q | 平均站点差异 | CI95 | 正向单元 | 负向单元 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in paired_rows:
        low = "NA" if row["station_bootstrap_ci95_low"] is None else f"[{row['station_bootstrap_ci95_low']:+.6f},{row['station_bootstrap_ci95_high']:+.6f}]"
        lines.append(f"| {row['comparison']} | {row['regime_variable']} | {row['regime']} | {row['q']:.3f} | {row['mean_station_diff']:+.6f} | {low} | {row['positive_cells']} | {row['negative_cells']} |")
    lines += ["", "状态分层只描述自然运行报警率差异，不能转译为攻击召回率或真实攻击识别能力。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "station_detail_rows": len(station), "paired_rows": len(paired_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
