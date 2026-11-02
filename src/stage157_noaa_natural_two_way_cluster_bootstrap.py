from __future__ import annotations

"""Stage 157: two-way station-by-year bootstrap for natural audits.

This stage does not retrain or reselect anything.  It reuses the station-level
contrasts from stages 155 and 156 and resamples both station and target-year
clusters.  The purpose is uncertainty sensitivity, not a new performance
claim.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
REPEATS = 10000


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def two_way_bootstrap(frame: pd.DataFrame, seed: int) -> tuple[float | None, float | None, int, int]:
    frame = frame.dropna(subset=["diff"]).copy()
    stations = sorted(frame["station_key"].astype(str).unique())
    years = sorted(frame["target_year"].astype(int).unique())
    if len(stations) < 2 or len(years) < 2:
        return None, None, len(stations), len(years)
    matrix = frame.assign(station_key=frame["station_key"].astype(str)).pivot_table(index="station_key", columns="target_year", values="diff", aggfunc="mean").reindex(index=stations, columns=years)
    values = matrix.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    station_idx = rng.integers(0, len(stations), size=(REPEATS, len(stations)))
    year_idx = rng.integers(0, len(years), size=(REPEATS, len(years)))
    sampled = values[station_idx[:, :, None], year_idx[:, None, :]]
    valid = np.isfinite(sampled)
    counts = valid.sum(axis=(1, 2))
    sums = np.where(valid, sampled, 0.0).sum(axis=(1, 2))
    means = np.full(REPEATS, np.nan, dtype=float)
    np.divide(sums, counts, out=means, where=counts > 0)
    means = means[np.isfinite(means)]
    if len(means) == 0:
        return None, None, len(stations), len(years)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), len(stations), len(years)


def station_bootstrap_ci(frame: pd.DataFrame, seed: int) -> tuple[float | None, float | None]:
    grouped = frame.groupby("station_key", sort=True)["diff"].mean().dropna().to_numpy(dtype=float)
    if len(grouped) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = grouped[rng.integers(0, len(grouped), size=(REPEATS, len(grouped)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def process_stage(stage: str, station_path: Path, paired_path: Path, context_columns: list[str]) -> list[dict[str, object]]:
    station = pd.read_csv(station_path)
    paired = pd.read_csv(paired_path)
    expanded = station[station["method"].eq("TrustMeteo-QC frozen expanded")]
    rows: list[dict[str, object]] = []
    for other in ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]:
        baseline = station[station["method"].eq(other)]
        keys = ["target_year", "q", *context_columns, "station_key"]
        left = expanded.set_index(keys)[["alarm_rate"]]
        right = baseline.set_index(keys)[["alarm_rate"]]
        common = left.join(right, lsuffix="_expanded", rsuffix="_other", how="inner").reset_index()
        common["diff"] = common["alarm_rate_expanded"] - common["alarm_rate_other"]
        for context_values, group in common.groupby(["q", *context_columns], sort=True):
            if not isinstance(context_values, tuple):
                context_values = (context_values,)
            q = float(context_values[0])
            context = dict(zip(context_columns, context_values[1:]))
            filter_mask = paired["comparison"].eq(f"expanded - {other}") & paired["q"].eq(q)
            for column, value in context.items():
                filter_mask &= paired[column].eq(value)
            old = paired.loc[filter_mask]
            low, high, station_count, year_count = two_way_bootstrap(group[["target_year", "station_key", "diff"]], stable_seed("stage157", stage, other, q, *context_values))
            station_low, station_high = station_bootstrap_ci(group[["station_key", "diff"]], stable_seed("stage157-station", stage, other, q, *context_values))
            rows.append({
                "source_stage": stage,
                "comparison": f"expanded - {other}",
                "q": q,
                **{key: str(value) for key, value in context.items()},
                "cell_count": int(len(group)),
                "station_count": station_count,
                "year_count": year_count,
                "mean_station_year_diff": float(group["diff"].mean()),
                "two_way_bootstrap_ci95_low": low,
                "two_way_bootstrap_ci95_high": high,
                "station_bootstrap_ci95_low": float(old.iloc[0]["station_bootstrap_ci95_low"]) if not old.empty else station_low,
                "station_bootstrap_ci95_high": float(old.iloc[0]["station_bootstrap_ci95_high"]) if not old.empty else station_high,
                "two_way_ci_width": None if low is None else float(high - low),
            })
    return rows


def main() -> int:
    specs = [
        ("stage155", TABLES / "stage155_noaa_natural_weather_regime_audit_station_detail.csv", TABLES / "stage155_noaa_natural_weather_regime_audit_paired.csv", ["regime_variable", "regime"]),
        ("stage156", TABLES / "stage156_noaa_natural_observation_gap_audit_station_detail.csv", TABLES / "stage156_noaa_natural_observation_gap_audit_paired.csv", ["gap_bucket"]),
    ]
    rows: list[dict[str, object]] = []
    for spec in specs:
        rows.extend(process_stage(*spec))
    if len(rows) != 144:
        raise ValueError(f"unexpected stage157 row count: {len(rows)}")
    frame = pd.DataFrame(rows)
    frame["station_ci_width"] = frame["station_bootstrap_ci95_high"] - frame["station_bootstrap_ci95_low"]
    if frame[["two_way_bootstrap_ci95_low", "two_way_bootstrap_ci95_high", "mean_station_year_diff"]].isna().any().any():
        raise ValueError("stage157 contains non-estimable two-way bootstrap rows")
    prefix = "stage157_noaa_natural_two_way_cluster_bootstrap"
    frame.to_csv(TABLES / f"{prefix}.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段155和156的自然观测站点明细与冻结方法结果；不注入攻击、不重新训练、不重新选择q、不读取攻击标签，NOAA质量字段仅作弱标签。对每个方法比较、固定q和状态/间隔分层同时重采样站点簇与目标年份簇，评估自然报警率差异区间对聚类单位的敏感性；自然报警率不是攻击召回率。",
        "source_stages": ["stage155", "stage156"],
        "comparisons": ["expanded - TrustMeteo-QC default", "expanded - Robust max deviation", "expanded - Isolation Forest", "expanded - One-Class SVM"],
        "repeats": REPEATS,
        "row_count": int(len(frame)),
        "detail": json.loads(frame.to_json(orient="records")),
        "limitations": ["双层bootstrap是当前站点和年份集合的敏感性分析，不创造新的独立站点或年份。", "长间隔状态样本较少，区间应按描述性证据解释。", "不把自然报警率区间解释为攻击性能或真实通信故障因果。"],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summary = frame.groupby(["source_stage", "comparison"], sort=True).agg(rows=("q", "size"), two_way_ci_width_mean=("two_way_ci_width", "mean"), station_ci_width_mean=("station_ci_width", "mean")).reset_index()
    lines = ["# 阶段157 NOAA自然运行双层聚类 bootstrap", "", "复用阶段155/156，不重新训练或调参；同时重采样站点和目标年份，检验自然报警率差异区间的聚类敏感性。", "", "| 来源阶段 | 比较 | 行数 | 双层CI宽度均值 | 站点CI宽度均值 |", "|---|---|---:|---:|---:|"]
    for row in summary.to_dict(orient="records"):
        lines.append(f"| {row['source_stage']} | {row['comparison']} | {int(row['rows'])} | {row['two_way_ci_width_mean']:.6f} | {row['station_ci_width_mean']:.6f} |")
    lines += ["", "双层区间仅用于不确定性敏感性，不代表新增独立样本，也不能转译为攻击召回率。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "row_count": len(frame), "repeats": REPEATS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
