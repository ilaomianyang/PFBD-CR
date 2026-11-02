from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage70_noaa_isd_multiyear_normalized.parquet"
AUDIT = TABLES / "stage71_noaa_multiyear_quality_audit.json"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
YEARS = [2021, 2022, 2023]
DETECTOR_COLS = [
    "Physical Rules",
    "Hampel MAD",
    "Kalman Innovation",
    "CUSUM",
    "Spatial Consistency",
    "structural_risk",
    "robust_fused_risk",
]
VALUE_COLS = ["original_value", *DETECTOR_COLS]
EPS = 1e-12

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, structural_features, structural_risk_from_norm  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage73_noaa_multiyear_controlled_benchmark import build_dataset  # noqa: E402


def add_structural(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    train = out["split"].eq("train")
    normal_train = train & (~out["combined_is_attack"].astype(bool))
    parts = []
    for split_name in ["train", "valid", "test"]:
        mask = out["split"].eq(split_name)
        part = structural_features(out.loc[mask], STRUCTURAL_BASE).set_axis(out.index[mask])
        parts.append(part)
    features = pd.concat(parts).reindex(out.index).fillna(0.0)
    features["station_key"] = out["station_key"].astype(str)
    scaler = AdaptiveScaler(STRUCTURAL_BASE).fit(features.loc[train], normal_train.loc[train])
    out["structural_risk"] = structural_risk_from_norm(scaler.transform(features), enhanced=False)
    return out


def finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def quantile_edges(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, qs)
    edges = np.unique(edges)
    if len(edges) < 2:
        center = float(reference[0]) if len(reference) else 0.0
        edges = np.array([center - 0.5, center + 0.5], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    if len(reference) < 2 or len(current) < 2:
        return float("nan")
    edges = quantile_edges(reference, bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_p = (ref_counts.astype(float) + 0.5) / (ref_counts.sum() + 0.5 * len(ref_counts))
    cur_p = (cur_counts.astype(float) + 0.5) / (cur_counts.sum() + 0.5 * len(cur_counts))
    return float(np.sum((cur_p - ref_p) * np.log((cur_p + EPS) / (ref_p + EPS))))


def robust_shift(reference: np.ndarray, current: np.ndarray) -> tuple[float, float, float, float]:
    ref_median = float(np.median(reference))
    cur_median = float(np.median(current))
    ref_mad = float(np.median(np.abs(reference - ref_median)))
    cur_mad = float(np.median(np.abs(current - cur_median)))
    scale = robust_scale(reference)
    return cur_median - ref_median, (cur_median - ref_median) / scale, cur_mad - ref_mad, cur_mad / max(ref_mad, EPS)


def robust_scale(values: np.ndarray) -> float:
    """Return a non-degenerate robust scale for continuous and discrete scores."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > EPS:
        return 1.4826 * mad
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr > EPS:
        return iqr / 1.3489795
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return max(std, 1.0)


def summarize_pair(reference: pd.DataFrame, current: pd.DataFrame, feature: str, station: str, ref_year: int, cur_year: int) -> dict[str, object]:
    ref = finite_values(reference, feature)
    cur = finite_values(current, feature)
    if len(ref) < 20 or len(cur) < 20:
        return {"station_key": station, "reference_year": ref_year, "current_year": cur_year, "feature": feature, "reference_n": int(len(ref)), "current_n": int(len(cur)), "psi": float("nan"), "wasserstein": float("nan"), "wasserstein_over_ref_mad": float("nan"), "median_shift": float("nan"), "median_shift_robust": float("nan"), "mad_shift": float("nan"), "mad_ratio": float("nan"), "status": "insufficient_n"}
    ref_median = float(np.median(ref))
    ref_mad = float(np.median(np.abs(ref - ref_median)))
    ref_scale = robust_scale(ref)
    median_shift, median_shift_robust, mad_shift, mad_ratio = robust_shift(ref, cur)
    return {"station_key": station, "reference_year": ref_year, "current_year": cur_year, "feature": feature, "reference_n": int(len(ref)), "current_n": int(len(cur)), "psi": psi(ref, cur), "wasserstein": float(wasserstein_distance(ref, cur)), "wasserstein_over_ref_scale": float(wasserstein_distance(ref, cur) / ref_scale), "reference_robust_scale": ref_scale, "median_shift": median_shift, "median_shift_robust": median_shift_robust, "mad_shift": mad_shift, "mad_ratio": mad_ratio, "status": "ok"}


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_multiyear_validation")}
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in YEARS)})
    if len(stations) < 2:
        raise RuntimeError("fewer than two stations have all three eligible years")
    source = source[source["station_id"].astype(str).isin(stations) & source["year"].isin(YEARS)].copy()

    detail_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    scored_by_seed: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        frame = build_dataset(source, seed, set(stations))
        scored, _ = score(frame)
        scored = add_structural(scored)
        scored["year"] = pd.to_numeric(scored["year"], errors="coerce").astype(int)
        scored["station_key"] = scored["station_key"].astype(str)
        # Exclude injected rows from distribution diagnostics. Detector values are retained
        # exactly as produced by the frozen pipeline; this is a natural-observation drift audit.
        normal = scored.loc[~scored["combined_is_attack"].astype(bool)].copy()
        scored_by_seed[seed] = normal
        for station in stations:
            for feature in VALUE_COLS:
                for ref_year, cur_year in [(2021, 2022), (2021, 2023), (2022, 2023)]:
                    ref = normal[(normal["station_key"].eq(station)) & (normal["year"].eq(ref_year))]
                    cur = normal[(normal["station_key"].eq(station)) & (normal["year"].eq(cur_year))]
                    row = summarize_pair(ref, cur, feature, station, ref_year, cur_year)
                    row["seed"] = seed
                    detail_rows.append(row)
        for station in stations:
            for year in YEARS:
                part = normal[(normal["station_key"].eq(station)) & (normal["year"].eq(year))]
                for feature in VALUE_COLS:
                    values = finite_values(part, feature)
                    if len(values):
                        distribution_rows.append({"seed": seed, "station_key": station, "year": year, "feature": feature, "n": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "median": float(np.median(values)), "mad": float(np.median(np.abs(values - np.median(values)))), "q01": float(np.quantile(values, 0.01)), "q99": float(np.quantile(values, 0.99))})
        print(json.dumps({"seed": seed, "stations": len(stations), "normal_rows": int(len(normal))}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(detail_rows)
    distributions = pd.DataFrame(distribution_rows)
    aggregate = detail[detail["status"].eq("ok")].groupby(["reference_year", "current_year", "feature"], sort=False).agg(station_count=("station_key", "nunique"), pair_count=("station_key", "size"), psi_mean=("psi", "mean"), psi_median=("psi", "median"), psi_p95=("psi", lambda x: float(x.quantile(0.95))), wasserstein_mean=("wasserstein", "mean"), wasserstein_over_ref_scale_mean=("wasserstein_over_ref_scale", "mean"), median_shift_mean=("median_shift", "mean"), median_shift_robust_mean=("median_shift_robust", "mean"), median_shift_robust_abs_p95=("median_shift_robust", lambda x: float(x.abs().quantile(0.95))), mad_ratio_median=("mad_ratio", "median"), mad_ratio_p95=("mad_ratio", lambda x: float(x.quantile(0.95)))).reset_index()
    aggregate.to_csv(TABLES / "stage78_noaa_multiyear_drift_aggregate.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(TABLES / "stage78_noaa_multiyear_drift_detail.csv", index=False, encoding="utf-8-sig")
    distributions.to_csv(TABLES / "stage78_noaa_multiyear_distributions.csv", index=False, encoding="utf-8-sig")

    severe = aggregate.sort_values(["psi_p95", "median_shift_robust_abs_p95"], ascending=False).head(10)
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方三年度数据；2021/2022/2023均为自然观测分布诊断；冻结TrustMeteo-QC评分器；排除透明受控注入行，不把受控攻击当自然漂移。每个种子独立评分，结果按站点配对汇总。", "stations": stations, "years": YEARS, "seeds": SEEDS, "features": VALUE_COLS, "normal_rows_by_seed": {str(seed): int(len(frame)) for seed, frame in scored_by_seed.items()}, "aggregate_rows": int(len(aggregate)), "severe_pairs": severe.to_dict(orient="records"), "interpretation_rules": {"psi": "仅作相对漂移指标；需结合站点样本量与业务阈值解释。", "wasserstein_over_ref_scale": "Wasserstein距离除以参考年非退化鲁棒尺度；尺度优先使用1.4826*MAD，MAD为零时回退IQR/1.34898，再回退标准差。", "median_shift_robust": "中位数差除以同一参考年非退化鲁棒尺度。", "use": "本诊断只用于预注册跨年份自适应方案，不自动修改9B主模型或锁定测试结果。"}}
    (TABLES / "stage78_noaa_multiyear_drift_diagnostic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")

    lines = ["# 阶段78 NOAA多年度分布漂移诊断", "", "冻结评分器，在三年度自然观测行上按站点配对计算 PSI、Wasserstein 距离、中位数/MAD 漂移。透明受控注入行被排除，不将其解释为自然攻击。", "", f"- 站点：`{len(stations)}`；种子：`{len(SEEDS)}`；特征：`{len(VALUE_COLS)}`；聚合行：`{len(aggregate)}`", "", "| 参考年→当前年 | 特征 | PSI均值 | PSI P95 | Wasserstein/鲁棒尺度均值 | 鲁棒中位数位移绝对值P95 |", "|---|---|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['reference_year']}→{row['current_year']} | {row['feature']} | {row['psi_mean']:.4f} | {row['psi_p95']:.4f} | {row['wasserstein_over_ref_scale_mean']:.4f} | {row['median_shift_robust_abs_p95']:.4f} |")
    lines += ["", "## 使用边界", "", "该阶段只量化自然观测的年份/站点分布漂移，为后续预先冻结的自适应阈值方案提供诊断证据。它不改变9B主模型、不覆盖锁定测试、不把弱质量标记或受控注入当攻击真值。离散/零MAD检测器的归一化距离使用非退化鲁棒尺度，避免数值伪影。", "", "## 最大漂移组合（前10）", "", "| 参考年→当前年 | 特征 | PSI P95 | 鲁棒中位数位移P95 |", "|---|---|---:|---:|"]
    for row in severe.to_dict(orient="records"):
        lines.append(f"| {row['reference_year']}→{row['current_year']} | {row['feature']} | {row['psi_p95']:.4f} | {row['median_shift_robust_abs_p95']:.4f} |")
    (REPORTS / "stage78_noaa_multiyear_drift_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(stations), "aggregate_rows": len(aggregate), "top_drift": severe.head(3).to_dict(orient="records")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
