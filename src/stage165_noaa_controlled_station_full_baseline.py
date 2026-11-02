from __future__ import annotations

"""Stage 165: station-level full-baseline heterogeneity audit."""

import hashlib
import json
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
TRUST_SOURCE = TABLES / "stage161_noaa_controlled_station_regime_performance_detail.csv"
FOLDS = [([2020], 2021, 2022), ([2020, 2021], 2022, 2023), ([2020, 2021, 2022], 2023, 2024), ([2020, 2021, 2022, 2023], 2024, 2025)]
SEEDS = [3407, 6257, 7919, 104729, 20260715]
TRUST_METHODS = ["TrustMeteo-QC default", "TrustMeteo-QC frozen expanded"]
BASELINES = ["Robust max deviation", "Isolation Forest", "One-Class SVM"]
ALL_METHODS = ["TrustMeteo-QC frozen expanded", "TrustMeteo-QC default", *BASELINES]
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]
CONTEXTS = {
    "temperature_regime": ["cold", "mild", "hot", "missing"],
    "gap_bucket": ["regular_le_3h", "moderate_3_6h", "long_6_24h", "very_long_gt_24h"],
}
REPEATS = 10000

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components  # noqa: E402
from stage161_noaa_controlled_station_regime_performance import clean_context  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage85_noaa_rolling_baseline_comparison import SCORE_COLS  # noqa: E402


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def finite_or_none(value: object) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def station_bootstrap(values: np.ndarray, seed: int) -> tuple[float | None, float | None]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(REPEATS, len(values)))]
    means = sample.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def station_rows(fold: str, seed: int, test_year: int, test_context: pd.DataFrame, y_test: np.ndarray, predictions: dict[str, tuple[np.ndarray, np.ndarray, float, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, (pred, risk, threshold, valid_f1) in predictions.items():
        for variable, labels in CONTEXTS.items():
            for label in labels:
                context_mask = test_context[variable].eq(label).to_numpy()
                for station in test_context.loc[context_mask, "station_key"].astype(str).unique():
                    mask = context_mask & test_context["station_key"].astype(str).eq(station).to_numpy()
                    if not mask.any():
                        continue
                    y = y_test[mask]
                    p = pred[mask]
                    metrics = _metrics(y, p, risk[mask])
                    rows.append({
                        "fold": fold, "seed": int(seed), "test_year": int(test_year), "method": method,
                        "station_key": str(station), "context_variable": variable, "context": label,
                        "rows": int(mask.sum()), "attack_rows": int(y.sum()), "clean_rows": int((~y).sum()),
                        "threshold": float(threshold), "valid_f1": float(valid_f1),
                        "test_f1": float(metrics["f1"]), "test_recall": float(metrics["recall"]),
                        "test_precision": float(metrics["precision"]), "test_auprc": finite_or_none(metrics["average_precision"]),
                        "test_false_alarm_rate": float(p[~y].mean()) if (~y).any() else None,
                    })
    return rows


def main() -> int:
    trust = pd.read_csv(TRUST_SOURCE)
    trust = trust[trust["method"].isin(TRUST_METHODS)].copy()
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(value) for value in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    baseline_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        fold = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            scored = clean_context(add_components(scored), source)
            train = scored["split"].eq("train").to_numpy()
            valid = scored["split"].eq("valid").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            normal_train = train & (~scored["combined_is_attack"].astype(bool).to_numpy())
            y_valid = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
            y_test = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            x = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            median = np.nanmedian(x[normal_train], axis=0)
            scale = np.maximum(1.4826 * np.nanmedian(np.abs(x[normal_train] - median), axis=0), 0.25)
            robust_score = np.maximum((x - median) / scale, 0.0).max(axis=1)
            isolation = IsolationForest(n_estimators=300, random_state=seed, n_jobs=1, contamination="auto").fit(x[normal_train])
            isolation_score = -isolation.score_samples(x)
            svm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)).fit(x[normal_train])
            svm_score = -svm.decision_function(x).ravel()
            predictions: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
            for method, scores in [("Robust max deviation", robust_score), ("Isolation Forest", isolation_score), ("One-Class SVM", svm_score)]:
                threshold, valid_f1 = _best_threshold(y_valid, scores[valid])
                predictions[method] = (scores[test] >= threshold, scores[test], float(threshold), float(valid_f1))
            test_context = scored.loc[test].reset_index(drop=True)
            baseline_rows.extend(station_rows(fold, seed, test_year, test_context, y_test, predictions))
        print(json.dumps({"fold": fold, "seeds": len(SEEDS), "baseline_rows": len(baseline_rows)}, ensure_ascii=False), flush=True)

    detail = pd.concat([trust, pd.DataFrame(baseline_rows)], ignore_index=True)
    detail["station_key"] = detail["station_key"].astype(str)
    if set(detail["method"].unique()) != set(ALL_METHODS):
        raise ValueError("stage165 method coverage mismatch")
    summary = detail.groupby(["method", "context_variable", "context", "station_key"], sort=True).agg(cells=("fold", "size"), attack_rows=("attack_rows", "sum"), test_f1_mean=("test_f1", "mean"), test_recall_mean=("test_recall", "mean"), test_far_mean=("test_false_alarm_rate", "mean")).reset_index()
    paired_rows: list[dict[str, object]] = []
    not_estimable: list[dict[str, object]] = []
    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    keys = ["fold", "seed", "test_year", "context_variable", "context", "station_key"]
    contexts = detail[["context_variable", "context"]].drop_duplicates().sort_values(["context_variable", "context"])
    for baseline in ["TrustMeteo-QC default", *BASELINES]:
        other = detail[detail["method"].eq(baseline)]
        common = expanded.set_index(keys).join(other.set_index(keys), lsuffix="_expanded", rsuffix="_baseline", how="inner").reset_index()
        for metric in METRICS:
            common["diff"] = pd.to_numeric(common[f"{metric}_expanded"], errors="coerce") - pd.to_numeric(common[f"{metric}_baseline"], errors="coerce")
            for context_row in contexts.itertuples(index=False):
                variable = str(context_row.context_variable)
                context = str(context_row.context)
                group = common[common["context_variable"].eq(variable) & common["context"].eq(context)].dropna(subset=["diff"])
                station_means = group.groupby("station_key", sort=True)["diff"].mean().dropna()
                if len(station_means) < 2:
                    not_estimable.append({"comparison": f"expanded - {baseline}", "metric": metric, "context_variable": variable, "context": context, "station_count": int(len(station_means)), "reason": "fewer than two stations with estimable paired metric"})
                    continue
                values = station_means.to_numpy(dtype=float)
                low, high = station_bootstrap(values, stable_seed("stage165", baseline, metric, variable, context))
                weak = station_means.index[values < 0].astype(str).tolist()
                paired_rows.append({
                    "comparison": f"expanded - {baseline}", "metric": metric, "context_variable": variable, "context": context,
                    "station_count": int(len(values)), "mean_station_diff": float(values.mean()), "median_station_diff": float(np.median(values)),
                    "station_bootstrap_ci95_low": low, "station_bootstrap_ci95_high": high,
                    "positive_station_count": int((values > 0).sum()), "negative_station_count": int((values < 0).sum()), "zero_station_count": int((values == 0).sum()),
                    "weak_station_count": len(weak), "weak_station_keys": ";".join(weak),
                })
    paired = pd.DataFrame(paired_rows)
    not_estimable_frame = pd.DataFrame(not_estimable)
    prefix = "stage165_noaa_controlled_station_full_baseline"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / f"{prefix}_station_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    not_estimable_frame.to_csv(TABLES / f"{prefix}_not_estimable.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段161站点×状态明细和阶段163完整基线协议；四折五种子；Trust default/扩展结果直接复用阶段161，Robust、Isolation Forest和One-Class SVM仅用train正常样本拟合、valid选择阈值，test只评价。对扩展方案相对四个基线按站点聚合并执行站点bootstrap；不重新训练、不重新选择候选、不根据站点test调参。攻击为透明受控注入，不是真实攻击日志；弱站点不能删除，不可估计指标保留为空。",
        "source_artifacts": [TRUST_SOURCE.name, "stage163_noaa_controlled_regime_full_baseline.py"],
        "stations": stations, "folds": [fold_name(*fold) for fold in FOLDS], "seeds": SEEDS, "methods": ALL_METHODS, "contexts": CONTEXTS,
        "detail_rows": int(len(detail)), "baseline_detail_rows": int(len(baseline_rows)), "station_summary_rows": int(len(summary)), "paired_rows": int(len(paired)), "not_estimable_count": int(len(not_estimable_frame)),
        "paired": paired.to_dict(orient="records"), "not_estimable": not_estimable_frame.to_dict(orient="records"),
        "limitations": ["站点bootstrap描述当前11站点的异质性，不创造新的独立站点。", "弱站点和基线优势必须完整报告，不能按结果删除。", "受控攻击不是真实攻击日志；站点层一致性不能替代真实外部攻击验证。"],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 阶段165 NOAA受控攻击站点层完整基线比较", "", "复用阶段161站点层Trust结果并补齐三个独立基线；站点bootstrap用于检查环境分层优势是否由少数站点贡献。", "", "| 比较 | 指标 | 状态变量 | 状态 | 站点数 | 平均差 | CI95 | 正向站点 | 负向站点 | 弱站点 |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in paired.to_dict(orient="records"):
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['context_variable']} | {row['context']} | {row['station_count']} | {row['mean_station_diff']:+.6f} | [{row['station_bootstrap_ci95_low']:+.6f},{row['station_bootstrap_ci95_high']:+.6f}] | {row['positive_station_count']} | {row['negative_station_count']} | {row['weak_station_count']} |")
    lines += ["", f"可估计成对单元：{len(paired)}；不可估计单元：{len(not_estimable_frame)}。", "受控攻击为透明注入；站点层结果用于异质性诊断，不用于重新选择模型。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "baseline_detail_rows": len(baseline_rows), "station_summary_rows": len(summary), "paired_rows": len(paired), "not_estimable": len(not_estimable_frame)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
