from __future__ import annotations

"""Stage 163: full fair-baseline comparison within NOAA environment strata.

All independent baselines are fitted on normal training rows only and select
their thresholds on validation data.  Test strata are evaluation-only.  The
NOAA attacks are transparent controlled injections, not real attack logs.
"""

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
REFERENCE = TABLES / "stage158_noaa_controlled_regime_gap_performance_detail.csv"
FOLDS = [([2020], 2021, 2022), ([2020, 2021], 2022, 2023), ([2020, 2021, 2022], 2023, 2024), ([2020, 2021, 2022, 2023], 2024, 2025)]
SEEDS = [3407, 6257, 7919, 104729, 20260715]
METHODS = ["TrustMeteo-QC frozen expanded", "TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
BASELINES = ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]
CONTEXTS = {
    "temperature_regime": ["cold", "mild", "hot", "missing"],
    "gap_bucket": ["regular_le_3h", "moderate_3_6h", "long_6_24h", "very_long_gt_24h"],
}
ALPHA = 0.25
WINDOW = 1
REPEATS = 10000

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components  # noqa: E402
from stage161_noaa_controlled_station_regime_performance import clean_context  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage85_noaa_rolling_baseline_comparison import SCORE_COLS  # noqa: E402


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def finite_or_none(value: object) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def fold_cluster_bootstrap(values: pd.DataFrame, seed: int) -> tuple[float | None, float | None]:
    fold_means = values.groupby("fold", sort=True)["diff"].mean().dropna().to_numpy(dtype=float)
    if len(fold_means) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = fold_means[rng.integers(0, len(fold_means), size=(REPEATS, len(fold_means)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def score_rows(
    fold: str,
    seed: int,
    test_year: int,
    test_context: pd.DataFrame,
    y_test: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray, float, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, (pred, risk, threshold, valid_f1) in predictions.items():
        for variable, labels in CONTEXTS.items():
            for label in labels:
                mask = test_context[variable].eq(label).to_numpy()
                if not mask.any():
                    continue
                y = y_test[mask]
                p = pred[mask]
                metrics = _metrics(y, p, risk[mask])
                rows.append({
                    "fold": fold,
                    "seed": int(seed),
                    "test_year": int(test_year),
                    "method": method,
                    "context_variable": variable,
                    "context": label,
                    "rows": int(mask.sum()),
                    "attack_rows": int(y.sum()),
                    "clean_rows": int((~y).sum()),
                    "threshold": float(threshold),
                    "valid_f1": float(valid_f1),
                    "test_f1": float(metrics["f1"]),
                    "test_recall": float(metrics["recall"]),
                    "test_precision": float(metrics["precision"]),
                    "test_auprc": finite_or_none(metrics["average_precision"]),
                    "test_false_alarm_rate": float(p[~y].mean()) if (~y).any() else None,
                })
    return rows


def reference_integrity(detail: pd.DataFrame) -> dict[str, object]:
    reference = pd.read_csv(REFERENCE)
    keys = ["fold", "seed", "test_year", "method", "context_variable", "context"]
    current = detail[detail["method"].isin(["TrustMeteo-QC frozen expanded", "TrustMeteo-QC default"])]
    merged = current.merge(reference, on=keys, suffixes=("_current", "_reference"), how="outer", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("stage163 Trust rows do not align with stage158 reference keys")
    deltas: dict[str, float] = {}
    for metric in METRICS:
        left = pd.to_numeric(merged[f"{metric}_current"], errors="coerce")
        right = pd.to_numeric(merged[f"{metric}_reference"], errors="coerce")
        both_missing = left.isna() & right.isna()
        mismatch_missing = left.isna() ^ right.isna()
        if mismatch_missing.any():
            raise ValueError(f"stage163 reference missingness mismatch for {metric}")
        finite = ~(both_missing | mismatch_missing)
        delta = float((left.loc[finite] - right.loc[finite]).abs().max()) if finite.any() else 0.0
        deltas[metric] = delta
        if delta > 1e-12:
            raise ValueError(f"stage163 reference drift for {metric}: {delta}")
    return {"reference_rows": int(len(reference)), "matched_rows": int(len(merged)), "max_abs_delta": deltas}


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(value) for value in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    rows: list[dict[str, object]] = []
    integrity: list[dict[str, object]] = []

    for train_years, valid_year, test_year in FOLDS:
        fold = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, default_result = score(frame)
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
            base = scored["base_fused_risk"].to_numpy(dtype=float)
            structural = scored["structural_risk"].to_numpy(dtype=float)
            expanded_raw = ALPHA * base + (1.0 - ALPHA) * structural
            expanded_valid_score = smooth_risk(scored.loc[valid], expanded_raw[valid], WINDOW)
            expanded_threshold, expanded_valid_f1 = _best_threshold(y_valid, expanded_valid_score)
            expanded_test_score = smooth_risk(scored.loc[test], expanded_raw[test], WINDOW)
            predictions["TrustMeteo-QC frozen expanded"] = (expanded_test_score >= expanded_threshold, expanded_test_score, float(expanded_threshold), float(expanded_valid_f1))
            default_test_score = scored.loc[test, "robust_fused_risk"].to_numpy(dtype=float)
            default_threshold = float(default_result["setting"]["threshold"])
            predictions["TrustMeteo-QC default"] = (scored.loc[test, "predicted_attack"].astype(bool).to_numpy(), default_test_score, default_threshold, float(default_result["valid"]["f1"]))

            for method, scores in [("Robust max deviation", robust_score), ("Isolation Forest", isolation_score), ("One-Class SVM", svm_score)]:
                threshold, valid_f1 = _best_threshold(y_valid, scores[valid])
                test_score = scores[test]
                predictions[method] = (test_score >= threshold, test_score, float(threshold), float(valid_f1))

            test_context = scored.loc[test].reset_index(drop=True)
            rows.extend(score_rows(fold, seed, test_year, test_context, y_test, predictions))
            integrity.append({
                "fold": fold,
                "seed": int(seed),
                "rows": int(len(frame)),
                "stations": int(frame["station_key"].nunique()),
                "train_attack_rows": int(frame.loc[frame["split"].eq("train"), "combined_is_attack"].sum()),
                "valid_attack_rows": int(frame.loc[frame["split"].eq("valid"), "combined_is_attack"].sum()),
                "test_attack_rows": int(frame.loc[frame["split"].eq("test"), "combined_is_attack"].sum()),
            })
        print(json.dumps({"fold": fold, "seeds": len(SEEDS), "detail_rows": len(rows)}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("stage163 produced no rows")
    reference_check = reference_integrity(detail)

    paired_rows: list[dict[str, object]] = []
    not_estimable: list[dict[str, object]] = []
    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    keys = ["fold", "seed", "test_year", "context_variable", "context"]
    context_values = detail[["context_variable", "context"]].drop_duplicates().sort_values(["context_variable", "context"])
    for baseline in BASELINES:
        other = detail[detail["method"].eq(baseline)]
        common = expanded.set_index(keys).join(other.set_index(keys), lsuffix="_expanded", rsuffix="_baseline", how="inner").reset_index()
        for metric in METRICS:
            common["diff"] = pd.to_numeric(common[f"{metric}_expanded"], errors="coerce") - pd.to_numeric(common[f"{metric}_baseline"], errors="coerce")
            for context_row in context_values.itertuples(index=False):
                variable = str(context_row.context_variable)
                context = str(context_row.context)
                group = common[common["context_variable"].eq(variable) & common["context"].eq(context)].dropna(subset=["diff"])
                fold_count = int(group["fold"].nunique())
                if fold_count < 2:
                    not_estimable.append({"comparison": f"expanded - {baseline}", "metric": metric, "context_variable": variable, "context": context, "fold_seed_cells": int(len(group)), "fold_count": fold_count, "reason": "fewer than two folds with estimable paired metric"})
                    continue
                low, high = fold_cluster_bootstrap(group, stable_seed("stage163", baseline, metric, variable, context))
                paired_rows.append({
                    "comparison": f"expanded - {baseline}",
                    "metric": metric,
                    "context_variable": variable,
                    "context": context,
                    "fold_seed_cells": int(len(group)),
                    "fold_count": fold_count,
                    "mean_diff": float(group["diff"].mean()),
                    "median_diff": float(group["diff"].median()),
                    "bootstrap_fold_ci95_low": low,
                    "bootstrap_fold_ci95_high": high,
                    "positive_cells": int((group["diff"] > 0).sum()),
                    "negative_cells": int((group["diff"] < 0).sum()),
                    "zero_cells": int((group["diff"] == 0).sum()),
                })

    paired = pd.DataFrame(paired_rows)
    not_estimable_frame = pd.DataFrame(not_estimable)
    summary = detail.groupby(["method", "context_variable", "context"], sort=True).agg(cells=("fold", "size"), attack_rows=("attack_rows", "sum"), test_f1_mean=("test_f1", "mean"), test_recall_mean=("test_recall", "mean"), test_precision_mean=("test_precision", "mean"), test_auprc_mean=("test_auprc", "mean"), test_far_mean=("test_false_alarm_rate", "mean")).reset_index()
    prefix = "stage163_noaa_controlled_regime_full_baseline"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    not_estimable_frame.to_csv(TABLES / f"{prefix}_not_estimable.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "NOAA官方2020–2025四折累计历史训练→valid→目标年test；五种子；攻击为透明受控注入，不是真实攻击日志。Trust default与冻结扩展方案复用阶段158设置；Robust、Isolation Forest和One-Class SVM仅用train正常样本拟合，各方法阈值只由valid选择，test只按原始温度状态和观测间隔评价。不重新训练、不重新选择候选、不根据test结果调参，NOAA质量字段仅作弱标签。",
        "stations": stations,
        "folds": [fold_name(*fold) for fold in FOLDS],
        "seeds": SEEDS,
        "methods": METHODS,
        "contexts": CONTEXTS,
        "detail_rows": int(len(detail)),
        "summary_rows": int(len(summary)),
        "paired_rows": int(len(paired)),
        "not_estimable_count": int(len(not_estimable_frame)),
        "reference_integrity": reference_check,
        "paired": paired.to_dict(orient="records"),
        "not_estimable": not_estimable_frame.to_dict(orient="records"),
        "limitations": [
            "环境分层属于冻结test的异质性评价，不用于反向选择模型或参数。",
            "受控攻击不是真实攻击日志；NOAA质量字段仍只是弱标签。",
            "四个滚动fold和11个站点限制统计功效；基线优势和扩展方案弱项必须完整报告。",
            "无攻击正例导致的不可估计AUPRC保留为空，不填0。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    lines = ["# 阶段163 NOAA受控攻击环境分层完整基线比较", "", "五种方法使用相同四折×五种子协议；独立基线只用train正常样本拟合，阈值只由valid选择，test只评价。", "", "| 方法 | 状态变量 | 状态 | F1 | Recall | Precision | AUPRC | FAR |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict(orient="records"):
        auprc = "NA" if pd.isna(row["test_auprc_mean"]) else f"{row['test_auprc_mean']:.4f}"
        lines.append(f"| {row['method']} | {row['context_variable']} | {row['context']} | {row['test_f1_mean']:.4f} | {row['test_recall_mean']:.4f} | {row['test_precision_mean']:.4f} | {auprc} | {row['test_far_mean']:.4f} |")
    lines += ["", "扩展方案相对各基线的完整成对差异和fold bootstrap区间见CSV；任何基线优势、跨零区间和不可估计指标均须保留。", "攻击为透明受控注入，不能写成真实攻击日志验证。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "summary_rows": len(summary), "paired_rows": len(paired), "not_estimable": len(not_estimable_frame)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
