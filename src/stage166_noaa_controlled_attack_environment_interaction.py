from __future__ import annotations

"""Stage 166: attack-type by environment interaction under the frozen protocol."""

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
FOLDS = [([2020], 2021, 2022), ([2020, 2021], 2022, 2023), ([2020, 2021, 2022], 2023, 2024), ([2020, 2021, 2022, 2023], 2024, 2025)]
SEEDS = [3407, 6257, 7919, 104729, 20260715]
ATTACK_TYPES = ["spike", "offset", "drift", "time_shift", "stuck", "gaussian_noise"]
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


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def finite_or_none(value: object) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def append_rows(fold: str, seed: int, test_year: int, test_context: pd.DataFrame, y_test: np.ndarray, predictions: dict[str, tuple[np.ndarray, np.ndarray, float, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    attack_labels = test_context["combined_attack_type"].astype(str)
    attack_mask_all = test_context["combined_is_attack"].astype(bool).to_numpy()
    for method, (pred, risk, threshold, valid_f1) in predictions.items():
        for variable, labels in CONTEXTS.items():
            for context in labels:
                context_mask = test_context[variable].eq(context).to_numpy()
                for attack_type in ATTACK_TYPES:
                    mask = context_mask & ((~attack_mask_all) | attack_labels.eq(attack_type).to_numpy())
                    if not mask.any() or int(y_test[mask].sum()) == 0:
                        continue
                    y = y_test[mask]
                    p = pred[mask]
                    metrics = _metrics(y, p, risk[mask])
                    rows.append({
                        "fold": fold, "seed": int(seed), "test_year": int(test_year), "method": method,
                        "context_variable": variable, "context": context, "attack_type": attack_type,
                        "rows": int(mask.sum()), "attack_rows": int(y.sum()), "clean_rows": int((~y).sum()),
                        "threshold": float(threshold), "valid_f1": float(valid_f1),
                        "test_f1": float(metrics["f1"]), "test_recall": float(metrics["recall"]),
                        "test_precision": float(metrics["precision"]), "test_auprc": finite_or_none(metrics["average_precision"]),
                        "test_false_alarm_rate": float(p[~y].mean()) if (~y).any() else None,
                    })
    return rows


def fold_bootstrap(group: pd.DataFrame, seed: int) -> tuple[float | None, float | None]:
    values = group.groupby("fold", sort=True)["diff"].mean().dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(REPEATS, len(values)))]
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(value) for value in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    rows: list[dict[str, object]] = []
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
            valid_expanded = smooth_risk(scored.loc[valid], expanded_raw[valid], WINDOW)
            expanded_threshold, expanded_valid_f1 = _best_threshold(y_valid, valid_expanded)
            test_expanded = smooth_risk(scored.loc[test], expanded_raw[test], WINDOW)
            predictions["TrustMeteo-QC frozen expanded"] = (test_expanded >= expanded_threshold, test_expanded, float(expanded_threshold), float(expanded_valid_f1))
            default_score = scored.loc[test, "robust_fused_risk"].to_numpy(dtype=float)
            predictions["TrustMeteo-QC default"] = (scored.loc[test, "predicted_attack"].astype(bool).to_numpy(), default_score, float(default_result["setting"]["threshold"]), float(default_result["valid"]["f1"]))
            for method, score_values in [("Robust max deviation", robust_score), ("Isolation Forest", isolation_score), ("One-Class SVM", svm_score)]:
                threshold, valid_f1 = _best_threshold(y_valid, score_values[valid])
                predictions[method] = (score_values[test] >= threshold, score_values[test], float(threshold), float(valid_f1))
            rows.extend(append_rows(fold, seed, test_year, scored.loc[test].reset_index(drop=True), y_test, predictions))
        print(json.dumps({"fold": fold, "seeds": len(SEEDS), "detail_rows": len(rows)}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("stage166 produced no rows")
    summary = detail.groupby(["method", "context_variable", "context", "attack_type"], sort=True).agg(cells=("fold", "size"), attack_rows=("attack_rows", "sum"), test_f1_mean=("test_f1", "mean"), test_recall_mean=("test_recall", "mean"), test_precision_mean=("test_precision", "mean"), test_auprc_mean=("test_auprc", "mean"), test_far_mean=("test_false_alarm_rate", "mean")).reset_index()
    paired_rows: list[dict[str, object]] = []
    not_estimable: list[dict[str, object]] = []
    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    keys = ["fold", "seed", "test_year", "context_variable", "context", "attack_type"]
    contexts = detail[["context_variable", "context", "attack_type"]].drop_duplicates().sort_values(["context_variable", "context", "attack_type"])
    for baseline in BASELINES:
        other = detail[detail["method"].eq(baseline)]
        common = expanded.set_index(keys).join(other.set_index(keys), lsuffix="_expanded", rsuffix="_baseline", how="inner").reset_index()
        for metric in METRICS:
            common["diff"] = pd.to_numeric(common[f"{metric}_expanded"], errors="coerce") - pd.to_numeric(common[f"{metric}_baseline"], errors="coerce")
            for context_row in contexts.itertuples(index=False):
                variable, context, attack_type = str(context_row.context_variable), str(context_row.context), str(context_row.attack_type)
                group = common[common["context_variable"].eq(variable) & common["context"].eq(context) & common["attack_type"].eq(attack_type)].dropna(subset=["diff"])
                fold_count = int(group["fold"].nunique())
                if fold_count < 2:
                    not_estimable.append({"comparison": f"expanded - {baseline}", "metric": metric, "context_variable": variable, "context": context, "attack_type": attack_type, "fold_count": fold_count, "reason": "fewer than two folds with estimable metric"})
                    continue
                low, high = fold_bootstrap(group, stable_seed("stage166", baseline, metric, variable, context, attack_type))
                paired_rows.append({"comparison": f"expanded - {baseline}", "metric": metric, "context_variable": variable, "context": context, "attack_type": attack_type, "fold_count": fold_count, "fold_seed_cells": int(len(group)), "mean_diff": float(group["diff"].mean()), "median_diff": float(group["diff"].median()), "bootstrap_fold_ci95_low": low, "bootstrap_fold_ci95_high": high, "positive_cells": int((group["diff"] > 0).sum()), "negative_cells": int((group["diff"] < 0).sum())})
    paired = pd.DataFrame(paired_rows)
    not_estimable_frame = pd.DataFrame(not_estimable)
    prefix = "stage166_noaa_controlled_attack_environment_interaction"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    not_estimable_frame.to_csv(TABLES / f"{prefix}_not_estimable.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "NOAA官方2020–2025四折滚动、五种子和冻结train/valid/test边界；Trust default与扩展复用冻结设置，Robust、Isolation Forest和One-Class SVM仅用train正常样本拟合、valid选择阈值。每个攻击类型×环境状态单元保留全部正常行作为误报分母，test只评价。不重新训练、不重新选择候选、不根据test结果调参（不根据交互test结果选方法或参数）。攻击为透明受控注入，不是真实攻击日志；NOAA质量字段仅作弱标签；不可估计指标保留为空。",
        "stations": stations, "folds": [fold_name(*fold) for fold in FOLDS], "seeds": SEEDS, "attack_types": ATTACK_TYPES, "methods": METHODS, "contexts": CONTEXTS,
        "detail_rows": int(len(detail)), "summary_rows": int(len(summary)), "paired_rows": int(len(paired)), "not_estimable_count": int(len(not_estimable_frame)),
        "paired": paired.to_dict(orient="records"), "not_estimable": not_estimable_frame.to_dict(orient="records"),
        "limitations": ["攻击类型×环境交互是受控注入下的异质性分析，不是真实攻击日志验证。", "四个滚动fold限制交互检验功效；不可估计单元不能填0或删除。", "交互结果不回写冻结模型。"],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 阶段166 NOAA攻击类型×环境状态交互", "", "每个攻击类型单元保留全部正常行作为误报分母；五种方法使用相同冻结协议。", "", "| 方法 | 状态变量 | 状态 | 攻击类型 | F1 | Recall | Precision | AUPRC | FAR |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict(orient="records"):
        auprc = "NA" if pd.isna(row["test_auprc_mean"]) else f"{row['test_auprc_mean']:.4f}"
        lines.append(f"| {row['method']} | {row['context_variable']} | {row['context']} | {row['attack_type']} | {row['test_f1_mean']:.4f} | {row['test_recall_mean']:.4f} | {row['test_precision_mean']:.4f} | {auprc} | {row['test_far_mean']:.4f} |")
    lines += ["", f"可估计成对单元：{len(paired)}；不可估计单元：{len(not_estimable_frame)}。", "交互结果只用于异质性分析；攻击为透明受控注入，不能写成真实攻击验证。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "summary_rows": len(summary), "paired_rows": len(paired), "not_estimable": len(not_estimable_frame)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
