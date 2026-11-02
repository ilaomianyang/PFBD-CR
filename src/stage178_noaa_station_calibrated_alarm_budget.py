from __future__ import annotations

"""Stage 178: equal-alarm-budget comparison for station calibration.

Stage 177 showed that station-wise thresholds can improve F1/Recall while
increasing the number of alarms.  This stage removes that confound by fixing
the total test alarm budget per fold, seed, and method, then comparing the
global threshold against station-wise ranking under the exact same budget.

Only frozen stage-172 risk scores are replayed.  No model, alpha, window, or q
is retrained or selected from test.
"""

import argparse
import itertools
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
FROZEN = TABLES / "stage141_noaa_frozen_external_validation_selected.csv"
STAGE172 = TABLES / "stage172_noaa_frozen_expanded_low_far_full_baseline.json"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
Q_GRID = [0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
BUDGET_QUANTILES = [0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
ALLOCATION_RULES = ["size", "uniform", "risk"]
METHODS = [
    "TrustMeteo-QC frozen expanded",
    "TrustMeteo-QC default",
    "Robust max deviation",
    "Isolation Forest",
    "One-Class SVM",
]
BASELINES = METHODS[1:]
METRICS = ["f1", "recall", "precision", "false_alarm_rate"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components  # noqa: E402
from stage151_noaa_attack_intensity_sensitivity import build_fold_scaled  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage85_noaa_rolling_baseline_comparison import SCORE_COLS  # noqa: E402


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def finite_or_none(value: object) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def top_k_mask(values: np.ndarray, budget: int) -> np.ndarray:
    """Select exactly budget rows by descending value with deterministic ties."""
    values = np.asarray(values, dtype=float)
    if budget < 0 or budget > len(values):
        raise ValueError(f"budget {budget} outside [0, {len(values)}]")
    order = np.lexsort((np.arange(len(values)), -values))
    selected = np.zeros(len(values), dtype=bool)
    selected[order[:budget]] = True
    return selected


def station_budget_mask(
    stations: np.ndarray,
    values: np.ndarray,
    budget: int,
    rule: str = "size",
) -> np.ndarray:
    """Allocate the fixed total budget by station-wise risk ranking.

    The largest-remainder allocation is deterministic and uses only the
    station sizes or unlabeled risk scores, not test labels.  Each station
    then takes its top-ranked rows by risk.
    """
    stations = np.asarray(stations).astype(str)
    values = np.asarray(values, dtype=float)
    unique = sorted(set(stations.tolist()))
    if budget < 0 or budget > len(values):
        raise ValueError(f"budget {budget} outside [0, {len(values)}]")
    sizes = np.asarray([int((stations == station).sum()) for station in unique])
    if rule == "size":
        weights = sizes.astype(float)
    elif rule == "uniform":
        weights = np.ones(len(unique), dtype=float)
    elif rule == "risk":
        station_means = np.asarray(
            [
                float(np.mean(values[stations == station]))
                for station in unique
            ],
            dtype=float,
        )
        # Anomaly scores can be negative.  Shift by the minimum so that
        # relative risk is preserved while allocation weights stay nonnegative.
        weights = station_means - float(np.min(station_means))
    else:
        raise ValueError(f"unknown allocation rule: {rule}")
    if weights.sum() <= 0:
        weights = np.ones(len(unique), dtype=float)
    raw = weights / weights.sum() * float(budget)
    allocation = np.floor(raw).astype(int)
    allocation = np.minimum(allocation, sizes)
    remainder = budget - int(allocation.sum())
    while remainder > 0:
        eligible = allocation < sizes
        if not bool(eligible.any()):
            raise RuntimeError("cannot allocate budget within station capacities")
        order = np.lexsort(
            (
                np.arange(len(unique)),
                -(raw - allocation),
                -(sizes - allocation),
            )
        )
        for idx in order:
            if remainder == 0:
                break
            if eligible[idx]:
                allocation[idx] += 1
                remainder -= 1
    selected = np.zeros(len(values), dtype=bool)
    for station, count in zip(unique, allocation):
        idx = np.flatnonzero(stations == station)
        if count:
            local = top_k_mask(values[idx], int(count))
            selected[idx[local]] = True
    return selected


def bundle(
    frame: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    risk: np.ndarray,
) -> dict[str, object]:
    metrics = _metrics(y, pred, risk)
    station = frame[["station_key"]].copy()
    station["y"] = y.astype(bool)
    station["pred"] = pred.astype(bool)
    f1_values: list[float] = []
    recall_values: list[float] = []
    precision_values: list[float] = []
    far_values: list[float] = []
    for _, group in station.groupby("station_key", sort=False):
        tp = int((group["pred"] & group["y"]).sum())
        fp = int((group["pred"] & ~group["y"]).sum())
        fn = int((~group["pred"] & group["y"]).sum())
        tn = int((~group["pred"] & ~group["y"]).sum())
        denominator = 2 * tp + fp + fn
        if denominator:
            f1_values.append(float(2 * tp / denominator))
        if int(group["y"].sum()):
            recall_values.append(float(tp / int(group["y"].sum())))
        if tp + fp:
            precision_values.append(float(tp / (tp + fp)))
        if int((~group["y"]).sum()):
            far_values.append(float(fp / int((~group["y"]).sum())))
    return {
        "f1": finite_or_none(metrics["f1"]),
        "recall": finite_or_none(metrics["recall"]),
        "precision": finite_or_none(metrics["precision"]),
        "auprc": finite_or_none(metrics["average_precision"]),
        "false_alarm_rate": finite_or_none(
            float(pred[~y].mean()) if (~y).any() else np.nan
        ),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "macro_recall": float(np.mean(recall_values)) if recall_values else None,
        "macro_precision": float(np.mean(precision_values)) if precision_values else None,
        "macro_far": float(np.mean(far_values)) if far_values else None,
        "alarm_count": int(pred.sum()),
        "false_positive_count": int((pred & ~y).sum()),
    }


def exact_sign_flip(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = [
        abs(float((values * np.asarray(signs)).mean()))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15)), float(values.mean())


def bh_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    order = np.argsort(np.asarray(values, dtype=float))
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        idx = int(order[rank])
        running = min(running, float(values[idx]) * len(order) / (rank + 1))
        adjusted[idx] = running
    return adjusted.tolist()


def clean_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        cleaned: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, float) and not np.isfinite(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def build_comparisons(detail: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if mode == "baseline":
        left = detail[detail["method"].eq(METHODS[0])]
        right_methods: list[str | None] = list(BASELINES)
        left_prefix = "station_budget_"
        right_prefix = "station_budget_"
        merge_keys = [
            "fold",
            "seed",
            "q",
            "alarm_budget",
            "budget_quantile",
            "allocation_rule",
        ]
    else:
        left = detail
        right_methods = [None]
        left_prefix = "station_budget_"
        right_prefix = "global_budget_"
        merge_keys = [
            "fold",
            "seed",
            "method",
            "q",
            "alarm_budget",
            "budget_quantile",
            "allocation_rule",
        ]
    for right_method in right_methods:
        if right_method is None:
            right = left
            comparison_name = "station-budget - global-budget"
        else:
            right = detail[detail["method"].eq(right_method)]
            comparison_name = (
                f"station-budget expanded - station-budget {right_method}"
            )
        common = left.merge(
            right,
            on=merge_keys,
            suffixes=("_left", "_right"),
            how="inner",
        )
        for rule in ALLOCATION_RULES:
            rule_common = common[common["allocation_rule"].eq(rule)]
            for q in sorted(float(x) for x in rule_common["q"].unique()):
                q_common = rule_common[np.isclose(rule_common["q"].astype(float), q)]
                for metric in METRICS:
                    left_col = f"{left_prefix}{metric}_left"
                    right_col = f"{right_prefix}{metric}_right"
                    diff = pd.to_numeric(
                        q_common[left_col], errors="coerce"
                    ) - pd.to_numeric(q_common[right_col], errors="coerce")
                    fold_values = (
                        diff.groupby(q_common["fold"], sort=True)
                        .mean()
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                    if len(fold_values) < 2:
                        continue
                    p_value, mean_diff = exact_sign_flip(fold_values)
                    rows.append(
                        {
                            "comparison": comparison_name,
                            "allocation_rule": rule,
                            "metric": metric,
                            "q": q,
                            "fold_count": int(len(fold_values)),
                            "fold_seed_cells": int(diff.notna().sum()),
                            "mean_fold_diff": mean_diff,
                            "p_value_exact_sign_flip": p_value,
                            "fold_differences": fold_values.tolist(),
                        }
                    )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["bh_q_value"] = bh_adjust(
            [float(x) for x in out["p_value_exact_sign_flip"]]
        )
    return out


def build_operating_points() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    frozen = pd.read_csv(FROZEN)
    eligible = {
        (int(row["year"]), str(row["station_id"]))
        for row in audit["detail"]
        if row.get("eligible_for_extended_validation")
    }
    years = sorted(int(x) for x in source["year"].unique())
    stations = {
        station
        for station in source["station_id"].astype(str).unique()
        if all((year, station) in eligible for year in years)
    }
    frozen = (
        frozen[frozen["method"].eq(METHODS[0])]
        .copy()
        .set_index(["fold", "seed"])
    )
    rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            alpha = float(frozen.loc[(fold, seed), "alpha_base"])
            window = int(frozen.loc[(fold, seed), "window"])
            frame = build_fold_scaled(
                source, seed, train_years, valid_year, test_year, stations, 1.0
            )
            scored, _ = score(frame)
            scored = add_components(scored)
            train = scored["split"].eq("train").to_numpy()
            test_mask = scored["split"].eq("test").to_numpy()
            normal_train = train & (~scored["combined_is_attack"].astype(bool).to_numpy())
            test_frame = scored.loc[test_mask].reset_index(drop=True)
            y_test = test_frame["combined_is_attack"].astype(bool).to_numpy()
            test_stations = test_frame["station_key"].astype(str).to_numpy()
            x = (
                scored[SCORE_COLS]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            med = np.nanmedian(x[normal_train], axis=0)
            scale = np.maximum(
                1.4826 * np.nanmedian(np.abs(x[normal_train] - med), axis=0),
                0.25,
            )
            robust = np.maximum((x - med) / scale, 0.0).max(axis=1)
            iso = IsolationForest(
                n_estimators=300,
                random_state=seed,
                n_jobs=1,
                contamination="auto",
            ).fit(x[normal_train])
            iso_risk = -iso.score_samples(x)
            svm = make_pipeline(
                StandardScaler(),
                OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
            ).fit(x[normal_train])
            svm_risk = -svm.decision_function(x).ravel()
            base = scored["base_fused_risk"].to_numpy(dtype=float)
            structural = scored["structural_risk"].to_numpy(dtype=float)
            expanded_raw = alpha * base + (1.0 - alpha) * structural
            expanded_risk = np.full(len(scored), np.nan, dtype=float)
            for split_name in ["train", "valid", "test"]:
                split_mask = scored["split"].eq(split_name).to_numpy()
                expanded_risk[split_mask] = smooth_risk(
                    scored.loc[split_mask], expanded_raw[split_mask], window
                )
            risks = {
                METHODS[0]: expanded_risk,
                "TrustMeteo-QC default": scored["robust_fused_risk"].to_numpy(dtype=float),
                "Robust max deviation": robust,
                "Isolation Forest": iso_risk,
                "One-Class SVM": svm_risk,
            }
            for method in METHODS:
                risk = risks[method]
                train_values = risk[normal_train & np.isfinite(risk)]
                if train_values.size == 0:
                    raise RuntimeError(f"no finite train-normal risk for {method}")
                test_risk = risk[test_mask]
                if not np.isfinite(test_risk).all():
                    raise RuntimeError(
                        f"non-finite test risk for {fold}/{seed}/{method}"
                    )
                for q, budget_quantile in zip(Q_GRID, BUDGET_QUANTILES):
                    global_threshold = float(np.nanquantile(train_values, q))
                    budget = int(
                        round(
                            float(len(test_risk))
                            * float(1.0 - budget_quantile)
                        )
                    )
                    budget = max(0, min(int(len(test_risk)), budget))
                    global_pred = top_k_mask(test_risk, budget)
                    if int(global_pred.sum()) != budget:
                        raise RuntimeError(
                            f"global budget mismatch for {fold}/{seed}/{method}/{q}: "
                            f"{int(global_pred.sum())} != {budget}"
                        )
                    global_metrics = bundle(
                        test_frame, y_test, global_pred, test_risk
                    )
                    for allocation_rule in ALLOCATION_RULES:
                        station_pred = station_budget_mask(
                            test_stations,
                            test_risk,
                            budget,
                            rule=allocation_rule,
                        )
                        if int(station_pred.sum()) != budget:
                            raise RuntimeError(
                                f"station budget mismatch for "
                                f"{fold}/{seed}/{method}/{q}/{allocation_rule}: "
                                f"{int(station_pred.sum())} != {budget}"
                            )
                        station_metrics = bundle(
                            test_frame, y_test, station_pred, test_risk
                        )
                        row = {
                            "fold": fold,
                            "seed": int(seed),
                            "method": method,
                            "q": float(q),
                            "alarm_budget": budget,
                            "budget_quantile": float(budget_quantile),
                            "allocation_rule": allocation_rule,
                            "global_threshold": global_threshold,
                            "global_topk_cutoff": (
                                float(np.min(test_risk[global_pred]))
                                if budget
                                else None
                            ),
                        }
                        for label, metrics in (
                            ("global_budget", global_metrics),
                            ("station_budget", station_metrics),
                        ):
                            for metric in [
                                "f1",
                                "recall",
                                "precision",
                                "auprc",
                                "false_alarm_rate",
                                "macro_f1",
                                "macro_recall",
                                "macro_precision",
                                "macro_far",
                                "alarm_count",
                                "false_positive_count",
                            ]:
                                row[f"{label}_{metric}"] = metrics[metric]
                        rows.append(row)
                        for station in sorted(set(test_stations.tolist())):
                            mask = test_stations == station
                            station_rows.append(
                                {
                                    "fold": fold,
                                    "seed": int(seed),
                                    "method": method,
                                    "q": float(q),
                                    "alarm_budget": budget,
                                    "budget_quantile": float(budget_quantile),
                                    "allocation_rule": allocation_rule,
                                    "station_id": station,
                                    "test_rows": int(mask.sum()),
                                    "test_attack_rows": int(y_test[mask].sum()),
                                    "global_budget_alarm_count": int(
                                        global_pred[mask].sum()
                                    ),
                                    "station_budget_alarm_count": int(
                                        station_pred[mask].sum()
                                    ),
                                    "global_budget_true_positive": int(
                                        (global_pred[mask] & y_test[mask]).sum()
                                    ),
                                    "global_budget_false_positive": int(
                                        (global_pred[mask] & ~y_test[mask]).sum()
                                    ),
                                    "station_budget_true_positive": int(
                                        (station_pred[mask] & y_test[mask]).sum()
                                    ),
                                    "station_budget_false_positive": int(
                                        (station_pred[mask] & ~y_test[mask]).sum()
                                    ),
                                }
                            )
        pd.DataFrame(rows).to_csv(
            TABLES / "stage178_noaa_station_calibrated_alarm_budget_detail.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(station_rows).to_csv(
            TABLES / "stage178_noaa_station_calibrated_alarm_budget_station_detail.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "seeds": len(SEEDS),
                    "q_values": len(Q_GRID),
                    "status": "completed",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return pd.DataFrame(rows), pd.DataFrame(station_rows)


def write_summaries(
    detail: pd.DataFrame,
    station_detail: pd.DataFrame,
) -> int:
    expected_detail_rows = len(FOLDS) * len(SEEDS) * len(METHODS) * len(Q_GRID) * len(ALLOCATION_RULES)
    expected_station_detail_rows = expected_detail_rows * 11
    if len(detail) != expected_detail_rows:
        raise RuntimeError(
            f"detail rows mismatch: {len(detail)} != {expected_detail_rows}"
        )
    if len(station_detail) != expected_station_detail_rows:
        raise RuntimeError(
            f"station detail rows mismatch: "
            f"{len(station_detail)} != {expected_station_detail_rows}"
        )
    expected_keys = [
        "fold",
        "seed",
        "method",
        "q",
        "allocation_rule",
    ]
    if detail.duplicated(expected_keys).any():
        raise RuntimeError("duplicate stage178 detail keys")
    station_keys = [*expected_keys, "station_id"]
    if station_detail.duplicated(station_keys).any():
        raise RuntimeError("duplicate stage178 station detail keys")
    if set(detail["allocation_rule"].unique()) != set(ALLOCATION_RULES):
        raise RuntimeError("allocation rule set mismatch")
    if not detail["global_budget_alarm_count"].eq(detail["alarm_budget"]).all():
        raise RuntimeError("global alarm budget is not conserved")
    if not detail["station_budget_alarm_count"].eq(detail["alarm_budget"]).all():
        raise RuntimeError("station alarm budget is not conserved")

    prefix = "stage178_noaa_station_calibrated_alarm_budget"
    detail.to_csv(
        TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig"
    )
    station_detail.to_csv(
        TABLES / f"{prefix}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline_comparisons = build_comparisons(detail, "baseline")
    calibration_comparisons = build_comparisons(detail, "calibration")
    expected_baseline_comparisons = (
        len(BASELINES) * len(Q_GRID) * len(METRICS) * len(ALLOCATION_RULES)
    )
    expected_calibration_comparisons = (
        len(Q_GRID) * len(METRICS) * len(ALLOCATION_RULES)
    )
    if len(baseline_comparisons) != expected_baseline_comparisons:
        raise RuntimeError(
            f"baseline comparison rows mismatch: "
            f"{len(baseline_comparisons)} != {expected_baseline_comparisons}"
        )
    if len(calibration_comparisons) != expected_calibration_comparisons:
        raise RuntimeError(
            f"calibration comparison rows mismatch: "
            f"{len(calibration_comparisons)} != "
            f"{expected_calibration_comparisons}"
        )
    baseline_comparisons.to_csv(
        TABLES / f"{prefix}_baseline_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibration_comparisons.to_csv(
        TABLES / f"{prefix}_calibration_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    expanded = detail[detail["method"].eq(METHODS[0])]
    expanded_995 = expanded[np.isclose(expanded["q"].astype(float), 0.995)]
    rule_metrics: dict[str, dict[str, float]] = {}
    for rule in ALLOCATION_RULES:
        rule_995 = expanded_995[
            expanded_995["allocation_rule"].eq(rule)
        ]
        if len(rule_995) != len(FOLDS) * len(SEEDS):
            raise RuntimeError(f"q=0.995 rows mismatch for rule {rule}")
        rule_metrics[rule] = {
            "global_budget_f1_mean": float(
                rule_995["global_budget_f1"].mean()
            ),
            "station_budget_f1_mean": float(
                rule_995["station_budget_f1"].mean()
            ),
            "global_budget_recall_mean": float(
                rule_995["global_budget_recall"].mean()
            ),
            "station_budget_recall_mean": float(
                rule_995["station_budget_recall"].mean()
            ),
            "global_budget_false_alarm_rate_mean": float(
                rule_995["global_budget_false_alarm_rate"].mean()
            ),
            "station_budget_false_alarm_rate_mean": float(
                rule_995["station_budget_false_alarm_rate"].mean()
            ),
        }
    size_metrics = rule_metrics["size"]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段172冻结风险分数、train/valid/test四折滚动、五种子和六个预先固定q；"
            "每个fold×seed×方法先由全局train正常分位数确定报警预算，再在同一"
            "总预算下比较全局阈值与站点内风险排序；站点预算分别按站点规模、"
            "站点等量和站点平均风险三种预先固定规则做最大余数分配，只使用"
            "站点大小和无标签风险，不使用test标签；"
            "不重新训练、不重新选择模型或q，不根据test调参；"
            "差异按fold执行精确sign-flip并使用BH校正。"
            "NOAA攻击是透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [
            STAGE172.name,
            "stage177_noaa_station_calibrated_operating_points_detail.csv",
        ],
        "methods": METHODS,
        "q_grid": Q_GRID,
        "allocation_rules": ALLOCATION_RULES,
        "expected_detail_rows": expected_detail_rows,
        "expected_station_detail_rows": expected_station_detail_rows,
        "detail_rows": int(len(detail)),
        "station_detail_rows": int(len(station_detail)),
        "baseline_comparison_rows": int(len(baseline_comparisons)),
        "calibration_comparison_rows": int(len(calibration_comparisons)),
        "expanded_q995_by_allocation_rule": rule_metrics,
        "expanded_q995_global_budget_f1_mean": size_metrics[
            "global_budget_f1_mean"
        ],
        "expanded_q995_station_budget_f1_mean": size_metrics[
            "station_budget_f1_mean"
        ],
        "expanded_q995_global_budget_recall_mean": size_metrics[
            "global_budget_recall_mean"
        ],
        "expanded_q995_station_budget_recall_mean": size_metrics[
            "station_budget_recall_mean"
        ],
        "expanded_q995_global_budget_false_alarm_rate_mean": size_metrics[
            "global_budget_false_alarm_rate_mean"
        ],
        "expanded_q995_station_budget_false_alarm_rate_mean": size_metrics[
            "station_budget_false_alarm_rate_mean"
        ],
        "baseline_bh_q_below_05": (
            int((baseline_comparisons["bh_q_value"] < 0.05).sum())
            if not baseline_comparisons.empty
            else 0
        ),
        "calibration_bh_q_below_05": (
            int((calibration_comparisons["bh_q_value"] < 0.05).sum())
            if not calibration_comparisons.empty
            else 0
        ),
        "min_baseline_bh_q_value": (
            float(baseline_comparisons["bh_q_value"].min())
            if not baseline_comparisons.empty
            else None
        ),
        "min_calibration_bh_q_value": (
            float(calibration_comparisons["bh_q_value"].min())
            if not calibration_comparisons.empty
            else None
        ),
        "detail": clean_records(detail),
        "station_detail": clean_records(station_detail),
        "baseline_comparisons": clean_records(baseline_comparisons),
        "calibration_comparisons": clean_records(calibration_comparisons),
        "limitations": [
            "等报警预算比较仍是受控注入下的运行点比较，不代表真实攻击召回率。",
            "站点预算的size、uniform、risk是预先固定的公平性检验，不是业务最优分配。",
            "四折功效有限；BH校正后无显著差异不能写成方法等效。",
            "NOAA攻击是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段178 NOAA站点校准等报警预算审计",
        "",
        "先由全局train正常分位数固定总报警预算，再在同一预算下比较全局阈值和三种站点内风险排序；test只评价。",
        "",
        "| 分配规则 | 方法 | q | 全局预算F1 | 站点预算F1 | 全局预算Recall | 站点预算Recall | 全局预算FAR | 站点预算FAR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary = (
        detail.groupby(["allocation_rule", "method", "q"], sort=True)[
            [
                "global_budget_f1",
                "station_budget_f1",
                "global_budget_recall",
                "station_budget_recall",
                "global_budget_false_alarm_rate",
                "station_budget_false_alarm_rate",
            ]
        ]
        .mean()
        .reset_index()
    )
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['allocation_rule']} | {row['method']} | {row['q']:.4f} | "
            f"{row['global_budget_f1']:.6f} | {row['station_budget_f1']:.6f} | "
            f"{row['global_budget_recall']:.6f} | {row['station_budget_recall']:.6f} | "
            f"{row['global_budget_false_alarm_rate']:.6f} | "
            f"{row['station_budget_false_alarm_rate']:.6f} |"
        )
    lines += [
        "",
        f"明细={len(detail)}；站点明细={len(station_detail)}；"
        f"基线比较={len(baseline_comparisons)}；校准比较={len(calibration_comparisons)}。",
        "等预算比较只改变报警在站点间的分配，不改变总报警数，也不能替代真实攻击验证。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": len(detail),
                "station_detail_rows": len(station_detail),
                "baseline_comparison_rows": len(baseline_comparisons),
                "calibration_comparison_rows": len(calibration_comparisons),
                "baseline_bh_q_below_05": payload["baseline_bh_q_below_05"],
                "calibration_bh_q_below_05": payload["calibration_bh_q_below_05"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Only rebuild summary artifacts from persisted fold details.",
    )
    args = parser.parse_args(argv)
    prefix = "stage178_noaa_station_calibrated_alarm_budget"
    if args.summarize_existing:
        detail = pd.read_csv(TABLES / f"{prefix}_detail.csv")
        station_detail = pd.read_csv(TABLES / f"{prefix}_station_detail.csv")
        return write_summaries(detail, station_detail)
    detail, station_detail = build_operating_points()
    return write_summaries(detail, station_detail)


if __name__ == "__main__":
    raise SystemExit(main())
