from __future__ import annotations

"""Stage 192: repeat-ratio gated event reranking audit.

Stage 190 and stage 191 showed that a causal rolling ``noise_ratio``
candidate improves spike, drift, and offset detection, but slightly reduces
row-level recall on stuck events.  This stage tests a pre-declared
``repeat_ratio`` gate intended to apply the rolling candidate only to
non-stuck-like rows.

The gate threshold is calibrated on valid-normal rows only.  For every
fold/seed, the gate threshold, rolling window, aggregation, blend weight,
and selection rule are chosen on valid; test labels are used only for
evaluation.  No model is retrained and the frozen model is not replaced.

NOAA attacks are transparent controlled injections, not real attack logs.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SCORE_DETAIL = (
    TABLES
    / "stage187_noaa_targeted_features_validation_calibrated_score_detail.parquet"
)
NOISE_FEATURE = "noise_ratio"
GATE_FEATURE = "repeat_ratio"
WINDOWS = [2, 3, 6]
AGGREGATIONS = ["rolling_max", "rolling_mean"]
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]
GATE_QUANTILES = [0.5, 0.75, 0.9, 0.95]
FAR_GRID = [0.005, 0.01, 0.02, 0.05]
SELECTION_RULES = [
    "valid_f1",
    "valid_event_macro",
    "valid_nonstuck_event_macro",
]
ATTACK_TYPES = [
    "spike",
    "offset",
    "drift",
    "time_shift",
    "stuck",
    "gaussian_noise",
]
NON_STUCK_TYPES = [name for name in ATTACK_TYPES if name != "stuck"]
FOLD_NAMES = [
    "2020->2021->2022",
    "2020-2021->2022->2023",
    "2020-2021-2022->2023->2024",
    "2020-2021-2022-2023->2024->2025",
]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402


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


def resolve_fold(value: str | None) -> str | None:
    if value is None:
        return None
    if value in FOLD_NAMES:
        return value
    if value.isdigit():
        index = int(value)
        if 0 <= index < len(FOLD_NAMES):
            return FOLD_NAMES[index]
    raise ValueError(
        f"unknown fold {value!r}; use 0-{len(FOLD_NAMES) - 1} or a fold name"
    )


def valid_far_threshold(
    scores: np.ndarray,
    normal_mask: np.ndarray,
    max_far: float,
) -> float:
    normal_scores = np.asarray(scores, dtype=float)[normal_mask]
    normal_scores = normal_scores[np.isfinite(normal_scores)]
    if normal_scores.size == 0:
        return float("inf")
    return float(
        np.quantile(normal_scores, 1.0 - max_far, method="higher")
    )


def macro_event_coverage(
    frame: pd.DataFrame,
    selected: np.ndarray,
    attack_types: list[str],
) -> float:
    attack = frame["is_attack"].astype(bool).to_numpy()
    attack_type = frame["attack_type"].astype(str).to_numpy()
    values: list[float] = []
    for current_type in attack_types:
        current = attack & (attack_type == current_type)
        if not current.any():
            continue
        work = frame.loc[current, ["event_id"]].copy()
        work["selected"] = selected[current]
        values.append(
            float(work.groupby("event_id")["selected"].any().mean())
        )
    return float(np.mean(values)) if values else float("nan")


def type_metrics(
    frame: pd.DataFrame,
    selected: np.ndarray,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    attack_type = frame["attack_type"].astype(str).to_numpy()
    attack = frame["is_attack"].astype(bool).to_numpy()
    for current_type in ATTACK_TYPES:
        current = attack & (attack_type == current_type)
        if not current.any():
            continue
        work = frame.loc[current, ["event_id"]].copy()
        work["selected"] = selected[current]
        output[current_type] = {
            "attack_rows": int(current.sum()),
            "selected_attack_rows": int((selected & current).sum()),
            "attack_recall": float(
                (selected & current).sum() / current.sum()
            ),
            "events": int(work["event_id"].nunique()),
            "event_coverage": float(
                work.groupby("event_id")["selected"].any().mean()
            ),
        }
    return output


def causal_roll(values: np.ndarray, window: int, aggregation: str) -> np.ndarray:
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation: {aggregation}")
    if window < 1:
        raise ValueError("window must be positive")
    series = pd.Series(np.asarray(values, dtype=float))
    rolling = series.rolling(window=window, min_periods=1)
    if aggregation == "rolling_max":
        return rolling.max().to_numpy(dtype=float)
    return rolling.mean().to_numpy(dtype=float)


def candidate_weights() -> list[tuple[float, float]]:
    return [
        (float(base_weight), float(candidate_weight))
        for base_weight, candidate_weight in itertools.product(
            [1.0 - weight for weight in WEIGHT_GRID],
            WEIGHT_GRID,
        )
        if abs((base_weight + candidate_weight) - 1.0) <= 1e-12
    ]


WEIGHT_COMBINATIONS = candidate_weights()


def prepare_wide(group: pd.DataFrame) -> pd.DataFrame:
    first = group[group["feature"].eq(NOISE_FEATURE)][
        [
            "fold",
            "seed",
            "split",
            "station_key",
            "timestamp_utc",
            "base_score",
            "candidate_score",
            "is_attack",
            "attack_type",
            "event_id",
        ]
    ].copy()
    wide = first.rename(columns={"candidate_score": "noise_ratio"})
    gate = group[group["feature"].eq(GATE_FEATURE)][
        ["fold", "seed", "split", "station_key", "timestamp_utc", "candidate_score"]
    ].rename(columns={"candidate_score": "repeat_ratio"})
    wide = wide.merge(
        gate,
        on=["fold", "seed", "split", "station_key", "timestamp_utc"],
        how="outer",
        validate="one_to_one",
        sort=False,
    )
    if len(wide) != len(first):
        raise ValueError("noise_ratio and repeat_ratio rows do not align")
    if wide[["base_score", "noise_ratio", "repeat_ratio"]].isna().any().any():
        raise ValueError("base, noise, or gate score is missing")
    return wide


def build_rolling_scores(
    frame: pd.DataFrame,
) -> dict[tuple[int, str], np.ndarray]:
    output: dict[tuple[int, str], np.ndarray] = {}
    for window in WINDOWS:
        for aggregation in AGGREGATIONS:
            values = np.empty(len(frame), dtype=float)
            for _, part in frame.groupby("station_key", sort=False):
                values[part.index.to_numpy()] = causal_roll(
                    part["noise_ratio"].to_numpy(dtype=float),
                    window,
                    aggregation,
                )
            output[(window, aggregation)] = values
    return output


def gate_thresholds(
    gate_valid: np.ndarray,
    valid_normal: np.ndarray,
) -> dict[float, float]:
    normal = np.asarray(gate_valid, dtype=float)[valid_normal]
    normal = normal[np.isfinite(normal)]
    if normal.size == 0:
        raise ValueError("no finite valid-normal gate values")
    return {
        float(quantile): float(
            np.quantile(normal, quantile, method="higher")
        )
        for quantile in GATE_QUANTILES
    }


def selection_key(
    rule: str,
    valid_metrics: dict[str, float],
    valid_event_macro: float,
    valid_nonstuck_macro: float,
    candidate_weight: float,
    window: int,
    gate_quantile: float,
) -> tuple[float, ...]:
    valid_event_macro = (
        float(valid_event_macro)
        if np.isfinite(valid_event_macro)
        else -1.0
    )
    valid_nonstuck_macro = (
        float(valid_nonstuck_macro)
        if np.isfinite(valid_nonstuck_macro)
        else -1.0
    )
    if rule == "valid_f1":
        return (
            float(valid_metrics["f1"]),
            valid_nonstuck_macro,
            valid_event_macro,
            -float(candidate_weight),
            -float(gate_quantile),
            -float(window),
        )
    if rule == "valid_event_macro":
        return (
            valid_event_macro,
            valid_nonstuck_macro,
            float(valid_metrics["f1"]),
            -float(candidate_weight),
            -float(gate_quantile),
            -float(window),
        )
    if rule == "valid_nonstuck_event_macro":
        return (
            valid_nonstuck_macro,
            valid_event_macro,
            float(valid_metrics["f1"]),
            -float(candidate_weight),
            -float(gate_quantile),
            -float(window),
        )
    raise ValueError(f"unknown selection rule: {rule}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-detail",
        type=Path,
        default=SCORE_DETAIL,
        help="Stage-187 valid/test score detail parquet.",
    )
    parser.add_argument("--fold", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    prefix = "stage192_noaa_repeat_gated_reranking"
    score_detail = pd.read_parquet(
        args.score_detail,
        columns=[
            "fold",
            "seed",
            "feature",
            "split",
            "station_key",
            "timestamp_utc",
            "base_score",
            "candidate_score",
            "is_attack",
            "attack_type",
            "event_id",
        ],
    )
    expected_features = {NOISE_FEATURE, GATE_FEATURE}
    score_detail = score_detail[
        score_detail["feature"].isin(expected_features)
    ].copy()
    if set(score_detail["feature"].unique()) != expected_features:
        raise ValueError("stage 192 expects noise_ratio and repeat_ratio")
    if set(score_detail["split"].unique()) != {"valid", "test"}:
        raise ValueError("stage 192 expects valid/test score detail only")
    fold = resolve_fold(args.fold)
    if fold is not None:
        score_detail = score_detail[score_detail["fold"].eq(fold)]
    if args.seed is not None:
        score_detail = score_detail[score_detail["seed"].eq(args.seed)]
    if score_detail.empty:
        raise ValueError("no score rows matched the requested fold/seed")

    detail_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for (fold, seed), group in score_detail.groupby(
        ["fold", "seed"], sort=True
    ):
        frame = prepare_wide(group)
        frame = frame.sort_values(
            ["station_key", "timestamp_utc"], kind="stable"
        ).reset_index(drop=True)
        valid = frame["split"].eq("valid").to_numpy()
        test = frame["split"].eq("test").to_numpy()
        if not valid.any() or not test.any():
            raise ValueError("both valid and test rows are required")
        y_valid = frame.loc[valid, "is_attack"].astype(bool).to_numpy()
        y_test = frame.loc[test, "is_attack"].astype(bool).to_numpy()
        valid_frame = frame.loc[valid].reset_index(drop=True)
        test_frame = frame.loc[test].reset_index(drop=True)
        valid_normal = ~y_valid
        base_valid = frame.loc[valid, "base_score"].to_numpy(dtype=float)
        base_test = frame.loc[test, "base_score"].to_numpy(dtype=float)
        gate_valid = frame.loc[valid, "repeat_ratio"].to_numpy(dtype=float)
        gate_test = frame.loc[test, "repeat_ratio"].to_numpy(dtype=float)
        rolling = build_rolling_scores(frame)
        rolling_valid = {key: values[valid] for key, values in rolling.items()}
        rolling_test = {key: values[test] for key, values in rolling.items()}
        thresholds = gate_thresholds(gate_valid, valid_normal)

        baseline_by_far: dict[float, dict[str, object]] = {}
        for max_far in FAR_GRID:
            threshold = valid_far_threshold(
                base_valid, valid_normal, max_far
            )
            baseline_pred = base_test >= threshold
            baseline_metrics = _metrics(y_test, baseline_pred, base_test)
            baseline_by_far[max_far] = {
                "test_f1": float(baseline_metrics["f1"]),
                "test_recall": float(baseline_metrics["recall"]),
                "test_precision": float(baseline_metrics["precision"]),
                "test_auprc": float(baseline_metrics["average_precision"]),
                "test_false_alarm_rate": (
                    float(baseline_pred[~y_test].mean())
                    if (~y_test).any()
                    else float("nan")
                ),
                "test_event_macro": macro_event_coverage(
                    test_frame, baseline_pred, ATTACK_TYPES
                ),
                "test_nonstuck_event_macro": macro_event_coverage(
                    test_frame, baseline_pred, NON_STUCK_TYPES
                ),
            }

        for max_far in FAR_GRID:
            candidate_stats: dict[str, dict[str, object]] = {}
            for gate_quantile, gate_threshold in thresholds.items():
                gate_valid_mask = gate_valid <= gate_threshold
                gate_test_mask = gate_test <= gate_threshold
                for window in WINDOWS:
                    for aggregation in AGGREGATIONS:
                        key = (window, aggregation)
                        for base_weight, candidate_weight in WEIGHT_COMBINATIONS:
                            valid_score = base_valid.copy()
                            test_score = base_test.copy()
                            if candidate_weight > 0.0:
                                valid_score = np.where(
                                    gate_valid_mask,
                                    (1.0 - candidate_weight) * base_valid
                                    + candidate_weight * rolling_valid[key],
                                    base_valid,
                                )
                                test_score = np.where(
                                    gate_test_mask,
                                    (1.0 - candidate_weight) * base_test
                                    + candidate_weight * rolling_test[key],
                                    base_test,
                                )
                            threshold = valid_far_threshold(
                                valid_score, valid_normal, max_far
                            )
                            valid_pred = valid_score >= threshold
                            valid_metrics = _metrics(
                                y_valid, valid_pred, valid_score
                            )
                            valid_event_macro = macro_event_coverage(
                                valid_frame, valid_pred, ATTACK_TYPES
                            )
                            valid_nonstuck_macro = macro_event_coverage(
                                valid_frame, valid_pred, NON_STUCK_TYPES
                            )
                            stat_key = (
                                f"{gate_quantile:.2f}|{window}|"
                                f"{aggregation}|{candidate_weight:.1f}"
                            )
                            candidate_stats[stat_key] = {
                                "gate_quantile": float(gate_quantile),
                                "gate_threshold": float(gate_threshold),
                                "window": int(window),
                                "aggregation": aggregation,
                                "base_weight": float(base_weight),
                                "candidate_weight": float(candidate_weight),
                                "threshold_valid": float(threshold),
                                "valid_score": valid_score,
                                "test_score": test_score,
                                "valid_metrics": valid_metrics,
                                "valid_event_macro": valid_event_macro,
                                "valid_nonstuck_event_macro": (
                                    valid_nonstuck_macro
                                ),
                            }

            selected_by_rule: dict[str, dict[str, object]] = {}
            for rule in SELECTION_RULES:
                selected_key = None
                selected_stat = None
                for stat in candidate_stats.values():
                    current_key = selection_key(
                        rule,
                        stat["valid_metrics"],
                        stat["valid_event_macro"],
                        stat["valid_nonstuck_event_macro"],
                        stat["candidate_weight"],
                        stat["window"],
                        stat["gate_quantile"],
                    )
                    if selected_key is None or current_key > selected_key:
                        selected_key = current_key
                        selected_stat = stat
                if selected_stat is None:
                    raise RuntimeError("no gated candidate setting was evaluated")
                selected_by_rule[rule] = selected_stat

            for rule in SELECTION_RULES:
                selected_stat = selected_by_rule[rule]
                test_score = selected_stat["test_score"]
                threshold = float(selected_stat["threshold_valid"])
                test_pred = test_score >= threshold
                test_metrics = _metrics(y_test, test_pred, test_score)
                baseline = baseline_by_far[max_far]
                row: dict[str, object] = {
                    "fold": fold,
                    "seed": int(seed),
                    "feature": NOISE_FEATURE,
                    "gate_feature": GATE_FEATURE,
                    "max_far": float(max_far),
                    "selection_rule": rule,
                    "gate_quantile": float(selected_stat["gate_quantile"]),
                    "gate_threshold": float(selected_stat["gate_threshold"]),
                    "window": int(selected_stat["window"]),
                    "aggregation": str(selected_stat["aggregation"]),
                    "base_weight": float(selected_stat["base_weight"]),
                    "candidate_weight": float(
                        selected_stat["candidate_weight"]
                    ),
                    "threshold_valid": threshold,
                    "valid_f1": float(selected_stat["valid_metrics"]["f1"]),
                    "valid_recall": float(
                        selected_stat["valid_metrics"]["recall"]
                    ),
                    "valid_precision": float(
                        selected_stat["valid_metrics"]["precision"]
                    ),
                    "valid_event_macro": float(
                        selected_stat["valid_event_macro"]
                    ),
                    "valid_nonstuck_event_macro": float(
                        selected_stat["valid_nonstuck_event_macro"]
                    ),
                    "test_f1": float(test_metrics["f1"]),
                    "test_recall": float(test_metrics["recall"]),
                    "test_precision": float(test_metrics["precision"]),
                    "test_auprc": float(test_metrics["average_precision"]),
                    "test_false_alarm_rate": (
                        float(test_pred[~y_test].mean())
                        if (~y_test).any()
                        else float("nan")
                    ),
                    "test_event_macro": macro_event_coverage(
                        test_frame, test_pred, ATTACK_TYPES
                    ),
                    "test_nonstuck_event_macro": macro_event_coverage(
                        test_frame, test_pred, NON_STUCK_TYPES
                    ),
                    "baseline_test_f1": float(baseline["test_f1"]),
                    "baseline_test_recall": float(baseline["test_recall"]),
                    "baseline_test_precision": float(
                        baseline["test_precision"]
                    ),
                    "baseline_test_auprc": float(baseline["test_auprc"]),
                    "baseline_test_false_alarm_rate": float(
                        baseline["test_false_alarm_rate"]
                    ),
                    "baseline_test_event_macro": float(
                        baseline["test_event_macro"]
                    ),
                    "baseline_test_nonstuck_event_macro": float(
                        baseline["test_nonstuck_event_macro"]
                    ),
                }
                detail_rows.append(row)
                for current_type, metrics in type_metrics(
                    test_frame, test_pred
                ).items():
                    type_rows.append(
                        {
                            "fold": fold,
                            "seed": int(seed),
                            "feature": NOISE_FEATURE,
                            "gate_feature": GATE_FEATURE,
                            "max_far": float(max_far),
                            "selection_rule": rule,
                            "attack_type": current_type,
                            **metrics,
                        }
                    )
                selection_rows.append(
                    {
                        "fold": fold,
                        "seed": int(seed),
                        "feature": NOISE_FEATURE,
                        "gate_feature": GATE_FEATURE,
                        "max_far": float(max_far),
                        "selection_rule": rule,
                        "gate_quantile": float(
                            selected_stat["gate_quantile"]
                        ),
                        "gate_threshold": float(
                            selected_stat["gate_threshold"]
                        ),
                        "window": int(selected_stat["window"]),
                        "aggregation": str(selected_stat["aggregation"]),
                        "base_weight": float(selected_stat["base_weight"]),
                        "candidate_weight": float(
                            selected_stat["candidate_weight"]
                        ),
                        "threshold_valid": threshold,
                    }
                )
        print(
            json.dumps(
                {"fold": fold, "seed": int(seed), "status": "completed"},
                ensure_ascii=False,
            ),
            flush=True,
        )

    detail = pd.DataFrame(detail_rows)
    type_detail = pd.DataFrame(type_rows)
    selection = pd.DataFrame(selection_rows)
    aggregate = (
        detail.groupby(["selection_rule", "max_far"], sort=True)
        .agg(
            fold_seed_pairs=("fold", "size"),
            valid_f1_mean=("valid_f1", "mean"),
            valid_event_macro_mean=("valid_event_macro", "mean"),
            valid_nonstuck_event_macro_mean=(
                "valid_nonstuck_event_macro",
                "mean",
            ),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
            test_recall_mean=("test_recall", "mean"),
            test_recall_std=("test_recall", "std"),
            test_precision_mean=("test_precision", "mean"),
            test_auprc_mean=("test_auprc", "mean"),
            test_false_alarm_rate_mean=("test_false_alarm_rate", "mean"),
            test_event_macro_mean=("test_event_macro", "mean"),
            test_nonstuck_event_macro_mean=(
                "test_nonstuck_event_macro",
                "mean",
            ),
            baseline_test_f1_mean=("baseline_test_f1", "mean"),
            baseline_test_recall_mean=("baseline_test_recall", "mean"),
            baseline_test_false_alarm_rate_mean=(
                "baseline_test_false_alarm_rate",
                "mean",
            ),
            baseline_test_event_macro_mean=(
                "baseline_test_event_macro",
                "mean",
            ),
            baseline_test_nonstuck_event_macro_mean=(
                "baseline_test_nonstuck_event_macro",
                "mean",
            ),
        )
        .reset_index()
    )
    paired_rows: list[dict[str, object]] = []
    for rule in SELECTION_RULES:
        for max_far in FAR_GRID:
            current = detail[
                detail["selection_rule"].eq(rule)
                & detail["max_far"].eq(max_far)
            ]
            for metric in [
                "test_f1",
                "test_recall",
                "test_precision",
                "test_auprc",
                "test_false_alarm_rate",
                "test_event_macro",
                "test_nonstuck_event_macro",
            ]:
                baseline_metric = metric.replace("test_", "baseline_test_")
                diff = current[metric] - current[baseline_metric]
                paired_rows.append(
                    {
                        "selection_rule": rule,
                        "max_far": float(max_far),
                        "metric": metric,
                        "fold_seed_pairs": int(len(current)),
                        "positive_pairs": int((diff > 0).sum()),
                        "negative_pairs": int((diff < 0).sum()),
                        "mean_diff": float(diff.mean()),
                        "std_diff": float(diff.std()),
                        "min_diff": float(diff.min()),
                        "max_diff": float(diff.max()),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    selection_aggregate = (
        selection.groupby(
            [
                "selection_rule",
                "max_far",
                "gate_quantile",
                "window",
                "aggregation",
                "base_weight",
                "candidate_weight",
            ],
            sort=True,
        )
        .size()
        .reset_index(name="selected_count")
    )
    stability = (
        selection.groupby(["selection_rule", "max_far"], sort=True)
        .agg(
            fold_seed_pairs=("fold", "size"),
            distinct_settings=(
                "aggregation",
                lambda values: int(
                    selection.loc[values.index]
                    .assign(
                        _setting=selection.loc[values.index, "gate_quantile"].astype(str)
                        + "|"
                        + selection.loc[values.index, "window"].astype(str)
                        + "|"
                        + selection.loc[values.index, "aggregation"].astype(str)
                        + "|"
                        + selection.loc[
                            values.index, "candidate_weight"
                        ].astype(str)
                    )["_setting"]
                    .nunique()
                ),
            ),
            mean_gate_quantile=("gate_quantile", "mean"),
            mean_candidate_weight=("candidate_weight", "mean"),
            std_candidate_weight=("candidate_weight", "std"),
            mean_window=("window", "mean"),
        )
        .reset_index()
    )

    detail.to_csv(
        TABLES / f"{prefix}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_detail.to_csv(
        TABLES / f"{prefix}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection.to_csv(
        TABLES / f"{prefix}_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection_aggregate.to_csv(
        TABLES / f"{prefix}_selection_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        TABLES / f"{prefix}_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        TABLES / f"{prefix}_paired_vs_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stability.to_csv(
        TABLES / f"{prefix}_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段187的noise_ratio和repeat_ratio valid/test候选分数，"
            "不重新训练；repeat_ratio门控阈值只由valid正常样本按预先固定"
            "分位0.5/0.75/0.9/0.95校准；滚动窗口预先固定为2/3/6，"
            "聚合只允许rolling_max/rolling_mean，权重组合预先固定；"
            "每个fold/seed只在valid选择规则：按valid_f1、全类型事件覆盖宏平均、"
            "非stuck事件覆盖宏平均三种规则选择门控分位、窗口、聚合和"
            "权重，并在固定FAR上限下只用valid正常样本校准阈值；"
            "test只评价，不参与门控、窗口、聚合、权重、阈值或规则选择；"
            "不根据test选择；候选不替换冻结主模型；五种子仅用于"
            "不确定性描述；NOAA攻击为透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [SCORE_DETAIL.name],
        "feature": NOISE_FEATURE,
        "gate_feature": GATE_FEATURE,
        "windows": WINDOWS,
        "aggregations": AGGREGATIONS,
        "weight_grid": WEIGHT_GRID,
        "weight_combination_count": int(len(WEIGHT_COMBINATIONS)),
        "gate_quantiles": GATE_QUANTILES,
        "far_grid": FAR_GRID,
        "selection_rules": SELECTION_RULES,
        "attack_types": ATTACK_TYPES,
        "detail_rows": int(len(detail)),
        "type_detail_rows": int(len(type_detail)),
        "selection_rows": int(len(selection)),
        "selection_aggregate_rows": int(len(selection_aggregate)),
        "aggregate_rows": int(len(aggregate)),
        "paired_rows": int(len(paired)),
        "stability_rows": int(len(stability)),
        "aggregate": clean_records(aggregate),
        "paired": clean_records(paired),
        "selection_aggregate": clean_records(selection_aggregate),
        "stability": clean_records(stability),
        "limitations": [
            "只审计预声明的repeat_ratio门控和noise_ratio滚动候选。",
            "门控分位、窗口、聚合、权重和阈值只在valid选择，test只评价。",
            "门控候选不替换冻结主模型，也不证明跨数据源普适。",
            "repeat_ratio门控可能误伤真实短时突刺或非重复异常。",
            "攻击仍为透明受控注入，不代表真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段192 NOAA重复率门控重排验证",
        "",
        "repeat_ratio门控和noise_ratio滚动重排只在valid选择，test只评价。",
        "",
        "| 选择规则 | FAR上限 | test F1 | test Recall | test FAR | 非stuck事件覆盖 | 相对基线F1 | 正向折种子 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        paired_row = paired[
            paired["selection_rule"].eq(row["selection_rule"])
            & paired["max_far"].eq(row["max_far"])
            & paired["metric"].eq("test_f1")
        ].iloc[0]
        lines.append(
            f"| {row['selection_rule']} | {row['max_far']:.3f} | "
            f"{row['test_f1_mean']:.6f} | {row['test_recall_mean']:.6f} | "
            f"{row['test_false_alarm_rate_mean']:.6f} | "
            f"{row['test_nonstuck_event_macro_mean']:.6f} | "
            f"{paired_row['mean_diff']:+.6f} | "
            f"{paired_row['positive_pairs']}/{paired_row['fold_seed_pairs']} |"
        )
    lines += [
        "",
        f"运行点={len(detail)}；类型明细={len(type_detail)}；"
        f"选择记录={len(selection)}。",
        "候选只在valid选择；NOAA攻击为透明受控注入，不代表真实攻击日志。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": len(detail),
                "type_detail_rows": len(type_detail),
                "selection_rows": len(selection),
                "aggregate_rows": len(aggregate),
                "paired_rows": len(paired),
                "stability_rows": len(stability),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
