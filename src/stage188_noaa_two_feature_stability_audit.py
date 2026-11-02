from __future__ import annotations

"""Stage 188: pre-declared stability audit for two targeted NOAA features.

Stage 187 evaluated eight targeted features one at a time under a
validation-calibrated low-FAR protocol.  Only ``repeat_ratio`` and
``noise_ratio`` showed stable positive signals.  This stage tests whether
those two features are complementary using a deliberately small,
pre-declared combination space:

* non-negative weights over the two features, total weight <= 0.5;
* FAR ceilings 0.5%, 1%, 2%, and 5%;
* selection rules declared before evaluation: valid F1, valid event-macro
  coverage, and valid non-stuck event-macro coverage.

The frozen base score, fold/seed splits, and candidate scores come from
stage 187.  Every weight and threshold is selected on validation only; test
labels are used only for evaluation.

不重新训练、不根据test选择特征、权重、阈值或规则；五种子仅用于不确定性描述。
NOAA攻击为透明受控注入，不是真实攻击日志。
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
    TABLES / "stage187_noaa_targeted_features_validation_calibrated_score_detail.parquet"
)
FEATURES = ["repeat_ratio", "noise_ratio"]
WEIGHT_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]
MAX_TOTAL_WEIGHT = 0.5
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

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402


def weight_combinations() -> list[tuple[float, float]]:
    combinations = [
        tuple(float(value) for value in values)
        for values in itertools.product(WEIGHT_VALUES, repeat=len(FEATURES))
    ]
    combinations = [
        values
        for values in combinations
        if sum(values) > 0.0
        and sum(values) <= MAX_TOTAL_WEIGHT + 1e-12
    ]
    return sorted(combinations, key=lambda values: (sum(values), values))


WEIGHT_COMBINATIONS = weight_combinations()


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


def prepare_wide(group: pd.DataFrame) -> pd.DataFrame:
    first = group[group["feature"].eq(FEATURES[0])][
        [
            "fold",
            "seed",
            "split",
            "station_key",
            "timestamp_utc",
            "base_score",
            "is_attack",
            "attack_type",
            "event_id",
        ]
    ].copy()
    wide = first.copy()
    for feature in FEATURES:
        part = group[group["feature"].eq(feature)][
            ["fold", "seed", "split", "station_key", "timestamp_utc", "candidate_score"]
        ].rename(columns={"candidate_score": feature})
        wide = wide.merge(
            part,
            on=["fold", "seed", "split", "station_key", "timestamp_utc"],
            how="outer",
            validate="one_to_one",
            sort=False,
        )
    if len(wide) != len(first):
        raise ValueError("candidate features do not align on rows")
    if wide[["base_score", *FEATURES]].isna().any().any():
        raise ValueError("candidate rows or base scores are missing")
    return wide


def selection_key(
    rule: str,
    valid_metrics: dict[str, float],
    valid_event_macro: float,
    valid_nonstuck_macro: float,
    total_weight: float,
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
            -float(total_weight),
        )
    if rule == "valid_event_macro":
        return (
            valid_event_macro,
            valid_nonstuck_macro,
            float(valid_metrics["f1"]),
            -float(total_weight),
        )
    if rule == "valid_nonstuck_event_macro":
        return (
            valid_nonstuck_macro,
            valid_event_macro,
            float(valid_metrics["f1"]),
            -float(total_weight),
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
    prefix = "stage188_noaa_two_feature_stability_audit"
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
    expected_features = set(FEATURES)
    score_detail = score_detail[
        score_detail["feature"].isin(FEATURES)
    ].copy()
    if set(score_detail["feature"].unique()) != expected_features:
        raise ValueError("unexpected candidate feature set")
    if set(score_detail["split"].unique()) != {"valid", "test"}:
        raise ValueError("stage 188 expects valid/test score detail only")
    if args.fold is not None:
        score_detail = score_detail[score_detail["fold"].eq(args.fold)]
    if args.seed is not None:
        score_detail = score_detail[score_detail["seed"].eq(args.seed)]
    if score_detail.empty:
        raise ValueError("no score rows matched the requested fold/seed")

    detail_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for (fold, seed), group in score_detail.groupby(
        ["fold", "seed"], sort=True
    ):
        wide = prepare_wide(group)
        valid = wide["split"].eq("valid").to_numpy()
        test = wide["split"].eq("test").to_numpy()
        y_valid = wide.loc[valid, "is_attack"].astype(bool).to_numpy()
        y_test = wide.loc[test, "is_attack"].astype(bool).to_numpy()
        valid_frame = wide.loc[valid].reset_index(drop=True)
        test_frame = wide.loc[test].reset_index(drop=True)
        valid_normal = ~y_valid
        base_valid = wide.loc[valid, "base_score"].to_numpy(dtype=float)
        base_test = wide.loc[test, "base_score"].to_numpy(dtype=float)
        feature_valid = {
            feature: wide.loc[valid, feature].to_numpy(dtype=float)
            for feature in FEATURES
        }
        feature_test = {
            feature: wide.loc[test, feature].to_numpy(dtype=float)
            for feature in FEATURES
        }
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
            for weights in WEIGHT_COMBINATIONS:
                total_weight = float(sum(weights))
                valid_score = (1.0 - total_weight) * base_valid
                test_score = (1.0 - total_weight) * base_test
                for feature, weight in zip(FEATURES, weights):
                    valid_score = valid_score + weight * feature_valid[feature]
                    test_score = test_score + weight * feature_test[feature]
                threshold = valid_far_threshold(
                    valid_score, valid_normal, max_far
                )
                valid_pred = valid_score >= threshold
                valid_metrics = _metrics(y_valid, valid_pred, valid_score)
                valid_event_macro = macro_event_coverage(
                    valid_frame, valid_pred, ATTACK_TYPES
                )
                valid_nonstuck_macro = macro_event_coverage(
                    valid_frame, valid_pred, NON_STUCK_TYPES
                )
                candidate_stats[str(weights)] = {
                    "weights": weights,
                    "total_weight": total_weight,
                    "threshold_valid": float(threshold),
                    "valid_score": valid_score,
                    "test_score": test_score,
                    "valid_metrics": valid_metrics,
                    "valid_event_macro": valid_event_macro,
                    "valid_nonstuck_event_macro": valid_nonstuck_macro,
                }

            selected_by_rule: dict[str, dict[str, object]] = {}
            for rule in SELECTION_RULES:
                selected_key = None
                selected_stat = None
                for key, stat in candidate_stats.items():
                    current_key = selection_key(
                        rule,
                        stat["valid_metrics"],
                        stat["valid_event_macro"],
                        stat["valid_nonstuck_event_macro"],
                        stat["total_weight"],
                    )
                    if selected_key is None or current_key > selected_key:
                        selected_key = current_key
                        selected_stat = stat
                if selected_stat is None:
                    raise RuntimeError("no candidate combination was evaluated")
                selected_by_rule[rule] = selected_stat

            for rule in SELECTION_RULES:
                selected_stat = selected_by_rule[rule]
                test_score = selected_stat["test_score"]
                threshold = float(selected_stat["threshold_valid"])
                test_pred = test_score >= threshold
                test_metrics = _metrics(y_test, test_pred, test_score)
                baseline = baseline_by_far[max_far]
                weights = selected_stat["weights"]
                row: dict[str, object] = {
                    "fold": fold,
                    "seed": int(seed),
                    "max_far": float(max_far),
                    "selection_rule": rule,
                    "weight_repeat_ratio": float(weights[0]),
                    "weight_noise_ratio": float(weights[1]),
                    "total_weight": float(selected_stat["total_weight"]),
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
                            "max_far": float(max_far),
                            "selection_rule": rule,
                            "attack_type": current_type,
                            **metrics,
                        }
                    )
                weight_rows.append(
                    {
                        "fold": fold,
                        "seed": int(seed),
                        "max_far": float(max_far),
                        "selection_rule": rule,
                        "weight_repeat_ratio": float(weights[0]),
                        "weight_noise_ratio": float(weights[1]),
                        "total_weight": float(selected_stat["total_weight"]),
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
    weight_selection = pd.DataFrame(weight_rows)
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
                if baseline_metric not in current.columns:
                    continue
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
    weight_aggregate = (
        weight_selection.groupby(
            [
                "selection_rule",
                "max_far",
                "weight_repeat_ratio",
                "weight_noise_ratio",
                "total_weight",
            ],
            sort=True,
        )
        .size()
        .reset_index(name="selected_count")
    )
    weight_selection["_weight_combo"] = (
        weight_selection["weight_repeat_ratio"].astype(str)
        + "|"
        + weight_selection["weight_noise_ratio"].astype(str)
    )
    stability = (
        weight_selection.groupby(
            ["selection_rule", "max_far"], sort=True
        )
        .agg(
            fold_seed_pairs=("fold", "size"),
            distinct_weight_combinations=("_weight_combo", "nunique"),
            mean_total_weight=("total_weight", "mean"),
            std_total_weight=("total_weight", "std"),
        )
        .reset_index()
    )
    weight_selection = weight_selection.drop(columns=["_weight_combo"])

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
    weight_selection.to_csv(
        TABLES / f"{prefix}_weight_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weight_aggregate.to_csv(
        TABLES / f"{prefix}_weight_aggregate.csv",
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
            "复用阶段187 valid/test候选分数，仅使用repeat_ratio和"
            "noise_ratio两个阶段187稳定候选；非负权重组合总权重不超过"
            "0.5；每个fold/seed只在valid上按预先声明的三种规则选择"
            "权重，并在预先固定的FAR上限下用valid正常样本校准阈值；"
            "test只评价，不参与特征、权重、阈值或规则选择；"
            "不根据test选择特征、权重、阈值或规则；"
            "不重新训练、不替换冻结主模型；五种子仅用于不确定性描述；"
            "NOAA攻击为透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [SCORE_DETAIL.name],
        "features": FEATURES,
        "weight_values": WEIGHT_VALUES,
        "max_total_weight": MAX_TOTAL_WEIGHT,
        "far_grid": FAR_GRID,
        "selection_rules": SELECTION_RULES,
        "attack_types": ATTACK_TYPES,
        "weight_combination_count": int(len(WEIGHT_COMBINATIONS)),
        "detail_rows": int(len(detail)),
        "type_detail_rows": int(len(type_detail)),
        "weight_selection_rows": int(len(weight_selection)),
        "weight_aggregate_rows": int(len(weight_aggregate)),
        "aggregate_rows": int(len(aggregate)),
        "paired_rows": int(len(paired)),
        "stability_rows": int(len(stability)),
        "aggregate": clean_records(aggregate),
        "paired": clean_records(paired),
        "weight_aggregate": clean_records(weight_aggregate),
        "stability": clean_records(stability),
        "limitations": [
            "只审计两个预先选出的特征，不进行全特征搜索。",
            "权重和阈值只在valid选择，test只评价；不能根据test改规则。",
            "候选组合不替换冻结主模型，也不证明跨数据源普适。",
            "事件覆盖宏平均对攻击类型等权，样本量小的类型会放大波动。",
            "攻击仍为透明受控注入，不代表真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段188 NOAA双特征稳定域审计",
        "",
        "复用阶段187分数；权重和阈值只在valid选择，test只评价。",
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
        f"组合数={len(WEIGHT_COMBINATIONS)}；运行点={len(detail)}；"
        f"类型明细={len(type_detail)}。",
        "候选组合只在valid选择；NOAA攻击为透明受控注入，不代表真实攻击日志。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "weight_combinations": len(WEIGHT_COMBINATIONS),
                "detail_rows": len(detail),
                "type_detail_rows": len(type_detail),
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
