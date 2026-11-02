from __future__ import annotations

"""Stage 198: cross-fold transfer of the seed-randomized protection choice.

Stages 195 and 196 re-selected the persistent-repeat protection setting
inside every fold/seed using that fold's validation data.  That is the
correct deployment protocol when validation data for the target period is
available, but it does not answer whether the *choice* transfers across
time folds.

This stage freezes the candidate grid and selection rule, then for each
held-out fold selects one fixed configuration using only the valid metrics
of the other three folds.  The held-out fold's own valid split is used only
to calibrate the FAR-constrained threshold for that transferred
configuration; its test split is evaluation-only.

No model, alpha, window, q, feature, attack position, or threshold is
selected on held-out test.  NOAA attacks remain transparent controlled
injections, not real attack logs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage193_noaa_persistent_repeat_protection import (  # noqa: E402
    build_run_lengths,
    build_rolling_scores,
    candidate_weights,
    gate_thresholds,
    macro_event_coverage,
    selection_key,
    valid_far_threshold,
)
from stage195_noaa_seed_randomized_placement import (  # noqa: E402
    AGGREGATIONS,
    ATTACK_TYPES,
    FAR_GRID,
    FOLDS,
    GATE_QUANTILES,
    NON_STUCK_TYPES,
    PRIMARY_FAR,
    RUN_LENGTHS,
    SELECTION_RULES,
    WINDOWS,
    build_score_frame,
    clean_records,
    fold_name,
    resolve_fold,
)

COHORTS = {
    "original": [3407, 6257, 7919, 104729, 20260715],
    "independent": [15485863, 32452843, 49979687, 67867967, 86028121],
}
METHOD = "TrustMeteo-QC frozen expanded"
PROTECTION_OPTIONS = [(0.0, 0)] + [
    (float(quantile), int(length))
    for quantile in GATE_QUANTILES
    for length in RUN_LENGTHS
]
WEIGHT_COMBINATIONS = candidate_weights()
CANDIDATE_CACHE_COLUMNS = [
    "cohort",
    "fold",
    "seed",
    "max_far",
    "gate_quantile",
    "run_length",
    "window",
    "aggregation",
    "base_weight",
    "candidate_weight",
    "valid_f1",
    "valid_event_macro",
    "valid_nonstuck_event_macro",
]


def prepare_group(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
    dict[tuple[int, str], np.ndarray],
    dict[tuple[int, str], np.ndarray],
]:
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
    thresholds = gate_thresholds(gate_valid, valid_normal)
    runs = build_run_lengths(frame, thresholds)
    rolling = build_rolling_scores(frame)
    rolling_valid = {key: values[valid] for key, values in rolling.items()}
    rolling_test = {key: values[test] for key, values in rolling.items()}
    runs_valid = {key: values[valid] for key, values in runs.items()}
    runs_test = {key: values[test] for key, values in runs.items()}
    context = {
        "valid": valid,
        "test": test,
        "y_valid": y_valid,
        "y_test": y_test,
        "valid_frame": valid_frame,
        "test_frame": test_frame,
        "valid_normal": valid_normal,
        "base_valid": base_valid,
        "base_test": base_test,
        "rolling_valid": rolling_valid,
        "rolling_test": rolling_test,
        "runs_valid": runs_valid,
        "runs_test": runs_test,
    }
    return frame, context, rolling, runs


def candidate_table(
    frame: pd.DataFrame,
    context: dict[str, object],
) -> pd.DataFrame:
    base_valid = context["base_valid"]
    valid_normal = context["valid_normal"]
    y_valid = context["y_valid"]
    valid_frame = context["valid_frame"]
    rolling_valid = context["rolling_valid"]
    runs_valid = context["runs_valid"]
    rows: list[dict[str, object]] = []
    for max_far in FAR_GRID:
        for gate_quantile, run_length in PROTECTION_OPTIONS:
            if gate_quantile > 0.0:
                protected_valid = runs_valid[gate_quantile] >= run_length
            else:
                protected_valid = np.zeros(len(base_valid), dtype=bool)
            for window in WINDOWS:
                for aggregation in AGGREGATIONS:
                    key = (window, aggregation)
                    for base_weight, candidate_weight in WEIGHT_COMBINATIONS:
                        valid_score = base_valid.copy()
                        if candidate_weight > 0.0:
                            blend_valid = (
                                (1.0 - candidate_weight) * base_valid
                                + candidate_weight * rolling_valid[key]
                            )
                            valid_score = np.where(
                                protected_valid, base_valid, blend_valid
                            )
                        threshold = valid_far_threshold(
                            valid_score, valid_normal, max_far
                        )
                        valid_pred = valid_score >= threshold
                        valid_metrics = _metrics(
                            y_valid, valid_pred, valid_score
                        )
                        rows.append(
                            {
                                "max_far": float(max_far),
                                "gate_quantile": float(gate_quantile),
                                "run_length": int(run_length),
                                "window": int(window),
                                "aggregation": aggregation,
                                "base_weight": float(base_weight),
                                "candidate_weight": float(candidate_weight),
                                "valid_f1": float(valid_metrics["f1"]),
                                "valid_event_macro": macro_event_coverage(
                                    valid_frame, valid_pred, ATTACK_TYPES
                                ),
                                "valid_nonstuck_event_macro": (
                                    macro_event_coverage(
                                        valid_frame,
                                        valid_pred,
                                        NON_STUCK_TYPES,
                                    )
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def select_transferred(
    candidates: pd.DataFrame,
    rule: str,
    max_far: float,
) -> dict[str, object]:
    current = candidates[
        candidates["max_far"].eq(float(max_far))
    ].copy()
    if current.empty:
        raise ValueError("no source-fold candidates at requested FAR")
    best = None
    best_key = None
    for row in current.to_dict(orient="records"):
        key = selection_key(
            rule,
            {
                "f1": float(row["valid_f1"]),
            },
            float(row["valid_event_macro"]),
            float(row["valid_nonstuck_event_macro"]),
            float(row["candidate_weight"]),
            int(row["window"]),
            float(row["gate_quantile"]),
            int(row["run_length"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = row
    if best is None:
        raise RuntimeError("no transferred candidate selected")
    return best


def evaluate_transferred(
    context: dict[str, object],
    selection: dict[str, object],
) -> dict[str, object]:
    base_valid = context["base_valid"]
    base_test = context["base_test"]
    valid_normal = context["valid_normal"]
    y_test = context["y_test"]
    test_frame = context["test_frame"]
    rolling_valid = context["rolling_valid"]
    rolling_test = context["rolling_test"]
    runs_valid = context["runs_valid"]
    runs_test = context["runs_test"]
    max_far = float(selection["max_far"])
    candidate_weight = float(selection["candidate_weight"])
    gate_quantile = float(selection["gate_quantile"])
    run_length = int(selection["run_length"])
    key = (int(selection["window"]), str(selection["aggregation"]))
    if gate_quantile > 0.0:
        protected_valid = runs_valid[gate_quantile] >= run_length
        protected_test = runs_test[gate_quantile] >= run_length
    else:
        protected_valid = np.zeros(len(base_valid), dtype=bool)
        protected_test = np.zeros(len(base_test), dtype=bool)
    valid_score = base_valid.copy()
    test_score = base_test.copy()
    if candidate_weight > 0.0:
        blend_valid = (
            (1.0 - candidate_weight) * base_valid
            + candidate_weight * rolling_valid[key]
        )
        blend_test = (
            (1.0 - candidate_weight) * base_test
            + candidate_weight * rolling_test[key]
        )
        valid_score = np.where(protected_valid, base_valid, blend_valid)
        test_score = np.where(protected_test, base_test, blend_test)
    baseline_threshold = valid_far_threshold(
        base_valid, valid_normal, max_far
    )
    baseline_pred = base_test >= baseline_threshold
    transferred_threshold = valid_far_threshold(
        valid_score, valid_normal, max_far
    )
    transferred_pred = test_score >= transferred_threshold
    transferred_metrics = _metrics(
        y_test, transferred_pred, test_score
    )
    baseline_metrics = _metrics(y_test, baseline_pred, base_test)
    return {
        "test_f1": float(transferred_metrics["f1"]),
        "test_recall": float(transferred_metrics["recall"]),
        "test_precision": float(transferred_metrics["precision"]),
        "test_auprc": float(transferred_metrics["average_precision"]),
        "test_false_alarm_rate": (
            float(transferred_pred[~y_test].mean())
            if (~y_test).any()
            else float("nan")
        ),
        "test_event_macro": macro_event_coverage(
            test_frame, transferred_pred, ATTACK_TYPES
        ),
        "test_nonstuck_event_macro": macro_event_coverage(
            test_frame, transferred_pred, NON_STUCK_TYPES
        ),
        "baseline_test_f1": float(baseline_metrics["f1"]),
        "baseline_test_recall": float(baseline_metrics["recall"]),
        "baseline_test_precision": float(baseline_metrics["precision"]),
        "baseline_test_auprc": float(
            baseline_metrics["average_precision"]
        ),
        "baseline_test_false_alarm_rate": (
            float(baseline_pred[~y_test].mean())
            if (~y_test).any()
            else float("nan")
        ),
        "baseline_test_event_macro": macro_event_coverage(
            test_frame, baseline_pred, ATTACK_TYPES
        ),
        "baseline_test_nonstuck_event_macro": macro_event_coverage(
            test_frame, baseline_pred, NON_STUCK_TYPES
        ),
    }


def summarize_and_write(
    detail: pd.DataFrame,
    selections: pd.DataFrame,
    cohort: str,
) -> int:
    prefix = f"stage198_noaa_seed_randomized_cross_fold_transfer_{cohort}"
    detail.to_csv(
        TABLES / f"{prefix}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selections.to_csv(
        TABLES / f"{prefix}_selections.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate = (
        detail.groupby(["selection_rule", "max_far"], sort=True)
        .agg(
            fold_seed_pairs=("held_out_fold", "size"),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
            test_recall_mean=("test_recall", "mean"),
            test_precision_mean=("test_precision", "mean"),
            test_auprc_mean=("test_auprc", "mean"),
            test_false_alarm_rate_mean=("test_false_alarm_rate", "mean"),
            test_event_macro_mean=("test_event_macro", "mean"),
            test_nonstuck_event_macro_mean=(
                "test_nonstuck_event_macro",
                "mean",
            ),
            baseline_test_f1_mean=("baseline_test_f1", "mean"),
            baseline_test_nonstuck_event_macro_mean=(
                "baseline_test_nonstuck_event_macro",
                "mean",
            ),
            protected_pairs=(
                "run_length",
                lambda values: int((values > 0).sum()),
            ),
            mean_candidate_weight=("candidate_weight", "mean"),
        )
        .reset_index()
    )
    aggregate["test_f1_diff_mean"] = (
        aggregate["test_f1_mean"] - aggregate["baseline_test_f1_mean"]
    )
    aggregate["test_nonstuck_diff_mean"] = (
        aggregate["test_nonstuck_event_macro_mean"]
        - aggregate["baseline_test_nonstuck_event_macro_mean"]
    )
    paired_rows: list[dict[str, object]] = []
    for rule in SELECTION_RULES:
        for max_far in FAR_GRID:
            current = detail[
                detail["selection_rule"].eq(rule)
                & detail["max_far"].eq(float(max_far))
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
                baseline_metric = metric.replace(
                    "test_", "baseline_test_"
                )
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
    paired.to_csv(
        TABLES / f"{prefix}_paired_vs_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        TABLES / f"{prefix}_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary = aggregate[
        aggregate["selection_rule"].eq("valid_f1")
        & aggregate["max_far"].eq(PRIMARY_FAR)
    ].iloc[0]
    primary_paired = paired[
        paired["selection_rule"].eq("valid_f1")
        & paired["max_far"].eq(PRIMARY_FAR)
    ]
    f1_paired = primary_paired[
        primary_paired["metric"].eq("test_f1")
    ].iloc[0]
    nonstuck_paired = primary_paired[
        primary_paired["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    selection_counts = (
        selections.groupby(
            [
                "held_out_fold",
                "gate_quantile",
                "run_length",
                "window",
                "aggregation",
                "candidate_weight",
            ],
            sort=True,
        )
        .size()
        .reset_index(name="seed_count")
    )
    selection_counts.to_csv(
        TABLES / f"{prefix}_selection_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段198检验种子随机化连续重复保护候选的跨折迁移：每个留出"
            "折只在其余三个折的valid结果上选择一个固定门控分位、连续"
            "长度、窗口、聚合和权重；留出折自身的valid正常样本只用于"
            "按固定FAR上限校准阈值；留出折test只评价；不根据test选择；"
            "不修改也不替换冻结协议与冻结主模型；NOAA攻击为透明受控"
            "注入，不是真实攻击日志。"
        ),
        "cohort": cohort,
        "seeds": COHORTS[cohort],
        "folds": [fold_name(*fold) for fold in FOLDS],
        "detail_rows": int(len(detail)),
        "selection_rows": int(len(selections)),
        "selection_count_rows": int(len(selection_counts)),
        "aggregate_rows": int(len(aggregate)),
        "paired_rows": int(len(paired)),
        "aggregate": clean_records(aggregate),
        "paired": clean_records(paired),
        "selection_counts": clean_records(selection_counts),
        "limitations": [
            "跨折迁移只覆盖当前NOAA站点、年份和受控注入生成器。",
            "源折valid选择减少折内选择自由，但不能替代独立数据源验证。",
            "候选仍不替换冻结主模型。",
            "NOAA攻击为透明受控注入，不代表真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段198 NOAA种子随机位置跨折迁移",
        "",
        f"- 队列：{cohort}",
        "",
        "| 选择规则 | FAR上限 | test F1 | test Recall | test FAR | 非stuck事件覆盖 | F1平均差 | 正向/负向 | 采用保护 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"{paired_row['positive_pairs']}/"
            f"{paired_row['negative_pairs']} | "
            f"{int(row['protected_pairs'])}/{int(row['fold_seed_pairs'])} |"
        )
    lines += [
        "",
        f"1%FAR下valid_f1规则：F1平均差="
        f"{f1_paired['mean_diff']:+.6f}，非stuck事件覆盖平均差="
        f"{nonstuck_paired['mean_diff']:+.6f}，"
        f"采用保护={int(primary['protected_pairs'])}/"
        f"{int(primary['fold_seed_pairs'])}。",
        "",
        "跨折选择只使用源折valid指标；留出折test不参与选择。",
        "",
        "NOAA攻击为透明受控注入，不代表真实攻击日志。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cohort": cohort,
                "detail_rows": payload["detail_rows"],
                "selection_rows": payload["selection_rows"],
                "aggregate_rows": payload["aggregate_rows"],
                "paired_rows": payload["paired_rows"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        choices=sorted(COHORTS),
        default="independent",
    )
    parser.add_argument("--fold", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    from stage195_noaa_seed_randomized_placement import (  # noqa: E402
        AUDIT,
        FROZEN,
        NOAA,
    )

    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
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
    frozen_source = (
        pd.read_csv(FROZEN)
        .loc[lambda x: x["method"].eq(METHOD)]
        .groupby("fold", sort=True)
        .agg(alpha_base=("alpha_base", "first"), window=("window", "first"))
    )
    fold_index = resolve_fold(args.fold)
    selected_folds = [FOLDS[fold_index]] if fold_index is not None else FOLDS
    selected_seeds = (
        [int(args.seed)] if args.seed is not None else COHORTS[args.cohort]
    )
    frozen_index = pd.MultiIndex.from_product(
        [
            [fold_name(*fold) for fold in selected_folds],
            selected_seeds,
        ],
        names=["fold", "seed"],
    )
    frozen = frozen_source.reindex(frozen_index.get_level_values("fold"))
    frozen.index = frozen_index
    if frozen[["alpha_base", "window"]].isna().any().any():
        raise ValueError("missing frozen alpha/window for requested cohort")

    candidates_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    context_by_key: dict[tuple[str, int], dict[str, object]] = {}
    candidate_cache_path = (
        TABLES
        / (
            "stage198_noaa_seed_randomized_cross_fold_transfer_"
            f"{args.cohort}_candidate_cache.parquet"
        )
    )
    if candidate_cache_path.exists():
        candidate_cache = pd.read_parquet(candidate_cache_path)
        missing_columns = [
            column
            for column in CANDIDATE_CACHE_COLUMNS
            if column not in candidate_cache.columns
        ]
        if missing_columns:
            raise ValueError(
                "stage198 candidate cache is missing columns: "
                + ", ".join(missing_columns)
            )
    else:
        candidate_cache = pd.DataFrame(columns=CANDIDATE_CACHE_COLUMNS)
    expected_candidate_rows = (
        len(FAR_GRID)
        * len(PROTECTION_OPTIONS)
        * len(WINDOWS)
        * len(AGGREGATIONS)
        * len(WEIGHT_COMBINATIONS)
    )
    for train_years, valid_year, test_year in selected_folds:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in selected_seeds:
            key = (fold, int(seed))
            cached = candidate_cache[
                candidate_cache["cohort"].eq(args.cohort)
                & candidate_cache["fold"].eq(fold)
                & candidate_cache["seed"].eq(int(seed))
            ]
            if len(cached) == expected_candidate_rows:
                candidates = cached[CANDIDATE_CACHE_COLUMNS].reset_index(
                    drop=True
                )
                frame, context, _, _ = prepare_group(
                    build_score_frame(
                        source,
                        stations,
                        frozen,
                        train_years,
                        valid_year,
                        test_year,
                        seed,
                    )[0]
                )
            else:
                frame, context, _, _ = prepare_group(
                    build_score_frame(
                        source,
                        stations,
                        frozen,
                        train_years,
                        valid_year,
                        test_year,
                        seed,
                    )[0]
                )
                candidates = candidate_table(frame, context)
                candidates.insert(0, "cohort", args.cohort)
                candidates.insert(1, "fold", fold)
                candidates.insert(2, "seed", int(seed))
                candidates = candidates[CANDIDATE_CACHE_COLUMNS]
                candidate_cache = pd.concat(
                    [
                        candidate_cache[
                            ~(
                                candidate_cache["cohort"].eq(args.cohort)
                                & candidate_cache["fold"].eq(fold)
                                & candidate_cache["seed"].eq(int(seed))
                            )
                        ],
                        candidates,
                    ],
                    ignore_index=True,
                )
                temporary = candidate_cache_path.with_suffix(
                    ".parquet.tmp"
                )
                candidate_cache.to_parquet(temporary, index=False)
                temporary.replace(candidate_cache_path)
            candidates_by_key[key] = candidates
            context_by_key[key] = context

    fold_names = [
        fold_name(*fold)
        for fold in selected_folds
        if any(
            key[0] == fold_name(*fold)
            for key in candidates_by_key
        )
    ]
    if len(fold_names) != 4:
        raise ValueError("cross-fold transfer expects all four folds")
    rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for held_out in fold_names:
        source_folds = [name for name in fold_names if name != held_out]
        source_candidates = pd.concat(
            [
                candidates_by_key[(name, int(seed))]
                for name in source_folds
                for seed in selected_seeds
            ],
            ignore_index=True,
        )
        source_aggregate = (
            source_candidates.groupby(
                [
                    "max_far",
                    "gate_quantile",
                    "run_length",
                    "window",
                    "aggregation",
                    "base_weight",
                    "candidate_weight",
                ],
                sort=False,
            )
            .agg(
                valid_f1=("valid_f1", "mean"),
                valid_event_macro=("valid_event_macro", "mean"),
                valid_nonstuck_event_macro=(
                    "valid_nonstuck_event_macro",
                    "mean",
                ),
            )
            .reset_index()
        )
        for rule in SELECTION_RULES:
            for max_far in FAR_GRID:
                selected = select_transferred(
                    source_aggregate, rule, float(max_far)
                )
                for seed in selected_seeds:
                    context = context_by_key[(held_out, int(seed))]
                    metrics = evaluate_transferred(context, selected)
                    rows.append(
                        {
                            "held_out_fold": held_out,
                            "seed": int(seed),
                            "source_folds": "|".join(source_folds),
                            "selection_rule": rule,
                            "max_far": float(max_far),
                            "gate_quantile": float(
                                selected["gate_quantile"]
                            ),
                            "run_length": int(selected["run_length"]),
                            "window": int(selected["window"]),
                            "aggregation": str(selected["aggregation"]),
                            "base_weight": float(
                                selected["base_weight"]
                            ),
                            "candidate_weight": float(
                                selected["candidate_weight"]
                            ),
                            "source_valid_f1": float(
                                selected["valid_f1"]
                            ),
                            "source_valid_event_macro": float(
                                selected["valid_event_macro"]
                            ),
                            "source_valid_nonstuck_event_macro": float(
                                selected["valid_nonstuck_event_macro"]
                            ),
                            **metrics,
                        }
                    )
                selection_rows.append(
                    {
                        "held_out_fold": held_out,
                        "source_folds": "|".join(source_folds),
                        "selection_rule": rule,
                        "max_far": float(max_far),
                        "gate_quantile": float(
                            selected["gate_quantile"]
                        ),
                        "run_length": int(selected["run_length"]),
                        "window": int(selected["window"]),
                        "aggregation": str(selected["aggregation"]),
                        "base_weight": float(selected["base_weight"]),
                        "candidate_weight": float(
                            selected["candidate_weight"]
                        ),
                        "source_valid_f1": float(selected["valid_f1"]),
                        "source_valid_event_macro": float(
                            selected["valid_event_macro"]
                        ),
                        "source_valid_nonstuck_event_macro": float(
                            selected["valid_nonstuck_event_macro"]
                        ),
                    }
                )
            print(
                json.dumps(
                    {
                        "held_out_fold": held_out,
                        "selection_rule": rule,
                        "status": "completed",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return summarize_and_write(
        pd.DataFrame(rows), pd.DataFrame(selection_rows), args.cohort
    )


if __name__ == "__main__":
    raise SystemExit(main())
