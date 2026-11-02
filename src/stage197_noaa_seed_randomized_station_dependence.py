from __future__ import annotations

"""Stage 197: station-dependence audit for stage 195/196.

The seed-randomized placement results are pooled across stations.  A pooled
gain can still be driven by one or two unusually easy stations.  This stage
reuses the exact stage-195 candidate grid and computes station-level and
leave-one-station-out (LOSO) deltas against the frozen base score.

No model, alpha, window, q, threshold, weight, or rule is selected here.
Every fold/seed still uses the validation-selected configuration from the
stage-195 candidate space; test labels are used only for evaluation.
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
    fold_name,
    resolve_fold,
)

REPLICATION_SEEDS = [3407, 6257, 7919, 104729, 20260715]
INDEPENDENT_SEEDS = [15485863, 32452843, 49979687, 67867967, 86028121]
METHOD = "TrustMeteo-QC frozen expanded"
PROTECTION_OPTIONS = [(0.0, 0)] + [
    (float(quantile), int(length))
    for quantile in GATE_QUANTILES
    for length in RUN_LENGTHS
]
WEIGHT_COMBINATIONS = candidate_weights()


def evaluate_station_aware(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = frame.sort_values(
        ["station_key", "timestamp_utc"], kind="stable"
    ).reset_index(drop=True)
    valid = frame["split"].eq("valid").to_numpy()
    test = frame["split"].eq("test").to_numpy()
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
    fold = str(frame["fold"].iloc[0])
    seed = int(frame["seed"].iloc[0])

    station_valid = valid_frame["station_key"].astype(str).to_numpy()
    station_test = test_frame["station_key"].astype(str).to_numpy()
    stations = sorted(set(station_valid))
    rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for max_far in FAR_GRID:
        baseline_threshold = valid_far_threshold(
            base_valid, valid_normal, max_far
        )
        baseline_pred = base_test >= baseline_threshold
        candidate_stats: dict[str, dict[str, object]] = {}
        for gate_quantile, run_length in PROTECTION_OPTIONS:
            if gate_quantile > 0.0:
                protected_valid = runs_valid[gate_quantile] >= run_length
                protected_test = runs_test[gate_quantile] >= run_length
            else:
                protected_valid = np.zeros(len(base_valid), dtype=bool)
                protected_test = np.zeros(len(base_test), dtype=bool)
            for window in WINDOWS:
                for aggregation in AGGREGATIONS:
                    key = (window, aggregation)
                    for base_weight, candidate_weight in WEIGHT_COMBINATIONS:
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
                            valid_score = np.where(
                                protected_valid, base_valid, blend_valid
                            )
                            test_score = np.where(
                                protected_test, base_test, blend_test
                            )
                        threshold = valid_far_threshold(
                            valid_score, valid_normal, max_far
                        )
                        valid_pred = valid_score >= threshold
                        valid_metrics = _metrics(
                            y_valid, valid_pred, valid_score
                        )
                        stat_key = (
                            f"{gate_quantile:.2f}|{run_length}|{window}|"
                            f"{aggregation}|{candidate_weight:.1f}"
                        )
                        candidate_stats[stat_key] = {
                            "gate_quantile": float(gate_quantile),
                            "run_length": int(run_length),
                            "window": int(window),
                            "aggregation": aggregation,
                            "base_weight": float(base_weight),
                            "candidate_weight": float(candidate_weight),
                            "threshold_valid": float(threshold),
                            "test_score": test_score,
                            "valid_metrics": valid_metrics,
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
                    stat["run_length"],
                )
                if selected_key is None or current_key > selected_key:
                    selected_key = current_key
                    selected_stat = stat
            if selected_stat is None:
                raise RuntimeError("no candidate setting was evaluated")
            test_score = selected_stat["test_score"]
            threshold = float(selected_stat["threshold_valid"])
            test_pred = test_score >= threshold
            test_metrics = _metrics(y_test, test_pred, test_score)
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "max_far": float(max_far),
                    "selection_rule": rule,
                    "gate_quantile": float(
                        selected_stat["gate_quantile"]
                    ),
                    "run_length": int(selected_stat["run_length"]),
                    "window": int(selected_stat["window"]),
                    "aggregation": str(selected_stat["aggregation"]),
                    "base_weight": float(selected_stat["base_weight"]),
                    "candidate_weight": float(
                        selected_stat["candidate_weight"]
                    ),
                    "threshold_valid": threshold,
                    "test_f1": float(test_metrics["f1"]),
                    "test_recall": float(test_metrics["recall"]),
                    "test_precision": float(test_metrics["precision"]),
                    "test_auprc": float(
                        test_metrics["average_precision"]
                    ),
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
                    "baseline_test_f1": float(
                        _metrics(y_test, baseline_pred, base_test)["f1"]
                    ),
                    "baseline_test_nonstuck_event_macro": (
                        macro_event_coverage(
                            test_frame, baseline_pred, NON_STUCK_TYPES
                        )
                    ),
                }
            )
            for station in stations:
                mask = station_test == station
                if not mask.any():
                    continue
                station_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "max_far": float(max_far),
                        "selection_rule": rule,
                        "station_key": station,
                        "test_rows": int(mask.sum()),
                        "test_attack_rows": int(y_test[mask].sum()),
                        "test_f1": float(
                            _metrics(
                                y_test[mask],
                                test_pred[mask],
                                test_score[mask],
                            )["f1"]
                        ),
                        "baseline_test_f1": float(
                            _metrics(
                                y_test[mask],
                                baseline_pred[mask],
                                base_test[mask],
                            )["f1"]
                        ),
                        "test_nonstuck_event_macro": macro_event_coverage(
                            test_frame.loc[mask].reset_index(drop=True),
                            test_pred[mask],
                            NON_STUCK_TYPES,
                        ),
                        "baseline_test_nonstuck_event_macro": (
                            macro_event_coverage(
                                test_frame.loc[mask].reset_index(drop=True),
                                baseline_pred[mask],
                                NON_STUCK_TYPES,
                            )
                        ),
                    }
                )
    return rows, station_rows


def summarize_and_write(
    rows: pd.DataFrame,
    station: pd.DataFrame,
    cohort: str,
) -> int:
    prefix = f"stage197_noaa_seed_randomized_station_dependence_{cohort}"
    detail = pd.DataFrame(rows)
    station_detail = pd.DataFrame(station)
    detail.to_csv(
        TABLES / f"{prefix}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_detail.to_csv(
        TABLES / f"{prefix}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_summary = (
        station_detail.groupby(
            ["station_key", "selection_rule", "max_far"], sort=True
        )
        .agg(
            fold_seed_pairs=("fold", "size"),
            f1_diff_mean=(
                "test_f1",
                lambda values: float(
                    (
                        values
                        - station_detail.loc[values.index, "baseline_test_f1"]
                    ).mean()
                ),
            ),
            f1_positive_pairs=(
                "test_f1",
                lambda values: int(
                    (
                        values
                        - station_detail.loc[values.index, "baseline_test_f1"]
                        > 0
                    ).sum()
                ),
            ),
            f1_negative_pairs=(
                "test_f1",
                lambda values: int(
                    (
                        values
                        - station_detail.loc[values.index, "baseline_test_f1"]
                        < 0
                    ).sum()
                ),
            ),
            nonstuck_diff_mean=(
                "test_nonstuck_event_macro",
                lambda values: float(
                    (
                        values
                        - station_detail.loc[
                            values.index,
                            "baseline_test_nonstuck_event_macro",
                        ]
                    ).mean()
                ),
            ),
            nonstuck_positive_pairs=(
                "test_nonstuck_event_macro",
                lambda values: int(
                    (
                        values
                        - station_detail.loc[
                            values.index,
                            "baseline_test_nonstuck_event_macro",
                        ]
                        > 0
                    ).sum()
                ),
            ),
            nonstuck_negative_pairs=(
                "test_nonstuck_event_macro",
                lambda values: int(
                    (
                        values
                        - station_detail.loc[
                            values.index,
                            "baseline_test_nonstuck_event_macro",
                        ]
                        < 0
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )
    station_summary.to_csv(
        TABLES / f"{prefix}_station_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    loso_rows: list[dict[str, object]] = []
    for (fold, seed, max_far, rule), group in detail.groupby(
        ["fold", "seed", "max_far", "selection_rule"], sort=True
    ):
        station_group = station_detail[
            station_detail["fold"].eq(fold)
            & station_detail["seed"].eq(seed)
            & station_detail["max_far"].eq(max_far)
            & station_detail["selection_rule"].eq(rule)
        ]
        total_f1 = float(group["test_f1"].iloc[0])
        baseline_f1 = float(group["baseline_test_f1"].iloc[0])
        total_nonstuck = float(
            group["test_nonstuck_event_macro"].iloc[0]
        )
        baseline_nonstuck = float(
            group["baseline_test_nonstuck_event_macro"].iloc[0]
        )
        for station_key in sorted(station_group["station_key"].unique()):
            kept = station_group[
                ~station_group["station_key"].eq(station_key)
            ]
            loso_rows.append(
                {
                    "fold": fold,
                    "seed": int(seed),
                    "max_far": float(max_far),
                    "selection_rule": rule,
                    "left_out_station": station_key,
                    "remaining_stations": int(
                        kept["station_key"].nunique()
                    ),
                    "full_f1_diff": total_f1 - baseline_f1,
                    "loso_f1_diff_mean": float(
                        (
                            kept["test_f1"] - kept["baseline_test_f1"]
                        ).mean()
                    ),
                    "full_nonstuck_diff": (
                        total_nonstuck - baseline_nonstuck
                    ),
                    "loso_nonstuck_diff_mean": float(
                        (
                            kept["test_nonstuck_event_macro"]
                            - kept["baseline_test_nonstuck_event_macro"]
                        ).mean()
                    ),
                }
            )
    loso = pd.DataFrame(loso_rows)
    loso.to_csv(
        TABLES / f"{prefix}_loso.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary = loso[
        loso["selection_rule"].eq("valid_f1")
        & loso["max_far"].eq(PRIMARY_FAR)
    ]
    station_primary = station_summary[
        station_summary["selection_rule"].eq("valid_f1")
        & station_summary["max_far"].eq(PRIMARY_FAR)
    ]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段197对阶段195/196的种子随机化位置结果做站点依赖审计："
            "每个fold/seed仍只在valid选择门控分位、连续长度、窗口、聚合和"
            "权重，并在固定FAR上限下只用valid正常样本校准阈值；test只评价；"
            "不根据test选择；逐站点和留一站点指标只用于诊断，不参与模型、"
            "阈值、窗口、权重或规则选择；本阶段不修改也不替换冻结协议与"
            "冻结主模型；NOAA攻击为透明受控注入，不是真实攻击日志。"
        ),
        "cohort": cohort,
        "detail_rows": int(len(detail)),
        "station_detail_rows": int(len(station_detail)),
        "station_summary_rows": int(len(station_summary)),
        "loso_rows": int(len(loso)),
        "primary_far": PRIMARY_FAR,
        "primary_rule": "valid_f1",
        "primary_station_count": int(
            station_primary["station_key"].nunique()
        ),
        "primary_f1_positive_stations": int(
            (station_primary["f1_diff_mean"] > 0).sum()
        ),
        "primary_f1_negative_stations": int(
            (station_primary["f1_diff_mean"] < 0).sum()
        ),
        "primary_nonstuck_positive_stations": int(
            (station_primary["nonstuck_diff_mean"] > 0).sum()
        ),
        "primary_nonstuck_negative_stations": int(
            (station_primary["nonstuck_diff_mean"] < 0).sum()
        ),
        "loso_f1_positive_rows": int(
            (primary["loso_f1_diff_mean"] > 0).sum()
        ),
        "loso_f1_negative_rows": int(
            (primary["loso_f1_diff_mean"] < 0).sum()
        ),
        "loso_nonstuck_positive_rows": int(
            (primary["loso_nonstuck_diff_mean"] > 0).sum()
        ),
        "loso_nonstuck_negative_rows": int(
            (primary["loso_nonstuck_diff_mean"] < 0).sum()
        ),
        "station_summary": station_primary.to_dict(orient="records"),
        "limitations": [
            "站点依赖审计只覆盖当前NOAA站点集合。",
            "站点级F1在小样本站点上波动较大。",
            "留一站点诊断不替代跨数据源外部验证。",
            "NOAA攻击为透明受控注入，不代表真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段197 NOAA种子随机位置站点依赖审计",
        "",
        f"- 队列：{cohort}",
        f"- 主运行点：valid_f1，{PRIMARY_FAR:.1%} FAR",
        f"- 逐站点F1平均差为正的站点："
        f"{payload['primary_f1_positive_stations']}/"
        f"{payload['primary_station_count']}",
        f"- 逐站点F1平均差为负的站点："
        f"{payload['primary_f1_negative_stations']}/"
        f"{payload['primary_station_count']}",
        f"- 留一站点F1平均差为正的行："
        f"{payload['loso_f1_positive_rows']}/{len(primary)}",
        f"- 留一站点F1平均差为负的行："
        f"{payload['loso_f1_negative_rows']}/{len(primary)}",
        f"- 留一站点非stuck事件覆盖平均差为正的行："
        f"{payload['loso_nonstuck_positive_rows']}/{len(primary)}",
        f"- 留一站点非stuck事件覆盖平均差为负的行："
        f"{payload['loso_nonstuck_negative_rows']}/{len(primary)}",
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
                "station_summary_rows": payload["station_summary_rows"],
                "loso_rows": payload["loso_rows"],
                "primary_f1_positive_stations": payload[
                    "primary_f1_positive_stations"
                ],
                "primary_station_count": payload[
                    "primary_station_count"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--cohort",
        choices=["original", "independent"],
        default="independent",
    )
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
    seed_pool = (
        REPLICATION_SEEDS
        if args.cohort == "original"
        else INDEPENDENT_SEEDS
    )
    selected_seeds = [int(args.seed)] if args.seed is not None else seed_pool
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

    rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in selected_folds:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in selected_seeds:
            frame, _ = build_score_frame(
                source,
                stations,
                frozen,
                train_years,
                valid_year,
                test_year,
                seed,
            )
            current_rows, current_station = evaluate_station_aware(frame)
            rows.extend(current_rows)
            station_rows.extend(current_station)
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "seed": int(seed),
                        "status": "completed",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return summarize_and_write(
        pd.DataFrame(rows),
        pd.DataFrame(station_rows),
        args.cohort,
    )


if __name__ == "__main__":
    raise SystemExit(main())
