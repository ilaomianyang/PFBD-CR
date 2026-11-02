from __future__ import annotations

"""Stage 196: independent-seed replication of stage 195.

Stage 195 showed that seed-randomized attack placement does not destroy the
stage-193 persistent-repeat candidate.  That result still used the same five
seeds that appear throughout the frozen protocol.  This stage freezes a new
seed set before evaluation and repeats the stage-195 pipeline without any
test-driven change.

The goal is replication, not model selection.  A candidate is supported only
if the direction of the validation-selected result survives on the new seeds.
The frozen protocol and frozen main model are not modified or replaced.
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

from stage195_noaa_seed_randomized_placement import (  # noqa: E402
    FOLDS,
    PRIMARY_FAR,
    SELECTION_RULES,
    build_score_frame,
    clean_records,
    evaluate_group,
    fold_name,
    resolve_fold,
)

REPLICATION_SEEDS = [15485863, 32452843, 49979687, 67867967, 86028121]
METHOD = "TrustMeteo-QC frozen expanded"


def summarize_and_write(detail: pd.DataFrame, placements: pd.DataFrame) -> int:
    prefix = "stage196_noaa_seed_randomized_independent_replication"
    detail.to_csv(
        TABLES / f"{prefix}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    placements.to_csv(
        TABLES / f"{prefix}_placement_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate = (
        detail.groupby(["selection_rule", "max_far"], sort=True)
        .agg(
            fold_seed_pairs=("fold", "size"),
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
        for max_far in sorted(detail["max_far"].unique()):
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
    placement_summary = (
        placements.groupby(["fold", "split", "attack_type"], sort=True)
        .agg(
            seeds=("seed", "nunique"),
            distinct_starts=("absolute_start", "nunique"),
            min_start=("absolute_start", "min"),
            max_start=("absolute_start", "max"),
        )
        .reset_index()
    )
    placement_summary.to_csv(
        TABLES / f"{prefix}_placement_summary.csv",
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
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段196是阶段195的独立种子复现：在新的一组预先冻结种子上"
            "重复种子随机化攻击位置协议；每个fold/seed把每个split按"
            "攻击类型划分成预先固定的连续区块，每个攻击在自己的区块内"
            "按种子均匀随机选择起点；复用阶段193的连续重复保护候选空间；"
            "每个fold/seed只在valid选择门控分位、连续长度、窗口、聚合和"
            "权重，并在固定FAR上限下只用valid正常样本校准阈值；test只评价，"
            "不参与位置、门控、窗口、聚合、权重、阈值或规则选择；"
            "不根据test选择；本阶段不修改也不替换冻结协议与冻结主模型；"
            "NOAA攻击为透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [
            "stage82_noaa_extended_normalized.parquet",
            "stage141_noaa_frozen_external_validation_selected.csv",
        ],
        "replication_seeds": REPLICATION_SEEDS,
        "folds": [fold_name(*fold) for fold in FOLDS],
        "detail_rows": int(len(detail)),
        "placement_detail_rows": int(len(placements)),
        "placement_summary_rows": int(len(placement_summary)),
        "aggregate_rows": int(len(aggregate)),
        "paired_rows": int(len(paired)),
        "aggregate": clean_records(aggregate),
        "paired": clean_records(paired),
        "placement_summary": clean_records(placement_summary),
        "limitations": [
            "阶段196只复现当前NOAA站点、年份和受控注入生成器。",
            "新种子减少偶然性，但不能替代独立数据源验证。",
            "候选仍不替换冻结主模型。",
            "NOAA攻击为透明受控注入，不代表真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段196 NOAA种子随机位置独立复现",
        "",
        "使用预先冻结的新种子重复阶段195的种子随机化攻击位置协议，"
        "valid选择、test只评价。",
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
        "NOAA攻击为透明受控注入，不代表真实攻击日志。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": payload["detail_rows"],
                "placement_detail_rows": payload["placement_detail_rows"],
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
    parser.add_argument("--fold", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild summaries from persisted stage-196 detail files.",
    )
    args = parser.parse_args(argv)
    prefix = "stage196_noaa_seed_randomized_independent_replication"
    if args.summarize_existing:
        detail = pd.read_csv(TABLES / f"{prefix}_detail.csv")
        placements = pd.read_csv(TABLES / f"{prefix}_placement_detail.csv")
        if args.fold is not None:
            fold_index = resolve_fold(args.fold)
            name = fold_name(*FOLDS[fold_index])
            detail = detail[detail["fold"].eq(name)]
            placements = placements[placements["fold"].eq(name)]
        if args.seed is not None:
            detail = detail[detail["seed"].eq(int(args.seed))]
            placements = placements[placements["seed"].eq(int(args.seed))]
        if detail.empty:
            raise ValueError("no detail rows matched the requested filter")
        return summarize_and_write(detail, placements)

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
        [int(args.seed)] if args.seed is not None else REPLICATION_SEEDS
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
        raise ValueError("missing frozen alpha/window for replication seeds")

    rows: list[dict[str, object]] = []
    placement_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in selected_folds:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in selected_seeds:
            frame, placements = build_score_frame(
                source,
                stations,
                frozen,
                train_years,
                valid_year,
                test_year,
                seed,
            )
            rows.extend(evaluate_group(frame))
            placement_rows.extend(placements)
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
        pd.DataFrame(rows), pd.DataFrame(placement_rows)
    )


if __name__ == "__main__":
    raise SystemExit(main())
