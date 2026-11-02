from __future__ import annotations

"""Stage 199: cross-station transfer of persistent-repeat protection.

The NOAA seed-randomized protection audit is internally strong, but it
still uses one data source.  The Liupanshan controlled benchmark has seven
stations and five pre-declared attack-construction seeds.  Stage 124
already provides a strict leave-one-station-out cold-start detector.

This stage asks a different question: can the persistent-repeat protection
choice transfer from the six source stations to the held-out station?

For each seed and held-out station:

* the frozen stage-124 robust risk is used as the base score;
* the same causal ``noise_ratio`` and ``repeat_ratio`` definitions as the
  NOAA protection work are recomputed from the local controlled data;
* the candidate grid is evaluated and one configuration is selected using
  only source-station valid rows;
* the same six source-station valid rows select the protection
  configuration and calibrate the fixed-FAR threshold;
* the held-out station does not participate in normalization, selection,
  or threshold calibration;
* the held-out station's test rows are evaluation-only.

No held-out test label, attack type, event, station, threshold, window,
weight, gate, run length, or feature is used for selection.  The attacks
remain transparent controlled injections, not real attack logs.
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
from stage186_noaa_targeted_nonstuck_feature_response import (  # noqa: E402
    ATTACK_TYPES,
)
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
    FAR_GRID,
    GATE_QUANTILES,
    PRIMARY_FAR,
    RUN_LENGTHS,
    SELECTION_RULES,
    WINDOWS,
    clean_records,
)
from stage186_noaa_targeted_nonstuck_feature_response import (  # noqa: E402
    targeted_features as targeted_feature_frame,
)

SEEDS = [3407, 6257, 7919, 104729, 20260715]
NON_STUCK_TYPES = [name for name in ATTACK_TYPES if name != "stuck"]
PROTECTION_OPTIONS = [(0.0, 0)] + [
    (float(quantile), int(length))
    for quantile in GATE_QUANTILES
    for length in RUN_LENGTHS
]
WEIGHT_COMBINATIONS = candidate_weights()


def local_score_frame(seed: int, held_out: str) -> pd.DataFrame:
    source_path = TABLES / f"stage09c_input_seed{seed}.parquet"
    detail_path = (
        TABLES / "stage124_leave_one_station_refit_multiseed_detail.parquet"
    )
    valid_path = (
        TABLES
        / "stage124_leave_one_station_refit_multiseed_valid_detail.parquet"
    )
    if not source_path.is_file() or not detail_path.is_file():
        raise FileNotFoundError(
            "stage199 requires stage09c inputs and stage124 detail"
        )
    source = pd.read_parquet(source_path).copy()
    source["timestamp_utc"] = pd.to_datetime(
        source["timestamp_utc"], utc=True, errors="coerce"
    )
    source["station_key"] = source["station_key"].astype(str)
    source = source.sort_values(
        ["station_key", "timestamp_utc", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    base_attack = source.get("is_anomaly", False).astype(bool)
    extension_attack = source.get("extension_is_attack", False).astype(bool)
    extension_value = pd.to_numeric(
        source.get("extension_observed_value"), errors="coerce"
    )
    observed = pd.to_numeric(source["observed_value"], errors="coerce")
    source["effective_observed_value"] = observed.where(
        ~extension_attack | extension_value.isna(), extension_value
    )
    source["is_attack"] = base_attack | extension_attack
    extension_type = (
        source.get("extension_attack_type", "normal")
        .fillna("normal")
        .astype(str)
    )
    base_type = (
        source.get("anomaly_type", "normal").fillna("normal").astype(str)
    )
    source["attack_type"] = np.where(
        extension_attack, extension_type, base_type
    )
    extension_event = (
        source.get("extension_event_id", "").fillna("").astype(str)
    )
    base_event = (
        source.get("anomaly_record_id", "").fillna("").astype(str)
    )
    source["event_id"] = np.where(
        extension_attack & extension_event.ne(""), extension_event, base_event
    )
    source["observed_value"] = source["effective_observed_value"]
    features = targeted_feature_frame(source)
    normal_train = (
        source["split"].eq("train").to_numpy()
        & ~source["is_attack"].to_numpy()
        & ~source["station_key"].astype(str).eq(str(held_out)).to_numpy()
    )
    reference = features.loc[normal_train]
    for column in ["noise_ratio", "repeat_ratio"]:
        center = float(reference[column].median())
        values = features[column].to_numpy(dtype=float)
        scale = max(
            1.4826
            * float(
                np.median(
                    np.abs(reference[column].to_numpy(dtype=float) - center)
                )
            ),
            0.25,
        )
        source[column] = np.maximum((values - center) / scale, 0.0)
    source["timestamp_utc"] = pd.to_datetime(
        source["timestamp_utc"], utc=True, errors="coerce"
    )
    detail = pd.read_parquet(detail_path)
    valid = pd.read_parquet(valid_path)
    detail = detail[detail["seed"].eq(int(seed))].copy()
    valid = valid[valid["seed"].eq(int(seed))].copy()
    detail = detail.rename(columns={"robust_fused_risk": "base_score"})
    valid = valid.rename(columns={"robust_fused_risk": "base_score"})
    detail = detail[
        [
            "held_out_station",
            "station_key",
            "timestamp_utc",
            "combined_attack_type",
            "combined_is_attack",
            "combined_event_id",
            "base_score",
        ]
    ].copy()
    valid = valid[
        [
            "held_out_station",
            "station_key",
            "timestamp_utc",
            "combined_attack_type",
            "combined_is_attack",
            "combined_event_id",
            "base_score",
        ]
    ].copy()
    detail["split"] = "test"
    valid["split"] = "valid"
    joined = pd.concat([valid, detail], ignore_index=True)
    joined["timestamp_utc"] = pd.to_datetime(
        joined["timestamp_utc"], utc=True, errors="coerce"
    )
    joined = joined.rename(
        columns={
            "combined_attack_type": "attack_type",
            "combined_is_attack": "is_attack",
            "combined_event_id": "event_id",
        }
    )
    feature_lookup = source[
        ["station_key", "timestamp_utc", "noise_ratio", "repeat_ratio"]
    ].drop_duplicates(["station_key", "timestamp_utc"])
    joined = joined.merge(
        feature_lookup,
        on=["station_key", "timestamp_utc"],
        how="left",
        validate="many_to_one",
    )
    if joined.empty:
        raise ValueError(f"no stage124 rows for seed {seed}")
    if joined[["base_score", "noise_ratio", "repeat_ratio"]].isna().any().any():
        raise ValueError(f"missing stage199 scores for seed {seed}")
    return joined


def prepare_context(
    frame: pd.DataFrame,
    held_out: str,
    seed: int,
) -> dict[str, object]:
    frame = frame.sort_values(
        ["station_key", "timestamp_utc"], kind="stable"
    ).reset_index(drop=True)
    source_mask = ~frame["station_key"].astype(str).eq(str(held_out)).to_numpy()
    held_mask = frame["station_key"].astype(str).eq(str(held_out)).to_numpy()
    source_valid = source_mask & frame["split"].eq("valid").to_numpy()
    source_test = source_mask & frame["split"].eq("test").to_numpy()
    held_test = held_mask & frame["split"].eq("test").to_numpy()
    for name, mask in [
        ("source_valid", source_valid),
        ("source_test", source_test),
        ("held_test", held_test),
    ]:
        if not mask.any():
            raise ValueError(f"{name} is empty for {held_out} seed {seed}")
    return {
        "frame": frame,
        "source_valid": source_valid,
        "source_test": source_test,
        "held_test": held_test,
        "seed": int(seed),
        "held_out": str(held_out),
    }


def candidate_table(context: dict[str, object]) -> pd.DataFrame:
    frame = context["frame"]
    source_valid = context["source_valid"]
    source_test = context["source_test"]
    y_valid = frame.loc[source_valid, "is_attack"].astype(bool).to_numpy()
    y_test = frame.loc[source_test, "is_attack"].astype(bool).to_numpy()
    valid_frame = frame.loc[source_valid].reset_index(drop=True)
    test_frame = frame.loc[source_test].reset_index(drop=True)
    base_valid = frame.loc[source_valid, "base_score"].to_numpy(dtype=float)
    base_test = frame.loc[source_test, "base_score"].to_numpy(dtype=float)
    gate_valid = frame.loc[source_valid, "repeat_ratio"].to_numpy(dtype=float)
    thresholds = gate_thresholds(gate_valid, ~y_valid)
    runs = build_run_lengths(frame, thresholds)
    rolling = build_rolling_scores(frame)
    rows: list[dict[str, object]] = []
    for max_far in FAR_GRID:
        for gate_quantile, run_length in PROTECTION_OPTIONS:
            protected_valid = (
                runs[gate_quantile][source_valid] >= run_length
                if gate_quantile > 0.0
                else np.zeros(len(base_valid), dtype=bool)
            )
            for window in WINDOWS:
                for aggregation in AGGREGATIONS:
                    key = (window, aggregation)
                    for base_weight, candidate_weight in WEIGHT_COMBINATIONS:
                        valid_score = base_valid.copy()
                        if candidate_weight > 0.0:
                            blend = (
                                (1.0 - candidate_weight) * base_valid
                                + candidate_weight * rolling[key][source_valid]
                            )
                            valid_score = np.where(
                                protected_valid, base_valid, blend
                            )
                        threshold = valid_far_threshold(
                            valid_score, ~y_valid, max_far
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
                                "aggregation": str(aggregation),
                                "base_weight": float(base_weight),
                                "candidate_weight": float(candidate_weight),
                                "source_valid_f1": float(
                                    valid_metrics["f1"]
                                ),
                                "source_valid_event_macro": (
                                    macro_event_coverage(
                                        valid_frame,
                                        valid_pred,
                                        ATTACK_TYPES,
                                    )
                                ),
                                "source_valid_nonstuck_event_macro": (
                                    macro_event_coverage(
                                        valid_frame,
                                        valid_pred,
                                        NON_STUCK_TYPES,
                                    )
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def evaluate_selection(
    context: dict[str, object],
    selection: dict[str, object],
) -> dict[str, object]:
    frame = context["frame"]
    source_valid = context["source_valid"]
    held_test = context["held_test"]
    y_valid = frame.loc[source_valid, "is_attack"].astype(bool).to_numpy()
    y_test = frame.loc[held_test, "is_attack"].astype(bool).to_numpy()
    valid_frame = frame.loc[source_valid].reset_index(drop=True)
    test_frame = frame.loc[held_test].reset_index(drop=True)
    base_valid = frame.loc[source_valid, "base_score"].to_numpy(dtype=float)
    base_test = frame.loc[held_test, "base_score"].to_numpy(dtype=float)
    gate_valid = frame.loc[source_valid, "repeat_ratio"].to_numpy(dtype=float)
    thresholds = gate_thresholds(gate_valid, ~y_valid)
    runs = build_run_lengths(frame, thresholds)
    rolling = build_rolling_scores(frame)
    max_far = float(selection["max_far"])
    candidate_weight = float(selection["candidate_weight"])
    gate_quantile = float(selection["gate_quantile"])
    run_length = int(selection["run_length"])
    key = (int(selection["window"]), str(selection["aggregation"]))
    protected_valid = (
        runs[gate_quantile][source_valid] >= run_length
        if gate_quantile > 0.0
        else np.zeros(len(base_valid), dtype=bool)
    )
    protected_test = (
        runs[gate_quantile][held_test] >= run_length
        if gate_quantile > 0.0
        else np.zeros(len(base_test), dtype=bool)
    )
    valid_score = base_valid.copy()
    test_score = base_test.copy()
    if candidate_weight > 0.0:
        valid_score = np.where(
            protected_valid,
            base_valid,
            (1.0 - candidate_weight) * base_valid
            + candidate_weight * rolling[key][source_valid],
        )
        test_score = np.where(
            protected_test,
            base_test,
            (1.0 - candidate_weight) * base_test
            + candidate_weight * rolling[key][held_test],
        )
    baseline_threshold = valid_far_threshold(
        base_valid, ~y_valid, max_far
    )
    baseline_pred = base_test >= baseline_threshold
    transferred_threshold = valid_far_threshold(
        valid_score, ~y_valid, max_far
    )
    transferred_pred = test_score >= transferred_threshold
    transferred = _metrics(y_test, transferred_pred, test_score)
    baseline = _metrics(y_test, baseline_pred, base_test)
    return {
        "test_f1": float(transferred["f1"]),
        "test_recall": float(transferred["recall"]),
        "test_precision": float(transferred["precision"]),
        "test_auprc": float(transferred["average_precision"]),
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
        "baseline_test_f1": float(baseline["f1"]),
        "baseline_test_recall": float(baseline["recall"]),
        "baseline_test_precision": float(baseline["precision"]),
        "baseline_test_auprc": float(baseline["average_precision"]),
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


def summarize(
    detail: pd.DataFrame,
    selections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate = (
        detail.groupby(["selection_rule", "max_far"], sort=True)
        .agg(
            fold_seed_pairs=("held_out_station", "size"),
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
    return aggregate, pd.DataFrame(paired_rows), pd.DataFrame()


def write_outputs(
    detail: pd.DataFrame,
    selections: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
) -> int:
    prefix = "stage199_liupanshan_cross_station_protection_transfer"
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
    station_summary = (
        detail.groupby(
            ["held_out_station", "selection_rule", "max_far"],
            sort=True,
        )
        .agg(
            seed_count=("seed", "nunique"),
            test_f1_mean=("test_f1", "mean"),
            baseline_test_f1_mean=("baseline_test_f1", "mean"),
            test_nonstuck_event_macro_mean=(
                "test_nonstuck_event_macro",
                "mean",
            ),
            baseline_test_nonstuck_event_macro_mean=(
                "baseline_test_nonstuck_event_macro",
                "mean",
            ),
            protected_seed_count=(
                "run_length",
                lambda values: int((values > 0).sum()),
            ),
        )
        .reset_index()
    )
    station_summary["test_f1_diff_mean"] = (
        station_summary["test_f1_mean"]
        - station_summary["baseline_test_f1_mean"]
    )
    station_summary["test_nonstuck_diff_mean"] = (
        station_summary["test_nonstuck_event_macro_mean"]
        - station_summary["baseline_test_nonstuck_event_macro_mean"]
    )
    station_summary.to_csv(
        TABLES / f"{prefix}_station_summary.csv",
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
    f1 = primary_paired[primary_paired["metric"].eq("test_f1")].iloc[0]
    nonstuck = primary_paired[
        primary_paired["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    payload = {
        "status": "NEGATIVE_RESULT_RETAINED",
        "protocol": (
            "阶段199在六盘山受控攻击数据上做跨站点保护配置迁移：对每个"
            "攻击构造种子和留出站点，只在其余六个源站点的valid结果上"
            "选择一个固定门控分位、连续长度、窗口、聚合和权重，并在"
            "同一组源站点valid正常样本上按固定FAR上限校准阈值；留出"
            "站点不参与特征标准化、配置选择或阈值校准；留出站点test只"
            "评价；不根据test选择；不修改也不替换冻结主模型；攻击为"
            "透明受控注入，不是真实攻击日志。"
        ),
        "seeds": SEEDS,
        "stations": sorted(detail["held_out_station"].unique().tolist()),
        "primary_far": PRIMARY_FAR,
        "primary_rule": "valid_f1",
        "primary_f1_diff": float(f1["mean_diff"]),
        "primary_f1_positive_pairs": int(f1["positive_pairs"]),
        "primary_f1_negative_pairs": int(f1["negative_pairs"]),
        "primary_nonstuck_diff": float(nonstuck["mean_diff"]),
        "primary_nonstuck_positive_pairs": int(
            nonstuck["positive_pairs"]
        ),
        "primary_nonstuck_negative_pairs": int(
            nonstuck["negative_pairs"]
        ),
        "primary_protected_pairs": int(primary["protected_pairs"]),
        "primary_pairs": int(primary["fold_seed_pairs"]),
        "detail_rows": int(len(detail)),
        "selection_rows": int(len(selections)),
        "aggregate_rows": int(len(aggregate)),
        "paired_rows": int(len(paired)),
        "station_summary_rows": int(len(station_summary)),
        "aggregate": clean_records(aggregate),
        "paired": clean_records(paired),
        "station_summary": clean_records(station_summary),
        "limitations": [
            "六盘山只有单一时段、单一地区的受控攻击数据。",
            "五个攻击构造种子不等于五个独立地区或年份。",
            "跨站点迁移不替代真实攻击日志或跨数据源外部验证。",
            "攻击为透明受控注入，不代表真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段199 六盘山跨站点保护配置迁移",
        "",
        "源站点valid选择固定保护配置并校准FAR阈值；留出站点不参与标准化、选择或校准；留出站点test只评价。",
        "",
        "| 选择规则 | FAR上限 | test F1 | Recall | FAR | 非stuck事件覆盖 | F1平均差 | 正向/负向 | 采用保护 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        current = paired[
            paired["selection_rule"].eq(row["selection_rule"])
            & paired["max_far"].eq(row["max_far"])
            & paired["metric"].eq("test_f1")
        ].iloc[0]
        lines.append(
            f"| {row['selection_rule']} | {row['max_far']:.3f} | "
            f"{row['test_f1_mean']:.6f} | {row['test_recall_mean']:.6f} | "
            f"{row['test_false_alarm_rate_mean']:.6f} | "
            f"{row['test_nonstuck_event_macro_mean']:.6f} | "
            f"{current['mean_diff']:+.6f} | "
            f"{current['positive_pairs']}/{current['negative_pairs']} | "
            f"{int(row['protected_pairs'])}/{int(row['fold_seed_pairs'])} |"
        )
    lines += [
        "",
        f"1%FAR下valid_f1规则：F1平均差={f1['mean_diff']:+.6f}，"
        f"非stuck事件覆盖平均差={nonstuck['mean_diff']:+.6f}，"
        f"采用保护={int(primary['protected_pairs'])}/"
        f"{int(primary['fold_seed_pairs'])}。",
        "",
        "该结果是保留的负结果：连续重复保护配置从源站点迁移到留出站点后，"
        "1%FAR下F1和非stuck覆盖平均均略为负，不能宣称跨站点稳定增益。",
        "",
        "该结果是受控攻击数据上的跨站点配置迁移审计，不是真实攻击日志验证。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": payload["detail_rows"],
                "selection_rows": payload["selection_rows"],
                "aggregate_rows": payload["aggregate_rows"],
                "paired_rows": payload["paired_rows"],
                "f1_diff": float(f1["mean_diff"]),
                "nonstuck_diff": float(nonstuck["mean_diff"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--station", type=str, default=None)
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize_existing:
        prefix = "stage199_liupanshan_cross_station_protection_transfer"
        detail = pd.read_csv(TABLES / f"{prefix}_detail.csv")
        selections = pd.read_csv(TABLES / f"{prefix}_selections.csv")
        aggregate, paired, _ = summarize(detail, selections)
        return write_outputs(detail, selections, aggregate, paired)
    rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for seed in [args.seed] if args.seed is not None else SEEDS:
        all_source = pd.read_parquet(
            TABLES / f"stage09c_input_seed{seed}.parquet"
        )
        stations = sorted(
            all_source["station_key"].astype(str).unique().tolist()
        )
        if args.station is not None:
            stations = [args.station]
        for station in stations:
            frame = local_score_frame(int(seed), station)
            current = frame.copy()
            context = prepare_context(current, station, int(seed))
            candidates = candidate_table(context)
            for rule in SELECTION_RULES:
                for max_far in FAR_GRID:
                    selected = None
                    selected_key = None
                    for row in candidates[
                        candidates["max_far"].eq(float(max_far))
                    ].to_dict(orient="records"):
                        key = selection_key(
                            rule,
                            {"f1": float(row["source_valid_f1"])},
                            float(row["source_valid_event_macro"]),
                            float(
                                row[
                                    "source_valid_nonstuck_event_macro"
                                ]
                            ),
                            float(row["candidate_weight"]),
                            int(row["window"]),
                            float(row["gate_quantile"]),
                            int(row["run_length"]),
                        )
                        if selected_key is None or key > selected_key:
                            selected_key = key
                            selected = row
                    if selected is None:
                        raise RuntimeError("no candidate selected")
                    metrics = evaluate_selection(context, selected)
                    rows.append(
                        {
                            "seed": int(seed),
                            "held_out_station": station,
                            "selection_rule": rule,
                            "max_far": float(max_far),
                            **selected,
                            **metrics,
                        }
                    )
                    selection_rows.append(
                        {
                            "seed": int(seed),
                            "held_out_station": station,
                            "selection_rule": rule,
                            **selected,
                        }
                    )
            print(
                json.dumps(
                    {
                        "seed": int(seed),
                        "held_out_station": station,
                        "status": "completed",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    detail = pd.DataFrame(rows)
    selections = pd.DataFrame(selection_rows)
    aggregate, paired, _ = summarize(detail, selections)
    return write_outputs(detail, selections, aggregate, paired)


if __name__ == "__main__":
    raise SystemExit(main())
