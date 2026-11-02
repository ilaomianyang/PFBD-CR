from __future__ import annotations

"""Stage 194: attack-position perturbation robustness for stage 193.

Stage 193 produced the strongest low-FAR reranking candidate so far: a
causal run-length ``repeat_ratio`` protection that kept the stage-190
non-stuck event-coverage gain while recovering the ``stuck`` row-level
recall.  A follow-up decomposition exposed a real limitation of the
existing controlled protocol:

* ``choose_start`` places each attack type at a fixed fraction of the
  split, independent of the seed;
* the frozen ``alpha``/``window`` settings are identical across the five
  seeds.

Consequently the twenty fold/seed pairs contain only four independent
attack placements, and five of the six attack types have no seed-driven
variation at all.  This stage does not modify the frozen protocol.  It
adds a pre-declared sensitivity axis: the same fixed-fraction placement
rule is shifted by a small deterministic offset, the whole pipeline is
rebuilt from the raw NOAA observations at each offset, and the stage-193
candidate is re-selected on valid and evaluated on test.

Offsets are pre-declared and symmetric, no offset is chosen from test, no
model is retrained, and the frozen protocol is not replaced.  NOAA attacks
remain transparent controlled injections, not real attack logs.
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
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"
AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
FROZEN = TABLES / "stage141_noaa_frozen_external_validation_selected.csv"
STAGE187_SCORE = (
    TABLES
    / "stage187_noaa_targeted_features_validation_calibrated_score_detail.parquet"
)

SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
BASE_FRACTIONS = [0.12, 0.26, 0.40, 0.54, 0.68, 0.82]
POSITION_SHIFTS = [-0.06, -0.03, 0.0, 0.03, 0.06]
WINDOWS = [2, 3, 6]
AGGREGATIONS = ["rolling_max", "rolling_mean"]
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]
GATE_QUANTILES = [0.75, 0.9, 0.95]
RUN_LENGTHS = [2, 3]
FAR_GRID = [0.005, 0.01, 0.02, 0.05]
PRIMARY_FAR = 0.01
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
METHOD = "TrustMeteo-QC frozen expanded"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import stable_seed  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components  # noqa: E402
from stage186_noaa_targeted_nonstuck_feature_response import (  # noqa: E402
    FEATURE_NAMES,
    fold_name,
    normalize_features,
    targeted_features,
)
from stage193_noaa_persistent_repeat_protection import (  # noqa: E402
    build_run_lengths,
    build_rolling_scores,
    candidate_weights,
    causal_roll,
    clean_records,
    gate_thresholds,
    macro_event_coverage,
    run_length_above,
    selection_key,
    type_metrics,
    valid_far_threshold,
)

WEIGHT_COMBINATIONS = candidate_weights()
PROTECTION_OPTIONS = [(0.0, 0)] + [
    (float(quantile), int(length))
    for quantile in GATE_QUANTILES
    for length in RUN_LENGTHS
]


def resolve_fold(value: str | None) -> int | None:
    if value is None:
        return None
    if value.isdigit():
        index = int(value)
        if 0 <= index < len(FOLDS):
            return index
    for index, (train_years, valid_year, test_year) in enumerate(FOLDS):
        if value == fold_name(train_years, valid_year, test_year):
            return index
    raise ValueError(
        f"unknown fold {value!r}; use 0-{len(FOLDS) - 1} or a fold name"
    )


def shifted_fractions(shift: float) -> list[float]:
    values = [min(max(value + shift, 0.02), 0.94) for value in BASE_FRACTIONS]
    if values != sorted(values):
        raise ValueError("position shift produced unordered fractions")
    return values


def shifted_start(
    group: pd.DataFrame, type_index: int, length: int, fractions: list[float]
) -> int:
    if len(group) < length + 4:
        raise ValueError("NOAA split too short for controlled attack")
    position = (
        int(round((len(group) - length - 2) * fractions[type_index])) + 1
    )
    return max(1, min(position, len(group) - length - 1))


def build_fold_shifted(
    source: pd.DataFrame,
    seed: int,
    train_years: list[int],
    valid_year: int,
    test_year: int,
    stations: set[str],
    shift: float,
) -> pd.DataFrame:
    all_years = [*train_years, valid_year, test_year]
    raw = source[
        source["station_id"].astype(str).isin(stations)
        & source["year"].isin(all_years)
    ].copy()
    raw = raw.dropna(subset=["timestamp_utc", "temperature_c"])
    raw["timestamp_utc"] = pd.to_datetime(
        raw["timestamp_utc"], utc=True, errors="coerce"
    )
    raw = raw.sort_values(
        ["station_id", "timestamp_utc"], kind="stable"
    ).reset_index(drop=True)
    raw = raw.rename(
        columns={"station_id": "station_key", "temperature_c": "original_value"}
    )
    raw["observed_value"] = raw["original_value"].astype(float)
    raw["effective_observed_value"] = raw["observed_value"]
    raw["combined_is_attack"] = False
    raw["combined_attack_type"] = "normal"
    raw["combined_event_id"] = raw.index.astype(str)
    raw["missing_observation"] = False
    split_map = {year: "train" for year in train_years}
    split_map.update({valid_year: "valid", test_year: "test"})
    raw["split"] = raw["year"].map(split_map)
    if raw["split"].isna().any():
        raise ValueError("unexpected year in rolling fold")
    fractions = shifted_fractions(shift)
    for split_name in ["valid", "test"]:
        for station, group in raw[
            raw["split"].eq(split_name)
        ].groupby("station_key", sort=True):
            idx = group.index.to_numpy()
            for type_index, attack_type in enumerate(ATTACK_TYPES):
                length = 12 if attack_type == "drift" else 8
                start = shifted_start(group, type_index, length, fractions)
                event_idx = idx[start : start + length]
                clean = raw.loc[event_idx, "original_value"].to_numpy(
                    dtype=float
                )
                rng = np.random.default_rng(
                    stable_seed(
                        seed,
                        *train_years,
                        valid_year,
                        test_year,
                        split_name,
                        station,
                        attack_type,
                    )
                )
                if attack_type == "spike":
                    values = clean + 5.0
                elif attack_type == "offset":
                    values = clean + 2.5
                elif attack_type == "drift":
                    values = clean + np.linspace(0.2, 2.4, length)
                elif attack_type == "time_shift":
                    prior = group["original_value"].to_numpy(dtype=float)
                    values = prior[start - 1 : start - 1 + length]
                elif attack_type == "stuck":
                    values = np.repeat(clean[0], length)
                else:
                    values = clean + rng.normal(0.0, 1.0, size=length)
                raw.loc[event_idx, "observed_value"] = values
                raw.loc[event_idx, "effective_observed_value"] = values
                raw.loc[event_idx, "combined_is_attack"] = True
                raw.loc[event_idx, "combined_attack_type"] = attack_type
                raw.loc[
                    event_idx, "combined_event_id"
                ] = f"{split_name}_{station}_{attack_type}_shift{shift:+.2f}"
    return raw


def build_score_frame(
    source: pd.DataFrame,
    stations: set[str],
    frozen: pd.DataFrame,
    train_years: list[int],
    valid_year: int,
    test_year: int,
    seed: int,
    shift: float,
) -> pd.DataFrame:
    fold = fold_name(train_years, valid_year, test_year)
    alpha = float(frozen.loc[(fold, seed), "alpha_base"])
    window = int(frozen.loc[(fold, seed), "window"])
    frame = build_fold_shifted(
        source,
        seed,
        train_years,
        valid_year,
        test_year,
        stations,
        shift,
    )
    scored, _ = score(frame)
    scored = add_components(scored)
    features = targeted_features(scored)
    train = scored["split"].eq("train").to_numpy()
    normal_train = train & (
        ~scored["combined_is_attack"].astype(bool).to_numpy()
    )
    normalized = normalize_features(features, normal_train)
    base = scored["base_fused_risk"].to_numpy(dtype=float)
    structural = scored["structural_risk"].to_numpy(dtype=float)
    raw_risk = alpha * base + (1.0 - alpha) * structural
    base_score = np.full(len(scored), np.nan, dtype=float)
    for split_name in ["train", "valid", "test"]:
        split_mask = scored["split"].eq(split_name).to_numpy()
        base_score[split_mask] = smooth_risk(
            scored.loc[split_mask], raw_risk[split_mask], window
        )
    keep = scored["split"].isin(["valid", "test"]).to_numpy()
    part = scored.loc[keep].copy()
    part["fold"] = fold
    part["seed"] = int(seed)
    part["position_shift"] = float(shift)
    part["base_score"] = base_score[keep]
    part["noise_ratio"] = normalized["noise_ratio"].to_numpy(dtype=float)[
        keep
    ]
    part["repeat_ratio"] = normalized["repeat_ratio"].to_numpy(dtype=float)[
        keep
    ]
    part["is_attack"] = part["combined_is_attack"].astype(bool)
    part["attack_type"] = part["combined_attack_type"].astype(str)
    part["event_id"] = part["combined_event_id"].astype(str)
    return part[
        [
            "fold",
            "seed",
            "position_shift",
            "split",
            "station_key",
            "timestamp_utc",
            "base_score",
            "noise_ratio",
            "repeat_ratio",
            "is_attack",
            "attack_type",
            "event_id",
        ]
    ].reset_index(drop=True)


def evaluate_group(frame: pd.DataFrame) -> list[dict[str, object]]:
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

    fold = str(frame["fold"].iloc[0])
    seed = int(frame["seed"].iloc[0])
    shift = float(frame["position_shift"].iloc[0])
    rows: list[dict[str, object]] = []
    for max_far in FAR_GRID:
        baseline_threshold = valid_far_threshold(
            base_valid, valid_normal, max_far
        )
        baseline_pred = base_test >= baseline_threshold
        baseline_metrics = _metrics(y_test, baseline_pred, base_test)
        baseline_row = {
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
                        valid_event_macro = macro_event_coverage(
                            valid_frame, valid_pred, ATTACK_TYPES
                        )
                        valid_nonstuck_macro = macro_event_coverage(
                            valid_frame, valid_pred, NON_STUCK_TYPES
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
                            "valid_event_macro": valid_event_macro,
                            "valid_nonstuck_event_macro": valid_nonstuck_macro,
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
            row: dict[str, object] = {
                "fold": fold,
                "seed": seed,
                "position_shift": shift,
                "max_far": float(max_far),
                "selection_rule": rule,
                "gate_quantile": float(selected_stat["gate_quantile"]),
                "run_length": int(selected_stat["run_length"]),
                "window": int(selected_stat["window"]),
                "aggregation": str(selected_stat["aggregation"]),
                "base_weight": float(selected_stat["base_weight"]),
                "candidate_weight": float(selected_stat["candidate_weight"]),
                "threshold_valid": threshold,
                "valid_f1": float(selected_stat["valid_metrics"]["f1"]),
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
                "baseline_test_f1": float(baseline_row["test_f1"]),
                "baseline_test_recall": float(baseline_row["test_recall"]),
                "baseline_test_precision": float(
                    baseline_row["test_precision"]
                ),
                "baseline_test_auprc": float(baseline_row["test_auprc"]),
                "baseline_test_false_alarm_rate": float(
                    baseline_row["test_false_alarm_rate"]
                ),
                "baseline_test_event_macro": float(
                    baseline_row["test_event_macro"]
                ),
                "baseline_test_nonstuck_event_macro": float(
                    baseline_row["test_nonstuck_event_macro"]
                ),
            }
            for current_type, metrics in type_metrics(
                test_frame, test_pred
            ).items():
                row[f"{current_type}_attack_rows"] = metrics["attack_rows"]
                row[f"{current_type}_recall"] = metrics["attack_recall"]
                row[f"{current_type}_event_coverage"] = metrics[
                    "event_coverage"
                ]
            rows.append(row)
    return rows


def redundancy_audit() -> pd.DataFrame:
    scores = pd.read_parquet(
        STAGE187_SCORE,
        columns=[
            "fold",
            "seed",
            "feature",
            "split",
            "station_key",
            "timestamp_utc",
            "is_attack",
            "attack_type",
        ],
        filters=[("feature", "==", "noise_ratio"), ("split", "==", "test")],
    )
    attack = scores[scores["is_attack"].astype(bool)].copy()
    attack["cell"] = (
        attack["station_key"].astype(str)
        + "|"
        + attack["timestamp_utc"].astype(str)
    )
    rows: list[dict[str, object]] = []
    for (fold, attack_type), group in attack.groupby(
        ["fold", "attack_type"], sort=True
    ):
        per_seed = {
            int(seed): frozenset(part["cell"])
            for seed, part in group.groupby("seed", sort=True)
        }
        reference = next(iter(per_seed.values()))
        rows.append(
            {
                "fold": fold,
                "attack_type": attack_type,
                "seed_count": int(len(per_seed)),
                "distinct_placements": int(len(set(per_seed.values()))),
                "identical_to_first": int(
                    sum(1 for cells in per_seed.values() if cells == reference)
                ),
                "attack_rows_per_seed": int(len(reference)),
            }
        )
    return pd.DataFrame(rows)


def summarize_and_write(detail: pd.DataFrame) -> int:
    prefix = "stage194_noaa_position_perturbation_robustness"
    detail.to_csv(
        TABLES / f"{prefix}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate = (
        detail.groupby(
            ["selection_rule", "max_far", "position_shift"], sort=True
        )
        .agg(
            folds=("fold", "size"),
            test_f1_mean=("test_f1", "mean"),
            test_recall_mean=("test_recall", "mean"),
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
            protected_folds=(
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
                        "folds": int(len(current)),
                        "positive_folds": int((diff > 0).sum()),
                        "negative_folds": int((diff < 0).sum()),
                        "mean_diff": float(diff.mean()),
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
    redundancy = redundancy_audit()
    redundancy.to_csv(
        TABLES / f"{prefix}_placement_redundancy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary = aggregate[
        aggregate["selection_rule"].eq("valid_f1")
        & aggregate["max_far"].eq(PRIMARY_FAR)
    ].sort_values("position_shift")
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
            "复用阶段193的连续重复保护候选空间；攻击位置在预先固定的"
            "对称偏移网格-0.06/-0.03/0/+0.03/+0.06上重建整条流程；"
            "每个fold/偏移只在valid选择门控分位、连续长度、窗口、"
            "聚合和权重，并在固定FAR上限下只用valid正常样本校准阈值；"
            "test只评价，不参与偏移、门控、窗口、聚合、权重、阈值或"
            "规则选择；不根据test选择；本阶段是位置敏感性审计，不修改"
            "也不替换冻结协议与冻结主模型；五种子仅用于不确定性描述；"
            "NOAA攻击为透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [NOAA.name, FROZEN.name, STAGE187_SCORE.name],
        "position_shifts": POSITION_SHIFTS,
        "base_fractions": BASE_FRACTIONS,
        "windows": WINDOWS,
        "aggregations": AGGREGATIONS,
        "gate_quantiles": GATE_QUANTILES,
        "run_lengths": RUN_LENGTHS,
        "far_grid": FAR_GRID,
        "primary_far": PRIMARY_FAR,
        "selection_rules": SELECTION_RULES,
        "attack_types": ATTACK_TYPES,
        "detail_rows": int(len(detail)),
        "aggregate_rows": int(len(aggregate)),
        "paired_rows": int(len(paired)),
        "placement_redundancy_rows": int(len(redundancy)),
        "aggregate": clean_records(aggregate),
        "paired": clean_records(paired),
        "placement_redundancy": clean_records(redundancy),
        "limitations": [
            "位置偏移是预先固定的敏感性轴，不替换冻结协议。",
            "现有受控协议中choose_start不依赖种子，五种子对多数攻击类型不构成独立位置实现。",
            "位置敏感性审计只覆盖当前NOAA站点、年份与受控注入生成器。",
            "攻击仍为透明受控注入，不代表真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段194 NOAA攻击位置扰动稳健性审计",
        "",
        "在预先固定的对称位置偏移网格上重建整条流程，valid选择、test只评价。",
        "",
        "| 位置偏移 | 折数 | test F1 | F1平均差 | 非stuck事件覆盖平均差 | 采用保护折数 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.to_dict(orient="records"):
        lines.append(
            f"| {row['position_shift']:+.2f} | {row['folds']} | "
            f"{row['test_f1_mean']:.6f} | "
            f"{row['test_f1_diff_mean']:+.6f} | "
            f"{row['test_nonstuck_diff_mean']:+.6f} | "
            f"{row['protected_folds']} |"
        )
    lines += [
        "",
        f"1%FAR下F1平均差跨位置={f1_paired['mean_diff']:+.6f}"
        f"（正向={f1_paired['positive_folds']}/"
        f"{f1_paired['folds']}）；非stuck事件覆盖平均差跨位置="
        f"{nonstuck_paired['mean_diff']:+.6f}"
        f"（正向={nonstuck_paired['positive_folds']}/"
        f"{nonstuck_paired['folds']}）。",
        "",
        "位置冗余审计：现有受控协议中同一折内5个种子的攻击位置完全相同，"
        "因此位置扰动是当前协议下最需要补强的敏感性轴。",
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
                "aggregate_rows": payload["aggregate_rows"],
                "paired_rows": payload["paired_rows"],
                "placement_redundancy_rows": payload[
                    "placement_redundancy_rows"
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
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild summaries from the persisted stage-194 detail CSV.",
    )
    args = parser.parse_args(argv)
    prefix = "stage194_noaa_position_perturbation_robustness"
    if args.summarize_existing:
        detail = pd.read_csv(TABLES / f"{prefix}_detail.csv")
        if args.fold is not None:
            fold_index = resolve_fold(args.fold)
            train_years, valid_year, test_year = FOLDS[fold_index]
            detail = detail[
                detail["fold"].eq(
                    fold_name(train_years, valid_year, test_year)
                )
            ]
        if args.shift is not None:
            detail = detail[detail["position_shift"].eq(float(args.shift))]
        if detail.empty:
            raise ValueError("no detail rows matched the requested filter")
        return summarize_and_write(detail)

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
    frozen = (
        pd.read_csv(FROZEN)
        .loc[lambda x: x["method"].eq(METHOD)]
        .set_index(["fold", "seed"])
    )
    fold_index = resolve_fold(args.fold)
    selected_folds = [FOLDS[fold_index]] if fold_index is not None else FOLDS
    selected_shifts = (
        [float(args.shift)] if args.shift is not None else POSITION_SHIFTS
    )

    rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in selected_folds:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for shift in selected_shifts:
            frame = build_score_frame(
                source,
                stations,
                frozen,
                train_years,
                valid_year,
                test_year,
                int(args.seed),
                float(shift),
            )
            rows.extend(evaluate_group(frame))
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "seed": int(args.seed),
                        "position_shift": float(shift),
                        "status": "completed",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return summarize_and_write(pd.DataFrame(rows))


if __name__ == "__main__":
    raise SystemExit(main())
