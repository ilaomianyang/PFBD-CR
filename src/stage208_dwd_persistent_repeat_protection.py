from __future__ import annotations

"""Stage 208: validation-selected persistent-repeat protection for DWD.

Stage 207 is retained unchanged as the frozen DWD controlled-attack
baseline.  This stage tests a pre-declared family of causal repeated-value
features that are expected to help the ``stuck`` attack while preserving
the Stage-207 test boundary.

The valid and test attack sequences are constructed separately:

* valid attacks are injected into 2024 only;
* test attacks are injected into 2025 only, using the exact Stage-207
  block generator and injection parameters;
* detector fitting uses only 2021-2023 train-normal observations;
* all feature, weight, window, and threshold selection uses valid only;
* test is evaluation-only.

The time-shift attack remains anchored to the clean Stage-206 series via
``source_row_position`` and ``clean_values``; it never reads the attacked
test values.  All attacks are transparent controlled injections, not real
attack logs.
"""

import hashlib
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
WORKSPACE_ROOT = ROOT.parents[1]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    ATTACK_PARAMETERS,
    COMPONENTS,
    DETECTORS,
    NORMALIZED,
    PROTOCOL,
    choose_attack_blocks,
    detector_frame,
    fit_detectors,
    fuse_scores,
    inject_attack,
    load_protocol,
    metrics,
    robust_standardize,
    score_components,
    sha256_file,
)

PREFIX = "stage208_dwd_persistent_repeat_protection"
STAGE207_JSON = TABLES / "stage207_dwd_controlled_attack.json"
STAGE207_EVALUATION = (
    TABLES / "stage207_dwd_controlled_attack_evaluation.csv"
)
STAGE207_EVENTS = TABLES / "stage207_dwd_controlled_attack_events.csv"

FEATURES = ["repeat_ratio", "repeat_run"]
WINDOWS = [2, 3, 6]
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
SELECTION_RULES = [
    "valid_f1",
    "valid_stuck_event",
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

# These offsets only define deterministic, pre-declared valid-side attack
# construction.  They never read test labels or test attack locations.
VALID_BLOCK_SEED_OFFSET = 104729
VALID_INJECTION_SEED_OFFSET = 20260715


def clean_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for record in records:
        cleaned: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(
                float(value)
            ):
                cleaned[key] = None
            elif isinstance(value, (np.integer,)):
                cleaned[key] = int(value)
            elif isinstance(value, (np.floating,)):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        output.append(cleaned)
    return output


def choose_split_attack_blocks(
    frame: pd.DataFrame,
    protocol: dict,
    seed: int,
    split: str,
) -> pd.DataFrame:
    """Choose non-overlapping blocks for one split.

    Test uses the exact Stage-207 generator.  Valid uses a separate,
    pre-declared RNG stream and is never used to choose test locations.
    """
    if split == "test":
        return choose_attack_blocks(frame, protocol, int(seed))
    if split != "valid":
        raise ValueError("split must be valid or test")

    rng = np.random.default_rng(int(seed) + VALID_BLOCK_SEED_OFFSET)
    attack_types = list(protocol["attack_types"])
    lengths = protocol["attack_length"]
    rows: list[dict[str, object]] = []
    for station_id, group in frame.groupby("station_id", sort=True):
        ordered = (
            group.sort_values("timestamp_utc")
            .reset_index()
            .rename(columns={"index": "global_row_position"})
        )
        positions = np.flatnonzero(
            ordered["split"].eq(split).to_numpy()
            & ordered["observation_present"].to_numpy(dtype=bool)
        )
        required = sum(int(lengths[name]) for name in attack_types)
        if positions.size < required:
            raise RuntimeError(
                f"not enough {split} rows for station {station_id}"
            )
        for attack_type in attack_types:
            length = int(lengths[attack_type])
            if positions.size < length:
                raise RuntimeError(
                    f"not enough {split} rows for {station_id}/{attack_type}"
                )
            start = int(rng.integers(0, positions.size - length + 1))
            chosen = positions[start : start + length]
            if chosen.size != length:
                raise RuntimeError("attack block does not match frozen length")
            for local_position in chosen:
                local_position = int(local_position)
                if attack_type == "time_shift":
                    source_local_position = local_position - 4
                    if source_local_position < 0:
                        raise RuntimeError(
                            "time_shift requires four clean historical rows"
                        )
                    source_global_position = int(
                        ordered.loc[
                            source_local_position, "global_row_position"
                        ]
                    )
                else:
                    source_global_position = -1
                rows.append(
                    {
                        "station_id": station_id,
                        "seed": int(seed),
                        "split": split,
                        "attack_type": attack_type,
                        "row_position": int(
                            ordered.loc[
                                local_position, "global_row_position"
                            ]
                        ),
                        "source_row_position": source_global_position,
                        "timestamp_utc": ordered.loc[
                            local_position, "timestamp_utc"
                        ],
                    }
                )
            positions = np.delete(
                positions,
                np.arange(start, start + length),
            )
    return pd.DataFrame(rows)


def repeat_feature(
    frame: pd.DataFrame,
    feature_name: str,
    window: int,
) -> pd.Series:
    """Return a causal repeated-value feature within each station."""
    if feature_name not in FEATURES:
        raise ValueError(f"unknown repeat feature: {feature_name}")
    if window < 1:
        raise ValueError("window must be positive")

    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, part in frame.groupby("station_id", sort=False):
        values = pd.to_numeric(part["temperature_c"], errors="coerce")
        indicator = values.diff().eq(0.0).astype(float)
        if feature_name == "repeat_ratio":
            series = indicator.rolling(
                window=window,
                min_periods=1,
            ).mean()
        else:
            run = np.zeros(len(part), dtype=float)
            current = 0
            for index, value in enumerate(
                indicator.to_numpy(dtype=float)
            ):
                if np.isfinite(value) and value > 0.5:
                    current += 1
                else:
                    current = 0
                run[index] = min(float(current), float(window))
            series = pd.Series(run, index=part.index, dtype=float)
        output.loc[part.index] = series
    return output.where(frame["observation_present"], 0.0)


def event_metrics(
    frame: pd.DataFrame,
    predicted: np.ndarray,
) -> dict[str, float | int]:
    attack = frame["is_attack"].astype(bool).to_numpy()
    if not attack.any():
        return {
            "events": 0,
            "detected_events": 0,
            "event_coverage": 0.0,
            "row_recall": 0.0,
        }
    event_rows = frame.loc[attack, ["attack_event_id"]].copy()
    event_rows["predicted"] = np.asarray(predicted, dtype=bool)[attack]
    grouped = event_rows.groupby("attack_event_id", sort=True)[
        "predicted"
    ].any()
    return {
        "events": int(len(grouped)),
        "detected_events": int(grouped.sum()),
        "event_coverage": float(grouped.mean()) if len(grouped) else 0.0,
        "row_recall": float(event_rows["predicted"].mean()),
    }


def type_metrics(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    score: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    attack_type = frame["attack_type"].astype(str).to_numpy()
    attack = frame["is_attack"].astype(bool).to_numpy()
    for current_type in ATTACK_TYPES:
        mask = attack & (attack_type == current_type)
        if not mask.any():
            output[current_type] = {
                "attack_rows": 0,
                "detected_rows": 0,
                "row_recall": 0.0,
                "events": 0,
                "event_coverage": 0.0,
                "mean_score_attack": 0.0,
            }
            continue
        work = frame.loc[mask, ["attack_event_id"]].copy()
        work["predicted"] = np.asarray(predicted, dtype=bool)[mask]
        grouped = work.groupby("attack_event_id", sort=True)[
            "predicted"
        ].any()
        output[current_type] = {
            "attack_rows": int(mask.sum()),
            "detected_rows": int(np.asarray(predicted, dtype=bool)[mask].sum()),
            "row_recall": float(
                np.asarray(predicted, dtype=bool)[mask].mean()
            ),
            "events": int(len(grouped)),
            "event_coverage": float(grouped.mean()) if len(grouped) else 0.0,
            "mean_score_attack": float(
                np.nanmean(np.asarray(score, dtype=float)[mask])
            ),
        }
    return output


def macro_event_coverage(
    type_values: dict[str, dict[str, float | int]],
    names: list[str],
) -> float:
    values = [
        float(type_values[name]["event_coverage"])
        for name in names
        if name in type_values
    ]
    return float(np.mean(values)) if values else 0.0


def station_metrics(
    frame: pd.DataFrame,
    predicted: np.ndarray,
) -> list[dict[str, object]]:
    """Return row-confusion and event counts for each station."""
    predicted = np.asarray(predicted, dtype=bool)
    y_true = frame["is_attack"].astype(bool).to_numpy()
    station_ids = frame["station_id"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for station_id in sorted(np.unique(station_ids).tolist()):
        mask = station_ids == station_id
        station_true = y_true[mask]
        station_pred = predicted[mask]
        event_part = frame.loc[mask, ["is_attack", "attack_event_id"]].copy()
        event_part = event_part.loc[event_part["is_attack"].astype(bool)]
        event_part["predicted"] = station_pred[
            frame.loc[mask, "is_attack"].astype(bool).to_numpy()
        ]
        grouped = event_part.groupby("attack_event_id", sort=True)[
            "predicted"
        ].any()
        rows.append(
            {
                "station_id": station_id,
                "rows": int(mask.sum()),
                "attack_rows": int(station_true.sum()),
                "normal_rows": int((~station_true).sum()),
                "tp": int((station_true & station_pred).sum()),
                "fp": int((~station_true & station_pred).sum()),
                "fn": int((station_true & ~station_pred).sum()),
                "tn": int((~station_true & ~station_pred).sum()),
                "events": int(len(grouped)),
                "detected_events": int(grouped.sum()),
                "event_coverage": (
                    float(grouped.mean()) if len(grouped) else float("nan")
                ),
                "row_recall": (
                    float(station_pred[station_true].mean())
                    if station_true.any()
                    else float("nan")
                ),
            }
        )
    return rows


def far_threshold(
    scores: np.ndarray,
    normal_mask: np.ndarray,
    target_far: float,
) -> float:
    reference = np.asarray(scores, dtype=float)[normal_mask]
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        raise RuntimeError("no valid-normal scores for threshold calibration")
    return float(np.quantile(reference, 1.0 - float(target_far)))


def selection_key(
    rule: str,
    valid_metrics: dict[str, float],
    valid_event_macro: float,
    valid_stuck_event: float,
    valid_nonstuck_event: float,
    feature_name: str,
    window: int,
    weight: float,
) -> tuple[float, ...]:
    if rule == "valid_f1":
        primary = float(valid_metrics["f1"])
    elif rule == "valid_stuck_event":
        primary = float(valid_stuck_event)
    elif rule == "valid_event_macro":
        primary = float(valid_event_macro)
    elif rule == "valid_nonstuck_event_macro":
        primary = float(valid_nonstuck_event)
    else:
        raise ValueError(f"unknown selection rule: {rule}")
    feature_rank = float(FEATURES.index(feature_name))
    return (
        primary,
        float(valid_metrics["f1"]),
        float(valid_stuck_event),
        float(valid_event_macro),
        float(valid_nonstuck_event),
        -float(weight),
        -float(window),
        -feature_rank,
    )


def candidate_grid() -> list[tuple[str, int, float]]:
    return [
        (feature_name, int(window), float(weight))
        for feature_name, window, weight in itertools.product(
            FEATURES,
            WINDOWS,
            WEIGHTS,
        )
    ]


def main() -> int:
    protocol = load_protocol()
    frame = pd.read_parquet(NORMALIZED)
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("unexpected Stage206 split labels")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("Stage206 contains duplicate station timestamps")

    base = detector_frame(frame)
    detectors = fit_detectors(base, protocol)
    train_normal = (
        base["split"].eq("train") & base["observation_present"]
    )

    stage207_payload = json.loads(
        STAGE207_JSON.read_text(encoding="utf-8")
    )
    stage207_evaluation = pd.read_csv(STAGE207_EVALUATION)
    stage207_events = pd.read_csv(STAGE207_EVENTS)

    candidates = candidate_grid()
    evaluation_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    attack_manifest: list[dict[str, object]] = []

    for seed in protocol["seeds"]:
        seed = int(seed)
        valid_blocks = choose_split_attack_blocks(
            base,
            protocol,
            seed,
            "valid",
        )
        test_blocks = choose_split_attack_blocks(
            base,
            protocol,
            seed,
            "test",
        )

        # Valid and test attacks are deliberately evaluated on separate
        # sequences.  This prevents valid-side Kalman state from entering
        # the frozen test evaluation.
        valid_attacked = inject_attack(
            base,
            valid_blocks,
            seed + VALID_INJECTION_SEED_OFFSET,
        )
        test_attacked = inject_attack(base, test_blocks, seed)
        for attacked, split in (
            (valid_attacked, "valid"),
            (test_attacked, "test"),
        ):
            attacked["attack_event_id"] = np.where(
                attacked["is_attack"],
                attacked["station_id"].astype(str)
                + "__"
                + attacked["attack_type"].astype(str)
                + "__"
                + split
                + "__"
                + str(seed),
                "",
            )

        valid_components = score_components(valid_attacked, detectors)
        valid_fused, _ = fuse_scores(valid_attacked, valid_components)
        valid_z = robust_standardize(valid_fused, train_normal).to_numpy(
            dtype=float
        )
        test_components = score_components(test_attacked, detectors)
        test_fused, _ = fuse_scores(test_attacked, test_components)
        test_z = robust_standardize(test_fused, train_normal).to_numpy(
            dtype=float
        )

        valid_feature_z: dict[tuple[str, int], np.ndarray] = {}
        test_feature_z: dict[tuple[str, int], np.ndarray] = {}
        for feature_name in FEATURES:
            for window in WINDOWS:
                key = (feature_name, int(window))
                valid_feature_z[key] = robust_standardize(
                    repeat_feature(
                        valid_attacked,
                        feature_name,
                        int(window),
                    ),
                    train_normal,
                ).to_numpy(dtype=float)
                test_feature_z[key] = robust_standardize(
                    repeat_feature(
                        test_attacked,
                        feature_name,
                        int(window),
                    ),
                    train_normal,
                ).to_numpy(dtype=float)

        valid_mask = valid_attacked["split"].eq("valid").to_numpy()
        test_mask = test_attacked["split"].eq("test").to_numpy()
        valid_frame = valid_attacked.loc[valid_mask].reset_index(drop=True)
        test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
        valid_normal = (
            valid_frame["observation_present"]
            & ~valid_frame["is_attack"]
        ).to_numpy(dtype=bool)
        test_normal = (
            test_frame["observation_present"]
            & ~test_frame["is_attack"]
        ).to_numpy(dtype=bool)
        y_valid = valid_frame["is_attack"].to_numpy(dtype=bool)
        y_test = test_frame["is_attack"].to_numpy(dtype=bool)
        valid_base_score = valid_z[valid_mask]
        test_base_score = test_z[test_mask]

        for far in protocol["far_grid"]:
            far = float(far)
            baseline_threshold = far_threshold(
                valid_base_score,
                valid_normal,
                far,
            )
            baseline_valid_pred = valid_base_score >= baseline_threshold
            baseline_valid_metrics = metrics(
                y_valid,
                baseline_valid_pred,
                valid_base_score,
            )
            baseline_valid_events = event_metrics(
                valid_frame,
                baseline_valid_pred,
            )
            baseline_test_pred = test_base_score >= baseline_threshold
            baseline_test_metrics = metrics(
                y_test,
                baseline_test_pred,
                test_base_score,
            )
            baseline_test_events = event_metrics(
                test_frame,
                baseline_test_pred,
            )

            best_by_rule: dict[str, dict[str, object]] = {}
            for feature_name, window, weight in candidates:
                valid_score = (
                    (1.0 - weight) * valid_base_score
                    + weight * valid_feature_z[(feature_name, window)][
                        valid_mask
                    ]
                )
                test_score = (
                    (1.0 - weight) * test_base_score
                    + weight * test_feature_z[(feature_name, window)][
                        test_mask
                    ]
                )
                threshold = far_threshold(
                    valid_score,
                    valid_normal,
                    far,
                )
                valid_pred = valid_score >= threshold
                valid_metrics = metrics(
                    y_valid,
                    valid_pred,
                    valid_score,
                )
                valid_type = type_metrics(
                    valid_frame,
                    valid_pred,
                    valid_score,
                )
                valid_event_macro = macro_event_coverage(
                    valid_type,
                    ATTACK_TYPES,
                )
                valid_stuck_event = float(
                    valid_type["stuck"]["event_coverage"]
                )
                valid_nonstuck_event = macro_event_coverage(
                    valid_type,
                    NON_STUCK_TYPES,
                )
                for rule in SELECTION_RULES:
                    key = selection_key(
                        rule,
                        valid_metrics,
                        valid_event_macro,
                        valid_stuck_event,
                        valid_nonstuck_event,
                        feature_name,
                        window,
                        weight,
                    )
                    current = best_by_rule.get(rule)
                    if current is None or key > current["selection_key"]:
                        best_by_rule[rule] = {
                            "selection_key": key,
                            "feature": feature_name,
                            "window": int(window),
                            "weight": float(weight),
                            "threshold": float(threshold),
                            "valid_score": valid_score,
                            "test_score": test_score,
                            "valid_metrics": valid_metrics,
                            "valid_event_macro": valid_event_macro,
                            "valid_stuck_event": valid_stuck_event,
                            "valid_nonstuck_event": valid_nonstuck_event,
                        }

            for rule in SELECTION_RULES:
                selected = best_by_rule[rule]
                selected_test_pred = (
                    selected["test_score"] >= selected["threshold"]
                )
                selected_test_metrics = metrics(
                    y_test,
                    selected_test_pred,
                    selected["test_score"],
                )
                selected_test_events = event_metrics(
                    test_frame,
                    selected_test_pred,
                )
                selected_test_type = type_metrics(
                    test_frame,
                    selected_test_pred,
                    selected["test_score"],
                )
                for station_row in station_metrics(
                    test_frame,
                    selected_test_pred,
                ):
                    station_rows.append(
                        {
                            "seed": seed,
                            "max_far": far,
                            "selection_rule": rule,
                            "arm": "candidate",
                            **station_row,
                        }
                    )
                for station_row in station_metrics(
                    test_frame,
                    baseline_test_pred,
                ):
                    station_rows.append(
                        {
                            "seed": seed,
                            "max_far": far,
                            "selection_rule": rule,
                            "arm": "stage207",
                            **station_row,
                        }
                    )
                evaluation_rows.append(
                    {
                        "seed": seed,
                        "max_far": far,
                        "selection_rule": rule,
                        "feature": selected["feature"],
                        "window": selected["window"],
                        "weight": selected["weight"],
                        "threshold": selected["threshold"],
                        "valid_f1": selected["valid_metrics"]["f1"],
                        "valid_recall": selected["valid_metrics"]["recall"],
                        "valid_precision": selected["valid_metrics"][
                            "precision"
                        ],
                        "valid_false_alarm_rate": selected["valid_metrics"][
                            "false_alarm_rate"
                        ],
                        "valid_event_macro": selected[
                            "valid_event_macro"
                        ],
                        "valid_stuck_event": selected[
                            "valid_stuck_event"
                        ],
                        "valid_nonstuck_event": selected[
                            "valid_nonstuck_event"
                        ],
                        "test_f1": selected_test_metrics["f1"],
                        "test_recall": selected_test_metrics["recall"],
                        "test_precision": selected_test_metrics[
                            "precision"
                        ],
                        "test_false_alarm_rate": selected_test_metrics[
                            "false_alarm_rate"
                        ],
                        "test_roc_auc": selected_test_metrics["roc_auc"],
                        "test_average_precision": selected_test_metrics[
                            "average_precision"
                        ],
                        "baseline_valid_f1": baseline_valid_metrics["f1"],
                        "baseline_valid_recall": baseline_valid_metrics[
                            "recall"
                        ],
                        "baseline_valid_false_alarm_rate": (
                            baseline_valid_metrics["false_alarm_rate"]
                        ),
                        "baseline_valid_event_coverage": baseline_valid_events[
                            "event_coverage"
                        ],
                        "baseline_test_f1": baseline_test_metrics["f1"],
                        "baseline_test_recall": baseline_test_metrics[
                            "recall"
                        ],
                        "baseline_test_precision": baseline_test_metrics[
                            "precision"
                        ],
                        "baseline_test_false_alarm_rate": (
                            baseline_test_metrics["false_alarm_rate"]
                        ),
                        "baseline_test_event_coverage": baseline_test_events[
                            "event_coverage"
                        ],
                        "baseline_test_row_recall": baseline_test_events[
                            "row_recall"
                        ],
                    }
                )
                event_rows.append(
                    {
                        "seed": seed,
                        "max_far": far,
                        "selection_rule": rule,
                        "events": selected_test_events["events"],
                        "detected_events": selected_test_events[
                            "detected_events"
                        ],
                        "event_coverage": selected_test_events[
                            "event_coverage"
                        ],
                        "row_recall": selected_test_events["row_recall"],
                        "baseline_events": baseline_test_events["events"],
                        "baseline_detected_events": baseline_test_events[
                            "detected_events"
                        ],
                        "baseline_event_coverage": baseline_test_events[
                            "event_coverage"
                        ],
                        "baseline_row_recall": baseline_test_events[
                            "row_recall"
                        ],
                    }
                )
                selection_rows.append(
                    {
                        "seed": seed,
                        "max_far": far,
                        "selection_rule": rule,
                        "feature": selected["feature"],
                        "window": selected["window"],
                        "weight": selected["weight"],
                        "threshold": selected["threshold"],
                        "valid_f1": selected["valid_metrics"]["f1"],
                        "valid_recall": selected["valid_metrics"]["recall"],
                        "valid_precision": selected["valid_metrics"][
                            "precision"
                        ],
                        "valid_false_alarm_rate": selected["valid_metrics"][
                            "false_alarm_rate"
                        ],
                        "valid_event_macro": selected[
                            "valid_event_macro"
                        ],
                        "valid_stuck_event": selected[
                            "valid_stuck_event"
                        ],
                        "valid_nonstuck_event": selected[
                            "valid_nonstuck_event"
                        ],
                    }
                )
                for attack_type in ATTACK_TYPES:
                    type_rows.append(
                        {
                            "seed": seed,
                            "max_far": far,
                            "selection_rule": rule,
                            "attack_type": attack_type,
                            **selected_test_type[attack_type],
                        }
                    )
        attack_manifest.append(
            {
                "seed": seed,
                "valid_attack_rows": int(valid_blocks.shape[0]),
                "test_attack_rows": int(test_blocks.shape[0]),
                "valid_stations": sorted(
                    valid_blocks["station_id"].unique().tolist()
                ),
                "test_stations": sorted(
                    test_blocks["station_id"].unique().tolist()
                ),
                "valid_attack_types": sorted(
                    valid_blocks["attack_type"].unique().tolist()
                ),
                "test_attack_types": sorted(
                    test_blocks["attack_type"].unique().tolist()
                ),
            }
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "status": "completed",
                    "valid_attack_rows": int(valid_blocks.shape[0]),
                    "test_attack_rows": int(test_blocks.shape[0]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    evaluation = pd.DataFrame(evaluation_rows)
    events = pd.DataFrame(event_rows)
    selection = pd.DataFrame(selection_rows)
    type_detail = pd.DataFrame(type_rows)
    station_detail = pd.DataFrame(station_rows)
    type_aggregate = (
        type_detail.groupby(
            ["selection_rule", "max_far", "attack_type"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            attack_rows=("attack_rows", "sum"),
            detected_rows=("detected_rows", "sum"),
            row_recall=("row_recall", "mean"),
            event_coverage=("event_coverage", "mean"),
            mean_score_attack=("mean_score_attack", "mean"),
        )
        .sort_values(["selection_rule", "max_far", "attack_type"])
    )

    aggregate = (
        evaluation.groupby(["selection_rule", "max_far"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            valid_f1=("valid_f1", "mean"),
            valid_false_alarm_rate=("valid_false_alarm_rate", "mean"),
            valid_event_macro=("valid_event_macro", "mean"),
            valid_stuck_event=("valid_stuck_event", "mean"),
            valid_nonstuck_event=("valid_nonstuck_event", "mean"),
            test_f1=("test_f1", "mean"),
            test_recall=("test_recall", "mean"),
            test_precision=("test_precision", "mean"),
            test_false_alarm_rate=("test_false_alarm_rate", "mean"),
            test_average_precision=("test_average_precision", "mean"),
            baseline_test_f1=("baseline_test_f1", "mean"),
            baseline_test_recall=("baseline_test_recall", "mean"),
            baseline_test_false_alarm_rate=(
                "baseline_test_false_alarm_rate",
                "mean",
            ),
            baseline_test_event_coverage=(
                "baseline_test_event_coverage",
                "mean",
            ),
            baseline_test_row_recall=(
                "baseline_test_row_recall",
                "mean",
            ),
        )
        .sort_values(["selection_rule", "max_far"])
    )
    candidate_event_aggregate = (
        events.groupby(["selection_rule", "max_far"], as_index=False)
        .agg(
            test_event_coverage=("event_coverage", "mean"),
            test_row_recall=("row_recall", "mean"),
        )
    )
    aggregate = aggregate.merge(
        candidate_event_aggregate,
        on=["selection_rule", "max_far"],
        how="inner",
        validate="one_to_one",
    )

    baseline_eval = stage207_evaluation.rename(
        columns={
            "f1": "stage207_f1",
            "recall": "stage207_recall",
            "precision": "stage207_precision",
            "false_alarm_rate": "stage207_false_alarm_rate",
            "average_precision": "stage207_average_precision",
        }
    )[
        [
            "seed",
            "max_far",
            "stage207_f1",
            "stage207_recall",
            "stage207_precision",
            "stage207_false_alarm_rate",
            "stage207_average_precision",
        ]
    ]
    baseline_events = stage207_events.rename(
        columns={
            "event_coverage": "stage207_event_coverage",
            "row_recall": "stage207_row_recall",
        }
    )[
        [
            "seed",
            "max_far",
            "stage207_event_coverage",
            "stage207_row_recall",
        ]
    ]
    paired_source = (
        evaluation.merge(
            baseline_eval,
            on=["seed", "max_far"],
            how="inner",
            validate="many_to_one",
        )
        .merge(
            baseline_events,
            on=["seed", "max_far"],
            how="inner",
            validate="many_to_one",
        )
        .merge(
            events[
                [
                    "seed",
                    "max_far",
                    "selection_rule",
                    "event_coverage",
                    "row_recall",
                ]
            ].rename(
                columns={
                    "event_coverage": "test_event_coverage",
                    "row_recall": "test_row_recall",
                }
            ),
            on=["seed", "max_far", "selection_rule"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(["selection_rule", "max_far", "seed"])
    )
    paired_rows: list[dict[str, object]] = []
    for (rule, far), group in paired_source.groupby(
        ["selection_rule", "max_far"],
        sort=True,
    ):
        for metric, baseline_metric in [
            ("test_f1", "stage207_f1"),
            ("test_recall", "stage207_recall"),
            ("test_precision", "stage207_precision"),
            ("test_false_alarm_rate", "stage207_false_alarm_rate"),
            ("test_average_precision", "stage207_average_precision"),
            ("test_event_coverage", "stage207_event_coverage"),
            ("test_row_recall", "stage207_row_recall"),
        ]:
            current = group[metric].to_numpy(dtype=float)
            baseline = group[baseline_metric].to_numpy(dtype=float)
            diff = current - baseline
            paired_rows.append(
                {
                    "selection_rule": rule,
                    "max_far": float(far),
                    "metric": metric,
                    "pairs": int(len(diff)),
                    "positive_pairs": int((diff > 0).sum()),
                    "negative_pairs": int((diff < 0).sum()),
                    "zero_pairs": int((diff == 0).sum()),
                    "mean_diff": float(diff.mean()),
                    "std_diff": float(diff.std(ddof=0)),
                    "min_diff": float(diff.min()),
                    "max_diff": float(diff.max()),
                }
            )
    paired = pd.DataFrame(paired_rows)

    selection_counts = (
        selection.groupby(
            [
                "selection_rule",
                "max_far",
                "feature",
                "window",
                "weight",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "selected_count"})
        .sort_values(["selection_rule", "max_far", "selected_count"])
    )

    primary_far = float(protocol["primary_far"])
    primary_aggregate = aggregate[
        aggregate["max_far"].eq(primary_far)
    ].to_dict(orient="records")
    primary_paired = paired[
        paired["max_far"].eq(primary_far)
    ].to_dict(orient="records")
    primary_type = type_aggregate[
        type_aggregate["max_far"].eq(primary_far)
    ].to_dict(orient="records")

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段208保留阶段207冻结基线不变，预先声明repeat_ratio和"
            "repeat_run两类因果重复特征，窗口2/3/6，权重0.0/0.1/0.2/"
            "0.3/0.5/0.7/0.9；三个无监督分量只在2021-2023 train正常段"
            "拟合；valid攻击只在2024注入，test攻击严格复用阶段207的"
            "2025攻击块生成器；valid和test分别评分，避免valid攻击通过"
            "Kalman状态污染test。valid和test分别构造攻击序列。"
            "特征、窗口、权重和阈值只在valid选择，"
            "test只评价；不根据test选择特征、权重、窗口、阈值、站点或"
            "攻击位置。time_shift只从阶段206干净序列的同一站点历史行"
            "取值，source_row_position和clean_values保留。所有攻击都是"
            "透明受控注入，不是真实攻击日志。"
        ),
        "source": "DWD_CDC",
        "provider": protocol["provider"],
        "dataset_version": protocol["dataset_version"],
        "normalized_sha256": sha256_file(NORMALIZED),
        "protocol_sha256": sha256_file(PROTOCOL),
        "stage207_json_sha256": sha256_file(STAGE207_JSON),
        "stage207_evaluation_sha256": sha256_file(STAGE207_EVALUATION),
        "stage207_events_sha256": sha256_file(STAGE207_EVENTS),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "detector_code_sha256": {
            path.name: sha256_file(path)
            for path in sorted((SRC / "detectors").glob("*.py"))
        },
        "attack_parameters": ATTACK_PARAMETERS,
        "valid_block_seed_offset": VALID_BLOCK_SEED_OFFSET,
        "valid_injection_seed_offset": VALID_INJECTION_SEED_OFFSET,
        "features": FEATURES,
        "windows": WINDOWS,
        "weights": WEIGHTS,
        "candidate_count": len(candidates),
        "selection_rules": SELECTION_RULES,
        "attack_types": ATTACK_TYPES,
        "far_grid": protocol["far_grid"],
        "primary_far": primary_far,
        "primary_endpoint": protocol["primary_endpoint"],
        "secondary_endpoint": protocol["secondary_endpoint"],
        "cluster_unit": protocol["cluster_unit"],
        "seeds": protocol["seeds"],
        "attack_manifest": attack_manifest,
        "evaluation_rows": int(len(evaluation)),
        "event_rows": int(len(events)),
        "selection_rows": int(len(selection)),
        "selection_count_rows": int(len(selection_counts)),
        "type_detail_rows": int(len(type_detail)),
        "station_detail_rows": int(len(station_detail)),
        "type_aggregate_rows": int(len(type_aggregate)),
        "paired_rows": int(len(paired)),
        "aggregate": clean_records(aggregate.to_dict(orient="records")),
        "primary_aggregate": clean_records(primary_aggregate),
        "primary_paired": clean_records(primary_paired),
        "primary_type_aggregate": clean_records(primary_type),
        "stage207_primary_event_coverage_mean": float(
            np.mean(
                [
                    row["event_coverage"]
                    for row in stage207_payload.get(
                        "primary_event_metrics",
                        [],
                    )
                ]
            )
        ),
        "stage207_primary_row_recall_mean": float(
            np.mean(
                [
                    row["row_recall"]
                    for row in stage207_payload.get(
                        "primary_event_metrics",
                        [],
                    )
                ]
            )
        ),
        "limitations": [
            "阶段207冻结基线和修正后的负结果全部保留，不能删除。",
            "valid选择只能降低过拟合风险，不能保证test方向一定为正。",
            "test只评价；任何根据test回看选择特征、窗口、权重、阈值、站点或攻击位置的做法都不允许。",
            "valid与test分别构造攻击序列，避免valid攻击通过顺序检测器污染test。",
            "攻击是透明受控注入，不是真实攻击日志；DWD质量码不是攻击标签。",
            "即使stuck改善，也必须同时报告其他攻击类型、FAR、行召回和事件覆盖的代价。",
        ],
    }

    evaluation.to_csv(
        TABLES / f"{PREFIX}_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    events.to_csv(
        TABLES / f"{PREFIX}_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection.to_csv(
        TABLES / f"{PREFIX}_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection_counts.to_csv(
        TABLES / f"{PREFIX}_selection_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_detail.to_csv(
        TABLES / f"{PREFIX}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_detail.to_csv(
        TABLES / f"{PREFIX}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_aggregate.to_csv(
        TABLES / f"{PREFIX}_type_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        TABLES / f"{PREFIX}_paired_vs_stage207.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段208 DWD连续重复保护valid选择评价",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 主FAR：`{primary_far}`",
        f"- 候选：`{len(candidates)}` 个（{FEATURES} × {WINDOWS} × {WEIGHTS}）",
        f"- 选择规则：`{SELECTION_RULES}`",
        "- Stage207冻结基线：`stage207_dwd_controlled_attack.json`",
        "",
        "## 主FAR聚合",
        "",
        "| 选择规则 | valid F1 | valid stuck事件 | test F1 | test Recall | test FAR | test事件覆盖 | Stage207 F1 | Stage207事件覆盖 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_aggregate:
        paired_f1 = next(
            (
                item
                for item in primary_paired
                if item["selection_rule"] == row["selection_rule"]
                and item["metric"] == "test_f1"
            ),
            None,
        )
        paired_event = next(
            (
                item
                for item in primary_paired
                if item["selection_rule"] == row["selection_rule"]
                and item["metric"] == "test_event_coverage"
            ),
            None,
        )
        lines.append(
            f"| {row['selection_rule']} | {row['valid_f1']:.4f} | "
            f"{row['valid_stuck_event']:.4f} | {row['test_f1']:.4f} | "
            f"{row['test_recall']:.4f} | "
            f"{row['test_false_alarm_rate']:.4f} | "
            f"{row['test_event_coverage']:.4f} | "
            f"{row['baseline_test_f1']:.4f} | "
            f"{row['baseline_test_event_coverage']:.4f} |"
        )
        if paired_f1 is not None:
            lines.append(
                f"|  |  |  |  |  |  |  | "
                f"F1差={paired_f1['mean_diff']:+.4f} "
                f"({paired_f1['positive_pairs']}/"
                f"{paired_f1['pairs']}正) | "
                f"事件差={paired_event['mean_diff']:+.4f} |"
            )
    lines += [
        "",
        "## 主FAR攻击类型",
        "",
        "| 选择规则 | 攻击类型 | 攻击行 | 行召回 | 事件覆盖 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in primary_type:
        lines.append(
            f"| {row['selection_rule']} | {row['attack_type']} | "
            f"{int(row['attack_rows'])} | {row['row_recall']:.4f} | "
            f"{row['event_coverage']:.4f} |"
        )
    lines += [
        "",
        "valid和test分别构造攻击序列；test只评价。所有攻击都是透明",
        "受控注入，不是真实攻击日志。阶段207负结果必须与阶段208并列",
        "报告，不能只挑选有利的规则或攻击类型。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "evaluation_rows": len(evaluation),
                "event_rows": len(events),
                "selection_rows": len(selection),
                "type_detail_rows": len(type_detail),
                "type_aggregate_rows": len(type_aggregate),
                "paired_rows": len(paired),
                "primary_far": primary_far,
                "primary_aggregate": clean_records(primary_aggregate),
                "primary_paired": clean_records(primary_paired),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
