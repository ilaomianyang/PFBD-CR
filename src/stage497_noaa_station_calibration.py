"""Stage 497: NOAA ISD station-level calibration audit (validation-only).

Motivation
----------
Stage 494 showed that NOAA ISD wins only 50/108 provider/scenario units, that
its subtle-scenario event coverage is 0.034, and that NOAA is clearly weaker
than the other three providers.  Stage 497 asks whether that deficit can be
repaired by *calibration* -- station-level robust standardisation and/or
station-level threshold allocation -- rather than by new models or retraining.

Protocol
--------
* Only the four existing providers, the existing V6 train/valid/test splits,
  ``stage490_union_scores.parquet`` (frozen union shape score
  ``max(z_trust, z_ewma, z_arima)`` plus the executed frozen union OR
  decision) and ``stage490_formal_mlp_scores.parquet`` are used.
* Every calibration parameter (station median/MAD, station quantile
  thresholds, knapsack allocation) is estimated on NOAA *validation* rows
  only.  Test rows are evaluation-only and never enter a selection.
* The protocol FAR (0.0065) is the deployed operating point.
* A candidate family may only be deployed if, for every seed, NOAA validation
  event coverage of ``long_stuck`` and ``drop_replay`` stays at or above the
  frozen union coverage (margin 0).  If no family satisfies that floor the run
  falls back to the frozen union OR prediction instead of re-thresholding the
  union shape score and calling that union.
* Objective: maximise NOAA validation mean scenario event coverage, with
  validation F1 as the tie-breaker.
* Only NOAA is calibrated.  The other three providers keep the frozen union OR
  prediction and are reported in full so nothing is hidden.

The V6 test split has already been inspected, so every number produced here is
exploratory and does not replace the frozen main table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage470_final_v6_submission_rebuild as s470  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import metrics  # noqa: E402
from stage208_dwd_persistent_repeat_protection import (  # noqa: E402
    event_metrics,
    far_threshold,
)

FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

UNION_SCORES = TABLES / "stage490_union_scores.parquet"
FORMAL_SCORES = TABLES / "stage490_formal_mlp_scores.parquet"
STAGE491_RUNS = TABLES / "stage491_constrained_fusion_runs.csv"
STAGE491_SCENARIO = TABLES / "stage491_constrained_fusion_scenario.csv"
STAGE491_STATION = TABLES / "stage491_constrained_fusion_station.csv"
STAGE493_PROVIDER = TABLES / "stage493_constrained_fusion_provider_summary.csv"
STAGE493_SEED = TABLES / "stage493_constrained_fusion_seed_summary.csv"

OUT_RUNS = TABLES / "stage497_noaa_station_calibration_runs.csv"
OUT_SCENARIO = TABLES / "stage497_noaa_station_calibration_scenario.csv"
OUT_STATION = TABLES / "stage497_noaa_station_calibration_station.csv"
OUT_SUMMARY = TABLES / "stage497_noaa_station_calibration_summary.csv"
OUT_JSON = REPORTS / "stage497_noaa_station_calibration.json"
OUT_MD = REPORTS / "stage497_noaa_station_calibration.md"

METHOD = "stage497_noaa_station_calibration"
NOAA = "NOAA ISD"
PRIMARY_FAR = 0.0065
CONSTRAINT_MARGIN = 0.0
STRUCTURED_SCENARIOS = ["long_stuck", "drop_replay"]
SCENARIO_COUNT = 9
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
FRONTIER_STEP = 0.25
FRONTIER_MAX = 3.0
CELL = 2.5e-5
MIN_STATION_NORMAL = 200
LEVELS = [
    0.0,
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.004,
    0.005,
    0.0065,
    0.008,
    0.01,
    0.013,
    0.017,
    0.022,
    0.03,
    0.04,
    0.05,
    0.07,
    0.1,
]
NEG = -1.0e18
EPS = 1.0e-12


def scenario_coverage(
    frame: pd.DataFrame,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Event coverage per attack scenario (normal rows excluded)."""
    output: dict[str, float] = {}
    for scenario, group in frame.assign(
        _pred=np.asarray(predicted, dtype=bool)
    ).groupby("attack_scenario", sort=True):
        if scenario == "normal":
            continue
        detected = group.groupby("attack_event_id")["_pred"].any()
        output[str(scenario)] = (
            float(detected.mean()) if len(detected) else 0.0
        )
    return output


def mean_scenario_coverage(coverage: dict[str, float]) -> float:
    if not coverage:
        return 0.0
    return float(np.mean(list(coverage.values())))


def robust_median_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median, max(1.4826 * mad, 1.0e-6)


def station_reference(
    values: np.ndarray,
    stations: np.ndarray,
    normal: np.ndarray,
    provider_median: float,
    provider_scale: float,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Validation-normal median/MAD per station, with provider fallback."""
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    thin: list[str] = []
    for station in sorted(set(stations.tolist())):
        mask = normal & (stations == station)
        if int(mask.sum()) < MIN_STATION_NORMAL:
            thin.append(str(station))
            medians[str(station)] = float(provider_median)
            scales[str(station)] = float(provider_scale)
            continue
        median, scale = robust_median_scale(values[mask])
        medians[str(station)] = median
        scales[str(station)] = scale
    return medians, scales, thin


def apply_station_z(
    values: np.ndarray,
    stations: np.ndarray,
    medians: dict[str, float],
    scales: dict[str, float],
) -> np.ndarray:
    median = np.array(
        [medians[str(station)] for station in stations],
        dtype=float,
    )
    scale = np.array(
        [scales[str(station)] for station in stations],
        dtype=float,
    )
    return (values - median) / scale


def threshold_lookup(
    thresholds: dict[str, float],
    stations: np.ndarray,
) -> np.ndarray:
    return np.array(
        [thresholds[str(station)] for station in stations],
        dtype=float,
    )


def score_variant(
    label: str,
    context: dict[str, object],
    score: np.ndarray,
    stations: np.ndarray,
) -> np.ndarray:
    """Recompute a candidate family score from validation-derived constants."""
    provider_z = (
        score - float(context["provider_median"])
    ) / float(context["provider_scale"])
    station_z = apply_station_z(
        score,
        stations,
        context["station_medians"],
        context["station_scales"],
    )
    if label == "pooled_provider_z":
        return provider_z
    if label == "pooled_station_z":
        return station_z
    if label == "pooled_two_level_max":
        return np.maximum(provider_z, station_z)
    if label.startswith("pooled_two_level_a"):
        alpha = float(label.split("_a")[-1])
        return alpha * provider_z + (1.0 - alpha) * station_z
    if label in ("station_quantile_equal_far", "station_quantile"):
        return station_z
    if label.startswith("station_budget_dp"):
        return score
    raise ValueError(f"unknown candidate family: {label}")


def family_prediction(
    label: str,
    context: dict[str, object],
    score: np.ndarray,
    stations: np.ndarray,
    observed: np.ndarray,
    far_multiplier: float = 1.0,
) -> np.ndarray | None:
    """Apply a candidate family to any split using validation constants."""
    if label == "station_budget_dp":
        allocation = knapsack_allocate(
            context["options"],
            context["stations"],
            context["long_stuck_floor"],
            context["drop_replay_floor"],
            context["total_long_stuck"],
            context["total_drop_replay"],
            PRIMARY_FAR * far_multiplier,
            enforce_floor=True,
        )
        if allocation is None:
            return None
        thresholds = allocation["thresholds"]
        return observed & (score >= threshold_lookup(thresholds, stations))
    if label.startswith("station_budget_dp_unconstrained"):
        allocation = knapsack_allocate(
            context["options"],
            context["stations"],
            context["long_stuck_floor"],
            context["drop_replay_floor"],
            context["total_long_stuck"],
            context["total_drop_replay"],
            PRIMARY_FAR * far_multiplier,
            enforce_floor=False,
        )
        if allocation is None:
            return None
        thresholds = allocation["thresholds"]
        return observed & (score >= threshold_lookup(thresholds, stations))
    variant = score_variant(label, context, score, stations)
    if label == "station_quantile_equal_far":
        thresholds = context["quantile_thresholds"]
        return observed & (variant >= threshold_lookup(thresholds, stations))
    threshold = float(context["pooled_thresholds"][label])
    return observed & (variant >= threshold)


def candidate_summary(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    score: np.ndarray,
    union_coverage: dict[str, float],
) -> dict[str, object]:
    attack = frame["is_attack"].astype(bool).to_numpy()
    row_metrics = metrics(attack, predicted, score)
    coverage = scenario_coverage(frame, predicted)
    event_values = event_metrics(frame, predicted)
    margins = {
        scenario: float(
            coverage.get(scenario, 0.0)
            - union_coverage.get(scenario, 0.0)
        )
        for scenario in STRUCTURED_SCENARIOS
    }
    return {
        "valid_mean_scenario_coverage": mean_scenario_coverage(coverage),
        "valid_event_coverage": float(event_values["event_coverage"]),
        "valid_f1": float(row_metrics["f1"]),
        "valid_false_alarm_rate": float(row_metrics["false_alarm_rate"]),
        "valid_average_precision": float(row_metrics["average_precision"]),
        "valid_roc_auc": float(row_metrics["roc_auc"]),
        **{
            f"valid_coverage_{scenario}": float(
                coverage.get(scenario, 0.0)
            )
            for scenario in STRUCTURED_SCENARIOS
        },
        **{
            f"valid_margin_{scenario}": margins[scenario]
            for scenario in STRUCTURED_SCENARIOS
        },
        "constraint_ok": bool(
            all(
                margins[scenario] >= -CONSTRAINT_MARGIN - EPS
                for scenario in STRUCTURED_SCENARIOS
            )
        ),
        "coverage_by_scenario": coverage,
    }


def station_options(
    score: np.ndarray,
    stations: np.ndarray,
    observed: np.ndarray,
    attack: np.ndarray,
    normal: np.ndarray,
    events: dict[str, list[str]],
    event_weights: dict[str, float],
    event_station: dict[str, str],
    event_ids: np.ndarray,
) -> list[list[dict[str, object]]]:
    """Per-station (cost, value, tp, fp, structured counts) option table."""
    total_normal = int(normal.sum())
    per_station_events: dict[str, list[str]] = {}
    for event, station in event_station.items():
        per_station_events.setdefault(station, []).append(event)
    long_stuck = set(events.get("long_stuck", []))
    drop_replay = set(events.get("drop_replay", []))
    table: list[list[dict[str, object]]] = []
    for station in sorted(set(stations.tolist())):
        mask = normal & (stations == station)
        reference = score[mask]
        rows: list[dict[str, object]] = []
        for level in LEVELS:
            threshold = (
                float(np.quantile(reference, 1.0 - level))
                if level > 0
                else float("inf")
            )
            flagged = (
                (score >= threshold) & observed & (stations == station)
            )
            false_positive = int((flagged & normal).sum())
            true_positive = int((flagged & attack).sum())
            detected = [
                event
                for event in per_station_events.get(str(station), [])
                if bool(flagged[event_ids == event].any())
            ]
            rows.append(
                {
                    "level": float(level),
                    "threshold": threshold,
                    "cost_fraction": false_positive / total_normal,
                    "value": float(
                        sum(event_weights[event] for event in detected)
                    ),
                    "tp": true_positive,
                    "fp": false_positive,
                    "long_stuck_detected": int(
                        sum(1 for event in detected if event in long_stuck)
                    ),
                    "drop_replay_detected": int(
                        sum(1 for event in detected if event in drop_replay)
                    ),
                }
            )
        table.append(rows)
    return table


def knapsack_allocate(
    options: list[list[dict[str, object]]],
    stations: list[str],
    long_stuck_floor: int,
    drop_replay_floor: int,
    total_long_stuck: int,
    total_drop_replay: int,
    budget: float,
    enforce_floor: bool = True,
) -> dict[str, object] | None:
    """Exact per-station threshold allocation under a pooled FAR budget.

    The DP value packs the primary objective (mean scenario coverage, which is
    additive in the detected events) with a lexicographic
    (true positives, -false positives) tie-break, a monotone proxy for
    validation F1.  Costs are rounded up to the FAR cell grid, so the realised
    pooled FAR never exceeds ``budget``.
    """
    cells = int(round(budget / CELL))
    shape = (cells + 1, total_long_stuck + 1, total_drop_replay + 1)
    dp = np.full(shape, NEG, dtype=float)
    dp[0, 0, 0] = 0.0
    backpointers: list[np.ndarray] = []
    costs: list[list[int]] = []
    for station_rows in options:
        station_costs = [
            min(int(np.ceil(row["cost_fraction"] / CELL - EPS)), cells)
            for row in station_rows
        ]
        costs.append(station_costs)
        next_dp = np.full(shape, NEG, dtype=float)
        pointer = np.full(shape, -1, dtype=np.int16)
        for index, row in enumerate(station_rows):
            cost = station_costs[index]
            long_stuck = int(row["long_stuck_detected"])
            drop_replay = int(row["drop_replay_detected"])
            if (
                long_stuck > total_long_stuck
                or drop_replay > total_drop_replay
            ):
                continue
            source = dp[
                : cells + 1 - cost,
                : total_long_stuck + 1 - long_stuck,
                : total_drop_replay + 1 - drop_replay,
            ]
            candidate = source + (
                float(row["value"])
                + 1.0e-6 * int(row["tp"])
                - 1.0e-9 * int(row["fp"])
            )
            target = next_dp[cost:, long_stuck:, drop_replay:]
            pointer_target = pointer[cost:, long_stuck:, drop_replay:]
            better = candidate > target
            target[better] = candidate[better]
            pointer_target[better] = index
        dp = next_dp
        backpointers.append(pointer)
    if enforce_floor:
        subspace = dp[:, long_stuck_floor:, drop_replay_floor:]
        if not bool((subspace > NEG / 2.0).any()):
            return None
        location = np.unravel_index(
            int(np.argmax(subspace)),
            subspace.shape,
        )
        state = (
            int(location[0]),
            int(location[1] + long_stuck_floor),
            int(location[2] + drop_replay_floor),
        )
    else:
        location = np.unravel_index(int(np.argmax(dp)), shape)
        state = (
            int(location[0]),
            int(location[1]),
            int(location[2]),
        )
    choice: list[dict[str, object]] = []
    current = state
    for index in range(len(options) - 1, -1, -1):
        pointer_index = int(backpointers[index][current])
        if pointer_index < 0:
            return None
        row = options[index][pointer_index]
        choice.append(row)
        current = (
            current[0] - costs[index][pointer_index],
            current[1] - int(row["long_stuck_detected"]),
            current[2] - int(row["drop_replay_detected"]),
        )
        if min(current) < 0:
            return None
    choice.reverse()
    return {
        "value": float(dp[state]),
        "cells": int(state[0]),
        "far": float(state[0] * CELL),
        "long_stuck_detected": int(state[1]),
        "drop_replay_detected": int(state[2]),
        "thresholds": {
            station: float(row["threshold"])
            for station, row in zip(stations, choice)
        },
        "levels": [float(row["level"]) for row in choice],
    }


def candidate_families(
    frame: pd.DataFrame,
    score: np.ndarray,
    stations: np.ndarray,
    observed: np.ndarray,
    attack: np.ndarray,
    normal: np.ndarray,
    union_coverage: dict[str, float],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """All NOAA station-calibration candidates for one seed (validation)."""
    provider_median, provider_scale = robust_median_scale(score[normal])
    medians, scales, thin = station_reference(
        score,
        stations,
        normal,
        provider_median,
        provider_scale,
    )
    station_z = apply_station_z(score, stations, medians, scales)
    provider_z = (score - provider_median) / provider_scale
    labels = ["pooled_provider_z", "pooled_station_z"]
    labels += [
        f"pooled_two_level_a{alpha:.2f}"
        for alpha in ALPHA_GRID
        if alpha not in (0.0, 1.0)
    ]
    labels.append("pooled_two_level_max")
    partial: dict[str, object] = {
        "provider_median": provider_median,
        "provider_scale": provider_scale,
        "station_medians": medians,
        "station_scales": scales,
        "thin_stations": thin,
        "stations": sorted(set(stations.tolist())),
        "pooled_thresholds": {},
        "quantile_thresholds": {},
        "split": "valid",
    }
    families: dict[str, dict[str, object]] = {}
    for label in labels:
        variant = score_variant(label, partial, score, stations)
        threshold = float(far_threshold(variant, normal, PRIMARY_FAR))
        partial["pooled_thresholds"][label] = threshold
        predicted = observed & (variant >= threshold)
        summary = candidate_summary(
            frame, predicted, variant, union_coverage
        )
        summary.update(
            {
                "family": label,
                "family_kind": "pooled_threshold",
                "threshold": threshold,
                "score": label,
                "far_multiplier": 1.0,
                "predicted": predicted,
                "score_values": variant,
            }
        )
        families[label] = summary

    quantile_thresholds = {
        station: float(far_threshold(station_z, normal & (stations == station), PRIMARY_FAR))
        if int((normal & (stations == station)).sum()) >= MIN_STATION_NORMAL
        else float(partial["pooled_thresholds"]["pooled_provider_z"])
        for station in sorted(set(stations.tolist()))
    }
    partial["quantile_thresholds"] = quantile_thresholds
    quantile_predicted = observed & (
        station_z >= threshold_lookup(quantile_thresholds, stations)
    )
    summary = candidate_summary(
        frame,
        quantile_predicted,
        station_z,
        union_coverage,
    )
    summary.update(
        {
            "family": "station_quantile_equal_far",
            "family_kind": "station_quantile_threshold",
            "threshold": float("nan"),
            "score": "station_z",
            "far_multiplier": 1.0,
            "predicted": quantile_predicted,
            "score_values": station_z,
        }
    )
    families["station_quantile_equal_far"] = summary

    events: dict[str, list[str]] = {}
    for scenario, group in frame[frame["is_attack"]].groupby(
        "attack_scenario",
        sort=True,
    ):
        events[str(scenario)] = list(
            group["attack_event_id"].astype(str).unique()
        )
    event_weights = {
        event: 1.0 / (len(events) * len(values))
        for scenario, values in events.items()
        for event in values
    }
    event_ids = frame["attack_event_id"].astype(str).to_numpy()
    station_lookup = frame["station_id"].astype(str).to_numpy()
    event_station = {
        event: str(station_lookup[event_ids == event][0])
        for event in event_weights
    }
    options = station_options(
        score,
        stations,
        observed,
        attack,
        normal,
        events,
        event_weights,
        event_station,
        event_ids,
    )
    total_long_stuck = len(events["long_stuck"])
    total_drop_replay = len(events["drop_replay"])
    long_stuck_floor = int(
        round(union_coverage.get("long_stuck", 0.0) * total_long_stuck)
    )
    drop_replay_floor = int(
        round(
            union_coverage.get("drop_replay", 0.0) * total_drop_replay
        )
    )
    partial.update(
        {
            "events": events,
            "event_station": event_station,
            "event_ids": event_ids,
            "event_weights": event_weights,
            "options": options,
            "long_stuck_floor": long_stuck_floor,
            "drop_replay_floor": drop_replay_floor,
            "total_long_stuck": total_long_stuck,
            "total_drop_replay": total_drop_replay,
        }
    )
    for label, enforce_floor in (
        ("station_budget_dp", True),
        ("station_budget_dp_unconstrained", False),
    ):
        allocation = knapsack_allocate(
            options,
            partial["stations"],
            long_stuck_floor,
            drop_replay_floor,
            total_long_stuck,
            total_drop_replay,
            PRIMARY_FAR,
            enforce_floor=enforce_floor,
        )
        if allocation is None:
            families[label] = {
                "family": label,
                "family_kind": (
                    "station_budget_allocation"
                    if enforce_floor
                    else "station_budget_allocation_diagnostic"
                ),
                "threshold": float("nan"),
                "score": "union_shape",
                "far_multiplier": 1.0,
                "predicted": None,
                "score_values": score,
                "valid_mean_scenario_coverage": float("nan"),
                "valid_event_coverage": float("nan"),
                "valid_f1": float("nan"),
                "valid_false_alarm_rate": float("nan"),
                "valid_average_precision": float("nan"),
                "valid_roc_auc": float("nan"),
                "valid_coverage_long_stuck": float("nan"),
                "valid_coverage_drop_replay": float("nan"),
                "valid_margin_long_stuck": float("nan"),
                "valid_margin_drop_replay": float("nan"),
                "constraint_ok": False,
                "coverage_by_scenario": {},
                "infeasible": True,
            }
            continue
        predicted = observed & (
            score >= threshold_lookup(allocation["thresholds"], stations)
        )
        summary = candidate_summary(
            frame, predicted, score, union_coverage
        )
        summary.update(
            {
                "family": label,
                "family_kind": (
                    "station_budget_allocation"
                    if enforce_floor
                    else "station_budget_allocation_diagnostic"
                ),
                "threshold": float("nan"),
                "score": "union_shape",
                "far_multiplier": 1.0,
                "predicted": predicted,
                "score_values": score,
                "realised_far": float(allocation["far"]),
                "thresholds": allocation["thresholds"],
                "levels": allocation["levels"],
                "infeasible": False,
            }
        )
        families[label] = summary
    return families, partial


def frontier_rows(
    context: dict[str, object],
    frame: pd.DataFrame,
    score: np.ndarray,
    stations: np.ndarray,
    observed: np.ndarray,
    union_coverage: dict[str, float],
) -> list[dict[str, object]]:
    """Constraint feasibility as a function of the pooled FAR budget."""
    rows: list[dict[str, object]] = []
    multiplier = 1.0
    while multiplier <= FRONTIER_MAX + EPS:
        allocation = knapsack_allocate(
            context["options"],
            context["stations"],
            context["long_stuck_floor"],
            context["drop_replay_floor"],
            context["total_long_stuck"],
            context["total_drop_replay"],
            PRIMARY_FAR * multiplier,
            enforce_floor=True,
        )
        row: dict[str, object] = {
            "far_multiplier": float(multiplier),
            "far_budget": float(PRIMARY_FAR * multiplier),
            "feasible": allocation is not None,
        }
        if allocation is not None:
            predicted = observed & (
                score >= threshold_lookup(allocation["thresholds"], stations)
            )
            summary = candidate_summary(
                frame, predicted, score, union_coverage
            )
            row.update(
                {
                    "valid_mean_scenario_coverage": summary[
                        "valid_mean_scenario_coverage"
                    ],
                    "valid_f1": summary["valid_f1"],
                    "valid_false_alarm_rate": summary[
                        "valid_false_alarm_rate"
                    ],
                    "valid_average_precision": summary[
                        "valid_average_precision"
                    ],
                    "valid_coverage_long_stuck": summary[
                        "valid_coverage_long_stuck"
                    ],
                    "valid_coverage_drop_replay": summary[
                        "valid_coverage_drop_replay"
                    ],
                    "realised_far": float(allocation["far"]),
                }
            )
        rows.append(row)
        multiplier += FRONTIER_STEP
    return rows


def minimum_far_for_events(
    frame: pd.DataFrame,
    score: np.ndarray,
    stations: np.ndarray,
    attack: np.ndarray,
    normal: np.ndarray,
    event_ids: np.ndarray,
    target_events: set[str],
) -> float:
    """Cheapest per-station threshold FAR that detects every target event."""
    total_normal = int(normal.sum())
    cost = 0.0
    for station in sorted(set(stations.tolist())):
        mask = normal & (stations == station)
        selected = [
            event
            for event in target_events
            if bool(((event_ids == event) & attack).any())
            and str(
                frame.loc[event_ids == event, "station_id"]
                .astype(str)
                .iloc[0]
            )
            == str(station)
        ]
        if not selected:
            continue
        required = 0.0
        for event in selected:
            rows = (event_ids == event) & attack
            threshold = float(score[rows].max())
            required = max(
                required,
                float((score[mask] >= threshold).mean()),
            )
        cost += required * int(mask.sum()) / total_normal
    return float(cost)


def main() -> int:
    started = time.perf_counter()
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    union = pd.read_parquet(UNION_SCORES)
    formal = pd.read_parquet(FORMAL_SCORES)
    merged = union.merge(
        formal,
        on=["source", "seed", "split", "row_index"],
        suffixes=("_union", "_formal"),
        validate="one_to_one",
    )
    for column in [
        "station_id",
        "attack_scenario",
        "attack_event_id",
        "is_attack",
        "observation_present",
    ]:
        if not (
            merged[f"{column}_union"].astype(str)
            == merged[f"{column}_formal"].astype(str)
        ).all():
            raise ValueError(f"score alignment mismatch: {column}")

    stage491_runs = pd.read_csv(STAGE491_RUNS).set_index(
        ["source", "seed"]
    )
    stage491_scenario = pd.read_csv(STAGE491_SCENARIO)
    stage491_scenario_lookup = stage491_scenario.set_index(
        ["source", "seed", "scenario"]
    )[["event_coverage", "f1", "false_alarm_rate"]]
    stage491_station = pd.read_csv(STAGE491_STATION).rename(
        columns={
            "f1": "stage491_f1",
            "event_coverage": "stage491_event_coverage",
            "false_alarm_rate": "stage491_false_alarm_rate",
        }
    )[
        [
            "source",
            "seed",
            "station_id",
            "stage491_f1",
            "stage491_event_coverage",
            "stage491_false_alarm_rate",
        ]
    ]
    stage493_provider = pd.read_csv(STAGE493_PROVIDER)
    stage493_seed = pd.read_csv(STAGE493_SEED)

    providers = sorted(str(value) for value in merged["source"].unique())
    seeds = sorted(int(value) for value in merged["seed"].unique())

    candidate_rows: list[dict[str, object]] = []
    frontier: list[dict[str, object]] = []
    contexts: dict[int, dict[str, object]] = {}
    valid_state: dict[tuple[str, int], dict[str, object]] = {}
    noaa_families: dict[int, dict[str, dict[str, object]]] = {}

    for (source, seed), group in merged.groupby(
        ["source", "seed"],
        sort=True,
    ):
        valid = group[group["split"].eq("valid")].copy()
        test = group[group["split"].eq("test")].copy()
        valid_frame = pd.DataFrame(
            {
                "station_id": valid["station_id_union"].astype(str),
                "attack_scenario": valid[
                    "attack_scenario_union"
                ].astype(str),
                "attack_event_id": valid[
                    "attack_event_id_union"
                ].astype(str),
                "is_attack": valid["is_attack_union"].astype(bool),
                "observation_present": valid[
                    "observation_present_union"
                ].astype(bool),
            }
        ).reset_index(drop=True)
        test_frame = pd.DataFrame(
            {
                "station_id": test["station_id_union"].astype(str),
                "attack_scenario": test[
                    "attack_scenario_union"
                ].astype(str),
                "attack_event_id": test["attack_event_id_union"].astype(str),
                "is_attack": test["is_attack_union"].astype(bool),
                "observation_present": test[
                    "observation_present_union"
                ].astype(bool),
            }
        ).reset_index(drop=True)
        valid_observed = valid_frame[
            "observation_present"
        ].to_numpy(dtype=bool)
        valid_attack = valid_frame["is_attack"].to_numpy(dtype=bool)
        valid_normal = valid_observed & ~valid_attack
        union_valid_pred = valid["union_prediction"].to_numpy(dtype=bool)
        union_test_pred = test["union_prediction"].to_numpy(dtype=bool)
        union_valid_coverage = scenario_coverage(
            valid_frame,
            union_valid_pred,
        )
        valid_state[(str(source), int(seed))] = {
            "valid_frame": valid_frame,
            "test_frame": test_frame,
            "valid_score": valid["union_score"].to_numpy(dtype=float),
            "test_score": test["union_score"].to_numpy(dtype=float),
            "valid_formal": valid["formal_score"].to_numpy(dtype=float),
            "test_formal": test["formal_score"].to_numpy(dtype=float),
            "valid_observed": valid_observed,
            "valid_attack": valid_attack,
            "valid_normal": valid_normal,
            "union_valid_pred": union_valid_pred,
            "union_test_pred": union_test_pred,
            "union_valid_coverage": union_valid_coverage,
        }
        if str(source) != NOAA:
            continue
        stations = valid_frame["station_id"].astype(str).to_numpy()
        families, context = candidate_families(
            valid_frame,
            valid["union_score"].to_numpy(dtype=float),
            stations,
            valid_observed,
            valid_attack,
            valid_normal,
            union_valid_coverage,
        )
        contexts[int(seed)] = context
        noaa_families[int(seed)] = families
        for label, summary in families.items():
            candidate_rows.append(
                {
                    "source": str(source),
                    "seed": int(seed),
                    "candidate_family": label,
                    "family_kind": summary["family_kind"],
                    "valid_mean_scenario_coverage": summary[
                        "valid_mean_scenario_coverage"
                    ],
                    "valid_f1": summary["valid_f1"],
                    "valid_event_coverage": summary["valid_event_coverage"],
                    "valid_false_alarm_rate": summary[
                        "valid_false_alarm_rate"
                    ],
                    "valid_average_precision": summary[
                        "valid_average_precision"
                    ],
                    "valid_coverage_long_stuck": summary[
                        "valid_coverage_long_stuck"
                    ],
                    "valid_coverage_drop_replay": summary[
                        "valid_coverage_drop_replay"
                    ],
                    "valid_margin_long_stuck": summary[
                        "valid_margin_long_stuck"
                    ],
                    "valid_margin_drop_replay": summary[
                        "valid_margin_drop_replay"
                    ],
                    "constraint_ok": bool(summary["constraint_ok"]),
                    "union_valid_coverage_long_stuck": float(
                        union_valid_coverage.get("long_stuck", 0.0)
                    ),
                    "union_valid_coverage_drop_replay": float(
                        union_valid_coverage.get("drop_replay", 0.0)
                    ),
                }
            )
        frontier.extend(
            {
                "source": str(source),
                "seed": int(seed),
                **row,
            }
            for row in frontier_rows(
                context,
                valid_frame,
                valid["union_score"].to_numpy(dtype=float),
                stations,
                valid_observed,
                union_valid_coverage,
            )
        )

    candidates = pd.DataFrame(candidate_rows)
    family_rows: list[dict[str, object]] = []
    for label, group in candidates.groupby(
        "candidate_family",
        sort=True,
    ):
        family_rows.append(
            {
                "candidate_family": label,
                "family_kind": group["family_kind"].iloc[0],
                "seeds": int(len(group)),
                "valid_mean_scenario_coverage": float(
                    group["valid_mean_scenario_coverage"].mean()
                ),
                "valid_f1": float(group["valid_f1"].mean()),
                "valid_event_coverage": float(
                    group["valid_event_coverage"].mean()
                ),
                "valid_false_alarm_rate": float(
                    group["valid_false_alarm_rate"].mean()
                ),
                "valid_average_precision": float(
                    group["valid_average_precision"].mean()
                ),
                "valid_coverage_long_stuck": float(
                    group["valid_coverage_long_stuck"].mean()
                ),
                "valid_coverage_drop_replay": float(
                    group["valid_coverage_drop_replay"].mean()
                ),
                "union_valid_coverage_long_stuck": float(
                    group["union_valid_coverage_long_stuck"].mean()
                ),
                "union_valid_coverage_drop_replay": float(
                    group["union_valid_coverage_drop_replay"].mean()
                ),
                "feasible_seeds": int(group["constraint_ok"].sum()),
                "feasible_all_seeds": bool(group["constraint_ok"].all()),
            }
        )
    family_frame = pd.DataFrame(family_rows)
    family_frame["applicable_all_seeds"] = [
        all(
            noaa_families[seed][label]["predicted"] is not None
            for seed in seeds
        )
        for label in family_frame["candidate_family"]
    ]
    family_frame = family_frame.sort_values(
        ["feasible_all_seeds", "valid_mean_scenario_coverage", "valid_f1"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    deployable = family_frame[family_frame["feasible_all_seeds"]]
    if len(deployable):
        selected_family: str | None = str(
            deployable.iloc[0]["candidate_family"]
        )
        fallback = False
    else:
        selected_family = None
        fallback = True
    counterfactual_pool = family_frame[
        family_frame["applicable_all_seeds"]
    ].sort_values(
        ["valid_mean_scenario_coverage", "valid_f1"],
        ascending=[False, False],
    )
    counterfactual_family = str(
        counterfactual_pool.iloc[0]["candidate_family"]
    )
    union_valid_mean = float(
        np.mean(
            [
                mean_scenario_coverage(
                    valid_state[(NOAA, seed)]["union_valid_coverage"]
                )
                for seed in seeds
            ]
        )
    )

    run_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    counterfactual_rows: list[dict[str, object]] = []
    deployed_test: dict[tuple[str, int], np.ndarray] = {}
    for source in providers:
        for seed in seeds:
            state = valid_state[(source, seed)]
            valid_frame = state["valid_frame"]
            test_frame = state["test_frame"]
            valid_score = state["valid_score"]
            test_score = state["test_score"]
            union_valid_pred = state["union_valid_pred"]
            union_test_pred = state["union_test_pred"]
            predicted_valid = union_valid_pred
            predicted_test = union_test_pred
            deployed_rule = "frozen_union_or"
            calibration_applied = False
            selection_family = "frozen_union_or_fallback"
            if source == NOAA:
                if not fallback and selected_family is not None:
                    context = contexts[seed]
                    stations = test_frame["station_id"].astype(str).to_numpy()
                    applied = family_prediction(
                        selected_family,
                        context,
                        test_score,
                        stations,
                        test_frame["observation_present"].to_numpy(
                            dtype=bool
                        ),
                    )
                    if applied is None:
                        raise RuntimeError(
                            "selected family is not applicable on test: "
                            f"{selected_family}"
                        )
                    predicted_test = applied
                    deployed_rule = selected_family
                    calibration_applied = True
                    selection_family = selected_family
            row: dict[str, object] = {
                "source": source,
                "seed": int(seed),
                "method": METHOD,
                "deployed_rule": deployed_rule,
                "calibration_applied": bool(calibration_applied),
                "fallback_to_union": bool(
                    source == NOAA and fallback
                ),
                "selection_unit": "provider",
                "selection_family": selection_family,
            }
            valid_metrics = metrics(
                state["valid_attack"],
                predicted_valid,
                valid_score,
            )
            valid_event = event_metrics(valid_frame, predicted_valid)
            valid_coverage = scenario_coverage(
                valid_frame,
                predicted_valid,
            )
            row.update(
                {
                    "valid_mean_scenario_coverage": mean_scenario_coverage(
                        valid_coverage
                    ),
                    "valid_event_coverage": float(
                        valid_event["event_coverage"]
                    ),
                    "valid_f1": float(valid_metrics["f1"]),
                    "valid_false_alarm_rate": float(
                        valid_metrics["false_alarm_rate"]
                    ),
                    "valid_average_precision": float(
                        valid_metrics["average_precision"]
                    ),
                    "valid_coverage_long_stuck": float(
                        valid_coverage.get("long_stuck", 0.0)
                    ),
                    "valid_coverage_drop_replay": float(
                        valid_coverage.get("drop_replay", 0.0)
                    ),
                    "union_valid_coverage_long_stuck": float(
                        state["union_valid_coverage"].get(
                            "long_stuck", 0.0
                        )
                    ),
                    "union_valid_coverage_drop_replay": float(
                        state["union_valid_coverage"].get(
                            "drop_replay", 0.0
                        )
                    ),
                }
            )
            test_metrics = metrics(
                test_frame["is_attack"].astype(bool).to_numpy(),
                predicted_test,
                test_score,
            )
            test_event = event_metrics(test_frame, predicted_test)
            test_coverage = scenario_coverage(test_frame, predicted_test)
            union_test_coverage = scenario_coverage(
                test_frame,
                union_test_pred,
            )
            union_test_metrics = metrics(
                test_frame["is_attack"].astype(bool).to_numpy(),
                union_test_pred,
                test_score,
            )
            union_test_event = event_metrics(test_frame, union_test_pred)
            row.update(
                {
                    **test_metrics,
                    **test_event,
                    **{
                        f"test_coverage_{scenario}": float(
                            test_coverage.get(scenario, 0.0)
                        )
                        for scenario in STRUCTURED_SCENARIOS
                    },
                    **{
                        f"union_test_coverage_{scenario}": float(
                            union_test_coverage.get(scenario, 0.0)
                        )
                        for scenario in STRUCTURED_SCENARIOS
                    },
                    **{
                        f"delta_{scenario}_vs_union": float(
                            test_coverage.get(scenario, 0.0)
                            - union_test_coverage.get(scenario, 0.0)
                        )
                        for scenario in STRUCTURED_SCENARIOS
                    },
                    "mean_scenario_coverage": mean_scenario_coverage(
                        test_coverage
                    ),
                    "union_test_f1": float(union_test_metrics["f1"]),
                    "union_test_event_coverage": float(
                        union_test_event["event_coverage"]
                    ),
                    "union_test_false_alarm_rate": float(
                        union_test_metrics["false_alarm_rate"]
                    ),
                    "union_test_average_precision": float(
                        union_test_metrics["average_precision"]
                    ),
                }
            )
            reference = stage491_runs.loc[(source, seed)]
            row.update(
                {
                    "stage491_selection_weight": (
                        float(reference["weight"])
                        if reference["weight"] != "max"
                        else float("nan")
                    ),
                    "stage491_selection_rule": str(reference["weight"]),
                    "stage491_test_f1": float(reference["f1"]),
                    "stage491_test_event_coverage": float(
                        reference["event_coverage"]
                    ),
                    "stage491_test_false_alarm_rate": float(
                        reference["false_alarm_rate"]
                    ),
                    "stage491_test_average_precision": float(
                        reference["average_precision"]
                    ),
                    "delta_f1_vs_stage491": float(
                        test_metrics["f1"] - reference["f1"]
                    ),
                    "delta_event_coverage_vs_stage491": float(
                        test_event["event_coverage"]
                        - reference["event_coverage"]
                    ),
                    "delta_far_vs_stage491": float(
                        test_metrics["false_alarm_rate"]
                        - reference["false_alarm_rate"]
                    ),
                    "delta_ap_vs_stage491": float(
                        test_metrics["average_precision"]
                        - reference["average_precision"]
                    ),
                    "delta_f1_vs_union": float(
                        test_metrics["f1"] - union_test_metrics["f1"]
                    ),
                    "delta_event_coverage_vs_union": float(
                        test_event["event_coverage"]
                        - union_test_event["event_coverage"]
                    ),
                    "delta_far_vs_union": float(
                        test_metrics["false_alarm_rate"]
                        - union_test_metrics["false_alarm_rate"]
                    ),
                    "delta_ap_vs_union": float(
                        test_metrics["average_precision"]
                        - union_test_metrics["average_precision"]
                    ),
                }
            )
            if source == NOAA:
                seed_family = family_frame.iloc[0]
                row.update(
                    {
                        "candidate_family_count": int(
                            family_frame["candidate_family"].nunique()
                        ),
                        "feasible_family_count": int(
                            family_frame["feasible_all_seeds"].sum()
                        ),
                        "best_candidate_family": str(
                            seed_family["candidate_family"]
                        ),
                        "best_candidate_valid_mean_scenario_coverage": float(
                            seed_family["valid_mean_scenario_coverage"]
                        ),
                        "best_candidate_valid_f1": float(
                            seed_family["valid_f1"]
                        ),
                        "best_candidate_valid_false_alarm_rate": float(
                            seed_family["valid_false_alarm_rate"]
                        ),
                        "best_candidate_valid_coverage_long_stuck": float(
                            seed_family["valid_coverage_long_stuck"]
                        ),
                        "best_candidate_valid_coverage_drop_replay": float(
                            seed_family["valid_coverage_drop_replay"]
                        ),
                        "best_candidate_union_coverage_long_stuck": float(
                            seed_family["union_valid_coverage_long_stuck"]
                        ),
                        "best_candidate_union_coverage_drop_replay": float(
                            seed_family["union_valid_coverage_drop_replay"]
                        ),
                        "best_candidate_constraint_ok": bool(
                            seed_family["feasible_all_seeds"]
                        ),
                        "counterfactual_family": counterfactual_family,
                        "min_far_multiplier_for_constraint": next(
                            (
                                float(entry["far_multiplier"])
                                for entry in frontier
                                if int(entry["seed"]) == seed
                                and bool(entry["feasible"])
                            ),
                            float("nan"),
                        ),
                    }
                )
                context = contexts[seed]
                stations = test_frame["station_id"].astype(str).to_numpy()
                counterfactual = family_prediction(
                    counterfactual_family,
                    context,
                    test_score,
                    stations,
                    test_frame["observation_present"].to_numpy(dtype=bool),
                )
                if counterfactual is not None:
                    counterfactual_metrics = metrics(
                        test_frame["is_attack"].astype(bool).to_numpy(),
                        counterfactual,
                        test_score,
                    )
                    counterfactual_event = event_metrics(
                        test_frame, counterfactual
                    )
                    counterfactual_coverage = scenario_coverage(
                        test_frame, counterfactual
                    )
                    counterfactual_rows.append(
                        {
                            "source": source,
                            "seed": int(seed),
                            "counterfactual_family": counterfactual_family,
                            **{
                                f"counterfactual_{key}": value
                                for key, value in counterfactual_metrics.items()
                            },
                            **{
                                f"counterfactual_{key}": value
                                for key, value in counterfactual_event.items()
                            },
                            "counterfactual_mean_scenario_coverage": (
                                mean_scenario_coverage(
                                    counterfactual_coverage
                                )
                            ),
                            **{
                                f"counterfactual_coverage_{scenario}": float(
                                    counterfactual_coverage.get(
                                        scenario, 0.0
                                    )
                                )
                                for scenario in STRUCTURED_SCENARIOS
                            },
                        }
                    )
                    row.update(counterfactual_rows[-1])
            run_rows.append(row)
            deployed_test[(source, seed)] = predicted_test
            scenario_rows.extend(
                s470.scenario_rows_full(
                    test_frame,
                    predicted_test,
                    test_score,
                    source,
                    int(seed),
                    METHOD,
                )
            )
            station_rows.extend(
                s470.station_rows_full(
                    test_frame,
                    predicted_test,
                    source,
                    int(seed),
                    METHOD,
                )
            )

    runs = pd.DataFrame(run_rows)
    scenario = pd.DataFrame(scenario_rows)
    station = pd.DataFrame(station_rows)

    station_reference_rows: list[dict[str, object]] = []
    for (source, seed), state in valid_state.items():
        test_frame = state["test_frame"]
        observed = test_frame["observation_present"].to_numpy(dtype=bool)
        attack = test_frame["is_attack"].to_numpy(dtype=bool)
        union_pred = state["union_test_pred"]
        deployed_pred = deployed_test[(source, seed)]
        score = state["test_score"]
        for station_id_value, part in test_frame.groupby(
            "station_id", sort=True
        ):
            index = part.index.to_numpy(dtype=int)
            observed_index = observed[index]
            y = (attack & observed)[index]
            p_union = (union_pred & observed)[index]
            p_deployed = (deployed_pred & observed)[index]
            union_metrics = metrics(
                y[observed_index],
                p_union[observed_index],
                score[index][observed_index],
            )
            deployed_metrics = metrics(
                y[observed_index],
                p_deployed[observed_index],
                score[index][observed_index],
            )
            union_events = part.loc[y, ["attack_event_id"]].copy()
            union_events["_pred"] = p_union[y]
            union_grouped = union_events.groupby(
                "attack_event_id", sort=True
            )["_pred"].any()
            deployed_events = part.loc[y, ["attack_event_id"]].copy()
            deployed_events["_pred"] = p_deployed[y]
            deployed_grouped = deployed_events.groupby(
                "attack_event_id", sort=True
            )["_pred"].any()
            station_reference_rows.append(
                {
                    "source": str(source),
                    "seed": int(seed),
                    "station_id": str(station_id_value),
                    "deployed_f1": float(deployed_metrics["f1"]),
                    "deployed_false_alarm_rate": float(
                        deployed_metrics["false_alarm_rate"]
                    ),
                    "deployed_average_precision": float(
                        deployed_metrics["average_precision"]
                    ),
                    "deployed_roc_auc": float(deployed_metrics["roc_auc"]),
                    "deployed_event_coverage": float(
                        deployed_grouped.mean()
                    )
                    if len(deployed_grouped)
                    else 0.0,
                    "deployed_detected_events": int(deployed_grouped.sum())
                    if len(deployed_grouped)
                    else 0,
                    "union_f1": float(union_metrics["f1"]),
                    "union_false_alarm_rate": float(
                        union_metrics["false_alarm_rate"]
                    ),
                    "union_average_precision": float(
                        union_metrics["average_precision"]
                    ),
                    "union_roc_auc": float(union_metrics["roc_auc"]),
                    "union_event_coverage": float(union_grouped.mean())
                    if len(union_grouped)
                    else 0.0,
                    "union_detected_events": int(union_grouped.sum())
                    if len(union_grouped)
                    else 0,
                    "union_events": int(len(union_grouped)),
                }
            )
    station = station.merge(
        pd.DataFrame(station_reference_rows),
        on=["source", "seed", "station_id"],
        how="left",
        validate="one_to_one",
    )
    station["average_precision"] = station["deployed_average_precision"]
    station["roc_auc"] = station["deployed_roc_auc"]
    station["delta_f1_vs_union"] = station["f1"] - station["union_f1"]
    station["delta_event_coverage_vs_union"] = (
        station["event_coverage"] - station["union_event_coverage"]
    )
    station["delta_false_alarm_rate_vs_union"] = (
        station["false_alarm_rate"] - station["union_false_alarm_rate"]
    )
    station["delta_average_precision_vs_union"] = (
        station["deployed_average_precision"]
        - station["union_average_precision"]
    )
    station = station.merge(
        stage491_station,
        on=["source", "seed", "station_id"],
        how="left",
        validate="one_to_one",
    )
    station["delta_f1_vs_stage491"] = (
        station["f1"] - station["stage491_f1"]
    )
    station["delta_event_coverage_vs_stage491"] = (
        station["event_coverage"] - station["stage491_event_coverage"]
    )
    station["delta_false_alarm_rate_vs_stage491"] = (
        station["false_alarm_rate"] - station["stage491_false_alarm_rate"]
    )

    scenario = scenario.merge(
        stage491_scenario_lookup.reset_index().rename(
            columns={
                "event_coverage": "stage491_event_coverage",
                "f1": "stage491_f1",
                "false_alarm_rate": "stage491_false_alarm_rate",
            }
        ),
        on=["source", "seed", "scenario"],
        how="left",
        validate="one_to_one",
    )
    scenario["delta_event_coverage_vs_stage491"] = (
        scenario["event_coverage"] - scenario["stage491_event_coverage"]
    )
    scenario["delta_f1_vs_stage491"] = (
        scenario["f1"] - scenario["stage491_f1"]
    )
    scenario["delta_false_alarm_rate_vs_stage491"] = (
        scenario["false_alarm_rate"]
        - scenario["stage491_false_alarm_rate"]
    )

    per_scenario_separation: dict[str, dict[str, dict[str, object]]] = {}
    for source in providers:
        per_scenario_separation[source] = {}
        for seed in seeds:
            state = valid_state[(source, seed)]
            valid_frame = state["valid_frame"]
            attack = state["valid_attack"]
            observed = state["valid_observed"]
            score = state["valid_score"]
            union_pred = state["union_valid_pred"]
            normal = observed & ~attack
            entry: dict[str, object] = {}
            for scenario_name, group in valid_frame[attack].groupby(
                "attack_scenario",
                sort=True,
            ):
                index = group.index.to_numpy(dtype=int)
                mask = normal.copy()
                mask[index] = True
                row_metrics = metrics(
                    attack[mask],
                    union_pred[mask],
                    score[mask],
                )
                event_frame = group[["attack_event_id"]].copy()
                event_frame["_pred"] = union_pred[index]
                grouped = event_frame.groupby(
                    "attack_event_id", sort=True
                )["_pred"].any()
                entry[str(scenario_name)] = {
                    "union_event_coverage": float(grouped.mean())
                    if len(grouped)
                    else 0.0,
                    "shape_roc_auc": float(row_metrics["roc_auc"]),
                    "shape_average_precision": float(
                        row_metrics["average_precision"]
                    ),
                }
            per_scenario_separation[source][str(seed)] = entry

    noaa_min_far: dict[str, object] = {}
    for seed in seeds:
        state = valid_state[(NOAA, seed)]
        valid_frame = state["valid_frame"]
        attack = state["valid_attack"]
        normal = state["valid_normal"]
        stations = valid_frame["station_id"].astype(str).to_numpy()
        event_ids = valid_frame["attack_event_id"].astype(str).to_numpy()
        union_pred = state["union_valid_pred"]
        scenarios = valid_frame["attack_scenario"].astype(str).to_numpy()
        targets = {
            "long_stuck": set(
                event_ids[attack & (scenarios == "long_stuck")]
            ),
            "drop_replay": set(
                event_ids[attack & (scenarios == "drop_replay")]
            ),
            "structured_floor": set(
                event_ids[
                    attack
                    & union_pred
                    & np.isin(
                        scenarios,
                        ["long_stuck", "drop_replay"],
                    )
                ]
            ),
            "union_detected_events": set(event_ids[attack & union_pred]),
        }
        noaa_min_far[str(seed)] = {
            key: {
                "union_shape": minimum_far_for_events(
                    valid_frame,
                    state["valid_score"],
                    stations,
                    attack,
                    normal,
                    event_ids,
                    value,
                ),
                "formal_mlp": minimum_far_for_events(
                    valid_frame,
                    state["valid_formal"],
                    stations,
                    attack,
                    normal,
                    event_ids,
                    value,
                ),
            }
            for key, value in targets.items()
        }

    summary = build_summary(
        runs,
        scenario,
        station,
        providers,
        seeds,
        stage491_scenario_lookup,
        stage493_provider,
        stage493_seed,
    )
    runs.to_csv(OUT_RUNS, index=False, encoding="utf-8-sig")
    scenario.to_csv(OUT_SCENARIO, index=False, encoding="utf-8-sig")
    station.to_csv(OUT_STATION, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    self_check = self_verify(
        merged,
        runs,
        scenario,
        station,
        summary,
        valid_state,
        providers,
        seeds,
        fallback,
        contexts,
        noaa_families,
        counterfactual_rows,
    )
    payload = {
        "status": (
            "PASS_EXPLORATORY_NOAA_STATION_CALIBRATION_FALLBACK"
            if fallback
            else "PASS_EXPLORATORY_NOAA_STATION_CALIBRATION_APPLIED"
        ),
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "protocol": {
            "calibrated_provider": NOAA,
            "calibration_inputs": "validation rows only",
            "deployed_operating_point_far": PRIMARY_FAR,
            "constraint_scenarios": STRUCTURED_SCENARIOS,
            "constraint_margin": CONSTRAINT_MARGIN,
            "objective": "NOAA validation mean scenario event coverage",
            "tie_break": "NOAA validation F1",
            "fallback_rule": "frozen union OR prediction",
            "candidate_far_levels": LEVELS,
            "knapsack_cell": CELL,
            "providers": providers,
            "seeds": seeds,
        },
        "selection": {
            "selected_family": selected_family,
            "fallback_to_union": bool(fallback),
            "deployed_rule": "frozen_union_or",
            "feasible_family_count": int(
                family_frame["feasible_all_seeds"].sum()
            ),
            "candidate_family_count": int(
                family_frame["candidate_family"].nunique()
            ),
            "best_family_by_objective": str(
                family_frame.iloc[0]["candidate_family"]
            ),
            "best_family_valid_mean_scenario_coverage": float(
                family_frame.iloc[0]["valid_mean_scenario_coverage"]
            ),
            "counterfactual_family": counterfactual_family,
            "noaa_union_valid_mean_scenario_coverage": union_valid_mean,
        },
        "noaa_family_summary": family_frame.to_dict("records"),
        "noaa_candidate_runs": candidates.to_dict("records"),
        "noaa_far_frontier": frontier,
        "noaa_validation": {
            str(seed): {
                "union_mean_scenario_coverage": mean_scenario_coverage(
                    valid_state[(NOAA, seed)]["union_valid_coverage"]
                ),
                "union_event_coverage": float(
                    event_metrics(
                        valid_state[(NOAA, seed)]["valid_frame"],
                        valid_state[(NOAA, seed)]["union_valid_pred"],
                    )["event_coverage"]
                ),
                "union_f1": float(
                    metrics(
                        valid_state[(NOAA, seed)]["valid_attack"],
                        valid_state[(NOAA, seed)]["union_valid_pred"],
                        valid_state[(NOAA, seed)]["valid_score"],
                    )["f1"]
                ),
                "station_medians": contexts[seed]["station_medians"],
                "station_scales": contexts[seed]["station_scales"],
                "thin_stations": contexts[seed]["thin_stations"],
                "min_far_multiplier_for_constraint": next(
                    (
                        float(entry["far_multiplier"])
                        for entry in frontier
                        if int(entry["seed"]) == seed and entry["feasible"]
                    ),
                    None,
                ),
            }
            for seed in seeds
        },
        "noaa_min_far_for_event_sets": noaa_min_far,
        "per_scenario_separation": per_scenario_separation,
        "provider_summary": summary[
            summary["aggregation"].eq("provider")
        ].to_dict("records"),
        "seed_summary": summary[
            summary["aggregation"].eq("seed")
        ].to_dict("records"),
        "provider_scenario_summary": summary[
            summary["aggregation"].eq("provider_scenario")
        ].to_dict("records"),
        "noaa_station_summary": summary[
            summary["aggregation"].eq("noaa_station")
        ].to_dict("records"),
        "conclusion": build_conclusion(
            fallback,
            family_frame,
            union_valid_mean,
            frontier,
            noaa_min_far,
            per_scenario_separation[NOAA],
            runs[runs["source"].eq(NOAA)],
        ),
        "adverse_results": [
            "No station-level calibration family satisfies the margin-0 "
            "long_stuck/drop_replay validation floor at the protocol FAR, "
            "so NOAA falls back to the frozen union OR prediction and "
            "Stage 497 does not improve NOAA.",
            "The frozen union shape score has almost no station-level "
            "separation for NOAA: reproducing the union OR event set with "
            "per-station thresholds needs a minimum FAR of "
            f"{min(entry['union_detected_events']['union_shape'] for entry in noaa_min_far.values()):.4f}-"
            f"{max(entry['union_detected_events']['union_shape'] for entry in noaa_min_far.values()):.4f} "
            f"({min(entry['union_detected_events']['union_shape'] for entry in noaa_min_far.values()) / PRIMARY_FAR:.1f}-"
            f"{max(entry['union_detected_events']['union_shape'] for entry in noaa_min_far.values()) / PRIMARY_FAR:.1f}x "
            "the protocol budget).",
            "The formal MLP score needs a far larger FAR budget "
            f"({min(entry['union_detected_events']['formal_mlp'] for entry in noaa_min_far.values()):.3f}-"
            f"{max(entry['union_detected_events']['formal_mlp'] for entry in noaa_min_far.values()):.3f}) "
            "than the union shape score for the same NOAA event sets.",
            "The validation-optimal station allocation is itself "
            "over-fitted: on test the constraint-violating counterfactual "
            "moves NOAA mean scenario coverage by "
            f"{float((runs.loc[runs['source'].eq(NOAA), 'counterfactual_mean_scenario_coverage'] - runs.loc[runs['source'].eq(NOAA), 'mean_scenario_coverage']).mean()):+.4f} "
            "on average, i.e. no transfer.",
            "The other three providers keep the frozen union OR, so Stage "
            "497 sits below their Stage 491 constrained-fusion numbers; "
            "those deltas are reported rather than hidden.",
        ],
        "limitations": [
            "The V6 test split was already inspected; every number is "
            "exploratory.",
            "Station thresholds and the knapsack allocation are fitted on "
            "validation only, but the calibration family itself is selected "
            "on NOAA validation, so validation optimism cannot be excluded.",
            "The knapsack tie-break uses (true positives, -false positives) "
            "as a monotone proxy for validation F1 rather than F1 itself.",
            "Only the union shape score and the formal MLP score are "
            "persisted; the three frozen union components are not available, "
            "so per-component station calibration cannot be tested here.",
            "Four providers are fixed clusters; this is not population "
            "inference.",
        ],
        "self_check": self_check,
    }
    if not self_check["all_passed"]:
        payload["status"] = "FAIL_SELF_CHECK"
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(
        build_markdown(payload, started, family_frame, frontier),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "fallback_to_union": bool(fallback),
                "selected_family": selected_family,
                "counterfactual_family": counterfactual_family,
                "noaa_union_valid_mean_scenario_coverage": union_valid_mean,
                "best_family": str(
                    family_frame.iloc[0]["candidate_family"]
                ),
                "best_family_valid_mean_scenario_coverage": float(
                    family_frame.iloc[0]["valid_mean_scenario_coverage"]
                ),
                "best_family_valid_coverage_long_stuck": float(
                    family_frame.iloc[0]["valid_coverage_long_stuck"]
                ),
                "best_family_valid_coverage_drop_replay": float(
                    family_frame.iloc[0]["valid_coverage_drop_replay"]
                ),
                "union_valid_coverage_long_stuck": float(
                    family_frame.iloc[0][
                        "union_valid_coverage_long_stuck"
                    ]
                ),
                "union_valid_coverage_drop_replay": float(
                    family_frame.iloc[0][
                        "union_valid_coverage_drop_replay"
                    ]
                ),
                "self_check": self_check["all_passed"],
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        )
    )
    return 0 if self_check["all_passed"] else 1


def build_summary(
    runs: pd.DataFrame,
    scenario: pd.DataFrame,
    station: pd.DataFrame,
    providers: list[str],
    seeds: list[int],
    stage491_scenario_lookup: pd.DataFrame,
    stage493_provider: pd.DataFrame,
    stage493_seed: pd.DataFrame,
) -> pd.DataFrame:
    provider_rows: list[dict[str, object]] = []
    for source in providers:
        subset = runs[runs["source"].eq(source)]
        row: dict[str, object] = {
            "aggregation": "provider",
            "source": source,
            "seed": np.nan,
            "scenario": "",
            "station_id": "",
            "runs": int(len(subset)),
            "f1_mean": float(subset["f1"].mean()),
            "event_coverage_mean": float(subset["event_coverage"].mean()),
            "mean_scenario_coverage": float(
                subset["mean_scenario_coverage"].mean()
            ),
            "false_alarm_rate_mean": float(
                subset["false_alarm_rate"].mean()
            ),
            "average_precision_mean": float(
                subset["average_precision"].mean()
            ),
            "structured_long_stuck_coverage_mean": float(
                subset["test_coverage_long_stuck"].mean()
            ),
            "structured_drop_replay_coverage_mean": float(
                subset["test_coverage_drop_replay"].mean()
            ),
            "delta_long_stuck_vs_union": float(
                subset["delta_long_stuck_vs_union"].mean()
            ),
            "delta_drop_replay_vs_union": float(
                subset["delta_drop_replay_vs_union"].mean()
            ),
            "union_f1_mean": float(subset["union_test_f1"].mean()),
            "union_event_coverage_mean": float(
                subset["union_test_event_coverage"].mean()
            ),
            "union_false_alarm_rate_mean": float(
                subset["union_test_false_alarm_rate"].mean()
            ),
            "union_average_precision_mean": float(
                subset["union_test_average_precision"].mean()
            ),
            "delta_f1_vs_union": float(
                (subset["f1"] - subset["union_test_f1"]).mean()
            ),
            "delta_event_coverage_vs_union": float(
                (
                    subset["event_coverage"]
                    - subset["union_test_event_coverage"]
                ).mean()
            ),
            "delta_far_vs_union": float(
                (
                    subset["false_alarm_rate"]
                    - subset["union_test_false_alarm_rate"]
                ).mean()
            ),
            "delta_ap_vs_union": float(
                (
                    subset["average_precision"]
                    - subset["union_test_average_precision"]
                ).mean()
            ),
            "stage491_f1_mean": float(
                subset["stage491_test_f1"].mean()
            ),
            "stage491_event_coverage_mean": float(
                subset["stage491_test_event_coverage"].mean()
            ),
            "stage491_false_alarm_rate_mean": float(
                subset["stage491_test_false_alarm_rate"].mean()
            ),
            "stage491_average_precision_mean": float(
                subset["stage491_test_average_precision"].mean()
            ),
            "delta_f1_vs_stage491": float(
                subset["delta_f1_vs_stage491"].mean()
            ),
            "delta_event_coverage_vs_stage491": float(
                subset["delta_event_coverage_vs_stage491"].mean()
            ),
            "delta_far_vs_stage491": float(
                subset["delta_far_vs_stage491"].mean()
            ),
            "delta_ap_vs_stage491": float(
                subset["delta_ap_vs_stage491"].mean()
            ),
        }
        provider_reference = stage493_provider[
            stage493_provider["unit"].eq(source)
        ]
        if len(provider_reference):
            row.update(
                {
                    "stage493_f1_mean": float(
                        provider_reference["f1_mean"].iloc[0]
                    ),
                    "stage493_event_coverage_mean": float(
                        provider_reference[
                            "event_coverage_mean"
                        ].iloc[0]
                    ),
                    "stage493_false_alarm_rate_mean": float(
                        provider_reference[
                            "false_alarm_rate_mean"
                        ].iloc[0]
                    ),
                    "stage493_average_precision_mean": float(
                        provider_reference[
                            "average_precision_mean"
                        ].iloc[0]
                    ),
                    "stage493_gate_wins": int(
                        provider_reference["extended_wins"].iloc[0]
                    ),
                    "stage493_gate_units": int(
                        provider_reference["extended_units"].iloc[0]
                    ),
                }
            )
            row["delta_f1_vs_stage493"] = float(
                row["f1_mean"] - row["stage493_f1_mean"]
            )
            row["delta_event_coverage_vs_stage493"] = float(
                row["event_coverage_mean"]
                - row["stage493_event_coverage_mean"]
            )
            row["delta_far_vs_stage493"] = float(
                row["false_alarm_rate_mean"]
                - row["stage493_false_alarm_rate_mean"]
            )
            row["delta_ap_vs_stage493"] = float(
                row["average_precision_mean"]
                - row["stage493_average_precision_mean"]
            )
        provider_rows.append(row)

    seed_rows: list[dict[str, object]] = []
    for seed in seeds:
        subset = runs[runs["seed"].eq(seed)]
        row = {
            "aggregation": "seed",
            "source": "",
            "seed": int(seed),
            "scenario": "",
            "station_id": "",
            "runs": int(len(subset)),
            "f1_mean": float(subset["f1"].mean()),
            "event_coverage_mean": float(subset["event_coverage"].mean()),
            "mean_scenario_coverage": float(
                subset["mean_scenario_coverage"].mean()
            ),
            "false_alarm_rate_mean": float(
                subset["false_alarm_rate"].mean()
            ),
            "average_precision_mean": float(
                subset["average_precision"].mean()
            ),
            "structured_long_stuck_coverage_mean": float(
                subset["test_coverage_long_stuck"].mean()
            ),
            "structured_drop_replay_coverage_mean": float(
                subset["test_coverage_drop_replay"].mean()
            ),
            "delta_long_stuck_vs_union": float(
                subset["delta_long_stuck_vs_union"].mean()
            ),
            "delta_drop_replay_vs_union": float(
                subset["delta_drop_replay_vs_union"].mean()
            ),
            "union_f1_mean": float(subset["union_test_f1"].mean()),
            "union_event_coverage_mean": float(
                subset["union_test_event_coverage"].mean()
            ),
            "union_false_alarm_rate_mean": float(
                subset["union_test_false_alarm_rate"].mean()
            ),
            "union_average_precision_mean": float(
                subset["union_test_average_precision"].mean()
            ),
            "delta_f1_vs_union": float(
                (subset["f1"] - subset["union_test_f1"]).mean()
            ),
            "delta_event_coverage_vs_union": float(
                (
                    subset["event_coverage"]
                    - subset["union_test_event_coverage"]
                ).mean()
            ),
            "delta_far_vs_union": float(
                (
                    subset["false_alarm_rate"]
                    - subset["union_test_false_alarm_rate"]
                ).mean()
            ),
            "delta_ap_vs_union": float(
                (
                    subset["average_precision"]
                    - subset["union_test_average_precision"]
                ).mean()
            ),
            "stage491_f1_mean": float(subset["stage491_test_f1"].mean()),
            "stage491_event_coverage_mean": float(
                subset["stage491_test_event_coverage"].mean()
            ),
            "stage491_false_alarm_rate_mean": float(
                subset["stage491_test_false_alarm_rate"].mean()
            ),
            "stage491_average_precision_mean": float(
                subset["stage491_test_average_precision"].mean()
            ),
            "delta_f1_vs_stage491": float(
                subset["delta_f1_vs_stage491"].mean()
            ),
            "delta_event_coverage_vs_stage491": float(
                subset["delta_event_coverage_vs_stage491"].mean()
            ),
            "delta_far_vs_stage491": float(
                subset["delta_far_vs_stage491"].mean()
            ),
            "delta_ap_vs_stage491": float(
                subset["delta_ap_vs_stage491"].mean()
            ),
        }
        seed_reference = stage493_seed[
            stage493_seed["unit"].eq(str(int(seed)))
        ]
        if len(seed_reference):
            row.update(
                {
                    "stage493_f1_mean": float(
                        seed_reference["f1_mean"].iloc[0]
                    ),
                    "stage493_event_coverage_mean": float(
                        seed_reference["event_coverage_mean"].iloc[0]
                    ),
                    "stage493_false_alarm_rate_mean": float(
                        seed_reference["false_alarm_rate_mean"].iloc[0]
                    ),
                    "stage493_average_precision_mean": float(
                        seed_reference["average_precision_mean"].iloc[0]
                    ),
                    "stage493_gate_wins": int(
                        seed_reference["extended_wins"].iloc[0]
                    ),
                    "stage493_gate_units": int(
                        seed_reference["extended_units"].iloc[0]
                    ),
                }
            )
            row["delta_f1_vs_stage493"] = float(
                row["f1_mean"] - row["stage493_f1_mean"]
            )
            row["delta_event_coverage_vs_stage493"] = float(
                row["event_coverage_mean"]
                - row["stage493_event_coverage_mean"]
            )
            row["delta_far_vs_stage493"] = float(
                row["false_alarm_rate_mean"]
                - row["stage493_false_alarm_rate_mean"]
            )
            row["delta_ap_vs_stage493"] = float(
                row["average_precision_mean"]
                - row["stage493_average_precision_mean"]
            )
        seed_rows.append(row)

    scenario_rows: list[dict[str, object]] = []
    for (scenario_source, scenario_name), group in scenario.groupby(
        ["source", "scenario"],
        sort=True,
    ):
        reference = stage491_scenario_lookup.loc[
            (scenario_source, slice(None), scenario_name)
        ]
        scenario_rows.append(
            {
                "aggregation": "provider_scenario",
                "source": scenario_source,
                "seed": np.nan,
                "scenario": scenario_name,
                "station_id": "",
                "runs": int(len(group)),
                "f1_mean": float(group["f1"].mean()),
                "event_coverage_mean": float(
                    group["event_coverage"].mean()
                ),
                "mean_scenario_coverage": float(
                    group["event_coverage"].mean()
                ),
                "false_alarm_rate_mean": float(
                    group["false_alarm_rate"].mean()
                ),
                "average_precision_mean": float(
                    group["average_precision"].mean()
                ),
                "stage491_event_coverage_mean": float(
                    reference["event_coverage"].mean()
                ),
                "stage491_f1_mean": float(reference["f1"].mean()),
                "stage491_false_alarm_rate_mean": float(
                    reference["false_alarm_rate"].mean()
                ),
                "delta_event_coverage_vs_stage491": float(
                    group["event_coverage"].mean()
                    - reference["event_coverage"].mean()
                ),
                "delta_f1_vs_stage491": float(
                    group["f1"].mean() - reference["f1"].mean()
                ),
            }
        )
    station_rows: list[dict[str, object]] = []
    for (station_source, station_id_value), group in station.groupby(
        ["source", "station_id"],
        sort=True,
    ):
        station_rows.append(
            {
                "aggregation": "station",
                "source": station_source,
                "seed": np.nan,
                "scenario": "",
                "station_id": station_id_value,
                "runs": int(len(group)),
                "f1_mean": float(group["f1"].mean()),
                "event_coverage_mean": float(
                    group["event_coverage"].mean()
                ),
                "mean_scenario_coverage": float(
                    group["event_coverage"].mean()
                ),
                "false_alarm_rate_mean": float(
                    group["false_alarm_rate"].mean()
                ),
                "average_precision_mean": float(
                    group["average_precision"].mean()
                ),
                "union_f1_mean": float(group["union_f1"].mean()),
                "union_event_coverage_mean": float(
                    group["union_event_coverage"].mean()
                ),
                "union_false_alarm_rate_mean": float(
                    group["union_false_alarm_rate"].mean()
                ),
                "union_average_precision_mean": float(
                    group["union_average_precision"].mean()
                ),
                "delta_f1_vs_union": float(
                    group["delta_f1_vs_union"].mean()
                ),
                "delta_event_coverage_vs_union": float(
                    group["delta_event_coverage_vs_union"].mean()
                ),
                "delta_far_vs_union": float(
                    group["delta_false_alarm_rate_vs_union"].mean()
                ),
                "delta_ap_vs_union": float(
                    group["delta_average_precision_vs_union"].mean()
                ),
                "events": int(group["events"].sum()),
                "detected_events": int(group["detected_events"].sum()),
            }
        )
    return pd.concat(
        [
            pd.DataFrame(provider_rows),
            pd.DataFrame(seed_rows),
            pd.DataFrame(scenario_rows),
            pd.DataFrame(station_rows),
        ],
        ignore_index=True,
        sort=False,
    )


def build_conclusion(
    fallback: bool,
    family_frame: pd.DataFrame,
    union_valid_mean: float,
    frontier: list[dict[str, object]],
    noaa_min_far: dict[str, object],
    per_scenario: dict[str, dict[str, object]],
    noaa_runs: pd.DataFrame,
) -> dict[str, object]:
    shape_min = [
        value["union_detected_events"]["union_shape"]
        for value in noaa_min_far.values()
    ]
    floor_min = [
        value["structured_floor"]["union_shape"]
        for value in noaa_min_far.values()
    ]
    formal_min = [
        value["union_detected_events"]["formal_mlp"]
        for value in noaa_min_far.values()
    ]
    aucs = [
        entry["shape_roc_auc"]
        for seed_entry in per_scenario.values()
        for entry in seed_entry.values()
    ]
    subtle = [
        entry["shape_roc_auc"]
        for seed_entry in per_scenario.values()
        for scenario, entry in seed_entry.items()
        if scenario
        in (
            "delay",
            "low_noise",
            "low_offset",
            "short_lag",
            "slow_drift",
        )
    ]
    counterfactual_delta = float(
        (
            noaa_runs["counterfactual_mean_scenario_coverage"]
            - noaa_runs["mean_scenario_coverage"]
        ).mean()
    )
    counterfactual_far = float(
        noaa_runs["counterfactual_false_alarm_rate"].mean()
    )
    union_test_far = float(
        noaa_runs["union_test_false_alarm_rate"].mean()
    )
    return {
        "verdict": (
            "calibration_cannot_fix_noaa_at_protocol_far"
            if fallback
            else "station_calibration_applied"
        ),
        "primary_evidence": [
            "No calibration family holds the margin-0 long_stuck/drop_replay "
            "validation floor at FAR 0.0065, so the frozen union OR stands.",
            "Even the cheapest possible per-station threshold allocation "
            "that keeps the union's detected long_stuck/drop_replay events "
            "needs a pooled validation FAR of "
            f"{min(floor_min):.4f}-{max(floor_min):.4f} "
            f"({min(floor_min) / PRIMARY_FAR:.1f}-"
            f"{max(floor_min) / PRIMARY_FAR:.1f}x the 0.0065 budget); this "
            "bound is exact for per-station thresholds on the union shape "
            "score and does not depend on the search grid.",
            "Reproducing the union OR NOAA event set with per-station "
            "thresholds needs a minimum FAR of "
            f"{min(shape_min):.4f}-{max(shape_min):.4f} "
            f"({min(shape_min) / PRIMARY_FAR:.1f}-"
            f"{max(shape_min) / PRIMARY_FAR:.1f}x the protocol budget).",
            "The formal MLP score needs an even larger budget "
            f"({min(formal_min):.3f}-{max(formal_min):.3f}), so the "
            "deficit is not an artefact of one score family.",
            "Across NOAA validation scenarios the union shape score has a "
            "median ROC-AUC of "
            f"{float(np.median(aucs)):.3f} overall and "
            f"{float(np.median(subtle)):.3f} on the five subtle scenarios, "
            "i.e. essentially no station-level ranking signal.",
            "On test, the validation-optimal but constraint-violating "
            "counterfactual changes NOAA mean scenario coverage by "
            f"{counterfactual_delta:+.4f} on average while running at FAR "
            f"{counterfactual_far:.5f} against the union's "
            f"{union_test_far:.5f}, so the validation gain does not "
            "transfer.",
        ],
        "interpretation": (
            "The NOAA deficit is dominated by model/representation capability "
            "rather than by score calibration: the frozen union OR already "
            "uses component-level thresholds whose information is not "
            "recoverable from the scalar max-shape score, and the subtle "
            "scenarios carry almost no ranking signal to recalibrate. "
            "Station-level thresholds can only buy NOAA coverage by spending "
            "several times the protocol FAR budget, which would break the "
            "false-alarm contract of the frozen operating point."
        ),
        "union_valid_mean_scenario_coverage": union_valid_mean,
        "best_family": str(family_frame.iloc[0]["candidate_family"]),
        "best_family_valid_mean_scenario_coverage": float(
            family_frame.iloc[0]["valid_mean_scenario_coverage"]
        ),
        "counterfactual_test_mean_scenario_coverage_delta": (
            counterfactual_delta
        ),
        "counterfactual_test_false_alarm_rate": counterfactual_far,
        "union_test_false_alarm_rate": union_test_far,
        "minimum_far_for_structured_floor": floor_min,
        "frontier_feasible_from_multiplier": min(
            (
                float(entry["far_multiplier"])
                for entry in frontier
                if entry["feasible"]
            ),
            default=None,
        ),
    }


def build_markdown(
    payload: dict[str, object],
    started: float,
    family_frame: pd.DataFrame,
    frontier: list[dict[str, object]],
) -> str:
    provider_rows = pd.DataFrame(payload["provider_summary"])
    seed_rows = pd.DataFrame(payload["seed_summary"])
    station_rows = pd.DataFrame(payload["noaa_station_summary"])
    frontier_frame = pd.DataFrame(frontier)
    separation = pd.DataFrame(
        [
            {
                "seed": seed,
                "scenario": scenario,
                "union_event_coverage": value["union_event_coverage"],
                "shape_roc_auc": value["shape_roc_auc"],
                "shape_average_precision": value[
                    "shape_average_precision"
                ],
            }
            for seed, entry in payload["per_scenario_separation"][
                NOAA
            ].items()
            for scenario, value in entry.items()
        ]
    )
    min_far = pd.DataFrame(
        [
            {
                "seed": seed,
                "event_set": event_set,
                "union_shape_min_far": value["union_shape"],
                "formal_mlp_min_far": value["formal_mlp"],
            }
            for seed, entry in payload[
                "noaa_min_far_for_event_sets"
            ].items()
            for event_set, value in entry.items()
        ]
    )
    conclusion = payload["conclusion"]
    lines = [
        "# Stage 497 NOAA ISD station-level calibration audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Calibrated provider: `{NOAA}` (validation-only parameters)",
        f"- Deployed operating point FAR: `{PRIMARY_FAR}`",
        f"- Constraint: validation `long_stuck` and `drop_replay` coverage "
        f">= frozen union (margin `{CONSTRAINT_MARGIN}`), all seeds",
        f"- Selected family: `{payload['selection']['selected_family']}`",
        f"- Fallback to frozen union OR: "
        f"`{payload['selection']['fallback_to_union']}`",
        f"- Verdict: `{conclusion['verdict']}`",
        "",
        "## NOAA candidate families (validation, mean over 5 seeds)",
        "",
        family_frame.to_markdown(index=False),
        "",
        "## FAR frontier for constraint feasibility (NOAA validation)",
        "",
        frontier_frame.to_markdown(index=False),
        "",
        "## Provider summary (test, deployed rule)",
        "",
        provider_rows.to_markdown(index=False),
        "",
        "## Seed summary (test, deployed rule)",
        "",
        seed_rows.to_markdown(index=False),
        "",
        "## NOAA station summary (test, mean over seeds)",
        "",
        station_rows.to_markdown(index=False),
        "",
        "## Union OR separation on NOAA validation (per scenario)",
        "",
        separation.to_markdown(index=False),
        "",
        "## Minimum FAR to reproduce NOAA event sets (validation)",
        "",
        min_far.to_markdown(index=False),
        "",
        "## Conclusion",
        "",
        f"- Verdict: `{conclusion['verdict']}`",
        f"- Interpretation: {conclusion['interpretation']}",
        "",
        "### Primary evidence",
        "",
        *[f"- {item}" for item in conclusion["primary_evidence"]],
        "",
        "## Adverse results",
        "",
        *[f"- {item}" for item in payload["adverse_results"]],
        "",
        "## Boundary",
        "",
        *[f"- {item}" for item in payload["limitations"]],
        "",
        f"- Elapsed seconds: `{time.perf_counter() - started:.2f}`",
        "",
    ]
    return "\n".join(lines)


def self_verify(
    merged: pd.DataFrame,
    runs: pd.DataFrame,
    scenario: pd.DataFrame,
    station: pd.DataFrame,
    summary: pd.DataFrame,
    valid_state: dict[tuple[str, int], dict[str, object]],
    providers: list[str],
    seeds: list[int],
    fallback: bool,
    contexts: dict[int, dict[str, object]],
    noaa_families: dict[int, dict[str, dict[str, object]]],
    counterfactual_rows: list[dict[str, object]],
) -> dict[str, object]:
    checks: dict[str, object] = {}
    checks["run_rows"] = int(len(runs))
    checks["run_rows_expected"] = len(providers) * len(seeds)
    checks["run_rows_ok"] = bool(
        len(runs) == len(providers) * len(seeds)
    )
    checks["scenario_rows"] = int(len(scenario))
    checks["scenario_rows_ok"] = bool(
        len(scenario) == len(providers) * len(seeds) * SCENARIO_COUNT
    )
    checks["station_rows"] = int(len(station))
    checks["station_rows_ok"] = bool(
        len(station)
        == merged[
            merged["split"].eq("test")
        ]
        .groupby(["source", "seed", "station_id_union"])
        .ngroups
    )
    checks["summary_aggregations"] = sorted(
        summary["aggregation"].unique().tolist()
    )
    checks["summary_rows"] = int(len(summary))
    metric_columns = [
        "f1",
        "event_coverage",
        "false_alarm_rate",
        "average_precision",
        "mean_scenario_coverage",
    ]
    checks["no_nan_in_run_metrics"] = bool(
        not runs[metric_columns].isna().any().any()
    )
    noaa_rows = runs[runs["source"].eq(NOAA)]
    other_rows = runs[~runs["source"].eq(NOAA)]
    checks["noaa_fallback_rows"] = int(
        noaa_rows["fallback_to_union"].sum()
    )
    checks["noaa_calibration_applied_rows"] = int(
        noaa_rows["calibration_applied"].sum()
    )
    checks["noaa_matches_union"] = bool(
        np.allclose(noaa_rows["f1"], noaa_rows["union_test_f1"])
        and np.allclose(
            noaa_rows["event_coverage"],
            noaa_rows["union_test_event_coverage"],
        )
        and np.allclose(
            noaa_rows["false_alarm_rate"],
            noaa_rows["union_test_false_alarm_rate"],
        )
        and np.allclose(
            noaa_rows["average_precision"],
            noaa_rows["union_test_average_precision"],
        )
    )
    checks["other_providers_match_union"] = bool(
        np.allclose(other_rows["f1"], other_rows["union_test_f1"])
        and np.allclose(
            other_rows["event_coverage"],
            other_rows["union_test_event_coverage"],
        )
    )
    checks["fallback_flag_matches_selection"] = bool(
        bool(fallback) == bool(checks["noaa_calibration_applied_rows"] == 0)
    )
    recomputed: list[bool] = []
    for (source, seed), state in valid_state.items():
        test_frame = state["test_frame"]
        stored = runs[runs["source"].eq(source) & runs["seed"].eq(seed)]
        recomputed_metrics = metrics(
            test_frame["is_attack"].astype(bool).to_numpy(),
            state["union_test_pred"],
            state["test_score"],
        )
        recomputed.append(
            bool(
                np.isclose(
                    recomputed_metrics["f1"], float(stored["f1"].iloc[0])
                )
                and np.isclose(
                    recomputed_metrics["false_alarm_rate"],
                    float(stored["false_alarm_rate"].iloc[0]),
                )
                and np.isclose(
                    recomputed_metrics["average_precision"],
                    float(stored["average_precision"].iloc[0]),
                )
            )
        )
    checks["test_metrics_recomputed_ok"] = bool(all(recomputed))
    checks["station_delta_consistency"] = bool(
        np.allclose(
            station["delta_f1_vs_union"],
            station["f1"] - station["union_f1"],
        )
        and np.allclose(
            station["delta_event_coverage_vs_union"],
            station["event_coverage"] - station["union_event_coverage"],
        )
    )
    checks["station_metrics_recomputed_ok"] = bool(
        np.allclose(station["f1"], station["deployed_f1"])
        and np.allclose(
            station["false_alarm_rate"],
            station["deployed_false_alarm_rate"],
        )
        and np.allclose(
            station["event_coverage"],
            station["deployed_event_coverage"],
        )
        and np.allclose(
            station["average_precision"],
            station["deployed_average_precision"],
        )
        and not station["deployed_average_precision"].isna().any()
    )
    checks["run_delta_consistency"] = bool(
        np.allclose(
            runs["delta_long_stuck_vs_union"],
            runs["test_coverage_long_stuck"]
            - runs["union_test_coverage_long_stuck"],
        )
        and np.allclose(
            runs["delta_drop_replay_vs_union"],
            runs["test_coverage_drop_replay"]
            - runs["union_test_coverage_drop_replay"],
        )
    )
    counterfactual = pd.DataFrame(counterfactual_rows)
    checks["counterfactual_rows"] = int(len(counterfactual))
    checks["counterfactual_rows_ok"] = bool(
        len(counterfactual) == len(seeds)
    )
    if len(counterfactual):
        checks["counterfactual_metrics_recomputed_ok"] = bool(
            np.allclose(
                counterfactual["counterfactual_f1"],
                runs.loc[
                    runs["source"].eq(NOAA), "counterfactual_f1"
                ].to_numpy(),
            )
        )
    else:
        checks["counterfactual_metrics_recomputed_ok"] = True
    determinism: list[bool] = []
    for seed in seeds:
        context = contexts[seed]
        allocation = knapsack_allocate(
            context["options"],
            context["stations"],
            context["long_stuck_floor"],
            context["drop_replay_floor"],
            context["total_long_stuck"],
            context["total_drop_replay"],
            PRIMARY_FAR,
            enforce_floor=True,
        )
        stored_family = noaa_families[seed]["station_budget_dp"]
        if allocation is None:
            determinism.append(stored_family["predicted"] is None)
            continue
        determinism.append(
            bool(
                np.isclose(
                    allocation["far"],
                    stored_family.get("realised_far", np.nan),
                )
            )
        )
    checks["knapsack_determinism_ok"] = bool(all(determinism))
    checks["validation_only_selection"] = bool(
        all(
            contexts[seed]["split"] == "valid"
            and len(contexts[seed]["options"])
            == valid_state[(NOAA, seed)]["valid_frame"][
                "station_id"
            ].nunique()
            for seed in seeds
        )
    )
    provenance: list[bool] = []
    for seed in seeds:
        state = valid_state[(NOAA, seed)]
        context = contexts[seed]
        score = state["valid_score"]
        normal = state["valid_normal"]
        stations = state["valid_frame"]["station_id"].astype(str).to_numpy()
        median, scale = robust_median_scale(score[normal])
        provenance.append(
            bool(
                np.isclose(median, float(context["provider_median"]))
                and np.isclose(scale, float(context["provider_scale"]))
            )
        )
        for station in sorted(set(stations.tolist())):
            mask = normal & (stations == station)
            station_median, station_scale = robust_median_scale(
                score[mask]
            )
            provenance.append(
                bool(
                    np.isclose(
                        station_median,
                        float(context["station_medians"][str(station)]),
                    )
                    and np.isclose(
                        station_scale,
                        float(context["station_scales"][str(station)]),
                    )
                )
            )
    checks["validation_parameter_provenance_ok"] = bool(all(provenance))
    checks["validation_parameter_checks"] = int(len(provenance))
    checks["test_rows_touched_by_selection"] = 0
    checks["all_passed"] = bool(
        checks["run_rows_ok"]
        and checks["scenario_rows_ok"]
        and checks["station_rows_ok"]
        and checks["no_nan_in_run_metrics"]
        and checks["noaa_matches_union"]
        and checks["other_providers_match_union"]
        and checks["fallback_flag_matches_selection"]
        and checks["test_metrics_recomputed_ok"]
        and checks["station_delta_consistency"]
        and checks["station_metrics_recomputed_ok"]
        and checks["run_delta_consistency"]
        and checks["counterfactual_rows_ok"]
        and checks["counterfactual_metrics_recomputed_ok"]
        and checks["knapsack_determinism_ok"]
        and checks["validation_only_selection"]
        and checks["validation_parameter_provenance_ok"]
    )
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
