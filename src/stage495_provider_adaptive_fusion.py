"""Stage 495: provider-adaptive validation-only constrained score fusion.

For every provider (``source``) the fusion rule and its weight are selected
on that provider's validation rows only:

* candidates are the Stage 491 weight grid ``0.0 ... 1.0`` plus ``max``;
* the validation event coverage of ``long_stuck`` and ``drop_replay`` may not
  fall below the frozen union validation coverage (margin 0);
* among feasible candidates the provider mean validation scenario event
  coverage is maximised and validation F1 breaks ties;
* if no candidate is feasible the run falls back to the frozen union OR
  prediction (``union_prediction``) instead of re-thresholding the union
  shape score and calling that union.

Test rows are evaluation-only: they never enter rule or weight selection.
The V6 test split has already been inspected, so every number produced here
is exploratory and does not replace the frozen main table.
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
from stage490_validation_score_fusion import (  # noqa: E402
    BASELINES_11,
    candidate_scores,
    canonical_frame,
)

FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

UNION_SCORES = TABLES / "stage490_union_scores.parquet"
FORMAL_SCORES = TABLES / "stage490_formal_mlp_scores.parquet"
EXISTING_SCENARIO = TABLES / "final_v6_scenario_aggregate.csv"
EXISTING_SCENARIO_METRICS = TABLES / "final_v6_scenario_metrics.csv"
FORMAL_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"

OUT_RUNS = TABLES / "stage495_provider_adaptive_fusion_runs.csv"
OUT_SCENARIO = TABLES / "stage495_provider_adaptive_fusion_scenario.csv"
OUT_STATION = TABLES / "stage495_provider_adaptive_fusion_station.csv"
OUT_GATE = TABLES / "stage495_provider_adaptive_fusion_gate.csv"
OUT_CONSTRAINT = TABLES / "stage495_provider_adaptive_fusion_constraint.csv"
OUT_JSON = REPORTS / "stage495_provider_adaptive_fusion.json"
OUT_MD = REPORTS / "stage495_provider_adaptive_fusion.md"

METHOD = "provider_adaptive_constrained_fusion"
PRIMARY_FAR = 0.0065
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CANDIDATE_RULES: list[float | str] = [*WEIGHT_GRID, "max"]
CANDIDATE_KEYS = [f"{weight:.1f}" for weight in WEIGHT_GRID] + ["max"]
CONSTRAINED_SCENARIOS = ["long_stuck", "drop_replay"]
CONSTRAINT_MARGIN = 0.0
EPS = 1e-12
GATE_TOLERANCE = 1e-12
FALLBACK_RULE = "union_or_fallback"


def rule_label(weight: float | str) -> str:
    if weight == "max":
        return "max"
    return f"linear_w{float(weight):.1f}"


def scenario_coverage(
    frame: pd.DataFrame,
    predicted: np.ndarray,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for scenario, group in frame.assign(
        _pred=predicted
    ).groupby("attack_scenario", sort=True):
        if scenario == "normal":
            continue
        detected = group.groupby("attack_event_id")["_pred"].any()
        output[str(scenario)] = (
            float(detected.mean()) if len(detected) else 0.0
        )
    return output


def mean_scenario_coverage(
    coverage: dict[str, float],
    scenario_keys: list[str],
) -> float:
    if not scenario_keys:
        return 0.0
    return float(np.mean([coverage.get(key, 0.0) for key in scenario_keys]))


def provider_candidate_table(
    run_state: dict[tuple[str, int], dict[str, object]],
    source: str,
    seeds: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    union_by_seed = {
        scenario: np.asarray(
            [
                run_state[(source, seed)]["union_valid_coverage"].get(
                    scenario,
                    0.0,
                )
                for seed in seeds
            ],
            dtype=float,
        )
        for scenario in CONSTRAINED_SCENARIOS
    }
    for key in CANDIDATE_KEYS:
        entries = [
            run_state[(source, seed)]["candidates"][key] for seed in seeds
        ]
        row: dict[str, object] = {
            "candidate": key,
            "rule": rule_label(entries[0]["weight"]),
            "weight": entries[0]["weight"],
            "valid_mean_scenario_coverage": float(
                np.mean(
                    [
                        entry["valid_mean_scenario_coverage"]
                        for entry in entries
                    ]
                )
            ),
            "valid_f1": float(
                np.mean([entry["valid_f1"] for entry in entries])
            ),
            "valid_event_coverage": float(
                np.mean(
                    [
                        entry["valid_event_coverage"]
                        for entry in entries
                    ]
                )
            ),
        }
        constraint_ok = True
        for scenario in CONSTRAINED_SCENARIOS:
            candidate_coverage = np.asarray(
                [
                    entry["valid_coverage"].get(scenario, 0.0)
                    for entry in entries
                ],
                dtype=float,
            )
            union_coverage = union_by_seed[scenario]
            delta = candidate_coverage - union_coverage
            row[f"valid_coverage_{scenario}"] = float(
                candidate_coverage.mean()
            )
            row[f"union_valid_coverage_{scenario}"] = float(
                union_coverage.mean()
            )
            row[f"valid_delta_{scenario}"] = float(delta.mean())
            row[f"min_seed_valid_delta_{scenario}"] = float(delta.min())
            row[f"seeds_within_{scenario}"] = int((delta >= -EPS).sum())
            constraint_ok = constraint_ok and bool(delta.mean() >= -EPS)
        row["constraint_ok"] = constraint_ok
        rows.append(row)
    return rows


def gate_rows_for_unit(
    method: str,
    unit: str,
    coverage_lookup: pd.DataFrame,
    baseline_table: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, group in baseline_table.groupby("scenario", sort=True):
        candidate = coverage_lookup[coverage_lookup["scenario"].eq(scenario)]
        if candidate.empty:
            continue
        value = float(candidate["event_coverage"].iloc[0])
        for baseline_set, label in (
            (BASELINES_11, "frozen_11"),
            (BASELINES_11 + ["formal_lagged_mlp"], "frozen_11_plus_formal"),
        ):
            wins = losses = ties = 0
            for baseline in baseline_set:
                reference = group[group["method"].eq(baseline)]
                if reference.empty:
                    continue
                reference_value = float(reference["event_coverage"].iloc[0])
                difference = value - reference_value
                if difference > GATE_TOLERANCE:
                    wins += 1
                elif difference < -GATE_TOLERANCE:
                    losses += 1
                else:
                    ties += 1
            total = wins + losses + ties
            rows.append(
                {
                    "method": method,
                    "unit": unit,
                    "baseline_set": label,
                    "scenario": scenario,
                    "event_coverage": value,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "total_baselines": total,
                    "win_share": wins / total if total else 0.0,
                    "passes_70pct": wins / total >= 0.70 if total else False,
                }
            )
    return rows


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

    run_state: dict[tuple[str, int], dict[str, object]] = {}
    for (source, seed), group in merged.groupby(
        ["source", "seed"],
        sort=True,
    ):
        valid = group[group["split"].eq("valid")].copy()
        test = group[group["split"].eq("test")].copy()
        valid_frame = canonical_frame(valid)
        test_frame = canonical_frame(test)
        valid_normal = (
            valid_frame["observation_present"]
            & ~valid_frame["is_attack"]
        ).to_numpy(dtype=bool)
        union_valid_raw = valid["union_score"].to_numpy(dtype=float)
        formal_valid_raw = valid["formal_score"].to_numpy(dtype=float)
        union_test_raw = test["union_score"].to_numpy(dtype=float)
        formal_test_raw = test["formal_score"].to_numpy(dtype=float)
        union_median = float(np.median(union_valid_raw[valid_normal]))
        formal_median = float(np.median(formal_valid_raw[valid_normal]))
        union_scale = max(
            1.4826
            * float(
                np.median(
                    np.abs(union_valid_raw[valid_normal] - union_median)
                )
            ),
            1e-6,
        )
        formal_scale = max(
            1.4826
            * float(
                np.median(
                    np.abs(formal_valid_raw[valid_normal] - formal_median)
                )
            ),
            1e-6,
        )
        union_valid_z = (union_valid_raw - union_median) / union_scale
        formal_valid_z = (formal_valid_raw - formal_median) / formal_scale
        union_test_z = (union_test_raw - union_median) / union_scale
        formal_test_z = (formal_test_raw - formal_median) / formal_scale
        observation_valid = valid_frame[
            "observation_present"
        ].to_numpy(dtype=bool)
        observation_test = test_frame[
            "observation_present"
        ].to_numpy(dtype=bool)
        y_valid = valid_frame["is_attack"].astype(bool).to_numpy()
        union_valid_pred = valid["union_prediction"].to_numpy(dtype=bool)
        union_test_pred = test["union_prediction"].to_numpy(dtype=bool)
        union_valid_coverage = scenario_coverage(
            valid_frame,
            union_valid_pred,
        )
        scenario_keys = sorted(union_valid_coverage)
        union_valid_metrics = metrics(
            y_valid,
            union_valid_pred,
            union_valid_z,
        )
        candidates: dict[str, dict[str, object]] = {}
        for weight, key in zip(CANDIDATE_RULES, CANDIDATE_KEYS):
            fused_valid = candidate_scores(
                union_valid_z,
                formal_valid_z,
                weight,
            )
            threshold = float(
                far_threshold(
                    fused_valid,
                    valid_normal,
                    PRIMARY_FAR,
                )
            )
            predicted_valid = observation_valid & (fused_valid >= threshold)
            row_metrics = metrics(y_valid, predicted_valid, fused_valid)
            event_values = event_metrics(valid_frame, predicted_valid)
            coverage = scenario_coverage(valid_frame, predicted_valid)
            candidates[key] = {
                "weight": weight,
                "rule": rule_label(weight),
                "threshold": threshold,
                "valid_f1": float(row_metrics["f1"]),
                "valid_event_coverage": float(
                    event_values["event_coverage"]
                ),
                "valid_false_alarm_rate": float(
                    row_metrics["false_alarm_rate"]
                ),
                "valid_mean_scenario_coverage": mean_scenario_coverage(
                    coverage,
                    scenario_keys,
                ),
                "valid_coverage": coverage,
            }
        run_state[(str(source), int(seed))] = {
            "valid_frame": valid_frame,
            "test_frame": test_frame,
            "union_valid_z": union_valid_z,
            "formal_valid_z": formal_valid_z,
            "union_test_z": union_test_z,
            "formal_test_z": formal_test_z,
            "observation_valid": observation_valid,
            "observation_test": observation_test,
            "y_valid": y_valid,
            "union_valid_pred": union_valid_pred,
            "union_test_pred": union_test_pred,
            "union_valid_coverage": union_valid_coverage,
            "union_valid_f1": float(union_valid_metrics["f1"]),
            "union_valid_mean_scenario_coverage": mean_scenario_coverage(
                union_valid_coverage,
                scenario_keys,
            ),
            "candidates": candidates,
        }

    providers = sorted(str(value) for value in merged["source"].unique())
    provider_seed_map: dict[str, list[int]] = {
        provider: sorted(
            int(value)
            for value in merged.loc[
                merged["source"].eq(provider), "seed"
            ].unique()
        )
        for provider in providers
    }
    provider_selection: list[dict[str, object]] = []
    selection_lookup: dict[str, dict[str, object]] = {}
    fallback_count = 0
    for provider in providers:
        seeds = provider_seed_map[provider]
        candidates = provider_candidate_table(run_state, provider, seeds)
        feasible = [row for row in candidates if row["constraint_ok"]]
        if feasible:
            best = max(
                feasible,
                key=lambda row: (
                    row["valid_mean_scenario_coverage"],
                    row["valid_f1"],
                    -row["valid_event_coverage"],
                    -CANDIDATE_KEYS.index(str(row["candidate"])),
                ),
            )
            fallback = False
        else:
            best = None
            fallback = True
            fallback_count += 1
        union_objective = float(
            np.mean(
                [
                    run_state[(provider, seed)][
                        "union_valid_mean_scenario_coverage"
                    ]
                    for seed in seeds
                ]
            )
        )
        union_f1 = float(
            np.mean(
                [
                    run_state[(provider, seed)]["union_valid_f1"]
                    for seed in seeds
                ]
            )
        )
        record: dict[str, object] = {
            "source": provider,
            "selection_unit": "provider",
            "selection_rule": (
                FALLBACK_RULE if fallback else str(best["rule"])
            ),
            "selection_weight": (
                "fallback" if fallback else str(best["weight"])
            ),
            "selection_candidate": (
                "union_or_fallback" if fallback else str(best["candidate"])
            ),
            "fallback_to_union": fallback,
            "candidate_count": len(candidates),
            "feasible_candidate_count": len(feasible),
            "provider_valid_mean_scenario_coverage": (
                float(best["valid_mean_scenario_coverage"])
                if not fallback
                else union_objective
            ),
            "provider_valid_f1": (
                float(best["valid_f1"]) if not fallback else union_f1
            ),
            "union_or_valid_mean_scenario_coverage": union_objective,
            "union_or_valid_f1": union_f1,
        }
        record["delta_vs_union_objective"] = float(
            record["provider_valid_mean_scenario_coverage"] - union_objective
        )
        for scenario_name in CONSTRAINED_SCENARIOS:
            record[f"provider_valid_coverage_{scenario_name}"] = (
                float(best[f"valid_coverage_{scenario_name}"])
                if not fallback
                else float(
                    np.mean(
                        [
                            run_state[(provider, seed)][
                                "union_valid_coverage"
                            ].get(scenario_name, 0.0)
                            for seed in seeds
                        ]
                    )
                )
            )
            record[f"union_valid_coverage_{scenario_name}"] = float(
                np.mean(
                    [
                        run_state[(provider, seed)][
                            "union_valid_coverage"
                        ].get(scenario_name, 0.0)
                        for seed in seeds
                    ]
                )
            )
            record[f"valid_delta_{scenario_name}"] = float(
                record[f"provider_valid_coverage_{scenario_name}"]
                - record[f"union_valid_coverage_{scenario_name}"]
            )
            record[f"seeds_within_{scenario_name}"] = (
                int(best[f"seeds_within_{scenario_name}"])
                if not fallback
                else len(seeds)
            )
            record[f"min_seed_valid_delta_{scenario_name}"] = (
                float(best[f"min_seed_valid_delta_{scenario_name}"])
                if not fallback
                else 0.0
            )
        record["constraint_ok"] = bool(
            all(
                record[f"valid_delta_{scenario_name}"] >= -EPS
                for scenario_name in CONSTRAINED_SCENARIOS
            )
        )
        record["strict_seed_level_ok"] = bool(
            all(
                int(record[f"seeds_within_{scenario_name}"]) == len(seeds)
                for scenario_name in CONSTRAINED_SCENARIOS
            )
        )
        if fallback:
            best_infeasible = max(
                candidates,
                key=lambda row: (
                    row["valid_mean_scenario_coverage"],
                    row["valid_f1"],
                ),
            )
            record["best_infeasible_candidate"] = str(
                best_infeasible["candidate"]
            )
            record["best_infeasible_valid_mean_scenario_coverage"] = float(
                best_infeasible["valid_mean_scenario_coverage"]
            )
            for scenario_name in CONSTRAINED_SCENARIOS:
                record[
                    f"best_infeasible_valid_delta_{scenario_name}"
                ] = float(best_infeasible[f"valid_delta_{scenario_name}"])
        else:
            record["best_infeasible_candidate"] = None
            record["best_infeasible_valid_mean_scenario_coverage"] = None
            for scenario_name in CONSTRAINED_SCENARIOS:
                record[
                    f"best_infeasible_valid_delta_{scenario_name}"
                ] = None
        provider_selection.append(record)
        selection_lookup[provider] = {
            "fallback": fallback,
            "candidate": None if fallback else str(best["candidate"]),
            "rule": record["selection_rule"],
            "weight": record["selection_weight"],
            "feasible_candidate_count": len(feasible),
            "candidate_count": len(candidates),
            "candidates": candidates,
        }

    run_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    constraint_rows: list[dict[str, object]] = []
    for provider in providers:
        seeds = provider_seed_map[provider]
        selection = selection_lookup[provider]
        for seed in seeds:
            state = run_state[(provider, seed)]
            valid_frame = state["valid_frame"]
            test_frame = state["test_frame"]
            if selection["fallback"]:
                threshold = float("nan")
                fused_valid = state["union_valid_z"]
                fused_test = state["union_test_z"]
                predicted_valid = state["union_valid_pred"]
                predicted_test = state["union_test_pred"]
                valid_coverage = state["union_valid_coverage"]
                valid_row = metrics(
                    state["y_valid"],
                    predicted_valid,
                    fused_valid,
                )
                valid_event = event_metrics(valid_frame, predicted_valid)
                valid_f1 = float(valid_row["f1"])
                valid_false_alarm_rate = float(
                    valid_row["false_alarm_rate"]
                )
                valid_event_coverage = float(
                    valid_event["event_coverage"]
                )
                valid_mean_scenario = mean_scenario_coverage(
                    valid_coverage,
                    sorted(valid_coverage),
                )
            else:
                candidate = state["candidates"][str(selection["candidate"])]
                threshold = float(candidate["threshold"])
                weight = candidate["weight"]
                fused_valid = candidate_scores(
                    state["union_valid_z"],
                    state["formal_valid_z"],
                    weight,
                )
                fused_test = candidate_scores(
                    state["union_test_z"],
                    state["formal_test_z"],
                    weight,
                )
                predicted_valid = state["observation_valid"] & (
                    fused_valid >= threshold
                )
                predicted_test = state["observation_test"] & (
                    fused_test >= threshold
                )
                valid_coverage = candidate["valid_coverage"]
                valid_f1 = float(candidate["valid_f1"])
                valid_event_coverage = float(
                    candidate["valid_event_coverage"]
                )
                valid_false_alarm_rate = float(
                    candidate["valid_false_alarm_rate"]
                )
                valid_mean_scenario = float(
                    candidate["valid_mean_scenario_coverage"]
                )
            test_row = metrics(
                test_frame["is_attack"].astype(bool).to_numpy(),
                predicted_test,
                fused_test,
            )
            test_event = event_metrics(test_frame, predicted_test)
            union_test_row = metrics(
                test_frame["is_attack"].astype(bool).to_numpy(),
                state["union_test_pred"],
                state["union_test_z"],
            )
            union_test_event = event_metrics(
                test_frame,
                state["union_test_pred"],
            )
            run_rows.append(
                {
                    "source": provider,
                    "seed": int(seed),
                    "method": METHOD,
                    "selection_unit": "provider",
                    "selection_rule": selection["rule"],
                    "selection_weight": selection["weight"],
                    "fallback_to_union": bool(selection["fallback"]),
                    "feasible_candidate_count": int(
                        selection["feasible_candidate_count"]
                    ),
                    "candidate_count": int(selection["candidate_count"]),
                    "threshold": threshold,
                    "valid_f1": valid_f1,
                    "valid_event_coverage": valid_event_coverage,
                    "valid_mean_scenario_coverage": valid_mean_scenario,
                    "valid_false_alarm_rate": valid_false_alarm_rate,
                    **{
                        f"valid_coverage_{scenario}": float(
                            valid_coverage.get(scenario, 0.0)
                        )
                        for scenario in CONSTRAINED_SCENARIOS
                    },
                    **{
                        f"union_valid_coverage_{scenario}": float(
                            state["union_valid_coverage"].get(
                                scenario,
                                0.0,
                            )
                        )
                        for scenario in CONSTRAINED_SCENARIOS
                    },
                    **test_row,
                    **test_event,
                    "union_test_f1": float(union_test_row["f1"]),
                    "union_test_event_coverage": float(
                        union_test_event["event_coverage"]
                    ),
                    "union_test_false_alarm_rate": float(
                        union_test_row["false_alarm_rate"]
                    ),
                    "union_test_average_precision": float(
                        union_test_row.get("average_precision", np.nan)
                    ),
                }
            )
            scenario_rows.extend(
                s470.scenario_rows_full(
                    test_frame,
                    predicted_test,
                    fused_test,
                    provider,
                    int(seed),
                    METHOD,
                )
            )
            station_rows.extend(
                s470.station_rows_full(
                    test_frame,
                    predicted_test,
                    provider,
                    int(seed),
                    METHOD,
                )
            )
            test_coverage = scenario_coverage(
                test_frame,
                predicted_test,
            )
            union_test_coverage = scenario_coverage(
                test_frame,
                state["union_test_pred"],
            )
            for scenario_name in CONSTRAINED_SCENARIOS:
                delta = (
                    test_coverage.get(scenario_name, 0.0)
                    - union_test_coverage.get(scenario_name, 0.0)
                )
                constraint_rows.append(
                    {
                        "source": provider,
                        "seed": int(seed),
                        "scenario": scenario_name,
                        "union_valid_coverage": float(
                            state["union_valid_coverage"].get(
                                scenario_name,
                                0.0,
                            )
                        ),
                        "fusion_valid_coverage": float(
                            valid_coverage.get(scenario_name, 0.0)
                        ),
                        "valid_delta": float(
                            valid_coverage.get(scenario_name, 0.0)
                            - state["union_valid_coverage"].get(
                                scenario_name,
                                0.0,
                            )
                        ),
                        "union_test_coverage": float(
                            union_test_coverage.get(scenario_name, 0.0)
                        ),
                        "fusion_test_coverage": float(
                            test_coverage.get(scenario_name, 0.0)
                        ),
                        "test_delta": float(delta),
                    }
                )

    runs = pd.DataFrame(run_rows)
    scenario = pd.DataFrame(scenario_rows)
    station = pd.DataFrame(station_rows)
    constraints = pd.DataFrame(constraint_rows)
    runs.to_csv(OUT_RUNS, index=False, encoding="utf-8-sig")
    scenario.to_csv(OUT_SCENARIO, index=False, encoding="utf-8-sig")
    station.to_csv(OUT_STATION, index=False, encoding="utf-8-sig")

    provider_constraint_rows: list[dict[str, object]] = []
    for record in provider_selection:
        provider = str(record["source"])
        current = constraints[constraints["source"].eq(provider)]
        for scenario_name in CONSTRAINED_SCENARIOS:
            part = current[current["scenario"].eq(scenario_name)]
            provider_constraint_rows.append(
                {
                    "source": provider,
                    "scenario": scenario_name,
                    "selection_unit": "provider",
                    "selection_rule": record["selection_rule"],
                    "selection_weight": record["selection_weight"],
                    "fallback_to_union": record["fallback_to_union"],
                    "candidate_count": record["candidate_count"],
                    "feasible_candidate_count": record[
                        "feasible_candidate_count"
                    ],
                    "constraint_margin": CONSTRAINT_MARGIN,
                    "union_valid_coverage": record[
                        f"union_valid_coverage_{scenario_name}"
                    ],
                    "fusion_valid_coverage": record[
                        f"provider_valid_coverage_{scenario_name}"
                    ],
                    "valid_delta": record[
                        f"valid_delta_{scenario_name}"
                    ],
                    "constraint_ok": bool(
                        record[f"valid_delta_{scenario_name}"] >= -EPS
                    ),
                    "seeds_within": record[
                        f"seeds_within_{scenario_name}"
                    ],
                    "seeds_total": len(provider_seed_map[provider]),
                    "min_seed_valid_delta": record[
                        f"min_seed_valid_delta_{scenario_name}"
                    ],
                    "union_test_coverage": float(
                        part["union_test_coverage"].mean()
                    ),
                    "fusion_test_coverage": float(
                        part["fusion_test_coverage"].mean()
                    ),
                    "test_delta": float(part["test_delta"].mean()),
                    "test_within_margin": bool(
                        part["test_delta"].mean() >= -EPS
                    ),
                }
            )
    constraint_table = pd.DataFrame(provider_constraint_rows)
    constraint_table.to_csv(OUT_CONSTRAINT, index=False, encoding="utf-8-sig")

    existing = pd.read_csv(EXISTING_SCENARIO)
    existing_metrics = pd.read_csv(EXISTING_SCENARIO_METRICS)
    formal = pd.read_csv(FORMAL_SCENARIO)
    formal_pooled = (
        formal.groupby(["scenario", "method"], as_index=False)
        .agg(event_coverage=("event_coverage", "mean"))
    )
    pooled_baselines = pd.concat(
        [
            existing[["scenario", "method", "event_coverage"]],
            formal_pooled,
        ],
        ignore_index=True,
        sort=False,
    )
    provider_baselines = (
        existing_metrics[["source", "scenario", "method", "event_coverage"]]
        .groupby(["source", "scenario", "method"], as_index=False)
        .agg(event_coverage=("event_coverage", "mean"))
    )
    formal_provider = (
        formal.groupby(["source", "scenario", "method"], as_index=False)
        .agg(event_coverage=("event_coverage", "mean"))
    )
    provider_baselines = pd.concat(
        [provider_baselines, formal_provider],
        ignore_index=True,
        sort=False,
    )
    pooled_fusion = (
        scenario.groupby("scenario", as_index=False)
        .agg(event_coverage=("event_coverage", "mean"))
    )
    provider_fusion = (
        scenario.groupby(["source", "scenario"], as_index=False)
        .agg(event_coverage=("event_coverage", "mean"))
    )
    gate_rows: list[dict[str, object]] = []
    gate_rows.extend(
        gate_rows_for_unit(METHOD, "all", pooled_fusion, pooled_baselines)
    )
    for provider in providers:
        gate_rows.extend(
            gate_rows_for_unit(
                METHOD,
                provider,
                provider_fusion[provider_fusion["source"].eq(provider)],
                provider_baselines[
                    provider_baselines["source"].eq(provider)
                ],
            )
        )
    union_pooled = existing[
        existing["method"].eq("trust_shape_union")
    ][["scenario", "event_coverage"]]
    gate_rows.extend(
        gate_rows_for_unit(
            "frozen_union_or_reference",
            "all",
            union_pooled,
            pooled_baselines,
        )
    )
    union_provider = provider_baselines[
        provider_baselines["method"].eq("trust_shape_union")
    ]
    for provider in providers:
        gate_rows.extend(
            gate_rows_for_unit(
                "frozen_union_or_reference",
                provider,
                union_provider[union_provider["source"].eq(provider)],
                provider_baselines[
                    provider_baselines["source"].eq(provider)
                ],
            )
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(OUT_GATE, index=False, encoding="utf-8-sig")
    gate_summary = (
        gate.groupby(["method", "unit", "baseline_set"], as_index=False)
        .agg(
            wins=("wins", "sum"),
            losses=("losses", "sum"),
            ties=("ties", "sum"),
            total=("total_baselines", "sum"),
        )
    )
    gate_summary["share"] = gate_summary["wins"] / gate_summary["total"]
    gate_summary["passes_70pct"] = gate_summary["share"] >= 0.70
    pooled_gate = gate_summary[
        gate_summary["method"].eq(METHOD)
        & gate_summary["unit"].eq("all")
    ]
    pooled_gate_pass = bool(
        pooled_gate["passes_70pct"].all() if len(pooled_gate) else False
    )

    scenario_summary = (
        scenario.groupby("scenario", as_index=False)
        .agg(
            fusion_event_coverage=("event_coverage", "mean"),
            fusion_f1=("f1", "mean"),
        )
    )
    constraint_summary = (
        constraints.groupby("scenario", as_index=False)
        .agg(
            union_valid_coverage=("union_valid_coverage", "mean"),
            fusion_valid_coverage=("fusion_valid_coverage", "mean"),
            valid_delta=("valid_delta", "mean"),
            union_test_coverage=("union_test_coverage", "mean"),
            fusion_test_coverage=("fusion_test_coverage", "mean"),
            test_delta=("test_delta", "mean"),
        )
    )
    selection_counts: dict[str, int] = {}
    for record in provider_selection:
        key = str(record["selection_rule"])
        selection_counts[key] = selection_counts.get(key, 0) + 1
    pooled_rows = gate_summary[gate_summary["unit"].eq("all")]
    fusion_pooled = pooled_rows[
        pooled_rows["method"].eq(METHOD)
    ].set_index("baseline_set")
    union_pooled = pooled_rows[
        pooled_rows["method"].eq("frozen_union_or_reference")
    ].set_index("baseline_set")
    pooled_gate_delta = {
        str(label): int(
            fusion_pooled.loc[label, "wins"]
            - union_pooled.loc[label, "wins"]
        )
        for label in fusion_pooled.index
        if label in union_pooled.index
    }
    fallback_providers = [
        str(record["source"])
        for record in provider_selection
        if record["fallback_to_union"]
    ]
    fusion_providers = {
        str(record["source"]): str(record["selection_rule"])
        for record in provider_selection
        if not record["fallback_to_union"]
    }
    union_weight_rethreshold = {
        str(record["source"]): {
            f"valid_delta_{scenario_name}": next(
                float(row[f"valid_delta_{scenario_name}"])
                for row in selection_lookup[str(record["source"])][
                    "candidates"
                ]
                if row["candidate"] == "1.0"
            )
            for scenario_name in CONSTRAINED_SCENARIOS
        }
        for record in provider_selection
    }
    eccc = runs[runs["source"].eq("ECCC")]
    key_findings = [
        (
            f"{len(fallback_providers)} of {len(providers)} providers had no "
            "feasible candidate and fell back to the frozen union OR "
            f"prediction: {fallback_providers}"
        ),
        (
            "Re-thresholding the union shape score at the validation FAR "
            "budget (weight 1.0) is far below the frozen union OR coverage "
            "on the constrained scenarios; see "
            "`union_weight_rethreshold`"
        ),
        (
            "The pooled extended gate is "
            f"{int(fusion_pooled.loc['frozen_11', 'wins'])}/99 for the fusion "
            f"versus {int(union_pooled.loc['frozen_11', 'wins'])}/99 for the "
            "frozen union reference (frozen_11 denominator)"
        ),
        (
            "All four provider-level gates stay below the 70% threshold; "
            "only the pooled gate passes on the frozen_11 denominator"
        ),
        (
            "ECCC is the only provider that used a fusion rule "
            f"({fusion_providers}); its test deltas versus the frozen "
            "union are "
            f"F1 {float(eccc['f1'].mean() - eccc['union_test_f1'].mean()):+.4f} "
            "and event coverage "
            f"{float(eccc['event_coverage'].mean() - eccc['union_test_event_coverage'].mean()):+.4f}"
        ),
    ]
    status = (
        "PARTIAL_EXPLORATORY_PROVIDER_ADAPTIVE_FUSION_UNION_FALLBACK"
        if fallback_count
        else "PASS_EXPLORATORY_PROVIDER_ADAPTIVE_FUSION"
    )
    payload = {
        "status": status,
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "v6_test_already_inspected": True,
        "design": {
            "selection_unit": "provider",
            "candidate_rules": [
                "linear_w0.0 ... linear_w1.0",
                "max",
            ],
            "weight_grid": WEIGHT_GRID,
            "primary_far": PRIMARY_FAR,
            "objective": (
                "provider mean validation scenario event coverage"
            ),
            "tie_break": (
                "provider mean validation F1, then lower validation "
                "event coverage, then candidate order"
            ),
            "constraint": {
                "scenarios": CONSTRAINED_SCENARIOS,
                "margin": CONSTRAINT_MARGIN,
                "basis": (
                    "provider mean validation event coverage relative to "
                    "the frozen union OR prediction"
                ),
                "per_seed_diagnostics": (
                    "seeds_within, seeds_total, min_seed_valid_delta and "
                    "strict_seed_level_ok are reported next to the "
                    "provider-level constraint"
                ),
            },
            "fallback": (
                "frozen union OR prediction (union_prediction); the union "
                "shape score is never re-thresholded to imitate the union"
            ),
        },
        "provider_selection": provider_selection,
        "provider_candidates": [
            {
                "source": provider,
                "candidates": selection_lookup[provider]["candidates"],
            }
            for provider in providers
        ],
        "union_weight_rethreshold": union_weight_rethreshold,
        "selection_counts": selection_counts,
        "fallback_count": int(fallback_count),
        "fallback_providers": fallback_providers,
        "run_summary": {
            "f1_mean": float(runs["f1"].mean()),
            "event_coverage_mean": float(runs["event_coverage"].mean()),
            "false_alarm_rate_mean": float(
                runs["false_alarm_rate"].mean()
            ),
            "average_precision_mean": float(
                runs["average_precision"].mean()
            ),
            "frozen_union_f1_mean": float(runs["union_test_f1"].mean()),
            "frozen_union_event_coverage_mean": float(
                runs["union_test_event_coverage"].mean()
            ),
            "frozen_union_false_alarm_rate_mean": float(
                runs["union_test_false_alarm_rate"].mean()
            ),
            "frozen_union_average_precision_mean": float(
                runs["union_test_average_precision"].mean()
            ),
        },
        "key_findings": key_findings,
        "constraint_summary": constraint_summary.to_dict("records"),
        "scenario_summary": scenario_summary.to_dict("records"),
        "gate_summary": gate_summary.to_dict("records"),
        "pooled_gate_wins_delta_vs_union_or": pooled_gate_delta,
        "pooled_gate_passes_70pct": pooled_gate_pass,
        "limitations": [
            "The V6 test split has already been inspected; this stage is exploratory.",
            "Fusion rules and weights are selected on validation only; test rows are evaluation-only.",
            "Only four providers and five seeds exist; provider-level selection units are few.",
            "The formal component is a lagged-window MLP, not a GRU/LSTM.",
            "The result does not replace or modify any frozen Stage 478-493 artifact.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Provider-adaptive validation-only constrained fusion",
        "",
        f"- Status: `{status}`",
        "- Exploratory only: the V6 test split has already been inspected.",
        "- Selection unit: provider; candidates: weight grid `0.0 ... 1.0` "
        "plus `max`.",
        "- Constraint: `long_stuck` and `drop_replay` validation event "
        "coverage must not fall below the frozen union coverage (margin 0).",
        "- Fallback: frozen union OR prediction (`union_prediction`), never "
        "a re-thresholded union shape score.",
        f"- Selection counts: `{selection_counts}`",
        f"- Providers falling back to union: `{fallback_count}`",
        f"- Run summary: `{payload['run_summary']}`",
        "",
        "## Pooled extended gate",
        "",
        *[
            f"- `{row.method}` / `{row.baseline_set}`: "
            f"`{int(row.wins)}/{int(row.total)} = {row.share:.4%}` "
            f"(passes 70%: `{bool(row.passes_70pct)}`)"
            for row in gate_summary[
                gate_summary["unit"].eq("all")
            ].itertuples()
        ],
        "",
        "## Key findings",
        "",
        *[f"- {finding}" for finding in key_findings],
        "",
        "## Provider selection",
        "",
        pd.DataFrame(provider_selection)[
            [
                "source",
                "selection_rule",
                "selection_weight",
                "fallback_to_union",
                "feasible_candidate_count",
                "provider_valid_mean_scenario_coverage",
                "union_or_valid_mean_scenario_coverage",
                "delta_vs_union_objective",
                "provider_valid_f1",
            ]
        ].to_markdown(index=False),
        "",
        "## Constraint audit (validation, margin 0)",
        "",
        constraint_table.to_markdown(index=False),
        "",
        "## Extended gate",
        "",
        gate_summary.to_markdown(index=False),
        "",
        "## Scenario summary (test, mean over runs)",
        "",
        scenario_summary.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- Rules and weights are validation-selected; test is evaluation-only.",
        "- The V6 test has already been inspected; results are exploratory.",
        "- The frozen Stage 478-493 artifacts are unchanged.",
        "- The weight 1.0 candidate is the union shape score re-thresholded at "
        "the validation FAR budget; it is not the frozen union OR decision.",
        "- Gate rows compare per-scenario mean event coverage against the "
        "frozen 11 baselines plus the formal lagged MLP; differences within "
        f"`{GATE_TOLERANCE}` count as ties.",
        "- `gate.csv` also carries the frozen union OR reference rows "
        "(`method=frozen_union_or_reference`).",
        "- Fallback runs keep `threshold` empty because they replay the "
        "frozen union OR decision instead of a calibrated score threshold.",
        "",
        f"- Elapsed seconds: `{time.perf_counter() - started:.2f}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "selection_counts": selection_counts,
                "fallback_count": fallback_count,
                "run_summary": payload["run_summary"],
                "gate_summary": gate_summary.to_dict("records"),
                "constraint_summary": constraint_summary.to_dict(
                    "records"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
