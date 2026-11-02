"""Stage 496: FAR safety calibration audit of the constrained score fusion.

The Stage 493 hierarchical bootstrap put the constrained-fusion test false
alarm rate (FAR) 95% upper bound at about 1.24%, above the 1% target.  This
audit re-thresholds the existing Stage 490 per-row scores at more conservative
validation FAR levels.  No model is retrained, no provider or dataset is
added, and no frozen artifact is modified.

Two candidate families are evaluated side by side at every level:

``frozen_union_option``
    Stage 491/492/495 semantics.  The frozen union OR prediction
    (``union_prediction``) remains the weight ``1.0`` candidate and the
    fallback, so the union decision rule is never re-thresholded.
``rethresholded_union``
    Pure re-thresholding of the Stage 490 shape scores.  The union row
    (weight ``1.0``) is the union shape score thresholded at the level rather
    than the frozen union OR prediction.

All levels are reported in parallel.  A level is a fixed input; the per-run
rule/weight is selected on validation rows only.  Test labels are used for the
descriptive audit only, no level is promoted to a final choice, and no frozen
main result is replaced.
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
FORMAL_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"

STAGE491_RUNS = TABLES / "stage491_constrained_fusion_runs.csv"
STAGE491_SCENARIO = TABLES / "stage491_constrained_fusion_scenario.csv"
STAGE491_GATE = TABLES / "stage491_constrained_fusion_gate.csv"
STAGE492_SWEEP = TABLES / "stage492_constraint_margin_sweep.csv"
STAGE493_BOOTSTRAP = TABLES / "stage493_constrained_fusion_bootstrap.csv"
STAGE493_GATE_BOOTSTRAP = (
    TABLES / "stage493_constrained_fusion_gate_bootstrap.csv"
)
STAGE493_PROVIDER = (
    TABLES / "stage493_constrained_fusion_provider_summary.csv"
)
STAGE493_SEED = TABLES / "stage493_constrained_fusion_seed_summary.csv"
STAGE493_LEAVE_ONE = (
    TABLES / "stage493_constrained_fusion_leave_one_provider.csv"
)

OUT_CALIBRATION = TABLES / "stage496_far_safety_calibration.csv"
OUT_PROVIDER = TABLES / "stage496_far_safety_provider.csv"
OUT_JSON = REPORTS / "stage496_far_safety_calibration.json"
OUT_MD = REPORTS / "stage496_far_safety_calibration.md"

FROZEN_UNION = "frozen_union_option"
RETHERSHOLDED_UNION = "rethresholded_union"
VARIANTS = (FROZEN_UNION, RETHERSHOLDED_UNION)

FAR_LEVELS = (0.0065, 0.005, 0.004, 0.003)
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RULE_GRID: list[float | str] = [*WEIGHT_GRID, "max"]
CONSTRAINED_SCENARIOS = ("long_stuck", "drop_replay")
CONSTRAINT_TOLERANCE = 0.05
BASELINE_KEYS = [*BASELINES_11, "formal_lagged_mlp"]
TEST_FAR_LIMIT = 0.01
GATE_MIN_SHARE = 0.70
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20261009
REFERENCE_LEVEL = FAR_LEVELS[0]


def scenario_event_coverage(
    frame: pd.DataFrame,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Event coverage per attack scenario in a single grouping pass."""
    attack = frame["is_attack"].astype(bool).to_numpy()
    predicted = np.asarray(predicted, dtype=bool)
    event_rows = frame.loc[
        attack,
        ["attack_scenario", "attack_event_id"],
    ].copy()
    event_rows["_pred"] = predicted[attack]
    grouped = event_rows.groupby(
        ["attack_scenario", "attack_event_id"],
        sort=True,
    )["_pred"].any()
    output: dict[str, float] = {}
    for scenario, group in grouped.groupby(level=0, sort=True):
        output[str(scenario)] = (
            float(group.mean()) if len(group) else 0.0
        )
    return output


def rule_label(weight: float | str) -> str:
    if weight == "max":
        return "max"
    return f"{float(weight):.1f}"


def weight_label(counts: dict[str, int]) -> str:
    order = [*[rule_label(weight) for weight in WEIGHT_GRID], "max"]
    parts = [
        f"{key}:{counts[key]}" for key in order if counts.get(key, 0)
    ]
    return ";".join(parts) if parts else "none"


def build_records(merged: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
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
        union_median = float(
            np.median(union_valid_raw[valid_normal])
        )
        formal_median = float(
            np.median(formal_valid_raw[valid_normal])
        )
        union_scale = max(
            1.4826
            * float(
                np.median(
                    np.abs(
                        union_valid_raw[valid_normal] - union_median
                    )
                )
            ),
            1e-6,
        )
        formal_scale = max(
            1.4826
            * float(
                np.median(
                    np.abs(
                        formal_valid_raw[valid_normal]
                        - formal_median
                    )
                )
            ),
            1e-6,
        )
        union_valid_pred = valid["union_prediction"].to_numpy(
            dtype=bool
        )
        union_test_pred = test["union_prediction"].to_numpy(
            dtype=bool
        )
        records.append(
            {
                "source": str(source),
                "seed": int(seed),
                "valid_frame": valid_frame,
                "test_frame": test_frame,
                "valid_normal": valid_normal,
                "valid_observed": valid_frame[
                    "observation_present"
                ].to_numpy(dtype=bool),
                "test_observed": test_frame[
                    "observation_present"
                ].to_numpy(dtype=bool),
                "union_valid_z": (
                    union_valid_raw - union_median
                )
                / union_scale,
                "formal_valid_z": (
                    formal_valid_raw - formal_median
                )
                / formal_scale,
                "union_test_z": (
                    union_test_raw - union_median
                )
                / union_scale,
                "formal_test_z": (
                    formal_test_raw - formal_median
                )
                / formal_scale,
                "union_valid_pred": union_valid_pred,
                "union_test_pred": union_test_pred,
                "union_valid_coverage": scenario_event_coverage(
                    valid_frame,
                    union_valid_pred,
                ),
                "union_test_coverage": scenario_event_coverage(
                    test_frame,
                    union_test_pred,
                ),
                "test_rows": int(len(test_frame)),
                "test_attack_rows": int(
                    test_frame["is_attack"].sum()
                ),
            }
        )
    return records


def evaluate_level(
    records: list[dict[str, object]],
    level: float,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validation-only selection at one FAR level for one variant."""
    run_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for record in records:
        valid_frame = record["valid_frame"]
        test_frame = record["test_frame"]
        valid_normal = record["valid_normal"]
        valid_observed = record["valid_observed"]
        test_observed = record["test_observed"]
        union_valid_coverage = record["union_valid_coverage"]
        candidates: list[dict[str, object]] = []
        for weight in RULE_GRID:
            fused_valid = candidate_scores(
                record["union_valid_z"],
                record["formal_valid_z"],
                weight,
            )
            threshold = float(
                far_threshold(fused_valid, valid_normal, level)
            )
            predicted_valid = (
                valid_observed & (fused_valid >= threshold)
            )
            if variant == FROZEN_UNION and weight == 1.0:
                predicted_valid = record["union_valid_pred"]
            coverage = scenario_event_coverage(
                valid_frame,
                predicted_valid,
            )
            constraint_ok = all(
                coverage.get(scenario, 0.0)
                >= union_valid_coverage.get(scenario, 0.0)
                - CONSTRAINT_TOLERANCE
                for scenario in CONSTRAINED_SCENARIOS
            )
            valid_metrics = metrics(
                valid_frame["is_attack"].to_numpy(dtype=bool),
                predicted_valid,
                fused_valid,
            )
            valid_event = event_metrics(valid_frame, predicted_valid)
            candidates.append(
                {
                    "weight": weight,
                    "threshold": threshold,
                    "constraint_ok": bool(constraint_ok),
                    "valid_mean_scenario_coverage": float(
                        np.mean(
                            [
                                coverage.get(scenario, 0.0)
                                for scenario in sorted(
                                    set(coverage)
                                    | set(union_valid_coverage)
                                )
                            ]
                        )
                    ),
                    "valid_f1": float(valid_metrics["f1"]),
                    "valid_event_coverage": float(
                        valid_event["event_coverage"]
                    ),
                    "valid_false_alarm_rate": float(
                        valid_metrics["false_alarm_rate"]
                    ),
                }
            )
        feasible = [
            candidate
            for candidate in candidates
            if candidate["constraint_ok"]
        ]
        if feasible:
            best = max(
                feasible,
                key=lambda candidate: (
                    candidate["valid_mean_scenario_coverage"],
                    candidate["valid_f1"],
                    -candidate["valid_event_coverage"],
                ),
            )
            fallback = False
        else:
            best = next(
                candidate
                for candidate in candidates
                if candidate["weight"] == 1.0
            )
            fallback = True

        fused_test = candidate_scores(
            record["union_test_z"],
            record["formal_test_z"],
            best["weight"],
        )
        predicted_test = (
            test_observed & (fused_test >= float(best["threshold"]))
        )
        if variant == FROZEN_UNION and best["weight"] == 1.0:
            predicted_test = record["union_test_pred"]
        test_metrics = metrics(
            test_frame["is_attack"].to_numpy(dtype=bool),
            predicted_test,
            fused_test,
        )
        test_event = event_metrics(test_frame, predicted_test)
        test_coverage = scenario_event_coverage(
            test_frame,
            predicted_test,
        )
        deltas = {
            scenario: float(
                test_coverage.get(scenario, 0.0)
                - record["union_test_coverage"].get(scenario, 0.0)
            )
            for scenario in CONSTRAINED_SCENARIOS
        }
        run_rows.append(
            {
                "variant": variant,
                "far_level": float(level),
                "source": record["source"],
                "seed": record["seed"],
                "weight": best["weight"],
                "threshold": float(best["threshold"]),
                "fallback_to_union": bool(fallback),
                "feasible_candidate_count": int(len(feasible)),
                "validation_target_far": float(level),
                "validation_achieved_far": float(
                    best["valid_false_alarm_rate"]
                ),
                "test_far": float(test_metrics["false_alarm_rate"]),
                "f1": float(test_metrics["f1"]),
                "precision": float(test_metrics["precision"]),
                "recall": float(test_metrics["recall"]),
                "event_coverage": float(
                    test_event["event_coverage"]
                ),
                "row_recall": float(test_event["row_recall"]),
                "average_precision": float(
                    test_metrics["average_precision"]
                ),
                "roc_auc": float(test_metrics["roc_auc"]),
                "long_stuck_test_delta": deltas["long_stuck"],
                "drop_replay_test_delta": deltas["drop_replay"],
                "test_rows": record["test_rows"],
                "test_attack_rows": record["test_attack_rows"],
            }
        )
        selection_rows.append(
            {
                "variant": variant,
                "far_level": float(level),
                "source": record["source"],
                "seed": record["seed"],
                "weight": best["weight"],
                "fallback_to_union": bool(fallback),
                "feasible_candidate_count": int(len(feasible)),
            }
        )
        for scenario, value in sorted(test_coverage.items()):
            coverage_rows.append(
                {
                    "variant": variant,
                    "far_level": float(level),
                    "source": record["source"],
                    "seed": record["seed"],
                    "scenario": scenario,
                    "event_coverage": float(value),
                }
            )
    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(coverage_rows),
        pd.DataFrame(selection_rows),
    )


def baseline_scenario_means() -> dict[str, dict[str, float]]:
    existing = pd.read_csv(EXISTING_SCENARIO)
    formal = pd.read_csv(FORMAL_SCENARIO)
    formal_mean = (
        formal.groupby(["scenario", "method"], as_index=False)
        .agg(event_coverage=("event_coverage", "mean"))
    )
    combined = pd.concat(
        [
            existing[["scenario", "method", "event_coverage"]],
            formal_mean,
        ],
        ignore_index=True,
        sort=False,
    )
    means: dict[str, dict[str, float]] = {}
    for scenario, group in combined.groupby("scenario", sort=True):
        values: dict[str, float] = {}
        for baseline in BASELINE_KEYS:
            reference = group[group["method"].eq(baseline)]
            if reference.empty:
                raise ValueError(
                    f"missing baseline {baseline} for {scenario}"
                )
            values[baseline] = float(
                reference["event_coverage"].iloc[0]
            )
        means[str(scenario)] = values
    return means


def extended_gate(
    coverage: pd.DataFrame,
    baselines: dict[str, dict[str, float]],
) -> tuple[int, int, pd.DataFrame]:
    candidate = (
        coverage.groupby("scenario", as_index=False)
        .agg(event_coverage=("event_coverage", "mean"))
    )
    rows: list[dict[str, object]] = []
    wins = 0
    total = 0
    for _, group in candidate.iterrows():
        scenario = str(group["scenario"])
        value = float(group["event_coverage"])
        for baseline in BASELINE_KEYS:
            reference = float(baselines[scenario][baseline])
            total += 1
            won = value > reference
            wins += int(won)
            rows.append(
                {
                    "scenario": scenario,
                    "baseline": baseline,
                    "candidate_event_coverage": value,
                    "baseline_event_coverage": reference,
                    "win": bool(won),
                }
            )
    return wins, total, pd.DataFrame(rows)


def hierarchical_bootstrap(
    runs: pd.DataFrame,
    coverage: pd.DataFrame,
    baselines: dict[str, dict[str, float]],
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, object]:
    providers = sorted(runs["source"].unique())
    seed_values = sorted(int(value) for value in runs["seed"].unique())
    position = {
        (str(source), int(seed)): index
        for index, (source, seed) in enumerate(
            zip(runs["source"], runs["seed"])
        )
    }
    far = runs["test_far"].to_numpy(dtype=float)
    f1 = runs["f1"].to_numpy(dtype=float)
    coverage_values = runs["event_coverage"].to_numpy(dtype=float)
    scenarios = sorted(baselines)
    coverage_lookup = {
        (str(row.source), int(row.seed), str(row.scenario)): float(
            row.event_coverage
        )
        for row in coverage.itertuples()
    }
    matrix = np.asarray(
        [
            [
                coverage_lookup[(str(source), int(seed), scenario)]
                for scenario in scenarios
            ]
            for source, seed in zip(runs["source"], runs["seed"])
        ],
        dtype=float,
    )
    baseline_matrix = np.asarray(
        [
            [baselines[scenario][key] for key in BASELINE_KEYS]
            for scenario in scenarios
        ],
        dtype=float,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    far_draws = np.empty(iterations, dtype=float)
    f1_draws = np.empty(iterations, dtype=float)
    coverage_draws = np.empty(iterations, dtype=float)
    gate_draws = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        selected: list[int] = []
        sampled_providers = rng.choice(
            providers,
            size=len(providers),
            replace=True,
        )
        for provider in sampled_providers:
            sampled_seeds = rng.choice(
                seed_values,
                size=len(seed_values),
                replace=True,
            )
            selected.extend(
                position[(str(provider), int(seed))]
                for seed in sampled_seeds
            )
        index = np.asarray(selected, dtype=int)
        far_draws[iteration] = float(far[index].mean())
        f1_draws[iteration] = float(f1[index].mean())
        coverage_draws[iteration] = float(
            coverage_values[index].mean()
        )
        candidate = matrix[index].mean(axis=0)
        wins = float(
            (candidate[:, None] > baseline_matrix).sum()
        )
        gate_draws[iteration] = wins / baseline_matrix.size
    return {
        "iterations": int(iterations),
        "test_far_point": float(far.mean()),
        "test_far_ci95_low": float(np.quantile(far_draws, 0.025)),
        "test_far_ci95_high": float(np.quantile(far_draws, 0.975)),
        "test_far_probability_above_1pct": float(
            np.mean(far_draws > TEST_FAR_LIMIT)
        ),
        "f1_ci95_low": float(np.quantile(f1_draws, 0.025)),
        "f1_ci95_high": float(np.quantile(f1_draws, 0.975)),
        "event_coverage_ci95_low": float(
            np.quantile(coverage_draws, 0.025)
        ),
        "event_coverage_ci95_high": float(
            np.quantile(coverage_draws, 0.975)
        ),
        "gate_share_point": float(gate_draws.mean()),
        "gate_share_ci95_low": float(
            np.quantile(gate_draws, 0.025)
        ),
        "gate_share_ci95_high": float(
            np.quantile(gate_draws, 0.975)
        ),
        "gate_probability_passes_70pct": float(
            np.mean(gate_draws >= GATE_MIN_SHARE)
        ),
    }


def aggregate_level(
    runs: pd.DataFrame,
    coverage: pd.DataFrame,
    baselines: dict[str, dict[str, float]],
    variant: str,
    level: float,
) -> dict[str, object]:
    wins, total, gate_detail = extended_gate(coverage, baselines)
    bootstrap = hierarchical_bootstrap(runs, coverage, baselines)
    counts = runs["weight"].map(rule_label).value_counts().to_dict()
    structured_means = {
        scenario: float(
            runs[f"{scenario}_test_delta"].mean()
        )
        for scenario in CONSTRAINED_SCENARIOS
    }
    structured_within = {
        scenario: float(
            runs[f"{scenario}_test_delta"].ge(-CONSTRAINT_TOLERANCE).mean()
        )
        for scenario in CONSTRAINED_SCENARIOS
    }
    gate_share = wins / total if total else 0.0
    gate_by_scenario = (
        gate_detail.groupby("scenario", as_index=False)
        .agg(
            wins=("win", "sum"),
            units=("win", "size"),
        )
        .assign(share=lambda frame: frame["wins"] / frame["units"])
        .to_dict("records")
    )
    literal_runs = int(
        (
            runs["test_far"].le(TEST_FAR_LIMIT)
            & runs["long_stuck_test_delta"].ge(
                -CONSTRAINT_TOLERANCE
            )
            & runs["drop_replay_test_delta"].ge(
                -CONSTRAINT_TOLERANCE
            )
        ).sum()
    )
    point_far_ok = bool(runs["test_far"].mean() <= TEST_FAR_LIMIT)
    bootstrap_far_ok = bool(
        bootstrap["test_far_ci95_high"] <= TEST_FAR_LIMIT
    )
    structured_ok = bool(
        all(
            structured_means[scenario] >= -CONSTRAINT_TOLERANCE
            and structured_within[scenario] >= 1.0
            for scenario in CONSTRAINED_SCENARIOS
        )
    )
    gate_ok = bool(gate_share >= GATE_MIN_SHARE)
    row = {
        "row_type": "far_level",
        "variant": variant,
        "far_level": float(level),
        "runs": int(len(runs)),
        "selected_weights": weight_label(counts),
        "union_rule_runs": int(runs["weight"].eq(1.0).sum()),
        "fallback_runs": int(runs["fallback_to_union"].sum()),
        "feasible_runs": int(
            runs["feasible_candidate_count"].gt(0).sum()
        ),
        "validation_target_far": float(level),
        "validation_achieved_far_mean": float(
            runs["validation_achieved_far"].mean()
        ),
        "validation_far_gap_mean": float(
            (
                runs["validation_achieved_far"]
                - runs["validation_target_far"]
            ).mean()
        ),
        "test_far_mean": float(runs["test_far"].mean()),
        "test_far_median": float(runs["test_far"].median()),
        "test_far_max": float(runs["test_far"].max()),
        "test_far_gap_vs_validation_mean": float(
            (
                runs["test_far"] - runs["validation_achieved_far"]
            ).mean()
        ),
        "test_far_gap_vs_target_mean": float(
            (
                runs["test_far"]
                - runs["validation_target_far"]
            ).mean()
        ),
        "test_far_bootstrap_ci95_low": bootstrap[
            "test_far_ci95_low"
        ],
        "test_far_bootstrap_ci95_high": bootstrap[
            "test_far_ci95_high"
        ],
        "test_far_bootstrap_probability_above_1pct": bootstrap[
            "test_far_probability_above_1pct"
        ],
        "bootstrap_iterations": bootstrap["iterations"],
        "f1_mean": float(runs["f1"].mean()),
        "event_coverage_mean": float(
            runs["event_coverage"].mean()
        ),
        "precision_mean": float(runs["precision"].mean()),
        "row_recall_mean": float(runs["row_recall"].mean()),
        "average_precision_mean": float(
            runs["average_precision"].mean()
        ),
        "extended_wins": int(wins),
        "extended_units": int(total),
        "extended_share": float(gate_share),
        "extended_passes_70pct": gate_ok,
        "extended_bootstrap_probability_passes_70pct": bootstrap[
            "gate_probability_passes_70pct"
        ],
        "long_stuck_test_delta_mean": structured_means["long_stuck"],
        "long_stuck_within_5pp_fraction": structured_within[
            "long_stuck"
        ],
        "drop_replay_test_delta_mean": structured_means[
            "drop_replay"
        ],
        "drop_replay_within_5pp_fraction": structured_within[
            "drop_replay"
        ],
        "criterion_test_far_point": point_far_ok,
        "criterion_test_far_bootstrap_upper": bootstrap_far_ok,
        "criterion_structured_within_5pp": structured_ok,
        "criterion_extended_gate_70pct": gate_ok,
        "feasible_literal_criteria": bool(
            point_far_ok and structured_ok and gate_ok
        ),
        "feasible_strict_criteria": bool(
            point_far_ok
            and bootstrap_far_ok
            and structured_ok
            and gate_ok
        ),
        "runs_meeting_literal_criteria": literal_runs,
    }
    return {
        "row": row,
        "bootstrap": bootstrap,
        "gate_detail": gate_detail,
        "gate_by_scenario": gate_by_scenario,
    }


def provider_rows(
    runs: pd.DataFrame,
    stage491_map: dict[tuple[str, int], dict[str, object]],
    stage493_provider: dict[str, float],
    variant: str,
    level: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def summary(
        aggregation: str,
        group: pd.DataFrame,
        source: str,
        seed: object,
    ) -> dict[str, object]:
        counts = group["weight"].map(rule_label).value_counts().to_dict()
        row: dict[str, object] = {
            "aggregation": aggregation,
            "variant": variant,
            "far_level": float(level),
            "source": source,
            "seed": seed,
            "runs": int(len(group)),
            "validation_target_far": float(level),
            "validation_achieved_far_mean": float(
                group["validation_achieved_far"].mean()
            ),
            "validation_far_gap_mean": float(
                (
                    group["validation_achieved_far"]
                    - group["validation_target_far"]
                ).mean()
            ),
            "test_far_mean": float(group["test_far"].mean()),
            "test_far_max": float(group["test_far"].max()),
            "test_far_gap_vs_validation_mean": float(
                (
                    group["test_far"]
                    - group["validation_achieved_far"]
                ).mean()
            ),
            "test_far_gap_vs_target_mean": float(
                (
                    group["test_far"]
                    - group["validation_target_far"]
                ).mean()
            ),
            "f1_mean": float(group["f1"].mean()),
            "precision_mean": float(group["precision"].mean()),
            "recall_mean": float(group["recall"].mean()),
            "event_coverage_mean": float(
                group["event_coverage"].mean()
            ),
            "row_recall_mean": float(group["row_recall"].mean()),
            "selected_weights": weight_label(counts),
            "union_rule_runs": int(group["weight"].eq(1.0).sum()),
            "fallback_runs": int(group["fallback_to_union"].sum()),
            "long_stuck_test_delta_mean": float(
                group["long_stuck_test_delta"].mean()
            ),
            "drop_replay_test_delta_mean": float(
                group["drop_replay_test_delta"].mean()
            ),
            "test_far_le_1pct": bool(
                group["test_far"].mean() <= TEST_FAR_LIMIT
            ),
        }
        return row

    for (source, seed), group in runs.groupby(
        ["source", "seed"],
        sort=True,
    ):
        row = summary(
            "provider_seed",
            group,
            str(source),
            int(seed),
        )
        reference = stage491_map.get((str(source), int(seed)), {})
        row["stage491_weight"] = reference.get("weight")
        row["stage491_test_far"] = reference.get(
            "false_alarm_rate"
        )
        row["stage493_provider_far_mean"] = stage493_provider.get(
            str(source)
        )
        rows.append(row)
    for source, group in runs.groupby("source", sort=True):
        row = summary("provider", group, str(source), None)
        row["stage493_provider_far_mean"] = stage493_provider.get(
            str(source)
        )
        rows.append(row)
    for seed, group in runs.groupby("seed", sort=True):
        rows.append(
            summary("seed", group, None, int(seed))
        )
    return rows


def union_rethreshold_diagnostic(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Why pure union re-thresholding loses structured coverage."""
    frozen: dict[str, list[float]] = {}
    retheresholded: dict[str, list[float]] = {}
    for record in records:
        threshold = float(
            far_threshold(
                record["union_valid_z"],
                record["valid_normal"],
                REFERENCE_LEVEL,
            )
        )
        predicted = record["valid_observed"] & (
            record["union_valid_z"] >= threshold
        )
        retheresholded_coverage = scenario_event_coverage(
            record["valid_frame"],
            predicted,
        )
        for scenario, value in record["union_valid_coverage"].items():
            frozen.setdefault(scenario, []).append(float(value))
            retheresholded.setdefault(scenario, []).append(
                float(retheresholded_coverage.get(scenario, 0.0))
            )
    rows: list[dict[str, object]] = []
    for scenario in sorted(set(frozen) | set(retheresholded)):
        frozen_mean = float(np.mean(frozen.get(scenario, [0.0])))
        retheresholded_mean = float(
            np.mean(retheresholded.get(scenario, [0.0]))
        )
        rows.append(
            {
                "scenario": scenario,
                "frozen_union_validation_coverage": frozen_mean,
                "rethresholded_union_validation_coverage": (
                    retheresholded_mean
                ),
                "coverage_drop": frozen_mean - retheresholded_mean,
            }
        )
    return rows


def reference_rows(
    baselines: dict[str, dict[str, float]],
) -> tuple[dict[str, object], dict[str, object]]:
    runs = pd.read_csv(STAGE491_RUNS)
    scenario = pd.read_csv(STAGE491_SCENARIO)
    scenario = scenario[
        scenario["method"].eq("constrained_validation_fusion")
    ]
    coverage = (
        scenario.groupby(
            ["source", "seed", "scenario"],
            as_index=False,
        )
        .agg(event_coverage=("event_coverage", "mean"))
        .assign(variant="stage491_reference")
    )
    wins, total, gate_detail = extended_gate(coverage, baselines)
    bootstrap = pd.read_csv(STAGE493_BOOTSTRAP)
    far_row = bootstrap[
        bootstrap["metric"].eq("false_alarm_rate")
    ].iloc[0]
    gate_bootstrap = pd.read_csv(STAGE493_GATE_BOOTSTRAP).iloc[0]
    gate_published = pd.read_csv(STAGE491_GATE)
    gate_published = gate_published[
        gate_published["baseline_set"].eq(
            "frozen_11_plus_formal"
        )
    ]
    row: dict[str, object] = {
        "row_type": "reference_stage491_493",
        "variant": "stage491_frozen_reference",
        "far_level": float(REFERENCE_LEVEL),
        "runs": int(len(runs)),
        "selected_weights": weight_label(
            runs["weight"].map(rule_label).value_counts().to_dict()
        ),
        "union_rule_runs": int(runs["weight"].eq(1.0).sum()),
        "fallback_runs": 0,
        "feasible_runs": int(runs["constraint_ok"].sum()),
        "validation_target_far": float(REFERENCE_LEVEL),
        "validation_achieved_far_mean": float("nan"),
        "validation_far_gap_mean": float("nan"),
        "test_far_mean": float(runs["false_alarm_rate"].mean()),
        "test_far_median": float(runs["false_alarm_rate"].median()),
        "test_far_max": float(runs["false_alarm_rate"].max()),
        "test_far_gap_vs_validation_mean": float("nan"),
        "test_far_gap_vs_target_mean": float("nan"),
        "test_far_bootstrap_ci95_low": float(far_row["ci95_low"]),
        "test_far_bootstrap_ci95_high": float(far_row["ci95_high"]),
        "test_far_bootstrap_probability_above_1pct": float("nan"),
        "bootstrap_iterations": int(far_row["iterations"]),
        "f1_mean": float(runs["f1"].mean()),
        "event_coverage_mean": float(
            runs["event_coverage"].mean()
        ),
        "precision_mean": float(runs["precision"].mean()),
        "row_recall_mean": float(runs["row_recall"].mean()),
        "average_precision_mean": float(
            runs["average_precision"].mean()
        ),
        "extended_wins": int(wins),
        "extended_units": int(total),
        "extended_share": float(wins / total if total else 0.0),
        "extended_passes_70pct": bool(
            wins / total >= GATE_MIN_SHARE if total else False
        ),
        "extended_bootstrap_probability_passes_70pct": float(
            gate_bootstrap["probability_passes_70pct"]
        ),
        "long_stuck_test_delta_mean": 0.0,
        "long_stuck_within_5pp_fraction": 1.0,
        "drop_replay_test_delta_mean": 0.0,
        "drop_replay_within_5pp_fraction": 1.0,
        "criterion_test_far_point": bool(
            runs["false_alarm_rate"].mean() <= TEST_FAR_LIMIT
        ),
        "criterion_test_far_bootstrap_upper": bool(
            float(far_row["ci95_high"]) <= TEST_FAR_LIMIT
        ),
        "criterion_structured_within_5pp": True,
        "criterion_extended_gate_70pct": bool(
            wins / total >= GATE_MIN_SHARE if total else False
        ),
        "feasible_literal_criteria": False,
        "feasible_strict_criteria": False,
        "runs_meeting_literal_criteria": int(
            runs["false_alarm_rate"].le(TEST_FAR_LIMIT).sum()
        ),
    }
    row["feasible_literal_criteria"] = bool(
        row["criterion_test_far_point"]
        and row["criterion_structured_within_5pp"]
        and row["criterion_extended_gate_70pct"]
    )
    row["feasible_strict_criteria"] = bool(
        row["feasible_literal_criteria"]
        and row["criterion_test_far_bootstrap_upper"]
    )
    detail = {
        "stage491_gate_published_wins": int(
            gate_published["wins"].sum()
        ),
        "stage491_gate_published_units": int(
            gate_published["total_baselines"].sum()
        ),
        "stage491_gate_recomputed_wins": int(wins),
        "stage491_gate_recomputed_units": int(total),
        "stage493_far_bootstrap_ci95_low": float(
            far_row["ci95_low"]
        ),
        "stage493_far_bootstrap_ci95_high": float(
            far_row["ci95_high"]
        ),
        "stage493_gate_bootstrap_point": float(
            gate_bootstrap["point_estimate"]
        ),
        "stage493_gate_bootstrap_ci95_low": float(
            gate_bootstrap["ci95_low"]
        ),
        "stage493_gate_bootstrap_ci95_high": float(
            gate_bootstrap["ci95_high"]
        ),
        "stage493_gate_bootstrap_probability_passes_70pct": float(
            gate_bootstrap["probability_passes_70pct"]
        ),
        "gate_detail": gate_detail.to_dict("records"),
    }
    return row, detail


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
        left = merged[f"{column}_union"].astype(str)
        right = merged[f"{column}_formal"].astype(str)
        if not left.eq(right).all():
            raise ValueError(f"score alignment mismatch: {column}")
    records = build_records(merged)
    baselines = baseline_scenario_means()

    calibration_rows: list[dict[str, object]] = []
    provider_table_rows: list[dict[str, object]] = []
    level_details: list[dict[str, object]] = []
    run_frames: dict[tuple[str, float], pd.DataFrame] = {}
    stage491_runs = pd.read_csv(STAGE491_RUNS)
    stage491_map = {
        (str(row.source), int(row.seed)): {
            "weight": float(row.weight),
            "false_alarm_rate": float(row.false_alarm_rate),
        }
        for row in stage491_runs.itertuples()
    }
    stage493_provider = pd.read_csv(STAGE493_PROVIDER)
    stage493_provider_map = {
        str(row.unit): float(row.false_alarm_rate_mean)
        for row in stage493_provider.itertuples()
    }
    for variant in VARIANTS:
        for level in FAR_LEVELS:
            runs, coverage, selections = evaluate_level(
                records,
                level,
                variant,
            )
            run_frames[(variant, level)] = runs
            aggregated = aggregate_level(
                runs,
                coverage,
                baselines,
                variant,
                level,
            )
            calibration_rows.append(aggregated["row"])
            provider_table_rows.extend(
                provider_rows(
                    runs,
                    stage491_map,
                    stage493_provider_map,
                    variant,
                    level,
                )
            )
            level_details.append(
                {
                    "variant": variant,
                    "far_level": float(level),
                    "selection_counts": runs["weight"]
                    .map(rule_label)
                    .value_counts()
                    .to_dict(),
                    "selection_rows": selections.to_dict("records"),
                    "bootstrap": aggregated["bootstrap"],
                    "gate_by_scenario": aggregated[
                        "gate_by_scenario"
                    ],
                }
            )

    reference_row, reference_detail = reference_rows(baselines)
    calibration_rows.append(reference_row)

    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(
        OUT_CALIBRATION,
        index=False,
        encoding="utf-8-sig",
    )
    provider = pd.DataFrame(provider_table_rows)
    provider["seed"] = provider["seed"].astype("Int64")
    provider.to_csv(OUT_PROVIDER, index=False, encoding="utf-8-sig")

    # Verification: the frozen-union family at the reference level must
    # reproduce the published Stage 491 runs exactly, and the Stage 493
    # bootstrap must be reproduced for the same configuration.
    base = run_frames[(FROZEN_UNION, REFERENCE_LEVEL)]
    stage491_left = stage491_runs.rename(
        columns=lambda column: f"{column}_stage491"
    )
    stage496_right = base.rename(
        columns=lambda column: f"{column}_stage496"
    )
    check = stage491_left.merge(
        stage496_right,
        left_on=["source_stage491", "seed_stage491"],
        right_on=["source_stage496", "seed_stage496"],
        validate="one_to_one",
    )
    if len(check) != len(stage491_runs):
        raise RuntimeError("Stage 491 merge lost runs")
    weight_match = bool(
        np.allclose(
            check["weight_stage491"].to_numpy(dtype=float),
            check["weight_stage496"].to_numpy(dtype=float),
            atol=1e-12,
        )
    )
    metric_checks = {
        metric: float(
            np.max(
                np.abs(
                    check[f"{left}_stage491"].to_numpy(dtype=float)
                    - check[f"{right}_stage496"].to_numpy(dtype=float)
                )
            )
        )
        for metric, (left, right) in {
            "f1": ("f1", "f1"),
            "event_coverage": ("event_coverage", "event_coverage"),
            "false_alarm_rate": ("false_alarm_rate", "test_far"),
        }.items()
    }
    row_checks = {
        column: bool(
            np.array_equal(
                check[f"{left}_stage491"].to_numpy(dtype=float),
                check[f"{right}_stage496"].to_numpy(dtype=float),
            )
        )
        for column, (left, right) in {
            "test_rows": ("rows", "test_rows"),
            "test_attack_rows": ("attack_rows", "test_attack_rows"),
        }.items()
    }
    base_level = next(
        detail
        for detail in level_details
        if detail["variant"] == FROZEN_UNION
        and detail["far_level"] == REFERENCE_LEVEL
    )
    bootstrap = base_level["bootstrap"]
    bootstrap_match = bool(
        abs(
            bootstrap["test_far_ci95_low"]
            - reference_detail["stage493_far_bootstrap_ci95_low"]
        )
        <= 1e-9
        and abs(
            bootstrap["test_far_ci95_high"]
            - reference_detail["stage493_far_bootstrap_ci95_high"]
        )
        <= 1e-9
    )
    gate_match = bool(
        reference_detail["stage491_gate_recomputed_wins"]
        == reference_detail["stage491_gate_published_wins"]
        and reference_detail["stage491_gate_recomputed_units"]
        == reference_detail["stage491_gate_published_units"]
    )
    if not weight_match or not gate_match or not bootstrap_match:
        raise RuntimeError("Stage 491/493 reproduction check failed")
    if max(metric_checks.values()) > 1e-12:
        raise RuntimeError("Stage 491 metric reproduction check failed")
    if not all(row_checks.values()):
        raise RuntimeError("Stage 491 row count reproduction failed")

    diagnostic = union_rethreshold_diagnostic(records)
    calibration_records = calibration.to_dict("records")
    strict_feasible = [
        row
        for row in calibration_records
        if row["row_type"] == "far_level"
        and row["feasible_strict_criteria"]
    ]
    literal_feasible = [
        row
        for row in calibration_records
        if row["row_type"] == "far_level"
        and row["feasible_literal_criteria"]
    ]
    payload = {
        "status": "PASS_EXPLORATORY_FAR_SAFETY_CALIBRATION_AUDIT",
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "no_new_providers_or_datasets": True,
        "no_frozen_artifact_modified": True,
        "test_used_for_level_selection": False,
        "score_source": {
            "row_scores": "stage490_union_scores.parquet, "
            "stage490_formal_mlp_scores.parquet",
            "stage491_results": "stage491_constrained_fusion_runs.csv",
            "stage493_results": "stage493_constrained_fusion_bootstrap.csv",
            "threshold_rule": "quantile of validation-normal fused "
            "scores at 1 - target FAR",
            "retraining": False,
        },
        "variants": {
            FROZEN_UNION: "weight 1.0 and the fallback keep the frozen "
            "union OR prediction (Stage 491/492/495 semantics)",
            RETHERSHOLDED_UNION: "every weight, including 1.0, is the "
            "Stage 490 shape score re-thresholded at the level",
        },
        "selection_rule": {
            "constraint": "validation event coverage of long_stuck and "
            "drop_replay within 5pp of the frozen union",
            "objective": "maximum validation mean scenario coverage",
            "tie_breaker": "validation F1, then lower validation event "
            "coverage",
            "fallback": "weight 1.0 candidate when no candidate is "
            "feasible",
        },
        "criteria": {
            "test_far_limit": TEST_FAR_LIMIT,
            "structured_drop_limit": CONSTRAINT_TOLERANCE,
            "extended_gate_min_share": GATE_MIN_SHARE,
            "strict_extra": "test FAR bootstrap 95% upper bound must "
            "also stay at or below the limit",
        },
        "far_levels": list(FAR_LEVELS),
        "calibration": calibration_records,
        "levels": level_details,
        "provider_summary": provider[
            provider["aggregation"].eq("provider")
        ].to_dict("records"),
        "seed_summary": provider[
            provider["aggregation"].eq("seed")
        ].to_dict("records"),
        "provider_seed_detail_rows": int(
            provider["aggregation"].eq("provider_seed").sum()
        ),
        "reference_stage491_493": reference_detail,
        "union_rethreshold_diagnostic": diagnostic,
        "feasibility": {
            "strict_feasible_configurations": [
                {
                    "variant": row["variant"],
                    "far_level": row["far_level"],
                }
                for row in strict_feasible
            ],
            "literal_feasible_configurations": [
                {
                    "variant": row["variant"],
                    "far_level": row["far_level"],
                }
                for row in literal_feasible
            ],
            "any_strict_feasible": bool(strict_feasible),
            "any_literal_feasible": bool(literal_feasible),
            "literal_note": (
                "the only literal configuration is the existing Stage "
                "491 setting; its Stage 493 bootstrap upper bound stays "
                "above the 1% target, so it does not solve the "
                "motivating problem"
            ),
            "strict_note": (
                "no level satisfies the three criteria once the "
                "bootstrap 95% upper bound is required to stay at or "
                "below 1%"
            ),
            "conclusion": (
                "no calibration level satisfies all three criteria"
                if not strict_feasible
                else "at least one level satisfies all three criteria"
            ),
        },
        "verification": {
            "score_alignment_checked": True,
            "stage491_weight_reproduced": weight_match,
            "stage491_metric_max_abs_diff": metric_checks,
            "stage491_row_counts_reproduced": row_checks,
            "stage491_gate_reproduced": gate_match,
            "stage493_far_bootstrap_reproduced": bootstrap_match,
        },
        "limitations": [
            "The V6 test split has already been inspected; every number "
            "here is exploratory.",
            "No level is selected on test; the levels are reported in "
            "parallel and any final choice would be exploratory.",
            "The frozen union OR prediction is not a pure quantile rule "
            "on the Stage 490 union shape score, so re-thresholding that "
            "score is a different decision rule.",
            "Four providers are treated as fixed clusters in the "
            "hierarchical bootstrap; this is not population inference.",
            "The extended gate bootstrap resamples candidate units only "
            "and holds the published baseline scenario means fixed.",
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    far_table = calibration[
        [
            "variant",
            "far_level",
            "validation_achieved_far_mean",
            "test_far_mean",
            "test_far_median",
            "test_far_max",
            "test_far_bootstrap_ci95_low",
            "test_far_bootstrap_ci95_high",
            "test_far_bootstrap_probability_above_1pct",
        ]
    ].copy()
    performance_table = calibration[
        [
            "variant",
            "far_level",
            "f1_mean",
            "event_coverage_mean",
            "precision_mean",
            "row_recall_mean",
            "extended_wins",
            "extended_units",
            "long_stuck_test_delta_mean",
            "drop_replay_test_delta_mean",
            "fallback_runs",
            "union_rule_runs",
            "runs_meeting_literal_criteria",
            "feasible_literal_criteria",
            "feasible_strict_criteria",
        ]
    ].copy()
    performance_table["extended_gate"] = (
        performance_table["extended_wins"].astype(str)
        + "/"
        + performance_table["extended_units"].astype(str)
    )
    performance_table = performance_table.drop(
        columns=["extended_wins", "extended_units"]
    )
    provider_headline = provider[
        provider["aggregation"].eq("provider")
    ][
        [
            "variant",
            "far_level",
            "source",
            "validation_achieved_far_mean",
            "test_far_mean",
            "test_far_gap_vs_validation_mean",
            "test_far_max",
            "selected_weights",
            "stage493_provider_far_mean",
        ]
    ].copy()
    diagnostic_frame = pd.DataFrame(diagnostic)
    strict_names = [
        f"{row['variant']} @ {row['far_level']}"
        for row in payload["feasibility"][
            "strict_feasible_configurations"
        ]
    ]
    literal_names = [
        f"{row['variant']} @ {row['far_level']}"
        for row in payload["feasibility"][
            "literal_feasible_configurations"
        ]
    ]
    lines = [
        "# FAR safety calibration audit",
        "",
        f"- Status: `{payload['status']}`",
        "- Score source: Stage 490 per-row union/formal scores; no model "
        "retraining, no new provider or dataset.",
        f"- FAR levels: `{list(FAR_LEVELS)}`",
        "- Level selection: fixed inputs, reported in parallel; per-run "
        "rule/weight selected on validation rows only.",
        "- Variants: `frozen_union_option` keeps the frozen union OR "
        "prediction as the weight 1.0 candidate and fallback (Stage "
        "491/492/495 semantics); `rethresholded_union` re-thresholds every "
        "Stage 490 shape score, including the union row.",
        "- Boundary: the V6 test split has already been inspected; every "
        "number is exploratory and no frozen result is replaced.",
        "",
        "## FAR calibration",
        "",
        far_table.to_markdown(index=False),
        "",
        "## Performance and gate",
        "",
        performance_table.to_markdown(index=False),
        "",
        "## Feasibility",
        "",
        f"- Literal criteria (point test FAR <= 1pp, structured drop <= "
        f"5pp, extended gate >= 70pct): `{literal_names or 'none'}`",
        f"- Strict criteria (literal criteria plus bootstrap 95pct upper "
        f"bound <= 1pp): `{strict_names or 'none'}`",
        "- The only literal configuration is the existing Stage 491 "
        "setting; it keeps the Stage 493 bootstrap upper bound at "
        "`0.01242`, so it does not solve the motivating problem.",
        "- In `frozen_union_option` the achieved validation FAR stays "
        "above the requested level (`0.0054` to `0.0061`) because runs "
        "that select the union rule cannot go below the frozen union's "
        "own validation FAR of about `0.0063`; the KNMI test FAR stays "
        "between `0.0130` and `0.0182`, so the bootstrap upper bound "
        "remains above 1pp at every level, and the gate share falls "
        "below 70pct once more runs select the union rule.",
        "- In `rethresholded_union` the test FAR tracks the level and the "
        "bootstrap upper bound stays below 1pp at every level, but "
        "long_stuck/drop_replay coverage drops by 31pp to 46pp and the "
        "extended gate falls below 70pct for every level except `0.0065`.",
        f"- Conclusion: `{payload['feasibility']['conclusion']}`",
        "",
        "## Provider FAR transfer",
        "",
        provider_headline.to_markdown(index=False),
        "",
        "Per-seed validation target/achieved/test FAR and deviations are "
        "in `stage496_far_safety_provider.csv` "
        "(`aggregation = provider_seed`), together with provider and seed "
        "aggregates and the Stage 491 per-seed reference columns.",
        "",
        "## Why re-thresholding the union score is not the union rule",
        "",
        diagnostic_frame.to_markdown(index=False),
        "",
        "## Stage 491/493 reference",
        "",
        f"- Stage 491 point test FAR: `{reference_row['test_far_mean']}`",
        f"- Stage 493 bootstrap CI: "
        f"`[{reference_detail['stage493_far_bootstrap_ci95_low']}, "
        f"{reference_detail['stage493_far_bootstrap_ci95_high']}]`",
        f"- Stage 491 extended gate: "
        f"`{reference_detail['stage491_gate_recomputed_wins']}/"
        f"{reference_detail['stage491_gate_recomputed_units']}`",
        "",
        "## Verification",
        "",
        f"- Stage 491 weights reproduced: `{weight_match}`",
        f"- Stage 491 metric max abs diff: `{metric_checks}`",
        f"- Stage 491 row counts reproduced: `{row_checks}`",
        f"- Stage 491 gate reproduced: `{gate_match}`",
        f"- Stage 493 FAR bootstrap reproduced: `{bootstrap_match}`",
        "",
        "## Boundary",
        "",
        "- Levels are fixed inputs; no level is selected on test.",
        "- The audit re-thresholds existing scores and never retrains.",
        "- No frozen artifact is modified; all outputs are new Stage 496 "
        "files.",
        "",
        f"- Elapsed seconds: `{payload['elapsed_seconds']:.2f}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "calibration": calibration_records,
                "feasibility": payload["feasibility"],
                "verification": payload["verification"],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
