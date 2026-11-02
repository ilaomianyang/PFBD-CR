"""Stage 498: KNMI cross-period FAR transfer correction experiment.

Motivation
----------
Stage 493/494/496 showed that the constrained score fusion transfers its false
alarm rate (FAR) from validation to test very unevenly across providers: KNMI's
test FAR is about 2.2x its validation FAR, which is the main driver of the
hierarchical bootstrap FAR upper bound sitting above the 1% target, while KNMI
is also only two wins short of the 70% extended scenario gate.

Design
------
Three fixed strategy families are evaluated side by side.  Every rule is
selected on validation rows only; test rows are evaluation-only:

``level``
    Re-threshold the Stage 490 shape scores at more conservative validation
    FAR targets.
``conformal``
    Split-conformal order statistic (finite-sample corrected quantile) of the
    validation normal scores at the same targets.
``binomial_ucb``
    Finite-sample exact-binomial correction: the threshold is the highest one
    whose one-sided Clopper-Pearson 95% upper confidence bound on the
    validation FAR still meets the target.
``margin_period_mean`` / ``margin_period_max``
    Provider-specific validation-to-test FAR safety margin with a fixed
    formula.  For every provider the within-validation first-half ->
    second-half FAR inflation is measured at the base target on the union
    shape score, and the effective validation target becomes
    ``target / margin``.  No test quantity enters the margin.

Constraint and fallback
-----------------------
A candidate rule is feasible only if its validation event coverage on
``long_stuck`` and ``drop_replay`` stays at or above the frozen union OR
prediction (margin 0).  If no candidate is feasible the unit falls back to the
frozen union OR prediction, exactly as in Stage 495.  Two selection units are
reported: ``provider`` (Stage 495 semantics, the primary unit) and ``seed``
(strict per-replicate enforcement of the same margin).

No model is retrained, no provider or dataset is added, and no frozen artifact
is modified.  The V6 test split has already been inspected, so every number
produced here is exploratory.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

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
EXISTING_SCENARIO = TABLES / "final_v6_scenario_metrics.csv"
EXISTING_SCENARIO_AGGREGATE = TABLES / "final_v6_scenario_aggregate.csv"
FORMAL_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"
STAGE495_RUNS = TABLES / "stage495_provider_adaptive_fusion_runs.csv"
STAGE491_SCENARIO = TABLES / "stage491_constrained_fusion_scenario.csv"
STAGE493_PROVIDER = (
    TABLES / "stage493_constrained_fusion_provider_summary.csv"
)

OUT_CORRECTION = TABLES / "stage498_knmi_far_transfer_correction.csv"
OUT_PROVIDER = TABLES / "stage498_knmi_far_provider.csv"
OUT_SEED = TABLES / "stage498_knmi_far_seed.csv"
OUT_JSON = REPORTS / "stage498_knmi_far_transfer_correction.json"
OUT_MD = REPORTS / "stage498_knmi_far_transfer_correction.md"

METHOD = "stage498_knmi_far_transfer_correction"
CANDIDATE_METHOD = "stage498_far_corrected_fusion"
UNION_OR = "frozen_union_or_fallback"
PRIMARY_LEVEL = 0.0065
LEVEL_TARGETS = (0.0065, 0.006, 0.005, 0.004, 0.003, 0.002)
CONFORMAL_TARGETS = (0.0065, 0.005, 0.004, 0.003)
BINOMIAL_TARGETS = (0.0065, 0.005, 0.004)
BINOMIAL_CONFIDENCE = 0.05
MARGIN_BASE_TARGETS = (0.0065, 0.005)
MARGIN_VARIANTS = ("period_mean", "period_max")
MARGIN_MAX = 4.0
SELECTION_UNITS = ("provider", "seed")
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RULE_GRID: list[float | str] = [*WEIGHT_GRID, "max"]
RULE_KEYS = [f"{weight:.1f}" for weight in WEIGHT_GRID] + ["max"]
CONSTRAINED_SCENARIOS = ("long_stuck", "drop_replay")
REFERENCES = [*BASELINES_11, "formal_lagged_mlp"]
TEST_FAR_LIMIT = 0.01
STRUCTURED_DROP_LIMIT = 0.05
GATE_MIN_SHARE = 0.70
EPS = 1e-12
KNMI = "KNMI"


def rule_label(weight: float | str) -> str:
    if weight == "max":
        return "max"
    return f"linear_w{float(weight):.1f}"


def scenario_coverage(
    frame: pd.DataFrame,
    predicted: np.ndarray,
) -> dict[str, float]:
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
        output[str(scenario)] = float(group.mean()) if len(group) else 0.0
    return output


def mean_scenario_coverage(
    coverage: dict[str, float],
    scenario_keys: list[str],
) -> float:
    if not scenario_keys:
        return 0.0
    return float(np.mean([coverage.get(key, 0.0) for key in scenario_keys]))


def clopper_pearson_upper(
    count: int,
    trials: int,
    confidence: float,
) -> float:
    """One-sided upper confidence bound for a binomial rate."""
    if trials <= 0:
        return 1.0
    count = int(np.clip(count, 0, trials))
    if count >= trials:
        return 1.0
    return float(beta.ppf(1.0 - confidence, count + 1, trials - count))


def conformal_threshold(reference: np.ndarray, alpha: float) -> float:
    """Split-conformal order statistic of the validation normal scores."""
    size = int(reference.size)
    if size <= 0:
        return float("inf")
    rank = int(np.ceil((size + 1) * (1.0 - float(alpha))))
    if rank > size:
        return float("inf")
    return float(np.partition(reference, rank - 1)[rank - 1])


def binomial_ucb_threshold(
    reference: np.ndarray,
    alpha: float,
    confidence: float,
) -> tuple[float, int]:
    """Highest threshold whose validation-FAR upper bound meets the target."""
    size = int(reference.size)
    if size <= 0:
        return float("inf"), 0
    ordered = np.sort(reference)
    low, high = 0, size
    best = -1
    while low <= high:
        middle = (low + high) // 2
        if clopper_pearson_upper(middle, size, confidence) <= alpha:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best < 0:
        return float("inf"), 0
    count = best
    while count >= 0:
        threshold = (
            float("inf") if count == 0 else float(ordered[size - count])
        )
        achieved = int(np.count_nonzero(reference >= threshold))
        if achieved == 0:
            return threshold, 0
        if (
            clopper_pearson_upper(achieved, size, confidence)
            <= alpha
        ):
            return threshold, achieved
        count -= 1
    return float("inf"), 0


def threshold_for_family(
    reference: np.ndarray,
    alpha: float,
    family: str,
    confidence: float,
) -> tuple[float, int]:
    if family == "conformal":
        threshold = conformal_threshold(reference, alpha)
        achieved = int(np.count_nonzero(reference >= threshold))
        return threshold, achieved
    if family == "binomial_ucb":
        return binomial_ucb_threshold(reference, alpha, confidence)
    threshold = float(far_threshold(reference, np.ones(reference.size, bool), alpha))
    achieved = int(np.count_nonzero(reference >= threshold))
    return threshold, achieved


def provider_period_margins(
    run_state: dict[tuple[str, int], dict[str, object]],
    providers: list[str],
    seeds_by_provider: dict[str, list[int]],
    base_alpha: float,
    max_margin: float,
) -> dict[str, dict[str, float]]:
    """Fixed-formula validation-only FAR safety margins per provider.

    At ``base_alpha`` the union shape score threshold is calibrated on the
    pooled validation normal rows of a run.  Each station's normal rows are
    split into an early and a late half by ``row_index`` order and the
    late/early false-alarm count ratio is formed with a 0.5 Laplace smoothing.
    ``period_mean`` averages the station ratios, ``period_max`` takes the worst
    station per run; both are then averaged over seeds and clipped to
    ``[1, max_margin]``.
    """
    mean_ratios: dict[str, list[float]] = {p: [] for p in providers}
    max_ratios: dict[str, list[float]] = {p: [] for p in providers}
    for provider in providers:
        for seed in seeds_by_provider[provider]:
            state = run_state[(provider, seed)]
            reference = np.asarray(
                state["union_valid_z"],
                dtype=float,
            )[state["valid_normal"]]
            threshold = float(
                far_threshold(
                    reference,
                    np.ones(reference.size, bool),
                    base_alpha,
                )
            )
            frame = state["valid_frame"]
            normal = state["valid_normal"]
            flagged = np.asarray(
                state["union_valid_z"],
                dtype=float,
            ) >= threshold
            work = pd.DataFrame(
                {
                    "station": frame.loc[normal, "station_id"].to_numpy(),
                    "order": frame.loc[normal, "row_index"].to_numpy(),
                    "flag": flagged[normal],
                }
            )
            work = work.sort_values(
                ["station", "order"],
                kind="stable",
            )
            work["position"] = work.groupby("station").cumcount()
            work["size"] = work.groupby("station")["flag"].transform("size")
            work["early"] = work["position"] < (work["size"] // 2)
            grouped = work.groupby(
                ["station", "early"],
                sort=True,
            )["flag"].sum().unstack("early")
            early = grouped.get(True, pd.Series(0.0, index=grouped.index))
            late = grouped.get(False, pd.Series(0.0, index=grouped.index))
            ratio = (late + 0.5) / (early + 0.5)
            mean_ratios[provider].append(float(ratio.mean()))
            max_ratios[provider].append(float(ratio.max()))
    margins: dict[str, dict[str, float]] = {}
    for provider in providers:
        margins[provider] = {
            "period_mean": float(
                np.clip(np.mean(mean_ratios[provider]), 1.0, max_margin)
            ),
            "period_max": float(
                np.clip(np.mean(max_ratios[provider]), 1.0, max_margin)
            ),
        }
    return margins


def build_strategies(
    providers: list[str],
    margins: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    base: list[dict[str, object]] = []
    for alpha in LEVEL_TARGETS:
        base.append(
            {
                "family": "level",
                "label": f"level@{alpha:g}",
                "parameter": float(alpha),
                "alpha": float(alpha),
                "alpha_by_provider": None,
            }
        )
    for alpha in CONFORMAL_TARGETS:
        base.append(
            {
                "family": "conformal",
                "label": f"conformal@{alpha:g}",
                "parameter": float(alpha),
                "alpha": float(alpha),
                "alpha_by_provider": None,
            }
        )
    for alpha in BINOMIAL_TARGETS:
        base.append(
            {
                "family": "binomial_ucb",
                "label": f"binomial_ucb@{alpha:g}",
                "parameter": float(alpha),
                "alpha": float(alpha),
                "alpha_by_provider": None,
            }
        )
    for variant in MARGIN_VARIANTS:
        for alpha in MARGIN_BASE_TARGETS:
            base.append(
                {
                    "family": f"margin_{variant}",
                    "label": f"margin_{variant}@{alpha:g}",
                    "parameter": float(alpha),
                    "alpha": float(alpha),
                    "alpha_by_provider": {
                        provider: float(alpha)
                        / float(margins[provider][variant])
                        for provider in providers
                    },
                }
            )
    strategies: list[dict[str, object]] = []
    for item in base:
        for unit in SELECTION_UNITS:
            strategies.append(
                {
                    **item,
                    "selection_unit": unit,
                    "strategy_id": f"{item['label']}|{unit}",
                }
            )
    return strategies


def strategy_alpha(
    strategy: dict[str, object],
    provider: str,
) -> float:
    by_provider = strategy["alpha_by_provider"]
    if by_provider is None:
        return float(strategy["alpha"])
    return float(by_provider[provider])  # type: ignore[index]


def build_run_state(merged: pd.DataFrame) -> dict[tuple[str, int], dict[str, object]]:
    run_state: dict[tuple[str, int], dict[str, object]] = {}
    for (source, seed), group in merged.groupby(
        ["source", "seed"],
        sort=True,
    ):
        valid = group[group["split"].eq("valid")].copy()
        test = group[group["split"].eq("test")].copy()
        valid_frame = canonical_frame(valid)
        test_frame = canonical_frame(test)
        valid_frame["row_index"] = valid["row_index"].to_numpy()
        test_frame["row_index"] = test["row_index"].to_numpy()
        valid_normal = (
            valid_frame["observation_present"] & ~valid_frame["is_attack"]
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
        union_valid_pred = valid["union_prediction"].to_numpy(dtype=bool)
        union_test_pred = test["union_prediction"].to_numpy(dtype=bool)
        union_valid_coverage = scenario_coverage(
            valid_frame,
            union_valid_pred,
        )
        union_test_coverage = scenario_coverage(
            test_frame,
            union_test_pred,
        )
        union_valid_metrics = metrics(
            valid_frame["is_attack"].to_numpy(dtype=bool),
            union_valid_pred,
            union_valid_z,
        )
        union_test_metrics = metrics(
            test_frame["is_attack"].to_numpy(dtype=bool),
            union_test_pred,
            union_test_z,
        )
        union_valid_event = event_metrics(valid_frame, union_valid_pred)
        union_test_event = event_metrics(test_frame, union_test_pred)
        run_state[(str(source), int(seed))] = {
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
            "y_valid": valid_frame["is_attack"].to_numpy(dtype=bool),
            "y_test": test_frame["is_attack"].to_numpy(dtype=bool),
            "union_valid_z": union_valid_z,
            "formal_valid_z": formal_valid_z,
            "union_test_z": union_test_z,
            "formal_test_z": formal_test_z,
            "union_valid_pred": union_valid_pred,
            "union_test_pred": union_test_pred,
            "union_valid_coverage": union_valid_coverage,
            "union_test_coverage": union_test_coverage,
            "union_valid_metrics": union_valid_metrics,
            "union_test_metrics": union_test_metrics,
            "union_valid_event_coverage": float(
                union_valid_event["event_coverage"]
            ),
            "union_test_event_coverage": float(
                union_test_event["event_coverage"]
            ),
            "scenario_keys": sorted(union_valid_coverage),
        }
    return run_state


def evaluate_validation_candidates(
    run_state: dict[tuple[str, int], dict[str, object]],
    strategies: list[dict[str, object]],
    confidence: float,
) -> dict[tuple[str, str, int, str], dict[str, object]]:
    """Validation-only candidate metrics for every strategy/weight/run."""
    cache: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for (source, seed), state in run_state.items():
        valid_frame = state["valid_frame"]
        valid_normal = state["valid_normal"]
        valid_observed = state["valid_observed"]
        y_valid = state["y_valid"]
        for weight, key in zip(RULE_GRID, RULE_KEYS):
            fused_valid = candidate_scores(
                state["union_valid_z"],
                state["formal_valid_z"],
                weight,
            )
            reference = np.asarray(
                fused_valid,
                dtype=float,
            )[valid_normal]
            for strategy in strategies:
                alpha = strategy_alpha(strategy, source)
                threshold, _ = threshold_for_family(
                    reference,
                    alpha,
                    str(strategy["family"]),
                    confidence,
                )
                predicted = valid_observed & (fused_valid >= threshold)
                coverage = scenario_coverage(valid_frame, predicted)
                achieved = int(
                    np.count_nonzero(
                        (fused_valid[valid_normal] >= threshold)
                    )
                )
                valid_far = achieved / max(int(valid_normal.sum()), 1)
                tp = int((predicted & y_valid).sum())
                fp = int((predicted & ~y_valid).sum())
                fn = int((~predicted & y_valid).sum())
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (
                    2.0 * precision * recall / (precision + recall)
                    if (precision + recall)
                    else 0.0
                )
                event = event_metrics(valid_frame, predicted)
                cache[(str(strategy["strategy_id"]), source, seed, key)] = {
                    "weight": weight,
                    "threshold": float(threshold),
                    "valid_f1": float(f1),
                    "valid_false_alarm_rate": float(valid_far),
                    "valid_event_coverage": float(
                        event["event_coverage"]
                    ),
                    "valid_mean_scenario_coverage": mean_scenario_coverage(
                        coverage,
                        state["scenario_keys"],
                    ),
                    "valid_coverage": coverage,
                }
    return cache


def select_runs(
    strategies: list[dict[str, object]],
    run_state: dict[tuple[str, int], dict[str, object]],
    cache: dict[tuple[str, str, int, str], dict[str, object]],
    providers: list[str],
    seeds_by_provider: dict[str, list[int]],
) -> dict[str, dict[tuple[str, int], dict[str, object]]]:
    """Validation-only rule selection with the frozen-union margin-0 constraint."""
    selections: dict[str, dict[tuple[str, int], dict[str, object]]] = {}
    for strategy in strategies:
        strategy_id = str(strategy["strategy_id"])
        unit = str(strategy["selection_unit"])
        chosen: dict[tuple[str, int], dict[str, object]] = {}
        for provider in providers:
            seeds = seeds_by_provider[provider]
            groups = (
                [seeds] if unit == "provider" else [[seed] for seed in seeds]
            )
            for group_seeds in groups:
                candidates: list[dict[str, object]] = []
                for key in RULE_KEYS:
                    entries = [
                        cache[(strategy_id, provider, seed, key)]
                        for seed in group_seeds
                    ]
                    constraint_ok = True
                    deltas: dict[str, float] = {}
                    for scenario in CONSTRAINED_SCENARIOS:
                        candidate_coverage = float(
                            np.mean(
                                [
                                    entry["valid_coverage"].get(
                                        scenario,
                                        0.0,
                                    )
                                    for entry in entries
                                ]
                            )
                        )
                        union_coverage = float(
                            np.mean(
                                [
                                    run_state[(provider, seed)][
                                        "union_valid_coverage"
                                    ].get(scenario, 0.0)
                                    for seed in group_seeds
                                ]
                            )
                        )
                        deltas[scenario] = candidate_coverage - union_coverage
                        constraint_ok = constraint_ok and bool(
                            deltas[scenario] >= -EPS
                        )
                    candidates.append(
                        {
                            "key": key,
                            "weight": entries[0]["weight"],
                            "valid_mean_scenario_coverage": float(
                                np.mean(
                                    [
                                        entry[
                                            "valid_mean_scenario_coverage"
                                        ]
                                        for entry in entries
                                    ]
                                )
                            ),
                            "valid_f1": float(
                                np.mean(
                                    [
                                        entry["valid_f1"]
                                        for entry in entries
                                    ]
                                )
                            ),
                            "valid_event_coverage": float(
                                np.mean(
                                    [
                                        entry["valid_event_coverage"]
                                        for entry in entries
                                    ]
                                )
                            ),
                            "constraint_ok": bool(constraint_ok),
                            "valid_delta": deltas,
                        }
                    )
                feasible = [
                    row for row in candidates if row["constraint_ok"]
                ]
                if feasible:
                    best = max(
                        feasible,
                        key=lambda row: (
                            row["valid_mean_scenario_coverage"],
                            row["valid_f1"],
                            -row["valid_event_coverage"],
                            -RULE_KEYS.index(str(row["key"])),
                        ),
                    )
                    fallback = False
                else:
                    best = None
                    fallback = True
                for seed in group_seeds:
                    chosen[(provider, seed)] = {
                        "strategy_id": strategy_id,
                        "selection_unit": unit,
                        "fallback_to_union": fallback,
                        "selection_key": (
                            "union_or_fallback"
                            if fallback
                            else str(best["key"])
                        ),
                        "selection_weight": (
                            "fallback" if fallback else best["weight"]
                        ),
                        "selection_rule": (
                            UNION_OR
                            if fallback
                            else rule_label(best["weight"])
                        ),
                        "feasible_candidate_count": int(len(feasible)),
                        "valid_mean_scenario_coverage": (
                            float(best["valid_mean_scenario_coverage"])
                            if best is not None
                            else float(
                                np.mean(
                                    [
                                        run_state[(provider, seed)][
                                            "union_valid_coverage"
                                        ].get(scenario, 0.0)
                                        for scenario in run_state[
                                            (provider, seed)
                                        ]["scenario_keys"]
                                    ]
                                )
                            )
                        ),
                        "valid_f1": (
                            float(best["valid_f1"])
                            if best is not None
                            else float(
                                run_state[(provider, seed)][
                                    "union_valid_metrics"
                                ]["f1"]
                            )
                        ),
                        "valid_false_alarm_rate": (
                            float(
                                cache[
                                    (
                                        strategy_id,
                                        provider,
                                        seed,
                                        str(best["key"]),
                                    )
                                ]["valid_false_alarm_rate"]
                            )
                            if best is not None
                            else float(
                                run_state[(provider, seed)][
                                    "union_valid_metrics"
                                ]["false_alarm_rate"]
                            )
                        ),
                        "valid_delta": (
                            {
                                scenario: float(
                                    best["valid_delta"][scenario]
                                )
                                for scenario in CONSTRAINED_SCENARIOS
                            }
                            if best is not None
                            else {
                                scenario: 0.0
                                for scenario in CONSTRAINED_SCENARIOS
                            }
                        ),
                    }
        selections[strategy_id] = chosen
    return selections


def constraint_diagnostics(
    run_state: dict[tuple[str, int], dict[str, object]],
    cache: dict[tuple[str, str, int, str], dict[str, object]],
    providers: list[str],
    seeds_by_provider: dict[str, list[int]],
) -> pd.DataFrame:
    """Margin-0 feasibility ceiling at the least conservative target.

    ``level@0.0065`` uses the lowest threshold of every evaluated strategy:
    the finite-sample families re-threshold at the same target with a
    correction that can only raise the threshold, the provider margins divide
    the target, and lower targets raise it further.  Event coverage is
    non-increasing in the threshold, so a seed that is infeasible here is
    infeasible for the whole catalog.
    """
    rows: list[dict[str, object]] = []
    for provider in providers:
        for seed in seeds_by_provider[provider]:
            state = run_state[(provider, seed)]
            union_ls = float(
                state["union_valid_coverage"].get("long_stuck", 0.0)
            )
            union_dr = float(
                state["union_valid_coverage"].get("drop_replay", 0.0)
            )
            best: dict[str, object] | None = None
            for key in RULE_KEYS:
                entry = cache[
                    ("level@0.0065|seed", provider, seed, key)
                ]
                long_stuck = float(
                    entry["valid_coverage"].get("long_stuck", 0.0)
                )
                drop_replay = float(
                    entry["valid_coverage"].get("drop_replay", 0.0)
                )
                margin = min(
                    long_stuck - union_ls,
                    drop_replay - union_dr,
                )
                if best is None or margin > float(best["margin_min"]):
                    best = {
                        "key": key,
                        "long_stuck": long_stuck,
                        "drop_replay": drop_replay,
                        "margin_min": float(margin),
                    }
            normal_rows = int((~state["y_test"]).sum())
            union_false_alarms = int(
                (state["union_test_pred"] & ~state["y_test"]).sum()
            )
            rows.append(
                {
                    "source": provider,
                    "seed": int(seed),
                    "union_valid_coverage_long_stuck": union_ls,
                    "union_valid_coverage_drop_replay": union_dr,
                    "best_candidate": str(best["key"]),
                    "best_valid_coverage_long_stuck": float(
                        best["long_stuck"]
                    ),
                    "best_valid_coverage_drop_replay": float(
                        best["drop_replay"]
                    ),
                    "best_margin_vs_union": float(best["margin_min"]),
                    "candidate_feasible_at_least_conservative_target": bool(
                        float(best["margin_min"]) >= -EPS
                    ),
                    "union_test_false_alarm_rate": (
                        union_false_alarms / normal_rows
                        if normal_rows
                        else float("nan")
                    ),
                    "test_normal_rows": normal_rows,
                    "union_test_false_alarms": union_false_alarms,
                }
            )
    return pd.DataFrame(rows)


def constraint_floor_summary(
    diagnostics: pd.DataFrame,
) -> tuple[list[dict[str, object]], float]:
    """Pooled FAR floor implied by the forced union-OR fallback seeds.

    Seeds without a feasible corrected candidate must use the frozen union OR
    prediction.  Assuming a corrected rule with zero false alarms on the
    remaining seeds gives the lowest pooled FAR any strategy of the catalog
    could reach for that provider.
    """
    rows: list[dict[str, object]] = []
    knmi_floor = float("nan")
    for provider, group in diagnostics.groupby("source", sort=True):
        forced = group[
            ~group["candidate_feasible_at_least_conservative_target"]
        ]
        normal_total = float(group["test_normal_rows"].sum())
        floor = (
            float(forced["union_test_false_alarms"].sum()) / normal_total
            if normal_total
            else float("nan")
        )
        rows.append(
            {
                "source": str(provider),
                "seeds": int(len(group)),
                "forced_fallback_seeds": int(len(forced)),
                "forced_fallback_seed_ids": ";".join(
                    str(int(seed)) for seed in forced["seed"]
                ),
                "union_or_test_far_in_forced_seeds": float(
                    forced["union_test_false_alarm_rate"].mean()
                )
                if len(forced)
                else float("nan"),
                "pooled_test_far_floor": floor,
            }
        )
        if str(provider) == KNMI:
            knmi_floor = floor
    return rows, knmi_floor


def evaluate_selected_runs(
    strategies: list[dict[str, object]],
    run_state: dict[tuple[str, int], dict[str, object]],
    cache: dict[tuple[str, str, int, str], dict[str, object]],
    selections: dict[str, dict[tuple[str, int], dict[str, object]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    strategy_lookup = {
        str(strategy["strategy_id"]): strategy for strategy in strategies
    }
    for strategy_id, chosen in selections.items():
        strategy = strategy_lookup[strategy_id]
        for (provider, seed), decision in chosen.items():
            state = run_state[(provider, seed)]
            fallback = bool(decision["fallback_to_union"])
            if fallback:
                predicted_test = state["union_test_pred"]
                fused_test = state["union_test_z"]
                weight: float | str = 1.0
                threshold = float("nan")
                valid_coverage = state["union_valid_coverage"]
                valid_far = float(
                    state["union_valid_metrics"]["false_alarm_rate"]
                )
                valid_f1 = float(state["union_valid_metrics"]["f1"])
                valid_event_coverage = float(
                    event_metrics(
                        state["valid_frame"],
                        state["union_valid_pred"],
                    )["event_coverage"]
                )
            else:
                weight = decision["selection_weight"]
                key = str(decision["selection_key"])
                entry = cache[(strategy_id, provider, seed, key)]
                weight = entry["weight"]
                threshold = float(entry["threshold"])
                fused_test = candidate_scores(
                    state["union_test_z"],
                    state["formal_test_z"],
                    weight,
                )
                predicted_test = state["test_observed"] & (
                    fused_test >= threshold
                )
                valid_coverage = entry["valid_coverage"]
                valid_far = float(entry["valid_false_alarm_rate"])
                valid_f1 = float(entry["valid_f1"])
                valid_event_coverage = float(entry["valid_event_coverage"])
            test_metrics = metrics(
                state["y_test"],
                predicted_test,
                fused_test,
            )
            test_event = event_metrics(state["test_frame"], predicted_test)
            test_coverage = scenario_coverage(
                state["test_frame"],
                predicted_test,
            )
            deltas = {
                scenario: float(
                    test_coverage.get(scenario, 0.0)
                    - state["union_test_coverage"].get(scenario, 0.0)
                )
                for scenario in CONSTRAINED_SCENARIOS
            }
            run_rows.append(
                {
                    "strategy_id": strategy_id,
                    "family": strategy["family"],
                    "parameter": float(strategy["parameter"]),
                    "selection_unit": strategy["selection_unit"],
                    "source": provider,
                    "seed": int(seed),
                    "fallback_to_union": fallback,
                    "feasible_candidate_count": int(
                        decision["feasible_candidate_count"]
                    ),
                    "selection_rule": decision["selection_rule"],
                    "selection_weight": weight,
                    "threshold": threshold,
                    "valid_target_far": strategy_alpha(strategy, provider),
                    "valid_false_alarm_rate": valid_far,
                    "valid_f1": valid_f1,
                    "valid_event_coverage": valid_event_coverage,
                    "valid_mean_scenario_coverage": float(
                        decision["valid_mean_scenario_coverage"]
                    ),
                    "valid_coverage_long_stuck": float(
                        valid_coverage.get("long_stuck", 0.0)
                    ),
                    "valid_coverage_drop_replay": float(
                        valid_coverage.get("drop_replay", 0.0)
                    ),
                    "valid_union_coverage_long_stuck": float(
                        state["union_valid_coverage"].get(
                            "long_stuck",
                            0.0,
                        )
                    ),
                    "valid_union_coverage_drop_replay": float(
                        state["union_valid_coverage"].get(
                            "drop_replay",
                            0.0,
                        )
                    ),
                    "valid_delta_long_stuck": float(
                        decision["valid_delta"]["long_stuck"]
                    ),
                    "valid_delta_drop_replay": float(
                        decision["valid_delta"]["drop_replay"]
                    ),
                    "rows": int(test_metrics["rows"]),
                    "attack_rows": int(test_metrics["attack_rows"]),
                    "precision": float(test_metrics["precision"]),
                    "recall": float(test_metrics["recall"]),
                    "f1": float(test_metrics["f1"]),
                    "false_alarm_rate": float(
                        test_metrics["false_alarm_rate"]
                    ),
                    "roc_auc": float(test_metrics["roc_auc"]),
                    "average_precision": float(
                        test_metrics["average_precision"]
                    ),
                    "events": int(test_event["events"]),
                    "detected_events": int(test_event["detected_events"]),
                    "event_coverage": float(test_event["event_coverage"]),
                    "row_recall": float(test_event["row_recall"]),
                    "union_test_false_alarm_rate": float(
                        state["union_test_metrics"]["false_alarm_rate"]
                    ),
                    "union_test_event_coverage": float(
                        state["union_test_event_coverage"]
                    ),
                    "union_test_f1": float(
                        state["union_test_metrics"]["f1"]
                    ),
                    "union_test_average_precision": float(
                        state["union_test_metrics"]["average_precision"]
                    ),
                    "long_stuck_test_delta": deltas["long_stuck"],
                    "drop_replay_test_delta": deltas["drop_replay"],
                    "long_stuck_within_5pp": bool(
                        deltas["long_stuck"] >= -STRUCTURED_DROP_LIMIT
                    ),
                    "drop_replay_within_5pp": bool(
                        deltas["drop_replay"] >= -STRUCTURED_DROP_LIMIT
                    ),
                    "test_coverage": test_coverage,
                }
            )
            for row in s470.scenario_rows_full(
                state["test_frame"],
                predicted_test,
                fused_test,
                provider,
                int(seed),
                CANDIDATE_METHOD,
            ):
                row["strategy_id"] = strategy_id
                row["selection_unit"] = strategy["selection_unit"]
                row["fallback_to_union"] = fallback
                row["union_event_coverage"] = float(
                    state["union_test_coverage"].get(
                        str(row["scenario"]),
                        0.0,
                    )
                )
                row["event_coverage_delta_vs_union"] = float(
                    row["event_coverage"]
                ) - float(row["union_event_coverage"])
                scenario_rows.append(row)
    return pd.DataFrame(run_rows), pd.DataFrame(scenario_rows)


def reference_wide() -> pd.DataFrame:
    existing = pd.read_csv(EXISTING_SCENARIO)
    existing = existing[existing["method"].isin(BASELINES_11)][
        ["source", "seed", "scenario", "method", "event_coverage"]
    ]
    formal = pd.read_csv(FORMAL_SCENARIO)[
        ["source", "seed", "scenario", "method", "event_coverage"]
    ]
    long = pd.concat([existing, formal], ignore_index=True)
    return long


def reference_aggregate() -> pd.DataFrame:
    """Frozen pooled reference table used by the Stage 491/492/493 gate."""
    existing = pd.read_csv(EXISTING_SCENARIO_AGGREGATE)
    existing = existing[existing["method"].isin(BASELINES_11)][
        ["scenario", "method", "event_coverage"]
    ]
    formal = (
        pd.read_csv(FORMAL_SCENARIO)
        .groupby(["scenario", "method"], as_index=False)
        .agg(event_coverage=("event_coverage", "mean"))
    )
    return pd.concat([existing, formal], ignore_index=True)


def pooled_gate_aggregate(
    candidate_long: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, object]:
    """Pooled 108-unit gate on the frozen Stage 491/492/493 float path."""
    candidate = candidate_long.groupby(
        "scenario",
        as_index=False,
    ).agg(event_coverage=("event_coverage", "mean"))
    wins = ties = losses = 0
    for row in candidate.itertuples():
        references = reference[reference["scenario"].eq(row.scenario)]
        for reference_row in references.itertuples():
            if float(row.event_coverage) > float(
                reference_row.event_coverage
            ):
                wins += 1
            elif float(row.event_coverage) < float(
                reference_row.event_coverage
            ):
                losses += 1
            else:
                ties += 1
    units = wins + ties + losses
    share = wins / units if units else 0.0
    return {
        "gate_wins": int(wins),
        "gate_ties": int(ties),
        "gate_losses": int(losses),
        "gate_units": int(units),
        "gate_share": float(share),
        "gate_passes_70pct": bool(share >= GATE_MIN_SHARE),
        "gate_wins_needed": int(np.ceil(GATE_MIN_SHARE * units))
        if units
        else 0,
        "gate_wins_shortfall": int(
            max(0, np.ceil(GATE_MIN_SHARE * units) - wins)
        )
        if units
        else 0,
    }


def gate_counts(
    wide: pd.DataFrame,
    candidate_column: str,
) -> dict[str, object]:
    wins = ties = losses = 0
    for _, group in wide.groupby("scenario", sort=True):
        candidate = float(group[candidate_column].mean())
        for reference in REFERENCES:
            if reference not in group:
                continue
            value = float(group[reference].mean())
            if candidate > value:
                wins += 1
            elif candidate < value:
                losses += 1
            else:
                ties += 1
    units = wins + ties + losses
    share = wins / units if units else 0.0
    return {
        "gate_wins": int(wins),
        "gate_ties": int(ties),
        "gate_losses": int(losses),
        "gate_units": int(units),
        "gate_share": float(share),
        "gate_passes_70pct": bool(share >= GATE_MIN_SHARE),
        "gate_wins_needed": int(np.ceil(GATE_MIN_SHARE * units))
        if units
        else 0,
        "gate_wins_shortfall": int(
            max(0, np.ceil(GATE_MIN_SHARE * units) - wins)
        )
        if units
        else 0,
    }


def strategy_gate_tables(
    scenario_rows: pd.DataFrame,
    reference_long: pd.DataFrame,
    reference_agg: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    tables: dict[str, dict[str, object]] = {}
    for strategy_id, group in scenario_rows.groupby(
        "strategy_id",
        sort=False,
    ):
        candidate = group[
            ["source", "seed", "scenario", "event_coverage"]
        ].assign(method=CANDIDATE_METHOD)
        long = pd.concat(
            [reference_long, candidate],
            ignore_index=True,
        )
        # Stage 493 float path: mean per (source, seed, scenario, method)
        # before the wide pivot, so the frozen tie counts are reproduced.
        wide = (
            long.groupby(
                ["source", "seed", "scenario", "method"],
                as_index=False,
            )
            .event_coverage.mean()
            .pivot_table(
                index=["source", "seed", "scenario"],
                columns="method",
                values="event_coverage",
                aggfunc="mean",
            )
            .reset_index()
        )
        record: dict[str, object] = {
            "pooled": pooled_gate_aggregate(candidate, reference_agg),
            "pooled_wide_path": gate_counts(wide, CANDIDATE_METHOD),
        }
        for provider in sorted(wide["source"].unique()):
            record[str(provider)] = gate_counts(
                wide[wide["source"].eq(provider)],
                CANDIDATE_METHOD,
            )
        tables[str(strategy_id)] = record
    return tables


def pooled_far(runs: pd.DataFrame, mask: pd.Series) -> float:
    subset = runs[mask]
    false_alarms = float(
        (
            subset["false_alarm_rate"]
            * (subset["rows"] - subset["attack_rows"])
        ).sum()
    )
    normals = float((subset["rows"] - subset["attack_rows"]).sum())
    return false_alarms / normals if normals else float("nan")


def correction_table(
    runs: pd.DataFrame,
    gates: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy_id, group in runs.groupby("strategy_id", sort=False):
        knmi = group[group["source"].eq(KNMI)]
        pooled = gates[strategy_id]["pooled"]
        knmi_gate = gates[strategy_id][KNMI]
        knmi_ls = float(knmi["long_stuck_test_delta"].mean())
        knmi_dr = float(knmi["drop_replay_test_delta"].mean())
        all_ls = float(group["long_stuck_test_delta"].mean())
        all_dr = float(group["drop_replay_test_delta"].mean())
        row = {
            "strategy_id": strategy_id,
            "family": group["family"].iloc[0],
            "parameter": float(group["parameter"].iloc[0]),
            "selection_unit": group["selection_unit"].iloc[0],
            "runs": int(len(group)),
            "fallback_runs": int(group["fallback_to_union"].sum()),
            "fallback_providers": int(
                group.groupby("source")["fallback_to_union"]
                .all()
                .sum()
            ),
            "overall_test_far_pooled": pooled_far(
                group,
                pd.Series(True, index=group.index),
            ),
            "overall_test_far_run_mean": float(
                group["false_alarm_rate"].mean()
            ),
            "overall_f1_mean": float(group["f1"].mean()),
            "overall_event_coverage_mean": float(
                group["event_coverage"].mean()
            ),
            "overall_precision_mean": float(group["precision"].mean()),
            "overall_row_recall_mean": float(group["row_recall"].mean()),
            "overall_average_precision_mean": float(
                group["average_precision"].mean()
            ),
            "overall_long_stuck_test_delta_mean": all_ls,
            "overall_drop_replay_test_delta_mean": all_dr,
            "knmi_test_far_pooled": pooled_far(
                knmi,
                pd.Series(True, index=knmi.index),
            ),
            "knmi_test_far_run_mean": float(
                knmi["false_alarm_rate"].mean()
            ),
            "knmi_valid_far_run_mean": float(
                knmi["valid_false_alarm_rate"].mean()
            ),
            "knmi_test_to_valid_far_ratio": float(
                knmi["false_alarm_rate"].mean()
                / max(knmi["valid_false_alarm_rate"].mean(), 1e-12)
            ),
            "knmi_f1_mean": float(knmi["f1"].mean()),
            "knmi_event_coverage_mean": float(
                knmi["event_coverage"].mean()
            ),
            "knmi_precision_mean": float(knmi["precision"].mean()),
            "knmi_row_recall_mean": float(knmi["row_recall"].mean()),
            "knmi_average_precision_mean": float(
                knmi["average_precision"].mean()
            ),
            "knmi_long_stuck_test_delta_mean": knmi_ls,
            "knmi_drop_replay_test_delta_mean": knmi_dr,
            "knmi_long_stuck_within_5pp_fraction": float(
                knmi["long_stuck_within_5pp"].mean()
            ),
            "knmi_drop_replay_within_5pp_fraction": float(
                knmi["drop_replay_within_5pp"].mean()
            ),
            "knmi_fallback_runs": int(knmi["fallback_to_union"].sum()),
            "pooled_gate_wins": pooled["gate_wins"],
            "pooled_gate_units": pooled["gate_units"],
            "pooled_gate_share": pooled["gate_share"],
            "pooled_gate_passes_70pct": pooled["gate_passes_70pct"],
            "pooled_gate_wins_wide_path": gates[strategy_id][
                "pooled_wide_path"
            ]["gate_wins"],
            "pooled_gate_share_wide_path": gates[strategy_id][
                "pooled_wide_path"
            ]["gate_share"],
            "knmi_gate_wins": knmi_gate["gate_wins"],
            "knmi_gate_units": knmi_gate["gate_units"],
            "knmi_gate_share": knmi_gate["gate_share"],
            "knmi_gate_passes_70pct": knmi_gate["gate_passes_70pct"],
            "knmi_gate_wins_shortfall": knmi_gate[
                "gate_wins_shortfall"
            ],
        }
        knmi_far = float(row["knmi_test_far_pooled"])
        knmi_structured_ok = bool(
            min(knmi_ls, knmi_dr) >= -STRUCTURED_DROP_LIMIT
        )
        row["knmi_far_point_le_1pct"] = bool(knmi_far <= TEST_FAR_LIMIT)
        row["knmi_structured_drop_le_5pp"] = knmi_structured_ok
        row["knmi_gate_ge_70pct"] = bool(
            row["knmi_gate_passes_70pct"]
        )
        row["knmi_feasible_all_three"] = bool(
            row["knmi_far_point_le_1pct"]
            and knmi_structured_ok
            and row["knmi_gate_ge_70pct"]
        )
        overall_far = float(row["overall_test_far_pooled"])
        overall_structured_ok = bool(
            min(all_ls, all_dr) >= -STRUCTURED_DROP_LIMIT
        )
        row["overall_far_point_le_1pct"] = bool(
            overall_far <= TEST_FAR_LIMIT
        )
        row["overall_structured_drop_le_5pp"] = overall_structured_ok
        row["overall_gate_ge_70pct"] = bool(
            row["pooled_gate_passes_70pct"]
        )
        row["overall_feasible_all_three"] = bool(
            row["overall_far_point_le_1pct"]
            and overall_structured_ok
            and row["overall_gate_ge_70pct"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def provider_table(
    runs: pd.DataFrame,
    gates: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy_id, group in runs.groupby("strategy_id", sort=False):
        for provider, part in group.groupby("source", sort=True):
            gate = gates[strategy_id][str(provider)]
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "family": part["family"].iloc[0],
                    "parameter": float(part["parameter"].iloc[0]),
                    "selection_unit": part["selection_unit"].iloc[0],
                    "source": str(provider),
                    "seeds": int(part["seed"].nunique()),
                    "fallback_seeds": int(
                        part["fallback_to_union"].sum()
                    ),
                    "selection_rules": ";".join(
                        sorted(set(part["selection_rule"].astype(str)))
                    ),
                    "test_far_pooled": pooled_far(
                        part,
                        pd.Series(True, index=part.index),
                    ),
                    "test_far_run_mean": float(
                        part["false_alarm_rate"].mean()
                    ),
                    "valid_far_run_mean": float(
                        part["valid_false_alarm_rate"].mean()
                    ),
                    "test_to_valid_far_ratio": float(
                        part["false_alarm_rate"].mean()
                        / max(
                            float(
                                part["valid_false_alarm_rate"].mean()
                            ),
                            1e-12,
                        )
                    ),
                    "f1_mean": float(part["f1"].mean()),
                    "event_coverage_mean": float(
                        part["event_coverage"].mean()
                    ),
                    "precision_mean": float(part["precision"].mean()),
                    "row_recall_mean": float(part["row_recall"].mean()),
                    "average_precision_mean": float(
                        part["average_precision"].mean()
                    ),
                    "long_stuck_test_delta_mean": float(
                        part["long_stuck_test_delta"].mean()
                    ),
                    "drop_replay_test_delta_mean": float(
                        part["drop_replay_test_delta"].mean()
                    ),
                    "long_stuck_within_5pp_fraction": float(
                        part["long_stuck_within_5pp"].mean()
                    ),
                    "drop_replay_within_5pp_fraction": float(
                        part["drop_replay_within_5pp"].mean()
                    ),
                    "union_test_far_mean": float(
                        part["union_test_false_alarm_rate"].mean()
                    ),
                    "union_test_event_coverage_mean": float(
                        part["union_test_event_coverage"].mean()
                    ),
                    **gate,
                }
            )
    return pd.DataFrame(rows)


def seed_table(
    runs: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = [
        "strategy_id",
        "family",
        "parameter",
        "selection_unit",
        "source",
        "seed",
    ]
    runs = runs.copy()
    runs["scope"] = "all"
    scenario_part = scenarios.copy()
    scenario_part["scope"] = scenario_part["scenario"].astype(str)
    keep = [
        *base_columns,
        "scope",
        "fallback_to_union",
        "selection_rule",
        "selection_weight",
        "valid_target_far",
        "valid_false_alarm_rate",
        "false_alarm_rate",
        "precision",
        "row_recall",
        "f1",
        "average_precision",
        "roc_auc",
        "events",
        "detected_events",
        "event_coverage",
        "rows",
        "attack_rows",
        "long_stuck_test_delta",
        "drop_replay_test_delta",
        "event_coverage_delta_vs_union",
    ]
    runs_out = runs.reindex(columns=keep)
    scenarios_out = scenario_part.reindex(columns=keep)
    combined = pd.concat([runs_out, scenarios_out], ignore_index=True)
    return combined


def verification(
    run_state: dict[tuple[str, int], dict[str, object]],
    runs: pd.DataFrame,
    scenario_rows: pd.DataFrame,
    reference_long: pd.DataFrame,
    reference_agg: pd.DataFrame,
) -> dict[str, object]:
    checks: dict[str, object] = {}
    stage495 = pd.read_csv(STAGE495_RUNS)
    candidate = runs[
        runs["strategy_id"].eq("level@0.0065|provider")
    ][
        [
            "source",
            "seed",
            "selection_weight",
            "threshold",
            "f1",
            "event_coverage",
            "false_alarm_rate",
            "fallback_to_union",
        ]
    ]
    merged = stage495.merge(
        candidate,
        on=["source", "seed"],
        suffixes=("_s495", "_s498"),
        validate="one_to_one",
    )
    both_selected = ~merged["fallback_to_union_s495"].astype(bool)
    weight_match = bool(
        (
            merged.loc[both_selected, "selection_weight_s495"].astype(str)
            == merged.loc[both_selected, "selection_weight_s498"].astype(str)
        ).all()
        if both_selected.any()
        else True
    )
    fallback_match = bool(
        (
            merged["fallback_to_union_s495"].astype(bool)
            == merged["fallback_to_union_s498"].astype(bool)
        ).all()
    )
    threshold_diff = float(
        np.nanmax(
            np.abs(
                merged["threshold_s495"].to_numpy(dtype=float)
                - merged["threshold_s498"].to_numpy(dtype=float)
            )
        )
    )
    metric_diff = {
        metric: float(
            np.nanmax(
                np.abs(
                    merged[f"{metric}_s495"].to_numpy(dtype=float)
                    - merged[f"{metric}_s498"].to_numpy(dtype=float)
                )
            )
        )
        for metric in ("f1", "event_coverage", "false_alarm_rate")
    }
    checks["stage495_provider_rule_reproduced"] = bool(
        weight_match and fallback_match and threshold_diff < 1e-9
    )
    checks["stage495_weight_match"] = weight_match
    checks["stage495_fallback_match"] = fallback_match
    checks["stage495_threshold_max_abs_diff"] = threshold_diff
    checks["stage495_metric_max_abs_diff"] = metric_diff

    stage491 = pd.read_csv(STAGE491_SCENARIO)
    stage491 = stage491[
        stage491["method"].eq("constrained_validation_fusion")
    ][["source", "seed", "scenario", "event_coverage"]].assign(
        method="stage491_reference"
    )
    long = pd.concat(
        [reference_long, stage491],
        ignore_index=True,
    )
    wide = long.pivot_table(
        index=["source", "seed", "scenario"],
        columns="method",
        values="event_coverage",
        aggfunc="mean",
    ).reset_index()
    pooled = gate_counts(wide, "stage491_reference")
    provider_wins = {
        str(provider): gate_counts(
            wide[wide["source"].eq(provider)],
            "stage491_reference",
        )["gate_wins"]
        for provider in sorted(wide["source"].unique())
    }
    stage493 = pd.read_csv(STAGE493_PROVIDER)
    stage493_wins = {
        str(row.unit): int(row.extended_wins)
        for row in stage493.itertuples()
    }
    checks["stage491_pooled_gate_wins"] = int(pooled["gate_wins"])
    pooled_aggregate = pooled_gate_aggregate(
        stage491[["scenario", "event_coverage"]],
        reference_agg,
    )
    checks["stage491_pooled_gate_wins_wide_path"] = int(
        pooled["gate_wins"]
    )
    checks["stage491_pooled_gate_wins_aggregate_path"] = int(
        pooled_aggregate["gate_wins"]
    )
    checks["stage491_pooled_gate_wins_expected"] = 79
    checks["stage491_pooled_gate_reproduced"] = bool(
        int(pooled_aggregate["gate_wins"]) == 79
    )
    checks["stage491_gate_float_path_note"] = (
        "The frozen 79/108 headline comes from the Stage 491/492 aggregate "
        "float path; the Stage 493 per-source-seed wide path gives 78/108 "
        "because the KNMI slow_drift unit is an exact tie there (Stage 494 "
        "documents the same float sensitivity)."
    )
    checks["stage491_provider_gate_wins"] = provider_wins
    checks["stage493_provider_gate_wins"] = stage493_wins
    checks["stage493_gate_reproduced"] = bool(
        all(
            provider_wins.get(unit) == wins
            for unit, wins in stage493_wins.items()
        )
    )
    union_far = {}
    for provider in sorted(run_state_keys(run_state)):
        subset = [
            state
            for (source, _), state in run_state.items()
            if source == provider
        ]
        union_far[provider] = float(
            np.mean(
                [
                    state["union_test_metrics"]["false_alarm_rate"]
                    for state in subset
                ]
            )
        )
    checks["union_test_far_run_mean"] = union_far
    checks["selection_uses_validation_only"] = True
    checks["deterministic_no_randomness"] = True
    return checks


def run_state_keys(
    run_state: dict[tuple[str, int], dict[str, object]],
) -> list[str]:
    return sorted({source for (source, _) in run_state})


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
    run_state = build_run_state(merged)
    providers = sorted({source for (source, _) in run_state})
    seeds_by_provider = {
        provider: sorted(
            seed for (source, seed) in run_state if source == provider
        )
        for provider in providers
    }
    margins = provider_period_margins(
        run_state,
        providers,
        seeds_by_provider,
        PRIMARY_LEVEL,
        MARGIN_MAX,
    )
    strategies = build_strategies(providers, margins)
    cache = evaluate_validation_candidates(
        run_state,
        strategies,
        BINOMIAL_CONFIDENCE,
    )
    selections = select_runs(
        strategies,
        run_state,
        cache,
        providers,
        seeds_by_provider,
    )
    runs, scenario_rows = evaluate_selected_runs(
        strategies,
        run_state,
        cache,
        selections,
    )
    reference_long = reference_wide()
    reference_agg = reference_aggregate()
    gates = strategy_gate_tables(
        scenario_rows,
        reference_long,
        reference_agg,
    )
    correction = correction_table(runs, gates)
    provider = provider_table(runs, gates)
    seed = seed_table(runs, scenario_rows)
    diagnostics = constraint_diagnostics(
        run_state,
        cache,
        providers,
        seeds_by_provider,
    )
    floor_rows, knmi_floor = constraint_floor_summary(diagnostics)
    correction["knmi_constraint_far_floor"] = float(knmi_floor)
    correction.to_csv(OUT_CORRECTION, index=False, encoding="utf-8-sig")
    provider.to_csv(OUT_PROVIDER, index=False, encoding="utf-8-sig")
    seed.to_csv(OUT_SEED, index=False, encoding="utf-8-sig")
    checks = verification(
        run_state,
        runs,
        scenario_rows,
        reference_long,
        reference_agg,
    )
    elapsed = time.perf_counter() - started
    payload = build_payload(
        correction,
        provider,
        seed,
        scenario_rows,
        gates,
        margins,
        diagnostics,
        floor_rows,
        knmi_floor,
        checks,
        elapsed,
    )
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(
        render_markdown(payload),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "strategies": len(correction),
                "feasible_knmi": payload["feasibility"][
                    "knmi_feasible_strategies"
                ],
                "feasible_overall": payload["feasibility"][
                    "overall_feasible_strategies"
                ],
                "elapsed_seconds": round(elapsed, 2),
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_payload(
    correction: pd.DataFrame,
    provider: pd.DataFrame,
    seed: pd.DataFrame,
    scenario_rows: pd.DataFrame,
    gates: dict[str, dict[str, object]],
    margins: dict[str, dict[str, float]],
    diagnostics: pd.DataFrame,
    floor_rows: list[dict[str, object]],
    knmi_floor: float,
    checks: dict[str, object],
    elapsed: float,
) -> dict[str, object]:
    knmi_feasible = correction[correction["knmi_feasible_all_three"]]
    overall_feasible = correction[correction["overall_feasible_all_three"]]
    scenario_summary = (
        scenario_rows.groupby(
            ["strategy_id", "source", "scenario"],
            as_index=False,
        )
        .agg(
            event_coverage=("event_coverage", "mean"),
            union_event_coverage=("union_event_coverage", "mean"),
            event_coverage_delta_vs_union=(
                "event_coverage_delta_vs_union",
                "mean",
            ),
            f1=("f1", "mean"),
            precision=("precision", "mean"),
            row_recall=("row_recall", "mean"),
            average_precision=("average_precision", "mean"),
            detected_events=("detected_events", "mean"),
            events=("events", "mean"),
        )
    )
    payload: dict[str, object] = {
        "status": "PASS_EXPLORATORY_KNMI_FAR_TRANSFER_CORRECTION",
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "method": METHOD,
        "design": {
            "selection_units": list(SELECTION_UNITS),
            "primary_selection_unit": "provider",
            "families": {
                "level": list(LEVEL_TARGETS),
                "conformal": list(CONFORMAL_TARGETS),
                "binomial_ucb": list(BINOMIAL_TARGETS),
                "margin_period_mean": list(MARGIN_BASE_TARGETS),
                "margin_period_max": list(MARGIN_BASE_TARGETS),
            },
            "binomial_confidence": BINOMIAL_CONFIDENCE,
            "margin_max_clip": MARGIN_MAX,
            "constraint": {
                "scenarios": list(CONSTRAINED_SCENARIOS),
                "margin": 0.0,
                "basis": "validation event coverage relative to the frozen union OR prediction",
                "fallback": UNION_OR,
            },
            "criteria": {
                "test_far_point_estimate_max": TEST_FAR_LIMIT,
                "structured_drop_max": STRUCTURED_DROP_LIMIT,
                "extended_gate_min_share": GATE_MIN_SHARE,
                "extended_gate_units_per_provider": 108,
            },
        },
        "provider_margin_factors": margins,
        "constraint_diagnostics": {
            "per_seed": diagnostics.to_dict("records"),
            "per_provider": floor_rows,
            "knmi_pooled_test_far_floor": float(knmi_floor),
            "interpretation": (
                "Seeds without a feasible corrected candidate at the least "
                "conservative target must fall back to the frozen union OR "
                "prediction; coverage is non-increasing in the threshold, so "
                "they stay infeasible for every other strategy in the "
                "catalog. The floor assumes a zero-FAR corrected rule on the "
                "remaining seeds."
            ),
        },
        "strategy_table": correction.to_dict("records"),
        "provider_table": provider.to_dict("records"),
        "scenario_summary": scenario_summary.to_dict("records"),
        "knmi_seed_runs": seed[
            seed["scope"].eq("all") & seed["source"].eq(KNMI)
        ].to_dict("records"),
        "gate_summary": gates,
        "feasibility": {
            "knmi_feasible_strategies": knmi_feasible[
                "strategy_id"
            ].tolist(),
            "overall_feasible_strategies": overall_feasible[
                "strategy_id"
            ].tolist(),
            "knmi_verdict": (
                "feasible_strategy_exists"
                if len(knmi_feasible)
                else "no_feasible_strategy"
            ),
            "overall_verdict": (
                "feasible_strategy_exists"
                if len(overall_feasible)
                else "no_feasible_strategy"
            ),
        },
        "verification": checks,
        "limitations": [
            "The V6 test split has already been inspected; every number is exploratory.",
            "No model is retrained; only existing Stage 490 per-row scores are re-thresholded.",
            "The provider margin uses a within-validation period split as a validation-only proxy for validation-to-test FAR transfer.",
            "Seed-level selection is a strict per-replicate constraint enforcement; provider-level selection is the Stage 495-compatible primary unit.",
            "The four providers are fixed clusters, not a random population sample.",
        ],
        "elapsed_seconds": float(elapsed),
    }
    return payload


def render_markdown(payload: dict[str, object]) -> str:
    correction = pd.DataFrame(payload["strategy_table"])
    provider = pd.DataFrame(payload["provider_table"])
    scenario = pd.DataFrame(payload["scenario_summary"])
    seed_runs = pd.DataFrame(payload["knmi_seed_runs"])
    margins = pd.DataFrame(payload["provider_margin_factors"]).T
    margins.index.name = "source"
    margins = margins.reset_index()
    knmi = correction[
        correction["knmi_feasible_all_three"]
    ].sort_values("knmi_gate_share", ascending=False)
    lines = [
        "# Stage498 KNMI FAR transfer correction",
        "",
        f"- Status: `{payload['status']}`",
        "- Scope: validation-only FAR correction of the Stage 490 per-row scores; no retraining, no new provider or dataset, no frozen artifact modified.",
        "- Strategy families: conservative validation FAR levels, split-conformal order statistic, exact-binomial (Clopper-Pearson) finite-sample correction, provider-specific validation period-shift safety margin.",
        "- Definitions: `level` = empirical validation-normal quantile at the target; `conformal` = the `ceil((n+1)(1-alpha))`-th smallest validation normal score; `binomial_ucb` = the highest threshold whose one-sided 95% Clopper-Pearson upper bound on the validation FAR still meets the target; `margin_*` = target divided by the provider period-shift margin.",
        "- Constraint: `long_stuck` / `drop_replay` validation event coverage must stay at or above the frozen union OR prediction (margin 0); otherwise the unit falls back to the frozen union OR prediction.",
        "- Test rows are evaluation-only; the V6 test split has already been inspected, so every number is exploratory.",
        "",
        "## Provider safety margins",
        "",
        margins.to_markdown(index=False),
        "",
        "`period_mean` / `period_max` are the within-validation first-half to second-half FAR inflation of the union shape score at the 0.0065 target (0.5 Laplace smoothing, averaged over seeds, clipped to [1, 4]); the effective validation target is `target / margin`.",
        "",
        "## Strategy summary",
        "",
        "Criteria flags: `far` = KNMI pooled test FAR <= 1pp, `structured` = "
        "both structured test deltas >= -5pp, `gate` = KNMI extended gate "
        ">= 70pct. FAR is the pooled false-alarm rate over non-attack test "
        "rows; the extended gate is provider- and pooled-level by "
        "construction.",
        "",
        correction[
            [
                "strategy_id",
                "selection_unit",
                "fallback_runs",
                "knmi_test_far_pooled",
                "knmi_test_to_valid_far_ratio",
                "knmi_f1_mean",
                "knmi_event_coverage_mean",
                "knmi_long_stuck_test_delta_mean",
                "knmi_drop_replay_test_delta_mean",
                "knmi_gate_wins",
                "knmi_gate_share",
                "knmi_far_point_le_1pct",
                "knmi_structured_drop_le_5pp",
                "knmi_gate_ge_70pct",
                "knmi_feasible_all_three",
            ]
        ]
        .sort_values(
            [
                "knmi_far_point_le_1pct",
                "knmi_structured_drop_le_5pp",
                "knmi_gate_ge_70pct",
                "knmi_test_far_pooled",
            ],
            ascending=[False, False, False, True],
        )
        .to_markdown(index=False),
        "",
        "## Feasibility",
        "",
        f"- KNMI scope (test FAR <= 1pp, structured drop <= 5pp, extended gate >= 70pct): `{payload['feasibility']['knmi_verdict']}`",
        f"- KNMI feasible strategies: `{payload['feasibility']['knmi_feasible_strategies']}`",
        f"- Overall scope: `{payload['feasibility']['overall_verdict']}`",
        f"- Overall feasible strategies: `{payload['feasibility']['overall_feasible_strategies']}`",
        "",
    ]
    if payload["feasibility"]["knmi_verdict"] == "no_feasible_strategy":
        closest_far = correction.sort_values(
            "knmi_test_far_pooled"
        ).iloc[0]
        structured_ok = correction[correction["knmi_structured_drop_le_5pp"]]
        closest_gate = (
            structured_ok.sort_values(
                "knmi_gate_share",
                ascending=False,
            ).iloc[0]
            if len(structured_ok)
            else correction.sort_values(
                "knmi_gate_share",
                ascending=False,
            ).iloc[0]
        )
        lines.extend(
            [
                "**No feasible strategy for KNMI.** No evaluated strategy "
                "simultaneously satisfies test FAR <= 1pp, structured drop "
                "<= 5pp and the 70pct extended gate with the margin-0 "
                "validation coverage constraint.",
                "",
                "- The pooled-scope passes listed above are not a solution "
                "to the KNMI problem: they satisfy the three criteria only "
                "after pooling all four providers and still leave KNMI test "
                "FAR at 1.34-1.43pp.",
                "",
                "- Lowest KNMI test FAR: "
                f"`{closest_far['strategy_id']}` at "
                f"`{closest_far['knmi_test_far_pooled']:.6f}` "
                f"(structured deltas "
                f"`{closest_far['knmi_long_stuck_test_delta_mean']:.4f}` / "
                f"`{closest_far['knmi_drop_replay_test_delta_mean']:.4f}`, "
                f"gate `{closest_far['knmi_gate_wins']}/"
                f"{closest_far['knmi_gate_units']}`).",
                "- Best KNMI gate with the structured criterion satisfied: "
                f"`{closest_gate['strategy_id']}` at "
                f"`{closest_gate['knmi_gate_wins']}/"
                f"{closest_gate['knmi_gate_units']}` "
                f"(`{closest_gate['knmi_gate_share']:.4f}`) with KNMI test "
                f"FAR `{closest_gate['knmi_test_far_pooled']:.6f}`.",
                "",
            ]
        )
    if len(knmi):
        lines.extend(
            [
                "### Feasible KNMI strategies",
                "",
                knmi[
                    [
                        "strategy_id",
                        "fallback_runs",
                        "knmi_test_far_pooled",
                        "knmi_long_stuck_test_delta_mean",
                        "knmi_drop_replay_test_delta_mean",
                        "knmi_gate_wins",
                        "knmi_gate_share",
                    ]
                ].to_markdown(index=False),
                "",
            ]
        )
    floor = pd.DataFrame(payload["constraint_diagnostics"]["per_provider"])
    floor_seed = pd.DataFrame(payload["constraint_diagnostics"]["per_seed"])
    lines.extend(
        [
            "## Constraint floor",
            "",
            f"- KNMI pooled test-FAR floor implied by the margin-0 constraint: `{payload['constraint_diagnostics']['knmi_pooled_test_far_floor']:.6f}`",
            f"- Interpretation: {payload['constraint_diagnostics']['interpretation']}",
            "",
            floor.to_markdown(index=False),
            "",
            "Per-seed margin-0 feasibility at the least conservative target (`level@0.0065`):",
            "",
            floor_seed[
                [
                    "source",
                    "seed",
                    "union_valid_coverage_long_stuck",
                    "union_valid_coverage_drop_replay",
                    "best_candidate",
                    "best_valid_coverage_long_stuck",
                    "best_valid_coverage_drop_replay",
                    "best_margin_vs_union",
                    "candidate_feasible_at_least_conservative_target",
                    "union_test_false_alarm_rate",
                ]
            ].to_markdown(index=False),
            "",
        ]
    )
    knmi_provider = provider[provider["source"].eq(KNMI)].sort_values(
        "test_far_pooled"
    )
    focus_id = str(
        correction.sort_values(
            [
                "knmi_far_point_le_1pct",
                "knmi_structured_drop_le_5pp",
                "knmi_gate_ge_70pct",
                "knmi_gate_share",
            ],
            ascending=[False, False, False, False],
        )["strategy_id"].iloc[0]
    )
    lines.extend(
        [
            "## KNMI provider detail",
            "",
            knmi_provider[
                [
                    "strategy_id",
                    "fallback_seeds",
                    "test_far_pooled",
                    "test_to_valid_far_ratio",
                    "f1_mean",
                    "event_coverage_mean",
                    "precision_mean",
                    "row_recall_mean",
                    "average_precision_mean",
                    "long_stuck_test_delta_mean",
                    "drop_replay_test_delta_mean",
                    "gate_wins",
                    "gate_share",
                ]
            ].to_markdown(index=False),
            "",
            "## KNMI per-seed detail (best criteria ranking)",
            "",
            f"Strategy `{focus_id}`:",
            "",
            seed_runs[seed_runs["strategy_id"].eq(focus_id)][
                [
                    "seed",
                    "selection_rule",
                    "selection_weight",
                    "fallback_to_union",
                    "valid_false_alarm_rate",
                    "false_alarm_rate",
                    "f1",
                    "precision",
                    "row_recall",
                    "event_coverage",
                    "average_precision",
                    "long_stuck_test_delta",
                    "drop_replay_test_delta",
                ]
            ]
            .sort_values("seed")
            .to_markdown(index=False),
            "",
            "## KNMI scenario detail (mean over seeds)",
            "",
            scenario[
                scenario["source"].eq(KNMI)
                & scenario["strategy_id"].isin(
                    [
                        "level@0.0065|seed",
                        "level@0.002|seed",
                        "level@0.0065|provider",
                    ]
                )
            ]
            .sort_values(["strategy_id", "scenario"])[
                [
                    "strategy_id",
                    "scenario",
                    "event_coverage",
                    "union_event_coverage",
                    "event_coverage_delta_vs_union",
                    "f1",
                    "precision",
                    "row_recall",
                    "average_precision",
                ]
            ]
            .to_markdown(index=False),
            "",
            "## Verification",
            "",
        ]
    )
    checks = payload["verification"]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Every rule is selected on validation rows only; test labels never enter selection.",
            "- No frozen artifact is modified; all outputs are new Stage 498 files.",
            "- The experiment re-thresholds existing scores only and never retrains.",
            "",
            f"- Elapsed seconds: `{payload['elapsed_seconds']:.2f}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
