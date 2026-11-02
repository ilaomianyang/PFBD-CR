"""Stage 494: exploratory provider-heterogeneity audit of the extended gate.

Read-only audit over existing Stage491 constrained fusion, frozen V6 union
(trust_shape_union) and formal lagged-MLP per provider/seed/scenario results.
It explains why ECCC, KNMI, MeteoSwiss and NOAA ISD each miss the 70%
extended scenario-by-baseline gate, and why NOAA ISD is clearly worse, by
decomposing the shortfall into gate definition (ties), threshold/FAR transfer,
attack type, event granularity, station/observation distribution, reference
strength and provider-specific detection difficulty.

No new providers, no downloads, no changes to frozen predictions.  The V6
test split has already been inspected, so every number here is exploratory
and none of it can be promoted to a confirmatory claim.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

EXISTING_SCENARIO = TABLES / "final_v6_scenario_metrics.csv"
FORMAL_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"
FUSION_SCENARIO = TABLES / "stage491_constrained_fusion_scenario.csv"
FUSION_RUNS = TABLES / "stage491_constrained_fusion_runs.csv"
FUSION_SELECTION = TABLES / "stage491_constrained_fusion_selection.csv"
FUSION_STATION = TABLES / "stage491_constrained_fusion_station.csv"
V6_DETAIL = TABLES / "final_v6_detail.csv"
FORMAL_RUNS = TABLES / "stage486_formal_lagged_mlp_runs.csv"

OUT_DETAIL = TABLES / "stage494_provider_heterogeneity_detail.csv"
OUT_SUMMARY = TABLES / "stage494_provider_heterogeneity_summary.csv"
OUT_JSON = REPORTS / "stage494_provider_heterogeneity.json"
OUT_MD = REPORTS / "stage494_provider_heterogeneity.md"

CANDIDATE = "constrained_validation_fusion"
UNION = "trust_shape_union"
FORMAL = "formal_lagged_mlp"

BASELINES_11 = [
    "robust_max_deviation",
    "isolation_forest",
    "one_class_svm",
    "pca_reconstruction",
    "mlp_autoencoder_reconstruction",
    "knn_distance",
    "ewma_deviation",
    "arima_innovation",
    "gru_predictor",
    "lightweight_autoencoder",
    "ics_cps_spc_proxy",
]
REFERENCES = BASELINES_11 + [FORMAL]

SCENARIOS = [
    "coordinated_offset",
    "delay",
    "drop_replay",
    "long_stuck",
    "low_noise",
    "low_offset",
    "low_spike",
    "short_lag",
    "slow_drift",
]
STRUCTURED = ["drop_replay", "long_stuck"]
COORDINATED = "coordinated_offset"
SUBTLE = [s for s in SCENARIOS if s not in STRUCTURED and s != COORDINATED]

FAR_TARGET = 0.0065
PASS_SHARE = 0.70
SCENARIO_METRIC_COLUMNS = [
    "rows",
    "attack_rows",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "row_recall",
    "f1",
    "false_alarm_rate",
    "roc_auc",
    "average_precision",
    "events",
    "detected_events",
    "event_coverage",
    "mean_event_length",
    "min_event_length",
    "max_event_length",
]


def outcome(candidate: float, reference: float) -> str:
    if candidate > reference:
        return "win"
    if candidate < reference:
        return "loss"
    return "tie"


def scenario_class(scenario: str) -> str:
    if scenario in STRUCTURED:
        return "structured_replay_stuck"
    if scenario == COORDINATED:
        return "coordinated_offset"
    return "subtle_other"


def load_long_frame() -> pd.DataFrame:
    existing = pd.read_csv(EXISTING_SCENARIO)
    existing = existing[
        existing["method"].isin(BASELINES_11 + [UNION])
    ].copy()
    existing["method_family"] = np.where(
        existing["method"].eq(UNION), "frozen_v6_union", "frozen_baseline"
    )
    formal = pd.read_csv(FORMAL_SCENARIO).copy()
    formal["method_family"] = "formal_lagged_mlp"
    fusion = pd.read_csv(FUSION_SCENARIO).copy()
    fusion["method_family"] = "stage491_constrained_fusion"
    columns = [
        "source",
        "seed",
        "scenario",
        "method",
        "method_family",
        *SCENARIO_METRIC_COLUMNS,
    ]
    long = pd.concat(
        [existing[columns], formal[columns], fusion[columns]],
        ignore_index=True,
    )
    long["seed"] = long["seed"].astype(int)
    long["source"] = long["source"].astype(str)
    long["scenario"] = long["scenario"].astype(str)
    return long


def provider_scenario_aggregate(long: pd.DataFrame) -> pd.DataFrame:
    """Seed-level mean metrics per provider, scenario and method."""
    return long.groupby(
        ["source", "scenario", "method"], as_index=False
    ).agg(
        coverage_mean=("event_coverage", "mean"),
        coverage_sd=("event_coverage", "std"),
        f1_mean=("f1", "mean"),
        far_mean=("false_alarm_rate", "mean"),
        events_mean=("events", "mean"),
        detected_events_mean=("detected_events", "mean"),
        detected_events_sum=("detected_events", "sum"),
    )


def provider_scenario_gate(long: pd.DataFrame) -> pd.DataFrame:
    """Provider-level gate, reproducing the frozen Stage493 float path.

    Stage493 builds a wide seed table and compares ``group[method].mean()``
    values, which is floating-point order sensitive for exact ties.  The same
    path is used here so the counts match the frozen artifacts, and an
    exact-integer comparison (summed detected events over seeds) is added as
    a reconciliation check.
    """
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
    grouped = provider_scenario_aggregate(long)
    rows: list[dict[str, object]] = []
    for (source, scenario), group in wide.groupby(
        ["source", "scenario"], sort=True
    ):
        candidate_coverage = float(group[CANDIDATE].mean())
        candidate_detected = float(
            grouped[
                grouped["source"].eq(source)
                & grouped["scenario"].eq(scenario)
                & grouped["method"].eq(CANDIDATE)
            ]["detected_events_sum"].iloc[0]
        )
        candidate_metrics = grouped[
            grouped["source"].eq(source)
            & grouped["scenario"].eq(scenario)
            & grouped["method"].eq(CANDIDATE)
        ].iloc[0]
        for reference in REFERENCES:
            reference_metrics = grouped[
                grouped["source"].eq(source)
                & grouped["scenario"].eq(scenario)
                & grouped["method"].eq(reference)
            ].iloc[0]
            reference_coverage = float(group[reference].mean())
            reference_detected = float(reference_metrics["detected_events_sum"])
            float_outcome = outcome(
                candidate_coverage, reference_coverage
            )
            exact_outcome = outcome(candidate_detected, reference_detected)
            rows.append(
                {
                    "source": source,
                    "scenario": scenario,
                    "method": reference,
                    "coverage_mean": reference_coverage,
                    "coverage_sd": float(reference_metrics["coverage_sd"]),
                    "f1_mean": float(reference_metrics["f1_mean"]),
                    "far_mean": float(reference_metrics["far_mean"]),
                    "events_mean": float(reference_metrics["events_mean"]),
                    "detected_events_mean": float(
                        reference_metrics["detected_events_mean"]
                    ),
                    "detected_events_sum": reference_detected,
                    "candidate_coverage": candidate_coverage,
                    "candidate_f1": float(candidate_metrics["f1_mean"]),
                    "candidate_far": float(candidate_metrics["far_mean"]),
                    "candidate_detected_events_sum": candidate_detected,
                    "coverage_gap": reference_coverage - candidate_coverage,
                    "coverage_gap_events": (
                        reference_coverage - candidate_coverage
                    )
                    * float(reference_metrics["events_mean"]),
                    "gate_outcome": float_outcome,
                    "gate_outcome_exact": exact_outcome,
                    "float_exact_agree": float_outcome == exact_outcome,
                    "ref_far_le_candidate": (
                        float(reference_metrics["far_mean"])
                        <= float(candidate_metrics["far_mean"])
                    ),
                    "candidate_f1_better": (
                        float(candidate_metrics["f1_mean"])
                        > float(reference_metrics["f1_mean"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def detail_table(
    long: pd.DataFrame,
    gate: pd.DataFrame,
) -> pd.DataFrame:
    seed_candidate = long[long["method"].eq(CANDIDATE)][
        [
            "source",
            "seed",
            "scenario",
            "event_coverage",
            "f1",
            "false_alarm_rate",
        ]
    ].rename(
        columns={
            "event_coverage": "candidate_coverage_seed",
            "f1": "candidate_f1_seed",
            "false_alarm_rate": "candidate_far_seed",
        }
    )
    detail = long.merge(
        seed_candidate,
        on=["source", "seed", "scenario"],
        validate="many_to_one",
    )
    detail["coverage_gap_seed"] = (
        detail["event_coverage"] - detail["candidate_coverage_seed"]
    )
    detail["seed_outcome"] = [
        (
            outcome(candidate_value, reference_value)
            if method in REFERENCES
            else ""
        )
        for candidate_value, reference_value, method in zip(
            detail["candidate_coverage_seed"],
            detail["event_coverage"],
            detail["method"],
        )
    ]
    detail["is_gate_reference"] = detail["method"].isin(REFERENCES).astype(int)
    aggregate = provider_scenario_aggregate(long).rename(
        columns={
            "coverage_mean": "provider_mean_event_coverage",
            "coverage_sd": "provider_coverage_sd",
            "f1_mean": "provider_mean_f1",
            "far_mean": "provider_mean_far",
            "events_mean": "provider_mean_events",
            "detected_events_mean": "provider_mean_detected_events",
            "detected_events_sum": "provider_detected_events_sum",
        }
    )
    detail = detail.merge(
        aggregate,
        on=["source", "scenario", "method"],
        how="left",
        validate="many_to_one",
    )
    gate_columns = gate[
        [
            "source",
            "scenario",
            "method",
            "candidate_coverage",
            "coverage_mean",
            "coverage_gap",
            "coverage_gap_events",
            "gate_outcome",
            "gate_outcome_exact",
            "float_exact_agree",
            "detected_events_sum",
            "candidate_detected_events_sum",
            "far_mean",
            "ref_far_le_candidate",
            "candidate_f1_better",
            "candidate_far",
            "candidate_f1",
        ]
    ].rename(
        columns={
            "candidate_coverage": "provider_mean_candidate_coverage",
            "coverage_mean": "provider_mean_reference_coverage",
            "coverage_gap": "provider_mean_coverage_gap",
            "coverage_gap_events": "provider_mean_coverage_gap_events",
            "gate_outcome": "provider_gate_outcome",
            "gate_outcome_exact": "provider_gate_outcome_exact",
            "float_exact_agree": "provider_float_exact_agree",
            "detected_events_sum": "provider_reference_detected_events_sum",
            "candidate_detected_events_sum": (
                "provider_candidate_detected_events_sum"
            ),
            "far_mean": "provider_mean_reference_far",
            "ref_far_le_candidate": "provider_reference_far_le_candidate",
            "candidate_f1_better": "provider_candidate_f1_better",
            "candidate_far": "provider_mean_candidate_far",
            "candidate_f1": "provider_mean_candidate_f1",
        }
    )
    detail = detail.merge(
        gate_columns,
        on=["source", "scenario", "method"],
        how="left",
        validate="many_to_one",
    )
    detail["scenario_class"] = detail["scenario"].map(scenario_class)
    ordered = [
        "source",
        "seed",
        "scenario",
        "scenario_class",
        "method",
        "method_family",
        "is_gate_reference",
        *SCENARIO_METRIC_COLUMNS,
        "candidate_coverage_seed",
        "candidate_f1_seed",
        "candidate_far_seed",
        "coverage_gap_seed",
        "seed_outcome",
        "provider_mean_event_coverage",
        "provider_coverage_sd",
        "provider_mean_f1",
        "provider_mean_far",
        "provider_mean_events",
        "provider_mean_detected_events",
        "provider_detected_events_sum",
        "provider_mean_candidate_coverage",
        "provider_mean_reference_coverage",
        "provider_mean_coverage_gap",
        "provider_mean_coverage_gap_events",
        "provider_gate_outcome",
        "provider_gate_outcome_exact",
        "provider_float_exact_agree",
        "provider_reference_detected_events_sum",
        "provider_candidate_detected_events_sum",
        "provider_mean_reference_far",
        "provider_reference_far_le_candidate",
        "provider_candidate_f1_better",
        "provider_mean_candidate_far",
        "provider_mean_candidate_f1",
    ]
    return detail[ordered].sort_values(
        ["source", "scenario", "method", "seed"], ignore_index=True
    )


def counterfactual_gates(gate: pd.DataFrame) -> pd.DataFrame:
    variants = {
        "strict_all_units": lambda frame: frame,
        "ties_counted_as_half": lambda frame: frame,
        "ties_excluded_from_denominator": lambda frame: frame,
        "drop_coordinated_offset": lambda frame: frame[
            ~frame["scenario"].eq(COORDINATED)
        ],
        "drop_structured_scenarios": lambda frame: frame[
            ~frame["scenario"].isin(STRUCTURED)
        ],
        "drop_two_killer_references": lambda frame: frame[
            ~frame["method"].isin(["arima_innovation", FORMAL])
        ],
        "only_references_at_or_below_candidate_far": lambda frame: frame[
            frame["ref_far_le_candidate"]
        ],
    }
    rows: list[dict[str, object]] = []
    for provider, provider_frame in gate.groupby("source", sort=True):
        for variant, selector in variants.items():
            frame = selector(provider_frame)
            wins = int(frame["gate_outcome"].eq("win").sum())
            ties = int(frame["gate_outcome"].eq("tie").sum())
            losses = int(frame["gate_outcome"].eq("loss").sum())
            units = int(len(frame))
            if variant == "ties_counted_as_half":
                share = (wins + 0.5 * ties) / units if units else 0.0
            elif variant == "ties_excluded_from_denominator":
                share = wins / (wins + losses) if wins + losses else 0.0
            else:
                share = wins / units if units else 0.0
            rows.append(
                {
                    "provider": provider,
                    "variant": variant,
                    "units": units,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "share": float(share),
                    "passes_70pct": bool(share >= PASS_SHARE),
                }
            )
    return pd.DataFrame(rows)


def provider_distribution(
    long: pd.DataFrame,
    runs: pd.DataFrame,
    station: pd.DataFrame,
) -> pd.DataFrame:
    stations = (
        station.groupby(["source", "seed"], as_index=False)
        .agg(
            stations=("station_id", "nunique"),
            true_negatives=("tn", "sum"),
            false_positives=("fp", "sum"),
            false_negatives=("fn", "sum"),
            true_positives=("tp", "sum"),
        )
    )
    stations["observed_normal_rows"] = (
        stations["true_negatives"] + stations["false_positives"]
    )
    stations["observed_attack_rows"] = (
        stations["false_negatives"] + stations["true_positives"]
    )
    stations["observed_rows"] = (
        stations["observed_normal_rows"]
        + stations["observed_attack_rows"]
    )
    merged = stations.merge(
        runs[["source", "seed", "rows", "attack_rows"]],
        on=["source", "seed"],
        validate="one_to_one",
    )
    merged["observed_share"] = merged["observed_rows"] / merged["rows"]
    merged["observed_normal_per_station"] = (
        merged["observed_normal_rows"] / merged["stations"]
    )
    merged["observed_attack_per_station"] = (
        merged["observed_attack_rows"] / merged["stations"]
    )
    events = (
        long[long["method"].eq(CANDIDATE)]
        .groupby(["source", "seed"], as_index=False)["events"]
        .sum()
        .rename(columns={"events": "events_total"})
    )
    merged = merged.merge(
        events, on=["source", "seed"], validate="one_to_one"
    )
    auc = (
        long[long["method"].isin([UNION, FORMAL])]
        .groupby(["source", "seed", "method"], as_index=False)["roc_auc"]
        .mean()
        .pivot_table(
            index=["source", "seed"],
            columns="method",
            values="roc_auc",
        )
        .reset_index()
        .rename(
            columns={
                UNION: "union_roc_auc",
                FORMAL: "formal_roc_auc",
            }
        )
    )
    merged = merged.merge(auc, on=["source", "seed"], validate="one_to_one")
    return merged


def build_summary(
    long: pd.DataFrame,
    gate: pd.DataFrame,
    detail: pd.DataFrame,
    runs: pd.DataFrame,
    selection: pd.DataFrame,
    distribution: pd.DataFrame,
    counterfactuals: pd.DataFrame,
) -> pd.DataFrame:
    v6_detail = pd.read_csv(V6_DETAIL)
    union_runs = v6_detail[v6_detail["method"].eq(UNION)]
    formal_runs = pd.read_csv(FORMAL_RUNS)
    rows: list[dict[str, object]] = []
    for provider in sorted(long["source"].unique()):
        provider_gate = gate[gate["source"].eq(provider)]
        provider_runs = runs[runs["source"].eq(provider)]
        provider_selection = selection[selection["source"].eq(provider)]
        provider_long = long[long["source"].eq(provider)]
        provider_dist = distribution[distribution["source"].eq(provider)]
        scenario_candidate = (
            provider_long[provider_long["method"].eq(CANDIDATE)]
            .groupby("scenario", as_index=False)["event_coverage"]
            .mean()
            .set_index("scenario")["event_coverage"]
        )
        provider_cf = counterfactuals[
            counterfactuals["provider"].eq(provider)
        ].set_index("variant")
        wins = int(provider_gate["gate_outcome"].eq("win").sum())
        ties = int(provider_gate["gate_outcome"].eq("tie").sum())
        losses = int(provider_gate["gate_outcome"].eq("loss").sum())
        wins_exact = int(
            provider_gate["gate_outcome_exact"].eq("win").sum()
        )
        ties_exact = int(
            provider_gate["gate_outcome_exact"].eq("tie").sum()
        )
        losses_exact = int(
            provider_gate["gate_outcome_exact"].eq("loss").sum()
        )
        units = int(len(provider_gate))
        wins_needed = int(np.ceil(PASS_SHARE * units))
        loss_units = provider_gate[provider_gate["gate_outcome"].eq("loss")]
        sensitive = provider_gate[~provider_gate["float_exact_agree"]]
        row: dict[str, object] = {
            "provider": provider,
            "seeds": int(provider_runs["seed"].nunique()),
            "stations": int(provider_dist["stations"].mean()),
            "events_total_mean": float(provider_dist["events_total"].mean()),
            "test_rows_mean": float(provider_dist["rows"].mean()),
            "observed_normal_rows_per_station": float(
                provider_dist["observed_normal_per_station"].mean()
            ),
            "observed_attack_rows_per_station": float(
                provider_dist["observed_attack_per_station"].mean()
            ),
            "observed_share": float(provider_dist["observed_share"].mean()),
            "union_roc_auc_mean": float(
                provider_dist["union_roc_auc"].mean()
            ),
            "formal_roc_auc_mean": float(
                provider_dist["formal_roc_auc"].mean()
            ),
            "candidate_run_f1_mean": float(provider_runs["f1"].mean()),
            "candidate_run_event_coverage_mean": float(
                provider_runs["event_coverage"].mean()
            ),
            "candidate_run_far_mean": float(
                provider_runs["false_alarm_rate"].mean()
            ),
            "candidate_run_ap_mean": float(
                provider_runs["average_precision"].mean()
            ),
            "candidate_scenario_mean_coverage": float(
                scenario_candidate.mean()
            ),
            "candidate_structured_mean_coverage": float(
                scenario_candidate[STRUCTURED].mean()
            ),
            "candidate_subtle_mean_coverage": float(
                scenario_candidate[
                    [s for s in scenario_candidate.index if s not in STRUCTURED]
                ].mean()
            ),
            "union_run_f1_mean": float(
                union_runs[union_runs["source"].eq(provider)]["f1"].mean()
            ),
            "union_run_event_coverage_mean": float(
                union_runs[union_runs["source"].eq(provider)][
                    "event_coverage"
                ].mean()
            ),
            "union_run_far_mean": float(
                union_runs[union_runs["source"].eq(provider)][
                    "false_alarm_rate"
                ].mean()
            ),
            "formal_run_f1_mean": float(
                formal_runs[formal_runs["source"].eq(provider)]["f1"].mean()
            ),
            "formal_run_event_coverage_mean": float(
                formal_runs[formal_runs["source"].eq(provider)][
                    "event_coverage"
                ].mean()
            ),
            "formal_run_far_mean": float(
                formal_runs[formal_runs["source"].eq(provider)][
                    "false_alarm_rate"
                ].mean()
            ),
            "extended_wins": wins,
            "extended_ties": ties,
            "extended_losses": losses,
            "extended_units": units,
            "extended_share": wins / units if units else 0.0,
            "extended_share_ties_half": (wins + 0.5 * ties) / units
            if units
            else 0.0,
            "extended_share_excl_ties": wins / (wins + losses)
            if wins + losses
            else 0.0,
            "passes_70pct": bool(wins / units >= PASS_SHARE if units else False),
            "passes_70pct_ties_half": bool(
                (wins + 0.5 * ties) / units >= PASS_SHARE if units else False
            ),
            "passes_70pct_excl_ties": bool(
                wins / (wins + losses) >= PASS_SHARE
                if wins + losses
                else False
            ),
            "wins_needed_70pct": wins_needed,
            "wins_shortfall": max(0, wins_needed - wins),
            "extended_wins_exact": wins_exact,
            "extended_ties_exact": ties_exact,
            "extended_losses_exact": losses_exact,
            "extended_share_exact": wins_exact / units if units else 0.0,
            "passes_70pct_exact": bool(
                wins_exact / units >= PASS_SHARE if units else False
            ),
            "float_order_sensitive_units": int(len(sensitive)),
            "float_order_sensitive_detail": "; ".join(
                f"{part.scenario}:{part.method}"
                for part in sensitive.itertuples()
            ),
            "fusion_seeds_weight_lt1": int(
                provider_selection["weight"].lt(1.0).sum()
            ),
            "fusion_seeds_union_fallback": int(
                provider_selection["fallback_to_union"].sum()
            ),
            "fusion_mean_weight": float(provider_selection["weight"].mean()),
            "fusion_mean_feasible_candidates": float(
                provider_selection["feasible_candidate_count"].mean()
            ),
            "validation_mean_scenario_coverage": float(
                provider_selection["valid_mean_scenario_coverage"].mean()
            ),
            "test_mean_scenario_coverage": float(scenario_candidate.mean()),
            "candidate_test_far_over_target": float(
                provider_runs["false_alarm_rate"].mean() / FAR_TARGET
            ),
            "union_test_far_over_target": float(
                union_runs[union_runs["source"].eq(provider)][
                    "false_alarm_rate"
                ].mean()
                / FAR_TARGET
            ),
            "losses_within_one_event": int(
                loss_units["coverage_gap_events"].le(1.0).sum()
            ),
            "losses_mean_gap_events": float(
                loss_units["coverage_gap_events"].mean()
            )
            if len(loss_units)
            else 0.0,
            "losses_reference_higher_far": int(
                (~loss_units["ref_far_le_candidate"]).sum()
            ),
            "losses_reference_far_le_candidate": int(
                loss_units["ref_far_le_candidate"].sum()
            ),
            "losses_candidate_f1_better": int(
                loss_units["candidate_f1_better"].sum()
            ),
        }
        for scenario in SCENARIOS:
            scenario_frame = provider_gate[provider_gate["scenario"].eq(scenario)]
            row[f"wins_{scenario}"] = int(
                scenario_frame["gate_outcome"].eq("win").sum()
            )
            row[f"ties_{scenario}"] = int(
                scenario_frame["gate_outcome"].eq("tie").sum()
            )
            row[f"losses_{scenario}"] = int(
                scenario_frame["gate_outcome"].eq("loss").sum()
            )
        for reference in REFERENCES:
            reference_frame = provider_gate[
                provider_gate["method"].eq(reference)
            ]
            row[f"losses_vs_{reference}"] = int(
                reference_frame["gate_outcome"].eq("loss").sum()
            )
            row[f"ties_vs_{reference}"] = int(
                reference_frame["gate_outcome"].eq("tie").sum()
            )
        for variant in [
            "drop_coordinated_offset",
            "drop_structured_scenarios",
            "drop_two_killer_references",
            "only_references_at_or_below_candidate_far",
        ]:
            row[f"counterfactual_share_{variant}"] = float(
                provider_cf.loc[variant, "share"]
            )
            row[f"counterfactual_units_{variant}"] = int(
                provider_cf.loc[variant, "units"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def tie_inventory(gate: pd.DataFrame) -> dict[str, object]:
    ties = gate[gate["gate_outcome"].eq("tie")]
    return {
        "total_units": int(len(gate)),
        "total_ties": int(len(ties)),
        "tie_share": float(len(ties) / len(gate)) if len(gate) else 0.0,
        "ties_at_zero_coverage": int(ties["candidate_coverage"].eq(0.0).sum()),
        "ties_at_full_coverage": int(
            ties["candidate_coverage"].eq(1.0).sum()
        ),
        "ties_at_intermediate_coverage": int(
            (
                ties["candidate_coverage"].gt(0.0)
                & ties["candidate_coverage"].lt(1.0)
            ).sum()
        ),
        "ties_by_scenario": {
            str(key): int(value)
            for key, value in ties["scenario"].value_counts().items()
        },
        "ties_by_provider": {
            str(key): int(value)
            for key, value in ties["source"].value_counts().items()
        },
    }


def build_payload(
    long: pd.DataFrame,
    gate: pd.DataFrame,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    distribution: pd.DataFrame,
    selection: pd.DataFrame,
) -> dict[str, object]:
    loss_units = gate[gate["gate_outcome"].eq("loss")].copy()
    loss_records = [
        {
            "provider": str(row.source),
            "scenario": str(row.scenario),
            "reference": str(row.method),
            "candidate_coverage": float(row.candidate_coverage),
            "reference_coverage": float(row.coverage_mean),
            "coverage_gap": float(row.coverage_gap),
            "coverage_gap_events": float(row.coverage_gap_events),
            "candidate_far": float(row.candidate_far),
            "reference_far": float(row.far_mean),
            "reference_far_le_candidate": bool(row.ref_far_le_candidate),
            "candidate_f1": float(row.candidate_f1),
            "reference_f1": float(row.f1_mean),
            "within_one_event": bool(row.coverage_gap_events <= 1.0),
        }
        for row in loss_units.itertuples()
    ]
    per_scenario_rows: list[dict[str, object]] = []
    for (provider, scenario), group in gate.groupby(
        ["source", "scenario"], sort=True
    ):
        best = group.loc[group["coverage_mean"].idxmax()]
        per_scenario_rows.append(
            {
                "provider": str(provider),
                "scenario": str(scenario),
                "scenario_class": scenario_class(str(scenario)),
                "wins": int(group["gate_outcome"].eq("win").sum()),
                "ties": int(group["gate_outcome"].eq("tie").sum()),
                "losses": int(group["gate_outcome"].eq("loss").sum()),
                "units": int(len(group)),
                "candidate_coverage": float(best["candidate_coverage"]),
                "mean_reference_coverage": float(group["coverage_mean"].mean()),
                "best_reference": str(best["method"]),
                "best_reference_coverage": float(best["coverage_mean"]),
                "candidate_minus_best_reference": float(
                    best["candidate_coverage"] - best["coverage_mean"]
                ),
                "events": float(best["events_mean"]),
                "coverage_step_per_event": float(1.0 / best["events_mean"])
                if best["events_mean"]
                else 0.0,
            }
        )
    per_reference_rows: list[dict[str, object]] = []
    for reference, group in gate.groupby("method", sort=True):
        per_reference_rows.append(
            {
                "reference": str(reference),
                "wins": int(group["gate_outcome"].eq("win").sum()),
                "ties": int(group["gate_outcome"].eq("tie").sum()),
                "losses": int(group["gate_outcome"].eq("loss").sum()),
                "units": int(len(group)),
                "mean_coverage": float(group["coverage_mean"].mean()),
                "mean_f1": float(group["f1_mean"].mean()),
                "mean_far": float(group["far_mean"].mean()),
                "losses_by_provider": {
                    str(provider): int(
                        part["gate_outcome"].eq("loss").sum()
                    )
                    for provider, part in group.groupby("source", sort=True)
                },
            }
        )
    fusion_rows: list[dict[str, object]] = []
    for provider, group in selection.groupby("source", sort=True):
        fusion_rows.append(
            {
                "provider": str(provider),
                "seeds": int(len(group)),
                "seeds_weight_lt1": int(group["weight"].lt(1.0).sum()),
                "seeds_union_fallback": int(group["fallback_to_union"].sum()),
                "mean_weight": float(group["weight"].mean()),
                "mean_feasible_candidates": float(
                    group["feasible_candidate_count"].mean()
                ),
                "mean_validation_scenario_coverage": float(
                    group["valid_mean_scenario_coverage"].mean()
                ),
                "weights": [
                    {"seed": int(seed), "weight": float(weight)}
                    for seed, weight in zip(group["seed"], group["weight"])
                ],
            }
        )
    distribution_rows: list[dict[str, object]] = []
    for provider, group in distribution.groupby("source", sort=True):
        distribution_rows.append(
            {
                "provider": str(provider),
                "stations": int(group["stations"].mean()),
                "events_total_mean": float(group["events_total"].mean()),
                "observed_normal_rows_per_station": float(
                    group["observed_normal_per_station"].mean()
                ),
                "observed_attack_rows_per_station": float(
                    group["observed_attack_per_station"].mean()
                ),
                "observed_share": float(group["observed_share"].mean()),
                "union_roc_auc_mean": float(group["union_roc_auc"].mean()),
                "formal_roc_auc_mean": float(group["formal_roc_auc"].mean()),
            }
        )
    seed_level_rows: list[dict[str, object]] = []
    reference_detail = detail[detail["is_gate_reference"].eq(1)]
    for provider, group in reference_detail.groupby("source", sort=True):
        per_seed_share = group.groupby("seed")["seed_outcome"].apply(
            lambda values: float(values.eq("win").mean())
        )
    attribution_rows: list[dict[str, object]] = []
    for row in summary.itertuples():
        provider_gate = gate[gate["source"].eq(row.provider)]
        loss_units = provider_gate[provider_gate["gate_outcome"].eq("loss")]
        attribution_rows.append(
            {
                "provider": row.provider,
                "extended_share": float(row.extended_share),
                "wins_shortfall": int(row.wins_shortfall),
                "ties": int(row.extended_ties),
                "share_excl_ties": float(row.extended_share_excl_ties),
                "losses": int(row.extended_losses),
                "losses_within_one_event": int(
                    row.losses_within_one_event
                ),
                "losses_to_higher_far_references": int(
                    row.losses_reference_higher_far
                ),
                "losses_to_more_conservative_references": int(
                    row.losses_reference_far_le_candidate
                ),
                "test_far_over_target": float(
                    row.candidate_test_far_over_target
                ),
                "top_loss_scenarios": [
                    {"scenario": str(scenario), "losses": int(count)}
                    for scenario, count in loss_units["scenario"]
                    .value_counts()
                    .head(3)
                    .items()
                ],
                "top_loss_references": [
                    {"reference": str(reference), "losses": int(count)}
                    for reference, count in loss_units["method"]
                    .value_counts()
                    .head(3)
                    .items()
                ],
                "stations": int(row.stations),
                "events_total": float(row.events_total_mean),
                "observed_normal_rows_per_station": float(
                    row.observed_normal_rows_per_station
                ),
                "union_roc_auc": float(row.union_roc_auc_mean),
                "candidate_structured_coverage": float(
                    row.candidate_structured_mean_coverage
                ),
                "candidate_subtle_coverage": float(
                    row.candidate_subtle_mean_coverage
                ),
                "fusion_mean_weight": float(row.fusion_mean_weight),
                "fusion_seeds_weight_lt1": int(
                    row.fusion_seeds_weight_lt1
                ),
                "fusion_mean_feasible_candidates": float(
                    row.fusion_mean_feasible_candidates
                ),
            }
        )
        seed_level_rows.append(
            {
                "provider": str(provider),
                "seed_units": int(len(group)),
                "wins": int(group["seed_outcome"].eq("win").sum()),
                "ties": int(group["seed_outcome"].eq("tie").sum()),
                "losses": int(group["seed_outcome"].eq("loss").sum()),
                "pooled_share": float(
                    group["seed_outcome"].eq("win").mean()
                ),
                "min_seed_share": float(per_seed_share.min()),
                "max_seed_share": float(per_seed_share.max()),
                "mean_seed_share": float(per_seed_share.mean()),
            }
        )
    provider_gate_records = summary[
        [
            "provider",
            "seeds",
            "stations",
            "events_total_mean",
            "extended_wins",
            "extended_ties",
            "extended_losses",
            "extended_units",
            "extended_share",
            "extended_share_ties_half",
            "extended_share_excl_ties",
            "passes_70pct",
            "passes_70pct_ties_half",
            "passes_70pct_excl_ties",
            "extended_wins_exact",
            "extended_ties_exact",
            "extended_losses_exact",
            "extended_share_exact",
            "passes_70pct_exact",
            "float_order_sensitive_units",
            "float_order_sensitive_detail",
            "wins_needed_70pct",
            "wins_shortfall",
            "candidate_run_f1_mean",
            "candidate_run_event_coverage_mean",
            "candidate_run_far_mean",
            "candidate_run_ap_mean",
            "union_run_event_coverage_mean",
            "union_run_far_mean",
            "formal_run_event_coverage_mean",
            "formal_run_far_mean",
            "candidate_structured_mean_coverage",
            "candidate_subtle_mean_coverage",
            "candidate_test_far_over_target",
            "union_test_far_over_target",
            "fusion_seeds_weight_lt1",
            "fusion_seeds_union_fallback",
            "fusion_mean_weight",
            "fusion_mean_feasible_candidates",
            "losses_within_one_event",
            "losses_mean_gap_events",
            "losses_reference_higher_far",
            "losses_reference_far_le_candidate",
            "union_roc_auc_mean",
            "formal_roc_auc_mean",
        ]
    ].to_dict("records")
    payload: dict[str, object] = {
        "status": "PASS_EXPLORATORY_PROVIDER_HETEROGENEITY_AUDIT",
        "exploratory_only": True,
        "confirmatory_claims": False,
        "date": "2026-10-09",
        "inputs": {
            "stage491_constrained_fusion_scenario": str(
                FUSION_SCENARIO.relative_to(ROOT)
            ),
            "stage491_constrained_fusion_runs": str(
                FUSION_RUNS.relative_to(ROOT)
            ),
            "stage491_constrained_fusion_selection": str(
                FUSION_SELECTION.relative_to(ROOT)
            ),
            "stage491_constrained_fusion_station": str(
                FUSION_STATION.relative_to(ROOT)
            ),
            "final_v6_scenario_metrics": str(
                EXISTING_SCENARIO.relative_to(ROOT)
            ),
            "final_v6_detail": str(V6_DETAIL.relative_to(ROOT)),
            "stage486_formal_lagged_mlp_scenario": str(
                FORMAL_SCENARIO.relative_to(ROOT)
            ),
            "stage486_formal_lagged_mlp_runs": str(
                FORMAL_RUNS.relative_to(ROOT)
            ),
        },
        "gate_definition": {
            "candidate": CANDIDATE,
            "references": REFERENCES,
            "scenarios": SCENARIOS,
            "units_per_provider": len(REFERENCES) * len(SCENARIOS),
            "win_rule": (
                "provider-scenario seed-mean candidate event coverage "
                "strictly greater than reference seed-mean event coverage"
            ),
            "ties_counted_as_non_wins": True,
            "pass_threshold": PASS_SHARE,
            "wins_needed": int(
                np.ceil(PASS_SHARE * len(REFERENCES) * len(SCENARIOS))
            ),
        },
        "provider_gate": provider_gate_records,
        "per_scenario_gate": per_scenario_rows,
        "per_reference_gate": per_reference_rows,
        "counterfactual_gates": counterfactuals.to_dict("records"),
        "gate_arithmetic_reconciliation": {
            "frozen_stage493_path": (
                "seed-mean event coverage compared with float > as in "
                "stage493 gate_for_units"
            ),
            "exact_integer_path": (
                "candidate and reference summed detected events over the 5 "
                "seeds; events per provider-scenario are identical across "
                "methods and seeds, so the comparison is exact"
            ),
            "sensitive_units": [
                {
                    "provider": str(row.source),
                    "scenario": str(row.scenario),
                    "reference": str(row.method),
                    "candidate_detected_events_sum": float(
                        row.candidate_detected_events_sum
                    ),
                    "reference_detected_events_sum": float(
                        row.detected_events_sum
                    ),
                    "float_outcome": str(row.gate_outcome),
                    "exact_outcome": str(row.gate_outcome_exact),
                }
                for row in gate[~gate["float_exact_agree"]].itertuples()
            ],
            "conclusion_unchanged": True,
        },
        "tie_inventory": tie_inventory(gate),
        "loss_units": loss_records,
        "fusion_engagement": fusion_rows,
        "provider_distribution": distribution_rows,
        "seed_level_gate": seed_level_rows,
        "failure_source_attribution": attribution_rows,
        "key_findings": [
            (
                "Strict-win gate: wins/108 = ECCC 66 (61.1%), KNMI 74 "
                "(68.5%), MeteoSwiss 67 (62.0%), NOAA ISD 50 (46.3%); "
                "76 wins are needed for 70%."
            ),
            (
                "Exact-integer reconciliation (summed detected events over "
                "seeds) flips three win units to exact ties - ECCC low_spike "
                "vs robust_max_deviation (19 vs 19 events), KNMI slow_drift "
                "vs pca_reconstruction and vs lightweight_autoencoder "
                "(4 vs 4 each) - giving ECCC 65, KNMI 72, MeteoSwiss 67 and "
                "NOAA ISD 50; every provider still fails 70%."
            ),
            (
                "Ties are counted as non-wins: 82/432 units (19.0%) are "
                "exact ties, 24 at zero coverage and 35 at full coverage; "
                "excluding ties lifts ECCC/KNMI/MeteoSwiss to 80.5/82.2/"
                "82.7% but leaves NOAA ISD at 51.5%."
            ),
            (
                "Losses concentrate in low_noise (18), slow_drift (19), "
                "coordinated_offset (14), delay (14) and low_offset (14); "
                "the two structured replay/stuck scenarios produce zero "
                "losses but 35 ties."
            ),
            (
                "The two references that beat the candidate most often are "
                "arima_innovation (22 losses) and formal_lagged_mlp (18), "
                "both coverage-oriented detectors with much lower run-level "
                "F1 than the candidate."
            ),
            (
                "ECCC/KNMI/MeteoSwiss losses are mostly one-event margins "
                "(81%/88%/57% within one event); 75%/13%/50% of their "
                "losses are to references with a higher test FAR than the "
                "candidate, so part of the shortfall is a coverage-only, "
                "uncalibrated-metric artifact."
            ),
            (
                "NOAA ISD is different: 47 genuine losses, 0% of them to "
                "references with a higher FAR, candidate identical to the "
                "frozen union in all 5 seeds (weight 1.0, only one feasible "
                "fusion candidate), lowest union ROC AUC (0.636) and only "
                "about 3,510 observed normal rows per station versus "
                "8,509-8,647 for the other providers."
            ),
            (
                "NOAA ISD is essentially blind to subtle attacks at the "
                "frozen operating point (mean coverage over the seven "
                "subtle scenarios 0.034 versus 0.133-0.255 elsewhere) while "
                "the formal lagged-MLP alone is stronger there on test "
                "(delay 0.309, short_lag 0.309, low_spike 0.164); the "
                "validation-only structured-scenario constraint forces the "
                "union fallback that leaves those gains unused."
            ),
        ],
        "uncertainties": [
            (
                "The V6 test split has already been inspected; all results "
                "are exploratory and post-hoc."
            ),
            (
                "The extended gate is coverage-only; it ignores F1, FAR and "
                "calibration, so ties and one-event margins decide most of "
                "the ECCC/KNMI/MeteoSwiss shortfall."
            ),
            (
                "Validation-to-test FAR transfer differs by provider (KNMI "
                "candidate test FAR 0.0143 = 2.2x the 0.0065 target), so "
                "cross-provider coverage comparisons are not at identical "
                "realized FAR."
            ),
            (
                "Four providers are fixed clusters; provider-level numbers "
                "are descriptive, not population inference."
            ),
            (
                "The formal component is a lagged-window MLP, not a "
                "GRU/LSTM; the NOAA fallback result does not establish what "
                "a different fusion rule would do on unseen data."
            ),
        ],
    }
    return payload


def markdown_report(
    summary: pd.DataFrame,
    gate: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    distribution: pd.DataFrame,
    payload: dict[str, object],
) -> str:
    lines: list[str] = [
        "# Stage494 provider heterogeneity audit (exploratory)",
        "",
        f"- Status: `{payload['status']}`",
        "- Scope: read-only audit of existing Stage491 constrained fusion, "
        "frozen V6 union and formal lagged-MLP results.",
        "- Exploratory only: the V6 test split has already been inspected; "
        "no result here is confirmatory.",
        "",
        "## Gate definition",
        "",
        "- Candidate: `constrained_validation_fusion` (Stage491).",
        "- References: 11 frozen baselines + `formal_lagged_mlp` (12).",
        "- Unit: provider x scenario x reference = 108 per provider.",
        "- Win: provider-scenario seed-mean candidate event coverage strictly "
        "greater than the reference seed-mean coverage.",
        "- Ties count as non-wins; 70% pass needs 76/108 wins.",
        "- Primary counts reproduce the frozen Stage493 float path. An exact-"
        "integer check (summed detected events over seeds) flips three win "
        "units to ties (ECCC low_spike vs robust_max_deviation; KNMI "
        "slow_drift vs pca_reconstruction and lightweight_autoencoder), so "
        "ECCC becomes 65/108 and KNMI 72/108; every provider still fails.",
        "",
        "## Provider gate results",
        "",
    ]
    table = summary[
        [
            "provider",
            "extended_wins",
            "extended_ties",
            "extended_losses",
            "extended_units",
            "extended_share",
            "extended_share_excl_ties",
            "wins_needed_70pct",
            "wins_shortfall",
            "passes_70pct",
        ]
    ].copy()
    for column in [
        "extended_share",
        "extended_share_excl_ties",
    ]:
        table[column] = table[column].map(lambda value: f"{value:.3f}")
    lines.append(table.to_markdown(index=False))
    lines.extend(["", "## Per-scenario win/tie/loss", ""])
    scenario_rows: list[dict[str, object]] = []
    for row in summary.itertuples():
        for scenario in SCENARIOS:
            scenario_rows.append(
                {
                    "provider": row.provider,
                    "scenario": scenario,
                    "wins": getattr(row, f"wins_{scenario}"),
                    "ties": getattr(row, f"ties_{scenario}"),
                    "losses": getattr(row, f"losses_{scenario}"),
                    "candidate_coverage": float(
                        gate[
                            gate["source"].eq(row.provider)
                            & gate["scenario"].eq(scenario)
                        ]["candidate_coverage"].iloc[0]
                    ),
                }
            )
    scenario_table = pd.DataFrame(scenario_rows)
    scenario_table["candidate_coverage"] = scenario_table[
        "candidate_coverage"
    ].map(lambda value: f"{value:.3f}")
    lines.append(scenario_table.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Attribution matrix",
            "",
        ]
    )
    attribution_table = pd.DataFrame(
        [
            {
                "provider": row["provider"],
                "wins_shortfall": row["wins_shortfall"],
                "ties": row["ties"],
                "share_excl_ties": f"{row['share_excl_ties']:.3f}",
                "losses": row["losses"],
                "within_one_event": row["losses_within_one_event"],
                "losses_to_higher_far_refs": row[
                    "losses_to_higher_far_references"
                ],
                "far_over_target": f"{row['test_far_over_target']:.2f}",
                "top_loss_scenarios": ", ".join(
                    f"{item['scenario']}({item['losses']})"
                    for item in row["top_loss_scenarios"]
                ),
                "top_loss_refs": ", ".join(
                    f"{item['reference']}({item['losses']})"
                    for item in row["top_loss_references"]
                ),
                "union_roc_auc": f"{row['union_roc_auc']:.3f}",
                "normal_rows_per_station": (
                    f"{row['observed_normal_rows_per_station']:.0f}"
                ),
            }
            for row in payload["failure_source_attribution"]
        ]
    )
    lines.append(attribution_table.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Failure-source decomposition",
            "",
            "### ECCC",
            "",
            "- 66 wins / 26 ties / 16 losses; excluding ties 80.5% (passes).",
            "- Exact-integer check: 65 wins / 27 ties (low_spike vs "
            "robust_max_deviation is a true 19-vs-19 event tie).",
            "- 11 ties are structured-scenario ceiling ties and 8 are "
            "`coordinated_offset` zero-coverage ties.",
            "- Losses: `low_noise` 8, `coordinated_offset` 4, then one each "
            "in delay/low_spike/short_lag/slow_drift; 13/16 within one event.",
            "- 12/16 losses are to references with a higher test FAR than "
            "the candidate, so threshold conservatism explains part of the "
            "shortfall.",
            "- Fusion engagement: weight 0.7 in 4/5 seeds; coverage gain "
            "over the union is only +0.003.",
            "",
            "### KNMI",
            "",
            "- 74 wins / 18 ties / 16 losses; 2 wins short; excluding ties "
            "82.2% (passes).",
            "- Exact-integer check: 72 wins / 20 ties (slow_drift vs "
            "pca_reconstruction and lightweight_autoencoder are true 4-vs-4 "
            "event ties).",
            "- Losses: `slow_drift` 6, `low_noise` 3, delay 2, low_spike 2, "
            "coordinated_offset 2, short_lag 1; 14/16 within one event.",
            "- Candidate test FAR is 0.0143 = 2.2x the 0.0065 target (union "
            "0.01675), so the coverage comes from a permissive operating "
            "point that still fails the gate; only 2/16 losses are to "
            "higher-FAR references.",
            "- Coarsest event granularity (6 stations, 6 events/scenario; "
            "one event = 0.167 coverage).",
            "",
            "### MeteoSwiss",
            "",
            "- 67 wins / 27 ties / 14 losses; 9 wins short; excluding ties "
            "82.7% (passes).",
            "- 12 ties are `coordinated_offset` (candidate covers 0 of 2 "
            "events) and 12 are structured-scenario ceiling ties.",
            "- Losses: `slow_drift` 4, `low_noise` 3, `low_offset` 3, delay "
            "2, low_spike 2; 8/14 within one event.",
            "- Fusion engaged in 3/5 seeds (weight 0.5) and adds +0.048 "
            "coverage over the union, the largest fusion gain.",
            "",
            "### NOAA ISD",
            "",
            "- 50 wins / 11 ties / 47 losses; 26 wins short; excluding ties "
            "51.5% (still fails).",
            "- All 24 structured-scenario units are wins; all 7 subtle "
            "scenarios contribute losses (delay 9, low_offset 11, "
            "coordinated_offset 8, slow_drift 8, low_noise 4, low_spike 4, "
            "short_lag 3).",
            "- Candidate equals the frozen union in all 5 seeds (weight "
            "1.0; only one feasible fusion candidate per seed), so the "
            "formal component is never used.",
            "- 0/47 losses are to references with a higher FAR; the "
            "reference FAR is lower than the candidate's in every NOAA "
            "unit, so this is a genuine detection deficit, not threshold "
            "conservatism.",
            "- NOAA union ROC AUC is the lowest (0.636) and the observed "
            "normal background is ~3,510 rows/station versus 8,509-8,647 "
            "for the other providers, with identical attack-row design.",
            "- On test the formal lagged-MLP alone is stronger than the "
            "union on subtle scenarios (delay 0.309, short_lag 0.309, "
            "low_spike 0.164 vs 0.036/0.055/0.055) but collapses on "
            "drop_replay/long_stuck (0.036/0.091 vs 0.891/0.945), which "
            "makes the validation structured-scenario constraint force "
            "the union fallback.",
            "",
            "## Cross-provider decomposition",
            "",
            "- Ties: 82/432 units (19.0%); 24 at zero coverage, 35 at full "
            "coverage, 23 intermediate. The structured scenarios alone "
            "produce 35 ties and zero losses.",
            "- Reference strength: `arima_innovation` (22 losses) and "
            "`formal_lagged_mlp` (18) cause the most losses; both are "
            "coverage-oriented with much lower run-level F1 than the "
            "candidate.",
            "- Attack type: losses are concentrated in low_noise (18), "
            "slow_drift (19), coordinated_offset (14), delay (14) and "
            "low_offset (14); `coordinated_offset` has only 2 events per "
            "provider, so its coverage grid is {0, 0.5, 1}.",
            "- Event granularity: 1/events per unit is 1/14 (ECCC), 1/6 "
            "(KNMI), 1/11 (MeteoSwiss, NOAA); 81%/88%/57%/87% of losses "
            "are within one event.",
            "- Station/observation distribution: ECCC 14 stations, KNMI 6, "
            "MeteoSwiss 11, NOAA 11; NOAA has ~41% of the other providers' "
            "observed normal rows per station with the same attack design.",
            "- FAR transfer: candidate test FAR / 0.0065 is 0.98 (ECCC), "
            "2.20 (KNMI), 1.09 (MeteoSwiss), 1.01 (NOAA); KNMI is the "
            "only clear calibration-transfer outlier.",
            "- Seed-level robustness: across the 2160 single-seed reference "
            "units, "
            f"{sum(row['ties'] for row in payload['seed_level_gate'])} are "
            "exact ties ("
            f"{sum(row['ties'] for row in payload['seed_level_gate']) / 2160:.1%}"
            ") and per-seed win shares range "
            f"{min(row['min_seed_share'] for row in payload['seed_level_gate']):.2f}-"
            f"{max(row['max_seed_share'] for row in payload['seed_level_gate']):.2f}; "
            "the provider-level gate averages the five seeds first and "
            "exceeds every single-seed share for the same provider.",
            "",
            "## Counterfactual gates (exploratory)",
            "",
        ]
    )
    counterfactual_table = counterfactuals.pivot_table(
        index="variant",
        columns="provider",
        values="share",
    ).reset_index()
    for column in counterfactual_table.columns[1:]:
        counterfactual_table[column] = counterfactual_table[column].map(
            lambda value: f"{value:.3f}"
        )
    lines.append(counterfactual_table.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Provider distribution context",
            "",
        ]
    )
    distribution_table = (
        distribution.groupby("source", as_index=False)
        .agg(
            stations=("stations", "mean"),
            events_total=("events_total", "mean"),
            observed_normal_rows_per_station=(
                "observed_normal_per_station",
                "mean",
            ),
            observed_attack_rows_per_station=(
                "observed_attack_per_station",
                "mean",
            ),
            union_roc_auc=("union_roc_auc", "mean"),
            formal_roc_auc=("formal_roc_auc", "mean"),
        )
        .rename(columns={"source": "provider"})
    )
    for column in [
        "observed_normal_rows_per_station",
        "observed_attack_rows_per_station",
        "union_roc_auc",
        "formal_roc_auc",
    ]:
        distribution_table[column] = distribution_table[column].map(
            lambda value: f"{value:.3f}"
        )
    lines.append(distribution_table.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- All numbers are exploratory; the test split was already "
            "inspected and the gate itself is post-hoc.",
            "- The gate is coverage-only; it ignores F1, FAR and "
            "calibration. Ties and one-event margins decide most of the "
            "non-NOAA shortfall.",
            "- Realized test FAR is not identical across providers, so "
            "cross-provider coverage is not compared at a matched "
            "operating point.",
            "- Four providers are fixed clusters; no population inference "
            "is intended.",
            "",
            "## Artifacts",
            "",
            "- `submission_final_20261009/tables/"
            "stage494_provider_heterogeneity_detail.csv`",
            "- `submission_final_20261009/tables/"
            "stage494_provider_heterogeneity_summary.csv`",
            "- `submission_final_20261009/reports/"
            "stage494_provider_heterogeneity.json`",
            "- `submission_final_20261009/reports/"
            "stage494_provider_heterogeneity.md`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    started = time.perf_counter()
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    long = load_long_frame()
    gate = provider_scenario_gate(long)
    detail = detail_table(long, gate)
    counterfactuals = counterfactual_gates(gate)
    runs = pd.read_csv(FUSION_RUNS)
    selection = pd.read_csv(FUSION_SELECTION)
    station = pd.read_csv(FUSION_STATION)
    distribution = provider_distribution(long, runs, station)
    summary = build_summary(
        long,
        gate,
        detail,
        runs,
        selection,
        distribution,
        counterfactuals,
    )
    payload = build_payload(
        long,
        gate,
        detail,
        summary,
        counterfactuals,
        distribution,
        selection,
    )
    detail.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(
        markdown_report(summary, gate, counterfactuals, distribution, payload),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "provider_gate": payload["provider_gate"],
                "tie_inventory": payload["tie_inventory"],
                "elapsed_seconds": float(time.perf_counter() - started),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
