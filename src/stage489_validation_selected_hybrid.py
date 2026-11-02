"""Stage 489: validation-selected Trust-union versus formal-MLP hybrid.

The choice between the frozen Trust union and the exploratory formal lagged
MLP is made per provider-seed using validation event coverage only.  Test
metrics are then read from the two already saved test evaluations.  The result
is exploratory because the V6 test has already been inspected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

SELECTION = TABLES / "final_v6_selection.csv"
UNION_DETAIL = TABLES / "final_v6_detail.csv"
UNION_SCENARIO = TABLES / "final_v6_scenario_metrics.csv"
FORMAL_RUNS = TABLES / "stage486_formal_lagged_mlp_runs.csv"
FORMAL_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"
EXISTING_SCENARIO = TABLES / "final_v6_scenario_aggregate.csv"

OUT_RUNS = TABLES / "stage489_validation_selected_hybrid_runs.csv"
OUT_SCENARIO = TABLES / "stage489_validation_selected_hybrid_scenario.csv"
OUT_GATE = TABLES / "stage489_validation_selected_hybrid_gate.csv"
OUT_JSON = REPORTS / "stage489_validation_selected_hybrid.json"
OUT_MD = REPORTS / "stage489_validation_selected_hybrid.md"

BASELINES = [
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
    "formal_lagged_mlp",
]
RUN_METRICS = [
    "precision",
    "recall",
    "f1",
    "false_alarm_rate",
    "roc_auc",
    "average_precision",
    "event_coverage",
    "row_recall",
]


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(SELECTION)
    union_runs = pd.read_csv(UNION_DETAIL)
    union_runs = union_runs[
        union_runs["method"].eq("trust_shape_union")
    ].copy()
    formal_runs = pd.read_csv(FORMAL_RUNS)

    choice = selection[
        [
            "source",
            "seed",
            "union_valid_f1",
            "union_valid_event_coverage",
        ]
    ].merge(
        formal_runs[
            [
                "source",
                "seed",
                "valid_f1",
                "valid_event_coverage",
            ]
        ].rename(
            columns={
                "valid_f1": "formal_valid_f1",
                "valid_event_coverage": "formal_valid_event_coverage",
            }
        ),
        on=["source", "seed"],
        validate="one_to_one",
    )
    choice["chosen"] = np.where(
        choice["union_valid_event_coverage"]
        > choice["formal_valid_event_coverage"],
        "trust_shape_union",
        np.where(
            choice["union_valid_event_coverage"]
            < choice["formal_valid_event_coverage"],
            "formal_lagged_mlp",
            np.where(
                choice["union_valid_f1"] >= choice["formal_valid_f1"],
                "trust_shape_union",
                "formal_lagged_mlp",
            ),
        ),
    )
    choice["choice_margin_event_coverage"] = (
        choice["union_valid_event_coverage"]
        - choice["formal_valid_event_coverage"]
    )

    union_selected = union_runs[
        ["source", "seed", *RUN_METRICS]
    ].rename(
        columns={metric: f"union_{metric}" for metric in RUN_METRICS}
    )
    formal_selected = formal_runs[
        ["source", "seed", *RUN_METRICS]
    ].rename(
        columns={metric: f"formal_{metric}" for metric in RUN_METRICS}
    )
    selected_runs = (
        choice.merge(union_selected, on=["source", "seed"])
        .merge(formal_selected, on=["source", "seed"])
    )
    for metric in RUN_METRICS:
        selected_runs[f"selected_{metric}"] = np.where(
            selected_runs["chosen"].eq("trust_shape_union"),
            selected_runs[f"union_{metric}"],
            selected_runs[f"formal_{metric}"],
        )
    selected_runs.to_csv(OUT_RUNS, index=False, encoding="utf-8-sig")

    selected_summary = {
        metric: float(selected_runs[f"selected_{metric}"].mean())
        for metric in RUN_METRICS
    }
    selection_counts = (
        selected_runs["chosen"].value_counts().to_dict()
    )

    union_scenario = pd.read_csv(UNION_SCENARIO)
    union_scenario = union_scenario[
        union_scenario["method"].eq("trust_shape_union")
    ].copy()
    formal_scenario = pd.read_csv(FORMAL_SCENARIO)
    scenario_choice = choice[["source", "seed", "chosen"]]
    union_scenario = union_scenario.merge(
        scenario_choice,
        on=["source", "seed"],
        validate="many_to_one",
    )
    formal_scenario = formal_scenario.merge(
        scenario_choice,
        on=["source", "seed"],
        validate="many_to_one",
    )
    scenario_metrics = [
        "precision",
        "row_recall",
        "f1",
        "false_alarm_rate",
        "roc_auc",
        "average_precision",
        "event_coverage",
        "mean_event_length",
        "events",
        "detected_events",
    ]
    union_scenario = union_scenario[
        ["source", "seed", "scenario", "chosen", *scenario_metrics]
    ].rename(
        columns={metric: f"union_{metric}" for metric in scenario_metrics}
    )
    formal_scenario = formal_scenario[
        ["source", "seed", "scenario", "chosen", *scenario_metrics]
    ].rename(
        columns={metric: f"formal_{metric}" for metric in scenario_metrics}
    )
    selected_scenario = union_scenario.merge(
        formal_scenario,
        on=["source", "seed", "scenario", "chosen"],
        validate="one_to_one",
    )
    for metric in scenario_metrics:
        selected_scenario[f"selected_{metric}"] = np.where(
            selected_scenario["chosen"].eq("trust_shape_union"),
            selected_scenario[f"union_{metric}"],
            selected_scenario[f"formal_{metric}"],
        )
    selected_scenario.to_csv(OUT_SCENARIO, index=False, encoding="utf-8-sig")
    scenario_summary = (
        selected_scenario.groupby("scenario", as_index=False)
        .agg(
            selected_event_coverage=("selected_event_coverage", "mean"),
            selected_f1=("selected_f1", "mean"),
        )
        .sort_values("scenario")
    )

    all_scenario = pd.read_csv(EXISTING_SCENARIO)
    formal_mean = (
        pd.read_csv(FORMAL_SCENARIO)
        .groupby(["scenario", "method"], as_index=False)
        .agg(event_coverage=("event_coverage", "mean"))
    )
    hybrid_mean = scenario_summary.rename(
        columns={
            "selected_event_coverage": "event_coverage",
        }
    ).assign(method="validation_selected_hybrid")[
        ["scenario", "method", "event_coverage"]
    ]
    all_scenario = pd.concat(
        [
            all_scenario[["scenario", "method", "event_coverage"]],
            formal_mean,
            hybrid_mean,
        ],
        ignore_index=True,
        sort=False,
    )
    gate_rows: list[dict[str, object]] = []
    for scenario, group in all_scenario.groupby("scenario", sort=True):
        candidate = group[
            group["method"].eq("validation_selected_hybrid")
        ]
        if candidate.empty:
            continue
        value = float(candidate["event_coverage"].iloc[0])
        wins = losses = ties = 0
        for baseline in BASELINES:
            reference = group[group["method"].eq(baseline)]
            if reference.empty:
                continue
            reference_value = float(reference["event_coverage"].iloc[0])
            if value > reference_value:
                wins += 1
            elif value < reference_value:
                losses += 1
            else:
                ties += 1
        total = wins + losses + ties
        gate_rows.append(
            {
                "scenario": scenario,
                "hybrid_event_coverage": value,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "total_baselines": total,
                "win_share": wins / total if total else 0.0,
                "passes_70pct": wins / total >= 0.70 if total else False,
            }
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(OUT_GATE, index=False, encoding="utf-8-sig")
    total_wins = int(gate["wins"].sum())
    total_units = int(gate["total_baselines"].sum())
    payload = {
        "status": "PASS_EXPLORATORY_VALIDATION_SELECTED_HYBRID",
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "selection_rule": (
            "Per provider-seed, choose the method with higher validation "
            "event coverage; break ties by validation F1."
        ),
        "selection_counts": selection_counts,
        "selected_summary": selected_summary,
        "scenario_summary": scenario_summary.to_dict("records"),
        "extended_gate": {
            "wins": total_wins,
            "units": total_units,
            "share": total_wins / total_units if total_units else 0.0,
            "passes_70pct": bool(
                total_wins / total_units >= 0.70
            )
            if total_units
            else False,
        },
        "limitations": [
            "The choice uses validation attacks only, not test labels.",
            "The formal MLP is a lagged-window MLP, not a GRU/LSTM.",
            "The V6 test has already been inspected; this is exploratory.",
            "The hybrid does not replace the frozen V6 main result.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Validation-selected Trust-union versus formal-MLP hybrid",
        "",
        f"- Status: `{payload['status']}`",
        f"- Selection counts: `{selection_counts}`",
        f"- Selected summary: `{selected_summary}`",
        f"- Extended gate: `{total_wins}/{total_units} = "
        f"{total_wins / total_units:.4%}`",
        f"- Passes 70%: `{payload['extended_gate']['passes_70pct']}`",
        "",
        "## Scenario summary",
        "",
        scenario_summary.to_markdown(index=False),
        "",
        "## Extended gate",
        "",
        gate.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- Selection uses validation event coverage only.",
        "- The formal MLP is exploratory and not a GRU/LSTM.",
        "- The hybrid is exploratory and does not replace the frozen result.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection_counts": selection_counts,
                "selected_summary": selected_summary,
                "extended_gate": payload["extended_gate"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
