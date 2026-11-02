"""Stage 487: merge the exploratory formal baseline into comparison tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

EXISTING_SUMMARY = TABLES / "final_v6_method_summary.csv"
EXISTING_SCENARIO = TABLES / "final_v6_scenario_aggregate.csv"
FORMAL_RUNS = TABLES / "stage486_formal_lagged_mlp_runs.csv"
FORMAL_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"

OUT_SUMMARY = TABLES / "final_v6_with_formal_lagged_mlp_summary.csv"
OUT_SCENARIO = TABLES / "final_v6_with_formal_lagged_mlp_scenario.csv"
OUT_WINS = TABLES / "final_v6_union_vs_formal_lagged_mlp_scenario_wins.csv"
OUT_JSON = REPORTS / "stage487_formal_baseline_merge.json"
OUT_MD = REPORTS / "stage487_formal_baseline_merge.md"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    existing_summary = pd.read_csv(EXISTING_SUMMARY)
    existing_summary["evidence_layer"] = "frozen_v6"
    existing_scenario = pd.read_csv(EXISTING_SCENARIO)
    existing_scenario["evidence_layer"] = "frozen_v6"
    formal_runs = pd.read_csv(FORMAL_RUNS)
    formal_scenario = pd.read_csv(FORMAL_SCENARIO)

    metric_columns = [
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "roc_auc",
        "average_precision",
        "event_coverage",
        "row_recall",
    ]
    formal_summary = (
        formal_runs.groupby("method", as_index=False)[metric_columns]
        .mean()
        .assign(evidence_layer="exploratory_formal_lagged_mlp")
    )
    summary = pd.concat(
        [existing_summary, formal_summary],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    formal_scenario_mean = (
        formal_scenario.groupby(
            ["scenario", "method"],
            as_index=False,
        )[
            [
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
        ]
        .mean()
        .assign(evidence_layer="exploratory_formal_lagged_mlp")
    )
    existing_scenario_columns = [
        column
        for column in formal_scenario_mean.columns
        if column in existing_scenario.columns
    ]
    scenario = pd.concat(
        [
            existing_scenario[existing_scenario_columns],
            formal_scenario_mean[existing_scenario_columns],
        ],
        ignore_index=True,
        sort=False,
    )
    scenario.to_csv(OUT_SCENARIO, index=False, encoding="utf-8-sig")

    union = (
        existing_scenario[existing_scenario["method"].eq("trust_shape_union")]
        .groupby("scenario", as_index=False)
        .agg(
            union_event_coverage=("event_coverage", "mean"),
            union_f1=("f1", "mean"),
        )
    )
    formal = (
        formal_scenario.groupby("scenario", as_index=False)
        .agg(
            formal_event_coverage=("event_coverage", "mean"),
            formal_f1=("f1", "mean"),
        )
    )
    wins = union.merge(formal, on="scenario", validate="one_to_one")
    wins["event_coverage_winner"] = wins.apply(
        lambda row: (
            "trust_shape_union"
            if row["union_event_coverage"] > row["formal_event_coverage"]
            else "formal_lagged_mlp"
            if row["formal_event_coverage"] > row["union_event_coverage"]
            else "tie"
        ),
        axis=1,
    )
    wins["f1_winner"] = wins.apply(
        lambda row: (
            "trust_shape_union"
            if row["union_f1"] > row["formal_f1"]
            else "formal_lagged_mlp"
            if row["formal_f1"] > row["union_f1"]
            else "tie"
        ),
        axis=1,
    )
    wins.to_csv(OUT_WINS, index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_EXPLORATORY_FORMAL_BASELINE_MERGE",
        "formal_method": "formal_lagged_mlp",
        "formal_run_count": int(len(formal_runs)),
        "formal_summary": formal_summary.to_dict("records"),
        "union_vs_formal_event_coverage_wins": int(
            wins["event_coverage_winner"].eq("trust_shape_union").sum()
        ),
        "formal_vs_union_event_coverage_wins": int(
            wins["event_coverage_winner"].eq("formal_lagged_mlp").sum()
        ),
        "ties_event_coverage": int(
            wins["event_coverage_winner"].eq("tie").sum()
        ),
        "union_vs_formal_f1_wins": int(
            wins["f1_winner"].eq("trust_shape_union").sum()
        ),
        "formal_vs_union_f1_wins": int(
            wins["f1_winner"].eq("formal_lagged_mlp").sum()
        ),
        "limitations": [
            "The formal baseline is a lagged-window MLP, not a GRU/LSTM.",
            "It is exploratory because V6 test was already inspected.",
            "It does not replace the frozen V6 summary or aggregation tables.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Formal baseline merge",
        "",
        f"- Status: `{payload['status']}`",
        f"- Formal method: `{payload['formal_method']}`",
        f"- Runs: `{payload['formal_run_count']}`",
        "",
        "## Formal summary",
        "",
        formal_summary.to_markdown(index=False),
        "",
        "## Union versus formal baseline",
        "",
        f"- Event coverage wins for union: `{payload['union_vs_formal_event_coverage_wins']}/9`",
        f"- Event coverage wins for formal MLP: `{payload['formal_vs_union_event_coverage_wins']}/9`",
        f"- F1 wins for union: `{payload['union_vs_formal_f1_wins']}/9`",
        f"- F1 wins for formal MLP: `{payload['formal_vs_union_f1_wins']}/9`",
        "",
        wins.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- This is an exploratory formal baseline.",
        "- It does not replace the frozen V6 main table.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "union_event_coverage_wins": payload[
                    "union_vs_formal_event_coverage_wins"
                ],
                "formal_event_coverage_wins": payload[
                    "formal_vs_union_event_coverage_wins"
                ],
                "union_f1_wins": payload["union_vs_formal_f1_wins"],
                "formal_f1_wins": payload[
                    "formal_vs_union_f1_wins"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
