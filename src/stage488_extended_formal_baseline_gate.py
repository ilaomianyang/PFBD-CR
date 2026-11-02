"""Stage 488: exploratory 70% gate after adding the formal MLP baseline.

The frozen 73/99 comparison is retained.  This stage adds the exploratory
formal lagged-MLP baseline to the denominator and reports the resulting
scenario-by-baseline gate without promoting the result to confirmatory status.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

EXISTING_SCENARIO = TABLES / "final_v6_scenario_aggregate.csv"
FORMAL_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"
OUT_TABLE = TABLES / "stage488_extended_formal_baseline_gate.csv"
OUT_JSON = REPORTS / "stage488_extended_formal_baseline_gate.json"
OUT_MD = REPORTS / "stage488_extended_formal_baseline_gate.md"

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
]


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(EXISTING_SCENARIO)
    formal = pd.read_csv(FORMAL_SCENARIO)
    formal_mean = (
        formal.groupby(["scenario", "method"], as_index=False)
        .agg(event_coverage=("event_coverage", "mean"))
    )
    all_methods = pd.concat(
        [
            existing[["scenario", "method", "event_coverage"]],
            formal_mean,
        ],
        ignore_index=True,
        sort=False,
    )
    union = all_methods[
        all_methods["method"].eq("trust_shape_union")
    ][["scenario", "event_coverage"]].rename(
        columns={"event_coverage": "union_event_coverage"}
    )
    rows: list[dict[str, object]] = []
    for scenario, group in all_methods.groupby("scenario", sort=True):
        candidate = union[union["scenario"].eq(scenario)]
        if candidate.empty:
            continue
        value = float(candidate["union_event_coverage"].iloc[0])
        wins = 0
        losses = 0
        ties = 0
        for baseline in BASELINES + ["formal_lagged_mlp"]:
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
        rows.append(
            {
                "scenario": scenario,
                "union_event_coverage": value,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "total_baselines": total,
                "win_share": wins / total if total else 0.0,
                "passes_70pct": wins / total >= 0.70 if total else False,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
    total_wins = int(table["wins"].sum())
    total_units = int(table["total_baselines"].sum())
    payload = {
        "status": "EXPLORATORY_EXTENDED_FORMAL_BASELINE_GATE",
        "frozen_main_result": "73/99 = 73.7%",
        "exploratory_extended_result": (
            f"{total_wins}/{total_units} = {total_wins / total_units:.4%}"
        ),
        "formal_baseline_added": "formal_lagged_mlp",
        "extended_gate_passes_70pct": bool(
            total_wins / total_units >= 0.70
        ),
        "scenario_rows": table.to_dict("records"),
        "limitations": [
            "The formal baseline is exploratory because V6 test was already inspected.",
            "The frozen 73/99 result is retained and is not rewritten.",
            "The extended denominator includes one formal lagged-window MLP, not a GRU/LSTM.",
            "This result shows that the project-specific 70% gate is sensitive to the baseline set.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Extended formal-baseline gate",
        "",
        f"- Frozen main result: `{payload['frozen_main_result']}`",
        f"- Exploratory extended result: `{payload['exploratory_extended_result']}`",
        f"- Extended gate passes 70%: `{payload['extended_gate_passes_70pct']}`",
        "",
        table.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- The formal MLP is exploratory.",
        "- The frozen 73/99 result is retained.",
        "- Adding the formal baseline changes the project-specific gate denominator.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "extended_result": payload[
                    "exploratory_extended_result"
                ],
                "passes": payload["extended_gate_passes_70pct"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
