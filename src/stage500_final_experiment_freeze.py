"""Stage 500: freeze the final exploratory candidate and stop experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage493_constrained_fusion_stability import (  # noqa: E402
    BASELINES,
    hierarchical_bootstrap,
    summary_for_units,
)

FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

STAGE491_RUNS = TABLES / "stage491_constrained_fusion_runs.csv"
STAGE491_SCENARIO = TABLES / "stage491_constrained_fusion_scenario.csv"
STAGE499_RUNS = TABLES / "stage499_noaa_representation_v2_runs.csv"
STAGE499_SCENARIO = TABLES / "stage499_noaa_representation_v2_scenario.csv"
EXISTING_SCENARIO = TABLES / "final_v6_scenario_metrics.csv"
FORMAL_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"

OUT_RUNS = TABLES / "stage500_final_candidate_runs.csv"
OUT_SCENARIO = TABLES / "stage500_final_candidate_scenario.csv"
OUT_PROVIDER = TABLES / "stage500_final_candidate_provider.csv"
OUT_LEAVE_ONE = TABLES / "stage500_final_candidate_leave_one_provider.csv"
OUT_BOOTSTRAP = TABLES / "stage500_final_candidate_bootstrap.csv"
OUT_GATE = TABLES / "stage500_final_candidate_gate_bootstrap.csv"
OUT_JSON = REPORTS / "stage500_final_experiment_freeze.json"
OUT_MD = REPORTS / "stage500_final_experiment_freeze.md"

STAGE499_METHOD = "mslag_hgb_residual_stuck"
STRUCTURED = ["long_stuck", "drop_replay"]


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    runs491 = pd.read_csv(STAGE491_RUNS)
    runs499 = pd.read_csv(STAGE499_RUNS)
    runs499 = runs499[
        runs499["method"].eq(STAGE499_METHOD)
        & runs499["deployed"].astype(bool)
    ].copy()
    final_runs = pd.concat(
        [
            runs491[~runs491["source"].eq("NOAA ISD")],
            runs499,
        ],
        ignore_index=True,
        sort=False,
    )
    final_runs["method"] = "final_exploratory_candidate"
    final_runs.to_csv(OUT_RUNS, index=False, encoding="utf-8-sig")

    scenario491 = pd.read_csv(STAGE491_SCENARIO)
    scenario499 = pd.read_csv(STAGE499_SCENARIO)
    scenario499 = scenario499[
        scenario499["method"].eq(STAGE499_METHOD)
    ].copy()
    final_scenario = pd.concat(
        [
            scenario491[~scenario491["source"].eq("NOAA ISD")],
            scenario499,
        ],
        ignore_index=True,
        sort=False,
    )
    final_scenario["method"] = "final_exploratory_candidate"
    final_scenario.to_csv(
        OUT_SCENARIO,
        index=False,
        encoding="utf-8-sig",
    )

    existing = pd.read_csv(EXISTING_SCENARIO)
    formal = pd.read_csv(FORMAL_SCENARIO)
    long = pd.concat(
        [
            existing[
                ["source", "seed", "scenario", "method", "event_coverage"]
            ],
            formal[
                ["source", "seed", "scenario", "method", "event_coverage"]
            ],
            final_scenario[
                ["source", "seed", "scenario", "method", "event_coverage"]
            ],
        ],
        ignore_index=True,
        sort=False,
    )
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
    if "final_exploratory_candidate" not in wide:
        raise ValueError("final candidate missing from scenario table")
    wide = wide.rename(
        columns={
            "final_exploratory_candidate": "constrained_validation_fusion"
        }
    )

    constraints: list[dict[str, object]] = []
    union_scenario = existing[
        existing["method"].eq("trust_shape_union")
    ]
    for (source, seed), group in final_scenario.groupby(
        ["source", "seed"],
        sort=True,
    ):
        union_group = union_scenario[
            union_scenario["source"].eq(source)
            & union_scenario["seed"].eq(seed)
        ]
        for scenario in STRUCTURED:
            current = group[group["scenario"].eq(scenario)]
            reference = union_group[
                union_group["scenario"].eq(scenario)
            ]
            current_value = float(current["event_coverage"].mean())
            reference_value = float(reference["event_coverage"].mean())
            delta = current_value - reference_value
            constraints.append(
                {
                    "source": source,
                    "seed": int(seed),
                    "scenario": scenario,
                    "union_test_coverage": reference_value,
                    "constrained_test_coverage": current_value,
                    "test_delta": delta,
                    "within_5pp": bool(delta >= -0.05),
                }
            )
    constraints_frame = pd.DataFrame(constraints)

    provider_rows: list[dict[str, object]] = []
    for provider in sorted(final_runs["source"].unique()):
        provider_rows.append(
            summary_for_units(
                final_runs[final_runs["source"].eq(provider)],
                constraints_frame[
                    constraints_frame["source"].eq(provider)
                ],
                wide[wide["source"].eq(provider)],
                str(provider),
            )
        )
    provider_summary = pd.DataFrame(provider_rows)
    provider_summary.to_csv(
        OUT_PROVIDER,
        index=False,
        encoding="utf-8-sig",
    )

    leave_rows: list[dict[str, object]] = []
    for provider in sorted(final_runs["source"].unique()):
        leave_rows.append(
            summary_for_units(
                final_runs[~final_runs["source"].eq(provider)],
                constraints_frame[
                    ~constraints_frame["source"].eq(provider)
                ],
                wide[~wide["source"].eq(provider)],
                f"without_{provider}",
            )
        )
    leave_one = pd.DataFrame(leave_rows)
    leave_one.to_csv(
        OUT_LEAVE_ONE,
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap, gate_bootstrap = hierarchical_bootstrap(
        final_runs,
        constraints_frame,
        wide,
    )
    bootstrap.to_csv(OUT_BOOTSTRAP, index=False, encoding="utf-8-sig")
    gate_bootstrap.to_csv(OUT_GATE, index=False, encoding="utf-8-sig")
    overall = summary_for_units(
        final_runs,
        constraints_frame,
        wide,
        "all",
    )
    noaa = provider_summary[
        provider_summary["unit"].eq("NOAA ISD")
    ].iloc[0]
    gate = gate_bootstrap.iloc[0]
    payload = {
        "status": "EXPLORATORY_FINAL_CANDIDATE_FROZEN_WITH_BOUNDARIES",
        "candidate": {
            "non_noaa": "Stage491 constrained validation fusion",
            "noaa": f"Stage499 {STAGE499_METHOD}",
        },
        "overall": overall,
        "provider_summary": provider_summary.to_dict("records"),
        "leave_one_provider": leave_one.to_dict("records"),
        "bootstrap": bootstrap.to_dict("records"),
        "gate_bootstrap": gate_bootstrap.to_dict("records"),
        "noaa_improved": bool(
            float(noaa["event_coverage_mean"]) > 0.253333
        ),
        "all_provider_gates_pass_70pct": bool(
            provider_summary["extended_passes_70pct"].all()
        ),
        "far_bootstrap_upper_le_1pct": bool(
            float(
                bootstrap[
                    bootstrap["metric"].eq("false_alarm_rate")
                ]["ci95_high"].iloc[0]
            )
            <= 0.01
        ),
        "stop_reason": (
            "The final candidate improves NOAA and passes the pooled extended "
            "gate at the point estimate, but provider-level gates and strict "
            "FAR stability remain unresolved. No further tuning on the "
            "already inspected V6 test is justified."
        ),
        "limitations": [
            "Exploratory only; V6 test was already inspected.",
            "The formal NOAA component is a HGB residual/stuck-run model, not a GRU/LSTM.",
            "Four providers are fixed clusters; no population inference.",
            "KNMI FAR transfer remains unresolved.",
            "No new provider or dataset was added.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Final experiment freeze",
        "",
        f"- Status: `{payload['status']}`",
        f"- Candidate: `{payload['candidate']}`",
        f"- NOAA improved: `{payload['noaa_improved']}`",
        f"- All provider gates pass 70%: `{payload['all_provider_gates_pass_70pct']}`",
        f"- FAR bootstrap upper <= 1%: `{payload['far_bootstrap_upper_le_1pct']}`",
        "",
        "## Overall",
        "",
        pd.DataFrame([overall]).to_markdown(index=False),
        "",
        "## Provider summary",
        "",
        provider_summary.to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        bootstrap.to_markdown(index=False),
        "",
        "## Gate bootstrap",
        "",
        gate_bootstrap.to_markdown(index=False),
        "",
        "## Stop reason",
        "",
        payload["stop_reason"],
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "overall": overall,
                "noaa": noaa.to_dict(),
                "gate_bootstrap": gate_bootstrap.to_dict("records"),
                "noaa_improved": payload["noaa_improved"],
                "all_provider_gates_pass_70pct": payload[
                    "all_provider_gates_pass_70pct"
                ],
                "far_bootstrap_upper_le_1pct": payload[
                    "far_bootstrap_upper_le_1pct"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
