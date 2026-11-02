"""Stage 493: provider/seed stability and hierarchical bootstrap for constrained fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

FUSION_RUNS = TABLES / "stage491_constrained_fusion_runs.csv"
FUSION_SCENARIO = TABLES / "stage491_constrained_fusion_scenario.csv"
FUSION_CONSTRAINT = TABLES / "stage491_constrained_fusion_constraint.csv"
EXISTING_SCENARIO = TABLES / "final_v6_scenario_metrics.csv"
FORMAL_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"

OUT_PROVIDER = TABLES / "stage493_constrained_fusion_provider_summary.csv"
OUT_SEED = TABLES / "stage493_constrained_fusion_seed_summary.csv"
OUT_LEAVE_ONE = TABLES / "stage493_constrained_fusion_leave_one_provider.csv"
OUT_BOOTSTRAP = TABLES / "stage493_constrained_fusion_bootstrap.csv"
OUT_GATE = TABLES / "stage493_constrained_fusion_gate_bootstrap.csv"
OUT_JSON = REPORTS / "stage493_constrained_fusion_stability.json"
OUT_MD = REPORTS / "stage493_constrained_fusion_stability.md"

METRICS = ["f1", "event_coverage", "false_alarm_rate", "average_precision"]
STRUCTURED = ["long_stuck", "drop_replay"]
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
ITERATIONS = 10000
SEED = 20261009


def scenario_wide() -> pd.DataFrame:
    fusion = pd.read_csv(FUSION_SCENARIO)
    fusion = fusion[
        fusion["method"].eq("constrained_validation_fusion")
    ][["source", "seed", "scenario", "event_coverage"]].assign(
        method="constrained_validation_fusion"
    )
    existing = pd.read_csv(EXISTING_SCENARIO)[
        ["source", "seed", "scenario", "method", "event_coverage"]
    ]
    formal = pd.read_csv(FORMAL_SCENARIO)[
        ["source", "seed", "scenario", "method", "event_coverage"]
    ]
    long = pd.concat([existing, formal, fusion], ignore_index=True)
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
    required = [
        "constrained_validation_fusion",
        *BASELINES,
    ]
    missing = [column for column in required if column not in wide]
    if missing:
        raise ValueError(f"missing scenario methods: {missing}")
    return wide


def gate_for_units(wide: pd.DataFrame) -> tuple[int, int, float]:
    wins = 0
    total = 0
    for _, group in wide.groupby("scenario", sort=True):
        candidate = float(
            group["constrained_validation_fusion"].mean()
        )
        for baseline in BASELINES:
            total += 1
            if candidate > float(group[baseline].mean()):
                wins += 1
    return wins, total, wins / total if total else 0.0


def summary_for_units(
    runs: pd.DataFrame,
    constraints: pd.DataFrame,
    wide: pd.DataFrame,
    label: str,
) -> dict[str, object]:
    row: dict[str, object] = {"unit": label, "runs": int(len(runs))}
    for metric in METRICS:
        row[f"{metric}_mean"] = float(runs[metric].mean())
    for scenario in STRUCTURED:
        current = constraints[constraints["scenario"].eq(scenario)]
        row[f"{scenario}_test_delta_mean"] = float(
            current["test_delta"].mean()
        )
        row[f"{scenario}_within_5pp_fraction"] = float(
            current["within_5pp"].mean()
        )
    wins, total, share = gate_for_units(wide)
    row.update(
        {
            "extended_wins": wins,
            "extended_units": total,
            "extended_share": share,
            "extended_passes_70pct": share >= 0.70,
        }
    )
    return row


def hierarchical_bootstrap(
    runs: pd.DataFrame,
    constraints: pd.DataFrame,
    wide: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    providers = sorted(runs["source"].unique())
    seed_values = sorted(runs["seed"].unique())
    rng = np.random.default_rng(SEED)
    metric_draws = {
        metric: np.empty(ITERATIONS, dtype=float) for metric in METRICS
    }
    delta_draws = {
        scenario: np.empty(ITERATIONS, dtype=float)
        for scenario in STRUCTURED
    }
    gate_draws = np.empty(ITERATIONS, dtype=float)
    for iteration in range(ITERATIONS):
        sampled_providers = rng.choice(
            providers,
            size=len(providers),
            replace=True,
        )
        selected: list[tuple[str, int]] = []
        for provider in sampled_providers:
            provider_seeds = rng.choice(
                seed_values,
                size=len(seed_values),
                replace=True,
            )
            selected.extend(
                (str(provider), int(seed))
                for seed in provider_seeds
            )
        selected_frame = pd.DataFrame(
            selected,
            columns=["source", "seed"],
        )
        run_sample = runs.merge(
            selected_frame,
            on=["source", "seed"],
            how="inner",
        )
        constraint_sample = constraints.merge(
            selected_frame,
            on=["source", "seed"],
            how="inner",
        )
        wide_sample = wide.merge(
            selected_frame,
            on=["source", "seed"],
            how="inner",
        )
        for metric in METRICS:
            metric_draws[metric][iteration] = float(
                run_sample[metric].mean()
            )
        for scenario in STRUCTURED:
            delta_draws[scenario][iteration] = float(
                constraint_sample[
                    constraint_sample["scenario"].eq(scenario)
                ]["test_delta"].mean()
            )
        _, _, share = gate_for_units(wide_sample)
        gate_draws[iteration] = share
    metric_rows: list[dict[str, object]] = []
    for metric, draws in metric_draws.items():
        metric_rows.append(
            {
                "metric": metric,
                "point_estimate": float(runs[metric].mean()),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "iterations": ITERATIONS,
            }
        )
    for scenario, draws in delta_draws.items():
        metric_rows.append(
            {
                "metric": f"{scenario}_test_delta",
                "point_estimate": float(
                    constraints[
                        constraints["scenario"].eq(scenario)
                    ]["test_delta"].mean()
                ),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "iterations": ITERATIONS,
            }
        )
    _, _, point_share = gate_for_units(wide)
    gate_rows = [
        {
            "metric": "extended_gate_share",
            "point_estimate": point_share,
            "ci95_low": float(np.quantile(gate_draws, 0.025)),
            "ci95_high": float(np.quantile(gate_draws, 0.975)),
            "probability_passes_70pct": float(
                np.mean(gate_draws >= 0.70)
            ),
            "iterations": ITERATIONS,
        }
    ]
    return pd.DataFrame(metric_rows), pd.DataFrame(gate_rows)


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(FUSION_RUNS)
    constraints = pd.read_csv(FUSION_CONSTRAINT)
    wide = scenario_wide()
    provider_rows: list[dict[str, object]] = []
    for provider in sorted(runs["source"].unique()):
        provider_rows.append(
            summary_for_units(
                runs[runs["source"].eq(provider)],
                constraints[constraints["source"].eq(provider)],
                wide[wide["source"].eq(provider)],
                str(provider),
            )
        )
    provider_summary = pd.DataFrame(provider_rows)
    provider_summary.to_csv(OUT_PROVIDER, index=False, encoding="utf-8-sig")

    seed_rows: list[dict[str, object]] = []
    for seed in sorted(runs["seed"].unique()):
        seed_rows.append(
            summary_for_units(
                runs[runs["seed"].eq(seed)],
                constraints[constraints["seed"].eq(seed)],
                wide[wide["seed"].eq(seed)],
                str(int(seed)),
            )
        )
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(OUT_SEED, index=False, encoding="utf-8-sig")

    leave_rows: list[dict[str, object]] = []
    for provider in sorted(runs["source"].unique()):
        leave_rows.append(
            summary_for_units(
                runs[~runs["source"].eq(provider)],
                constraints[~constraints["source"].eq(provider)],
                wide[~wide["source"].eq(provider)],
                f"without_{provider}",
            )
        )
    leave_one = pd.DataFrame(leave_rows)
    leave_one.to_csv(OUT_LEAVE_ONE, index=False, encoding="utf-8-sig")

    bootstrap, gate_bootstrap = hierarchical_bootstrap(
        runs,
        constraints,
        wide,
    )
    bootstrap.to_csv(OUT_BOOTSTRAP, index=False, encoding="utf-8-sig")
    gate_bootstrap.to_csv(OUT_GATE, index=False, encoding="utf-8-sig")

    payload = {
        "status": "PASS_EXPLORATORY_CONSTRAINED_FUSION_STABILITY",
        "exploratory_only": True,
        "provider_count": int(runs["source"].nunique()),
        "seed_count": int(runs["seed"].nunique()),
        "bootstrap_iterations": ITERATIONS,
        "provider_summary": provider_summary.to_dict("records"),
        "seed_summary": seed_summary.to_dict("records"),
        "leave_one_provider": leave_one.to_dict("records"),
        "bootstrap": bootstrap.to_dict("records"),
        "gate_bootstrap": gate_bootstrap.to_dict("records"),
        "limitations": [
            "Four observed providers are treated as fixed clusters; this is not population inference.",
            "The V6 test was already inspected; the result is exploratory.",
            "The formal component is a lagged-window MLP, not a GRU/LSTM.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Constrained fusion stability audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Providers: `{payload['provider_count']}`; seeds: `{payload['seed_count']}`",
        f"- Bootstrap iterations: `{ITERATIONS}`",
        "",
        "## Provider summary",
        "",
        provider_summary.to_markdown(index=False),
        "",
        "## Leave-one-provider",
        "",
        leave_one.to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        bootstrap.to_markdown(index=False),
        "",
        "## Gate bootstrap",
        "",
        gate_bootstrap.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- Four providers are fixed clusters, not a random population.",
        "- The V6 test was already inspected; all results are exploratory.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "provider_summary": provider_summary.to_dict("records"),
                "leave_one_provider": leave_one.to_dict("records"),
                "bootstrap": bootstrap.to_dict("records"),
                "gate_bootstrap": gate_bootstrap.to_dict("records"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
