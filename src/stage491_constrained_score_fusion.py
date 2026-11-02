"""Stage 491: constrained validation-only score fusion.

Constraint: validation event coverage for long_stuck and drop_replay must not
fall more than 5 percentage points below the frozen union prediction.
Objective: maximize mean validation event coverage across all nine scenarios,
with validation F1 as the tie-breaker.  Test is evaluation only.
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
FORMAL_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"

OUT_RUNS = TABLES / "stage491_constrained_fusion_runs.csv"
OUT_SCENARIO = TABLES / "stage491_constrained_fusion_scenario.csv"
OUT_STATION = TABLES / "stage491_constrained_fusion_station.csv"
OUT_SELECTION = TABLES / "stage491_constrained_fusion_selection.csv"
OUT_GATE = TABLES / "stage491_constrained_fusion_gate.csv"
OUT_CONSTRAINT = TABLES / "stage491_constrained_fusion_constraint.csv"
OUT_JSON = REPORTS / "stage491_constrained_fusion.json"
OUT_MD = REPORTS / "stage491_constrained_fusion.md"

PRIMARY_FAR = 0.0065
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CONSTRAINED_SCENARIOS = ["long_stuck", "drop_replay"]
CONSTRAINT_TOLERANCE = 0.05


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
        events = group["attack_event_id"].astype(str)
        detected = group.groupby("attack_event_id")["_pred"].any()
        output[str(scenario)] = (
            float(detected.mean()) if len(detected) else 0.0
        )
    return output


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
    run_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    constraint_rows: list[dict[str, object]] = []

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
        union_med = float(np.median(union_valid_raw[valid_normal]))
        formal_med = float(np.median(formal_valid_raw[valid_normal]))
        union_scale = max(
            1.4826
            * float(
                np.median(
                    np.abs(
                        union_valid_raw[valid_normal] - union_med
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
                        formal_valid_raw[valid_normal] - formal_med
                    )
                )
            ),
            1e-6,
        )
        union_valid_z = (union_valid_raw - union_med) / union_scale
        formal_valid_z = (formal_valid_raw - formal_med) / formal_scale
        union_test_z = (union_test_raw - union_med) / union_scale
        formal_test_z = (formal_test_raw - formal_med) / formal_scale
        union_valid_pred = valid["union_prediction"].to_numpy(dtype=bool)
        union_coverage = scenario_coverage(
            valid_frame,
            union_valid_pred,
        )

        candidates: list[dict[str, object]] = []
        for weight in [*WEIGHT_GRID, "max"]:
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
            predicted_valid = (
                valid_frame["observation_present"].to_numpy(dtype=bool)
                & (fused_valid >= threshold)
            )
            if weight == 1.0:
                predicted_valid = union_valid_pred
            valid_metrics = metrics(
                valid_frame["is_attack"].astype(bool).to_numpy(),
                predicted_valid,
                fused_valid,
            )
            valid_event = event_metrics(
                valid_frame,
                predicted_valid,
            )
            coverage = scenario_coverage(
                valid_frame,
                predicted_valid,
            )
            constraint_ok = all(
                coverage.get(scenario, 0.0)
                >= union_coverage.get(scenario, 0.0)
                - CONSTRAINT_TOLERANCE
                for scenario in CONSTRAINED_SCENARIOS
            )
            mean_coverage = float(
                np.mean(
                    [
                        coverage.get(scenario, 0.0)
                        for scenario in sorted(
                            set(coverage) | set(union_coverage)
                        )
                    ]
                )
            )
            candidates.append(
                {
                    "weight": weight,
                    "threshold": threshold,
                    "valid_f1": float(valid_metrics["f1"]),
                    "valid_event_coverage": float(
                        valid_event["event_coverage"]
                    ),
                    "valid_mean_scenario_coverage": mean_coverage,
                    "constraint_ok": bool(constraint_ok),
                    **{
                        f"valid_coverage_{scenario}": float(
                            coverage.get(scenario, 0.0)
                        )
                        for scenario in CONSTRAINED_SCENARIOS
                    },
                    **{
                        f"valid_union_coverage_{scenario}": float(
                            union_coverage.get(scenario, 0.0)
                        )
                        for scenario in CONSTRAINED_SCENARIOS
                    },
                }
            )
        feasible = [
            candidate
            for candidate in candidates
            if candidate["constraint_ok"]
        ]
        if not feasible:
            best = next(
                candidate
                for candidate in candidates
                if candidate["weight"] == 1.0
            )
            fallback = True
        else:
            best = max(
                feasible,
                key=lambda candidate: (
                    candidate["valid_mean_scenario_coverage"],
                    candidate["valid_f1"],
                    -candidate["valid_event_coverage"],
                ),
            )
            fallback = False

        fused_test = candidate_scores(
            union_test_z,
            formal_test_z,
            best["weight"],
        )
        if best["weight"] == 1.0:
            predicted_test = test["union_prediction"].to_numpy(
                dtype=bool
            )
        else:
            predicted_test = (
                test_frame["observation_present"].to_numpy(dtype=bool)
                & (fused_test >= float(best["threshold"]))
            )
        test_metrics = metrics(
            test_frame["is_attack"].astype(bool).to_numpy(),
            predicted_test,
            fused_test,
        )
        test_event = event_metrics(test_frame, predicted_test)
        test_coverage = scenario_coverage(
            test_frame,
            predicted_test,
        )
        union_test_coverage = scenario_coverage(
            test_frame,
            test["union_prediction"].to_numpy(dtype=bool),
        )
        for scenario in CONSTRAINED_SCENARIOS:
            delta = (
                test_coverage.get(scenario, 0.0)
                - union_test_coverage.get(scenario, 0.0)
            )
            constraint_rows.append(
                {
                    "source": source,
                    "seed": int(seed),
                    "scenario": scenario,
                    "union_test_coverage": float(
                        union_test_coverage.get(scenario, 0.0)
                    ),
                    "constrained_test_coverage": float(
                        test_coverage.get(scenario, 0.0)
                    ),
                    "test_delta": float(delta),
                    "within_5pp": bool(delta >= -CONSTRAINT_TOLERANCE),
                }
            )
        selection_rows.append(
            {
                "source": source,
                "seed": int(seed),
                **best,
                "fallback_to_union": fallback,
                "feasible_candidate_count": len(feasible),
            }
        )
        run_rows.append(
            {
                "source": source,
                "seed": int(seed),
                "method": "constrained_validation_fusion",
                **best,
                **test_metrics,
                **test_event,
            }
        )
        scenario_rows.extend(
            s470.scenario_rows_full(
                test_frame,
                predicted_test,
                fused_test,
                source,
                int(seed),
                "constrained_validation_fusion",
            )
        )
        station_rows.extend(
            s470.station_rows_full(
                test_frame,
                predicted_test,
                source,
                int(seed),
                "constrained_validation_fusion",
            )
        )

    runs = pd.DataFrame(run_rows)
    scenario = pd.DataFrame(scenario_rows)
    station = pd.DataFrame(station_rows)
    selection = pd.DataFrame(selection_rows)
    constraints = pd.DataFrame(constraint_rows)
    runs.to_csv(OUT_RUNS, index=False, encoding="utf-8-sig")
    scenario.to_csv(OUT_SCENARIO, index=False, encoding="utf-8-sig")
    station.to_csv(OUT_STATION, index=False, encoding="utf-8-sig")
    selection.to_csv(OUT_SELECTION, index=False, encoding="utf-8-sig")
    constraints.to_csv(OUT_CONSTRAINT, index=False, encoding="utf-8-sig")

    fusion_scenario = (
        scenario.groupby("scenario", as_index=False)
        .agg(
            fusion_event_coverage=("event_coverage", "mean"),
            fusion_f1=("f1", "mean"),
        )
    )
    existing = pd.read_csv(EXISTING_SCENARIO)
    formal = pd.read_csv(FORMAL_SCENARIO)
    formal_mean = (
        formal.groupby(["scenario", "method"], as_index=False)
        .agg(event_coverage=("event_coverage", "mean"))
    )
    all_scenario = pd.concat(
        [
            existing[["scenario", "method", "event_coverage"]],
            formal_mean,
            fusion_scenario.rename(
                columns={"fusion_event_coverage": "event_coverage"}
            ).assign(method="constrained_validation_fusion")[
                ["scenario", "method", "event_coverage"]
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    gate_rows: list[dict[str, object]] = []
    for scenario_name, group in all_scenario.groupby(
        "scenario",
        sort=True,
    ):
        candidate = group[
            group["method"].eq("constrained_validation_fusion")
        ]
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
                reference_value = float(
                    reference["event_coverage"].iloc[0]
                )
                if value > reference_value:
                    wins += 1
                elif value < reference_value:
                    losses += 1
                else:
                    ties += 1
            total = wins + losses + ties
            gate_rows.append(
                {
                    "baseline_set": label,
                    "scenario": scenario_name,
                    "fusion_event_coverage": value,
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
    gate_summary = (
        gate.groupby("baseline_set", as_index=False)
        .agg(
            wins=("wins", "sum"),
            total=("total_baselines", "sum"),
        )
    )
    gate_summary["share"] = (
        gate_summary["wins"] / gate_summary["total"]
    )
    gate_summary["passes_70pct"] = gate_summary["share"] >= 0.70
    constraint_summary = (
        constraints.groupby("scenario", as_index=False)
        .agg(
            union_test_coverage=("union_test_coverage", "mean"),
            constrained_test_coverage=(
                "constrained_test_coverage",
                "mean",
            ),
            test_delta=("test_delta", "mean"),
            within_5pp_fraction=("within_5pp", "mean"),
        )
    )
    payload = {
        "status": "PASS_EXPLORATORY_CONSTRAINED_FUSION",
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "constraint": {
            "scenarios": CONSTRAINED_SCENARIOS,
            "tolerance": CONSTRAINT_TOLERANCE,
            "basis": "validation event coverage relative to union",
        },
        "selection_counts": selection["weight"].value_counts().to_dict(),
        "fallback_count": int(selection["fallback_to_union"].sum()),
        "run_summary": {
            "f1_mean": float(runs["f1"].mean()),
            "event_coverage_mean": float(runs["event_coverage"].mean()),
            "false_alarm_rate_mean": float(
                runs["false_alarm_rate"].mean()
            ),
            "average_precision_mean": float(
                runs["average_precision"].mean()
            ),
        },
        "gate_summary": gate_summary.to_dict("records"),
        "constraint_summary": constraint_summary.to_dict("records"),
        "scenario_summary": fusion_scenario.to_dict("records"),
        "limitations": [
            "Constraint is enforced on validation only; test deltas are reported separately.",
            "The V6 test has already been inspected; this is exploratory.",
            "The formal component is a lagged-window MLP, not a GRU/LSTM.",
            "The result does not replace the frozen V6 main table.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Constrained validation-only score fusion",
        "",
        f"- Status: `{payload['status']}`",
        f"- Constraint: `{CONSTRAINED_SCENARIOS}` within "
        f"`{CONSTRAINT_TOLERANCE}` validation coverage of union",
        f"- Selection counts: `{payload['selection_counts']}`",
        f"- Fallback to union: `{payload['fallback_count']}`",
        f"- Run summary: `{payload['run_summary']}`",
        "",
        "## Gate summary",
        "",
        gate_summary.to_markdown(index=False),
        "",
        "## Constraint summary",
        "",
        constraint_summary.to_markdown(index=False),
        "",
        "## Scenario summary",
        "",
        fusion_scenario.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- Constraint is validation-only; test deltas are descriptive.",
        "- The V6 test has already been inspected; the result is exploratory.",
        "- The result does not replace the frozen V6 main table.",
        "",
        f"- Elapsed seconds: `{time.perf_counter() - started:.2f}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection_counts": payload["selection_counts"],
                "fallback_count": payload["fallback_count"],
                "run_summary": payload["run_summary"],
                "gate_summary": payload["gate_summary"],
                "constraint_summary": payload["constraint_summary"],
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
