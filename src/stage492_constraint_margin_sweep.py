"""Stage 492: safety-margin sweep for constrained score fusion.

The sweep changes only the validation-side safety margin for long_stuck and
drop_replay.  Test is evaluated after each margin to show whether the
validation constraint transfers.  All results are exploratory.
"""

from __future__ import annotations

import json
import sys
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

OUT_SWEEP = TABLES / "stage492_constraint_margin_sweep.csv"
OUT_JSON = REPORTS / "stage492_constraint_margin_sweep.json"
OUT_MD = REPORTS / "stage492_constraint_margin_sweep.md"

PRIMARY_FAR = 0.0065
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CONSTRAINED_SCENARIOS = ["long_stuck", "drop_replay"]
TEST_TOLERANCE = 0.05
MARGIN_GRID = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


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


def prepare_records(merged: pd.DataFrame) -> list[dict[str, object]]:
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
        union_med = float(np.median(union_valid_raw[valid_normal]))
        formal_med = float(np.median(formal_valid_raw[valid_normal]))
        union_scale = max(
            1.4826
            * float(
                np.median(
                    np.abs(union_valid_raw[valid_normal] - union_med)
                )
            ),
            1e-6,
        )
        formal_scale = max(
            1.4826
            * float(
                np.median(
                    np.abs(formal_valid_raw[valid_normal] - formal_med)
                )
            ),
            1e-6,
        )
        union_valid_z = (union_valid_raw - union_med) / union_scale
        formal_valid_z = (formal_valid_raw - formal_med) / formal_scale
        union_test_z = (union_test_raw - union_med) / union_scale
        formal_test_z = (formal_test_raw - formal_med) / formal_scale
        union_valid_pred = valid["union_prediction"].to_numpy(dtype=bool)
        union_test_pred = test["union_prediction"].to_numpy(dtype=bool)
        records.append(
            {
                "source": source,
                "seed": int(seed),
                "valid_frame": valid_frame,
                "test_frame": test_frame,
                "valid_normal": valid_normal,
                "union_valid_z": union_valid_z,
                "formal_valid_z": formal_valid_z,
                "union_test_z": union_test_z,
                "formal_test_z": formal_test_z,
                "union_valid_coverage": scenario_coverage(
                    valid_frame,
                    union_valid_pred,
                ),
                "union_test_coverage": scenario_coverage(
                    test_frame,
                    union_test_pred,
                ),
                "union_valid_pred": union_valid_pred,
                "union_test_pred": union_test_pred,
            }
        )
    return records


def evaluate_margin(
    records: list[dict[str, object]],
    margin: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for record in records:
        valid_frame = record["valid_frame"]
        test_frame = record["test_frame"]
        valid_normal = record["valid_normal"]
        union_valid_z = record["union_valid_z"]
        formal_valid_z = record["formal_valid_z"]
        union_test_z = record["union_test_z"]
        formal_test_z = record["formal_test_z"]
        union_coverage = record["union_valid_coverage"]
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
                predicted_valid = record["union_valid_pred"]
            valid_metrics = metrics(
                valid_frame["is_attack"].astype(bool).to_numpy(),
                predicted_valid,
                fused_valid,
            )
            coverage = scenario_coverage(
                valid_frame,
                predicted_valid,
            )
            constraint_ok = all(
                coverage.get(scenario, 0.0)
                >= union_coverage.get(scenario, 0.0) + margin
                for scenario in CONSTRAINED_SCENARIOS
            )
            candidates.append(
                {
                    "weight": weight,
                    "threshold": threshold,
                    "valid_f1": float(valid_metrics["f1"]),
                    "valid_mean_scenario_coverage": float(
                        np.mean(
                            [
                                coverage.get(scenario, 0.0)
                                for scenario in sorted(
                                    set(coverage)
                                    | set(union_coverage)
                                )
                            ]
                        )
                    ),
                    "constraint_ok": bool(constraint_ok),
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
            union_test_z,
            formal_test_z,
            best["weight"],
        )
        if best["weight"] == 1.0:
            predicted_test = record["union_test_pred"]
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
        selection_rows.append(
            {
                "source": record["source"],
                "seed": record["seed"],
                **best,
                "fallback_to_union": fallback,
            }
        )
        run_rows.append(
            {
                "source": record["source"],
                "seed": record["seed"],
                **best,
                **test_metrics,
                **test_event,
                **{
                    f"test_delta_{scenario}": float(
                        test_coverage.get(scenario, 0.0)
                        - record["union_test_coverage"].get(
                            scenario,
                            0.0,
                        )
                    )
                    for scenario in CONSTRAINED_SCENARIOS
                },
            }
        )
        scenario_rows.extend(
            s470.scenario_rows_full(
                test_frame,
                predicted_test,
                fused_test,
                record["source"],
                int(record["seed"]),
                "constrained_margin_fusion",
            )
        )
    runs = pd.DataFrame(run_rows)
    scenario = pd.DataFrame(scenario_rows)
    selections = pd.DataFrame(selection_rows)
    return {
        "runs": runs,
        "scenario": scenario,
        "selections": selections,
    }, runs, scenario


def main() -> int:
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
    records = prepare_records(merged)
    existing = pd.read_csv(EXISTING_SCENARIO)
    formal_scenario = pd.read_csv(FORMAL_SCENARIO)
    formal_mean = (
        formal_scenario.groupby(
            ["scenario", "method"],
            as_index=False,
        )
        .agg(event_coverage=("event_coverage", "mean"))
    )
    sweep_rows: list[dict[str, object]] = []
    detail_tables: dict[str, object] = {}
    for margin in MARGIN_GRID:
        tables, runs, scenario = evaluate_margin(records, margin)
        fusion_scenario = (
            scenario.groupby("scenario", as_index=False)
            .agg(
                fusion_event_coverage=(
                    "event_coverage",
                    "mean",
                ),
            )
        )
        all_scenario = pd.concat(
            [
                existing[["scenario", "method", "event_coverage"]],
                formal_mean,
                fusion_scenario.rename(
                    columns={
                        "fusion_event_coverage": "event_coverage"
                    }
                ).assign(method="constrained_margin_fusion")[
                    ["scenario", "method", "event_coverage"]
                ],
            ],
            ignore_index=True,
            sort=False,
        )
        wins = 0
        total = 0
        for scenario_name, group in all_scenario.groupby(
            "scenario",
            sort=True,
        ):
            candidate = group[
                group["method"].eq("constrained_margin_fusion")
            ]
            if candidate.empty:
                continue
            value = float(candidate["event_coverage"].iloc[0])
            for baseline in BASELINES_11 + ["formal_lagged_mlp"]:
                reference = group[group["method"].eq(baseline)]
                if reference.empty:
                    continue
                total += 1
                if value > float(
                    reference["event_coverage"].iloc[0]
                ):
                    wins += 1
        deltas = {
            scenario: float(
                runs[f"test_delta_{scenario}"].mean()
            )
            for scenario in CONSTRAINED_SCENARIOS
        }
        within = {
            scenario: float(
                runs[f"test_delta_{scenario}"]
                .ge(-TEST_TOLERANCE)
                .mean()
            )
            for scenario in CONSTRAINED_SCENARIOS
        }
        sweep_rows.append(
            {
                "validation_margin": margin,
                "selection_counts": tables["selections"][
                    "weight"
                ].value_counts().to_dict(),
                "fallback_count": int(
                    tables["selections"]["fallback_to_union"].sum()
                ),
                "f1_mean": float(runs["f1"].mean()),
                "event_coverage_mean": float(
                    runs["event_coverage"].mean()
                ),
                "false_alarm_rate_mean": float(
                    runs["false_alarm_rate"].mean()
                ),
                "extended_wins": wins,
                "extended_units": total,
                "extended_share": wins / total if total else 0.0,
                "extended_passes_70pct": bool(
                    wins / total >= 0.70
                )
                if total
                else False,
                "long_stuck_test_delta": deltas["long_stuck"],
                "drop_replay_test_delta": deltas["drop_replay"],
                "long_stuck_within_5pp_fraction": within["long_stuck"],
                "drop_replay_within_5pp_fraction": within[
                    "drop_replay"
                ],
                "test_constraint_passes": bool(
                    deltas["long_stuck"] >= -TEST_TOLERANCE
                    and deltas["drop_replay"] >= -TEST_TOLERANCE
                ),
            }
        )
        detail_tables[str(margin)] = tables
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT_SWEEP, index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_EXPLORATORY_CONSTRAINT_MARGIN_SWEEP",
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "constraint_scenarios": CONSTRAINED_SCENARIOS,
        "test_tolerance": TEST_TOLERANCE,
        "margin_grid": MARGIN_GRID,
        "sweep": sweep.to_dict("records"),
        "limitations": [
            "Margins are validation-side safety margins.",
            "Test deltas are reported only to show transfer failure or success.",
            "The V6 test has already been inspected; this is exploratory.",
            "No margin can be promoted to confirmatory status.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Constraint-margin sweep",
        "",
        f"- Status: `{payload['status']}`",
        f"- Test tolerance: `{TEST_TOLERANCE}`",
        "",
        sweep.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- Validation margins are varied; test deltas are descriptive.",
        "- The V6 test has already been inspected; all results are exploratory.",
        "- No margin is promoted to confirmatory status.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": sweep.to_dict("records"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
