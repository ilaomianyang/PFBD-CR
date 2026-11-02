"""Stage 490: validation-only score-level fusion of Trust union and MLP.

The fusion weight is selected on validation-normal and validation-attack rows
only.  Test labels are used only for final exploratory evaluation.  The
result does not replace the frozen V6 main table.
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

FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

UNION_SCORES = TABLES / "stage490_union_scores.parquet"
FORMAL_SCORES = TABLES / "stage490_formal_mlp_scores.parquet"
EXISTING_SCENARIO = TABLES / "final_v6_scenario_aggregate.csv"
FORMAL_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"

OUT_RUNS = TABLES / "stage490_score_fusion_runs.csv"
OUT_SCENARIO = TABLES / "stage490_score_fusion_scenario.csv"
OUT_STATION = TABLES / "stage490_score_fusion_station.csv"
OUT_GATE = TABLES / "stage490_score_fusion_gate.csv"
OUT_JSON = REPORTS / "stage490_score_fusion.json"
OUT_MD = REPORTS / "stage490_score_fusion.md"

PRIMARY_FAR = 0.0065
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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


def robust_z(values: np.ndarray, normal: np.ndarray) -> np.ndarray:
    reference = np.asarray(values, dtype=float)[normal]
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        raise RuntimeError("no valid-normal reference scores")
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    scale = max(1.4826 * mad, 1e-6)
    return np.nan_to_num((values - median) / scale)


def canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": frame["station_id_union"].astype(str),
            "attack_scenario": frame[
                "attack_scenario_union"
            ].astype(str),
            "attack_event_id": frame[
                "attack_event_id_union"
            ].astype(str),
            "is_attack": frame["is_attack_union"].astype(bool),
            "observation_present": frame[
                "observation_present_union"
            ].astype(bool),
        }
    ).reset_index(drop=True)


def candidate_scores(
    union_z: np.ndarray,
    formal_z: np.ndarray,
    weight: float | str,
) -> np.ndarray:
    if weight == "max":
        return np.maximum(union_z, formal_z)
    value = float(weight)
    return value * union_z + (1.0 - value) * formal_z


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

    run_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
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
        valid_union_median = float(
            np.median(union_valid_raw[valid_normal])
        )
        valid_formal_median = float(
            np.median(formal_valid_raw[valid_normal])
        )
        valid_union_scale = max(
            1.4826
            * float(
                np.median(
                    np.abs(
                        union_valid_raw[valid_normal]
                        - valid_union_median
                    )
                )
            ),
            1e-6,
        )
        valid_formal_scale = max(
            1.4826
            * float(
                np.median(
                    np.abs(
                        formal_valid_raw[valid_normal]
                        - valid_formal_median
                    )
                )
            ),
            1e-6,
        )
        union_valid_z = (
            union_valid_raw - valid_union_median
        ) / valid_union_scale
        formal_valid_z = (
            formal_valid_raw - valid_formal_median
        ) / valid_formal_scale
        union_test_z = (
            union_test_raw - valid_union_median
        ) / valid_union_scale
        formal_test_z = (
            formal_test_raw - valid_formal_median
        ) / valid_formal_scale

        best: dict[str, object] | None = None
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
            row_metrics = metrics(
                valid_frame["is_attack"].astype(bool).to_numpy(),
                predicted_valid,
                fused_valid,
            )
            event_values = event_metrics(valid_frame, predicted_valid)
            candidate = {
                "weight": weight,
                "threshold": threshold,
                "valid_f1": float(row_metrics["f1"]),
                "valid_event_coverage": float(
                    event_values["event_coverage"]
                ),
                "valid_false_alarm_rate": float(
                    row_metrics["false_alarm_rate"]
                ),
            }
            if best is None or (
                candidate["valid_event_coverage"],
                candidate["valid_f1"],
                -candidate["valid_false_alarm_rate"],
            ) > (
                best["valid_event_coverage"],
                best["valid_f1"],
                -best["valid_false_alarm_rate"],
            ):
                best = candidate
        assert best is not None
        fused_test = candidate_scores(
            union_test_z,
            formal_test_z,
            best["weight"],
        )
        predicted_test = (
            test_frame["observation_present"].to_numpy(dtype=bool)
            & (fused_test >= float(best["threshold"]))
        )
        row_metrics = metrics(
            test_frame["is_attack"].astype(bool).to_numpy(),
            predicted_test,
            fused_test,
        )
        event_values = event_metrics(test_frame, predicted_test)
        selections.append(
            {
                "source": source,
                "seed": int(seed),
                **best,
            }
        )
        run_rows.append(
            {
                "source": source,
                "seed": int(seed),
                "method": "validation_score_fusion",
                **best,
                **row_metrics,
                **event_values,
            }
        )
        scenario_rows.extend(
            s470.scenario_rows_full(
                test_frame,
                predicted_test,
                fused_test,
                source,
                int(seed),
                "validation_score_fusion",
            )
        )
        station_rows.extend(
            s470.station_rows_full(
                test_frame,
                predicted_test,
                source,
                int(seed),
                "validation_score_fusion",
            )
        )

    runs = pd.DataFrame(run_rows)
    scenario = pd.DataFrame(scenario_rows)
    station = pd.DataFrame(station_rows)
    selections_frame = pd.DataFrame(selections)
    runs.to_csv(OUT_RUNS, index=False, encoding="utf-8-sig")
    scenario.to_csv(OUT_SCENARIO, index=False, encoding="utf-8-sig")
    station.to_csv(OUT_STATION, index=False, encoding="utf-8-sig")
    selections_frame.to_csv(
        TABLES / "stage490_score_fusion_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )

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
            ).assign(method="validation_score_fusion")[
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
            group["method"].eq("validation_score_fusion")
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
    payload = {
        "status": "PASS_EXPLORATORY_VALIDATION_SCORE_FUSION",
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "primary_far": PRIMARY_FAR,
        "weight_grid": WEIGHT_GRID,
        "selection_counts": selections_frame["weight"].value_counts().to_dict(),
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
        "scenario_summary": fusion_scenario.to_dict("records"),
        "gate_summary": gate_summary.to_dict("records"),
        "limitations": [
            "Fusion weights are selected on validation only.",
            "The V6 test has already been inspected; this is exploratory.",
            "The result does not replace the frozen V6 main table.",
            "The formal component is a lagged-window MLP, not a GRU/LSTM.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Validation-only score fusion",
        "",
        f"- Status: `{payload['status']}`",
        f"- Weight grid: `{WEIGHT_GRID}`",
        f"- Selection counts: `{payload['selection_counts']}`",
        f"- Run summary: `{payload['run_summary']}`",
        "",
        "## Gate summary",
        "",
        gate_summary.to_markdown(index=False),
        "",
        "## Scenario summary",
        "",
        fusion_scenario.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- Fusion weights are validation-selected only.",
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
                "run_summary": payload["run_summary"],
                "gate_summary": payload["gate_summary"],
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
