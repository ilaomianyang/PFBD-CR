"""Stage 486: formal lagged-window MLP sequence predictor baseline.

This is an exploratory, same-contract baseline for the existing V6 providers.
It uses only past observations and missingness flags to predict the current
observation.  The model is trained on train-normal rows, the alarm threshold
is calibrated on valid-normal rows, and the frozen V6 test attacks are used
for evaluation only.  It is not a GRU/LSTM and is not part of the frozen
confirmatory package.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage340_v4_route_locked_evaluation as base  # noqa: E402
import stage470_final_v6_submission_rebuild as s470  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    detector_frame,
    metrics,
)
from stage208_dwd_persistent_repeat_protection import (  # noqa: E402
    event_metrics,
    far_threshold,
)
from stage240_nonstuck_balanced_selection import (  # noqa: E402
    PROTOCOLS,
    make_frame,
)

FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"
PROTOCOL_PATH = (
    ROOT
    / "confirmatory_package"
    / "simulated_attack_v6"
    / "protocol_freeze.json"
)

OUT_RUNS = TABLES / "stage486_formal_lagged_mlp_runs.csv"
OUT_SCENARIO = TABLES / "stage486_formal_lagged_mlp_scenario.csv"
OUT_STATION = TABLES / "stage486_formal_lagged_mlp_station.csv"
OUT_COMPARISON = TABLES / "stage486_formal_lagged_mlp_comparison.csv"
OUT_JSON = REPORTS / "stage486_formal_lagged_mlp_baseline.json"
OUT_MD = REPORTS / "stage486_formal_lagged_mlp_baseline.md"
OUT_SCORES = TABLES / "stage490_formal_mlp_scores.parquet"

VALID_OFFSET = 1010001
TEST_OFFSET = 1020001
PRIMARY_FAR = 0.0065
SEQUENCE_LENGTH = 12
MAX_TRAIN_WINDOWS = 30000
MODEL_SEED = 3407


def station_stats(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    train = frame[
        frame["split"].eq("train") & frame["observation_present"]
    ].copy()
    stats: dict[str, tuple[float, float]] = {}
    for station, group in train.groupby("station_id", sort=True):
        values = pd.to_numeric(
            group["temperature_c"],
            errors="coerce",
        ).dropna()
        if values.empty:
            stats[str(station)] = (0.0, 1.0)
            continue
        median = float(values.median())
        mad = float((values - median).abs().median())
        stats[str(station)] = (median, max(1.4826 * mad, 0.25))
    return stats


def sequence_matrix(
    frame: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.zeros(
        (len(frame), 2 * SEQUENCE_LENGTH),
        dtype=np.float32,
    )
    target = np.zeros(len(frame), dtype=np.float32)
    available = np.zeros(len(frame), dtype=bool)
    for station, part in frame.groupby("station_id", sort=True):
        station_id = str(station)
        median, scale = stats.get(station_id, (0.0, 1.0))
        indices = part.index.to_numpy(dtype=int)
        values = pd.to_numeric(
            part["temperature_c"],
            errors="coerce",
        ).to_numpy(dtype=float)
        observed = part["observation_present"].to_numpy(dtype=bool)
        usable = observed & np.isfinite(values)
        normalized = np.where(
            usable,
            (values - median) / scale,
            0.0,
        )
        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        missing = (~usable).astype(np.float32)
        target[indices] = normalized
        if len(part) <= SEQUENCE_LENGTH:
            continue
        for position in range(SEQUENCE_LENGTH, len(part)):
            row = indices[position]
            start = position - SEQUENCE_LENGTH
            features[row, :SEQUENCE_LENGTH] = normalized[start:position]
            features[row, SEQUENCE_LENGTH:] = missing[start:position]
            available[row] = True
    return features, target, available


def fit_model(
    features: np.ndarray,
    target: np.ndarray,
    available: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[MLPRegressor, dict[str, object]]:
    train_rows = np.flatnonzero(available & train_mask)
    if train_rows.size > MAX_TRAIN_WINDOWS:
        rng = np.random.default_rng(MODEL_SEED)
        train_rows = np.sort(
            rng.choice(
                train_rows,
                size=MAX_TRAIN_WINDOWS,
                replace=False,
            )
        )
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=MODEL_SEED,
    )
    model.fit(features[train_rows], target[train_rows])
    metadata = {
        "train_windows_available": int(train_rows.size),
        "max_train_windows": int(MAX_TRAIN_WINDOWS),
        "sequence_length": int(SEQUENCE_LENGTH),
        "hidden_layer_sizes": [32, 16],
        "max_iter": 60,
        "model_seed": int(MODEL_SEED),
        "n_iter": int(model.n_iter_),
        "loss": float(model.loss_),
    }
    return model, metadata


def score_frame(
    model: MLPRegressor,
    features: np.ndarray,
    target: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(len(target), dtype=float)
    rows = np.flatnonzero(available)
    if rows.size:
        prediction = model.predict(features[rows])
        scores[rows] = np.abs(target[rows] - prediction)
    return scores


def main() -> int:
    started = time.perf_counter()
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    scenarios = {
        str(row["name"]): row for row in frozen["attack_scenarios"]
    }
    run_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []

    for source, protocol_path in PROTOCOLS.items():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        frame = detector_frame(make_frame(source, protocol))
        stats = station_stats(frame)
        features, target, available = sequence_matrix(frame, stats)
        train_normal = (
            frame["split"].eq("train")
            & frame["observation_present"]
        ).to_numpy(dtype=bool)
        model, metadata = fit_model(
            features,
            target,
            available,
            train_normal,
        )
        fit_rows.append({"source": source, **metadata})
        print(
            json.dumps(
                {
                    "event": "provider_model_fit",
                    "source": source,
                    **metadata,
                }
            ),
            flush=True,
        )
        for seed in base.SEEDS:
            valid_blocks = base.simbase.make_simulated_blocks(
                frame,
                frozen,
                seed,
                "valid",
                VALID_OFFSET,
            )
            test_blocks = base.simbase.make_simulated_blocks(
                frame,
                frozen,
                seed,
                "test",
                TEST_OFFSET,
            )
            valid = base.simbase.inject_simulated_attack(
                frame,
                valid_blocks,
                scenarios,
            )
            test = base.simbase.inject_simulated_attack(
                frame,
                test_blocks,
                scenarios,
            )
            valid_features, valid_target, valid_available = sequence_matrix(
                valid,
                stats,
            )
            test_features, test_target, test_available = sequence_matrix(
                test,
                stats,
            )
            valid_scores = score_frame(
                model,
                valid_features,
                valid_target,
                valid_available,
            )
            test_scores = score_frame(
                model,
                test_features,
                test_target,
                test_available,
            )
            valid_mask = valid["split"].eq("valid").to_numpy(dtype=bool)
            test_mask = test["split"].eq("test").to_numpy(dtype=bool)
            valid_frame = valid.loc[valid_mask].reset_index(drop=True)
            test_frame = test.loc[test_mask].reset_index(drop=True)
            valid_scores = valid_scores[valid_mask]
            test_scores = test_scores[test_mask]
            valid_normal = (
                valid_frame["observation_present"]
                & ~valid_frame["is_attack"]
            ).to_numpy(dtype=bool)
            threshold = float(
                far_threshold(
                    valid_scores,
                    valid_normal,
                    PRIMARY_FAR,
                )
            )
            valid_predicted = (
                valid_frame["observation_present"].to_numpy(dtype=bool)
                & (valid_scores >= threshold)
            )
            valid_row_metrics = metrics(
                valid_frame["is_attack"].astype(bool).to_numpy(),
                valid_predicted,
                valid_scores,
            )
            valid_event_values = event_metrics(
                valid_frame,
                valid_predicted,
            )
            predicted = (
                test_frame["observation_present"].to_numpy(dtype=bool)
                & (test_scores >= threshold)
            )
            score_frames.append(
                pd.DataFrame(
                    {
                        "source": source,
                        "seed": int(seed),
                        "split": "valid",
                        "row_index": np.arange(len(valid_frame)),
                        "station_id": valid_frame["station_id"].astype(str),
                        "attack_scenario": valid_frame[
                            "attack_scenario"
                        ].astype(str),
                        "attack_event_id": valid_frame[
                            "attack_event_id"
                        ].astype(str),
                        "is_attack": valid_frame["is_attack"].astype(bool),
                        "observation_present": valid_frame[
                            "observation_present"
                        ].astype(bool),
                        "formal_score": valid_scores,
                        "formal_prediction": valid_predicted,
                    }
                )
            )
            score_frames.append(
                pd.DataFrame(
                    {
                        "source": source,
                        "seed": int(seed),
                        "split": "test",
                        "row_index": np.arange(len(test_frame)),
                        "station_id": test_frame["station_id"].astype(str),
                        "attack_scenario": test_frame[
                            "attack_scenario"
                        ].astype(str),
                        "attack_event_id": test_frame[
                            "attack_event_id"
                        ].astype(str),
                        "is_attack": test_frame["is_attack"].astype(bool),
                        "observation_present": test_frame[
                            "observation_present"
                        ].astype(bool),
                        "formal_score": test_scores,
                        "formal_prediction": predicted,
                    }
                )
            )
            row_metrics = metrics(
                test_frame["is_attack"].astype(bool).to_numpy(),
                predicted,
                test_scores,
            )
            event_values = event_metrics(test_frame, predicted)
            run_rows.append(
                {
                    "source": source,
                    "seed": int(seed),
                    "method": "formal_lagged_mlp",
                    "threshold": threshold,
                    "valid_normal_rows": int(valid_normal.sum()),
                    "test_rows": int(len(test_frame)),
                    "valid_f1": float(valid_row_metrics["f1"]),
                    "valid_event_coverage": float(
                        valid_event_values["event_coverage"]
                    ),
                    "valid_false_alarm_rate": float(
                        valid_row_metrics["false_alarm_rate"]
                    ),
                    **row_metrics,
                    **event_values,
                }
            )
            scenario_rows.extend(
                s470.scenario_rows_full(
                    test_frame,
                    predicted,
                    test_scores,
                    source,
                    int(seed),
                    "formal_lagged_mlp",
                )
            )
            station_rows.extend(
                s470.station_rows_full(
                    test_frame,
                    predicted,
                    source,
                    int(seed),
                    "formal_lagged_mlp",
                )
            )
            print(
                json.dumps(
                    {
                        "event": "seed_complete",
                        "source": source,
                        "seed": int(seed),
                    }
                ),
                flush=True,
            )

    runs = pd.DataFrame(run_rows)
    scenario = pd.DataFrame(scenario_rows)
    station = pd.DataFrame(station_rows)
    runs.to_csv(OUT_RUNS, index=False, encoding="utf-8-sig")
    scenario.to_csv(OUT_SCENARIO, index=False, encoding="utf-8-sig")
    station.to_csv(OUT_STATION, index=False, encoding="utf-8-sig")
    pd.concat(score_frames, ignore_index=True).to_parquet(
        OUT_SCORES,
        index=False,
    )

    comparison_rows: list[dict[str, object]] = []
    existing_scenario = pd.read_csv(
        TABLES / "final_v6_scenario_metrics.csv"
    )
    formal_scenario = (
        scenario.groupby(["scenario"], as_index=False)
        .agg(
            formal_event_coverage=("event_coverage", "mean"),
            formal_f1=("f1", "mean"),
        )
    )
    for method in sorted(existing_scenario["method"].unique()):
        existing = (
            existing_scenario[existing_scenario["method"].eq(method)]
            .groupby("scenario", as_index=False)
            .agg(
                existing_event_coverage=("event_coverage", "mean"),
                existing_f1=("f1", "mean"),
            )
        )
        merged = formal_scenario.merge(existing, on="scenario")
        comparison_rows.append(
            {
                "method": method,
                "scenario_count": int(len(merged)),
                "formal_event_coverage_wins": int(
                    (
                        merged["formal_event_coverage"]
                        > merged["existing_event_coverage"]
                    ).sum()
                ),
                "formal_f1_wins": int(
                    (merged["formal_f1"] > merged["existing_f1"]).sum()
                ),
                "formal_event_coverage_mean_difference": float(
                    (
                        merged["formal_event_coverage"]
                        - merged["existing_event_coverage"]
                    ).mean()
                ),
                "formal_f1_mean_difference": float(
                    (merged["formal_f1"] - merged["existing_f1"]).mean()
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT_COMPARISON, index=False, encoding="utf-8-sig")

    payload = {
        "status": "PASS_EXPLORATORY_FORMAL_LAGGED_MLP_BASELINE",
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "protocol_id": frozen["protocol_id"],
        "method": "formal_lagged_mlp",
        "sequence_length": SEQUENCE_LENGTH,
        "primary_far": PRIMARY_FAR,
        "providers": list(PROTOCOLS),
        "seed_count": len(base.SEEDS),
        "run_count": int(len(runs)),
        "fit_metadata": fit_rows,
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
        "comparison": comparison.to_dict("records"),
        "limitations": [
            "This is a lagged-window MLP, not a GRU/LSTM or Transformer.",
            "V6 test has already been inspected; this baseline is exploratory.",
            "It uses the same providers, splits, attacks, FAR calibration, and metrics as V6.",
            "It does not replace the frozen main table or create a new confirmatory test.",
            "No provider or dataset was added.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage486 formal lagged-window MLP baseline",
        "",
        f"- Status: `{payload['status']}`",
        f"- Exploratory only: `{payload['exploratory_only']}`",
        f"- Sequence length: `{SEQUENCE_LENGTH}`",
        f"- Primary FAR: `{PRIMARY_FAR}`",
        f"- Runs: `{payload['run_count']}`",
        "",
        "## Run summary",
        "",
        pd.DataFrame([payload["run_summary"]]).to_markdown(index=False),
        "",
        "## Scenario-level comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- This is a formal trained lagged-window MLP, not a GRU/LSTM.",
        "- It is exploratory because the V6 test was already inspected.",
        "- It does not replace the frozen V6 main table.",
        "",
        f"- Elapsed seconds: `{time.perf_counter() - started:.2f}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "runs": payload["run_count"],
                "summary": payload["run_summary"],
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
