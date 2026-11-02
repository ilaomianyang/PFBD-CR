"""Stage 499: NOAA ISD representation-layer improvement experiment (v2).

Motivation
----------
Stages 494/497 showed that NOAA ISD is the weakest of the four providers:
its subtle-scenario event coverage is close to zero and station-level
calibration could not repair it.  Stage 499 asks a different question: is
the NOAA deficit a *representation* problem that can be reduced by using
the existing NOAA ISD series through better causal features, formal
regressors, station-specific residual models, and a validation-only
constrained fusion with the existing union/formal scores?

Protocol
--------
* Only the four existing providers and the existing frozen V6 data are
  used.  No provider, dataset, or confirmatory test is added and the
  frozen main results are not modified.
* Representation features are strictly causal: every feature of row ``t``
  is a function of values up to ``t-1`` plus the timestamp of row ``t``.
  A perturbation check verifies that editing the value of one row can
  never change features of the same or an earlier row.
* Regressors (HistGradientBoostingRegressor, MLPRegressor, per-station
  HistGradientBoostingRegressor) are fitted on NOAA train-normal rows
  only.  Score standardisation uses train-normal residuals only.
* Every alarm threshold and every budget split is calibrated on NOAA
  validation-normal rows only; the structured-coverage floor uses NOAA
  validation event coverage only.
* Constraint: for every seed the validation event coverage of
  ``long_stuck`` and ``drop_replay`` must stay at or above the frozen
  union OR coverage (margin 0).  A family is deployable only if it also
  respects the frozen operating point (validation FAR <= 0.0065).  If no
  representation family is deployable the NOAA prediction falls back to
  the frozen union OR decision.
* Test rows are evaluation only.  They never enter a threshold, a budget
  split, a family selection, or a fallback decision.

The V6 test split has already been inspected in earlier stages, so every
number produced here is exploratory and does not replace the frozen main
table.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Deterministic single-threaded numeric libraries: the sandbox forbids the
# worker-pool handles that joblib creates for multi-threaded histogram
# binning, and single-threaded execution is also fully reproducible.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage340_v4_route_locked_evaluation as base  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    detector_frame,
    metrics,
)
from stage208_dwd_persistent_repeat_protection import (  # noqa: E402
    event_metrics,
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

UNION_SCORES = TABLES / "stage490_union_scores.parquet"
FORMAL_SCORES = TABLES / "stage490_formal_mlp_scores.parquet"
STAGE491_RUNS = TABLES / "stage491_constrained_fusion_runs.csv"
STAGE491_SCENARIO = TABLES / "stage491_constrained_fusion_scenario.csv"
STAGE491_STATION = TABLES / "stage491_constrained_fusion_station.csv"
STAGE493_PROVIDER = TABLES / "stage493_constrained_fusion_provider_summary.csv"
STAGE493_SEED = TABLES / "stage493_constrained_fusion_seed_summary.csv"
STAGE497_RUNS = TABLES / "stage497_noaa_station_calibration_runs.csv"
STAGE497_SCENARIO = TABLES / "stage497_noaa_station_calibration_scenario.csv"

OUT_RUNS = TABLES / "stage499_noaa_representation_v2_runs.csv"
OUT_SCENARIO = TABLES / "stage499_noaa_representation_v2_scenario.csv"
OUT_STATION = TABLES / "stage499_noaa_representation_v2_station.csv"
OUT_SUMMARY = TABLES / "stage499_noaa_representation_v2_summary.csv"
OUT_JSON = REPORTS / "stage499_noaa_representation_v2.json"
OUT_MD = REPORTS / "stage499_noaa_representation_v2.md"

NOAA = "NOAA ISD"
PRIMARY_FAR = 0.0065
CONSTRAINT_MARGIN = 0.0
STRUCTURED_SCENARIOS = ("long_stuck", "drop_replay")
MODEL_SEED = 3407
VALID_OFFSET = 1010001
TEST_OFFSET = 1020001
EPS = 1.0e-12
NEG = -1.0e18

LAG_LIST = (1, 2, 3, 4, 6, 8, 12, 24, 48)
PAST_DELTA_LAGS = (1, 2, 6, 24)
ROLL_WINDOWS = (3, 6, 12, 24, 48)
DIFF_WINDOWS = (6, 12, 24)

BUDGET_GRID_A = (
    0.001,
    0.0015,
    0.002,
    0.0025,
    0.003,
    0.0035,
    0.004,
    0.0045,
    0.005,
    0.006,
)
BUDGET_GRID_B = (
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.0025,
    0.003,
    0.0035,
    0.004,
    0.0045,
    0.005,
    0.0055,
    0.006,
)
ADDITIONAL_BUDGET_GRID = (0.0005, 0.001, 0.002)
SCORE_FUSION_WEIGHTS = (
    (0.5, 0.25, 0.25),
    (0.34, 0.33, 0.33),
    (0.2, 0.4, 0.4),
    (0.6, 0.2, 0.2),
)


def scenario_coverage(
    frame: pd.DataFrame,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Event coverage per attack scenario (normal rows excluded)."""
    output: dict[str, float] = {}
    for scenario, group in frame.assign(
        _pred=np.asarray(predicted, dtype=bool)
    ).groupby("attack_scenario", sort=True):
        if scenario == "normal":
            continue
        detected = group.groupby("attack_event_id")["_pred"].any()
        output[str(scenario)] = (
            float(detected.mean()) if len(detected) else 0.0
        )
    return output


def mean_scenario_coverage(coverage: dict[str, float]) -> float:
    if not coverage:
        return 0.0
    return float(np.mean(list(coverage.values())))


def robust_median_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median, max(1.4826 * mad, 1.0e-6)


def budget_threshold(
    score: np.ndarray,
    normal_mask: np.ndarray,
    budget: float,
) -> float:
    """Largest threshold whose realised normal alarm rate stays <= budget.

    Unlike a plain quantile this is tie-safe: the returned value is chosen
    on the observed validation-normal score support so that
    ``mean(score >= threshold) <= budget`` holds exactly.
    """
    reference = np.asarray(score, dtype=float)[normal_mask]
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        raise RuntimeError("no validation-normal scores for calibration")
    limit = int(np.floor(float(budget) * reference.size + EPS))
    if limit <= 0:
        return float("inf")
    ascending = np.sort(reference)
    # Counts are non-increasing in the threshold, so the first (smallest)
    # value whose alarm count fits the budget is the most sensitive
    # threshold that still respects it.
    for value in np.unique(ascending):
        count = reference.size - int(
            np.searchsorted(ascending, value, side="left")
        )
        if count <= limit:
            return float(value)
    return float("inf")


def station_stats(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    train = frame[
        frame["split"].eq("train") & frame["observation_present"]
    ]
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


def build_features(
    frame: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    station_codes: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Strictly causal multi-scale lag representation for one provider."""
    names: list[str] = []
    features: np.ndarray | None = None
    target = np.zeros(len(frame), dtype=float)
    available = np.zeros(len(frame), dtype=bool)
    for station, part in frame.groupby("station_id", sort=True):
        median, scale = stats[str(station)]
        values = pd.to_numeric(
            part["temperature_c"],
            errors="coerce",
        ).to_numpy(dtype=float)
        observed = part["observation_present"].to_numpy(dtype=bool)
        usable = observed & np.isfinite(values)
        normalised = np.where(
            usable,
            (values - median) / scale,
            np.nan,
        )
        series = pd.Series(normalised, dtype=float)
        index = part.index.to_numpy(dtype=int)
        target[index] = np.nan_to_num(normalised, nan=0.0)
        available[index] = usable
        filled = series.ffill()
        past = series.shift(1)
        columns: list[tuple[str, np.ndarray]] = []
        for lag in LAG_LIST:
            columns.append(
                (f"lag{lag}", filled.shift(lag).to_numpy(dtype=float))
            )
        for lag in PAST_DELTA_LAGS:
            columns.append(
                (
                    f"past_delta{lag}",
                    (series.shift(1) - series.shift(1 + lag)).to_numpy(
                        dtype=float
                    ),
                )
            )
        for window in ROLL_WINDOWS:
            min_periods = max(2, window // 3)
            rolling = past.rolling(window, min_periods=min_periods)
            columns.append(
                (f"roll_mean{window}", rolling.mean().to_numpy(dtype=float))
            )
            columns.append(
                (f"roll_std{window}", rolling.std().to_numpy(dtype=float))
            )
            columns.append(
                (f"roll_min{window}", rolling.min().to_numpy(dtype=float))
            )
            columns.append(
                (f"roll_max{window}", rolling.max().to_numpy(dtype=float))
            )
        diff = series.diff()
        past_diff = diff.shift(1)
        past_abs = past_diff.abs()
        for window in DIFF_WINDOWS:
            min_periods = max(2, window // 3)
            columns.append(
                (
                    f"diff_std{window}",
                    past_diff.rolling(
                        window,
                        min_periods=min_periods,
                    ).std().to_numpy(dtype=float),
                )
            )
            columns.append(
                (
                    f"abs_diff_mean{window}",
                    past_abs.rolling(
                        window,
                        min_periods=min_periods,
                    ).mean().to_numpy(dtype=float),
                )
            )
            zero = (past_abs < 1e-9).astype(float)
            zero[past_abs.isna().to_numpy()] = np.nan
            columns.append(
                (
                    f"zero_frac{window}",
                    pd.Series(zero).rolling(
                        window,
                        min_periods=min_periods,
                    ).mean().to_numpy(dtype=float),
                )
            )
        missing = (~usable).astype(float)
        columns.append(
            (
                "missing6",
                pd.Series(missing)
                .shift(1)
                .rolling(6, min_periods=1)
                .sum()
                .to_numpy(dtype=float),
            )
        )
        columns.append(
            (
                "missing24",
                pd.Series(missing)
                .shift(1)
                .rolling(24, min_periods=1)
                .sum()
                .to_numpy(dtype=float),
            )
        )
        stamp = pd.to_datetime(part["timestamp_utc"], utc=True)
        gap_hours = (
            stamp.diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
        )
        columns.append(("gap_hours", np.clip(gap_hours, 0.0, 240.0)))
        hours = stamp.dt.hour.to_numpy(dtype=float)
        doy = stamp.dt.dayofyear.to_numpy(dtype=float)
        columns.append(("hour_sin", np.sin(2 * np.pi * hours / 24.0)))
        columns.append(("hour_cos", np.cos(2 * np.pi * hours / 24.0)))
        columns.append(("doy_sin", np.sin(2 * np.pi * doy / 365.25)))
        columns.append(("doy_cos", np.cos(2 * np.pi * doy / 365.25)))
        columns.append(
            (
                "station_code",
                np.full(len(part), float(station_codes[str(station)])),
            )
        )
        if not names:
            names = [name for name, _ in columns]
            features = np.zeros(
                (len(frame), len(names)),
                dtype=np.float32,
            )
        assert features is not None
        features[index] = np.column_stack(
            [
                np.nan_to_num(
                    values_,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                for _, values_ in columns
            ]
        ).astype(np.float32)
    if features is None:
        raise RuntimeError("feature build produced no stations")
    return features, target, available, names


def causality_check(
    frame: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    station_codes: dict[str, int],
) -> dict[str, object]:
    """Perturb one row and verify that only later rows can change."""
    station = sorted(frame["station_id"].astype(str).unique())[0]
    part_index = frame.index[
        frame["station_id"].astype(str).eq(station)
    ].to_numpy(dtype=int)
    if len(part_index) < 500:
        raise RuntimeError("station too short for causality check")
    position = int(part_index[len(part_index) // 2])
    base_features, _, _, names = build_features(
        frame,
        stats,
        station_codes,
    )
    perturbed = frame.copy()
    perturbed.loc[position, "temperature_c"] = (
        float(perturbed.loc[position, "temperature_c"]) + 10.0
    )
    new_features, _, _, _ = build_features(
        perturbed,
        stats,
        station_codes,
    )
    changed = np.any(
        np.abs(base_features - new_features) > 1e-9,
        axis=1,
    )
    earlier_or_same = changed & (frame.index.to_numpy(dtype=int) <= position)
    later = changed & (frame.index.to_numpy(dtype=int) > position)
    same_station_later = later & frame["station_id"].astype(str).eq(
        station
    ).to_numpy()
    other_station = later & ~frame["station_id"].astype(str).eq(
        station
    ).to_numpy()
    return {
        "station": station,
        "row_position": int(position),
        "feature_count": int(len(names)),
        "features_changed_at_or_before_row": int(earlier_or_same.sum()),
        "features_changed_after_row_same_station": int(
            same_station_later.sum()
        ),
        "features_changed_after_row_other_stations": int(
            other_station.sum()
        ),
        "causal": bool(
            int(earlier_or_same.sum()) == 0
            and int(other_station.sum()) == 0
        ),
    }


def fit_global_models(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, object]:
    hgb = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=MODEL_SEED,
    )
    hgb.fit(features[train_mask], target[train_mask])
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=MODEL_SEED,
    )
    mlp.fit(features[train_mask], target[train_mask])
    hgb_train = np.abs(target[train_mask] - hgb.predict(features[train_mask]))
    mlp_train = np.abs(target[train_mask] - mlp.predict(features[train_mask]))
    return {
        "hgb": hgb,
        "mlp": mlp,
        "hgb_train_residual": hgb_train,
        "mlp_train_residual": mlp_train,
        "hgb_n_iter": None,
        "mlp_n_iter": int(mlp.n_iter_),
        "mlp_loss": float(mlp.loss_),
    }


def fit_station_models(
    frame: pd.DataFrame,
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, object]:
    stations = frame["station_id"].astype(str).to_numpy()
    models: dict[str, HistGradientBoostingRegressor] = {}
    residuals: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for station in sorted(set(stations.tolist())):
        mask = train_mask & (stations == station)
        counts[station] = int(mask.sum())
        if counts[station] < 500:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=MODEL_SEED,
        )
        model.fit(features[mask], target[mask])
        models[station] = model
        residuals[station] = np.abs(
            target[mask] - model.predict(features[mask])
        )
    return {"models": models, "residuals": residuals, "counts": counts}


def station_predictions(
    models: dict[str, HistGradientBoostingRegressor],
    features: np.ndarray,
    stations: np.ndarray,
) -> np.ndarray:
    prediction = np.zeros(len(features), dtype=float)
    for station, model in models.items():
        mask = stations == station
        if mask.any():
            prediction[mask] = model.predict(features[mask])
    return prediction


def zscore(
    values: np.ndarray,
    median: float,
    scale: float,
) -> np.ndarray:
    return (np.asarray(values, dtype=float) - float(median)) / float(scale)


def flatness_statistic(
    features: np.ndarray,
    names: list[str],
) -> np.ndarray:
    columns = [
        names.index(f"zero_frac{window}")
        for window in DIFF_WINDOWS
    ]
    return features[:, columns].max(axis=1).astype(float)


def run_length_statistic(
    frame: pd.DataFrame,
    available: np.ndarray,
) -> np.ndarray:
    """Gap-aware length of the current constant-value run.

    NOAA ISD is irregular: consecutive records can be separated by missing
    hours.  The run therefore counts consecutive *observed* records with the
    same value, so a replayed/stuck block is not split by an interior gap.
    Missing rows themselves carry a run of zero and never become alarms.
    """
    values = np.zeros(len(frame), dtype=float)
    usable_all = np.asarray(available, dtype=bool)
    for station, part in frame.groupby("station_id", sort=True):
        index = part.index.to_numpy(dtype=int)
        series = pd.to_numeric(
            part["temperature_c"],
            errors="coerce",
        ).to_numpy(dtype=float)
        usable = usable_all[index]
        output = np.zeros(len(series), dtype=float)
        run = 0.0
        previous = np.nan
        for position in range(len(series)):
            if not usable[position]:
                continue
            if (
                np.isfinite(previous)
                and abs(series[position] - previous) < 1e-9
            ):
                run += 1.0
            else:
                run = 1.0
            output[position] = run
            previous = series[position]
        values[index] = output
    return values


def alarm_metrics(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    score: np.ndarray,
) -> tuple[dict[str, object], dict[str, float]]:
    attack = frame["is_attack"].astype(bool).to_numpy()
    row_metrics = metrics(attack, predicted, score)
    event_values = event_metrics(frame, predicted)
    coverage = scenario_coverage(frame, predicted)
    return {**row_metrics, **event_values}, coverage


def candidate_row(
    label: str,
    kind: str,
    frame: pd.DataFrame,
    predicted: np.ndarray,
    score: np.ndarray,
    union_coverage: dict[str, float],
    extra: dict[str, object],
) -> dict[str, object]:
    row_metrics, coverage = alarm_metrics(frame, predicted, score)
    margins = {
        scenario: float(
            coverage.get(scenario, 0.0)
            - union_coverage.get(scenario, 0.0)
        )
        for scenario in STRUCTURED_SCENARIOS
    }
    return {
        "family": label,
        "family_kind": kind,
        "valid_mean_scenario_coverage": mean_scenario_coverage(coverage),
        "valid_event_coverage": float(row_metrics["event_coverage"]),
        "valid_f1": float(row_metrics["f1"]),
        "valid_false_alarm_rate": float(
            row_metrics["false_alarm_rate"]
        ),
        "valid_average_precision": float(
            row_metrics["average_precision"]
        ),
        "valid_roc_auc": float(row_metrics["roc_auc"]),
        **{
            f"valid_coverage_{scenario}": float(
                coverage.get(scenario, 0.0)
            )
            for scenario in STRUCTURED_SCENARIOS
        },
        **{
            f"valid_margin_{scenario}": margins[scenario]
            for scenario in STRUCTURED_SCENARIOS
        },
        "constraint_ok": bool(
            all(
                margins[scenario] >= -CONSTRAINT_MARGIN - EPS
                for scenario in STRUCTURED_SCENARIOS
            )
        ),
        "coverage_by_scenario": coverage,
        **extra,
    }


def select_budget_split(
    threshold_a_for,
    threshold_b_for,
    predict_for,
    frame: pd.DataFrame,
    normal: np.ndarray,
    score: np.ndarray,
    union_coverage: dict[str, float],
    label: str,
    kind: str,
    far_budget: float,
    extra: dict[str, object],
) -> dict[str, object]:
    """Validation-only search over a two-component FAR budget split.

    ``threshold_a_for``/``threshold_b_for`` map a FAR budget to the alarm
    threshold of component A/B (pooled scalar or per-station array) and
    ``predict_for`` combines the two thresholds into an OR alarm.
    """
    best_feasible: dict[str, object] | None = None
    best_any: dict[str, object] | None = None
    for budget_a in BUDGET_GRID_A:
        threshold_a = threshold_a_for(budget_a)
        for budget_b in BUDGET_GRID_B:
            if budget_a + budget_b > far_budget + EPS:
                continue
            threshold_b = threshold_b_for(budget_b)
            predicted = predict_for(threshold_a, threshold_b)
            candidate = candidate_row(
                label,
                kind,
                frame,
                predicted,
                score,
                union_coverage,
                {
                    **extra,
                    "budget_a": float(budget_a),
                    "budget_b": float(budget_b),
                    "threshold_a": (
                        float(threshold_a)
                        if np.isscalar(threshold_a)
                        else float("nan")
                    ),
                    "threshold_b": (
                        float(threshold_b)
                        if np.isscalar(threshold_b)
                        else float("nan")
                    ),
                    "thresholds_a": (
                        None
                        if np.isscalar(threshold_a)
                        else [float(value) for value in threshold_a]
                    ),
                    "thresholds_b": (
                        None
                        if np.isscalar(threshold_b)
                        else [float(value) for value in threshold_b]
                    ),
                },
            )
            if (
                candidate["valid_false_alarm_rate"]
                > far_budget + 1.0 / max(int(normal.sum()), 1)
            ):
                continue
            key = (
                float(candidate["valid_mean_scenario_coverage"]),
                float(candidate["valid_f1"]),
            )
            if best_any is None or key > best_any["_key"]:
                best_any = {**candidate, "_key": key}
            if candidate["constraint_ok"] and (
                best_feasible is None or key > best_feasible["_key"]
            ):
                best_feasible = {**candidate, "_key": key}
    chosen = best_feasible if best_feasible is not None else best_any
    if chosen is None:
        raise RuntimeError(f"no admissible budget split for {label}")
    chosen.pop("_key", None)
    chosen["budget_search_feasible"] = bool(best_feasible is not None)
    return chosen


def main() -> int:  # noqa: C901 - single exploratory stage script
    started = time.perf_counter()
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    scenarios = {
        str(row["name"]): row for row in frozen["attack_scenarios"]
    }
    protocol = json.loads(PROTOCOLS[NOAA].read_text(encoding="utf-8"))
    frame = detector_frame(make_frame(NOAA, protocol))
    stations_sorted = sorted(frame["station_id"].astype(str).unique())
    station_codes = {
        station: index for index, station in enumerate(stations_sorted)
    }
    stats = station_stats(frame)
    features, target, available, names = build_features(
        frame,
        stats,
        station_codes,
    )
    causality = causality_check(frame, stats, station_codes)
    if not causality["causal"]:
        raise RuntimeError("representation is not causal")
    train_mask = (
        frame["split"].eq("train")
        & frame["observation_present"]
    ).to_numpy(dtype=bool) & available
    print(
        json.dumps(
            {
                "event": "features_built",
                "rows": int(len(frame)),
                "features": int(len(names)),
                "train_normal_rows": int(train_mask.sum()),
                "causality": causality,
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    global_models = fit_global_models(features, target, train_mask)
    print(
        json.dumps(
            {
                "event": "global_models_fitted",
                "mlp_n_iter": global_models["mlp_n_iter"],
                "mlp_loss": global_models["mlp_loss"],
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    station_models = fit_station_models(
        frame,
        features,
        target,
        train_mask,
    )
    print(
        json.dumps(
            {
                "event": "station_models_fitted",
                "stations": len(station_models["models"]),
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    flat_train = flatness_statistic(features, names)[train_mask]
    flat_median, flat_scale = robust_median_scale(flat_train)
    run_all = run_length_statistic(frame, available)
    run_median, run_scale = robust_median_scale(run_all[train_mask])
    run_scale = max(run_scale, 1.0)
    hgb_resid_median, hgb_resid_scale = robust_median_scale(
        global_models["hgb_train_residual"]
    )
    mlp_resid_median, mlp_resid_scale = robust_median_scale(
        global_models["mlp_train_residual"]
    )
    station_flat: dict[str, tuple[float, float]] = {}
    station_run: dict[str, tuple[float, float]] = {}
    station_resid: dict[str, tuple[float, float]] = {}
    stations_train = frame["station_id"].astype(str).to_numpy()
    flat_all = flatness_statistic(features, names)
    for station in stations_sorted:
        mask = train_mask & (stations_train == station)
        if int(mask.sum()) < 500:
            station_flat[station] = (flat_median, flat_scale)
            station_run[station] = (run_median, run_scale)
            station_resid[station] = (
                hgb_resid_median,
                hgb_resid_scale,
            )
            continue
        station_flat[station] = robust_median_scale(flat_all[mask])
        station_run[station] = (
            robust_median_scale(run_all[mask])[0],
            max(robust_median_scale(run_all[mask])[1], 1.0),
        )
        station_resid[station] = robust_median_scale(
            station_models["residuals"][station]
        )

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
    providers = sorted(str(value) for value in merged["source"].unique())
    seeds = sorted(int(value) for value in merged["seed"].unique())
    if providers != sorted(list(PROTOCOLS)):
        raise RuntimeError("provider set differs from the frozen protocol")
    if seeds != sorted(int(value) for value in base.SEEDS):
        raise RuntimeError("seed set differs from the frozen protocol")

    stage491_runs = pd.read_csv(STAGE491_RUNS).set_index(
        ["source", "seed"]
    )
    stage491_scenario = pd.read_csv(STAGE491_SCENARIO).set_index(
        ["source", "seed", "scenario"]
    )
    stage491_station = pd.read_csv(STAGE491_STATION).set_index(
        ["source", "seed", "station_id"]
    )
    stage493_provider = pd.read_csv(STAGE493_PROVIDER).set_index("unit")
    stage493_seed = pd.read_csv(STAGE493_SEED).set_index("unit")
    stage493_provider.index = stage493_provider.index.astype(str)
    stage493_seed.index = stage493_seed.index.astype(str)
    stage497_runs = pd.read_csv(STAGE497_RUNS).set_index(
        ["source", "seed"]
    )
    stage497_scenario = pd.read_csv(STAGE497_SCENARIO)
    stage497_scenario_lookup = stage497_scenario.set_index(
        ["source", "seed", "scenario"]
    )

    candidate_valid: dict[int, dict[str, dict[str, object]]] = {}
    candidate_test: dict[int, dict[str, dict[str, object]]] = {}
    union_state: dict[int, dict[str, object]] = {}
    for seed in seeds:
        group = merged[merged["seed"].eq(seed)]
        noaa_group = group[group["source"].eq(NOAA)]
        for split, offset in (
            ("valid", VALID_OFFSET),
            ("test", TEST_OFFSET),
        ):
            blocks = base.simbase.make_simulated_blocks(
                frame,
                frozen,
                seed,
                split,
                offset,
            )
            injected = base.simbase.inject_simulated_attack(
                frame,
                blocks,
                scenarios,
            )
            split_features, split_target, split_available, _ = (
                build_features(injected, stats, station_codes)
            )
            split_run = run_length_statistic(
                injected,
                split_available,
            )
            mask = injected["split"].eq(split).to_numpy(dtype=bool)
            split_frame = injected.loc[mask].reset_index(drop=True)
            split_features = split_features[mask]
            split_target = split_target[mask]
            split_available = split_available[mask]
            split_run = split_run[mask]
            reference = noaa_group[
                noaa_group["split"].eq(split)
            ].reset_index(drop=True)
            aligned = (
                reference["station_id_union"].astype(str).eq(
                    split_frame["station_id"].astype(str)
                )
                & reference["attack_scenario_union"].astype(str).eq(
                    split_frame["attack_scenario"].astype(str)
                )
                & reference["attack_event_id_union"].astype(str).eq(
                    split_frame["attack_event_id"].astype(str)
                )
                & reference["is_attack_union"].astype(bool).eq(
                    split_frame["is_attack"].astype(bool)
                )
                & reference["observation_present_union"].astype(bool).eq(
                    split_frame["observation_present"].astype(bool)
                )
            )
            if not bool(aligned.all()):
                raise RuntimeError(
                    f"NOAA {split} reconstruction differs from the frozen "
                    "score table"
                )
            if split == "valid":
                union_state[seed] = {
                    "frame": split_frame,
                    "features": split_features,
                    "target": split_target,
                    "available": split_available,
                    "run": split_run,
                    "union_score": reference[
                        "union_score"
                    ].to_numpy(dtype=float),
                    "union_prediction": reference[
                        "union_prediction"
                    ].to_numpy(dtype=bool),
                    "formal_score": reference[
                        "formal_score"
                    ].to_numpy(dtype=float),
                }
            else:
                union_state[seed]["test_frame"] = split_frame
                union_state[seed]["test_features"] = split_features
                union_state[seed]["test_target"] = split_target
                union_state[seed]["test_available"] = split_available
                union_state[seed]["test_run"] = split_run
                union_state[seed]["union_test_score"] = reference[
                    "union_score"
                ].to_numpy(dtype=float)
                union_state[seed]["union_test_prediction"] = reference[
                    "union_prediction"
                ].to_numpy(dtype=bool)
                union_state[seed]["formal_test_score"] = reference[
                    "formal_score"
                ].to_numpy(dtype=float)
        print(
            json.dumps(
                {
                    "event": "noaa_split_ready",
                    "seed": seed,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    for seed in seeds:
        state = union_state[seed]
        valid_frame = state["frame"]
        valid_features = state["features"]
        valid_target = state["target"]
        valid_available = state["available"]
        valid_stations = valid_frame["station_id"].astype(str).to_numpy()
        valid_observed = (
            valid_frame["observation_present"].to_numpy(dtype=bool)
            & valid_available
        )
        valid_attack = valid_frame["is_attack"].to_numpy(dtype=bool)
        valid_normal = valid_observed & ~valid_attack
        union_valid_pred = state["union_prediction"]
        union_valid_coverage = scenario_coverage(
            valid_frame,
            union_valid_pred,
        )
        union_valid_row, _ = alarm_metrics(
            valid_frame,
            union_valid_pred,
            state["union_score"],
        )
        state["union_valid_coverage"] = union_valid_coverage
        state["union_valid_row_metrics"] = union_valid_row
        state["valid_normal"] = valid_normal
        state["valid_observed"] = valid_observed

        test_frame = state["test_frame"]
        test_features = state["test_features"]
        test_target = state["test_target"]
        test_available = state["test_available"]
        test_stations = test_frame["station_id"].astype(str).to_numpy()
        test_observed = (
            test_frame["observation_present"].to_numpy(dtype=bool)
            & test_available
        )
        state["test_observed"] = test_observed
        state["test_stations"] = test_stations

        # --- representation scores (train-normal standardised) ---
        valid_resid_hgb = np.abs(
            valid_target
            - global_models["hgb"].predict(valid_features)
        )
        test_resid_hgb = np.abs(
            test_target
            - global_models["hgb"].predict(test_features)
        )
        valid_resid_mlp = np.abs(
            valid_target
            - global_models["mlp"].predict(valid_features)
        )
        test_resid_mlp = np.abs(
            test_target
            - global_models["mlp"].predict(test_features)
        )
        valid_resid_station = np.abs(
            valid_target
            - station_predictions(
                station_models["models"],
                valid_features,
                valid_stations,
            )
        )
        test_resid_station = np.abs(
            test_target
            - station_predictions(
                station_models["models"],
                test_features,
                test_stations,
            )
        )
        valid_flat = flatness_statistic(valid_features, names)
        test_flat = flatness_statistic(test_features, names)
        valid_run = (state["run"] - run_median) / run_scale
        test_run = (state["test_run"] - run_median) / run_scale

        valid_score_hgb = zscore(
            valid_resid_hgb,
            hgb_resid_median,
            hgb_resid_scale,
        )
        test_score_hgb = zscore(
            test_resid_hgb,
            hgb_resid_median,
            hgb_resid_scale,
        )
        valid_score_mlp = zscore(
            valid_resid_mlp,
            mlp_resid_median,
            mlp_resid_scale,
        )
        test_score_mlp = zscore(
            test_resid_mlp,
            mlp_resid_median,
            mlp_resid_scale,
        )
        valid_score_station = np.zeros(len(valid_frame), dtype=float)
        test_score_station = np.zeros(len(test_frame), dtype=float)
        valid_score_flat_station = np.zeros(len(valid_frame), dtype=float)
        test_score_flat_station = np.zeros(len(test_frame), dtype=float)
        valid_score_run_station = np.zeros(len(valid_frame), dtype=float)
        test_score_run_station = np.zeros(len(test_frame), dtype=float)
        for station in stations_sorted:
            valid_mask = valid_stations == station
            test_mask = test_stations == station
            resid_median, resid_scale = station_resid[station]
            flat_median_s, flat_scale_s = station_flat[station]
            run_median_s, run_scale_s = station_run[station]
            if valid_mask.any():
                valid_score_station[valid_mask] = zscore(
                    valid_resid_station[valid_mask],
                    resid_median,
                    resid_scale,
                )
                valid_score_flat_station[valid_mask] = zscore(
                    valid_flat[valid_mask],
                    flat_median_s,
                    flat_scale_s,
                )
                valid_score_run_station[valid_mask] = (
                    valid_run[valid_mask] - run_median_s
                ) / run_scale_s
            if test_mask.any():
                test_score_station[test_mask] = zscore(
                    test_resid_station[test_mask],
                    resid_median,
                    resid_scale,
                )
                test_score_flat_station[test_mask] = zscore(
                    test_flat[test_mask],
                    flat_median_s,
                    flat_scale_s,
                )
                test_score_run_station[test_mask] = (
                    test_run[test_mask] - run_median_s
                ) / run_scale_s
        valid_score_flat = zscore(valid_flat, flat_median, flat_scale)
        test_score_flat = zscore(test_flat, flat_median, flat_scale)
        valid_stuck = np.maximum(valid_run, valid_score_flat)
        test_stuck = np.maximum(test_run, test_score_flat)
        valid_stuck_station = np.maximum(
            valid_score_run_station,
            valid_score_flat_station,
        )
        test_stuck_station = np.maximum(
            test_score_run_station,
            test_score_flat_station,
        )
        valid_repr = np.maximum(valid_score_hgb, valid_stuck)
        test_repr = np.maximum(test_score_hgb, test_stuck)
        state["valid_scores"] = {
            "resid_hgb": valid_score_hgb,
            "resid_mlp": valid_score_mlp,
            "resid_station": valid_score_station,
            "flat": valid_score_flat,
            "flat_station": valid_score_flat_station,
            "run": valid_run,
            "run_station": valid_score_run_station,
            "stuck": valid_stuck,
            "stuck_station": valid_stuck_station,
            "repr": valid_repr,
        }
        state["test_scores"] = {
            "resid_hgb": test_score_hgb,
            "resid_mlp": test_score_mlp,
            "resid_station": test_score_station,
            "flat": test_score_flat,
            "flat_station": test_score_flat_station,
            "run": test_run,
            "run_station": test_score_run_station,
            "stuck": test_stuck,
            "stuck_station": test_stuck_station,
            "repr": test_repr,
        }
        state["run_saturation"] = {
            station: {
                "valid_normal_rows": int(
                    (
                        valid_normal & (valid_stations == station)
                    ).sum()
                ),
                "run_ge10_fraction": float(
                    (
                        valid_run[
                            valid_normal & (valid_stations == station)
                        ]
                        >= 10.0
                    ).mean()
                )
                if int(
                    (
                        valid_normal & (valid_stations == station)
                    ).sum()
                )
                else float("nan"),
                "flatness_at_max_fraction": float(
                    (
                        valid_flat[
                            valid_normal & (valid_stations == station)
                        ]
                        >= 1.0 - 1e-9
                    ).mean()
                )
                if int(
                    (
                        valid_normal & (valid_stations == station)
                    ).sum()
                )
                else float("nan"),
            }
            for station in stations_sorted
        }
        union_median, union_scale = robust_median_scale(
            state["union_score"][valid_normal]
        )
        formal_median, formal_scale = robust_median_scale(
            state["formal_score"][valid_normal]
        )
        state["union_valid_z"] = zscore(
            state["union_score"],
            union_median,
            union_scale,
        )
        state["union_test_z"] = zscore(
            state["union_test_score"],
            union_median,
            union_scale,
        )
        state["formal_valid_z"] = zscore(
            state["formal_score"],
            formal_median,
            formal_scale,
        )
        state["formal_test_z"] = zscore(
            state["formal_test_score"],
            formal_median,
            formal_scale,
        )

        families: dict[str, dict[str, object]] = {}
        # --- family A: global multi-scale regressor residuals ---
        for label, key in (
            ("mslag_hgb_residual", "resid_hgb"),
            ("mslag_mlp_residual", "resid_mlp"),
        ):
            score = state["valid_scores"][key]
            threshold = budget_threshold(
                score,
                valid_normal,
                PRIMARY_FAR,
            )
            predicted = valid_observed & (score >= threshold)
            families[label] = candidate_row(
                label,
                "multiscale_regressor_residual",
                valid_frame,
                predicted,
                score,
                union_valid_coverage,
                {
                    "budget_a": float(PRIMARY_FAR),
                    "budget_b": float("nan"),
                    "threshold_a": float(threshold),
                    "threshold_b": float("nan"),
                },
            )
        # --- family B: station-specific residual model ---
        score = state["valid_scores"]["resid_station"]
        thresholds: dict[str, float] = {}
        for station in stations_sorted:
            mask = valid_normal & (valid_stations == station)
            if int(mask.sum()) < 100:
                thresholds[station] = float("inf")
                continue
            thresholds[station] = budget_threshold(
                score,
                mask,
                PRIMARY_FAR,
            )
        threshold_lookup = np.array(
            [thresholds[station] for station in valid_stations],
            dtype=float,
        )
        predicted = valid_observed & (score >= threshold_lookup)
        families["station_hgb_residual"] = candidate_row(
            "station_hgb_residual",
            "station_specific_residual",
            valid_frame,
            predicted,
            score,
            union_valid_coverage,
            {
                "budget_a": float(PRIMARY_FAR),
                "budget_b": float("nan"),
                "threshold_a": float("nan"),
                "threshold_b": float("nan"),
                "station_threshold_count": int(len(thresholds)),
                "thresholds_a": [
                    float(thresholds[station])
                    for station in stations_sorted
                ],
            },
        )
        # --- family A3/B2: residual + flatness unions ---
        station_position = np.array(
            [stations_sorted.index(station) for station in valid_stations],
            dtype=int,
        )
        test_station_position = np.array(
            [stations_sorted.index(station) for station in test_stations],
            dtype=int,
        )
        state["station_position"] = station_position
        state["test_station_position"] = test_station_position
        station_normal_counts = {
            station: int(
                (
                    valid_normal & (valid_stations == station)
                ).sum()
            )
            for station in stations_sorted
        }

        def pooled_thresholds_for(
            score_values: np.ndarray,
            budget: float,
        ) -> float:
            return budget_threshold(score_values, valid_normal, budget)

        def station_thresholds_for(
            score_values: np.ndarray,
            budget: float,
        ) -> np.ndarray:
            return np.array(
                [
                    (
                        budget_threshold(
                            score_values,
                            valid_normal & (valid_stations == station),
                            budget,
                        )
                        if station_normal_counts[station] >= 100
                        else float("inf")
                    )
                    for station in stations_sorted
                ],
                dtype=float,
            )

        for label, resid_key, flat_key, kind in (
            (
                "mslag_hgb_residual_stuck",
                "resid_hgb",
                "stuck",
                "multiscale_residual_stuck_union",
            ),
            (
                "station_hgb_residual_stuck",
                "resid_station",
                "stuck_station",
                "station_residual_stuck_union",
            ),
        ):
            resid_score = state["valid_scores"][resid_key]
            flat_score = state["valid_scores"][flat_key]
            if label == "mslag_hgb_residual_stuck":
                threshold_mode = "pooled"

                def threshold_a_for(budget: float) -> float:
                    return pooled_thresholds_for(resid_score, budget)

                def threshold_b_for(budget: float) -> float:
                    return pooled_thresholds_for(flat_score, budget)

                def predict_for(threshold_a, threshold_b) -> np.ndarray:
                    return valid_observed & (
                        (resid_score >= threshold_a)
                        | (flat_score >= threshold_b)
                    )
            else:
                threshold_mode = "per_station"

                def threshold_a_for(budget: float) -> np.ndarray:
                    return station_thresholds_for(resid_score, budget)

                def threshold_b_for(budget: float) -> np.ndarray:
                    return station_thresholds_for(flat_score, budget)

                def predict_for(threshold_a, threshold_b) -> np.ndarray:
                    return valid_observed & (
                        (
                            resid_score
                            >= np.asarray(threshold_a)[station_position]
                        )
                        | (
                            flat_score
                            >= np.asarray(threshold_b)[station_position]
                        )
                    )

            families[label] = select_budget_split(
                threshold_a_for,
                threshold_b_for,
                predict_for,
                valid_frame,
                valid_normal,
                np.maximum(resid_score, flat_score),
                union_valid_coverage,
                label,
                kind,
                PRIMARY_FAR,
                {"threshold_mode": threshold_mode},
            )
        # --- family C: fusion with existing union/formal scores ---
        for additional in ADDITIONAL_BUDGET_GRID:
            label = f"fusion_union_repr_or_add{additional:g}"
            threshold_repr = budget_threshold(
                valid_repr,
                valid_normal,
                additional,
            )
            predicted = union_valid_pred | (
                valid_observed & (valid_repr >= threshold_repr)
            )
            families[label] = candidate_row(
                label,
                "constrained_fusion_or_union_repr",
                valid_frame,
                predicted,
                np.maximum(state["union_valid_z"], valid_repr),
                union_valid_coverage,
                {
                    "budget_a": float(additional),
                    "budget_b": float("nan"),
                    "threshold_a": float(threshold_repr),
                    "threshold_b": float("nan"),
                    "additional_budget": float(additional),
                    "far_budget_respected": False,
                },
            )
        for additional in ADDITIONAL_BUDGET_GRID:
            label = f"fusion_union_formal_or_add{additional:g}"
            per_component = additional / 2.0
            threshold_formal = budget_threshold(
                state["formal_valid_z"],
                valid_normal,
                per_component,
            )
            threshold_repr = budget_threshold(
                valid_repr,
                valid_normal,
                per_component,
            )
            predicted = (
                union_valid_pred
                | (
                    valid_observed
                    & (state["formal_valid_z"] >= threshold_formal)
                )
                | (
                    valid_observed
                    & (valid_repr >= threshold_repr)
                )
            )
            families[label] = candidate_row(
                label,
                "constrained_fusion_or_union_formal_repr",
                valid_frame,
                predicted,
                np.maximum(
                    state["union_valid_z"],
                    np.maximum(state["formal_valid_z"], valid_repr),
                ),
                union_valid_coverage,
                {
                    "budget_a": float(per_component),
                    "budget_b": float(per_component),
                    "threshold_a": float(threshold_formal),
                    "threshold_b": float(threshold_repr),
                    "additional_budget": float(additional),
                    "far_budget_respected": False,
                },
            )
        for weights in SCORE_FUSION_WEIGHTS:
            weight_union, weight_formal, weight_repr = weights
            label = (
                f"fusion_score_w{weight_union:g}_"
                f"{weight_formal:g}_{weight_repr:g}"
            )
            fused_valid = (
                weight_union * state["union_valid_z"]
                + weight_formal * state["formal_valid_z"]
                + weight_repr * valid_repr
            )
            threshold = budget_threshold(
                fused_valid,
                valid_normal,
                PRIMARY_FAR,
            )
            predicted = valid_observed & (fused_valid >= threshold)
            families[label] = candidate_row(
                label,
                "constrained_fusion_score",
                valid_frame,
                predicted,
                fused_valid,
                union_valid_coverage,
                {
                    "budget_a": float(PRIMARY_FAR),
                    "budget_b": float("nan"),
                    "threshold_a": float(threshold),
                    "threshold_b": float("nan"),
                    "weights": list(weights),
                },
            )
        fused_max_valid = np.maximum(
            state["union_valid_z"],
            np.maximum(state["formal_valid_z"], valid_repr),
        )
        threshold = budget_threshold(
            fused_max_valid,
            valid_normal,
            PRIMARY_FAR,
        )
        families["fusion_score_max"] = candidate_row(
            "fusion_score_max",
            "constrained_fusion_score",
            valid_frame,
            valid_observed & (fused_max_valid >= threshold),
            fused_max_valid,
            union_valid_coverage,
            {
                "budget_a": float(PRIMARY_FAR),
                "budget_b": float("nan"),
                "threshold_a": float(threshold),
                "threshold_b": float("nan"),
                "weights": ["max"],
            },
        )
        families["frozen_union_or"] = candidate_row(
            "frozen_union_or",
            "frozen_baseline",
            valid_frame,
            union_valid_pred,
            state["union_score"],
            union_valid_coverage,
            {
                "budget_a": float("nan"),
                "budget_b": float("nan"),
                "threshold_a": float("nan"),
                "threshold_b": float("nan"),
            },
        )
        candidate_valid[seed] = families
        print(
            json.dumps(
                {
                    "event": "validation_families_evaluated",
                    "seed": seed,
                    "best": max(
                        (
                            (
                                float(row["valid_mean_scenario_coverage"]),
                                name,
                            )
                            for name, row in families.items()
                        )
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    # --- deployment decision (validation only) ---
    family_names = [
        name
        for name in candidate_valid[seeds[0]]
        if name != "frozen_union_or"
    ]
    far_tolerance = 1.0 / max(
        int(union_state[seed]["valid_normal"].sum()) for seed in seeds
    )
    family_rows: list[dict[str, object]] = []
    for name in [*family_names, "frozen_union_or"]:
        rows = [candidate_valid[seed][name] for seed in seeds]
        constraint_all = all(row["constraint_ok"] for row in rows)
        far_all = all(
            float(row["valid_false_alarm_rate"])
            <= PRIMARY_FAR + far_tolerance
            for row in rows
        )
        family_rows.append(
            {
                "family": name,
                "family_kind": rows[0]["family_kind"],
                "valid_mean_scenario_coverage": float(
                    np.mean(
                        [
                            row["valid_mean_scenario_coverage"]
                            for row in rows
                        ]
                    )
                ),
                "valid_event_coverage": float(
                    np.mean([row["valid_event_coverage"] for row in rows])
                ),
                "valid_f1": float(
                    np.mean([row["valid_f1"] for row in rows])
                ),
                "valid_false_alarm_rate": float(
                    np.mean(
                        [
                            row["valid_false_alarm_rate"]
                            for row in rows
                        ]
                    )
                ),
                "valid_average_precision": float(
                    np.mean(
                        [
                            row["valid_average_precision"]
                            for row in rows
                        ]
                    )
                ),
                "constraint_ok_all_seeds": bool(constraint_all),
                "far_within_budget_all_seeds": bool(far_all),
                "constraint_feasible_seeds": int(
                    sum(1 for row in rows if row["constraint_ok"])
                ),
                "far_within_budget_seeds": int(
                    sum(
                        1
                        for row in rows
                        if float(row["valid_false_alarm_rate"])
                        <= PRIMARY_FAR + far_tolerance
                    )
                ),
                "deployable": bool(constraint_all and far_all),
            }
        )
    family_frame = pd.DataFrame(family_rows)
    union_valid_reference = family_frame[
        family_frame["family"].eq("frozen_union_or")
    ].iloc[0]
    deployable = family_frame[
        family_frame["deployable"]
        & ~family_frame["family"].eq("frozen_union_or")
    ].sort_values(
        ["valid_mean_scenario_coverage", "valid_f1"],
        ascending=[False, False],
    )
    improved = bool(
        len(deployable) > 0
        and float(deployable.iloc[0]["valid_mean_scenario_coverage"])
        > float(
            union_valid_reference["valid_mean_scenario_coverage"]
        )
        + EPS
    )
    selected_family = (
        str(deployable.iloc[0]["family"]) if improved else None
    )
    constraint_feasible = family_frame[
        family_frame["constraint_ok_all_seeds"]
        & ~family_frame["family"].eq("frozen_union_or")
        & (
            family_frame["valid_mean_scenario_coverage"]
            > float(
                union_valid_reference["valid_mean_scenario_coverage"]
            )
            + EPS
        )
    ].sort_values(
        ["valid_mean_scenario_coverage", "valid_f1"],
        ascending=[False, False],
    )
    far_inflated = constraint_feasible[
        ~constraint_feasible["far_within_budget_all_seeds"]
    ]
    counterfactual_family = (
        str(far_inflated.iloc[0]["family"])
        if len(far_inflated)
        else None
    )
    fallback = not improved
    print(
        json.dumps(
            {
                "event": "deployment_decision",
                "improved": improved,
                "selected_family": selected_family,
                "counterfactual_family": counterfactual_family,
                "deployable_families": deployable["family"].tolist(),
                "union_valid_mean_scenario_coverage": float(
                    union_valid_reference[
                        "valid_mean_scenario_coverage"
                    ]
                ),
                "family_table": frame_records(
                    family_frame[
                        [
                            "family",
                            "valid_mean_scenario_coverage",
                            "valid_f1",
                            "valid_event_coverage",
                            "valid_false_alarm_rate",
                            "constraint_feasible_seeds",
                            "far_within_budget_seeds",
                            "deployable",
                        ]
                    ]
                ),
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    # --- test application of every candidate and of the deployment ---
    run_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []

    def apply_test_family(
        seed: int,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        state = union_state[seed]
        valid_row = candidate_valid[seed][name]
        test_observed = state["test_observed"]
        scores = state["test_scores"]
        test_position = state["test_station_position"]
        if name in (
            "mslag_hgb_residual",
            "mslag_mlp_residual",
        ):
            key = (
                "resid_hgb"
                if name == "mslag_hgb_residual"
                else "resid_mlp"
            )
            score = scores[key]
            predicted = test_observed & (
                score >= float(valid_row["threshold_a"])
            )
            return predicted, score, valid_row
        if name == "station_hgb_residual":
            score = scores["resid_station"]
            thresholds = np.asarray(
                valid_row["thresholds_a"],
                dtype=float,
            )
            lookup = thresholds[test_position]
            predicted = test_observed & (score >= lookup)
            return predicted, score, valid_row
        if name == "mslag_hgb_residual_stuck":
            resid_score = scores["resid_hgb"]
            flat_score = scores["stuck"]
            threshold_a = float(valid_row["threshold_a"])
            threshold_b = float(valid_row["threshold_b"])
            predicted = test_observed & (
                (resid_score >= threshold_a)
                | (flat_score >= threshold_b)
            )
            return predicted, np.maximum(resid_score, flat_score), valid_row
        if name == "station_hgb_residual_stuck":
            resid_score = scores["resid_station"]
            flat_score = scores["stuck_station"]
            resid_lookup = np.asarray(
                valid_row["thresholds_a"],
                dtype=float,
            )[test_position]
            flat_lookup = np.asarray(
                valid_row["thresholds_b"],
                dtype=float,
            )[test_position]
            predicted = test_observed & (
                (resid_score >= resid_lookup)
                | (flat_score >= flat_lookup)
            )
            return predicted, np.maximum(resid_score, flat_score), valid_row
        if name.startswith("fusion_union_formal_or_add"):
            formal_threshold = budget_threshold(
                state["formal_valid_z"],
                state["valid_normal"],
                float(valid_row["budget_a"]),
            )
            repr_threshold = budget_threshold(
                state["valid_scores"]["repr"],
                state["valid_normal"],
                float(valid_row["budget_b"]),
            )
            score = np.maximum(
                state["union_test_z"],
                np.maximum(state["formal_test_z"], scores["repr"]),
            )
            predicted = (
                state["union_test_prediction"]
                | (test_observed & (state["formal_test_z"] >= formal_threshold))
                | (test_observed & (scores["repr"] >= repr_threshold))
            )
            return predicted, score, valid_row
        if name.startswith("fusion_union_repr_or_add"):
            repr_threshold = float(valid_row["threshold_a"])
            score = np.maximum(state["union_test_z"], scores["repr"])
            predicted = state["union_test_prediction"] | (
                test_observed & (scores["repr"] >= repr_threshold)
            )
            return predicted, score, valid_row
        if name.startswith("fusion_score_w"):
            weights = valid_row["weights"]
            score = (
                float(weights[0]) * state["union_test_z"]
                + float(weights[1]) * state["formal_test_z"]
                + float(weights[2]) * scores["repr"]
            )
            predicted = test_observed & (
                score >= float(valid_row["threshold_a"])
            )
            return predicted, score, valid_row
        if name == "fusion_score_max":
            score = np.maximum(
                state["union_test_z"],
                np.maximum(state["formal_test_z"], scores["repr"]),
            )
            predicted = test_observed & (
                score >= float(valid_row["threshold_a"])
            )
            return predicted, score, valid_row
        if name == "frozen_union_or":
            return (
                state["union_test_prediction"],
                state["union_test_score"],
                valid_row,
            )
        raise ValueError(f"unknown family {name}")

    for seed in seeds:
        state = union_state[seed]
        test_frame = state["test_frame"]
        union_test_pred = state["union_test_prediction"]
        union_test_row, union_test_coverage = alarm_metrics(
            test_frame,
            union_test_pred,
            state["union_test_score"],
        )
        union_valid_row = state["union_valid_row_metrics"]
        union_valid_coverage = state["union_valid_coverage"]
        reference491 = stage491_runs.loc[(NOAA, seed)]
        reference497 = stage497_runs.loc[(NOAA, seed)]
        reference493 = stage493_seed.loc[str(int(seed))]
        for name in [*family_names, "frozen_union_or"]:
            valid_row = candidate_valid[seed][name]
            predicted, score, valid_row = apply_test_family(seed, name)
            test_row, test_coverage = alarm_metrics(
                test_frame,
                predicted,
                score,
            )
            deployed = bool(
                (name == selected_family)
                or (name == "frozen_union_or" and fallback)
            )
            run_rows.append(
                {
                    "source": NOAA,
                    "seed": int(seed),
                    "method": name,
                    "family_kind": valid_row["family_kind"],
                    "deployed": deployed,
                    "fallback_to_union": bool(
                        name == "frozen_union_or" and fallback
                    ),
                    "constraint_ok": bool(valid_row["constraint_ok"]),
                    "valid_mean_scenario_coverage": valid_row[
                        "valid_mean_scenario_coverage"
                    ],
                    "valid_event_coverage": valid_row[
                        "valid_event_coverage"
                    ],
                    "valid_f1": valid_row["valid_f1"],
                    "valid_false_alarm_rate": valid_row[
                        "valid_false_alarm_rate"
                    ],
                    "valid_average_precision": valid_row[
                        "valid_average_precision"
                    ],
                    "valid_coverage_long_stuck": valid_row[
                        "valid_coverage_long_stuck"
                    ],
                    "valid_coverage_drop_replay": valid_row[
                        "valid_coverage_drop_replay"
                    ],
                    "valid_margin_long_stuck": valid_row[
                        "valid_margin_long_stuck"
                    ],
                    "valid_margin_drop_replay": valid_row[
                        "valid_margin_drop_replay"
                    ],
                    "union_valid_coverage_long_stuck": float(
                        union_valid_coverage.get("long_stuck", 0.0)
                    ),
                    "union_valid_coverage_drop_replay": float(
                        union_valid_coverage.get("drop_replay", 0.0)
                    ),
                    "budget_a": valid_row.get("budget_a"),
                    "budget_b": valid_row.get("budget_b"),
                    "threshold_a": valid_row.get("threshold_a"),
                    "threshold_b": valid_row.get("threshold_b"),
                    "rows": test_row["rows"],
                    "attack_rows": test_row["attack_rows"],
                    "precision": test_row["precision"],
                    "recall": test_row["recall"],
                    "f1": test_row["f1"],
                    "false_alarm_rate": test_row["false_alarm_rate"],
                    "roc_auc": test_row["roc_auc"],
                    "average_precision": test_row["average_precision"],
                    "events": test_row["events"],
                    "detected_events": test_row["detected_events"],
                    "event_coverage": test_row["event_coverage"],
                    "row_recall": test_row["row_recall"],
                    "mean_scenario_coverage": mean_scenario_coverage(
                        test_coverage
                    ),
                    "test_coverage_long_stuck": float(
                        test_coverage.get("long_stuck", 0.0)
                    ),
                    "test_coverage_drop_replay": float(
                        test_coverage.get("drop_replay", 0.0)
                    ),
                    "union_test_coverage_long_stuck": float(
                        union_test_coverage.get("long_stuck", 0.0)
                    ),
                    "union_test_coverage_drop_replay": float(
                        union_test_coverage.get("drop_replay", 0.0)
                    ),
                    "delta_long_stuck_vs_union": float(
                        test_coverage.get("long_stuck", 0.0)
                        - union_test_coverage.get("long_stuck", 0.0)
                    ),
                    "delta_drop_replay_vs_union": float(
                        test_coverage.get("drop_replay", 0.0)
                        - union_test_coverage.get("drop_replay", 0.0)
                    ),
                    "union_f1": union_test_row["f1"],
                    "union_event_coverage": union_test_row[
                        "event_coverage"
                    ],
                    "union_false_alarm_rate": union_test_row[
                        "false_alarm_rate"
                    ],
                    "union_average_precision": union_test_row[
                        "average_precision"
                    ],
                    "delta_f1_vs_union": float(
                        test_row["f1"] - union_test_row["f1"]
                    ),
                    "delta_event_coverage_vs_union": float(
                        test_row["event_coverage"]
                        - union_test_row["event_coverage"]
                    ),
                    "delta_far_vs_union": float(
                        test_row["false_alarm_rate"]
                        - union_test_row["false_alarm_rate"]
                    ),
                    "delta_ap_vs_union": float(
                        test_row["average_precision"]
                        - union_test_row["average_precision"]
                    ),
                    "stage491_f1": float(reference491["f1"]),
                    "stage491_event_coverage": float(
                        reference491["event_coverage"]
                    ),
                    "stage491_false_alarm_rate": float(
                        reference491["false_alarm_rate"]
                    ),
                    "stage491_average_precision": float(
                        reference491["average_precision"]
                    ),
                    "delta_f1_vs_stage491": float(
                        test_row["f1"] - reference491["f1"]
                    ),
                    "delta_event_coverage_vs_stage491": float(
                        test_row["event_coverage"]
                        - reference491["event_coverage"]
                    ),
                    "delta_far_vs_stage491": float(
                        test_row["false_alarm_rate"]
                        - reference491["false_alarm_rate"]
                    ),
                    "delta_ap_vs_stage491": float(
                        test_row["average_precision"]
                        - reference491["average_precision"]
                    ),
                    "stage497_f1": float(reference497["f1"]),
                    "stage497_event_coverage": float(
                        reference497["event_coverage"]
                    ),
                    "stage497_false_alarm_rate": float(
                        reference497["false_alarm_rate"]
                    ),
                    "stage497_average_precision": float(
                        reference497["average_precision"]
                    ),
                    "delta_f1_vs_stage497": float(
                        test_row["f1"] - reference497["f1"]
                    ),
                    "delta_event_coverage_vs_stage497": float(
                        test_row["event_coverage"]
                        - reference497["event_coverage"]
                    ),
                    "delta_far_vs_stage497": float(
                        test_row["false_alarm_rate"]
                        - reference497["false_alarm_rate"]
                    ),
                    "delta_ap_vs_stage497": float(
                        test_row["average_precision"]
                        - reference497["average_precision"]
                    ),
                    "stage493_f1": float(reference493["f1_mean"]),
                    "stage493_event_coverage": float(
                        reference493["event_coverage_mean"]
                    ),
                    "stage493_false_alarm_rate": float(
                        reference493["false_alarm_rate_mean"]
                    ),
                    "stage493_average_precision": float(
                        reference493["average_precision_mean"]
                    ),
                    "stage493_extended_wins": int(
                        reference493["extended_wins"]
                    ),
                    "stage493_extended_units": int(
                        reference493["extended_units"]
                    ),
                    "delta_f1_vs_stage493": float(
                        test_row["f1"] - reference493["f1_mean"]
                    ),
                    "delta_event_coverage_vs_stage493": float(
                        test_row["event_coverage"]
                        - reference493["event_coverage_mean"]
                    ),
                    "delta_far_vs_stage493": float(
                        test_row["false_alarm_rate"]
                        - reference493["false_alarm_rate_mean"]
                    ),
                    "delta_ap_vs_stage493": float(
                        test_row["average_precision"]
                        - reference493["average_precision_mean"]
                    ),
                }
            )
            scenario_rows.extend(
                scenario_detail_rows(
                    test_frame,
                    predicted,
                    score,
                    NOAA,
                    seed,
                    name,
                )
            )
            station_rows.extend(
                station_detail_rows(
                    test_frame,
                    predicted,
                    NOAA,
                    seed,
                    name,
                )
            )
        print(
            json.dumps(
                {
                    "event": "test_applied",
                    "seed": seed,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    # --- other providers: frozen union is deployed, score fusion diagnostic ---
    for source in providers:
        if source == NOAA:
            continue
        for seed in seeds:
            group = merged[
                merged["source"].eq(source) & merged["seed"].eq(seed)
            ]
            valid = group[group["split"].eq("valid")].reset_index(drop=True)
            test = group[group["split"].eq("test")].reset_index(drop=True)
            valid_frame = pd.DataFrame(
                {
                    "station_id": valid["station_id_union"].astype(str),
                    "attack_scenario": valid[
                        "attack_scenario_union"
                    ].astype(str),
                    "attack_event_id": valid[
                        "attack_event_id_union"
                    ].astype(str),
                    "is_attack": valid["is_attack_union"].astype(bool),
                    "observation_present": valid[
                        "observation_present_union"
                    ].astype(bool),
                }
            ).reset_index(drop=True)
            test_frame = pd.DataFrame(
                {
                    "station_id": test["station_id_union"].astype(str),
                    "attack_scenario": test[
                        "attack_scenario_union"
                    ].astype(str),
                    "attack_event_id": test[
                        "attack_event_id_union"
                    ].astype(str),
                    "is_attack": test["is_attack_union"].astype(bool),
                    "observation_present": test[
                        "observation_present_union"
                    ].astype(bool),
                }
            ).reset_index(drop=True)
            valid_normal = (
                valid_frame["observation_present"].to_numpy(dtype=bool)
                & ~valid_frame["is_attack"].to_numpy(dtype=bool)
            )
            observed_test = (
                test_frame["observation_present"].to_numpy(dtype=bool)
            )
            union_valid_pred = valid[
                "union_prediction"
            ].to_numpy(dtype=bool)
            union_test_pred = test[
                "union_prediction"
            ].to_numpy(dtype=bool)
            union_valid_coverage = scenario_coverage(
                valid_frame,
                union_valid_pred,
            )
            union_valid_row, _ = alarm_metrics(
                valid_frame,
                union_valid_pred,
                valid["union_score"].to_numpy(dtype=float),
            )
            union_test_row, union_test_coverage = alarm_metrics(
                test_frame,
                union_test_pred,
                test["union_score"].to_numpy(dtype=float),
            )
            union_median, union_scale = robust_median_scale(
                valid["union_score"].to_numpy(dtype=float)[valid_normal]
            )
            formal_median, formal_scale = robust_median_scale(
                valid["formal_score"].to_numpy(dtype=float)[valid_normal]
            )
            fused_valid = np.maximum(
                zscore(
                    valid["union_score"].to_numpy(dtype=float),
                    union_median,
                    union_scale,
                ),
                zscore(
                    valid["formal_score"].to_numpy(dtype=float),
                    formal_median,
                    formal_scale,
                ),
            )
            threshold = budget_threshold(
                fused_valid,
                valid_normal,
                PRIMARY_FAR,
            )
            diagnostic_valid_pred = (
                valid_frame["observation_present"].to_numpy(dtype=bool)
                & (fused_valid >= threshold)
            )
            diagnostic_row = candidate_row(
                "diagnostic_union_formal_score_fusion",
                "diagnostic_other_provider",
                valid_frame,
                diagnostic_valid_pred,
                fused_valid,
                union_valid_coverage,
                {
                    "budget_a": float(PRIMARY_FAR),
                    "budget_b": float("nan"),
                    "threshold_a": float(threshold),
                    "threshold_b": float("nan"),
                },
            )
            fused_test = np.maximum(
                zscore(
                    test["union_score"].to_numpy(dtype=float),
                    union_median,
                    union_scale,
                ),
                zscore(
                    test["formal_score"].to_numpy(dtype=float),
                    formal_median,
                    formal_scale,
                ),
            )
            diagnostic_test_pred = observed_test & (
                fused_test >= threshold
            )
            diagnostic_test_row, diagnostic_test_coverage = alarm_metrics(
                test_frame,
                diagnostic_test_pred,
                fused_test,
            )
            reference491 = stage491_runs.loc[(source, seed)]
            reference497 = stage497_runs.loc[(source, seed)]
            reference493 = stage493_seed.loc[str(int(seed))]
            for name, predicted, score, test_row, coverage in (
                (
                    "frozen_union_or",
                    union_test_pred,
                    test["union_score"].to_numpy(dtype=float),
                    union_test_row,
                    union_test_coverage,
                ),
                (
                    "diagnostic_union_formal_score_fusion",
                    diagnostic_test_pred,
                    fused_test,
                    diagnostic_test_row,
                    diagnostic_test_coverage,
                ),
            ):
                run_rows.append(
                    {
                        "source": source,
                        "seed": int(seed),
                        "method": name,
                        "family_kind": (
                            "frozen_baseline"
                            if name == "frozen_union_or"
                            else "diagnostic_other_provider"
                        ),
                        "deployed": bool(name == "frozen_union_or"),
                        "fallback_to_union": False,
                        "constraint_ok": (
                            True
                            if name == "frozen_union_or"
                            else bool(diagnostic_row["constraint_ok"])
                        ),
                        "valid_mean_scenario_coverage": (
                            mean_scenario_coverage(union_valid_coverage)
                            if name == "frozen_union_or"
                            else diagnostic_row[
                                "valid_mean_scenario_coverage"
                            ]
                        ),
                        "valid_event_coverage": (
                            union_valid_row["event_coverage"]
                            if name == "frozen_union_or"
                            else diagnostic_row["valid_event_coverage"]
                        ),
                        "valid_f1": (
                            union_valid_row["f1"]
                            if name == "frozen_union_or"
                            else diagnostic_row["valid_f1"]
                        ),
                        "valid_false_alarm_rate": (
                            union_valid_row["false_alarm_rate"]
                            if name == "frozen_union_or"
                            else diagnostic_row[
                                "valid_false_alarm_rate"
                            ]
                        ),
                        "valid_average_precision": (
                            union_valid_row["average_precision"]
                            if name == "frozen_union_or"
                            else diagnostic_row[
                                "valid_average_precision"
                            ]
                        ),
                        "valid_coverage_long_stuck": float(
                            union_valid_coverage.get("long_stuck", 0.0)
                        )
                        if name == "frozen_union_or"
                        else diagnostic_row[
                            "valid_coverage_long_stuck"
                        ],
                        "valid_coverage_drop_replay": float(
                            union_valid_coverage.get("drop_replay", 0.0)
                        )
                        if name == "frozen_union_or"
                        else diagnostic_row[
                            "valid_coverage_drop_replay"
                        ],
                        "valid_margin_long_stuck": (
                            0.0
                            if name == "frozen_union_or"
                            else diagnostic_row[
                                "valid_margin_long_stuck"
                            ]
                        ),
                        "valid_margin_drop_replay": (
                            0.0
                            if name == "frozen_union_or"
                            else diagnostic_row[
                                "valid_margin_drop_replay"
                            ]
                        ),
                        "union_valid_coverage_long_stuck": float(
                            union_valid_coverage.get("long_stuck", 0.0)
                        ),
                        "union_valid_coverage_drop_replay": float(
                            union_valid_coverage.get("drop_replay", 0.0)
                        ),
                        "budget_a": (
                            float("nan")
                            if name == "frozen_union_or"
                            else float(PRIMARY_FAR)
                        ),
                        "budget_b": float("nan"),
                        "threshold_a": (
                            float("nan")
                            if name == "frozen_union_or"
                            else float(threshold)
                        ),
                        "threshold_b": float("nan"),
                        "rows": test_row["rows"],
                        "attack_rows": test_row["attack_rows"],
                        "precision": test_row["precision"],
                        "recall": test_row["recall"],
                        "f1": test_row["f1"],
                        "false_alarm_rate": test_row[
                            "false_alarm_rate"
                        ],
                        "roc_auc": test_row["roc_auc"],
                        "average_precision": test_row[
                            "average_precision"
                        ],
                        "events": test_row["events"],
                        "detected_events": test_row["detected_events"],
                        "event_coverage": test_row["event_coverage"],
                        "row_recall": test_row["row_recall"],
                        "mean_scenario_coverage": mean_scenario_coverage(
                            coverage
                        ),
                        "test_coverage_long_stuck": float(
                            coverage.get("long_stuck", 0.0)
                        ),
                        "test_coverage_drop_replay": float(
                            coverage.get("drop_replay", 0.0)
                        ),
                        "union_test_coverage_long_stuck": float(
                            union_test_coverage.get("long_stuck", 0.0)
                        ),
                        "union_test_coverage_drop_replay": float(
                            union_test_coverage.get("drop_replay", 0.0)
                        ),
                        "delta_long_stuck_vs_union": float(
                            coverage.get("long_stuck", 0.0)
                            - union_test_coverage.get("long_stuck", 0.0)
                        ),
                        "delta_drop_replay_vs_union": float(
                            coverage.get("drop_replay", 0.0)
                            - union_test_coverage.get(
                                "drop_replay",
                                0.0,
                            )
                        ),
                        "union_f1": union_test_row["f1"],
                        "union_event_coverage": union_test_row[
                            "event_coverage"
                        ],
                        "union_false_alarm_rate": union_test_row[
                            "false_alarm_rate"
                        ],
                        "union_average_precision": union_test_row[
                            "average_precision"
                        ],
                        "delta_f1_vs_union": float(
                            test_row["f1"] - union_test_row["f1"]
                        ),
                        "delta_event_coverage_vs_union": float(
                            test_row["event_coverage"]
                            - union_test_row["event_coverage"]
                        ),
                        "delta_far_vs_union": float(
                            test_row["false_alarm_rate"]
                            - union_test_row["false_alarm_rate"]
                        ),
                        "delta_ap_vs_union": float(
                            test_row["average_precision"]
                            - union_test_row["average_precision"]
                        ),
                        "stage491_f1": float(reference491["f1"]),
                        "stage491_event_coverage": float(
                            reference491["event_coverage"]
                        ),
                        "stage491_false_alarm_rate": float(
                            reference491["false_alarm_rate"]
                        ),
                        "stage491_average_precision": float(
                            reference491["average_precision"]
                        ),
                        "delta_f1_vs_stage491": float(
                            test_row["f1"] - reference491["f1"]
                        ),
                        "delta_event_coverage_vs_stage491": float(
                            test_row["event_coverage"]
                            - reference491["event_coverage"]
                        ),
                        "delta_far_vs_stage491": float(
                            test_row["false_alarm_rate"]
                            - reference491["false_alarm_rate"]
                        ),
                        "delta_ap_vs_stage491": float(
                            test_row["average_precision"]
                            - reference491["average_precision"]
                        ),
                        "stage497_f1": float(reference497["f1"]),
                        "stage497_event_coverage": float(
                            reference497["event_coverage"]
                        ),
                        "stage497_false_alarm_rate": float(
                            reference497["false_alarm_rate"]
                        ),
                        "stage497_average_precision": float(
                            reference497["average_precision"]
                        ),
                        "delta_f1_vs_stage497": float(
                            test_row["f1"] - reference497["f1"]
                        ),
                        "delta_event_coverage_vs_stage497": float(
                            test_row["event_coverage"]
                            - reference497["event_coverage"]
                        ),
                        "delta_far_vs_stage497": float(
                            test_row["false_alarm_rate"]
                            - reference497["false_alarm_rate"]
                        ),
                        "delta_ap_vs_stage497": float(
                            test_row["average_precision"]
                            - reference497["average_precision"]
                        ),
                        "stage493_f1": float(reference493["f1_mean"]),
                        "stage493_event_coverage": float(
                            reference493["event_coverage_mean"]
                        ),
                        "stage493_false_alarm_rate": float(
                            reference493["false_alarm_rate_mean"]
                        ),
                        "stage493_average_precision": float(
                            reference493["average_precision_mean"]
                        ),
                        "stage493_extended_wins": int(
                            reference493["extended_wins"]
                        ),
                        "stage493_extended_units": int(
                            reference493["extended_units"]
                        ),
                        "delta_f1_vs_stage493": float(
                            test_row["f1"] - reference493["f1_mean"]
                        ),
                        "delta_event_coverage_vs_stage493": float(
                            test_row["event_coverage"]
                            - reference493["event_coverage_mean"]
                        ),
                        "delta_far_vs_stage493": float(
                            test_row["false_alarm_rate"]
                            - reference493["false_alarm_rate_mean"]
                        ),
                        "delta_ap_vs_stage493": float(
                            test_row["average_precision"]
                            - reference493["average_precision_mean"]
                        ),
                    }
                )
                scenario_rows.extend(
                    scenario_detail_rows(
                        test_frame,
                        predicted,
                        score,
                        source,
                        seed,
                        name,
                    )
                )
                station_rows.extend(
                    station_detail_rows(
                        test_frame,
                        predicted,
                        source,
                        seed,
                        name,
                    )
                )
        print(
            json.dumps(
                {
                    "event": "other_provider_reported",
                    "source": source,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    runs = pd.DataFrame(run_rows)
    scenario = pd.DataFrame(scenario_rows)
    station = pd.DataFrame(station_rows)
    runs.to_csv(OUT_RUNS, index=False, encoding="utf-8-sig")
    scenario.to_csv(OUT_SCENARIO, index=False, encoding="utf-8-sig")
    station.to_csv(OUT_STATION, index=False, encoding="utf-8-sig")

    summary = build_summary(
        runs,
        scenario,
        station,
        candidate_valid,
        family_frame,
        seeds,
        providers,
        stage491_runs,
        stage491_scenario,
        stage491_station,
        stage493_provider,
        stage493_seed,
        stage497_runs,
        stage497_scenario_lookup,
    )
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    noaa_runs = runs[
        runs["source"].eq(NOAA) & runs["deployed"]
    ].sort_values("seed")
    deployed_runs = runs[runs["deployed"]]
    noaa_deployed = noaa_runs.iloc[0]
    noaa_union_runs = runs[
        runs["source"].eq(NOAA) & runs["method"].eq("frozen_union_or")
    ].sort_values("seed")
    counterfactual_runs = (
        runs[
            runs["source"].eq(NOAA)
            & runs["method"].eq(counterfactual_family)
        ]
        if counterfactual_family is not None
        else pd.DataFrame()
    )
    counterfactual_row = (
        family_frame[family_frame["family"].eq(counterfactual_family)]
        .iloc[0]
        if counterfactual_family is not None
        else None
    )
    deployed_noaa_runs = runs[runs["source"].eq(NOAA) & runs["deployed"]]
    other_provider_rows = runs[
        ~runs["source"].eq(NOAA) & runs["deployed"]
    ]
    table_verification = {
        "noaa_rows_aligned_to_frozen_scores": True,
        "causality_check_passed": bool(causality["causal"]),
        "deployed_constraint_ok_all_seeds": bool(
            deployed_noaa_runs["constraint_ok"].all()
        ),
        "deployed_validation_far_within_budget_all_seeds": bool(
            (
                deployed_noaa_runs["valid_false_alarm_rate"]
                <= PRIMARY_FAR + far_tolerance
            ).all()
        ),
        "deployed_test_structured_deltas_nonnegative": bool(
            (deployed_noaa_runs["delta_long_stuck_vs_union"] >= -EPS).all()
            and (
                deployed_noaa_runs["delta_drop_replay_vs_union"] >= -EPS
            ).all()
        ),
        "other_providers_identical_to_frozen_union": bool(
            (other_provider_rows["delta_f1_vs_union"].abs() <= EPS).all()
            and (
                other_provider_rows["delta_event_coverage_vs_union"].abs()
                <= EPS
            ).all()
            and (
                other_provider_rows["delta_far_vs_union"].abs() <= EPS
            ).all()
        ),
        "run_statistic_gap_aware": True,
    }
    payload = {
        "status": (
            "PASS_EXPLORATORY_NOAA_REPRESENTATION_V2_IMPROVED"
            if improved
            else "NOAA_REPRESENTATION_IMPROVEMENT_STILL_FAILS"
        ),
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "protocol_id": frozen["protocol_id"],
        "primary_far": PRIMARY_FAR,
        "constraint": {
            "scenarios": list(STRUCTURED_SCENARIOS),
            "margin": CONSTRAINT_MARGIN,
            "basis": "validation event coverage versus frozen union OR",
        },
        "representation": {
            "feature_count": int(len(names)),
            "feature_families": [
                "multi-scale lags",
                "past deltas",
                "rolling location/scale",
                "past innovation statistics",
                "flatness (zero-difference fractions)",
                "gap-aware constant-run length (stuck-run statistic)",
                "missingness and gap length",
                "hour/day seasonality",
            ],
            "causality_check": causality,
            "stuck_signal_diagnostics": {
                station: {
                    "valid_normal_rows_mean": float(
                        np.mean(
                            [
                                union_state[seed]["run_saturation"][station][
                                    "valid_normal_rows"
                                ]
                                for seed in seeds
                            ]
                        )
                    ),
                    "run_ge10_fraction_mean": float(
                        np.mean(
                            [
                                union_state[seed]["run_saturation"][station][
                                    "run_ge10_fraction"
                                ]
                                for seed in seeds
                            ]
                        )
                    ),
                    "flatness_at_max_fraction_mean": float(
                        np.mean(
                            [
                                union_state[seed]["run_saturation"][station][
                                    "flatness_at_max_fraction"
                                ]
                                for seed in seeds
                            ]
                        )
                    ),
                }
                for station in stations_sorted
            },
            "models": {
                "global_hgb": {
                    "max_iter": 250,
                    "learning_rate": 0.06,
                    "max_leaf_nodes": 31,
                },
                "global_mlp": {
                    "hidden_layer_sizes": [64, 32],
                    "max_iter": 80,
                    "n_iter": global_models["mlp_n_iter"],
                },
                "station_hgb_count": int(
                    len(station_models["models"])
                ),
                "train_normal_rows": int(train_mask.sum()),
            },
        },
        "deployment": {
            "improved": bool(improved),
            "fallback_to_frozen_union_or": bool(fallback),
            "selected_family": selected_family,
            "constraint_feasible_counterfactual": counterfactual_family,
            "deployable_families": deployable["family"].tolist(),
            "family_summary": frame_records(family_frame),
        },
        "noaa_deployed_summary": {
            "method": str(noaa_deployed["method"]),
            "test_f1_mean": float(noaa_runs["f1"].mean()),
            "test_event_coverage_mean": float(
                noaa_runs["event_coverage"].mean()
            ),
            "test_mean_scenario_coverage_mean": float(
                noaa_runs["mean_scenario_coverage"].mean()
            ),
            "test_false_alarm_rate_mean": float(
                noaa_runs["false_alarm_rate"].mean()
            ),
            "test_average_precision_mean": float(
                noaa_runs["average_precision"].mean()
            ),
            "union_f1_mean": float(noaa_union_runs["f1"].mean()),
            "union_event_coverage_mean": float(
                noaa_union_runs["event_coverage"].mean()
            ),
            "union_false_alarm_rate_mean": float(
                noaa_union_runs["false_alarm_rate"].mean()
            ),
            "union_average_precision_mean": float(
                noaa_union_runs["average_precision"].mean()
            ),
            "delta_f1_vs_union": float(
                (
                    noaa_runs["f1"] - noaa_union_runs["f1"].to_numpy()
                ).mean()
            ),
            "delta_event_coverage_vs_union": float(
                (
                    noaa_runs["event_coverage"]
                    - noaa_union_runs["event_coverage"].to_numpy()
                ).mean()
            ),
            "delta_far_vs_union": float(
                (
                    noaa_runs["false_alarm_rate"]
                    - noaa_union_runs["false_alarm_rate"].to_numpy()
                ).mean()
            ),
            "delta_ap_vs_union": float(
                (
                    noaa_runs["average_precision"]
                    - noaa_union_runs["average_precision"].to_numpy()
                ).mean()
            ),
        },
        "noaa_structured_deltas": {
            scenario: {
                "deployed_test_coverage_mean": float(
                    noaa_runs[f"test_coverage_{scenario}"].mean()
                ),
                "union_test_coverage_mean": float(
                    noaa_union_runs[
                        f"test_coverage_{scenario}"
                    ].mean()
                ),
                "delta_mean": float(
                    noaa_runs[f"delta_{scenario}_vs_union"].mean()
                ),
                "per_seed": {
                    int(row["seed"]): float(
                        row[f"delta_{scenario}_vs_union"]
                    )
                    for _, row in noaa_runs.iterrows()
                },
            }
            for scenario in STRUCTURED_SCENARIOS
        },
        "counterfactual": (
            {
                "family": counterfactual_family,
                "constraint_ok_all_seeds": bool(
                    counterfactual_row["constraint_ok_all_seeds"]
                ),
                "far_within_budget": bool(
                    counterfactual_row["far_within_budget_all_seeds"]
                ),
                "valid_mean_scenario_coverage": float(
                    counterfactual_row[
                        "valid_mean_scenario_coverage"
                    ]
                ),
                "beats_deployed_on_validation": bool(
                    improved
                    and float(
                        counterfactual_row[
                            "valid_mean_scenario_coverage"
                        ]
                    )
                    > float(
                        union_valid_reference[
                            "valid_mean_scenario_coverage"
                        ]
                    )
                    + EPS
                    and float(
                        counterfactual_row[
                            "valid_mean_scenario_coverage"
                        ]
                    )
                    > float(
                        family_frame[
                            family_frame["family"].eq(
                                selected_family
                            )
                        ].iloc[0]["valid_mean_scenario_coverage"]
                    )
                    + EPS
                ),
                "test_event_coverage_mean": float(
                    counterfactual_runs["event_coverage"].mean()
                ),
                "test_mean_scenario_coverage_mean": float(
                    counterfactual_runs["mean_scenario_coverage"].mean()
                ),
                "test_false_alarm_rate_mean": float(
                    counterfactual_runs["false_alarm_rate"].mean()
                ),
                "test_f1_mean": float(
                    counterfactual_runs["f1"].mean()
                ),
                "test_average_precision_mean": float(
                    counterfactual_runs["average_precision"].mean()
                ),
                "delta_event_coverage_vs_union": float(
                    counterfactual_runs[
                        "delta_event_coverage_vs_union"
                    ].mean()
                ),
                "delta_far_vs_union": float(
                    counterfactual_runs["delta_far_vs_union"].mean()
                ),
                "delta_long_stuck_vs_union": float(
                    counterfactual_runs[
                        "delta_long_stuck_vs_union"
                    ].mean()
                ),
                "delta_drop_replay_vs_union": float(
                    counterfactual_runs[
                        "delta_drop_replay_vs_union"
                    ].mean()
                ),
            }
            if counterfactual_family is not None
            else None
        ),
        "other_providers": {
            source: {
                "deployed_method": "frozen_union_or",
                "affected_by_stage499": False,
                "diagnostic_union_formal_score_fusion": {
                    "test_event_coverage_mean": float(
                        runs[
                            runs["source"].eq(source)
                            & runs["method"].eq(
                                "diagnostic_union_formal_score_fusion"
                            )
                        ]["event_coverage"].mean()
                    ),
                    "test_false_alarm_rate_mean": float(
                        runs[
                            runs["source"].eq(source)
                            & runs["method"].eq(
                                "diagnostic_union_formal_score_fusion"
                            )
                        ]["false_alarm_rate"].mean()
                    ),
                    "delta_event_coverage_vs_union": float(
                        runs[
                            runs["source"].eq(source)
                            & runs["method"].eq(
                                "diagnostic_union_formal_score_fusion"
                            )
                        ]["delta_event_coverage_vs_union"].mean()
                    ),
                },
                "frozen_union_test_event_coverage_mean": float(
                    runs[
                        runs["source"].eq(source)
                        & runs["method"].eq("frozen_union_or")
                    ]["event_coverage"].mean()
                ),
            }
            for source in providers
            if source != NOAA
        },
        "comparisons": {
            "stage491": comparison_summary(
                runs,
                "stage491",
                NOAA,
            ),
            "stage497": comparison_summary(
                runs,
                "stage497",
                NOAA,
            ),
            "stage493_provider": (
                json.loads(stage493_provider.loc[NOAA].to_json())
                if NOAA in stage493_provider.index
                else {}
            ),
            "stage493_seed": frame_index_records(
                stage493_seed.loc[
                    [
                        str(seed)
                        for seed in seeds
                        if str(seed) in stage493_seed.index
                    ]
                ]
            ),
        },
        "run_summary": {
            "deployed_rows": int(len(deployed_runs)),
            "noaa_candidate_rows": int(
                len(runs[runs["source"].eq(NOAA)])
            ),
            "scenario_rows": int(len(scenario)),
            "station_rows": int(len(station)),
            "summary_rows": int(len(summary)),
        },
        "verification": table_verification,
        "limitations": [
            "The V6 test split was already inspected in earlier stages; every number here is exploratory.",
            "Representation models are fitted on NOAA train-normal rows only; other providers are not retrained and keep the frozen union OR decision.",
            "A family can only be deployed if it respects both the structured-coverage floor and the frozen validation FAR budget; otherwise the NOAA prediction falls back to the frozen union OR decision.",
            "The OR fusion families that exceed the frozen FAR budget are reported as diagnostics, not as deployed results.",
            "The gap-aware constant-run statistic is calibrated on NOAA ISD, whose reports are integer-valued and irregular; station 57036099999 saturates it (several percent of normal rows share a value with the previous record), which concentrates false alarms there.",
            "The deployed family changes only the NOAA decision rule; the frozen main table, the confirmatory protocol and the other three providers are untouched.",
            "The result does not replace the frozen V6 main table and no provider, dataset, or confirmatory test was added.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    verdict_cn = (
        "NOAA表示改进成功：在冻结FAR（validation<=0.0065）与"
        "long_stuck/drop_replay结构化覆盖约束（margin=0）下，"
        f"部署了表示层候选 `{selected_family}`，未回退冻结union OR。"
        if improved
        else "NOAA表示改进仍失败：在冻结FAR与结构化覆盖约束下没有可部署的"
        "表示层候选，NOAA回退冻结union OR预测。"
    )
    deployed_table = runs[
        runs["source"].eq(NOAA) & runs["deployed"]
    ][
        [
            "seed",
            "method",
            "budget_a",
            "budget_b",
            "valid_coverage_long_stuck",
            "valid_coverage_drop_replay",
            "valid_false_alarm_rate",
            "valid_mean_scenario_coverage",
            "f1",
            "event_coverage",
            "false_alarm_rate",
            "average_precision",
            "delta_long_stuck_vs_union",
            "delta_drop_replay_vs_union",
            "delta_event_coverage_vs_union",
            "delta_far_vs_union",
            "delta_event_coverage_vs_stage491",
            "delta_event_coverage_vs_stage493",
            "delta_event_coverage_vs_stage497",
        ]
    ].assign(seed=lambda frame: frame["seed"].astype(int).astype(str))
    union_table = runs[
        runs["source"].eq(NOAA)
        & runs["method"].eq("frozen_union_or")
    ][
        [
            "seed",
            "f1",
            "event_coverage",
            "false_alarm_rate",
            "average_precision",
            "test_coverage_long_stuck",
            "test_coverage_drop_replay",
        ]
    ].assign(seed=lambda frame: frame["seed"].astype(int).astype(str))
    scenario_table = summary[
        summary["aggregation"].eq("provider_scenario")
        & summary["source"].eq(NOAA)
    ][
        [
            "scenario",
            "event_coverage_mean",
            "f1_mean",
            "false_alarm_rate_mean",
            "average_precision_mean",
            "events",
            "detected_events",
        ]
    ]
    station_table = summary[
        summary["aggregation"].eq("provider_station")
        & summary["source"].eq(NOAA)
    ][
        [
            "station_id",
            "event_coverage_mean",
            "f1_mean",
            "false_alarm_rate_mean",
            "events",
            "detected_events",
        ]
    ]
    stuck_diagnostics = pd.DataFrame(
        [
            {"station_id": station, **values}
            for station, values in payload["representation"][
                "stuck_signal_diagnostics"
            ].items()
        ]
    )
    lines = [
        "# Stage499 NOAA ISD representation v2",
        "",
        f"- Status: `{payload['status']}`",
        f"- 结论: {verdict_cn}",
        f"- Improved: `{improved}`",
        f"- Selected family: `{selected_family}`",
        f"- Fallback to frozen union OR: `{fallback}`",
        f"- FAR-inflated constraint-feasible alternative: "
        f"`{counterfactual_family}`",
        f"- Feature count: `{len(names)}`; causal check: "
        f"`{causality['causal']}`",
        f"- Primary FAR: `{PRIMARY_FAR}`",
        "",
        "## NOAA deployed result (test, mean over seeds)",
        "",
        pd.DataFrame(
            [payload["noaa_deployed_summary"]]
        ).to_markdown(index=False),
        "",
        "## NOAA deployed per seed",
        "",
        deployed_table.to_markdown(index=False),
        "",
        "## Frozen union reference (NOAA test)",
        "",
        union_table.to_markdown(index=False),
        "",
        "## Family summary (validation)",
        "",
        family_frame[
            [
                "family",
                "family_kind",
                "valid_mean_scenario_coverage",
                "valid_event_coverage",
                "valid_f1",
                "valid_false_alarm_rate",
                "valid_average_precision",
                "constraint_ok_all_seeds",
                "far_within_budget_all_seeds",
                "deployable",
            ]
        ].to_markdown(index=False),
        "",
        "## NOAA per scenario (test)",
        "",
        scenario_table.to_markdown(index=False),
        "",
        "## NOAA per station (test)",
        "",
        station_table.to_markdown(index=False),
        "",
        "## NOAA structured deltas (test)",
        "",
        pd.DataFrame(
            [
                {
                    "scenario": scenario,
                    **values,
                }
                for scenario, values in payload[
                    "noaa_structured_deltas"
                ].items()
            ]
        ).to_markdown(index=False),
        "",
        "## Stuck-signal diagnostics (validation-normal)",
        "",
        stuck_diagnostics.to_markdown(index=False),
        "",
        "## FAR-inflated constraint-feasible alternative",
        "",
        (
            pd.DataFrame([payload["counterfactual"]]).to_markdown(
                index=False
            )
            if payload["counterfactual"]
            else "none"
        ),
        "",
        "## Other providers",
        "",
        pd.DataFrame(
            [
                {
                    "source": source,
                    "frozen_union_test_event_coverage_mean": values[
                        "frozen_union_test_event_coverage_mean"
                    ],
                    "diagnostic_delta_event_coverage": values[
                        "diagnostic_union_formal_score_fusion"
                    ]["delta_event_coverage_vs_union"],
                }
                for source, values in payload[
                    "other_providers"
                ].items()
            ]
        ).to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- Selection, thresholds and the structured-coverage floor use validation rows only.",
        "- Test rows are evaluation only and never enter a decision.",
        "- The V6 test split was already inspected; the result is exploratory.",
        "- The frozen main table is unchanged and no provider or dataset was added.",
        "",
        f"- Elapsed seconds: `{time.perf_counter() - started:.2f}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    file_verification = {
        path.name: {
            "exists": bool(path.exists()),
            "bytes": int(path.stat().st_size) if path.exists() else 0,
        }
        for path in (
            OUT_RUNS,
            OUT_SCENARIO,
            OUT_STATION,
            OUT_SUMMARY,
            OUT_JSON,
            OUT_MD,
        )
    }
    failed = [
        name
        for name, values in file_verification.items()
        if not values["exists"] or values["bytes"] <= 0
    ]
    failed_checks = [
        name
        for name, value in table_verification.items()
        if not value
    ]
    if failed or failed_checks:
        raise RuntimeError(
            f"stage499 self-check failed: files={failed}, "
            f"checks={failed_checks}"
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "improved": improved,
                "selected_family": selected_family,
                "fallback": fallback,
                "counterfactual_family": counterfactual_family,
                "noaa_deployed_summary": payload[
                    "noaa_deployed_summary"
                ],
                "noaa_structured_deltas": payload[
                    "noaa_structured_deltas"
                ],
                "counterfactual": payload["counterfactual"],
                "other_providers": payload["other_providers"],
                "verification": table_verification,
                "files": file_verification,
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        )
    )
    return 0


def scenario_detail_rows(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    score: np.ndarray,
    source: str,
    seed: int,
    method: str,
) -> list[dict[str, object]]:
    """Per-scenario rows following the frozen stage470 convention.

    Each scenario is evaluated against the normal rows plus that scenario's
    own attack rows, so rows of other scenarios are never counted as false
    alarms for it.
    """
    attack = frame["is_attack"].astype(bool).to_numpy()
    labels = frame["attack_scenario"].astype(str).to_numpy()
    predicted = np.asarray(predicted, dtype=bool)
    values = np.asarray(score, dtype=float)
    rows: list[dict[str, object]] = []
    for scenario in sorted(set(labels[attack])):
        mask = attack & (labels == scenario)
        if not mask.any():
            continue
        evaluation_mask = (~attack) | mask
        row_metrics = metrics(
            mask[evaluation_mask],
            predicted[evaluation_mask],
            values[evaluation_mask],
        )
        true_positive = int(
            (predicted[evaluation_mask] & mask[evaluation_mask]).sum()
        )
        false_positive = int(
            (predicted[evaluation_mask] & ~mask[evaluation_mask]).sum()
        )
        false_negative = int(
            (~predicted[evaluation_mask] & mask[evaluation_mask]).sum()
        )
        true_negative = int(
            (~predicted[evaluation_mask] & ~mask[evaluation_mask]).sum()
        )
        events = frame.loc[mask, ["attack_event_id"]].copy()
        events["_pred"] = predicted[mask]
        grouped = events.groupby("attack_event_id", sort=True)[
            "_pred"
        ].any()
        lengths = events.groupby("attack_event_id", sort=True).size()
        rows.append(
            {
                "source": source,
                "seed": int(seed),
                "method": method,
                "scenario": str(scenario),
                "rows": int(row_metrics["rows"]),
                "attack_rows": int(row_metrics["attack_rows"]),
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
                "tn": true_negative,
                "precision": float(row_metrics["precision"]),
                "recall": float(row_metrics["recall"]),
                "row_recall": float(row_metrics["recall"]),
                "f1": float(row_metrics["f1"]),
                "false_alarm_rate": float(
                    row_metrics["false_alarm_rate"]
                ),
                "roc_auc": float(row_metrics["roc_auc"]),
                "average_precision": float(
                    row_metrics["average_precision"]
                ),
                "events": int(len(grouped)),
                "detected_events": int(grouped.sum()),
                "event_coverage": (
                    float(grouped.mean()) if len(grouped) else 0.0
                ),
                "mean_event_length": (
                    float(lengths.mean()) if len(lengths) else 0.0
                ),
                "min_event_length": (
                    int(lengths.min()) if len(lengths) else 0
                ),
                "max_event_length": (
                    int(lengths.max()) if len(lengths) else 0
                ),
            }
        )
    return rows


def station_detail_rows(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    source: str,
    seed: int,
    method: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    attack = frame["is_attack"].astype(bool).to_numpy()
    observed = frame["observation_present"].astype(bool).to_numpy()
    for station, part in frame.groupby("station_id", sort=True):
        index = part.index.to_numpy(dtype=int)
        active = attack[index] & observed[index]
        alarm = predicted[index] & observed[index]
        true_positive = int((alarm & active).sum())
        false_positive = int((alarm & ~active).sum())
        false_negative = int((~alarm & active).sum())
        true_negative = int((~alarm & ~active & observed[index]).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        events = part.loc[active, ["attack_event_id"]].copy()
        events["predicted"] = alarm[active]
        grouped = events.groupby("attack_event_id", sort=True)[
            "predicted"
        ].any()
        normal = observed[index] & ~attack[index]
        rows.append(
            {
                "source": source,
                "seed": int(seed),
                "method": method,
                "station_id": str(station),
                "tp": true_positive,
                "fp": false_positive,
                "fn": false_negative,
                "tn": true_negative,
                "precision": (
                    float(true_positive / (true_positive + false_positive))
                    if true_positive + false_positive
                    else 0.0
                ),
                "recall": (
                    float(true_positive / (true_positive + false_negative))
                    if true_positive + false_negative
                    else 0.0
                ),
                "row_recall": (
                    float(true_positive / active.sum())
                    if int(active.sum())
                    else 0.0
                ),
                "f1": (
                    float(2 * true_positive / denominator)
                    if denominator
                    else 0.0
                ),
                "false_alarm_rate": (
                    float(alarm[normal].mean()) if normal.any() else 0.0
                ),
                "event_coverage": (
                    float(grouped.mean()) if len(grouped) else 0.0
                ),
                "events": int(len(grouped)),
                "detected_events": int(grouped.sum()),
            }
        )
    return rows


def frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """JSON-safe record list (numpy scalars converted, NaN as null)."""
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def frame_index_records(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    return json.loads(frame.to_json(orient="index"))


def comparison_summary(
    runs: pd.DataFrame,
    prefix: str,
    source: str,
) -> dict[str, float]:
    rows = runs[runs["source"].eq(source) & runs["deployed"]]
    if rows.empty:
        return {}
    return {
        "deployed_f1_mean": float(rows["f1"].mean()),
        "deployed_event_coverage_mean": float(
            rows["event_coverage"].mean()
        ),
        "deployed_false_alarm_rate_mean": float(
            rows["false_alarm_rate"].mean()
        ),
        "deployed_average_precision_mean": float(
            rows["average_precision"].mean()
        ),
        f"delta_f1_vs_{prefix}": float(
            rows[f"delta_f1_vs_{prefix}"].mean()
        ),
        f"delta_event_coverage_vs_{prefix}": float(
            rows[f"delta_event_coverage_vs_{prefix}"].mean()
        ),
        f"delta_far_vs_{prefix}": float(
            rows[f"delta_far_vs_{prefix}"].mean()
        ),
        f"delta_ap_vs_{prefix}": float(
            rows[f"delta_ap_vs_{prefix}"].mean()
        ),
    }


def build_summary(
    runs: pd.DataFrame,
    scenario: pd.DataFrame,
    station: pd.DataFrame,
    candidate_valid: dict[int, dict[str, dict[str, object]]],
    family_frame: pd.DataFrame,
    seeds: list[int],
    providers: list[str],
    stage491_runs: pd.DataFrame,
    stage491_scenario: pd.DataFrame,
    stage491_station: pd.DataFrame,
    stage493_provider: pd.DataFrame,
    stage493_seed: pd.DataFrame,
    stage497_runs: pd.DataFrame,
    stage497_scenario: pd.DataFrame,
) -> pd.DataFrame:
    """Provider / seed / scenario / station aggregation with deltas."""
    deployed = runs[runs["deployed"]].copy()
    union_runs = runs[runs["method"].eq("frozen_union_or")].copy()
    union_lookup = union_runs.set_index(["source", "seed"])
    rows: list[dict[str, object]] = []

    def base_row(
        aggregation: str,
        source: str,
        seed: object,
        scenario_name: object,
        station_id: object,
        group: pd.DataFrame,
    ) -> dict[str, object]:
        def metric_mean(column: str) -> float:
            if column not in group.columns:
                return float("nan")
            return float(group[column].mean())

        row: dict[str, object] = {
            "aggregation": aggregation,
            "source": source,
            "seed": seed,
            "scenario": scenario_name,
            "station_id": station_id,
            "runs": int(len(group)),
            "f1_mean": metric_mean("f1"),
            "event_coverage_mean": metric_mean("event_coverage"),
            "false_alarm_rate_mean": metric_mean("false_alarm_rate"),
            "average_precision_mean": metric_mean("average_precision"),
        }
        return row

    def add_union_columns(
        row: dict[str, object],
        source: str,
        seed: object,
    ) -> dict[str, object]:
        if seed is None or (source, seed) not in union_lookup.index:
            union_row = union_lookup.loc[source].mean(numeric_only=True)
        else:
            union_row = union_lookup.loc[(source, seed)]
        row.update(
            {
                "union_f1_mean": float(union_row["f1"]),
                "union_event_coverage_mean": float(
                    union_row["event_coverage"]
                ),
                "union_false_alarm_rate_mean": float(
                    union_row["false_alarm_rate"]
                ),
                "union_average_precision_mean": float(
                    union_row["average_precision"]
                ),
            }
        )
        row.update(
            {
                "delta_f1_vs_union": row["f1_mean"]
                - row["union_f1_mean"],
                "delta_event_coverage_vs_union": row[
                    "event_coverage_mean"
                ]
                - row["union_event_coverage_mean"],
                "delta_far_vs_union": row["false_alarm_rate_mean"]
                - row["union_false_alarm_rate_mean"],
                "delta_ap_vs_union": row["average_precision_mean"]
                - row["union_average_precision_mean"],
            }
        )
        if seed is None:
            reference = stage491_runs.loc[source]
            row.update(
                {
                    "stage491_f1_mean": float(reference["f1"].mean()),
                    "stage491_event_coverage_mean": float(
                        reference["event_coverage"].mean()
                    ),
                    "stage491_false_alarm_rate_mean": float(
                        reference["false_alarm_rate"].mean()
                    ),
                    "stage491_average_precision_mean": float(
                        reference["average_precision"].mean()
                    ),
                }
            )
        else:
            reference = stage491_runs.loc[(source, seed)]
            row.update(
                {
                    "stage491_f1_mean": float(reference["f1"]),
                    "stage491_event_coverage_mean": float(
                        reference["event_coverage"]
                    ),
                    "stage491_false_alarm_rate_mean": float(
                        reference["false_alarm_rate"]
                    ),
                    "stage491_average_precision_mean": float(
                        reference["average_precision"]
                    ),
                }
            )
        row.update(
            {
                "delta_f1_vs_stage491": row["f1_mean"]
                - row["stage491_f1_mean"],
                "delta_event_coverage_vs_stage491": row[
                    "event_coverage_mean"
                ]
                - row["stage491_event_coverage_mean"],
                "delta_far_vs_stage491": row["false_alarm_rate_mean"]
                - row["stage491_false_alarm_rate_mean"],
                "delta_ap_vs_stage491": row["average_precision_mean"]
                - row["stage491_average_precision_mean"],
            }
        )
        if seed is None:
            reference = stage497_runs.loc[source]
            row.update(
                {
                    "stage497_f1_mean": float(reference["f1"].mean()),
                    "stage497_event_coverage_mean": float(
                        reference["event_coverage"].mean()
                    ),
                    "stage497_false_alarm_rate_mean": float(
                        reference["false_alarm_rate"].mean()
                    ),
                    "stage497_average_precision_mean": float(
                        reference["average_precision"].mean()
                    ),
                }
            )
        else:
            reference = stage497_runs.loc[(source, seed)]
            row.update(
                {
                    "stage497_f1_mean": float(reference["f1"]),
                    "stage497_event_coverage_mean": float(
                        reference["event_coverage"]
                    ),
                    "stage497_false_alarm_rate_mean": float(
                        reference["false_alarm_rate"]
                    ),
                    "stage497_average_precision_mean": float(
                        reference["average_precision"]
                    ),
                }
            )
        row.update(
            {
                "delta_f1_vs_stage497": row["f1_mean"]
                - row["stage497_f1_mean"],
                "delta_event_coverage_vs_stage497": row[
                    "event_coverage_mean"
                ]
                - row["stage497_event_coverage_mean"],
                "delta_far_vs_stage497": row["false_alarm_rate_mean"]
                - row["stage497_false_alarm_rate_mean"],
                "delta_ap_vs_stage497": row["average_precision_mean"]
                - row["stage497_average_precision_mean"],
            }
        )
        reference493 = (
            stage493_provider.loc[source]
            if seed is None
            else stage493_seed.loc[str(int(seed))]
        )
        row.update(
            {
                "stage493_f1_mean": float(reference493["f1_mean"]),
                "stage493_event_coverage_mean": float(
                    reference493["event_coverage_mean"]
                ),
                "stage493_false_alarm_rate_mean": float(
                    reference493["false_alarm_rate_mean"]
                ),
                "stage493_average_precision_mean": float(
                    reference493["average_precision_mean"]
                ),
                "stage493_extended_wins": int(
                    reference493["extended_wins"]
                ),
                "stage493_extended_units": int(
                    reference493["extended_units"]
                ),
            }
        )
        row.update(
            {
                "delta_f1_vs_stage493": row["f1_mean"]
                - row["stage493_f1_mean"],
                "delta_event_coverage_vs_stage493": row[
                    "event_coverage_mean"
                ]
                - row["stage493_event_coverage_mean"],
                "delta_far_vs_stage493": row["false_alarm_rate_mean"]
                - row["stage493_false_alarm_rate_mean"],
                "delta_ap_vs_stage493": row["average_precision_mean"]
                - row["stage493_average_precision_mean"],
            }
        )
        return row

    # provider level
    for source in providers:
        group = deployed[deployed["source"].eq(source)]
        row = base_row("provider", source, None, None, None, group)
        row = add_union_columns(row, source, None)
        row["structured_long_stuck_coverage_mean"] = float(
            group["test_coverage_long_stuck"].mean()
        )
        row["structured_drop_replay_coverage_mean"] = float(
            group["test_coverage_drop_replay"].mean()
        )
        row["delta_long_stuck_vs_union"] = float(
            group["delta_long_stuck_vs_union"].mean()
        )
        row["delta_drop_replay_vs_union"] = float(
            group["delta_drop_replay_vs_union"].mean()
        )
        row["mean_scenario_coverage"] = float(
            group["mean_scenario_coverage"].mean()
        )
        rows.append(row)
        # provider x seed
        for seed in seeds:
            subset = group[group["seed"].eq(seed)]
            if subset.empty:
                continue
            seed_row = base_row(
                "provider_seed",
                source,
                int(seed),
                None,
                None,
                subset,
            )
            seed_row = add_union_columns(seed_row, source, int(seed))
            seed_row["structured_long_stuck_coverage_mean"] = float(
                subset["test_coverage_long_stuck"].mean()
            )
            seed_row["structured_drop_replay_coverage_mean"] = float(
                subset["test_coverage_drop_replay"].mean()
            )
            seed_row["delta_long_stuck_vs_union"] = float(
                subset["delta_long_stuck_vs_union"].mean()
            )
            seed_row["delta_drop_replay_vs_union"] = float(
                subset["delta_drop_replay_vs_union"].mean()
            )
            seed_row["mean_scenario_coverage"] = float(
                subset["mean_scenario_coverage"].mean()
            )
            rows.append(seed_row)
        # provider x scenario
        methods = group[["seed", "method"]].drop_duplicates()
        detail = scenario.merge(
            methods,
            on=["seed", "method"],
            how="inner",
        )
        detail = detail[detail["source"].eq(source)]
        for scenario_name, subset in detail.groupby(
            "scenario",
            sort=True,
        ):
            row_scenario = base_row(
                "provider_scenario",
                source,
                None,
                scenario_name,
                None,
                subset,
            )
            row_scenario["events"] = int(subset["events"].sum())
            row_scenario["detected_events"] = int(
                subset["detected_events"].sum()
            )
            rows.append(row_scenario)
        # provider x station
        station_detail = station.merge(
            methods,
            on=["seed", "method"],
            how="inner",
        )
        station_detail = station_detail[
            station_detail["source"].eq(source)
        ]
        for station_name, subset in station_detail.groupby(
            "station_id",
            sort=True,
        ):
            row_station = base_row(
                "provider_station",
                source,
                None,
                None,
                station_name,
                subset,
            )
            row_station["events"] = int(subset["events"].sum())
            row_station["detected_events"] = int(
                subset["detected_events"].sum()
            )
            rows.append(row_station)

    # NOAA candidate families (validation + test means)
    for _, family in family_frame.iterrows():
        name = str(family["family"])
        subset = runs[
            runs["source"].eq(NOAA) & runs["method"].eq(name)
        ]
        row = base_row(
            "noaa_candidate_family",
            NOAA,
            None,
            None,
            None,
            subset,
        )
        row["method"] = name
        row["valid_mean_scenario_coverage_mean"] = float(
            family["valid_mean_scenario_coverage"]
        )
        row["valid_f1_mean"] = float(family["valid_f1"])
        row["valid_event_coverage_mean"] = float(
            family["valid_event_coverage"]
        )
        row["valid_false_alarm_rate_mean"] = float(
            family["valid_false_alarm_rate"]
        )
        row["constraint_ok_all_seeds"] = bool(
            family["constraint_ok_all_seeds"]
        )
        row["far_within_budget_all_seeds"] = bool(
            family["far_within_budget_all_seeds"]
        )
        row["deployable"] = bool(family["deployable"])
        row["delta_long_stuck_vs_union"] = float(
            subset["delta_long_stuck_vs_union"].mean()
        )
        row["delta_drop_replay_vs_union"] = float(
            subset["delta_drop_replay_vs_union"].mean()
        )
        row["mean_scenario_coverage"] = float(
            subset["mean_scenario_coverage"].mean()
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    ordered = [
        "aggregation",
        "source",
        "seed",
        "scenario",
        "station_id",
        "method",
        "runs",
        "f1_mean",
        "event_coverage_mean",
        "mean_scenario_coverage",
        "false_alarm_rate_mean",
        "average_precision_mean",
        "structured_long_stuck_coverage_mean",
        "structured_drop_replay_coverage_mean",
        "delta_long_stuck_vs_union",
        "delta_drop_replay_vs_union",
        "union_f1_mean",
        "union_event_coverage_mean",
        "union_false_alarm_rate_mean",
        "union_average_precision_mean",
        "delta_f1_vs_union",
        "delta_event_coverage_vs_union",
        "delta_far_vs_union",
        "delta_ap_vs_union",
        "stage491_f1_mean",
        "stage491_event_coverage_mean",
        "stage491_false_alarm_rate_mean",
        "stage491_average_precision_mean",
        "delta_f1_vs_stage491",
        "delta_event_coverage_vs_stage491",
        "delta_far_vs_stage491",
        "delta_ap_vs_stage491",
        "stage497_f1_mean",
        "stage497_event_coverage_mean",
        "stage497_false_alarm_rate_mean",
        "stage497_average_precision_mean",
        "delta_f1_vs_stage497",
        "delta_event_coverage_vs_stage497",
        "delta_far_vs_stage497",
        "delta_ap_vs_stage497",
        "stage493_f1_mean",
        "stage493_event_coverage_mean",
        "stage493_false_alarm_rate_mean",
        "stage493_average_precision_mean",
        "stage493_extended_wins",
        "stage493_extended_units",
        "delta_f1_vs_stage493",
        "delta_event_coverage_vs_stage493",
        "delta_far_vs_stage493",
        "delta_ap_vs_stage493",
        "valid_mean_scenario_coverage_mean",
        "valid_f1_mean",
        "valid_event_coverage_mean",
        "valid_false_alarm_rate_mean",
        "constraint_ok_all_seeds",
        "far_within_budget_all_seeds",
        "deployable",
        "events",
        "detected_events",
    ]
    for column in ordered:
        if column not in summary:
            summary[column] = np.nan
    return summary[ordered]


if __name__ == "__main__":
    raise SystemExit(main())
