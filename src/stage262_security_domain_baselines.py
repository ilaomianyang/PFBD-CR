from __future__ import annotations

"""Stage 262: exploratory security-domain baseline audit.

This stage adds five reproducible security/time-series baselines to the
Stage248 frozen evaluation contract: EWMA deviation, low-order ARIMA
innovation, a lightweight NumPy GRU predictor, a lightweight bottleneck
autoencoder, and a transparent ICS/CPS statistical-process-control proxy.

All models are fitted only on train-normal rows. Thresholds are calibrated
only on valid-normal rows at FAR=0.01. Test rows are evaluation-only.
All attacks are transparent controlled injections, not real attack logs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage262_security_domain_baselines"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    ATTACK_PARAMETERS,
    COMPONENTS,
    detector_frame,
    fit_detectors,
    inject_attack,
    metrics,
    score_components,
    sha256_file,
)
from stage208_dwd_persistent_repeat_protection import (  # noqa: E402
    VALID_INJECTION_SEED_OFFSET,
    choose_split_attack_blocks,
    event_metrics,
    far_threshold,
    type_metrics,
)
from stage217_swiss_controlled_attack_evaluation import (  # noqa: E402
    fuse_scores,
    robust_standardize,
)
from stage240_nonstuck_balanced_selection import (  # noqa: E402
    AUDITS,
    NORMALIZED,
    PROTOCOLS,
    make_frame,
)
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_features,
    select_candidate_stage244,
)

TRUST_METHODS = s248.TRUST_METHODS
EXISTING_BASELINES = s248.BASELINE_METHODS
SECURITY_BASELINES = [
    "ewma_deviation",
    "arima_innovation",
    "gru_predictor",
    "lightweight_autoencoder",
    "ics_cps_spc_proxy",
]
METHODS = [*TRUST_METHODS, *SECURITY_BASELINES]
SELECTION_RULE_BY_METHOD = s248.SELECTION_RULE_BY_METHOD

MAX_FIT_ROWS = 100_000
MODEL_SEED = 20260919
EWMA_ALPHA = 0.20
AR_ORDER = 4
GRU_HIDDEN = 8
AE_COMPONENTS = 12


def _fit_reference(
    values: np.ndarray,
    train_normal: np.ndarray,
) -> tuple[float, float]:
    reference = np.asarray(values, dtype=float)[train_normal]
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        return 0.0, 1.0
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    scale = max(1.4826 * mad, 1e-6)
    return median, scale


def _apply_reference(
    values: np.ndarray,
    median: float,
    scale: float,
) -> np.ndarray:
    return (
        np.asarray(values, dtype=float) - float(median)
    ) / max(float(scale), 1e-6)


def _ewma_scores(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(
        frame["observed_value"], errors="coerce"
    ).to_numpy(dtype=float)
    scores = np.zeros(len(values), dtype=float)
    for _, part in frame.groupby("station_id", sort=False):
        index = part.index.to_numpy(dtype=int)
        state = np.nan
        for position, value in enumerate(values[index]):
            if not np.isfinite(value):
                continue
            if np.isfinite(state):
                scores[index[position]] = abs(value - state)
            state = value if not np.isfinite(state) else (
                EWMA_ALPHA * value + (1.0 - EWMA_ALPHA) * state
            )
    return scores


def _fit_ar_coefficients(
    frame: pd.DataFrame,
    train_normal: np.ndarray,
) -> dict[str, np.ndarray]:
    values = pd.to_numeric(
        frame["observed_value"], errors="coerce"
    ).to_numpy(dtype=float)
    train_normal = np.asarray(train_normal, dtype=bool)
    coefficients_by_station: dict[str, np.ndarray] = {}
    for _, part in frame.groupby("station_id", sort=False):
        index = part.index.to_numpy(dtype=int)
        series = values[index]
        local_train = train_normal[index]
        positions = np.arange(AR_ORDER, len(series))
        if positions.size == 0:
            coefficients_by_station[str(part["station_id"].iloc[0])] = (
                np.zeros(AR_ORDER, dtype=float)
            )
            continue
        design = np.column_stack(
            [series[positions - lag] for lag in range(1, AR_ORDER + 1)]
        )
        target = series[positions]
        valid = local_train[positions] & np.isfinite(target) & np.isfinite(
            design
        ).all(axis=1)
        for lag in range(1, AR_ORDER + 1):
            valid &= local_train[positions - lag]
        coefficients = np.zeros(AR_ORDER, dtype=float)
        if valid.sum() > AR_ORDER:
            coefficients = np.linalg.lstsq(
                design[valid], target[valid], rcond=None
            )[0]
        coefficients_by_station[str(part["station_id"].iloc[0])] = (
            coefficients
        )
    return coefficients_by_station


def _ar_scores(
    frame: pd.DataFrame,
    coefficients_by_station: dict[str, np.ndarray],
) -> np.ndarray:
    values = pd.to_numeric(
        frame["observed_value"], errors="coerce"
    ).to_numpy(dtype=float)
    scores = np.zeros(len(values), dtype=float)
    for _, part in frame.groupby("station_id", sort=False):
        index = part.index.to_numpy(dtype=int)
        series = values[index]
        positions = np.arange(AR_ORDER, len(series))
        if positions.size == 0:
            continue
        design = np.column_stack(
            [series[positions - lag] for lag in range(1, AR_ORDER + 1)]
        )
        target = series[positions]
        valid = np.isfinite(target) & np.isfinite(design).all(axis=1)
        coefficients = coefficients_by_station.get(
            str(part["station_id"].iloc[0]),
            np.zeros(AR_ORDER, dtype=float),
        )
        prediction = design @ coefficients
        residual = np.abs(target - prediction)
        residual[~valid] = 0.0
        scores[index[positions]] = residual
    return scores


def _gru_weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(MODEL_SEED)
    input_size = 1 + GRU_HIDDEN
    bound = 1.0 / np.sqrt(input_size)
    return {
        "weight_z": rng.uniform(
            -bound, bound, size=(GRU_HIDDEN, input_size)
        ),
        "weight_r": rng.uniform(
            -bound, bound, size=(GRU_HIDDEN, input_size)
        ),
        "weight_h": rng.uniform(
            -bound, bound, size=(GRU_HIDDEN, input_size)
        ),
        "bias_z": np.zeros(GRU_HIDDEN, dtype=float),
        "bias_r": np.zeros(GRU_HIDDEN, dtype=float),
        "bias_h": np.zeros(GRU_HIDDEN, dtype=float),
    }


def _gru_hidden_rows(
    series: np.ndarray,
    weights: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(series)
    hidden = np.zeros(GRU_HIDDEN, dtype=float)
    hidden_rows = np.zeros((len(series), GRU_HIDDEN), dtype=float)
    for position, value in enumerate(series):
        if valid[position]:
            features = np.concatenate(([float(value)], hidden))
            update = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        weights["weight_z"] @ features + weights["bias_z"],
                        -30.0,
                        30.0,
                    )
                )
            )
            reset = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        weights["weight_r"] @ features + weights["bias_r"],
                        -30.0,
                        30.0,
                    )
                )
            )
            candidate_features = np.concatenate(
                ([float(value)], reset * hidden)
            )
            candidate = np.tanh(
                weights["weight_h"] @ candidate_features
                + weights["bias_h"]
            )
            hidden = (1.0 - update) * hidden + update * candidate
        hidden_rows[position] = hidden
    return hidden_rows, valid


def _fit_gru(
    frame: pd.DataFrame,
    train_normal: np.ndarray,
) -> dict[str, object]:
    values = pd.to_numeric(
        frame["observed_value"], errors="coerce"
    ).to_numpy(dtype=float)
    train_normal = np.asarray(train_normal, dtype=bool)
    input_median, input_scale = _fit_reference(values, train_normal)
    scaled_values = _apply_reference(
        values, input_median, input_scale
    )
    weights = _gru_weights()
    coefficients_by_station: dict[str, np.ndarray] = {}
    for _, part in frame.groupby("station_id", sort=False):
        index = part.index.to_numpy(dtype=int)
        series = scaled_values[index]
        hidden_rows, valid = _gru_hidden_rows(series, weights)
        train_positions = np.flatnonzero(
            train_normal[index] & valid
        )
        if train_positions.size <= GRU_HIDDEN:
            coefficients_by_station[
                str(part["station_id"].iloc[0])
            ] = np.zeros(GRU_HIDDEN + 1, dtype=float)
            continue
        design = np.column_stack(
            (
                hidden_rows[train_positions],
                np.ones(train_positions.size, dtype=float),
            )
        )
        target = series[train_positions]
        coefficients_by_station[
            str(part["station_id"].iloc[0])
        ] = np.linalg.lstsq(design, target, rcond=None)[0]
    return {
        "weights": weights,
        "coefficients_by_station": coefficients_by_station,
        "input_median": input_median,
        "input_scale": input_scale,
    }


def _gru_scores(
    frame: pd.DataFrame,
    fitted: dict[str, object],
) -> np.ndarray:
    values = pd.to_numeric(
        frame["observed_value"], errors="coerce"
    ).to_numpy(dtype=float)
    scaled_values = _apply_reference(
        values,
        float(fitted["input_median"]),
        float(fitted["input_scale"]),
    )
    weights = fitted["weights"]
    coefficients_by_station = fitted["coefficients_by_station"]
    assert isinstance(weights, dict)
    assert isinstance(coefficients_by_station, dict)
    scores = np.zeros(len(values), dtype=float)
    for _, part in frame.groupby("station_id", sort=False):
        index = part.index.to_numpy(dtype=int)
        series = scaled_values[index]
        hidden_rows, valid = _gru_hidden_rows(series, weights)
        coefficients = coefficients_by_station.get(
            str(part["station_id"].iloc[0]),
            np.zeros(GRU_HIDDEN + 1, dtype=float),
        )
        prediction = (
            np.column_stack(
                (hidden_rows, np.ones(len(series), dtype=float))
            )
            @ coefficients
        )
        residual = np.abs(series - prediction)
        residual[~valid] = 0.0
        scores[index] = residual
    return scores


def _fit_ae(
    matrix: np.ndarray,
    fit_indices: np.ndarray,
) -> object:
    return make_pipeline(
        StandardScaler(),
        PCA(
            n_components=min(AE_COMPONENTS, matrix.shape[1]),
            svd_solver="randomized",
            random_state=MODEL_SEED,
        ),
    ).fit(matrix[fit_indices])


def _ae_scores(model: object, matrix: np.ndarray) -> np.ndarray:
    scaled = model.named_steps["standardscaler"].transform(matrix)
    reconstructed = model.named_steps["pca"].inverse_transform(
        model.named_steps["pca"].transform(scaled)
    )
    return np.mean((scaled - reconstructed) ** 2, axis=1)


def _fit_spc(
    frame: pd.DataFrame,
    train_normal: np.ndarray,
) -> dict[str, tuple[float, float]]:
    values = pd.to_numeric(
        frame["observed_value"], errors="coerce"
    ).to_numpy(dtype=float)
    train_normal = np.asarray(train_normal, dtype=bool)
    parameters: dict[str, tuple[float, float]] = {}
    for _, part in frame.groupby("station_id", sort=False):
        index = part.index.to_numpy(dtype=int)
        reference = values[index][train_normal[index]]
        reference = reference[np.isfinite(reference)]
        center = float(np.median(reference)) if reference.size else 0.0
        scale = float(np.median(np.abs(reference - center))) if reference.size else 1.0
        scale = max(1.4826 * scale, 1e-6)
        parameters[str(part["station_id"].iloc[0])] = (center, scale)
    return parameters


def _spc_scores(
    frame: pd.DataFrame,
    parameters: dict[str, tuple[float, float]],
) -> np.ndarray:
    values = pd.to_numeric(
        frame["observed_value"], errors="coerce"
    ).to_numpy(dtype=float)
    scores = np.zeros(len(values), dtype=float)
    for _, part in frame.groupby("station_id", sort=False):
        index = part.index.to_numpy(dtype=int)
        center, scale = parameters.get(
            str(part["station_id"].iloc[0]), (0.0, 1.0)
        )
        scores[index] = np.abs(values[index] - center) / scale
    return scores


def fit_security_baselines(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    train_normal: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    fit_indices = s248.stratified_fit_indices(
        frame, train_normal, MAX_FIT_ROWS, MODEL_SEED
    )
    ar_coefficients = _fit_ar_coefficients(frame, train_normal)
    gru = _fit_gru(frame, train_normal)
    ae = _fit_ae(matrix, fit_indices)
    spc_parameters = _fit_spc(frame, train_normal)
    raw_clean = {
        "ewma_deviation": _ewma_scores(frame),
        "arima_innovation": _ar_scores(frame, ar_coefficients),
        "gru_predictor": _gru_scores(frame, gru),
        "lightweight_autoencoder": _ae_scores(ae, matrix),
        "ics_cps_spc_proxy": _spc_scores(frame, spc_parameters),
    }
    references = {
        method: _fit_reference(values, train_normal)
        for method, values in raw_clean.items()
    }
    fitted = {
        "ar_coefficients": ar_coefficients,
        "gru": gru,
        "ae": ae,
        "spc_parameters": spc_parameters,
        "references": references,
    }
    fit_info = {
        "fit_rows": int(len(fit_indices)),
        "train_normal_rows": int(train_normal.sum()),
        "ewma_alpha": EWMA_ALPHA,
        "ar_order": AR_ORDER,
        "gru_hidden": GRU_HIDDEN,
        "gru_kind": "fixed_random_gated_recurrent_reservoir_with_train_normal_readout",
        "ae_components": int(min(AE_COMPONENTS, matrix.shape[1])),
        "model_seed": MODEL_SEED,
    }
    return fitted, fit_info


def score_security_baselines(
    fitted: dict[str, object],
    frame: pd.DataFrame,
    matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    ar_coefficients = fitted["ar_coefficients"]
    gru = fitted["gru"]
    ae = fitted["ae"]
    spc_parameters = fitted["spc_parameters"]
    references = fitted["references"]
    assert isinstance(ar_coefficients, dict)
    assert isinstance(gru, dict)
    assert isinstance(spc_parameters, dict)
    assert isinstance(references, dict)
    raw = {
        "ewma_deviation": _ewma_scores(frame),
        "arima_innovation": _ar_scores(frame, ar_coefficients),
        "gru_predictor": _gru_scores(frame, gru),
        "lightweight_autoencoder": _ae_scores(ae, matrix),
        "ics_cps_spc_proxy": _spc_scores(frame, spc_parameters),
    }
    return {
        method: _apply_reference(
            values,
            float(references[method][0]),
            float(references[method][1]),
        )
        for method, values in raw.items()
    }


def paired_security_station_table(
    station_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in SECURITY_BASELINES:
            trust = station_metrics[
                station_metrics["method"].eq(candidate)
            ]
            reference = station_metrics[
                station_metrics["method"].eq(baseline)
            ]
            aligned = trust.merge(
                reference,
                on=["source", "seed", "station_id"],
                suffixes=("_trust", "_baseline"),
                validate="one_to_one",
            )
            for row in aligned.to_dict("records"):
                rows.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "source": str(row["source"]),
                        "seed": int(row["seed"]),
                        "station_id": str(row["station_id"]),
                        "f1_difference": float(
                            row["f1_trust"] - row["f1_baseline"]
                        ),
                        "event_coverage_difference": float(
                            row["event_coverage_trust"]
                            - row["event_coverage_baseline"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []

    for source, protocol_path in PROTOCOLS.items():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        frame = make_frame(source, protocol)
        base = detector_frame(frame)
        detectors = fit_detectors(base, protocol)
        train_normal = (
            base["split"].eq("train") & base["observation_present"]
        ).to_numpy(dtype=bool)
        base_components = score_components(base, detectors)
        base_fused, _ = fuse_scores(base, base_components)
        base_z = robust_standardize(base_fused, train_normal).to_numpy(dtype=float)
        base_features = candidate_features(base, train_normal)
        base_matrix, feature_names = s248.feature_matrix(
            base_components, base_features, train_normal
        )
        fitted_security, fit_info = fit_security_baselines(
            base, base_matrix, train_normal
        )
        primary_far = float(protocol["primary_far"])
        station_ids = sorted(frame["station_id"].astype(str).unique().tolist())
        fit_rows.append(
            {
                "source": source,
                "feature_names": feature_names,
                **fit_info,
            }
        )

        for seed in protocol["seeds"]:
            seed = int(seed)
            valid_blocks = choose_split_attack_blocks(
                base, protocol, seed, "valid"
            )
            test_blocks = choose_split_attack_blocks(
                base, protocol, seed, "test"
            )
            valid_attacked = inject_attack(
                base, valid_blocks, seed + VALID_INJECTION_SEED_OFFSET
            )
            test_attacked = inject_attack(base, test_blocks, seed)
            for attacked, split in (
                (valid_attacked, "valid"),
                (test_attacked, "test"),
            ):
                attacked["attack_event_id"] = np.where(
                    attacked["is_attack"],
                    attacked["station_id"].astype(str)
                    + "__"
                    + attacked["attack_type"].astype(str)
                    + "__"
                    + split
                    + "__"
                    + str(seed),
                    "",
                )
            valid_mask = valid_attacked["split"].eq("valid").to_numpy(
                dtype=bool
            )
            test_mask = test_attacked["split"].eq("test").to_numpy(
                dtype=bool
            )
            valid_frame = valid_attacked.loc[valid_mask].reset_index(
                drop=True
            )
            test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
            valid_normal = (
                valid_frame["observation_present"]
                & ~valid_frame["is_attack"]
            ).to_numpy(dtype=bool)
            valid_components = score_components(valid_attacked, detectors)
            valid_fused, _ = fuse_scores(valid_attacked, valid_components)
            valid_base_z = robust_standardize(
                valid_fused, train_normal
            ).to_numpy(dtype=float)
            test_components = score_components(test_attacked, detectors)
            test_fused, _ = fuse_scores(test_attacked, test_components)
            test_base_z = robust_standardize(
                test_fused, train_normal
            ).to_numpy(dtype=float)
            valid_features = candidate_features(
                valid_attacked, train_normal
            )
            test_features = candidate_features(
                test_attacked, train_normal
            )
            valid_matrix, _ = s248.feature_matrix(
                valid_components,
                valid_features,
                train_normal,
            )
            test_matrix, _ = s248.feature_matrix(
                test_components,
                test_features,
                train_normal,
            )
            valid_security = score_security_baselines(
                fitted_security,
                valid_attacked,
                valid_matrix,
            )
            test_security = score_security_baselines(
                fitted_security,
                test_attacked,
                test_matrix,
            )
            valid_base = valid_base_z[valid_mask]
            test_base = test_base_z[test_mask]
            valid_feature_masked = {
                key: values[valid_mask]
                for key, values in valid_features.items()
            }
            test_feature_masked = {
                key: values[test_mask]
                for key, values in test_features.items()
            }

            valid_scores: dict[str, np.ndarray] = {}
            test_scores: dict[str, np.ndarray] = {}
            for method, rule in SELECTION_RULE_BY_METHOD.items():
                selected = select_candidate_stage244(
                    rule,
                    valid_base,
                    valid_feature_masked,
                    valid_frame,
                    valid_normal,
                    primary_far,
                )
                key = (
                    str(selected["feature"]),
                    int(selected["window"]),
                )
                valid_scores[method] = (
                    (1.0 - float(selected["weight"])) * valid_base
                    + float(selected["weight"]) * valid_feature_masked[key]
                )
                test_scores[method] = (
                    (1.0 - float(selected["weight"])) * test_base
                    + float(selected["weight"]) * test_feature_masked[key]
                )
                selection_rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "method": method,
                        "rule": rule,
                        "feature": selected["feature"],
                        "window": int(selected["window"]),
                        "weight": float(selected["weight"]),
                        "threshold": float(selected["threshold"]),
                    }
                )
            for method in SECURITY_BASELINES:
                valid_scores[method] = valid_security[method][valid_mask]
                test_scores[method] = test_security[method][test_mask]

            y_test = test_frame["is_attack"].astype(bool).to_numpy()
            station_values = test_frame["station_id"].astype(str).to_numpy()
            for method in METHODS:
                valid_score = np.asarray(valid_scores[method], dtype=float)
                test_score = np.asarray(test_scores[method], dtype=float)
                threshold = far_threshold(
                    valid_score, valid_normal, primary_far
                )
                predicted = (
                    test_frame["observation_present"].to_numpy(dtype=bool)
                    & (test_score >= threshold)
                )
                row_metrics = metrics(y_test, predicted, test_score)
                event_values = event_metrics(test_frame, predicted)
                type_values = type_metrics(
                    test_frame, predicted, test_score
                )
                evaluation_rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "method": method,
                        "threshold": float(threshold),
                        **row_metrics,
                        **event_values,
                    }
                )
                for attack_type, values in type_values.items():
                    type_rows.append(
                        {
                            "source": source,
                            "seed": seed,
                            "method": method,
                            "attack_type": attack_type,
                            **values,
                        }
                    )
                event_frame = test_frame.loc[y_test].copy()
                event_frame["predicted"] = predicted[y_test]
                event_grouped = (
                    event_frame.groupby(
                        ["station_id", "attack_event_id"],
                        sort=True,
                        as_index=False,
                    )["predicted"]
                    .any()
                )
                for station_id in station_ids:
                    mask = station_values == station_id
                    y_station = y_test[mask]
                    pred_station = predicted[mask]
                    station_events = event_grouped[
                        event_grouped["station_id"].eq(station_id)
                    ]
                    station_rows.append(
                        {
                            "source": source,
                            "seed": seed,
                            "method": method,
                            "station_id": station_id,
                            "rows": int(mask.sum()),
                            "attack_rows": int(y_station.sum()),
                            "normal_rows": int((~y_station).sum()),
                            "tp": int((y_station & pred_station).sum()),
                            "fp": int((~y_station & pred_station).sum()),
                            "fn": int((y_station & ~pred_station).sum()),
                            "tn": int((~y_station & ~pred_station).sum()),
                            "events": int(len(station_events)),
                            "detected_events": int(
                                station_events["predicted"].sum()
                            ),
                        }
                    )
        print(
            json.dumps(
                {
                    "source": source,
                    "status": "completed",
                    "seeds": len(protocol["seeds"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    evaluation = pd.DataFrame(evaluation_rows)
    station = pd.DataFrame(station_rows)
    type_detail = pd.DataFrame(type_rows)
    selections = pd.DataFrame(selection_rows)
    evaluation_aggregate = (
        evaluation.groupby(["source", "method"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            false_alarm_rate_mean=("false_alarm_rate", "mean"),
            average_precision_mean=("average_precision", "mean"),
            event_coverage_mean=("event_coverage", "mean"),
            event_coverage_std=("event_coverage", "std"),
            row_recall_mean=("row_recall", "mean"),
        )
        .sort_values(["source", "method"])
    )
    provider_equal = (
        evaluation_aggregate.groupby("method", as_index=False)
        .agg(
            provider_count=("source", "nunique"),
            f1_mean=("f1_mean", "mean"),
            event_coverage_mean=("event_coverage_mean", "mean"),
            row_recall_mean=("row_recall_mean", "mean"),
        )
        .sort_values("method")
    )
    station_metrics = s248.station_metric_table(station)
    paired = paired_security_station_table(station_metrics)
    comparison_rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in SECURITY_BASELINES:
            comparison_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    **s248.hierarchical_bootstrap(
                        paired, candidate, baseline
                    ),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["f1_bootstrap_bh_q"] = s248.benjamini_hochberg(
        comparisons["f1_bootstrap_two_sided_p"].tolist()
    )
    comparisons["event_coverage_bootstrap_bh_q"] = (
        s248.benjamini_hochberg(
            comparisons["event_coverage_bootstrap_two_sided_p"].tolist()
        )
    )
    comparisons["f1_primary_supported"] = (
        comparisons["f1_ci95_low"].gt(0.0)
        & comparisons["f1_bootstrap_bh_q"].le(0.05)
    )
    comparisons["event_coverage_primary_supported"] = (
        comparisons["event_ci95_low"].gt(0.0)
        & comparisons["event_coverage_bootstrap_bh_q"].le(0.05)
    )
    leave_one_rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in SECURITY_BASELINES:
            leave_one_rows.extend(
                s248.leave_one_provider_bootstrap(
                    paired, candidate, baseline
                )
            )
    leave_one = pd.DataFrame(leave_one_rows)
    type_aggregate = (
        type_detail.groupby(
            ["source", "method", "attack_type"], as_index=False
        )
        .agg(
            seeds=("seed", "nunique"),
            attack_rows=("attack_rows", "sum"),
            detected_rows=("detected_rows", "sum"),
            row_recall=("row_recall", "mean"),
            event_coverage=("event_coverage", "mean"),
        )
        .sort_values(["source", "method", "attack_type"])
    )
    payload = {
        "status": "PASS_EXPLORATORY_SECURITY_BASELINE_AUDIT",
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "protocol": (
            "阶段262在阶段248冻结的四个目标机构、train/valid/test边界、"
            "透明受控攻击和FAR=0.01主终点上补充五个安全/时序强基线。"
            "所有模型只在train正常行拟合；阈值只由valid正常行校准；"
            "test只评价。所有新结果均标为探索性，不替换阶段246/248。"
        ),
        "primary_far": 0.01,
        "primary_endpoint": "event_coverage",
        "provider_count": int(len(PROTOCOLS)),
        "station_count": int(
            station[["source", "station_id"]].drop_duplicates().shape[0]
        ),
        "seed_count": int(evaluation["seed"].nunique()),
        "method_count": int(len(METHODS)),
        "methods": METHODS,
        "new_baseline_methods": SECURITY_BASELINES,
        "existing_baseline_methods": EXISTING_BASELINES,
        "fit_rows": fit_rows,
        "evaluation_rows": int(len(evaluation)),
        "station_rows": int(len(station)),
        "type_rows": int(len(type_detail)),
        "selection_rows": int(len(selections)),
        "provider_equal": s248.clean_records(
            provider_equal.to_dict("records")
        ),
        "comparisons": s248.clean_records(
            comparisons.to_dict("records")
        ),
        "leave_one_provider_out": s248.clean_records(
            leave_one.to_dict("records")
        ),
        "attack_parameters": ATTACK_PARAMETERS,
        "components": COMPONENTS,
        "protocol_sha256": {
            source: sha256_file(path)
            for source, path in PROTOCOLS.items()
        },
        "normalized_sha256": {
            source: sha256_file(path)
            for source, path in NORMALIZED.items()
        },
        "audit_sha256": {
            source: sha256_file(path)
            for source, path in AUDITS.items()
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "limitations": [
            "这是探索性安全基线审计，不替换阶段246/248的确认性主结果。",
            "ICS/CPS SPC proxy是透明统计过程控制代理，不是原论文模型的逐位复现。",
            "GRU predictor是固定随机门控循环、仅训练线性读出的轻量reservoir代理，不代表大规模深度模型上界。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
            "如果新基线占优，结果必须原样保留。",
            "四个机构不能支持机构无关或站点无关的普适声称。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    evaluation.to_csv(
        TABLES / f"{PREFIX}_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station.to_csv(
        TABLES / f"{PREFIX}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_detail.to_csv(
        TABLES / f"{PREFIX}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selections.to_csv(
        TABLES / f"{PREFIX}_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evaluation_aggregate.to_csv(
        TABLES / f"{PREFIX}_evaluation_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provider_equal.to_csv(
        TABLES / f"{PREFIX}_provider_equal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        TABLES / f"{PREFIX}_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one.to_csv(
        TABLES / f"{PREFIX}_leave_one_provider_out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_aggregate.to_csv(
        TABLES / f"{PREFIX}_type_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段262 安全领域强基线探索性补充",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 探索性：`{payload['exploratory_only']}`",
        f"- 不属于冻结主结果：`{payload['not_part_of_frozen_main_results']}`",
        f"- 机构数：`{payload['provider_count']}`",
        f"- 站点数：`{payload['station_count']}`",
        f"- 种子数：`{payload['seed_count']}`",
        f"- 新增基线：`{', '.join(SECURITY_BASELINES)}`",
        "",
        "## 边界",
        "",
        "- 所有模型只在train正常行拟合，阈值只由valid正常行校准。",
        "- test只评价，不用于选择模型、特征、窗口、权重或阈值。",
        "- ICS/CPS基线是可复现SPC代理，不声称原论文逐位复现。",
        "- 所有攻击均为透明受控注入，不是真实攻击日志。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "provider_count": payload["provider_count"],
                "method_count": payload["method_count"],
                "new_baselines": SECURITY_BASELINES,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
