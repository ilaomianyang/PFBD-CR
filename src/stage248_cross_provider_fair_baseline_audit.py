from __future__ import annotations

"""Stage 248: cross-provider fair low-FAR baseline audit.

The frozen Stage219/222/228/238 transfer results compare the proposed
target-domain-selected rule against a target-domain-recalibrated transfer
arm.  This stage adds the missing reviewer-facing comparison: the same
four external target providers, the same train/valid/test boundaries, the
same transparent controlled attack sequences, and the same primary
``FAR=0.01`` endpoint, but now with unsupervised baselines fitted only on
train-normal rows.

Methods
-------
* ``trust_valid_f1``: the proposed candidate selected on target valid by F1.
* ``trust_nonstuck_event_macro``: the proposed candidate selected on target
  valid by non-stuck event coverage.
* ``robust_max_deviation``: maximum positive robust standardized feature
  deviation.
* ``isolation_forest``: Isolation Forest anomaly score.
* ``one_class_svm``: RBF One-Class SVM anomaly score.

All thresholds are calibrated from target valid-normal rows only.  Test is
evaluation-only.  The One-Class SVM fit uses a deterministic train-normal
subsample because kernel SVM fitting on several hundred thousand rows is
not computationally practical; this cap is explicit in the payload and
does not use valid or test labels.  All attacks are transparent controlled
injections, not real attack logs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage248_cross_provider_fair_baseline_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
    ATTACK_TYPES,
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
from stage218_swiss_negative_result_uncertainty import (  # noqa: E402
    exact_sign_flip_two_sided,
)
from stage219_swiss_dwd_protection_transfer import (  # noqa: E402
    clean_records,
)
from stage240_nonstuck_balanced_selection import (  # noqa: E402
    AUDITS,
    NORMALIZED,
    PROTOCOLS,
    make_frame,
)
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    ALL_FEATURES,
    candidate_features,
    select_candidate_stage244,
)

TRUST_METHODS = [
    "trust_valid_f1",
    "trust_nonstuck_event_macro",
]
BASELINE_METHODS = [
    "robust_max_deviation",
    "isolation_forest",
    "one_class_svm",
]
METHODS = [*TRUST_METHODS, *BASELINE_METHODS]
SELECTION_RULE_BY_METHOD = {
    "trust_valid_f1": "valid_f1",
    "trust_nonstuck_event_macro": "valid_nonstuck_event_macro",
}
MAX_OCSVM_FIT_ROWS = 20000
BASELINE_FIT_SEED = 20260919


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def component_matrix(
    components: pd.DataFrame,
    train_normal: np.ndarray,
) -> np.ndarray:
    columns = [
        robust_standardize(
            components[name],
            pd.Series(train_normal, index=components.index),
        ).to_numpy(dtype=float)
        for name in COMPONENTS
    ]
    return np.column_stack(columns)


def candidate_matrix(
    features: dict[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    columns: list[np.ndarray] = []
    for feature_name in ALL_FEATURES:
        windows = (2, 3, 6) if feature_name in {"repeat_ratio", "repeat_run"} else (1,)
        for window in windows:
            key = (str(feature_name), int(window))
            names.append(f"{feature_name}_w{window}")
            columns.append(np.asarray(features[key], dtype=float))
    return np.column_stack(columns), names


def feature_matrix(
    components: pd.DataFrame,
    candidate_values: dict[tuple[str, int], np.ndarray],
    train_normal: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    component_values = component_matrix(components, train_normal)
    component_names = [f"component_{name}" for name in COMPONENTS]
    candidate_values_matrix, candidate_names = candidate_matrix(candidate_values)
    matrix = np.column_stack([component_values, candidate_values_matrix])
    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return matrix, [*component_names, *candidate_names]


def stratified_fit_indices(
    frame: pd.DataFrame,
    train_normal: np.ndarray,
    limit: int,
    seed: int,
) -> np.ndarray:
    positions = np.flatnonzero(train_normal)
    if positions.size <= limit:
        return positions
    station_ids = frame["station_id"].astype(str).to_numpy()
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    per_station = max(1, int(np.ceil(limit / frame["station_id"].nunique())))
    for station_id in sorted(frame["station_id"].astype(str).unique()):
        station_positions = positions[station_ids[positions] == station_id]
        if station_positions.size > per_station:
            draw = rng.choice(
                station_positions,
                size=per_station,
                replace=False,
            )
        else:
            draw = station_positions
        selected.extend(int(value) for value in draw)
    if len(selected) > limit:
        selected = [
            int(value)
            for value in rng.choice(
                np.asarray(selected, dtype=int),
                size=limit,
                replace=False,
            )
        ]
    return np.asarray(sorted(selected), dtype=int)


def fit_baselines(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    train_normal: np.ndarray,
) -> dict[str, object]:
    train_x = matrix[train_normal]
    robust = {
        "name": "robust_max_deviation",
        "kind": "maximum_positive_robust_deviation",
    }
    isolation = IsolationForest(
        n_estimators=300,
        random_state=BASELINE_FIT_SEED,
        n_jobs=1,
        contamination="auto",
    ).fit(train_x)
    fit_indices = stratified_fit_indices(
        frame,
        train_normal,
        MAX_OCSVM_FIT_ROWS,
        BASELINE_FIT_SEED,
    )
    ocsvm = make_pipeline(
        StandardScaler(),
        OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05,
            cache_size=1000,
            max_iter=-1,
        ),
    ).fit(matrix[fit_indices])
    return {
        "robust_max_deviation": robust,
        "isolation_forest": isolation,
        "one_class_svm": ocsvm,
    }


def score_baselines(
    fitted: dict[str, object],
    matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "robust_max_deviation": np.max(matrix, axis=1),
        "isolation_forest": -fitted["isolation_forest"].score_samples(matrix),
        "one_class_svm": -fitted["one_class_svm"].decision_function(
            matrix
        ).ravel(),
    }


def station_metric_table(station: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        station.groupby(
            ["source", "seed", "method", "station_id"],
            as_index=False,
        )
        .agg(
            rows=("rows", "sum"),
            attack_rows=("attack_rows", "sum"),
            normal_rows=("normal_rows", "sum"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tn=("tn", "sum"),
            events=("events", "sum"),
            detected_events=("detected_events", "sum"),
        )
    )
    grouped["f1"] = np.divide(
        2.0 * grouped["tp"],
        2.0 * grouped["tp"] + grouped["fp"] + grouped["fn"],
        out=np.zeros(len(grouped), dtype=float),
        where=(
            2.0 * grouped["tp"] + grouped["fp"] + grouped["fn"]
        ).to_numpy()
        > 0,
    )
    grouped["event_coverage"] = np.divide(
        grouped["detected_events"],
        grouped["events"],
        out=np.zeros(len(grouped), dtype=float),
        where=grouped["events"].to_numpy() > 0,
    )
    return grouped


def paired_station_table(
    station_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in BASELINE_METHODS:
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


def comparison_summary(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby(["candidate", "baseline", "source"], as_index=False)
        .agg(
            stations=("station_id", "nunique"),
            f1_difference=("f1_difference", "mean"),
            event_coverage_difference=("event_coverage_difference", "mean"),
        )
        .sort_values(["candidate", "baseline", "source"])
    )


def hierarchical_bootstrap(
    paired: pd.DataFrame,
    candidate: str,
    baseline: str,
    iterations: int = 10000,
) -> dict[str, float]:
    current = paired[
        paired["candidate"].eq(candidate)
        & paired["baseline"].eq(baseline)
    ]
    providers = sorted(current["source"].unique())
    by_provider = {
        provider: current.loc[
            current["source"].eq(provider),
            ["f1_difference", "event_coverage_difference"],
        ].to_numpy(dtype=float)
        for provider in providers
    }
    rng = np.random.default_rng(20260920)
    f1_draws = np.empty(iterations, dtype=float)
    event_draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        f1_means: list[float] = []
        event_means: list[float] = []
        for provider_index in rng.integers(
            0,
            len(providers),
            size=len(providers),
        ):
            values = by_provider[providers[int(provider_index)]]
            sampled = values[
                rng.integers(0, len(values), size=len(values))
            ]
            f1_means.append(float(sampled[:, 0].mean()))
            event_means.append(float(sampled[:, 1].mean()))
        f1_draws[index] = float(np.mean(f1_means))
        event_draws[index] = float(np.mean(event_means))
    provider_f1 = (
        current.groupby("source", as_index=False)["f1_difference"].mean()[
            "f1_difference"
        ].to_numpy(dtype=float)
    )
    provider_event = (
        current.groupby("source", as_index=False)[
            "event_coverage_difference"
        ].mean()["event_coverage_difference"].to_numpy(dtype=float)
    )
    f1_bootstrap_p = min(
        1.0,
        2.0
        * min(
            float(np.mean(f1_draws <= 0.0)),
            float(np.mean(f1_draws >= 0.0)),
        ),
    )
    event_bootstrap_p = min(
        1.0,
        2.0
        * min(
            float(np.mean(event_draws <= 0.0)),
            float(np.mean(event_draws >= 0.0)),
        ),
    )
    return {
        "provider_count": int(len(providers)),
        "station_rows": int(len(current)),
        "iterations": int(iterations),
        "f1_difference": float(np.mean(provider_f1)),
        "f1_ci95_low": float(np.quantile(f1_draws, 0.025)),
        "f1_ci95_high": float(np.quantile(f1_draws, 0.975)),
        "event_coverage_difference": float(np.mean(provider_event)),
        "event_ci95_low": float(np.quantile(event_draws, 0.025)),
        "event_ci95_high": float(np.quantile(event_draws, 0.975)),
        "provider_sign_flip_f1_p": float(
            exact_sign_flip_two_sided(provider_f1)
        ),
        "provider_sign_flip_event_p": float(
            exact_sign_flip_two_sided(provider_event)
        ),
        "f1_bootstrap_two_sided_p": float(f1_bootstrap_p),
        "event_coverage_bootstrap_two_sided_p": float(
            event_bootstrap_p
        ),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    count = len(values)
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        rank = reverse_index + 1
        running = min(running, float(ranked[reverse_index]) * count / rank)
        adjusted[reverse_index] = min(1.0, running)
    output = np.empty(count, dtype=float)
    output[order] = adjusted
    return output.tolist()


def leave_one_provider_bootstrap(
    paired: pd.DataFrame,
    candidate: str,
    baseline: str,
    iterations: int = 10000,
) -> list[dict[str, object]]:
    current = paired[
        paired["candidate"].eq(candidate)
        & paired["baseline"].eq(baseline)
    ]
    rows: list[dict[str, object]] = []
    for removed_provider in sorted(current["source"].unique()):
        retained = current[current["source"].ne(removed_provider)]
        providers = sorted(retained["source"].unique())
        by_provider = {
            provider: retained.loc[
                retained["source"].eq(provider),
                ["f1_difference", "event_coverage_difference"],
            ].to_numpy(dtype=float)
            for provider in providers
        }
        rng = np.random.default_rng(20260920)
        f1_draws = np.empty(iterations, dtype=float)
        event_draws = np.empty(iterations, dtype=float)
        for index in range(iterations):
            f1_means: list[float] = []
            event_means: list[float] = []
            for provider_index in rng.integers(
                0,
                len(providers),
                size=len(providers),
            ):
                values = by_provider[providers[int(provider_index)]]
                sampled = values[
                    rng.integers(0, len(values), size=len(values))
                ]
                f1_means.append(float(sampled[:, 0].mean()))
                event_means.append(float(sampled[:, 1].mean()))
            f1_draws[index] = float(np.mean(f1_means))
            event_draws[index] = float(np.mean(event_means))
        provider_f1 = (
            retained.groupby("source", as_index=False)["f1_difference"].mean()[
                "f1_difference"
            ].to_numpy(dtype=float)
        )
        provider_event = (
            retained.groupby("source", as_index=False)[
                "event_coverage_difference"
            ].mean()["event_coverage_difference"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "removed_provider": str(removed_provider),
                "retained_provider_count": int(len(providers)),
                "station_rows": int(len(retained)),
                "iterations": int(iterations),
                "f1_difference": float(np.mean(provider_f1)),
                "f1_ci95_low": float(np.quantile(f1_draws, 0.025)),
                "f1_ci95_high": float(np.quantile(f1_draws, 0.975)),
                "f1_direction_positive": bool(np.mean(provider_f1) > 0.0),
                "event_coverage_difference": float(np.mean(provider_event)),
                "event_ci95_low": float(np.quantile(event_draws, 0.025)),
                "event_ci95_high": float(np.quantile(event_draws, 0.975)),
                "event_direction_positive": bool(
                    np.mean(provider_event) > 0.0
                ),
            }
        )
    return rows


def main() -> int:
    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []

    for source, protocol_path in PROTOCOLS.items():
        protocol = load_json(protocol_path)
        frame = make_frame(source, protocol)
        base = detector_frame(frame)
        detectors = fit_detectors(base, protocol)
        train_normal = (
            base["split"].eq("train") & base["observation_present"]
        ).to_numpy(dtype=bool)
        base_components = score_components(base, detectors)
        base_fused, _ = fuse_scores(base, base_components)
        base_z = robust_standardize(base_fused, train_normal).to_numpy(
            dtype=float
        )
        base_features = candidate_features(base, train_normal)
        base_matrix, feature_names = feature_matrix(
            base_components,
            base_features,
            train_normal,
        )
        fitted = fit_baselines(base, base_matrix, train_normal)
        primary_far = float(protocol["primary_far"])
        station_ids = sorted(
            frame["station_id"].astype(str).unique().tolist()
        )
        fit_rows.append(
            {
                "source": source,
                "train_normal_rows": int(train_normal.sum()),
                "feature_count": int(base_matrix.shape[1]),
                "feature_names": feature_names,
                "ocsvm_fit_rows": int(
                    min(MAX_OCSVM_FIT_ROWS, train_normal.sum())
                ),
                "ocsvm_subsampled": bool(
                    train_normal.sum() > MAX_OCSVM_FIT_ROWS
                ),
                "fit_seed": BASELINE_FIT_SEED,
            }
        )
        for seed in protocol["seeds"]:
            seed = int(seed)
            valid_blocks = choose_split_attack_blocks(
                base,
                protocol,
                seed,
                "valid",
            )
            test_blocks = choose_split_attack_blocks(
                base,
                protocol,
                seed,
                "test",
            )
            valid_attacked = inject_attack(
                base,
                valid_blocks,
                seed + VALID_INJECTION_SEED_OFFSET,
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

            valid_components = score_components(valid_attacked, detectors)
            valid_fused, _ = fuse_scores(valid_attacked, valid_components)
            valid_base_z = robust_standardize(
                valid_fused,
                train_normal,
            ).to_numpy(dtype=float)
            test_components = score_components(test_attacked, detectors)
            test_fused, _ = fuse_scores(test_attacked, test_components)
            test_base_z = robust_standardize(
                test_fused,
                train_normal,
            ).to_numpy(dtype=float)
            valid_features = candidate_features(
                valid_attacked,
                train_normal,
            )
            test_features = candidate_features(
                test_attacked,
                train_normal,
            )
            valid_matrix, _ = feature_matrix(
                valid_components,
                valid_features,
                train_normal,
            )
            test_matrix, _ = feature_matrix(
                test_components,
                test_features,
                train_normal,
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
            if not valid_frame["is_attack"].any():
                raise RuntimeError(f"{source} valid attack injection is empty")
            if not test_frame["is_attack"].any():
                raise RuntimeError(f"{source} test attack injection is empty")
            valid_normal = (
                valid_frame["observation_present"]
                & ~valid_frame["is_attack"]
            ).to_numpy(dtype=bool)
            valid_base = valid_base_z[valid_mask]
            test_base = test_base_z[test_mask]
            valid_baseline_scores = score_baselines(
                fitted,
                valid_matrix[valid_mask],
            )
            test_baseline_scores = score_baselines(
                fitted,
                test_matrix[test_mask],
            )
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
                test_scores[method] = (
                    (1.0 - float(selected["weight"])) * test_base
                    + float(selected["weight"])
                    * test_feature_masked[key]
                )
                valid_scores[method] = (
                    (1.0 - float(selected["weight"])) * valid_base
                    + float(selected["weight"])
                    * valid_feature_masked[key]
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
                        "valid_f1": float(selected["valid_f1"]),
                        "valid_nonstuck_event_macro": float(
                            selected["valid_nonstuck_event_macro"]
                        ),
                        "valid_event_macro": float(
                            selected["valid_event_macro"]
                        ),
                        "valid_stuck_event": float(
                            selected["valid_stuck_event"]
                        ),
                    }
                )
            for method in BASELINE_METHODS:
                valid_scores[method] = valid_baseline_scores[method]
                test_scores[method] = test_baseline_scores[method]

            y_test = test_frame["is_attack"].astype(bool).to_numpy()
            station_values = test_frame["station_id"].astype(str).to_numpy()
            for method in METHODS:
                valid_score = np.asarray(valid_scores[method], dtype=float)
                test_score = np.asarray(test_scores[method], dtype=float)
                threshold = far_threshold(
                    valid_score,
                    valid_normal,
                    primary_far,
                )
                predicted = (
                    test_frame["observation_present"].to_numpy(dtype=bool)
                    & (test_score >= threshold)
                )
                row_metrics = metrics(y_test, predicted, test_score)
                event_values = event_metrics(test_frame, predicted)
                type_values = type_metrics(
                    test_frame,
                    predicted,
                    test_score,
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
    station_metrics = station_metric_table(station)
    paired = paired_station_table(station_metrics)
    provider_comparison = comparison_summary(paired)
    comparison_rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in BASELINE_METHODS:
            comparison_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    **hierarchical_bootstrap(
                        paired,
                        candidate,
                        baseline,
                    ),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["f1_bootstrap_bh_q"] = benjamini_hochberg(
        comparisons["f1_bootstrap_two_sided_p"].tolist()
    )
    comparisons["event_coverage_bootstrap_bh_q"] = benjamini_hochberg(
        comparisons["event_coverage_bootstrap_two_sided_p"].tolist()
    )
    comparisons["provider_sign_flip_f1_bh_q"] = benjamini_hochberg(
        comparisons["provider_sign_flip_f1_p"].tolist()
    )
    comparisons["provider_sign_flip_event_bh_q"] = benjamini_hochberg(
        comparisons["provider_sign_flip_event_p"].tolist()
    )
    comparisons["f1_primary_supported"] = (
        comparisons["f1_ci95_low"].gt(0.0)
        & comparisons["f1_bootstrap_bh_q"].le(0.05)
    )
    comparisons["event_coverage_primary_supported"] = (
        comparisons["event_ci95_low"].gt(0.0)
        & comparisons["event_coverage_bootstrap_bh_q"].le(0.05)
    )
    leave_one_provider_rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in BASELINE_METHODS:
            leave_one_provider_rows.extend(
                leave_one_provider_bootstrap(
                    paired,
                    candidate,
                    baseline,
                )
            )
    leave_one_provider = pd.DataFrame(leave_one_provider_rows)
    type_aggregate = (
        type_detail.groupby(
            ["source", "method", "attack_type"],
            as_index=False,
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
    wins = (
        paired.assign(
            f1_win=paired["f1_difference"] > 0.0,
            event_win=paired["event_coverage_difference"] > 0.0,
        )
        .groupby(["candidate", "baseline"], as_index=False)
        .agg(
            paired_count=("f1_difference", "size"),
            f1_win_rate=("f1_win", "mean"),
            event_win_rate=("event_win", "mean"),
            f1_difference=("f1_difference", "mean"),
            event_coverage_difference=("event_coverage_difference", "mean"),
        )
        .sort_values(["candidate", "baseline"])
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段248在四个外部目标机构上补齐低FAR公平基线比较。"
            "所有方法使用相同train/valid/test边界、相同透明受控攻击"
            "序列、相同valid正常样本阈值的FAR=0.01主终点，test只评价。"
            "Trust候选按valid_f1或valid_nonstuck_event_macro选择；"
            "Robust max deviation、Isolation Forest和One-Class SVM"
            "只在train正常行拟合。One-Class SVM仅因训练规模使用"
            "确定性子采样，不使用valid/test标签，并在fit_rows中披露。"
            "结果只支持对Robust max deviation和Isolation Forest的"
            "稳定主终点优势；相对One-Class SVM仅保留互补性，"
            "不能宣称全面领先。"
            "所有攻击都是透明受控注入，不是真实攻击日志。"
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
        "baseline_methods": BASELINE_METHODS,
        "ocsvm_fit_cap": MAX_OCSVM_FIT_ROWS,
        "fit_rows": fit_rows,
        "evaluation_rows": int(len(evaluation)),
        "station_rows": int(len(station)),
        "type_rows": int(len(type_detail)),
        "selection_rows": int(len(selections)),
        "provider_equal_rows": clean_records(
            provider_equal.to_dict("records")
        ),
        "comparison_rows": clean_records(
            comparisons.to_dict("records")
        ),
        "leave_one_provider_rows": clean_records(
            leave_one_provider.to_dict("records")
        ),
        "paired_win_rows": clean_records(wins.to_dict("records")),
        "provider_comparison_rows": clean_records(
            provider_comparison.to_dict("records")
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
            "这是跨机构公平基线补充审计，不替换阶段219/222/228/238的冻结主结果。",
            "One-Class SVM训练子采样是计算预算约束，必须与全训练集结果分开解释。",
            "四个目标机构的站点网络、缺失结构和攻击注入位置不同。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
            "如果某基线占优，该结果必须保留，不能只报告有利对比。",
            "本审计不能宣称对所有公平基线全面领先。",
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
    evaluation_aggregate.to_csv(
        TABLES / f"{PREFIX}_evaluation_aggregate.csv",
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
    type_aggregate.to_csv(
        TABLES / f"{PREFIX}_type_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selections.to_csv(
        TABLES / f"{PREFIX}_selection.csv",
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
    leave_one_provider.to_csv(
        TABLES / f"{PREFIX}_leave_one_provider_out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provider_comparison.to_csv(
        TABLES / f"{PREFIX}_provider_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wins.to_csv(
        TABLES / f"{PREFIX}_paired_win.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段248 跨机构低FAR公平基线审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 机构数：`{payload['provider_count']}`；站点数：`{payload['station_count']}`；方法数：`{payload['method_count']}`",
        f"- 主FAR：`{payload['primary_far']}`；主终点：`{payload['primary_endpoint']}`",
        f"- One-Class SVM训练上限：`{MAX_OCSVM_FIT_ROWS}` 行",
        "",
        "## 机构等权结果",
        "",
        "| 方法 | F1均值 | 事件覆盖均值 | 行召回均值 |",
        "|---|---:|---:|---:|",
    ]
    for row in provider_equal.to_dict("records"):
        lines.append(
            f"| {row['method']} | {row['f1_mean']:.6f} | "
            f"{row['event_coverage_mean']:.6f} | "
            f"{row['row_recall_mean']:.6f} |"
        )
    lines += [
        "",
        "## Trust与基线配对比较",
        "",
        "| Trust | 基线 | F1差值 | F1 95% CI | 事件覆盖差值 | 事件95% CI | F1胜率 | 事件胜率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    win_lookup = wins.set_index(["candidate", "baseline"])
    for row in comparisons.to_dict("records"):
        win = win_lookup.loc[(row["candidate"], row["baseline"])]
        lines.append(
            f"| {row['candidate']} | {row['baseline']} | "
            f"{row['f1_difference']:+.6f} | "
            f"[{row['f1_ci95_low']:+.6f}, {row['f1_ci95_high']:+.6f}] | "
            f"{row['event_coverage_difference']:+.6f} | "
            f"[{row['event_ci95_low']:+.6f}, {row['event_ci95_high']:+.6f}] | "
            f"{win['f1_win_rate']:.3f} | {win['event_win_rate']:.3f} |"
        )
    lines += [
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "provider_equal": provider_equal.to_dict("records"),
                "comparisons": comparisons.to_dict("records"),
                "wins": wins.to_dict("records"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
