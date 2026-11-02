from __future__ import annotations

"""Stage 244: cross-provider non-stuck candidate feature evaluation.

Stage 232 showed that the frozen cross-provider advantage is concentrated in
``stuck``.  Stage 240 re-used the repeat-only candidate grid and failed to
find a non-stuck advantage.  This stage therefore asks the next pre-declared
question:

Can a richer but still causal feature family improve non-stuck event coverage
across the four external transfer targets without using test labels?

The stage reuses the frozen Stage 219/222/228/238 protocols, normalized
tables, split boundaries, transparent controlled attack generator, detector
fitting, and target-domain valid selection rule.  Candidate features are
causal within station.  All scaling statistics are fitted on train-normal
rows only.  Threshold calibration and candidate selection use valid only.
Test is evaluation-only.

The candidate set deliberately includes the Stage 208 repeat features as a
fallback so valid can reject the broader feature family.  The experiment
does not replace the frozen main result.  All attacks are transparent
controlled injections, not real attack logs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage244_cross_provider_nonstuck_candidate_features"

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
    WINDOWS,
    WEIGHTS,
    candidate_grid,
    choose_split_attack_blocks,
    far_threshold,
    macro_event_coverage,
    repeat_feature,
    selection_key,
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
    evaluate_arm,
    frozen_dwd_selection,
)
from stage222_noaa_dwd_protection_transfer import (  # noqa: E402
    make_noaa_frame,
)
from stage240_nonstuck_balanced_selection import (  # noqa: E402
    AUDITS,
    NORMALIZED,
    PROTOCOLS,
    RECALIBRATED_ARMS,
    SELECTION_RULES,
    make_frame,
)


NEW_FEATURES = [
    "level_shift",
    "slope_6",
    "trend_consistency",
    "noise_ratio",
    "diff_zscore",
    "second_diff_zscore",
    "drift_acceleration",
]
ALL_FEATURES = [
    "repeat_ratio",
    "repeat_run",
    *NEW_FEATURES,
]


def _robust_scale(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return max(1.4826 * mad, 0.25)


def nonstuck_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return causal per-station non-stuck candidate features."""
    work = frame[
        ["station_id", "timestamp_utc", "temperature_c"]
    ].copy()
    work["_value"] = pd.to_numeric(
        work["temperature_c"],
        errors="coerce",
    )
    work["_timestamp"] = pd.to_datetime(
        work["timestamp_utc"],
        utc=True,
        errors="coerce",
    )
    work = work.sort_values(
        ["station_id", "_timestamp"],
        kind="stable",
    )
    out = pd.DataFrame(
        index=frame.index,
        columns=NEW_FEATURES,
        dtype=float,
    )
    for _, group in work.groupby("station_id", sort=False):
        values = group["_value"]
        difference = values.diff()
        abs_difference = difference.abs()
        diff_scale = (
            abs_difference.rolling(24, min_periods=6)
            .median()
            .clip(lower=0.1)
        )
        rolling_median_24 = values.rolling(24, min_periods=6).median()
        residual = values - rolling_median_24
        local_residual_std = residual.rolling(6, min_periods=3).std()
        long_residual_std = residual.rolling(24, min_periods=8).std()
        second_difference = (
            values - 2.0 * values.shift(1) + values.shift(2)
        )
        drift_acceleration = (
            values - 2.0 * values.shift(6) + values.shift(12)
        ).abs() / 6.0
        sign_change = np.sign(difference)
        sign_consistency = (
            sign_change.rolling(6, min_periods=3).mean().abs()
        )
        part = pd.DataFrame(index=group.index)
        part["level_shift"] = (
            values
            - values.shift(1).rolling(8, min_periods=4).median()
        ).abs() / diff_scale
        part["slope_6"] = (
            (values - values.shift(6)).abs() / 6.0 / diff_scale
        )
        part["trend_consistency"] = sign_consistency * (
            (values - values.shift(6)).abs() / 6.0 / diff_scale
        )
        part["noise_ratio"] = (
            local_residual_std / long_residual_std.replace(0.0, np.nan)
        )
        part["diff_zscore"] = (
            abs_difference.rolling(4, min_periods=2).max() / diff_scale
        )
        part["second_diff_zscore"] = (
            second_difference.abs() / diff_scale
        )
        part["drift_acceleration"] = drift_acceleration / diff_scale
        out.loc[group.index, NEW_FEATURES] = part[
            NEW_FEATURES
        ].to_numpy(dtype=float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def standardize_nonnegative(
    values: pd.Series,
    train_normal: np.ndarray,
) -> np.ndarray:
    reference = values.loc[train_normal]
    center = float(reference.median())
    scale = _robust_scale(reference)
    return np.maximum(
        (values.to_numpy(dtype=float) - center) / scale,
        0.0,
    )


def candidate_features(
    base: pd.DataFrame,
    train_normal: np.ndarray,
) -> dict[tuple[str, int], np.ndarray]:
    output: dict[tuple[str, int], np.ndarray] = {}
    for feature_name in ("repeat_ratio", "repeat_run"):
        for window in WINDOWS:
            output[(feature_name, int(window))] = robust_standardize(
                repeat_feature(base, feature_name, int(window)),
                train_normal,
            ).to_numpy(dtype=float)
    nonstuck = nonstuck_features(base)
    for feature_name in NEW_FEATURES:
        series = nonstuck[feature_name]
        output[(feature_name, 1)] = standardize_nonnegative(
            series,
            train_normal,
        )
    return output


def candidate_grid_stage244() -> list[tuple[str, int, float]]:
    return [
        (feature_name, int(window), float(weight))
        for feature_name in ALL_FEATURES
        for window in (
            WINDOWS if feature_name in {"repeat_ratio", "repeat_run"} else [1]
        )
        for weight in WEIGHTS
    ]


def selection_key_stage244(
    rule: str,
    valid_metrics: dict[str, float],
    valid_event_macro: float,
    valid_stuck_event: float,
    valid_nonstuck_event: float,
    feature_name: str,
    window: int,
    weight: float,
) -> tuple[float, ...]:
    if rule == "valid_f1":
        primary = float(valid_metrics["f1"])
    elif rule == "valid_event_macro":
        primary = float(valid_event_macro)
    elif rule == "valid_nonstuck_event_macro":
        primary = float(valid_nonstuck_event)
    else:
        raise ValueError(f"unknown selection rule: {rule}")
    return (
        primary,
        float(valid_metrics["f1"]),
        float(valid_stuck_event),
        float(valid_event_macro),
        -float(weight),
        -float(ALL_FEATURES.index(feature_name)),
        -int(window),
    )


def select_candidate_stage244(
    rule: str,
    valid_base: np.ndarray,
    valid_features: dict[tuple[str, int], np.ndarray],
    valid_frame: pd.DataFrame,
    valid_normal: np.ndarray,
    primary_far: float,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for feature_name, window, weight in candidate_grid_stage244():
        score = (
            (1.0 - float(weight)) * valid_base
            + float(weight)
            * valid_features[(str(feature_name), int(window))]
        )
        threshold = far_threshold(score, valid_normal, primary_far)
        predicted = score >= threshold
        valid_metrics = metrics(
            valid_frame["is_attack"].astype(bool).to_numpy(),
            predicted,
            score,
        )
        valid_types = type_metrics(valid_frame, predicted, score)
        event_macro = macro_event_coverage(valid_types, ATTACK_TYPES)
        stuck_event = float(valid_types["stuck"]["event_coverage"])
        nonstuck_event = macro_event_coverage(
            valid_types,
            [name for name in ATTACK_TYPES if name != "stuck"],
        )
        key = selection_key_stage244(
            rule,
            valid_metrics,
            event_macro,
            stuck_event,
            nonstuck_event,
            str(feature_name),
            int(window),
            float(weight),
        )
        if best is None or key > best["selection_key"]:
            best = {
                "selection_key": key,
                "feature": str(feature_name),
                "window": int(window),
                "weight": float(weight),
                "threshold": float(threshold),
                "valid_f1": float(valid_metrics["f1"]),
                "valid_nonstuck_event_macro": float(nonstuck_event),
                "valid_event_macro": float(event_macro),
                "valid_stuck_event": float(stuck_event),
            }
    if best is None:
        raise RuntimeError(f"candidate selection failed for rule={rule}")
    return best


def evaluate_source(
    source: str,
    protocol: dict,
    selection: pd.DataFrame,
    frozen_selection_by_seed: dict[int, dict[str, object]] | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    frame = make_frame(source, protocol)
    base = detector_frame(frame)
    detectors = fit_detectors(base, protocol)
    train_normal = base["split"].eq("train") & base["observation_present"]
    components = score_components(base, detectors)
    fused, _ = fuse_scores(base, components)
    base_z = robust_standardize(fused, train_normal).to_numpy(dtype=float)
    features = candidate_features(base, train_normal)

    primary_far = float(protocol["primary_far"])
    station_ids = sorted(frame["station_id"].astype(str).unique().tolist())
    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for seed in protocol["seeds"]:
        seed = int(seed)
        valid_blocks = choose_split_attack_blocks(base, protocol, seed, "valid")
        test_blocks = choose_split_attack_blocks(base, protocol, seed, "test")
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

        valid_features = candidate_features(valid_attacked, train_normal)
        test_features = candidate_features(test_attacked, train_normal)

        valid_mask = valid_attacked["split"].eq("valid").to_numpy(dtype=bool)
        test_mask = test_attacked["split"].eq("test").to_numpy(dtype=bool)
        valid_frame = valid_attacked.loc[valid_mask].reset_index(drop=True)
        test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
        if not valid_frame["is_attack"].any():
            raise RuntimeError(f"{source} valid attack injection is empty")
        if not test_frame["is_attack"].any():
            raise RuntimeError(f"{source} test attack injection is empty")
        valid_normal = (
            valid_frame["observation_present"] & ~valid_frame["is_attack"]
        ).to_numpy(dtype=bool)
        valid_base = valid_base_z[valid_mask]
        test_base = test_base_z[test_mask]

        if frozen_selection_by_seed is None:
            frozen = frozen_dwd_selection(selection, seed, primary_far)
        else:
            if seed not in frozen_selection_by_seed:
                raise RuntimeError(
                    f"missing frozen DWD selection for replication seed={seed}"
                )
            frozen = frozen_selection_by_seed[seed]
        frozen_key = (str(frozen["feature"]), int(frozen["window"]))
        valid_transferred = (
            (1.0 - float(frozen["weight"])) * valid_base
            + float(frozen["weight"])
            * valid_features[frozen_key][valid_mask]
        )
        test_transferred = (
            (1.0 - float(frozen["weight"])) * test_base
            + float(frozen["weight"]) * test_features[frozen_key][test_mask]
        )
        recalibrated_threshold = far_threshold(
            valid_transferred,
            valid_normal,
            primary_far,
        )

        selected = {
            rule: select_candidate_stage244(
                rule,
                valid_base,
                {
                    key: values[valid_mask]
                    for key, values in valid_features.items()
                },
                valid_frame,
                valid_normal,
                primary_far,
            )
            for rule in SELECTION_RULES
        }
        arms: dict[str, dict[str, object]] = {
            RECALIBRATED_ARMS[source]: {
                "feature": frozen["feature"],
                "window": frozen["window"],
                "weight": frozen["weight"],
                "threshold": recalibrated_threshold,
                "score": test_transferred,
                "selection_rule": "dwd_frozen_recalibrated",
            }
        }
        for rule, best in selected.items():
            key = (str(best["feature"]), int(best["window"]))
            score = (
                (1.0 - float(best["weight"])) * test_base
                + float(best["weight"]) * test_features[key][test_mask]
            )
            arms[f"{source.lower().replace(' ', '_')}_{rule}"] = {
                "feature": best["feature"],
                "window": best["window"],
                "weight": best["weight"],
                "threshold": best["threshold"],
                "score": score,
                "selection_rule": rule,
            }
            selection_rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "rule": rule,
                    "feature": best["feature"],
                    "window": best["window"],
                    "weight": best["weight"],
                    "threshold": best["threshold"],
                    "valid_f1": best["valid_f1"],
                    "valid_nonstuck_event_macro": best[
                        "valid_nonstuck_event_macro"
                    ],
                    "valid_event_macro": best["valid_event_macro"],
                    "valid_stuck_event": best["valid_stuck_event"],
                }
            )

        for arm_name, arm in arms.items():
            score = np.asarray(arm["score"], dtype=float)
            predicted = (
                test_frame["observation_present"].to_numpy(dtype=bool)
                & (score >= float(arm["threshold"]))
            )
            arm_metrics, arm_types = evaluate_arm(
                test_frame,
                predicted,
                score,
            )
            evaluation_rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "arm": arm_name,
                    "selection_rule": arm["selection_rule"],
                    "feature": arm["feature"],
                    "window": arm["window"],
                    "weight": arm["weight"],
                    "threshold": arm["threshold"],
                    **arm_metrics,
                }
            )
            for attack_type, values in arm_types.items():
                type_rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "arm": arm_name,
                        "attack_type": attack_type,
                        **values,
                    }
                )
            y_test = test_frame["is_attack"].astype(bool).to_numpy()
            station_values = test_frame["station_id"].astype(str).to_numpy()
            event_frame = test_frame.loc[y_test].copy()
            event_frame["predicted"] = predicted[y_test]
            grouped = (
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
                station_rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "arm": arm_name,
                        "selection_rule": arm["selection_rule"],
                        "station_id": station_id,
                        "rows": int(mask.sum()),
                        "attack_rows": int(y_station.sum()),
                        "normal_rows": int((~y_station).sum()),
                        "tp": int((y_station & pred_station).sum()),
                        "fp": int((~y_station & pred_station).sum()),
                        "fn": int((y_station & ~pred_station).sum()),
                        "tn": int((~y_station & ~pred_station).sum()),
                        "events": int(
                            len(grouped[grouped["station_id"].eq(station_id)])
                        ),
                        "detected_events": int(
                            grouped[
                                grouped["station_id"].eq(station_id)
                            ]["predicted"].sum()
                        ),
                    }
                )
    return evaluation_rows, station_rows, type_rows, selection_rows


def station_metric_table(station: pd.DataFrame) -> pd.DataFrame:
    station_metric = (
        station.groupby(
            ["source", "seed", "arm", "selection_rule", "station_id"],
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
    station_metric["f1"] = np.divide(
        2.0 * station_metric["tp"],
        2.0 * station_metric["tp"]
        + station_metric["fp"]
        + station_metric["fn"],
        out=np.zeros(len(station_metric), dtype=float),
        where=(
            2.0 * station_metric["tp"]
            + station_metric["fp"]
            + station_metric["fn"]
        ).to_numpy()
        > 0,
    )
    station_metric["event_coverage"] = np.divide(
        station_metric["detected_events"],
        station_metric["events"],
        out=np.zeros(len(station_metric), dtype=float),
        where=station_metric["events"].to_numpy() > 0,
    )
    return station_metric


def paired_station_table(
    station_metric: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nonstuck_arm = {
        source: f"{source.lower().replace(' ', '_')}_valid_nonstuck_event_macro"
        for source in PROTOCOLS
    }
    rows: list[dict[str, object]] = []
    for source in PROTOCOLS:
        baseline = station_metric[
            station_metric["arm"].eq(RECALIBRATED_ARMS[source])
        ]
        candidate = station_metric[
            station_metric["arm"].eq(nonstuck_arm[source])
        ]
        aligned = baseline.merge(
            candidate,
            on=["source", "seed", "station_id"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        aligned["f1_difference"] = (
            aligned["f1_candidate"] - aligned["f1_baseline"]
        )
        aligned["event_difference"] = (
            aligned["event_coverage_candidate"]
            - aligned["event_coverage_baseline"]
        )
        for row in aligned.to_dict("records"):
            rows.append(
                {
                    "source": source,
                    "seed": int(row["seed"]),
                    "station_id": str(row["station_id"]),
                    "f1_difference": float(row["f1_difference"]),
                    "event_coverage_difference": float(
                        row["event_difference"]
                    ),
                }
            )
    paired = pd.DataFrame(rows)
    provider = (
        paired.groupby("source", as_index=False)
        .agg(
            stations=("station_id", "nunique"),
            f1_difference=("f1_difference", "mean"),
            event_coverage_difference=("event_coverage_difference", "mean"),
        )
    )
    leave_one: list[dict[str, object]] = []
    for removed in sorted(provider["source"].unique()):
        retained = provider[~provider["source"].eq(removed)]
        leave_one.append(
            {
                "removed_source": removed,
                "retained_sources": ",".join(
                    sorted(retained["source"].unique())
                ),
                "f1_difference": float(retained["f1_difference"].mean()),
                "event_coverage_difference": float(
                    retained["event_coverage_difference"].mean()
                ),
                "f1_direction_positive": bool(
                    float(retained["f1_difference"].mean()) > 0.0
                ),
                "event_direction_positive": bool(
                    float(retained["event_coverage_difference"].mean()) > 0.0
                ),
            }
        )
    return paired, provider, pd.DataFrame(leave_one)


def hierarchical_bootstrap(
    paired: pd.DataFrame,
    iterations: int = 10000,
    seed: int = 20261004,
) -> dict[str, float]:
    providers = sorted(paired["source"].unique())
    by_provider = {
        provider: paired.loc[
            paired["source"].eq(provider),
            ["f1_difference", "event_coverage_difference"],
        ].to_numpy(dtype=float)
        for provider in providers
    }
    rng = np.random.default_rng(seed)
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
        paired.groupby("source", as_index=False)["f1_difference"].mean()[
            "f1_difference"
        ].to_numpy(dtype=float)
    )
    provider_event = (
        paired.groupby("source", as_index=False)[
            "event_coverage_difference"
        ].mean()["event_coverage_difference"].to_numpy(dtype=float)
    )
    return {
        "iterations": int(iterations),
        "provider_count": int(len(providers)),
        "station_rows": int(len(paired)),
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
    }


def main() -> int:
    selection = pd.read_csv(
        TABLES / "stage208_dwd_persistent_repeat_protection_selection.csv"
    )
    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for source, protocol_path in PROTOCOLS.items():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        source_eval, source_station, source_type, source_selection = (
            evaluate_source(source, protocol, selection)
        )
        evaluation_rows.extend(source_eval)
        station_rows.extend(source_station)
        type_rows.extend(source_type)
        selection_rows.extend(source_selection)
        print(
            json.dumps(
                {
                    "source": source,
                    "status": "completed",
                    "rows": len(source_eval),
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
        evaluation.groupby(["source", "arm", "selection_rule"], as_index=False)
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
        .sort_values(["source", "arm"])
    )
    station_metric = station_metric_table(station)
    paired, provider, leave_one = paired_station_table(station_metric)
    bootstrap = hierarchical_bootstrap(paired)

    type_aggregate = (
        type_detail.groupby(["source", "arm", "attack_type"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            attack_rows=("attack_rows", "sum"),
            detected_rows=("detected_rows", "sum"),
            row_recall=("row_recall", "mean"),
            event_coverage=("event_coverage", "mean"),
        )
        .sort_values(["source", "arm", "attack_type"])
    )
    nonstuck_rows: list[dict[str, object]] = []
    nonstuck_arm = {
        source: f"{source.lower().replace(' ', '_')}_valid_nonstuck_event_macro"
        for source in PROTOCOLS
    }
    for source in PROTOCOLS:
        baseline = type_aggregate[
            type_aggregate["source"].eq(source)
            & type_aggregate["arm"].eq(RECALIBRATED_ARMS[source])
        ].set_index("attack_type")
        candidate = type_aggregate[
            type_aggregate["source"].eq(source)
            & type_aggregate["arm"].eq(nonstuck_arm[source])
        ].set_index("attack_type")
        for attack_type in ATTACK_TYPES:
            if attack_type == "stuck":
                continue
            nonstuck_rows.append(
                {
                    "source": source,
                    "attack_type": attack_type,
                    "row_recall_difference": float(
                        candidate.loc[attack_type, "row_recall"]
                        - baseline.loc[attack_type, "row_recall"]
                    ),
                    "event_coverage_difference": float(
                        candidate.loc[attack_type, "event_coverage"]
                        - baseline.loc[attack_type, "event_coverage"]
                    ),
                }
            )
    nonstuck_type = pd.DataFrame(nonstuck_rows)
    nonstuck_type_equal = (
        nonstuck_type.groupby("attack_type", as_index=False)
        .agg(
            provider_count=("source", "nunique"),
            row_recall_difference=("row_recall_difference", "mean"),
            event_coverage_difference=("event_coverage_difference", "mean"),
        )
    )
    selection_counts = (
        selections.groupby(
            ["source", "rule", "feature", "window", "weight"],
            as_index=False,
        )
        .agg(seeds=("seed", "nunique"))
        .sort_values(
            ["source", "rule", "seeds"],
            ascending=[True, True, False],
        )
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段244在四个外部目标域上扩展非stuck候选特征，不替换"
            "阶段219/222/228/238冻结主结果。复用冻结协议、标准化数据、"
            "train/valid/test split、透明受控攻击生成器和检测器拟合；"
            "新增特征均为站点内因果量，标准化统计仅用train正常行，"
            "候选权重和阈值只用valid选择，test只评价。候选集合包含"
            "阶段208的repeat特征作为可拒绝回退，因此valid可以完全不"
            "选择新特征。所有攻击都是透明受控注入，不是真实攻击日志。"
        ),
        "selection_rules": list(SELECTION_RULES),
        "provider_count": int(len(PROTOCOLS)),
        "station_count": int(
            station[["source", "station_id"]].drop_duplicates().shape[0]
        ),
        "candidate_count": int(len(candidate_grid_stage244())),
        "new_feature_count": int(len(NEW_FEATURES)),
        "evaluation_rows": int(len(evaluation)),
        "station_rows": int(len(station)),
        "type_rows": int(len(type_detail)),
        "selection_rows": int(len(selections)),
        "hierarchical_bootstrap": bootstrap,
        "provider_station_rows": clean_records(provider.to_dict("records")),
        "leave_one_provider_out_rows": clean_records(
            leave_one.to_dict("records")
        ),
        "nonstuck_type_equal_rows": clean_records(
            nonstuck_type_equal.to_dict("records")
        ),
        "selection_count_rows": clean_records(
            selection_counts.to_dict("records")
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
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "limitations": [
            "这是非stuck候选扩展，不替换冻结主结果。",
            "只有四个外部目标机构；候选级有效不等于跨机构普适。",
            "攻击全部为透明受控注入，不代表真实攻击日志。",
            "如果valid拒绝新特征，结果应报告为负结果而不是删除。",
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
    provider.to_csv(
        TABLES / f"{PREFIX}_provider_station.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one.to_csv(
        TABLES / f"{PREFIX}_leave_one_provider_out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nonstuck_type.to_csv(
        TABLES / f"{PREFIX}_nonstuck_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nonstuck_type_equal.to_csv(
        TABLES / f"{PREFIX}_nonstuck_type_equal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection_counts.to_csv(
        TABLES / f"{PREFIX}_selection_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段244 跨机构非stuck候选特征评估",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 机构数：`{payload['provider_count']}`；站点数：`{payload['station_count']}`；候选数：`{payload['candidate_count']}`",
        "",
        "## 机构等权差值（非stuck规则 − valid重校准）",
        "",
        "| 指标 | 差值 | 95% CI |",
        "|---|---:|---:|",
        f"| F1 | {bootstrap['f1_difference']:+.6f} | [{bootstrap['f1_ci95_low']:+.6f}, {bootstrap['f1_ci95_high']:+.6f}] |",
        f"| 事件覆盖 | {bootstrap['event_coverage_difference']:+.6f} | [{bootstrap['event_ci95_low']:+.6f}, {bootstrap['event_ci95_high']:+.6f}] |",
        "",
        "## 非stuck攻击类型",
        "",
        "| 攻击类型 | 行召回差值 | 事件覆盖差值 |",
        "|---|---:|---:|",
    ]
    for row in nonstuck_type_equal.to_dict("records"):
        lines.append(
            f"| {row['attack_type']} | {row['row_recall_difference']:+.6f} | "
            f"{row['event_coverage_difference']:+.6f} |"
        )
    lines += [
        "",
        "## 限制",
        "",
        "- 本阶段是非stuck候选扩展，不替换冻结主结果。",
        "- 只有四个外部目标机构。",
        "- 所有攻击均为透明受控注入，不是真实攻击日志。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "hierarchical_bootstrap": bootstrap,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
