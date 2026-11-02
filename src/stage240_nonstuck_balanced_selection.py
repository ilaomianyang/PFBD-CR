from __future__ import annotations

"""Stage 240: non-stuck-balanced target-domain selection audit.

The frozen Stage219/222/228/238 transfer results select the target-domain
local arm by ``valid_f1``.  Stage232 showed that the cross-provider
direction is concentrated in ``stuck``: removing ``stuck`` makes both row
recall and event coverage turn negative.

This stage asks a narrower and pre-declared question: if the target-domain
valid selection rule prioritizes non-``stuck`` event coverage instead of
overall F1, does the cross-provider advantage survive?

The stage reuses the frozen protocols, normalized tables, split labels,
attack generators, detector fitting, and the same 42-candidate grid.  It
does not modify or replace the frozen main results.  Test labels are used
only for evaluation after the valid selection has been fixed.  All
attacks are transparent controlled injections, not real attack logs.
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
PREFIX = "stage240_nonstuck_balanced_selection"

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
    FEATURES,
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


PROTOCOLS = {
    "MeteoSwiss": TABLES / "stage214_swiss_protocol_preregistration.json",
    "NOAA ISD": TABLES / "stage222_noaa_transfer_protocol_preregistration.json",
    "KNMI": TABLES / "stage224_knmi_protocol_preregistration.json",
    "ECCC": TABLES / "stage234_eccc_protocol_preregistration.json",
}
NORMALIZED = {
    "MeteoSwiss": TABLES / "stage216_swiss_normalized.parquet",
    "NOAA ISD": TABLES / "stage82_noaa_extended_normalized.parquet",
    "KNMI": TABLES / "stage226_knmi_normalized.parquet",
    "ECCC": TABLES / "stage236_eccc_normalized.parquet",
}
AUDITS = {
    "MeteoSwiss": TABLES / "stage216a_swiss_protocol_correction.json",
    "NOAA ISD": TABLES / "stage83_noaa_extended_quality_audit.json",
    "KNMI": TABLES / "stage226a_knmi_missingness_audit.json",
    "ECCC": TABLES / "stage236_eccc_normalized.json",
}
LOCAL_ARMS = {
    "MeteoSwiss": "swiss_valid_selected",
    "NOAA ISD": "noaa_valid_selected",
    "KNMI": "knmi_valid_selected",
    "ECCC": "eccc_valid_selected",
}
RECALIBRATED_ARMS = {
    "MeteoSwiss": "swiss_valid_recalibrated",
    "NOAA ISD": "noaa_valid_recalibrated",
    "KNMI": "knmi_valid_recalibrated",
    "ECCC": "eccc_valid_recalibrated",
}
SELECTION_RULES = ("valid_f1", "valid_nonstuck_event_macro")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_frame(source: str, protocol: dict) -> pd.DataFrame:
    if source == "NOAA ISD":
        return make_noaa_frame(protocol)
    frame = pd.read_parquet(NORMALIZED[source])
    frame["station_id"] = frame["station_id"].astype(str)
    audit = load_json(AUDITS[source])
    retained = set(str(value) for value in audit["retained_station_ids"])
    frame = frame[frame["station_id"].isin(retained)].copy()
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    if set(frame["station_id"]) != retained:
        raise RuntimeError(f"{source} retained station set changed")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError(f"{source} split labels are incomplete")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError(f"{source} contains duplicate station timestamps")
    return frame


def select_candidate(
    rule: str,
    valid_base: np.ndarray,
    valid_feature_z: dict[tuple[str, int], np.ndarray],
    valid_frame: pd.DataFrame,
    valid_normal: np.ndarray,
    primary_far: float,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for feature_name, window, weight in candidate_grid():
        score = (
            (1.0 - float(weight)) * valid_base
            + float(weight)
            * valid_feature_z[(str(feature_name), int(window))]
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
        key = selection_key(
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

    feature_z: dict[tuple[str, int], np.ndarray] = {}
    for feature_name in FEATURES:
        for window in WINDOWS:
            feature_z[(str(feature_name), int(window))] = robust_standardize(
                repeat_feature(base, str(feature_name), int(window)),
                train_normal,
            ).to_numpy(dtype=float)

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
        valid_feature_z: dict[tuple[str, int], np.ndarray] = {}
        test_feature_z: dict[tuple[str, int], np.ndarray] = {}
        for feature_name in FEATURES:
            for window in WINDOWS:
                key = (str(feature_name), int(window))
                valid_feature_z[key] = robust_standardize(
                    repeat_feature(
                        valid_attacked,
                        str(feature_name),
                        int(window),
                    ),
                    train_normal,
                ).to_numpy(dtype=float)
                test_feature_z[key] = robust_standardize(
                    repeat_feature(
                        test_attacked,
                        str(feature_name),
                        int(window),
                    ),
                    train_normal,
                ).to_numpy(dtype=float)

        valid_mask = valid_attacked["split"].eq("valid").to_numpy(dtype=bool)
        test_mask = test_attacked["split"].eq("test").to_numpy(dtype=bool)
        valid_frame = valid_attacked.loc[valid_mask].reset_index(drop=True)
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

        frozen = frozen_dwd_selection(selection, seed, primary_far)
        frozen_key = (str(frozen["feature"]), int(frozen["window"]))
        valid_transferred = (
            (1.0 - float(frozen["weight"])) * valid_base
            + float(frozen["weight"])
            * valid_feature_z[frozen_key][valid_mask]
        )
        test_transferred = (
            (1.0 - float(frozen["weight"])) * test_base
            + float(frozen["weight"]) * test_feature_z[frozen_key][test_mask]
        )
        recalibrated_threshold = far_threshold(
            valid_transferred,
            valid_normal,
            primary_far,
        )

        selected = {
            rule: select_candidate(
                rule,
                valid_base,
                {
                    key: values[valid_mask]
                    for key, values in valid_feature_z.items()
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
                + float(best["weight"]) * test_feature_z[key][test_mask]
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


def paired_summary(
    evaluation: pd.DataFrame,
    baseline_arm: str,
    candidate_arm: str,
) -> list[dict[str, object]]:
    baseline = evaluation[evaluation["arm"].eq(baseline_arm)]
    candidate = evaluation[evaluation["arm"].eq(candidate_arm)]
    aligned = baseline.merge(
        candidate,
        on=["source", "seed"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for metric in [
        "f1",
        "recall",
        "precision",
        "false_alarm_rate",
        "average_precision",
        "event_coverage",
        "row_recall",
    ]:
        values = (
            aligned[f"{metric}_candidate"].to_numpy(dtype=float)
            - aligned[f"{metric}_baseline"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "comparison": f"{candidate_arm}_minus_{baseline_arm}",
                "metric": metric,
                "pairs": int(len(values)),
                "mean_diff": float(np.mean(values)),
                "std_diff": float(np.std(values, ddof=0)),
                "positive_pairs": int(np.sum(values > 0)),
                "negative_pairs": int(np.sum(values < 0)),
                "zero_pairs": int(np.sum(values == 0)),
            }
        )
    return rows


def main() -> int:
    selection = pd.read_csv(
        TABLES / "stage208_dwd_persistent_repeat_protection_selection.csv"
    )
    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for source, protocol_path in PROTOCOLS.items():
        protocol = load_json(protocol_path)
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
    provider_equal_rows: list[dict[str, object]] = []
    for arm in sorted(evaluation["arm"].unique()):
        rows = evaluation_aggregate[evaluation_aggregate["arm"].eq(arm)]
        provider_equal_rows.append(
            {
                "arm": arm,
                "provider_count": int(len(rows)),
                "station_count": int(
                    station[station["arm"].eq(arm)]["station_id"].nunique()
                ),
                "f1_provider_equal": float(rows["f1_mean"].mean()),
                "event_coverage_provider_equal": float(
                    rows["event_coverage_mean"].mean()
                ),
                "row_recall_provider_equal": float(
                    rows["row_recall_mean"].mean()
                ),
            }
        )
    provider_equal = pd.DataFrame(provider_equal_rows)

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
    recalibrated_arm = {
        source: RECALIBRATED_ARMS[source] for source in PROTOCOLS
    }
    nonstuck_arm = {
        source: f"{source.lower().replace(' ', '_')}_valid_nonstuck_event_macro"
        for source in PROTOCOLS
    }
    paired_station_rows: list[dict[str, object]] = []
    for source in PROTOCOLS:
        baseline = station_metric[
            station_metric["arm"].eq(recalibrated_arm[source])
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
            paired_station_rows.append(
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
    paired_station = pd.DataFrame(paired_station_rows)
    provider_station = (
        paired_station.groupby("source", as_index=False)
        .agg(
            stations=("station_id", "nunique"),
            f1_difference=("f1_difference", "mean"),
            event_coverage_difference=("event_coverage_difference", "mean"),
        )
    )
    provider_equal_difference = {
        "f1_difference": float(provider_station["f1_difference"].mean()),
        "event_coverage_difference": float(
            provider_station["event_coverage_difference"].mean()
        ),
    }
    leave_one_provider_rows: list[dict[str, object]] = []
    for removed in sorted(provider_station["source"].unique()):
        retained = provider_station[
            ~provider_station["source"].eq(removed)
        ]
        leave_one_provider_rows.append(
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
    leave_one_provider = pd.DataFrame(leave_one_provider_rows)

    bootstrap_iterations = 10000
    bootstrap_seed = 20261003
    providers = sorted(paired_station["source"].unique())
    by_provider = {
        provider: paired_station.loc[
            paired_station["source"].eq(provider),
            ["f1_difference", "event_coverage_difference"],
        ].to_numpy(dtype=float)
        for provider in providers
    }
    rng = np.random.default_rng(bootstrap_seed)
    provider_draws = rng.integers(
        0,
        len(providers),
        size=(bootstrap_iterations, len(providers)),
    )
    f1_boot = np.empty(bootstrap_iterations, dtype=float)
    event_boot = np.empty(bootstrap_iterations, dtype=float)
    for index in range(bootstrap_iterations):
        f1_means: list[float] = []
        event_means: list[float] = []
        for provider_index in provider_draws[index]:
            values = by_provider[providers[int(provider_index)]]
            draws = rng.integers(0, len(values), size=len(values))
            sampled = values[draws]
            f1_means.append(float(sampled[:, 0].mean()))
            event_means.append(float(sampled[:, 1].mean()))
        f1_boot[index] = float(np.mean(f1_means))
        event_boot[index] = float(np.mean(event_means))
    provider_values_f1 = provider_station["f1_difference"].to_numpy(
        dtype=float
    )
    provider_values_event = provider_station[
        "event_coverage_difference"
    ].to_numpy(dtype=float)
    hierarchical_bootstrap = {
        "iterations": bootstrap_iterations,
        "provider_count": int(len(providers)),
        "station_rows": int(len(paired_station)),
        "f1_difference": float(np.mean(provider_values_f1)),
        "f1_ci95_low": float(np.quantile(f1_boot, 0.025)),
        "f1_ci95_high": float(np.quantile(f1_boot, 0.975)),
        "event_coverage_difference": float(
            np.mean(provider_values_event)
        ),
        "event_ci95_low": float(np.quantile(event_boot, 0.025)),
        "event_ci95_high": float(np.quantile(event_boot, 0.975)),
        "provider_sign_flip_f1_p": float(
            exact_sign_flip_two_sided(provider_values_f1)
        ),
        "provider_sign_flip_event_p": float(
            exact_sign_flip_two_sided(provider_values_event)
        ),
    }

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
    nonstuck_type_rows: list[dict[str, object]] = []
    for source in PROTOCOLS:
        baseline = type_aggregate[
            type_aggregate["source"].eq(source)
            & type_aggregate["arm"].eq(recalibrated_arm[source])
        ].set_index("attack_type")
        candidate = type_aggregate[
            type_aggregate["source"].eq(source)
            & type_aggregate["arm"].eq(nonstuck_arm[source])
        ].set_index("attack_type")
        for attack_type in ATTACK_TYPES:
            if attack_type == "stuck":
                continue
            nonstuck_type_rows.append(
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
    nonstuck_type = pd.DataFrame(nonstuck_type_rows)
    nonstuck_type_equal = (
        nonstuck_type.groupby("attack_type", as_index=False)
        .agg(
            provider_count=("source", "nunique"),
            row_recall_difference=("row_recall_difference", "mean"),
            event_coverage_difference=(
                "event_coverage_difference",
                "mean",
            ),
        )
    )

    selection_counts = (
        selections.groupby(
            ["source", "rule", "feature", "window", "weight"],
            as_index=False,
        )
        .agg(seeds=("seed", "nunique"))
        .sort_values(["source", "rule", "seeds"], ascending=[True, True, False])
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段240只审计非stuck平衡选择，不替换阶段219/222/228/238的"
            "冻结主结果。对四个目标域复用冻结协议、标准化数据、"
            "train/valid/test split、"
            "受控攻击生成器、检测器拟合和42个候选网格；每个seed在目标域"
            "valid上分别按valid_f1和valid_nonstuck_event_macro选择，"
            "test只评价。非stuck规则优先最大化valid非stuck事件覆盖宏平均，"
            "随后依次为valid F1、总事件宏平均、stuck事件覆盖、权重/窗口/"
            "特征稳定排序。不使用test标签做任何选择。所有攻击都是透明"
            "受控注入，不是真实攻击日志。"
        ),
        "selection_rules": list(SELECTION_RULES),
        "provider_count": int(len(PROTOCOLS)),
        "station_count": int(
            station[["source", "station_id"]].drop_duplicates().shape[0]
        ),
        "candidate_count": int(len(candidate_grid())),
        "evaluation_rows": int(len(evaluation)),
        "station_rows": int(len(station)),
        "type_rows": int(len(type_detail)),
        "selection_rows": int(len(selections)),
        "provider_equal_difference": provider_equal_difference,
        "hierarchical_bootstrap": hierarchical_bootstrap,
        "provider_equal_rows": clean_records(
            provider_equal.to_dict("records")
        ),
        "provider_station_rows": clean_records(
            provider_station.to_dict("records")
        ),
        "leave_one_provider_out_rows": clean_records(
            leave_one_provider.to_dict("records")
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
            "这是非stuck平衡的补充审计，不替换冻结主结果。",
            "只有四个迁移目标，机构级精确sign-flip检验最小双侧p=0.125。",
            "非stuck优先规则可能降低总体F1或stuck覆盖。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
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
    provider_station.to_csv(
        TABLES / f"{PREFIX}_provider_station.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one_provider.to_csv(
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
        "# 阶段240 非stuck平衡目标域选择审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 机构数：`{payload['provider_count']}`；站点数：`{payload['station_count']}`；候选数：`{payload['candidate_count']}`",
        "",
        "## 机构等权差值（非stuck规则 − valid重校准）",
        "",
        "| 指标 | 差值 | 95% CI |",
        "|---|---:|---:|",
        f"| F1 | {hierarchical_bootstrap['f1_difference']:+.6f} | [{hierarchical_bootstrap['f1_ci95_low']:+.6f}, {hierarchical_bootstrap['f1_ci95_high']:+.6f}] |",
        f"| 事件覆盖 | {hierarchical_bootstrap['event_coverage_difference']:+.6f} | [{hierarchical_bootstrap['event_ci95_low']:+.6f}, {hierarchical_bootstrap['event_ci95_high']:+.6f}] |",
        "",
        "## 留一机构",
        "",
        "| 移除机构 | F1差值 | 事件覆盖差值 |",
        "|---|---:|---:|",
    ]
    for row in leave_one_provider.to_dict("records"):
        lines.append(
            f"| {row['removed_source']} | {row['f1_difference']:+.6f} | "
            f"{row['event_coverage_difference']:+.6f} |"
        )
    lines += [
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
        "- 本阶段是补充审计，不替换冻结主结果。",
        "- 只有四个迁移目标，机构级精确检验功效有限。",
        "- 非stuck优先可能降低总体F1或stuck事件覆盖。",
        "- 所有攻击都是透明受控注入，不是真实攻击日志。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "provider_equal_difference": provider_equal_difference,
                "hierarchical_bootstrap": hierarchical_bootstrap,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
