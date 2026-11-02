from __future__ import annotations

"""Stage 242: preregistered multisource transfer matrix.

The frozen Stage219/222/228/238 experiments used DWD as the sole source
provider.  This stage asks whether that design choice created a
DWD-specific result.  It reuses the five frozen provider protocols and
the five source-domain ``valid_f1`` selections:

* DWD from Stage208;
* MeteoSwiss, NOAA ISD, KNMI, and ECCC from Stage240.

For every ordered source-target pair, the source feature, window, and
weight are transferred unchanged.  Only the threshold is recalibrated
on target-domain valid-normal rows.  The target test split is used only
for evaluation.  Target-domain local selection is reconstructed from the
same frozen Stage240 candidate grid as a reference arm.

This is a post-hoc matrix over already frozen selections, not a new
test-time selection experiment.  All attacks are transparent controlled
injections, not real attack logs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage242_multisource_transfer_matrix"
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20261002

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
from stage219_swiss_dwd_protection_transfer import (  # noqa: E402
    clean_records,
    frozen_dwd_selection,
)
from stage222_noaa_dwd_protection_transfer import (  # noqa: E402
    make_noaa_frame,
)
from stage240_nonstuck_balanced_selection import (  # noqa: E402
    PROTOCOLS,
    NORMALIZED,
    make_frame,
    select_candidate,
)


PROVIDERS = ["MeteoSwiss", "NOAA ISD", "KNMI", "ECCC", "DWD"]
DWD_SELECTION = (
    TABLES / "stage208_dwd_persistent_repeat_protection_selection.csv"
)
DWD_NORMALIZED = TABLES / "stage206_dwd_normalized.parquet"
STAGE240_SELECTION = (
    TABLES / "stage240_nonstuck_balanced_selection_selection.csv"
)
STAGE241 = TABLES / "stage241_transfer_heterogeneity_audit.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_selection(
    source: str,
    seed: int,
    primary_far: float,
) -> dict[str, object]:
    if source == "DWD":
        selection = pd.read_csv(DWD_SELECTION)
        return frozen_dwd_selection(selection, seed, primary_far)
    selection = pd.read_csv(STAGE240_SELECTION)
    rows = selection[
        selection["source"].eq(source)
        & selection["seed"].eq(int(seed))
        & selection["rule"].eq("valid_f1")
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one frozen source selection for {source} seed={seed}"
        )
    row = rows.iloc[0]
    return {
        "feature": str(row["feature"]),
        "window": int(row["window"]),
        "weight": float(row["weight"]),
        "threshold": float(row["threshold"]),
    }


def target_data(source: str) -> tuple[pd.DataFrame, dict]:
    if source == "DWD":
        protocol = load_json(TABLES / "stage204_dwd_protocol_preregistration.json")
        frame = pd.read_parquet(DWD_NORMALIZED)
        frame["station_id"] = frame["station_id"].astype(str)
        frame = frame.sort_values(
            ["station_id", "timestamp_utc"],
            kind="stable",
        ).reset_index(drop=True)
        if set(frame["station_id"]) != {
            str(value) for value in protocol["station_ids"]
        }:
            raise RuntimeError("DWD station set changed")
        if set(frame["split"]) != {"train", "valid", "test"}:
            raise RuntimeError("DWD split labels are incomplete")
        if frame.duplicated(["station_id", "timestamp_utc"]).any():
            raise RuntimeError("DWD contains duplicate station timestamps")
        return frame, protocol
    protocol = load_json(PROTOCOLS[source])
    frame = make_frame(source, protocol)
    return frame, protocol


def station_f1(frame: pd.DataFrame) -> pd.Series:
    denominator = (
        2.0 * frame["tp"] + frame["fp"] + frame["fn"]
    ).to_numpy(dtype=float)
    values = np.divide(
        2.0 * frame["tp"].to_numpy(dtype=float),
        denominator,
        out=np.zeros(len(frame), dtype=float),
        where=denominator > 0.0,
    )
    return pd.Series(values, index=frame.index, dtype=float)


def evaluate_target(
    source: str,
    protocol: dict,
    frame: pd.DataFrame,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
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
    transfer_rows: list[dict[str, object]] = []

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
        valid_normal = (
            valid_frame["observation_present"]
            & ~valid_frame["is_attack"]
        ).to_numpy(dtype=bool)
        valid_base = valid_base_z[valid_mask]
        test_base = test_base_z[test_mask]
        valid_feature_mask = {
            key: values[valid_mask]
            for key, values in valid_feature_z.items()
        }
        test_feature_mask = {
            key: values[test_mask]
            for key, values in test_feature_z.items()
        }

        selected = select_candidate(
            "valid_f1",
            valid_base,
            valid_feature_mask,
            valid_frame,
            valid_normal,
            primary_far,
        )
        local_key = (
            str(selected["feature"]),
            int(selected["window"]),
        )
        local_score = (
            (1.0 - float(selected["weight"])) * test_base
            + float(selected["weight"]) * test_feature_mask[local_key]
        )
        transferred_by_source: dict[str, np.ndarray] = {}
        for candidate_source in PROVIDERS:
            if candidate_source == source:
                continue
            source_selected = source_selection(
                candidate_source,
                seed,
                primary_far,
            )
            key = (
                str(source_selected["feature"]),
                int(source_selected["window"]),
            )
            valid_transferred = (
                (1.0 - float(source_selected["weight"])) * valid_base
                + float(source_selected["weight"])
                * valid_feature_mask[key]
            )
            threshold = far_threshold(
                valid_transferred,
                valid_normal,
                primary_far,
            )
            test_score = (
                (1.0 - float(source_selected["weight"])) * test_base
                + float(source_selected["weight"]) * test_feature_mask[key]
            )
            transferred_by_source[candidate_source] = test_score
            transfer_rows.append(
                {
                    "target": source,
                    "source": candidate_source,
                    "seed": seed,
                    "feature": source_selected["feature"],
                    "window": int(source_selected["window"]),
                    "weight": float(source_selected["weight"]),
                    "recalibrated_threshold": float(threshold),
                    "target_far": primary_far,
                }
            )

        arms: dict[str, dict[str, object]] = {
            "target_valid_selected": {
                "score": local_score,
                "threshold": float(selected["threshold"]),
            }
        }
        for candidate_source, score in transferred_by_source.items():
            key = f"{candidate_source.lower().replace(' ', '_')}_recalibrated"
            row = next(
                row
                for row in transfer_rows
                if row["target"] == source
                and row["source"] == candidate_source
                and row["seed"] == seed
            )
            arms[key] = {
                "score": score,
                "threshold": float(row["recalibrated_threshold"]),
            }

        for arm_name, arm in arms.items():
            score = np.asarray(arm["score"], dtype=float)
            predicted = (
                test_frame["observation_present"].to_numpy(dtype=bool)
                & (score >= float(arm["threshold"]))
            )
            y_test = test_frame["is_attack"].astype(bool).to_numpy()
            arm_metrics = metrics(y_test, predicted, score)
            evaluation_rows.append(
                {
                    "target": source,
                    "seed": seed,
                    "arm": arm_name,
                    **arm_metrics,
                }
            )
            station_values = test_frame["station_id"].astype(str).to_numpy()
            for station_id in station_ids:
                mask = station_values == station_id
                y_station = y_test[mask]
                pred_station = predicted[mask]
                tp = int((y_station & pred_station).sum())
                fp = int((~y_station & pred_station).sum())
                fn = int((y_station & ~pred_station).sum())
                tn = int((~y_station & ~pred_station).sum())
                station_rows.append(
                    {
                        "target": source,
                        "seed": seed,
                        "arm": arm_name,
                        "station_id": station_id,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                    }
                )
    return evaluation_rows, station_rows, transfer_rows


def load_local_reference_rows() -> dict[str, pd.DataFrame]:
    """Load frozen upstream local-selection evaluations."""

    stage240 = pd.read_csv(
        TABLES / "stage240_nonstuck_balanced_selection_evaluation.csv"
    )
    stage240 = stage240[stage240["selection_rule"].eq("valid_f1")].copy()
    stage240["seed"] = stage240["seed"].astype(int)

    stage208 = pd.read_csv(
        TABLES / "stage208_dwd_persistent_repeat_protection_evaluation.csv"
    )
    protocol = load_json(
        TABLES / "stage204_dwd_protocol_preregistration.json"
    )
    primary_far = float(protocol["primary_far"])
    stage208 = stage208[
        stage208["selection_rule"].eq("valid_f1")
        & stage208["max_far"].astype(float).eq(primary_far)
    ].copy()
    stage208["seed"] = stage208["seed"].astype(int)

    return {
        "stage240": stage240,
        "stage208": stage208,
    }


def local_arm_reproduction_audit(
    evaluation: pd.DataFrame,
    reference: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    """Compare local-selection arms with frozen upstream artifacts.

    The four non-DWD providers are expected to reproduce Stage240 exactly.
    DWD remains a same-rule reference only: Stage242 reconstructs its local
    arm from the Stage204 protocol on the full candidate grid, whereas
    Stage208's frozen ``valid_f1`` selection is nested in its FAR grid.
    The audit records that distinction instead of claiming an exact replay.
    """

    metric_pairs = [
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "f1"),
        ("false_alarm_rate", "false_alarm_rate"),
        ("roc_auc", "roc_auc"),
        ("average_precision", "average_precision"),
    ]
    rows: list[dict[str, object]] = []
    stage240 = reference["stage240"]
    stage208 = reference["stage208"]

    for provider in PROVIDERS:
        local = evaluation[
            evaluation["target"].eq(provider)
            & evaluation["arm"].eq("target_valid_selected")
        ].copy()
        local["seed"] = local["seed"].astype(int)

        if provider == "DWD":
            upstream = stage208.copy()
            upstream_columns = {
                "precision": "test_precision",
                "recall": "test_recall",
                "f1": "test_f1",
                "false_alarm_rate": "test_false_alarm_rate",
                "roc_auc": "test_roc_auc",
                "average_precision": "test_average_precision",
            }
            reference_label = "stage208_valid_f1_primary_far"
            exact_upstream_replay_expected = False
        else:
            upstream = stage240[stage240["source"].eq(provider)].copy()
            upstream_columns = {
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "false_alarm_rate": "false_alarm_rate",
                "roc_auc": "roc_auc",
                "average_precision": "average_precision",
            }
            reference_label = "stage240_valid_f1"
            exact_upstream_replay_expected = True

        local = local.sort_values("seed", kind="stable").reset_index(drop=True)
        upstream = (
            upstream.sort_values("seed", kind="stable").reset_index(drop=True)
        )
        if local["seed"].tolist() != upstream["seed"].tolist():
            raise RuntimeError(
                f"local-arm seed mismatch for {provider}: "
                f"{local['seed'].tolist()} vs {upstream['seed'].tolist()}"
            )

        max_abs_differences: dict[str, float] = {}
        exact_match = True
        for local_column, _ in metric_pairs:
            upstream_column = upstream_columns[local_column]
            difference = np.abs(
                local[local_column].to_numpy(dtype=float)
                - upstream[upstream_column].to_numpy(dtype=float)
            )
            max_difference = float(difference.max()) if len(difference) else 0.0
            max_abs_differences[local_column] = max_difference
            if max_difference > 1e-12:
                exact_match = False

        rows.append(
            {
                "provider": provider,
                "reference": reference_label,
                "seed_count": int(len(local)),
                "exact_upstream_replay_expected": (
                    exact_upstream_replay_expected
                ),
                "exact_match": exact_match,
                "max_abs_differences": max_abs_differences,
                "max_abs_difference": max(max_abs_differences.values()),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Stage242 multisource transfer matrix."
    )
    parser.add_argument(
        "--reuse-evaluation",
        action="store_true",
        help=(
            "Reuse the frozen Stage242 evaluation, station, and transfer "
            "manifest tables; skip detector fitting and attack replay."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = [
        DWD_SELECTION,
        STAGE240_SELECTION,
        STAGE241,
        *PROTOCOLS.values(),
        *NORMALIZED.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing frozen artifacts for Stage242: " + ", ".join(missing)
        )

    reuse_paths = {
        "evaluation": TABLES / f"{PREFIX}_evaluation.csv",
        "station": TABLES / f"{PREFIX}_station_detail.csv",
        "transfer": TABLES / f"{PREFIX}_transfer_manifest.csv",
    }
    if args.reuse_evaluation:
        reuse_missing = [
            str(path)
            for path in reuse_paths.values()
            if not path.is_file() or path.stat().st_size == 0
        ]
        if reuse_missing:
            raise FileNotFoundError(
                "cannot reuse Stage242 evaluation; missing: "
                + ", ".join(reuse_missing)
            )
        evaluation = pd.read_csv(reuse_paths["evaluation"])
        station = pd.read_csv(reuse_paths["station"])
        transfer = pd.read_csv(reuse_paths["transfer"])
        for path in reuse_paths.values():
            print(
                json.dumps(
                    {
                        "status": "reused",
                        "path": str(path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    else:
        evaluations: list[pd.DataFrame] = []
        stations: list[pd.DataFrame] = []
        transfers: list[pd.DataFrame] = []
        for target in PROVIDERS:
            frame, protocol = target_data(target)
            eval_rows, station_rows, transfer_rows = evaluate_target(
                target,
                protocol,
                frame,
            )
            evaluations.append(pd.DataFrame(eval_rows))
            stations.append(pd.DataFrame(station_rows))
            transfers.append(pd.DataFrame(transfer_rows))
            print(
                json.dumps(
                    {
                        "target": target,
                        "status": "completed",
                        "evaluation_rows": len(eval_rows),
                        "station_rows": len(station_rows),
                        "transfer_rows": len(transfer_rows),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        evaluation = pd.concat(evaluations, ignore_index=True)
        station = pd.concat(stations, ignore_index=True)
        transfer = pd.concat(transfers, ignore_index=True)
        evaluation.to_csv(
            reuse_paths["evaluation"],
            index=False,
            encoding="utf-8-sig",
        )
        station.to_csv(
            reuse_paths["station"],
            index=False,
            encoding="utf-8-sig",
        )
        transfer.to_csv(
            reuse_paths["transfer"],
            index=False,
            encoding="utf-8-sig",
        )
    station["f1"] = station_f1(station)
    station_aggregate = (
        station.groupby(
            ["target", "arm", "station_id"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tn=("tn", "sum"),
        )
    )
    station_aggregate["f1"] = station_f1(station_aggregate)

    local = station_aggregate[
        station_aggregate["arm"].eq("target_valid_selected")
    ][["target", "station_id", "f1"]].rename(
        columns={"f1": "f1_target_valid_selected"}
    )
    pairwise_rows: list[dict[str, object]] = []
    pairwise_specs = (
        transfer[["target", "source"]]
        .drop_duplicates()
        .sort_values(["target", "source"], kind="stable")
        .to_dict("records")
    )
    for row in pairwise_specs:
        arm_name = (
            f"{str(row['source']).lower().replace(' ', '_')}_recalibrated"
        )
        candidate = station_aggregate[
            station_aggregate["target"].eq(str(row["target"]))
            & station_aggregate["arm"].eq(arm_name)
        ][["target", "station_id", "f1"]].rename(
            columns={"f1": "f1_source_recalibrated"}
        )
        aligned = local.merge(
            candidate,
            on=["target", "station_id"],
            validate="one_to_one",
        )
        aligned["difference"] = (
            aligned["f1_target_valid_selected"]
            - aligned["f1_source_recalibrated"]
        )
        values = aligned["difference"].to_numpy(dtype=float)
        pairwise_rows.append(
            {
                "target": str(row["target"]),
                "source": str(row["source"]),
                "station_count": int(len(aligned)),
                "f1_difference": float(values.mean()),
                "positive_stations": int((values > 0.0).sum()),
                "negative_stations": int((values < 0.0).sum()),
                "zero_stations": int((values == 0.0).sum()),
                "local_direction_positive": bool(values.mean() > 0.0),
            }
        )
    pairwise = pd.DataFrame(pairwise_rows)
    local_reproduction = local_arm_reproduction_audit(
        evaluation,
        load_local_reference_rows(),
    )
    exact_provider_count = sum(
        bool(row["exact_match"]) for row in local_reproduction
    )
    expected_exact_provider_count = len(PROVIDERS) - 1

    target_rows: list[dict[str, object]] = []
    for target in PROVIDERS:
        subset = pairwise[pairwise["target"].eq(target)]
        target_rows.append(
            {
                "target": target,
                "source_count": int(len(subset)),
                "local_direction_positive_sources": int(
                    subset["local_direction_positive"].sum()
                ),
                "mean_f1_difference": float(subset["f1_difference"].mean()),
                "min_f1_difference": float(subset["f1_difference"].min()),
                "max_f1_difference": float(subset["f1_difference"].max()),
            }
        )
    target_summary = pd.DataFrame(target_rows)

    source_rows: list[dict[str, object]] = []
    for source in PROVIDERS:
        subset = pairwise[pairwise["source"].eq(source)]
        source_rows.append(
            {
                "source": source,
                "target_count": int(len(subset)),
                "local_direction_positive_targets": int(
                    subset["local_direction_positive"].sum()
                ),
                "mean_f1_difference": float(subset["f1_difference"].mean()),
                "min_f1_difference": float(subset["f1_difference"].min()),
                "max_f1_difference": float(subset["f1_difference"].max()),
            }
        )
    source_summary = pd.DataFrame(source_rows)
    pairwise_positive = int(pairwise["local_direction_positive"].sum())
    target_all_positive = bool(
        target_summary["local_direction_positive_sources"]
        .eq(4)
        .all()
    )
    source_all_positive = bool(
        source_summary["local_direction_positive_targets"]
        .eq(4)
        .all()
    )

    checks: list[dict[str, object]] = [
        {
            "check": "provider_count",
            "status": "PASS" if len(PROVIDERS) == 5 else "FAIL",
            "value": len(PROVIDERS),
        },
        {
            "check": "pairwise_transfer_count",
            "status": "PASS" if len(pairwise) == 20 else "FAIL",
            "value": len(pairwise),
        },
        {
            "check": "pairwise_station_count",
            "status": "PASS"
            if int(pairwise["station_count"].sum()) == 216
            else "FAIL",
            "value": int(pairwise["station_count"].sum()),
        },
        {
            "check": "target_direction_heterogeneity_retained",
            "status": "PASS" if not target_all_positive else "FAIL",
            "value": target_summary[
                ["target", "local_direction_positive_sources"]
            ].to_dict("records"),
        },
        {
            "check": "source_direction_heterogeneity_retained",
            "status": "PASS" if not source_all_positive else "FAIL",
            "value": source_summary[
                ["source", "local_direction_positive_targets"]
            ].to_dict("records"),
        },
        {
            "check": "pairwise_negative_cases_retained",
            "status": "PASS" if pairwise_positive == 18 else "FAIL",
            "value": pairwise_positive,
        },
        {
            "check": "stage240_local_arm_exact_reproduction",
            "status": (
                "PASS"
                if exact_provider_count == expected_exact_provider_count
                else "FAIL"
            ),
            "value": {
                "exact_provider_count": exact_provider_count,
                "expected_exact_provider_count": expected_exact_provider_count,
                "providers": [
                    {
                        "provider": row["provider"],
                        "reference": row["reference"],
                        "exact_match": row["exact_match"],
                        "exact_upstream_replay_expected": row[
                            "exact_upstream_replay_expected"
                        ],
                    }
                    for row in local_reproduction
                ],
            },
        },
    ]
    failed = [check for check in checks if check["status"] == "FAIL"]
    payload = {
        "status": (
            "PASS_PIPELINE_REVIEW_REQUIRED"
            if not failed
            else "FAIL"
        ),
        "protocol": (
            "Preregistered-source, read-only multisource transfer matrix. "
            "Source feature/window/weight are frozen; target-domain valid "
            "normal rows are used only for threshold recalibration; target "
            "test labels are evaluation-only. All attacks are transparent "
            "controlled injections."
        ),
        "provider_count": len(PROVIDERS),
        "pairwise_count": int(len(pairwise)),
        "pairwise_station_count": int(pairwise["station_count"].sum()),
        "pairwise_local_positive": pairwise_positive,
        "all_targets_local_positive": target_all_positive,
        "all_sources_local_positive": source_all_positive,
        "negative_pair_count": int(20 - pairwise_positive),
        "negative_pairs": clean_records(
            pairwise.loc[
                ~pairwise["local_direction_positive"],
                ["target", "source", "f1_difference"],
            ].to_dict("records")
        ),
        "pairwise_rows": clean_records(pairwise.to_dict("records")),
        "target_summary_rows": clean_records(
            target_summary.to_dict("records")
        ),
        "source_summary_rows": clean_records(
            source_summary.to_dict("records")
        ),
        "local_arm_reproduction": local_reproduction,
        "local_arm_exact_provider_count": int(exact_provider_count),
        "checks": checks,
        "artifacts": {
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    }

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pairwise.to_csv(
        TABLES / f"{PREFIX}_pairwise.csv",
        index=False,
        encoding="utf-8-sig",
    )
    target_summary.to_csv(
        TABLES / f"{PREFIX}_target_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    source_summary.to_csv(
        TABLES / f"{PREFIX}_source_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    transfer.to_csv(
        TABLES / f"{PREFIX}_transfer_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "provider": row["provider"],
                "reference": row["reference"],
                "seed_count": row["seed_count"],
                "exact_upstream_replay_expected": row[
                    "exact_upstream_replay_expected"
                ],
                "exact_match": row["exact_match"],
                "max_abs_difference": row["max_abs_difference"],
                "max_abs_differences": json.dumps(
                    row["max_abs_differences"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for row in local_reproduction
        ]
    ).to_csv(
        TABLES / f"{PREFIX}_local_reproduction.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段242 多源迁移矩阵",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 机构数：`{payload['provider_count']}`；有向迁移对：`{payload['pairwise_count']}`；站点-对：`{payload['pairwise_station_count']}`",
        f"- 目标域本地选择方向为正的迁移对：`{pairwise_positive}/20`；反向或持平：`{20 - pairwise_positive}/20`",
        f"- Stage240 本地臂逐行精确复现：`{exact_provider_count}/4` 个来源机构；DWD 为同规则但不同冻结运行点的参照，不按 Stage240 复现口径计算",
        "",
        "## 目标域视角",
        "",
        "| 目标机构 | 来源数 | 本地方向为正 | 平均F1差值 | 最小 | 最大 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in target_summary.to_dict("records"):
        lines.append(
            f"| {row['target']} | {row['source_count']} | "
            f"{row['local_direction_positive_sources']} | "
            f"{row['mean_f1_difference']:+.6f} | "
            f"{row['min_f1_difference']:+.6f} | "
            f"{row['max_f1_difference']:+.6f} |"
        )
    lines += [
        "",
        "## 来源域视角",
        "",
        "| 来源机构 | 目标数 | 本地方向为正 | 平均F1差值 | 最小 | 最大 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in source_summary.to_dict("records"):
        lines.append(
            f"| {row['source']} | {row['target_count']} | "
            f"{row['local_direction_positive_targets']} | "
            f"{row['mean_f1_difference']:+.6f} | "
            f"{row['min_f1_difference']:+.6f} | "
            f"{row['max_f1_difference']:+.6f} |"
        )
    lines += [
        "",
        "## 边界",
        "",
        "- 迁移矩阵只重用五个来源域已冻结的 `valid_f1` 特征、窗口和权重；阈值始终在目标域 valid 正常样本上重校准，目标域 test 只评价。",
        "- MeteoSwiss、NOAA ISD、KNMI 和 ECCC 的本地臂与阶段240的 `valid_f1` 评价逐行完全一致；DWD 本地臂按阶段204协议重选，与阶段208在固定 FAR 网格内的冻结 `valid_f1` 选择不是同一复现口径。",
        "- 该矩阵不是新的 test-time 选择实验，也没有把 DWD 绝对阈值直接迁移。",
        "- 多源扩展没有被写成全部正向：`KNMI` 作为来源域时出现两个反向或持平结果，分别是 `DWD ← KNMI` 和 `ECCC ← KNMI`，必须保留。",
        "- 20 个有向迁移对不是 20 个独立机构样本，机构级依赖仍然存在。",
        "- 单个迁移对只有一个目标机构，未构造或报告机构级 sign-flip p 值。",
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
                "provider_count": payload["provider_count"],
                "pairwise_count": payload["pairwise_count"],
                "pairwise_local_positive": pairwise_positive,
                "negative_pair_count": int(20 - pairwise_positive),
                "all_targets_local_positive": target_all_positive,
                "all_sources_local_positive": source_all_positive,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
