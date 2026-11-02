from __future__ import annotations

"""Stage 302: complete the frozen Japan One-Class SVM budget grid.

Stage295 pre-registered 20k/50k/100k One-Class SVM fit budgets for the
Japan locked evaluation, but Stage297 only evaluated the 100k budget.
This stage reuses the frozen Stage294 data protocol, Stage295 evaluation
contract, Stage296 normalized data, and the Stage297 attack/threshold
semantics while changing only the One-Class SVM fit size.

It does not modify Trust, any other baseline, or the Stage297 primary
table.  All attacks remain transparent controlled injections.
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
PREFIX = "stage302_japan_ocsvm_budget_grid"
DATA_PROTOCOL_PREFIX = "stage294_japan_isd_protocol_preregistration"
CONTRACT_PREFIX = "stage295_japan_isd_locked_evaluation_contract"
DOWNLOAD_PREFIX = "stage296_japan_isd_official_download"
STAGE297_PREFIX = "stage297_japan_isd_locked_transfer"
PROTOCOL_ID = "TMQC-JAPAN-OCSVM-BUDGET-GRID-20260921-v1.0"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
import stage254_ocsvm_training_scale_sensitivity as s254  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    detector_frame,
    fit_detectors,
    inject_attack,
    score_components,
    sha256_file,
)
from stage208_dwd_persistent_repeat_protection import (  # noqa: E402
    VALID_INJECTION_SEED_OFFSET,
    choose_split_attack_blocks,
    event_metrics,
    far_threshold,
    metrics,
    type_metrics,
)
from stage217_swiss_controlled_attack_evaluation import (  # noqa: E402
    fuse_scores,
    robust_standardize,
)
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_features,
    select_candidate_stage244,
)

TRUST_METHODS = list(s248.TRUST_METHODS)
SELECTION_RULE_BY_METHOD = dict(s248.SELECTION_RULE_BY_METHOD)


def load_json(name: str) -> dict:
    return json.loads((TABLES / f"{name}.json").read_text(encoding="utf-8"))


def make_japan_frame(data_protocol: dict, download: dict) -> pd.DataFrame:
    frame = pd.read_parquet(
        TABLES / f"{DOWNLOAD_PREFIX}_normalized.parquet"
    )
    frame["station_id"] = frame["station_id"].astype(str)
    retained = set(str(value) for value in download["retained_station_ids"])
    frame = frame[frame["station_id"].isin(retained)].copy()
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    if set(frame["station_id"]) != retained:
        raise RuntimeError("Japan retained station set changed")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("Japan split labels are incomplete")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("Japan duplicate station timestamps")
    if float(data_protocol["primary_far"]) != 0.01:
        raise RuntimeError("Japan primary FAR changed")
    return frame


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
    candidate: str,
    baseline: str,
) -> pd.DataFrame:
    current = station_metrics[station_metrics["method"].eq(candidate)]
    reference = station_metrics[
        station_metrics["method"].eq(baseline)
    ]
    aligned = current.merge(
        reference,
        on=["source", "seed", "station_id"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    return pd.DataFrame(
        {
            "source": aligned["source"],
            "seed": aligned["seed"].astype(int),
            "station_id": aligned["station_id"].astype(str),
            "f1_difference": (
                aligned["f1_candidate"] - aligned["f1_baseline"]
            ),
            "event_coverage_difference": (
                aligned["event_coverage_candidate"]
                - aligned["event_coverage_baseline"]
            ),
        }
    )


def station_cluster_bootstrap(
    paired: pd.DataFrame,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    stations = sorted(paired["station_id"].astype(str).unique())
    by_station = {
        station: paired.loc[
            paired["station_id"].astype(str).eq(station),
            ["f1_difference", "event_coverage_difference"],
        ].to_numpy(dtype=float)
        for station in stations
    }
    rng = np.random.default_rng(int(seed))
    f1_draws = np.empty(iterations, dtype=float)
    event_draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        chosen = rng.integers(0, len(stations), size=len(stations))
        f1_values: list[float] = []
        event_values: list[float] = []
        for station_index in chosen:
            values = by_station[stations[int(station_index)]]
            sampled = values[
                rng.integers(0, len(values), size=len(values))
            ]
            f1_values.append(float(sampled[:, 0].mean()))
            event_values.append(float(sampled[:, 1].mean()))
        f1_draws[index] = float(np.mean(f1_values))
        event_draws[index] = float(np.mean(event_values))
    return {
        "station_count": int(len(stations)),
        "paired_rows": int(len(paired)),
        "iterations": int(iterations),
        "f1_difference": float(paired["f1_difference"].mean()),
        "f1_ci95_low": float(np.quantile(f1_draws, 0.025)),
        "f1_ci95_high": float(np.quantile(f1_draws, 0.975)),
        "event_coverage_difference": float(
            paired["event_coverage_difference"].mean()
        ),
        "event_ci95_low": float(np.quantile(event_draws, 0.025)),
        "event_ci95_high": float(np.quantile(event_draws, 0.975)),
        "f1_bootstrap_two_sided_p": min(
            1.0,
            2.0
            * min(
                float(np.mean(f1_draws <= 0.0)),
                float(np.mean(f1_draws >= 0.0)),
            ),
        ),
        "event_coverage_bootstrap_two_sided_p": min(
            1.0,
            2.0
            * min(
                float(np.mean(event_draws <= 0.0)),
                float(np.mean(event_draws >= 0.0)),
            ),
        ),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data_protocol = load_json(DATA_PROTOCOL_PREFIX)
    contract = load_json(CONTRACT_PREFIX)
    download = load_json(DOWNLOAD_PREFIX)
    stage297 = load_json(STAGE297_PREFIX)
    if data_protocol["status"] != "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD":
        raise RuntimeError("Japan data protocol is not frozen")
    if contract["status"] != (
        "EVALUATION_CONTRACT_FROZEN_BEFORE_FULL_DOWNLOAD"
    ):
        raise RuntimeError("Japan evaluation contract is not frozen")
    if download["status"] != (
        "PASS_JAPAN_ISD_DOWNLOAD_AND_COVERAGE_AUDIT"
    ):
        raise RuntimeError("Japan download audit did not pass")
    if stage297["status"] != "COMPLETED_JAPAN_ISD_LOCKED_TRANSFER":
        raise RuntimeError("Japan locked transfer is not complete")
    required_budgets = [
        int(value)
        for value in contract["method_contract"]["one_class_svm"][
            "reported_fit_budgets"
        ]
    ]
    if required_budgets != [20000, 50000, 100000]:
        raise RuntimeError("Japan One-Class SVM budget contract changed")
    if int(stage297["station_count"]) != 12:
        raise RuntimeError("Japan station count changed")
    if int(stage297["seed_count"]) != 5:
        raise RuntimeError("Japan seed count changed")

    frame = make_japan_frame(data_protocol, download)
    base = detector_frame(frame)
    detectors = fit_detectors(base, data_protocol)
    train_normal = (
        base["split"].eq("train") & base["observation_present"]
    ).to_numpy(dtype=bool)
    base_components = score_components(base, detectors)
    base_fused, _ = fuse_scores(base, base_components)
    base_z = robust_standardize(base_fused, train_normal).to_numpy(
        dtype=float
    )
    base_features = candidate_features(base, train_normal)
    base_matrix, feature_names = s248.feature_matrix(
        base_components,
        base_features,
        train_normal,
    )
    train_rows = int(train_normal.sum())
    fit_rows_by_budget = {
        budget: int(min(budget, train_rows)) for budget in required_budgets
    }

    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    models: dict[int, object] = {}
    for budget in required_budgets:
        model, fit_meta = s254.fit_ocsvm(
            base_matrix,
            base,
            train_normal,
            int(budget),
        )
        models[budget] = model
        fit_rows.append(
            {
                "requested_rows": int(budget),
                "fit_rows": int(fit_meta["fit_rows"]),
                "train_normal_rows": int(train_rows),
                "fit_seconds": float(fit_meta["fit_seconds"]),
                "peak_rss_mb": float(fit_meta["peak_rss_mb"]),
                "support_vectors": int(fit_meta["support_vectors"]),
                "support_vector_ratio": float(
                    fit_meta["support_vector_ratio"]
                ),
                "fit_seed": int(s254.FIT_SEED),
            }
        )
        print(
            json.dumps(
                {
                    "event": "fit_completed",
                    "requested_rows": int(budget),
                    "fit_rows": int(fit_meta["fit_rows"]),
                    "fit_seconds": float(fit_meta["fit_seconds"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    primary_far = float(contract["primary_far"])
    for raw_seed in data_protocol["seeds"]:
        seed = int(raw_seed)
        valid_blocks = choose_split_attack_blocks(
            base,
            data_protocol,
            seed,
            "valid",
        )
        test_blocks = choose_split_attack_blocks(
            base,
            data_protocol,
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
        valid_z = robust_standardize(
            valid_fused,
            train_normal,
        ).to_numpy(dtype=float)
        test_components = score_components(test_attacked, detectors)
        test_fused, _ = fuse_scores(test_attacked, test_components)
        test_z = robust_standardize(
            test_fused,
            train_normal,
        ).to_numpy(dtype=float)
        valid_features = candidate_features(valid_attacked, train_normal)
        test_features = candidate_features(test_attacked, train_normal)
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
        valid_mask = valid_attacked["split"].eq("valid").to_numpy(
            dtype=bool
        )
        test_mask = test_attacked["split"].eq("test").to_numpy(
            dtype=bool
        )
        valid_frame = valid_attacked.loc[valid_mask].reset_index(drop=True)
        test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
        valid_normal = (
            valid_frame["observation_present"]
            & ~valid_frame["is_attack"]
        ).to_numpy(dtype=bool)
        y_test = test_frame["is_attack"].to_numpy(dtype=bool)
        station_values = test_frame["station_id"].astype(str).to_numpy()
        valid_values = valid_matrix[valid_mask]
        test_values = test_matrix[test_mask]
        valid_base_score = valid_z[valid_mask]
        test_base_score = test_z[test_mask]
        valid_feature_masked = {
            key: values[valid_mask]
            for key, values in valid_features.items()
        }
        test_feature_masked = {
            key: values[test_mask]
            for key, values in test_features.items()
        }

        trust_valid_scores: dict[str, np.ndarray] = {}
        trust_test_scores: dict[str, np.ndarray] = {}
        for method, rule in SELECTION_RULE_BY_METHOD.items():
            selected = select_candidate_stage244(
                rule,
                valid_base_score,
                valid_feature_masked,
                valid_frame,
                valid_normal,
                primary_far,
            )
            key = (
                str(selected["feature"]),
                int(selected["window"]),
            )
            trust_valid_scores[method] = (
                (1.0 - float(selected["weight"])) * valid_base_score
                + float(selected["weight"]) * valid_feature_masked[key]
            )
            trust_test_scores[method] = (
                (1.0 - float(selected["weight"])) * test_base_score
                + float(selected["weight"]) * test_feature_masked[key]
            )

        score_by_method: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for budget in required_budgets:
            method = f"one_class_svm_{int(budget) // 1000}k"
            score_by_method[method] = (
                s254.score_in_chunks(
                    models[budget],
                    valid_values[valid_normal],
                ),
                s254.score_in_chunks(
                    models[budget],
                    test_values,
                ),
            )
        for method in TRUST_METHODS:
            score_by_method[method] = (
                trust_valid_scores[method],
                trust_test_scores[method],
            )

        for method, (valid_score, test_score) in score_by_method.items():
            threshold = far_threshold(
                valid_score,
                np.ones(valid_score.size, dtype=bool),
                primary_far,
            )
            predicted = (
                test_frame["observation_present"].to_numpy(dtype=bool)
                & (test_score >= threshold)
            )
            evaluation_rows.append(
                {
                    "source": "NOAA_ISD_JAPAN",
                    "seed": seed,
                    "method": method,
                    "requested_rows": (
                        int(method.split("_")[-1][:-1]) * 1000
                        if method.startswith("one_class_svm_")
                        else np.nan
                    ),
                    "fit_rows": (
                        int(
                            fit_rows_by_budget[
                                int(method.split("_")[-1][:-1]) * 1000
                            ]
                        )
                        if method.startswith("one_class_svm_")
                        else np.nan
                    ),
                    "max_far": primary_far,
                    "threshold": float(threshold),
                    **metrics(y_test, predicted, test_score),
                    **event_metrics(test_frame, predicted),
                }
            )
            type_values = type_metrics(
                test_frame,
                predicted,
                test_score,
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
            for station_id in sorted(
                frame["station_id"].astype(str).unique()
            ):
                mask = station_values == station_id
                y_station = y_test[mask]
                pred_station = predicted[mask]
                station_events = event_grouped[
                    event_grouped["station_id"].eq(station_id)
                ]
                station_rows.append(
                    {
                        "source": "NOAA_ISD_JAPAN",
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
                    "event": "seed_completed",
                    "seed": seed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        pd.DataFrame(evaluation_rows).to_csv(
            TABLES / f"{PREFIX}_evaluation_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(station_rows).to_csv(
            TABLES / f"{PREFIX}_station_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

    evaluation = pd.DataFrame(evaluation_rows)
    station_detail = pd.DataFrame(station_rows)
    station_metrics = station_metric_table(station_detail)

    comparison_rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for budget in required_budgets:
            baseline = f"one_class_svm_{int(budget) // 1000}k"
            paired = paired_station_table(
                station_metrics,
                candidate,
                baseline,
            )
            comparison_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "requested_rows": int(budget),
                    **station_cluster_bootstrap(
                        paired,
                        int(contract["bootstrap"]["iterations"]),
                        int(contract["bootstrap"]["seed"]),
                    ),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["f1_bootstrap_bh_q"] = s248.benjamini_hochberg(
        comparisons["f1_bootstrap_two_sided_p"].tolist()
    )
    comparisons["event_coverage_bootstrap_bh_q"] = (
        s248.benjamini_hochberg(
            comparisons[
                "event_coverage_bootstrap_two_sided_p"
            ].tolist()
        )
    )
    comparisons["f1_non_inferior"] = (
        comparisons["f1_ci95_low"].gt(
            float(contract["non_inferiority_margin"])
        )
        & comparisons["f1_bootstrap_bh_q"].le(0.05)
    )
    comparisons["event_coverage_non_inferior"] = (
        comparisons["event_ci95_low"].gt(
            float(contract["non_inferiority_margin"])
        )
        & comparisons["event_coverage_bootstrap_bh_q"].le(0.05)
    )

    aggregate = (
        evaluation.groupby(
            ["source", "method", "requested_rows", "fit_rows"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            f1_mean=("f1", "mean"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            false_alarm_rate_mean=("false_alarm_rate", "mean"),
            average_precision_mean=("average_precision", "mean"),
            event_coverage_mean=("event_coverage", "mean"),
            row_recall_mean=("row_recall", "mean"),
        )
        .sort_values(["requested_rows"])
    )
    ocsvm_evaluation = evaluation.loc[
        evaluation["method"].astype(str).str.startswith(
            "one_class_svm_"
        )
    ]
    complete = bool(
        set(required_budgets)
        == set(
            int(value)
            for value in ocsvm_evaluation["requested_rows"].unique()
        )
        and ocsvm_evaluation.groupby("requested_rows")["seed"]
        .nunique()
        .eq(5)
        .all()
    )
    all_non_inferior = bool(
        not comparisons.empty
        and comparisons["f1_non_inferior"].all()
        and comparisons["event_coverage_non_inferior"].all()
    )
    status = (
        "PASS_JAPAN_OCSVM_BUDGET_GRID_COMPLETE"
        if complete
        else "FAIL_JAPAN_OCSVM_BUDGET_GRID_INCOMPLETE"
    )
    payload = {
        "status": status,
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "Stage302 reuses the frozen Stage294/295/296 Japan protocol, "
            "attacks, seeds, threshold semantics, and evaluation code. It "
            "changes only the One-Class SVM fit size to complete the "
            "pre-registered 20k/50k/100k grid and compares each smaller "
            "budget with the frozen Stage297 100k result."
        ),
        "provider": stage297["provider"],
        "station_count": int(stage297["station_count"]),
        "seed_count": int(stage297["seed_count"]),
        "required_budgets": required_budgets,
        "observed_budgets": sorted(
            int(value)
            for value in evaluation.loc[
                evaluation["method"].astype(str).str.startswith(
                    "one_class_svm_"
                ),
                "requested_rows",
            ]
            .dropna()
            .unique()
        ),
        "budget_grid_complete": complete,
        "fit_rows": fit_rows,
        "feature_count": int(base_matrix.shape[1]),
        "feature_names": feature_names,
        "train_normal_rows": train_rows,
        "primary_far": primary_far,
        "primary_endpoint": contract["primary_endpoint"],
        "non_inferiority_margin": float(
            contract["non_inferiority_margin"]
        ),
        "evaluation_rows": int(len(evaluation)),
        "comparison_count": int(len(comparisons)),
        "all_smaller_budgets_non_inferior": all_non_inferior,
        "aggregate": s248.clean_records(aggregate.to_dict("records")),
        "comparisons": s248.clean_records(
            comparisons.to_dict("records")
        ),
        "input_sha256": {
            "stage294": sha256_file(
                TABLES / f"{DATA_PROTOCOL_PREFIX}.json"
            ),
            "stage295": sha256_file(
                TABLES / f"{CONTRACT_PREFIX}.json"
            ),
            "stage296": sha256_file(
                TABLES / f"{DOWNLOAD_PREFIX}.json"
            ),
            "stage296_normalized": sha256_file(
                TABLES / f"{DOWNLOAD_PREFIX}_normalized.parquet"
            ),
            "stage297": sha256_file(
                TABLES / f"{STAGE297_PREFIX}.json"
            ),
            "stage297_evaluation": sha256_file(
                TABLES / f"{STAGE297_PREFIX}_evaluation.csv"
            ),
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "limitations": [
            "Japan is a new NOAA station network, not an independent "
            "provider.",
            "All attacks are transparent controlled injections.",
            "This stage only completes the pre-registered One-Class SVM "
            "budget grid; it does not change the Stage297 primary table.",
            "The smaller budgets are compared with the frozen 100k result "
            "under the same station-clustered bootstrap and BH correction.",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    fit_frame = pd.DataFrame(fit_rows)
    fit_frame.to_csv(
        TABLES / f"{PREFIX}_fit_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evaluation.to_csv(
        TABLES / f"{PREFIX}_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        TABLES / f"{PREFIX}_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        TABLES / f"{PREFIX}_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage 302 Japan One-Class SVM budget grid",
        "",
        f"- Status: `{status}`",
        f"- Required budgets: `{required_budgets}`",
        f"- Observed budgets: `{payload['observed_budgets']}`",
        f"- Budget grid complete: `{complete}`",
        f"- Smaller budgets all non-inferior: `{all_non_inferior}`",
        "",
        "## Boundary",
        "",
        "- Japan is a new NOAA station network, not an independent provider.",
        "- Attacks are transparent controlled injections.",
        "- Stage297 remains the primary locked-transfer result.",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "required_budgets": required_budgets,
                "observed_budgets": payload["observed_budgets"],
                "budget_grid_complete": complete,
                "all_smaller_budgets_non_inferior": all_non_inferior,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
