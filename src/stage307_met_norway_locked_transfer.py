from __future__ import annotations

"""Stage 307: locked MET Norway Frost transfer under frozen contracts.

This is the first stage that evaluates MET Norway test labels.  It uses
the frozen Stage304 station network, split years, controlled attacks,
seeds, primary FAR, and endpoint, plus the frozen Stage305 method set,
training budgets, threshold rule, station-clustered bootstrap, BH
multiplicity rule, and non-inferiority margin.

All attacks are transparent controlled injections.  This stage cannot
establish station independence or real-attack recall; those claims remain
outside its evidence boundary.
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
PREFIX = "stage307_met_norway_locked_transfer"
DATA_PROTOCOL_PREFIX = "stage304_met_norway_protocol_preregistration"
CONTRACT_PREFIX = "stage305_met_norway_locked_evaluation_contract"
DOWNLOAD_PREFIX = "stage306_met_norway_official_download"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
import stage254_ocsvm_training_scale_sensitivity as s254  # noqa: E402
import stage256_lightweight_strong_baselines as s256  # noqa: E402
import stage262_security_domain_baselines as s262  # noqa: E402
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
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_features,
    select_candidate_stage244,
)

TRUST_METHODS = list(s248.TRUST_METHODS)
CONFIRMATORY_BASELINES = list(s248.BASELINE_METHODS)
RECONSTRUCTION_BASELINES = list(s256.NEW_BASELINES)
SECURITY_BASELINES = list(s262.SECURITY_BASELINES)
OCSVM_BUDGETS = [20000, 50000, 100000]
OCSVM_METHODS = [
    f"one_class_svm_{budget // 1000}k" for budget in OCSVM_BUDGETS
]
CONTRACT_METHODS = [
    *TRUST_METHODS,
    *CONFIRMATORY_BASELINES,
    *RECONSTRUCTION_BASELINES,
    *SECURITY_BASELINES,
]
ALL_BASELINES = [
    *[
        method
        for method in CONFIRMATORY_BASELINES
        if method != "one_class_svm"
    ],
    *OCSVM_METHODS,
    *RECONSTRUCTION_BASELINES,
    *SECURITY_BASELINES,
]
METHODS = [
    *TRUST_METHODS,
    *ALL_BASELINES,
]
SELECTION_RULE_BY_METHOD = dict(s248.SELECTION_RULE_BY_METHOD)
PRIMARY_TRUST_CANDIDATES = [
    "trust_valid_f1",
    "trust_nonstuck_event_macro",
]


def load_json(name: str) -> dict:
    return json.loads((TABLES / f"{name}.json").read_text(encoding="utf-8"))


def make_frame(
    data_protocol: dict,
    download: dict,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        TABLES / f"{DOWNLOAD_PREFIX}_normalized.parquet"
    )
    frame["station_id"] = frame["station_id"].astype(str)
    retained = set(
        str(value) for value in download["retained_station_ids"]
    )
    frame = frame[frame["station_id"].isin(retained)].copy()
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    if set(frame["station_id"]) != retained:
        raise RuntimeError("MET Norway retained station set changed")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("MET Norway split labels are incomplete")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("MET Norway duplicate station timestamps")
    if float(data_protocol["primary_far"]) != 0.01:
        raise RuntimeError("MET Norway primary FAR changed")
    return frame


def paired_station_table(
    station_metrics: pd.DataFrame,
    baselines: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in baselines:
            current = station_metrics[
                station_metrics["method"].eq(candidate)
            ]
            reference = station_metrics[
                station_metrics["method"].eq(baseline)
            ]
            aligned = current.merge(
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


def station_cluster_bootstrap(
    paired: pd.DataFrame,
    candidate: str,
    baseline: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    current = paired[
        paired["candidate"].eq(candidate)
        & paired["baseline"].eq(baseline)
    ].copy()
    if current.empty:
        raise RuntimeError(
            f"empty paired comparison: {candidate} vs {baseline}"
        )
    stations = sorted(current["station_id"].astype(str).unique())
    by_station = {
        station: current.loc[
            current["station_id"].astype(str).eq(station),
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
        "paired_rows": int(len(current)),
        "iterations": int(iterations),
        "f1_difference": float(current["f1_difference"].mean()),
        "f1_ci95_low": float(np.quantile(f1_draws, 0.025)),
        "f1_ci95_high": float(np.quantile(f1_draws, 0.975)),
        "event_coverage_difference": float(
            current["event_coverage_difference"].mean()
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
    if data_protocol["status"] != "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD":
        raise RuntimeError("MET Norway data protocol is not frozen")
    if contract["status"] != (
        "EVALUATION_CONTRACT_FROZEN_BEFORE_FULL_DOWNLOAD"
    ):
        raise RuntimeError("MET Norway evaluation contract is not frozen")
    if download["status"] != (
        "PASS_MET_NORWAY_DOWNLOAD_AND_COVERAGE_AUDIT"
    ):
        raise RuntimeError("MET Norway download audit did not pass")
    if download["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("MET Norway transfer was already evaluated")
    if list(contract["methods"]) != CONTRACT_METHODS:
        raise RuntimeError("MET Norway evaluation method set changed")
    if contract["primary_endpoint"] != "event_coverage":
        raise RuntimeError("MET Norway primary endpoint changed")
    if int(download["station_count"]) != 12:
        raise RuntimeError("MET Norway station count changed")
    if int(download["retained_station_count"]) != 12:
        raise RuntimeError("MET Norway retained station count changed")

    frame = make_frame(data_protocol, download)
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
    existing_fitted = s248.fit_baselines(base, base_matrix, train_normal)
    lightweight_fitted, lightweight_fit_info = (
        s256.fit_lightweight_baselines(base, base_matrix, train_normal)
    )
    security_fitted, security_fit_info = s262.fit_security_baselines(
        base,
        base_matrix,
        train_normal,
    )
    required_budgets = [
        int(value)
        for value in contract["method_contract"]["one_class_svm"][
            "reported_fit_budgets"
        ]
    ]
    if required_budgets != OCSVM_BUDGETS:
        raise RuntimeError("MET Norway One-Class SVM budget contract changed")
    train_rows = int(train_normal.sum())
    ocsvm_models: dict[int, object] = {}
    ocsvm_fit_rows: list[dict[str, object]] = []
    for budget in required_budgets:
        model, fit_meta = s254.fit_ocsvm(
            base_matrix,
            base,
            train_normal,
            int(budget),
        )
        ocsvm_models[int(budget)] = model
        ocsvm_fit_rows.append(
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
    station_ids = sorted(frame["station_id"].astype(str).unique().tolist())
    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

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
        valid_z = robust_standardize(valid_fused, train_normal).to_numpy(
            dtype=float
        )
        test_components = score_components(test_attacked, detectors)
        test_fused, _ = fuse_scores(test_attacked, test_components)
        test_z = robust_standardize(test_fused, train_normal).to_numpy(
            dtype=float
        )
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
        test_mask = test_attacked["split"].eq("test").to_numpy(dtype=bool)
        valid_frame = valid_attacked.loc[valid_mask].reset_index(drop=True)
        test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
        valid_normal = (
            valid_frame["observation_present"]
            & ~valid_frame["is_attack"]
        ).to_numpy(dtype=bool)
        y_test = test_frame["is_attack"].to_numpy(dtype=bool)
        station_values = test_frame["station_id"].astype(str).to_numpy()
        valid_base_score = valid_z[valid_mask]
        test_base_score = test_z[test_mask]
        valid_values = valid_matrix[valid_mask]
        test_values = test_matrix[test_mask]
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
            valid_scores[method] = (
                (1.0 - float(selected["weight"])) * valid_base_score
                + float(selected["weight"]) * valid_feature_masked[key]
            )
            test_scores[method] = (
                (1.0 - float(selected["weight"])) * test_base_score
                + float(selected["weight"]) * test_feature_masked[key]
            )
            selection_rows.append(
                {
                    "source": "MET_NORWAY_FROST",
                    "seed": seed,
                    "method": method,
                    "rule": rule,
                    "feature": selected["feature"],
                    "window": int(selected["window"]),
                    "weight": float(selected["weight"]),
                    "threshold": float(selected["threshold"]),
                }
            )
        existing_valid = s248.score_baselines(
            existing_fitted,
            valid_values,
        )
        existing_test = s248.score_baselines(
            existing_fitted,
            test_values,
        )
        lightweight_valid = s256.score_lightweight_baselines(
            lightweight_fitted,
            valid_values,
        )
        lightweight_test = s256.score_lightweight_baselines(
            lightweight_fitted,
            test_values,
        )
        security_valid = s262.score_security_baselines(
            security_fitted,
            valid_attacked,
            valid_matrix,
        )
        security_test = s262.score_security_baselines(
            security_fitted,
            test_attacked,
            test_matrix,
        )
        for method in CONFIRMATORY_BASELINES:
            if method == "one_class_svm":
                continue
            valid_scores[method] = existing_valid[method]
            test_scores[method] = existing_test[method]
        for method in RECONSTRUCTION_BASELINES:
            valid_scores[method] = lightweight_valid[method]
            test_scores[method] = lightweight_test[method]
        for method in SECURITY_BASELINES:
            valid_scores[method] = security_valid[method][valid_mask]
            test_scores[method] = security_test[method][test_mask]
        for budget in required_budgets:
            method = f"one_class_svm_{int(budget) // 1000}k"
            valid_scores[method] = s254.score_in_chunks(
                ocsvm_models[int(budget)],
                valid_values,
            )
            test_scores[method] = s254.score_in_chunks(
                ocsvm_models[int(budget)],
                test_values,
            )

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
            evaluation_rows.append(
                {
                    "source": "MET_NORWAY_FROST",
                    "seed": seed,
                    "method": method,
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
            for attack_type, values in type_values.items():
                type_rows.append(
                    {
                        "source": "MET_NORWAY_FROST",
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
                        "source": "MET_NORWAY_FROST",
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
                    "source": "MET_NORWAY_FROST",
                    "seed": seed,
                    "status": "completed",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    evaluation = pd.DataFrame(evaluation_rows)
    evaluation_aggregate = (
        evaluation.groupby(
            ["source", "method", "max_far"],
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
        .sort_values(["source", "method", "max_far"])
    )
    station_detail = pd.DataFrame(station_rows)
    station_metrics = s248.station_metric_table(station_detail)
    paired = paired_station_table(station_metrics, ALL_BASELINES)
    comparison_rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in ALL_BASELINES:
            comparison_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    **station_cluster_bootstrap(
                        paired,
                        candidate,
                        baseline,
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
    comparisons["event_coverage_primary_supported"] = (
        comparisons["event_ci95_low"].gt(0.0)
        & comparisons["event_coverage_bootstrap_bh_q"].le(0.05)
    )
    comparisons["f1_primary_supported"] = (
        comparisons["f1_ci95_low"].gt(0.0)
        & comparisons["f1_bootstrap_bh_q"].le(0.05)
    )
    support_count = int(
        comparisons["event_coverage_primary_supported"].sum()
    )
    noninferior_count = int(
        comparisons["event_coverage_non_inferior"].sum()
    )
    primary_pass_count = int(
        (
            comparisons["event_coverage_primary_supported"]
            | comparisons["event_coverage_non_inferior"]
        ).sum()
    )
    comparison_count = int(len(comparisons))
    all_supported = bool(primary_pass_count == comparison_count)
    payload = {
        "status": "COMPLETED_MET_NORWAY_LOCKED_TRANSFER",
        "transfer_gate_status": (
            "PASS_ALL_PRE_REGISTERED_CANDIDATE_BASELINE_PAIRS"
            if all_supported
            else "NOT_PASSED"
        ),
        "protocol": (
            "阶段307在阶段304数据协议、阶段305评价契约和阶段306"
            "下载审计之上执行MET Norway Frost独立机构锁定迁移。"
            "13个方法使用相同train/valid/test、透明受控攻击、五个种子"
            "和FAR=0.01主终点；One-Class SVM完整报告20k/50k/100k"
            "预算。所有基线与Trust在同一MET Norway test上评价，"
            "不利结果原样保留。"
        ),
        "provider": data_protocol["provider"],
        "protocol_id": data_protocol["protocol_id"],
        "contract_id": contract["contract_id"],
        "primary_far": primary_far,
        "primary_endpoint": contract["primary_endpoint"],
        "provider_count": 1,
        "station_count": int(len(station_ids)),
        "seed_count": int(evaluation["seed"].nunique()),
        "method_count": int(evaluation["method"].nunique()),
        "methods": METHODS,
        "primary_trust_candidates": PRIMARY_TRUST_CANDIDATES,
        "all_baselines": ALL_BASELINES,
        "required_ocsvm_budgets": required_budgets,
        "ocsvm_fit_rows": ocsvm_fit_rows,
        "feature_count": int(base_matrix.shape[1]),
        "feature_names": feature_names,
        "lightweight_fit_info": lightweight_fit_info,
        "security_fit_info": security_fit_info,
        "evaluation_rows": int(len(evaluation)),
        "selection_rows": int(len(selection_rows)),
        "provider_equal": s248.clean_records(
            evaluation_aggregate.to_dict("records")
        ),
        "comparisons": s248.clean_records(
            comparisons.to_dict("records")
        ),
        "primary_support_count": support_count,
        "primary_noninferior_count": noninferior_count,
        "primary_pass_count": primary_pass_count,
        "primary_comparison_count": comparison_count,
        "all_baselines_supported": all_supported,
        "attack_parameters": ATTACK_PARAMETERS,
        "components": COMPONENTS,
        "data_protocol_sha256": sha256_file(
            TABLES / f"{DATA_PROTOCOL_PREFIX}.json"
        ),
        "evaluation_contract_sha256": sha256_file(
            TABLES / f"{CONTRACT_PREFIX}.json"
        ),
        "download_sha256": sha256_file(
            TABLES / f"{DOWNLOAD_PREFIX}.json"
        ),
        "normalized_sha256": sha256_file(
            TABLES / f"{DOWNLOAD_PREFIX}_normalized.parquet"
        ),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "promotion_status": {
            "provider_dependence": (
                "INDEPENDENT_PROVIDER_TRANSFER_TESTED"
                if all_supported
                else "BOUNDED_BY_ADVERSE_MET_NORWAY_TRANSFER"
            ),
            "station_network_dependence": "BOUNDED_BY_ONE_PROVIDER",
            "resolved_requires": (
                "Additional independent providers and independent "
                "production incident or fault logs with auditable labels."
            ),
        },
        "limitations": [
            "MET Norway is one independent provider but one station network.",
            "It cannot by itself establish station independence.",
            "Controlled injections are not real production attacks.",
            "The 2025 test period is partial and ends on 2025-08-24.",
            "A baseline win must be retained and the claim reduced.",
            "Provider transfer does not replace production-incident evidence.",
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
        TABLES / f"{PREFIX}_provider_equal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        TABLES / f"{PREFIX}_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_detail.to_csv(
        TABLES / f"{PREFIX}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(selection_rows).to_csv(
        TABLES / f"{PREFIX}_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(type_rows).to_csv(
        TABLES / f"{PREFIX}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段307 MET Norway Frost锁定迁移",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 网络：`{data_protocol['network_id']}`",
        f"- 站点数：`{payload['station_count']}`",
        f"- 种子数：`{payload['seed_count']}`",
        f"- 方法数：`{payload['method_count']}`",
        f"- 主FAR：`{payload['primary_far']}`",
        f"- 主终点：`{payload['primary_endpoint']}`",
        f"- 主比较支持数：`{support_count}/{comparison_count}`",
        f"- 主比较非劣数：`{noninferior_count}/{comparison_count}`",
        f"- 优势或非劣通过数：`{primary_pass_count}/"
        f"{comparison_count}`",
        f"- 迁移门槛：`{payload['transfer_gate_status']}`",
        "",
        "## 边界",
        "",
        "- MET Norway是独立机构，但仍只有一个provider网络。",
        "- 所有攻击都是透明受控注入，不是真实攻击日志。",
        "- 2025 test为部分年度，截止到2025-08-24。",
        "- 若任一基线占优，结果原样保留并降低声称。",
        "- 本阶段不替换Stage246、FMI、SMHI或日本结果。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "station_count": payload["station_count"],
                "seed_count": payload["seed_count"],
                "method_count": payload["method_count"],
                "primary_support_count": support_count,
                "primary_noninferior_count": noninferior_count,
                "primary_pass_count": primary_pass_count,
                "primary_comparison_count": comparison_count,
                "all_baselines_supported": all_supported,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
