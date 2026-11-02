from __future__ import annotations

"""Stage 315: NOAA 2025 temporal locked transfer.

This stage reuses the frozen Stage307 detector, feature, baseline,
threshold, event, bootstrap, and multiplicity implementations. It replaces
only the evaluation period with the pre-existing NOAA 2025 temporal
holdout. NOAA is not a new provider, and 2024 remains excluded because it
was the previous NOAA test year.
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
PACKAGE_ROOT = ROOT / "confirmatory_package" / "noaa_2025_temporal"
PREFIX = "stage315_noaa_2025_temporal_locked_transfer"
DOWNLOAD_PREFIX = "stage315_noaa_2025_temporal_data_audit"
SOURCE_NETWORK_ID = "NOAA_ISD_EXTENDED_11_STATION_NETWORK"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
import stage254_ocsvm_training_scale_sensitivity as s254  # noqa: E402
import stage256_lightweight_strong_baselines as s256  # noqa: E402
import stage262_security_domain_baselines as s262  # noqa: E402
import stage307_met_norway_locked_transfer as s307  # noqa: E402
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_frame(protocol: dict, download: dict) -> pd.DataFrame:
    frame = pd.read_parquet(
        TABLES / f"{DOWNLOAD_PREFIX}_normalized.parquet"
    )
    frame["station_id"] = frame["station_id"].astype(str)
    retained = set(str(value) for value in protocol["station_ids"])
    frame = frame[frame["station_id"].isin(retained)].copy()
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    if set(frame["station_id"]) != retained:
        raise RuntimeError("NOAA 2025 station set changed")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("NOAA 2025 split labels incomplete")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("NOAA 2025 duplicate timestamps")
    if float(protocol["primary_far"]) != 0.01:
        raise RuntimeError("NOAA 2025 primary FAR changed")
    return frame


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    protocol = load_json(PACKAGE_ROOT / "protocol_freeze.json")
    analysis = load_json(PACKAGE_ROOT / "analysis_freeze.json")
    access = load_json(PACKAGE_ROOT / "access_gate.json")
    download = load_json(TABLES / f"{DOWNLOAD_PREFIX}.json")
    if protocol["status"] != (
        "PROTOCOL_FROZEN_BEFORE_2025_PERFORMANCE_EVALUATION"
    ):
        raise RuntimeError("Stage315 protocol is not frozen")
    if protocol["provider_is_new"] is not False:
        raise RuntimeError("Stage315 provider boundary changed")
    if protocol["network_is_disjoint_from_stage304"] is not True:
        raise RuntimeError("Stage315 network boundary changed")
    if analysis["status"] != (
        "ANALYSIS_FROZEN_BEFORE_2025_PERFORMANCE_EVALUATION"
    ):
        raise RuntimeError("Stage315 analysis freeze is missing")
    if analysis["analysis_frozen"] is not True:
        raise RuntimeError("Stage315 analysis is not frozen")
    if access["status"] != "ACCESS_GATE_OPENED_AFTER_PROTOCOL_AND_ANALYSIS_FREEZE":
        raise RuntimeError("Stage315 access gate is not open")
    if download["status"] != (
        "PASS_NOAA_2025_TEMPORAL_DATA_AUDIT"
    ):
        raise RuntimeError("Stage315 data audit did not pass")
    if download["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("Stage315 data audit was already evaluated")
    if int(download["retained_station_count"]) < 2:
        raise RuntimeError("too few retained NOAA stations")
    if int(download["retained_station_count"]) != len(
        download["retained_station_ids"]
    ):
        raise RuntimeError("retained station count mismatch")
    frozen_methods = set(str(value) for value in analysis["methods"])
    expanded_methods = set(str(value) for value in s307.METHODS)
    expected_expansion = {
        "one_class_svm_20k",
        "one_class_svm_50k",
        "one_class_svm_100k",
    }
    if "one_class_svm" not in frozen_methods:
        raise RuntimeError(
            "Stage315 contract lost the One-Class SVM expansion anchor"
        )
    frozen_expanded = (
        frozen_methods - {"one_class_svm"}
    ) | expected_expansion
    if frozen_expanded != expanded_methods:
        raise RuntimeError(
            "Stage315 method set changed outside the frozen "
            "One-Class SVM expansion"
        )
    if analysis["primary_far"] != 0.01:
        raise RuntimeError("Stage315 primary FAR changed")
    if analysis["primary_endpoint"] != "event_coverage":
        raise RuntimeError("Stage315 primary endpoint changed")
    if int(analysis["bootstrap"]["iterations"]) != 10000:
        raise RuntimeError("Stage315 bootstrap iterations changed")
    if int(analysis["bootstrap"]["seed"]) != 20260921:
        raise RuntimeError("Stage315 bootstrap seed changed")

    frame = make_frame(protocol, download)
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
    base_matrix, _ = s248.feature_matrix(
        base_components,
        base_features,
        train_normal,
    )
    existing_fitted = s248.fit_baselines(base, base_matrix, train_normal)
    lightweight_fitted, _ = s256.fit_lightweight_baselines(
        base,
        base_matrix,
        train_normal,
    )
    security_fitted, _ = s262.fit_security_baselines(
        base,
        base_matrix,
        train_normal,
    )
    required_budgets = [
        int(value)
        for value in analysis["method_contract"]["one_class_svm"][
            "reported_fit_budgets"
        ]
    ]
    if required_budgets != s307.OCSVM_BUDGETS:
        raise RuntimeError("Stage315 One-Class SVM budgets changed")
    ocsvm_models: dict[int, object] = {}
    for budget in required_budgets:
        model, fit_meta = s254.fit_ocsvm(
            base_matrix,
            base,
            train_normal,
            int(budget),
        )
        ocsvm_models[int(budget)] = model
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

    primary_far = float(analysis["primary_far"])
    station_ids = sorted(frame["station_id"].astype(str).unique().tolist())
    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for raw_seed in protocol["seeds"]:
        seed = int(raw_seed)
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
        for method, rule in s307.SELECTION_RULE_BY_METHOD.items():
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
                    "source": SOURCE_NETWORK_ID,
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
        for method in s307.CONFIRMATORY_BASELINES:
            if method == "one_class_svm":
                continue
            valid_scores[method] = existing_valid[method]
            test_scores[method] = existing_test[method]
        for method in s307.RECONSTRUCTION_BASELINES:
            valid_scores[method] = lightweight_valid[method]
            test_scores[method] = lightweight_test[method]
        for method in s307.SECURITY_BASELINES:
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

        for method in s307.METHODS:
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
                    "source": SOURCE_NETWORK_ID,
                    "seed": seed,
                    "method": method,
                    "max_far": primary_far,
                    "threshold": float(threshold),
                    **metrics(y_test, predicted, test_score),
                    **event_metrics(test_frame, predicted),
                }
            )
            for attack_type, values in type_metrics(
                test_frame,
                predicted,
                test_score,
            ).items():
                type_rows.append(
                    {
                        "source": SOURCE_NETWORK_ID,
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
                        "source": SOURCE_NETWORK_ID,
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
                    "source": SOURCE_NETWORK_ID,
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
    paired = s307.paired_station_table(
        station_metrics,
        s307.ALL_BASELINES,
    )
    comparison_rows: list[dict[str, object]] = []
    for candidate in s307.TRUST_METHODS:
        for baseline in s307.ALL_BASELINES:
            comparison_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    **s307.station_cluster_bootstrap(
                        paired,
                        candidate,
                        baseline,
                        int(analysis["bootstrap"]["iterations"]),
                        int(analysis["bootstrap"]["seed"]),
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
    margin = float(analysis["non_inferiority_margin"])
    comparisons["f1_superior"] = (
        comparisons["f1_ci95_low"] > 0.0
    )
    comparisons["event_coverage_superior"] = (
        comparisons["event_ci95_low"] > 0.0
    )
    comparisons["f1_non_inferior"] = (
        comparisons["f1_ci95_low"] > margin
    )
    comparisons["event_coverage_non_inferior"] = (
        comparisons["event_ci95_low"] > margin
    )
    comparisons["primary_superior"] = comparisons[
        "event_coverage_superior"
    ]
    comparisons["primary_non_inferior"] = comparisons[
        "event_coverage_non_inferior"
    ]
    comparisons["primary_pass"] = (
        comparisons["primary_superior"]
        | comparisons["primary_non_inferior"]
    )
    support_count = int(comparisons["primary_superior"].sum())
    noninferior_count = int(
        comparisons["primary_non_inferior"].sum()
    )
    primary_pass_count = int(comparisons["primary_pass"].sum())
    comparison_count = int(len(comparisons))
    all_supported = bool(primary_pass_count == comparison_count)
    payload = {
        "status": "COMPLETED_NOAA_2025_TEMPORAL_LOCKED_TRANSFER",
        "protocol": (
            "在预先冻结的NOAA ISD 2025时间窗上执行锁定迁移。"
            "复用Stage307的方法、特征、阈值、事件和统计契约；"
            "2024旧test年份完全排除，2025仅用于评价。"
            "NOAA仍为同一provider，结果不能证明机构无关。"
            "所有攻击为透明受控注入，不是真实攻击日志。"
        ),
        "package_id": protocol["package_id"],
        "provider_id": protocol["provider_id"],
        "provider_is_new": False,
        "network_id": protocol["network_id"],
        "network_is_disjoint_from_stage304": True,
        "evaluation_period": "2025-01-01_to_2025-08-24",
        "excluded_previous_test_year": 2024,
        "station_count": int(len(station_ids)),
        "seed_count": int(evaluation["seed"].nunique()),
        "method_count": int(evaluation["method"].nunique()),
        "primary_far": primary_far,
        "primary_endpoint": analysis["primary_endpoint"],
        "non_inferiority_margin": margin,
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
        "transfer_gate_status": (
            "PASSED"
            if all_supported
            else "NOT_PASSED"
        ),
        "attack_parameters": ATTACK_PARAMETERS,
        "components": COMPONENTS,
        "protocol_freeze_sha256": sha256_file(
            PACKAGE_ROOT / "protocol_freeze.json"
        ),
        "analysis_freeze_sha256": sha256_file(
            PACKAGE_ROOT / "analysis_freeze.json"
        ),
        "download_sha256": sha256_file(
            TABLES / f"{DOWNLOAD_PREFIX}.json"
        ),
        "normalized_sha256": sha256_file(
            TABLES / f"{DOWNLOAD_PREFIX}_normalized.parquet"
        ),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "promotion_status": {
            "provider_dependence": "NOT_TESTED_BY_THIS_NETWORK",
            "temporal_dependence": (
                "TEMPORAL_HOLDOUT_TESTED"
                if all_supported
                else "BOUNDED_BY_ADVERSE_2025_TEMPORAL_HOLDOUT"
            ),
            "resolved_requires": (
                "A new independent provider and independent production "
                "incident or fault logs with auditable labels."
            ),
        },
        "limitations": [
            "NOAA remains the same provider and the same 11-station network.",
            "The 2025 temporal holdout cannot close P0-1 by itself.",
            "All attacks are transparent controlled injections.",
            "The 2025 test period is partial and ends on 2025-08-24.",
            "Any baseline win must be retained and the claim reduced.",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
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
    report = [
        "# Stage315 NOAA 2025 时间锁定迁移",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 网络：`{payload['network_id']}`",
        "- provider是否为新机构：`False`",
        "- 评价时间窗：`2025-01-01` 至 `2025-08-24`",
        "- 2024旧test年份：`EXCLUDED`",
        f"- 站点数：`{payload['station_count']}`",
        f"- 种子数：`{payload['seed_count']}`",
        f"- 方法数：`{payload['method_count']}`",
        f"- 主比较通过：`{primary_pass_count}/{comparison_count}`",
        f"- 迁移门槛：`{payload['transfer_gate_status']}`",
        "",
        "## 边界",
        "",
        "- NOAA仍为同一provider、同一11站网络，不能关闭P0-1。",
        "- 这是新时间窗，不是新机构或新生产日志。",
        "- 所有攻击为透明受控注入，不是真实攻击日志。",
        "- 基线占优结果必须原样保留并降低声称。",
        "- 本阶段不替换Stage222/248的冻结结果。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "station_count": payload["station_count"],
                "primary_pass_count": primary_pass_count,
                "primary_comparison_count": comparison_count,
                "all_baselines_supported": all_supported,
                "transfer_gate_status": payload[
                    "transfer_gate_status"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
