from __future__ import annotations

"""Stage 267: locked SMHI transfer under the Stage264/265 contracts.

This stage is the first time SMHI test labels are evaluated.  It uses the
frozen 18 stations, four parameters, split years, controlled attacks,
seeds, detector components, candidate features, 13 methods, FAR=0.01
endpoint, station-clustered bootstrap, and BH multiplicity rule.  No
result may be used to change the frozen contract.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = os.environ.get("TMQC_TRANSFER_PREFIX", "stage267_smhi_locked_transfer")
DATA_PROTOCOL_PREFIX = os.environ.get(
    "TMQC_DATA_PROTOCOL_PREFIX",
    "stage264_smhi_protocol_preregistration",
)
EVALUATION_CONTRACT_PREFIX = os.environ.get(
    "TMQC_EVALUATION_CONTRACT_PREFIX",
    "stage265_smhi_locked_evaluation_contract",
)
DOWNLOAD_PREFIX = os.environ.get(
    "TMQC_DOWNLOAD_PREFIX",
    "stage266_smhi_official_download",
)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
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
EXISTING_BASELINES = list(s248.BASELINE_METHODS)
LIGHTWEIGHT_BASELINES = list(s256.NEW_BASELINES)
SECURITY_BASELINES = list(s262.SECURITY_BASELINES)
ALL_BASELINES = [
    *EXISTING_BASELINES,
    *LIGHTWEIGHT_BASELINES,
    *SECURITY_BASELINES,
]
METHODS = [*TRUST_METHODS, *ALL_BASELINES]
SELECTION_RULE_BY_METHOD = dict(s248.SELECTION_RULE_BY_METHOD)
PRIMARY_TRUST_CANDIDATES = [
    "trust_valid_f1",
    "trust_nonstuck_event_macro",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_smhi_frame(
    data_protocol: dict,
    download: dict,
) -> pd.DataFrame:
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
        raise RuntimeError("SMHI retained station set changed")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("SMHI split labels are incomplete")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("SMHI duplicate station timestamps")
    if float(data_protocol["primary_far"]) != 0.01:
        raise RuntimeError("SMHI primary FAR changed")
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


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data_protocol = load_json(
        TABLES / f"{DATA_PROTOCOL_PREFIX}.json"
    )
    evaluation_contract = load_json(
        TABLES / f"{EVALUATION_CONTRACT_PREFIX}.json"
    )
    download = load_json(TABLES / f"{DOWNLOAD_PREFIX}.json")
    if data_protocol["status"] not in {
        "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD",
        "PROTOCOL_FROZEN_BEFORE_OBSERVATION_ACCESS",
    }:
        raise RuntimeError("SMHI data protocol is not frozen")
    if evaluation_contract["status"] not in {
        "EVALUATION_CONTRACT_FROZEN_BEFORE_FULL_DOWNLOAD",
        "EVALUATION_CONTRACT_FROZEN_BEFORE_PERFORMANCE",
    }:
        raise RuntimeError("SMHI evaluation contract is not frozen")
    if download["status"] not in {
        "PASS_SMHI_DOWNLOAD_AND_COVERAGE_AUDIT",
        "PASS_SMHI_STATION_HOLDOUT_DOWNLOAD_AND_COVERAGE_AUDIT",
    }:
        raise RuntimeError("SMHI download audit did not pass")
    if download["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("SMHI transfer was already evaluated")
    if list(evaluation_contract["methods"]) != METHODS:
        raise RuntimeError("SMHI evaluation method set changed")
    if evaluation_contract["primary_endpoint"] != "event_coverage":
        raise RuntimeError("SMHI primary endpoint changed")

    frame = make_smhi_frame(data_protocol, download)
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
    primary_far = float(evaluation_contract["primary_far"])
    station_ids = sorted(frame["station_id"].astype(str).unique().tolist())

    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for seed in data_protocol["seeds"]:
        seed = int(seed)
        valid_blocks = choose_split_attack_blocks(
            base, data_protocol, seed, "valid"
        )
        test_blocks = choose_split_attack_blocks(
            base, data_protocol, seed, "test"
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
            valid_fused, train_normal
        ).to_numpy(dtype=float)
        test_components = score_components(test_attacked, detectors)
        test_fused, _ = fuse_scores(test_attacked, test_components)
        test_base_z = robust_standardize(
            test_fused, train_normal
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
        test_mask = test_attacked["split"].eq("test").to_numpy(dtype=bool)
        valid_frame = valid_attacked.loc[valid_mask].reset_index(drop=True)
        test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
        if not valid_frame["is_attack"].any():
            raise RuntimeError("SMHI valid attack injection is empty")
        if not test_frame["is_attack"].any():
            raise RuntimeError("SMHI test attack injection is empty")
        valid_normal = (
            valid_frame["observation_present"]
            & ~valid_frame["is_attack"]
        ).to_numpy(dtype=bool)
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
                    "source": "SMHI",
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
            valid_matrix[valid_mask],
        )
        existing_test = s248.score_baselines(
            existing_fitted,
            test_matrix[test_mask],
        )
        lightweight_valid = s256.score_lightweight_baselines(
            lightweight_fitted,
            valid_matrix[valid_mask],
        )
        lightweight_test = s256.score_lightweight_baselines(
            lightweight_fitted,
            test_matrix[test_mask],
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
        for method in EXISTING_BASELINES:
            valid_scores[method] = existing_valid[method]
            test_scores[method] = existing_test[method]
        for method in LIGHTWEIGHT_BASELINES:
            valid_scores[method] = lightweight_valid[method]
            test_scores[method] = lightweight_test[method]
        for method in SECURITY_BASELINES:
            valid_scores[method] = security_valid[method][valid_mask]
            test_scores[method] = security_test[method][test_mask]

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
                    "source": "SMHI",
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
                        "source": "SMHI",
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
                        "source": "SMHI",
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
                {"source": "SMHI", "seed": seed, "status": "completed"},
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
    station_metrics = s248.station_metric_table(station)
    paired = paired_station_table(station_metrics, ALL_BASELINES)
    comparison_rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in ALL_BASELINES:
            comparison_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    **s248.hierarchical_bootstrap(
                        paired,
                        candidate,
                        baseline,
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
    comparisons["event_coverage_non_inferior"] = (
        comparisons["event_ci95_low"].gt(
            float(evaluation_contract["non_inferiority_margin"])
        )
        & comparisons["event_coverage_bootstrap_bh_q"].le(0.05)
    )
    if len(evaluation["source"].unique()) != 1:
        raise RuntimeError(
            "Stage267 leave-one-provider semantics require one provider"
        )
    leave_one = pd.DataFrame(
        columns=[
            "candidate",
            "baseline",
            "removed_source",
            "status",
            "provider_count",
            "station_rows",
            "iterations",
            "f1_difference",
            "f1_ci95_low",
            "f1_ci95_high",
            "event_coverage_difference",
            "event_ci95_low",
            "event_ci95_high",
            "f1_bootstrap_two_sided_p",
            "event_coverage_bootstrap_two_sided_p",
        ]
    )
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
    primary_comparison = comparisons[
        comparisons["candidate"].isin(PRIMARY_TRUST_CANDIDATES)
    ].copy()
    primary_comparison = primary_comparison.sort_values(
        "event_coverage_difference",
        ascending=False,
    )
    support_count = int(
        primary_comparison["event_coverage_primary_supported"].sum()
    )
    noninferior_count = int(
        primary_comparison["event_coverage_non_inferior"].sum()
    )
    primary_pass_count = int(
        (
            primary_comparison["event_coverage_primary_supported"]
            | primary_comparison["event_coverage_non_inferior"]
        ).sum()
    )
    primary_comparison_count = int(len(primary_comparison))
    all_supported = bool(
        primary_pass_count == primary_comparison_count
    )
    payload = {
        "status": "COMPLETED_SMHI_LOCKED_TRANSFER",
        "transfer_gate_status": (
            "PASS_ALL_PRE_REGISTERED_CANDIDATE_BASELINE_PAIRS"
            if all_supported
            else "NOT_PASSED"
        ),
        "protocol": (
            "阶段267在阶段264数据协议、阶段265评价契约和阶段266下载审计"
            "之上执行第六机构SMHI锁定迁移。13个方法使用相同train/valid/"
            "test、透明受控攻击、五个种子和FAR=0.01主终点；所有基线"
            "与Trust在同一SMHI test上评价，不利结果原样保留。"
        ),
        "provider": data_protocol["provider"],
        "protocol_id": data_protocol["protocol_id"],
        "contract_id": evaluation_contract["contract_id"],
        "primary_far": primary_far,
        "primary_endpoint": evaluation_contract["primary_endpoint"],
        "provider_count": 1,
        "provider_level": evaluation_contract["provider_level"],
        "station_count": int(
            station[["source", "station_id"]].drop_duplicates().shape[0]
        ),
        "seed_count": int(evaluation["seed"].nunique()),
        "method_count": int(len(METHODS)),
        "methods": METHODS,
        "primary_trust_candidates": PRIMARY_TRUST_CANDIDATES,
        "primary_candidate_rule": (
            "阶段246的历史冻结材料以trust_valid_f1作为主要检测候选，"
            "并将trust_nonstuck_event_macro列为非stuck扩展；阶段265在"
            "SMHI下载和评价前并列冻结两者。为避免按SMHI结果事后择优，"
            "本阶段同时报告两个候选，并要求两个候选对所有预注册基线均"
            "通过主终点优势或非劣效门槛，才允许标记外部迁移门槛通过。"
        ),
        "all_baselines": ALL_BASELINES,
        "existing_baseline_methods": EXISTING_BASELINES,
        "lightweight_baseline_methods": LIGHTWEIGHT_BASELINES,
        "security_baseline_methods": SECURITY_BASELINES,
        "feature_count": int(base_matrix.shape[1]),
        "feature_names": feature_names,
        "lightweight_fit_info": lightweight_fit_info,
        "security_fit_info": security_fit_info,
        "evaluation_rows": int(len(evaluation)),
        "station_rows": int(len(station)),
        "type_rows": int(len(type_detail)),
        "selection_rows": int(len(selections)),
        "provider_equal": s248.clean_records(
            evaluation_aggregate.to_dict("records")
        ),
        "comparisons": s248.clean_records(
            comparisons.to_dict("records")
        ),
        "leave_one_provider_status": "NOT_APPLICABLE_SINGLE_PROVIDER",
        "leave_one_provider_note": (
            "SMHI本阶段只有一个provider，移除该provider后没有保留机构，"
            "因此留一机构区间不适用，不能作为机构稳健性证据。"
        ),
        "leave_one_provider_out": s248.clean_records(
            leave_one.to_dict("records")
        ),
        "primary_comparison": s248.clean_records(
            primary_comparison.to_dict("records")
        ),
        "primary_support_count": support_count,
        "primary_noninferior_count": noninferior_count,
        "primary_pass_count": primary_pass_count,
        "primary_comparison_count": primary_comparison_count,
        "all_baselines_supported": all_supported,
        "attack_parameters": ATTACK_PARAMETERS,
        "components": COMPONENTS,
        "data_protocol_sha256": sha256_file(
            TABLES / f"{DATA_PROTOCOL_PREFIX}.json"
        ),
        "evaluation_contract_sha256": sha256_file(
            TABLES / f"{EVALUATION_CONTRACT_PREFIX}.json"
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
                "PARTIALLY_CLOSED"
                if all_supported
                else "BOUNDED_BY_ADVERSE_SMHI_TRANSFER"
            ),
            "resolved_requires": (
                "More independent providers and independent production "
                "incident or fault logs with auditable labels and governance."
            ),
        },
        "limitations": [
            "SMHI only provides one external provider in this stage.",
            "Leave-one-provider bootstrap is not applicable within SMHI.",
            "SMHI controlled injections are not real production attacks.",
            "The SMHI test cannot prove provider or station independence.",
            "A baseline win must be retained and the claim reduced.",
            "Online causal recovery is not equivalent to offline interpolation.",
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
        "# 阶段267 SMHI第六机构锁定迁移",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 机构：`{payload['provider']}`",
        f"- 站点数：`{payload['station_count']}`",
        f"- 种子数：`{payload['seed_count']}`",
        f"- 方法数：`{payload['method_count']}`",
        f"- 主FAR：`{payload['primary_far']}`",
        f"- 主终点：`{payload['primary_endpoint']}`",
        f"- 主候选：`{', '.join(PRIMARY_TRUST_CANDIDATES)}`",
        f"- 主比较支持数：`{support_count}/{primary_comparison_count}`",
        f"- 主比较非劣数：`{noninferior_count}/{primary_comparison_count}`",
        f"- 主比较优势或非劣通过数：`{primary_pass_count}/"
        f"{primary_comparison_count}`",
        f"- 外部迁移门槛：`{payload['transfer_gate_status']}`",
        "- 留一机构：`NOT_APPLICABLE_SINGLE_PROVIDER`",
        "",
        "## 边界",
        "",
        "- SMHI只是一个外部机构，不能证明机构无关或站点无关。",
        "- 所有攻击都是透明受控注入，不是真实攻击日志。",
        "- 本阶段不替换阶段246冻结主结果，也不回写旧test。",
        "- 若任一基线占优，结果必须原样保留并降低声称。",
        "- 在线因果恢复仍不等同于离线双向插值。",
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
                "primary_comparison_count": primary_comparison_count,
                "all_baselines_supported": all_supported,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
