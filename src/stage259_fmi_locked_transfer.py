from __future__ import annotations

"""Stage 259: locked fifth-provider transfer on the frozen FMI protocol.

The FMI protocol was frozen in Stage257 before observation download, and
Stage258 completed the official download and coverage audit.  This stage
now performs the pre-declared locked transfer evaluation.  It uses the
same detector, attack generator, split boundary, seeds, and ``FAR=0.01``
endpoint as Stage248/256, but the target data are the retained FMI
stations only.

The result is a new locked external test.  It can reduce provider-
dependence risk to ``PARTIALLY_CLOSED``; it cannot prove provider
independence, cannot replace production incident logs, and cannot
overwrite the frozen Stage246 endpoint.
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
PREFIX = "stage259_fmi_locked_transfer"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
import stage256_lightweight_strong_baselines as s256  # noqa: E402
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

PROTOCOL_PREFIX = "stage257_fmi_protocol_preregistration"
DOWNLOAD_PREFIX = "stage258_fmi_official_download"
TRUST_METHODS = s248.TRUST_METHODS
EXISTING_BASELINES = s248.BASELINE_METHODS
NEW_BASELINES = s256.NEW_BASELINES
BASELINES = [*EXISTING_BASELINES, *NEW_BASELINES]
METHODS = [*TRUST_METHODS, *BASELINES]
SELECTION_RULE_BY_METHOD = s248.SELECTION_RULE_BY_METHOD


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_fmi_frame(protocol: dict, audit: dict) -> pd.DataFrame:
    normalized = pd.read_parquet(
        TABLES / f"{DOWNLOAD_PREFIX}_normalized.parquet"
    )
    normalized["station_id"] = normalized["station_id"].astype(str)
    retained = set(str(value) for value in audit["retained_station_ids"])
    frame = normalized[
        normalized["station_id"].isin(retained)
    ].copy()
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    if set(frame["station_id"]) != retained:
        raise RuntimeError("FMI retained station set changed")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("FMI split labels are incomplete")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("FMI duplicate station timestamps")
    if protocol["primary_far"] != 0.01:
        raise RuntimeError("FMI primary FAR changed")
    return frame


def paired_station_table(
    station_metrics: pd.DataFrame,
    baselines: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in baselines:
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
    protocol = load_json(TABLES / f"{PROTOCOL_PREFIX}.json")
    audit = load_json(TABLES / f"{DOWNLOAD_PREFIX}.json")
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_DOWNLOAD":
        raise RuntimeError("FMI protocol is not frozen")
    if audit["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("FMI data were already used for performance")

    frame = make_fmi_frame(protocol, audit)
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
    base_matrix, feature_names = s248.feature_matrix(
        base_components,
        base_features,
        train_normal,
    )
    existing_fitted = s248.fit_baselines(base, base_matrix, train_normal)
    new_fitted, new_fit_info = s256.fit_lightweight_baselines(
        base,
        base_matrix,
        train_normal,
    )
    primary_far = float(protocol["primary_far"])
    station_ids = sorted(frame["station_id"].astype(str).unique().tolist())

    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for seed in protocol["seeds"]:
        seed = int(seed)
        valid_blocks = choose_split_attack_blocks(
            base, protocol, seed, "valid"
        )
        test_blocks = choose_split_attack_blocks(
            base, protocol, seed, "test"
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
            raise RuntimeError("FMI valid attack injection is empty")
        if not test_frame["is_attack"].any():
            raise RuntimeError("FMI test attack injection is empty")
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
                    "source": "FMI",
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
        new_valid = s256.score_lightweight_baselines(
            new_fitted,
            valid_matrix[valid_mask],
        )
        new_test = s256.score_lightweight_baselines(
            new_fitted,
            test_matrix[test_mask],
        )
        for method in EXISTING_BASELINES:
            valid_scores[method] = existing_valid[method]
            test_scores[method] = existing_test[method]
        for method in NEW_BASELINES:
            valid_scores[method] = new_valid[method]
            test_scores[method] = new_test[method]

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
                    "source": "FMI",
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
                        "source": "FMI",
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
                        "source": "FMI",
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
                {"source": "FMI", "seed": seed, "status": "completed"},
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
    paired = paired_station_table(station_metrics, BASELINES)
    comparison_rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in BASELINES:
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
    comparisons["event_coverage_bootstrap_bh_q"] = s248.benjamini_hochberg(
        comparisons["event_coverage_bootstrap_two_sided_p"].tolist()
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
        for baseline in BASELINES:
            leave_one_rows.extend(
                s248.leave_one_provider_bootstrap(
                    paired,
                    candidate,
                    baseline,
                )
            )
    leave_one = pd.DataFrame(leave_one_rows)
    payload = {
        "status": "PASS_LOCKED_EXTERNAL_TRANSFER",
        "protocol": (
            "阶段259在阶段257冻结、阶段258下载并通过覆盖率审计的FMI"
            "第五机构数据上执行锁定迁移评价。使用相同train/valid/test、"
            "透明受控攻击、五个种子和FAR=0.01主终点；Trust与三个原"
            "基线、三个轻量强基线在同一目标域比较。FMI test只评价，"
            "不用于选择特征、窗口、权重、阈值或模型。所有攻击仍是"
            "透明受控注入，不是真实攻击日志。"
        ),
        "provider": protocol["provider"],
        "protocol_id": protocol["protocol_id"],
        "primary_far": 0.01,
        "primary_endpoint": "event_coverage",
        "provider_count": 1,
        "station_count": int(
            station[["source", "station_id"]].drop_duplicates().shape[0]
        ),
        "seed_count": int(evaluation["seed"].nunique()),
        "method_count": int(len(METHODS)),
        "methods": METHODS,
        "baseline_methods": BASELINES,
        "new_baseline_methods": NEW_BASELINES,
        "existing_baseline_methods": EXISTING_BASELINES,
        "fit_info": new_fit_info,
        "feature_count": int(base_matrix.shape[1]),
        "feature_names": feature_names,
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
        "leave_one_provider_out": s248.clean_records(
            leave_one.to_dict("records")
        ),
        "attack_parameters": ATTACK_PARAMETERS,
        "components": COMPONENTS,
        "protocol_sha256": sha256_file(
            TABLES / f"{PROTOCOL_PREFIX}.json"
        ),
        "download_sha256": sha256_file(
            TABLES / f"{DOWNLOAD_PREFIX}.json"
        ),
        "normalized_sha256": sha256_file(
            TABLES / f"{DOWNLOAD_PREFIX}_normalized.parquet"
        ),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "promotion_status": {
            "provider_dependence": "PARTIALLY_CLOSED",
            "resolved_requires": (
                "Independent production incident or fault logs with "
                "auditable labels and governance."
            ),
        },
        "limitations": [
            "FMI是第五个独立机构，但不能证明机构无关或站点无关。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
            "本阶段不能替换阶段246冻结主结果，也不能回写旧test。",
            "在线因果恢复仍不等同于离线双向插值。",
            "受控注入、弱标签或测试床结果不能写成真实事故召回率。",
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
    lines = [
        "# 阶段259 FMI第五机构锁定迁移",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 机构：`{payload['provider']}`",
        f"- 站点数：`{payload['station_count']}`",
        f"- 种子数：`{payload['seed_count']}`",
        f"- 方法数：`{payload['method_count']}`",
        f"- 主FAR：`{payload['primary_far']}`",
        f"- 主终点：`{payload['primary_endpoint']}`",
        "",
        "## 边界",
        "",
        "- FMI 是第五个独立机构，但不能证明机构无关或站点无关。",
        "- 所有攻击都是透明受控注入，不是真实攻击日志。",
        "- 本阶段不替换阶段246冻结主结果，也不回写旧test。",
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
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
