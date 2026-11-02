from __future__ import annotations

"""Stage 228: cross-provider transfer of the frozen DWD protection rule to KNMI.

The DWD protection rule selected by ``valid_f1`` at the primary FAR is
transferred to KNMI without re-selecting the feature, window, or weight.
Three deterministic arms are reported:

* ``dwd_threshold_transfer``: the DWD-selected feature/window/weight and
  the DWD-valid threshold are applied unchanged to KNMI test rows.
* ``knmi_valid_recalibrated``: only the threshold is recalibrated on KNMI
  valid-normal rows; the feature/window/weight stay DWD-selected.
* ``knmi_valid_selected``: the same pre-declared candidate grid is
  selected locally on KNMI valid rows, as an upper-bound reference.

The KNMI test rows are evaluation-only.  All attacks are transparent
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

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    ATTACK_PARAMETERS,
    COMPONENTS,
    choose_attack_blocks,
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
    event_metrics,
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
    evaluate_arm,
    frozen_dwd_selection,
)

PROTOCOL = TABLES / "stage224_knmi_protocol_preregistration.json"
NORMALIZED = TABLES / "stage226_knmi_normalized.parquet"
CORRECTION = TABLES / "stage226a_knmi_missingness_audit.json"
DWD_SELECTION = (
    TABLES / "stage208_dwd_persistent_repeat_protection_selection.csv"
)
PREFIX = "stage228_knmi_dwd_protection_transfer"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    protocol = load_json(PROTOCOL)
    correction = load_json(CORRECTION)
    retained = set(
        str(value) for value in correction["retained_station_ids"]
    )
    frame = pd.read_parquet(NORMALIZED)
    frame["station_id"] = frame["station_id"].astype(str)
    frame = frame[frame["station_id"].isin(retained)].copy()
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"], kind="stable"
    ).reset_index(drop=True)
    if set(frame["station_id"]) != retained:
        raise RuntimeError("retained station set differs from Stage226A")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("unexpected Stage226 split labels")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("Stage226 contains duplicate station timestamps")

    selection = pd.read_csv(DWD_SELECTION)
    base = detector_frame(frame)
    detectors = fit_detectors(base, protocol)
    train_normal = (
        base["split"].eq("train") & base["observation_present"]
    )
    components = score_components(base, detectors)
    fused, weights = fuse_scores(base, components)
    base_z = robust_standardize(fused, train_normal).to_numpy(dtype=float)

    feature_z: dict[tuple[str, int], np.ndarray] = {}
    for feature_name in FEATURES:
        for window in WINDOWS:
            feature_z[(feature_name, int(window))] = robust_standardize(
                repeat_feature(base, feature_name, int(window)),
                train_normal,
            ).to_numpy(dtype=float)

    candidates = candidate_grid()
    primary_far = float(protocol["primary_far"])
    evaluation_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    station_event_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    transfer_manifest: list[dict[str, object]] = []

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
                key = (feature_name, int(window))
                valid_feature_z[key] = robust_standardize(
                    repeat_feature(
                        valid_attacked,
                        feature_name,
                        int(window),
                    ),
                    train_normal,
                ).to_numpy(dtype=float)
                test_feature_z[key] = robust_standardize(
                    repeat_feature(
                        test_attacked,
                        feature_name,
                        int(window),
                    ),
                    train_normal,
                ).to_numpy(dtype=float)

        valid_mask = valid_attacked["split"].eq("valid").to_numpy(dtype=bool)
        test_mask = test_attacked["split"].eq("test").to_numpy(dtype=bool)
        valid_frame = valid_attacked.loc[valid_mask].reset_index(drop=True)
        test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
        if not valid_frame["is_attack"].any():
            raise RuntimeError("KNMI valid attack injection is empty")
        if not test_frame["is_attack"].any():
            raise RuntimeError("KNMI test attack injection is empty")
        valid_normal = (
            valid_frame["observation_present"]
            & ~valid_frame["is_attack"]
        ).to_numpy(dtype=bool)
        valid_base = valid_base_z[valid_mask]
        test_base = test_base_z[test_mask]

        frozen = frozen_dwd_selection(selection, seed, primary_far)
        key = (str(frozen["feature"]), int(frozen["window"]))
        valid_transferred = (
            (1.0 - float(frozen["weight"])) * valid_base
            + float(frozen["weight"]) * valid_feature_z[key][valid_mask]
        )
        test_transferred = (
            (1.0 - float(frozen["weight"])) * test_base
            + float(frozen["weight"]) * test_feature_z[key][test_mask]
        )
        knmi_threshold = far_threshold(
            valid_transferred,
            valid_normal,
            primary_far,
        )

        best_local: dict[str, object] | None = None
        for feature_name, window, weight in candidates:
            valid_score = (
                (1.0 - weight) * valid_base
                + weight
                * valid_feature_z[(feature_name, int(window))][valid_mask]
            )
            threshold = far_threshold(
                valid_score,
                valid_normal,
                primary_far,
            )
            valid_pred = valid_score >= threshold
            valid_metrics = metrics(
                valid_frame["is_attack"].astype(bool).to_numpy(),
                valid_pred,
                valid_score,
            )
            valid_type = type_metrics(
                valid_frame,
                valid_pred,
                valid_score,
            )
            key_local = selection_key(
                "valid_f1",
                valid_metrics,
                macro_event_coverage(valid_type, ATTACK_TYPES),
                float(valid_type["stuck"]["event_coverage"]),
                macro_event_coverage(
                    valid_type,
                    [name for name in ATTACK_TYPES if name != "stuck"],
                ),
                feature_name,
                int(window),
                float(weight),
            )
            if best_local is None or key_local > best_local["selection_key"]:
                best_local = {
                    "selection_key": key_local,
                    "feature": feature_name,
                    "window": int(window),
                    "weight": float(weight),
                    "threshold": float(threshold),
                }
        if best_local is None:
            raise RuntimeError("local KNMI candidate selection failed")

        arms = {
            "dwd_threshold_transfer": {
                "feature": frozen["feature"],
                "window": frozen["window"],
                "weight": frozen["weight"],
                "threshold": frozen["dwd_threshold"],
                "score": test_transferred,
            },
            "knmi_valid_recalibrated": {
                "feature": frozen["feature"],
                "window": frozen["window"],
                "weight": frozen["weight"],
                "threshold": knmi_threshold,
                "score": test_transferred,
            },
            "knmi_valid_selected": {
                "feature": best_local["feature"],
                "window": best_local["window"],
                "weight": best_local["weight"],
                "threshold": best_local["threshold"],
                    "score": (
                        (1.0 - float(best_local["weight"])) * test_base
                        + float(best_local["weight"])
                        * test_feature_z[
                            (
                                str(best_local["feature"]),
                                int(best_local["window"]),
                        )
                    ][test_mask]
                ),
            },
        }

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
                    "seed": seed,
                    "arm": arm_name,
                    "feature": arm["feature"],
                    "window": arm["window"],
                    "weight": arm["weight"],
                    "threshold": arm["threshold"],
                    **arm_metrics,
                }
            )
            event_rows.append(
                {
                    "seed": seed,
                    "arm": arm_name,
                    "events": arm_metrics["events"],
                    "detected_events": arm_metrics["detected_events"],
                    "event_coverage": arm_metrics["event_coverage"],
                    "row_recall": arm_metrics["row_recall"],
                }
            )
            for attack_type, values in arm_types.items():
                type_rows.append(
                    {
                        "seed": seed,
                        "arm": arm_name,
                        "attack_type": attack_type,
                        **values,
                    }
                )
            y_test = test_frame["is_attack"].astype(bool).to_numpy()
            station_ids = test_frame["station_id"].astype(str).to_numpy()
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
            for station_id in sorted(retained):
                mask = station_ids == station_id
                y_station = y_test[mask]
                pred_station = predicted[mask]
                station_rows.append(
                    {
                        "seed": seed,
                        "arm": arm_name,
                        "station_id": station_id,
                        "rows": int(mask.sum()),
                        "attack_rows": int(y_station.sum()),
                        "normal_rows": int((~y_station).sum()),
                        "tp": int((y_station & pred_station).sum()),
                        "fp": int((~y_station & pred_station).sum()),
                        "fn": int((y_station & ~pred_station).sum()),
                        "tn": int((~y_station & ~pred_station).sum()),
                    }
                )
                station_events = event_grouped[
                    event_grouped["station_id"].eq(station_id)
                ]
                station_event_rows.append(
                    {
                        "seed": seed,
                        "arm": arm_name,
                        "station_id": station_id,
                        "events": int(len(station_events)),
                        "detected_events": int(
                            station_events["predicted"].sum()
                        ),
                        "event_coverage": (
                            float(station_events["predicted"].mean())
                            if len(station_events)
                            else float("nan")
                        ),
                    }
                )
            selection_rows.append(
                {
                    "seed": seed,
                    "arm": arm_name,
                    "feature": arm["feature"],
                    "window": arm["window"],
                    "weight": arm["weight"],
                    "threshold": arm["threshold"],
                }
            )
        transfer_manifest.append(
            {
                "seed": seed,
                "dwd_feature": frozen["feature"],
                "dwd_window": frozen["window"],
                "dwd_weight": frozen["weight"],
                "dwd_threshold": frozen["dwd_threshold"],
                "knmi_recalibrated_threshold": knmi_threshold,
                "knmi_local_feature": best_local["feature"],
                "knmi_local_window": best_local["window"],
                "knmi_local_weight": best_local["weight"],
                "knmi_local_threshold": best_local["threshold"],
                "valid_attack_rows": int(valid_frame["is_attack"].sum()),
                "test_attack_rows": int(test_frame["is_attack"].sum()),
            }
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "status": "completed",
                    "dwd_feature": frozen["feature"],
                    "dwd_window": frozen["window"],
                    "dwd_weight": frozen["weight"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    evaluation = pd.DataFrame(evaluation_rows)
    events = pd.DataFrame(event_rows)
    type_detail = pd.DataFrame(type_rows)
    station_detail = pd.DataFrame(station_rows)
    station_event_detail = pd.DataFrame(station_event_rows)
    selections = pd.DataFrame(selection_rows)
    type_aggregate = (
        type_detail.groupby(["arm", "attack_type"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            attack_rows=("attack_rows", "sum"),
            detected_rows=("detected_rows", "sum"),
            row_recall=("row_recall", "mean"),
            event_coverage=("event_coverage", "mean"),
        )
        .sort_values(["arm", "attack_type"])
    )
    aggregate = (
        evaluation.groupby("arm", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            false_alarm_rate_mean=("false_alarm_rate", "mean"),
            average_precision_mean=("average_precision", "mean"),
        )
        .sort_values("arm")
    )
    event_aggregate = (
        events.groupby("arm", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            event_coverage_mean=("event_coverage", "mean"),
            event_coverage_std=("event_coverage", "std"),
            row_recall_mean=("row_recall", "mean"),
        )
        .sort_values("arm")
    )
    combined = aggregate.merge(
        event_aggregate,
        on=["arm", "seeds"],
        how="inner",
        validate="one_to_one",
    )
    paired_rows: list[dict[str, object]] = []
    baseline = evaluation[evaluation["arm"].eq("dwd_threshold_transfer")]
    for arm_name in [
        "knmi_valid_recalibrated",
        "knmi_valid_selected",
    ]:
        candidate = evaluation[evaluation["arm"].eq(arm_name)]
        aligned = baseline.merge(
            candidate,
            on="seed",
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
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
            paired_rows.append(
                {
                    "comparison": f"{arm_name}_minus_dwd_threshold_transfer",
                    "metric": metric,
                    "seeds": int(len(values)),
                    "mean_diff": float(np.mean(values)),
                    "std_diff": float(np.std(values, ddof=0)),
                    "positive_seeds": int(np.sum(values > 0)),
                    "negative_seeds": int(np.sum(values < 0)),
                    "zero_seeds": int(np.sum(values == 0)),
                }
            )
    paired = pd.DataFrame(paired_rows)
    primary_row = combined[
        combined["arm"].eq("dwd_threshold_transfer")
    ].iloc[0]
    recalibrated_row = combined[
        combined["arm"].eq("knmi_valid_recalibrated")
    ].iloc[0]
    local_row = combined[
        combined["arm"].eq("knmi_valid_selected")
    ].iloc[0]

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段228检验阶段208在DWD上由valid_f1选出的连续重复保护"
            "候选能否迁移到KNMI。DWD的feature、window、weight从"
            "阶段208冻结selection表逐种子读取，不重新选择；"
            "dwd_threshold_transfer直接使用DWD阈值；"
            "knmi_valid_recalibrated仅用KNMI valid正常样本重校准阈值；"
            "knmi_valid_selected在KNMI valid上使用独立注入攻击序列，"
            "按相同网格本地选择；valid攻击序列和test攻击序列分开生成。"
            "KNMI 2025 test只评价，不用于特征、窗口、权重或阈值选择。"
            "所有攻击都是透明受控注入，不是真实攻击日志。"
        ),
        "source": "KNMI_hourly",
        "transfer_source": "Deutscher Wetterdienst (DWD)",
        "primary_far": primary_far,
        "primary_endpoint": protocol["primary_endpoint"],
        "cluster_unit": protocol["cluster_unit"],
        "seeds": protocol["seeds"],
        "attack_types": protocol["attack_types"],
        "attack_parameters": ATTACK_PARAMETERS,
        "components": COMPONENTS,
        "component_weights": weights,
        "normalized_sha256": sha256_file(NORMALIZED),
        "protocol_sha256": sha256_file(PROTOCOL),
        "correction_sha256": sha256_file(CORRECTION),
        "dwd_selection_sha256": sha256_file(DWD_SELECTION),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "detector_code_sha256": {
            path.name: sha256_file(path)
            for path in sorted((SRC / "detectors").glob("*.py"))
        },
        "aggregate": clean_records(combined.to_dict("records")),
        "type_aggregate": clean_records(type_aggregate.to_dict("records")),
        "paired": clean_records(paired.to_dict("records")),
        "transfer_manifest": clean_records(transfer_manifest),
        "station_event_detail_rows": int(len(station_event_detail)),
        "dwd_threshold_transfer_f1_mean": float(primary_row["f1_mean"]),
        "knmi_valid_recalibrated_f1_mean": float(
            recalibrated_row["f1_mean"]
        ),
        "knmi_valid_selected_f1_mean": float(local_row["f1_mean"]),
        "dwd_threshold_transfer_event_coverage_mean": float(
            primary_row["event_coverage_mean"]
        ),
        "knmi_valid_recalibrated_event_coverage_mean": float(
            recalibrated_row["event_coverage_mean"]
        ),
        "knmi_valid_selected_event_coverage_mean": float(
            local_row["event_coverage_mean"]
        ),
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
    events.to_csv(
        TABLES / f"{PREFIX}_events.csv",
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
    station_detail.to_csv(
        TABLES / f"{PREFIX}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_event_detail.to_csv(
        TABLES / f"{PREFIX}_station_event_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selections.to_csv(
        TABLES / f"{PREFIX}_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        TABLES / f"{PREFIX}_paired.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段228 DWD保护候选跨机构迁移到KNMI",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 来源：`{payload['source']}`；迁移来源：`{payload['transfer_source']}`",
        f"- 主FAR：`{primary_far}`；主终点：`{protocol['primary_endpoint']}`",
        "",
        "## 三臂对照",
        "",
        "| 方案 | F1均值 | F1标准差 | Recall | FAR | 事件覆盖 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in combined.to_dict("records"):
        lines.append(
            f"| {row['arm']} | {row['f1_mean']:.6f} | "
            f"{row['f1_std']:.6f} | {row['recall_mean']:.6f} | "
            f"{row['false_alarm_rate_mean']:.6f} | "
            f"{row['event_coverage_mean']:.6f} |"
        )
    lines += [
        "",
        "## 逐种子迁移配置",
        "",
        "| 种子 | DWD特征 | 窗口 | 权重 | DWD阈值 | KNMI重校准阈值 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in transfer_manifest:
        lines.append(
            f"| {row['seed']} | {row['dwd_feature']} | "
            f"{row['dwd_window']} | {row['dwd_weight']:.2f} | "
            f"{row['dwd_threshold']:.6f} | "
            f"{row['knmi_recalibrated_threshold']:.6f} |"
        )
    lines += [
        "",
        "## 攻击类型事件覆盖",
        "",
        "| 方案 | 攻击类型 | 事件覆盖均值 | 行召回均值 |",
        "|---|---|---:|---:|",
    ]
    for row in type_aggregate.to_dict("records"):
        lines.append(
            f"| {row['arm']} | {row['attack_type']} | "
            f"{row['event_coverage']:.6f} | {row['row_recall']:.6f} |"
        )
    lines += [
        "",
        "## 限制",
        "",
        "- DWD与KNMI的站点、传感器、处理链和攻击块不同，迁移结果只描述该冻结协议下的跨机构表现，不构成所有机构的泛化声明。",
        "- `knmi_valid_selected` 是目标域本地选择参考，不代表无目标域标签的部署性能。",
        "- 所有攻击都是透明受控注入，不是真实攻击日志，不能解释为真实入侵率。",
        "- KNMI没有与DWD QN_9等价的逐行质量码；空字段只表示缺失。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "primary_far": primary_far,
                "aggregate": payload["aggregate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
