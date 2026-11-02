from __future__ import annotations

"""Stage 222: third-provider transfer to NOAA ISD.

The Stage208 DWD protection rule selected by ``valid_f1`` is transferred
to an independent NOAA/NCEI Global Hourly ISD subset without re-selecting
the feature, window, or weight.  Three deterministic arms are reported:

* ``dwd_threshold_transfer``: the DWD feature/window/weight and the
  DWD-valid threshold are applied unchanged to NOAA test rows.
* ``noaa_valid_recalibrated``: only the threshold is recalibrated on
  NOAA valid-normal rows; the feature/window/weight stay DWD-selected.
* ``noaa_valid_selected``: the same pre-declared candidate grid is
  selected locally on NOAA valid rows, as a target-domain reference.

The NOAA test year (2024) is evaluation-only.  The DWD selection is read
from the frozen Stage208 selection table.  The NOAA split is frozen in
Stage222 before any transfer performance is computed: 2020-2022 train,
2023 valid, and 2024 test.  Station 57014099999 is excluded using the
pre-existing Stage83 coverage audit, not test performance.

All attacks are transparent controlled injections, not real attack logs.
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

PROTOCOL = TABLES / "stage222_noaa_transfer_protocol_preregistration.json"
NORMALIZED = TABLES / "stage82_noaa_extended_normalized.parquet"
QUALITY_AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
DWD_SELECTION = (
    TABLES / "stage208_dwd_persistent_repeat_protection_selection.csv"
)
PREFIX = "stage222_noaa_dwd_protection_transfer"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for record in records:
        cleaned: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(
                float(value)
            ):
                cleaned[key] = None
            elif isinstance(value, np.integer):
                cleaned[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        output.append(cleaned)
    return output


def frozen_dwd_selection(
    selection: pd.DataFrame,
    seed: int,
    far: float,
) -> dict[str, object]:
    rows = selection[
        selection["seed"].eq(int(seed))
        & selection["max_far"].astype(float).eq(float(far))
        & selection["selection_rule"].eq("valid_f1")
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one DWD valid_f1 selection for seed={seed}, far={far}"
        )
    row = rows.iloc[0]
    feature = str(row["feature"])
    window = int(row["window"])
    weight = float(row["weight"])
    if feature not in FEATURES or window not in WINDOWS or weight not in WEIGHTS:
        raise RuntimeError("DWD selection is outside the frozen candidate grid")
    return {
        "feature": feature,
        "window": window,
        "weight": weight,
        "dwd_threshold": float(row["threshold"]),
    }


def make_noaa_frame(protocol: dict) -> pd.DataFrame:
    stations = set(str(value) for value in protocol["station_ids"])
    split = protocol["split"]
    years = {int(value) for value in protocol["years"]}
    source = pd.read_parquet(NORMALIZED)
    frame = source[
        source["station_id"].astype(str).isin(stations)
        & source["year"].astype(int).isin(years)
    ].copy()
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    if set(frame["station_id"].astype(str)) != stations:
        raise RuntimeError("NOAA station set differs from Stage222 protocol")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("NOAA contains duplicate station timestamps")

    split_by_year: dict[int, str] = {}
    for name in ["train", "valid", "test"]:
        for year in split[name]:
            split_by_year[int(year)] = name
    if set(split_by_year) != years:
        raise RuntimeError("NOAA split years and protocol years differ")
    frame["split"] = frame["year"].astype(int).map(split_by_year)
    frame["observation_present"] = pd.to_numeric(
        frame["temperature_c"], errors="coerce"
    ).notna()
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("NOAA split labels are incomplete")
    return frame


def evaluate_arm(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    score: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    y = frame["is_attack"].astype(bool).to_numpy()
    return (
        {
            **metrics(y, predicted, score),
            **event_metrics(frame, predicted),
        },
        type_metrics(frame, predicted, score),
    )


def main() -> int:
    protocol = load_json(PROTOCOL)
    quality = load_json(QUALITY_AUDIT)
    frame = make_noaa_frame(protocol)
    stations = set(str(value) for value in protocol["station_ids"])
    if set(frame["station_id"].astype(str)) != stations:
        raise RuntimeError("NOAA retained station set changed")

    excluded = set(str(value) for value in protocol["excluded_station_ids"])
    if excluded & stations:
        raise RuntimeError("NOAA excluded station is still retained")
    audit_excluded = {
        str(row["station_id"])
        for row in quality.get("excluded_station_years", [])
    }
    if excluded != audit_excluded:
        raise RuntimeError("NOAA exclusion differs from Stage83 audit")

    selection = pd.read_csv(DWD_SELECTION)
    base = detector_frame(frame)
    detectors = fit_detectors(base, protocol)
    train_normal = (
        base["split"].eq("train") & base["observation_present"]
    )
    components = pd.DataFrame(index=base.index)
    for name, detector in detectors.items():
        components[name] = detector.score(base)
    fused, _ = fuse_scores(base, components)
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

        valid_components = pd.DataFrame(index=valid_attacked.index)
        for name, detector in detectors.items():
            valid_components[name] = detector.score(valid_attacked)
        valid_fused, _ = fuse_scores(valid_attacked, valid_components)
        valid_base_z = robust_standardize(
            valid_fused,
            train_normal,
        ).to_numpy(dtype=float)
        test_components = pd.DataFrame(index=test_attacked.index)
        for name, detector in detectors.items():
            test_components[name] = detector.score(test_attacked)
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
            raise RuntimeError("NOAA valid attack injection is empty")
        if not test_frame["is_attack"].any():
            raise RuntimeError("NOAA test attack injection is empty")
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
        noaa_threshold = far_threshold(
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
            valid_types = type_metrics(
                valid_frame,
                valid_pred,
                valid_score,
            )
            key_local = selection_key(
                "valid_f1",
                valid_metrics,
                macro_event_coverage(valid_types, ATTACK_TYPES),
                float(valid_types["stuck"]["event_coverage"]),
                macro_event_coverage(
                    valid_types,
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
            raise RuntimeError("local NOAA candidate selection failed")

        arms = {
            "dwd_threshold_transfer": {
                "feature": frozen["feature"],
                "window": frozen["window"],
                "weight": frozen["weight"],
                "threshold": frozen["dwd_threshold"],
                "score": test_transferred,
            },
            "noaa_valid_recalibrated": {
                "feature": frozen["feature"],
                "window": frozen["window"],
                "weight": frozen["weight"],
                "threshold": noaa_threshold,
                "score": test_transferred,
            },
            "noaa_valid_selected": {
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
            for station_id in sorted(stations):
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
                "noaa_recalibrated_threshold": noaa_threshold,
                "noaa_local_feature": best_local["feature"],
                "noaa_local_window": best_local["window"],
                "noaa_local_weight": best_local["weight"],
                "noaa_local_threshold": best_local["threshold"],
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

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段222在NOAA/NCEI Global Hourly ISD第三来源上执行"
            "DWD冻结保护候选迁移。使用阶段83质量审计后2020-2024"
            "可用的11个站点，2020-2022 train、2023 valid、2024 test；"
            "排除57014099999，因为2020和2021覆盖低于30%。DWD的"
            "feature、window、weight从阶段208冻结selection表逐种子"
            "读取，不重新选择；三臂为DWD绝对阈值直接迁移、保留DWD"
            "特征/窗口/权重并在NOAA valid正常样本重校准阈值、以及"
            "NOAA valid本地选择参考；valid攻击序列和test攻击序列"
            "分开生成。test只评价，不根据test选择特征、"
            "窗口、权重、阈值、站点或攻击位置。所有攻击都是透明受控"
            "注入，不是真实攻击日志。"
        ),
        "source": protocol["source"],
        "transfer_source": "Deutscher Wetterdienst (DWD)",
        "primary_far": primary_far,
        "primary_endpoint": protocol["primary_endpoint"],
        "cluster_unit": protocol["cluster_unit"],
        "seeds": protocol["seeds"],
        "station_count": len(stations),
        "excluded_station_ids": sorted(excluded),
        "normalized_sha256": sha256_file(NORMALIZED),
        "quality_audit_sha256": sha256_file(QUALITY_AUDIT),
        "protocol_sha256": sha256_file(PROTOCOL),
        "dwd_selection_sha256": sha256_file(DWD_SELECTION),
        "stage208_json_sha256": sha256_file(
            TABLES / "stage208_dwd_persistent_repeat_protection.json"
        ),
        "stage83_json_sha256": sha256_file(QUALITY_AUDIT),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "attack_parameters": ATTACK_PARAMETERS,
        "components": COMPONENTS,
        "transfer_manifest": clean_records(transfer_manifest),
        "aggregate": clean_records(combined.to_dict("records")),
        "type_aggregate": clean_records(type_aggregate.to_dict("records")),
        "evaluation_rows": int(len(evaluation)),
        "event_rows": int(len(events)),
        "station_detail_rows": int(len(station_detail)),
        "station_event_detail_rows": int(len(station_event_detail)),
        "selection_rows": int(len(selections)),
        "limitations": [
            "NOAA观测是真实外部数据，但攻击标签是透明受控注入，不是真实网络攻击日志。",
            "这是第三个来源的冻结迁移结果，不能写成所有机构的普适领先。",
            "NOAA站点主要位于中国西部，与DWD和MeteoSwiss的站点网络、传感器和处理链不同。",
            "2025本地文件不完整，因此test固定为完整的2024年，主运行点仍是FAR=0.01。",
            "DWD绝对阈值直接迁移失败必须保留；NOAA valid重校准不代表无目标域标签部署。",
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
    lines = [
        "# 阶段222 NOAA第三来源DWD保护迁移",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 主FAR：`{primary_far}`；主终点：`{protocol['primary_endpoint']}`",
        f"- 站点：`{len(stations)}`；排除：`{sorted(excluded)}`",
        "- 切分：`2020-2022 train / 2023 valid / 2024 test`",
        "- DWD选择：`Stage208 valid_f1 feature/window/weight` 逐种子冻结读取",
        "",
        "## 三臂对照",
        "",
        "| 方案 | F1均值 | F1标准差 | Recall | Precision | FAR | 事件覆盖 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined.to_dict("records"):
        lines.append(
            f"| {row['arm']} | {row['f1_mean']:.6f} | "
            f"{row['f1_std']:.6f} | {row['recall_mean']:.6f} | "
            f"{row['precision_mean']:.6f} | "
            f"{row['false_alarm_rate_mean']:.6f} | "
            f"{row['event_coverage_mean']:.6f} |"
        )
    lines += [
        "",
        "## 攻击类型事件覆盖",
        "",
        "| 方案 | 攻击类型 | 事件覆盖 | 行召回 |",
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
        *[f"- {item}" for item in payload["limitations"]],
        "",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "aggregate": payload["aggregate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
