from __future__ import annotations

"""Stage 210: seed-randomized replication of the Stage-208 candidate.

Stage 208 selected the ``repeat_ratio``/``repeat_run`` family and the
``valid_f1`` rule on the frozen five-seed DWD controlled-attack protocol.
This stage keeps that rule family and the train/valid/test boundary fixed,
but evaluates a new pre-declared set of attack-block seeds.  It therefore
tests whether the Stage-208 gain depends on the original attack
placements.

The new seeds are fixed in this file before evaluation.  Features,
windows, weights, and thresholds are still selected only on the 2024
valid split.  The 2025 test split is evaluation-only.  All attacks remain
transparent controlled injections, not real attack logs.
"""

import hashlib
import itertools
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
    NORMALIZED,
    PROTOCOL,
    choose_attack_blocks,
    detector_frame,
    fit_detectors,
    fuse_scores,
    inject_attack,
    load_protocol,
    metrics,
    robust_standardize,
    score_components,
    sha256_file,
)
from stage208_dwd_persistent_repeat_protection import (  # noqa: E402
    ATTACK_TYPES,
    FEATURES,
    SELECTION_RULES,
    WEIGHTS,
    WINDOWS,
    candidate_grid,
    choose_split_attack_blocks,
    clean_records,
    event_metrics,
    far_threshold,
    macro_event_coverage,
    repeat_feature,
    selection_key,
    station_metrics,
    type_metrics,
)

PREFIX = "stage210_dwd_seed_randomized_replication"
STAGE208_PREFIX = "stage208_dwd_persistent_repeat_protection"
STAGE208_JSON = TABLES / f"{STAGE208_PREFIX}.json"
REPLICATION_SEEDS = [
    271828,
    314159,
    8675309,
    1047290,
    20260918,
]
PRIMARY_RULE = "valid_f1"


def main() -> int:
    protocol = load_protocol()
    frame = pd.read_parquet(NORMALIZED)
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("Stage206 contains duplicate station timestamps")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("unexpected Stage206 split labels")

    stage208 = json.loads(STAGE208_JSON.read_text(encoding="utf-8"))
    original_seeds = [int(value) for value in stage208["seeds"]]
    if set(original_seeds) & set(REPLICATION_SEEDS):
        raise RuntimeError("replication seeds overlap the original seeds")

    base = detector_frame(frame)
    detectors = fit_detectors(base, protocol)
    train_normal = (
        base["split"].eq("train") & base["observation_present"]
    )
    candidates = candidate_grid()
    evaluation_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []

    for seed in REPLICATION_SEEDS:
        seed = int(seed)
        valid_blocks = choose_split_attack_blocks(
            base,
            protocol,
            seed,
            "valid",
        )
        test_blocks = choose_attack_blocks(base, protocol, seed)
        valid_attacked = inject_attack(
            base,
            valid_blocks,
            seed + 20260715,
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

        valid_mask = valid_attacked["split"].eq("valid").to_numpy()
        test_mask = test_attacked["split"].eq("test").to_numpy()
        valid_frame = valid_attacked.loc[valid_mask].reset_index(drop=True)
        test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
        valid_normal = (
            valid_frame["observation_present"]
            & ~valid_frame["is_attack"]
        ).to_numpy(dtype=bool)
        y_valid = valid_frame["is_attack"].to_numpy(dtype=bool)
        y_test = test_frame["is_attack"].to_numpy(dtype=bool)
        valid_base_score = valid_z[valid_mask]
        test_base_score = test_z[test_mask]

        for far in protocol["far_grid"]:
            far = float(far)
            baseline_threshold = far_threshold(
                valid_base_score,
                valid_normal,
                far,
            )
            baseline_test_pred = test_base_score >= baseline_threshold
            baseline_test_metrics = metrics(
                y_test,
                baseline_test_pred,
                test_base_score,
            )
            baseline_test_events = event_metrics(
                test_frame,
                baseline_test_pred,
            )

            best_by_rule: dict[str, dict[str, object]] = {}
            for feature_name, window, weight in candidates:
                valid_score = (
                    (1.0 - weight) * valid_base_score
                    + weight * valid_feature_z[(feature_name, window)][
                        valid_mask
                    ]
                )
                test_score = (
                    (1.0 - weight) * test_base_score
                    + weight * test_feature_z[(feature_name, window)][
                        test_mask
                    ]
                )
                threshold = far_threshold(
                    valid_score,
                    valid_normal,
                    far,
                )
                valid_pred = valid_score >= threshold
                valid_metrics = metrics(
                    y_valid,
                    valid_pred,
                    valid_score,
                )
                valid_type = type_metrics(
                    valid_frame,
                    valid_pred,
                    valid_score,
                )
                valid_event_macro = macro_event_coverage(
                    valid_type,
                    ATTACK_TYPES,
                )
                valid_stuck_event = float(
                    valid_type["stuck"]["event_coverage"]
                )
                valid_nonstuck_event = macro_event_coverage(
                    valid_type,
                    [name for name in ATTACK_TYPES if name != "stuck"],
                )
                for rule in SELECTION_RULES:
                    key = selection_key(
                        rule,
                        valid_metrics,
                        valid_event_macro,
                        valid_stuck_event,
                        valid_nonstuck_event,
                        feature_name,
                        window,
                        weight,
                    )
                    current = best_by_rule.get(rule)
                    if current is None or key > current["selection_key"]:
                        best_by_rule[rule] = {
                            "selection_key": key,
                            "feature": feature_name,
                            "window": int(window),
                            "weight": float(weight),
                            "threshold": float(threshold),
                            "test_score": test_score,
                            "valid_metrics": valid_metrics,
                            "valid_event_macro": valid_event_macro,
                            "valid_stuck_event": valid_stuck_event,
                            "valid_nonstuck_event": valid_nonstuck_event,
                        }

            for rule in SELECTION_RULES:
                selected = best_by_rule[rule]
                selected_test_pred = (
                    selected["test_score"] >= selected["threshold"]
                )
                selected_test_metrics = metrics(
                    y_test,
                    selected_test_pred,
                    selected["test_score"],
                )
                selected_test_events = event_metrics(
                    test_frame,
                    selected_test_pred,
                )
                selected_test_type = type_metrics(
                    test_frame,
                    selected_test_pred,
                    selected["test_score"],
                )
                for station_row in station_metrics(
                    test_frame,
                    selected_test_pred,
                ):
                    station_rows.append(
                        {
                            "seed": seed,
                            "max_far": far,
                            "selection_rule": rule,
                            "arm": "candidate",
                            **station_row,
                        }
                    )
                for station_row in station_metrics(
                    test_frame,
                    baseline_test_pred,
                ):
                    station_rows.append(
                        {
                            "seed": seed,
                            "max_far": far,
                            "selection_rule": rule,
                            "arm": "stage207",
                            **station_row,
                        }
                    )
                evaluation_rows.append(
                    {
                        "seed": seed,
                        "max_far": far,
                        "selection_rule": rule,
                        "feature": selected["feature"],
                        "window": selected["window"],
                        "weight": selected["weight"],
                        "threshold": selected["threshold"],
                        "valid_f1": selected["valid_metrics"]["f1"],
                        "valid_recall": selected["valid_metrics"]["recall"],
                        "valid_precision": selected["valid_metrics"][
                            "precision"
                        ],
                        "valid_false_alarm_rate": selected[
                            "valid_metrics"
                        ]["false_alarm_rate"],
                        "valid_event_macro": selected[
                            "valid_event_macro"
                        ],
                        "valid_stuck_event": selected[
                            "valid_stuck_event"
                        ],
                        "valid_nonstuck_event": selected[
                            "valid_nonstuck_event"
                        ],
                        "test_f1": selected_test_metrics["f1"],
                        "test_recall": selected_test_metrics["recall"],
                        "test_precision": selected_test_metrics[
                            "precision"
                        ],
                        "test_false_alarm_rate": selected_test_metrics[
                            "false_alarm_rate"
                        ],
                        "test_roc_auc": selected_test_metrics["roc_auc"],
                        "test_average_precision": selected_test_metrics[
                            "average_precision"
                        ],
                        "baseline_test_f1": baseline_test_metrics["f1"],
                        "baseline_test_recall": baseline_test_metrics[
                            "recall"
                        ],
                        "baseline_test_precision": baseline_test_metrics[
                            "precision"
                        ],
                        "baseline_test_false_alarm_rate": (
                            baseline_test_metrics["false_alarm_rate"]
                        ),
                        "baseline_test_event_coverage": (
                            baseline_test_events["event_coverage"]
                        ),
                        "baseline_test_row_recall": baseline_test_events[
                            "row_recall"
                        ],
                    }
                )
                event_rows.append(
                    {
                        "seed": seed,
                        "max_far": far,
                        "selection_rule": rule,
                        "events": selected_test_events["events"],
                        "detected_events": selected_test_events[
                            "detected_events"
                        ],
                        "event_coverage": selected_test_events[
                            "event_coverage"
                        ],
                        "row_recall": selected_test_events["row_recall"],
                        "baseline_events": baseline_test_events["events"],
                        "baseline_detected_events": baseline_test_events[
                            "detected_events"
                        ],
                        "baseline_event_coverage": baseline_test_events[
                            "event_coverage"
                        ],
                        "baseline_row_recall": baseline_test_events[
                            "row_recall"
                        ],
                    }
                )
                selection_rows.append(
                    {
                        "seed": seed,
                        "max_far": far,
                        "selection_rule": rule,
                        "feature": selected["feature"],
                        "window": selected["window"],
                        "weight": selected["weight"],
                        "threshold": selected["threshold"],
                        "valid_f1": selected["valid_metrics"]["f1"],
                        "valid_recall": selected["valid_metrics"]["recall"],
                        "valid_precision": selected["valid_metrics"][
                            "precision"
                        ],
                        "valid_false_alarm_rate": selected[
                            "valid_metrics"
                        ]["false_alarm_rate"],
                        "valid_event_macro": selected[
                            "valid_event_macro"
                        ],
                        "valid_stuck_event": selected[
                            "valid_stuck_event"
                        ],
                        "valid_nonstuck_event": selected[
                            "valid_nonstuck_event"
                        ],
                    }
                )
                for attack_type in ATTACK_TYPES:
                    type_rows.append(
                        {
                            "seed": seed,
                            "max_far": far,
                            "selection_rule": rule,
                            "attack_type": attack_type,
                            **selected_test_type[attack_type],
                        }
                    )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "status": "completed",
                    "valid_attack_rows": int(valid_blocks.shape[0]),
                    "test_attack_rows": int(test_blocks.shape[0]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    evaluation = pd.DataFrame(evaluation_rows)
    events = pd.DataFrame(event_rows)
    selection = pd.DataFrame(selection_rows)
    type_detail = pd.DataFrame(type_rows)
    station_detail = pd.DataFrame(station_rows)
    primary_far = float(protocol["primary_far"])
    primary = evaluation.loc[
        evaluation["max_far"].eq(primary_far)
        & evaluation["selection_rule"].eq(PRIMARY_RULE)
    ].copy()
    primary_events = events.loc[
        events["max_far"].eq(primary_far)
        & events["selection_rule"].eq(PRIMARY_RULE)
    ].copy()
    primary_type = type_detail.loc[
        type_detail["max_far"].eq(primary_far)
        & type_detail["selection_rule"].eq(PRIMARY_RULE)
    ].copy()

    summary = {
        "seeds": int(primary["seed"].nunique()),
        "candidate_test_f1_mean": float(primary["test_f1"].mean()),
        "baseline_test_f1_mean": float(primary["baseline_test_f1"].mean()),
        "candidate_test_recall_mean": float(primary["test_recall"].mean()),
        "baseline_test_recall_mean": float(
            primary["baseline_test_recall"].mean()
        ),
        "candidate_test_false_alarm_rate_mean": float(
            primary["test_false_alarm_rate"].mean()
        ),
        "baseline_test_false_alarm_rate_mean": float(
            primary["baseline_test_false_alarm_rate"].mean()
        ),
        "candidate_event_coverage_mean": float(
            primary_events["event_coverage"].mean()
        ),
        "baseline_event_coverage_mean": float(
            primary_events["baseline_event_coverage"].mean()
        ),
        "candidate_row_recall_mean": float(
            primary_events["row_recall"].mean()
        ),
        "baseline_row_recall_mean": float(
            primary_events["baseline_row_recall"].mean()
        ),
    }
    paired_rows: list[dict[str, object]] = []
    for metric, baseline_metric in [
        ("test_f1", "baseline_test_f1"),
        ("test_recall", "baseline_test_recall"),
        ("test_precision", "baseline_test_precision"),
        ("test_false_alarm_rate", "baseline_test_false_alarm_rate"),
    ]:
        diff = (
            primary[metric].to_numpy(dtype=float)
            - primary[baseline_metric].to_numpy(dtype=float)
        )
        paired_rows.append(
            {
                "metric": metric,
                "pairs": int(len(diff)),
                "positive_pairs": int((diff > 0).sum()),
                "negative_pairs": int((diff < 0).sum()),
                "zero_pairs": int((diff == 0).sum()),
                "mean_diff": float(diff.mean()),
                "std_diff": float(diff.std(ddof=0)),
                "min_diff": float(diff.min()),
                "max_diff": float(diff.max()),
            }
        )
    for metric, baseline_metric in [
        ("event_coverage", "baseline_event_coverage"),
        ("row_recall", "baseline_row_recall"),
    ]:
        diff = (
            primary_events[metric].to_numpy(dtype=float)
            - primary_events[baseline_metric].to_numpy(dtype=float)
        )
        paired_rows.append(
            {
                "metric": metric,
                "pairs": int(len(diff)),
                "positive_pairs": int((diff > 0).sum()),
                "negative_pairs": int((diff < 0).sum()),
                "zero_pairs": int((diff == 0).sum()),
                "mean_diff": float(diff.mean()),
                "std_diff": float(diff.std(ddof=0)),
                "min_diff": float(diff.min()),
                "max_diff": float(diff.max()),
            }
        )
    paired = pd.DataFrame(paired_rows)

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段210保持阶段208的候选特征族、窗口、权重网格、选择规则"
            "和train/valid/test边界不变，只预先声明五个新的攻击块种子"
            "进行独立复现。特征、窗口、权重和阈值仍只在2024 valid选择，"
            "不根据test选择，2025 test只评价；新种子不与原五种子重叠。"
            "time_shift只从阶段206干净序列取值，source_row_position和"
            "clean_values保留。攻击仍是透明受控注入，不是真实攻击日志，"
            "不能替代真实攻击日志或所有气象网络。"
        ),
        "source": "DWD_CDC",
        "normalized_sha256": sha256_file(NORMALIZED),
        "protocol_sha256": sha256_file(PROTOCOL),
        "stage208_json_sha256": sha256_file(STAGE208_JSON),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "original_seeds": original_seeds,
        "replication_seeds": REPLICATION_SEEDS,
        "primary_rule": PRIMARY_RULE,
        "primary_far": primary_far,
        "evaluation_rows": int(len(evaluation)),
        "event_rows": int(len(events)),
        "selection_rows": int(len(selection)),
        "type_detail_rows": int(len(type_detail)),
        "station_detail_rows": int(len(station_detail)),
        "summary": summary,
        "paired": clean_records(
            paired.to_dict(orient="records")
        ),
        "primary_type_aggregate": clean_records(
            primary_type.groupby(
                "attack_type",
                as_index=False,
            )
            .agg(
                attack_rows=("attack_rows", "sum"),
                detected_rows=("detected_rows", "sum"),
                row_recall=("row_recall", "mean"),
                event_coverage=("event_coverage", "mean"),
            )
            .sort_values("attack_type")
            .to_dict(orient="records")
        ),
        "limitations": [
            "新种子只改变攻击块位置，不是新的机构、站点或真实攻击数据。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
            "特征、窗口、权重和阈值仍由2024 valid选择，test只评价。",
            "该复现不能替代独立站点外部复现。",
        ],
    }
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
    selection.to_csv(
        TABLES / f"{PREFIX}_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_detail.to_csv(
        TABLES / f"{PREFIX}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_detail.to_csv(
        TABLES / f"{PREFIX}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        TABLES / f"{PREFIX}_paired.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段210 DWD新攻击块种子独立复现",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 原种子：`{original_seeds}`",
        f"- 复现种子：`{REPLICATION_SEEDS}`",
        f"- 固定规则：`{PRIMARY_RULE}`；主FAR：`{primary_far}`",
        "",
        "## 主结果",
        "",
        "| 指标 | 候选均值 | 基线均值 | 差值 | 正向种子 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        candidate_key = (
            "candidate_test_f1_mean"
            if row["metric"] == "test_f1"
            else "candidate_test_recall_mean"
            if row["metric"] == "test_recall"
            else "candidate_event_coverage_mean"
            if row["metric"] == "event_coverage"
            else "candidate_row_recall_mean"
            if row["metric"] == "row_recall"
            else None
        )
        baseline_key = (
            "baseline_test_f1_mean"
            if row["metric"] == "test_f1"
            else "baseline_test_recall_mean"
            if row["metric"] == "test_recall"
            else "baseline_event_coverage_mean"
            if row["metric"] == "event_coverage"
            else "baseline_row_recall_mean"
            if row["metric"] == "row_recall"
            else None
        )
        if candidate_key is None:
            continue
        lines.append(
            f"| {row['metric']} | {summary[candidate_key]:.4f} | "
            f"{summary[baseline_key]:.4f} | {row['mean_diff']:+.4f} | "
            f"{row['positive_pairs']}/{row['pairs']} |"
        )
    lines += [
        "",
        "## 攻击类型",
        "",
        "| 攻击类型 | 攻击行 | 行召回 | 事件覆盖 |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["primary_type_aggregate"]:
        lines.append(
            f"| {row['attack_type']} | {row['attack_rows']} | "
            f"{row['row_recall']:.4f} | {row['event_coverage']:.4f} |"
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
                "replication_seeds": REPLICATION_SEEDS,
                "summary": summary,
                "paired": paired_rows,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
