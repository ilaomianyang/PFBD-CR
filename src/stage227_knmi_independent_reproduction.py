from __future__ import annotations

"""Stage 227: independent reproduction on the KNMI third source.

The three unsupervised components are fitted again on KNMI train-normal
rows only.  Valid-normal rows calibrate the frozen FAR-constrained
threshold; 2025 test rows are used for evaluation only.  DWD weights are
not transferred.  The attack generator and parameters are the frozen
Stage-224 protocol, and all attacks are transparent controlled
injections, not real attack logs.
"""

import hashlib
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
    event_metrics,
    far_threshold,
    fit_detectors,
    inject_attack,
    metrics,
    score_components,
    sha256_file,
)
from stage217_swiss_controlled_attack_evaluation import (  # noqa: E402
    fuse_scores,
)

PROTOCOL = TABLES / "stage224_knmi_protocol_preregistration.json"
NORMALIZED = TABLES / "stage226_knmi_normalized.parquet"
CORRECTION = TABLES / "stage226a_knmi_missingness_audit.json"
PREFIX = "stage227_knmi_controlled_attack"


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

    base = detector_frame(frame)
    detectors = fit_detectors(base, protocol)
    components = score_components(base, detectors)
    fused, weights = fuse_scores(base, components)
    valid_normal = (
        base["split"].eq("valid") & base["observation_present"]
    )
    thresholds = {
        str(far): far_threshold(fused, valid_normal, float(far))
        for far in protocol["far_grid"]
    }
    primary_far = float(protocol["primary_far"])

    evaluation_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    attack_manifest: list[dict[str, object]] = []

    for seed in protocol["seeds"]:
        seed = int(seed)
        block = choose_attack_blocks(base, protocol, seed)
        attacked = inject_attack(base, block, seed)
        attacked_components = score_components(attacked, detectors)
        attacked_fused, _ = fuse_scores(attacked, attacked_components)
        for far in protocol["far_grid"]:
            threshold = thresholds[str(far)]
            predicted = (
                attacked["observation_present"].to_numpy(dtype=bool)
                & attacked_fused.to_numpy(dtype=float).__ge__(threshold)
            )
            test = attacked["split"].eq("test").to_numpy(dtype=bool)
            y_test = attacked.loc[test, "is_attack"].to_numpy(dtype=bool)
            pred_test = predicted[test]
            score_test = attacked_fused.loc[test].to_numpy(dtype=float)
            evaluation_rows.append(
                {
                    "seed": seed,
                    "max_far": float(far),
                    "threshold": float(threshold),
                    **metrics(y_test, pred_test, score_test),
                }
            )
            test_frame = attacked.loc[test].reset_index(drop=True)
            event_rows.append(
                {
                    "seed": seed,
                    "max_far": float(far),
                    **event_metrics(test_frame, pred_test),
                }
            )
            if abs(float(far) - primary_far) < 1e-12:
                for attack_type in protocol["attack_types"]:
                    mask = test_frame["attack_type"].eq(attack_type).to_numpy()
                    event_coverage = float("nan")
                    if mask.any():
                        grouped = (
                            pd.DataFrame(
                                {
                                    "station_id": test_frame.loc[
                                        mask, "station_id"
                                    ].to_numpy(),
                                    "detected": pred_test[mask],
                                }
                            )
                            .groupby("station_id", sort=True)["detected"]
                            .any()
                        )
                        event_coverage = float(grouped.mean())
                    type_rows.append(
                        {
                            "seed": seed,
                            "attack_type": attack_type,
                            "attack_rows": int(mask.sum()),
                            "detected_rows": int(pred_test[mask].sum()),
                            "row_recall": (
                                float(pred_test[mask].mean())
                                if mask.any()
                                else float("nan")
                            ),
                            "event_coverage": event_coverage,
                            "mean_score_attack": (
                                float(np.nanmean(score_test[mask]))
                                if mask.any()
                                else float("nan")
                            ),
                        }
                    )
                for station_id in sorted(retained):
                    mask = test_frame["station_id"].eq(station_id).to_numpy()
                    y_station = y_test[mask]
                    pred_station = pred_test[mask]
                    station_rows.append(
                        {
                            "seed": seed,
                            "station_id": station_id,
                            "rows": int(mask.sum()),
                            "attack_rows": int(y_station.sum()),
                            "normal_rows": int((~y_station).sum()),
                            "tp": int((y_station & pred_station).sum()),
                            "fp": int((~y_station & pred_station).sum()),
                            "fn": int((y_station & ~pred_station).sum()),
                            "tn": int((~y_station & ~pred_station).sum()),
                            "f1": metrics(
                                y_station,
                                pred_station,
                                score_test[mask],
                            )["f1"],
                            "row_recall": (
                                float(pred_station[y_station].mean())
                                if y_station.any()
                                else float("nan")
                            ),
                            "false_alarm_rate": (
                                float(pred_station[~y_station].mean())
                                if (~y_station).any()
                                else float("nan")
                            ),
                        }
                    )
        attack_manifest.append(
            {
                "seed": seed,
                "attack_rows": int(block.shape[0]),
                "attack_types": sorted(block["attack_type"].unique().tolist()),
                "stations": sorted(block["station_id"].unique().tolist()),
            }
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "status": "completed",
                    "attack_rows": int(block.shape[0]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    evaluation = pd.DataFrame(evaluation_rows)
    events = pd.DataFrame(event_rows)
    type_detail = pd.DataFrame(type_rows)
    station_detail = pd.DataFrame(station_rows)
    type_aggregate = (
        type_detail.groupby("attack_type", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            attack_rows=("attack_rows", "sum"),
            detected_rows=("detected_rows", "sum"),
            row_recall=("row_recall", "mean"),
            event_coverage=("event_coverage", "mean"),
            mean_score_attack=("mean_score_attack", "mean"),
        )
        .sort_values("attack_type")
    )
    primary_eval = evaluation[
        evaluation["max_far"].eq(primary_far)
    ].copy()
    primary_events = events[events["max_far"].eq(primary_far)].copy()
    paired_rows: list[dict[str, object]] = []
    for metric in [
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "roc_auc",
        "average_precision",
    ]:
        values = primary_eval[metric].to_numpy(dtype=float)
        paired_rows.append(
            {
                "metric": metric,
                "seeds": int(len(values)),
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=0)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )
    for metric in ["event_coverage", "row_recall"]:
        values = primary_events[metric].to_numpy(dtype=float)
        paired_rows.append(
            {
                "metric": metric,
                "seeds": int(len(values)),
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=0)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )
    paired = pd.DataFrame(paired_rows)

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段227在阶段226标准化的KNMI第三来源数据上执行独立复现："
            "三个无监督分量只在2021-2023 train正常段重新拟合；"
            "2024 valid正常段按冻结FAR网格校准阈值；2025 test只评价。"
            "使用阶段224冻结的攻击类型、长度、块数、种子、FAR、"
            "主终点和聚类单位；使用阶段226A记录的6个主变量可用站点；"
            "不迁移DWD权重，不根据test选择特征、权重、阈值、站点或"
            "攻击位置。所有攻击都是透明受控注入，不是真实攻击日志。"
        ),
        "source": "KNMI_hourly",
        "provider": protocol["provider"],
        "dataset": protocol["dataset"],
        "normalized_sha256": sha256_file(NORMALIZED),
        "protocol_sha256": sha256_file(PROTOCOL),
        "correction_sha256": sha256_file(CORRECTION),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "detector_code_sha256": {
            path.name: sha256_file(path)
            for path in sorted((SRC / "detectors").glob("*.py"))
        },
        "attack_parameters": ATTACK_PARAMETERS,
        "components": COMPONENTS,
        "component_weights": weights,
        "threshold_calibration": {
            "source": "valid_normal_observations",
            "method": "empirical_quantile",
            "quantile": "1 - target_far",
            "scores": "equal_weight_standardized_components",
        },
        "thresholds": thresholds,
        "primary_far": primary_far,
        "primary_endpoint": protocol["primary_endpoint"],
        "secondary_endpoint": protocol["secondary_endpoint"],
        "cluster_unit": protocol["cluster_unit"],
        "multiplicity": protocol["multiplicity"],
        "seeds": protocol["seeds"],
        "attack_types": protocol["attack_types"],
        "attack_length": protocol["attack_length"],
        "attack_blocks": protocol["attack_blocks"],
        "station_count": int(len(retained)),
        "station_ids": sorted(retained),
        "excluded_station_ids": correction["excluded_station_ids"],
        "evaluation_rows": int(len(evaluation)),
        "event_rows": int(len(events)),
        "type_detail_rows": int(len(type_detail)),
        "station_detail_rows": int(len(station_detail)),
        "primary_evaluation": primary_eval.to_dict(orient="records"),
        "primary_events": primary_events.to_dict(orient="records"),
        "primary_summary": paired.to_dict(orient="records"),
        "type_aggregate": type_aggregate.to_dict(orient="records"),
        "attack_manifest": attack_manifest,
        "limitations": [
            "这是第三个跨机构来源的独立复现，不是多机构泛化结论。",
            "KNMI没有与DWD QN_9等价的逐行数值质量码；空字段只表示缺失，不是攻击标签。",
            "所有攻击都是透明受控注入，不是真实攻击日志，不能解释为真实入侵率。",
            "6个冻结站主变量全空，已在阶段226A排除；不得把它们作为性能站点。",
            "阶段227不迁移DWD权重；组件在本来源train正常段重新拟合。",
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
    paired.to_csv(
        TABLES / f"{PREFIX}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段227 KNMI第三来源透明受控攻击评价",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 来源：`{payload['source']}`；站点：`{payload['station_count']}`（排除 `{payload['excluded_station_ids']}`）",
        f"- 主FAR：`{primary_far}`；主终点：`{protocol['primary_endpoint']}`；聚类单位：`{protocol['cluster_unit']}`",
        f"- 分量：`{', '.join(COMPONENTS)}`；权重：`{weights}`",
        f"- 种子：`{protocol['seeds']}`；攻击类型：`{protocol['attack_types']}`",
        "",
        "## 主FAR行级与事件级结果",
        "",
        "| 种子 | 行级F1 | 行级Recall | 行级FAR | 事件覆盖 | 行召回 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_eval.to_dict(orient="records"):
        event = primary_events[
            primary_events["seed"].eq(row["seed"])
        ].iloc[0]
        lines.append(
            f"| {int(row['seed'])} | {row['f1']:.4f} | "
            f"{row['recall']:.4f} | {row['false_alarm_rate']:.4f} | "
            f"{event['event_coverage']:.4f} | "
            f"{event['row_recall']:.4f} |"
        )
    lines += [
        "",
        "## 攻击类型平均",
        "",
        "| 攻击类型 | 种子 | 攻击行 | 行召回 | 事件覆盖 | 平均攻击分数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in type_aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['attack_type']} | {int(row['seeds'])} | "
            f"{int(row['attack_rows'])} | {row['row_recall']:.4f} | "
            f"{row['event_coverage']:.4f} | "
            f"{row['mean_score_attack']:.4f} |"
        )
    lines += [
        "",
        "本阶段不迁移DWD权重，组件在本来源train正常段重新拟合；",
        "valid只校准，test只评价。所有攻击都是透明受控注入，不是真实攻击日志。",
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
                "evaluation_rows": payload["evaluation_rows"],
                "event_rows": payload["event_rows"],
                "primary_far": primary_far,
                "primary_event_coverage_mean": float(
                    primary_events["event_coverage"].mean()
                ),
                "primary_row_recall_mean": float(
                    primary_events["row_recall"].mean()
                ),
                "primary_f1_mean": float(primary_eval["f1"].mean()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
