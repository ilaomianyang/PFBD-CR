from __future__ import annotations

"""Stage 207: first DWD transparent controlled-attack evaluation.

The source, stations, years, split, attack types, attack lengths, block
count, seeds, FAR grid, primary FAR, endpoints, clustering unit, and
multiplicity rule are all taken from the Stage-204 frozen protocol.

Only the train-normal portion is used to fit the unsupervised detectors.
Only valid-normal rows are used to set the FAR-constrained threshold.
The 2025 test rows are used for evaluation only.  Attacks are transparent
controlled injections, not real attack logs.
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
WORKSPACE_ROOT = ROOT.parents[1]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors import (  # noqa: E402
    HampelMADDetector,
    KalmanInnovationDetector,
    SpatialConsistencyDetector,
)

PROTOCOL = TABLES / "stage204_dwd_protocol_preregistration.json"
NORMALIZED = TABLES / "stage206_dwd_normalized.parquet"
PREFIX = "stage207_dwd_controlled_attack"

DETECTORS = [
    HampelMADDetector(),
    KalmanInnovationDetector(),
    SpatialConsistencyDetector(),
]
COMPONENTS = [detector.name for detector in DETECTORS]

ATTACK_PARAMETERS = {
    "spike": {
        "kind": "additive",
        "magnitude": 8.0,
        "unit": "degC",
    },
    "offset": {
        "kind": "additive",
        "magnitude": 4.0,
        "unit": "degC",
    },
    "drift": {
        "kind": "linear_additive",
        "start": 0.0,
        "end": 6.0,
        "unit": "degC",
    },
    "time_shift": {
        "kind": "lag_copy",
        "lag_rows": 4,
        "direction": "past",
        "source_series": "clean_stage206_temperature_c",
    },
    "stuck": {
        "kind": "constant_hold",
        "source": "clean_value_at_first_attack_row",
    },
    "gaussian_noise": {
        "kind": "additive_noise",
        "sigma": 2.0,
        "unit": "degC",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def detector_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["observed_value"] = pd.to_numeric(
        out["temperature_c"], errors="coerce"
    )
    out["station_key"] = out["station_id"].astype(str)
    return out


def fit_detectors(
    frame: pd.DataFrame, protocol: dict
) -> dict[str, object]:
    train = frame["split"].eq("train")
    normal = train & frame["observation_present"]
    fitted: dict[str, object] = {}
    for detector in DETECTORS:
        detector.fit(frame.loc[train], normal.loc[train])
        fitted[detector.name] = detector
    return fitted


def score_components(
    frame: pd.DataFrame, detectors: dict[str, object]
) -> pd.DataFrame:
    scores = pd.DataFrame(index=frame.index)
    for name, detector in detectors.items():
        scores[name] = detector.score(frame)
    return scores


def robust_standardize(
    values: pd.Series, normal_mask: pd.Series
) -> pd.Series:
    reference = pd.to_numeric(
        values.loc[normal_mask], errors="coerce"
    ).dropna()
    if reference.empty:
        return pd.Series(0.0, index=values.index, dtype=float)
    median = float(reference.median())
    mad = float((reference - median).abs().median())
    scale = max(1.4826 * mad, 1e-6)
    return (pd.to_numeric(values, errors="coerce") - median) / scale


def fuse_scores(
    frame: pd.DataFrame, components: pd.DataFrame
) -> tuple[pd.Series, dict[str, float]]:
    train_normal = frame["split"].eq("train") & frame[
        "observation_present"
    ]
    standardized = pd.DataFrame(index=frame.index)
    for name in COMPONENTS:
        standardized[name] = robust_standardize(
            components[name], train_normal
        )
    weights = {
        "Hampel MAD": 1.0 / 3.0,
        "Kalman Innovation": 1.0 / 3.0,
        "Spatial Consistency": 1.0 / 3.0,
    }
    fused = sum(
        standardized[name] * weight
        for name, weight in weights.items()
    )
    fused = fused.where(frame["observation_present"], 0.0)
    return fused.astype(float), weights


def far_threshold(
    scores: pd.Series, normal_mask: pd.Series, target_far: float
) -> float:
    reference = pd.to_numeric(
        scores.loc[normal_mask], errors="coerce"
    ).dropna()
    if reference.empty:
        raise RuntimeError("no valid-normal scores for threshold calibration")
    return float(np.quantile(reference, 1.0 - float(target_far)))


def choose_attack_blocks(
    frame: pd.DataFrame,
    protocol: dict,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    attack_types = list(protocol["attack_types"])
    lengths = protocol["attack_length"]
    block_count = int(protocol["attack_blocks"])
    if block_count != len(attack_types):
        raise ValueError(
            "Stage204 expects one non-overlapping block per attack type"
        )
    rows: list[dict[str, object]] = []
    for station_id, group in frame.groupby("station_id", sort=True):
        ordered = (
            group.sort_values("timestamp_utc")
            .reset_index()
            .rename(columns={"index": "global_row_position"})
        )
        test_positions = np.flatnonzero(
            ordered["split"].eq("test").to_numpy()
            & ordered["observation_present"].to_numpy(dtype=bool)
        )
        if len(test_positions) < sum(int(lengths[t]) for t in attack_types):
            raise RuntimeError(
                f"not enough test rows for station {station_id}"
            )
        for attack_type in attack_types:
            length = int(lengths[attack_type])
            if len(test_positions) < length:
                raise RuntimeError(
                    f"not enough test rows for {station_id}/{attack_type}"
                )
            start_pos = int(rng.integers(0, len(test_positions) - length + 1))
            positions = test_positions[start_pos : start_pos + length]
            if positions.size != length:
                raise RuntimeError("attack block does not match frozen length")
            for local_position in positions:
                global_position = int(
                    ordered.loc[local_position, "global_row_position"]
                )
                source_local_position = int(local_position) - 4
                if attack_type == "time_shift":
                    if source_local_position < 0:
                        raise RuntimeError(
                            "time_shift requires four clean historical rows"
                        )
                    source_global_position = int(
                        ordered.loc[
                            source_local_position, "global_row_position"
                        ]
                    )
                else:
                    source_global_position = -1
                rows.append(
                    {
                        "station_id": station_id,
                        "seed": int(seed),
                        "attack_type": attack_type,
                        "row_position": global_position,
                        "source_row_position": source_global_position,
                        "timestamp_utc": ordered.loc[
                            local_position, "timestamp_utc"
                        ],
                    }
                )
            test_positions = np.delete(
                test_positions,
                np.arange(start_pos, start_pos + length),
            )
    return pd.DataFrame(rows)


def inject_attack(
    frame: pd.DataFrame,
    block: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    attacked = frame.copy()
    attacked["is_attack"] = False
    attacked["attack_type"] = "normal"
    attacked["attack_seed"] = int(seed)
    rng = np.random.default_rng(int(seed) + 1000003)
    clean_values = pd.to_numeric(
        frame["temperature_c"], errors="coerce"
    ).to_numpy(dtype=float).copy()
    attacked_values = clean_values.copy()
    for attack_type, type_part in block.groupby("attack_type", sort=True):
        if attack_type not in ATTACK_PARAMETERS:
            raise ValueError(f"unexpected attack type: {attack_type}")
        for _, part in type_part.groupby("station_id", sort=True):
            positions = part["row_position"].to_numpy(dtype=int)
            if attack_type == "spike":
                attacked_values[positions] += 8.0
            elif attack_type == "offset":
                attacked_values[positions] += 4.0
            elif attack_type == "drift":
                attacked_values[positions] += np.linspace(
                    0.0, 6.0, len(positions)
                )
            elif attack_type == "time_shift":
                source_positions = part[
                    "source_row_position"
                ].to_numpy(dtype=int)
                source_values = clean_values[source_positions]
                if not np.isfinite(source_values).all():
                    source_values = (
                        pd.Series(clean_values)
                        .ffill()
                        .bfill()
                        .to_numpy(dtype=float)[source_positions]
                    )
                attacked_values[positions] = source_values
            elif attack_type == "stuck":
                first_value = clean_values[positions[0]]
                if not np.isfinite(first_value):
                    first_value = float(
                        pd.Series(clean_values)
                        .ffill()
                        .bfill()
                        .iloc[positions[0]]
                    )
                attacked_values[positions] = first_value
            elif attack_type == "gaussian_noise":
                attacked_values[positions] += rng.normal(
                    0.0, 2.0, size=len(positions)
                )
            attacked.loc[positions, "is_attack"] = True
            attacked.loc[positions, "attack_type"] = attack_type
    attacked["temperature_c"] = attacked_values
    attacked["observed_value"] = pd.to_numeric(
        attacked["temperature_c"], errors="coerce"
    )
    attacked["station_key"] = attacked["station_id"].astype(str)
    attacked["attack_event_id"] = np.where(
        attacked["is_attack"],
        attacked["station_id"].astype(str)
        + "__"
        + attacked["attack_type"].astype(str)
        + "__"
        + attacked["attack_seed"].astype(str),
        "",
    )
    return attacked


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    score = np.asarray(score, dtype=float)
    out = {
        "rows": int(len(y_true)),
        "attack_rows": int(y_true.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_alarm_rate": float(
            y_pred[~y_true].mean() if (~y_true).any() else np.nan
        ),
    }
    finite = np.isfinite(score)
    if finite.any() and y_true[finite].any() and (~y_true[finite]).any():
        out["roc_auc"] = float(roc_auc_score(y_true[finite], score[finite]))
        out["average_precision"] = float(
            average_precision_score(y_true[finite], score[finite])
        )
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
    return out


def event_metrics(
    frame: pd.DataFrame, predicted: np.ndarray
) -> dict[str, float | int]:
    event_rows = frame[frame["is_attack"]].copy()
    if event_rows.empty:
        return {
            "events": 0,
            "detected_events": 0,
            "event_coverage": float("nan"),
            "row_recall": float("nan"),
        }
    event_rows["_pred"] = np.asarray(predicted, dtype=bool)[
        event_rows.index
    ]
    grouped = event_rows.groupby("attack_event_id", sort=True)["_pred"].any()
    return {
        "events": int(len(grouped)),
        "detected_events": int(grouped.sum()),
        "event_coverage": float(grouped.mean()),
        "row_recall": float(event_rows["_pred"].mean()),
    }


def main() -> int:
    protocol = load_protocol()
    frame = pd.read_parquet(NORMALIZED)
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"], kind="stable"
    ).reset_index(drop=True)
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("unexpected Stage206 split labels")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("Stage206 contains duplicate station timestamps")

    base = detector_frame(frame)
    detectors = fit_detectors(base, protocol)
    components = score_components(base, detectors)
    fused, weights = fuse_scores(base, components)
    valid_normal = (
        base["split"].eq("valid")
        & base["observation_present"]
    )
    thresholds = {
        str(far): far_threshold(fused, valid_normal, float(far))
        for far in protocol["far_grid"]
    }
    primary_far = float(protocol["primary_far"])
    primary_threshold = thresholds[str(primary_far)]

    evaluation_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    attack_manifest: list[dict[str, object]] = []
    attacked_by_seed: dict[int, pd.DataFrame] = {}
    fused_by_seed: dict[int, pd.Series] = {}
    for seed in protocol["seeds"]:
        block = choose_attack_blocks(base, protocol, int(seed))
        attacked = inject_attack(base, block, int(seed))
        attacked_components = score_components(attacked, detectors)
        attacked_fused, _ = fuse_scores(attacked, attacked_components)
        attacked_by_seed[int(seed)] = attacked
        fused_by_seed[int(seed)] = attacked_fused
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
            row = {
                "seed": int(seed),
                "max_far": float(far),
                "threshold": float(threshold),
                **metrics(y_test, pred_test, score_test),
            }
            evaluation_rows.append(row)
            event_rows.append(
                {
                    "seed": int(seed),
                    "max_far": float(far),
                    **event_metrics(
                        attacked.loc[test].reset_index(drop=True),
                        pred_test,
                    ),
                }
            )
        attack_manifest.append(
            {
                "seed": int(seed),
                "attack_rows": int(block.shape[0]),
                "attack_types": sorted(block["attack_type"].unique().tolist()),
                "stations": sorted(block["station_id"].unique().tolist()),
            }
        )

    evaluation = pd.DataFrame(evaluation_rows)
    events = pd.DataFrame(event_rows)
    primary_rows = evaluation[
        evaluation["max_far"].eq(primary_far)
    ]
    primary_events = events[events["max_far"].eq(primary_far)]
    type_rows: list[dict[str, object]] = []
    for seed in protocol["seeds"]:
        attacked = attacked_by_seed[int(seed)]
        attacked_fused = fused_by_seed[int(seed)]
        threshold = thresholds[str(primary_far)]
        predicted = (
            attacked["observation_present"].to_numpy(dtype=bool)
            & attacked_fused.to_numpy(dtype=float).__ge__(threshold)
        )
        test = attacked["split"].eq("test").to_numpy(dtype=bool)
        test_frame = attacked.loc[test].reset_index(drop=True)
        test_pred = predicted[test]
        test_score = attacked_fused.loc[test].to_numpy(dtype=float)
        for attack_type in protocol["attack_types"]:
            mask = test_frame["attack_type"].eq(attack_type).to_numpy()
            event_coverage = float("nan")
            if mask.any():
                station_events = (
                    pd.DataFrame(
                        {
                            "station_id": test_frame.loc[
                                mask, "station_id"
                            ].to_numpy(),
                            "detected": test_pred[mask],
                        }
                    )
                    .groupby("station_id", sort=True)["detected"]
                    .any()
                )
                event_coverage = float(station_events.mean())
            type_rows.append(
                {
                    "seed": int(seed),
                    "attack_type": attack_type,
                    "attack_rows": int(mask.sum()),
                    "detected_rows": int(test_pred[mask].sum()),
                    "row_recall": float(test_pred[mask].mean())
                    if mask.any()
                    else float("nan"),
                    "event_coverage": float(
                        event_coverage
                    ),
                    "mean_score_attack": float(
                        np.nanmean(test_score[mask])
                    )
                    if mask.any()
                    else float("nan"),
                }
            )
    type_frame = pd.DataFrame(type_rows)
    type_aggregate = (
        type_frame.groupby("attack_type", as_index=False)
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
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段207严格复用阶段204冻结协议，在阶段206标准化的DWD数据上"
            "执行透明受控攻击：Hampel MAD、Kalman Innovation和Spatial "
            "Consistency只在train正常段拟合；valid正常段按冻结FAR网格"
            "校准阈值；test只评价。攻击按站点和种子在2025 test年份"
            "非重叠注入，类型、长度、块数和种子来自阶段204；不根据test"
            "选择特征、权重、阈值、站点或攻击位置。攻击参数在运行前"
            "固定并写入attack_parameters；time_shift只从阶段206干净序列"
            "的同一站点历史行取值。攻击不是真实攻击日志。"
        ),
        "source": "DWD_CDC",
        "provider": protocol["provider"],
        "dataset_version": protocol["dataset_version"],
        "normalized_sha256": sha256_file(NORMALIZED),
        "protocol_sha256": sha256_file(PROTOCOL),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "detector_code_sha256": {
            path.name: sha256_file(path)
            for path in sorted((SRC / "detectors").glob("*.py"))
        },
        "attack_parameters": ATTACK_PARAMETERS,
        "threshold_calibration": {
            "source": "valid_normal_observations",
            "method": "empirical_quantile",
            "quantile": "1 - target_far",
            "scores": "equal_weight_standardized_components",
        },
        "components": COMPONENTS,
        "component_weights": weights,
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
        "primary_row_metrics": primary_rows.to_dict(orient="records"),
        "primary_event_metrics": primary_events.to_dict(orient="records"),
        "attack_manifest": attack_manifest,
        "evaluation_rows": len(evaluation),
        "event_rows": len(events),
        "type_rows": len(type_frame),
        "type_aggregate": type_aggregate.to_dict(orient="records"),
        "limitations": [
            "攻击是透明受控注入，不是真实攻击日志。",
            "修正前的站内索引误用会把攻击行写到其它站点并虚高事件覆盖；当前产物是修正后的可追溯基线。",
            "本阶段只使用DWD冻结站点和年份，不根据test选择任何组件。",
            "DWD质量码不是攻击标签；低质量行没有作为正样本。",
            "这是第一份DWD外部受控性能证据，不代表所有攻击类型都稳定占优。",
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
    type_frame.to_csv(
        TABLES / f"{PREFIX}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_aggregate.to_csv(
        TABLES / f"{PREFIX}_type_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段207 DWD透明受控攻击第一轮评价",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 来源：`{payload['source']}`；主FAR：`{primary_far}`",
        f"- 分量：`{', '.join(COMPONENTS)}`；权重：`{weights}`",
        f"- 阈值：`{thresholds}`",
        f"- 种子：`{protocol['seeds']}`；攻击类型：`{protocol['attack_types']}`",
        f"- 攻击参数：`{ATTACK_PARAMETERS}`",
        "",
        "## 主FAR行级与事件级结果",
        "",
        "| 种子 | 行级F1 | 行级Recall | 行级FAR | 事件覆盖 | 行召回 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows.to_dict(orient="records"):
        event = primary_events[
            primary_events["seed"].eq(row["seed"])
        ].iloc[0]
        lines.append(
            f"| {row['seed']} | {row['f1']:.4f} | "
            f"{row['recall']:.4f} | {row['false_alarm_rate']:.4f} | "
            f"{event['event_coverage']:.4f} | "
            f"{event['row_recall']:.4f} |"
        )
    lines += [
        "",
        "## 攻击类型平均",
        "",
        "| 攻击类型 | 种子 | 攻击行 | 检出率 | 事件覆盖 | 平均攻击分数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in type_aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['attack_type']} | {row['seeds']} | "
            f"{row['attack_rows']} | {row['row_recall']:.4f} | "
            f"{row['event_coverage']:.4f} | "
            f"{row['mean_score_attack']:.4f} |"
        )
    lines += [
        "",
        "本阶段所有攻击都是透明受控注入，不是真实攻击日志；test只评价，",
        "不用于选择特征、权重、阈值、站点或攻击位置。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "evaluation_rows": len(evaluation),
                "event_rows": len(events),
                "type_rows": len(type_frame),
                "primary_far": primary_far,
                "primary_event_coverage_mean": float(
                    primary_events["event_coverage"].mean()
                ),
                "primary_row_recall_mean": float(
                    primary_events["row_recall"].mean()
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
