"""Stage 218: uncertainty and failure attribution for the Swiss negative result.

This stage reads the frozen Stage-217 MeteoSwiss transparent controlled
attack result and performs a deterministic, evaluation-only replay at the
single pre-registered primary FAR.  It does not fit detectors, select
weights, calibrate thresholds, choose stations, or choose attack positions.

The replay is needed for two quantities that Stage 217 did not persist:
station-level event coverage and a seed-by-station-by-attack-type event
table.  Scores and predictions are recreated from the frozen Stage-214
protocol, Stage-216 normalized data, and the Stage-217 detector settings.
The replay must reproduce the Stage-217 row-level and event-level primary
metrics before any uncertainty analysis is accepted.
"""

import hashlib
import json
import math
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
    choose_attack_blocks,
    detector_frame,
    event_metrics,
    far_threshold,
    fit_detectors,
    inject_attack,
    metrics,
    score_components,
)
from stage217_swiss_controlled_attack_evaluation import (  # noqa: E402
    fuse_scores,
)

PROTOCOL = TABLES / "stage214_swiss_protocol_preregistration.json"
NORMALIZED = TABLES / "stage216_swiss_normalized.parquet"
CORRECTION = TABLES / "stage216a_swiss_protocol_correction.json"
STAGE217 = TABLES / "stage217_swiss_controlled_attack.json"
STAGE207 = TABLES / "stage207_dwd_controlled_attack.json"
PREFIX = "stage218_swiss_negative_result_uncertainty"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH q-values while preserving input order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("BH input must be a finite one-dimensional array")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("BH input must be within [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    count = len(ranked)
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    output = np.empty(count, dtype=float)
    output[order] = adjusted
    return output.tolist()


def cluster_bootstrap_mean(
    values_by_station: dict[str, float],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the mean over stations, treating stations as clusters."""
    stations = sorted(values_by_station)
    values = np.asarray(
        [values_by_station[station] for station in stations],
        dtype=float,
    )
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("cluster bootstrap values must be finite and non-empty")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(
        0,
        values.size,
        size=(int(iterations), values.size),
    )
    boot = values[draws].mean(axis=1)
    return {
        "stations": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_mean": float(boot.mean()),
        "bootstrap_std": float(boot.std(ddof=0)),
    }


def exact_sign_flip_two_sided(differences: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value for a paired sample.

    The implementation enumerates all sign assignments.  It is intended
    for the small station-level samples used in this stage, not for large
    row-level data.
    """
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    nonzero = values[values != 0.0]
    if nonzero.size == 0:
        return 1.0
    observed = abs(float(nonzero.mean()))
    signs = np.arange(1 << nonzero.size, dtype=np.uint64)
    total = 0
    extreme = 0
    for mask in signs:
        signed = np.where(
            ((mask >> np.arange(nonzero.size, dtype=np.uint64)) & 1) == 1,
            nonzero,
            -nonzero,
        )
        statistic = abs(float(signed.mean()))
        total += 1
        if statistic >= observed - 1e-15:
            extreme += 1
    return float(extreme / total)


def main() -> int:
    protocol = load_json(PROTOCOL)
    correction = load_json(CORRECTION)
    stage217 = load_json(STAGE217)
    stage207 = load_json(STAGE207)
    retained = set(correction["retained_station_ids"])

    frame = pd.read_parquet(NORMALIZED)
    frame = frame[
        frame["station_id"].astype(str).isin(retained)
    ].copy()
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)

    if set(frame["station_id"]) != retained:
        raise RuntimeError("retained station set differs from Stage216A")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("unexpected Stage216 split labels")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("Stage216 contains duplicate station timestamps")
    if int(stage217["station_count"]) != len(retained):
        raise RuntimeError("Stage217 station count differs from Stage216A")
    if float(stage217["primary_far"]) != float(protocol["primary_far"]):
        raise RuntimeError("Stage217 primary FAR differs from protocol")

    base = detector_frame(frame)
    detectors = fit_detectors(base, protocol)
    components = score_components(base, detectors)
    fused, _ = fuse_scores(base, components)
    valid_normal = (
        base["split"].eq("valid") & base["observation_present"]
    )
    primary_far = float(protocol["primary_far"])
    threshold = far_threshold(fused, valid_normal, primary_far)
    frozen_threshold = float(stage217["thresholds"][str(primary_far)])
    if not math.isclose(threshold, frozen_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("replayed primary threshold differs from Stage217")

    seed_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    station_event_rows: list[dict[str, object]] = []
    type_event_rows: list[dict[str, object]] = []
    replay_eval_rows: list[dict[str, object]] = []
    replay_event_rows: list[dict[str, object]] = []

    for seed in protocol["seeds"]:
        seed = int(seed)
        block = choose_attack_blocks(base, protocol, seed)
        attacked = inject_attack(base, block, seed)
        attacked_components = score_components(attacked, detectors)
        attacked_fused, _ = fuse_scores(attacked, attacked_components)
        predicted = (
            attacked["observation_present"].to_numpy(dtype=bool)
            & attacked_fused.to_numpy(dtype=float).__ge__(threshold)
        )
        test = attacked["split"].eq("test").to_numpy(dtype=bool)
        y_test = attacked.loc[test, "is_attack"].to_numpy(dtype=bool)
        pred_test = predicted[test]
        score_test = attacked_fused.loc[test].to_numpy(dtype=float)
        test_frame = attacked.loc[test].reset_index(drop=True)

        row_metrics = metrics(y_test, pred_test, score_test)
        event = event_metrics(test_frame, pred_test)
        replay_eval_rows.append({"seed": seed, **row_metrics})
        replay_event_rows.append({"seed": seed, **event})

        event_frame = test_frame[test_frame["is_attack"]].copy()
        event_frame["_pred"] = pred_test[event_frame.index]
        event_frame["attack_event_id"] = (
            event_frame["station_id"].astype(str)
            + "__"
            + event_frame["attack_type"].astype(str)
        )
        grouped = (
            event_frame.groupby(
                ["station_id", "attack_type", "attack_event_id"],
                sort=True,
                as_index=False,
            )["_pred"]
            .any()
            .rename(columns={"_pred": "detected"})
        )
        for row in grouped.to_dict(orient="records"):
            station_event_rows.append(
                {
                    "seed": seed,
                    "station_id": str(row["station_id"]),
                    "attack_type": str(row["attack_type"]),
                    "attack_event_id": str(row["attack_event_id"]),
                    "detected": bool(row["detected"]),
                }
            )
        station_rows = (
            grouped.groupby("station_id", sort=True, as_index=False)["detected"]
            .mean()
            .rename(columns={"detected": "event_coverage"})
        )
        for row in station_rows.to_dict(orient="records"):
            seed_rows.append(
                {
                    "seed": seed,
                    "station_id": str(row["station_id"]),
                    "events": int(
                        grouped[
                            grouped["station_id"].eq(row["station_id"])
                        ].shape[0]
                    ),
                    "event_coverage": float(row["event_coverage"]),
                    "row_recall": float(
                        pred_test[
                            test_frame["station_id"].eq(
                                row["station_id"]
                            ).to_numpy()
                            & y_test
                        ].mean()
                    )
                    if (
                        test_frame["station_id"].eq(row["station_id"])
                        & y_test
                    ).any()
                    else float("nan"),
                }
            )
        type_station_rows = (
            grouped.groupby(
                ["station_id", "attack_type"],
                sort=True,
                as_index=False,
            )["detected"]
            .mean()
            .rename(columns={"detected": "event_coverage"})
        )
        for row in type_station_rows.to_dict(orient="records"):
            type_event_rows.append(
                {
                    "seed": seed,
                    "station_id": str(row["station_id"]),
                    "attack_type": str(row["attack_type"]),
                    "events": int(
                        grouped[
                            grouped["station_id"].eq(row["station_id"])
                            & grouped["attack_type"].eq(row["attack_type"])
                        ].shape[0]
                    ),
                    "event_coverage": float(row["event_coverage"]),
                }
            )

    replay_eval = pd.DataFrame(replay_eval_rows)
    replay_events = pd.DataFrame(replay_event_rows)
    station_seed = pd.DataFrame(seed_rows)
    station_type_seed = pd.DataFrame(type_event_rows)

    frozen_eval = pd.DataFrame(stage217["primary_evaluation"])
    frozen_events = pd.DataFrame(stage217["primary_events"])
    replay_eval = replay_eval.sort_values("seed").reset_index(drop=True)
    frozen_eval = frozen_eval.sort_values("seed").reset_index(drop=True)
    replay_events = replay_events.sort_values("seed").reset_index(drop=True)
    frozen_events = frozen_events.sort_values("seed").reset_index(drop=True)

    replay_eval_diff = float(
        np.max(
            np.abs(
                replay_eval[
                    [
                        "rows",
                        "attack_rows",
                        "precision",
                        "recall",
                        "f1",
                        "false_alarm_rate",
                        "roc_auc",
                        "average_precision",
                    ]
                ].to_numpy(dtype=float)
                - frozen_eval[
                    [
                        "rows",
                        "attack_rows",
                        "precision",
                        "recall",
                        "f1",
                        "false_alarm_rate",
                        "roc_auc",
                        "average_precision",
                    ]
                ].to_numpy(dtype=float)
            )
        )
    )
    replay_event_diff = float(
        np.max(
            np.abs(
                replay_events[
                    ["events", "detected_events", "event_coverage", "row_recall"]
                ].to_numpy(dtype=float)
                - frozen_events[
                    ["events", "detected_events", "event_coverage", "row_recall"]
                ].to_numpy(dtype=float)
            )
        )
    )
    if replay_eval_diff > 1e-12 or replay_event_diff > 1e-12:
        raise RuntimeError(
            "Stage218 replay does not reproduce the frozen Stage217 primary result"
        )

    # Station-cluster uncertainty for the primary event-coverage endpoint.
    station_seed = station_seed.sort_values(
        ["station_id", "seed"]
    ).reset_index(drop=True)
    station_event = (
        station_seed.groupby("station_id", sort=True, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            events=("events", "sum"),
            event_coverage=("event_coverage", "mean"),
            row_recall=("row_recall", "mean"),
        )
        .sort_values("station_id")
    )
    station_event_values = dict(
        zip(
            station_event["station_id"].astype(str),
            station_event["event_coverage"].astype(float),
        )
    )
    station_row_values = dict(
        zip(
            station_event["station_id"].astype(str),
            station_event["row_recall"].astype(float),
        )
    )
    event_bootstrap = cluster_bootstrap_mean(
        station_event_values,
        iterations=10000,
        seed=20260918,
    )
    row_bootstrap = cluster_bootstrap_mean(
        station_row_values,
        iterations=10000,
        seed=20260919,
    )

    seed_summary = (
        replay_events.groupby("seed", as_index=False)
        .agg(
            event_coverage=("event_coverage", "mean"),
            row_recall=("row_recall", "mean"),
        )
        .sort_values("seed")
    )
    seed_summary["event_coverage_delta"] = (
        seed_summary["event_coverage"] - float(event_bootstrap["mean"])
    )

    # Attack-type uncertainty uses station-cluster means per type.
    type_station = (
        station_type_seed.groupby(
            ["attack_type", "station_id"],
            sort=True,
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            events=("events", "sum"),
            event_coverage=("event_coverage", "mean"),
        )
        .sort_values(["attack_type", "station_id"])
    )
    type_rows: list[dict[str, object]] = []
    for attack_type, group in type_station.groupby(
        "attack_type",
        sort=True,
    ):
        values = dict(
            zip(
                group["station_id"].astype(str),
                group["event_coverage"].astype(float),
            )
        )
        bootstrap = cluster_bootstrap_mean(
            values,
            iterations=10000,
            seed=20260920 + len(type_rows),
        )
        type_rows.append(
            {
                "attack_type": str(attack_type),
                "stations": int(bootstrap["stations"]),
                "events": int(group["events"].sum()),
                "event_coverage_mean": float(bootstrap["mean"]),
                "event_coverage_std": float(bootstrap["std"]),
                "event_coverage_ci95_low": float(bootstrap["ci95_low"]),
                "event_coverage_ci95_high": float(bootstrap["ci95_high"]),
            }
        )
    type_summary = pd.DataFrame(type_rows).sort_values("attack_type")

    # Exact station sign-flip tests against zero for the primary endpoint.
    sign_flip_rows: list[dict[str, object]] = []
    for attack_type, group in type_station.groupby(
        "attack_type",
        sort=True,
    ):
        differences = group["event_coverage"].to_numpy(dtype=float)
        sign_flip_rows.append(
            {
                "attack_type": str(attack_type),
                "stations": int(len(differences)),
                "mean_difference": float(np.nanmean(differences)),
                "p_value_exact_sign_flip": exact_sign_flip_two_sided(
                    differences
                ),
            }
        )
    sign_flip = pd.DataFrame(sign_flip_rows)
    sign_flip["bh_q_value"] = benjamini_hochberg(
        sign_flip["p_value_exact_sign_flip"].astype(float).tolist()
    )
    sign_flip["bh_significant_05"] = (
        sign_flip["bh_q_value"] < 0.05
    )

    # Cross-provider descriptive contrast.  The DWD and Swiss sources have
    # different station grids, sensors, processing chains, and attack-block
    # placements, so this is not a paired provider effect estimate.
    dwd_summary: dict[str, dict[str, object]] = {}
    for metric in [
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "average_precision",
    ]:
        values = np.asarray(
            [
                row[metric]
                for row in stage207["primary_row_metrics"]
            ],
            dtype=float,
        )
        dwd_summary[metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=0)),
        }
    for metric in ["event_coverage", "row_recall"]:
        values = np.asarray(
            [
                row[metric]
                for row in stage207["primary_event_metrics"]
            ],
            dtype=float,
        )
        dwd_summary[metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=0)),
        }
    swiss_summary = {
        row["metric"]: row for row in stage217["primary_summary"]
    }
    provider_rows: list[dict[str, object]] = []
    for metric in [
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "average_precision",
        "event_coverage",
        "row_recall",
    ]:
        provider_rows.append(
            {
                "metric": metric,
                "dwd_mean": float(dwd_summary[metric]["mean"]),
                "dwd_std": float(dwd_summary[metric]["std"]),
                "swiss_mean": float(swiss_summary[metric]["mean"]),
                "swiss_std": float(swiss_summary[metric]["std"]),
                "descriptive_difference_swiss_minus_dwd": float(
                    swiss_summary[metric]["mean"]
                    - dwd_summary[metric]["mean"]
                ),
            }
        )
    provider_summary = pd.DataFrame(provider_rows)

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段218只读取阶段214冻结的MeteoSwiss协议、阶段216标准化"
            "数据、阶段216A可用性修正和阶段217冻结结果。由于阶段217"
            "未保存站点级事件覆盖明细，本阶段在冻结主FAR=0.01上做一次"
            "确定性评价重放，仅用于恢复事件覆盖的站点级不确定性；"
            "不训练模型、不重新选择特征或权重、不重新校准阈值、"
            "不选择站点或攻击位置。重放必须先逐种子复现阶段217的行级"
            "和事件级主结果，否则本阶段失败。站点是重采样簇，种子是"
            "簇内重复测量。所有攻击都是透明受控注入，不是真实攻击日志。"
        ),
        "source": stage217["source"],
        "provider": stage217["provider"],
        "protocol_sha256": sha256_file(PROTOCOL),
        "normalized_sha256": sha256_file(NORMALIZED),
        "correction_sha256": sha256_file(CORRECTION),
        "stage217_sha256": sha256_file(STAGE217),
        "stage207_sha256": sha256_file(STAGE207),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "primary_far": primary_far,
        "primary_endpoint": protocol["primary_endpoint"],
        "secondary_endpoint": protocol["secondary_endpoint"],
        "cluster_unit": protocol["cluster_unit"],
        "seeds": [int(seed) for seed in protocol["seeds"]],
        "attack_types": list(protocol["attack_types"]),
        "station_count": int(len(retained)),
        "station_ids": sorted(retained),
        "excluded_station_ids": stage217["excluded_station_ids"],
        "replay_max_abs_row_metric_difference": replay_eval_diff,
        "replay_max_abs_event_metric_difference": replay_event_diff,
        "replay_reproduces_stage217": True,
        "primary_station_event_coverage": station_event.to_dict(
            orient="records"
        ),
        "event_coverage_station_cluster_bootstrap": event_bootstrap,
        "row_recall_station_cluster_bootstrap": row_bootstrap,
        "seed_summary": seed_summary.to_dict(orient="records"),
        "attack_type_station_cluster": type_summary.to_dict(
            orient="records"
        ),
        "attack_type_sign_flip": sign_flip.to_dict(orient="records"),
        "provider_descriptive_contrast": provider_summary.to_dict(
            orient="records"
        ),
        "provider_contrast_is_paired": False,
        "provider_contrast_interpretation": (
            "DWD和MeteoSwiss的站点网格、传感器网络、处理链和攻击块"
            "位置不同；该对照不是配对比较，只用于描述性异质性说明，"
            "不能解释为机构配对效应，不能扩展为多机构泛化结论。"
        ),
        "limitations": [
            "这是对阶段217负结果的只读不确定性审计，不是新的模型选择。",
            "站点级事件覆盖由确定性重放恢复；重放必须逐种子复现阶段217主结果。",
            "MeteoSwiss没有与DWD QN_9等价的逐行质量码；空字段只表示缺失。",
            "所有攻击都是透明受控注入，不是真实攻击日志，不能解释为真实入侵率。",
            "DWD与MeteoSwiss的对照不是配对比较，不能写成多机构泛化领先。",
            "只有11个站点，站点簇bootstrap区间不能替代新的独立机构或独立传感器网络。",
        ],
    }

    station_event.to_csv(
        TABLES / f"{PREFIX}_station.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_seed.to_csv(
        TABLES / f"{PREFIX}_station_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_type_seed.to_csv(
        TABLES / f"{PREFIX}_station_attack_type_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_summary.to_csv(
        TABLES / f"{PREFIX}_attack_type.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sign_flip.to_csv(
        TABLES / f"{PREFIX}_sign_flip.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provider_summary.to_csv(
        TABLES / f"{PREFIX}_provider_contrast.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段218 MeteoSwiss负结果不确定性审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 来源：`{payload['source']}`；站点：`{payload['station_count']}`；主FAR：`{payload['primary_far']}`",
        f"- 主终点：`{payload['primary_endpoint']}`；聚类单位：`{payload['cluster_unit']}`",
        f"- 重放最大行级差：`{replay_eval_diff:.3e}`；最大事件级差：`{replay_event_diff:.3e}`",
        "",
        "## 事件覆盖站点簇不确定性",
        "",
        f"- 站点均值：`{event_bootstrap['mean']:.6f}`",
        f"- 站点簇bootstrap 95%区间：`[{event_bootstrap['ci95_low']:.6f}, {event_bootstrap['ci95_high']:.6f}]`",
        f"- 站点标准差：`{event_bootstrap['std']:.6f}`",
        "",
        "## 攻击类型事件覆盖",
        "",
        "| 攻击类型 | 站点 | 事件数 | 均值 | 站点簇95%区间 | BH q值 |",
        "|---|---:|---:|---:|---|---:|",
    ]
    sign_flip_by_type = {
        row["attack_type"]: row
        for row in sign_flip.to_dict(orient="records")
    }
    for row in type_summary.to_dict(orient="records"):
        sign = sign_flip_by_type[row["attack_type"]]
        lines.append(
            f"| {row['attack_type']} | {int(row['stations'])} | "
            f"{int(row['events'])} | {row['event_coverage_mean']:.6f} | "
            f"[{row['event_coverage_ci95_low']:.6f}, "
            f"{row['event_coverage_ci95_high']:.6f}] | "
            f"{sign['bh_q_value']:.6f} |"
        )
    lines += [
        "",
        "## 跨机构描述性对照",
        "",
        "该对照不是配对比较。DWD与MeteoSwiss的站点网格、传感器网络、处理链和攻击块位置不同，只能用于描述性异质性说明，不能写成多机构泛化领先。",
        "",
        "| 指标 | DWD均值 | MeteoSwiss均值 | 描述性差值 |",
        "|---|---:|---:|---:|",
    ]
    for row in provider_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['metric']} | {row['dwd_mean']:.6f} | "
            f"{row['swiss_mean']:.6f} | "
            f"{row['descriptive_difference_swiss_minus_dwd']:+.6f} |"
        )
    lines += [
        "",
        "所有攻击都是透明受控注入，不是真实攻击日志；阶段218不重新选择模型、阈值、站点或攻击位置。",
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
                "event_coverage_mean": event_bootstrap["mean"],
                "event_coverage_ci95": [
                    event_bootstrap["ci95_low"],
                    event_bootstrap["ci95_high"],
                ],
                "replay_max_abs_row_metric_difference": replay_eval_diff,
                "replay_max_abs_event_metric_difference": replay_event_diff,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
