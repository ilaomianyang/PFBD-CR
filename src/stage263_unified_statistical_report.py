from __future__ import annotations

"""Stage 263: unified statistical reporting for baseline audits.

This stage only consumes existing evaluation, station-detail, and
type-detail artifacts.  It does not train a model, recalibrate a threshold,
or touch any test prediction.  The report separates pooled, station-equal,
and provider-equal summaries and retains the exploratory status of
Stage256 and Stage262.
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
PREFIX = "stage263_unified_statistical_report"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)

SOURCES = {
    "stage248_fair_baseline": {
        "prefix": "stage248_cross_provider_fair_baseline_audit",
        "layer": "confirmatory_comparison",
        "exploratory_only": False,
        "not_part_of_frozen_main_results": False,
    },
    "stage256_lightweight": {
        "prefix": "stage256_lightweight_strong_baselines",
        "layer": "exploratory_supplement",
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
    },
    "stage262_security_domain": {
        "prefix": "stage262_security_domain_baselines",
        "layer": "exploratory_supplement",
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
    },
}

PRIMARY_METHODS = ["trust_valid_f1", "trust_nonstuck_event_macro"]
ALL_METRICS = [
    "f1",
    "recall",
    "precision",
    "false_alarm_rate",
    "event_coverage",
    "row_recall",
]
POOLED_ONLY_METRICS = ["average_precision"]
BOOTSTRAP_ITERATIONS = 10000


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _aggregate_station(station: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        station.groupby(
            ["stage", "source", "seed", "method", "station_id"],
            as_index=False,
        )
        .agg(
            rows=("rows", "sum"),
            attack_rows=("attack_rows", "sum"),
            normal_rows=("normal_rows", "sum"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tn=("tn", "sum"),
            events=("events", "sum"),
            detected_events=("detected_events", "sum"),
        )
    )
    grouped["f1"] = np.divide(
        2.0 * grouped["tp"],
        2.0 * grouped["tp"] + grouped["fp"] + grouped["fn"],
        out=np.zeros(len(grouped), dtype=float),
        where=(
            2.0 * grouped["tp"] + grouped["fp"] + grouped["fn"]
        ).to_numpy()
        > 0,
    )
    grouped["recall"] = np.divide(
        grouped["tp"],
        grouped["tp"] + grouped["fn"],
        out=np.zeros(len(grouped), dtype=float),
        where=(grouped["tp"] + grouped["fn"]).to_numpy() > 0,
    )
    grouped["precision"] = np.divide(
        grouped["tp"],
        grouped["tp"] + grouped["fp"],
        out=np.zeros(len(grouped), dtype=float),
        where=(grouped["tp"] + grouped["fp"]).to_numpy() > 0,
    )
    grouped["false_alarm_rate"] = np.divide(
        grouped["fp"],
        grouped["fp"] + grouped["tn"],
        out=np.zeros(len(grouped), dtype=float),
        where=(grouped["fp"] + grouped["tn"]).to_numpy() > 0,
    )
    grouped["event_coverage"] = np.divide(
        grouped["detected_events"],
        grouped["events"],
        out=np.zeros(len(grouped), dtype=float),
        where=grouped["events"].to_numpy() > 0,
    )
    grouped["row_recall"] = grouped["recall"]
    return grouped


def _load_layer(
    stage: str,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluation = _load_csv(TABLES / f"{prefix}_evaluation.csv")
    station = _load_csv(TABLES / f"{prefix}_station_detail.csv")
    type_detail = _load_csv(TABLES / f"{prefix}_type_detail.csv")
    for frame in (evaluation, station, type_detail):
        frame.insert(0, "stage", stage)
    return evaluation, station, type_detail


def _pooled_metrics(
    station_metrics: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    pooled = (
        station_metrics.groupby(
            ["stage", "source", "seed", "method"],
            as_index=False,
        )
        .agg(
            rows=("rows", "sum"),
            attack_rows=("attack_rows", "sum"),
            normal_rows=("normal_rows", "sum"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tn=("tn", "sum"),
            events=("events", "sum"),
            detected_events=("detected_events", "sum"),
        )
    )
    pooled["precision"] = np.divide(
        pooled["tp"],
        pooled["tp"] + pooled["fp"],
        out=np.zeros(len(pooled), dtype=float),
        where=(pooled["tp"] + pooled["fp"]).to_numpy() > 0,
    )
    pooled["recall"] = np.divide(
        pooled["tp"],
        pooled["tp"] + pooled["fn"],
        out=np.zeros(len(pooled), dtype=float),
        where=(pooled["tp"] + pooled["fn"]).to_numpy() > 0,
    )
    pooled["f1"] = np.divide(
        2.0 * pooled["tp"],
        2.0 * pooled["tp"] + pooled["fp"] + pooled["fn"],
        out=np.zeros(len(pooled), dtype=float),
        where=(
            2.0 * pooled["tp"] + pooled["fp"] + pooled["fn"]
        ).to_numpy()
        > 0,
    )
    pooled["false_alarm_rate"] = np.divide(
        pooled["fp"],
        pooled["fp"] + pooled["tn"],
        out=np.zeros(len(pooled), dtype=float),
        where=(pooled["fp"] + pooled["tn"]).to_numpy() > 0,
    )
    pooled["event_coverage"] = np.divide(
        pooled["detected_events"],
        pooled["events"],
        out=np.zeros(len(pooled), dtype=float),
        where=pooled["events"].to_numpy() > 0,
    )
    pooled["row_recall"] = pooled["recall"]
    pooled = pooled.merge(
        evaluation[
            [
                "stage",
                "source",
                "seed",
                "method",
                "average_precision",
            ]
        ],
        on=["stage", "source", "seed", "method"],
        how="left",
        validate="one_to_one",
    )
    return pooled


def _station_equal_metrics(
    station_metrics: pd.DataFrame,
    pooled: pd.DataFrame,
) -> pd.DataFrame:
    event_counts = (
        station_metrics.groupby(
            ["stage", "source", "seed", "method"],
            as_index=False,
        )
        .agg(
            stations=("station_id", "nunique"),
            events=("events", "sum"),
            detected_events=("detected_events", "sum"),
        )
    )
    equal = (
        station_metrics.groupby(
            ["stage", "source", "seed", "method"],
            as_index=False,
        )[ALL_METRICS]
        .mean()
    )
    equal = equal.rename(
        columns={metric: f"station_equal_{metric}" for metric in ALL_METRICS}
    )
    pooled_columns = [
        "stage",
        "source",
        "seed",
        "method",
        *ALL_METRICS,
        *POOLED_ONLY_METRICS,
    ]
    output = pooled[pooled_columns].merge(
        equal,
        on=["stage", "source", "seed", "method"],
        how="inner",
        validate="one_to_one",
    )
    output = output.merge(
        event_counts,
        on=["stage", "source", "seed", "method"],
        how="inner",
        validate="one_to_one",
    )
    return output


def _provider_equal_metrics(
    station_equal: pd.DataFrame,
    pooled: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (stage, method), group in station_equal.groupby(
        ["stage", "method"], sort=True
    ):
        provider_means = (
            group.groupby("source", as_index=False)[
                [f"station_equal_{metric}" for metric in ALL_METRICS]
            ]
            .mean()
        )
        row: dict[str, object] = {
            "stage": stage,
            "method": method,
            "provider_count": int(provider_means["source"].nunique()),
        }
        for metric in ALL_METRICS:
            values = provider_means[
                f"station_equal_{metric}"
            ].to_numpy(dtype=float)
            row[f"provider_equal_{metric}"] = float(np.mean(values))
        pooled_ap = (
            pooled[
                pooled["stage"].eq(stage)
                & pooled["method"].eq(method)
            ]
            .groupby("source", as_index=False)["average_precision"]
            .mean()["average_precision"]
            .to_numpy(dtype=float)
        )
        row["provider_equal_average_precision"] = float(
            np.mean(pooled_ap)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _station_direction(
    station_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage, group in station_metrics.groupby("stage", sort=True):
        stage_comparisons = comparisons[comparisons["stage"].eq(stage)]
        pairs = (
            stage_comparisons[["candidate", "baseline"]]
            .drop_duplicates()
            .sort_values(["candidate", "baseline"])
        )
        for candidate, reference in pairs.itertuples(index=False):
            current = group[group["method"].eq(candidate)]
            baseline = group[group["method"].eq(reference)]
            if current.empty or baseline.empty:
                continue
            aligned = current.merge(
                baseline,
                on=["source", "seed", "station_id"],
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            aligned["f1_difference"] = (
                aligned["f1_candidate"] - aligned["f1_baseline"]
            )
            aligned["event_coverage_difference"] = (
                aligned["event_coverage_candidate"]
                - aligned["event_coverage_baseline"]
            )
            for row in aligned.to_dict("records"):
                rows.append(
                    {
                        "stage": stage,
                        "candidate": candidate,
                        "reference": reference,
                        "source": str(row["source"]),
                        "seed": int(row["seed"]),
                        "station_id": str(row["station_id"]),
                        "f1_difference": float(row["f1_difference"]),
                        "event_coverage_difference": float(
                            row["event_coverage_difference"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _mde_table(
    station_equal: pd.DataFrame,
    station_metrics: pd.DataFrame,
    alpha: float = 0.05,
    power: float = 0.80,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143
    for (stage, method), group in station_equal.groupby(
        ["stage", "method"], sort=True
    ):
        stage_station = station_metrics[
            station_metrics["stage"].eq(stage)
            & station_metrics["method"].eq(method)
        ]
        providers = int(group["source"].nunique())
        stations = int(
            stage_station[["source", "station_id"]]
            .drop_duplicates()
            .shape[0]
        )
        seeds = int(group["seed"].nunique())
        for metric in ALL_METRICS:
            provider_values = (
                group.groupby("source", as_index=False)[
                    f"station_equal_{metric}"
                ]
                .mean()[f"station_equal_{metric}"]
                .to_numpy(dtype=float)
            )
            values = provider_values[np.isfinite(provider_values)]
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            mde = (
                (z_alpha + z_power) * std / np.sqrt(values.size)
                if values.size > 0
                else float("nan")
            )
            rows.append(
                {
                    "stage": stage,
                    "method": method,
                    "metric": metric,
                    "provider_count": providers,
                    "station_count": stations,
                    "seed_count": seeds,
                    "independent_units": int(values.size),
                    "independent_unit_definition": (
                        "机构等权均值；独立单元为机构"
                    ),
                    "alpha": float(alpha),
                    "power": float(power),
                    "standard_deviation": std,
                    "minimum_detectable_effect": float(mde),
                }
            )
    return pd.DataFrame(rows)


def _attack_type_table(type_detail: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        type_detail.groupby(
            ["stage", "method", "attack_type"],
            as_index=False,
        )
        .agg(
            source_count=("source", "nunique"),
            seeds=("seed", "nunique"),
            attack_rows=("attack_rows", "sum"),
            detected_rows=("detected_rows", "sum"),
            events=("events", "sum"),
            row_recall=("row_recall", "mean"),
            event_coverage=("event_coverage", "mean"),
        )
        .sort_values(["stage", "method", "attack_type"])
    )
    aggregate["pooled_row_recall"] = np.divide(
        aggregate["detected_rows"],
        aggregate["attack_rows"],
        out=np.zeros(len(aggregate), dtype=float),
        where=aggregate["attack_rows"].to_numpy() > 0,
    )
    return aggregate


def _leave_one_union() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for stage, info in SOURCES.items():
        path = TABLES / f"{info['prefix']}_leave_one_provider_out.csv"
        frame = _load_csv(path)
        frame.insert(0, "stage", stage)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _event_definition_audit(
    type_detail: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (stage, source, method), group in type_detail.groupby(
        ["stage", "source", "method"], sort=True
    ):
        event_rows: list[dict[str, object]] = []
        for row in group.to_dict("records"):
            events = int(row["events"])
            attack_rows = int(row["attack_rows"])
            if events <= 0:
                continue
            if attack_rows % events != 0:
                raise RuntimeError(
                    "attack rows are not divisible by event count for "
                    f"{stage}/{source}/{method}/{row['attack_type']}"
                )
            event_rows.append(
                {
                    "attack_type": str(row["attack_type"]),
                    "events": events,
                    "attack_rows": attack_rows,
                    "rows_per_event": attack_rows / events,
                }
            )
        if not event_rows:
            continue
        event_frame = pd.DataFrame(event_rows)
        rows.append(
            {
                "stage": stage,
                "source": source,
                "method": method,
                "attack_types": int(event_frame["attack_type"].nunique()),
                "events": int(event_frame["events"].sum()),
                "attack_rows": int(event_frame["attack_rows"].sum()),
                "mean_rows_per_event": float(
                    event_frame["rows_per_event"].mean()
                ),
                "min_rows_per_event": float(
                    event_frame["rows_per_event"].min()
                ),
                "max_rows_per_event": float(
                    event_frame["rows_per_event"].max()
                ),
                "single_row_event_types": int(
                    event_frame["rows_per_event"].le(1.0).sum()
                ),
                "event_denominator_definition": (
                    "每个(station_id, attack_type, split, seed)攻击块为一个事件"
                ),
                "event_coverage_definition": (
                    "事件内任一行被预测为攻击则该事件计为检出"
                ),
            }
        )
    return pd.DataFrame(rows)


def _low_far_point(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (stage, source, method), group in evaluation.groupby(
        ["stage", "source", "method"], sort=True
    ):
        for metric, label in [
            ("false_alarm_rate", "low_far_operating_point"),
            ("f1", "f1_optimal_operating_point"),
        ]:
            best = group.sort_values(
                [metric, "f1"],
                ascending=[True, False],
            ).iloc[0]
            rows.append(
                {
                    "stage": stage,
                    "source": source,
                    "method": method,
                    "operating_point": label,
                    "seed": int(best["seed"]),
                    "threshold": float(best["threshold"]),
                    "f1": float(best["f1"]),
                    "false_alarm_rate": float(best["false_alarm_rate"]),
                    "event_coverage": float(best["event_coverage"]),
                    "row_recall": float(best["row_recall"]),
                }
            )
    return pd.DataFrame(rows)


def _comparison_union() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for stage, info in SOURCES.items():
        path = TABLES / f"{info['prefix']}_comparisons.csv"
        frame = _load_csv(path)
        frame.insert(0, "stage", stage)
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    output["multiple_comparison_family"] = (
        output["stage"] + "::" + output["candidate"] + "::" + output["baseline"]
    )
    return output


def _bootstrap_structure() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage, info in SOURCES.items():
        evaluation = _load_csv(
            TABLES / f"{info['prefix']}_evaluation.csv"
        )
        station = _load_csv(
            TABLES / f"{info['prefix']}_station_detail.csv"
        )
        type_detail = _load_csv(
            TABLES / f"{info['prefix']}_type_detail.csv"
        )
        rows.append(
            {
                "stage": stage,
                "layer": info["layer"],
                "exploratory_only": info["exploratory_only"],
                "not_part_of_frozen_main_results": info[
                    "not_part_of_frozen_main_results"
                ],
                "provider_count": int(evaluation["source"].nunique()),
                "station_count": int(
                    station[["source", "station_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "seed_count": int(evaluation["seed"].nunique()),
                "evaluation_rows": int(len(evaluation)),
                "station_rows": int(len(station)),
                "attack_type_rows": int(len(type_detail)),
                "bootstrap_hierarchy": (
                    "provider -> station -> seed；provider等权后再"
                    "计算机构间均值"
                ),
                "primary_endpoint": "event_coverage",
                "secondary_endpoints": (
                    "f1;precision;recall;row_recall;"
                    "false_alarm_rate;average_precision"
                ),
                "multiple_comparison": (
                    "Benjamini-Hochberg，按stage/candidate/baseline族"
                ),
            }
        )
    return pd.DataFrame(rows)


def _pooled_cross_source(
    pooled: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (stage, method), group in pooled.groupby(
        ["stage", "method"], sort=True
    ):
        for metric in ALL_METRICS:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "stage": stage,
                    "method": method,
                    "metric": metric,
                    "seed_rows": int(values.size),
                    "mean": float(np.mean(values)) if values.size else 0.0,
                    "std": (
                        float(np.std(values, ddof=1))
                        if values.size > 1
                        else 0.0
                    ),
                    "min": float(np.min(values)) if values.size else 0.0,
                    "max": float(np.max(values)) if values.size else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    evaluation_frames: list[pd.DataFrame] = []
    station_frames: list[pd.DataFrame] = []
    type_frames: list[pd.DataFrame] = []
    for stage, info in SOURCES.items():
        evaluation, station, type_detail = _load_layer(
            stage, info["prefix"]
        )
        evaluation_frames.append(evaluation)
        station_frames.append(station)
        type_frames.append(type_detail)
    evaluation = pd.concat(evaluation_frames, ignore_index=True)
    station = pd.concat(station_frames, ignore_index=True)
    type_detail = pd.concat(type_frames, ignore_index=True)
    station_metrics = _aggregate_station(station)
    pooled = _pooled_metrics(station_metrics, evaluation)
    station_equal = _station_equal_metrics(station_metrics, pooled)
    provider_equal = _provider_equal_metrics(station_equal, pooled)
    comparisons = _comparison_union()
    station_direction = _station_direction(station_metrics, comparisons)
    mde = _mde_table(station_equal, station_metrics)
    attack_type = _attack_type_table(type_detail)
    event_audit = _event_definition_audit(type_detail)
    low_far = _low_far_point(evaluation)
    leave_one = _leave_one_union()
    bootstrap = _bootstrap_structure()
    pooled_cross_source = _pooled_cross_source(pooled)
    pooled.to_csv(
        TABLES / f"{PREFIX}_pooled.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_equal.to_csv(
        TABLES / f"{PREFIX}_station_equal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provider_equal.to_csv(
        TABLES / f"{PREFIX}_provider_equal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_direction.to_csv(
        TABLES / f"{PREFIX}_station_direction.csv",
        index=False,
        encoding="utf-8-sig",
    )
    attack_type.to_csv(
        TABLES / f"{PREFIX}_attack_type.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mde.to_csv(
        TABLES / f"{PREFIX}_power_mde.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_audit.to_csv(
        TABLES / f"{PREFIX}_event_definition_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    low_far.to_csv(
        TABLES / f"{PREFIX}_low_far_and_f1_optimum.csv",
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
    bootstrap.to_csv(
        TABLES / f"{PREFIX}_bootstrap_structure.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pooled_cross_source.to_csv(
        TABLES / f"{PREFIX}_pooled_cross_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_UNIFIED_STATISTICAL_REPORT",
        "protocol": (
            "阶段263只读取Stage248、Stage256和Stage262的已落盘评价产物，"
            "不训练、不重新选择阈值、不修改test预测。pooled、站点等权、"
            "机构等权分别报告；事件覆盖和行召回并列；Stage256/262保持"
            "探索性标签；所有受控攻击仍不是真实攻击日志。"
        ),
        "source_count": int(len(SOURCES)),
        "stage_count": int(len(SOURCES)),
        "provider_count": int(
            station[["stage", "source"]].drop_duplicates().shape[0]
        ),
        "station_count": int(
            station[["stage", "source", "station_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "seed_count": int(evaluation["seed"].nunique()),
        "method_count": int(evaluation["method"].nunique()),
        "pooled_rows": int(len(pooled)),
        "station_equal_rows": int(len(station_equal)),
        "provider_equal_rows": int(len(provider_equal)),
        "station_direction_rows": int(len(station_direction)),
        "attack_type_rows": int(len(attack_type)),
        "power_mde_rows": int(len(mde)),
        "event_definition_audit_rows": int(len(event_audit)),
        "low_far_and_f1_optimum_rows": int(len(low_far)),
        "comparison_rows": int(len(comparisons)),
        "leave_one_provider_rows": int(len(leave_one)),
        "bootstrap_structure_rows": int(len(bootstrap)),
        "layers": {
            stage: {
                "layer": info["layer"],
                "exploratory_only": info["exploratory_only"],
                "not_part_of_frozen_main_results": info[
                    "not_part_of_frozen_main_results"
                ],
            }
            for stage, info in SOURCES.items()
        },
        "metrics": ALL_METRICS,
        "bootstrap_hierarchy": (
            "机构 -> 站点 -> 种子；bootstrap按机构等权后再汇总"
        ),
        "multiple_comparison": (
            "Benjamini-Hochberg；每个stage/candidate/baseline为一个族"
        ),
        "event_definition": (
            "每个(station_id, attack_type, split, seed)攻击块为一个事件；"
            "事件覆盖分母为攻击事件数，事件内任一行命中即视为检出。"
        ),
        "low_far_definition": (
            "低误报点为每个stage/source/method在种子内FAR最低的运行点；"
            "F1最优点为F1最高的运行点；两者并列报告。"
        ),
        "limitations": [
            "所有攻击标签均为透明受控注入，不是真实攻击日志。",
            "Stage256和Stage262是探索性补充，不能替换Stage246/248。",
            "事件覆盖容易受短事件或单行事件结构影响，必须与行召回并列。",
            "机构数量有限，MDE和机构等权结果不能支持机构无关声称。",
            "如果某基线在某个终点占优，该结果必须原样保留。",
        ],
        "input_sha256": {
            stage: sha256_file(
                TABLES / f"{info['prefix']}_evaluation.csv"
            )
            for stage, info in SOURCES.items()
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段263 统一统计报告",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 阶段数：`{payload['stage_count']}`；机构记录数：`{payload['provider_count']}`；站点记录数：`{payload['station_count']}`",
        f"- 种子数：`{payload['seed_count']}`；方法数：`{payload['method_count']}`",
        "",
        "## 统计边界",
        "",
        f"- pooled 行数：`{payload['pooled_rows']}`",
        f"- 站点等权行数：`{payload['station_equal_rows']}`",
        f"- 机构等权行数：`{payload['provider_equal_rows']}`",
        f"- 站点方向行数：`{payload['station_direction_rows']}`",
        f"- 攻击类型行数：`{payload['attack_type_rows']}`",
        f"- 功效/MDE 行数：`{payload['power_mde_rows']}`",
        f"- 事件定义审计行数：`{payload['event_definition_audit_rows']}`",
        f"- 低误报/F1最优点行数：`{payload['low_far_and_f1_optimum_rows']}`",
        f"- 比较行数：`{payload['comparison_rows']}`",
        f"- 留一机构行数：`{payload['leave_one_provider_rows']}`",
        "",
        "## 事件定义",
        "",
        f"- {payload['event_definition']}",
        "",
        "## Bootstrap 层级",
        "",
        f"- {payload['bootstrap_hierarchy']}",
        f"- {payload['multiple_comparison']}",
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "stage_count": payload["stage_count"],
                "pooled_rows": payload["pooled_rows"],
                "station_equal_rows": payload["station_equal_rows"],
                "provider_equal_rows": payload["provider_equal_rows"],
                "station_direction_rows": payload[
                    "station_direction_rows"
                ],
                "attack_type_rows": payload["attack_type_rows"],
                "power_mde_rows": payload["power_mde_rows"],
                "event_definition_audit_rows": payload[
                    "event_definition_audit_rows"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
