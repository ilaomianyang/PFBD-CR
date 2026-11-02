from __future__ import annotations

"""Stage 209: station-clustered uncertainty for the Stage-208 candidate.

This stage does not refit detectors, reselect features, or recalibrate
thresholds.  It reads the station-level evaluation detail emitted by
Stage 208 and treats ``station_id`` as the resampling cluster.  The five
attack seeds are repeated measurements within each station, so a
station bootstrap is more conservative than resampling all station-seed
rows independently.

The reported intervals describe uncertainty over the 12 DWD stations in
this frozen protocol.  They do not turn controlled injections into real
attack logs and do not justify selecting a different Stage-208 rule after
looking at test results.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
STAGE208_PREFIX = "stage208_dwd_persistent_repeat_protection"
PREFIX = "stage209_dwd_station_cluster_uncertainty"
STAGE208_JSON = TABLES / f"{STAGE208_PREFIX}.json"
STAGE208_STATION = TABLES / f"{STAGE208_PREFIX}_station_detail.csv"
PRIMARY_RULE = "valid_f1"
METRICS = ["f1", "recall", "precision", "false_alarm_rate"]
EVENT_METRICS = ["event_coverage", "row_recall"]


def ci(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def confusion_metrics(
    totals: dict[str, float],
) -> dict[str, float]:
    tp = float(totals["tp"])
    fp = float(totals["fp"])
    fn = float(totals["fn"])
    tn = float(totals["tn"])
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "f1": safe_divide(
            2.0 * precision * recall,
            precision + recall,
        ),
        "recall": recall,
        "precision": precision,
        "false_alarm_rate": safe_divide(fp, fp + tn),
    }


def event_metrics(
    events: float,
    detected_events: float,
    attack_rows: float,
    detected_rows: float,
) -> dict[str, float]:
    return {
        "event_coverage": safe_divide(detected_events, events),
        "row_recall": safe_divide(detected_rows, attack_rows),
    }


def aggregate_arm(
    frame: pd.DataFrame,
) -> dict[str, float]:
    totals = {
        name: float(frame[name].sum())
        for name in ["tp", "fp", "fn", "tn"]
    }
    totals.update(
        {
            "events": float(frame["events"].sum()),
            "detected_events": float(frame["detected_events"].sum()),
            "attack_rows": float(frame["attack_rows"].sum()),
            "detected_rows": float(
                (
                    frame["attack_rows"] - frame["fn"]
                ).sum()
            ),
        }
    )
    output = confusion_metrics(totals)
    output.update(
        event_metrics(
            totals["events"],
            totals["detected_events"],
            totals["attack_rows"],
            totals["detected_rows"],
        )
    )
    return output


def aggregate_arm_by_seed(
    frame: pd.DataFrame,
) -> dict[str, float]:
    """Average seed-level metrics, matching the Stage-208 estimand."""
    values = [
        aggregate_arm(part)
        for _, part in frame.groupby("seed", sort=True)
    ]
    if not values:
        raise ValueError("cannot aggregate an empty arm")
    return {
        name: float(np.mean([value[name] for value in values]))
        for name in METRICS + EVENT_METRICS
    }


def bootstrap_station_metrics(
    candidate_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    stations = sorted(candidate_rows["station_id"].astype(str).unique())
    candidate_by_station = {
        station: candidate_rows.loc[
            candidate_rows["station_id"].astype(str).eq(station)
        ]
        for station in stations
    }
    baseline_by_station = {
        station: baseline_rows.loc[
            baseline_rows["station_id"].astype(str).eq(station)
        ]
        for station in stations
    }
    candidate_values = {
        name: np.empty(iterations, dtype=float)
        for name in METRICS + EVENT_METRICS
    }
    baseline_values = {
        name: np.empty(iterations, dtype=float)
        for name in METRICS + EVENT_METRICS
    }
    differences = {
        name: np.empty(iterations, dtype=float)
        for name in METRICS + EVENT_METRICS
    }
    for iteration in range(iterations):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        candidate = pd.concat(
            [
                candidate_by_station[str(station)]
                for station in sampled
            ],
            ignore_index=True,
        )
        candidate_metrics = aggregate_arm_by_seed(candidate)
        baseline = pd.concat(
            [
                baseline_by_station[str(station)]
                for station in sampled
            ],
            ignore_index=True,
        )
        baseline_metrics = aggregate_arm_by_seed(baseline)
        for name in METRICS + EVENT_METRICS:
            candidate_values[name][iteration] = candidate_metrics[name]
            baseline_values[name][iteration] = baseline_metrics[name]
            differences[name][iteration] = (
                candidate_metrics[name] - baseline_metrics[name]
            )
    return candidate_values, baseline_values, differences


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage209 DWD station-cluster uncertainty"
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.iterations < 100:
        raise ValueError("iterations must be at least 100")

    stage208 = json.loads(STAGE208_JSON.read_text(encoding="utf-8"))
    detail = pd.read_csv(STAGE208_STATION)
    detail = detail.loc[
        detail["selection_rule"].eq(PRIMARY_RULE)
        & detail["max_far"].eq(float(stage208["primary_far"]))
    ].copy()
    if detail.empty:
        raise RuntimeError("Stage208 station detail is empty")
    if detail.duplicated(
        ["seed", "station_id", "arm"]
    ).any():
        raise RuntimeError("duplicate station-seed-arm rows")

    stations = sorted(detail["station_id"].astype(str).unique())
    seeds = sorted(int(value) for value in detail["seed"].unique())
    expected_rows = len(stations) * len(seeds) * 2
    if len(detail) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} station rows, found {len(detail)}"
        )

    candidate = detail.loc[detail["arm"].eq("candidate")].copy()
    baseline = detail.loc[detail["arm"].eq("stage207")].copy()
    candidate_metrics = aggregate_arm_by_seed(candidate)
    baseline_metrics = aggregate_arm_by_seed(baseline)
    if int(candidate["attack_rows"].sum()) != 5 * 624:
        raise RuntimeError(
            "station detail attack-row total does not match the "
            "five-seed Stage208 protocol"
        )
    if int(candidate["events"].sum()) != 5 * 72:
        raise RuntimeError(
            "station detail event total does not match the "
            "five-seed Stage208 protocol"
        )

    rng = np.random.default_rng(int(args.seed))
    (
        bootstrap_candidate,
        bootstrap_baseline,
        bootstrap_differences,
    ) = bootstrap_station_metrics(
        candidate,
        baseline,
        rng,
        int(args.iterations),
    )
    station_rows: list[dict[str, object]] = []
    for station_id in stations:
        part = candidate.loc[
            candidate["station_id"].astype(str).eq(station_id)
        ]
        metrics_row = aggregate_arm_by_seed(part)
        station_rows.append(
            {
                "station_id": station_id,
                "seeds": int(part["seed"].nunique()),
                "rows": int(part["rows"].sum()),
                "attack_rows": int(part["attack_rows"].sum()),
                **metrics_row,
            }
        )

    result_rows: list[dict[str, object]] = []
    for name in METRICS + EVENT_METRICS:
        result_rows.append(
            {
                "metric": name,
                "candidate_mean": float(candidate_metrics[name]),
                "candidate_ci95": ci(bootstrap_candidate[name]),
                "stage207_mean": float(baseline_metrics[name]),
                "stage207_ci95": ci(bootstrap_baseline[name]),
                "difference_mean": float(
                    candidate_metrics[name] - baseline_metrics[name]
                ),
                "difference_ci95": ci(bootstrap_differences[name]),
                "candidate_greater_probability": float(
                    np.mean(bootstrap_differences[name] > 0)
                ),
                "candidate_lower_probability": float(
                    np.mean(bootstrap_differences[name] < 0)
                ),
            }
        )

    paired_rows: list[dict[str, object]] = []
    for name in METRICS + EVENT_METRICS:
        station_diffs = []
        for station_id in stations:
            current = candidate.loc[
                candidate["station_id"].astype(str).eq(station_id)
            ]
            base = baseline.loc[
                baseline["station_id"].astype(str).eq(station_id)
            ]
            station_diffs.append(
                aggregate_arm_by_seed(current)[name]
                - aggregate_arm_by_seed(base)[name]
            )
        station_diffs_array = np.asarray(station_diffs, dtype=float)
        paired_rows.append(
            {
                "metric": name,
                "stations": int(len(station_diffs_array)),
                "positive_stations": int(
                    np.sum(station_diffs_array > 0)
                ),
                "negative_stations": int(
                    np.sum(station_diffs_array < 0)
                ),
                "zero_stations": int(
                    np.sum(station_diffs_array == 0)
                ),
                "mean_station_difference": float(
                    station_diffs_array.mean()
                ),
                "min_station_difference": float(
                    station_diffs_array.min()
                ),
                "max_station_difference": float(
                    station_diffs_array.max()
                ),
            }
        )

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段209只读取阶段208固定valid_f1规则、主FAR和test站点级"
            "明细，不重新拟合、不重新选择、不重新校准阈值。以12个DWD"
            "站点为重采样簇，五个攻击种子作为站点内重复测量；每次"
            "重采样站点并对候选与阶段207做配对比较。区间只描述当前"
            "冻结协议下跨站点的不确定性；攻击仍是透明受控注入，"
            "不是真实攻击日志，不能替代独立站点外部复现，也不能"
            "外推为所有气象网络。"
        ),
        "source": str(STAGE208_STATION),
        "selection_rule": PRIMARY_RULE,
        "primary_far": float(stage208["primary_far"]),
        "iterations": int(args.iterations),
        "bootstrap_seed": int(args.seed),
        "stations": stations,
        "seeds": seeds,
        "station_rows": int(len(detail)),
        "candidate_metrics": candidate_metrics,
        "stage207_metrics": baseline_metrics,
        "bootstrap": result_rows,
        "paired_station": paired_rows,
        "limitations": [
            "只有12个DWD站点，站点簇bootstrap的区间不能替代独立站点外部复现。",
            "五个种子是同一冻结攻击生成器的重复测量，不是五套独立真实攻击数据。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
            "阶段207修正后的负结果必须与阶段208候选并列报告。",
            "该分析不允许根据test结果回看并改选特征、权重、窗口或阈值。",
        ],
    }
    pd.DataFrame(station_rows).to_csv(
        TABLES / f"{PREFIX}_station.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(result_rows).to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(paired_rows).to_csv(
        TABLES / f"{PREFIX}_paired_station.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段209 DWD站点簇级不确定性",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 固定规则：`{PRIMARY_RULE}`",
        f"- 主FAR：`{payload['primary_far']}`",
        f"- 站点：`{len(stations)}`",
        f"- 种子：`{len(seeds)}`",
        f"- Bootstrap：`{args.iterations}` 次，种子 `{args.seed}`",
        "",
        "## 指标区间",
        "",
        "| 指标 | 候选均值 | 候选CI95% | Stage207均值 | 差值均值 |",
        "|---|---:|---|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['metric']} | {row['candidate_mean']:.4f} | "
            f"[{row['candidate_ci95'][0]:.4f}, "
            f"{row['candidate_ci95'][1]:.4f}] | "
            f"{row['stage207_mean']:.4f} | "
            f"{row['difference_mean']:+.4f} |"
        )
    lines += [
        "",
        "## 站点配对方向",
        "",
        "| 指标 | 正向站点 | 负向站点 | 零差站点 | 站点差值均值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            f"| {row['metric']} | {row['positive_stations']} | "
            f"{row['negative_stations']} | {row['zero_stations']} | "
            f"{row['mean_station_difference']:+.4f} |"
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
                "stations": len(stations),
                "seeds": len(seeds),
                "iterations": int(args.iterations),
                "candidate_f1": candidate_metrics["f1"],
                "stage207_f1": baseline_metrics["f1"],
                "f1_difference_ci95": next(
                    row["difference_ci95"]
                    for row in result_rows
                    if row["metric"] == "f1"
                ),
                "f1_candidate_greater_probability": next(
                    row["candidate_greater_probability"]
                    for row in result_rows
                    if row["metric"] == "f1"
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
