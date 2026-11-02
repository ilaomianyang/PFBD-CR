from __future__ import annotations

"""Stage 211: station-clustered uncertainty for the Stage-210 replication.

This stage reads the frozen Stage-210 station-level detail and performs the
same station-cluster paired bootstrap used by Stage 209.  It does not
refit, reselect, or recalibrate anything.  The five new attack seeds are
repeated measurements within each station; the 12 stations are the
resampling clusters.

The interval describes cross-station uncertainty in this controlled DWD
replication only.  It is not evidence from real attack logs and does not
replace an independent station-network replication.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage211_dwd_replication_station_uncertainty"
STAGE210_PREFIX = "stage210_dwd_seed_randomized_replication"
STAGE210_JSON = TABLES / f"{STAGE210_PREFIX}.json"
STAGE210_STATION = TABLES / f"{STAGE210_PREFIX}_station_detail.csv"
PRIMARY_RULE = "valid_f1"

from stage209_dwd_station_cluster_uncertainty import (  # noqa: E402
    EVENT_METRICS,
    METRICS,
    aggregate_arm_by_seed,
    bootstrap_station_metrics,
    ci,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage211 DWD replication station-cluster uncertainty"
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.iterations < 100:
        raise ValueError("iterations must be at least 100")

    stage210 = json.loads(STAGE210_JSON.read_text(encoding="utf-8"))
    detail = pd.read_csv(STAGE210_STATION)
    detail = detail.loc[
        detail["selection_rule"].eq(PRIMARY_RULE)
        & detail["max_far"].eq(float(stage210["primary_far"]))
    ].copy()
    if detail.empty:
        raise RuntimeError("Stage210 station detail is empty")
    if detail.duplicated(["seed", "station_id", "arm"]).any():
        raise RuntimeError("duplicate station-seed-arm rows")

    stations = sorted(detail["station_id"].astype(str).unique())
    seeds = sorted(int(value) for value in detail["seed"].unique())
    if len(stations) != 12 or len(seeds) != 5:
        raise RuntimeError(
            "Stage211 expects 12 stations and 5 replication seeds"
        )
    if len(detail) != len(stations) * len(seeds) * 2:
        raise RuntimeError("unexpected station-detail row count")

    candidate = detail.loc[detail["arm"].eq("candidate")].copy()
    baseline = detail.loc[detail["arm"].eq("stage207")].copy()
    if int(candidate["attack_rows"].sum()) != 5 * 624:
        raise RuntimeError("unexpected candidate attack-row total")
    if int(candidate["events"].sum()) != 5 * 72:
        raise RuntimeError("unexpected candidate event total")

    candidate_metrics = aggregate_arm_by_seed(candidate)
    baseline_metrics = aggregate_arm_by_seed(baseline)
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
        values = np.asarray(station_diffs, dtype=float)
        paired_rows.append(
            {
                "metric": name,
                "stations": int(len(values)),
                "positive_stations": int(np.sum(values > 0)),
                "negative_stations": int(np.sum(values < 0)),
                "zero_stations": int(np.sum(values == 0)),
                "mean_station_difference": float(values.mean()),
                "min_station_difference": float(values.min()),
                "max_station_difference": float(values.max()),
            }
        )

    station_rows: list[dict[str, object]] = []
    for station_id in stations:
        part = candidate.loc[
            candidate["station_id"].astype(str).eq(station_id)
        ]
        station_rows.append(
            {
                "station_id": station_id,
                "seeds": int(part["seed"].nunique()),
                "rows": int(part["rows"].sum()),
                "attack_rows": int(part["attack_rows"].sum()),
                **aggregate_arm_by_seed(part),
            }
        )

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段211只读取阶段210固定valid_f1规则、主FAR和test站点级"
            "明细，不重新拟合、不重新选择、不重新校准阈值。以12个DWD"
            "站点为重采样簇，五个新攻击种子作为站点内重复测量，对"
            "候选与阶段207做配对bootstrap。攻击仍是透明受控注入，"
            "不是真实攻击日志，不能替代独立站点外部复现，也不能"
            "外推为所有气象网络。"
        ),
        "source": str(STAGE210_STATION),
        "selection_rule": PRIMARY_RULE,
        "primary_far": float(stage210["primary_far"]),
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
            "新种子只改变攻击块位置，不是新的机构、站点或真实攻击数据。",
            "只有12个DWD站点，不能替代独立站点外部复现。",
            "五个新种子是同一冻结攻击生成器的重复测量。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
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
        "# 阶段211 DWD复现站点簇级不确定性",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 固定规则：`{PRIMARY_RULE}`",
        f"- 主FAR：`{payload['primary_far']}`",
        f"- 站点：`{len(stations)}`；种子：`{len(seeds)}`",
        f"- Bootstrap：`{args.iterations}` 次，种子 `{args.seed}`",
        "",
        "## 指标区间",
        "",
        "| 指标 | 候选均值 | 候选CI95% | Stage207均值 | 差值均值 | 差值CI95% |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['metric']} | {row['candidate_mean']:.4f} | "
            f"[{row['candidate_ci95'][0]:.4f}, "
            f"{row['candidate_ci95'][1]:.4f}] | "
            f"{row['stage207_mean']:.4f} | "
            f"{row['difference_mean']:+.4f} | "
            f"[{row['difference_ci95'][0]:+.4f}, "
            f"{row['difference_ci95'][1]:+.4f}] |"
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
                "event_difference_ci95": next(
                    row["difference_ci95"]
                    for row in result_rows
                    if row["metric"] == "event_coverage"
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
