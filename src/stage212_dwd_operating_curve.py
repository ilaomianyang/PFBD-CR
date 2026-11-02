from __future__ import annotations

"""Stage 212: frozen DWD operating curve for the repeat-protection candidate.

Stage 208 and Stage 210 already contain evaluations at the frozen FAR grid
``0.005/0.01/0.02/0.05``.  This stage only reads those frozen results,
consolidates them into one auditable operating curve, and adds
station-clustered paired intervals for every point.  It does not refit,
train, select, or recalibrate anything.

The primary operating point remains the pre-registered ``0.01`` FAR.  The
other grid points are sensitivity evidence only; looking at the test curve
is not permission to choose a different primary FAR after the fact.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage212_dwd_operating_curve"
PRIMARY_RULE = "valid_f1"
SOURCES = {
    "stage208": {
        "prefix": "stage208_dwd_persistent_repeat_protection",
        "label": "original_five_seeds",
    },
    "stage210": {
        "prefix": "stage210_dwd_seed_randomized_replication",
        "label": "replication_five_seeds",
    },
}

from stage209_dwd_station_cluster_uncertainty import (  # noqa: E402
    EVENT_METRICS,
    METRICS,
    aggregate_arm_by_seed,
    bootstrap_station_metrics,
    ci,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage212 DWD frozen operating curve"
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.iterations < 100:
        raise ValueError("iterations must be at least 100")

    curve_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    source_audit: list[dict[str, object]] = []
    rng = np.random.default_rng(int(args.seed))
    protocol = json.loads(
        (
            TABLES / "stage204_dwd_protocol_preregistration.json"
        ).read_text(encoding="utf-8")
    )
    far_grid = [float(value) for value in protocol["far_grid"]]
    if float(protocol["primary_far"]) != 0.01:
        raise RuntimeError("unexpected primary FAR in Stage204 protocol")

    for source_name, config in SOURCES.items():
        prefix = config["prefix"]
        evaluation = pd.read_csv(TABLES / f"{prefix}_evaluation.csv")
        events = pd.read_csv(TABLES / f"{prefix}_events.csv")
        station_detail = pd.read_csv(
            TABLES / f"{prefix}_station_detail.csv"
        )
        payload = json.loads(
            (TABLES / f"{prefix}.json").read_text(encoding="utf-8")
        )
        if float(payload["primary_far"]) != 0.01:
            raise RuntimeError("unexpected primary FAR")
        observed_fars = sorted(
            float(value)
            for value in evaluation["max_far"].unique()
        )
        if observed_fars != sorted(far_grid):
            raise RuntimeError(
                f"{source_name} does not cover the frozen FAR grid"
            )
        source_audit.append(
            {
                "source": source_name,
                "label": config["label"],
                "far_grid": far_grid,
                "primary_far": float(payload["primary_far"]),
                "evaluation_rows": int(len(evaluation)),
                "event_rows": int(len(events)),
                "station_detail_rows": int(len(station_detail)),
            }
        )

        for far in far_grid:
            eval_part = evaluation.loc[
                evaluation["selection_rule"].eq(PRIMARY_RULE)
                & evaluation["max_far"].eq(far)
            ].copy()
            event_part = events.loc[
                events["selection_rule"].eq(PRIMARY_RULE)
                & events["max_far"].eq(far)
            ].copy()
            station_part = station_detail.loc[
                station_detail["selection_rule"].eq(PRIMARY_RULE)
                & station_detail["max_far"].eq(far)
            ].copy()
            if eval_part.empty or event_part.empty or station_part.empty:
                raise RuntimeError(
                    f"missing operating-curve rows for {source_name}/{far}"
                )
            candidate = station_part.loc[
                station_part["arm"].eq("candidate")
            ].copy()
            baseline = station_part.loc[
                station_part["arm"].eq("stage207")
            ].copy()
            candidate_metrics = aggregate_arm_by_seed(candidate)
            baseline_metrics = aggregate_arm_by_seed(baseline)
            if not np.isclose(
                candidate_metrics["f1"],
                float(eval_part["test_f1"].mean()),
            ):
                raise RuntimeError(
                    f"station detail F1 does not match evaluation for "
                    f"{source_name}/{far}"
                )
            if not np.isclose(
                candidate_metrics["event_coverage"],
                float(event_part["event_coverage"].mean()),
            ):
                raise RuntimeError(
                    f"station detail event coverage does not match events "
                    f"for {source_name}/{far}"
                )
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
            seed_f1_diff = (
                eval_part["test_f1"].to_numpy(dtype=float)
                - eval_part["baseline_test_f1"].to_numpy(dtype=float)
            )
            seed_event_diff = (
                event_part["event_coverage"].to_numpy(dtype=float)
                - event_part["baseline_event_coverage"].to_numpy(
                    dtype=float
                )
            )
            for metric in METRICS + EVENT_METRICS:
                curve_rows.append(
                    {
                        "source": source_name,
                        "source_label": config["label"],
                        "selection_rule": PRIMARY_RULE,
                        "max_far": far,
                        "is_primary": bool(
                            np.isclose(far, float(payload["primary_far"]))
                        ),
                        "metric": metric,
                        "candidate_mean": float(candidate_metrics[metric]),
                        "candidate_ci95": ci(bootstrap_candidate[metric]),
                        "stage207_mean": float(baseline_metrics[metric]),
                        "stage207_ci95": ci(bootstrap_baseline[metric]),
                        "difference_mean": float(
                            candidate_metrics[metric]
                            - baseline_metrics[metric]
                        ),
                        "difference_ci95": ci(
                            bootstrap_differences[metric]
                        ),
                        "candidate_greater_probability": float(
                            np.mean(bootstrap_differences[metric] > 0)
                        ),
                        "candidate_lower_probability": float(
                            np.mean(bootstrap_differences[metric] < 0)
                        ),
                    }
                )
            paired_rows.append(
                {
                    "source": source_name,
                    "source_label": config["label"],
                    "max_far": far,
                    "is_primary": bool(
                        np.isclose(far, float(payload["primary_far"]))
                    ),
                    "seed_count": int(eval_part["seed"].nunique()),
                    "f1_up_count": int((seed_f1_diff > 0).sum()),
                    "f1_down_count": int((seed_f1_diff < 0).sum()),
                    "f1_tie_count": int((seed_f1_diff == 0).sum()),
                    "event_coverage_up_count": int(
                        (seed_event_diff > 0).sum()
                    ),
                    "event_coverage_down_count": int(
                        (seed_event_diff < 0).sum()
                    ),
                    "event_coverage_tie_count": int(
                        (seed_event_diff == 0).sum()
                    ),
                }
            )

    curve = pd.DataFrame(curve_rows)
    paired = pd.DataFrame(paired_rows)
    primary_f1 = curve.loc[
        curve["metric"].eq("f1")
        & curve["is_primary"]
    ]
    low_far_f1 = curve.loc[
        curve["metric"].eq("f1")
        & curve["max_far"].eq(0.005)
    ]
    if len(primary_f1) != len(SOURCES):
        raise RuntimeError("primary FAR must have one F1 row per source")
    if not (primary_f1["difference_ci95"].map(lambda x: x[0] > 0)).all():
        raise RuntimeError("primary F1 interval must remain positive")

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段212只读取阶段208和阶段210已冻结的FAR网格结果，并读取"
            "对应站点级明细；以12个DWD站点为重采样簇做配对bootstrap。"
            "不重新拟合、不训练、不重新选择、不重新校准阈值。"
            "主运行点仍是预先冻结的FAR=0.01，"
            "0.005/0.02/0.05仅作敏感性运行曲线，不根据test选择FAR。"
            "只有12个DWD站点，不能替代独立站点外部复现。"
            "攻击仍是透明受控注入，不是真实攻击日志。"
        ),
        "primary_rule": PRIMARY_RULE,
        "primary_far": 0.01,
        "iterations": int(args.iterations),
        "bootstrap_seed": int(args.seed),
        "sources": source_audit,
        "curve_rows": int(len(curve)),
        "paired_rows": int(len(paired)),
        "curve": curve.to_dict(orient="records"),
        "seed_direction": paired.to_dict(orient="records"),
        "limitations": [
            "FAR=0.005/0.02/0.05只是冻结网格上的敏感性分析，不是test选出的最优运行点。",
            "Stage210只是新攻击块种子的复现，不是新的机构或站点数据。",
            "只有12个DWD站点，站点簇区间不能替代独立站点外部复现。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
        ],
    }
    curve.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        TABLES / f"{PREFIX}_seed_direction.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段212 DWD冻结运行曲线",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 固定规则：`{PRIMARY_RULE}`",
        "- 主运行点：`FAR=0.01`（预先冻结）",
        f"- Bootstrap：`{args.iterations}` 次，种子 `{args.seed}`",
        "",
        "## F1运行曲线",
        "",
        "| 数据来源 | FAR | 候选F1 | Stage207 F1 | F1差值 | 差值CI95 | 种子正向 |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for _, row in curve.loc[curve["metric"].eq("f1")].iterrows():
        direction = paired.loc[
            paired["source"].eq(row["source"])
            & paired["max_far"].eq(row["max_far"])
        ].iloc[0]
        lines.append(
            f"| {row['source_label']} | {row['max_far']:.3f} | "
            f"{row['candidate_mean']:.4f} | {row['stage207_mean']:.4f} | "
            f"{row['difference_mean']:+.4f} | "
            f"[{row['difference_ci95'][0]:+.4f}, "
            f"{row['difference_ci95'][1]:+.4f}] | "
            f"{direction['f1_up_count']}/"
            f"{direction['seed_count']} |"
        )
    lines += [
        "",
        "## 事件覆盖运行曲线",
        "",
        "| 数据来源 | FAR | 候选事件覆盖 | Stage207事件覆盖 | 差值 | 差值CI95 | 种子正向 |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for _, row in curve.loc[
        curve["metric"].eq("event_coverage")
    ].iterrows():
        direction = paired.loc[
            paired["source"].eq(row["source"])
            & paired["max_far"].eq(row["max_far"])
        ].iloc[0]
        lines.append(
            f"| {row['source_label']} | {row['max_far']:.3f} | "
            f"{row['candidate_mean']:.4f} | {row['stage207_mean']:.4f} | "
            f"{row['difference_mean']:+.4f} | "
            f"[{row['difference_ci95'][0]:+.4f}, "
            f"{row['difference_ci95'][1]:+.4f}] | "
            f"{direction['event_coverage_up_count']}/"
            f"{direction['seed_count']} |"
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
                "iterations": int(args.iterations),
                "primary_f1": primary_f1[
                    [
                        "source_label",
                        "candidate_mean",
                        "stage207_mean",
                        "difference_mean",
                        "difference_ci95",
                    ]
                ].to_dict(orient="records"),
                "low_far_f1": low_far_f1[
                    [
                        "source_label",
                        "candidate_mean",
                        "stage207_mean",
                        "difference_mean",
                        "difference_ci95",
                    ]
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
