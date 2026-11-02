from __future__ import annotations

"""Stage 280: read-only stability audit for the non-stuck claim.

The audit consumes only the frozen Stage244 and Stage245 summaries and
type-detail tables. It does not train, select thresholds, or inspect new
test predictions. It separates three questions that are often conflated:

1. Are the required non-stuck attack types represented with enough events?
2. Does each attack type retain its direction across the five frozen seeds?
3. Does the provider-balanced direction survive leave-one-provider-out?

A positive point estimate with a negative leave-one-provider direction is
retained as boundary evidence, not promoted to a confirmatory claim.
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
PREFIX = "stage280_nonstuck_stability_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)

STAGES = {
    "stage244": "stage244_cross_provider_nonstuck_candidate_features",
    "stage245": "stage245_cross_provider_nonstuck_seed_replication",
}
NONSTUCK_TYPES = [
    "spike",
    "offset",
    "drift",
    "time_shift",
    "gaussian_noise",
]
MIN_EVENTS_PER_PROVIDER_TYPE = 5
MIN_SEEDS = 5


def _read(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _provider_type_rows(
    stage: str,
) -> pd.DataFrame:
    prefix = STAGES[stage]
    detail = _read(f"{prefix}_type_detail.csv")
    detail = detail[detail["attack_type"].isin(NONSTUCK_TYPES)].copy()
    provider = (
        detail.groupby(
            ["source", "attack_type"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            events=("events", "sum"),
            detected_events=("detected_events", "sum")
            if "detected_events" in detail.columns
            else ("events", "sum"),
            row_recall_mean=("row_recall", "mean"),
            row_recall_min=("row_recall", "min"),
            event_coverage_mean=("event_coverage", "mean"),
            event_coverage_min=("event_coverage", "min"),
        )
    )
    if "detected_events" in detail.columns:
        provider["detected_events"] = pd.to_numeric(
            provider["detected_events"], errors="coerce"
        )
    provider["stage"] = stage
    return provider


def _seed_direction_rows(
    stage: str,
) -> pd.DataFrame:
    prefix = STAGES[stage]
    detail = _read(f"{prefix}_type_detail.csv")
    detail = detail[detail["attack_type"].isin(NONSTUCK_TYPES)].copy()
    rows: list[dict[str, object]] = []
    for (attack_type, seed), group in detail.groupby(
        ["attack_type", "seed"], sort=True
    ):
        event = group.groupby("source", as_index=False)[
            ["event_coverage", "row_recall"]
        ].mean()
        rows.append(
            {
                "stage": stage,
                "attack_type": attack_type,
                "seed": int(seed),
                "provider_count": int(event["source"].nunique()),
                "event_coverage": float(event["event_coverage"].mean()),
                "row_recall": float(event["row_recall"].mean()),
                "event_direction_positive": bool(
                    float(event["event_coverage"].mean()) > 0.0
                ),
                "row_direction_positive": bool(
                    float(event["row_recall"].mean()) > 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _type_aggregate(
    provider: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (stage, attack_type), group in provider.groupby(
        ["stage", "attack_type"], sort=True
    ):
        rows.append(
            {
                "stage": stage,
                "attack_type": attack_type,
                "provider_count": int(group["source"].nunique()),
                "seeds_min": int(group["seeds"].min()),
                "events": int(group["events"].sum()),
                "event_coverage": float(
                    group["event_coverage_mean"].mean()
                ),
                "event_coverage_min": float(
                    group["event_coverage_min"].min()
                ),
                "row_recall": float(group["row_recall_mean"].mean()),
                "row_recall_min": float(group["row_recall_min"].min()),
            }
        )
    return pd.DataFrame(rows)


def _checks(
    provider: pd.DataFrame,
    seeds: pd.DataFrame,
    types: pd.DataFrame,
    leave_one: pd.DataFrame,
) -> list[dict[str, object]]:
    expected_pairs = len(STAGES) * 4 * len(NONSTUCK_TYPES)
    checks: list[dict[str, object]] = [
        {
            "check": "provider_type_coverage",
            "status": (
                "PASS" if len(provider) == expected_pairs else "FAIL"
            ),
            "detail": (
                f"provider_type_rows={len(provider)} expected={expected_pairs}"
            ),
        },
        {
            "check": "minimum_events_per_provider_type",
            "status": (
                "PASS"
                if (
                    provider["events"] >= MIN_EVENTS_PER_PROVIDER_TYPE
                ).all()
                else "FAIL"
            ),
            "detail": (
                "min_events="
                f"{int(provider['events'].min())} threshold="
                f"{MIN_EVENTS_PER_PROVIDER_TYPE}"
            ),
        },
        {
            "check": "five_seeds_per_type",
            "status": (
                "PASS"
                if (
                    seeds.groupby(["stage", "attack_type"])["seed"]
                    .nunique()
                    .eq(MIN_SEEDS)
                    .all()
                )
                else "FAIL"
            ),
            "detail": (
                "seed_counts="
                + ",".join(
                    str(value)
                    for value in (
                        seeds.groupby(["stage", "attack_type"])["seed"]
                        .nunique()
                        .tolist()
                    )
                )
            ),
        },
        {
            "check": "type_event_coverage_positive",
            "status": (
                "PASS"
                if (provider["event_coverage_min"] > 0.0).all()
                else "FAIL"
            ),
            "detail": (
                "min_provider_type_event_coverage="
                f"{float(provider['event_coverage_min'].min())}"
            ),
        },
        {
            "check": "type_row_recall_positive",
            "status": (
                "PASS"
                if (provider["row_recall_min"] > 0.0).all()
                else "FAIL"
            ),
            "detail": (
                "min_provider_type_row_recall="
                f"{float(provider['row_recall_min'].min())}"
            ),
        },
        {
            "check": "all_seed_event_directions_positive",
            "status": (
                "PASS"
                if bool(seeds["event_direction_positive"].all())
                else "FAIL"
            ),
            "detail": (
                "positive_seed_types="
                f"{int(seeds['event_direction_positive'].sum())}/"
                f"{len(seeds)}"
            ),
        },
        {
            "check": "all_seed_row_directions_positive",
            "status": (
                "PASS"
                if bool(seeds["row_direction_positive"].all())
                else "FAIL"
            ),
            "detail": (
                "positive_seed_types="
                f"{int(seeds['row_direction_positive'].sum())}/"
                f"{len(seeds)}"
            ),
        },
        {
            "check": "leave_one_provider_event_positive",
            "status": (
                "PASS"
                if (
                    bool(leave_one["event_direction_positive"].all())
                    and bool(
                        (
                            provider["event_coverage_min"] > 0.0
                        ).all()
                    )
                )
                else "FAIL"
            ),
            "detail": (
                "positive_leave_one="
                f"{int(leave_one['event_direction_positive'].sum())}/"
                f"{len(leave_one)}"
            ),
        },
        {
            "check": "leave_one_provider_row_positive",
            "status": (
                "PASS"
                if (
                    bool(leave_one["row_direction_positive"].all())
                    and bool(
                        (provider["row_recall_min"] > 0.0).all()
                    )
                )
                else "FAIL"
            ),
            "detail": (
                "positive_leave_one="
                f"{int(leave_one['row_direction_positive'].sum())}/"
                f"{len(leave_one)}"
            ),
        },
    ]
    return checks


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    provider = pd.concat(
        [_provider_type_rows(stage) for stage in STAGES],
        ignore_index=True,
    )
    seeds = pd.concat(
        [_seed_direction_rows(stage) for stage in STAGES],
        ignore_index=True,
    )
    types = _type_aggregate(provider)
    leave_rows: list[dict[str, object]] = []
    for stage in STAGES:
        stage_provider = provider[provider["stage"].eq(stage)]
        providers = sorted(stage_provider["source"].unique().tolist())
        for removed in providers:
            retained = stage_provider[
                ~stage_provider["source"].eq(removed)
            ]
            leave_rows.append(
                {
                    "stage": stage,
                    "removed_source": removed,
                    "retained_sources": ",".join(
                        name for name in providers if name != removed
                    ),
                    "event_coverage": float(
                        retained["event_coverage_mean"].mean()
                    ),
                    "row_recall": float(
                        retained["row_recall_mean"].mean()
                    ),
                    "event_direction_positive": bool(
                        float(
                            retained["event_coverage_mean"].mean()
                        )
                        > 0.0
                    ),
                    "row_direction_positive": bool(
                        float(retained["row_recall_mean"].mean()) > 0.0
                    ),
                }
            )
    leave = pd.DataFrame(leave_rows)
    checks = _checks(provider, seeds, types, leave)
    failed = [row for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_NONSTUCK_STABILITY_AUDIT"
        if not failed
        else "BOUNDARY_NONSTUCK_STABILITY_REMAINS"
    )
    provider_path = TABLES / f"{PREFIX}_provider_type.csv"
    seed_path = TABLES / f"{PREFIX}_seed_direction.csv"
    type_path = TABLES / f"{PREFIX}_type_summary.csv"
    leave_path = TABLES / f"{PREFIX}_leave_one_provider.csv"
    provider.to_csv(provider_path, index=False, encoding="utf-8-sig")
    seeds.to_csv(seed_path, index=False, encoding="utf-8-sig")
    types.to_csv(type_path, index=False, encoding="utf-8-sig")
    leave.to_csv(leave_path, index=False, encoding="utf-8-sig")
    payload = {
        "status": status,
        "protocol": (
            "Stage280只读取Stage244/245冻结的type-detail结果，重建"
            "非stuck类型的事件数、5种子方向和留一机构方向；不训练、"
            "不重新选择特征、窗口、权重或阈值，不接触新的test预测。"
        ),
        "attack_types": NONSTUCK_TYPES,
        "provider_type_rows": int(len(provider)),
        "seed_direction_rows": int(len(seeds)),
        "type_summary_rows": int(len(types)),
        "leave_one_provider_rows": int(len(leave)),
        "checks": checks,
        "failed_checks": [row["check"] for row in failed],
        "paper_action": (
            "只有全部检查通过时才允许把非stuck稳定性升级为主要优势；"
            "否则主贡献必须限定为stuck/重复值/冻结状态异常。"
        ),
        "limitations": [
            "这是透明受控注入，不是真实攻击日志。",
            "四个目标机构不能代表机构总体分布。",
            "事件数充分不等于效果显著。",
            "任何负的留一机构方向必须保留，不能隐藏或重新选择。",
        ],
        "input_sha256": {
            stage: sha256_file(
                TABLES / f"{prefix}_type_detail.csv"
            )
            for stage, prefix in STAGES.items()
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage 280 非 stuck 稳定性审计",
        "",
        f"- 状态：`{status}`",
        f"- 非 stuck 类型：`{', '.join(NONSTUCK_TYPES)}`",
        f"- 类型汇总行数：`{len(types)}`",
        f"- 种子方向行数：`{len(seeds)}`",
        f"- 留一机构行数：`{len(leave)}`",
        "",
        "## 检查",
        "",
    ]
    for row in checks:
        report.append(
            f"- `{row['check']}`：`{row['status']}`；{row['detail']}"
        )
    report += [
        "",
        "## 处理规则",
        "",
        f"- {payload['paper_action']}",
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "failed_checks": payload["failed_checks"],
                "provider_type_rows": len(provider),
                "seed_direction_rows": len(seeds),
                "type_summary_rows": len(types),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
