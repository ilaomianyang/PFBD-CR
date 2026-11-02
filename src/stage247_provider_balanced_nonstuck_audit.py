from __future__ import annotations

"""Stage 247: provider-balanced audit of the frozen non-stuck candidates.

This stage is deliberately analysis-only.  It does not train a detector,
select a feature, choose a threshold, or read a test label for selection.
It takes the already frozen Stage244 and Stage245 provider-level results and
asks a narrower submission question:

    Is the positive non-stuck point estimate robust when each external
    provider is given equal weight and one provider is removed at a time?

The output separates directional replication from provider-balanced
robustness.  A positive point estimate with a negative leave-one-provider
direction remains boundary evidence, not a universal superiority claim.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage247_provider_balanced_nonstuck_audit"


def load_json(name: str) -> dict:
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


def provider_table(payload: dict, stage: str) -> pd.DataFrame:
    frame = pd.DataFrame(payload["provider_station_rows"]).copy()
    frame["stage"] = stage
    frame["source"] = frame["source"].astype(str)
    for column in (
        "f1_difference",
        "event_coverage_difference",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def provider_summary(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    summary = (
        frame.groupby("source", as_index=False)
        .agg(
            stations=("stations", "sum"),
            f1_difference=("f1_difference", "mean"),
            event_coverage_difference=("event_coverage_difference", "mean"),
        )
        .sort_values("source")
    )
    summary["stage"] = stage
    return summary[
        [
            "stage",
            "source",
            "stations",
            "f1_difference",
            "event_coverage_difference",
        ]
    ]


def equal_weight(summary: pd.DataFrame) -> dict[str, float]:
    return {
        "provider_count": int(summary["source"].nunique()),
        "f1_difference": float(summary["f1_difference"].mean()),
        "event_coverage_difference": float(
            summary["event_coverage_difference"].mean()
        ),
    }


def leave_one_provider(summary: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    providers = sorted(summary["source"].unique().tolist())
    for removed in providers:
        retained = summary[~summary["source"].eq(removed)]
        rows.append(
            {
                "removed_source": removed,
                "retained_sources": ",".join(
                    name for name in providers if name != removed
                ),
                "f1_difference": float(retained["f1_difference"].mean()),
                "event_coverage_difference": float(
                    retained["event_coverage_difference"].mean()
                ),
                "f1_direction_positive": bool(
                    float(retained["f1_difference"].mean()) > 0.0
                ),
                "event_direction_positive": bool(
                    float(retained["event_coverage_difference"].mean()) > 0.0
                ),
            }
        )
    return rows


def provider_rank_stability(summary: pd.DataFrame) -> dict[str, object]:
    f1_rank = summary.sort_values(
        ["f1_difference", "source"],
        ascending=[False, True],
    )["source"].tolist()
    event_rank = summary.sort_values(
        ["event_coverage_difference", "source"],
        ascending=[False, True],
    )["source"].tolist()
    return {
        "f1_rank_descending": f1_rank,
        "event_rank_descending": event_rank,
        "largest_f1_provider": f1_rank[0] if f1_rank else None,
        "largest_event_provider": event_rank[0] if event_rank else None,
    }


def main() -> int:
    stage244 = load_json(
        "stage244_cross_provider_nonstuck_candidate_features.json"
    )
    stage245 = load_json(
        "stage245_cross_provider_nonstuck_seed_replication.json"
    )
    frames = [
        provider_table(stage244, "stage244"),
        provider_table(stage245, "stage245"),
    ]
    combined = pd.concat(frames, ignore_index=True)
    summaries = pd.concat(
        [
            provider_summary(frames[0], "stage244"),
            provider_summary(frames[1], "stage245"),
        ],
        ignore_index=True,
    )
    equal_rows: list[dict[str, object]] = []
    leave_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for stage in ("stage244", "stage245"):
        summary = summaries[summaries["stage"].eq(stage)].copy()
        equal = equal_weight(summary)
        equal_rows.append({"stage": stage, **equal})
        leave_rows.extend(
            {"stage": stage, **row} for row in leave_one_provider(summary)
        )
        stability_rows.append(
            {"stage": stage, **provider_rank_stability(summary)}
        )

    equal = pd.DataFrame(equal_rows)
    leave = pd.DataFrame(leave_rows)
    combined["stage"] = combined["stage"].astype(str)
    stage_direction = (
        equal.groupby("stage", as_index=False)
        .agg(
            f1_difference=("f1_difference", "mean"),
            event_coverage_difference=("event_coverage_difference", "mean"),
        )
        .sort_values("stage")
    )
    checks = {
        "both_stages_f1_point_positive": bool(
            (equal["f1_difference"] > 0.0).all()
        ),
        "both_stages_event_point_positive": bool(
            (equal["event_coverage_difference"] > 0.0).all()
        ),
        "all_leave_one_f1_positive": bool(
            leave["f1_direction_positive"].all()
        ),
        "all_leave_one_event_positive": bool(
            leave["event_direction_positive"].all()
        ),
        "single_provider_not_dominant": bool(
            leave["f1_direction_positive"].all()
            and leave["event_direction_positive"].all()
        ),
    }
    payload = {
        "status": (
            "PASS_PIPELINE_REVIEW_REQUIRED"
            if all(checks.values())
            else "BOUNDARY_EVIDENCE_RETAINED"
        ),
        "protocol": (
            "阶段247只读取阶段244和阶段245已经冻结的provider级结果，"
            "重建机构等权点估计和留一机构方向；不训练、不重新选择特征、"
            "窗口、权重或阈值，不使用test标签做任何选择。"
            "所有攻击都是透明受控注入，不是真实攻击日志。"
        ),
        "source_rows": int(len(combined)),
        "provider_equal_rows": equal.to_dict("records"),
        "leave_one_provider_out_rows": leave.to_dict("records"),
        "provider_rank_rows": stability_rows,
        "checks": [
            {"check": name, "status": "PASS" if value else "FAIL"}
            for name, value in checks.items()
        ],
        "stage_direction_rows": stage_direction.to_dict("records"),
        "limitations": [
            "该阶段只审计阶段244/245冻结结果，不产生新的检测性能。",
            "四个目标机构之间的站点规模和网络结构不同，机构等权不是总体分布估计。",
            "若任何留一机构方向为负，必须保留该边界，不能宣称单一机构不驱动结论。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    equal.to_csv(
        TABLES / f"{PREFIX}_provider_equal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave.to_csv(
        TABLES / f"{PREFIX}_leave_one_provider_out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(stability_rows).to_csv(
        TABLES / f"{PREFIX}_provider_rank.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps({
        "status": payload["status"],
        "checks": checks,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
