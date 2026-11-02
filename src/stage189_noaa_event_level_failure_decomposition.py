from __future__ import annotations

"""Stage 189: event-level decomposition of low-FAR NOAA misses.

Stage 180 preserved the frozen expanded-risk row detail for the two
pre-declared low-FAR operating points and three station budget rules.  This
stage reuses those rows without retraining and separates each controlled
attack event into one of two mutually exclusive failure classes:

* ``ranking_failure``: no row of the event is in the global fixed budget;
* ``station_budget_suppression``: at least one row is in the global fixed
  budget, but the station budget excludes every row of the event.

The second class is further split into ``zero_station_budget`` and
``local_displacement`` using the station's total selected-row count.  No
model, threshold, q, allocation rule, or event definition is selected from
test.  The audit is diagnostic and does not replace the frozen model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
ROW_DETAIL = (
    TABLES / "stage180_noaa_station_budget_row_tail_audit_row_detail.parquet"
)
EVENT_DETAIL = (
    TABLES / "stage180_noaa_station_budget_row_tail_audit_event_detail.csv"
)
KEYS = ["fold", "seed", "q", "allocation_rule"]
EVENT_KEYS = [*KEYS, "event_id"]
STATION_KEYS = [*KEYS, "station_id"]
FAILURE_CLASSES = [
    "covered",
    "ranking_failure",
    "station_budget_suppression",
]
SUPPRESSION_CLASSES = ["zero_station_budget", "local_displacement"]
RISK_BINS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0000001]
RISK_LABELS = [
    "[0.000,0.500)",
    "[0.500,0.800)",
    "[0.800,0.900)",
    "[0.900,0.950)",
    "[0.950,0.980)",
    "[0.980,0.990)",
    "[0.990,0.995)",
    "[0.995,0.999)",
    "[0.999,1.000]",
]
FOLD_NAMES = [
    "2020->2021->2022",
    "2020-2021->2022->2023",
    "2020-2021-2022->2023->2024",
    "2020-2021-2022-2023->2024->2025",
]


def resolve_fold(value: str | None) -> str | None:
    if value is None:
        return None
    if value in FOLD_NAMES:
        return value
    if value.isdigit():
        index = int(value)
        if 0 <= index < len(FOLD_NAMES):
            return FOLD_NAMES[index]
    raise ValueError(
        f"unknown fold {value!r}; use 0-{len(FOLD_NAMES) - 1} or a fold name"
    )


def clean_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        cleaned: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, float) and not np.isfinite(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def failure_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    output = (
        frame.groupby(keys, sort=True)
        .agg(
            events=("event_id", "size"),
            covered_events=("event_covered", "sum"),
            ranking_failure_events=(
                "failure_class",
                lambda values: int((values == "ranking_failure").sum()),
            ),
            station_budget_suppression_events=(
                "failure_class",
                lambda values: int(
                    (values == "station_budget_suppression").sum()
                ),
            ),
            zero_station_budget_events=(
                "suppression_class",
                lambda values: int((values == "zero_station_budget").sum()),
            ),
            local_displacement_events=(
                "suppression_class",
                lambda values: int((values == "local_displacement").sum()),
            ),
            event_rows_total=("event_rows", "sum"),
            selected_event_rows=("selected_rows", "sum"),
            global_selected_event_rows=("global_selected_rows", "sum"),
            mean_event_recall=("event_recall", "mean"),
            mean_event_rows=("event_rows", "mean"),
            mean_risk_percentile_max=("risk_percentile_max", "mean"),
            median_risk_percentile_max=("risk_percentile_max", "median"),
        )
        .reset_index()
    )
    output["uncovered_events"] = output["events"] - output["covered_events"]
    output["event_coverage"] = (
        output["covered_events"] / output["events"]
    )
    output["ranking_failure_rate"] = (
        output["ranking_failure_events"] / output["events"]
    )
    output["station_budget_suppression_rate"] = (
        output["station_budget_suppression_events"] / output["events"]
    )
    output["ranking_failure_share_of_uncovered"] = (
        output["ranking_failure_events"] / output["uncovered_events"]
    )
    output["station_budget_suppression_share_of_uncovered"] = (
        output["station_budget_suppression_events"] / output["uncovered_events"]
    )
    output["zero_station_budget_share_of_suppression"] = (
        output["zero_station_budget_events"]
        / output["station_budget_suppression_events"]
    )
    output["local_displacement_share_of_suppression"] = (
        output["local_displacement_events"]
        / output["station_budget_suppression_events"]
    )
    output["event_recall"] = (
        output["selected_event_rows"] / output["event_rows_total"]
    )
    return output


def load_event_frame(
    row_detail: pd.DataFrame,
    event_detail: pd.DataFrame,
    fold: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    rows = row_detail.copy()
    if fold is not None:
        rows = rows[rows["fold"].eq(fold)]
    if seed is not None:
        rows = rows[rows["seed"].eq(seed)]
    if rows.empty:
        raise ValueError("no row detail matched the requested fold/seed")

    rows["station_id"] = rows["station_id"].astype(str)
    rows["event_id"] = rows["event_id"].astype(str)
    rows["event_key"] = (
        rows["fold"].astype(str)
        + "|"
        + rows["station_id"]
        + "|"
        + rows["attack_type"].astype(str)
        + "|"
        + rows["seed"].astype(str)
    )
    rows["selected"] = rows["selected"].astype(bool)
    rows["global_selected"] = rows["global_selected"].astype(bool)
    rows["is_attack"] = rows["is_attack"].astype(bool)
    attacks = rows[rows["is_attack"]].copy()
    if attacks.empty:
        raise ValueError("no attack rows in the requested row detail")

    event = (
        attacks.groupby(
            [*KEYS, "event_key", "station_id", "attack_type"], sort=True
        )
        .agg(
            event_rows=("selected", "size"),
            selected_rows=("selected", "sum"),
            global_selected_rows=("global_selected", "sum"),
            risk_max=("risk", "max"),
            risk_percentile_max=("risk_percentile", "max"),
            risk_median=("risk", "median"),
            risk_percentile_median=("risk_percentile", "median"),
        )
        .reset_index()
        .rename(columns={"event_key": "event_id"})
    )
    station_budget = (
        rows[rows["selected"]]
        .groupby(STATION_KEYS, sort=True)
        .agg(
            station_total_selected_rows=("selected", "size"),
            station_budget_threshold_risk=("risk", "min"),
            station_budget_threshold_percentile=(
                "risk_percentile",
                "min",
            ),
        )
        .reset_index()
    )
    event = event.merge(
        station_budget,
        on=STATION_KEYS,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    event["station_total_selected_rows"] = (
        event["station_total_selected_rows"].fillna(0).astype(int)
    )
    event["station_has_budget"] = event["station_total_selected_rows"] > 0
    event["event_covered"] = event["selected_rows"] > 0
    event["event_recall"] = event["selected_rows"] / event["event_rows"]
    event["failure_class"] = np.select(
        [
            event["event_covered"],
            event["global_selected_rows"] > 0,
        ],
        ["covered", "station_budget_suppression"],
        default="ranking_failure",
    )
    event["suppression_class"] = np.select(
        [
            event["failure_class"].ne("station_budget_suppression"),
            event["station_has_budget"],
        ],
        ["not_applicable", "local_displacement"],
        default="zero_station_budget",
    )
    event["above_station_budget_threshold"] = (
        event["risk_max"] >= event["station_budget_threshold_risk"]
    )

    expected = event_detail.copy()
    expected["station_id"] = expected["station_id"].astype(str)
    expected["event_id"] = expected["event_id"].astype(str)
    if fold is not None:
        expected = expected[expected["fold"].eq(fold)]
    if seed is not None:
        expected = expected[expected["seed"].eq(seed)]
    if expected[EVENT_KEYS].duplicated().any():
        raise ValueError("event detail has duplicate event keys")
    check = event.merge(
        expected[
            [
                *EVENT_KEYS,
                "event_rows",
                "selected_rows",
                "event_recall",
                "event_covered",
                "risk_percentile_max",
            ]
        ],
        on=EVENT_KEYS,
        how="outer",
        validate="one_to_one",
        suffixes=("", "_expected"),
        indicator=True,
    )
    if len(check) != len(event) or not check["_merge"].eq("both").all():
        raise ValueError("event rows do not align with stage 180 event detail")
    if not check["event_rows"].eq(check["event_rows_expected"]).all():
        raise ValueError("event row counts do not match stage 180")
    if not check["selected_rows"].eq(check["selected_rows_expected"]).all():
        raise ValueError("selected row counts do not match stage 180")
    if not check["event_covered"].eq(check["event_covered_expected"]).all():
        raise ValueError("event coverage does not match stage 180")
    if not np.allclose(
        check["event_recall"],
        check["event_recall_expected"],
        atol=1e-12,
    ):
        raise ValueError("event recall does not match stage 180")
    if not np.allclose(
        check["risk_percentile_max"],
        check["risk_percentile_max_expected"],
        atol=1e-12,
    ):
        raise ValueError("event risk percentile does not match stage 180")
    if not check["failure_class"].isin(FAILURE_CLASSES).all():
        raise ValueError("unknown event failure class")
    if not set(check["suppression_class"].unique()).issubset(
        {"not_applicable", *SUPPRESSION_CLASSES}
    ):
        raise ValueError("unknown suppression class")
    if (
        check["failure_class"].eq("ranking_failure")
        & check["global_selected_rows"].gt(0)
    ).any():
        raise ValueError("ranking failure cannot contain globally selected rows")
    if (
        check["failure_class"].eq("station_budget_suppression")
        & ~check["selected_rows"].eq(0)
    ).any():
        raise ValueError("budget suppression cannot contain selected event rows")
    return check.drop(
        columns=[
            "_merge",
            "event_rows_expected",
            "selected_rows_expected",
            "event_recall_expected",
            "event_covered_expected",
            "risk_percentile_max_expected",
        ]
    )


def write_outputs(event: pd.DataFrame, prefix: str) -> int:
    event = event.copy()
    event["risk_bin"] = pd.cut(
        event["risk_percentile_max"],
        bins=RISK_BINS,
        labels=RISK_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)
    overall = failure_summary(event, ["q", "allocation_rule"])
    by_type = failure_summary(
        event, ["q", "allocation_rule", "attack_type"]
    )
    by_length = failure_summary(
        event, ["q", "allocation_rule", "event_rows"]
    )
    by_risk = failure_summary(
        event, ["q", "allocation_rule", "risk_bin"]
    )
    by_station = failure_summary(
        event, ["q", "allocation_rule", "station_id"]
    )

    event.to_csv(
        TABLES / f"{prefix}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall.to_csv(
        TABLES / f"{prefix}_overall_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_type.to_csv(
        TABLES / f"{prefix}_type_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_length.to_csv(
        TABLES / f"{prefix}_length_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_risk.to_csv(
        TABLES / f"{prefix}_risk_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_station.to_csv(
        TABLES / f"{prefix}_station_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段180冻结扩展风险的逐行test明细和事件明细；不重新训练、"
            "不改变模型、alpha、window、q、阈值或分配规则，不根据test选择"
            "任何规则；四折五种子仅用于不确定性描述。每个受控攻击事件按互斥规则分解：全局固定预算内没有"
            "该事件任何行记为ranking_failure；全局预算内有事件行但站点预算"
            "未选中任何事件行记为station_budget_suppression，并进一步区分为"
            "zero_station_budget或local_displacement。事件覆盖是诊断性事件级"
            "证据，不替代逐行检测指标。NOAA攻击为透明受控注入，不是真实"
            "攻击日志。"
        ),
        "source_artifacts": [ROW_DETAIL.name, EVENT_DETAIL.name],
        "failure_classes": FAILURE_CLASSES,
        "suppression_classes": SUPPRESSION_CLASSES,
        "risk_bins": RISK_LABELS,
        "detail_rows": int(len(event)),
        "overall_aggregate_rows": int(len(overall)),
        "type_aggregate_rows": int(len(by_type)),
        "length_aggregate_rows": int(len(by_length)),
        "risk_aggregate_rows": int(len(by_risk)),
        "station_aggregate_rows": int(len(by_station)),
        "overall_aggregate": clean_records(overall),
        "type_aggregate": clean_records(by_type),
        "length_aggregate": clean_records(by_length),
        "risk_aggregate": clean_records(by_risk),
        "station_aggregate": clean_records(by_station),
        "limitations": [
            "事件级覆盖不能替代逐行F1、Recall、Precision或FAR。",
            "事件上限、风险分位和失败分类只用于解释阶段180的固定预算漏检。",
            "攻击事件来自透明受控注入，不代表真实攻击日志。",
            "zero_station_budget和local_displacement是诊断子类，不用于部署选择。",
            "站点预算抑制为零或未覆盖事件为零时，相应比例不可估计，保留为空值。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段189 NOAA事件级漏检分解",
        "",
        "复用阶段180逐行test明细；不重新训练、不根据test选择q、阈值或分配规则。",
        "",
        "| q | 分配规则 | 事件数 | 事件覆盖 | 排序失败 | 预算抑制 | 零站点预算 | 站内置换 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.to_dict(orient="records"):
        lines.append(
            f"| {row['q']:.4f} | {row['allocation_rule']} | "
            f"{int(row['events'])} | {row['event_coverage']:.6f} | "
            f"{row['ranking_failure_rate']:.6f} | "
            f"{row['station_budget_suppression_rate']:.6f} | "
            f"{int(row['zero_station_budget_events'])} | "
            f"{int(row['local_displacement_events'])} |"
        )
    lines += [
        "",
        "## 攻击类型分解",
        "",
        "| q | 分配规则 | 攻击类型 | 事件数 | 事件覆盖 | 排序失败率 | 预算抑制率 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in by_type.to_dict(orient="records"):
        lines.append(
            f"| {row['q']:.4f} | {row['allocation_rule']} | "
            f"{row['attack_type']} | {int(row['events'])} | "
            f"{row['event_coverage']:.6f} | "
            f"{row['ranking_failure_rate']:.6f} | "
            f"{row['station_budget_suppression_rate']:.6f} |"
        )
    lines += [
        "",
        "## 事件长度分解",
        "",
        "| q | 分配规则 | 事件行数 | 事件数 | 事件覆盖 | 排序失败率 | 预算抑制率 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in by_length.to_dict(orient="records"):
        lines.append(
            f"| {row['q']:.4f} | {row['allocation_rule']} | "
            f"{int(row['event_rows'])} | {int(row['events'])} | "
            f"{row['event_coverage']:.6f} | "
            f"{row['ranking_failure_rate']:.6f} | "
            f"{row['station_budget_suppression_rate']:.6f} |"
        )
    lines += [
        "",
        "## 风险上界分位分解",
        "",
        "| q | 分配规则 | 风险上界分位 | 事件数 | 事件覆盖 | 排序失败率 | 预算抑制率 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in by_risk.to_dict(orient="records"):
        lines.append(
            f"| {row['q']:.4f} | {row['allocation_rule']} | "
            f"{row['risk_bin']} | {int(row['events'])} | "
            f"{row['event_coverage']:.6f} | "
            f"{row['ranking_failure_rate']:.6f} | "
            f"{row['station_budget_suppression_rate']:.6f} |"
        )
    lines += [
        "",
        f"事件明细={len(event)}；总体聚合={len(overall)}；"
        f"类型聚合={len(by_type)}；长度聚合={len(by_length)}；"
        f"风险聚合={len(by_risk)}；站点聚合={len(by_station)}。",
        "失败分类互斥；事件覆盖不是逐行性能指标。",
        "NOAA攻击为透明受控注入，不是真实攻击日志。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": len(event),
                "overall_aggregate_rows": len(overall),
                "type_aggregate_rows": len(by_type),
                "length_aggregate_rows": len(by_length),
                "risk_aggregate_rows": len(by_risk),
                "station_aggregate_rows": len(by_station),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row-detail",
        type=Path,
        default=ROW_DETAIL,
        help="Stage-180 row-level test detail parquet.",
    )
    parser.add_argument(
        "--event-detail",
        type=Path,
        default=EVENT_DETAIL,
        help="Stage-180 event-level detail CSV.",
    )
    parser.add_argument("--fold", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="stage189_noaa_event_level_failure_decomposition",
    )
    args = parser.parse_args(argv)
    row_detail = pd.read_parquet(
        args.row_detail,
        columns=[
            *STATION_KEYS,
            "risk",
            "risk_percentile",
            "selected",
            "global_selected",
            "is_attack",
            "attack_type",
            "event_id",
        ],
    )
    event_detail = pd.read_csv(args.event_detail)
    fold = resolve_fold(args.fold)
    event = load_event_frame(
        row_detail,
        event_detail,
        fold=fold,
        seed=args.seed,
    )
    return write_outputs(event, args.output_prefix)


if __name__ == "__main__":
    raise SystemExit(main())
