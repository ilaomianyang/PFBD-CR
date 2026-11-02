from __future__ import annotations

"""Stage 316: read-only decomposition of the NOAA 2025 temporal holdout.

The stage does not train, tune, inject new attacks, or inspect any new
test period. It reads Stage315 and records which baselines and attack types
remain adverse so the claim ceiling stays bounded.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage316_noaa_2025_failure_decomposition"
STAGE315 = TABLES / "stage315_noaa_2025_temporal_locked_transfer.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    payload315 = load_json(STAGE315)
    comparisons = pd.read_csv(
        TABLES / "stage315_noaa_2025_temporal_locked_transfer_comparisons.csv"
    )
    type_detail = pd.read_csv(
        TABLES / "stage315_noaa_2025_temporal_locked_transfer_type_detail.csv"
    )
    station_detail = pd.read_csv(
        TABLES / "stage315_noaa_2025_temporal_locked_transfer_station_detail.csv"
    )

    failed = comparisons[~comparisons["primary_pass"].astype(bool)].copy()
    failed = failed.sort_values(
        ["candidate", "baseline"],
        kind="stable",
    ).reset_index(drop=True)

    type_summary = (
        type_detail.groupby(
            ["method", "attack_type"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            row_recall_mean=("row_recall", "mean"),
            event_coverage_mean=("event_coverage", "mean"),
        )
        .sort_values(["method", "attack_type"], kind="stable")
    )

    station_summary = (
        station_detail.groupby(
            ["method", "station_id"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            rows=("rows", "mean"),
            attack_rows=("attack_rows", "mean"),
            normal_rows=("normal_rows", "mean"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tn=("tn", "sum"),
            events=("events", "sum"),
            detected_events=("detected_events", "sum"),
        )
        .sort_values(["method", "station_id"], kind="stable")
    )
    station_summary["row_recall"] = station_summary.apply(
        lambda row: (
            float(row["tp"]) / float(row["tp"] + row["fn"])
            if (row["tp"] + row["fn"]) > 0
            else 0.0
        ),
        axis=1,
    )
    station_summary["event_coverage"] = station_summary.apply(
        lambda row: (
            float(row["detected_events"]) / float(row["events"])
            if row["events"] > 0
            else 0.0
        ),
        axis=1,
    )

    nonstuck_types = [
        "spike",
        "offset",
        "drift",
        "time_shift",
        "gaussian_noise",
    ]
    nonstuck = type_summary[
        type_summary["attack_type"].isin(nonstuck_types)
    ].copy()
    trust_nonstuck = nonstuck[
        nonstuck["method"].isin(
            ["trust_valid_f1", "trust_nonstuck_event_macro"]
        )
    ].copy()
    baseline_nonstuck = nonstuck[
        ~nonstuck["method"].isin(
            ["trust_valid_f1", "trust_nonstuck_event_macro"]
        )
    ].copy()
    best_baseline_by_type = (
        baseline_nonstuck.sort_values(
            ["attack_type", "event_coverage_mean"],
            ascending=[True, False],
            kind="stable",
        )
        .groupby("attack_type", as_index=False)
        .first()[["attack_type", "method", "event_coverage_mean"]]
        .rename(
            columns={
                "method": "best_baseline",
                "event_coverage_mean": "best_baseline_event_coverage",
            }
        )
    )
    trust_by_type = (
        trust_nonstuck.groupby("attack_type", as_index=False)[
            ["row_recall_mean", "event_coverage_mean"]
        ]
        .mean()
        .rename(
            columns={
                "row_recall_mean": "trust_mean_row_recall",
                "event_coverage_mean": "trust_mean_event_coverage",
            }
        )
    )
    nonstuck_comparison = trust_by_type.merge(
        best_baseline_by_type,
        on="attack_type",
        how="left",
        validate="one_to_one",
    )
    nonstuck_comparison["trust_minus_best_baseline_event"] = (
        nonstuck_comparison["trust_mean_event_coverage"]
        - nonstuck_comparison["best_baseline_event_coverage"]
    )

    failed.to_csv(
        TABLES / f"{PREFIX}_failed_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_summary.to_csv(
        TABLES / f"{PREFIX}_attack_type_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_summary.to_csv(
        TABLES / f"{PREFIX}_station_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nonstuck_comparison.to_csv(
        TABLES / f"{PREFIX}_nonstuck_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "status": "PASS_NOAA_2025_FAILURE_DECOMPOSITION_AUDIT",
        "protocol": "只读分解Stage315 NOAA 2025时间锁定结果；不训练、不调参、不新增test、不改变任何预测。",
        "source_stage": "stage315_noaa_2025_temporal_locked_transfer",
        "source_status": payload315["status"],
        "provider_is_new": False,
        "evaluation_period": "2025-01-01_to_2025-08-24",
        "station_count": int(payload315["station_count"]),
        "seed_count": int(payload315["seed_count"]),
        "method_count": int(payload315["method_count"]),
        "primary_pass_count": int(payload315["primary_pass_count"]),
        "primary_comparison_count": int(
            payload315["primary_comparison_count"]
        ),
        "primary_fail_count": int(len(failed)),
        "all_baselines_supported": bool(
            payload315["all_baselines_supported"]
        ),
        "failed_comparison_rows": int(len(failed)),
        "nonstuck_types": nonstuck_types,
        "p0_2_status_after": "OPEN",
        "p0_3_status_after": "OPEN",
        "p0_4_status_after": "OPEN",
        "adverse_results_preserved": True,
        "limitations": [
            "NOAA remains the same provider and the same 11-station network.",
            "The 2025 test period is partial.",
            "The decomposition is diagnostic and cannot promote a failed gate.",
            "All attacks are transparent controlled injections.",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Stage316 NOAA 2025 失败分解",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 来源：`{payload['source_stage']}`",
        f"- 主终点通过：`{payload['primary_pass_count']}/"
        f"{payload['primary_comparison_count']}`",
        f"- 未通过比较：`{payload['primary_fail_count']}`",
        f"- 全面支持：`{payload['all_baselines_supported']}`",
        "",
        "## 未通过比较",
        "",
        "| 候选 | 基线 | F1差 | Event差 |",
        "|---|---|---:|---:|",
    ]
    for row in failed.to_dict("records"):
        lines.append(
            f"| {row['candidate']} | {row['baseline']} | "
            f"{row['f1_difference']:.6f} | "
            f"{row['event_coverage_difference']:.6f} |"
        )
    lines += [
        "",
        "## 边界",
        "",
        "- 该分解只读取Stage315，不产生新的确认性性能。",
        "- 非 stuck 的失败必须保留，P0-2、P0-3、P0-4继续OPEN。",
        "- NOAA 2025不是新机构，不能关闭P0-1。",
        "- 受控注入不能写成真实生产事故召回率。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "primary_pass_count": payload["primary_pass_count"],
                "primary_comparison_count": payload[
                    "primary_comparison_count"
                ],
                "primary_fail_count": payload["primary_fail_count"],
                "all_baselines_supported": payload[
                    "all_baselines_supported"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
