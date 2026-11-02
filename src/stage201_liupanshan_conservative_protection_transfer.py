from __future__ import annotations

"""Stage 201: conservative cross-station protection transfer.

Stage 199 selected the best source-valid persistent-repeat protection
configuration for each held-out station and seed.  Stage 200 showed that
the failure is not mainly a threshold-shift problem: even a test-label
oracle threshold did not restore non-stuck event coverage.

This stage tests one pre-declared conservative rule.  A protection
configuration is used only when, on the six source stations' valid rows,
it improves pooled F1 by at least ``F1_MARGIN`` and does not reduce
pooled non-stuck event coverage relative to the frozen ungated score.
If no protection candidate satisfies that rule, the frozen baseline is
used.  Selection uses source valid rows only; the held-out station's
test rows are evaluation-only.

The result is exploratory transfer evidence, not a replacement for the
frozen model and not a real-attack-log validation.
"""

import argparse
import json
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

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage193_noaa_persistent_repeat_protection import (  # noqa: E402
    selection_key,
)
from stage195_noaa_seed_randomized_placement import (  # noqa: E402
    FAR_GRID,
    PRIMARY_FAR,
    SELECTION_RULES,
    clean_records,
)
from stage199_liupanshan_cross_station_protection_transfer import (  # noqa: E402
    SEEDS,
    candidate_table,
    evaluate_selection,
    local_score_frame,
    prepare_context,
)

PREFIX = "stage201_liupanshan_conservative_protection_transfer"
F1_MARGIN = 0.005
NONSTUCK_MARGIN = 0.0
PRIMARY_RULE = "valid_f1"


def select_conservative(
    candidates: pd.DataFrame,
    rule: str,
    max_far: float,
) -> tuple[dict[str, object], bool]:
    current = candidates[
        candidates["max_far"].eq(float(max_far))
    ].copy()
    if current.empty:
        raise ValueError("no candidate rows")
    baseline = current[
        current["gate_quantile"].eq(0.0)
        & current["run_length"].eq(0)
        & current["candidate_weight"].eq(0.0)
    ]
    if baseline.empty:
        raise ValueError("missing ungated baseline candidate")
    baseline_row = baseline.iloc[0].to_dict()
    eligible = current[
        current["gate_quantile"].gt(0.0)
        & current["run_length"].gt(0)
        & current["candidate_weight"].gt(0.0)
        & current["source_valid_f1"].ge(
            float(baseline_row["source_valid_f1"]) + F1_MARGIN
        )
        & current["source_valid_nonstuck_event_macro"].ge(
            float(baseline_row["source_valid_nonstuck_event_macro"])
            + NONSTUCK_MARGIN
        )
    ].copy()
    if eligible.empty:
        return baseline_row, False
    best = None
    best_key = None
    for row in eligible.to_dict(orient="records"):
        key = selection_key(
            rule,
            {"f1": float(row["source_valid_f1"])},
            float(row["source_valid_event_macro"]),
            float(row["source_valid_nonstuck_event_macro"]),
            float(row["candidate_weight"]),
            int(row["window"]),
            float(row["gate_quantile"]),
            int(row["run_length"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = row
    if best is None:
        raise RuntimeError("eligible candidate selection failed")
    return best, True


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = (
        detail.groupby(["selection_rule", "max_far"], sort=True)
        .agg(
            pairs=("held_out_station", "size"),
            protected_pairs=("protection_used", "sum"),
            test_f1_mean=("test_f1", "mean"),
            baseline_test_f1_mean=("baseline_test_f1", "mean"),
            test_nonstuck_mean=("test_nonstuck_event_macro", "mean"),
            baseline_nonstuck_mean=(
                "baseline_test_nonstuck_event_macro", "mean"
            ),
            test_false_alarm_rate_mean=(
                "test_false_alarm_rate", "mean"
            ),
        )
        .reset_index()
    )
    aggregate["f1_diff_mean"] = (
        aggregate["test_f1_mean"] - aggregate["baseline_test_f1_mean"]
    )
    aggregate["nonstuck_diff_mean"] = (
        aggregate["test_nonstuck_mean"]
        - aggregate["baseline_nonstuck_mean"]
    )
    paired_rows: list[dict[str, object]] = []
    for rule in sorted(detail["selection_rule"].unique().tolist()):
        for max_far in sorted(detail["max_far"].unique().tolist()):
            current = detail[
                detail["selection_rule"].eq(rule)
                & detail["max_far"].eq(float(max_far))
            ]
            for metric, baseline_metric in [
                ("test_f1", "baseline_test_f1"),
                (
                    "test_nonstuck_event_macro",
                    "baseline_test_nonstuck_event_macro",
                ),
                ("test_recall", "baseline_test_recall"),
                ("test_precision", "baseline_test_precision"),
                ("test_auprc", "baseline_test_auprc"),
                (
                    "test_false_alarm_rate",
                    "baseline_test_false_alarm_rate",
                ),
            ]:
                values = (
                    current[metric].to_numpy(dtype=float)
                    - current[baseline_metric].to_numpy(dtype=float)
                )
                paired_rows.append(
                    {
                        "selection_rule": rule,
                        "max_far": float(max_far),
                        "metric": metric,
                        "pairs": int(len(values)),
                        "positive_pairs": int((values > 0).sum()),
                        "negative_pairs": int((values < 0).sum()),
                        "mean_diff": float(np.mean(values)),
                        "std_diff": float(np.std(values, ddof=1)),
                        "min_diff": float(np.min(values)),
                        "max_diff": float(np.max(values)),
                    }
                )
    return aggregate, pd.DataFrame(paired_rows)


def write_outputs(
    detail: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
) -> int:
    detail.to_csv(
        TABLES / f"{PREFIX}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        TABLES / f"{PREFIX}_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        TABLES / f"{PREFIX}_paired.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary = aggregate[
        aggregate["selection_rule"].eq(PRIMARY_RULE)
        & aggregate["max_far"].eq(PRIMARY_FAR)
    ].iloc[0]
    primary_paired = paired[
        paired["selection_rule"].eq(PRIMARY_RULE)
        & paired["max_far"].eq(PRIMARY_FAR)
    ]
    f1 = primary_paired[primary_paired["metric"].eq("test_f1")].iloc[0]
    nonstuck = primary_paired[
        primary_paired["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    status = (
        "CANDIDATE_TRANSFER_REQUIRES_REPLICATION"
        if (
            float(f1["mean_diff"]) > 0.0
            and float(nonstuck["mean_diff"]) >= 0.0
            and int(primary["protected_pairs"]) > 0
        )
        else "NEGATIVE_RESULT_RETAINED"
    )
    payload = {
        "status": status,
        "protocol": (
            "阶段201在阶段199冻结分数上测试一个预先固定的保守迁移规则："
            "使用阶段199的五个固定攻击构造种子和七个留出站点；"
            "仅当保护候选在六个源站valid上的F1相对无门控基线至少提高"
            f"{F1_MARGIN:.3f}，且非stuck事件覆盖不低于基线时，才采用"
            "该保护配置；否则退回冻结基线。配置选择只用源站valid，"
            "留出站点test只评价；不根据test选择；不修改主模型。"
            "攻击为透明受控注入，不是真实攻击日志。"
        ),
        "seeds": SEEDS,
        "stations": sorted(
            detail["held_out_station"].astype(str).unique().tolist()
        ),
        "f1_margin": F1_MARGIN,
        "nonstuck_margin": NONSTUCK_MARGIN,
        "primary_rule": PRIMARY_RULE,
        "primary_far": PRIMARY_FAR,
        "detail_rows": int(len(detail)),
        "aggregate_rows": int(len(aggregate)),
        "paired_rows": int(len(paired)),
        "primary_protected_pairs": int(primary["protected_pairs"]),
        "primary_pairs": int(primary["pairs"]),
        "primary_f1_diff": float(f1["mean_diff"]),
        "primary_f1_positive_pairs": int(f1["positive_pairs"]),
        "primary_f1_negative_pairs": int(f1["negative_pairs"]),
        "primary_nonstuck_diff": float(nonstuck["mean_diff"]),
        "primary_nonstuck_positive_pairs": int(
            nonstuck["positive_pairs"]
        ),
        "primary_nonstuck_negative_pairs": int(
            nonstuck["negative_pairs"]
        ),
        "aggregate": clean_records(aggregate),
        "paired": clean_records(paired),
        "limitations": [
            "该阶段是阶段199负结果的保守规则诊断，不是新主模型。",
            "六盘山仍是单一时段、单一地区的受控攻击数据。",
            "五个攻击构造种子不等于五个独立地区或年份。",
            "攻击为透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段201 六盘山保守保护迁移",
        "",
        "保护仅在源站valid上同时满足预声明F1和非stuck覆盖条件时启用，否则退回基线。",
        "",
        "| 选择规则 | FAR上限 | 采用保护 | test F1 | 基线F1 | F1平均差 | 非stuck平均差 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['selection_rule']} | {row['max_far']:.3f} | "
            f"{int(row['protected_pairs'])}/{int(row['pairs'])} | "
            f"{row['test_f1_mean']:.6f} | "
            f"{row['baseline_test_f1_mean']:.6f} | "
            f"{row['f1_diff_mean']:+.6f} | "
            f"{row['nonstuck_diff_mean']:+.6f} |"
        )
    lines += [
        "",
        f"1%FAR下{PRIMARY_RULE}：F1平均差={f1['mean_diff']:+.6f}，"
        f"非stuck事件覆盖平均差={nonstuck['mean_diff']:+.6f}，"
        f"采用保护={int(primary['protected_pairs'])}/"
        f"{int(primary['pairs'])}。",
        "",
        "该结果只用于判断保守迁移规则是否值得进一步复现，不替换主模型。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": payload["detail_rows"],
                "aggregate_rows": payload["aggregate_rows"],
                "paired_rows": payload["paired_rows"],
                "protected_pairs": payload["primary_protected_pairs"],
                "f1_diff": payload["primary_f1_diff"],
                "nonstuck_diff": payload["primary_nonstuck_diff"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args(argv)
    detail_path = TABLES / f"{PREFIX}_detail.csv"
    if args.summarize_existing:
        detail = pd.read_csv(detail_path)
        aggregate, paired = summarize(detail)
        return write_outputs(detail, aggregate, paired)

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        source_path = TABLES / f"stage09c_input_seed{seed}.parquet"
        all_source = pd.read_parquet(source_path)
        stations = sorted(
            all_source["station_key"].astype(str).unique().tolist()
        )
        for station in stations:
            frame = local_score_frame(int(seed), station)
            context = prepare_context(frame, station, int(seed))
            candidates = candidate_table(context)
            for rule in SELECTION_RULES:
                for max_far in FAR_GRID:
                    selected, protection_used = select_conservative(
                        candidates, rule, float(max_far)
                    )
                    metrics = evaluate_selection(context, selected)
                    rows.append(
                        {
                            "seed": int(seed),
                            "held_out_station": station,
                            "selection_rule": rule,
                            "max_far": float(max_far),
                            "protection_used": bool(protection_used),
                            **{
                                key: selected[key]
                                for key in [
                                    "gate_quantile",
                                    "run_length",
                                    "window",
                                    "aggregation",
                                    "base_weight",
                                    "candidate_weight",
                                    "source_valid_f1",
                                    "source_valid_event_macro",
                                    "source_valid_nonstuck_event_macro",
                                ]
                            },
                            **metrics,
                        }
                    )
            print(
                json.dumps(
                    {
                        "seed": int(seed),
                        "held_out_station": station,
                        "status": "completed",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    detail = pd.DataFrame(rows)
    aggregate, paired = summarize(detail)
    return write_outputs(detail, aggregate, paired)


if __name__ == "__main__":
    raise SystemExit(main())
