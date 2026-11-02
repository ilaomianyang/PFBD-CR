from __future__ import annotations

"""Stage 200: failure attribution for Stage 199 cross-station transfer.

Stage 199 fixed both the persistent-repeat protection configuration and
the fixed-FAR threshold on six source stations, then evaluated the
held-out station without any target-domain adaptation.  Its primary
result was negative.

This diagnostic asks whether that failure is explained mainly by a
score-distribution / threshold shift rather than by the protection
configuration itself.  For every Stage 199 selection it compares:

* ``source_threshold``: the deployed Stage 199 rule (source valid only);
* ``target_oracle_threshold``: a diagnostic-only upper bound computed
  from held-out test normal labels at the same FAR cap.

The oracle threshold is not a deployable method.  It uses test labels,
never enters model or configuration selection, and must not be reported
as a performance result.  NOAA and Liupanshan attacks remain transparent
controlled injections, not real attack logs.
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
from stage186_noaa_targeted_nonstuck_feature_response import (  # noqa: E402
    ATTACK_TYPES,
)
from stage193_noaa_persistent_repeat_protection import (  # noqa: E402
    build_run_lengths,
    build_rolling_scores,
    gate_thresholds,
    macro_event_coverage,
    valid_far_threshold,
)
from stage195_noaa_seed_randomized_placement import (  # noqa: E402
    clean_records,
)
from stage199_liupanshan_cross_station_protection_transfer import (  # noqa: E402
    NON_STUCK_TYPES,
    SEEDS,
    local_score_frame,
    prepare_context,
)

STAGE199_PREFIX = (
    "stage199_liupanshan_cross_station_protection_transfer"
)
PREFIX = "stage200_liupanshan_protection_failure_attribution"
PRIMARY_RULE = "valid_f1"
PRIMARY_FAR = 0.01


def reconstruct_test_score(
    context: dict[str, object],
    selection: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    frame = context["frame"]
    source_valid = context["source_valid"]
    held_test = context["held_test"]
    y_valid = frame.loc[source_valid, "is_attack"].astype(bool).to_numpy()
    base_test = frame.loc[held_test, "base_score"].to_numpy(dtype=float)
    gate_valid = frame.loc[source_valid, "repeat_ratio"].to_numpy(dtype=float)
    thresholds = gate_thresholds(gate_valid, ~y_valid)
    runs = build_run_lengths(frame, thresholds)
    rolling = build_rolling_scores(frame)
    candidate_weight = float(selection["candidate_weight"])
    gate_quantile = float(selection["gate_quantile"])
    run_length = int(selection["run_length"])
    key = (int(selection["window"]), str(selection["aggregation"]))
    protected_test = (
        runs[gate_quantile][held_test] >= run_length
        if gate_quantile > 0.0
        else np.zeros(len(base_test), dtype=bool)
    )
    test_score = base_test.copy()
    if candidate_weight > 0.0:
        test_score = np.where(
            protected_test,
            base_test,
            (1.0 - candidate_weight) * base_test
            + candidate_weight * rolling[key][held_test],
        )
    return test_score, protected_test


def evaluate_selection(
    context: dict[str, object],
    selection: dict[str, object],
) -> dict[str, object]:
    frame = context["frame"]
    source_valid = context["source_valid"]
    held_test = context["held_test"]
    y_valid = frame.loc[source_valid, "is_attack"].astype(bool).to_numpy()
    y_test = frame.loc[held_test, "is_attack"].astype(bool).to_numpy()
    valid_frame = frame.loc[source_valid].reset_index(drop=True)
    test_frame = frame.loc[held_test].reset_index(drop=True)
    base_valid = frame.loc[source_valid, "base_score"].to_numpy(dtype=float)
    base_test = frame.loc[held_test, "base_score"].to_numpy(dtype=float)
    gate_valid = frame.loc[source_valid, "repeat_ratio"].to_numpy(dtype=float)
    thresholds = gate_thresholds(gate_valid, ~y_valid)
    runs = build_run_lengths(frame, thresholds)
    rolling = build_rolling_scores(frame)
    max_far = float(selection["max_far"])
    candidate_weight = float(selection["candidate_weight"])
    gate_quantile = float(selection["gate_quantile"])
    run_length = int(selection["run_length"])
    key = (int(selection["window"]), str(selection["aggregation"]))
    protected_valid = (
        runs[gate_quantile][source_valid] >= run_length
        if gate_quantile > 0.0
        else np.zeros(len(base_valid), dtype=bool)
    )
    protected_test = (
        runs[gate_quantile][held_test] >= run_length
        if gate_quantile > 0.0
        else np.zeros(len(base_test), dtype=bool)
    )
    valid_score = base_valid.copy()
    test_score = base_test.copy()
    if candidate_weight > 0.0:
        valid_score = np.where(
            protected_valid,
            base_valid,
            (1.0 - candidate_weight) * base_valid
            + candidate_weight * rolling[key][source_valid],
        )
        test_score = np.where(
            protected_test,
            base_test,
            (1.0 - candidate_weight) * base_test
            + candidate_weight * rolling[key][held_test],
        )

    source_threshold = valid_far_threshold(
        valid_score, ~y_valid, max_far
    )
    source_pred = test_score >= source_threshold
    oracle_threshold = valid_far_threshold(
        test_score, ~y_test, max_far
    )
    oracle_pred = test_score >= oracle_threshold

    baseline_threshold = valid_far_threshold(
        base_valid, ~y_valid, max_far
    )
    baseline_pred = base_test >= baseline_threshold
    source_metrics = _metrics(y_test, source_pred, test_score)
    oracle_metrics = _metrics(y_test, oracle_pred, test_score)
    baseline_metrics = _metrics(y_test, baseline_pred, base_test)
    return {
        "source_threshold": float(source_threshold),
        "oracle_threshold": float(oracle_threshold),
        "threshold_shift": float(oracle_threshold - source_threshold),
        "source_test_f1": float(source_metrics["f1"]),
        "source_test_recall": float(source_metrics["recall"]),
        "source_test_precision": float(source_metrics["precision"]),
        "source_test_auprc": float(
            source_metrics["average_precision"]
        ),
        "source_test_false_alarm_rate": (
            float(source_pred[~y_test].mean())
            if (~y_test).any()
            else float("nan")
        ),
        "source_test_nonstuck_event_macro": macro_event_coverage(
            test_frame, source_pred, NON_STUCK_TYPES
        ),
        "oracle_test_f1": float(oracle_metrics["f1"]),
        "oracle_test_recall": float(oracle_metrics["recall"]),
        "oracle_test_precision": float(oracle_metrics["precision"]),
        "oracle_test_auprc": float(
            oracle_metrics["average_precision"]
        ),
        "oracle_test_false_alarm_rate": (
            float(oracle_pred[~y_test].mean())
            if (~y_test).any()
            else float("nan")
        ),
        "oracle_test_nonstuck_event_macro": macro_event_coverage(
            test_frame, oracle_pred, NON_STUCK_TYPES
        ),
        "baseline_test_f1": float(baseline_metrics["f1"]),
        "baseline_test_nonstuck_event_macro": macro_event_coverage(
            test_frame, baseline_pred, NON_STUCK_TYPES
        ),
        "oracle_gain_over_source_f1": float(
            oracle_metrics["f1"] - source_metrics["f1"]
        ),
        "oracle_gain_over_source_nonstuck": float(
            macro_event_coverage(test_frame, oracle_pred, NON_STUCK_TYPES)
            - macro_event_coverage(
                test_frame, source_pred, NON_STUCK_TYPES
            )
        ),
        "source_minus_baseline_f1": float(
            source_metrics["f1"] - baseline_metrics["f1"]
        ),
        "source_minus_baseline_nonstuck": float(
            macro_event_coverage(
                test_frame, source_pred, NON_STUCK_TYPES
            )
            - macro_event_coverage(
                test_frame, baseline_pred, NON_STUCK_TYPES
            )
        ),
    }


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = (
        detail.groupby(["selection_rule", "max_far"], sort=True)
        .agg(
            pairs=("held_out_station", "size"),
            source_f1_mean=("source_test_f1", "mean"),
            oracle_f1_mean=("oracle_test_f1", "mean"),
            baseline_f1_mean=("baseline_test_f1", "mean"),
            source_nonstuck_mean=(
                "source_test_nonstuck_event_macro", "mean"
            ),
            oracle_nonstuck_mean=(
                "oracle_test_nonstuck_event_macro", "mean"
            ),
            baseline_nonstuck_mean=(
                "baseline_test_nonstuck_event_macro", "mean"
            ),
            mean_threshold_shift=("threshold_shift", "mean"),
            mean_oracle_gain_f1=("oracle_gain_over_source_f1", "mean"),
            mean_oracle_gain_nonstuck=(
                "oracle_gain_over_source_nonstuck", "mean"
            ),
            mean_source_minus_baseline_f1=(
                "source_minus_baseline_f1", "mean"
            ),
            mean_source_minus_baseline_nonstuck=(
                "source_minus_baseline_nonstuck", "mean"
            ),
        )
        .reset_index()
    )
    paired_rows: list[dict[str, object]] = []
    for rule in sorted(detail["selection_rule"].unique().tolist()):
        for max_far in sorted(detail["max_far"].unique().tolist()):
            current = detail[
                detail["selection_rule"].eq(rule)
                & detail["max_far"].eq(float(max_far))
            ]
            for metric in [
                "oracle_gain_over_source_f1",
                "oracle_gain_over_source_nonstuck",
                "source_minus_baseline_f1",
                "source_minus_baseline_nonstuck",
                "threshold_shift",
            ]:
                values = current[metric].to_numpy(dtype=float)
                paired_rows.append(
                    {
                        "selection_rule": rule,
                        "max_far": float(max_far),
                        "metric": metric,
                        "pairs": int(len(values)),
                        "positive_pairs": int((values > 0).sum()),
                        "negative_pairs": int((values < 0).sum()),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
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
    oracle_f1 = primary_paired[
        primary_paired["metric"].eq("oracle_gain_over_source_f1")
    ].iloc[0]
    oracle_nonstuck = primary_paired[
        primary_paired["metric"].eq(
            "oracle_gain_over_source_nonstuck"
        )
    ].iloc[0]
    source_f1 = primary_paired[
        primary_paired["metric"].eq("source_minus_baseline_f1")
    ].iloc[0]
    payload = {
        "status": "DIAGNOSTIC_ONLY_ORACLE_THRESHOLD",
        "protocol": (
            "阶段200只复用阶段199已选保护配置和冻结分数，做失败归因；"
            "不重新选择候选、不重新训练模型、不修改阈值规则。"
            "使用五个固定攻击构造种子和七个留出站点；"
            "source_threshold仅由六个源站valid正常样本按固定FAR校准；"
            "target_oracle_threshold使用留出站点test正常标签，仅作为"
            "不可部署的诊断上界；oracle不参与模型、配置、阈值或规则"
            "选择，不根据test选择，也不得作为论文性能结果；"
            "不得作为可部署方法。"
            "留出站点test只用于诊断评价。"
            "攻击为透明受控注入，不是真实攻击日志。"
        ),
        "source_stage": STAGE199_PREFIX,
        "seeds": SEEDS,
        "stations": sorted(
            detail["held_out_station"].astype(str).unique().tolist()
        ),
        "primary_rule": PRIMARY_RULE,
        "primary_far": PRIMARY_FAR,
        "detail_rows": int(len(detail)),
        "aggregate_rows": int(len(aggregate)),
        "paired_rows": int(len(paired)),
        "primary_source_f1_mean": float(primary["source_f1_mean"]),
        "primary_oracle_f1_mean": float(primary["oracle_f1_mean"]),
        "primary_source_minus_baseline_f1": float(
            source_f1["mean"]
        ),
        "primary_oracle_gain_f1": float(oracle_f1["mean"]),
        "primary_oracle_gain_f1_positive_pairs": int(
            oracle_f1["positive_pairs"]
        ),
        "primary_oracle_gain_f1_negative_pairs": int(
            oracle_f1["negative_pairs"]
        ),
        "primary_oracle_gain_nonstuck": float(
            oracle_nonstuck["mean"]
        ),
        "primary_oracle_gain_nonstuck_positive_pairs": int(
            oracle_nonstuck["positive_pairs"]
        ),
        "primary_oracle_gain_nonstuck_negative_pairs": int(
            oracle_nonstuck["negative_pairs"]
        ),
        "mean_threshold_shift": float(primary["mean_threshold_shift"]),
        "aggregate": clean_records(aggregate),
        "paired": clean_records(paired),
        "limitations": [
            "oracle阈值使用test正常标签，只能作为失败归因上界。",
            "oracle结果不能作为可部署方法或论文性能结果。",
            "六盘山仍是单一时段、单一地区的受控攻击数据。",
            "攻击为透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段200 六盘山保护迁移失败归因",
        "",
        "只复用阶段199冻结配置，比较源站阈值与不可部署的oracle阈值上界。",
        "",
        "| 选择规则 | FAR上限 | 源站F1 | oracle F1 | 源站-基线F1 | oracle增益F1 | oracle增益非stuck | 平均阈值偏移 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['selection_rule']} | {row['max_far']:.3f} | "
            f"{row['source_f1_mean']:.6f} | "
            f"{row['oracle_f1_mean']:.6f} | "
            f"{row['mean_source_minus_baseline_f1']:+.6f} | "
            f"{row['mean_oracle_gain_f1']:+.6f} | "
            f"{row['mean_oracle_gain_nonstuck']:+.6f} | "
            f"{row['mean_threshold_shift']:+.6f} |"
        )
    lines += [
        "",
        f"1%FAR下valid_f1：源站迁移相对基线F1平均差="
        f"{source_f1['mean']:+.6f}；oracle阈值相对源站阈值F1平均增益="
        f"{oracle_f1['mean']:+.6f}"
        f"（正向={int(oracle_f1['positive_pairs'])}/"
        f"{int(oracle_f1['pairs'])}）；非stuck平均增益="
        f"{oracle_nonstuck['mean']:+.6f}。",
        "",
        "oracle阈值使用test正常标签，仅用于诊断失败来源，不是可部署结果。",
        "该诊断不改变阶段199的负结果，也不能用于论文性能声明。",
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
                "source_minus_baseline_f1": payload[
                    "primary_source_minus_baseline_f1"
                ],
                "oracle_gain_f1": payload["primary_oracle_gain_f1"],
                "oracle_gain_nonstuck": payload[
                    "primary_oracle_gain_nonstuck"
                ],
                "mean_threshold_shift": payload["mean_threshold_shift"],
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

    selection_path = TABLES / f"{STAGE199_PREFIX}_selections.csv"
    if not selection_path.is_file():
        raise FileNotFoundError(
            "stage200 requires stage199 selections; run stage199 first"
        )
    selections = pd.read_csv(selection_path)
    selections = selections[
        selections["selection_rule"].eq(PRIMARY_RULE)
    ].copy()
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for station in sorted(
            selections.loc[
                selections["seed"].eq(int(seed)),
                "held_out_station",
            ]
            .astype(str)
            .unique()
            .tolist()
        ):
            frame = local_score_frame(int(seed), station)
            context = prepare_context(frame, station, int(seed))
            current = selections[
                selections["seed"].eq(int(seed))
                & selections["held_out_station"].astype(str).eq(station)
            ]
            if current.empty:
                raise ValueError(
                    f"missing stage199 selection for {seed}/{station}"
                )
            for selection in current.to_dict(orient="records"):
                metrics = evaluate_selection(context, selection)
                rows.append(
                    {
                        "seed": int(seed),
                        "held_out_station": station,
                        **{
                            key: selection[key]
                            for key in [
                                "selection_rule",
                                "max_far",
                                "gate_quantile",
                                "run_length",
                                "window",
                                "aggregation",
                                "base_weight",
                                "candidate_weight",
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
