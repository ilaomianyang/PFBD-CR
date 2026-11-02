from __future__ import annotations

"""Stage 191: focused diagnostics for the stage-190 reranking candidate.

Stage 190 evaluated a pre-declared causal rolling reranking space and found
a small positive mean F1 difference at low FAR together with a consistent
increase in non-stuck event coverage.  This stage does not select anything
new.  It reuses the stage-190 valid-only selections and diagnoses:

* which rolling settings were selected across fold/seed pairs;
* attack-type-level changes relative to the frozen baseline;
* the cost on ``stuck`` events;
* whether the apparent gain is concentrated in a few folds or seeds.

Baseline thresholds are recomputed only from valid-normal rows, exactly as
in stage 190.  Test labels are used only for the diagnostic comparison.
NOAA attacks are transparent controlled injections, not real attack logs.
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
SCORE_DETAIL = (
    TABLES
    / "stage187_noaa_targeted_features_validation_calibrated_score_detail.parquet"
)
STAGE190_SELECTION = (
    TABLES / "stage190_noaa_event_reranking_validation_selection.csv"
)
STAGE190_TYPE_DETAIL = (
    TABLES / "stage190_noaa_event_reranking_validation_type_detail.csv"
)
STAGE190_DETAIL = TABLES / "stage190_noaa_event_reranking_validation_detail.csv"
FEATURE = "noise_ratio"
FAR_GRID = [0.01, 0.02]
PRIMARY_FAR = 0.01
SELECTION_RULES = [
    "valid_f1",
    "valid_event_macro",
    "valid_nonstuck_event_macro",
]
ATTACK_TYPES = [
    "spike",
    "offset",
    "drift",
    "time_shift",
    "stuck",
    "gaussian_noise",
]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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


def valid_far_threshold(
    scores: np.ndarray,
    normal_mask: np.ndarray,
    max_far: float,
) -> float:
    normal_scores = np.asarray(scores, dtype=float)[normal_mask]
    normal_scores = normal_scores[np.isfinite(normal_scores)]
    if normal_scores.size == 0:
        return float("inf")
    return float(
        np.quantile(normal_scores, 1.0 - max_far, method="higher")
    )


def type_metrics(
    frame: pd.DataFrame,
    selected: np.ndarray,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    attack_type = frame["attack_type"].astype(str).to_numpy()
    attack = frame["is_attack"].astype(bool).to_numpy()
    for current_type in ATTACK_TYPES:
        current = attack & (attack_type == current_type)
        if not current.any():
            continue
        work = frame.loc[current, ["event_id"]].copy()
        work["selected"] = selected[current]
        output[current_type] = {
            "attack_rows": int(current.sum()),
            "selected_attack_rows": int((selected & current).sum()),
            "attack_recall": float(
                (selected & current).sum() / current.sum()
            ),
            "events": int(work["event_id"].nunique()),
            "event_coverage": float(
                work.groupby("event_id")["selected"].any().mean()
            ),
        }
    return output


def load_stage190_selection() -> pd.DataFrame:
    selection = pd.read_csv(STAGE190_SELECTION)
    required = {
        "fold",
        "seed",
        "max_far",
        "selection_rule",
        "window",
        "aggregation",
        "base_weight",
        "candidate_weight",
        "threshold_valid",
    }
    missing = required.difference(selection.columns)
    if missing:
        raise ValueError(f"stage190 selection is missing columns: {sorted(missing)}")
    selection = selection[selection["max_far"].isin(FAR_GRID)].copy()
    if selection.empty:
        raise ValueError("no stage190 selection rows at the requested FAR grid")
    return selection


def load_stage190_type_detail() -> pd.DataFrame:
    type_detail = pd.read_csv(STAGE190_TYPE_DETAIL)
    required = {
        "fold",
        "seed",
        "max_far",
        "selection_rule",
        "attack_type",
        "attack_rows",
        "selected_attack_rows",
        "attack_recall",
        "events",
        "event_coverage",
    }
    missing = required.difference(type_detail.columns)
    if missing:
        raise ValueError(
            f"stage190 type detail is missing columns: {sorted(missing)}"
        )
    type_detail = type_detail[type_detail["max_far"].isin(FAR_GRID)].copy()
    if type_detail.empty:
        raise ValueError("no stage190 type rows at the requested FAR grid")
    return type_detail


def baseline_type_metrics() -> pd.DataFrame:
    scores = pd.read_parquet(
        SCORE_DETAIL,
        columns=[
            "fold",
            "seed",
            "feature",
            "split",
            "is_attack",
            "attack_type",
            "event_id",
            "base_score",
        ],
        filters=[("feature", "==", FEATURE)],
    )
    if scores.empty:
        raise ValueError(f"no stage187 scores found for {FEATURE!r}")
    rows: list[dict[str, object]] = []
    for (fold, seed), group in scores.groupby(["fold", "seed"], sort=True):
        valid = group["split"].eq("valid").to_numpy()
        test = group["split"].eq("test").to_numpy()
        if not valid.any() or not test.any():
            raise ValueError("baseline requires valid and test rows")
        y_valid = group.loc[valid, "is_attack"].astype(bool).to_numpy()
        valid_normal = ~y_valid
        base_valid = group.loc[valid, "base_score"].to_numpy(dtype=float)
        base_test = group.loc[test, "base_score"].to_numpy(dtype=float)
        test_frame = group.loc[test].reset_index(drop=True)
        for max_far in FAR_GRID:
            threshold = valid_far_threshold(
                base_valid, valid_normal, max_far
            )
            baseline_pred = base_test >= threshold
            for attack_type, metrics in type_metrics(
                test_frame, baseline_pred
            ).items():
                rows.append(
                    {
                        "fold": fold,
                        "seed": int(seed),
                        "max_far": float(max_far),
                        "attack_type": attack_type,
                        "baseline_attack_rows": metrics["attack_rows"],
                        "baseline_selected_attack_rows": metrics[
                            "selected_attack_rows"
                        ],
                        "baseline_attack_recall": metrics["attack_recall"],
                        "baseline_events": metrics["events"],
                        "baseline_event_coverage": metrics["event_coverage"],
                    }
                )
    return pd.DataFrame(rows)


def build_pair_detail(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["fold", "seed", "selection_rule", "max_far", "attack_type"]
    baseline = baseline.copy()
    baseline["selection_rule"] = np.nan
    baseline_rows: list[pd.DataFrame] = []
    for rule in sorted(candidate["selection_rule"].unique()):
        part = baseline.copy()
        part["selection_rule"] = rule
        baseline_rows.append(part)
    baseline = pd.concat(baseline_rows, ignore_index=True)
    merged = candidate.merge(
        baseline,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("candidate and baseline attack-type rows do not align")
    merged = merged.drop(columns=["_merge"])
    merged["delta_selected_attack_rows"] = (
        merged["selected_attack_rows"]
        - merged["baseline_selected_attack_rows"]
    )
    merged["delta_attack_recall"] = (
        merged["attack_recall"] - merged["baseline_attack_recall"]
    )
    merged["delta_event_coverage"] = (
        merged["event_coverage"] - merged["baseline_event_coverage"]
    )
    return merged


def aggregate_pair_detail(detail: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        detail.groupby(
            ["selection_rule", "max_far", "attack_type"], sort=True
        )
        .agg(
            fold_seed_pairs=("fold", "size"),
            attack_rows=("attack_rows", "sum"),
            candidate_selected_attack_rows=(
                "selected_attack_rows",
                "sum",
            ),
            baseline_selected_attack_rows=(
                "baseline_selected_attack_rows",
                "sum",
            ),
            mean_candidate_attack_recall=("attack_recall", "mean"),
            mean_baseline_attack_recall=(
                "baseline_attack_recall",
                "mean",
            ),
            mean_delta_attack_recall=("delta_attack_recall", "mean"),
            mean_candidate_event_coverage=("event_coverage", "mean"),
            mean_baseline_event_coverage=(
                "baseline_event_coverage",
                "mean",
            ),
            mean_delta_event_coverage=("delta_event_coverage", "mean"),
            positive_recall_pairs=(
                "delta_attack_recall",
                lambda values: int((values > 0).sum()),
            ),
            negative_recall_pairs=(
                "delta_attack_recall",
                lambda values: int((values < 0).sum()),
            ),
            positive_coverage_pairs=(
                "delta_event_coverage",
                lambda values: int((values > 0).sum()),
            ),
            negative_coverage_pairs=(
                "delta_event_coverage",
                lambda values: int((values < 0).sum()),
            ),
        )
        .reset_index()
    )
    aggregate["pooled_candidate_attack_recall"] = (
        aggregate["candidate_selected_attack_rows"]
        / aggregate["attack_rows"]
    )
    aggregate["pooled_baseline_attack_recall"] = (
        aggregate["baseline_selected_attack_rows"]
        / aggregate["attack_rows"]
    )
    aggregate["pooled_delta_attack_recall"] = (
        aggregate["pooled_candidate_attack_recall"]
        - aggregate["pooled_baseline_attack_recall"]
    )
    return aggregate


def aggregate_settings(selection: pd.DataFrame) -> pd.DataFrame:
    setting = (
        selection.groupby(
            [
                "selection_rule",
                "max_far",
                "window",
                "aggregation",
                "base_weight",
                "candidate_weight",
            ],
            sort=True,
        )
        .size()
        .reset_index(name="selected_count")
    )
    totals = (
        selection.groupby(["selection_rule", "max_far"], sort=True)
        .size()
        .reset_index(name="fold_seed_pairs")
    )
    setting = setting.merge(
        totals,
        on=["selection_rule", "max_far"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    setting["selection_share"] = (
        setting["selected_count"] / setting["fold_seed_pairs"]
    )
    return setting


def setting_consensus(selection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (rule, max_far), group in selection.groupby(
        ["selection_rule", "max_far"], sort=True
    ):
        counts = (
            group.groupby(
                [
                    "window",
                    "aggregation",
                    "base_weight",
                    "candidate_weight",
                ],
                sort=True,
            )
            .size()
            .sort_values(ascending=False)
        )
        top = counts.index[0]
        rows.append(
            {
                "selection_rule": rule,
                "max_far": float(max_far),
                "fold_seed_pairs": int(len(group)),
                "distinct_settings": int(len(counts)),
                "consensus_window": int(top[0]),
                "consensus_aggregation": str(top[1]),
                "consensus_base_weight": float(top[2]),
                "consensus_candidate_weight": float(top[3]),
                "consensus_count": int(counts.iloc[0]),
                "consensus_share": float(counts.iloc[0] / len(group)),
            }
        )
    return pd.DataFrame(rows)


def fold_stability(selection: pd.DataFrame) -> pd.DataFrame:
    return (
        selection.groupby(
            ["selection_rule", "max_far", "fold"], sort=True
        )
        .agg(
            fold_seed_pairs=("seed", "size"),
            distinct_settings=(
                "aggregation",
                lambda values: int(
                    selection.loc[values.index]
                    .assign(
                        _setting=selection.loc[values.index, "window"].astype(str)
                        + "|"
                        + selection.loc[values.index, "aggregation"].astype(str)
                        + "|"
                        + selection.loc[
                            values.index, "candidate_weight"
                        ].astype(str)
                    )["_setting"]
                    .nunique()
                ),
            ),
            mean_candidate_weight=("candidate_weight", "mean"),
            mean_window=("window", "mean"),
        )
        .reset_index()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=STAGE190_SELECTION)
    parser.add_argument("--type-detail", type=Path, default=STAGE190_TYPE_DETAIL)
    parser.add_argument("--score-detail", type=Path, default=SCORE_DETAIL)
    args = parser.parse_args(argv)
    prefix = "stage191_noaa_reranking_focused_diagnostics"

    selection = pd.read_csv(args.selection)
    selection = selection[selection["max_far"].isin(FAR_GRID)].copy()
    if selection.empty:
        raise ValueError("no stage190 selection rows at the requested FAR grid")
    candidate = pd.read_csv(args.type_detail)
    candidate = candidate[candidate["max_far"].isin(FAR_GRID)].copy()
    if candidate.empty:
        raise ValueError("no stage190 type rows at the requested FAR grid")

    baseline = baseline_type_metrics()
    detail = build_pair_detail(candidate, baseline)
    type_aggregate = aggregate_pair_detail(detail)
    settings = aggregate_settings(selection)
    consensus = setting_consensus(selection)
    by_fold = fold_stability(selection)

    overall = pd.read_csv(STAGE190_DETAIL)
    overall = overall[overall["max_far"].isin(FAR_GRID)].copy()
    overall = overall[
        [
            "fold",
            "seed",
            "selection_rule",
            "max_far",
            "test_f1",
            "test_recall",
            "test_precision",
            "test_auprc",
            "test_false_alarm_rate",
            "test_event_macro",
            "test_nonstuck_event_macro",
            "baseline_test_f1",
            "baseline_test_recall",
            "baseline_test_precision",
            "baseline_test_auprc",
            "baseline_test_false_alarm_rate",
            "baseline_test_event_macro",
            "baseline_test_nonstuck_event_macro",
        ]
    ].copy()
    overall["delta_test_f1"] = overall["test_f1"] - overall["baseline_test_f1"]
    overall["delta_test_recall"] = (
        overall["test_recall"] - overall["baseline_test_recall"]
    )
    overall["delta_test_false_alarm_rate"] = (
        overall["test_false_alarm_rate"]
        - overall["baseline_test_false_alarm_rate"]
    )
    overall["delta_test_event_macro"] = (
        overall["test_event_macro"] - overall["baseline_test_event_macro"]
    )
    overall["delta_test_nonstuck_event_macro"] = (
        overall["test_nonstuck_event_macro"]
        - overall["baseline_test_nonstuck_event_macro"]
    )
    overall_aggregate = (
        overall.groupby(["selection_rule", "max_far"], sort=True)
        .agg(
            fold_seed_pairs=("fold", "size"),
            mean_delta_test_f1=("delta_test_f1", "mean"),
            positive_f1_pairs=(
                "delta_test_f1",
                lambda values: int((values > 0).sum()),
            ),
            negative_f1_pairs=(
                "delta_test_f1",
                lambda values: int((values < 0).sum()),
            ),
            mean_delta_test_recall=("delta_test_recall", "mean"),
            mean_delta_test_false_alarm_rate=(
                "delta_test_false_alarm_rate",
                "mean",
            ),
            mean_delta_test_event_macro=(
                "delta_test_event_macro",
                "mean",
            ),
            mean_delta_test_nonstuck_event_macro=(
                "delta_test_nonstuck_event_macro",
                "mean",
            ),
            positive_nonstuck_pairs=(
                "delta_test_nonstuck_event_macro",
                lambda values: int((values > 0).sum()),
            ),
            negative_nonstuck_pairs=(
                "delta_test_nonstuck_event_macro",
                lambda values: int((values < 0).sum()),
            ),
        )
        .reset_index()
    )

    detail.to_csv(
        TABLES / f"{prefix}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_aggregate.to_csv(
        TABLES / f"{prefix}_type_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    settings.to_csv(
        TABLES / f"{prefix}_settings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    consensus.to_csv(
        TABLES / f"{prefix}_consensus.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_fold.to_csv(
        TABLES / f"{prefix}_fold_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_aggregate.to_csv(
        TABLES / f"{prefix}_overall_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段190的valid-only选择和test评价结果，不重新训练、"
            "不重新选择窗口、聚合、权重或阈值；基线阈值只由阶段187"
            "valid正常样本按固定FAR上限重算；本阶段只诊断1%/2%FAR下"
            "的配置集中度、攻击类型增益和stuck代价；test只评价，"
            "不参与任何选择；五种子仅用于不确定性描述；"
            "NOAA攻击为透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [
            STAGE190_SELECTION.name,
            STAGE190_TYPE_DETAIL.name,
            STAGE190_DETAIL.name,
            SCORE_DETAIL.name,
        ],
        "feature": FEATURE,
        "far_grid": FAR_GRID,
        "primary_far": PRIMARY_FAR,
        "selection_rules": SELECTION_RULES,
        "attack_types": ATTACK_TYPES,
        "detail_rows": int(len(detail)),
        "type_aggregate_rows": int(len(type_aggregate)),
        "settings_rows": int(len(settings)),
        "consensus_rows": int(len(consensus)),
        "fold_stability_rows": int(len(by_fold)),
        "overall_aggregate_rows": int(len(overall_aggregate)),
        "type_aggregate": clean_records(type_aggregate),
        "consensus": clean_records(consensus),
        "overall_aggregate": clean_records(overall_aggregate),
        "limitations": [
            "只诊断阶段190已选择的配置，不新增候选搜索。",
            "攻击类型指标来自透明受控注入，不代表真实攻击分布。",
            "折种子配对不是独立样本，正负计数仅用于稳定性描述。",
            "本阶段不替换冻结主模型，也不把诊断结果写成部署增益。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段191 NOAA事件级重排聚焦诊断",
        "",
        "复用阶段190选择；只诊断配置集中度、攻击类型增益和stuck代价，不新增选择。",
        "",
        "## 配置共识",
        "",
        "| 选择规则 | FAR | 折种子数 | 不同配置数 | 共识窗口 | 共识聚合 | 共识候选权重 | 共识占比 |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in consensus.to_dict(orient="records"):
        lines.append(
            f"| {row['selection_rule']} | {row['max_far']:.3f} | "
            f"{int(row['fold_seed_pairs'])} | {int(row['distinct_settings'])} | "
            f"{int(row['consensus_window'])} | {row['consensus_aggregation']} | "
            f"{row['consensus_candidate_weight']:.2f} | "
            f"{row['consensus_share']:.4f} |"
        )
    lines += [
        "",
        "## 攻击类型配对差值",
        "",
        "| 选择规则 | FAR | 攻击类型 | 候选Recall | 基线Recall | Recall差 | 候选事件覆盖 | 基线事件覆盖 | 覆盖差 | 正/负覆盖折种子 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in type_aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['selection_rule']} | {row['max_far']:.3f} | "
            f"{row['attack_type']} | {row['mean_candidate_attack_recall']:.6f} | "
            f"{row['mean_baseline_attack_recall']:.6f} | "
            f"{row['mean_delta_attack_recall']:+.6f} | "
            f"{row['mean_candidate_event_coverage']:.6f} | "
            f"{row['mean_baseline_event_coverage']:.6f} | "
            f"{row['mean_delta_event_coverage']:+.6f} | "
            f"{int(row['positive_coverage_pairs'])}/"
            f"{int(row['negative_coverage_pairs'])} |"
        )
    lines += [
        "",
        "## 总体配对",
        "",
        "| 选择规则 | FAR | F1差 | F1正/负 | Recall差 | FAR差 | 事件覆盖差 | 非stuck覆盖差 | 非stuck正/负 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['selection_rule']} | {row['max_far']:.3f} | "
            f"{row['mean_delta_test_f1']:+.6f} | "
            f"{int(row['positive_f1_pairs'])}/"
            f"{int(row['negative_f1_pairs'])} | "
            f"{row['mean_delta_test_recall']:+.6f} | "
            f"{row['mean_delta_test_false_alarm_rate']:+.6f} | "
            f"{row['mean_delta_test_event_macro']:+.6f} | "
            f"{row['mean_delta_test_nonstuck_event_macro']:+.6f} | "
            f"{int(row['positive_nonstuck_pairs'])}/"
            f"{int(row['negative_nonstuck_pairs'])} |"
        )
    lines += [
        "",
        f"攻击类型明细={len(detail)}；类型聚合={len(type_aggregate)}；"
        f"配置行={len(settings)}；共识行={len(consensus)}。",
        "本阶段只解释阶段190，不替换冻结主模型。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": len(detail),
                "type_aggregate_rows": len(type_aggregate),
                "settings_rows": len(settings),
                "consensus_rows": len(consensus),
                "fold_stability_rows": len(by_fold),
                "overall_aggregate_rows": len(overall_aggregate),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
