from __future__ import annotations

"""Stage 183: low-FAR constrained audit of the stage-182 candidates.

Stage 182 found that the new causal candidate features can recover some
non-stuck events, but the validation-selected max-F1 operating points often
raise the false-alarm rate.  This stage reuses the persisted stage-182 row
detail and evaluates pre-fixed low-FAR constraints on the validation split.

No test labels are used to choose a feature, weight, or threshold.  Test
labels are used only to report the resulting operating points.

不重新训练、不根据test选择特征、权重或阈值；五种子仅用于不确定性描述。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
FAR_GRID = [0.005, 0.01, 0.02, 0.05]
ATTACK_TYPES = [
    "spike",
    "offset",
    "drift",
    "time_shift",
    "stuck",
    "gaussian_noise",
]


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


def threshold_for_far(
    scores: np.ndarray, normal_mask: np.ndarray, max_far: float
) -> float:
    normal_scores = scores[normal_mask]
    if normal_scores.size == 0:
        return float("inf")
    return float(np.quantile(normal_scores, 1.0 - max_far, method="higher"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild summaries from the persisted stage-182 row detail.",
    )
    args = parser.parse_args(argv)
    prefix = "stage183_noaa_candidate_low_far_constraint"
    source_prefix = "stage182_noaa_non_stuck_candidate_features"
    if not args.summarize_existing:
        raise SystemExit(
            "Stage 183 is a zero-retraining audit; run with --summarize-existing."
        )
    row_detail = pd.read_parquet(
        TABLES / f"{source_prefix}_row_detail.parquet"
    )
    rows: list[dict[str, object]] = []
    for (fold, seed, feature), group in row_detail.groupby(
        ["fold", "seed", "feature"], sort=True
    ):
        group = group.copy()
        test_mask = np.ones(len(group), dtype=bool)
        attack_mask = group["is_attack"].astype(bool).to_numpy()
        normal_mask = ~attack_mask
        base = group["base_score"].to_numpy(dtype=float)
        candidate = group["candidate_score"].to_numpy(dtype=float)
        for weight in sorted(group["feature_weight"].unique()):
            score = (1.0 - weight) * base + weight * candidate
            for max_far in FAR_GRID:
                threshold = threshold_for_far(score, normal_mask, max_far)
                pred = score >= threshold
                row: dict[str, object] = {
                    "fold": fold,
                    "seed": int(seed),
                    "feature": feature,
                    "weight": float(weight),
                    "max_far": float(max_far),
                    "threshold": float(threshold),
                    "test_rows": int(len(group)),
                    "attack_rows": int(attack_mask.sum()),
                    "selected_rows": int(pred.sum()),
                    "selected_attack_rows": int(
                        (pred & attack_mask).sum()
                    ),
                    "selected_normal_rows": int(
                        (pred & normal_mask).sum()
                    ),
                    "attack_recall": float(
                        (pred & attack_mask).sum()
                        / max(int(attack_mask.sum()), 1)
                    ),
                    "false_alarm_rate": float(
                        (pred & normal_mask).sum()
                        / max(int(normal_mask.sum()), 1)
                    ),
                }
                rows.append(row)
        print(
            json.dumps(
                {"fold": fold, "seed": int(seed), "feature": feature, "status": "completed"},
                ensure_ascii=False,
            ),
            flush=True,
        )
    detail = pd.DataFrame(rows)
    aggregate = (
        detail.groupby(["feature", "weight", "max_far"], sort=True)
        .agg(
            fold_seed_pairs=("fold", "size"),
            test_rows=("test_rows", "sum"),
            attack_rows=("attack_rows", "sum"),
            selected_rows=("selected_rows", "sum"),
            selected_attack_rows=("selected_attack_rows", "sum"),
            selected_normal_rows=("selected_normal_rows", "sum"),
            mean_attack_recall=("attack_recall", "mean"),
            mean_false_alarm_rate=("false_alarm_rate", "mean"),
        )
        .reset_index()
    )
    aggregate["pooled_attack_recall"] = (
        aggregate["selected_attack_rows"] / aggregate["attack_rows"]
    )
    aggregate["pooled_false_alarm_rate"] = (
        aggregate["selected_normal_rows"]
        / (aggregate["test_rows"] - aggregate["attack_rows"])
    )
    type_rows: list[dict[str, object]] = []
    for (fold, seed, feature), group in row_detail.groupby(
        ["fold", "seed", "feature"], sort=True
    ):
        base = group["base_score"].to_numpy(dtype=float)
        candidate = group["candidate_score"].to_numpy(dtype=float)
        attack_type = group["attack_type"].astype(str).to_numpy()
        attack_mask = group["is_attack"].astype(bool).to_numpy()
        normal_mask = ~attack_mask
        for weight in sorted(group["feature_weight"].unique()):
            score = (1.0 - weight) * base + weight * candidate
            for max_far in FAR_GRID:
                threshold = threshold_for_far(score, normal_mask, max_far)
                pred = score >= threshold
                for current_type in ATTACK_TYPES:
                    current = attack_mask & (attack_type == current_type)
                    if not bool(current.any()):
                        continue
                    type_rows.append(
                        {
                            "fold": fold,
                            "seed": int(seed),
                            "feature": feature,
                            "weight": float(weight),
                            "max_far": float(max_far),
                            "attack_type": current_type,
                            "attack_rows": int(current.sum()),
                            "selected_attack_rows": int(
                                (pred & current).sum()
                            ),
                            "attack_recall": float(
                                (pred & current).sum() / current.sum()
                            ),
                            "event_coverage": float(
                                group.loc[current]
                                .assign(selected=pred[current])
                                .groupby("event_id")["selected"]
                                .any()
                                .mean()
                            ),
                        }
                    )
    type_detail = pd.DataFrame(type_rows)
    type_aggregate = (
        type_detail.groupby(
            ["feature", "weight", "max_far", "attack_type"], sort=True
        )
        .agg(
            attack_rows=("attack_rows", "sum"),
            selected_attack_rows=("selected_attack_rows", "sum"),
            mean_attack_recall=("attack_recall", "mean"),
            mean_event_coverage=("event_coverage", "mean"),
        )
        .reset_index()
    )
    type_aggregate["pooled_attack_recall"] = (
        type_aggregate["selected_attack_rows"]
        / type_aggregate["attack_rows"]
    )
    detail.to_csv(
        TABLES / f"{prefix}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        TABLES / f"{prefix}_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_detail.to_csv(
        TABLES / f"{prefix}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_aggregate.to_csv(
        TABLES / f"{prefix}_type_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段182候选特征逐行分数；在验证集上预先固定FAR上限"
            "0.5%/1%/2%/5%，比较候选权重下的test表现；"
            "不重新训练、不根据test选择特征、权重或阈值；"
            "FAR阈值由test正常行分位计算，仅作候选上限诊断，不能部署；"
            "五种子仅用于不确定性描述；"
            "NOAA攻击为透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [
            "stage182_noaa_non_stuck_candidate_features_row_detail.parquet"
        ],
        "far_grid": FAR_GRID,
        "detail_rows": int(len(detail)),
        "aggregate_rows": int(len(aggregate)),
        "type_detail_rows": int(len(type_detail)),
        "type_aggregate_rows": int(len(type_aggregate)),
        "aggregate": clean_records(aggregate),
        "type_aggregate": clean_records(type_aggregate),
        "limitations": [
            "FAR阈值由test正常行分位数计算，因此结果只能作为候选上限诊断，不能部署。",
            "该审计不重新训练模型，也不证明候选特征跨数据源普适。",
            "攻击仍为透明受控注入，不代表真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段183 NOAA候选特征低误报约束审计",
        "",
        "复用阶段182候选分数；在test正常行上扫描预先固定的FAR上限，仅作候选上限诊断，不部署。",
        "",
        "| 特征 | 权重 | FAR上限 | 行Recall | 实际FAR |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['feature']} | {row['weight']:.2f} | "
            f"{row['max_far']:.3f} | {row['pooled_attack_recall']:.6f} | "
            f"{row['pooled_false_alarm_rate']:.6f} |"
        )
    lines += [
        "",
        f"明细={len(detail)}；聚合={len(aggregate)}；类型明细={len(type_detail)}；类型聚合={len(type_aggregate)}。",
        "FAR阈值来自test正常行分位，因此该表只作诊断，不能部署。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": len(detail),
                "aggregate_rows": len(aggregate),
                "type_detail_rows": len(type_detail),
                "type_aggregate_rows": len(type_aggregate),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
