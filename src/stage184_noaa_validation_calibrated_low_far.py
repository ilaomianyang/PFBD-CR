from __future__ import annotations

"""Stage 184: validation-calibrated low-FAR audit for NOAA candidates.

Stage 183 showed promising low-FAR diagnostics, but its thresholds were
computed from test-normal quantiles and are therefore not deployable.  This
stage reruns the same frozen NOAA rolling protocol and persists validation
and test candidate scores separately.  Every threshold is calibrated on the
validation split only; the test split is used only for final evaluation.

不重新训练、不根据test选择特征、权重或阈值；五种子仅用于不确定性描述。
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
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"
AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
FROZEN = TABLES / "stage141_noaa_frozen_external_validation_selected.csv"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
FEATURE_NAMES = [
    "local_residual_volatility",
    "difference_burst",
    "variance_jump",
]
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]
FAR_GRID = [0.005, 0.01, 0.02, 0.05]
ATTACK_TYPES = [
    "spike",
    "offset",
    "drift",
    "time_shift",
    "stuck",
    "gaussian_noise",
]
METHOD = "TrustMeteo-QC frozen expanded"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components  # noqa: E402
from stage151_noaa_attack_intensity_sensitivity import build_fold_scaled  # noqa: E402
from stage182_noaa_non_stuck_candidate_features import (  # noqa: E402
    candidate_features,
    fold_name,
    normalize_candidates,
)
from stage38_noaa_controlled_benchmark import score  # noqa: E402


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
    return float(np.quantile(normal_scores, 1.0 - max_far, method="higher"))


def summarize_and_write(score_detail: pd.DataFrame) -> int:
    prefix = "stage184_noaa_validation_calibrated_low_far"
    score_detail.to_parquet(TABLES / f"{prefix}_score_detail.parquet", index=False)

    rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    for (fold, seed, feature), group in score_detail.groupby(
        ["fold", "seed", "feature"], sort=True
    ):
        valid = group["split"].eq("valid").to_numpy()
        test = group["split"].eq("test").to_numpy()
        if not valid.any() or not test.any():
            continue
        y_valid = group.loc[valid, "is_attack"].astype(bool).to_numpy()
        y_test = group.loc[test, "is_attack"].astype(bool).to_numpy()
        base_valid = group.loc[valid, "base_score"].to_numpy(dtype=float)
        base_test = group.loc[test, "base_score"].to_numpy(dtype=float)
        candidate_valid = group.loc[valid, "candidate_score"].to_numpy(dtype=float)
        candidate_test = group.loc[test, "candidate_score"].to_numpy(dtype=float)
        valid_normal = ~y_valid
        for weight in WEIGHT_GRID:
            valid_score = (
                (1.0 - weight) * base_valid + weight * candidate_valid
            )
            test_score = (1.0 - weight) * base_test + weight * candidate_test
            for max_far in FAR_GRID:
                threshold = valid_far_threshold(
                    valid_score, valid_normal, max_far
                )
                valid_pred = valid_score >= threshold
                test_pred = test_score >= threshold
                metrics = _metrics(y_test, test_pred, test_score)
                valid_far = (
                    float(valid_pred[valid_normal].mean())
                    if valid_normal.any()
                    else float("nan")
                )
                test_far = (
                    float(test_pred[~y_test].mean())
                    if (~y_test).any()
                    else float("nan")
                )
                rows.append(
                    {
                        "fold": fold,
                        "seed": int(seed),
                        "feature": feature,
                        "weight": float(weight),
                        "max_far": float(max_far),
                        "threshold_valid": float(threshold),
                        "valid_rows": int(valid.sum()),
                        "valid_attack_rows": int(y_valid.sum()),
                        "valid_normal_rows": int(valid_normal.sum()),
                        "valid_selected_rows": int(valid_pred.sum()),
                        "valid_false_alarm_rate": valid_far,
                        "test_rows": int(test.sum()),
                        "test_attack_rows": int(y_test.sum()),
                        "test_selected_rows": int(test_pred.sum()),
                        "test_selected_attack_rows": int(
                            (test_pred & y_test).sum()
                        ),
                        "test_selected_normal_rows": int(
                            (test_pred & ~y_test).sum()
                        ),
                        "test_f1": float(metrics["f1"]),
                        "test_recall": float(metrics["recall"]),
                        "test_precision": float(metrics["precision"]),
                        "test_auprc": float(metrics["average_precision"]),
                        "test_false_alarm_rate": test_far,
                    }
                )
                test_frame = group.loc[test].copy()
                test_frame["selected"] = test_pred
                attack_type = test_frame["attack_type"].astype(str).to_numpy()
                for current_type in ATTACK_TYPES:
                    current = y_test & (attack_type == current_type)
                    if not bool(current.any()):
                        continue
                    event_ids = test_frame.loc[current, "event_id"].astype(str)
                    selected_events = (
                        test_frame.loc[current]
                        .assign(_selected=test_pred[current])
                        .groupby("event_id")["_selected"]
                        .any()
                    )
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
                                (test_pred & current).sum()
                            ),
                            "attack_recall": float(
                                (test_pred & current).sum() / current.sum()
                            ),
                            "events": int(event_ids.nunique()),
                            "event_coverage": float(selected_events.mean()),
                        }
                    )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "seed": int(seed),
                    "feature": feature,
                    "status": "completed",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    detail = pd.DataFrame(rows)
    type_detail = pd.DataFrame(type_rows)
    aggregate = (
        detail.groupby(["feature", "weight", "max_far"], sort=True)
        .agg(
            fold_seed_pairs=("fold", "size"),
            valid_rows=("valid_rows", "sum"),
            valid_attack_rows=("valid_attack_rows", "sum"),
            valid_normal_rows=("valid_normal_rows", "sum"),
            valid_selected_rows=("valid_selected_rows", "sum"),
            test_rows=("test_rows", "sum"),
            test_attack_rows=("test_attack_rows", "sum"),
            test_selected_rows=("test_selected_rows", "sum"),
            test_selected_attack_rows=("test_selected_attack_rows", "sum"),
            test_selected_normal_rows=("test_selected_normal_rows", "sum"),
            mean_valid_false_alarm_rate=("valid_false_alarm_rate", "mean"),
            mean_test_f1=("test_f1", "mean"),
            mean_test_recall=("test_recall", "mean"),
            mean_test_precision=("test_precision", "mean"),
            mean_test_auprc=("test_auprc", "mean"),
            mean_test_false_alarm_rate=("test_false_alarm_rate", "mean"),
        )
        .reset_index()
    )
    aggregate["pooled_test_recall"] = (
        aggregate["test_selected_attack_rows"] / aggregate["test_attack_rows"]
    )
    aggregate["pooled_test_false_alarm_rate"] = (
        aggregate["test_selected_normal_rows"]
        / (aggregate["test_rows"] - aggregate["test_attack_rows"])
    )
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

    baseline = detail[detail["weight"].eq(0.0)].set_index(
        ["fold", "seed", "feature", "max_far"]
    )
    paired_rows: list[dict[str, object]] = []
    for key, current in detail.set_index(
        ["fold", "seed", "feature", "max_far"]
    ).iterrows():
        if float(current["weight"]) == 0.0:
            continue
        ref = baseline.loc[key]
        for metric in [
            "test_f1",
            "test_recall",
            "test_precision",
            "test_auprc",
            "test_false_alarm_rate",
        ]:
            paired_rows.append(
                {
                    "fold": key[0],
                    "seed": int(key[1]),
                    "feature": key[2],
                    "max_far": float(key[3]),
                    "weight": float(current["weight"]),
                    "metric": metric,
                    "candidate_minus_baseline": float(
                        current[metric] - ref[metric]
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired_aggregate = (
        paired.groupby(["feature", "max_far", "weight", "metric"], sort=True)
        .agg(
            fold_seed_pairs=("candidate_minus_baseline", "size"),
            mean_diff=("candidate_minus_baseline", "mean"),
            std_diff=("candidate_minus_baseline", "std"),
            min_diff=("candidate_minus_baseline", "min"),
            max_diff=("candidate_minus_baseline", "max"),
        )
        .reset_index()
    )

    detail.to_csv(
        TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig"
    )
    aggregate.to_csv(
        TABLES / f"{prefix}_aggregate.csv", index=False, encoding="utf-8-sig"
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
    paired.to_csv(
        TABLES / f"{prefix}_paired_vs_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired_aggregate.to_csv(
        TABLES / f"{prefix}_paired_vs_baseline_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段182三类因果候选特征和阶段141冻结alpha/window；"
            "每个fold/seed只在valid正常样本上按预先固定的FAR上限"
            "0.5%/1%/2%/5%校准阈值；test只评价，不参与特征、权重、"
            "阈值或模型选择；不重新训练、不根据test选择特征或权重，"
            "不替换冻结主模型；五种子仅用于不确定性描述；"
            "NOAA攻击为透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [
            NOAA.name,
            AUDIT.name,
            FROZEN.name,
            "stage182_noaa_non_stuck_candidate_features_row_detail.parquet",
        ],
        "method": METHOD,
        "features": FEATURE_NAMES,
        "weight_grid": WEIGHT_GRID,
        "far_grid": FAR_GRID,
        "attack_types": ATTACK_TYPES,
        "score_detail_rows": int(len(score_detail)),
        "detail_rows": int(len(detail)),
        "aggregate_rows": int(len(aggregate)),
        "type_detail_rows": int(len(type_detail)),
        "type_aggregate_rows": int(len(type_aggregate)),
        "paired_rows": int(len(paired)),
        "paired_aggregate_rows": int(len(paired_aggregate)),
        "aggregate": clean_records(aggregate),
        "type_aggregate": clean_records(type_aggregate),
        "paired_aggregate": clean_records(paired_aggregate),
        "limitations": [
            "阈值只在valid正常样本上校准，test标签只用于最终评价。",
            "候选审计不替换冻结主模型，也不证明跨数据源普适。",
            "攻击仍为透明受控注入，不代表真实攻击日志。",
            "低FAR运行点可能因valid正常样本有限而阈值粗化，需结合样本量解释。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段184 NOAA验证集校准低误报审计",
        "",
        "复用阶段182候选特征和阶段141冻结alpha/window；阈值只在valid正常样本上校准，test只评价。",
        "",
        "| 特征 | 权重 | FAR上限 | valid FAR | test Recall | test FAR | test F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['feature']} | {row['weight']:.2f} | "
            f"{row['max_far']:.3f} | {row['mean_valid_false_alarm_rate']:.6f} | "
            f"{row['pooled_test_recall']:.6f} | "
            f"{row['pooled_test_false_alarm_rate']:.6f} | "
            f"{row['mean_test_f1']:.6f} |"
        )
    lines += [
        "",
        f"分数明细={len(score_detail)}；运行点={len(detail)}；类型聚合={len(type_aggregate)}。",
        "该阶段是验证集校准的候选审计；不替换冻结主模型，不代表真实攻击日志。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "score_detail_rows": len(score_detail),
                "detail_rows": len(detail),
                "aggregate_rows": len(aggregate),
                "type_aggregate_rows": len(type_aggregate),
                "paired_aggregate_rows": len(paired_aggregate),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild summaries from persisted stage-184 score detail.",
    )
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    prefix = "stage184_noaa_validation_calibrated_low_far"
    if args.summarize_existing:
        score_detail = pd.read_parquet(
            TABLES / f"{prefix}_score_detail.parquet"
        )
        return summarize_and_write(score_detail)

    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {
        (int(row["year"]), str(row["station_id"]))
        for row in audit["detail"]
        if row.get("eligible_for_extended_validation")
    }
    years = sorted(int(x) for x in source["year"].unique())
    stations = {
        station
        for station in source["station_id"].astype(str).unique()
        if all((year, station) in eligible for year in years)
    }
    frozen = (
        pd.read_csv(FROZEN)
        .loc[lambda x: x["method"].eq(METHOD)]
        .set_index(["fold", "seed"])
    )
    selected_folds = (
        [FOLDS[args.fold]]
        if args.fold is not None
        else FOLDS
    )
    selected_seeds = [args.seed] if args.seed is not None else SEEDS
    score_frames: list[pd.DataFrame] = []
    for train_years, valid_year, test_year in selected_folds:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in selected_seeds:
            alpha = float(frozen.loc[(fold, seed), "alpha_base"])
            window = int(frozen.loc[(fold, seed), "window"])
            frame = build_fold_scaled(
                source,
                seed,
                train_years,
                valid_year,
                test_year,
                stations,
                1.0,
            )
            scored, _ = score(frame)
            scored = add_components(scored)
            features = candidate_features(scored)
            train = scored["split"].eq("train").to_numpy()
            normal_train = train & (
                ~scored["combined_is_attack"].astype(bool).to_numpy()
            )
            normalized = normalize_candidates(features, train, normal_train)
            base = scored["base_fused_risk"].to_numpy(dtype=float)
            structural = scored["structural_risk"].to_numpy(dtype=float)
            raw_risk = alpha * base + (1.0 - alpha) * structural
            base_score = np.full(len(scored), np.nan, dtype=float)
            for split_name in ["train", "valid", "test"]:
                split_mask = scored["split"].eq(split_name).to_numpy()
                base_score[split_mask] = smooth_risk(
                    scored.loc[split_mask], raw_risk[split_mask], window
                )
            for feature in FEATURE_NAMES:
                keep = scored["split"].isin(["valid", "test"]).to_numpy()
                part = scored.loc[keep].copy()
                part["fold"] = fold
                part["seed"] = int(seed)
                part["feature"] = feature
                part["base_score"] = base_score[keep]
                part["candidate_score"] = normalized[feature].to_numpy(
                    dtype=float
                )[keep]
                part["is_attack"] = part["combined_is_attack"].astype(bool)
                part["attack_type"] = part["combined_attack_type"].astype(str)
                part["event_id"] = part["combined_event_id"].astype(str)
                score_frames.append(
                    part[
                        [
                            "fold",
                            "seed",
                            "feature",
                            "split",
                            "station_key",
                            "timestamp_utc",
                            "year",
                            "base_score",
                            "candidate_score",
                            "is_attack",
                            "attack_type",
                            "event_id",
                        ]
                    ]
                )
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "seed": int(seed),
                        "alpha_base": alpha,
                        "window": window,
                        "status": "scored",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    score_detail = pd.concat(score_frames, ignore_index=True)
    return summarize_and_write(score_detail)


if __name__ == "__main__":
    raise SystemExit(main())
