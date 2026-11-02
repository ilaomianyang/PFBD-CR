from __future__ import annotations

"""Stage 182: validation-selected candidate features for non-stuck misses.

Stage 181 showed that the frozen structural features respond strongly to
stuck events, weakly to spikes, and not at all to gaussian noise, drift,
offset, or time shifts under the pre-fixed low-FAR budgets.

This stage adds three causal, label-free candidate features:

* local residual volatility;
* adjacent-difference burst;
* rolling variance jump.

The candidates are evaluated with the same rolling folds and seeds, but the
feature weight and threshold are selected only on each fold's validation
split.  Test labels are used only for final evaluation.  The existing frozen
model is retained as the reference; this stage is a candidate audit, not a
replacement.

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
METHOD = "TrustMeteo-QC frozen expanded"
ATTACK_TYPES = [
    "spike",
    "offset",
    "drift",
    "time_shift",
    "stuck",
    "gaussian_noise",
]
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]
FEATURE_NAMES = [
    "local_residual_volatility",
    "difference_burst",
    "variance_jump",
]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components  # noqa: E402
from stage151_noaa_attack_intensity_sensitivity import build_fold_scaled  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def robust_positive_scale(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return max(1.4826 * mad, 0.25)


def candidate_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[["station_key", "timestamp_utc", "observed_value"]].copy()
    work["_value"] = pd.to_numeric(work["observed_value"], errors="coerce")
    work["_timestamp"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
    work = work.sort_values(["station_key", "_timestamp"], kind="stable")
    out = pd.DataFrame(index=frame.index, columns=FEATURE_NAMES, dtype=float)
    for _, group in work.groupby("station_key", sort=False):
        values = group["_value"]
        centered = values - values.rolling(24, min_periods=4).median()
        local_residual = centered.abs()
        difference = values.diff().abs()
        rolling_var = values.rolling(12, min_periods=4).var()
        previous_var = rolling_var.shift(12)
        variance_jump = (rolling_var - previous_var).clip(lower=0.0)
        part = pd.DataFrame(index=group.index)
        part["local_residual_volatility"] = local_residual.rolling(
            6, min_periods=2
        ).std()
        part["difference_burst"] = difference.rolling(6, min_periods=2).max()
        part["variance_jump"] = variance_jump.rolling(6, min_periods=2).max()
        out.loc[group.index, FEATURE_NAMES] = part[
            FEATURE_NAMES
        ].to_numpy(dtype=float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def normalize_candidates(
    features: pd.DataFrame,
    train_mask: np.ndarray,
    normal_train: np.ndarray,
) -> pd.DataFrame:
    normalized = pd.DataFrame(index=features.index, columns=FEATURE_NAMES, dtype=float)
    ref = features.loc[normal_train]
    for column in FEATURE_NAMES:
        center = float(ref[column].median())
        scale = robust_positive_scale(ref[column])
        normalized[column] = np.maximum(
            (features[column].to_numpy(dtype=float) - center) / scale,
            0.0,
        )
    return normalized


def select_candidate(
    frame: pd.DataFrame,
    base_score: np.ndarray,
    candidate_score: np.ndarray,
) -> tuple[float, float, float]:
    valid = frame["split"].eq("valid").to_numpy()
    y_valid = frame.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    base_valid = base_score[valid]
    candidate_valid = candidate_score[valid]
    best: tuple[float, float, float] | None = None
    for weight in WEIGHT_GRID:
        score = (1.0 - weight) * base_valid + weight * candidate_valid
        threshold, f1 = _best_threshold(y_valid, score)
        far = float((score[~y_valid] >= threshold).mean()) if (~y_valid).any() else 1.0
        candidate = (f1, -far, -weight)
        if best is None or candidate > best:
            best = candidate
            selected = (float(weight), float(threshold), float(f1))
    if best is None:
        return 0.0, 1.0, 0.0
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild summaries from persisted row detail.",
    )
    args = parser.parse_args(argv)
    prefix = "stage182_noaa_non_stuck_candidate_features"
    if args.summarize_existing:
        row_detail = pd.read_parquet(TABLES / f"{prefix}_row_detail.parquet")
        return summarize_and_write(row_detail)

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
    row_frames: list[pd.DataFrame] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
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
            raw_candidate = normalized.max(axis=1).to_numpy(dtype=float)
            candidate_score = raw_candidate
            for feature in FEATURE_NAMES:
                current = normalized[feature].to_numpy(dtype=float)
                selected = select_candidate(scored, base_score, current)
                weight, threshold, valid_f1 = selected
                test_mask = scored["split"].eq("test").to_numpy()
                y_test = scored.loc[test_mask, "combined_is_attack"].astype(bool).to_numpy()
                test_base = base_score[test_mask]
                test_candidate = current[test_mask]
                test_score = (1.0 - weight) * test_base + weight * test_candidate
                test_pred = test_score >= threshold
                metrics = _metrics(y_test, test_pred, test_score)
                test_frame = scored.loc[test_mask].reset_index(drop=True).copy()
                test_frame["fold"] = fold
                test_frame["seed"] = int(seed)
                test_frame["feature"] = feature
                test_frame["feature_weight"] = weight
                test_frame["candidate_threshold"] = threshold
                test_frame["candidate_valid_f1"] = valid_f1
                test_frame["base_score"] = test_base
                test_frame["candidate_score"] = test_candidate
                test_frame["combined_score"] = test_score
                test_frame["selected"] = test_pred
                test_frame["is_attack"] = test_frame[
                    "combined_is_attack"
                ].astype(bool)
                test_frame["attack_type"] = test_frame[
                    "combined_attack_type"
                ].astype(str)
                test_frame["event_id"] = test_frame["combined_event_id"].astype(str)
                test_frame["test_f1"] = float(metrics["f1"])
                test_frame["test_recall"] = float(metrics["recall"])
                test_frame["test_precision"] = float(metrics["precision"])
                test_frame["test_auprc"] = float(metrics["average_precision"])
                test_frame["test_false_alarm_rate"] = float(
                    test_pred[~y_test].mean()
                ) if (~y_test).any() else float("nan")
                row_frames.append(test_frame)
        print(
            json.dumps(
                {"fold": fold, "features": FEATURE_NAMES, "status": "completed"},
                ensure_ascii=False,
            ),
            flush=True,
        )
    row_detail = pd.concat(row_frames, ignore_index=True)
    return summarize_and_write(row_detail)


def summarize_and_write(row_detail: pd.DataFrame) -> int:
    prefix = "stage182_noaa_non_stuck_candidate_features"
    row_detail.to_parquet(TABLES / f"{prefix}_row_detail.parquet", index=False)
    type_rows: list[dict[str, object]] = []
    for (fold, seed, feature), group in row_detail.groupby(
        ["fold", "seed", "feature"], sort=True
    ):
        for attack_type in ATTACK_TYPES:
            current = group[group["attack_type"].eq(attack_type)]
            if current.empty:
                continue
            selected = current["selected"].astype(bool)
            type_rows.append(
                {
                    "fold": fold,
                    "seed": int(seed),
                    "feature": feature,
                    "attack_type": attack_type,
                    "rows": int(len(current)),
                    "selected_rows": int(selected.sum()),
                    "recall": float(selected.mean()),
                    "event_coverage": float(
                        current.groupby("event_id")["selected"].any().mean()
                    ),
                    "weight": float(current["feature_weight"].iloc[0]),
                    "threshold": float(current["candidate_threshold"].iloc[0]),
                    "valid_f1": float(current["candidate_valid_f1"].iloc[0]),
                    "test_f1": float(current["test_f1"].iloc[0]),
                    "test_recall": float(current["test_recall"].iloc[0]),
                    "test_precision": float(current["test_precision"].iloc[0]),
                    "test_auprc": float(current["test_auprc"].iloc[0]),
                    "test_false_alarm_rate": float(
                        current["test_false_alarm_rate"].iloc[0]
                    ),
                }
            )
    detail = pd.DataFrame(type_rows)
    aggregate = (
        detail.groupby(["feature", "attack_type"], sort=True)
        .agg(
            rows=("rows", "sum"),
            selected_rows=("selected_rows", "sum"),
            mean_event_coverage=("event_coverage", "mean"),
            mean_weight=("weight", "mean"),
            mean_valid_f1=("valid_f1", "mean"),
            mean_test_f1=("test_f1", "mean"),
            mean_test_recall=("test_recall", "mean"),
            mean_test_precision=("test_precision", "mean"),
            mean_test_auprc=("test_auprc", "mean"),
            mean_test_false_alarm_rate=("test_false_alarm_rate", "mean"),
        )
        .reset_index()
    )
    aggregate["pooled_recall"] = (
        aggregate["selected_rows"] / aggregate["rows"]
    )
    detail.to_csv(
        TABLES / f"{prefix}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        TABLES / f"{prefix}_type_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段178冻结扩展评分设置和同一透明受控注入；"
            "新增局部残差波动、相邻差分突变、滚动方差跃迁三类因果候选特征；"
            "每个特征权重和阈值只在valid选择，test只评价；"
            "不重新训练、不根据test选择特征或权重；候选仅作审计，"
            "不替换冻结主模型；五种子仅用于不确定性描述；"
            "NOAA攻击为透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [NOAA.name, AUDIT.name, FROZEN.name],
        "method": METHOD,
        "features": FEATURE_NAMES,
        "weight_grid": WEIGHT_GRID,
        "attack_types": ATTACK_TYPES,
        "row_detail_rows": int(len(row_detail)),
        "type_detail_rows": int(len(detail)),
        "type_aggregate_rows": int(len(aggregate)),
        "type_aggregate": aggregate.to_dict(orient="records"),
        "limitations": [
            "候选特征在valid上选择权重和阈值，test只评价，不能据test再选特征。",
            "候选审计不替换现有冻结主模型，也不证明跨数据源普适。",
            "攻击仍为透明受控注入，不代表真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段182 NOAA非stuck候选特征审计",
        "",
        "复用冻结扩展评分设置；新增三类因果候选特征，权重和阈值只在valid选择，test只评价。",
        "",
        "| 特征 | 攻击类型 | 行Recall | 事件覆盖 | 平均权重 | valid F1 | test F1 | test FAR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['feature']} | {row['attack_type']} | "
            f"{row['pooled_recall']:.6f} | {row['mean_event_coverage']:.6f} | "
            f"{row['mean_weight']:.2f} | {row['mean_valid_f1']:.6f} | "
            f"{row['mean_test_f1']:.6f} | "
            f"{row['mean_test_false_alarm_rate']:.6f} |"
        )
    lines += [
        "",
        f"逐行={len(row_detail)}；类型明细={len(detail)}；类型聚合={len(aggregate)}。",
        "候选特征只在valid选择参数；所有攻击均为透明受控注入。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "row_detail_rows": len(row_detail),
                "type_detail_rows": len(detail),
                "type_aggregate_rows": len(aggregate),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
