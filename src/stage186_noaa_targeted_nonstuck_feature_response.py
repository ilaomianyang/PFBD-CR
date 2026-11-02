from __future__ import annotations

"""Stage 186: targeted causal feature response audit for non-stuck misses.

Stage 184 and stage 185 showed that the three original candidate features
have useful but unstable signal outside stuck events.  This stage adds eight
targeted causal features aimed at level shifts, sustained drift, high-
frequency noise, short-lag breaks, and repeated/abrupt changes.

The audit uses only train-normal quantiles as fixed diagnostic thresholds.
Validation and test rows are reported for feature response only; no model,
feature weight, threshold, or selection rule is chosen from test.
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
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
FEATURE_NAMES = [
    "level_shift",
    "slope_6",
    "trend_consistency",
    "noise_ratio",
    "diff_zscore",
    "repeat_ratio",
    "second_diff_zscore",
    "drift_acceleration",
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

from stage151_noaa_attack_intensity_sensitivity import build_fold_scaled  # noqa: E402


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


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


def robust_scale(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return max(1.4826 * mad, 0.25)


def targeted_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[
        ["station_key", "timestamp_utc", "observed_value"]
    ].copy()
    work["_value"] = pd.to_numeric(
        work["observed_value"], errors="coerce"
    )
    work["_timestamp"] = pd.to_datetime(
        work["timestamp_utc"], utc=True, errors="coerce"
    )
    work = work.sort_values(
        ["station_key", "_timestamp"], kind="stable"
    )
    out = pd.DataFrame(
        index=frame.index, columns=FEATURE_NAMES, dtype=float
    )
    for _, group in work.groupby("station_key", sort=False):
        values = group["_value"]
        difference = values.diff()
        abs_difference = difference.abs()
        median_abs_diff = abs_difference.rolling(
            24, min_periods=6
        ).median()
        diff_scale = median_abs_diff.clip(lower=0.1)
        rolling_median_24 = values.rolling(24, min_periods=6).median()
        residual = values - rolling_median_24
        local_residual_std = residual.rolling(6, min_periods=3).std()
        long_residual_std = residual.rolling(24, min_periods=8).std()
        second_difference = values - 2.0 * values.shift(1) + values.shift(2)
        drift_acceleration = (
            values - 2.0 * values.shift(6) + values.shift(12)
        ).abs() / 6.0
        sign_change = np.sign(difference)
        sign_consistency = sign_change.rolling(
            6, min_periods=3
        ).mean().abs()
        repeat_ratio = (
            abs_difference.le(1e-8).astype(float)
            .rolling(4, min_periods=2)
            .mean()
        )
        part = pd.DataFrame(index=group.index)
        part["level_shift"] = (
            values
            - values.shift(1).rolling(8, min_periods=4).median()
        ).abs() / diff_scale
        part["slope_6"] = (values - values.shift(6)).abs() / 6.0 / diff_scale
        part["trend_consistency"] = sign_consistency * (
            (values - values.shift(6)).abs() / 6.0 / diff_scale
        )
        part["noise_ratio"] = (
            local_residual_std / long_residual_std.replace(0.0, np.nan)
        )
        part["diff_zscore"] = abs_difference.rolling(
            4, min_periods=2
        ).max() / diff_scale
        part["repeat_ratio"] = repeat_ratio
        part["second_diff_zscore"] = (
            second_difference.abs() / diff_scale
        )
        part["drift_acceleration"] = drift_acceleration / diff_scale
        out.loc[group.index, FEATURE_NAMES] = part[
            FEATURE_NAMES
        ].to_numpy(dtype=float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def normalize_features(
    features: pd.DataFrame,
    normal_train: np.ndarray,
) -> pd.DataFrame:
    normalized = pd.DataFrame(
        index=features.index, columns=FEATURE_NAMES, dtype=float
    )
    reference = features.loc[normal_train]
    for column in FEATURE_NAMES:
        center = float(reference[column].median())
        scale = robust_scale(reference[column])
        normalized[column] = np.maximum(
            (features[column].to_numpy(dtype=float) - center) / scale,
            0.0,
        )
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    prefix = "stage186_noaa_targeted_nonstuck_feature_response"
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
    selected_folds = [FOLDS[args.fold]] if args.fold is not None else FOLDS
    selected_seeds = [args.seed] if args.seed is not None else SEEDS
    detail_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in selected_folds:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in selected_seeds:
            frame = build_fold_scaled(
                source,
                seed,
                train_years,
                valid_year,
                test_year,
                stations,
                1.0,
            )
            raw_features = targeted_features(frame)
            train = frame["split"].eq("train").to_numpy()
            normal_train = train & (
                ~frame["combined_is_attack"].astype(bool).to_numpy()
            )
            features = normalize_features(raw_features, normal_train)
            normal_train_values = features.loc[normal_train]
            thresholds = {
                column: float(
                    normal_train_values[column].quantile(
                        0.95, interpolation="higher"
                    )
                )
                for column in FEATURE_NAMES
            }
            for split_name in ["valid", "test"]:
                split_mask = frame["split"].eq(split_name).to_numpy()
                normal_mask = split_mask & (
                    ~frame["combined_is_attack"].astype(bool).to_numpy()
                )
                attack_type = frame["combined_attack_type"].astype(str).to_numpy()
                attack_mask = frame["combined_is_attack"].astype(bool).to_numpy()
                for column in FEATURE_NAMES:
                    values = features[column].to_numpy(dtype=float)
                    normal_values = values[normal_mask]
                    threshold = thresholds[column]
                    for current_type in ATTACK_TYPES:
                        current = split_mask & attack_mask & (
                            attack_type == current_type
                        )
                        if not bool(current.any()):
                            continue
                        current_values = values[current]
                        event_ids = frame.loc[
                            current, "combined_event_id"
                        ].astype(str)
                        selected = current_values >= threshold
                        selected_events = (
                            frame.loc[current, ["combined_event_id"]]
                            .assign(_selected=selected)
                            .groupby("combined_event_id")["_selected"]
                            .any()
                        )
                        detail_rows.append(
                            {
                                "fold": fold,
                                "seed": int(seed),
                                "split": split_name,
                                "feature": column,
                                "attack_type": current_type,
                                "train_normal_p95": float(threshold),
                                "attack_rows": int(current.sum()),
                                "attack_mean": float(
                                    np.nanmean(current_values)
                                ),
                                "normal_mean": float(
                                    np.nanmean(normal_values)
                                ),
                                "attack_above_p95_rate": float(
                                    selected.mean()
                                ),
                                "events": int(event_ids.nunique()),
                                "event_coverage": float(
                                    selected_events.mean()
                                ),
                            }
                        )
            normal_frame = features.loc[normal_train]
            correlation = normal_frame.corr(method="spearman")
            for row_feature in FEATURE_NAMES:
                for column_feature in FEATURE_NAMES:
                    correlation_rows.append(
                        {
                            "fold": fold,
                            "seed": int(seed),
                            "feature_row": row_feature,
                            "feature_column": column_feature,
                            "spearman": float(
                                correlation.loc[row_feature, column_feature]
                            ),
                        }
                    )
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "seed": int(seed),
                        "features": len(FEATURE_NAMES),
                        "status": "completed",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    detail = pd.DataFrame(detail_rows)
    aggregate = (
        detail.groupby(
            ["feature", "split", "attack_type"], sort=True
        )
        .agg(
            fold_seed_pairs=("fold", "size"),
            attack_rows=("attack_rows", "sum"),
            attack_mean=("attack_mean", "mean"),
            normal_mean=("normal_mean", "mean"),
            train_normal_p95=("train_normal_p95", "mean"),
            attack_above_p95_rate=("attack_above_p95_rate", "mean"),
            mean_event_coverage=("event_coverage", "mean"),
        )
        .reset_index()
    )
    correlation = pd.DataFrame(correlation_rows)
    correlation_aggregate = (
        correlation.groupby(
            ["feature_row", "feature_column"], sort=True
        )
        .agg(
            fold_seed_pairs=("fold", "size"),
            mean_spearman=("spearman", "mean"),
            std_spearman=("spearman", "std"),
        )
        .reset_index()
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
    correlation.to_csv(
        TABLES / f"{prefix}_correlation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlation_aggregate.to_csv(
        TABLES / f"{prefix}_correlation_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段178冻结评分协议和同一透明受控注入；新增8个针对"
            "非stuck类型设计的因果特征；诊断阈值只由train正常样本的"
            "95%分位固定，valid和test只报告特征响应与事件覆盖；"
            "不重新训练、不根据test选择特征、权重或阈值；"
            "五种子仅用于不确定性描述；NOAA攻击为透明受控注入，"
            "不是真实攻击日志。"
        ),
        "source_artifacts": [NOAA.name, AUDIT.name],
        "features": FEATURE_NAMES,
        "attack_types": ATTACK_TYPES,
        "detail_rows": int(len(detail)),
        "aggregate_rows": int(len(aggregate)),
        "correlation_rows": int(len(correlation)),
        "correlation_aggregate_rows": int(len(correlation_aggregate)),
        "aggregate": clean_records(aggregate),
        "correlation_aggregate": clean_records(correlation_aggregate),
        "limitations": [
            "该阶段只做train正常分位固定的特征响应诊断，不选择模型。",
            "特征响应强不等于低误报检测增益，仍需独立valid校准审计。",
            "事件覆盖按攻击类型分别计算，小样本类型波动较大。",
            "攻击仍为透明受控注入，不代表真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段186 NOAA非stuck定向特征响应审计",
        "",
        "新增8个因果特征；阈值由train正常样本95%分位固定，valid/test只报告响应。",
        "",
        "| 特征 | 攻击类型 | split | 攻击均值 | 正常均值 | 超过train p95比例 | 事件覆盖 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['feature']} | {row['attack_type']} | {row['split']} | "
            f"{row['attack_mean']:.6f} | {row['normal_mean']:.6f} | "
            f"{row['attack_above_p95_rate']:.6f} | "
            f"{row['mean_event_coverage']:.6f} |"
        )
    lines += [
        "",
        f"明细={len(detail)}；聚合={len(aggregate)}；相关矩阵={len(correlation)}。",
        "该阶段是特征响应诊断，不是模型性能结论。",
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
                "correlation_rows": len(correlation),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
