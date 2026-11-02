from __future__ import annotations

"""Stage 177: station-calibrated low-FAR operating points.

The frozen risk scores from the stage-172 protocol are replayed exactly.  For
each station, the operating threshold is the train-normal risk quantile within
that station instead of the global train-normal quantile.  Test data are used
only for evaluation.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"
AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
FROZEN = TABLES / "stage141_noaa_frozen_external_validation_selected.csv"
STAGE172 = TABLES / "stage172_noaa_frozen_expanded_low_far_full_baseline.json"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
Q_GRID = [0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
METHODS = [
    "TrustMeteo-QC frozen expanded",
    "TrustMeteo-QC default",
    "Robust max deviation",
    "Isolation Forest",
    "One-Class SVM",
]
BASELINES = METHODS[1:]
METRICS = ["f1", "recall", "precision", "false_alarm_rate"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components  # noqa: E402
from stage151_noaa_attack_intensity_sensitivity import build_fold_scaled  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage85_noaa_rolling_baseline_comparison import SCORE_COLS  # noqa: E402


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def finite_or_none(value: object) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def mean_or_none(values: list[float | None]) -> float | None:
    finite = [
        value
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def f1_from_counts(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else None


def station_thresholds(
    frame: pd.DataFrame,
    risk: np.ndarray,
    train_normal: np.ndarray,
    q: float,
) -> dict[str, float]:
    station_values = frame["station_key"].astype(str).to_numpy()
    out: dict[str, float] = {}
    for station in sorted(set(station_values)):
        mask = (station_values == station) & train_normal & np.isfinite(risk)
        values = risk[mask]
        if values.size == 0:
            raise RuntimeError(f"no finite train-normal risk for station {station}")
        out[station] = float(np.nanquantile(values, q))
    return out


def station_confusion(
    frame: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
) -> pd.DataFrame:
    work = frame[["station_key"]].copy()
    work["y"] = y.astype(bool)
    work["pred"] = pred.astype(bool)
    grouped = work.groupby("station_key", sort=False)
    rows: list[dict[str, object]] = []
    for station, group in grouped:
        tp = int((group["pred"] & group["y"]).sum())
        fp = int((group["pred"] & ~group["y"]).sum())
        fn = int((~group["pred"] & group["y"]).sum())
        tn = int((~group["pred"] & ~group["y"]).sum())
        rows.append(
            {
                "station_id": str(station),
                "test_rows": int(len(group)),
                "test_attack_rows": int(group["y"].sum()),
                "test_normal_rows": int((~group["y"]).sum()),
                "alarm_count": tp + fp,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "recall": finite_or_none(tp / int(group["y"].sum())) if int(group["y"].sum()) else None,
                "precision": finite_or_none(tp / (tp + fp)) if (tp + fp) else None,
                "far": finite_or_none(fp / int((~group["y"]).sum())) if int((~group["y"]).sum()) else None,
                "f1": f1_from_counts(tp, fp, fn),
            }
        )
    return pd.DataFrame(rows)


def bundle(
    frame: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    risk: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame]:
    metrics = _metrics(y, pred, risk)
    station = station_confusion(frame, y, pred)
    return (
        {
            "f1": finite_or_none(metrics["f1"]),
            "recall": finite_or_none(metrics["recall"]),
            "precision": finite_or_none(metrics["precision"]),
            "auprc": finite_or_none(metrics["average_precision"]),
            "false_alarm_rate": finite_or_none(float(pred[~y].mean()) if (~y).any() else np.nan),
            "macro_f1": mean_or_none(station["f1"].tolist()),
            "macro_recall": mean_or_none(station["recall"].tolist()),
            "macro_precision": mean_or_none(station["precision"].tolist()),
            "macro_far": mean_or_none(station["far"].tolist()),
            "station_count": int(len(station)),
            "alarm_count": int(station["alarm_count"].sum()),
            "false_positive_count": int(station["false_positive"].sum()),
        },
        station,
    )


def exact_sign_flip(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = [
        abs(float((values * np.asarray(signs)).mean()))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15)), float(values.mean())


def bh_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    order = np.argsort(np.asarray(values, dtype=float))
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        idx = int(order[rank])
        running = min(running, float(values[idx]) * len(order) / (rank + 1))
        adjusted[idx] = running
    return adjusted.tolist()


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


def build_comparisons(
    detail: pd.DataFrame,
    baseline_comparisons: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if baseline_comparisons:
        left = detail[detail["method"].eq(METHODS[0])]
        right_methods = BASELINES
        left_prefix = "station_"
        right_prefix = "station_"
    else:
        left = detail
        right_methods = [None]
        left_prefix = "station_"
        right_prefix = "global_"
    for right_method in right_methods:
        if right_method is None:
            right = left
            merge_keys = ["fold", "seed", "method", "q"]
            comparison_name = "station-calibrated - global"
        else:
            right = detail[detail["method"].eq(right_method)]
            merge_keys = ["fold", "seed", "q"]
            comparison_name = f"station-calibrated expanded - station-calibrated {right_method}"
        common = left.merge(
            right,
            on=merge_keys,
            suffixes=("_left", "_right"),
            how="inner",
        )
        for q in sorted(float(x) for x in common["q"].unique()):
            q_common = common[np.isclose(common["q"].astype(float), q)]
            for metric in METRICS:
                left_col = f"{left_prefix}{metric}_left"
                right_col = f"{right_prefix}{metric}_right"
                diff = pd.to_numeric(q_common[left_col], errors="coerce") - pd.to_numeric(
                    q_common[right_col], errors="coerce"
                )
                fold_values = (
                    diff.groupby(q_common["fold"], sort=True)
                    .mean()
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if len(fold_values) < 2:
                    continue
                p_value, mean_diff = exact_sign_flip(fold_values)
                rows.append(
                    {
                        "comparison": comparison_name,
                        "metric": metric,
                        "q": q,
                        "fold_count": int(len(fold_values)),
                        "fold_seed_cells": int(diff.notna().sum()),
                        "mean_fold_diff": mean_diff,
                        "p_value_exact_sign_flip": p_value,
                        "fold_differences": fold_values.tolist(),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["bh_q_value"] = bh_adjust(
            [float(x) for x in out["p_value_exact_sign_flip"]]
        )
    return out


def write_summaries(
    detail: pd.DataFrame,
    station_detail: pd.DataFrame,
) -> int:
    prefix = "stage177_noaa_station_calibrated_operating_points"
    detail.to_csv(
        TABLES / f"{prefix}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_detail.to_csv(
        TABLES / f"{prefix}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline_comparisons = build_comparisons(detail, baseline_comparisons=True)
    calibration_comparisons = build_comparisons(detail, baseline_comparisons=False)
    baseline_comparisons.to_csv(
        TABLES / f"{prefix}_baseline_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibration_comparisons.to_csv(
        TABLES / f"{prefix}_calibration_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    expanded = detail[detail["method"].eq(METHODS[0])]
    expanded_995 = expanded[np.isclose(expanded["q"].astype(float), 0.995)]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段172的train/valid/test四折滚动、五种子和冻结alpha/window；"
            "对每个站点只用该站点train正常样本的冻结风险分位数计算站点阈值，"
            "test只评价。全局阈值和站点校准阈值在同一风险分数上并列比较；"
            "不重新训练、不重新选择模型或q，不根据test调参。"
            "站点校准与全局阈值及基线之间的差异按fold执行精确sign-flip，"
            "并使用BH校正控制多重比较。"
            "NOAA攻击是透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [
            STAGE172.name,
            "stage174_noaa_low_far_alarm_burden_audit_station_detail.csv",
        ],
        "methods": METHODS,
        "q_grid": Q_GRID,
        "detail_rows": int(len(detail)),
        "station_detail_rows": int(len(station_detail)),
        "baseline_comparison_rows": int(len(baseline_comparisons)),
        "calibration_comparison_rows": int(len(calibration_comparisons)),
        "global_replay_mismatch_count": 0,
        "expanded_q995_station_f1_mean": float(expanded_995["station_f1"].mean()),
        "expanded_q995_global_f1_mean": float(expanded_995["global_f1"].mean()),
        "expanded_q995_station_macro_f1_mean": float(expanded_995["station_macro_f1"].mean()),
        "expanded_q995_station_alarm_count_mean": float(expanded_995["station_alarm_count"].mean()),
        "baseline_bh_q_below_05": (
            int((baseline_comparisons["bh_q_value"] < 0.05).sum())
            if not baseline_comparisons.empty
            else 0
        ),
        "calibration_bh_q_below_05": (
            int((calibration_comparisons["bh_q_value"] < 0.05).sum())
            if not calibration_comparisons.empty
            else 0
        ),
        "min_baseline_bh_q_value": (
            float(baseline_comparisons["bh_q_value"].min())
            if not baseline_comparisons.empty
            else None
        ),
        "min_calibration_bh_q_value": (
            float(calibration_comparisons["bh_q_value"].min())
            if not calibration_comparisons.empty
            else None
        ),
        "detail": clean_records(detail),
        "station_detail": clean_records(station_detail),
        "baseline_comparisons": clean_records(baseline_comparisons),
        "calibration_comparisons": clean_records(calibration_comparisons),
        "limitations": [
            "站点阈值只改变运行点，不改变风险分数；不能把站点校准解释成新的模型训练。",
            "每个站点的正常样本量不同，分位数目标不等于每站点完全相同报警数。",
            "四折功效有限；BH校正后无显著差异不等于方法等效。",
            "NOAA攻击是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段177 NOAA站点校准低误报运行点",
        "",
        "复用阶段172冻结风险分数；每站只用本站train正常分位数选阈值，test只评价。",
        "",
        "| 方法 | q | 全局F1 | 站点F1 | 站点macro F1 | 站点报警数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summary = (
        detail.groupby(["method", "q"], sort=True)[
            [
                "global_f1",
                "station_f1",
                "station_macro_f1",
                "station_alarm_count",
            ]
        ]
        .mean()
        .reset_index()
    )
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {row['q']:.4f} | {row['global_f1']:.6f} | "
            f"{row['station_f1']:.6f} | {row['station_macro_f1']:.6f} | "
            f"{row['station_alarm_count']:.1f} |"
        )
    lines += [
        "",
        f"明细={len(detail)}；站点明细={len(station_detail)}；"
        f"基线成对比较={len(baseline_comparisons)}；校准成对比较={len(calibration_comparisons)}。",
        "站点阈值只改变运行点，不改变风险分数，也不能替代真实攻击验证。",
    ]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": len(detail),
                "station_detail_rows": len(station_detail),
                "baseline_comparison_rows": len(baseline_comparisons),
                "calibration_comparison_rows": len(calibration_comparisons),
                "baseline_bh_q_below_05": payload["baseline_bh_q_below_05"],
                "calibration_bh_q_below_05": payload["calibration_bh_q_below_05"],
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
        help="Only rebuild summary artifacts from persisted fold details.",
    )
    args = parser.parse_args(argv)
    prefix = "stage177_noaa_station_calibrated_operating_points"
    if args.summarize_existing:
        detail_path = TABLES / f"{prefix}_detail.csv"
        station_path = TABLES / f"{prefix}_station_detail.csv"
        if not detail_path.exists():
            legacy_detail = TABLES / "station_detail.csv"
            legacy_station = TABLES / "station_station_detail.csv"
            if not legacy_detail.exists() or not legacy_station.exists():
                raise FileNotFoundError(
                    "No persisted stage177 detail files were found."
                )
            detail_path = legacy_detail
            station_path = legacy_station
        detail = pd.read_csv(detail_path)
        station_detail = pd.read_csv(station_path)
        return write_summaries(detail, station_detail)

    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    frozen = pd.read_csv(FROZEN)
    stage172 = json.loads(STAGE172.read_text(encoding="utf-8"))
    expected = {
        (str(row["fold"]), int(row["seed"]), str(row["method"]), round(float(row["q"]), 10)): float(row["test_f1"])
        for row in stage172["detail"]
    }
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
        frozen[frozen["method"].eq(METHODS[0])]
        .copy()
        .set_index(["fold", "seed"])
    )

    rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            alpha = float(frozen.loc[(fold, seed), "alpha_base"])
            window = int(frozen.loc[(fold, seed), "window"])
            frame = build_fold_scaled(source, seed, train_years, valid_year, test_year, stations, 1.0)
            scored, _ = score(frame)
            scored = add_components(scored)
            train = scored["split"].eq("train").to_numpy()
            test_mask = scored["split"].eq("test").to_numpy()
            normal_train = train & (~scored["combined_is_attack"].astype(bool).to_numpy())
            test_frame = scored.loc[test_mask].reset_index(drop=True)
            y_test = test_frame["combined_is_attack"].astype(bool).to_numpy()
            x = (
                scored[SCORE_COLS]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            med = np.nanmedian(x[normal_train], axis=0)
            scale = np.maximum(
                1.4826 * np.nanmedian(np.abs(x[normal_train] - med), axis=0),
                0.25,
            )
            robust = np.maximum((x - med) / scale, 0.0).max(axis=1)
            iso = IsolationForest(
                n_estimators=300,
                random_state=seed,
                n_jobs=1,
                contamination="auto",
            ).fit(x[normal_train])
            iso_risk = -iso.score_samples(x)
            svm = make_pipeline(
                StandardScaler(),
                OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
            ).fit(x[normal_train])
            svm_risk = -svm.decision_function(x).ravel()
            base = scored["base_fused_risk"].to_numpy(dtype=float)
            structural = scored["structural_risk"].to_numpy(dtype=float)
            expanded_raw = alpha * base + (1.0 - alpha) * structural
            expanded_risk = np.full(len(scored), np.nan, dtype=float)
            for split_name in ["train", "valid", "test"]:
                split_mask = scored["split"].eq(split_name).to_numpy()
                expanded_risk[split_mask] = smooth_risk(
                    scored.loc[split_mask], expanded_raw[split_mask], window
                )
            risks = {
                METHODS[0]: expanded_risk,
                "TrustMeteo-QC default": scored["robust_fused_risk"].to_numpy(dtype=float),
                "Robust max deviation": robust,
                "Isolation Forest": iso_risk,
                "One-Class SVM": svm_risk,
            }
            for method in METHODS:
                risk = risks[method]
                train_values = risk[normal_train & np.isfinite(risk)]
                if train_values.size == 0:
                    raise RuntimeError(f"no finite train-normal risk for {method}")
                for q in Q_GRID:
                    global_threshold = float(np.nanquantile(train_values, q))
                    thresholds = station_thresholds(scored, risk, normal_train, q)
                    global_pred = risk[test_mask] >= global_threshold
                    station_pred = np.zeros(len(test_frame), dtype=bool)
                    test_stations = test_frame["station_key"].astype(str).to_numpy()
                    for station, threshold in thresholds.items():
                        station_pred[test_stations == station] = (
                            risk[test_mask][test_stations == station] >= threshold
                        )
                    global_metrics, global_station = bundle(
                        test_frame, y_test, global_pred, risk[test_mask]
                    )
                    station_metrics, station_station = bundle(
                        test_frame, y_test, station_pred, risk[test_mask]
                    )
                    key = (fold, seed, method, round(float(q), 10))
                    if key not in expected:
                        raise RuntimeError(f"missing stage172 key: {key}")
                    if abs(float(global_metrics["f1"]) - expected[key]) > 1e-10:
                        raise RuntimeError(
                            f"stage172 replay mismatch for {key}: "
                            f"{global_metrics['f1']} vs {expected[key]}"
                        )
                    row = {
                        "fold": fold,
                        "seed": int(seed),
                        "method": method,
                        "q": float(q),
                        "global_threshold": global_threshold,
                        "station_threshold_min": float(min(thresholds.values())),
                        "station_threshold_max": float(max(thresholds.values())),
                        "station_threshold_median": float(np.median(list(thresholds.values()))),
                        "station_count": int(len(thresholds)),
                        "missing_station_thresholds": 0,
                    }
                    for label, metrics in (("global", global_metrics), ("station", station_metrics)):
                        for metric in [
                            "f1",
                            "recall",
                            "precision",
                            "auprc",
                            "false_alarm_rate",
                            "macro_f1",
                            "macro_recall",
                            "macro_precision",
                            "macro_far",
                            "alarm_count",
                            "false_positive_count",
                        ]:
                            row[f"{label}_{metric}"] = metrics[metric]
                    rows.append(row)
                    station_counts = global_station.merge(
                        station_station,
                        on="station_id",
                        suffixes=("_global", "_station"),
                        how="outer",
                    )
                    for _, station_row in station_counts.iterrows():
                        station_rows.append(
                            {
                                "fold": fold,
                                "seed": int(seed),
                                "method": method,
                                "q": float(q),
                                "station_id": str(station_row["station_id"]),
                                "global_threshold": global_threshold,
                                "station_threshold": thresholds[str(station_row["station_id"])],
                                "global_alarm_count": int(station_row["alarm_count_global"]),
                                "station_alarm_count": int(station_row["alarm_count_station"]),
                                "global_true_positive": int(station_row["true_positive_global"]),
                                "global_false_positive": int(station_row["false_positive_global"]),
                                "station_true_positive": int(station_row["true_positive_station"]),
                                "station_false_positive": int(station_row["false_positive_station"]),
                            }
                        )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "seeds": len(SEEDS),
                    "q_values": len(Q_GRID),
                    "status": "completed",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        pd.DataFrame(rows).to_csv(
            TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(station_rows).to_csv(
            TABLES / f"{prefix}_station_detail.csv",
            index=False,
            encoding="utf-8-sig",
        )

    detail = pd.DataFrame(rows)
    station_detail = pd.DataFrame(station_rows)
    # Persist the deterministic four-fold replay before post-processing so a
    # reporting-stage failure does not require recomputing all folds.
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    station_detail.to_csv(
        TABLES / f"{prefix}_station_detail.csv", index=False, encoding="utf-8-sig"
    )
    return write_summaries(detail, station_detail)


if __name__ == "__main__":
    raise SystemExit(main())
