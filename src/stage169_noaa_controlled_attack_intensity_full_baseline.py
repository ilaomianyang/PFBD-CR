from __future__ import annotations

"""Stage 169: full-baseline attack-intensity robustness under the frozen protocol."""

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
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
AMPLITUDES = [0.5, 1.0, 1.5, 2.0]
ATTACK_TYPES = ["spike", "offset", "drift", "time_shift", "stuck", "gaussian_noise"]
METHODS = ["TrustMeteo-QC frozen expanded", "TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
BASELINES = ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components, candidate_scores  # noqa: E402
from stage151_noaa_attack_intensity_sensitivity import build_fold_scaled  # noqa: E402
from stage151_noaa_attack_intensity_sensitivity import stable_seed as intensity_seed  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage85_noaa_rolling_baseline_comparison import SCORE_COLS  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def finite_or_none(value: object) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = {station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)}
    frozen = pd.read_csv(FROZEN)
    frozen = frozen[frozen["method"].eq("TrustMeteo-QC frozen expanded")].copy()
    if len(frozen) != len(FOLDS) * len(SEEDS):
        raise RuntimeError("frozen expanded settings are incomplete")
    settings = frozen.set_index(["fold", "seed"])
    rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            key = (fold, seed)
            alpha = float(settings.loc[key, "alpha_base"])
            window = int(settings.loc[key, "window"])
            for amplitude in AMPLITUDES:
                frame = build_fold_scaled(source, seed, train_years, valid_year, test_year, stations, amplitude)
                scored, default_result = score(frame)
                scored = add_components(scored)
                valid = scored["split"].eq("valid").to_numpy()
                test = scored["split"].eq("test").to_numpy()
                y_valid = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
                y_test = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
                train = scored["split"].eq("train").to_numpy()
                normal_train = train & (~scored["combined_is_attack"].astype(bool).to_numpy())
                x = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                med = np.nanmedian(x[normal_train], axis=0)
                scale = np.maximum(1.4826 * np.nanmedian(np.abs(x[normal_train] - med), axis=0), 0.25)
                robust_score = np.maximum((x - med) / scale, 0.0).max(axis=1)
                isolation = IsolationForest(n_estimators=300, random_state=seed, n_jobs=1, contamination="auto").fit(x[normal_train])
                isolation_score = -isolation.score_samples(x)
                svm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)).fit(x[normal_train])
                svm_score = -svm.decision_function(x).ravel()

                base = scored["base_fused_risk"].to_numpy(dtype=float)
                structural = scored["structural_risk"].to_numpy(dtype=float)
                expanded_raw = alpha * base + (1.0 - alpha) * structural
                expanded_valid = smooth_risk(scored.loc[valid], expanded_raw[valid], window)
                expanded_threshold, expanded_valid_f1 = _best_threshold(y_valid, expanded_valid)
                expanded_test = smooth_risk(scored.loc[test], expanded_raw[test], window)
                predictions: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {
                    "TrustMeteo-QC frozen expanded": (expanded_test >= expanded_threshold, expanded_test, float(expanded_threshold), float(expanded_valid_f1)),
                    "TrustMeteo-QC default": (scored.loc[test, "predicted_attack"].astype(bool).to_numpy(), scored.loc[test, "robust_fused_risk"].to_numpy(dtype=float), float(default_result["setting"]["threshold"]), float(default_result["valid"]["f1"])),
                }
                for method, values in [("Robust max deviation", robust_score), ("Isolation Forest", isolation_score), ("One-Class SVM", svm_score)]:
                    threshold, valid_f1 = _best_threshold(y_valid, values[valid])
                    predictions[method] = (values[test] >= threshold, values[test], float(threshold), float(valid_f1))

                test_frame = scored.loc[test].reset_index(drop=True)
                for method in METHODS:
                    pred, risk, threshold, valid_f1 = predictions[method]
                    metrics = _metrics(y_test, pred, risk)
                    rows.append({
                        "fold": fold, "seed": int(seed), "test_year": int(test_year), "amplitude": float(amplitude), "method": method,
                        "threshold_valid": float(threshold), "valid_f1": float(valid_f1), "test_f1": float(metrics["f1"]),
                        "test_recall": float(metrics["recall"]), "test_precision": float(metrics["precision"]), "test_auprc": finite_or_none(metrics["average_precision"]),
                        "test_false_alarm_rate": float(pred[~y_test].mean()) if (~y_test).any() else None,
                    })
                    for attack_type in ATTACK_TYPES:
                        mask = test_frame["combined_attack_type"].eq(attack_type).to_numpy()
                        type_rows.append({
                            "fold": fold, "seed": int(seed), "test_year": int(test_year), "amplitude": float(amplitude), "method": method,
                            "attack_type": attack_type, "rows": int(mask.sum()), "detected_rows": int(pred[mask].sum()),
                            "recall": float(pred[mask].mean()) if mask.any() else None,
                        })
        print(json.dumps({"fold": fold, "amplitudes": len(AMPLITUDES), "seeds": len(SEEDS), "status": "completed"}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(rows)
    type_detail = pd.DataFrame(type_rows)
    summary = detail.groupby(["method", "amplitude"], sort=True)[["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = ["method", "amplitude"] + [f"{a}_{b}" for a, b in summary.columns.tolist()[2:]]
    type_summary = type_detail.groupby(["method", "amplitude", "attack_type"], sort=True)[["rows", "detected_rows", "recall"]].agg(["mean", "std", "min", "max"]).reset_index()
    type_summary.columns = ["method", "amplitude", "attack_type"] + [f"{a}_{b}" for a, b in type_summary.columns.tolist()[3:]]

    paired_rows: list[dict[str, object]] = []
    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    for baseline in BASELINES:
        other = detail[detail["method"].eq(baseline)]
        common = expanded.merge(other, on=["fold", "seed", "test_year", "amplitude"], suffixes=("_expanded", "_baseline"), how="inner")
        for amplitude in AMPLITUDES:
            group = common[common["amplitude"].eq(amplitude)]
            for metric in METRICS:
                diff = pd.to_numeric(group[f"{metric}_expanded"], errors="coerce") - pd.to_numeric(group[f"{metric}_baseline"], errors="coerce")
                fold_means = group.assign(diff=diff).groupby("fold", sort=True)["diff"].mean().dropna()
                paired_rows.append({
                    "comparison": f"expanded - {baseline}", "amplitude": float(amplitude), "metric": metric,
                    "fold_count": int(len(fold_means)), "fold_seed_cells": int(diff.notna().sum()),
                    "mean_fold_diff": float(fold_means.mean()), "median_fold_diff": float(fold_means.median()),
                    "min_fold_diff": float(fold_means.min()), "max_fold_diff": float(fold_means.max()),
                })
    paired = pd.DataFrame(paired_rows)
    not_estimable = pd.DataFrame(columns=["comparison", "amplitude", "metric", "reason"])
    prefix = "stage169_noaa_controlled_attack_intensity_full_baseline"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    type_detail.to_csv(TABLES / f"{prefix}_type_detail.csv", index=False, encoding="utf-8-sig")
    type_summary.to_csv(TABLES / f"{prefix}_type_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    not_estimable.to_csv(TABLES / f"{prefix}_not_estimable.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "NOAA官方2020–2025四折滚动、五种子和冻结train/valid/test边界；攻击强度网格预先固定为0.5/1.0/1.5/2.0；五种方法在每个强度下使用相同数据、相同攻击位置和相同评价边界。Trust扩展与default复用冻结设置，Robust、Isolation Forest和One-Class SVM只用train正常样本拟合、valid选择阈值，test只评价；不根据test重新选择强度、方法或参数。攻击为透明受控注入，不是真实攻击日志；不可估计单元保留为空。",
        "source_artifact": "stage151_noaa_attack_intensity_sensitivity_detail.csv",
        "stations": sorted(stations), "folds": [fold_name(*fold) for fold in FOLDS], "seeds": SEEDS,
        "amplitudes": AMPLITUDES, "attack_types": ATTACK_TYPES, "methods": METHODS,
        "detail_rows": int(len(detail)), "summary_rows": int(len(summary)), "type_detail_rows": int(len(type_detail)), "type_summary_rows": int(len(type_summary)),
        "paired_rows": int(len(paired)), "not_estimable_count": int(len(not_estimable)),
        "detail": detail.to_dict(orient="records"), "summary": summary.to_dict(orient="records"), "type_summary": type_summary.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"), "not_estimable": not_estimable.to_dict(orient="records"),
        "limitations": ["攻击强度只改变预先定义的扰动幅度；time_shift和stuck的数值幅度不改变，需按攻击机制解释。", "攻击仍为透明受控注入，不是真实网络攻击日志。", "四折和五种子用于不确定性描述，不支持把任一强度或方法写成普适最优。"],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 阶段169 NOAA攻击强度完整基线比较", "", "五种方法在相同四折、五种子、攻击位置和强度网格下比较；每个强度的阈值只由valid选择，test只评价。", "", "| 方法 | 强度 | F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict(orient="records"):
        auprc = "NA" if pd.isna(row["test_auprc_mean"]) else f"{row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f}"
        lines.append(f"| {row['method']} | {row['amplitude']:.1f} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {auprc} | {row['test_false_alarm_rate_mean']:.4f}±{row['test_false_alarm_rate_std']:.4f} |")
    lines += ["", f"明细={len(detail)}；攻击类型明细={len(type_detail)}；成对比较={len(paired)}；不可估计={len(not_estimable)}。", "攻击为透明受控注入，不是真实攻击日志；负结果和基线优势必须与正向结果并列报告。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "summary_rows": len(summary), "type_detail_rows": len(type_detail), "paired_rows": len(paired)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
