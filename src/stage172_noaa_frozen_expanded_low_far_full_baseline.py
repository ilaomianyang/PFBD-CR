from __future__ import annotations

"""Stage 172: low-FAR operating points for the frozen expanded method and baselines."""

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
FOLDS = [([2020], 2021, 2022), ([2020, 2021], 2022, 2023), ([2020, 2021, 2022], 2023, 2024), ([2020, 2021, 2022, 2023], 2024, 2025)]
Q_GRID = [0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
METHODS = ["TrustMeteo-QC frozen expanded", "TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]

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


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = {station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)}
    frozen = pd.read_csv(FROZEN)
    frozen = frozen[frozen["method"].eq("TrustMeteo-QC frozen expanded")].copy().set_index(["fold", "seed"])
    rows: list[dict[str, object]] = []
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
            test = scored["split"].eq("test").to_numpy()
            normal_train = train & (~scored["combined_is_attack"].astype(bool).to_numpy())
            y_test = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            x = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            med = np.nanmedian(x[normal_train], axis=0)
            scale = np.maximum(1.4826 * np.nanmedian(np.abs(x[normal_train] - med), axis=0), 0.25)
            robust = np.maximum((x - med) / scale, 0.0).max(axis=1)
            iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=1, contamination="auto").fit(x[normal_train])
            iso_risk = -iso.score_samples(x)
            svm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)).fit(x[normal_train])
            svm_risk = -svm.decision_function(x).ravel()
            base = scored["base_fused_risk"].to_numpy(dtype=float)
            structural = scored["structural_risk"].to_numpy(dtype=float)
            expanded_raw = alpha * base + (1.0 - alpha) * structural
            expanded_risk = np.full(len(scored), np.nan, dtype=float)
            for split in ["train", "valid", "test"]:
                mask = scored["split"].eq(split)
                expanded_risk[mask.to_numpy()] = smooth_risk(scored.loc[mask], expanded_raw[mask.to_numpy()], window)
            risks = {"TrustMeteo-QC frozen expanded": expanded_risk, "TrustMeteo-QC default": scored["robust_fused_risk"].to_numpy(dtype=float), "Robust max deviation": robust, "Isolation Forest": iso_risk, "One-Class SVM": svm_risk}
            for method in METHODS:
                risk = risks[method]
                train_values = risk[normal_train & np.isfinite(risk)]
                for q in Q_GRID:
                    threshold = float(np.nanquantile(train_values, q))
                    pred = risk[test] >= threshold
                    metrics = _metrics(y_test, pred, risk[test])
                    rows.append({"fold": fold, "seed": int(seed), "test_year": int(test_year), "method": method, "q": float(q), "threshold_train_normal": threshold, "test_f1": float(metrics["f1"]), "test_recall": float(metrics["recall"]), "test_precision": float(metrics["precision"]), "test_auprc": finite_or_none(metrics["average_precision"]), "test_false_alarm_rate": float(pred[~y_test].mean()) if (~y_test).any() else None})
        print(json.dumps({"fold": fold, "seeds": len(SEEDS), "q_values": len(Q_GRID), "status": "completed"}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["method", "q"], sort=True)[METRICS].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = ["method", "q"] + [f"{a}_{b}" for a, b in summary.columns.tolist()[2:]]
    paired_rows: list[dict[str, object]] = []
    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    for baseline in METHODS[1:]:
        other = detail[detail["method"].eq(baseline)]
        common = expanded.merge(other, on=["fold", "seed", "test_year", "q"], suffixes=("_expanded", "_baseline"), how="inner")
        for q in Q_GRID:
            group = common[common["q"].eq(q)]
            for metric in METRICS:
                diff = pd.to_numeric(group[f"{metric}_expanded"], errors="coerce") - pd.to_numeric(group[f"{metric}_baseline"], errors="coerce")
                paired_rows.append({"comparison": f"expanded - {baseline}", "q": float(q), "metric": metric, "fold_count": int(group["fold"].nunique()), "fold_seed_cells": int(diff.notna().sum()), "mean_diff": float(diff.mean()), "median_diff": float(diff.median()), "min_diff": float(diff.min()), "max_diff": float(diff.max())})
    paired = pd.DataFrame(paired_rows)
    not_estimable = pd.DataFrame(columns=["comparison", "q", "metric", "reason"])
    prefix = "stage172_noaa_frozen_expanded_low_far_full_baseline"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    not_estimable.to_csv(TABLES / f"{prefix}_not_estimable.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020–2025四折滚动、五种子、固定幅度1.0和冻结train/valid/test边界；五种方法在相同攻击数据和相同训练正常风险分位数q下比较，q网格预先固定为0.95/0.975/0.99/0.995/0.9975/0.999。冻结扩展与default沿用冻结评分设置，Robust、Isolation Forest和One-Class SVM只用train正常样本拟合；阈值只由train正常样本分位数确定，test只评价；不根据test选择方法、q或参数。攻击为透明受控注入，不是真实攻击日志；不可估计单元保留为空。", "stations": sorted(stations), "folds": [fold_name(*fold) for fold in FOLDS], "seeds": SEEDS, "q_grid": Q_GRID, "methods": METHODS, "detail_rows": int(len(detail)), "summary_rows": int(len(summary)), "paired_rows": int(len(paired)), "not_estimable_count": int(len(not_estimable)), "detail": detail.to_dict(orient="records"), "summary": summary.to_dict(orient="records"), "paired": paired.to_dict(orient="records"), "not_estimable": not_estimable.to_dict(orient="records"), "limitations": ["低误报运行点必须同时报告Recall、F1和AUPRC，不能只报告FAR。", "q是统一运行点参数，不代表每种方法的最优test阈值。", "攻击为透明受控注入，不是真实网络攻击日志。"]}
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 阶段172 NOAA冻结扩展低误报完整基线比较", "", "五种方法在相同训练正常风险分位数q下比较；阈值只由train正常样本确定，test只评价。", "", "| 方法 | q | F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict(orient="records"):
        auprc = "NA" if pd.isna(row["test_auprc_mean"]) else f"{row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f}"
        lines.append(f"| {row['method']} | {row['q']:.4f} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {auprc} | {row['test_false_alarm_rate_mean']:.4f}±{row['test_false_alarm_rate_std']:.4f} |")
    lines += ["", f"明细={len(detail)}；汇总={len(summary)}；成对比较={len(paired)}；不可估计={len(not_estimable)}。", "低误报运行点不能被单独包装成全面领先；必须并列报告Recall、F1和AUPRC。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "summary_rows": len(summary), "paired_rows": len(paired)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
