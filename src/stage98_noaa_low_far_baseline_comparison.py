from __future__ import annotations

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
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
Q_GRID = [0.90, 0.95, 0.975, 0.99, 0.995]
METHODS = ["Trust legacy", "Trust component_zmax", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
SCORE_COLS = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency", "structural_risk"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, structural_features, structural_risk_from_norm  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage91_noaa_physical_component_repair import component_physical_scores  # noqa: E402
from stage92_noaa_physical_component_bootstrap import output_for_method  # noqa: E402


def add_structural(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    train = out["split"].eq("train")
    parts = []
    for split in ["train", "valid", "test"]:
        mask = out["split"].eq(split)
        parts.append(structural_features(out.loc[mask], STRUCTURAL_BASE).set_axis(out.index[mask]))
    features = pd.concat(parts).reindex(out.index).fillna(0.0)
    features["station_key"] = out["station_key"].astype(str)
    normal_train = train & ~out["combined_is_attack"].astype(bool)
    scaler = AdaptiveScaler(STRUCTURAL_BASE).fit(features.loc[train], normal_train.loc[train])
    out["structural_risk"] = structural_risk_from_norm(scaler.transform(features), enhanced=False)
    return out


def baseline_risks(scored: pd.DataFrame) -> dict[str, np.ndarray]:
    train = scored["split"].eq("train").to_numpy()
    normal_train = train & ~scored["combined_is_attack"].astype(bool).to_numpy()
    X = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    med = np.nanmedian(X[normal_train], axis=0)
    scale = np.maximum(1.4826 * np.nanmedian(np.abs(X[normal_train] - med), axis=0), 0.25)
    robust = np.maximum((X - med) / scale, 0.0).max(axis=1)
    iso = IsolationForest(n_estimators=300, random_state=20260918, n_jobs=1, contamination="auto").fit(X[normal_train])
    svm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)).fit(X[normal_train])
    return {"Robust max deviation": robust, "Isolation Forest": -iso.score_samples(X), "One-Class SVM": -svm.decision_function(X).ravel()}


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            physical, _ = component_physical_scores(frame)
            scored = add_structural(scored)
            trust_risks = {"Trust legacy": output_for_method(scored, physical["legacy"], "legacy")[2], "Trust component_zmax": output_for_method(scored, physical["component_zmax"], "component_zmax")[2]}
            risks = {**trust_risks, **baseline_risks(scored)}
            train = scored["split"].eq("train").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            for method in METHODS:
                risk = risks[method]
                for q in Q_GRID:
                    threshold = float(np.nanquantile(risk[train & np.isfinite(risk)], q))
                    pred = risk[test] >= threshold
                    metrics = _metrics(yt, pred, risk[test])
                    rows.append({"fold": fold, "seed": seed, "method": method, "q": q, "threshold": threshold, "test_f1": float(metrics["f1"]), "test_recall": float(metrics["recall"]), "test_precision": float(metrics["precision"]), "test_auprc": float(metrics["average_precision"]), "test_far": float(pred[~yt].mean()) if (~yt).any() else float("nan")})
            print(json.dumps({"fold": fold, "seed": seed, "methods": len(METHODS), "q_values": len(Q_GRID)}, ensure_ascii=False), flush=True)
        print(json.dumps({"fold_completed": fold, "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby(["method", "q"], sort=False)[["test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["method", "q"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[2:]]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020–2025扩展表；11站点；四折滚动、5种子；Trust候选与Robust/Isolation Forest/One-Class SVM在相同训练正常风险分位数q下比较；Trust alpha/window由valid选择，阈值只由train正常样本确定；locked test仅作最终评价。", "stations": stations, "q_grid": Q_GRID, "methods": METHODS, "detail": rows, "aggregate": aggregate.to_dict(orient="records"), "limitations": ["比较用于公平运行点分析，不根据test选择主模型。", "攻击为透明受控注入，不是真实攻击日志。", "低FAR运行点必须同时报告Recall损失。"]}
    detail.to_csv(TABLES / "stage98_noaa_low_far_baseline_comparison_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage98_noaa_low_far_baseline_comparison.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage98_noaa_low_far_baseline_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段98 NOAA低误报运行点公平基线比较", "", "在相同训练正常风险分位数q下比较Trust legacy、component_zmax与三个无监督基线。", "", "| 方法 | q | test F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['q']:.3f} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} |")
    lines += ["", "该表用于低误报运行点的公平比较，不能替代各方法的F1最优运行点。"]
    (REPORTS / "stage98_noaa_low_far_baseline_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(detail), "methods": len(METHODS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
