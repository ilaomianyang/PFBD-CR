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
FOLDS = [([2020], 2021, 2022), ([2020, 2021], 2022, 2023), ([2020, 2021, 2022], 2023, 2024), ([2020, 2021, 2022, 2023], 2024, 2025)]
SCORE_COLS = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency", "structural_risk"]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402
from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, structural_features, structural_risk_from_norm  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402


def add_structural(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    train = out["split"].eq("train")
    normal_train = train & (~out["combined_is_attack"].astype(bool))
    parts = []
    for split_name in ["train", "valid", "test"]:
        mask = out["split"].eq(split_name)
        parts.append(structural_features(out.loc[mask], STRUCTURAL_BASE).set_axis(out.index[mask]))
    features = pd.concat(parts).reindex(out.index).fillna(0.0)
    features["station_key"] = out["station_key"].astype(str)
    scaler = AdaptiveScaler(STRUCTURAL_BASE).fit(features.loc[train], normal_train.loc[train])
    out["structural_risk"] = structural_risk_from_norm(scaler.transform(features), enhanced=False)
    return out


def evaluate(yv: np.ndarray, yt: np.ndarray, scores: np.ndarray, valid: np.ndarray, test: np.ndarray, name: str, fold: str, seed: int) -> dict[str, object]:
    threshold, valid_f1 = _best_threshold(yv, scores[valid])
    pred = scores[test] >= threshold
    m = _metrics(yt, pred, scores[test])
    return {"fold": fold, "seed": seed, "method": name, "valid_f1": float(valid_f1), "test_f1": float(m["f1"]), "test_recall": float(m["recall"]), "test_precision": float(m["precision"]), "test_auprc": float(m["average_precision"]), "test_false_alarm_rate": float(pred[~yt].mean()) if (~yt).any() else float("nan"), "threshold": float(threshold)}


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    rows: list[dict[str, object]] = []
    integrity: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold_name = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, trust = score(frame)
            scored = add_structural(scored)
            train = scored["split"].eq("train").to_numpy()
            valid = scored["split"].eq("valid").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            normal_train = train & (~scored["combined_is_attack"].astype(bool).to_numpy())
            yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
            yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            X = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            med = np.nanmedian(X[normal_train], axis=0)
            scale = np.maximum(1.4826 * np.nanmedian(np.abs(X[normal_train] - med), axis=0), 0.25)
            robust = np.maximum((X - med) / scale, 0.0).max(axis=1)
            iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=1, contamination="auto").fit(X[normal_train])
            iso_score = -iso.score_samples(X)
            svm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)).fit(X[normal_train])
            svm_score = -svm.decision_function(X).ravel()
            rows.append(evaluate(yv, yt, robust, valid, test, "Robust max deviation", fold_name, seed))
            rows.append(evaluate(yv, yt, iso_score, valid, test, "Isolation Forest", fold_name, seed))
            rows.append(evaluate(yv, yt, svm_score, valid, test, "One-Class SVM", fold_name, seed))
            rows.append({"fold": fold_name, "seed": seed, "method": "TrustMeteo-QC", "valid_f1": float(trust["valid"]["f1"]), "test_f1": float(trust["test"]["f1"]), "test_recall": float(trust["test"]["recall"]), "test_precision": float(trust["test"]["precision"]), "test_auprc": float(trust["test"]["average_precision"]), "test_false_alarm_rate": float(trust["test_false_alarm_rate"]), "threshold": float(trust["setting"]["threshold"])})
            integrity.append({"fold": fold_name, "train_years": train_years, "valid_year": valid_year, "test_year": test_year, "seed": seed, "rows": int(len(frame)), "train_attack_rows": int(frame.loc[frame["split"].eq("train"), "combined_is_attack"].sum()), "valid_attack_rows": int(frame.loc[frame["split"].eq("valid"), "combined_is_attack"].sum()), "test_attack_rows": int(frame.loc[frame["split"].eq("test"), "combined_is_attack"].sum())})
        print(json.dumps({"fold": fold_name, "methods": 4, "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("method", sort=False)[["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["method"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    fold_aggregate = detail.groupby(["fold", "method"], sort=False)[["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]].agg(["mean", "std"]).reset_index()
    fold_aggregate.columns = ["fold", "method"] + [f"{a}_{b}" for a, b in fold_aggregate.columns.tolist()[2:]]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020-2025扩展表；11个所有年度均通过质量审计的站点；同一累计历史训练→前一年验证→目标年测试滚动折；Robust/IF/SVM仅用train正常样本拟合、valid选阈值，test只评价；攻击为透明受控注入。", "stations": stations, "folds": [f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}" for train_years, valid_year, test_year in FOLDS], "seeds": SEEDS, "detail": rows, "aggregate": aggregate.to_dict(orient="records"), "fold_aggregate": fold_aggregate.to_dict(orient="records"), "integrity": integrity}
    detail.to_csv(TABLES / "stage85_noaa_rolling_baseline_comparison_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage85_noaa_rolling_baseline_comparison.csv", index=False, encoding="utf-8-sig")
    fold_aggregate.to_csv(TABLES / "stage85_noaa_rolling_baseline_by_fold.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage85_noaa_rolling_baseline_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段85 NOAA滚动多年度公平基线比较", "", "在阶段84相同的累计历史训练→前一年验证→目标年测试滚动折上比较四种方法。", "", "| 方法 | test F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_false_alarm_rate_mean']:.4f}±{row['test_false_alarm_rate_std']:.4f} |")
    lines += ["", "跨年份基线差异必须结合每个滚动折报告；不能将受控攻击结果描述为真实攻击检测。"]
    (REPORTS / "stage85_noaa_rolling_baseline_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(stations), "methods": int(detail["method"].nunique()), "folds": len(payload["folds"]), "seeds": len(SEEDS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
