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
METHODS = ["Robust max deviation", "Isolation Forest", "One-Class SVM", "TrustMeteo-QC"]
BOOTSTRAPS = 5000
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


def metric(y: np.ndarray, p: np.ndarray, s: np.ndarray) -> tuple[float, float, float, float]:
    m = _metrics(y, p, s)
    far = float(p[~y].mean()) if (~y).any() else float("nan")
    return float(m["f1"]), float(m["recall"]), float(m["average_precision"]), far


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    by_method: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {method: {} for method in METHODS}
    fold_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
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
            score_map = {"Robust max deviation": robust, "Isolation Forest": iso_score, "One-Class SVM": svm_score, "TrustMeteo-QC": scored["robust_fused_risk"].to_numpy(dtype=float)}
            prediction_map: dict[str, np.ndarray] = {}
            threshold_map: dict[str, float] = {}
            for method in ["Robust max deviation", "Isolation Forest", "One-Class SVM"]:
                threshold, _ = _best_threshold(yv, score_map[method][valid])
                threshold_map[method] = float(threshold)
                prediction_map[method] = score_map[method] >= threshold
            threshold_map["TrustMeteo-QC"] = float(trust["setting"]["threshold"])
            prediction_map["TrustMeteo-QC"] = score_map["TrustMeteo-QC"] >= threshold_map["TrustMeteo-QC"]
            fold_rows.append({"fold": fold_name, "seed": seed, "test_rows": int(test.sum()), "test_attack_rows": int(yt.sum()), **{f"{method}_f1": metric(yt, prediction_map[method][test], score_map[method][test])[0] for method in METHODS}, **{f"{method}_far": metric(yt, prediction_map[method][test], score_map[method][test])[3] for method in METHODS}})
            for method in METHODS:
                for station in stations:
                    mask = test & scored["station_key"].eq(station).to_numpy()
                    key = f"{fold_name}|{seed}|{station}"
                    by_method[method][key] = {"y": scored.loc[mask, "combined_is_attack"].astype(bool).to_numpy(), "p": prediction_map[method][mask], "s": score_map[method][mask]}
            selected_rows.append({"fold": fold_name, "seed": seed, "thresholds": threshold_map})
        print(json.dumps({"fold": fold_name, "seeds": len(SEEDS), "stations": len(stations)}, ensure_ascii=False), flush=True)

    station_counts: dict[str, dict[str, dict[str, int]]] = {method: {} for method in METHODS}
    for method in METHODS:
        for station in stations:
            totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            suffix = f"|{station}"
            for key, part in by_method[method].items():
                if not key.endswith(suffix):
                    continue
                y = np.asarray(part["y"], dtype=bool)
                p = np.asarray(part["p"], dtype=bool)
                totals["tp"] += int(np.sum(y & p))
                totals["fp"] += int(np.sum((~y) & p))
                totals["fn"] += int(np.sum(y & (~p)))
                totals["tn"] += int(np.sum((~y) & (~p)))
            station_counts[method][station] = totals

    rng = np.random.default_rng(20260912)
    boot_rows: list[dict[str, object]] = []
    for b in range(BOOTSTRAPS):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        metrics = {}
        for method in METHODS:
            totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for station in sampled:
                for field in totals:
                    totals[field] += station_counts[method][str(station)][field]
            precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
            recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            far = totals["fp"] / max(totals["fp"] + totals["tn"], 1)
            metrics[method] = {"f1": f1, "recall": recall, "far": far}
        row = {"bootstrap": b}
        for method in METHODS:
            safe = method.lower().replace(" ", "_").replace("-", "").replace("_qc", "qc")
            for metric_name in ["f1", "recall", "far"]:
                row[f"{safe}_{metric_name}"] = metrics[method][metric_name]
        for method in METHODS[:-1]:
            safe = method.lower().replace(" ", "_").replace("-", "").replace("_qc", "qc")
            row[f"{safe}_minus_trust_f1"] = metrics[method]["f1"] - metrics["TrustMeteo-QC"]["f1"]
            row[f"{safe}_minus_trust_far"] = metrics[method]["far"] - metrics["TrustMeteo-QC"]["far"]
        boot_rows.append(row)
    boot = pd.DataFrame(boot_rows)
    summary: dict[str, object] = {}
    for method in METHODS[:-1]:
        safe = method.lower().replace(" ", "_").replace("-", "").replace("_qc", "qc")
        f1_col = f"{safe}_minus_trust_f1"
        far_col = f"{safe}_minus_trust_far"
        summary[method] = {"f1_diff_mean": float(boot[f1_col].mean()), "f1_diff_ci95": [float(boot[f1_col].quantile(0.025)), float(boot[f1_col].quantile(0.975))], "far_diff_mean": float(boot[far_col].mean()), "far_diff_ci95": [float(boot[far_col].quantile(0.025)), float(boot[far_col].quantile(0.975))], "prob_f1_greater": float((boot[f1_col] > 0).mean()), "prob_far_lower": float((boot[far_col] < 0).mean())}
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020-2025扩展表；4个累计历史训练→前一年验证→目标年测试滚动折；11站点、5种子；以站点为重采样簇对预聚合TP/FP/FN/TN进行5000次配对bootstrap，报告F1/Recall/FAR；每种子阈值只由valid选择，test只评价。", "stations": stations, "folds": [f"{'-'.join(map(str, a))}->{b}->{c}" for a, b, c in FOLDS], "seeds": SEEDS, "bootstraps": BOOTSTRAPS, "fold_rows": fold_rows, "selected_thresholds": selected_rows, "station_counts": station_counts, "summary_vs_trust": summary}
    pd.DataFrame(fold_rows).to_csv(TABLES / "stage86_noaa_rolling_station_bootstrap_fold_detail.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(TABLES / "stage86_noaa_rolling_station_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage86_noaa_rolling_station_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段86 NOAA滚动站点簇级配对bootstrap", "", "在阶段85相同的四个滚动折上，以11个站点为重采样簇，比较基线与TrustMeteo-QC。", "", "| 方法 | F1差值均值 | F1差值95% CI | FAR差值均值 | FAR差值95% CI | F1更高概率 | FAR更低概率 |", "|---|---:|---|---:|---|---:|---:|"]
    for method, row in summary.items():
        lines.append(f"| {method}−TrustMeteo-QC | {row['f1_diff_mean']:.4f} | [{row['f1_diff_ci95'][0]:.4f}, {row['f1_diff_ci95'][1]:.4f}] | {row['far_diff_mean']:.4f} | [{row['far_diff_ci95'][0]:.4f}, {row['far_diff_ci95'][1]:.4f}] | {row['prob_f1_greater']:.4f} | {row['prob_far_lower']:.4f} |")
    lines += ["", "该统计只支持本扩展NOAA滚动协议下的站点簇级差异，不能外推为所有数据源或真实攻击场景的普遍结论。"]
    (REPORTS / "stage86_noaa_rolling_station_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary_vs_trust": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
