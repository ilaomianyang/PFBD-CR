from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage70_noaa_isd_multiyear_normalized.parquet"
AUDIT = TABLES / "stage71_noaa_multiyear_quality_audit.json"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
SCORE_COLS = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency", "structural_risk"]
BOOTSTRAPS = 5000
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _best_threshold  # noqa: E402
from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, structural_features, structural_risk_from_norm  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage73_noaa_multiyear_controlled_benchmark import build_dataset  # noqa: E402


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


def metric(y: np.ndarray, p: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    normal = ~y
    return float(f1_score(y, p, zero_division=0)), float(average_precision_score(y, score)), float(p[normal].mean()) if normal.any() else float("nan")


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_multiyear_validation")}
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in [2021, 2022, 2023])})
    source = source[source["station_id"].astype(str).isin(stations) & source["year"].isin([2021, 2022, 2023])].copy()
    by_seed_station: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for seed in SEEDS:
        frame = build_dataset(source, seed, set(stations))
        scored, trust = score(frame)
        scored = add_structural(scored)
        train = scored["split"].eq("train").to_numpy()
        valid = scored["split"].eq("valid").to_numpy()
        test = scored["split"].eq("test").to_numpy()
        normal_train = train & (~scored["combined_is_attack"].astype(bool).to_numpy())
        X = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
        yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
        iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=1, contamination="auto").fit(X[normal_train])
        svm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)).fit(X[normal_train])
        svm_score = -svm.decision_function(X).ravel()
        svm_threshold, _ = _best_threshold(yv, svm_score[valid])
        trust_score = scored["robust_fused_risk"].to_numpy(dtype=float)
        trust_pred = trust_score >= float(trust["setting"]["threshold"])
        svm_pred = svm_score >= svm_threshold
        seed_map: dict[str, dict[str, np.ndarray]] = {}
        for station in stations:
            mask = test & scored["station_key"].astype(str).eq(station).to_numpy()
            seed_map[station] = {"y": scored.loc[mask, "combined_is_attack"].astype(bool).to_numpy(), "trust_pred": trust_pred[mask], "trust_score": trust_score[mask], "svm_pred": svm_pred[mask], "svm_score": svm_score[mask]}
        by_seed_station[seed] = seed_map
        print(json.dumps({"seed": seed, "stations": len(seed_map)}, ensure_ascii=False), flush=True)
    rng = np.random.default_rng(20260912)
    boot_rows: list[dict[str, object]] = []
    for b in range(BOOTSTRAPS):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        arrays = {"trust": {"y": [], "p": [], "s": []}, "svm": {"y": [], "p": [], "s": []}}
        for station in sampled:
            for seed in SEEDS:
                part = by_seed_station[seed][station]
                arrays["trust"]["y"].append(part["y"]); arrays["trust"]["p"].append(part["trust_pred"]); arrays["trust"]["s"].append(part["trust_score"])
                arrays["svm"]["y"].append(part["y"]); arrays["svm"]["p"].append(part["svm_pred"]); arrays["svm"]["s"].append(part["svm_score"])
        ty, tp, ts = (np.concatenate(arrays["trust"][key]) for key in ["y", "p", "s"])
        sy, sp, ss = (np.concatenate(arrays["svm"][key]) for key in ["y", "p", "s"])
        tf1, tap, tfar = metric(ty, tp, ts)
        sf1, sap, sfar = metric(sy, sp, ss)
        boot_rows.append({"bootstrap": b, "trust_f1": tf1, "svm_f1": sf1, "f1_diff_svm_minus_trust": sf1 - tf1, "trust_auprc": tap, "svm_auprc": sap, "auprc_diff_svm_minus_trust": sap - tap, "trust_far": tfar, "svm_far": sfar, "far_diff_svm_minus_trust": sfar - tfar})
    boot = pd.DataFrame(boot_rows)
    def ci(series: pd.Series) -> list[float]:
        return [float(series.quantile(0.025)), float(series.quantile(0.975))]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "跨年份2023 locked test；按11个站点簇、5种子进行5000次配对bootstrap；每种子阈值仅由2022 valid选择；不把逐行记录视为独立站点重复。", "stations": stations, "seeds": SEEDS, "bootstraps": BOOTSTRAPS, "summary": {"f1_diff_mean": float(boot["f1_diff_svm_minus_trust"].mean()), "f1_diff_ci95": ci(boot["f1_diff_svm_minus_trust"]), "auprc_diff_mean": float(boot["auprc_diff_svm_minus_trust"].mean()), "auprc_diff_ci95": ci(boot["auprc_diff_svm_minus_trust"]), "far_diff_mean": float(boot["far_diff_svm_minus_trust"].mean()), "far_diff_ci95": ci(boot["far_diff_svm_minus_trust"]), "prob_svm_f1_greater": float((boot["f1_diff_svm_minus_trust"] > 0).mean()), "prob_svm_far_lower": float((boot["far_diff_svm_minus_trust"] < 0).mean())}}
    boot.to_csv(TABLES / "stage77_noaa_multiyear_cluster_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage77_noaa_multiyear_cluster_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    s = payload["summary"]
    lines = ["# 阶段77 NOAA多年度站点簇级配对bootstrap", "", "以站点为重采样簇，5种子固定使用valid选择的阈值，比较One-Class SVM与TrustMeteo-QC在2023 locked test上的差异。", "", f"- F1差值（SVM−TrustMeteo）：`{s['f1_diff_mean']:.4f}`，95% CI `[{s['f1_diff_ci95'][0]:.4f}, {s['f1_diff_ci95'][1]:.4f}]`", f"- AUPRC差值：`{s['auprc_diff_mean']:.4f}`，95% CI `[{s['auprc_diff_ci95'][0]:.4f}, {s['auprc_diff_ci95'][1]:.4f}]`", f"- FAR差值（SVM−TrustMeteo）：`{s['far_diff_mean']:.4f}`，95% CI `[{s['far_diff_ci95'][0]:.4f}, {s['far_diff_ci95'][1]:.4f}]`", f"- bootstrap中SVM F1更高概率：`{s['prob_svm_f1_greater']:.4f}`；FAR更低概率：`{s['prob_svm_far_lower']:.4f}`", "", "该统计只支持跨年份外部比较，不改变主模型，也不把受控攻击当作真实攻击日志。"]
    (REPORTS / "stage77_noaa_multiyear_cluster_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
