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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, structural_features, structural_risk_from_norm  # noqa: E402
from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402

TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
SCORE_COLS = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency", "structural_risk"]


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


def evaluate_score(yv: np.ndarray, yt: np.ndarray, score_all: np.ndarray, valid_mask: np.ndarray, test_mask: np.ndarray, name: str, seed: int) -> dict[str, object]:
    threshold, valid_f1 = _best_threshold(yv, score_all[valid_mask])
    test_score = score_all[test_mask]
    pred = test_score >= threshold
    mt = _metrics(yt, pred, test_score)
    return {"seed": seed, "method": name, "valid_f1": valid_f1, "test_f1": mt["f1"], "test_recall": mt["recall"], "test_precision": mt["precision"], "test_auprc": mt["average_precision"], "test_false_alarm_rate": float(pred[~yt].mean()) if (~yt).any() else float("nan")}


def main() -> int:
    source = pd.read_parquet(NOAA)
    rows = []
    integrity = []
    for seed in SEEDS:
        frame = build_dataset(source, seed)
        scored, trust = score(frame)
        scored = add_structural(scored)
        train = scored["split"].eq("train").to_numpy()
        valid = scored["split"].eq("valid").to_numpy()
        test = scored["split"].eq("test").to_numpy()
        normal_train = train & (~scored["combined_is_attack"].astype(bool).to_numpy())
        X = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
        yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
        med = np.nanmedian(X[normal_train], axis=0)
        scale = np.maximum(1.4826 * np.nanmedian(np.abs(X[normal_train] - med), axis=0), 0.25)
        robust = np.maximum((X - med) / scale, 0.0).max(axis=1)
        iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=1, contamination="auto").fit(X[normal_train])
        iso_score = -iso.score_samples(X)
        svm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)).fit(X[normal_train])
        svm_score = -svm.decision_function(X).ravel()
        rows.append(evaluate_score(yv, yt, robust, valid, test, "Robust max deviation", seed))
        rows.append(evaluate_score(yv, yt, iso_score, valid, test, "Isolation Forest", seed))
        rows.append(evaluate_score(yv, yt, svm_score, valid, test, "One-Class SVM", seed))
        rows.append({"seed": seed, "method": "TrustMeteo-QC", "valid_f1": trust["valid"]["f1"], "test_f1": trust["test"]["f1"], "test_recall": trust["test"]["recall"], "test_precision": trust["test"]["precision"], "test_auprc": trust["test"]["average_precision"], "test_false_alarm_rate": trust["test_false_alarm_rate"]})
        integrity.append({"seed": seed, "train_attack_rows": int(scored.loc[train, "combined_is_attack"].sum()), "valid_rows": int(valid.sum()), "test_rows": int(test.sum())})
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("method", sort=False)[["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["method"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA 12站点同一时间切分、同一受控攻击、五种子；所有无监督基线仅用train正常样本拟合，valid选阈值，test评价。", "seeds": SEEDS, "integrity": integrity, "detail": rows, "aggregate": aggregate.to_dict(orient="records")}
    detail.to_csv(TABLES / "stage41_noaa_baseline_comparison_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage41_noaa_baseline_comparison.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage41_noaa_baseline_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段41 NOAA外部受控基线比较", "", "所有方法使用相同NOAA时间切分和受控攻击；无监督基线只用train正常样本拟合。", "", "| 方法 | test F1均值±标准差 | Recall均值±标准差 | AUPRC均值±标准差 | FAR均值±标准差 |", "|---|---:|---:|---:|---:|"]
    for r in aggregate.to_dict(orient="records"):
        lines.append(f"| {r['method']} | {r['test_f1_mean']:.4f}±{r['test_f1_std']:.4f} | {r['test_recall_mean']:.4f}±{r['test_recall_std']:.4f} | {r['test_auprc_mean']:.4f}±{r['test_auprc_std']:.4f} | {r['test_false_alarm_rate_mean']:.4f}±{r['test_false_alarm_rate_std']:.4f} |")
    lines += ["", "该表用于外部数据源的公平比较，不替代真实攻击日志；完整每种子结果见 `tables/stage41_noaa_baseline_comparison_detail.csv`。"]
    (REPORTS / "stage41_noaa_baseline_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "methods": int(detail["method"].nunique()), "seeds": len(SEEDS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
