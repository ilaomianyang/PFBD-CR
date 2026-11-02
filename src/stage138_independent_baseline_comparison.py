from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SEEDS = [271828, 314159, 161803, 8675309, 424242]
SCORE_COLS = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency", "ERA5 Background", "structural_risk"]
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402


def evaluate(seed: int) -> list[dict[str, object]]:
    path = TABLES / f"stage09b_local_robust_scores_stage137_default_seed{seed}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    X = df[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    train = df["split"].eq("train").to_numpy()
    valid = df["split"].eq("valid").to_numpy()
    test = df["split"].eq("test").to_numpy()
    normal_train = train & ~df["combined_is_attack"].astype(bool).to_numpy()
    y_valid = df.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    y_test = df.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    scores: list[tuple[str, np.ndarray]] = []
    med = np.nanmedian(X[normal_train], axis=0)
    scale = np.maximum(np.nanmedian(np.abs(X[normal_train] - med), axis=0) * 1.4826, 0.25)
    scores.append(("Robust max deviation", np.maximum((X - med) / scale, 0.0).max(axis=1)))
    iso = IsolationForest(n_estimators=300, random_state=20260911, n_jobs=1, contamination="auto")
    iso.fit(X[normal_train])
    scores.append(("Isolation Forest", -iso.score_samples(X)))
    ocsvm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05))
    ocsvm.fit(X[normal_train])
    scores.append(("One-Class SVM", -ocsvm.decision_function(X).ravel()))
    rows: list[dict[str, object]] = []
    for name, score in scores:
        threshold, valid_f1 = _best_threshold(y_valid, score[valid])
        test_pred = score[test] >= threshold
        metrics = _metrics(y_test, test_pred, score[test])
        rows.append({"seed": seed, "method": name, "valid_f1": float(valid_f1), "valid_threshold": float(threshold), **{f"test_{key}": float(value) for key, value in {"f1": metrics["f1"], "recall": metrics["recall"], "precision": metrics["precision"], "auprc": metrics["average_precision"]}.items()}, "test_false_alarm_rate": float(test_pred[~y_test].mean())})
    payload = json.loads((TABLES / f"stage09b_robust_fusion_stage137_default_seed{seed}.json").read_text(encoding="utf-8"))
    rows.append({"seed": seed, "method": "Stage09B robust fusion", "valid_f1": float(payload["local_valid"]["f1"]), "valid_threshold": float(payload["setting"]["threshold"]), "test_f1": float(payload["local_test"]["f1"]), "test_recall": float(payload["local_test"]["recall"]), "test_precision": float(payload["local_test"]["precision"]), "test_auprc": float(payload["local_test"]["average_precision"]), "test_false_alarm_rate": float(payload["test_false_alarm_rate_on_clean_rows"])})
    return rows


def bootstrap_mean(values: np.ndarray, seed: int, repeats: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    rows = [row for seed in SEEDS for row in evaluate(seed)]
    detail = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, object]] = []
    for method_index, (method, group) in enumerate(detail.groupby("method", sort=True)):
        row: dict[str, object] = {"method": method, "seed_count": int(len(group))}
        for index, metric in enumerate(METRICS):
            values = group[metric].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean(values, 138138 + method_index * 100 + index)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    trust = detail[detail["method"] == "Stage09B robust fusion"].set_index("seed")
    paired_rows: list[dict[str, object]] = []
    for method in sorted(set(detail["method"]) - {"Stage09B robust fusion"}):
        candidate = detail[detail["method"] == method].set_index("seed")
        for metric in METRICS:
            diff = trust[metric] - candidate[metric]
            ci_low, ci_high = bootstrap_mean(diff.to_numpy(dtype=float), 138900 + len(paired_rows))
            paired_rows.append({"candidate": method, "reference": "Stage09B robust fusion", "metric": metric, "reference_minus_candidate_mean": float(diff.mean()), "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high})
    paired = pd.DataFrame(paired_rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段138在阶段135五个新种子上比较Stage09B与Robust max deviation、Isolation Forest、One-Class SVM；所有基线只用train正常样本拟合，阈值只由valid选择，test只评价。不同方法使用完全相同的输入特征和split，不根据test重新调参。",
        "seeds": SEEDS,
        "result_count": len(detail),
        "aggregate_count": len(aggregate),
        "paired_count": len(paired),
        "limitations": [
            "基线比较仍在同一气象数据源和同一受控攻击生成器上进行。",
            "五个种子级bootstrap只量化当前复现协议的不确定性，不能证明跨域普适。",
            "Stage09B若未全面优于所有基线，必须保留逐指标负结果。",
        ],
    }
    detail.to_csv(TABLES / "stage138_independent_baseline_comparison_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage138_independent_baseline_comparison_aggregate.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / "stage138_independent_baseline_comparison_paired.csv", index=False, encoding="utf-8-sig")
    payload["aggregate"] = aggregate.to_dict(orient="records")
    payload["paired"] = paired.to_dict(orient="records")
    (TABLES / "stage138_independent_baseline_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段138新种子公平基线比较", "", "所有方法使用相同输入和split；基线只用train正常样本拟合、valid选择阈值。", "", "| 方法 | F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate_rows:
        lines.append(f"| {row['method']} | {row['test_f1_mean']:.4f} | {row['test_recall_mean']:.4f} | {row['test_precision_mean']:.4f} | {row['test_auprc_mean']:.4f} | {row['test_false_alarm_rate_mean']:.4f} |")
    lines += ["", "逐种子结果和相对Stage09B的配对bootstrap见CSV；不能只报告对自己有利的指标。"]
    (REPORTS / "stage138_independent_baseline_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "result_count": len(detail), "aggregate_count": len(aggregate), "paired_count": len(paired)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
