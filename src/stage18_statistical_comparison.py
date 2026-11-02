from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from stage09_trust_pipeline import _best_threshold  # noqa: E402


def metric_vector(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    clean = ~y
    return {"f1": float(f1_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)), "auprc": float(average_precision_score(y, score)), "false_alarm_rate": float(pred[clean].mean()) if clean.any() else float("nan")}


def main() -> int:
    df = pd.read_parquet(ROOT / "tables" / "stage09b_local_robust_scores.parquet")
    cols = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency", "ERA5 Background", "structural_risk"]
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = df["combined_is_attack"].to_numpy(dtype=bool)
    train = df["split"].eq("train").to_numpy(); valid = df["split"].eq("valid").to_numpy(); test = df["split"].eq("test").to_numpy(); normal_train = train & ~y
    robust_json = json.loads((ROOT / "tables" / "stage09b_robust_fusion.json").read_text(encoding="utf-8"))
    scores = {"Stage09B robust fusion": df["robust_fused_risk"].to_numpy(dtype=float)}
    thresholds = {"Stage09B robust fusion": float(robust_json["setting"]["threshold"])}
    iso = IsolationForest(n_estimators=300, random_state=20260911, n_jobs=1, contamination="auto").fit(X[normal_train])
    scores["Isolation Forest"] = -iso.score_samples(X)
    thresholds["Isolation Forest"], _ = _best_threshold(y[valid], scores["Isolation Forest"][valid])
    pos_weight = max((y[train] == 0).sum() / max((y[train] == 1).sum(), 1), 1.0)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260911))
    lr.fit(X[train], y[train])
    scores["Logistic Regression supervised"] = lr.predict_proba(X)[:, 1]
    thresholds["Logistic Regression supervised"], _ = _best_threshold(y[valid], scores["Logistic Regression supervised"][valid])
    test_y = y[test]
    point = {name: metric_vector(test_y, score[test], thresholds[name]) for name, score in scores.items()}
    rng = np.random.default_rng(20260911)
    n_boot = 2000
    names = list(scores)
    diffs = []
    for comparator in names[1:]:
        vals = {"f1": [], "recall": [], "precision": [], "auprc": [], "false_alarm_rate": []}
        for _ in range(n_boot):
            idx = rng.integers(0, len(test_y), len(test_y))
            yb = test_y[idx]
            a = metric_vector(yb, scores["Stage09B robust fusion"][test][idx], thresholds["Stage09B robust fusion"])
            b = metric_vector(yb, scores[comparator][test][idx], thresholds[comparator])
            for metric in vals:
                vals[metric].append(a[metric] - b[metric])
        row = {"comparison": f"Stage09B robust fusion - {comparator}"}
        for metric, values in vals.items():
            arr = np.asarray(values, dtype=float)
            row[f"{metric}_difference_mean"] = float(np.mean(arr))
            row[f"{metric}_difference_ci_low"] = float(np.quantile(arr, 0.025))
            row[f"{metric}_difference_ci_high"] = float(np.quantile(arr, 0.975))
            row[f"{metric}_probability_positive"] = float(np.mean(arr > 0))
        diffs.append(row)
    detail = pd.DataFrame(diffs)
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    detail.to_csv(out_dir / "stage18_statistical_comparison.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "固定test预测与valid阈值；2000次配对bootstrap；未使用test重新调阈值", "point_metrics": point, "comparisons": detail.to_dict(orient="records")}
    (out_dir / "stage18_statistical_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段18配对统计比较", "", "使用固定 test 预测和 valid 选择的阈值，进行 2000 次配对 bootstrap。差值定义为 Stage09B 减去对比方法。", "", "| 比较 | F1差值95%CI | AUPRC差值95%CI | 误伤率差值95%CI |", "|---|---:|---:|---:|"]
    for _, r in detail.iterrows():
        lines.append(f"| {r['comparison']} | [{r['f1_difference_ci_low']:.4f}, {r['f1_difference_ci_high']:.4f}] | [{r['auprc_difference_ci_low']:.4f}, {r['auprc_difference_ci_high']:.4f}] | [{r['false_alarm_rate_difference_ci_low']:.4f}, {r['false_alarm_rate_difference_ci_high']:.4f}] |")
    lines += ["", "点估计", "", "| 方法 | F1 | Recall | Precision | AUPRC | 正常误伤率 |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in point.items():
        lines.append(f"| {name} | {m['f1']:.4f} | {m['recall']:.4f} | {m['precision']:.4f} | {m['auprc']:.4f} | {m['false_alarm_rate']:.4f} |")
    lines += ["", "若差值 CI 跨过 0，应在论文中写成‘未观察到统计显著优势’，不能写成确定性领先。"]
    report = report_dir / "stage18_statistical_comparison.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
