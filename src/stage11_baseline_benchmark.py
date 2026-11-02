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
sys.path.insert(0, str(SRC))
from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402
from trust_fusion import robust_location_scale  # noqa: E402


def main() -> int:
    path = ROOT / "tables" / "stage09b_local_robust_scores.parquet"
    df = pd.read_parquet(path)
    score_cols = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency", "ERA5 Background", "structural_risk"]
    X = df[score_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    train = df["split"].eq("train").to_numpy(); valid = df["split"].eq("valid").to_numpy(); test = df["split"].eq("test").to_numpy()
    normal_train = train & ~df["combined_is_attack"].to_numpy(dtype=bool)
    y_valid = df.loc[valid, "combined_is_attack"].to_numpy(dtype=bool); y_test = df.loc[test, "combined_is_attack"].to_numpy(dtype=bool)
    outputs = []

    med = np.nanmedian(X[normal_train], axis=0)
    scale = np.nanmedian(np.abs(X[normal_train] - med), axis=0) * 1.4826
    scale = np.maximum(scale, 0.25)
    robust_score = np.maximum((X - med) / scale, 0.0).max(axis=1)
    outputs.append(("Robust max deviation", robust_score))

    # 单进程以保证 Windows/受限运行环境下的可复现性。
    iso = IsolationForest(n_estimators=300, random_state=20260911, n_jobs=1, contamination="auto")
    iso.fit(X[normal_train])
    outputs.append(("Isolation Forest", -iso.score_samples(X)))

    ocsvm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05))
    ocsvm.fit(X[normal_train])
    outputs.append(("One-Class SVM", -ocsvm.decision_function(X).ravel()))

    rows = []
    for name, score in outputs:
        threshold, valid_f1 = _best_threshold(y_valid, score[valid])
        valid_pred = score[valid] >= threshold; test_pred = score[test] >= threshold
        clean = ~y_test
        rows.append({"method": name, "valid_f1": valid_f1, "valid_threshold": threshold, "test_f1": _metrics(y_test, test_pred, score[test])["f1"], "test_recall": _metrics(y_test, test_pred, score[test])["recall"], "test_precision": _metrics(y_test, test_pred, score[test])["precision"], "test_auprc": _metrics(y_test, test_pred, score[test])["average_precision"], "test_false_alarm_rate": float(test_pred[clean].mean()) if clean.any() else float("nan")})
    robust = json.loads((ROOT / "tables" / "stage09b_robust_fusion.json").read_text(encoding="utf-8"))
    rows.append({"method": "Stage09B robust fusion", "valid_f1": robust["local_valid"]["f1"], "valid_threshold": robust["setting"]["threshold"], "test_f1": robust["local_test"]["f1"], "test_recall": robust["local_test"]["recall"], "test_precision": robust["local_test"]["precision"], "test_auprc": robust["local_test"]["average_precision"], "test_false_alarm_rate": robust["test_false_alarm_rate_on_clean_rows"]})
    result = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    result.to_csv(out_dir / "stage11_baseline_benchmark.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "所有方法均只用train正常样本拟合、valid选阈值、test评价；非监督基线不使用攻击标签训练", "results": result.to_dict(orient="records")}
    (out_dir / "stage11_baseline_benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段11公平无监督基线比较", "", "所有方法使用相同的 train/valid/test 划分；Isolation Forest 与 One-Class SVM 只在 train 正常样本上拟合，阈值只在 valid 选择。", "", "| 方法 | test F1 | Recall | Precision | AUPRC | 正常误伤率 |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in result.iterrows():
        lines.append(f"| {r['method']} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_precision']:.4f} | {r['test_auprc']:.4f} | {r['test_false_alarm_rate']:.4f} |")
    lines += ["", "基线结果用于公平比较，不应删除表现较弱的方法。Stage09B 的结构化特征仍需结合多种子和外部验证解释。"]
    report = report_dir / "stage11_baseline_benchmark.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
