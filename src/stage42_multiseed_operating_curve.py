from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage09_trust_pipeline import _best_threshold, _metrics

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
TARGETS = [0.005, 0.01, 0.02, 0.05]


def evaluate_point(yv: np.ndarray, rv: np.ndarray, yt: np.ndarray, rt: np.ndarray, label: str, threshold: float, target: float | None) -> dict[str, object]:
    mv = _metrics(yv, rv >= threshold, rv)
    mt = _metrics(yt, rt >= threshold, rt)
    return {"operating_point": label, "target_valid_far": target, "threshold": float(threshold), "valid_f1": mv["f1"], "valid_far": float((rv[~yv] >= threshold).mean()), "test_f1": mt["f1"], "test_recall": mt["recall"], "test_precision": mt["precision"], "test_auprc": mt["average_precision"], "test_far": float((rt[~yt] >= threshold).mean())}


def main() -> int:
    rows = []
    for seed in SEEDS:
        path = TABLES / f"stage09b_local_robust_scores_seed{seed}.parquet"
        df = pd.read_parquet(path)
        valid = df["split"].eq("valid").to_numpy()
        test = df["split"].eq("test").to_numpy()
        yv = df.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
        yt = df.loc[test, "combined_is_attack"].astype(bool).to_numpy()
        rv = df.loc[valid, "robust_fused_risk"].to_numpy(dtype=float)
        rt = df.loc[test, "robust_fused_risk"].to_numpy(dtype=float)
        clean_valid = rv[~yv]
        for target in TARGETS:
            threshold = float(np.quantile(clean_valid, 1.0 - target))
            rows.append({"seed": seed, **evaluate_point(yv, rv, yt, rt, f"valid_FAR_{target:.1%}", threshold, target)})
        threshold, _ = _best_threshold(yv, rv)
        rows.append({"seed": seed, **evaluate_point(yv, rv, yt, rt, "valid_F1_optimal", threshold, None)})
    detail = pd.DataFrame(rows)
    metrics = ["valid_f1", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]
    aggregate = detail.groupby("operating_point", sort=False)[metrics].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["operating_point"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "读取已完成的9B五种子锁定分数；阈值只由各自valid正常样本分位数或valid F1选择；test只评价。", "seeds": SEEDS, "targets": TARGETS, "detail": rows, "aggregate": aggregate.to_dict(orient="records")}
    detail.to_csv(TABLES / "stage42_multiseed_operating_curve_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage42_multiseed_operating_curve.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage42_multiseed_operating_curve.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段42 9B多种子低误报运行曲线", "", "各种子独立由valid选择运行点，locked test只评价。", "", "| 运行点 | test FAR均值±标准差 | Precision均值±标准差 | Recall均值±标准差 | F1均值±标准差 | AUPRC均值±标准差 |", "|---|---:|---:|---:|---:|---:|"]
    for r in aggregate.to_dict(orient="records"):
        lines.append(f"| {r['operating_point']} | {r['test_far_mean']:.4f}±{r['test_far_std']:.4f} | {r['test_precision_mean']:.4f}±{r['test_precision_std']:.4f} | {r['test_recall_mean']:.4f}±{r['test_recall_std']:.4f} | {r['test_f1_mean']:.4f}±{r['test_f1_std']:.4f} | {r['test_auprc_mean']:.4f}±{r['test_auprc_std']:.4f} |")
    lines += ["", "完整每种子结果见 `tables/stage42_multiseed_operating_curve_detail.csv`。"]
    (REPORTS / "stage42_multiseed_operating_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "points": len(aggregate), "seeds": len(SEEDS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
