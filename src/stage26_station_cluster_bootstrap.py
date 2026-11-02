from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段26：留一站点簇级配对bootstrap")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    cold = pd.read_csv(TABLES / "stage22_leave_one_station_refit.csv")
    calibrated = pd.read_csv(TABLES / "stage23_unlabeled_site_calibration.csv")
    key = "held_out_station"
    merged = cold.merge(calibrated, on=key, suffixes=("_cold", "_calibrated"), validate="one_to_one")
    stations = merged[key].astype(str).to_numpy()
    rng = np.random.default_rng(args.seed)
    metrics = [
        ("test_f1", "F1"),
        ("test_recall", "Recall"),
        ("test_auprc", "AUPRC"),
        ("test_false_alarm_rate", "误伤率"),
        ("recovery_mae_after", "恢复MAE"),
        ("event_recall", "事件召回"),
    ]
    diffs = {name: np.empty(args.iterations, dtype=float) for name, _ in metrics}
    cold_means = {name: np.empty(args.iterations, dtype=float) for name, _ in metrics}
    cal_means = {name: np.empty(args.iterations, dtype=float) for name, _ in metrics}
    n = len(merged)
    for i in range(args.iterations):
        idx = rng.integers(0, n, size=n)
        for name, _ in metrics:
            c = merged[f"{name}_cold"].to_numpy(dtype=float)[idx]
            a = merged[f"{name}_calibrated"].to_numpy(dtype=float)[idx]
            cold_means[name][i] = float(np.mean(c))
            cal_means[name][i] = float(np.mean(a))
            diffs[name][i] = cal_means[name][i] - cold_means[name][i]
    rows = []
    for name, label in metrics:
        rows.append({
            "metric": name,
            "label": label,
            "cold_mean": float(merged[f"{name}_cold"].mean()),
            "cold_ci95": ci(cold_means[name]),
            "calibrated_mean": float(merged[f"{name}_calibrated"].mean()),
            "calibrated_ci95": ci(cal_means[name]),
            "difference_calibrated_minus_cold": float(merged[f"{name}_calibrated"].mean() - merged[f"{name}_cold"].mean()),
            "difference_ci95": ci(diffs[name]),
        })
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "以站点为重采样簇进行配对bootstrap；每次重采样包含7个留出站点，比较阶段23无标签校准与阶段22冷启动严格重拟合。",
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "stations": stations.tolist(),
        "results": rows,
        "interpretation": "站点簇数量为7，置信区间用于描述跨站点不确定性，不替代更多独立站点数据。",
    }
    (TABLES / "stage26_station_cluster_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage26_station_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 阶段26留一站点簇级配对bootstrap",
        "",
        "以7个留出站点为重采样簇，比较阶段23无标签校准相对阶段22冷启动的均值差异。该分析描述站点间不确定性，不把7个站点当作大量独立样本。",
        "",
        "| 指标 | 阶段22均值 | 阶段22 CI95% | 阶段23均值 | 阶段23 CI95% | 23−22差值 | 差值 CI95% |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for r in rows:
        lines.append(f"| {r['label']} | {r['cold_mean']:.4f} | [{r['cold_ci95'][0]:.4f}, {r['cold_ci95'][1]:.4f}] | {r['calibrated_mean']:.4f} | [{r['calibrated_ci95'][0]:.4f}, {r['calibrated_ci95'][1]:.4f}] | {r['difference_calibrated_minus_cold']:+.4f} | [{r['difference_ci95'][0]:+.4f}, {r['difference_ci95'][1]:+.4f}] |")
    lines += ["", "置信区间受站点数限制，应在论文中作为跨站点审计的不确定性描述，并与逐站点结果同时给出。"]
    (REPORTS / "stage26_station_cluster_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "iterations": args.iterations, "results": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
