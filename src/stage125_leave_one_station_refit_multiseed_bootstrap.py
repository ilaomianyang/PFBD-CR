from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
N_BOOT = 10000
SEED = 20260915


def _bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    indices = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    samples = values[indices].mean(axis=1)
    return observed, float(samples.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> int:
    source = TABLES / "stage124_leave_one_station_refit_multiseed.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    detail = pd.read_csv(source)
    if len(detail) != 35 or detail["seed"].nunique() != 5 or detail["held_out_station"].nunique() != 7:
        raise ValueError("stage124 must contain 5 seeds × 7 stations")
    metrics = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate", "recovery_mae_before", "recovery_mae_after", "event_recall"]
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for level, grouped in [("seed", detail.groupby("seed", sort=True)), ("station", detail.groupby("held_out_station", sort=True))]:
        aggregate = grouped[metrics].mean(numeric_only=True).reset_index()
        for metric in metrics:
            observed, boot_mean, low, high = _bootstrap(aggregate[metric].to_numpy(dtype=float), rng)
            rows.append({
                "cluster_level": level,
                "clusters": int(len(aggregate)),
                "metric": metric,
                "observed_mean": observed,
                "bootstrap_mean": boot_mean,
                "ci95_low": low,
                "ci95_high": high,
                "n_boot": N_BOOT,
                "bootstrap_seed": SEED,
            })
    result = pd.DataFrame(rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "读取阶段124五种子×七站点严格留一站点重拟合的locked test结果；先按种子或留出站点聚合，再分别对5个种子簇和7个站点簇执行10000次均值bootstrap；不重新训练、不重新选择模型或阈值，所有负方向和跨零区间保留。",
        "source": str(source),
        "n_boot": N_BOOT,
        "bootstrap_seed": SEED,
        "results": result.to_dict(orient="records"),
        "limitations": [
            "种子级只有5个簇，站点级只有7个簇，区间不能解释为大规模泛化保证。",
            "阶段124使用受控攻击构造，不能替代真实网络攻击数据。",
            "恢复MAE与检测指标的簇级不确定性应分开解释，不应只挑选改善最大的站点。",
        ],
    }
    result.to_csv(TABLES / "stage125_leave_one_station_refit_multiseed_bootstrap.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage125_leave_one_station_refit_multiseed_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段125严格留一站点多种子bootstrap", "",
        f"- 状态：**{payload['status']}**", f"- bootstrap：`{N_BOOT}`次；种子簇=`5`；站点簇=`7`", "",
        "| 聚合层级 | 指标 | 观察均值 | 95% CI |", "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['cluster_level']}({row['clusters']}簇) | {row['metric']} | {row['observed_mean']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |")
    lines += ["", "该bootstrap只量化阶段124的种子/站点簇波动，不把均值包装成普适性证明。"]
    (REPORTS / "stage125_leave_one_station_refit_multiseed_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(result), "seed_clusters": 5, "station_clusters": 7, "n_boot": N_BOOT}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
