from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
N_BOOT = 10000
SEED = 20260914


def main() -> int:
    source = TABLES / "stage122_leave_one_station_threshold_transfer_selected.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    detail = pd.read_csv(source)
    required = {"heldout_station", "baseline_f1", "source_baseline_f1", "strict_f1", "adapted_f1", "baseline_recall", "source_baseline_recall", "strict_recall", "adapted_recall", "baseline_far", "source_baseline_far", "strict_far", "adapted_far"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    stations = sorted(detail["heldout_station"].astype(str).unique().tolist())
    grouped = detail.assign(heldout_station=detail["heldout_station"].astype(str)).groupby("heldout_station", sort=True)
    station = grouped.mean(numeric_only=True).reset_index()
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for reference in ["baseline", "source_baseline"]:
        for method in ["strict", "adapted"]:
            for metric in ["f1", "recall", "far"]:
                difference = station[f"{method}_{metric}"].to_numpy(dtype=float) - station[f"{reference}_{metric}"].to_numpy(dtype=float)
                observed = float(difference.mean())
                samples = rng.integers(0, len(station), size=(N_BOOT, len(station)))
                boot = difference[samples].mean(axis=1)
                rows.append({
                    "comparison": f"{method}_vs_{reference}",
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "stations": len(station),
                    "observed_diff": observed,
                    "bootstrap_mean": float(boot.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "prob_diff_positive": float(np.mean(boot > 0.0)),
                    "prob_diff_negative": float(np.mean(boot < 0.0)),
                    "n_boot": N_BOOT,
                })
    result = pd.DataFrame(rows)
    strict_f1 = float(result.loc[(result["method"] == "strict") & (result["reference"] == "baseline") & (result["metric"] == "f1"), "observed_diff"].iloc[0])
    adapted_f1 = float(result.loc[(result["method"] == "adapted") & (result["reference"] == "baseline") & (result["metric"] == "f1"), "observed_diff"].iloc[0])
    strict_source_f1 = float(result.loc[(result["method"] == "strict") & (result["reference"] == "source_baseline") & (result["metric"] == "f1"), "observed_diff"].iloc[0])
    adapted_source_f1 = float(result.loc[(result["method"] == "adapted") & (result["reference"] == "source_baseline") & (result["metric"] == "f1"), "observed_diff"].iloc[0])
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "读取阶段122每个留出站点×种子的locked test结果，先在站点内对五个种子取均值，再以7个留出站点为独立簇做10000次有放回配对bootstrap；不重新选择q、不重新训练、不修改阶段9B/9C结果。严格迁移与目标历史无标签适配分别相对全站点原基线和源站点valid-only基线比较。",
        "source": str(source),
        "stations": stations,
        "n_boot": N_BOOT,
        "seed": SEED,
        "results": result.to_dict(orient="records"),
        "interpretation": {
            "strict_f1": "严格留一站点迁移的差值若区间跨零，只能视为未证实改善；负差值应保留。",
            "adapted_f1": "目标历史适配若差值为正，也只能说明当前7站点集合上的轻量适配候选，不能替代严格冷启动。",
        },
        "limitations": [
            "只有7个站点簇，bootstrap区间反映站点集合不确定性，不等同于大规模跨区域泛化。",
            "阶段122没有重新拟合检测器，因此该检验只针对阈值迁移/适配，不是完整模型留一站点重训。",
            "攻击标签为受控注入，不能解释为真实攻击检测能力。",
        ],
    }
    result.to_csv(TABLES / "stage123_threshold_transfer_bootstrap.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage123_threshold_transfer_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段123留一站点阈值迁移站点簇bootstrap", "",
        f"- 状态：**{payload['status']}**", f"- 站点簇：`{len(station)}`；bootstrap：`{N_BOOT}`次",
        f"- 严格迁移相对全站点基线F1差值：`{strict_f1:+.4f}`", f"- 历史适配相对全站点基线F1差值：`{adapted_f1:+.4f}`", f"- 严格迁移相对源站点valid-only基线F1差值：`{strict_source_f1:+.4f}`", f"- 历史适配相对源站点valid-only基线F1差值：`{adapted_source_f1:+.4f}`", "",
        "| 比较 | 指标 | 观察差值 | 95% CI | 差值为正概率 | 差值为负概率 |", "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['observed_diff']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['prob_diff_positive']:.4f} | {row['prob_diff_negative']:.4f} |")
    lines += ["", "严格迁移与目标历史适配不得合并为同一冷启动结论；区间跨零或站点方向不一致时保留不确定性。"]
    (REPORTS / "stage123_threshold_transfer_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(result), "stations": len(station), "n_boot": N_BOOT}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
