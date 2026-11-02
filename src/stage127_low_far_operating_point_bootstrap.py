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


def main() -> int:
    source = TABLES / "stage126_low_far_leave_one_station_operating_points_selected.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    detail = pd.read_csv(source)
    required = {"far_cap", "held_out_station", "baseline_test_f1", "test_f1", "baseline_test_recall", "test_recall", "baseline_test_false_alarm_rate", "test_false_alarm_rate"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for far_cap, group in detail.groupby("far_cap", sort=True):
        station = group.groupby("held_out_station", as_index=False).mean(numeric_only=True)
        for metric in ["f1", "recall", "false_alarm_rate"]:
            difference = station[f"test_{metric}"].to_numpy(dtype=float) - station[f"baseline_test_{metric}"].to_numpy(dtype=float)
            indices = rng.integers(0, len(station), size=(N_BOOT, len(station)))
            boot = difference[indices].mean(axis=1)
            rows.append({
                "far_cap": float(far_cap),
                "metric": metric,
                "stations": len(station),
                "observed_diff": float(difference.mean()),
                "bootstrap_mean": float(boot.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "prob_diff_positive": float(np.mean(boot > 0.0)),
                "prob_diff_negative": float(np.mean(boot < 0.0)),
                "n_boot": N_BOOT,
                "bootstrap_seed": SEED,
            })
    result = pd.DataFrame(rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "读取阶段126每个FAR上限的5种子×7站点locked test结果；先在留出站点内对种子取均值，再以7个站点为独立簇做10000次配对bootstrap，比较低误报运行点与阶段124 F1最优阈值；不重新选阈值、不重新训练。",
        "source": str(source),
        "far_caps": sorted(detail["far_cap"].unique().tolist()),
        "n_boot": N_BOOT,
        "bootstrap_seed": SEED,
        "results": result.to_dict(orient="records"),
        "limitations": [
            "只有7个站点簇；区间描述当前站点集合，不是跨区域泛化保证。",
            "FAR上限只在源站点valid宏平均上约束，留出站点test可能超出上限。",
            "阶段124检测器和恢复策略未改变，受控攻击标签不等同于真实攻击。",
        ],
    }
    result.to_csv(TABLES / "stage127_low_far_operating_point_bootstrap.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage127_low_far_operating_point_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段127低误报运行点站点簇bootstrap", "",
        f"- 状态：**{payload['status']}**", f"- 站点簇：`7`；bootstrap：`{N_BOOT}`次", "",
        "| valid FAR上限 | 指标 | 观察差值 | 95% CI | 差值为正概率 | 差值为负概率 |", "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['far_cap']:.3f} | {row['metric']} | {row['observed_diff']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['prob_diff_positive']:.4f} | {row['prob_diff_negative']:.4f} |")
    lines += ["", "低误报运行点与F1最优点必须并列报告；任何跨零区间和薄弱站点均保留。"]
    (REPORTS / "stage127_low_far_operating_point_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(result), "caps": len(payload["far_caps"]), "stations": 7}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
