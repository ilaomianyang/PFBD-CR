from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
INPUT = TABLES / "stage107_noaa_natural_svm_validation_station_detail.csv"
METHODS = ["Trust legacy", "Trust component_zmax", "One-Class SVM"]
REFERENCES = ["Trust legacy", "Trust component_zmax"]
Q_GRID = [0.99, 0.995, 0.999]
BOOTSTRAPS = 5000


def main() -> int:
    detail = pd.read_csv(INPUT)
    required = {"target_year", "method", "q", "station_key", "alarm_rate"}
    if not required.issubset(detail.columns):
        raise ValueError(f"stage107 station detail missing columns: {sorted(required.difference(detail.columns))}")
    if set(detail["method"].unique()) != set(METHODS):
        raise ValueError(f"unexpected methods: {sorted(detail['method'].unique())}")
    rng = np.random.default_rng(20260923)
    result_rows: list[dict[str, object]] = []
    for target_year in sorted(detail["target_year"].unique()):
        for q in Q_GRID:
            subset = detail.loc[(detail["target_year"] == target_year) & np.isclose(detail["q"].astype(float), q)]
            wide = subset.pivot_table(index="station_key", columns="method", values="alarm_rate")
            if wide.isna().any().any() or len(wide) < 5:
                raise ValueError(f"incomplete station alarm table for year={target_year}, q={q}")
            stations = wide.index.to_numpy()
            for reference in REFERENCES:
                diff = wide["One-Class SVM"].to_numpy(dtype=float) - wide[reference].to_numpy(dtype=float)
                boot = np.asarray([float(diff[rng.integers(0, len(stations), size=len(stations))].mean()) for _ in range(BOOTSTRAPS)])
                result_rows.append({"target_year": int(target_year), "q": q, "candidate": "One-Class SVM", "reference": reference, "candidate_alarm_rate_mean": float(wide["One-Class SVM"].mean()), "reference_alarm_rate_mean": float(wide[reference].mean()), "observed_diff": float(diff.mean()), "ci95_low": float(np.quantile(boot, 0.025)), "ci95_high": float(np.quantile(boot, 0.975)), "prob_candidate_lower": float((boot < 0).mean()), "stations": int(len(stations)), "bootstraps": BOOTSTRAPS})
    out = pd.DataFrame(result_rows)
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "读取阶段107无攻击自然观测站点报警率；按目标年、q和站点做5000次等权配对bootstrap；不使用攻击标签、不重新选择阈值。", "source": str(INPUT), "methods": METHODS, "q_grid": Q_GRID, "bootstraps": BOOTSTRAPS, "results": out.to_dict(orient="records"), "limitations": ["自然报警率不是攻击召回率。", "站点簇bootstrap以站点等权，不替代按行数加权的总体报警率。", "NOAA质量标记仅是弱标签。"]}
    out.to_csv(TABLES / "stage108_noaa_natural_alarm_station_bootstrap.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage108_noaa_natural_alarm_station_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段108 NOAA自然报警率站点簇bootstrap", "", "对阶段107的自然报警率按目标年、q和站点执行5000次等权配对bootstrap；差值为 One-Class SVM 减参考。", "", "| 目标年 | q | 参考 | SVM报警率 | 参考报警率 | 差值 | 95% CI | SVM更低概率 |", "|---:|---:|---|---:|---:|---:|---|---:|"]
    for row in result_rows:
        lines.append(f"| {row['target_year']} | {row['q']:.3f} | {row['reference']} | {row['candidate_alarm_rate_mean']:.4f} | {row['reference_alarm_rate_mean']:.4f} | {row['observed_diff']:+.4f} | [{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}] | {row['prob_candidate_lower']:.4f} |")
    lines += ["", "该结果只支持自然观测报警率的站点稳定性，不支持攻击召回或真实攻击检测结论。"]
    (REPORTS / "stage108_noaa_natural_alarm_station_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
