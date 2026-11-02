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
Q_GRID = [0.99, 0.995, 0.999]
BOOTSTRAPS = 5000


def main() -> int:
    detail = pd.read_csv(INPUT)
    required = {"target_year", "method", "q", "station_key", "quality_flagged_rows", "quality_clean_rows", "quality_flagged_alarm_rate", "quality_clean_alarm_rate"}
    if not required.issubset(detail.columns):
        raise ValueError(f"stage107 station detail missing columns: {sorted(required.difference(detail.columns))}")
    rng = np.random.default_rng(20260924)
    rows: list[dict[str, object]] = []
    for target_year in sorted(detail["target_year"].unique()):
        for q in Q_GRID:
            for method in METHODS:
                subset = detail.loc[(detail["target_year"] == target_year) & (detail["method"] == method) & np.isclose(detail["q"].astype(float), q)].copy()
                usable = subset[(subset["quality_flagged_rows"] > 0) & (subset["quality_clean_rows"] > 0) & subset["quality_flagged_alarm_rate"].notna() & subset["quality_clean_alarm_rate"].notna()].copy()
                if len(usable) < 5:
                    raise ValueError(f"too few usable stations for year={target_year}, q={q}, method={method}: {len(usable)}")
                contrast = usable["quality_flagged_alarm_rate"].to_numpy(dtype=float) - usable["quality_clean_alarm_rate"].to_numpy(dtype=float)
                boot = np.asarray([float(contrast[rng.integers(0, len(contrast), size=len(contrast))].mean()) for _ in range(BOOTSTRAPS)])
                rows.append({"target_year": int(target_year), "q": q, "method": method, "stations_usable": int(len(usable)), "flagged_rows": int(usable["quality_flagged_rows"].sum()), "clean_rows": int(usable["quality_clean_rows"].sum()), "contrast_mean": float(contrast.mean()), "ci95_low": float(np.quantile(boot, 0.025)), "ci95_high": float(np.quantile(boot, 0.975)), "prob_contrast_positive": float((boot > 0).mean()), "bootstraps": BOOTSTRAPS})
    out = pd.DataFrame(rows)
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "读取阶段107自然观测站点明细；仅对同时存在质量可疑和质量正常样本的站点，按目标年/q/方法做5000次等权站点bootstrap；NOAA质量标记仅作弱标签，不作攻击真值。", "source": str(INPUT), "methods": METHODS, "q_grid": Q_GRID, "bootstraps": BOOTSTRAPS, "results": out.to_dict(orient="records"), "limitations": ["质量标记为弱标签，不能解释为攻击召回或真实攻击标签。", "仅使用同时有两组样本的站点，样本数和站点异质性仍需披露。", "站点等权bootstrap不替代按行数加权的总体报警率。"]}
    out.to_csv(TABLES / "stage109_noaa_quality_weak_label_bootstrap.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage109_noaa_quality_weak_label_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段109 NOAA质量弱标签站点簇bootstrap", "", "对阶段107的质量可疑/正常报警率差异做5000次等权站点bootstrap；差值为质量可疑报警率减质量正常报警率。", "", "| 目标年 | 方法 | q | 可用站点 | 可疑行数 | 正常行数 | 差异 | 95% CI | 差异为正概率 |", "|---:|---|---:|---:|---:|---:|---:|---|---:|"]
    for row in rows:
        lines.append(f"| {row['target_year']} | {row['method']} | {row['q']:.3f} | {row['stations_usable']} | {row['flagged_rows']} | {row['clean_rows']} | {row['contrast_mean']:+.4f} | [{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}] | {row['prob_contrast_positive']:.4f} |")
    lines += ["", "差异为正只表示质量可疑样本的报警率更高，不表示检测到攻击；质量标记是弱标签。"]
    (REPORTS / "stage109_noaa_quality_weak_label_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
