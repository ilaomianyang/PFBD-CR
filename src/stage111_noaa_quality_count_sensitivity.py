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
MIN_FLAGGED_ROWS = [1, 3, 5, 10, 20]
BOOTSTRAPS = 5000
SEED = 20260926


def station_bootstrap(values_by_year: dict[int, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    years = np.array(sorted(values_by_year), dtype=int)
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for i in range(BOOTSTRAPS):
        sampled_years = years[rng.integers(0, len(years), size=len(years))]
        year_means: list[float] = []
        for year in sampled_years:
            values = values_by_year[int(year)]
            year_means.append(float(values[rng.integers(0, len(values), size=len(values))].mean()))
        draws[i] = float(np.mean(year_means))
    return draws


def main() -> int:
    detail = pd.read_csv(INPUT)
    required = {
        "target_year",
        "method",
        "q",
        "station_key",
        "quality_flagged_rows",
        "quality_clean_rows",
        "quality_flagged_alarm_rate",
        "quality_clean_alarm_rate",
    }
    if not required.issubset(detail.columns):
        raise ValueError(f"stage107 station detail missing columns: {sorted(required.difference(detail.columns))}")
    if set(detail["method"].unique()) != set(METHODS):
        raise ValueError(f"unexpected methods: {sorted(detail['method'].unique())}")
    for col in ["target_year", "quality_flagged_rows", "quality_clean_rows"]:
        detail[col] = detail[col].astype(int)
    detail["q"] = detail["q"].astype(float)
    detail["quality_flagged_alarm_rate"] = pd.to_numeric(detail["quality_flagged_alarm_rate"], errors="coerce")
    detail["quality_clean_alarm_rate"] = pd.to_numeric(detail["quality_clean_alarm_rate"], errors="coerce")
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for q in Q_GRID:
        q_detail = detail.loc[np.isclose(detail["q"], q)].copy()
        for method in METHODS:
            method_detail = q_detail.loc[
                (q_detail["method"] == method)
                & (q_detail["quality_clean_rows"] > 0)
                & q_detail["quality_flagged_alarm_rate"].notna()
                & q_detail["quality_clean_alarm_rate"].notna()
            ].copy()
            for minimum in MIN_FLAGGED_ROWS:
                subset = method_detail.loc[method_detail["quality_flagged_rows"] >= minimum].copy()
                if len(subset) < 5:
                    raise ValueError(f"too few station-year rows for q={q}, method={method}, minimum={minimum}: {len(subset)}")
                subset["contrast"] = subset["quality_flagged_alarm_rate"] - subset["quality_clean_alarm_rate"]
                by_year = {int(year): group["contrast"].to_numpy(dtype=float) for year, group in subset.groupby("target_year")}
                if len(by_year) < 3 or any(len(values) < 1 for values in by_year.values()):
                    raise ValueError(f"insufficient year/station coverage for q={q}, method={method}, minimum={minimum}")
                boot = station_bootstrap(by_year, rng)
                flagged_alarm_count = float((subset["quality_flagged_alarm_rate"] * subset["quality_flagged_rows"]).sum())
                clean_alarm_count = float((subset["quality_clean_alarm_rate"] * subset["quality_clean_rows"]).sum())
                flagged_rows = int(subset["quality_flagged_rows"].sum())
                clean_rows = int(subset["quality_clean_rows"].sum())
                rows.append(
                    {
                        "q": q,
                        "method": method,
                        "min_flagged_rows": minimum,
                        "target_years": len(by_year),
                        "min_stations_per_year": int(min(len(values) for values in by_year.values())),
                        "station_year_pairs": int(len(subset)),
                        "flagged_rows": flagged_rows,
                        "clean_rows": clean_rows,
                        "station_equal_contrast": float(subset["contrast"].mean()),
                        "station_equal_ci95_low": float(np.quantile(boot, 0.025)),
                        "station_equal_ci95_high": float(np.quantile(boot, 0.975)),
                        "row_weighted_contrast": flagged_alarm_count / flagged_rows - clean_alarm_count / clean_rows,
                        "bootstraps": BOOTSTRAPS,
                    }
                )

    out = pd.DataFrame(rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "读取阶段107自然观测站点明细；预先固定质量可疑记录数阈值为1/3/5/10/20；不删除主结果，只报告在阈值下仍有足够站点年份的敏感性；站点等权差异用年份一级、站点二级5000次bootstrap，另报告按行数加权差异；不使用攻击标签、不重新选择阈值。",
        "source": str(INPUT),
        "methods": METHODS,
        "q_grid": Q_GRID,
        "min_flagged_rows": MIN_FLAGGED_ROWS,
        "bootstraps": BOOTSTRAPS,
        "seed": SEED,
        "results": out.to_dict(orient="records"),
        "limitations": [
            "NOAA质量标记只是弱标签，任何差异都不能解释为攻击召回或真实攻击检测。",
            "阈值敏感性只能检验小样本记录影响，不能修复质量标记本身的不完美或站点选择偏差。",
            "行数加权和站点等权回答不同问题，必须并列呈现，不能挑选其一作为唯一结论。",
            "较高可疑行数阈值下部分年份的站点数较少，结果应作为诊断性敏感性分析解读。",
        ],
    }
    out.to_csv(TABLES / "stage111_noaa_quality_count_sensitivity.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage111_noaa_quality_count_sensitivity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 阶段111 NOAA质量弱标签可疑记录数敏感性",
        "",
        "预先固定最少质量可疑记录数阈值（1、3、5、10、20），同时报告站点等权层级bootstrap和行数加权差异。阈值只是敏感性分析，不用于删除主结果。",
        "",
        "| q | 方法 | 最少可疑行数 | 年份数 | 每年最少站点 | 站点-年份对 | 可疑行数 | 正常行数 | 站点等权差异 | 95% CI | 行数加权差异 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['q']:.3f} | {row['method']} | {row['min_flagged_rows']} | {row['target_years']} | {row['min_stations_per_year']} | {row['station_year_pairs']} | {row['flagged_rows']} | {row['clean_rows']} | {row['station_equal_contrast']:+.4f} | [{row['station_equal_ci95_low']:+.4f},{row['station_equal_ci95_high']:+.4f}] | {row['row_weighted_contrast']:+.4f} |"
        )
    lines += ["", "质量标记仅作弱标签；敏感性差异不能解释为攻击召回、真实攻击或跨域普适性。"]
    (REPORTS / "stage111_noaa_quality_count_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
