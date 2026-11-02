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
SEED = 20260925


def hierarchical_bootstrap(values_by_year: dict[int, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    years = np.array(sorted(values_by_year), dtype=int)
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for i in range(BOOTSTRAPS):
        sampled_years = years[rng.integers(0, len(years), size=len(years))]
        year_means: list[float] = []
        for year in sampled_years:
            values = values_by_year[int(year)]
            sampled_values = values[rng.integers(0, len(values), size=len(values))]
            year_means.append(float(sampled_values.mean()))
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

    detail["target_year"] = detail["target_year"].astype(int)
    detail["q"] = detail["q"].astype(float)
    usable = detail.loc[
        (detail["quality_flagged_rows"] > 0)
        & (detail["quality_clean_rows"] > 0)
        & detail["quality_flagged_alarm_rate"].notna()
        & detail["quality_clean_alarm_rate"].notna()
    ].copy()
    usable["quality_contrast"] = usable["quality_flagged_alarm_rate"] - usable["quality_clean_alarm_rate"]
    rng = np.random.default_rng(SEED)
    result_rows: list[dict[str, object]] = []

    for q in Q_GRID:
        q_detail = usable.loc[np.isclose(usable["q"], q)].copy()
        contrast_wide = q_detail.pivot_table(index=["target_year", "station_key"], columns="method", values="quality_contrast")
        if contrast_wide.empty or contrast_wide[METHODS].isna().any().any():
            raise ValueError(f"incomplete paired quality contrast table for q={q}")
        if contrast_wide.index.get_level_values("target_year").nunique() < 3:
            raise ValueError(f"too few target years for q={q}")

        for reference in REFERENCES:
            paired = contrast_wide["One-Class SVM"] - contrast_wide[reference]
            by_year = {
                int(year): group.to_numpy(dtype=float)
                for year, group in paired.groupby(level="target_year")
            }
            if any(len(values) < 5 for values in by_year.values()):
                raise ValueError(f"too few usable stations in q={q}, reference={reference}")
            year_means = {year: float(values.mean()) for year, values in by_year.items()}
            observed = float(np.mean(list(year_means.values())))
            boot = hierarchical_bootstrap(by_year, rng)
            result_rows.append(
                {
                    "q": q,
                    "candidate": "One-Class SVM",
                    "reference": reference,
                    "target_years": len(by_year),
                    "station_year_pairs": int(sum(len(values) for values in by_year.values())),
                    "observed_diff_in_contrast": observed,
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "prob_candidate_more_quality_aligned": float((boot > 0).mean()),
                    "bootstraps": BOOTSTRAPS,
                }
            )

    out = pd.DataFrame(result_rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "读取阶段107自然观测站点明细；先计算质量可疑报警率−质量正常报警率，再比较One-Class SVM与Trust的差异；按目标年份一级、站点二级做5000次层级bootstrap，每个年份等权、年份内站点等权；不使用攻击标签、不重新选择阈值。",
        "source": str(INPUT),
        "methods": METHODS,
        "references": REFERENCES,
        "q_grid": Q_GRID,
        "bootstraps": BOOTSTRAPS,
        "seed": SEED,
        "results": out.to_dict(orient="records"),
        "limitations": [
            "NOAA质量标记只是弱标签，质量关联不能解释为攻击召回或真实攻击检测。",
            "层级bootstrap只量化当前5个目标年份和站点集合的不确定性，不替代更多年份或独立数据源。",
            "差异为正表示One-Class SVM的质量可疑−正常报警率对比更高，不代表综合性能更优。",
        ],
    }
    out.to_csv(TABLES / "stage110_noaa_quality_hierarchical_bootstrap.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage110_noaa_quality_hierarchical_bootstrap.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 阶段110 NOAA质量弱标签年份—站点层级bootstrap",
        "",
        "先计算每个站点年份的（质量可疑报警率−质量正常报警率），再比较 One-Class SVM 与 Trust 的差异。bootstrap 以年份为一级、站点为二级，每个年份等权、年份内站点等权。",
        "",
        "| q | 候选 | 参考 | 目标年数 | 站点-年份对 | 差异（候选−参考） | 95% CI | 候选更高概率 |",
        "|---:|---|---|---:|---:|---:|---|---:|",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['q']:.3f} | {row['candidate']} | {row['reference']} | {row['target_years']} | {row['station_year_pairs']} | {row['observed_diff_in_contrast']:+.4f} | [{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}] | {row['prob_candidate_more_quality_aligned']:.4f} |"
        )
    lines += [
        "",
        "差异为正仅表示质量弱标签下的报警率对比更高；不表示攻击召回率、真实攻击检测能力或跨域普适性。",
    ]
    (REPORTS / "stage110_noaa_quality_hierarchical_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
