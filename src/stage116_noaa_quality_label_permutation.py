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
PERMUTATIONS = 10000
SEED = 20260914


def _recover_count(rate: float, rows: int) -> int:
    count = int(round(float(rate) * int(rows)))
    if rows <= 0 or count < 0 or count > rows:
        raise ValueError(f"invalid recovered count: rate={rate}, rows={rows}, count={count}")
    if not np.isclose(count / rows, rate, atol=1e-10, rtol=0):
        raise ValueError(f"alarm rate does not map to an integer count: rate={rate}, rows={rows}")
    return count


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def _permuted_contrasts(group: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    draws = np.zeros(PERMUTATIONS, dtype=float)
    for row in group.itertuples(index=False):
        total_rows = int(row.quality_flagged_rows + row.quality_clean_rows)
        total_alarms = int(row.quality_flagged_alarm_count + row.quality_clean_alarm_count)
        flagged_alarms = rng.hypergeometric(
            ngood=total_alarms,
            nbad=total_rows - total_alarms,
            nsample=int(row.quality_flagged_rows),
            size=PERMUTATIONS,
        )
        clean_alarms = total_alarms - flagged_alarms
        draws += flagged_alarms / row.quality_flagged_rows - clean_alarms / row.quality_clean_rows
    return draws / len(group)


def _result_row(group: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    observed = float(group["quality_contrast"].mean())
    null = _permuted_contrasts(group, rng)
    return {
        "stations_or_pairs": int(len(group)),
        "quality_flagged_rows": int(group["quality_flagged_rows"].sum()),
        "quality_clean_rows": int(group["quality_clean_rows"].sum()),
        "observed_contrast": observed,
        "null_mean": float(null.mean()),
        "null_ci95_low": float(np.quantile(null, 0.025)),
        "null_ci95_high": float(np.quantile(null, 0.975)),
        "permutation_p_one_sided": float((1 + np.count_nonzero(null >= observed)) / (PERMUTATIONS + 1)),
        "permutation_p_two_sided": float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (PERMUTATIONS + 1)),
        "permutations": PERMUTATIONS,
    }


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
    usable["quality_flagged_alarm_count"] = [
        _recover_count(rate, rows)
        for rate, rows in zip(usable["quality_flagged_alarm_rate"], usable["quality_flagged_rows"])
    ]
    usable["quality_clean_alarm_count"] = [
        _recover_count(rate, rows)
        for rate, rows in zip(usable["quality_clean_alarm_rate"], usable["quality_clean_rows"])
    ]
    usable["quality_contrast"] = usable["quality_flagged_alarm_rate"] - usable["quality_clean_alarm_rate"]

    rng = np.random.default_rng(SEED)
    yearly_rows: list[dict[str, object]] = []
    for target_year in sorted(usable["target_year"].unique()):
        for q in Q_GRID:
            for method in METHODS:
                group = usable.loc[
                    (usable["target_year"] == target_year)
                    & np.isclose(usable["q"], q)
                    & (usable["method"] == method)
                ].copy()
                if len(group) < 5:
                    raise ValueError(f"too few usable stations: year={target_year}, q={q}, method={method}")
                yearly_rows.append({"target_year": int(target_year), "q": q, "method": method, **_result_row(group, rng)})

    yearly = pd.DataFrame(yearly_rows)
    yearly["bh_q_one_sided_45"] = _bh_adjust(yearly["permutation_p_one_sided"].to_numpy(dtype=float))
    yearly["bh_q_two_sided_45"] = _bh_adjust(yearly["permutation_p_two_sided"].to_numpy(dtype=float))

    aggregate_rows: list[dict[str, object]] = []
    for q in Q_GRID:
        for method in METHODS:
            group = usable.loc[np.isclose(usable["q"], q) & (usable["method"] == method)].copy()
            years = sorted(group["target_year"].unique())
            if len(years) != 5 or any((group["target_year"] == year).sum() < 5 for year in years):
                raise ValueError(f"incomplete aggregate coverage: q={q}, method={method}")
            observed_year_means = group.groupby("target_year")["quality_contrast"].mean()
            null_by_year = [
                _permuted_contrasts(group.loc[group["target_year"] == year], rng)
                for year in years
            ]
            null = np.vstack(null_by_year).mean(axis=0)
            observed = float(observed_year_means.mean())
            aggregate_rows.append(
                {
                    "q": q,
                    "method": method,
                    "target_years": len(years),
                    "station_year_pairs": int(len(group)),
                    "quality_flagged_rows": int(group["quality_flagged_rows"].sum()),
                    "quality_clean_rows": int(group["quality_clean_rows"].sum()),
                    "observed_year_equal_contrast": observed,
                    "null_mean": float(null.mean()),
                    "null_ci95_low": float(np.quantile(null, 0.025)),
                    "null_ci95_high": float(np.quantile(null, 0.975)),
                    "permutation_p_one_sided": float((1 + np.count_nonzero(null >= observed)) / (PERMUTATIONS + 1)),
                    "permutation_p_two_sided": float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (PERMUTATIONS + 1)),
                    "permutations": PERMUTATIONS,
                }
            )

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["bh_q_one_sided_9"] = _bh_adjust(aggregate["permutation_p_one_sided"].to_numpy(dtype=float))
    aggregate["bh_q_two_sided_9"] = _bh_adjust(aggregate["permutation_p_two_sided"].to_numpy(dtype=float))
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "读取阶段107的NOAA自然观测站点明细；在每个站点×目标年×方法×q内固定报警总数、质量可疑行数和质量正常行数，仅随机置换质量弱标签；用10000次条件随机化检验质量可疑报警率−质量正常报警率，并对45个逐年检验和9个跨年检验分别做Benjamini-Hochberg校正。不注入攻击、不把质量字段当攻击真值。",
        "source": str(INPUT),
        "methods": METHODS,
        "q_grid": Q_GRID,
        "permutations": PERMUTATIONS,
        "seed": SEED,
        "yearly_results": yearly.to_dict(orient="records"),
        "aggregate_results": aggregate.to_dict(orient="records"),
        "limitations": [
            "NOAA质量字段只是弱标签；显著关联不能解释为攻击检测、攻击召回或因果关系。",
            "条件随机化只排除站点年份内随机标签分配这一零假设，不能消除未观测的设备、维护或气候混杂。",
            "检验固定既有报警输出，不根据置换结果重新选择模型、q或阈值。",
            "逐年和跨年多重比较分别校正，不能只挑选未校正的显著结果。",
        ],
    }
    yearly.to_csv(TABLES / "stage116_noaa_quality_label_permutation_yearly.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage116_noaa_quality_label_permutation_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage116_noaa_quality_label_permutation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 阶段116 NOAA质量弱标签条件随机化检验",
        "",
        "在每个站点和年份内部固定报警数及质量可疑/正常行数，只随机置换质量弱标签。该检验用于判断观察到的报警率差异是否超出随机标签分配可解释的范围。",
        "",
        "| q | 方法 | 年份 | 站点-年份对 | 观察差异 | 零分布95%区间 | 单侧p | 单侧BH q |",
        "|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['q']:.3f} | {row['method']} | {row['target_years']} | {row['station_year_pairs']} | {row['observed_year_equal_contrast']:+.4f} | [{row['null_ci95_low']:+.4f},{row['null_ci95_high']:+.4f}] | {row['permutation_p_one_sided']:.5f} | {row['bh_q_one_sided_9']:.5f} |"
        )
    lines += [
        "",
        "质量字段仅作弱标签。即使校正后显著，也只能说明自然观测中的统计关联，不能称为真实攻击检测或攻击召回。",
    ]
    (REPORTS / "stage116_noaa_quality_label_permutation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "yearly_tests": len(yearly),
                "aggregate_tests": len(aggregate),
                "aggregate_bh_significant": int((aggregate["bh_q_one_sided_9"] < 0.05).sum()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
