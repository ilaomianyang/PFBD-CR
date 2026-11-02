from __future__ import annotations

"""Stage 175: fold-seed stability of low-FAR alarm concentration.

This stage reuses the stage-174 station/month alarm accounting. It does not
retrain a model, recompute a score, or change any threshold.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
STATION = TABLES / "stage174_noaa_low_far_alarm_burden_audit_station_detail.csv"
MONTH = TABLES / "stage174_noaa_low_far_alarm_burden_audit_month_detail.csv"
BASELINES = [
    "TrustMeteo-QC default",
    "Robust max deviation",
    "Isolation Forest",
    "One-Class SVM",
]
ENTITY_COLUMNS = {"station": "station_id", "month": "month"}


def share(counts: pd.Series, fraction: float) -> float | None:
    total = float(counts.sum())
    if total <= 0.0:
        return None
    ordered = np.sort(counts.to_numpy(dtype=float))[::-1]
    count = max(1, int(np.ceil(len(ordered) * fraction)))
    return float(ordered[:count].sum() / total)


def gini(shares: np.ndarray) -> float | None:
    shares = np.sort(np.asarray(shares, dtype=float))
    n = len(shares)
    if n == 0:
        return None
    return float((2.0 * np.dot(np.arange(1, n + 1), shares) - (n + 1)) / n)


def concentration(frame: pd.DataFrame, entity: str) -> dict[str, object]:
    column = ENTITY_COLUMNS[entity]
    counts = frame.groupby(column, sort=False)["alarm_count"].sum()
    total = float(counts.sum())
    if total <= 0.0:
        return {
            f"{entity}_count": int(len(counts)),
            f"{entity}_alarm_share_top10pct": None,
            f"{entity}_alarm_share_max": None,
            f"{entity}_alarm_gini": None,
        }
    shares = counts / total
    return {
        f"{entity}_count": int(len(counts)),
        f"{entity}_alarm_share_top10pct": share(counts, 0.10),
        f"{entity}_alarm_share_max": float(shares.max()),
        f"{entity}_alarm_gini": gini(shares.to_numpy(dtype=float)),
    }


def exact_sign_flip(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = [
        abs(float((values * np.asarray(signs)).mean()))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15)), float(values.mean())


def bh_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    order = np.argsort(np.asarray(values, dtype=float))
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        idx = int(order[rank])
        running = min(running, float(values[idx]) * len(order) / (rank + 1))
        adjusted[idx] = running
    return adjusted.tolist()


def clean_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        cleaned: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, float) and not np.isfinite(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def main() -> int:
    station = pd.read_csv(STATION)
    month = pd.read_csv(MONTH)
    required_station = {"fold", "seed", "method", "q", "station_id", "alarm_count"}
    required_month = {"fold", "seed", "method", "q", "month", "alarm_count"}
    if not required_station.issubset(station.columns):
        raise ValueError(f"station detail missing columns: {sorted(required_station - set(station.columns))}")
    if not required_month.issubset(month.columns):
        raise ValueError(f"month detail missing columns: {sorted(required_month - set(month.columns))}")

    rows: list[dict[str, object]] = []
    for entity, frame in (("station", station), ("month", month)):
        for key, group in frame.groupby(["fold", "seed", "method", "q"], sort=True):
            fold, seed, method, q = key
            row = {
                "entity": entity,
                "fold": fold,
                "seed": int(seed),
                "method": method,
                "q": float(q),
                "alarm_count": int(group["alarm_count"].sum()),
            }
            row.update(concentration(group, entity))
            rows.append(row)
    detail = pd.DataFrame(rows)

    comparisons: list[dict[str, object]] = []
    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    for baseline in BASELINES:
        other = detail[detail["method"].eq(baseline)]
        keys = ["entity", "fold", "seed", "q"]
        common = (
            expanded.set_index(keys)
            .join(other.set_index(keys), lsuffix="_expanded", rsuffix="_baseline", how="inner")
            .reset_index()
        )
        for entity in ("station", "month"):
            entity_common = common[common["entity"].eq(entity)]
            for q in sorted(float(x) for x in entity_common["q"].unique()):
                q_common = entity_common[np.isclose(entity_common["q"].astype(float), q)]
                for metric in ("alarm_share_top10pct", "alarm_share_max", "alarm_gini"):
                    exp_col = f"{entity}_{metric}_expanded"
                    base_col = f"{entity}_{metric}_baseline"
                    diff = pd.to_numeric(q_common[exp_col], errors="coerce") - pd.to_numeric(
                        q_common[base_col], errors="coerce"
                    )
                    fold_values = diff.groupby(q_common["fold"], sort=True).mean().dropna().to_numpy(dtype=float)
                    if len(fold_values) < 2:
                        continue
                    p_value, mean_diff = exact_sign_flip(fold_values)
                    comparisons.append(
                        {
                            "comparison": f"expanded - {baseline}",
                            "entity": entity,
                            "metric": metric,
                            "q": q,
                            "fold_count": int(len(fold_values)),
                            "fold_seed_cells": int(diff.notna().sum()),
                            "mean_fold_diff": mean_diff,
                            "p_value_exact_sign_flip": p_value,
                            "fold_differences": fold_values.tolist(),
                        }
                    )
    comparison_frame = pd.DataFrame(comparisons)
    if not comparison_frame.empty:
        adjusted = bh_adjust(
            [float(x) for x in comparison_frame["p_value_exact_sign_flip"].tolist()]
        )
        comparison_frame["bh_q_value"] = adjusted

    prefix = "stage175_noaa_low_far_alarm_burden_stability"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    comparison_frame.to_csv(TABLES / f"{prefix}_comparisons.csv", index=False, encoding="utf-8-sig")

    expanded_q995 = detail[
        detail["method"].eq("TrustMeteo-QC frozen expanded")
        & detail["q"].eq(0.995)
        & detail["entity"].eq("station")
    ]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段174的train/valid/test逐站点/逐月报警计数；按fold和种子分别计算前10%报警份额、"
            "最大单实体份额与Gini。扩展方案相对四个基线按fold汇总五种子的差，"
            "执行精确sign-flip并统一BH校正。不重新训练、不重新选择阈值或q、"
            "不根据test结果调参；受控攻击不是真实攻击日志。"
        ),
        "source_artifacts": [STATION.name, MONTH.name],
        "entities": ["station", "month"],
        "methods": ["TrustMeteo-QC frozen expanded", *BASELINES],
        "q_grid": sorted(float(x) for x in detail["q"].unique()),
        "detail_rows": int(len(detail)),
        "comparison_rows": int(len(comparison_frame)),
        "expanded_q995_station_cells": int(len(expanded_q995)),
        "expanded_q995_station_top10pct_mean": (
            float(expanded_q995["station_alarm_share_top10pct"].mean())
            if not expanded_q995.empty
            else None
        ),
        "expanded_q995_station_top10pct_std": (
            float(expanded_q995["station_alarm_share_top10pct"].std(ddof=1))
            if len(expanded_q995) > 1
            else None
        ),
        "detail": clean_records(detail),
        "comparisons": clean_records(comparison_frame),
        "bh_q_below_05": (
            int((comparison_frame["bh_q_value"] < 0.05).sum())
            if not comparison_frame.empty
            else 0
        ),
        "min_bh_q_value": (
            float(comparison_frame["bh_q_value"].min())
            if not comparison_frame.empty
            else None
        ),
        "limitations": [
            "集中度差异只描述报警运行负担，不是检测召回或攻击真值。",
            "四个滚动fold的精确检验功效有限；BH校正后无显著差异不等于等效。",
            "不可估计集中度保留为空；NOAA攻击为透明受控注入。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    lines = [
        "# 阶段175 NOAA低误报报警负担稳定性",
        "",
        "复用阶段174逐站点/逐月计数，按fold×种子计算报警集中度；不重新训练或调阈值。",
        "",
        "| 实体 | 比较 | q | 指标 | fold数 | 平均fold差 | 精确p | BH q |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in comparison_frame.to_dict(orient="records"):
        lines.append(
            f"| {row['entity']} | {row['comparison']} | {row['q']:.4f} | {row['metric']} | "
            f"{row['fold_count']} | {row['mean_fold_diff']:+.6f} | "
            f"{row['p_value_exact_sign_flip']:.6f} | {row['bh_q_value']:.6f} |"
        )
    lines += [
        "",
        f"明细={len(detail)}；成对比较={len(comparison_frame)}；"
        f"扩展q=0.995站点前10%份额均值={payload['expanded_q995_station_top10pct_mean']:.6f}。",
        "集中度是运行负担诊断，不能解释为攻击召回或真实攻击率。",
    ]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": len(detail),
                "comparison_rows": len(comparison_frame),
                "bh_q_below_05": payload["bh_q_below_05"],
                "min_bh_q_value": payload["min_bh_q_value"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
