from __future__ import annotations

"""Stage 176: station-equal-weight metrics for the frozen low-FAR grid.

Stage 174 records station-level TP/FP/FN/TN. This stage compares row-pooled
metrics with station-equal-weight metrics and quantifies false-positive
over-allocation relative to each station's share of normal rows.
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
BASELINES = [
    "TrustMeteo-QC default",
    "Robust max deviation",
    "Isolation Forest",
    "One-Class SVM",
]
EXPANDED = "TrustMeteo-QC frozen expanded"
METRICS = ["pooled_f1", "macro_f1", "macro_recall", "macro_precision", "macro_far"]


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def f1_from_counts(tp: float, fp: float, fn: float) -> float | None:
    denominator = 2.0 * tp + fp + fn
    if denominator <= 0.0:
        return None
    return float(2.0 * tp / denominator)


def mean_or_none(values: list[float | None]) -> float | None:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return float(finite.mean())


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
    required = {
        "fold",
        "seed",
        "method",
        "q",
        "station_id",
        "test_rows",
        "test_attack_rows",
        "test_normal_rows",
        "alarm_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    }
    missing = required - set(station.columns)
    if missing:
        raise ValueError(f"stage174 station detail missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for key, group in station.groupby(["fold", "seed", "method", "q"], sort=True):
        fold, seed, method, q = key
        tp = float(group["true_positive"].sum())
        fp = float(group["false_positive"].sum())
        fn = float(group["false_negative"].sum())
        tn = float(group["true_negative"].sum())
        normal_rows = float(group["test_normal_rows"].sum())
        attack_rows = float(group["test_attack_rows"].sum())

        recall_values = group.apply(
            lambda x: ratio(float(x["true_positive"]), float(x["test_attack_rows"])),
            axis=1,
        )
        precision_values = group.apply(
            lambda x: ratio(float(x["true_positive"]), float(x["alarm_count"])),
            axis=1,
        )
        far_values = group.apply(
            lambda x: ratio(float(x["false_positive"]), float(x["test_normal_rows"])),
            axis=1,
        )
        f1_values = [
            f1_from_counts(float(x["true_positive"]), float(x["false_positive"]), float(x["false_negative"]))
            for _, x in group.iterrows()
        ]
        expected_fp = normal_rows / len(group)
        over_allocation = [
            ratio(float(x["false_positive"]), expected_fp)
            for _, x in group.iterrows()
        ]
        row = {
            "fold": fold,
            "seed": int(seed),
            "method": method,
            "q": float(q),
            "station_count": int(len(group)),
            "attack_rows": int(attack_rows),
            "normal_rows": int(normal_rows),
            "alarm_count": int(group["alarm_count"].sum()),
            "false_positive_count": int(fp),
            "pooled_f1": f1_from_counts(tp, fp, fn),
            "pooled_recall": ratio(tp, attack_rows),
            "pooled_precision": ratio(tp, tp + fp),
            "pooled_far": ratio(fp, normal_rows),
            "macro_f1": mean_or_none(f1_values),
            "macro_recall": mean_or_none(recall_values.tolist()),
            "macro_precision": mean_or_none(precision_values.tolist()),
            "macro_far": mean_or_none(far_values.tolist()),
            "fp_over_allocation_max": float(np.nanmax(np.asarray(over_allocation, dtype=float))),
            "fp_over_allocation_top1_share": float(
                np.nanmax(group["false_positive"].to_numpy(dtype=float))
                / max(float(group["false_positive"].sum()), 1.0)
            ),
            "stations_with_zero_alarm": int((group["alarm_count"] == 0).sum()),
            "stations_with_zero_fp": int((group["false_positive"] == 0).sum()),
        }
        for metric in METRICS:
            if not metric.startswith("macro_"):
                continue
            pooled_value = row[metric.replace("macro_", "pooled_")]
            macro_value = row[metric]
            row[f"pooled_minus_macro_{metric}"] = (
                float(pooled_value - macro_value)
                if pooled_value is not None and macro_value is not None
                else None
            )
        rows.append(row)

        normal_share = group["test_normal_rows"].astype(float) / normal_rows
        fp_share = group["false_positive"].astype(float) / max(fp, 1.0)
        for _, x in group.assign(normal_share=normal_share.to_numpy(), fp_share=fp_share.to_numpy()).iterrows():
            station_rows.append(
                {
                    "fold": fold,
                    "seed": int(seed),
                    "method": method,
                    "q": float(q),
                    "station_id": str(x["station_id"]),
                    "attack_rows": int(x["test_attack_rows"]),
                    "normal_rows": int(x["test_normal_rows"]),
                    "alarm_count": int(x["alarm_count"]),
                    "false_positive": int(x["false_positive"]),
                    "recall": ratio(float(x["true_positive"]), float(x["test_attack_rows"])),
                    "precision": ratio(float(x["true_positive"]), float(x["alarm_count"])),
                    "far": ratio(float(x["false_positive"]), float(x["test_normal_rows"])),
                    "normal_share": float(x["normal_share"]),
                    "fp_share": float(x["fp_share"]),
                    "fp_over_allocation": ratio(float(x["false_positive"]), expected_fp),
                }
            )

    detail = pd.DataFrame(rows)
    station_detail = pd.DataFrame(station_rows)
    comparisons: list[dict[str, object]] = []
    expanded = detail[detail["method"].eq(EXPANDED)]
    for baseline in BASELINES:
        other = detail[detail["method"].eq(baseline)]
        common = expanded.merge(
            other,
            on=["fold", "seed", "q"],
            suffixes=("_expanded", "_baseline"),
            how="inner",
        )
        for q in sorted(float(x) for x in common["q"].unique()):
            q_common = common[np.isclose(common["q"].astype(float), q)]
            for metric in METRICS + [
                "fp_over_allocation_max",
                "fp_over_allocation_top1_share",
                "stations_with_zero_alarm",
            ]:
                diff = pd.to_numeric(
                    q_common[f"{metric}_expanded"], errors="coerce"
                ) - pd.to_numeric(q_common[f"{metric}_baseline"], errors="coerce")
                fold_values = (
                    diff.groupby(q_common["fold"], sort=True)
                    .mean()
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if len(fold_values) < 2:
                    continue
                p_value, mean_diff = exact_sign_flip(fold_values)
                comparisons.append(
                    {
                        "comparison": f"expanded - {baseline}",
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
        comparison_frame["bh_q_value"] = bh_adjust(
            [float(x) for x in comparison_frame["p_value_exact_sign_flip"]]
        )

    prefix = "stage176_noaa_low_far_station_equal_weight_metrics"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    station_detail.to_csv(
        TABLES / f"{prefix}_station_detail.csv", index=False, encoding="utf-8-sig"
    )
    comparison_frame.to_csv(
        TABLES / f"{prefix}_comparisons.csv", index=False, encoding="utf-8-sig"
    )

    expanded_995 = detail[detail["method"].eq(EXPANDED) & detail["q"].eq(0.995)]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段174的train/valid/test站点级TP/FP/FN/TN；每个折×种子单元同时计算"
            "行汇总pooled指标和站点等权macro指标，并计算各站点误报数相对等额正常行份额的"
            "超配比。扩展方案相对四个基线按fold汇总并执行精确sign-flip与BH校正。"
            "不重新训练、不重新选择阈值或q；不可估计指标保留为空。"
            "NOAA攻击是透明受控注入，不是真实攻击日志。"
        ),
        "source_artifact": STATION.name,
        "methods": [EXPANDED, *BASELINES],
        "q_grid": sorted(float(x) for x in detail["q"].unique()),
        "metrics": METRICS,
        "detail_rows": int(len(detail)),
        "station_detail_rows": int(len(station_detail)),
        "comparison_rows": int(len(comparison_frame)),
        "expanded_q995_pooled_f1_mean": float(expanded_995["pooled_f1"].mean()),
        "expanded_q995_macro_f1_mean": float(expanded_995["macro_f1"].mean()),
        "expanded_q995_pooled_minus_macro_f1": float(
            expanded_995["pooled_minus_macro_macro_f1"].mean()
        ),
        "detail": clean_records(detail),
        "station_detail": clean_records(station_detail),
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
            "站点等权macro指标只改变站点权重，不创造新的独立样本，也不替代行加权总体指标。",
            "误报超配比是运行负担诊断，不是攻击召回或真实入侵率。",
            "四个滚动fold的功效有限；BH校正后无显著差异不等于方法等效。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    lines = [
        "# 阶段176 NOAA低误报站点等权指标与误报超配审计",
        "",
        "复用阶段174站点级TP/FP/FN/TN；同时报告行汇总与站点等权指标，不重新训练或调阈值。",
        "",
        "| 方法 | q | pooled F1 | macro F1 | macro Recall | macro Precision | macro FAR | 最大误报超配 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary = (
        detail.groupby(["method", "q"], sort=True)[
            ["pooled_f1", "macro_f1", "macro_recall", "macro_precision", "macro_far", "fp_over_allocation_max"]
        ]
        .mean()
        .reset_index()
    )
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {row['q']:.4f} | {row['pooled_f1']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['macro_recall']:.6f} | "
            f"{row['macro_precision']:.6f} | {row['macro_far']:.6f} | "
            f"{row['fp_over_allocation_max']:.4f} |"
        )
    lines += [
        "",
        f"明细={len(detail)}；站点明细={len(station_detail)}；成对比较={len(comparison_frame)}；"
        f"BH q<0.05={payload['bh_q_below_05']}；最小BH q={payload['min_bh_q_value']:.4f}。",
        "站点等权结果只用于暴露行加权全局指标被站点样本量支配的风险，不能包装成另一种普遍优势。",
    ]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": len(detail),
                "station_detail_rows": len(station_detail),
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
