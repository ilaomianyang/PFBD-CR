from __future__ import annotations

"""Stage 179: decompose station-budget loss into allocation and ranking.

Stage 178 compared three fixed station-budget allocation rules under an equal
alarm budget.  This stage reuses its station-level counts only.  For each
fold, five fixed seed, method, q, and rule, it computes:

* oracle recall: the best possible recall if budget could be allocated with
  test labels;
* allocation-ceiling recall: the best recall attainable by the fixed station
  allocation before ranking rows within each station;
* actual recall: the observed recall after station-wise risk ranking.

The decomposition is diagnostic only.  It never selects an allocation rule,
model, threshold, or q from test results, and it does not retrain anything.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
STATION = (
    TABLES
    / "stage178_noaa_station_calibrated_alarm_budget_station_detail.csv"
)
DETAIL = TABLES / "stage178_noaa_station_calibrated_alarm_budget_detail.csv"
METHODS = [
    "TrustMeteo-QC frozen expanded",
    "TrustMeteo-QC default",
    "Robust max deviation",
    "Isolation Forest",
    "One-Class SVM",
]
RULES = ["size", "uniform", "risk"]
Q_GRID = [0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
METRICS = [
    "actual_recall",
    "allocation_ceiling_recall",
    "oracle_recall",
    "allocation_loss_count",
    "ranking_loss_count",
]


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


def build_detail(station: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fold",
        "seed",
        "method",
        "q",
        "allocation_rule",
        "station_id",
        "alarm_budget",
        "test_attack_rows",
        "station_budget_alarm_count",
        "station_budget_true_positive",
        "station_budget_false_positive",
    }
    missing = required - set(station.columns)
    if missing:
        raise ValueError(f"stage178 station detail missing columns: {sorted(missing)}")

    frame = station.copy()
    frame["allocation_ceiling_tp"] = frame[
        ["station_budget_alarm_count", "test_attack_rows"]
    ].min(axis=1)
    rows: list[dict[str, object]] = []
    keys = ["fold", "seed", "method", "q", "allocation_rule"]
    for key, group in frame.groupby(keys, sort=True):
        fold, seed, method, q, rule = key
        budget = int(group["alarm_budget"].iloc[0])
        attack_rows = int(group["test_attack_rows"].sum())
        alarm_rows = int(group["station_budget_alarm_count"].sum())
        true_positive = int(group["station_budget_true_positive"].sum())
        false_positive = int(group["station_budget_false_positive"].sum())
        allocation_ceiling_tp = int(group["allocation_ceiling_tp"].sum())
        oracle_tp = min(budget, attack_rows)
        if alarm_rows != budget:
            raise RuntimeError(
                f"station alarm budget mismatch for {key}: {alarm_rows} != {budget}"
            )
        if true_positive + false_positive != alarm_rows:
            raise RuntimeError(f"alarm count decomposition mismatch for {key}")
        rows.append(
            {
                "fold": fold,
                "seed": int(seed),
                "method": method,
                "q": float(q),
                "allocation_rule": rule,
                "alarm_budget": budget,
                "test_attack_rows": attack_rows,
                "station_count": int(len(group)),
                "actual_true_positive": true_positive,
                "actual_false_positive": false_positive,
                "actual_recall": (
                    float(true_positive / attack_rows) if attack_rows else None
                ),
                "allocation_ceiling_true_positive": allocation_ceiling_tp,
                "allocation_ceiling_recall": (
                    float(allocation_ceiling_tp / attack_rows)
                    if attack_rows
                    else None
                ),
                "oracle_true_positive": int(oracle_tp),
                "oracle_recall": (
                    float(oracle_tp / attack_rows) if attack_rows else None
                ),
                "allocation_loss_count": int(oracle_tp - allocation_ceiling_tp),
                "ranking_loss_count": int(allocation_ceiling_tp - true_positive),
                "allocation_loss_share": (
                    float((oracle_tp - allocation_ceiling_tp) / oracle_tp)
                    if oracle_tp
                    else None
                ),
                "ranking_loss_share": (
                    float((allocation_ceiling_tp - true_positive) / oracle_tp)
                    if oracle_tp
                    else None
                ),
                "station_allocation_gini": gini(
                    group["station_budget_alarm_count"].to_numpy(dtype=float)
                ),
                "station_attack_gini": gini(
                    group["test_attack_rows"].to_numpy(dtype=float)
                ),
            }
        )
    return pd.DataFrame(rows)


def gini(values: np.ndarray) -> float | None:
    values = np.sort(np.asarray(values, dtype=float))
    total = float(values.sum())
    if len(values) == 0 or total <= 0.0:
        return None
    n = len(values)
    return float(
        (2.0 * np.dot(np.arange(1, n + 1), values) - (n + 1) * total)
        / (n * total)
    )


def build_comparisons(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expanded = detail[
        detail["method"].eq("TrustMeteo-QC frozen expanded")
    ].copy()
    baselines = [method for method in METHODS if method != METHODS[0]]
    for baseline in baselines:
        other = detail[detail["method"].eq(baseline)].copy()
        keys = ["fold", "seed", "q", "allocation_rule"]
        common = expanded.merge(
            other,
            on=keys,
            suffixes=("_expanded", "_baseline"),
            how="inner",
            validate="one_to_one",
        )
        for q in sorted(float(x) for x in common["q"].unique()):
            q_common = common[np.isclose(common["q"].astype(float), q)]
            for rule in RULES:
                rule_common = q_common[q_common["allocation_rule"].eq(rule)]
                for metric in METRICS:
                    diff = pd.to_numeric(
                        rule_common[f"{metric}_expanded"], errors="coerce"
                    ) - pd.to_numeric(
                        rule_common[f"{metric}_baseline"], errors="coerce"
                    )
                    fold_values = (
                        diff.groupby(rule_common["fold"], sort=True)
                        .mean()
                        .dropna()
                        .to_numpy(dtype=float)
                    )
                    if len(fold_values) < 2:
                        continue
                    p_value, mean_diff = exact_sign_flip(fold_values)
                    rows.append(
                        {
                            "comparison": f"expanded - {baseline}",
                            "allocation_rule": rule,
                            "metric": metric,
                            "q": q,
                            "fold_count": int(len(fold_values)),
                            "fold_seed_cells": int(diff.notna().sum()),
                            "mean_fold_diff": mean_diff,
                            "p_value_exact_sign_flip": p_value,
                            "fold_differences": fold_values.tolist(),
                        }
                    )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["bh_q_value"] = bh_adjust(
            [float(x) for x in out["p_value_exact_sign_flip"]]
        )
    return out


def main() -> int:
    station = pd.read_csv(STATION)
    detail = build_detail(station)
    expected_detail_rows = (
        len(METHODS) * len(Q_GRID) * len(RULES) * 4 * 5
    )
    if len(detail) != expected_detail_rows:
        raise RuntimeError(
            f"detail rows mismatch: {len(detail)} != {expected_detail_rows}"
        )
    comparisons = build_comparisons(detail)
    expected_comparisons = (
        (len(METHODS) - 1) * len(Q_GRID) * len(RULES) * len(METRICS)
    )
    if len(comparisons) != expected_comparisons:
        raise RuntimeError(
            f"comparison rows mismatch: "
            f"{len(comparisons)} != {expected_comparisons}"
        )
    if detail.duplicated(
        ["fold", "seed", "method", "q", "allocation_rule"]
    ).any():
        raise RuntimeError("duplicate stage179 keys")
    for column in [
        "actual_recall",
        "allocation_ceiling_recall",
        "oracle_recall",
    ]:
        valid = detail[column].dropna()
        if not valid.between(0, 1).all():
            raise RuntimeError(f"{column} outside [0,1]")
    if not (
        detail["oracle_true_positive"]
        >= detail["allocation_ceiling_true_positive"]
    ).all():
        raise RuntimeError("allocation ceiling exceeds oracle")
    if not (
        detail["allocation_ceiling_true_positive"]
        >= detail["actual_true_positive"]
    ).all():
        raise RuntimeError("actual TP exceeds allocation ceiling")

    prefix = "stage179_noaa_station_budget_loss_decomposition"
    detail.to_csv(
        TABLES / f"{prefix}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        TABLES / f"{prefix}_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    expanded = detail[
        detail["method"].eq("TrustMeteo-QC frozen expanded")
        & np.isclose(detail["q"].astype(float), 0.995)
    ]
    rule_summary: dict[str, dict[str, float]] = {}
    for rule in RULES:
        group = expanded[expanded["allocation_rule"].eq(rule)]
        if len(group) != 20:
            raise RuntimeError(f"expanded q=0.995 rows mismatch for {rule}")
        rule_summary[rule] = {
            "actual_recall_mean": float(group["actual_recall"].mean()),
            "allocation_ceiling_recall_mean": float(
                group["allocation_ceiling_recall"].mean()
            ),
            "oracle_recall_mean": float(group["oracle_recall"].mean()),
            "allocation_loss_count_mean": float(
                group["allocation_loss_count"].mean()
            ),
            "ranking_loss_count_mean": float(
                group["ranking_loss_count"].mean()
            ),
        }

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段178的逐站点报警计数和五种子结果，不重新训练、不重新计算风险分数、"
            "不重新选择模型、阈值、q或分配规则。对每个fold×种子×方法×q×"
            "分配规则，将固定预算下的Recall损失分解为：oracle上限、固定站点"
            "分配上限、站内风险排序后的实际值。分配损失只表示预算在站点间"
            "错配，排序损失只表示站内top-k未命中攻击；oracle使用test标签，"
            "仅作为事后不可部署上界，不能用于选择。NOAA攻击是透明受控注入，"
            "不是真实攻击日志。"
        ),
        "source_artifacts": [STATION.name],
        "methods": METHODS,
        "q_grid": Q_GRID,
        "allocation_rules": RULES,
        "detail_rows": int(len(detail)),
        "comparison_rows": int(len(comparisons)),
        "expanded_q995_by_allocation_rule": rule_summary,
        "bh_q_below_05": (
            int((comparisons["bh_q_value"] < 0.05).sum())
            if not comparisons.empty
            else 0
        ),
        "min_bh_q_value": (
            float(comparisons["bh_q_value"].min())
            if not comparisons.empty
            else None
        ),
        "detail": clean_records(detail),
        "comparisons": clean_records(comparisons),
        "limitations": [
            "oracle上界使用test攻击标签，仅用于损失分解，不可部署、不可选规则。",
            "固定站点分配上限仍依赖test每站攻击数，只是诊断上界。",
            "四折精确检验功效有限；BH校正后无显著差异不等于方法等效。",
            "NOAA攻击是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段179 NOAA站点预算损失分解",
        "",
        "复用阶段178站点计数；将固定预算下的召回损失分解为站点分配损失和站内排序损失，不重新训练或调参。",
        "",
        "| 分配规则 | 实际Recall | 分配上限Recall | Oracle Recall | 分配损失TP | 排序损失TP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rule in RULES:
        row = rule_summary[rule]
        lines.append(
            f"| {rule} | {row['actual_recall_mean']:.6f} | "
            f"{row['allocation_ceiling_recall_mean']:.6f} | "
            f"{row['oracle_recall_mean']:.6f} | "
            f"{row['allocation_loss_count_mean']:.4f} | "
            f"{row['ranking_loss_count_mean']:.4f} |"
        )
    lines += [
        "",
        f"明细={len(detail)}；比较={len(comparisons)}；"
        f"BH q<0.05={payload['bh_q_below_05']}；"
        f"最小BH q={payload['min_bh_q_value']:.6f}。",
        "Oracle上界使用test标签，仅作事后诊断，不能用于部署或选择规则。",
        "NOAA攻击是透明受控注入，不是真实攻击日志。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": len(detail),
                "comparison_rows": len(comparisons),
                "bh_q_below_05": payload["bh_q_below_05"],
                "min_bh_q_value": payload["min_bh_q_value"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
