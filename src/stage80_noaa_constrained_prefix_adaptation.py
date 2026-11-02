from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage70_noaa_isd_multiyear_normalized.parquet"
AUDIT = TABLES / "stage71_noaa_multiyear_quality_audit.json"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
PREFIX_DAYS = [7, 30, 90]
QUANTILES = [0.95, 0.975, 0.99, 0.995, 0.999]
SHRINKAGES = [0.25, 0.5, 0.75, 1.0]
OPERATING_POINTS = ["max_f1", "recall_ge_0p5", "far_le_0p10"]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage73_noaa_multiyear_controlled_benchmark import build_dataset  # noqa: E402
from stage79_noaa_prefix_adaptation import prefix_mask  # noqa: E402


def blended_predictions(scored: pd.DataFrame, year: int, prefix_days: int, quantile: float, shrinkage: float, fixed_threshold: float) -> tuple[np.ndarray, dict[str, float]]:
    in_year = scored["year"].eq(year).to_numpy()
    prefix = prefix_mask(scored, year, prefix_days)
    risk = scored["robust_fused_risk"].to_numpy(dtype=float)
    pred = np.zeros(len(scored), dtype=bool)
    thresholds: dict[str, float] = {}
    for station, group in scored.loc[in_year].groupby("station_key", sort=True):
        idx = group.index.to_numpy()
        ref_idx = idx[prefix[idx] & np.isfinite(risk[idx])]
        local = float(np.quantile(risk[ref_idx], quantile)) if len(ref_idx) >= 20 else fixed_threshold
        threshold = float((1.0 - shrinkage) * fixed_threshold + shrinkage * local)
        thresholds[str(station)] = threshold
        eligible = idx[np.isfinite(risk[idx]) & ~prefix[idx]]
        pred[eligible] = risk[eligible] >= threshold
    return pred, thresholds


def evaluate(scored: pd.DataFrame, pred: np.ndarray, year: int, prefix_days: int) -> dict[str, float | int]:
    prefix = prefix_mask(scored, year, prefix_days)
    region = scored["year"].eq(year).to_numpy() & ~prefix
    y = scored.loc[region, "combined_is_attack"].astype(bool).to_numpy()
    p = pred[region]
    s = scored.loc[region, "robust_fused_risk"].to_numpy(dtype=float)
    m = _metrics(y, p, s)
    prefix_region = scored["year"].eq(year).to_numpy() & prefix
    return {"f1": float(m["f1"]), "recall": float(m["recall"]), "precision": float(m["precision"]), "auprc": float(m["average_precision"]), "far": float(p[~y].mean()) if (~y).any() else float("nan"), "rows": int(region.sum()), "attack_rows": int(y.sum()), "prefix_rows": int(prefix_region.sum()), "prefix_attack_rows": int(scored.loc[prefix_region, "combined_is_attack"].astype(bool).sum())}


def select_point(candidates: list[dict[str, object]], point: str) -> dict[str, object] | None:
    if point == "max_f1":
        pool = candidates
        return max(pool, key=lambda x: (x["valid_f1"], -x["valid_far"], x["valid_recall"]))
    if point == "recall_ge_0p5":
        pool = [x for x in candidates if float(x["valid_recall"]) >= 0.5]
        return min(pool, key=lambda x: (x["valid_far"], -x["valid_f1"], -x["valid_recall"])) if pool else None
    if point == "far_le_0p10":
        pool = [x for x in candidates if float(x["valid_far"]) <= 0.10]
        return max(pool, key=lambda x: (x["valid_f1"], x["valid_recall"], -x["valid_far"])) if pool else None
    raise ValueError(f"unknown operating point: {point}")


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_multiyear_validation")}
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in [2021, 2022, 2023])})
    source = source[source["station_id"].astype(str).isin(stations) & source["year"].isin([2021, 2022, 2023])].copy()
    detail_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        frame = build_dataset(source, seed, set(stations))
        scored, fixed = score(frame)
        scored["year"] = pd.to_numeric(scored["year"], errors="coerce").astype(int)
        scored["station_key"] = scored["station_key"].astype(str)
        fixed_threshold = float(fixed["setting"]["threshold"])
        candidates: list[dict[str, object]] = []
        for days in PREFIX_DAYS:
            for q in QUANTILES:
                for shrinkage in SHRINKAGES:
                    pred, thresholds = blended_predictions(scored, 2022, days, q, shrinkage, fixed_threshold)
                    valid = evaluate(scored, pred, 2022, days)
                    row = {"seed": seed, "prefix_days": days, "quantile": q, "shrinkage": shrinkage, "valid_f1": valid["f1"], "valid_recall": valid["recall"], "valid_precision": valid["precision"], "valid_auprc": valid["auprc"], "valid_far": valid["far"], "valid_rows": valid["rows"], "valid_attack_rows": valid["attack_rows"], "valid_prefix_rows": valid["prefix_rows"], "valid_prefix_attack_rows": valid["prefix_attack_rows"], "threshold_min": float(min(thresholds.values())), "threshold_median": float(np.median(list(thresholds.values()))), "threshold_max": float(max(thresholds.values())), "fixed_threshold": fixed_threshold}
                    candidates.append(row)
                    detail_rows.append(row)
        for point in OPERATING_POINTS:
            chosen = select_point(candidates, point)
            if chosen is None:
                selected_rows.append({"seed": seed, "operating_point": point, "status": "NO_VALID_CANDIDATE"})
                continue
            days = int(chosen["prefix_days"])
            q = float(chosen["quantile"])
            shrinkage = float(chosen["shrinkage"])
            test_pred, test_thresholds = blended_predictions(scored, 2023, days, q, shrinkage, fixed_threshold)
            test = evaluate(scored, test_pred, 2023, days)
            fixed_pred = scored["robust_fused_risk"].to_numpy(dtype=float) >= fixed_threshold
            fixed_test = evaluate(scored, fixed_pred, 2023, days)
            selected_rows.append({**chosen, "operating_point": point, "status": "EVALUATED", "test_f1": test["f1"], "test_recall": test["recall"], "test_precision": test["precision"], "test_auprc": test["auprc"], "test_far": test["far"], "test_rows": test["rows"], "test_attack_rows": test["attack_rows"], "test_prefix_rows": test["prefix_rows"], "test_prefix_attack_rows": test["prefix_attack_rows"], "test_threshold_min": float(min(test_thresholds.values())), "test_threshold_median": float(np.median(list(test_thresholds.values()))), "test_threshold_max": float(max(test_thresholds.values())), "fixed_test_f1": fixed_test["f1"], "fixed_test_recall": fixed_test["recall"], "fixed_test_precision": fixed_test["precision"], "fixed_test_auprc": fixed_test["auprc"], "fixed_test_far": fixed_test["far"]})
        print(json.dumps({"seed": seed, "candidates": len(candidates), "points": {row["operating_point"]: {"valid_f1": row.get("valid_f1"), "valid_recall": row.get("valid_recall"), "valid_far": row.get("valid_far"), "test_f1": row.get("test_f1"), "test_far": row.get("test_far")} for row in selected_rows if row["seed"] == seed}}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(detail_rows)
    selected = pd.DataFrame(selected_rows)
    evaluated = selected[selected["status"].eq("EVALUATED")].copy()
    summary_rows = []
    for point, group in evaluated.groupby("operating_point", sort=False):
        summary_rows.append({"operating_point": point, "seeds": int(group["seed"].nunique()), "test_f1_mean": float(group["test_f1"].mean()), "test_f1_std": float(group["test_f1"].std(ddof=1)), "test_recall_mean": float(group["test_recall"].mean()), "test_precision_mean": float(group["test_precision"].mean()), "test_auprc_mean": float(group["test_auprc"].mean()), "test_far_mean": float(group["test_far"].mean()), "fixed_test_f1_mean": float(group["fixed_test_f1"].mean()), "fixed_test_recall_mean": float(group["fixed_test_recall"].mean()), "fixed_test_far_mean": float(group["fixed_test_far"].mean()), "f1_gain_vs_fixed": float((group["test_f1"] - group["fixed_test_f1"]).mean()), "far_gain_vs_fixed": float((group["test_far"] - group["fixed_test_far"]).mean()), "max_prefix_attack_rows": int(group[["valid_prefix_attack_rows", "test_prefix_attack_rows"]].max().max())})
    summary = pd.DataFrame(summary_rows)
    status = "CANDIDATE_GAIN_REQUIRES_MORE_YEARS" if ((summary["f1_gain_vs_fixed"] > 0) & (summary["far_gain_vs_fixed"] <= 0)).any() else "NEGATIVE_RESULT_RETAINED"
    payload = {"status": status, "protocol": "2021训练；2022验证选择目标年前缀天数、站点分位数和固定阈值收缩系数；2023仅评价。报告max-F1、valid Recall>=0.5最低FAR、valid FAR<=0.10最高F1三类预定义运行点。前缀不使用标签做阈值，但透明记录其中受控注入行数。", "stations": stations, "seeds": SEEDS, "grid": {"prefix_days": PREFIX_DAYS, "quantiles": QUANTILES, "shrinkages": SHRINKAGES}, "summary": summary_rows, "selected": selected_rows}
    detail.to_csv(TABLES / "stage80_noaa_constrained_prefix_adaptation_detail.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(TABLES / "stage80_noaa_constrained_prefix_adaptation_selected.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "stage80_noaa_constrained_prefix_adaptation_summary.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage80_noaa_constrained_prefix_adaptation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段80 NOAA受约束前缀自适应运行点", "", "目标年前缀站点分位数与冻结阈值做收缩融合。参数只由2022验证期选择，2023仅评价；前缀内受控注入行数被显式审计。", "", f"- 状态：**{status}**", "", "| 运行点 | seeds | test F1 | Recall | Precision | AUPRC | FAR | 固定F1 | 固定Recall | 固定FAR | F1差值 | FAR差值 | 前缀攻击行上限 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append(f"| {row['operating_point']} | {row['seeds']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f} | {row['test_precision_mean']:.4f} | {row['test_auprc_mean']:.4f} | {row['test_far_mean']:.4f} | {row['fixed_test_f1_mean']:.4f} | {row['fixed_test_recall_mean']:.4f} | {row['fixed_test_far_mean']:.4f} | {row['f1_gain_vs_fixed']:.4f} | {row['far_gain_vs_fixed']:.4f} | {row['max_prefix_attack_rows']} |")
    lines += ["", "无候选可因test结果重新选择。只有在验证约束与跨年测试同时成立时，候选才可进入后续更多年份复核；否则作为负结果和运行点权衡保留。"]
    (REPORTS / "stage80_noaa_constrained_prefix_adaptation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary_rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
