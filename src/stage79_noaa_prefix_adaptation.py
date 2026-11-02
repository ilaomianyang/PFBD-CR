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
QUANTILES = [0.95, 0.975, 0.99, 0.995]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage73_noaa_multiyear_controlled_benchmark import build_dataset  # noqa: E402


def prefix_mask(frame: pd.DataFrame, year: int, days: int) -> np.ndarray:
    out = np.zeros(len(frame), dtype=bool)
    part = frame[frame["year"].eq(year)].copy()
    part["timestamp_utc"] = pd.to_datetime(part["timestamp_utc"], utc=True, errors="coerce")
    for _, group in part.groupby("station_key", sort=False):
        start = group["timestamp_utc"].min()
        mask = group["timestamp_utc"].lt(start + pd.Timedelta(days=int(days)))
        out[group.index.to_numpy()[mask.to_numpy()]] = True
    return out


def adaptive_predictions(scored: pd.DataFrame, year: int, prefix_days: int, quantile: float) -> tuple[np.ndarray, dict[str, float]]:
    scored = scored.copy()
    scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
    in_year = scored["year"].eq(year).to_numpy()
    prefix = prefix_mask(scored, year, prefix_days)
    risk = scored["robust_fused_risk"].to_numpy(dtype=float)
    pred = np.zeros(len(scored), dtype=bool)
    thresholds: dict[str, float] = {}
    for station, group in scored.loc[in_year].groupby("station_key", sort=True):
        idx = group.index.to_numpy()
        ref_idx = idx[prefix[idx] & np.isfinite(risk[idx])]
        threshold = float(np.quantile(risk[ref_idx], quantile)) if len(ref_idx) >= 20 else float("inf")
        thresholds[str(station)] = threshold
        eligible = idx[np.isfinite(risk[idx]) & ~prefix[idx]]
        pred[eligible] = risk[eligible] >= threshold
    return pred, thresholds


def evaluate_region(scored: pd.DataFrame, pred: np.ndarray, year: int, prefix_days: int) -> dict[str, float]:
    region = scored["year"].eq(year).to_numpy() & ~prefix_mask(scored, year, prefix_days)
    y = scored.loc[region, "combined_is_attack"].astype(bool).to_numpy()
    p = pred[region]
    s = scored.loc[region, "robust_fused_risk"].to_numpy(dtype=float)
    m = _metrics(y, p, s)
    return {"f1": float(m["f1"]), "recall": float(m["recall"]), "precision": float(m["precision"]), "auprc": float(m["average_precision"]), "far": float(p[~y].mean()) if (~y).any() else float("nan"), "rows": int(region.sum()), "attack_rows": int(y.sum())}


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_multiyear_validation")}
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in [2021, 2022, 2023])})
    source = source[source["station_id"].astype(str).isin(stations) & source["year"].isin([2021, 2022, 2023])].copy()
    rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for seed in SEEDS:
        frame = build_dataset(source, seed, set(stations))
        scored, fixed = score(frame)
        scored["year"] = pd.to_numeric(scored["year"], errors="coerce").astype(int)
        scored["station_key"] = scored["station_key"].astype(str)
        fixed_threshold = float(fixed["setting"]["threshold"])
        valid_region = scored["year"].eq(2022).to_numpy() & ~prefix_mask(scored, 2022, 30)
        test_region = scored["year"].eq(2023).to_numpy() & ~prefix_mask(scored, 2023, 30)
        fixed_pred = scored["robust_fused_risk"].to_numpy(dtype=float) >= fixed_threshold
        fixed_valid = evaluate_region(scored, fixed_pred, 2022, 30)
        fixed_test = evaluate_region(scored, fixed_pred, 2023, 30)
        candidates: list[dict[str, object]] = []
        for days in PREFIX_DAYS:
            for q in QUANTILES:
                pred, thresholds = adaptive_predictions(scored, 2022, days, q)
                valid = evaluate_region(scored, pred, 2022, days)
                candidates.append({"seed": seed, "prefix_days": days, "quantile": q, "valid_f1": valid["f1"], "valid_far": valid["far"], "valid_recall": valid["recall"], "valid_auprc": valid["auprc"], "station_threshold_min": float(min(thresholds.values())) if thresholds else float("nan"), "station_threshold_median": float(np.median(list(thresholds.values()))) if thresholds else float("nan"), "station_threshold_max": float(max(thresholds.values())) if thresholds else float("nan"), "fixed_valid_f1": fixed_valid["f1"], "fixed_valid_far": fixed_valid["far"]})
        best = max(candidates, key=lambda x: (x["valid_f1"], -x["valid_far"]))
        test_pred, test_thresholds = adaptive_predictions(scored, 2023, int(best["prefix_days"]), float(best["quantile"]))
        adaptive_test = evaluate_region(scored, test_pred, 2023, int(best["prefix_days"]))
        selected_row = {**best, "test_f1": adaptive_test["f1"], "test_recall": adaptive_test["recall"], "test_precision": adaptive_test["precision"], "test_auprc": adaptive_test["auprc"], "test_far": adaptive_test["far"], "test_rows": adaptive_test["rows"], "test_attack_rows": adaptive_test["attack_rows"], "fixed_test_f1": fixed_test["f1"], "fixed_test_recall": fixed_test["recall"], "fixed_test_precision": fixed_test["precision"], "fixed_test_auprc": fixed_test["auprc"], "fixed_test_far": fixed_test["far"], "test_station_threshold_min": float(min(test_thresholds.values())) if test_thresholds else float("nan"), "test_station_threshold_median": float(np.median(list(test_thresholds.values()))) if test_thresholds else float("nan"), "test_station_threshold_max": float(max(test_thresholds.values())) if test_thresholds else float("nan")}
        selected.append(selected_row)
        rows.extend(candidates)
        print(json.dumps({"seed": seed, "selected": {"prefix_days": best["prefix_days"], "quantile": best["quantile"], "valid_f1": best["valid_f1"], "valid_far": best["valid_far"]}, "test_f1": adaptive_test["f1"], "fixed_test_f1": fixed_test["f1"]}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected)
    summary_metrics = ["valid_f1", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far", "fixed_test_f1", "fixed_test_recall", "fixed_test_precision", "fixed_test_auprc", "fixed_test_far"]
    summary = {metric: {"mean": float(selected_df[metric].mean()), "std": float(selected_df[metric].std(ddof=1)), "min": float(selected_df[metric].min()), "max": float(selected_df[metric].max())} for metric in summary_metrics}
    f1_gain = summary["test_f1"]["mean"] - summary["fixed_test_f1"]["mean"]
    auprc_gain = summary["test_auprc"]["mean"] - summary["fixed_test_auprc"]["mean"]
    far_gain = summary["test_far"]["mean"] - summary["fixed_test_far"]["mean"]
    status = "CANDIDATE_GAIN_REQUIRES_MORE_YEARS" if f1_gain > 0 and auprc_gain >= 0 and far_gain <= 0 else "NEGATIVE_RESULT_RETAINED"
    payload = {"status": status, "protocol": "NOAA官方三年度数据；11个三年质量合格站点；每个种子使用2022目标年份前缀的无标签风险分位数选择prefix_days和q，固定后应用于2023目标年份前缀后的锁定测试；不使用2023标签调参；攻击仅为透明受控注入。", "stations": stations, "seeds": SEEDS, "prefix_days": PREFIX_DAYS, "quantiles": QUANTILES, "summary": summary, "gains_vs_fixed": {"f1": f1_gain, "auprc": auprc_gain, "far": far_gain}, "selected_by_valid": selected}
    detail.to_csv(TABLES / "stage79_noaa_prefix_adaptation_detail.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage79_noaa_prefix_adaptation_selected.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage79_noaa_prefix_adaptation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段79 NOAA目标年份无标签前缀自适应", "", "只用目标年份最初前缀的风险分布估计每站点阈值；prefix_days和分位数由2022验证期选择，冻结后应用于2023前缀之后的锁定测试。", "", f"- 状态：**{status}**", f"- 2023 test F1：`{summary['test_f1']['mean']:.4f}±{summary['test_f1']['std']:.4f}`；固定阈值：`{summary['fixed_test_f1']['mean']:.4f}`；差值：`{f1_gain:.4f}`", f"- AUPRC差值：`{auprc_gain:.4f}`；FAR差值：`{far_gain:.4f}`（负值表示误报下降）", "", "| seed | 前缀天数 | 分位数 | valid F1 | test F1 | Recall | AUPRC | FAR | 固定F1 | 固定FAR |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in selected:
        lines.append(f"| {int(row['seed'])} | {int(row['prefix_days'])} | {row['quantile']:.3f} | {row['valid_f1']:.4f} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | {row['test_auprc']:.4f} | {row['test_far']:.4f} | {row['fixed_test_f1']:.4f} | {row['fixed_test_far']:.4f} |")
    lines += ["", "结果只用于判断跨年份无标签自适应是否值得进入后续预注册方案；无论正负结果，都不覆盖9B主模型或修改锁定测试数据。"]
    (REPORTS / "stage79_noaa_prefix_adaptation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary, "gains_vs_fixed": payload["gains_vs_fixed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
