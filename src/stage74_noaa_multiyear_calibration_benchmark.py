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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage64_noaa_empirical_percentile_calibration import calibrated_percentile, fit_empirical  # noqa: E402
from stage73_noaa_multiyear_controlled_benchmark import build_dataset  # noqa: E402


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_multiyear_validation")}
    stations = {station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in [2021, 2022, 2023])}
    source = source[source["station_id"].astype(str).isin(stations) & source["year"].isin([2021, 2022, 2023])].copy()
    selected_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        frame = build_dataset(source, seed, stations)
        scored, reference = score(frame)
        scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
        train = scored["split"].eq("train")
        valid = scored["split"].eq("valid")
        test = scored["split"].eq("test")
        yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
        yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
        calibration = fit_empirical(scored.loc[train])
        candidates: list[dict[str, object]] = []
        weights = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.25, 0.25), (0.5, 0.25), (0.25, 0.5)]
        for station_weight, hour_weight in weights:
            calibrated = calibrated_percentile(scored, calibration, station_weight, hour_weight)
            for q in [0.90, 0.925, 0.95, 0.975, 0.99, 0.995]:
                threshold = float(np.nanquantile(calibrated[train.to_numpy()], q))
                pv = calibrated[valid.to_numpy()] >= threshold
                pt = calibrated[test.to_numpy()] >= threshold
                mv = _metrics(yv, pv, calibrated[valid.to_numpy()])
                mt = _metrics(yt, pt, calibrated[test.to_numpy()])
                candidates.append({"seed": seed, "station_weight": station_weight, "hour_weight": hour_weight, "global_weight": max(0.0, 1.0 - station_weight - hour_weight), "q": q, "threshold": threshold, "valid_f1": mv["f1"], "valid_far": float(pv[~yv].mean()), "test_f1": mt["f1"], "test_recall": mt["recall"], "test_precision": mt["precision"], "test_auprc": mt["average_precision"], "test_far": float(pt[~yt].mean()), "fixed_test_f1": reference["test"]["f1"], "fixed_test_recall": reference["test"]["recall"], "fixed_test_auprc": reference["test"]["average_precision"], "fixed_test_far": reference["test_false_alarm_rate"]})
        best = max(candidates, key=lambda x: (x["valid_f1"], -x["valid_far"]))
        selected_rows.append(best)
        detail_rows.extend(candidates)
        print(json.dumps({"seed": seed, "selected": best, "candidate_count": len(candidates)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(detail_rows)
    selected = pd.DataFrame(selected_rows)
    metrics = ["valid_f1", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far", "fixed_test_f1", "fixed_test_recall", "fixed_test_auprc", "fixed_test_far"]
    summary = {metric: {"mean": float(selected[metric].mean()), "std": float(selected[metric].std(ddof=1)), "min": float(selected[metric].min()), "max": float(selected[metric].max())} for metric in metrics}
    f1_gain = summary["test_f1"]["mean"] - summary["fixed_test_f1"]["mean"]
    auprc_gain = summary["test_auprc"]["mean"] - summary["fixed_test_auprc"]["mean"]
    status = "CANDIDATE_GAIN_REQUIRES_MORE_YEARS" if f1_gain > 0 and auprc_gain >= 0 else "NEGATIVE_RESULT_RETAINED"
    payload = {"status": status, "protocol": "NOAA三年度、11个三年质量合格站点、5种子；2021训练、2022验证选择经验分位数校准、2023锁定测试；test只评价；攻击为透明受控注入。", "stations": sorted(stations), "seeds": SEEDS, "summary": summary, "selected_by_valid": selected_rows, "f1_gain_vs_fixed": f1_gain, "auprc_gain_vs_fixed": auprc_gain}
    detail.to_csv(TABLES / "stage74_noaa_multiyear_calibration_benchmark_detail.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(TABLES / "stage74_noaa_multiyear_calibration_benchmark_selected.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage74_noaa_multiyear_calibration_benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段74 NOAA多年度站点分位数校准基准", "", "2021训练、2022验证选择校准、2023锁定测试；攻击为透明受控注入。", "", f"- 状态：**{status}**", f"- 校准方案test F1：`{summary['test_f1']['mean']:.4f}±{summary['test_f1']['std']:.4f}`；固定方案：`{summary['fixed_test_f1']['mean']:.4f}`；差值：`{f1_gain:.4f}`", f"- AUPRC差值：`{auprc_gain:.4f}`；校准FAR：`{summary['test_far']['mean']:.4f}±{summary['test_far']['std']:.4f}`；固定FAR：`{summary['fixed_test_far']['mean']:.4f}`", "", "| seed | 站点权重 | 小时权重 | q | valid F1 | test F1 | Recall | AUPRC | FAR |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in selected_rows:
        lines.append(f"| {row['seed']} | {row['station_weight']:.2f} | {row['hour_weight']:.2f} | {row['q']:.3f} | {row['valid_f1']:.4f} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | {row['test_auprc']:.4f} | {row['test_far']:.4f} |")
    lines += ["", "该校准只利用2022验证期选择参数；若不能稳定降低2023误报并保持检测能力，不进入主模型。"]
    (REPORTS / "stage74_noaa_multiyear_calibration_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary, "f1_gain_vs_fixed": f1_gain, "auprc_gain_vs_fixed": auprc_gain}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
