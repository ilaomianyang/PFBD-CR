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

from stage38_noaa_controlled_benchmark import COMPLETE_TYPES, score  # noqa: E402
from stage73_noaa_multiyear_controlled_benchmark import build_dataset  # noqa: E402
from stage64_noaa_empirical_percentile_calibration import calibrated_percentile, fit_empirical  # noqa: E402


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_multiyear_validation")}
    stations = {station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in [2021, 2022, 2023])}
    source = source[source["station_id"].astype(str).isin(stations) & source["year"].isin([2021, 2022, 2023])].copy()
    selected = json.loads((TABLES / "stage74_noaa_multiyear_calibration_benchmark.json").read_text(encoding="utf-8"))["selected_by_valid"]
    selected_map = {int(row["seed"]): row for row in selected}
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        frame = build_dataset(source, seed, stations)
        scored, _ = score(frame)
        scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
        calibration = fit_empirical(scored.loc[scored["split"].eq("train")])
        choice = selected_map[seed]
        calibrated = calibrated_percentile(scored, calibration, float(choice["station_weight"]), float(choice["hour_weight"]))
        fixed = scored["predicted_attack"].astype(bool).to_numpy()
        threshold = float(choice["threshold"])
        calibrated_pred = calibrated >= threshold
        test = scored["split"].eq("test").to_numpy()
        for attack_type in COMPLETE_TYPES:
            mask = test & scored["combined_attack_type"].eq(attack_type).to_numpy()
            rows.append({"seed": seed, "attack_type": attack_type, "rows": int(mask.sum()), "fixed_recall": float(fixed[mask].mean()), "calibrated_recall": float(calibrated_pred[mask].mean()), "recall_change": float(calibrated_pred[mask].mean() - fixed[mask].mean())})
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("attack_type", sort=False)[["fixed_recall", "calibrated_recall", "recall_change"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["attack_type"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "使用阶段74每种子由2022 valid选择的校准参数；只统计2023 locked test攻击类型Recall；不重新调参。", "stations": sorted(stations), "seeds": SEEDS, "detail": rows, "aggregate": aggregate.to_dict(orient="records"), "interpretation": "校准Recall变化按攻击类型报告，不能删除低Recall类型。"}
    detail.to_csv(TABLES / "stage75_noaa_multiyear_calibration_attack_type_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage75_noaa_multiyear_calibration_attack_type_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage75_noaa_multiyear_calibration_attack_type.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段75 NOAA多年度校准攻击类型分层", "", "使用阶段74的验证集选择参数，统计2023 locked test逐类Recall；不重新调参。", "", "| 攻击类型 | 固定Recall | 校准Recall | 变化 |", "|---|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['attack_type']} | {row['fixed_recall_mean']:.4f}±{row['fixed_recall_std']:.4f} | {row['calibrated_recall_mean']:.4f}±{row['calibrated_recall_std']:.4f} | {row['recall_change_mean']:.4f} |")
    lines += ["", "该表用于解释跨年份校准的Recall/FAR权衡，不代表真实攻击日志。"]
    (REPORTS / "stage75_noaa_multiyear_calibration_attack_type.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "attack_types": len(aggregate), "rows": len(detail)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
