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
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402
from stage64_noaa_empirical_percentile_calibration import calibrated_percentile, fit_empirical  # noqa: E402


def main() -> int:
    source = pd.read_parquet(NOAA)
    selected = pd.read_csv(TABLES / "stage64_noaa_empirical_percentile_calibration_selected.csv")
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        choice = selected.loc[selected["seed"].eq(seed)].iloc[0].to_dict()
        frame = build_dataset(source, seed)
        scored, _ = score(frame)
        scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
        calibration = fit_empirical(scored.loc[scored["split"].eq("train")])
        calibrated = calibrated_percentile(scored, calibration, float(choice["station_weight"]), float(choice["hour_weight"]))
        test = scored["split"].eq("test")
        natural = test & ~scored["combined_is_attack"].astype(bool)
        quality_flagged = pd.to_numeric(scored["temperature_quality"], errors="coerce").fillna(9).ne(1)
        for station, station_mask in scored.loc[natural].groupby(scored.loc[natural, "station_key"].astype(str), sort=True).groups.items():
            idx = scored.index.take(station_mask)
            sf = idx.intersection(scored.index[quality_flagged])
            sc = idx.difference(sf)
            for group, group_idx in [("quality_flagged", sf), ("quality_clean", sc)]:
                values = calibrated[group_idx.to_numpy()] if len(group_idx) else np.array([], dtype=float)
                rows.append({"seed": seed, "station_key": str(station), "group": group, "rows": int(len(group_idx)), "mean_percentile": float(values.mean()) if len(values) else float("nan"), "alarm_rate": float((values >= float(choice["threshold"])).mean()) if len(values) else float("nan")})
    detail = pd.DataFrame(rows)
    pivot = detail.pivot_table(index=["seed", "station_key"], columns="group", values=["rows", "mean_percentile", "alarm_rate"], aggfunc="first")
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    pivot["alarm_rate_diff_flagged_minus_clean"] = pivot["alarm_rate_quality_flagged"] - pivot["alarm_rate_quality_clean"]
    aggregate = pivot.groupby("station_key", sort=True)[["rows_quality_flagged", "rows_quality_clean", "alarm_rate_quality_flagged", "alarm_rate_quality_clean", "alarm_rate_diff_flagged_minus_clean"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["station_key"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    positive_stations = int((aggregate["alarm_rate_diff_flagged_minus_clean_mean"] > 0).sum())
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "使用阶段64每种子valid选定参数；只统计test中未注入的自然观测；按站点分层；NOAA质量标记仅作弱标签。", "seeds": SEEDS, "detail": rows, "aggregate_by_station": aggregate.to_dict(orient="records"), "stations_with_positive_alarm_rate_difference": positive_stations, "station_count": int(len(aggregate))}
    detail.to_csv(TABLES / "stage68_noaa_quality_station_stratified_detail.csv", index=False, encoding="utf-8-sig")
    pivot.to_csv(TABLES / "stage68_noaa_quality_station_stratified_seed_station.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage68_noaa_quality_station_stratified_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage68_noaa_quality_station_stratified.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段68 NOAA质量弱标签站点分层", "", "只统计test中未注入的自然观测；质量标记不是攻击真值。", "", "| 站点 | 可疑行数 | 正常行数 | 可疑报警率 | 正常报警率 | 差值 |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['station_key']} | {row['rows_quality_flagged_mean']:.1f}±{row['rows_quality_flagged_std']:.1f} | {row['rows_quality_clean_mean']:.1f}±{row['rows_quality_clean_std']:.1f} | {row['alarm_rate_quality_flagged_mean']:.4f} | {row['alarm_rate_quality_clean_mean']:.4f} | {row['alarm_rate_diff_flagged_minus_clean_mean']:.4f} |")
    lines += ["", f"12站点中质量可疑组报警率高于正常组的站点数：`{positive_stations}/{len(aggregate)}`。", "该结果用于站点异质性审计，不能写成真实攻击识别率。"]
    (REPORTS / "stage68_noaa_quality_station_stratified.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(aggregate), "positive_stations": positive_stations}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
