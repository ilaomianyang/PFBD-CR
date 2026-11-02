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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors import CUSUMDetector, HampelMADDetector, KalmanInnovationDetector, PhysicalRulesDetector, SpatialConsistencyDetector  # noqa: E402
from trust_fusion import robust_location_scale  # noqa: E402


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_multiyear_validation")}
    frame = source[source.apply(lambda r: (int(r["year"]), str(r["station_id"])) in eligible, axis=1)].copy()
    frame["station_key"] = frame["station_id"].astype(str)
    frame["observed_value"] = pd.to_numeric(frame["temperature_c"], errors="coerce")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame["missing_observation"] = frame["observed_value"].isna()
    frame["split"] = frame["year"].map({2021: "train", 2022: "valid", 2023: "test"})
    train = frame["split"].eq("train")
    detectors = [PhysicalRulesDetector(), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    for detector in detectors:
        detector.fit(frame.loc[train], ~frame.loc[train, "missing_observation"])
        values = np.full(len(frame), np.nan, dtype=float)
        for split in ["train", "valid", "test"]:
            mask = frame["split"].eq(split)
            if mask.any():
                values[mask.to_numpy()] = detector.score(frame.loc[mask])
        frame[detector.name] = values
    component_names = [detector.name for detector in detectors]
    locations = {}
    normalized = []
    for name in component_names:
        loc, scale = robust_location_scale(frame.loc[train & ~frame["missing_observation"], name], floor=0.25)
        locations[name] = {"location": loc, "scale": scale}
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        normalized.append(np.nan_to_num(np.maximum(values - loc, 0.0) / scale, nan=0.0, posinf=1e6))
    frame["equal_weight_risk"] = np.mean(np.vstack(normalized), axis=0)
    quality_flagged = pd.to_numeric(frame["temperature_quality"], errors="coerce").fillna(9).ne(1)
    rows: list[dict[str, object]] = []
    for q in [0.99, 0.995, 0.999]:
        threshold = float(np.nanquantile(frame.loc[train, "equal_weight_risk"], q))
        for year, split in [(2021, "train"), (2022, "valid"), (2023, "test")]:
            mask = frame["split"].eq(split)
            pred = frame["equal_weight_risk"].to_numpy(dtype=float) >= threshold
            values = frame.loc[mask, "equal_weight_risk"].to_numpy(dtype=float)
            flagged = mask & quality_flagged
            clean = mask & ~quality_flagged
            rows.append({"q": q, "threshold": threshold, "year": year, "split": split, "rows": int(mask.sum()), "mean_risk": float(values.mean()) if len(values) else float("nan"), "p95_risk": float(np.quantile(values, 0.95)) if len(values) else float("nan"), "alarm_rate": float(pred[mask.to_numpy()].mean()) if mask.any() else float("nan"), "quality_flagged_rows": int(flagged.sum()), "quality_flagged_alarm_rate": float(pred[flagged.to_numpy()].mean()) if flagged.any() else float("nan"), "quality_clean_rows": int(clean.sum()), "quality_clean_alarm_rate": float(pred[clean.to_numpy()].mean()) if clean.any() else float("nan")})
    detail = pd.DataFrame(rows)
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "2021训练、2022无标签验证、2023无标签测试；仅训练正常观测拟合检测器和等权融合，阈值为训练风险分位数；不注入攻击、不使用测试标签；排除阶段71不合格站点-年份。", "eligible_station_years": len(eligible), "component_locations": locations, "threshold_grid": [0.99, 0.995, 0.999], "detail": rows, "excluded_station_years": audit["excluded_station_years"]}
    detail.to_csv(TABLES / "stage72_noaa_multiyear_natural_validation.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage72_noaa_multiyear_natural_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段72 NOAA多年度自然观测验证", "", "2021训练、2022无标签验证、2023无标签测试；不注入攻击，不使用测试标签。", "", f"- 状态：**{payload['status']}**；合格站点-年份：`{len(eligible)}`；排除项：`{len(audit['excluded_station_years'])}`", "", "| q | 年份 | 行数 | 平均风险 | P95风险 | 报警率 | 质量可疑报警率 | 质量正常报警率 |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['q']:.3f} | {row['year']} | {row['rows']} | {row['mean_risk']:.4f} | {row['p95_risk']:.4f} | {row['alarm_rate']:.4f} | {row['quality_flagged_alarm_rate']:.4f} ({row['quality_flagged_rows']}) | {row['quality_clean_alarm_rate']:.4f} ({row['quality_clean_rows']}) |")
    lines += ["", "该实验只量化自然观测报警率和质量弱标签差异，不提供攻击召回率。2021-57014099999因覆盖率不足未进入验证。"]
    (REPORTS / "stage72_noaa_multiyear_natural_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "eligible_station_years": len(eligible), "rows": len(frame)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
