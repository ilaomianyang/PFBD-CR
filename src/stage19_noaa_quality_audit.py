from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from detectors import CUSUMDetector, HampelMADDetector, KalmanInnovationDetector, PhysicalRulesDetector, SpatialConsistencyDetector  # noqa: E402
from stage09_trust_pipeline import _prepare_local, _score_splits  # noqa: E402
from trust_fusion import TrustFusion  # noqa: E402


def main() -> int:
    local = _prepare_local(ROOT / "tables" / "stage05_controlled_attack_extensions.parquet")
    frame = local.copy(); frame["observed_value"] = frame["effective_observed_value"]
    train = frame["split"].eq("train"); normal = train & ~frame["combined_is_attack"]
    detectors = [PhysicalRulesDetector(), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    for detector in detectors:
        detector.fit(frame.loc[train], normal.loc[train])
        local[detector.name] = _score_splits(detector, frame)
    local["ERA5 Background"] = 0.0
    valid = local["split"].eq("valid")
    fusion = TrustFusion(include_era5=False).fit(local.loc[train], local.loc[valid], local.loc[valid, "combined_is_attack"])
    noaa = pd.read_parquet(ROOT / "tables" / "stage06_noaa_isd_normalized.parquet").rename(columns={"station_id": "station_key", "temperature_c": "observed_value"}).copy()
    noaa["timestamp_utc"] = pd.to_datetime(noaa["timestamp_utc"], utc=True, errors="coerce")
    noaa = noaa.dropna(subset=["timestamp_utc", "station_key"]).copy()
    for detector in detectors:
        noaa[detector.name] = detector.score(noaa)
    noaa["ERA5 Background"] = 0.0; noaa["missing_observation"] = noaa["observed_value"].isna()
    fused = fusion.transform(noaa, extreme_background=pd.Series(False, index=noaa.index))
    noaa["fused_risk"] = fused
    noaa["quality_flagged"] = pd.to_numeric(noaa["temperature_quality"], errors="coerce").fillna(9).ne(1) | noaa["observed_value"].isna()
    local_cal = float(np.nanquantile(fused[noaa["timestamp_utc"] < pd.Timestamp("2022-07-01", tz="UTC")], 0.995))
    noaa["alarm_local_valid_threshold"] = fused >= fusion.threshold
    noaa["alarm_external_calibrated"] = fused >= local_cal
    rows = []
    for label, mask in [("quality_flagged", noaa["quality_flagged"]), ("quality_clean", ~noaa["quality_flagged"])]:
        part = noaa.loc[mask]
        rows.append({"group": label, "rows": len(part), "mean_risk": float(part["fused_risk"].mean()) if len(part) else float("nan"), "p95_risk": float(part["fused_risk"].quantile(0.95)) if len(part) else float("nan"), "alarm_rate_local_valid_threshold": float(part["alarm_local_valid_threshold"].mean()) if len(part) else float("nan"), "alarm_rate_external_calibrated": float(part["alarm_external_calibrated"].mean()) if len(part) else float("nan")})
    result = pd.DataFrame(rows)
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    result.to_csv(out_dir / "stage19_noaa_quality_audit.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方质量标记仅作弱标签审计，不等同攻击真值；阈值由本地valid或NOAA前期校准得到", "fusion_threshold_local": fusion.threshold, "external_calibrated_threshold": local_cal, "results": result.to_dict(orient="records"), "flagged_rows": int(noaa["quality_flagged"].sum())}
    (out_dir / "stage19_noaa_quality_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段19 NOAA官方质量标记审计", "", "NOAA质量标记是自然观测质量控制信息，不是网络攻击真值；本表只用于弱标签审计。", "", "| 分组 | 行数 | 平均风险 | P95风险 | 本地valid阈值报警率 | 外部校准阈值报警率 |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in result.iterrows():
        lines.append(f"| {r['group']} | {int(r['rows'])} | {r['mean_risk']:.4f} | {r['p95_risk']:.4f} | {r['alarm_rate_local_valid_threshold']:.4f} | {r['alarm_rate_external_calibrated']:.4f} |")
    lines += ["", f"质量标记或缺失记录数：`{payload['flagged_rows']}`", "该结果不能写成攻击检测召回率，只能作为外部自然观测质量审计。"]
    report = report_dir / "stage19_noaa_quality_audit.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
