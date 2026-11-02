from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
ALIGNMENT = TABLES / "stage07_era5_alignment.parquet"
BIAS_DETAIL = TABLES / "stage08_era5_bias_detail.parquet"
REFERENCE = TABLES / "stage06_era5_station_reference.parquet"
CALIBRATION_SOURCE = ROOT / "src" / "stage08_external_calibration.py"


def metrics(label: str, data: pd.DataFrame) -> dict[str, object]:
    if data.empty:
        return {"subset": label, "rows": 0, "stations": 0, "raw_mae_c": None, "corrected_mae_c": None, "raw_rmse_c": None, "corrected_rmse_c": None}
    return {
        "subset": label,
        "rows": int(len(data)),
        "stations": int(data["station_key"].nunique()),
        "raw_mae_c": float(data["error_raw_c"].abs().mean()),
        "corrected_mae_c": float(data["error_corrected_c"].abs().mean()),
        "raw_rmse_c": float(np.sqrt(np.mean(data["error_raw_c"] ** 2))),
        "corrected_rmse_c": float(np.sqrt(np.mean(data["error_corrected_c"] ** 2))),
    }


def main() -> int:
    alignment = pd.read_parquet(ALIGNMENT)
    bias = pd.read_parquet(BIAS_DETAIL)
    reference = pd.read_parquet(REFERENCE)
    required_alignment = {"station_key", "hour_utc", "local_original_c", "local_observed_c", "era5_t2m_c", "grid_distance_km"}
    required_bias = {"station_key", "hour_utc", "anomaly_rows", "error_raw_c", "error_corrected_c", "median_bias_c", "era5_corrected_t2m_c"}
    if not required_alignment.issubset(alignment.columns):
        raise ValueError(f"stage07 alignment missing columns: {sorted(required_alignment.difference(alignment.columns))}")
    if not required_bias.issubset(bias.columns):
        raise ValueError(f"stage08 bias detail missing columns: {sorted(required_bias.difference(bias.columns))}")

    alignment["hour_utc"] = pd.to_datetime(alignment["hour_utc"], utc=True, errors="coerce")
    bias["hour_utc"] = pd.to_datetime(bias["hour_utc"], utc=True, errors="coerce")
    finite_cols = ["era5_t2m_c", "era5_u10_ms", "era5_v10_ms", "era5_wind_speed_ms", "era5_surface_pressure_hpa"]
    finite_available = [col for col in finite_cols if col in alignment.columns]
    calibration_text = CALIBRATION_SOURCE.read_text(encoding="utf-8")
    source_fit_check = 'aggregate("valid")' in calibration_text and '~local["is_anomaly"].astype(bool)' in calibration_text

    station_bias_counts = bias.groupby("station_key")["median_bias_c"].nunique(dropna=False)
    clean = bias.loc[bias["anomaly_rows"] == 0].copy()
    controlled = bias.loc[bias["anomaly_rows"] > 0].copy()
    checks = {
        "alignment_nonempty": len(alignment) > 0,
        "bias_nonempty": len(bias) > 0,
        "alignment_unique_station_hour": int(alignment.duplicated(["station_key", "hour_utc"]).sum()) == 0,
        "bias_unique_station_hour": int(bias.duplicated(["station_key", "hour_utc"]).sum()) == 0,
        "alignment_timestamps_valid": bool(alignment["hour_utc"].notna().all()),
        "bias_timestamps_valid": bool(bias["hour_utc"].notna().all()),
        "era5_variables_finite": bool(alignment[finite_available].notna().all().all()),
        "grid_distance_bounded_25km": bool(alignment["grid_distance_km"].between(0, 25).all()),
        "bias_station_set_in_reference": set(bias["station_key"].unique()).issubset(set(reference["station_key"].unique())),
        "median_bias_constant_per_station": bool((station_bias_counts <= 1).all()),
        "clean_and_controlled_hours_present": len(clean) > 0 and len(controlled) > 0,
        "calibration_source_uses_valid_normal_only": source_fit_check,
        "corrected_clean_mae_not_worse": bool(clean["error_corrected_c"].abs().mean() <= clean["error_raw_c"].abs().mean()),
    }
    metric_rows = [metrics("all_test_aligned_hours", bias), metrics("clean_hours_anomaly_rows_zero", clean), metrics("controlled_hours_anomaly_rows_positive", controlled)]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED" if all(checks.values()) else "FAIL",
        "protocol": "审计阶段07 ERA5小时对齐和阶段08偏差订正产物；检查站点—小时唯一性、时间有效性、ERA5变量有限值、网格距离、站点集合、每站固定偏差和自然/受控小时分层；同时静态核对阶段08源码仅用valid正常样本估计偏差，测试期受控注入不参与校准。",
        "alignment_source": str(ALIGNMENT),
        "bias_source": str(BIAS_DETAIL),
        "reference_source": str(REFERENCE),
        "checks": checks,
        "alignment_rows": int(len(alignment)),
        "bias_rows": int(len(bias)),
        "alignment_stations": int(alignment["station_key"].nunique()),
        "bias_stations": int(bias["station_key"].nunique()),
        "timestamp_min": str(bias["hour_utc"].min()),
        "timestamp_max": str(bias["hour_utc"].max()),
        "grid_distance_km_min": float(alignment["grid_distance_km"].min()),
        "grid_distance_km_max": float(alignment["grid_distance_km"].max()),
        "metrics": metric_rows,
        "limitations": [
            "ERA5是区域背景，不是现场传感器真值；对齐误差不能解释为传感器攻击检测性能。",
            "阶段08偏差订正的源码静态检查只能证明拟合数据边界，不能证明跨区域代表性。",
            "受控小时的订正误差只用于分层审计，不能当作真实攻击恢复效果。",
        ],
    }
    out = pd.DataFrame(metric_rows)
    out.to_csv(TABLES / "stage113_era5_alignment_leakage_audit.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage113_era5_alignment_leakage_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段113 ERA5对齐与偏差订正泄漏审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 阶段07对齐行数：`{payload['alignment_rows']}`；站点数：`{payload['alignment_stations']}`",
        f"- 阶段08偏差明细行数：`{payload['bias_rows']}`；站点数：`{payload['bias_stations']}`",
        f"- 时间范围：`{payload['timestamp_min']}` 至 `{payload['timestamp_max']}`",
        f"- ERA5网格距离：`{payload['grid_distance_km_min']:.3f}`–`{payload['grid_distance_km_max']:.3f}` km",
        "",
        "## 检查项",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in checks.items())
    lines += ["", "## 分层误差", "", "| 子集 | 行数 | 站点 | 原始MAE (°C) | 订正MAE (°C) | 原始RMSE (°C) | 订正RMSE (°C) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in metric_rows:
        lines.append(f"| {row['subset']} | {row['rows']} | {row['stations']} | {row['raw_mae_c']:.4f} | {row['corrected_mae_c']:.4f} | {row['raw_rmse_c']:.4f} | {row['corrected_rmse_c']:.4f} |")
    lines += ["", "ERA5仅作为区域背景变量；源码检查确认偏差由valid正常样本估计，测试期受控注入未参与偏差拟合。"]
    (REPORTS / "stage113_era5_alignment_leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks), "failures": sum(not x for x in checks.values())}, ensure_ascii=False))
    return 0 if payload["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
