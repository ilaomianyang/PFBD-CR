from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"


def audit_station(group: pd.DataFrame) -> dict[str, object]:
    station = str(group["station_id"].iloc[0])
    ts = pd.to_datetime(group["timestamp_utc"], utc=True, errors="coerce").dropna().sort_values()
    unique = ts.drop_duplicates()
    date_min = unique.min()
    date_max = unique.max()
    expected = int(((date_max - date_min).total_seconds() // 3600) + 1) if len(unique) else 0
    gaps = unique.diff().dropna().dt.total_seconds().div(3600.0)
    nominal_interval = float(gaps.round().mode().iloc[0]) if len(gaps) else 1.0
    nominal_interval = max(nominal_interval, 1.0)
    expected_nominal = int(((date_max - date_min).total_seconds() // (nominal_interval * 3600)) + 1) if len(unique) else 0
    quality = pd.to_numeric(group["temperature_quality"], errors="coerce").fillna(9)
    return {
        "station_id": station,
        "rows": int(len(group)),
        "unique_timestamps": int(unique.nunique()),
        "duplicate_rows": int(len(ts) - unique.nunique()),
        "date_min": str(date_min),
        "date_max": str(date_max),
        "expected_hourly_rows": expected,
        "hourly_coverage": float(len(unique) / expected) if expected else float("nan"),
        "nominal_interval_hours": nominal_interval,
        "expected_nominal_rows": expected_nominal,
        "nominal_coverage": float(len(unique) / expected_nominal) if expected_nominal else float("nan"),
        "temperature_missing_rate": float(group["temperature_c"].isna().mean()),
        "temperature_quality_bad_rate": float(quality.ne(1).mean()),
        "max_gap_hours": float(gaps.max()) if len(gaps) else 0.0,
        "latitude_nunique": int(group["latitude"].dropna().nunique()),
        "longitude_nunique": int(group["longitude"].dropna().nunique()),
        "elevation_nunique": int(group["elevation_m"].dropna().nunique()),
    }


def main() -> int:
    frame = pd.read_parquet(NOAA)
    rows = [audit_station(group) for _, group in frame.groupby("station_id", sort=True)]
    detail = pd.DataFrame(rows)
    checks = {
        "no_duplicate_timestamps": bool((detail["duplicate_rows"] == 0).all()),
        "nominal_coverage_at_least_0p8": bool((detail["nominal_coverage"] >= 0.8).all()),
        "temperature_missing_below_0p2": bool((detail["temperature_missing_rate"] < 0.2).all()),
        "coordinates_stable": bool(((detail["latitude_nunique"] <= 1) & (detail["longitude_nunique"] <= 1)).all()),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "REVIEW_REQUIRED",
        "protocol": "对阶段6标准化NOAA表逐站审计时间唯一性、小时覆盖、温度缺失、官方质量标记、最大时间间隔和坐标稳定性；不涉及攻击标签。",
        "stations": int(len(detail)),
        "checks": checks,
        "summary": {
            "rows": int(len(frame)),
            "duplicate_rows": int(detail["duplicate_rows"].sum()),
            "min_hourly_coverage": float(detail["hourly_coverage"].min()),
            "min_nominal_coverage": float(detail["nominal_coverage"].min()),
            "max_temperature_missing_rate": float(detail["temperature_missing_rate"].max()),
            "max_quality_bad_rate": float(detail["temperature_quality_bad_rate"].max()),
            "max_gap_hours": float(detail["max_gap_hours"].max()),
        },
        "detail": rows,
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage58_noaa_data_quality_audit.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage58_noaa_data_quality_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段58 NOAA标准化数据质量审计", "", "逐站检查时间唯一性、名义采样间隔覆盖、温度缺失、官方质量标记、最大时间间隔和坐标稳定性。小时覆盖率按每小时计算；对3小时报站站点同时报告按名义间隔计算的覆盖率。", "", f"- 状态：**{payload['status']}**", f"- 站点数：`{payload['stations']}`；总行数：`{payload['summary']['rows']}`", f"- 重复行：`{payload['summary']['duplicate_rows']}`；最小小时覆盖率：`{payload['summary']['min_hourly_coverage']:.4f}`；最小名义覆盖率：`{payload['summary']['min_nominal_coverage']:.4f}`；最大温度缺失率：`{payload['summary']['max_temperature_missing_rate']:.4f}`；最大间隔：`{payload['summary']['max_gap_hours']:.1f} h`", "", "| 站点 | 行数 | 名义间隔(h) | 小时覆盖率 | 名义覆盖率 | 温度缺失率 | 质量标记异常率 | 最大间隔(h) | 重复行 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['station_id']} | {row['rows']} | {row['nominal_interval_hours']:.0f} | {row['hourly_coverage']:.4f} | {row['nominal_coverage']:.4f} | {row['temperature_missing_rate']:.4f} | {row['temperature_quality_bad_rate']:.4f} | {row['max_gap_hours']:.1f} | {row['duplicate_rows']} |")
    lines += ["", "该审计只说明数据可进入实验，不等同于攻击真值或性能通过。"]
    (REPORTS / "stage58_noaa_data_quality_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": payload["stations"], "checks": checks}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
