from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage70_noaa_isd_multiyear_normalized.parquet"


def audit_group(group: pd.DataFrame) -> dict[str, object]:
    year = int(group["year"].iloc[0])
    station = str(group["station_id"].iloc[0])
    ts = pd.to_datetime(group["timestamp_utc"], utc=True, errors="coerce").dropna().sort_values()
    unique = ts.drop_duplicates()
    date_min, date_max = unique.min(), unique.max()
    gaps = unique.diff().dropna().dt.total_seconds().div(3600.0)
    nominal = float(gaps.round().mode().iloc[0]) if len(gaps) else 1.0
    nominal = max(nominal, 1.0)
    expected = int(((date_max - date_min).total_seconds() // (nominal * 3600)) + 1) if len(unique) else 0
    quality = pd.to_numeric(group["temperature_quality"], errors="coerce").fillna(9)
    return {"year": year, "station_id": station, "rows": int(len(group)), "unique_timestamps": int(unique.nunique()), "duplicate_rows": int(len(ts) - unique.nunique()), "date_min": str(date_min), "date_max": str(date_max), "nominal_interval_hours": nominal, "nominal_coverage": float(len(unique) / expected) if expected else float("nan"), "temperature_missing_rate": float(group["temperature_c"].isna().mean()), "temperature_quality_bad_rate": float(quality.ne(1).mean()), "max_gap_hours": float(gaps.max()) if len(gaps) else 0.0, "latitude_nunique": int(group["latitude"].dropna().nunique()), "longitude_nunique": int(group["longitude"].dropna().nunique()), "elevation_nunique": int(group["elevation_m"].dropna().nunique())}


def main() -> int:
    frame = pd.read_parquet(NOAA)
    detail = pd.DataFrame([audit_group(group) for _, group in frame.groupby(["year", "station_id"], sort=True)])
    checks = {"expected_file_rows": bool(len(detail) == 36), "no_duplicate_timestamps": bool((detail["duplicate_rows"] == 0).all()), "temperature_missing_below_0p2": bool((detail["temperature_missing_rate"] < 0.2).all()), "coordinates_stable": bool(((detail["latitude_nunique"] <= 1) & (detail["longitude_nunique"] <= 1)).all()), "each_station_has_three_years": bool(frame.groupby("station_id")["year"].nunique().eq(3).all())}
    detail["eligible_for_multiyear_validation"] = (detail["duplicate_rows"] == 0) & (detail["nominal_coverage"] >= 0.8) & (detail["temperature_missing_rate"] < 0.2) & (detail["latitude_nunique"] <= 1) & (detail["longitude_nunique"] <= 1)
    checks["eligible_station_years_exist"] = bool(detail["eligible_for_multiyear_validation"].any())
    excluded = detail.loc[~detail["eligible_for_multiyear_validation"]].to_dict(orient="records")
    yearly = detail.groupby("year", sort=True).agg(stations=("station_id", "nunique"), rows=("rows", "sum"), min_nominal_coverage=("nominal_coverage", "min"), max_temperature_missing_rate=("temperature_missing_rate", "max"), duplicate_rows=("duplicate_rows", "sum")).reset_index()
    status = "PASS" if all(checks.values()) and not excluded else ("PASS_WITH_EXCLUSIONS" if all(checks.values()) else "REVIEW_REQUIRED")
    payload = {"status": status, "protocol": "对阶段70三年度NOAA标准化表逐年逐站审计时间唯一性、名义采样覆盖、温度缺失、质量标记、最大间隔和坐标稳定性；不涉及攻击标签；不合格站点-年份排除出后续验证。", "checks": checks, "years": sorted(frame["year"].unique().tolist()), "stations": int(frame["station_id"].nunique()), "rows": int(len(frame)), "eligible_station_years": int(detail["eligible_for_multiyear_validation"].sum()), "excluded_station_years": excluded, "detail": detail.to_dict(orient="records"), "yearly": yearly.to_dict(orient="records")}
    detail.to_csv(TABLES / "stage71_noaa_multiyear_quality_audit_detail.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(TABLES / "stage71_noaa_multiyear_quality_audit_yearly.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage71_noaa_multiyear_quality_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段71 NOAA多年度逐年逐站质量审计", "", "独立审计2021、2022、2023每个站点-年份；不涉及攻击标签。", "", f"- 状态：**{payload['status']}**；站点：`{payload['stations']}`；年份：`{payload['years']}`；总行数：`{payload['rows']}`", f"- 可进入多年度验证的站点-年份：`{payload['eligible_station_years']}`；排除：`{len(excluded)}`", f"- 检查：`{checks}`", "", "| 年份 | 站点数 | 行数 | 最小名义覆盖率 | 最大温度缺失率 | 重复行 |", "|---:|---:|---:|---:|---:|---:|"]
    for row in yearly.to_dict(orient="records"):
        lines.append(f"| {row['year']} | {row['stations']} | {row['rows']} | {row['min_nominal_coverage']:.4f} | {row['max_temperature_missing_rate']:.4f} | {row['duplicate_rows']} |")
    if excluded:
        lines += ["", "## 排除的站点-年份", "", "| 年份 | 站点 | 名义覆盖率 | 最大间隔(h) | 原因 |", "|---:|---|---:|---:|---|"]
        for row in excluded:
            reason = "名义覆盖率低于0.8" if row["nominal_coverage"] < 0.8 else "其他质量条件未通过"
            lines.append(f"| {row['year']} | {row['station_id']} | {row['nominal_coverage']:.4f} | {row['max_gap_hours']:.1f} | {reason} |")
    lines += ["", "只有通过本审计的年份和站点，才允许进入多年度自然观测验证；排除项必须在论文和补充材料中披露。"]
    (REPORTS / "stage71_noaa_multiyear_quality_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": payload["stations"], "years": payload["years"], "checks": checks}, ensure_ascii=False))
    return 0 if payload["status"] in {"PASS", "PASS_WITH_EXCLUSIONS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
