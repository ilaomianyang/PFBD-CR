from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"


def audit_group(group: pd.DataFrame) -> dict[str, object]:
    ts = pd.to_datetime(group["timestamp_utc"], utc=True, errors="coerce").dropna().sort_values()
    unique = ts.drop_duplicates()
    date_min, date_max = unique.min(), unique.max()
    gaps = unique.diff().dropna().dt.total_seconds().div(3600.0)
    nominal = float(gaps.round().mode().iloc[0]) if len(gaps) else 1.0
    nominal = max(nominal, 1.0)
    expected = int(((date_max - date_min).total_seconds() // (nominal * 3600)) + 1) if len(unique) else 0
    quality = pd.to_numeric(group["temperature_quality"], errors="coerce").fillna(9)
    return {"year": int(group["year"].iloc[0]), "station_id": str(group["station_id"].iloc[0]), "rows": int(len(group)), "unique_timestamps": int(unique.nunique()), "duplicate_rows": int(len(ts) - unique.nunique()), "date_min": str(date_min), "date_max": str(date_max), "nominal_interval_hours": nominal, "nominal_coverage": float(len(unique) / expected) if expected else float("nan"), "temperature_missing_rate": float(group["temperature_c"].isna().mean()), "temperature_quality_bad_rate": float(quality.ne(1).mean()), "max_gap_hours": float(gaps.max()) if len(gaps) else 0.0, "latitude_nunique": int(group["latitude"].dropna().nunique()), "longitude_nunique": int(group["longitude"].dropna().nunique()), "elevation_nunique": int(group["elevation_m"].dropna().nunique())}


def main() -> int:
    frame = pd.read_parquet(NOAA)
    detail = pd.DataFrame([audit_group(group) for _, group in frame.groupby(["year", "station_id"], sort=True)])
    detail["eligible_for_extended_validation"] = (detail["duplicate_rows"] == 0) & (detail["nominal_coverage"] >= 0.8) & (detail["temperature_missing_rate"] < 0.2) & (detail["latitude_nunique"] <= 1) & (detail["longitude_nunique"] <= 1)
    years = sorted(int(x) for x in frame["year"].unique())
    stations = sorted(frame["station_id"].astype(str).unique())
    checks = {"no_duplicate_timestamps": bool((detail["duplicate_rows"] == 0).all()), "temperature_missing_below_0p2": bool((detail["temperature_missing_rate"] < 0.2).all()), "coordinates_stable": bool(((detail["latitude_nunique"] <= 1) & (detail["longitude_nunique"] <= 1)).all()), "all_expected_years_present": bool(set(years) == set(range(min(years), max(years) + 1))), "each_station_has_all_years": bool(frame.groupby("station_id")["year"].nunique().eq(len(years)).all())}
    excluded = detail.loc[~detail["eligible_for_extended_validation"]].to_dict(orient="records")
    yearly = detail.groupby("year", sort=True).agg(stations=("station_id", "nunique"), rows=("rows", "sum"), eligible=("eligible_for_extended_validation", "sum"), min_nominal_coverage=("nominal_coverage", "min"), max_temperature_missing_rate=("temperature_missing_rate", "max"), duplicate_rows=("duplicate_rows", "sum")).reset_index()
    status = "PASS" if all(checks.values()) and not excluded else ("PASS_WITH_EXCLUSIONS" if all(checks.values()) else "REVIEW_REQUIRED")
    payload = {"status": status, "protocol": "对阶段82扩展表逐年逐站审计重复时间戳、名义覆盖、温度缺失、质量标记、最大间隔和坐标稳定性；不合格站点-年份排除，不修改原有2021-2023审计。", "years": years, "stations": len(stations), "rows": int(len(frame)), "checks": checks, "eligible_station_years": int(detail["eligible_for_extended_validation"].sum()), "excluded_station_years": excluded, "detail": detail.to_dict(orient="records"), "yearly": yearly.to_dict(orient="records")}
    detail.to_csv(TABLES / "stage83_noaa_extended_quality_audit_detail.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(TABLES / "stage83_noaa_extended_quality_audit_yearly.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage83_noaa_extended_quality_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段83 NOAA扩展多年度质量审计", "", "对阶段82的所有年份和站点逐项审计；排除项不进入后续留一年验证。", "", f"- 状态：**{status}**；年份：`{years}`；站点：`{len(stations)}`；可用站点-年份：`{payload['eligible_station_years']}`；排除：`{len(excluded)}`", f"- 检查：`{checks}`", "", "| 年份 | 站点数 | 行数 | 可用站点-年份 | 最小覆盖率 | 最大温度缺失率 | 重复行 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in yearly.to_dict(orient="records"):
        lines.append(f"| {row['year']} | {row['stations']} | {row['rows']} | {row['eligible']} | {row['min_nominal_coverage']:.4f} | {row['max_temperature_missing_rate']:.4f} | {row['duplicate_rows']} |")
    if excluded:
        lines += ["", "## 排除站点-年份", "", "| 年份 | 站点 | 覆盖率 | 最大间隔(h) |", "|---:|---|---:|---:|"]
        for row in excluded:
            lines.append(f"| {row['year']} | {row['station_id']} | {row['nominal_coverage']:.4f} | {row['max_gap_hours']:.1f} |")
    (REPORTS / "stage83_noaa_extended_quality_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "years": years, "stations": len(stations), "eligible_station_years": payload["eligible_station_years"], "excluded": len(excluded), "checks": checks}, ensure_ascii=False))
    return 0 if status in {"PASS", "PASS_WITH_EXCLUSIONS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
