from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_DIR = ROOT.parents[1] / "dataset" / "external" / "noaa_isd"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage06_external_ingest import parse_noaa_component  # noqa: E402


def ingest_file(path: Path, year: int) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, low_memory=False)
    required = {"STATION", "DATE", "TMP", "WND", "REPORT_TYPE", "QUALITY_CONTROL"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    out = pd.DataFrame({
        "source": "NOAA_ISD",
        "year": int(year),
        "station_id": raw["STATION"].astype(str),
        "station_name": raw.get("NAME", pd.Series("", index=raw.index)).astype(str),
        "timestamp_utc": pd.to_datetime(raw["DATE"], errors="coerce", utc=True),
        "latitude": pd.to_numeric(raw.get("LATITUDE"), errors="coerce"),
        "longitude": pd.to_numeric(raw.get("LONGITUDE"), errors="coerce"),
        "elevation_m": pd.to_numeric(raw.get("ELEVATION"), errors="coerce"),
        "temperature_c": raw["TMP"].map(lambda x: parse_noaa_component(x, 0, 10.0)),
        "temperature_quality": raw["TMP"].map(lambda x: str(x).split(",")[1] if "," in str(x) else ""),
        "dewpoint_c": raw.get("DEW", pd.Series(np.nan, index=raw.index)).map(lambda x: parse_noaa_component(x, 0, 10.0)),
        "pressure_hpa": raw.get("SLP", pd.Series(np.nan, index=raw.index)).map(lambda x: parse_noaa_component(x, 0, 10.0)),
        "wind_direction_deg": raw["WND"].map(lambda x: parse_noaa_component(x, 0, 1.0)),
        "wind_speed_ms": raw["WND"].map(lambda x: parse_noaa_component(x, 3, 10.0)),
        "visibility_m": raw.get("VIS", pd.Series(np.nan, index=raw.index)).map(lambda x: parse_noaa_component(x, 0, 10.0)),
        "report_type": raw["REPORT_TYPE"].astype(str),
        "quality_control": raw["QUALITY_CONTROL"].astype(str),
    })
    out.loc[~out["wind_direction_deg"].between(0, 360), "wind_direction_deg"] = np.nan
    return out.dropna(subset=["timestamp_utc"]).sort_values(["station_id", "timestamp_utc"], kind="stable")


def main() -> int:
    files: list[tuple[Path, int]] = []
    files.extend((p, 2022) for p in sorted(DATA_DIR.glob("*99999.csv")))
    for year_dir in [DATA_DIR / "multiyear" / "2021", DATA_DIR / "multiyear" / "2023"]:
        year = int(year_dir.name)
        files.extend((p, year) for p in sorted(year_dir.glob("*99999.csv")))
    if not files:
        raise FileNotFoundError(f"NOAA multiyear files not found in {DATA_DIR}")
    frames = []
    file_rows = []
    for path, year in files:
        frame = ingest_file(path, year)
        frames.append(frame)
        file_rows.append({"year": year, "file": str(path), "rows": int(len(frame)), "station_ids": sorted(frame["station_id"].astype(str).unique().tolist()), "date_min": str(frame["timestamp_utc"].min()), "date_max": str(frame["timestamp_utc"].max())})
    data = pd.concat(frames, ignore_index=True).sort_values(["station_id", "timestamp_utc"], kind="stable").reset_index(drop=True)
    duplicate_rows = int(data.duplicated(["station_id", "timestamp_utc"]).sum())
    station_years = data.groupby("station_id")["year"].nunique()
    coverage = data.groupby(["year", "station_id"])["timestamp_utc"].agg(["min", "max", "count"]).reset_index()
    required_columns = ["source", "year", "station_id", "station_name", "timestamp_utc", "latitude", "longitude", "elevation_m", "temperature_c", "temperature_quality", "dewpoint_c", "pressure_hpa", "wind_direction_deg", "wind_speed_ms", "visibility_m", "report_type", "quality_control"]
    payload = {"status": "PASS" if duplicate_rows == 0 and len(files) == 36 and station_years.min() >= 3 else "REVIEW_REQUIRED", "protocol": "递归接入NOAA/NCEI官方2021、2022、2023 Global Hourly文件；保留year字段；不修改2022阶段6基线文件。", "files": file_rows, "file_count": len(files), "rows": int(len(data)), "stations": int(data["station_id"].nunique()), "years": sorted(data["year"].unique().tolist()), "duplicate_station_timestamp_rows": duplicate_rows, "required_columns": required_columns, "missing_required_columns": sorted(set(required_columns).difference(data.columns)), "station_year_counts": {str(k): int(v) for k, v in station_years.to_dict().items()}, "coverage": coverage.to_dict(orient="records")}
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = TABLES / "stage70_noaa_isd_multiyear_normalized.parquet"
    data.to_parquet(out_path, index=False)
    (TABLES / "stage70_noaa_multiyear_ingest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(file_rows).to_csv(TABLES / "stage70_noaa_multiyear_file_audit.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(TABLES / "stage70_noaa_multiyear_coverage.csv", index=False, encoding="utf-8-sig")
    lines = ["# 阶段70 NOAA多年度接入与数据审计", "", "独立接入2021/2022/2023 NOAA官方Global Hourly文件；原2022阶段6基线文件保持不变。", "", f"- 状态：**{payload['status']}**；文件：`{payload['file_count']}`；站点：`{payload['stations']}`；年份：`{payload['years']}`；总行数：`{payload['rows']}`", f"- 重复站点-时间戳：`{duplicate_rows}`；每站覆盖年份最少：`{min(station_years)}`", "", "| 年份 | 文件数 | 总行数 | 最早时间 | 最晚时间 |", "|---:|---:|---:|---|---|"]
    for year, part in coverage.groupby("year", sort=True):
        lines.append(f"| {year} | {part['station_id'].nunique()} | {int(part['count'].sum())} | {part['min'].min()} | {part['max'].max()} |")
    lines += ["", "多年度文件在通过字段、重复键、站点覆盖和质量审计前，不得进入后续性能实验。", f"标准化数据：`{out_path}`"]
    (REPORTS / "stage70_noaa_multiyear_ingest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "file_count": payload["file_count"], "stations": payload["stations"], "years": payload["years"], "rows": payload["rows"], "duplicate_rows": duplicate_rows}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
