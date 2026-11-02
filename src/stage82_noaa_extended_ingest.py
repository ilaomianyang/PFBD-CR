from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_DIR = ROOT.parents[1] / "dataset" / "external" / "noaa_isd"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage70_noaa_multiyear_ingest import ingest_file  # noqa: E402


def discover() -> list[tuple[Path, int]]:
    files: list[tuple[Path, int]] = []
    files.extend((p, 2022) for p in sorted(DATA_DIR.glob("*99999.csv")))
    for year_dir in sorted(DATA_DIR.glob("multiyear/[0-9][0-9][0-9][0-9]")):
        year = int(year_dir.name)
        if year == 2022:
            continue
        files.extend((p, year) for p in sorted(year_dir.glob("*99999.csv")))
    return files


def main() -> int:
    files = discover()
    if not files:
        raise FileNotFoundError("NOAA extended files not found")
    frames = []
    file_rows = []
    for path, year in files:
        frame = ingest_file(path, year)
        frames.append(frame)
        file_rows.append({"year": year, "file": str(path), "rows": int(len(frame)), "station_ids": sorted(frame["station_id"].astype(str).unique().tolist()), "date_min": str(frame["timestamp_utc"].min()), "date_max": str(frame["timestamp_utc"].max())})
    data = pd.concat(frames, ignore_index=True).sort_values(["station_id", "timestamp_utc"], kind="stable").reset_index(drop=True)
    duplicate_rows = int(data.duplicated(["station_id", "timestamp_utc"]).sum())
    years = sorted(int(x) for x in data["year"].dropna().unique())
    stations = sorted(data["station_id"].astype(str).unique())
    station_year = data.groupby(["station_id", "year"]).size().reset_index(name="rows")
    expected = len(stations) * len(years)
    status = "PASS" if duplicate_rows == 0 and len(files) == expected and len(station_year) == expected else "REVIEW_REQUIRED"
    payload = {"status": status, "protocol": "独立接入NOAA/NCEI官方Global Hourly文件；保留2022原始文件不覆盖；扩展表覆盖本地已下载的所有完整年度目录。", "files": file_rows, "file_count": len(files), "rows": int(len(data)), "stations": len(stations), "years": years, "duplicate_station_timestamp_rows": duplicate_rows, "station_year_pairs": int(len(station_year)), "expected_station_year_pairs": expected, "station_year_counts": station_year.to_dict(orient="records")}
    out = TABLES / "stage82_noaa_extended_normalized.parquet"
    data.to_parquet(out, index=False)
    (TABLES / "stage82_noaa_extended_ingest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(file_rows).to_csv(TABLES / "stage82_noaa_extended_file_audit.csv", index=False, encoding="utf-8-sig")
    station_year.to_csv(TABLES / "stage82_noaa_extended_station_year_counts.csv", index=False, encoding="utf-8-sig")
    lines = ["# 阶段82 NOAA扩展多年度标准化接入", "", "独立接入本地已下载的 NOAA/NCEI 官方 Global Hourly 文件；不覆盖阶段70旧表和2022锁定数据。", "", f"- 状态：**{status}**；文件：`{len(files)}`；站点：`{len(stations)}`；年份：`{years}`；总行数：`{len(data)}`；重复站点-时间键：`{duplicate_rows}`", "", "| 年份 | 站点-年份对数 | 总行数 |", "|---:|---:|---:|"]
    for year, part in data.groupby("year", sort=True):
        lines.append(f"| {year} | {part['station_id'].nunique()} | {len(part)} |")
    lines += ["", f"扩展标准化表：`{out}`", "阶段83必须在此表上完成逐年逐站质量审计后，才能进入留一年性能验证。"]
    (REPORTS / "stage82_noaa_extended_ingest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "file_count": len(files), "stations": len(stations), "years": years, "rows": len(data), "duplicate_rows": duplicate_rows}, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
