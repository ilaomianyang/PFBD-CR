from __future__ import annotations

"""Stage 235: download the pre-registered ECCC hourly observations.

Stage 234 froze the ECCC protocol before any hourly observation values
were inspected for model selection or performance.  This stage reads the
official ECCC GeoMet CSV endpoint for each of the 16 frozen stations,
using offset pagination, and records URLs, byte counts, SHA256 digests,
columns, row counts, station IDs, and UTC timestamp ranges.

It does not train a detector, select a threshold, or compute
performance.
"""

import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
DATA_DIR = ROOT.parents[1] / "dataset" / "external" / "eccc"
PROTOCOL = TABLES / "stage234_eccc_protocol_preregistration.json"
STATIONS = TABLES / "stage234_eccc_protocol_preregistration_stations.csv"
PREFIX = "stage235_eccc_official_download"

USER_AGENT = "TrustMeteo-QC/0.1 (reproducible research; ECCC Open Data)"
BASE_URL = "https://api.weather.gc.ca/collections/climate-hourly/items"
PAGE_LIMIT = 10000
REQUIRED_COLUMNS = {
    "STN_ID",
    "CLIMATE_IDENTIFIER",
    "LOCAL_DATE",
    "UTC_DATE",
    "TEMP",
    "RELATIVE_HUMIDITY",
    "DEW_POINT_TEMP",
    "STATION_PRESSURE",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_url(
    station_id: int, year: int, offset: int, limit: int = PAGE_LIMIT
) -> str:
    params = {
        "f": "csv",
        "STN_ID": station_id,
        "datetime": (
            f"{year}-01-01T00:00:00Z/"
            f"{year}-12-31T23:59:59Z"
        ),
        "limit": limit,
        "offset": offset,
    }
    return BASE_URL + "?" + urllib.parse.urlencode(params)


def download(url: str, retries: int = 4, timeout: int = 240) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if not payload:
                raise ValueError("empty response")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(str(last_error))


def read_csv(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload), dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def audit_station_year(
    station_id: int, year: int
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    pages: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        url = build_url(station_id, year, offset)
        payload = download(url)
        frame = read_csv(payload)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"ECCC station {station_id} year {year} missing columns: "
                f"{sorted(missing)}"
            )
        page_path = (
            DATA_DIR
            / str(station_id)
            / f"{year}_offset_{offset:08d}.csv"
        )
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_bytes(payload)
        frames.append(frame)
        pages.append(
            {
                "station_id": station_id,
                "year": year,
                "offset": offset,
                "url": url,
                "target": str(page_path),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "rows": int(len(frame)),
                "columns": frame.columns.tolist(),
                "required_columns_ok": True,
                "error": None,
            }
        )
        if len(frame) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        if offset > 200000:
            raise RuntimeError(
                f"ECCC station {station_id} year {year} pagination overflow"
            )
    combined = pd.concat(frames, ignore_index=True)
    return pages, combined


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_ECCC_PERFORMANCE":
        raise RuntimeError("Stage234 protocol is not frozen before download")
    station_table = pd.read_csv(STATIONS, dtype={"station_id": int})
    expected_ids = protocol["station_ids"]
    if station_table["station_id"].astype(int).tolist() != expected_ids:
        raise RuntimeError("Stage234 station order changed")
    years = [int(value) for value in protocol["years"]]

    page_rows: list[dict[str, object]] = []
    station_year_rows: list[dict[str, object]] = []
    all_frames: list[pd.DataFrame] = []
    for station_id in expected_ids:
        station_id = int(station_id)
        station_frames: list[pd.DataFrame] = []
        for year in years:
            try:
                pages, frame = audit_station_year(station_id, year)
                page_rows.extend(pages)
                frame["_download_station_id"] = station_id
                frame["_download_year"] = year
                station_frames.append(frame)
                all_frames.append(frame)
                station_year_rows.append(
                    {
                        "station_id": station_id,
                        "year": year,
                        "status": "downloaded",
                        "pages": len(pages),
                        "rows": int(len(frame)),
                        "bytes": int(
                            sum(int(page["bytes"]) for page in pages)
                        ),
                        "error": None,
                    }
                )
                print(
                    json.dumps(
                        {
                            "station_id": station_id,
                            "year": year,
                            "status": "downloaded",
                            "rows": int(len(frame)),
                            "pages": len(pages),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                station_year_rows.append(
                    {
                        "station_id": station_id,
                        "year": year,
                        "status": "failed",
                        "pages": 0,
                        "rows": 0,
                        "bytes": 0,
                        "error": repr(exc),
                    }
                )
                print(
                    json.dumps(
                        {
                            "station_id": station_id,
                            "year": year,
                            "status": "failed",
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if station_frames:
            combined = pd.concat(station_frames, ignore_index=True)
            combined.to_csv(
                DATA_DIR / str(station_id) / "raw_combined.csv",
                index=False,
                encoding="utf-8-sig",
            )

    pages = pd.DataFrame(page_rows)
    station_years = pd.DataFrame(station_year_rows)
    pages.to_csv(
        TABLES / f"{PREFIX}_pages.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_years.to_csv(
        TABLES / f"{PREFIX}_station_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined["_utc_timestamp"] = pd.to_datetime(
            combined["UTC_DATE"], errors="coerce", utc=True
        )
        combined["_station_id"] = pd.to_numeric(
            combined["STN_ID"], errors="coerce"
        )
        combined["_temperature"] = pd.to_numeric(
            combined["TEMP"], errors="coerce"
        )
        combined["_humidity"] = pd.to_numeric(
            combined["RELATIVE_HUMIDITY"], errors="coerce"
        )
        station_audit = (
            combined.groupby("_station_id", sort=True, as_index=False)
            .agg(
                rows=("_station_id", "size"),
                invalid_utc=("_utc_timestamp", lambda s: int(s.isna().sum())),
                timestamp_min=("_utc_timestamp", "min"),
                timestamp_max=("_utc_timestamp", "max"),
                temperature_present=(
                    "_temperature",
                    lambda s: int(s.notna().sum()),
                ),
                humidity_present=(
                    "_humidity",
                    lambda s: int(s.notna().sum()),
                ),
            )
            .rename(columns={"_station_id": "station_id"})
        )
    else:
        station_audit = pd.DataFrame()
    station_audit.to_csv(
        TABLES / f"{PREFIX}_station_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_audit_records = station_audit.copy()
    for column in ["timestamp_min", "timestamp_max"]:
        if column in station_audit_records:
            station_audit_records[column] = station_audit_records[
                column
            ].map(
                lambda value: (
                    None if pd.isna(value) else str(value)
                )
            )

    failed = int(station_years["status"].eq("failed").sum())
    expected_station_years = len(expected_ids) * len(years)
    payload = {
        "status": (
            "PASS_PIPELINE_REVIEW_REQUIRED"
            if failed == 0 and len(station_years) == expected_station_years
            else "REVIEW_REQUIRED"
        ),
        "protocol": (
            "阶段235按阶段234冻结的16个ECCC站，从官方GeoMet"
            "climate-hourly CSV端点按年下载2021-2025小时观测。"
            "GeoMet CSV每页最多10000行，因此使用offset分页；"
            "保存每页URL、文件、字节数、SHA256、列、行数和站年审计。"
            "不训练模型、不选择阈值、不计算性能，不插补缺失。"
        ),
        "protocol_status": protocol["status"],
        "provider": protocol["provider"],
        "dataset": protocol["dataset"],
        "collection_url": protocol["collection_url"],
        "license": protocol["license"],
        "license_url": protocol["license_url"],
        "page_limit": PAGE_LIMIT,
        "station_count": len(expected_ids),
        "station_ids": expected_ids,
        "years": years,
        "expected_station_year_jobs": expected_station_years,
        "completed_station_year_jobs": int(
            station_years["status"].eq("downloaded").sum()
        ),
        "failed_station_year_jobs": failed,
        "page_count": int(len(pages)),
        "rows": int(station_years["rows"].sum()),
        "station_audit": station_audit_records.to_dict(orient="records"),
        "station_year_detail": station_year_rows,
        "limitations": [
            "本阶段只验证官方下载、分页、字段和覆盖，不评价模型性能。",
            "ECCC小时序列可能因站址迁移、仪器变化和报告规则变化而不均质。",
            "空或非数值字段按缺失保留，不插补；缺失不是攻击标签。",
            "所有攻击仍须由后续阶段按冻结协议透明注入。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段235 ECCC官方小时数据下载与审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 来源：{payload['provider']}",
        f"- 官方集合：{payload['collection_url']}",
        f"- 许可：`{payload['license']}`；{payload['license_url']}",
        f"- 站点：`{payload['station_count']}`；年份：`{years}`",
        f"- 站年任务：`{payload['completed_station_year_jobs']}/"
        f"{payload['expected_station_year_jobs']}`；失败：`{failed}`",
        f"- 分页：`{payload['page_count']}`；总行数：`{payload['rows']}`",
        "",
        "| 站号 | 行数 | 无效UTC | UTC最小 | UTC最大 | 温度有效 | 湿度有效 |",
        "|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in station_audit.to_dict(orient="records"):
        lines.append(
            f"| {int(row['station_id'])} | {int(row['rows'])} | "
            f"{int(row['invalid_utc'])} | {row['timestamp_min']} | "
            f"{row['timestamp_max']} | "
            f"{int(row['temperature_present'])} | "
            f"{int(row['humidity_present'])} |"
        )
    lines += [
        "",
        "本阶段只保存官方原始分页和下载审计，不训练检测器、不选择阈值、",
        "不计算攻击召回率；空字段不插补，缺失不冒充攻击标签。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "station_count": payload["station_count"],
                "completed_station_year_jobs": payload[
                    "completed_station_year_jobs"
                ],
                "failed_station_year_jobs": failed,
                "page_count": payload["page_count"],
                "rows": payload["rows"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if payload["status"] == "PASS_PIPELINE_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
