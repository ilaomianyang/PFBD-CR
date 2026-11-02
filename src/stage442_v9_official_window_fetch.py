from __future__ import annotations

"""Fetch the frozen V9 official October-2026 source files after window close.

The command is intentionally conservative:

* ``--dry-run`` never performs network I/O.
* Real downloading is refused before 2026-11-01T00:00Z, so a partial
  October window cannot be mistaken for the locked test.
* Only the four frozen providers and frozen station IDs are used.
* Raw files are archived with URL, byte count and SHA256; normalization and
  evaluation remain separate stages.
"""

import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
RAW = PACKAGE / "raw_official_2026_10"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
OUT_JSON = TABLES / "stage442_v9_official_window_fetch.json"
OUT_REPORT = REPORTS / "stage442_v9_official_window_fetch.md"
DRY_RUN_JSON = TABLES / "stage442_v9_official_window_fetch_dry_run.json"
DRY_RUN_REPORT = REPORTS / "stage442_v9_official_window_fetch_dry_run.md"

START = "2026-10-01T00:00:00Z"
END = "2026-10-31T23:00:00Z"
WINDOW_CLOSE = datetime(2026, 11, 1, tzinfo=timezone.utc)
USER_AGENT = "TrustMeteo-QC/0.1 (locked V9 official-window fetch)"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str, timeout: int = 240) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"empty official response: {url}")
    return payload


def frozen_station_ids(protocol: dict, provider: str) -> list[str]:
    provider_protocol = ROOT / str(protocol["providers"][provider]["protocol"])
    meta = json.loads(provider_protocol.read_text(encoding="utf-8"))
    ids = meta.get("station_ids")
    if ids:
        return [str(value) for value in ids]
    if provider == "KNMI":
        table = pd.read_csv(TABLES / "stage225_knmi_stations.csv", dtype=str)
        return table["station_id"].astype(str).tolist()
    raise RuntimeError(f"{provider}: frozen station IDs are unavailable")


def plans(protocol: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for provider in ("MeteoSwiss", "NOAA ISD", "KNMI", "ECCC"):
        for station_id in frozen_station_ids(protocol, provider):
            if provider == "MeteoSwiss":
                url = (
                    "https://data.geo.admin.ch/api/stac/v0.9/collections/"
                    f"ch.meteoschweiz.ogd-smn/items/{station_id}"
                )
                suffix = "csv"
            elif provider == "NOAA ISD":
                url = (
                    "https://www.ncei.noaa.gov/data/global-hourly/access/"
                    f"2026/{station_id}.csv"
                )
                suffix = "csv"
            elif provider == "KNMI":
                url = (
                    "https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/"
                    f"uurgegevens/uurgeg_{station_id}_2021-2030.zip"
                )
                suffix = "zip"
            else:
                query = urllib.parse.urlencode(
                    {
                        "f": "csv",
                        "STN_ID": station_id,
                        "datetime": f"{START}/{END}",
                        "limit": 10000,
                        "offset": 0,
                    }
                )
                url = (
                    "https://api.weather.gc.ca/collections/climate-hourly/items?"
                    + query
                )
                suffix = "csv"
            rows.append(
                {
                    "provider": provider,
                    "station_id": station_id,
                    "url": url,
                    "window_start_utc": START,
                    "window_end_utc": END,
                    "raw_suffix": suffix,
                }
            )
    return rows


def meteoswiss_asset(stac_payload: dict) -> str:
    assets = stac_payload.get("assets", {})
    matches = [
        value.get("href")
        for key, value in assets.items()
        if str(key).endswith("h_historical_2020-2029.csv")
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(f"expected one MeteoSwiss hourly asset, got {matches}")
    return str(matches[0])


def download_provider_row(row: dict[str, object]) -> list[dict[str, object]]:
    provider = str(row["provider"])
    station_id = str(row["station_id"])
    target_dir = RAW / provider.replace(" ", "_").lower()
    target_dir.mkdir(parents=True, exist_ok=True)
    if provider == "ECCC":
        # Keep each official page separately; the V9 adapter accepts multiple
        # CSV pages and the raw hashes remain individually auditable.
        offset = 0
        output: list[dict[str, object]] = []
        while True:
            query = urllib.parse.urlencode(
                {
                    "f": "csv",
                    "STN_ID": station_id,
                    "datetime": f"{START}/{END}",
                    "limit": 10000,
                    "offset": offset,
                }
            )
            url = (
                "https://api.weather.gc.ca/collections/climate-hourly/items?"
                + query
            )
            payload = fetch(url)
            target = target_dir / f"{station_id}_offset_{offset:08d}.csv"
            target.write_bytes(payload)
            output.append(
                {
                    "provider": provider,
                    "station_id": station_id,
                    "offset": offset,
                    "url": url,
                    "path": str(target),
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            )
            rows = pd.read_csv(io.BytesIO(payload), dtype=str)
            if len(rows) < 10000:
                break
            offset += 10000
        return output

    if provider == "MeteoSwiss":
        stac = json.loads(fetch(str(row["url"])).decode("utf-8"))
        source_url = meteoswiss_asset(stac)
        payload = fetch(source_url)
        target = target_dir / f"{station_id}_2026_10.csv"
    else:
        source_url = str(row["url"])
        payload = fetch(source_url)
        target = target_dir / f"{station_id}_2026_10.{row['raw_suffix']}"
    target.write_bytes(payload)
    return [
        {
            "provider": provider,
            "station_id": station_id,
            "url": source_url,
            "path": str(target),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "CONFIRMATORY_PROTOCOL_FROZEN_PENDING_NEW_OFFICIAL_WINDOW":
        raise RuntimeError("V9 protocol is not in the frozen pending state")
    planned = plans(protocol)
    now = datetime.now(timezone.utc)
    if args.dry_run:
        status = "PASS_V9_OFFICIAL_FETCH_DRY_RUN"
        rows = planned
        message = "未联网；仅生成四 provider 冻结站点下载计划。"
    elif now < WINDOW_CLOSE:
        status = "BLOCKED_V9_FETCH_BEFORE_WINDOW_CLOSE"
        rows = []
        message = (
            "当前时间早于 2026-11-01T00:00:00Z，拒绝下载，避免使用未闭合的 "
            "2026-10 时间窗作为锁定 test。"
        )
    else:
        rows = []
        failures: list[str] = []
        for row in planned:
            try:
                rows.extend(download_provider_row(row))
            except Exception as exc:
                failures.append(
                    f"{row['provider']}/{row['station_id']}: {type(exc).__name__}: {exc}"
                )
        status = (
            "PASS_V9_OFFICIAL_FETCH_COMPLETED"
            if not failures
            else "FAIL_V9_OFFICIAL_FETCH_PARTIAL"
        )
        message = (
            "官方 raw 文件已归档，下一步必须运行 V9 模板规范化和数据门禁。"
            if not failures
            else f"下载失败数：{len(failures)}"
        )
    payload = {
        "stage": 442,
        "status": status,
        "protocol_id": protocol["protocol_id"],
        "dry_run": bool(args.dry_run),
        "planned_station_count": len(planned),
        "planned_file_count": len(planned) if args.dry_run else None,
        "archived_file_count": 0 if args.dry_run else len(rows),
        "window_close_utc": WINDOW_CLOSE.isoformat().replace("+00:00", "Z"),
        "rows": rows,
        "message": message,
        "observation_values_read": bool(rows) and not args.dry_run,
        "performance_evaluated": False,
        "historical_substitution": False,
        "claim_boundary": [
            "本阶段只归档冻结官方 raw 文件，不做规范化、攻击生成或性能评价。",
            "在窗口结束前运行真实下载会被硬拒绝。",
        ],
    }
    output_json = DRY_RUN_JSON if args.dry_run else OUT_JSON
    output_report = DRY_RUN_REPORT if args.dry_run else OUT_REPORT
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_report.write_text(
        "\n".join(
            [
                "# Stage442 V9 官方窗口采集器",
                "",
                f"- 状态：`{status}`",
                f"- dry-run：`{bool(args.dry_run)}`",
                f"- 冻结站点计划：`{len(planned)}`",
                f"- 计划文件数：`{len(planned) if args.dry_run else '不适用'}`",
                f"- 已归档文件：`{0 if args.dry_run else len(rows)}`",
                f"- {message}",
                "- 本阶段不产生性能结果。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": 442,
                "status": status,
                "planned_station_count": len(planned),
                "planned_file_count": len(planned) if args.dry_run else None,
                "archived_file_count": 0 if args.dry_run else len(rows),
            },
            ensure_ascii=False,
        )
    )
    return 0 if status.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
