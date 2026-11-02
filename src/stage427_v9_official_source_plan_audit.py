from __future__ import annotations

"""Stage 427: audit the official-source plan for the V9 October window.

This is metadata-only.  It reads frozen protocol metadata and source-code
constants, but never requests an HTTP body and never reads observation rows.
It exists to make the future data-access step reproducible and provider
specific rather than relying on a generic CSV convention.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
OUT_JSON = ROOT / "tables" / "stage427_v9_official_source_plan_audit.json"
OUT_REPORT = ROOT / "reports" / "stage427_v9_official_source_plan_audit.md"

SOURCES = {
    "MeteoSwiss": {
        "script": "stage215_swiss_official_download.py",
        "url_template": (
            "https://data.geo.admin.ch/api/stac/v0.9/collections/"
            "ch.meteoschweiz.ogd-smn/items/{station_id}"
        ),
        "asset_rule": "hourly h_historical_2020-2029.csv STAC asset",
        "station_field": "station_abbr",
        "timestamp_field": "reference_timestamp",
        "temperature_field": "tre200h0",
        "station_count": 12,
    },
    "NOAA ISD": {
        "script": "stage376_noaa_2026_download_audit.py",
        "url_template": (
            "https://www.ncei.noaa.gov/data/global-hourly/access/2026/"
            "{station_id}.csv"
        ),
        "asset_rule": "official 2026 station CSV",
        "station_field": "station_id",
        "timestamp_field": "timestamp_utc",
        "temperature_field": "temperature_c",
        "station_count": 11,
    },
    "KNMI": {
        "script": "stage225_knmi_official_download.py",
        "url_template": (
            "https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/"
            "uurgegevens/uurgeg_{station_id}_2021-2030.zip"
        ),
        "asset_rule": "official 2021-2030 hourly ZIP; filter UTC timestamps",
        "station_field": "STN",
        "timestamp_field": "YYYYMMDD+HH",
        "temperature_field": "T",
        "station_count": 12,
    },
    "ECCC": {
        "script": "stage235_eccc_official_download.py",
        "url_template": (
            "https://api.weather.gc.ca/collections/climate-hourly/items"
            "?f=csv&STN_ID={station_id}&datetime="
            "2026-10-01T00%3A00%3A00Z%2F2026-10-31T23%3A00%3A00Z"
            "&limit=10000&offset={offset}"
        ),
        "asset_rule": "official GeoMet climate-hourly paginated CSV",
        "station_field": "STN_ID",
        "timestamp_field": "UTC_DATE",
        "temperature_field": "TEMP",
        "station_count": 16,
    },
}


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []
    for provider, source in SOURCES.items():
        path = SRC / source["script"]
        provider_protocol = ROOT / protocol["providers"][provider]["protocol"]
        provider_meta = json.loads(provider_protocol.read_text(encoding="utf-8"))
        frozen_station_count = int(
            provider_meta.get("station_count", len(provider_meta.get("station_ids", [])))
        )
        checks.append(
            {
                "provider": provider,
                "script_present": path.exists(),
                "url_template_present": bool(source["url_template"]),
                "frozen_station_count": frozen_station_count,
                "expected_station_count": source["station_count"],
                "station_count_matches": frozen_station_count
                == source["station_count"],
                "required_fields": {
                    "station": source["station_field"],
                    "timestamp": source["timestamp_field"],
                    "temperature": source["temperature_field"],
                },
            }
        )
    ok = all(
        row["script_present"]
        and row["url_template_present"]
        and row["station_count_matches"]
        for row in checks
    )
    payload = {
        "stage": 427,
        "status": "PASS_V9_OFFICIAL_SOURCE_PLAN_AUDIT" if ok else "FAIL_V9_OFFICIAL_SOURCE_PLAN_AUDIT",
        "observation_values_read": False,
        "http_response_bodies_read": False,
        "performance_evaluated": False,
        "test_labels_read": False,
        "window": {
            "test_start_utc": "2026-10-01T00:00:00Z",
            "test_end_utc": "2026-10-31T23:00:00Z",
        },
        "providers": checks,
        "rules": [
            "只使用冻结四 provider 和既定站点顺序。",
            "下载后必须用 UTC 双边界过滤 2026-10-01 至 2026-10-31。",
            "任一 provider 缺失时保持 NOT_READY，不得用旧年份替代。",
            "本计划不读取 HTTP 响应正文，不产生性能结果。",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage427 V9 官方来源计划审计",
                "",
                f"- 状态：`{payload['status']}`",
                "- 仅审计官方 URL 模板、脚本存在性和字段契约。",
                "- 未读取 HTTP 响应正文、观测值、攻击标签或 test。",
                "- 该阶段不产生检测性能结果。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": 427, "status": payload["status"]}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
