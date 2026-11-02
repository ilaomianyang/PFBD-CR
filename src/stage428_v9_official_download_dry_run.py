from __future__ import annotations

"""Stage 428: provider-specific V9 download dry-run.

Only frozen protocol metadata and station metadata are read.  No network
request is made and no observation body is opened.  The output is an
auditable URL/file plan for the 2026-10 window.
"""

import json
import urllib.parse
from pathlib import Path

import pandas as pd

from stage427_v9_official_source_plan_audit import SOURCES


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
OUT_JSON = TABLES / "stage428_v9_official_download_dry_run.json"
OUT_DETAIL = TABLES / "stage428_v9_official_download_dry_run.csv"
OUT_REPORT = ROOT / "reports" / "stage428_v9_official_download_dry_run.md"


def stations_for(provider: str, protocol: dict) -> list[str]:
    # The provider protocol field is a path, not a station list.
    protocol_path = ROOT / protocol["providers"][provider]["protocol"]
    meta = json.loads(protocol_path.read_text(encoding="utf-8"))
    ids = meta.get("station_ids")
    if ids:
        return [str(x) for x in ids]
    if provider == "KNMI":
        station_table = TABLES / "stage225_knmi_stations.csv"
        frame = pd.read_csv(station_table, dtype={"station_id": str})
        return frame["station_id"].astype(str).tolist()
    raise RuntimeError(f"no frozen station metadata for {provider}")


def build_url(provider: str, station_id: str) -> str:
    if provider == "MeteoSwiss":
        return SOURCES[provider]["url_template"].format(station_id=station_id)
    if provider == "NOAA ISD":
        return SOURCES[provider]["url_template"].format(station_id=station_id)
    if provider == "KNMI":
        return SOURCES[provider]["url_template"].format(station_id=station_id)
    if provider == "ECCC":
        query = urllib.parse.urlencode(
            {
                "f": "csv",
                "STN_ID": station_id,
                "datetime": "2026-10-01T00:00:00Z/2026-10-31T23:00:00Z",
                "limit": 10000,
                "offset": 0,
            }
        )
        return (
            "https://api.weather.gc.ca/collections/climate-hourly/items?"
            + query
        )
    raise KeyError(provider)


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    provider_summary: list[dict[str, object]] = []
    for provider in SOURCES:
        station_ids = stations_for(provider, protocol)
        expected = int(
            json.loads(
                (ROOT / protocol["providers"][provider]["protocol"]).read_text(
                    encoding="utf-8"
                )
            ).get("station_count", len(station_ids))
        )
        if len(station_ids) != expected:
            raise RuntimeError(
                f"{provider}: frozen station count mismatch {len(station_ids)} != {expected}"
            )
        for station_id in station_ids:
            rows.append(
                {
                    "provider": provider,
                    "station_id": station_id,
                    "url": build_url(provider, station_id),
                    "window_start_utc": "2026-10-01T00:00:00Z",
                    "window_end_utc": "2026-10-31T23:00:00Z",
                    "request_made": False,
                    "response_body_read": False,
                    "expected_raw_directory": str(
                        PACKAGE
                        / "raw_official_2026_10"
                        / provider.replace(" ", "_").lower()
                    ),
                    "raw_naming_contract": (
                        f"{station_id}_offset_*.csv"
                        if provider == "ECCC"
                        else f"{station_id}_2026_10."
                        + ("zip" if provider == "KNMI" else "csv")
                    ),
                }
            )
        provider_summary.append(
            {
                "provider": provider,
                "station_count": len(station_ids),
                "url_count": len(station_ids),
                "request_made": False,
                "response_body_read": False,
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
    payload = {
        "stage": 428,
        "status": "PASS_V9_OFFICIAL_DOWNLOAD_DRY_RUN",
        "protocol_id": protocol["protocol_id"],
        "window": {
            "test_start_utc": "2026-10-01T00:00:00Z",
            "test_end_utc": "2026-10-31T23:00:00Z",
        },
        "provider_count": len(provider_summary),
        "station_count": int(len(detail)),
        "provider_summary": provider_summary,
        "request_made": False,
        "response_body_read": False,
        "observation_values_read": False,
        "performance_evaluated": False,
        "claim_boundary": [
            "这是下载计划 dry-run，不是数据下载，不是性能结果。",
            "官方文件缺失时不得用旧年份或旧 test 替代。",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage428 V9 官方下载 dry-run",
                "",
                f"- 状态：`{payload['status']}`",
                f"- provider：`{payload['provider_count']}`；站点/URL：`{payload['station_count']}`",
                "- 未发起网络请求，未读取响应正文，未读取观测值。",
                "- 该阶段不产生检测性能结果。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": 428,
                "status": payload["status"],
                "provider_count": payload["provider_count"],
                "station_count": payload["station_count"],
                "request_made": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
