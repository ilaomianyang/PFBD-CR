from __future__ import annotations

"""Stage 418: read-only feasibility audit for a future independent lock.

This stage deliberately does not download, parse, or cache observation values.
It inspects frozen protocol metadata, local filenames, and official HTTP
response status/headers only.  A positive response is not a performance result
and cannot be used to tune a detector.
"""

import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "raw"
OUT_JSON = TABLES / "stage418_unseen_window_feasibility_audit.json"
OUT_REPORT = REPORTS / "stage418_unseen_window_feasibility_audit.md"

FROZEN_PROTOCOLS = {
    "ECCC": TABLES / "stage234_eccc_protocol_preregistration.json",
    "KNMI": TABLES / "stage224_knmi_protocol_preregistration.json",
    "MeteoSwiss": TABLES / "stage214_swiss_protocol_preregistration.json",
    "NOAA_ISD": TABLES / "stage375_noaa_2026_protocol.json",
    "MET_Norway": TABLES / "stage384_met_norway_2026_factor_protocol.json",
    "FMI": TABLES / "stage390_fmi_2026_factor_protocol.json",
}

# These are metadata/catalogue endpoints or a single file-status request.  The
# NOAA URL is intentionally a HEAD request: it records whether an official
# yearly object exists without reading its CSV content.
OFFICIAL_CHECKS = {
    "ECCC": "https://api.weather.gc.ca/collections/climate-hourly",
    "KNMI": "https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens",
    "MeteoSwiss": "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.meteoschweiz.ogd-smn",
    "NOAA_ISD_2026_status": "https://www.ncei.noaa.gov/data/global-hourly/access/2026/47911099999.csv",
    "MET_Norway": "https://frost.met.no/sources/v0.jsonld?types=SensorSystem&country=NO&fields=id",
    "FMI": "https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0&request=GetCapabilities",
}

PROVIDER_RAW_DIRS = {
    "ECCC": ["eccc"],
    "KNMI": ["knmi"],
    "MeteoSwiss": ["swiss", "meteo_swiss"],
    "NOAA_ISD": ["noaa_isd_japan", "noaa_isd_japan_2026"],
    "MET_Norway": ["met_norway_frost", "met_norway_frost_2026_factor", "met_norway_frost_2026_sep2026_factor"],
    "FMI": ["fmi", "fmi_2026_factor", "fmi_sep2026_factor"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = payload.get("split", {})
    years = payload.get("years", [])
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "years": years,
        "split": split,
        "test_period_start_utc": payload.get("test_period_start_utc"),
        "test_period_end_utc": payload.get("test_period_end_utc"),
        "observation_access": payload.get(
            "observation_access",
            payload.get("observations_accessed_before_freeze"),
        ),
        "performance_status": payload.get("performance_status"),
        "provider_boundary": payload.get("provider_boundary"),
    }


def local_year_tokens(names: list[str]) -> list[int]:
    years: set[int] = set()
    for name in names:
        for token in re.findall(r"(?<!\d)(20\d{2})(?!\d)", name):
            years.add(int(token))
    return sorted(years)


def local_inventory() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for provider, directory_names in PROVIDER_RAW_DIRS.items():
        files: list[str] = []
        missing_dirs: list[str] = []
        for directory_name in directory_names:
            directory = RAW / directory_name
            if not directory.is_dir():
                missing_dirs.append(directory_name)
                continue
            # Names only: no observation file is opened.
            files.extend(path.name for path in directory.rglob("*") if path.is_file())
        result[provider] = {
            "directories": directory_names,
            "missing_directories": missing_dirs,
            "file_count": len(files),
            "year_tokens_in_filenames": local_year_tokens(files),
            "files_sample": sorted(files)[:20],
            "observation_values_read": False,
        }
    return result


def head_status(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "TrustMeteo-QC read-only feasibility audit"},
    )
    started = datetime.now(timezone.utc).isoformat()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return {
                "url": url,
                "status_code": int(response.status),
                "content_type": str(response.headers.get("content-type", "")),
                "content_length": str(response.headers.get("content-length", "")),
                "retrieved_at_utc": started,
                "body_read": False,
                "error": "",
            }
    except urllib.error.HTTPError as error:
        return {
            "url": url,
            "status_code": int(error.code),
            "content_type": str(error.headers.get("content-type", "")),
            "content_length": str(error.headers.get("content-length", "")),
            "retrieved_at_utc": started,
            "body_read": False,
            "error": f"HTTP {error.code}",
        }
    except Exception as error:
        return {
            "url": url,
            "status_code": None,
            "content_type": "",
            "content_length": "",
            "retrieved_at_utc": started,
            "body_read": False,
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> int:
    protocols = {
        provider: protocol_summary(path)
        for provider, path in FROZEN_PROTOCOLS.items()
    }
    checks = {name: head_status(url) for name, url in OFFICIAL_CHECKS.items()}
    inventory = local_inventory()

    # A future window is not yet a usable experiment.  A file-status 200 for
    # NOAA 2026 would only make it eligible for a new protocol/data freeze; it
    # never promotes a result or authorizes reading the file in this stage.
    noaa_status = checks["NOAA_ISD_2026_status"]["status_code"]
    noaa_2026_available = noaa_status == 200
    network_unverified = all(
        item["status_code"] is None for item in checks.values()
    )
    eligible_for_new_freeze = bool(noaa_2026_available)
    if network_unverified:
        reason = (
            "当前运行环境无法连接官方端点，未能确认任何新窗口；这不是 "
            "数据缺失或 404 结论，必须在网络恢复后仅重做本只读状态审计。"
        )
        status = "BLOCKED_NETWORK_UNVERIFIED"
    elif noaa_status == 404:
        reason = (
            "NOAA 2026 官方文件明确返回 HTTP 404；其余端点仅证明官方目录状态，"
            "不能证明测试观测覆盖。"
        )
        status = "BLOCKED_NOAA_2026_NOT_PUBLISHED"
    elif not eligible_for_new_freeze:
        reason = (
            "NOAA 2026 文件未返回 HTTP 200，且没有其他端点提供可冻结的未访问"
            "观测窗口；不能进入新协议或性能评价。"
        )
        status = "BLOCKED_NO_UNSEEN_WINDOW_CONFIRMED"
    else:
        reason = (
            "NOAA 2026 官方文件状态可访问，但仍必须重新冻结 protocol/data/"
            "analysis 后才能读取和评价；本阶段没有读取观测值。"
        )
        status = "ELIGIBLE_FOR_NEW_FREEZE_NOT_EVALUATED"

    payload = {
        "status": status,
        "stage": 418,
        "protocol": (
            "只读可行性审计：读取冻结协议、raw 文件名和官方 HTTP 状态/响应头；"
            "不读取观测响应正文，不解析观测值，不训练、不调阈值、不注入攻击。"
        ),
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "current_date_boundary": "2026-09-24",
        "frozen_protocols": protocols,
        "official_status_checks": checks,
        "local_raw_inventory": inventory,
        "noaa_2026_file_status_code": noaa_status,
        "noaa_2026_file_status_200": noaa_2026_available,
        "network_unverified": network_unverified,
        "new_window_confirmed": False,
        "performance_status": "NOT_EVALUATED",
        "observation_values_read": False,
        "eligible_for_new_protocol_freeze": eligible_for_new_freeze,
        "reason": reason,
        "stop_rule": (
            "若需要读取响应正文、解析观测值、复用已查看窗口或根据旧 test "
            "反馈改变方法/阈值，则本阶段立即停止并保留未完成门槛。"
        ),
        "claim_boundary": [
            "HTTP 状态或目录可访问不等于观测覆盖合格。",
            "本审计没有任何性能结果。",
            "所有后续攻击仍只能是透明受控模拟攻击。",
            "不得把该阶段标记为全面领先或确认性验证。",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage418 未访问时间窗可行性只读审计",
        "",
        f"- 状态：`{status}`",
        f"- 审计时间：`{payload['audit_time_utc']}`",
        "- 读取范围：冻结协议、raw 文件名、官方 HTTP 响应状态和响应头。",
        "- 观测响应正文：未读取；训练、阈值校准、攻击注入和性能评价：未执行。",
        "",
        "## 官方状态",
        "",
        "| 来源 | HTTP 状态 | 是否读取正文 | 说明 |",
        "|---|---:|---|---|",
    ]
    for name, item in checks.items():
        lines.append(
            f"| {name} | {item['status_code'] if item['status_code'] is not None else 'ERROR'} "
            f"| {'否' if not item['body_read'] else '是'} | {item['error'] or '仅状态/响应头'} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        f"- {reason}",
        "- 旧 test、已查看的 2026 时间窗和历史 raw 文件不能重新升级为确认性结果。",
        "- 若未来出现新的可下载窗口，必须先单独冻结 protocol/data/analysis，再进入下载和评价。",
        "",
        "## 产物边界",
        "",
        f"- JSON：`{OUT_JSON}`",
        f"- 报告：`{OUT_REPORT}`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "noaa_2026_file_status": checks["NOAA_ISD_2026_status"]["status_code"],
                "eligible_for_new_freeze": eligible_for_new_freeze,
                "observation_values_read": False,
                "performance_status": "NOT_EVALUATED",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
