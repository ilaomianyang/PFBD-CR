from __future__ import annotations

"""Stage 303: audit candidate independent provider sources before freezing.

This stage performs a source-level feasibility audit only. It does not
download observations, freeze a station network, fit a detector, calibrate a
threshold, inject an attack, or compute a performance result.

The audit exists because a provider name alone is not sufficient evidence:
the source must be an official, reproducible, machine-accessible historical
hourly archive with a license that permits research reuse. Candidate sources
that require interactive account creation, prohibit automated access, expose
only gridded products, or cannot be reached from the execution environment
must remain blocked rather than being silently promoted.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage303_independent_provider_source_feasibility"
PROTOCOL_ID = "TMQC-INDEPENDENT-PROVIDER-SOURCE-FEASIBILITY-20260921-v1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = [
        {
            "provider_id": "met_norway_frost",
            "provider": "Norwegian Meteorological Institute (MET Norway)",
            "country_or_region": "Norway",
            "official_entry": "https://frost.met.no/",
            "official_api_reference": "https://frost.met.no/swagger.json",
            "license": "Norwegian Licence for Open Government Data (NLOD) 2.0",
            "historical_hourly_capability": (
                "YES: Frost observation and available-time-series endpoints "
                "support time-resolution filtering, including PT1H."
            ),
            "machine_access": (
                "OFFICIAL_API_REQUIRES_CLIENT_ID: anonymous requests return "
                "HTTP 401 Missing authentication token; client ID issuance "
                "requires an email-based account."
            ),
            "bulk_historical_archive": (
                "PARTIAL: API supports paged historical queries; no "
                "unauthenticated frozen station-year CSV archive was found."
            ),
            "reproducibility_status": "BLOCKED_REQUIRES_USER_CREDENTIAL",
            "can_freeze_now": False,
            "can_download_without_interactive_account": False,
            "can_promote_to_new_provider": False,
            "blocking_reason": (
                "The API is official and licensed, but a user-specific "
                "client ID is required. The project must not fabricate or "
                "borrow credentials."
            ),
        },
        {
            "provider_id": "australia_bom",
            "provider": "Australian Bureau of Meteorology (BOM)",
            "country_or_region": "Australia",
            "official_entry": "https://www.bom.gov.au/climate/data/",
            "official_api_reference": (
                "https://www.bom.gov.au/climate/data-services/station-data.shtml"
            ),
            "license": "BOM copyright; registered-user service may apply",
            "historical_hourly_capability": (
                "PARTIAL: station data products exist, but hourly product "
                "coverage must be selected per product."
            ),
            "machine_access": (
                "AUTOMATED_ACCESS_BLOCKED: official response explicitly "
                "states that the website does not support web scraping and "
                "directs automated users to FTP or a registered service."
            ),
            "bulk_historical_archive": (
                "NOT_CONFIRMED_AS_HOURLY: ACORN-SAT is daily; the CDO "
                "endpoint is interactive and not a frozen hourly archive."
            ),
            "reproducibility_status": "BLOCKED_AUTOMATION_POLICY",
            "can_freeze_now": False,
            "can_download_without_interactive_account": False,
            "can_promote_to_new_provider": False,
            "blocking_reason": (
                "The official site prohibits automated scraping. A "
                "registered service or FTP product would need explicit "
                "terms and a separately frozen download contract."
            ),
        },
        {
            "provider_id": "belgium_rmi",
            "provider": "Royal Meteorological Institute of Belgium (RMI)",
            "country_or_region": "Belgium",
            "official_entry": "https://opendata.meteo.be/",
            "official_api_reference": (
                "https://opendata.meteo.be/ (WFS/WCS endpoints exposed "
                "after account registration)"
            ),
            "license": (
                "RMI open data terms; HVD subset CC BY 4.0; gridded "
                "climate data require an academic agreement"
            ),
            "historical_hourly_capability": (
                "PARTIAL: gridded observational products are offered; "
                "station-hour CSV availability is not the primary contract."
            ),
            "machine_access": (
                "ACCOUNT_REQUIRED: registration and academic agreement "
                "acknowledgment are required before download."
            ),
            "bulk_historical_archive": (
                "PARTIAL: WCS/WFS services, not an unauthenticated frozen "
                "station-year hourly archive."
            ),
            "reproducibility_status": "BLOCKED_REQUIRES_ACCOUNT_AND_TERMS",
            "can_freeze_now": False,
            "can_download_without_interactive_account": False,
            "can_promote_to_new_provider": False,
            "blocking_reason": (
                "The service requires an account and agreement acceptance; "
                "gridded products are not equivalent to the frozen station "
                "hourly contract used by this project."
            ),
        },
        {
            "provider_id": "korea_kma",
            "provider": "Korea Meteorological Administration (KMA)",
            "country_or_region": "Republic of Korea",
            "official_entry": "https://data.kma.go.kr/",
            "official_api_reference": (
                "https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36"
            ),
            "license": "KOGL Type 1 (source attribution) on the portal",
            "historical_hourly_capability": (
                "YES: ASOS hourly elements and CSV/Excel export are exposed."
            ),
            "machine_access": (
                "SESSION_AND_LOGIN_BOUND: the download flow is tied to a "
                "web session and member login; no documented unauthenticated "
                "station-year API contract was confirmed."
            ),
            "bulk_historical_archive": (
                "PARTIAL: interactive export and FTP-style flows exist, but "
                "the exact reproducible contract was not established."
            ),
            "reproducibility_status": "BLOCKED_REQUIRES_SESSION_OR_ACCOUNT",
            "can_freeze_now": False,
            "can_download_without_interactive_account": False,
            "can_promote_to_new_provider": False,
            "blocking_reason": (
                "A reproducible download contract could not be established "
                "without a session or account, so the source cannot be "
                "used as an independent locked provider yet."
            ),
        },
        {
            "provider_id": "singapore_mss",
            "provider": "Meteorological Service Singapore (MSS)",
            "country_or_region": "Singapore",
            "official_entry": "https://api.data.gov.sg/",
            "official_api_reference": (
                "https://api.data.gov.sg/v1/environment/air-temperature"
            ),
            "license": "Singapore Open Data Licence",
            "historical_hourly_capability": (
                "PARTIAL: real-time/nearest-hour observation endpoints are "
                "documented; long historical station archives are not the "
                "primary contract."
            ),
            "machine_access": (
                "AUTHENTICATION_REQUIRED: anonymous request returned HTTP "
                "403 MissingAuthenticationTokenException."
            ),
            "bulk_historical_archive": (
                "NOT_CONFIRMED: the accessible contract is not a frozen "
                "historical station-year hourly archive."
            ),
            "reproducibility_status": "BLOCKED_AUTH_AND_HISTORICAL_SCOPE",
            "can_freeze_now": False,
            "can_download_without_interactive_account": False,
            "can_promote_to_new_provider": False,
            "blocking_reason": (
                "Authentication is required and the confirmed endpoint is "
                "not a long historical hourly station archive."
            ),
        },
        {
            "provider_id": "hong_kong_hko",
            "provider": "Hong Kong Observatory (HKO)",
            "country_or_region": "Hong Kong SAR",
            "official_entry": "https://www.hko.gov.hk/en/cis/climahk.htm",
            "official_api_reference": (
                "https://data.weather.gov.hk/weatherAPI/doc/"
                "HKO_Open_Data_API_Documentation.pdf"
            ),
            "license": "Hong Kong Observatory open data terms",
            "historical_hourly_capability": (
                "PARTIAL: hourly observations are available, but the "
                "confirmed public contract is centred on one principal "
                "station rather than a multi-station network."
            ),
            "machine_access": (
                "OFFICIAL_API_REACHABLE: the data API and documentation "
                "endpoint are reachable without an account."
            ),
            "bulk_historical_archive": (
                "NOT_CONFIRMED_AS_MULTI_STATION: single-station hourly "
                "history is not sufficient for the multi-station locked "
                "network contract."
            ),
            "reproducibility_status": "BLOCKED_SINGLE_STATION_SCOPE",
            "can_freeze_now": False,
            "can_download_without_interactive_account": True,
            "can_promote_to_new_provider": False,
            "blocking_reason": (
                "The confirmed public hourly contract is not a "
                "multi-station network comparable to the frozen protocol."
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    if frame["provider_id"].duplicated().any():
        raise RuntimeError("duplicate provider id")
    if frame["can_promote_to_new_provider"].any():
        raise RuntimeError(
            "Stage303 must not promote a provider without a frozen "
            "download and evaluation contract"
        )
    blocked = frame[
        frame["reproducibility_status"].str.startswith("BLOCKED")
    ]
    payload = {
        "status": "PASS_READ_ONLY_PROVIDER_FEASIBILITY_AUDIT",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "阶段303只审计未使用官方机构候选是否具备可重复、可合法批量"
            "获取历史小时观测的条件；不下载观测、不冻结站点、不训练模型、"
            "不校准阈值、不注入攻击、不计算性能。"
        ),
        "candidate_count": int(len(frame)),
        "blocked_candidate_count": int(len(blocked)),
        "promotable_candidate_count": int(
            frame["can_promote_to_new_provider"].sum()
        ),
        "requires_account_count": int(
            frame["machine_access"].str.contains(
                "ACCOUNT|CLIENT_ID|LOGIN|AUTHENTICATION",
                case=False,
                regex=True,
            ).sum()
        ),
        "automation_blocked_count": int(
            frame["machine_access"].str.contains(
                "AUTOMATED_ACCESS_BLOCKED",
                case=False,
                regex=False,
            ).sum()
        ),
        "providers": rows,
        "conclusion": (
            "本轮没有发现可直接升级为第七个独立机构的源。MET Norway "
            "Frost 的许可和小时能力最接近要求，但必须先由用户以真实"
            "邮箱申请 client ID；BOM 明确禁止自动抓取；KMA 需要会话/账号；"
            "新加坡需要认证；比利时 RMI 需要学术账号与协议确认；HKO "
            "公开小时合同主要是单站，不能替代多站网络。"
        ),
        "promotion_boundary": (
            "该审计不能关闭 P0-1、P0-5 或 P0-6，也不能作为新机构性能"
            "证据。只有拿到可重复下载凭据或官方冻结归档，并先完成"
            "protocol/data/analysis 冻结后，才能进入新的锁定迁移评价。"
        ),
        "forbidden_claims": [
            "A reachable official website is evidence of provider independence.",
            "A candidate source may be promoted without a frozen download contract.",
            "A gridded product is equivalent to a frozen station-hour network.",
            "An account-gated source may be described as publicly reproducible.",
        ],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    table_path = TABLES / f"{PREFIX}.csv"
    frame.to_csv(table_path, index=False, encoding="utf-8-sig")
    payload["source_table_sha256"] = sha256_file(table_path)
    json_path = TABLES / f"{PREFIX}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段303 独立机构数据源可行性审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 候选数：`{payload['candidate_count']}`",
        f"- 阻断候选：`{payload['blocked_candidate_count']}`",
        f"- 可直接升级候选：`{payload['promotable_candidate_count']}`",
        "",
        "## 结论",
        "",
        payload["conclusion"],
        "",
        "## 边界",
        "",
        f"- {payload['promotion_boundary']}",
        "- 本阶段不下载观测、不冻结站点、不训练、不调阈值、不计算性能。",
        "- 官方页面可达不等于可重复批量下载，也不等于新机构性能证据。",
        "",
        "## 候选摘要",
        "",
        "| 机构 | 地区 | 可重复状态 | 小时能力 | 可否直接升级 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['provider']} | {row['country_or_region']} | "
            f"`{row['reproducibility_status']}` | "
            f"{row['historical_hourly_capability']} | "
            f"`{row['can_promote_to_new_provider']}` |"
        )
    report_path = REPORTS / f"{PREFIX}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "candidate_count": payload["candidate_count"],
                "blocked_candidate_count": payload["blocked_candidate_count"],
                "promotable_candidate_count": payload[
                    "promotable_candidate_count"
                ],
                "json_sha256": sha256_file(json_path),
                "source_table_sha256": payload["source_table_sha256"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
