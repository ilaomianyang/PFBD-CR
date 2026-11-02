from __future__ import annotations

"""Stage 306: official MET Norway Frost download and coverage audit.

This stage consumes the frozen Stage304 data protocol and Stage305
evaluation contract.  It downloads only the official Frost
``/observations/v0.jsonld`` endpoint for the frozen station IDs and
reference-time intervals, normalizes the pre-registered air_temperature
values to UTC, and audits split coverage.  It does not fit a detector,
select a threshold, inject an attack, or compute any performance result.
"""

import base64
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "raw" / "met_norway_frost"
SECRET = ROOT / ".secrets" / "met_norway_frost.json"
PREFIX = "stage306_met_norway_official_download"
DATA_PROTOCOL_PREFIX = "stage304_met_norway_protocol_preregistration"
CONTRACT_PREFIX = "stage305_met_norway_locked_evaluation_contract"

OFFICIAL_ROOT = "https://frost.met.no"
OBSERVATIONS_ENDPOINT = f"{OFFICIAL_ROOT}/observations/v0.jsonld"
YEARS = [2021, 2022, 2023, 2024, 2025]
COVERAGE_MIN = 0.70
ACCEPTED_QUALITY_CODE = "0"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_credentials() -> dict[str, str]:
    if not SECRET.is_file():
        raise FileNotFoundError(
            "MET Norway Frost credentials are not available in .secrets"
        )
    payload = json.loads(SECRET.read_text(encoding="utf-8"))
    client_id = str(payload.get("client_id", "")).strip()
    client_secret = str(payload.get("client_secret", "")).strip()
    if not client_id or not client_secret:
        raise RuntimeError("MET Norway Frost credentials are incomplete")
    return {"client_id": client_id, "client_secret": client_secret}


def authorization_header(credentials: dict[str, str]) -> dict[str, str]:
    token = base64.b64encode(
        (
            credentials["client_id"] + ":" + credentials["client_secret"]
        ).encode("ascii")
    ).decode("ascii")
    return {
        "Authorization": f"Basic {token}",
        "User-Agent": "TrustMeteo-QC reproducible research MET Norway audit",
    }


def build_query(
    source_id: str,
    exposure_category: str,
    start_utc: str,
    end_utc: str,
) -> dict[str, str]:
    return {
        "sources": source_id,
        "elements": "air_temperature",
        "referencetime": f"{start_utc}/{end_utc}",
        "timeoffsets": "PT0H",
        "timeresolutions": "PT1H",
        "levels": "2",
        "qualities": ACCEPTED_QUALITY_CODE,
        "performancecategories": "C",
        "exposurecategories": exposure_category,
    }


def download_json(
    query: dict[str, str],
    headers: dict[str, str],
    retries: int = 5,
    timeout: int = 180,
) -> tuple[dict, bytes, dict[str, str]]:
    url = OBSERVATIONS_ENDPOINT + "?" + urllib.parse.urlencode(query)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                response_headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                }
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise RuntimeError(
                    f"unexpected Frost observation response: {url}"
                )
            return parsed, payload, response_headers
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(
                f"Frost observation request failed: {url} "
                f"HTTP {error.code}: {body[:4000]}"
            )
            if error.code not in {429, 500, 502, 503, 504}:
                raise last_error from error
        except Exception as error:
            last_error = error
        if attempt + 1 < retries:
            time.sleep(2.0 + 2.0 * attempt)
    raise RuntimeError(f"Frost observation download failed: {url}") from last_error


def normalize_observations(
    payload: dict,
    station_id: str,
    accepted_quality_code: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in payload.get("data", []):
        reference_time = item.get("referenceTime")
        observations = item.get("observations") or []
        if not reference_time or not observations:
            continue
        timestamp = pd.to_datetime(reference_time, errors="coerce", utc=True)
        if pd.isna(timestamp):
            continue
        for observation in observations:
            value = observation.get("value")
            quality = observation.get("qualityCode")
            rows.append(
                {
                    "station_id": station_id,
                    "timestamp_utc": timestamp,
                    "temperature_c": pd.to_numeric(
                        value,
                        errors="coerce",
                    ),
                    "quality_code": (
                        "" if quality is None else str(quality)
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "timestamp_utc",
                "temperature_c",
                "quality_code",
                "observation_present",
            ]
        )
    frame = frame.dropna(subset=["timestamp_utc"]).copy()
    frame = frame.sort_values("timestamp_utc", kind="stable")
    frame = frame.drop_duplicates("timestamp_utc", keep="last")
    frame.loc[
        frame["quality_code"].astype(str).ne(accepted_quality_code),
        "temperature_c",
    ] = np.nan
    frame["observation_present"] = frame["temperature_c"].notna()
    frame["year"] = frame["timestamp_utc"].dt.year.astype(int)
    return frame.reset_index(drop=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data_protocol = load_json(TABLES / f"{DATA_PROTOCOL_PREFIX}.json")
    evaluation_contract = load_json(
        TABLES / f"{CONTRACT_PREFIX}.json"
    )
    if data_protocol["status"] != "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD":
        raise RuntimeError("Stage304 MET Norway protocol is not frozen")
    if data_protocol["download_status"] != "NOT_DOWNLOADED":
        raise RuntimeError("Stage304 MET Norway download was already consumed")
    if data_protocol["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError(
            "Stage304 MET Norway performance was already evaluated"
        )
    if evaluation_contract["status"] != (
        "EVALUATION_CONTRACT_FROZEN_BEFORE_FULL_DOWNLOAD"
    ):
        raise RuntimeError("Stage305 MET Norway contract is not frozen")
    if evaluation_contract["download_status"] != "NOT_DOWNLOADED":
        raise RuntimeError("Stage305 MET Norway download was already consumed")
    if evaluation_contract["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError(
            "Stage305 MET Norway performance was already evaluated"
        )
    if data_protocol["quality_accepted_codes"] != [0]:
        raise RuntimeError("MET Norway accepted quality-code rule changed")
    if data_protocol["test_period_end_utc"] != "2025-08-24T23:59:59Z":
        raise RuntimeError("MET Norway test-period boundary changed")

    stations = pd.read_csv(
        TABLES / f"{DATA_PROTOCOL_PREFIX}_stations.csv",
        dtype={"station_id": str},
    )
    frozen_station_ids = [
        str(value) for value in data_protocol["station_ids"]
    ]
    if stations["station_id"].astype(str).tolist() != frozen_station_ids:
        raise RuntimeError("MET Norway station table changed")
    source_contract_by_station = {
        str(row["station_id"]): row
        for row in stations.to_dict("records")
    }
    if len(source_contract_by_station) != len(frozen_station_ids):
        raise RuntimeError("MET Norway source-id mapping is incomplete")
    if list(data_protocol["years"]) != YEARS:
        raise RuntimeError("MET Norway years changed")

    credentials = load_credentials()
    headers = authorization_header(credentials)
    RAW.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.now(timezone.utc).isoformat()
    detail_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    for station_id in frozen_station_ids:
        station_contract = source_contract_by_station[station_id]
        source_id = str(station_contract["source_id"])
        exposure_category = str(
            station_contract["exposure_category"]
        )
        for year in YEARS:
            start_utc = f"{int(year)}-01-01T00:00:00Z"
            end_utc = (
                data_protocol["test_period_end_utc"]
                if int(year) == 2025
                else f"{int(year)}-12-31T23:59:59Z"
            )
            query = build_query(
                source_id,
                exposure_category,
                start_utc,
                end_utc,
            )
            url = OBSERVATIONS_ENDPOINT + "?" + urllib.parse.urlencode(query)
            raw_path = RAW / f"{station_id}_{int(year)}.json"
            if raw_path.is_file():
                payload_raw = raw_path.read_bytes()
                payload = json.loads(payload_raw)
                status = "cached"
                response_headers: dict[str, str] = {"source": "cache"}
            else:
                payload, payload_raw, response_headers = download_json(
                    query,
                    headers,
                )
                raw_path.write_bytes(payload_raw)
                status = "downloaded"
            try:
                parsed = normalize_observations(
                    payload,
                    station_id,
                    ACCEPTED_QUALITY_CODE,
                )
                frames.append(parsed)
                detail_rows.append(
                    {
                        "station_id": station_id,
                        "source_id": source_id,
                        "exposure_category": exposure_category,
                        "year": int(year),
                        "url": url,
                        "status": status,
                        "downloaded_at_utc": downloaded_at,
                        "content_type": response_headers.get(
                            "content-type",
                            "",
                        ),
                        "bytes": int(len(payload_raw)),
                        "sha256": sha256_bytes(payload_raw),
                        "rows": int(len(parsed)),
                        "nonmissing_rows": int(
                            parsed["observation_present"].sum()
                        ),
                        "error": "",
                    }
                )
            except Exception as error:
                detail_rows.append(
                    {
                        "station_id": station_id,
                        "source_id": source_id,
                        "exposure_category": exposure_category,
                        "year": int(year),
                        "url": url,
                        "status": "failed",
                        "downloaded_at_utc": downloaded_at,
                        "content_type": response_headers.get(
                            "content-type",
                            "",
                        ),
                        "bytes": int(len(payload_raw)),
                        "sha256": sha256_bytes(payload_raw),
                        "rows": 0,
                        "nonmissing_rows": 0,
                        "error": str(error),
                    }
                )
            print(
                json.dumps(
                    {
                        "station_id": station_id,
                        "year": int(year),
                        "status": detail_rows[-1]["status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    detail = pd.DataFrame(detail_rows)
    if frames:
        long = pd.concat(frames, ignore_index=True)
    else:
        long = pd.DataFrame(
            columns=[
                "station_id",
                "timestamp_utc",
                "temperature_c",
                "quality_code",
                "observation_present",
            ]
        )
    long = long.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).drop_duplicates(
        ["station_id", "timestamp_utc"],
        keep="last",
    )
    long = long[
        long["timestamp_utc"].le(
            pd.Timestamp(data_protocol["test_period_end_utc"])
        )
    ].copy()
    long["split"] = np.select(
        [
            long["year"].between(2021, 2023),
            long["year"].eq(2024),
            long["year"].eq(2025),
        ],
        ["train", "valid", "test"],
        default="outside_protocol",
    )
    long = long[long["split"].ne("outside_protocol")].copy()
    long = long.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    long.to_parquet(
        TABLES / f"{PREFIX}_normalized.parquet",
        index=False,
    )
    detail.to_csv(
        TABLES / f"{PREFIX}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_rows: list[dict[str, object]] = []
    retained_ids: list[str] = []
    for station_id in frozen_station_ids:
        station_frame = long[long["station_id"].astype(str).eq(station_id)]
        row: dict[str, object] = {"station_id": station_id}
        retained = True
        for split in ("train", "valid", "test"):
            split_frame = station_frame[station_frame["split"].eq(split)]
            expected = int(len(split_frame))
            present = int(split_frame["observation_present"].sum())
            coverage = present / expected if expected else 0.0
            row[f"{split}_rows"] = expected
            row[f"{split}_present"] = present
            row[f"{split}_coverage"] = float(coverage)
            if expected == 0 or present == 0 or coverage < COVERAGE_MIN:
                retained = False
        row["retained"] = bool(retained)
        audit_rows.append(row)
        if retained:
            retained_ids.append(station_id)
    station_audit = pd.DataFrame(audit_rows)
    station_audit.to_csv(
        TABLES / f"{PREFIX}_station_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "status": "PASS_MET_NORWAY_DOWNLOAD_AND_COVERAGE_AUDIT",
        "protocol": (
            "阶段306按阶段304/305冻结的12个MET Norway Frost站点、"
            "2021-2025切分和质量码规则，从官方/observations/v0.jsonld"
            "下载并规范化；2025 test截止到2025-08-24T23:59:59Z。"
            "只做覆盖率审计，不训练模型、不校阈值、不注入攻击、"
            "不计算性能。"
        ),
        "data_protocol_sha256": sha256_file(
            TABLES / f"{DATA_PROTOCOL_PREFIX}.json"
        ),
        "evaluation_contract_sha256": sha256_file(
            TABLES / f"{CONTRACT_PREFIX}.json"
        ),
        "station_count": int(len(stations)),
        "expected_requests": int(len(stations) * len(YEARS)),
        "downloaded_count": int(detail["status"].eq("downloaded").sum()),
        "cached_count": int(detail["status"].eq("cached").sum()),
        "failed_count": int(detail["status"].eq("failed").sum()),
        "normalized_rows": int(len(long)),
        "retained_station_count": int(len(retained_ids)),
        "retained_station_ids": retained_ids,
        "station_audit": station_audit.to_dict("records"),
        "normalized_sha256": sha256_file(
            TABLES / f"{PREFIX}_normalized.parquet"
        ),
        "detail_sha256": sha256_file(TABLES / f"{PREFIX}_detail.csv"),
        "coverage_rule": data_protocol["coverage_rule"],
        "quality_boundary": data_protocol["quality_rule"],
        "download_status": "DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "limitations": [
            "本阶段只完成官方下载、规范化与覆盖率审计。",
            "未训练模型、未选择阈值、未注入攻击、未计算性能。",
            "未通过覆盖率规则的站点保留在审计表中并排除。",
            "2025 test为部分年度，截止到2025-08-24，不能声称全年覆盖。",
            "MET Norway是独立机构，但仍只有一个provider网络，不能证明站点无关。",
            "后续攻击仍为透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段306 MET Norway Frost官方下载与覆盖率审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 预期请求：`{payload['expected_requests']}`",
        f"- 下载成功：`{payload['downloaded_count']}`",
        f"- 缓存命中：`{payload['cached_count']}`",
        f"- 下载失败：`{payload['failed_count']}`",
        f"- 规范化行数：`{payload['normalized_rows']}`",
        f"- 保留站点：`{payload['retained_station_count']}`",
        f"- 性能状态：`{payload['performance_status']}`",
        "",
        "## 边界",
        "",
        "- 只接受官方质量码0；质量码1可能表示缺失后插值，按缺失处理。",
        "- 本阶段不训练模型、不选择阈值、不注入攻击、不计算性能。",
        "- 未通过覆盖率规则的站点保留在审计表中并排除。",
        "- 2025 test是部分年度，截止到2025-08-24。",
        "- 独立机构不自动等于站点无关或真实攻击召回率。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "downloaded": payload["downloaded_count"],
                "cached": payload["cached_count"],
                "failed": payload["failed_count"],
                "retained_stations": payload["retained_station_count"],
                "normalized_rows": payload["normalized_rows"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
