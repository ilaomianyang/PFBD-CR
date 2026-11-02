from __future__ import annotations

"""Stage 296: official NOAA ISD Japan download and coverage audit.

This stage consumes the frozen Stage294 protocol and Stage295 evaluation
contract.  It downloads only the official NCEI global-hourly archives for
the frozen Japanese station IDs and years, normalizes the pre-registered
ISD fields to UTC, and audits split coverage.  It does not fit a detector,
select a threshold, inject an attack, or compute any performance result.
"""

import gzip
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "raw" / "noaa_isd_japan"
PREFIX = "stage296_japan_isd_official_download"
PROTOCOL_PREFIX = "stage294_japan_isd_protocol_preregistration"
CONTRACT_PREFIX = "stage295_japan_isd_locked_evaluation_contract"

USER_AGENT = "TrustMeteo-QC reproducible research NOAA ISD Japan download"
OFFICIAL_ROOT = "https://www.ncei.noaa.gov/data/global-hourly/access"
YEARS = [2021, 2022, 2023, 2024, 2025]
COVERAGE_MIN = 0.70

FIELD_TO_VARIABLE = {
    "TMP": "temperature_c",
    "DEW": "dew_point_c",
    "WND": "wind_speed_ms",
    "SLP": "pressure_hpa",
}


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


def download(url: str, retries: int = 5, timeout: int = 180) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                }
            if not payload:
                raise RuntimeError(f"empty NOAA ISD response: {url}")
            return payload, headers
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2.0 + 2.0 * attempt)
    raise RuntimeError(f"NOAA ISD download failed: {url}") from last_error


def decompress_payload(payload: bytes) -> bytes:
    if payload.startswith(b"\x1f\x8b"):
        return gzip.decompress(payload)
    return payload


def parse_isd_component(value: object, index: int, scale: float = 1.0) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    parts = text.split(",")
    if index >= len(parts):
        return np.nan
    number_text = parts[index].strip()
    if not number_text:
        return np.nan
    try:
        number = float(number_text)
    except ValueError:
        return np.nan
    if not np.isfinite(number) or number >= 9990:
        return np.nan
    return number / scale


def parse_isd_quality(value: object, index: int = 1) -> str:
    if value is None:
        return ""
    parts = str(value).split(",")
    if index >= len(parts):
        return ""
    return parts[index].strip()


def parse_station_year(
    payload: bytes,
    station_id: str,
    year: int,
    accepted_quality_code: str,
) -> pd.DataFrame:
    text = decompress_payload(payload).decode("utf-8-sig", errors="replace")
    raw = pd.read_csv(pd.io.common.StringIO(text), dtype=str)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "timestamp_utc",
                "temperature_c",
                "dew_point_c",
                "wind_speed_ms",
                "pressure_hpa",
                "temperature_quality",
                "dew_point_quality",
                "wind_speed_quality",
                "pressure_quality",
            ]
        )
    required = {"DATE", "TMP", "DEW", "WND", "SLP"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise RuntimeError(
            f"NOAA ISD archive {station_id}/{year} missing fields: {missing}"
        )
    frame = pd.DataFrame(
        {
            "station_id": station_id,
            "timestamp_utc": pd.to_datetime(
                raw["DATE"],
                errors="coerce",
                utc=True,
            ),
            "temperature_c": raw["TMP"].map(
                lambda value: parse_isd_component(value, 0, 10.0)
            ),
            "dew_point_c": raw["DEW"].map(
                lambda value: parse_isd_component(value, 0, 10.0)
            ),
            "wind_speed_ms": raw["WND"].map(
                lambda value: parse_isd_component(value, 3, 10.0)
            ),
            "pressure_hpa": raw["SLP"].map(
                lambda value: parse_isd_component(value, 0, 10.0)
            ),
            "temperature_quality": raw["TMP"].map(
                lambda value: parse_isd_quality(value, 1)
            ),
            "dew_point_quality": raw["DEW"].map(
                lambda value: parse_isd_quality(value, 1)
            ),
            "wind_speed_quality": raw["WND"].map(
                lambda value: parse_isd_quality(value, 1)
            ),
            "pressure_quality": raw["SLP"].map(
                lambda value: parse_isd_quality(value, 1)
            ),
        }
    )
    frame = frame.dropna(subset=["timestamp_utc"]).copy()
    frame["year"] = frame["timestamp_utc"].dt.year.astype(int)
    frame = frame[frame["year"].eq(int(year))].copy()
    for variable in FIELD_TO_VARIABLE.values():
        quality_column = f"{variable.replace('_c', '').replace('_ms', '').replace('_hpa', '')}_quality"
        if quality_column not in frame.columns:
            continue
        # NOAA ISD QC value 1 means the observation passed the named check.
        frame.loc[
            frame[quality_column].astype(str).ne(str(accepted_quality_code)),
            variable,
        ] = np.nan
    frame["observation_present"] = frame["temperature_c"].notna()
    return frame.reset_index(drop=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data_protocol = load_json(TABLES / f"{PROTOCOL_PREFIX}.json")
    evaluation_contract = load_json(TABLES / f"{CONTRACT_PREFIX}.json")
    if data_protocol["status"] != "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD":
        raise RuntimeError("Stage294 Japan protocol is not frozen")
    if data_protocol["download_status"] != "NOT_DOWNLOADED":
        raise RuntimeError("Stage294 Japan download was already consumed")
    if data_protocol["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("Stage294 Japan performance was already evaluated")
    if evaluation_contract["status"] != (
        "EVALUATION_CONTRACT_FROZEN_BEFORE_FULL_DOWNLOAD"
    ):
        raise RuntimeError("Stage295 Japan evaluation contract is not frozen")
    if evaluation_contract["download_status"] != "NOT_DOWNLOADED":
        raise RuntimeError("Stage295 Japan download was already consumed")
    if evaluation_contract["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("Stage295 Japan performance was already evaluated")

    stations = pd.read_csv(
        TABLES / f"{PROTOCOL_PREFIX}_stations.csv",
        dtype={"station_id": str},
    )
    frozen_station_ids = [
        str(value) for value in data_protocol["station_ids"]
    ]
    if stations["station_id"].astype(str).tolist() != frozen_station_ids:
        raise RuntimeError("Japan station table changed")
    if list(data_protocol["years"]) != YEARS:
        raise RuntimeError("Japan years changed")
    accepted_quality_code = str(data_protocol["quality_accepted_code"])
    if accepted_quality_code != "1":
        raise RuntimeError("Japan ISD accepted quality code changed")
    if data_protocol["download_contract"] != (
        "Use only https://www.ncei.noaa.gov/data/global-hourly/access/"
        "<year>/<station_id>.csv for the frozen station IDs and years "
        "2021-2025. Record URL, download time, bytes, SHA256, station id, "
        "year, and response headers. Do not use any other mirror or derived "
        "observation source as the locked historical source."
    ):
        raise RuntimeError("Japan download contract changed")

    RAW.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.now(timezone.utc).isoformat()
    detail_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    for station_id in frozen_station_ids:
        for year in YEARS:
            url = f"{OFFICIAL_ROOT}/{year}/{station_id}.csv"
            raw_path = RAW / f"{station_id}_{year}.csv"
            if raw_path.is_file():
                payload = raw_path.read_bytes()
                headers: dict[str, str] = {"source": "cache"}
                status = "cached"
            else:
                payload, headers = download(url)
                raw_path.write_bytes(payload)
                status = "downloaded"
            try:
                parsed = parse_station_year(
                    payload,
                    station_id,
                    int(year),
                    accepted_quality_code,
                )
                frames.append(parsed)
                detail_rows.append(
                    {
                        "station_id": station_id,
                        "year": int(year),
                        "url": url,
                        "status": status,
                        "downloaded_at_utc": downloaded_at,
                        "content_type": headers.get("content-type", ""),
                        "bytes": int(len(payload)),
                        "sha256": sha256_bytes(payload),
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
                        "year": int(year),
                        "url": url,
                        "status": "failed",
                        "downloaded_at_utc": downloaded_at,
                        "content_type": headers.get("content-type", ""),
                        "bytes": int(len(payload)),
                        "sha256": sha256_bytes(payload),
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
                "dew_point_c",
                "wind_speed_ms",
                "pressure_hpa",
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
        "status": "PASS_JAPAN_ISD_DOWNLOAD_AND_COVERAGE_AUDIT",
        "protocol": (
            "阶段296按阶段294/295冻结的12个NOAA ISD日本站点、"
            "2021-2025切分和变量规则，从官方global-hourly路径下载并"
            "规范化；2025 test截止到2025-08-24T23:59:59Z。只做覆盖率"
            "审计，不训练模型、不校阈值、不注入攻击、不计算性能。"
        ),
        "data_protocol_sha256": sha256_file(
            TABLES / f"{PROTOCOL_PREFIX}.json"
        ),
        "evaluation_contract_sha256": sha256_file(
            TABLES / f"{CONTRACT_PREFIX}.json"
        ),
        "station_count": int(len(stations)),
        "expected_archives": int(len(stations) * len(YEARS)),
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
        "quality_boundary": (
            data_protocol["quality_rule"]
        ),
        "download_status": "DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "limitations": [
            "本阶段只完成官方下载、规范化与覆盖率审计。",
            "未训练模型、未选择阈值、未注入攻击、未计算性能。",
            "未通过覆盖率规则的站点保留在审计表中并排除。",
            "2025 test为部分年度，截止到2025-08-24，不能声称全年覆盖。",
            "日本网络与六盘山NOAA网络空间不相交，但仍属于同一NOAA provider，不能单独证明机构无关。",
            "后续攻击仍为透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段296 NOAA ISD日本官方下载与覆盖率审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 预期文件：`{payload['expected_archives']}`",
        f"- 下载成功：`{payload['downloaded_count']}`",
        f"- 缓存命中：`{payload['cached_count']}`",
        f"- 下载失败：`{payload['failed_count']}`",
        f"- 规范化行数：`{payload['normalized_rows']}`",
        f"- 保留站点：`{payload['retained_station_count']}`",
        f"- 性能状态：`{payload['performance_status']}`",
        "",
        "## 边界",
        "",
        "- 只保留通过官方质量检查的观测，质量标记不是攻击标签。",
        "- 本阶段不训练模型、不选择阈值、不注入攻击、不计算性能。",
        "- 未通过覆盖率规则的站点保留在审计表中并排除。",
        "- 2025 test是部分年度，截止到2025-08-24。",
        "- 同一NOAA provider下的新站点网络不能证明机构无关。",
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
