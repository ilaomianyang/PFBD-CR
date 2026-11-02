from __future__ import annotations

"""Stage 365: download and audit the frozen SMHI station holdout."""

import csv
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "raw" / os.environ.get(
    "TMQC_HOLDOUT_RAW_NAME",
    "smhi_station_holdout",
)
PROTOCOL_PREFIX = os.environ.get(
    "TMQC_HOLDOUT_PROTOCOL_PREFIX",
    "stage364_smhi_station_holdout_protocol",
)
PREFIX = os.environ.get(
    "TMQC_HOLDOUT_DOWNLOAD_PREFIX",
    "stage365_smhi_station_holdout_download",
)
PARAMETER_OUTPUT = {
    1: "temperature_c",
    6: "humidity_pct",
    4: "wind_speed_ms",
    9: "pressure_hpa",
}
URL = (
    "https://opendata-download-metobs.smhi.se/api/version/1.0/"
    "parameter/{parameter_id}/station/{station_id}/period/"
    "corrected-archive/data.csv"
)
USER_AGENT = "TrustMeteo-QC Stage365 official SMHI holdout download"
COVERAGE_MIN = 0.70


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, retries: int = 2) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=(20, 120),
                verify=True,
            )
            response.raise_for_status()
            payload = response.content
            headers = {
                key.lower(): value for key, value in response.headers.items()
            }
            if not payload:
                raise RuntimeError(f"empty response: {url}")
            return payload, headers
        except requests.HTTPError:
            # A 4xx/5xx response is a deterministic source failure for this
            # archive and should be recorded immediately, not retried five
            # times as though it were a transient network timeout.
            raise
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(1.0 + 1.0 * attempt)
    raise RuntimeError(f"SMHI download failed: {url}") from last_error


def parse_station_csv(payload: bytes) -> tuple[pd.DataFrame, list[str]]:
    if payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)
    text = payload.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if line.startswith("Datum;")),
        None,
    )
    if header_index is None:
        raise RuntimeError("SMHI CSV has no Datum header")
    metadata = [line.rstrip("\r\n") for line in lines[:header_index]]
    header = lines[header_index].split(";")
    if len(header) < 3:
        raise RuntimeError("SMHI CSV has no value column")
    quality_column = next(
        (
            index
            for index, name in enumerate(header)
            if name.strip().lower() in {"kvalitet", "kvalitetskod"}
        ),
        None,
    )
    if quality_column is None:
        raise RuntimeError("SMHI CSV has no quality column")
    frame = pd.read_csv(
        io.StringIO("\n".join(lines[header_index:])),
        sep=";",
        dtype=str,
        keep_default_na=False,
        engine="c",
    )
    if frame.shape[1] <= max(2, quality_column):
        raise RuntimeError("SMHI CSV rows have insufficient columns")
    date_column = frame.columns[0]
    time_column = frame.columns[1]
    value_column = frame.columns[2]
    frame["timestamp_utc"] = pd.to_datetime(
        frame[date_column].str.strip()
        + " "
        + frame[time_column].str.strip(),
        utc=True,
        errors="coerce",
    )
    frame["value"] = pd.to_numeric(
        frame[value_column].str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )
    frame["quality_code"] = (
        frame.iloc[:, quality_column].astype(str).str.strip().str.upper()
    )
    output = frame[["timestamp_utc", "value", "quality_code"]].dropna(
        subset=["timestamp_utc"]
    )
    output = output[output["quality_code"].eq("G")].reset_index(drop=True)
    return output, metadata


def station_name(metadata: list[str]) -> str:
    for line in metadata:
        if line.startswith("Stationsnamn;"):
            return line.split(";", 1)[1].strip()
    return ""


def fetch_archive(
    station: object,
    parameter_id: int,
    variable: str,
    downloaded_at: str,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    station_id = str(station.station_id)
    path = RAW / f"{station_id}_{parameter_id}.csv"
    url = URL.format(parameter_id=parameter_id, station_id=station_id)
    try:
        if path.is_file():
            payload = path.read_bytes()
            headers = {"source": "cache"}
            status = "cached"
        else:
            payload, headers = download(url)
            path.write_bytes(payload)
            status = "downloaded"
        parsed, metadata = parse_station_csv(payload)
        parsed = parsed[parsed["quality_code"].eq("G")].copy()
        parsed["station_id"] = station_id
        parsed["station_name"] = station_name(metadata) or str(
            station.station_name
        )
        parsed["parameter_id"] = int(parameter_id)
        parsed["variable"] = variable
        detail = {
            "station_id": station_id,
            "station_name": (
                parsed["station_name"].iloc[0]
                if len(parsed)
                else str(station.station_name)
            ),
            "parameter_id": int(parameter_id),
            "variable": variable,
            "url": url,
            "status": status,
            "downloaded_at_utc": downloaded_at,
            "bytes": int(len(payload)),
            "sha256": sha256_bytes(payload),
            "rows": int(len(parsed)),
            "nonmissing_rows": int(parsed["value"].notna().sum()),
            "error": "",
            "content_type": headers.get("content-type", ""),
        }
        return detail, parsed
    except Exception as error:
        detail = {
            "station_id": station_id,
            "station_name": str(station.station_name),
            "parameter_id": int(parameter_id),
            "variable": variable,
            "url": url,
            "status": "failed",
            "downloaded_at_utc": downloaded_at,
            "bytes": int(path.stat().st_size) if path.is_file() else 0,
            "sha256": sha256_file(path) if path.is_file() else "",
            "rows": 0,
            "nonmissing_rows": 0,
            "error": str(error),
            "content_type": "",
        }
        return detail, None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    protocol_path = TABLES / f"{PROTOCOL_PREFIX}.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_OBSERVATION_ACCESS":
        raise RuntimeError("Stage364 protocol is not frozen")
    stations = pd.read_csv(
        TABLES / f"{PROTOCOL_PREFIX}_stations.csv",
        dtype={"station_id": str},
    )
    if stations["station_id"].astype(str).tolist() != protocol["station_ids"]:
        raise RuntimeError("Stage364 station table changed")
    RAW.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    downloaded_at = datetime.now(timezone.utc).isoformat()
    tasks = [
        (station, parameter_id, variable)
        for station in stations.itertuples()
        for parameter_id, variable in PARAMETER_OUTPUT.items()
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(fetch_archive, station, parameter_id, variable, downloaded_at)
            for station, parameter_id, variable in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            detail, parsed = future.result()
            detail_rows.append(detail)
            if parsed is not None:
                frames.append(parsed)
            print(
                json.dumps(
                    {
                        "station_id": detail["station_id"],
                        "parameter_id": detail["parameter_id"],
                        "status": detail["status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    detail_rows.sort(
        key=lambda row: (str(row["station_id"]), int(row["parameter_id"]))
    )
    detail = pd.DataFrame(detail_rows)
    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not long.empty:
        long = long.drop_duplicates(
            ["station_id", "variable", "timestamp_utc"], keep="last"
        )
        wide = long.pivot(
            index=["station_id", "station_name", "timestamp_utc"],
            columns="variable",
            values="value",
        ).reset_index()
        wide.columns.name = None
    else:
        wide = pd.DataFrame(
            columns=["station_id", "station_name", "timestamp_utc"]
        )
    for variable in PARAMETER_OUTPUT.values():
        if variable not in wide.columns:
            wide[variable] = np.nan
    if not wide.empty:
        wide["year"] = wide["timestamp_utc"].dt.year
        wide["observation_present"] = wide["temperature_c"].notna()
        wide["split"] = np.select(
            [
                wide["year"].between(2021, 2023),
                wide["year"].eq(2024),
                wide["year"].eq(2025),
            ],
            ["train", "valid", "test"],
            default="outside_protocol",
        )
        wide = wide[wide["split"].ne("outside_protocol")].copy()
        wide = wide.sort_values(["station_id", "timestamp_utc"], kind="stable")
    normalized = TABLES / f"{PREFIX}_normalized.parquet"
    detail_path = TABLES / f"{PREFIX}_detail.csv"
    audit_path = TABLES / f"{PREFIX}_station_audit.csv"
    wide.to_parquet(normalized, index=False)
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    audit_rows: list[dict[str, object]] = []
    retained: list[str] = []
    for station_id in protocol["station_ids"]:
        current = wide[wide["station_id"].astype(str).eq(str(station_id))]
        row: dict[str, object] = {
            "station_id": str(station_id),
            "station_name": str(current["station_name"].iloc[0]) if not current.empty else "",
        }
        keep = True
        for split in ("train", "valid", "test"):
            subset = current[current["split"].eq(split)]
            expected = int(len(subset))
            present = int(subset["observation_present"].sum()) if expected else 0
            coverage = present / expected if expected else 0.0
            row[f"{split}_rows"] = expected
            row[f"{split}_present"] = present
            row[f"{split}_coverage"] = float(coverage)
            keep = keep and expected > 0 and present > 0 and coverage >= COVERAGE_MIN
        row["retained"] = bool(keep)
        audit_rows.append(row)
        if keep:
            retained.append(str(station_id))
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_SMHI_STATION_HOLDOUT_DOWNLOAD_AND_COVERAGE_AUDIT",
        "protocol_id": protocol["protocol_id"],
        "data_protocol_sha256": sha256_file(protocol_path),
        "station_count": int(len(stations)),
        "expected_archives": int(len(stations) * len(PARAMETER_OUTPUT)),
        "downloaded_count": int(detail["status"].eq("downloaded").sum()),
        "cached_count": int(detail["status"].eq("cached").sum()),
        "failed_count": int(detail["status"].eq("failed").sum()),
        "normalized_rows": int(len(wide)),
        "retained_station_count": int(len(retained)),
        "retained_station_ids": retained,
        "normalized_sha256": sha256_file(normalized),
        "detail_sha256": sha256_file(detail_path),
        "station_audit_sha256": sha256_file(audit_path),
        "performance_status": "NOT_EVALUATED",
        "quality_boundary": "Only official G-quality rows are retained; quality codes are not attack labels.",
        "limitations": [
            "This stage downloads and audits the frozen station holdout only.",
            "No model fitting, threshold selection, attack injection, or performance evaluation was performed.",
            "The holdout is within SMHI and cannot establish a new independent provider.",
        ],
    }
    output = TABLES / f"{PREFIX}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = REPORTS / f"{PREFIX}.md"
    report.write_text(
        "\n".join(
            [
                "# Stage365 SMHI 未使用站点官方下载与覆盖率审计",
                "",
                f"- 状态：`{payload['status']}`",
                f"- 预期归档：`{payload['expected_archives']}`",
                f"- 下载成功：`{payload['downloaded_count']}`",
                f"- 下载失败：`{payload['failed_count']}`",
                f"- 规范化行数：`{payload['normalized_rows']}`",
                f"- 覆盖率合格站点：`{payload['retained_station_count']}`",
                "- 性能评价：未执行。",
                "",
                "所有攻击仍需在后续锁定评价中作为透明受控模拟，不能写成真实事故。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
