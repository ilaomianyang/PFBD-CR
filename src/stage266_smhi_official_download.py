from __future__ import annotations

"""Stage 266: official SMHI download, normalization, and coverage audit.

This stage consumes the frozen Stage264/265 contracts.  It downloads the
corrected-archive CSV endpoint only for the frozen 18 stations and four
parameter IDs, normalizes dates to UTC, retains only official G-quality
rows, and audits split coverage.  It does not fit a detector, select a
threshold, or compute any performance result.
"""

import csv
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
RAW = ROOT / "raw" / "smhi"
PREFIX = "stage266_smhi_official_download"
DATA_PROTOCOL_PREFIX = "stage264_smhi_protocol_preregistration"
EVALUATION_CONTRACT_PREFIX = "stage265_smhi_locked_evaluation_contract"
USER_AGENT = "TrustMeteo-QC official SMHI download"
CORRECTED_ARCHIVE_URL = (
    "https://opendata-download-metobs.smhi.se/api/version/1.0/"
    "parameter/{parameter_id}/station/{station_id}/period/"
    "corrected-archive/data.csv"
)
PARAMETER_OUTPUT = {
    1: "temperature_c",
    6: "humidity_pct",
    4: "wind_speed_ms",
    9: "pressure_hpa",
}
VALUE_QUALITY = "G"
COVERAGE_MIN = 0.70


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


def download(url: str, retries: int = 5) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
                headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                }
            if not payload:
                raise RuntimeError(f"empty SMHI response: {url}")
            return payload, headers
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2.0 + 2.0 * attempt)
    raise RuntimeError(f"SMHI download failed: {url}") from last_error


def decompress_payload(payload: bytes) -> bytes:
    if payload.startswith(b"\x1f\x8b"):
        return gzip.decompress(payload)
    return payload


def metadata_block(lines: list[str]) -> list[str]:
    output: list[str] = []
    for line in lines:
        if line.startswith("Datum;"):
            break
        output.append(line.rstrip("\r\n"))
    return output


def parse_station_csv(payload: bytes) -> tuple[pd.DataFrame, list[str]]:
    text = decompress_payload(payload).decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("Datum;")
        ),
        None,
    )
    if header_index is None:
        raise RuntimeError("SMHI CSV has no Datum header")
    block = metadata_block(lines)
    reader = csv.reader(lines[header_index:], delimiter=";")
    header = next(reader)
    value_column = 2
    date_column = 0
    time_column = 1
    quality_column = None
    for index, name in enumerate(header):
        normalized = name.strip().lower()
        if normalized in {"kvalitet", "kvalitetskod"}:
            quality_column = index
            break
    if quality_column is None:
        raise RuntimeError("SMHI CSV has no quality column")
    rows: list[dict[str, object]] = []
    for values in reader:
        if len(values) <= max(date_column, time_column, value_column):
            continue
        date_text = values[date_column].strip()
        time_text = values[time_column].strip()
        if not date_text or not time_text:
            continue
        value_text = values[value_column].strip().replace(",", ".")
        quality = (
            values[quality_column].strip().upper()
            if len(values) > quality_column
            else ""
        )
        try:
            timestamp = pd.Timestamp(
                f"{date_text} {time_text}",
                tz="UTC",
            )
            value = float(value_text) if value_text else np.nan
        except ValueError:
            timestamp = pd.NaT
            value = np.nan
        rows.append(
            {
                "timestamp_utc": timestamp,
                "value": value,
                "quality_code": quality,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=["timestamp_utc", "value", "quality_code"]
        )
    frame = frame.dropna(subset=["timestamp_utc"])
    frame = frame[
        frame["quality_code"].eq(VALUE_QUALITY)
    ].reset_index(drop=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame, block


def parse_station_name(metadata: list[str]) -> str:
    for line in metadata:
        if line and not line.startswith("Stationsnamn"):
            fields = next(csv.reader([line], delimiter=";"))
            if fields and fields[0].strip():
                return fields[0].strip()
    return ""


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data_protocol = load_json(
        TABLES / f"{DATA_PROTOCOL_PREFIX}.json"
    )
    evaluation_contract = load_json(
        TABLES / f"{EVALUATION_CONTRACT_PREFIX}.json"
    )
    if data_protocol["status"] != "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD":
        raise RuntimeError("Stage264 protocol is not frozen")
    if data_protocol["download_status"] != "NOT_DOWNLOADED":
        raise RuntimeError("Stage264 download was already consumed")
    if evaluation_contract["status"] != (
        "EVALUATION_CONTRACT_FROZEN_BEFORE_FULL_DOWNLOAD"
    ):
        raise RuntimeError("Stage265 evaluation contract is not frozen")
    if evaluation_contract["download_status"] != "NOT_DOWNLOADED":
        raise RuntimeError("Stage265 download was already consumed")
    if evaluation_contract["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("Stage265 performance was already evaluated")

    stations = pd.read_csv(
        TABLES / f"{DATA_PROTOCOL_PREFIX}_stations.csv",
        dtype={"station_id": str},
    )
    frozen_station_ids = data_protocol["station_ids"]
    if stations["station_id"].astype(str).tolist() != frozen_station_ids:
        raise RuntimeError("SMHI station table changed")
    parameter_ids = [
        int(data_protocol["parameter_ids"][variable])
        for variable in [
            "temperature_c",
            "humidity_pct",
            "wind_speed_ms",
            "pressure_hpa",
        ]
    ]
    if parameter_ids != list(PARAMETER_OUTPUT):
        raise RuntimeError("SMHI parameter IDs changed")
    if "0.70" not in str(data_protocol["coverage_rule"]):
        raise RuntimeError("SMHI coverage threshold changed")

    RAW.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    downloaded_at = datetime.now(timezone.utc).isoformat()

    for station in stations.itertuples():
        station_id = str(station.station_id)
        for parameter_id in parameter_ids:
            variable = PARAMETER_OUTPUT[parameter_id]
            raw_path = RAW / f"{station_id}_{parameter_id}.csv"
            url = CORRECTED_ARCHIVE_URL.format(
                parameter_id=parameter_id,
                station_id=station_id,
            )
            if raw_path.is_file():
                payload = raw_path.read_bytes()
                headers: dict[str, str] = {"source": "cache"}
                status = "cached"
                error = ""
            else:
                payload, headers = download(url)
                raw_path.write_bytes(payload)
                status = "downloaded"
                error = ""
            try:
                parsed, metadata = parse_station_csv(payload)
                station_name = parse_station_name(metadata) or (
                    str(station.station_name)
                )
                parsed["station_id"] = station_id
                parsed["station_name"] = station_name
                parsed["parameter_id"] = int(parameter_id)
                parsed["variable"] = variable
                frames.append(parsed)
                detail_rows.append(
                    {
                        "station_id": station_id,
                        "station_name": station_name,
                        "parameter_id": int(parameter_id),
                        "variable": variable,
                        "url": url,
                        "status": status,
                        "downloaded_at_utc": downloaded_at,
                        "content_type": headers.get("content-type", ""),
                        "bytes": int(len(payload)),
                        "sha256": sha256_bytes(payload),
                        "quality_code": VALUE_QUALITY,
                        "rows": int(len(parsed)),
                        "nonmissing_rows": int(parsed["value"].notna().sum()),
                        "error": error,
                    }
                )
            except Exception as error_obj:
                detail_rows.append(
                    {
                        "station_id": station_id,
                        "station_name": str(station.station_name),
                        "parameter_id": int(parameter_id),
                        "variable": variable,
                        "url": url,
                        "status": "failed",
                        "downloaded_at_utc": downloaded_at,
                        "content_type": headers.get("content-type", ""),
                        "bytes": int(len(payload)),
                        "sha256": sha256_bytes(payload),
                        "quality_code": VALUE_QUALITY,
                        "rows": 0,
                        "nonmissing_rows": 0,
                        "error": str(error_obj),
                    }
                )
            print(
                json.dumps(
                    {
                        "station_id": station_id,
                        "parameter_id": parameter_id,
                        "variable": variable,
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
                "timestamp_utc",
                "value",
                "quality_code",
                "station_id",
                "station_name",
                "parameter_id",
                "variable",
            ]
        )
    long = long.sort_values(
        ["station_id", "variable", "timestamp_utc"],
        kind="stable",
    ).drop_duplicates(
        ["station_id", "variable", "timestamp_utc"],
        keep="last",
    )
    wide = long.pivot(
        index=["station_id", "station_name", "timestamp_utc"],
        columns="variable",
        values="value",
    ).reset_index()
    wide.columns.name = None
    for variable in PARAMETER_OUTPUT.values():
        if variable not in wide.columns:
            wide[variable] = np.nan
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
    wide = wide.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    wide.to_parquet(
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
        station_frame = wide[wide["station_id"].astype(str).eq(station_id)]
        row: dict[str, object] = {
            "station_id": station_id,
            "station_name": (
                str(station_frame["station_name"].iloc[0])
                if not station_frame.empty
                else ""
            ),
        }
        retained = True
        for split in ("train", "valid", "test"):
            split_frame = station_frame[station_frame["split"].eq(split)]
            expected = int(len(split_frame))
            present = int(split_frame["observation_present"].sum())
            coverage = present / expected if expected else 0.0
            row[f"{split}_rows"] = expected
            row[f"{split}_present"] = present
            row[f"{split}_coverage"] = float(coverage)
            if (
                expected == 0
                or present == 0
                or coverage < COVERAGE_MIN
            ):
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
        "status": "PASS_SMHI_DOWNLOAD_AND_COVERAGE_AUDIT",
        "protocol": (
            "阶段266按阶段264/265冻结的18个SMHI站点、4个参数、"
            "2021-2025切分和G质量码规则，从官方corrected-archive下载"
            "并规范化，只做覆盖率审计；不训练模型、不校阈值、不评价性能。"
        ),
        "data_protocol_sha256": sha256_file(
            TABLES / f"{DATA_PROTOCOL_PREFIX}.json"
        ),
        "evaluation_contract_sha256": sha256_file(
            TABLES / f"{EVALUATION_CONTRACT_PREFIX}.json"
        ),
        "station_count": int(len(stations)),
        "expected_archives": int(len(stations) * len(parameter_ids)),
        "downloaded_count": int(detail["status"].eq("downloaded").sum()),
        "cached_count": int(detail["status"].eq("cached").sum()),
        "failed_count": int(detail["status"].eq("failed").sum()),
        "normalized_rows": int(len(wide)),
        "retained_station_count": int(len(retained_ids)),
        "retained_station_ids": retained_ids,
        "station_audit": station_audit.to_dict("records"),
        "normalized_sha256": sha256_file(
            TABLES / f"{PREFIX}_normalized.parquet"
        ),
        "detail_sha256": sha256_file(TABLES / f"{PREFIX}_detail.csv"),
        "coverage_rule": data_protocol["coverage_rule"],
        "quality_boundary": (
            "Only official G-quality rows are retained as normal "
            "observations. Quality codes are not attack labels."
        ),
        "download_status": "DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "limitations": [
            "本阶段只完成官方下载、规范化与覆盖率审计。",
            "未训练模型、未选择阈值、未计算任何性能结果。",
            "不通过覆盖率的站点保留在审计表中并排除。",
            "所有攻击实验仍为后续透明受控注入，不是真实攻击日志。",
            "SMHI结果不能单独证明机构无关。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段266 SMHI官方下载与覆盖率审计",
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
        "- 只保留官方 G 质量码行，质量码不是攻击标签。",
        "- 本阶段不训练模型、不选择阈值、不计算性能。",
        "- 未通过覆盖率规则的站点保留在审计表中并排除。",
        "- 后续攻击均为透明受控注入，不是真实攻击日志。",
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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
