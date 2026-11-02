from __future__ import annotations

"""Stage 258: download and normalize the pre-registered FMI observations.

The protocol in Stage257 must already be frozen.  This stage downloads
hourly WFS observations for the frozen stations and years, records URLs and
SHA256 digests, normalizes the four primary variables, and audits coverage.
It does not train a detector, select a threshold, or compute any
performance result.
"""

import hashlib
import json
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "raw" / "fmi"
PREFIX = "stage258_fmi_official_download"
PROTOCOL_PREFIX = "stage257_fmi_protocol_preregistration"
USER_AGENT = "TrustMeteo-QC official FMI download"

PARAMETER_BY_VARIABLE = {
    "temperature_c": "TA_PT1H_AVG",
    "humidity_pct": "RH_PT1H_AVG",
    "wind_speed_ms": "WS_PT1H_AVG",
    "pressure_hpa": "PA_PT1H_AVG",
}
PARAMETERS = list(PARAMETER_BY_VARIABLE.values())


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


def observation_url(
    station_id: str,
    start: str,
    end: str,
) -> str:
    return (
        "https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0"
        "&request=GetFeature"
        "&storedquery_id=fmi::observations::weather::hourly::simple"
        f"&fmisid={station_id}"
        f"&starttime={start}&endtime={end}"
        f"&parameters={','.join(PARAMETERS)}"
    )


def year_windows(year: int) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    for month in range(1, 13):
        if month == 12:
            next_year = year + 1
            next_month = 1
        else:
            next_year = year
            next_month = month + 1
        windows.append(
            (
                f"{year}-{month:02d}-01T00:00:00Z",
                f"{next_year}-{next_month:02d}-01T00:00:00Z",
            )
        )
    return windows


def download(url: str, retries: int = 3) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError("empty response")
            return payload, sha256_bytes(payload)
        except (
            urllib.error.URLError,
            TimeoutError,
            RuntimeError,
        ) as error:
            last_error = error
    raise RuntimeError(f"FMI download failed: {url}") from last_error


def parse_observations(
    payload: bytes,
    station_id: str,
    station_name: str,
) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(
            columns=["timestamp_utc", *PARAMETERS]
        )
    root = ET.fromstring(payload)
    namespace = {
        "wfs": "http://www.opengis.net/wfs/2.0",
        "gml": "http://www.opengis.net/gml/3.2",
        "bswfs": "http://xml.fmi.fi/schema/wfs/2.0",
    }
    rows: list[dict[str, object]] = []
    for member in root.findall("wfs:member", namespace):
        element = member.find("bswfs:BsWfsElement", namespace)
        if element is None:
            continue
        time_node = element.find("bswfs:Time", namespace)
        name_node = element.find("bswfs:ParameterName", namespace)
        value_node = element.find("bswfs:ParameterValue", namespace)
        if (
            time_node is None
            or name_node is None
            or value_node is None
        ):
            continue
        value_text = (value_node.text or "").strip()
        try:
            value = float(value_text) if value_text else np.nan
        except ValueError:
            value = np.nan
        rows.append(
            {
                "timestamp_utc": pd.Timestamp(
                    time_node.text,
                    tz="UTC",
                ),
                "station_id": str(station_id),
                "station_name": str(station_name),
                "parameter": (name_node.text or "").strip(),
                "value": value,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["timestamp_utc", "parameter", "value"]
        )
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(
        ["station_id", "timestamp_utc", "parameter"],
        keep="last",
    )
    wide = frame.pivot(
        index=["station_id", "station_name", "timestamp_utc"],
        columns="parameter",
        values="value",
    )
    wide = wide.reindex(columns=PARAMETERS)
    wide.columns.name = None
    return wide.reset_index()


def split_xml_documents(payload: bytes) -> list[bytes]:
    marker = b"</wfs:FeatureCollection>"
    documents: list[bytes] = []
    start = 0
    while True:
        end = payload.find(marker, start)
        if end < 0:
            break
        end += len(marker)
        documents.append(payload[start:end])
        start = end
    return documents


def normalize_observations(
    year: int,
    payload: bytes,
    station_id: str,
    station_name: str,
) -> pd.DataFrame:
    wide = parse_observations(payload, station_id, station_name)
    if wide.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "timestamp_utc",
                "temperature_c",
                "humidity_pct",
                "wind_speed_ms",
                "pressure_hpa",
                "observation_present",
                "split",
            ]
        )
    wide = wide.rename(
        columns={
            PARAMETER_BY_VARIABLE[target]: target
            for target in PARAMETER_BY_VARIABLE
        }
    )
    for target in PARAMETER_BY_VARIABLE:
        if target not in wide.columns:
            wide[target] = np.nan
        wide[target] = pd.to_numeric(wide[target], errors="coerce")
    wide["year"] = int(year)
    wide["observation_present"] = wide["temperature_c"].notna()
    if year <= 2023:
        wide["split"] = "train"
    elif year == 2024:
        wide["split"] = "valid"
    elif year == 2025:
        wide["split"] = "test"
    else:
        raise RuntimeError(f"unexpected FMI year {year}")
    return wide[
        [
            "station_id",
            "station_name",
            "timestamp_utc",
            "year",
            "temperature_c",
            "humidity_pct",
            "wind_speed_ms",
            "pressure_hpa",
            "observation_present",
            "split",
        ]
    ]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--station-id", default="")
    parser.add_argument("--year", type=int, default=0)
    args = parser.parse_args()
    protocol_path = TABLES / f"{PROTOCOL_PREFIX}.json"
    protocol = load_json(protocol_path)
    if protocol.get("status") != "PROTOCOL_FROZEN_BEFORE_DOWNLOAD":
        raise RuntimeError("Stage257 FMI protocol is not frozen")
    if protocol.get("download_status") != "NOT_DOWNLOADED":
        raise RuntimeError("Stage257 FMI protocol was already consumed")
    station_path = TABLES / f"{PROTOCOL_PREFIX}_stations.csv"
    stations = pd.read_csv(
        station_path,
        dtype={"station_id": str},
    )
    stations["station_id"] = stations["station_id"].str.zfill(6)
    RAW.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    def process_station_year(
        station_id: str,
        station_name: str,
        year: int,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        raw_path = RAW / f"{station_id}_{year}.xml"
        month_paths = [
            RAW / f"{station_id}_{year}_{month_index:02d}.xml"
            for month_index in range(1, 13)
        ]
        parts: list[bytes] = []
        urls: list[str] = []
        digests: list[str] = []
        for month_index, (start, end) in enumerate(
            year_windows(year),
            start=1,
        ):
            url = observation_url(station_id, start, end)
            month_path = month_paths[month_index - 1]
            if month_path.is_file():
                payload = month_path.read_bytes()
                digest = sha256_bytes(payload)
            else:
                payload, digest = download(url)
                month_path.write_bytes(payload)
            parts.append(payload)
            urls.append(url)
            digests.append(digest)
        payload = b"\n".join(parts)
        raw_path.write_bytes(payload)
        normalized_parts = [
            normalize_observations(
                year,
                part,
                station_id,
                station_name,
            )
            for part in parts
        ]
        frame = (
            pd.concat(normalized_parts, ignore_index=True)
            .drop_duplicates(
                ["station_id", "timestamp_utc"],
                keep="last",
            )
            .sort_values("timestamp_utc", kind="stable")
            .reset_index(drop=True)
        )
        detail = {
            "station_id": station_id,
            "station_name": station_name,
            "year": int(year),
            "url": ";".join(urls),
            "status": "downloaded",
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "part_sha256": ";".join(digests),
            "rows": int(len(frame)),
            "temperature_present": int(
                frame["temperature_c"].notna().sum()
            ),
            "error": "",
        }
        return frame, detail

    tasks: list[tuple[str, str, int]] = []
    for station in stations.itertuples():
        for year in protocol["years"]:
            if args.station_id and str(station.station_id) != args.station_id:
                continue
            if args.year and int(year) != args.year:
                continue
            tasks.append(
                (
                    str(station.station_id),
                    str(station.station_name),
                    int(year),
                )
            )
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(
                process_station_year,
                station_id,
                station_name,
                year,
            ): (station_id, station_name, year)
            for station_id, station_name, year in tasks
        }
        for future in as_completed(future_map):
            station_id, station_name, year = future_map[future]
            try:
                frame, detail = future.result()
                frames.append(frame)
                detail_rows.append(detail)
                status = "downloaded"
            except Exception as error:
                frames.append(
                    normalize_observations(
                        year,
                        b"",
                        station_id,
                        station_name,
                    )
                )
                detail_rows.append(
                    {
                        "station_id": station_id,
                        "station_name": station_name,
                        "year": year,
                        "url": "",
                        "status": "failed",
                        "bytes": 0,
                        "sha256": "",
                        "part_sha256": "",
                        "rows": 0,
                        "temperature_present": 0,
                        "error": str(error),
                    }
                )
                status = "failed"
            print(
                json.dumps(
                    {
                        "station_id": station_id,
                        "year": year,
                        "status": status,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    detail = pd.DataFrame(detail_rows)
    normalized = pd.concat(frames, ignore_index=True)
    value_columns = list(PARAMETER_BY_VARIABLE)
    normalized["_nonmissing"] = normalized[value_columns].notna().sum(axis=1)
    normalized = normalized.sort_values(
        ["station_id", "timestamp_utc", "_nonmissing"],
        kind="stable",
    ).drop_duplicates(
        ["station_id", "timestamp_utc"],
        keep="last",
    )
    normalized = normalized.drop(columns=["_nonmissing"])
    normalized = normalized.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    normalized.to_parquet(
        TABLES / f"{PREFIX}_normalized.parquet",
        index=False,
    )
    detail.to_csv(
        TABLES / f"{PREFIX}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    station_rows: list[dict[str, object]] = []
    for station in stations.itertuples():
        station_frame = normalized[
            normalized["station_id"].eq(str(station.station_id))
        ]
        row: dict[str, object] = {
            "station_id": str(station.station_id),
            "station_name": str(station.station_name),
        }
        retained = True
        for split in ("train", "valid", "test"):
            split_frame = station_frame[station_frame["split"].eq(split)]
            present = int(split_frame["observation_present"].sum())
            expected = int(len(split_frame))
            coverage = present / expected if expected else 0.0
            row[f"{split}_rows"] = expected
            row[f"{split}_present"] = present
            row[f"{split}_coverage"] = float(coverage)
            if expected == 0 or present == 0 or coverage < 0.70:
                retained = False
        row["retained"] = bool(retained)
        station_rows.append(row)
    station_audit = pd.DataFrame(station_rows)
    station_audit.to_csv(
        TABLES / f"{PREFIX}_station_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    retained_ids = station_audit.loc[
        station_audit["retained"],
        "station_id",
    ].astype(str).tolist()
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段258按阶段257冻结的FMI站点、年份、变量和覆盖率规则"
            "下载官方逐小时观测并规范化；不训练模型、不选择阈值、"
            "不计算性能。下载URL、时间、字节数、SHA256、行数和"
            "覆盖率均保留。"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "station_count": int(len(stations)),
        "expected_archives": int(len(stations) * len(protocol["years"])),
        "downloaded_count": int(detail["status"].eq("downloaded").sum()),
        "failed_count": int(detail["status"].eq("failed").sum()),
        "normalized_rows": int(len(normalized)),
        "retained_station_count": int(len(retained_ids)),
        "retained_station_ids": retained_ids,
        "station_audit": station_audit.to_dict("records"),
        "normalized_sha256": sha256_file(
            TABLES / f"{PREFIX}_normalized.parquet"
        ),
        "detail_sha256": sha256_file(
            TABLES / f"{PREFIX}_detail.csv"
        ),
        "download_status": "DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "limitations": [
            "本阶段只完成官方数据下载、规范化与覆盖率审计。",
            "未训练模型、未选择阈值、未计算任何检测性能。",
            "未达到覆盖率规则的站点必须保留在审计表中并排除。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段258 FMI官方观测下载与规范化",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 预期文件：`{payload['expected_archives']}`",
        f"- 下载成功：`{payload['downloaded_count']}`",
        f"- 下载失败：`{payload['failed_count']}`",
        f"- 规范化行数：`{payload['normalized_rows']}`",
        f"- 保留站点：`{payload['retained_station_count']}`",
        f"- 性能状态：`{payload['performance_status']}`",
        "",
        "## 边界",
        "",
        "- 本阶段不训练模型、不选择阈值、不计算性能。",
        "- 未通过覆盖率规则的站点保留在审计表中并排除。",
        "- 所有攻击实验仍为后续透明受控注入，不是真实攻击日志。",
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
