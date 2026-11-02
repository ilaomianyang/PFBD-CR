from __future__ import annotations

"""Stage 225: download the pre-registered KNMI station archives.

Stage 224 froze the KNMI protocol before any hourly observation values
were inspected for model selection or performance.  This stage reads the
official KNMI station-coordinate snapshot through KNMI's public Open
Data API, selects the pre-registered 12 stations with the frozen 0.75
degree grid rule, and downloads each station's official 2021-2030
hourly archive from the KNMI CDN.

The stage records URLs, download times, byte counts, SHA256 digests,
archive members, fields, row counts, and timestamp ranges.  It does not
train a detector, select a threshold, or compute performance.
"""

import hashlib
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
DATA_DIR = ROOT.parents[1] / "dataset" / "external" / "knmi"
PROTOCOL = TABLES / "stage224_knmi_protocol_preregistration.json"
STATIONS = TABLES / "stage225_knmi_stations.csv"
PREFIX = "stage225_knmi_official_download"

USER_AGENT = "TrustMeteo-QC/0.1 (reproducible research; KNMI OpenData)"
CATALOGUE_URL = (
    "https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens"
)
CDN_BASE = (
    "https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/"
    "uurgegevens"
)
OPEN_DATA_API = "https://api.dataplatform.knmi.nl/open-data/v1"
STATION_DATASET = "waarneemstations_csv"
STATION_VERSION = "1.0"

# KNMI publishes this anonymous key on its official Open Data API page.
# It is intended for unregistered access to open data, not a credential
# bypass.  The key is read from the official page at run time so the
# script does not silently depend on an expired hard-coded token.
ANONYMOUS_KEY_PAGE = "https://developer.dataplatform.knmi.nl/open-data-api"

REQUIRED_COLUMNS = {"STN", "YYYYMMDD", "HH", "T", "U", "TD", "P"}
SELECTED_STATION_IDS = [
    209,
    215,
    229,
    251,
    258,
    277,
    278,
    285,
    286,
    290,
    308,
    315,
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_text(url: str, retries: int = 3, timeout: int = 90) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(str(last_error))


def download(url: str, retries: int = 3, timeout: int = 120) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if not payload:
                raise ValueError("empty response")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(str(last_error))


def fetch_json(
    url: str,
    headers: dict[str, str] | None = None,
    retries: int = 4,
) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, **(headers or {})},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < retries:
                time.sleep(20 * attempt)
                continue
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(str(last_error))


def anonymous_api_key() -> str:
    page = fetch_text(ANONYMOUS_KEY_PAGE)
    match = re.search(
        r"<pre><code>\s*([A-Za-z0-9_.=-]{40,})\s*</code></pre>",
        page,
    )
    if match is None:
        raise RuntimeError("KNMI anonymous Open Data API key not found")
    return match.group(1)


def official_station_ids() -> list[int]:
    catalogue = fetch_text(CATALOGUE_URL)
    matches = re.findall(r"uurgeg_(\d+)_2021-2030\.zip", catalogue)
    station_ids = sorted({int(value) for value in matches})
    if len(station_ids) < 12:
        raise RuntimeError("KNMI catalogue station set is unexpectedly small")
    return station_ids


def latest_station_metadata(api_key: str) -> tuple[str, str]:
    base = (
        f"{OPEN_DATA_API}/datasets/{STATION_DATASET}/versions/"
        f"{STATION_VERSION}/files"
    )
    headers = {"Authorization": api_key}
    payload = fetch_json(base, headers=headers)
    files = list(payload.get("files", []))
    while payload.get("nextPageToken"):
        token = urllib.parse.quote(str(payload["nextPageToken"]), safe="")
        payload = fetch_json(
            f"{base}?nextPageToken={token}", headers=headers
        )
        files.extend(payload.get("files", []))
    if not files:
        raise RuntimeError("KNMI station metadata API returned no files")
    latest = max(files, key=lambda item: str(item["created"]))
    filename = str(latest["filename"])
    url_payload = fetch_json(
        f"{base}/{urllib.parse.quote(filename, safe='')}/url",
        headers=headers,
    )
    download_url = str(url_payload["temporaryDownloadUrl"])
    return filename, download_url


def select_stations(
    metadata: pd.DataFrame, station_ids: list[int]
) -> pd.DataFrame:
    frame = metadata.loc[metadata["STN"].isin(station_ids)].copy()
    if frame["STN"].duplicated().any():
        raise RuntimeError("duplicate STN rows in KNMI station metadata")
    frame["latitude_bin"] = (
        (frame["POS_NB"] - 50.5) // 0.75
    ).map(math.floor)
    frame["longitude_bin"] = (
        (frame["POS_OL"] - 3.0) // 0.75
    ).map(math.floor)
    frame["spatial_cell"] = (
        frame["latitude_bin"].astype(str)
        + "_"
        + frame["longitude_bin"].astype(str)
    )
    first_per_cell = (
        frame.sort_values(["spatial_cell", "STN"])
        .drop_duplicates("spatial_cell", keep="first")
        .sort_values("STN")
    )
    chosen = first_per_cell.head(12).copy()
    if len(chosen) < 12:
        remaining = frame.loc[
            ~frame["STN"].isin(chosen["STN"])
        ].sort_values("STN")
        chosen = pd.concat([chosen, remaining], ignore_index=True)
        chosen = chosen.drop_duplicates("STN").head(12)
    chosen = chosen.sort_values("STN").reset_index(drop=True)
    chosen["selection_rank"] = range(1, len(chosen) + 1)
    return chosen


def audit_archive(payload: bytes) -> dict[str, object]:
    result: dict[str, object] = {
        "archive_members": [],
        "data_members": [],
        "required_columns_ok": False,
        "rows": 0,
        "columns": [],
        "timestamp_min": None,
        "timestamp_max": None,
        "station_ids": [],
        "error": None,
    }
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            result["archive_members"] = names
            data_members = [
                name for name in names if name.lower().endswith(".txt")
            ]
            result["data_members"] = data_members
            if not data_members:
                raise ValueError("no text member found")
            text = archive.read(data_members[0]).decode(
                "utf-8", errors="replace"
            )
            lines = text.splitlines()
            header_index = next(
                (
                    index
                    for index, line in enumerate(lines)
                    if line.lstrip("# ").startswith("STN,")
                ),
                None,
            )
            if header_index is None:
                raise ValueError("KNMI data header not found")
            csv_text = "\n".join(
                [lines[header_index].lstrip("# ").strip()]
                + lines[header_index + 1 :]
            )
            frame = pd.read_csv(io.StringIO(csv_text), dtype=str)
            frame.columns = [str(column).strip() for column in frame.columns]
            result["columns"] = frame.columns.tolist()
            result["required_columns_ok"] = REQUIRED_COLUMNS.issubset(
                frame.columns
            )
            result["rows"] = int(len(frame))
            if "STN" in frame:
                result["station_ids"] = sorted(
                    frame["STN"].dropna().astype(str).unique().tolist()
                )
            if {"YYYYMMDD", "HH"}.issubset(frame.columns):
                timestamps = pd.to_datetime(
                    frame["YYYYMMDD"].astype(str).str.strip()
                    + frame["HH"].astype(str).str.strip().str.zfill(2),
                    format="%Y%m%d%H",
                    errors="coerce",
                    utc=True,
                )
                if not timestamps.dropna().empty:
                    result["timestamp_min"] = str(timestamps.min())
                    result["timestamp_max"] = str(timestamps.max())
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_KNMI_PERFORMANCE":
        raise RuntimeError("Stage224 protocol is not frozen before download")
    if protocol["station_count"] != 12:
        raise RuntimeError("Stage224 station count changed")

    api_key = anonymous_api_key()
    metadata_filename, metadata_url = latest_station_metadata(api_key)
    metadata_payload = download(metadata_url)
    metadata_path = DATA_DIR / metadata_filename
    metadata_path.write_bytes(metadata_payload)
    metadata = pd.read_csv(io.BytesIO(metadata_payload))
    metadata.columns = [str(column).strip() for column in metadata.columns]
    if not {"STN", "POS_NB", "POS_OL"}.issubset(metadata.columns):
        raise RuntimeError("KNMI station metadata fields changed")

    catalogue_ids = official_station_ids()
    selected = select_stations(metadata, catalogue_ids)
    if selected["STN"].astype(int).tolist() != SELECTED_STATION_IDS:
        raise RuntimeError(
            "KNMI frozen selection changed: "
            f"{selected['STN'].astype(int).tolist()}"
        )
    station_output = selected[
        [
            "STN",
            "LOCATIE",
            "STARTT",
            "STOPT",
            "POS_NB",
            "POS_OL",
            "spatial_cell",
            "selection_rank",
        ]
    ].rename(
        columns={
            "STN": "station_id",
            "LOCATIE": "station_name",
            "STARTT": "metadata_start",
            "STOPT": "metadata_stop",
            "POS_NB": "latitude",
            "POS_OL": "longitude",
        }
    )
    station_output.to_csv(
        STATIONS, index=False, encoding="utf-8-sig"
    )

    rows: list[dict[str, object]] = []
    for station_id in selected["STN"].astype(int):
        filename = f"uurgeg_{station_id}_2021-2030.zip"
        url = f"{CDN_BASE}/{filename}"
        target = DATA_DIR / filename
        item: dict[str, object] = {
            "station_id": station_id,
            "url": url,
            "target": str(target),
            "status": "pending",
            "downloaded_at_utc": None,
            "bytes": None,
            "sha256": None,
            "error": None,
        }
        try:
            if target.exists() and target.stat().st_size > 0:
                payload = target.read_bytes()
                item["status"] = "already_present"
                item["bytes"] = len(payload)
                item["sha256"] = sha256_bytes(payload)
            else:
                payload = download(url)
                target.write_bytes(payload)
                item["status"] = "downloaded"
                item["downloaded_at_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()
                item["bytes"] = len(payload)
                item["sha256"] = sha256_bytes(payload)
            item.update(
                {
                    f"audit_{key}": value
                    for key, value in audit_archive(payload).items()
                }
            )
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = repr(exc)
        rows.append(item)

    detail = pd.DataFrame(rows)
    detail.to_csv(
        TABLES / f"{PREFIX}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    passed = (
        detail["status"].isin(["downloaded", "already_present"])
        & detail["audit_required_columns_ok"].fillna(False).astype(bool)
        & detail["audit_error"].isna()
    )
    payload = {
        "status": (
            "PASS_PIPELINE_REVIEW_REQUIRED"
            if bool(passed.all())
            else "REVIEW_REQUIRED"
        ),
        "protocol": (
            "阶段225在下载KNMI小时观测前，先用KNMI官方Open Data API"
            "读取官方站点坐标最新快照，并按阶段224冻结的0.75度网格"
            "和站号规则选择12站；随后只从KNMI官方CDN下载每站"
            "2021-2030小时ZIP。保留URL、下载时间、字节数、SHA256、"
            "压缩包成员、字段、行数、站号和时间范围。不训练模型、"
            "不选择阈值、不计算性能。"
        ),
        "protocol_status": protocol["status"],
        "provider": protocol["provider"],
        "catalogue_url": CATALOGUE_URL,
        "cdn_base": CDN_BASE,
        "station_metadata_dataset": STATION_DATASET,
        "station_metadata_version": STATION_VERSION,
        "station_metadata_filename": metadata_filename,
        "station_metadata_api_url": (
            f"{OPEN_DATA_API}/datasets/{STATION_DATASET}/versions/"
            f"{STATION_VERSION}/files/{metadata_filename}/url"
        ),
        "station_metadata_temporary_url_used": True,
        "station_metadata_sha256": sha256_bytes(metadata_payload),
        "station_metadata_columns": metadata.columns.tolist(),
        "official_catalogue_station_count": len(catalogue_ids),
        "station_count": len(selected),
        "expected_archives": len(selected),
        "downloaded_count": int(detail["status"].eq("downloaded").sum()),
        "already_present_count": int(
            detail["status"].eq("already_present").sum()
        ),
        "failed_count": int(detail["status"].eq("failed").sum()),
        "required_columns_ok_count": int(
            detail["audit_required_columns_ok"].fillna(False).sum()
        ),
        "archive_audit_failure_count": int(
            detail["audit_error"].notna().sum()
        ),
        "selected_station_ids": selected["STN"].astype(int).tolist(),
        "detail": rows,
        "limitations": [
            "阶段225只验证官方下载、归档结构和站点覆盖，不评价模型性能。",
            "KNMI官方说明小时序列可能因站址迁移和观测方式变化而不均质，不能用于趋势分析。",
            "空字段按缺失保留，不插补；缺失不等于攻击标签。",
            "所有攻击仍须是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段225 KNMI官方小时数据下载与站点冻结",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 来源：{payload['provider']}",
        f"- 官方站点坐标快照：`{metadata_filename}`",
        f"- 坐标快照SHA256：`{payload['station_metadata_sha256']}`",
        f"- 官方小时站点总数：`{len(catalogue_ids)}`；冻结站点：`{len(selected)}`",
        f"- 本次下载：`{payload['downloaded_count']}`；已存在：`{payload['already_present_count']}`；失败：`{payload['failed_count']}`",
        f"- 字段审计通过：`{payload['required_columns_ok_count']}`；归档审计失败：`{payload['archive_audit_failure_count']}`",
        "",
        "| 站号 | 站名 | 纬度 | 经度 | 空间单元 | 状态 | 字节 | 行数 | 时间范围 | 字段完整 |",
        "|---|---|---:|---:|---|---|---:|---:|---|---|",
    ]
    station_lookup = station_output.set_index("station_id").to_dict(
        orient="index"
    )
    for row in rows:
        station = station_lookup[int(row["station_id"])]
        time_range = (
            f"{row.get('audit_timestamp_min')} -> "
            f"{row.get('audit_timestamp_max')}"
        )
        lines.append(
            f"| {row['station_id']} | {station['station_name']} | "
            f"{float(station['latitude']):.4f} | "
            f"{float(station['longitude']):.4f} | "
            f"{station['spatial_cell']} | {row['status']} | "
            f"{row.get('bytes') or 0} | {row.get('audit_rows') or 0} | "
            f"{time_range} | "
            f"{'是' if row.get('audit_required_columns_ok') else '否'} |"
        )
    lines += [
        "",
        "本阶段只保存KNMI官方原始坐标快照和小时归档，不训练检测器、",
        "不选择阈值、不计算攻击召回率；空字段不插补，质量缺失不冒充攻击标签。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "official_catalogue_station_count": len(catalogue_ids),
                "station_count": len(selected),
                "selected_station_ids": payload["selected_station_ids"],
                "downloaded_count": payload["downloaded_count"],
                "already_present_count": payload["already_present_count"],
                "failed_count": payload["failed_count"],
                "required_columns_ok_count": payload[
                    "required_columns_ok_count"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if payload["status"] == "PASS_PIPELINE_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
