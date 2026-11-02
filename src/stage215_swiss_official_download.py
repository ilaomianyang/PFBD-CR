from __future__ import annotations

"""Stage 215: download the pre-registered MeteoSwiss station CSV files.

Only the official Swiss federal STAC assets for the 12 Stage-214 stations
are used.  The stage downloads the hourly 2020-2029 historical CSV for
each station, records URL, timestamp, byte count, and SHA256 digest, and
audits the required columns and timestamp coverage.  It does not train a
detector, select a threshold, or compute performance.
"""

import hashlib
import io
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
DATA_DIR = ROOT.parents[1] / "dataset" / "external" / "meteoswiss"
PROTOCOL = TABLES / "stage214_swiss_protocol_preregistration.json"
STATIONS = TABLES / "stage214_swiss_protocol_preregistration_stations.csv"
PREFIX = "stage215_swiss_official_download"
USER_AGENT = "TrustMeteo-QC/0.1 (reproducible research; MeteoSwiss OpenData)"
STAC_BASE = "https://data.geo.admin.ch/api/stac/v0.9"
COLLECTION = "ch.meteoschweiz.ogd-smn"
ASSET_SUFFIX = "h_historical_2020-2029.csv"
REQUIRED_COLUMNS = {
    "station_abbr",
    "reference_timestamp",
    "tre200h0",
    "ure200h0",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, retries: int = 3) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if not payload:
                raise ValueError("empty response")
            return payload, sha256_bytes(payload)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(str(last_error))


def asset_url(station_id: str) -> str:
    item = fetch_json(
        f"{STAC_BASE}/collections/{COLLECTION}/items/{station_id}"
    )
    matches = [
        value["href"]
        for key, value in item.get("assets", {}).items()
        if key.endswith(ASSET_SUFFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one hourly asset for {station_id}, found {matches}"
        )
    return str(matches[0])


def audit_csv(payload: bytes) -> dict[str, object]:
    result: dict[str, object] = {
        "rows": 0,
        "columns": [],
        "required_columns_ok": False,
        "timestamp_min": None,
        "timestamp_max": None,
        "station_ids": [],
        "error": None,
    }
    try:
        frame = pd.read_csv(
            io.BytesIO(payload),
            sep=";",
            dtype=str,
            encoding="utf-8",
        )
        frame.columns = [str(column).strip() for column in frame.columns]
        result["columns"] = frame.columns.tolist()
        result["required_columns_ok"] = REQUIRED_COLUMNS.issubset(
            frame.columns
        )
        result["rows"] = int(len(frame))
        if "station_abbr" in frame:
            result["station_ids"] = sorted(
                frame["station_abbr"].astype(str).str.lower().unique().tolist()
            )
        if "reference_timestamp" in frame:
            timestamps = pd.to_datetime(
                frame["reference_timestamp"].astype(str).str.strip(),
                format="%d.%m.%Y %H:%M",
                errors="coerce",
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
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_DOWNLOAD":
        raise RuntimeError("Stage214 protocol is not frozen before download")
    station_table = pd.read_csv(STATIONS, dtype={"station_id": str})
    station_ids = station_table["station_id"].str.lower().tolist()

    rows: list[dict[str, object]] = []
    for station_id in station_ids:
        target = DATA_DIR / f"ogd-smn_{station_id}_{ASSET_SUFFIX}"
        item: dict[str, object] = {
            "station_id": station_id,
            "url": None,
            "target": str(target),
            "status": "pending",
            "downloaded_at_utc": None,
            "bytes": None,
            "sha256": None,
            "error": None,
        }
        try:
            url = asset_url(station_id)
            item["url"] = url
            if target.exists() and target.stat().st_size > 0:
                payload = target.read_bytes()
                item["status"] = "already_present"
                item["bytes"] = len(payload)
                item["sha256"] = sha256_bytes(payload)
            else:
                payload, digest = download(url)
                target.write_bytes(payload)
                item["status"] = "downloaded"
                item["downloaded_at_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()
                item["bytes"] = len(payload)
                item["sha256"] = digest
            item.update(
                {
                    f"audit_{key}": value
                    for key, value in audit_csv(payload).items()
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
            "阶段215只从瑞士联邦官方STAC下载阶段214冻结的12个"
            "MeteoSwiss SwissMetNet站点的小时历史CSV（2020-2029）；"
            "保留URL、下载时间、字节数、SHA256、字段、行数、站号和时间范围；"
            "不训练模型、不选择阈值、不计算性能；所有后续攻击必须是"
            "透明受控注入，不是真实攻击日志。"
        ),
        "protocol_status": protocol["status"],
        "provider": protocol["provider"],
        "dataset_id": protocol["dataset_id"],
        "license": protocol["license"],
        "station_count": len(station_ids),
        "expected_archives": len(station_ids),
        "downloaded_count": int(detail["status"].eq("downloaded").sum()),
        "already_present_count": int(
            detail["status"].eq("already_present").sum()
        ),
        "failed_count": int(detail["status"].eq("failed").sum()),
        "required_columns_ok_count": int(
            detail["audit_required_columns_ok"].fillna(False).sum()
        ),
        "audit_failure_count": int(detail["audit_error"].notna().sum()),
        "detail": rows,
        "limitations": [
            "阶段215只验证官方下载和CSV结构，不评价模型性能。",
            "MeteoSwiss小时CSV没有与DWD QN_9等价的逐行数值质量码；不得伪造质量码。",
            "空字段按缺失保留，不插补；缺失不等于攻击标签。",
            "所有攻击仍须是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段215 MeteoSwiss官方小时数据下载与归档审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 来源：{payload['provider']}",
        f"- 数据集：`{payload['dataset_id']}`；许可：`{payload['license']}`",
        f"- 冻结站点：`{payload['station_count']}`；预期文件：`{payload['expected_archives']}`",
        f"- 本次下载：`{payload['downloaded_count']}`；已存在：`{payload['already_present_count']}`；失败：`{payload['failed_count']}`",
        f"- 字段审计通过：`{payload['required_columns_ok_count']}`；审计失败：`{payload['audit_failure_count']}`",
        "",
        "| 站号 | 状态 | 字节 | 行数 | 时间范围 | 字段完整 |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        time_range = (
            f"{row.get('audit_timestamp_min')} -> "
            f"{row.get('audit_timestamp_max')}"
        )
        lines.append(
            f"| {row['station_id']} | {row['status']} | "
            f"{row.get('bytes') or 0} | {row.get('audit_rows') or 0} | "
            f"{time_range} | "
            f"{'是' if row.get('audit_required_columns_ok') else '否'} |"
        )
    lines += [
        "",
        "本阶段只保存官方原始CSV和可追溯审计信息，不训练检测器、",
        "不选择阈值、不计算攻击召回率；空字段不插补，质量缺失不冒充攻击标签。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "expected_archives": payload["expected_archives"],
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
