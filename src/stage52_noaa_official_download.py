from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parents[1] / "dataset" / "external" / "noaa_isd"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
CANDIDATES = TABLES / "stage51_noaa_station_candidates.csv"
REQUIRED_COLUMNS = {"STATION", "DATE", "TMP", "WND", "REPORT_TYPE", "QUALITY_CONTROL"}
USER_AGENT = "TrustMeteo-QC/0.1 (reproducible research; NOAA ISD official download)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, retries: int = 3) -> tuple[str, int]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=60) as response:
                payload = response.read()
            # NCEI's official CSV is quoted (and may include a UTF-8 BOM); validate
            # the normalized header rather than assuming an unquoted first byte.
            header = payload[:512].lstrip(b"\xef\xbb\xbf").splitlines()[0].replace(b'"', b"")
            if not header.startswith(b"STATION,DATE,"):
                raise ValueError("downloaded content does not start with NOAA CSV header")
            target.write_bytes(payload)
            return sha256(target), len(payload)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(str(last_error))


def audit_file(path: Path, station_id: str) -> dict[str, object]:
    result: dict[str, object] = {
        "station_id": station_id,
        "file": str(path),
        "exists": path.exists(),
        "readable": False,
        "rows": 0,
        "columns_ok": False,
        "station_ids": [],
        "date_min": None,
        "date_max": None,
        "quality_columns": [],
        "error": None,
    }
    if not path.exists():
        return result
    try:
        frame = pd.read_csv(path, dtype=str, low_memory=False)
        result["readable"] = True
        result["rows"] = int(len(frame))
        result["columns_ok"] = REQUIRED_COLUMNS.issubset(frame.columns)
        result["station_ids"] = sorted(frame["STATION"].dropna().astype(str).unique().tolist()) if "STATION" in frame else []
        if "DATE" in frame:
            dates = pd.to_datetime(frame["DATE"], errors="coerce", utc=True).dropna()
            if len(dates):
                result["date_min"] = str(dates.min())
                result["date_max"] = str(dates.max())
        result["quality_columns"] = [c for c in ["TMP", "WND", "SLP", "DEW", "VIS", "QUALITY_CONTROL"] if c in frame.columns]
    except Exception as exc:  # audit should record a failure rather than hide it
        result["error"] = repr(exc)
    return result


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(CANDIDATES, dtype=str)
    rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        station_id = str(row["station_id"])
        url = str(row["official_2022_url"])
        target = DATA_DIR / f"{station_id}.csv"
        item: dict[str, object] = {
            "station_id": station_id,
            "station_name": str(row["STATION NAME"]),
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
                item["status"] = "already_present"
                item["bytes"] = int(target.stat().st_size)
                item["sha256"] = sha256(target)
            else:
                digest, size = download(url, target)
                item["status"] = "downloaded"
                item["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()
                item["bytes"] = size
                item["sha256"] = digest
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = repr(exc)
        item.update({f"audit_{k}": v for k, v in audit_file(target, station_id).items() if k not in {"station_id", "file"}})
        rows.append(item)

    detail = pd.DataFrame(rows)
    detail.to_csv(TABLES / "stage52_noaa_official_download.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS" if bool((detail["audit_readable"] & detail["audit_columns_ok"]).all()) else "REVIEW_REQUIRED",
        "protocol": "仅下载NOAA/NCEI官方2022 Global Hourly CSV；保留URL、下载时间、字节数和SHA256；下载后进行字段、站点和时间范围初检。",
        "candidate_count": int(len(detail)),
        "downloaded_count": int(detail["status"].eq("downloaded").sum()),
        "already_present_count": int(detail["status"].eq("already_present").sum()),
        "failed_count": int(detail["status"].eq("failed").sum()),
        "audit_readable_count": int(detail["audit_readable"].sum()),
        "audit_columns_ok_count": int(detail["audit_columns_ok"].sum()),
        "detail": rows,
    }
    (TABLES / "stage52_noaa_official_download.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# 阶段52 NOAA官方扩展站点下载与初检",
        "",
        "仅使用 NOAA/NCEI 官方 2022 Global Hourly 地址；每个文件保存下载状态、SHA256、字段检查和时间覆盖。",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 候选站点：`{payload['candidate_count']}`；本次下载：`{payload['downloaded_count']}`；失败：`{payload['failed_count']}`",
        "",
        "| 站点 | 状态 | 行数 | 字段完整 | 日期起点 | 日期终点 | SHA256前16位 |",
        "|---|---|---:|---|---|---|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item['station_id']} | {item['status']} | {item['audit_rows']} | "
            f"{'是' if item['audit_columns_ok'] else '否'} | {item['audit_date_min']} | {item['audit_date_max']} | "
            f"{str(item['sha256'] or '')[:16]} |"
        )
    lines += ["", "下载失败或字段/时间审计不通过的站点不得进入后续融合实验；原始文件和SHA256清单必须保留。"]
    (REPORTS / "stage52_noaa_official_download.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["status", "candidate_count", "downloaded_count", "failed_count", "audit_columns_ok_count"]}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
