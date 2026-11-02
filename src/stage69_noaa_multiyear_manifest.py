from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parents[1] / "dataset" / "external" / "noaa_isd" / "multiyear"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
INVENTORY = TABLES / "stage51_noaa_station_inventory.csv"
USER_AGENT = "TrustMeteo-QC/0.1 (reproducible research; NOAA ISD official multiyear download)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> tuple[str, int]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    header = payload[:512].lstrip(b"\xef\xbb\xbf").splitlines()[0].replace(b'"', b"")
    if not header.startswith(b"STATION,DATE,"):
        raise ValueError("downloaded content does not start with NOAA CSV header")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return sha256(target), len(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段69：建立NOAA多年度官方文件清单；默认不下载")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2023])
    parser.add_argument("--download", action="store_true", help="下载官方文件；网络不可用时记录失败")
    args = parser.parse_args()
    stations = pd.read_csv(INVENTORY, dtype=str)
    rows: list[dict[str, object]] = []
    for year in args.years:
        for _, station in stations.iterrows():
            station_id = str(station["station_id"])
            url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_id}.csv"
            target = DATA_DIR / str(year) / f"{station_id}.csv"
            item: dict[str, object] = {"year": year, "station_id": station_id, "station_name": str(station["STATION NAME"]), "url": url, "target": str(target), "status": "missing", "checked_at_utc": datetime.now(timezone.utc).isoformat(), "bytes": None, "sha256": None, "error": None}
            if target.exists() and target.stat().st_size > 0:
                item["status"] = "already_present"
                item["bytes"] = int(target.stat().st_size)
                item["sha256"] = sha256(target)
            elif args.download:
                try:
                    digest, size = download(url, target)
                    item["status"] = "downloaded"
                    item["bytes"] = size
                    item["sha256"] = digest
                except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
                    item["status"] = "failed"
                    item["error"] = repr(exc)
            rows.append(item)
    detail = pd.DataFrame(rows)
    status_counts = detail["status"].value_counts().to_dict()
    payload = {"status": "PASS_MANIFEST_ONLY" if not args.download else ("PASS" if not detail["status"].eq("failed").any() else "DOWNLOAD_REVIEW_REQUIRED"), "protocol": "仅生成NOAA/NCEI官方Global Hourly多年度URL清单；2022锁定数据不修改；下载时保存时间、字节数和SHA256。", "years": args.years, "download_requested": bool(args.download), "status_counts": {str(k): int(v) for k, v in status_counts.items()}, "detail": rows}
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage69_noaa_multiyear_manifest.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage69_noaa_multiyear_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段69 NOAA多年度官方数据清单", "", "默认只生成清单，不下载；下载时仅使用NCEI官方Global Hourly地址，并保存SHA256。", "", f"- 状态：**{payload['status']}**；年份：`{args.years}`；下载请求：`{args.download}`", "", "| 年份 | 站点 | 状态 | 官方URL | 目标文件 | SHA256前16位 |", "|---:|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['year']} | {row['station_id']} | {row['status']} | {row['url']} | {row['target']} | {str(row['sha256'] or '')[:16]} |")
    lines += ["", "2022已有文件不被覆盖；下载失败或字段/时间审计不通过的文件不得进入性能实验。"]
    (REPORTS / "stage69_noaa_multiyear_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "years": args.years, "status_counts": payload["status_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
