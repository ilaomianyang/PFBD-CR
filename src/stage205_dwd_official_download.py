from __future__ import annotations

"""Stage 205: download the pre-registered DWD station archives.

The station list and product scope are read from the Stage-204 protocol.
Only DWD's official HTTPS OpenData historical directory is used.  Raw
archives are retained, and every file records its URL, download time,
byte count, and SHA256 digest.  The stage validates the archive layout
and required CSV columns but does not train a detector or evaluate
performance.
"""

import hashlib
import io
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
DATA_DIR = ROOT.parents[1] / "dataset" / "external" / "dwd"
PROTOCOL = TABLES / "stage204_dwd_protocol_preregistration.json"
STATIONS = TABLES / "stage204_dwd_protocol_preregistration_stations.csv"
PREFIX = "stage205_dwd_official_download"
USER_AGENT = "TrustMeteo-QC/0.1 (reproducible research; DWD official OpenData)"
PRODUCTS = {
    "TU": {
        "directory": "air_temperature",
        "required_columns": {"STATIONS_ID", "MESS_DATUM", "QN_9", "TT_TU", "RF_TU"},
        "role": "primary",
        "primary_variables": ["TT_TU", "RF_TU"],
        "quality_field": "QN_9",
    },
    "TF": {
        "directory": "moisture",
        "required_columns": {
            "STATIONS_ID",
            "MESS_DATUM",
            "QN_8",
            "TF_STD",
            "RF_STD",
        },
        "role": "auxiliary",
        "primary_variables": [],
        "quality_field": "QN_8",
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download(url: str, retries: int = 3) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if not payload:
                raise ValueError("empty response")
            return payload, sha256_bytes(payload)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(str(last_error))


def product_base_url(product: str) -> str:
    directory = PRODUCTS[product]["directory"]
    return (
        "https://opendata.dwd.de/climate_environment/CDC/"
        f"observations_germany/climate/hourly/{directory}/historical/"
    )


def discover_archive_url(product: str, station_id: str) -> str:
    base = product_base_url(product)
    prefix = f"stundenwerte_{product}_{station_id}_"
    listing = urllib.request.urlopen(
        urllib.request.Request(base, headers={"User-Agent": USER_AGENT}),
        timeout=90,
    ).read().decode("utf-8", errors="replace")
    matches = [
        line.split('href="', 1)[1].split('"', 1)[0]
        for line in listing.splitlines()
        if prefix in line and "_hist.zip" in line
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one historical archive for {product}/{station_id}, "
            f"found {matches}"
        )
    return base + matches[0]


def find_existing_archive(
    product_dir: Path, product: str, station_id: str
) -> Path | None:
    matches = sorted(
        product_dir.glob(
            f"stundenwerte_{product}_{station_id}_*_hist.zip"
        )
    )
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple local archives for {product}/{station_id}: "
            f"{[path.name for path in matches]}"
        )
    return matches[0] if matches else None


def audit_archive(payload: bytes, product: str) -> dict[str, object]:
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
                name
                for name in names
                if name.lower().endswith(".txt")
                and "produkt" in name.lower()
            ]
            result["data_members"] = data_members
            if not data_members:
                raise ValueError("no product data member found")
            frames = [
                pd.read_csv(
                    archive.open(name),
                    sep=";",
                    dtype=str,
                    encoding="latin1",
                )
                for name in data_members
            ]
            frame = pd.concat(frames, ignore_index=True)
            frame.columns = [str(column).strip() for column in frame.columns]
            result["columns"] = frame.columns.tolist()
            result["required_columns_ok"] = (
                PRODUCTS[product]["required_columns"].issubset(frame.columns)
            )
            result["rows"] = int(len(frame))
            if "MESS_DATUM" in frame:
                timestamps = pd.to_datetime(
                    frame["MESS_DATUM"].astype(str).str.strip(),
                    format="%Y%m%d%H",
                    errors="coerce",
                    utc=True,
                )
                result["timestamp_min"] = (
                    None if timestamps.dropna().empty else str(timestamps.min())
                )
                result["timestamp_max"] = (
                    None if timestamps.dropna().empty else str(timestamps.max())
                )
            if "STATIONS_ID" in frame:
                result["station_ids"] = sorted(
                    frame["STATIONS_ID"].astype(str).str.zfill(5).unique().tolist()
                )
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    station_table = pd.read_csv(STATIONS, dtype={"station_id": str})
    station_ids = station_table["station_id"].astype(str).str.zfill(5).tolist()
    rows: list[dict[str, object]] = []
    for product, settings in PRODUCTS.items():
        product_dir = DATA_DIR / product
        product_dir.mkdir(parents=True, exist_ok=True)
        for station_id in station_ids:
            target = find_existing_archive(
                product_dir, product, station_id
            )
            if target is None:
                url = discover_archive_url(product, station_id)
                filename = url.rsplit("/", 1)[-1]
                target = product_dir / filename
            else:
                url = product_base_url(product) + target.name
            item: dict[str, object] = {
                "product": product,
                "role": settings["role"],
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
                        for key, value in audit_archive(payload, product).items()
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
        "status": "PASS_PIPELINE_REVIEW_REQUIRED" if bool(passed.all()) else "REVIEW_REQUIRED",
        "protocol": (
            "阶段205只从DWD官方OpenData历史目录下载阶段204冻结的"
            "12个站点、TU主产品和TF辅助归档；主分析变量是TU中的TT_TU和"
            "RF_TU，TF中的RF_STD只作为辅助归档保留；保留URL、下载时间、字节数、"
            "SHA256、压缩包成员、字段、时间范围和站号；不训练模型、"
            "不选择阈值、不计算性能。"
        ),
        "protocol_status": protocol["status"],
        "station_count": len(station_ids),
        "product_count": len(PRODUCTS),
        "expected_archives": len(station_ids) * len(PRODUCTS),
        "primary_archives": len(station_ids),
        "auxiliary_archives": len(station_ids) * (len(PRODUCTS) - 1),
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
        "detail": rows,
        "limitations": [
            "阶段205只验证下载和归档结构，不评价模型性能。",
            "DWD质量码不是攻击标签；TU主产品的低质量行将在后续接入阶段按预注册规则处理。",
            "TF归档是辅助归档，不进入主分析，也不替代TU中的RF_TU。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段205 DWD官方历史文件下载与归档审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 冻结站点：`{len(station_ids)}`；产品：`{len(PRODUCTS)}`；预期归档：`{payload['expected_archives']}`",
        f"- 主产品归档：`{payload['primary_archives']}`；辅助归档：`{payload['auxiliary_archives']}`",
        f"- 本次下载：`{payload['downloaded_count']}`；已存在：`{payload['already_present_count']}`；失败：`{payload['failed_count']}`",
        f"- 字段审计通过：`{payload['required_columns_ok_count']}`；归档审计失败：`{payload['archive_audit_failure_count']}`",
        "",
        "| 产品 | 用途 | 站号 | 状态 | 字节 | 行数 | 时间范围 | 字段完整 |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        time_range = (
            f"{row.get('audit_timestamp_min')} -> "
            f"{row.get('audit_timestamp_max')}"
        )
        lines.append(
            f"| {row['product']} | {row['role']} | {row['station_id']} | {row['status']} | "
            f"{row.get('bytes') or 0} | {row.get('audit_rows') or 0} | "
            f"{time_range} | "
            f"{'是' if row.get('audit_required_columns_ok') else '否'} |"
        )
    lines += [
        "",
        "本阶段只保存DWD官方原始归档和可追溯审计信息，不训练检测器、",
        "不选择阈值、不计算攻击召回率。所有后续攻击仍须是透明受控注入。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "expected_archives": payload["expected_archives"],
                "primary_archives": payload["primary_archives"],
                "auxiliary_archives": payload["auxiliary_archives"],
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
