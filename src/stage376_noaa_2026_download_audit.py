from __future__ import annotations

"""Download and audit the frozen NOAA ISD 2026 partial-year holdout.

Only the 12 frozen Japan stations and 2026 official NCEI archives are
requested.  No model, threshold, attack label, or performance result is
created here.
"""

import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "raw" / "noaa_isd_japan_2026"
PROTOCOL = ROOT / "confirmatory_package" / "noaa_2026_temporal" / "protocol_freeze.json"
SOURCE_NORMALIZED = TABLES / "stage296_japan_isd_official_download_normalized.parquet"
OUT_PREFIX = "stage376_noaa_2026_download_audit"
USER_AGENT = "TrustMeteo-QC reproducible research NOAA ISD 2026 audit"
OFFICIAL_ROOT = "https://www.ncei.noaa.gov/data/global-hourly/access"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from stage296_japan_isd_official_download import parse_station_year  # noqa: E402


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, retries: int = 4, timeout: int = 180) -> tuple[bytes, dict[str, str]]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = response.read()
                headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
            if not payload:
                raise RuntimeError(f"empty response: {url}")
            return payload, headers
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(2 + attempt * 2)
    raise RuntimeError(f"download failed: {url}") from last


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_2026_OBSERVATION_ACCESS":
        raise RuntimeError("Stage375 protocol is not frozen")
    if protocol["observations_accessed_before_freeze"] is not False:
        raise RuntimeError("Stage375 observation boundary changed")
    stations = [str(x) for x in protocol["station_ids"]]
    RAW.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    parts: list[pd.DataFrame] = []
    for station_id in stations:
        url = f"{OFFICIAL_ROOT}/2026/{station_id}.csv"
        path = RAW / f"{station_id}_2026.csv"
        started = datetime.now(timezone.utc).isoformat()
        error_text = ""
        try:
            if path.is_file():
                payload = path.read_bytes()
                headers = {"source": "cache"}
                status = "cached"
            else:
                payload, headers = download(url)
                path.write_bytes(payload)
                status = "downloaded"
            parsed = parse_station_year(payload, station_id, 2026, "1")
        except Exception as exc:
            payload = b""
            headers = {}
            parsed = pd.DataFrame()
            status = "failed"
            error_text = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "station_id": station_id,
                "year": 2026,
                "url": url,
                "status": status,
                "retrieved_at_utc": started,
                "bytes": int(len(payload)),
                "sha256": sha256_bytes(payload),
                "path": str(path),
                "parsed_rows": int(len(parsed)),
                "observation_present_rows": int(parsed["observation_present"].sum())
                if not parsed.empty
                else 0,
                "date_min": str(parsed["timestamp_utc"].min()) if not parsed.empty else "",
                "date_max": str(parsed["timestamp_utc"].max()) if not parsed.empty else "",
                "headers": headers,
                "error": error_text,
            }
        )
        if not parsed.empty:
            parts.append(parsed)
        print(json.dumps({"station_id": station_id, "status": status, "rows": len(parsed), "error": error_text}, ensure_ascii=False), flush=True)

    new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if new.empty:
        combined = pd.read_parquet(SOURCE_NORMALIZED)
        combined = combined.iloc[0:0].copy()
        combined["split"] = pd.Series(dtype="object")
        duplicates = 0
        station_audit = pd.DataFrame(
            [
                {
                    "station_id": station_id,
                    "year": 2026,
                    "split": "test",
                    "rows": 0,
                    "observation_present_rows": 0,
                    "coverage": 0.0,
                    "date_min": "",
                    "date_max": "",
                    "duplicate_timestamps": 0,
                }
                for station_id in stations
            ]
        )
        normalized_out = TABLES / f"{OUT_PREFIX}_normalized.parquet"
        combined.to_parquet(normalized_out, index=False)
        detail_out = TABLES / f"{OUT_PREFIX}_detail.json"
        detail_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        station_out = TABLES / f"{OUT_PREFIX}_station_audit.csv"
        station_audit.to_csv(station_out, index=False, encoding="utf-8-sig")
        payload = {
            "status": "FAIL_NOAA_2026_DOWNLOAD_AND_COVERAGE_AUDIT",
            "protocol_id": protocol["protocol_id"],
            "station_count": len(stations),
            "years": [2021, 2022, 2023, 2024, 2026],
            "test_period": [protocol["test_period_start_utc"], protocol["test_period_end_utc"]],
            "downloaded_2026_files": rows,
            "normalized_rows": 0,
            "test_rows": 0,
            "test_station_count": 0,
            "test_min_coverage": 0.0,
            "test_max_coverage": 0.0,
            "duplicate_station_timestamp_rows": 0,
            "coverage_failures": station_audit.to_dict("records"),
            "performance_status": "NOT_EVALUATED",
            "labels_inspected": False,
            "protocol_sha256": sha256_file(PROTOCOL),
            "normalized_sha256": sha256_file(normalized_out),
            "source_normalized_sha256": sha256_file(SOURCE_NORMALIZED),
            "limitations": [
                "2026 官方归档未能下载或解析，不能进入性能评价。",
                "这是数据可行性失败，不是算法性能结果。",
                "同一 NOAA provider 和日本站网，不能证明 provider 独立性。",
            ],
        }
        (TABLES / f"{OUT_PREFIX}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (REPORTS / f"{OUT_PREFIX}.md").write_text(
            "\n".join(
                [
                    "# Stage376 NOAA ISD 2026 下载与覆盖率审计",
                    "",
                    "- 状态：`FAIL_NOAA_2026_DOWNLOAD_AND_COVERAGE_AUDIT`",
                    "- 2026 官方年度归档未能形成可评价的测试数据。",
                    "- 本阶段没有训练、阈值校准、攻击注入或性能评价。",
                    "- 不得将该失败解释为 Trust 或任何基线的性能失败。",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": payload["status"], "test_rows": 0, "test_min_coverage": 0.0}, ensure_ascii=False))
        return 2
    start = pd.Timestamp(protocol["test_period_start_utc"])
    end = pd.Timestamp(protocol["test_period_end_utc"])
    new = new[(new["timestamp_utc"] >= start) & (new["timestamp_utc"] <= end)].copy()
    new["year"] = 2026
    new["split"] = "test"

    old = pd.read_parquet(SOURCE_NORMALIZED)
    old["station_id"] = old["station_id"].astype(str)
    old = old[
        old["station_id"].isin(stations) & old["year"].isin([2021, 2022, 2023, 2024])
    ].copy()
    split_map = {2021: "train", 2022: "train", 2023: "train", 2024: "valid"}
    old["split"] = old["year"].map(split_map)
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    combined = combined.sort_values(["station_id", "timestamp_utc"], kind="stable").reset_index(drop=True)
    duplicates = int(combined.duplicated(["station_id", "timestamp_utc"]).sum())
    station_rows = []
    for (station_id, year), part in combined.groupby(["station_id", "year"], sort=True):
        present = int(part["observation_present"].sum())
        station_rows.append(
            {
                "station_id": str(station_id),
                "year": int(year),
                "split": str(part["split"].iloc[0]),
                "rows": int(len(part)),
                "observation_present_rows": present,
                "coverage": float(present / len(part)) if len(part) else 0.0,
                "date_min": str(part["timestamp_utc"].min()),
                "date_max": str(part["timestamp_utc"].max()),
                "duplicate_timestamps": int(part.duplicated(["station_id", "timestamp_utc"]).sum()),
            }
        )
    audit = pd.DataFrame(station_rows)
    test_audit = audit[audit["year"].eq(2026)]
    failures = audit[(audit["coverage"] < 0.70) | audit["rows"].eq(0)].to_dict("records")
    performance_status = "NOT_EVALUATED"
    download_status = "PASS_NOAA_2026_DOWNLOAD_AND_COVERAGE_AUDIT" if not failures and duplicates == 0 else "FAIL_NOAA_2026_COVERAGE_AUDIT"
    normalized_out = TABLES / f"{OUT_PREFIX}_normalized.parquet"
    combined.to_parquet(normalized_out, index=False)
    detail_out = TABLES / f"{OUT_PREFIX}_detail.json"
    detail_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    station_out = TABLES / f"{OUT_PREFIX}_station_audit.csv"
    audit.to_csv(station_out, index=False, encoding="utf-8-sig")
    payload = {
        "status": download_status,
        "protocol_id": protocol["protocol_id"],
        "station_count": len(stations),
        "years": [2021, 2022, 2023, 2024, 2026],
        "test_period": [protocol["test_period_start_utc"], protocol["test_period_end_utc"]],
        "downloaded_2026_files": rows,
        "normalized_rows": int(len(combined)),
        "test_rows": int(len(new)),
        "test_station_count": int(test_audit["station_id"].nunique()),
        "test_min_coverage": float(test_audit["coverage"].min()) if not test_audit.empty else 0.0,
        "test_max_coverage": float(test_audit["coverage"].max()) if not test_audit.empty else 0.0,
        "duplicate_station_timestamp_rows": duplicates,
        "coverage_failures": failures,
        "performance_status": performance_status,
        "labels_inspected": False,
        "protocol_sha256": sha256_file(PROTOCOL),
        "normalized_sha256": sha256_file(normalized_out),
        "source_normalized_sha256": sha256_file(SOURCE_NORMALIZED),
        "limitations": [
            "同一 NOAA provider 和日本站网的新时间窗，不是新 provider。",
            "2026 test 仅覆盖冻结的部分年度，不能代表全年。",
            "本阶段没有训练、阈值校准、攻击注入或性能评价。",
            "若覆盖率门失败，不得进入性能运行。",
        ],
    }
    (TABLES / f"{OUT_PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage376 NOAA ISD 2026 下载与覆盖率审计",
        "",
        f"- 状态：`{download_status}`",
        f"- 2026 文件：`{len(rows)}`；测试行：`{len(new)}`；测试站点：`{test_audit['station_id'].nunique() if not test_audit.empty else 0}`",
        f"- 测试覆盖率范围：`{payload['test_min_coverage']:.4f}` 至 `{payload['test_max_coverage']:.4f}`",
        f"- 重复站点时间戳：`{duplicates}`",
        "- 本阶段仅做官方归档下载和覆盖率审计，没有性能结果。",
    ]
    (REPORTS / f"{OUT_PREFIX}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": download_status, "test_rows": len(new), "test_min_coverage": payload["test_min_coverage"]}, ensure_ascii=False))
    return 0 if download_status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
