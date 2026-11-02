from __future__ import annotations

"""Stage 284: fetch and audit the author-linked SWaT CSVs.

The USAD notebook points to two public Google Drive file IDs. This stage
downloads only those files, verifies that they are real CSV payloads rather
than Google confirmation pages, and records hashes and structural metadata.
It does not train or evaluate a model.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "raw" / "public_ics_cps"
PREFIX = "stage284_swat_author_data_download"

FILES = {
    "normal": "1rVJ5ry5GG-ZZi5yI4x9lICB8VhErXwCw",
    "attack": "1iDYc0OEmidN712fquOBRFjln90SbpaE7",
}
EXPECTED = {
    "normal": "SWaT_Dataset_Normal_v1.csv",
    "attack": "SWaT_Dataset_Attack_v0.csv",
}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(session: requests.Session, file_id: str, destination: Path) -> dict:
    endpoint = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download"}
    first = session.get(endpoint, params=params, timeout=60)
    first.raise_for_status()
    confirm = re.search(r'name="confirm" value="([^"]+)"', first.text)
    uuid = re.search(r'name="uuid" value="([^"]+)"', first.text)
    if confirm and uuid:
        params["confirm"] = confirm.group(1)
        params["uuid"] = uuid.group(1)
        response = session.get(
            endpoint, params=params, timeout=300, stream=True
        )
    else:
        response = first
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        raise RuntimeError(
            f"Google Drive returned HTML instead of CSV: {file_id}"
        )
    with destination.open("wb") as handle:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                handle.write(chunk)
    return {
        "content_type": content_type,
        "content_length_header": response.headers.get("content-length"),
    }


def _inspect_csv(path: Path) -> dict:
    try:
        frame = pd.read_csv(path, sep=None, engine="python", nrows=2000)
    except Exception:
        frame = pd.read_csv(path, sep=";", nrows=2000)
    return {
        "rows_sampled": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": [str(column) for column in frame.columns],
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    rows = []
    for key, file_id in FILES.items():
        destination = RAW / EXPECTED[key]
        try:
            metadata = _download(session, file_id, destination)
            inspection = _inspect_csv(destination)
            status = "DOWNLOADED_AND_CSV_AUDITED"
            error = None
        except Exception as exc:
            metadata = {}
            inspection = {}
            status = "DOWNLOAD_OR_AUDIT_FAILED"
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "dataset": key,
                "google_drive_file_id": file_id,
                "expected_filename": EXPECTED[key],
                "status": status,
                "bytes": destination.stat().st_size
                if destination.is_file()
                else 0,
                "sha256": _sha256(destination)
                if destination.is_file()
                else None,
                "content_type": metadata.get("content_type"),
                "content_length_header": metadata.get(
                    "content_length_header"
                ),
                "rows_sampled": inspection.get("rows_sampled"),
                "columns": inspection.get("columns"),
                "column_names": inspection.get("column_names", []),
                "error": error,
            }
        )
    passed = all(row["status"] == "DOWNLOADED_AND_CSV_AUDITED" for row in rows)
    payload = {
        "status": (
            "PASS_AUTHOR_SWAT_DATA_DOWNLOADED_AND_AUDITED"
            if passed
            else "BOUNDARY_AUTHOR_SWAT_DATA_NOT_OBTAINED"
        ),
        "protocol": (
            "Download only the two Google Drive IDs declared by the fixed "
            "USAD notebook; reject HTML confirmation pages; record SHA256 "
            "and structural CSV metadata. No model training or evaluation."
        ),
        "source_notebook": "vendor/USAD.ipynb",
        "rows": rows,
        "paper_faithful_reproduction_ready": passed,
        "p0_3_status_after": "IN_PROGRESS" if passed else "OPEN",
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(
        TABLES / f"{PREFIX}.csv", index=False, encoding="utf-8-sig"
    )
    lines = [
        "# Stage 284 USAD 作者链接 SWaT 数据下载审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 论文忠实复现就绪：`{payload['paper_faithful_reproduction_ready']}`",
        f"- P0-3：`{payload['p0_3_status_after']}`",
        "",
        "| dataset | status | bytes | rows sampled | columns | SHA256 |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| {row['dataset']} | {row['status']} | {row['bytes']} | "
        f"{row['rows_sampled']} | {row['columns']} | `{row['sha256']}` |"
        for row in rows
    )
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "paper_faithful_reproduction_ready": (
                    payload["paper_faithful_reproduction_ready"]
                ),
                "p0_3_status_after": payload["p0_3_status_after"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
