from __future__ import annotations

"""Stage 283: audit the fixed-commit USAD source boundary.

This stage verifies the provenance of the public ICS/CPS baseline source
files that are present locally. It does not train, evaluate, tune, or infer
missing data. It records whether the author-linked SWaT CSVs are available;
P0-3 remains open because the same-contract baseline comparison is not
complete even when the source and data are available.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
VENDOR = ROOT / "vendor"
PREFIX = "stage283_public_ics_cps_source_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import sha256_file  # noqa: E402


COMMIT = "e25af45c8e1c32783aed94fc7e5ab85effef2b1a"
REPOSITORY = "https://github.com/manigalati/usad"
EXPECTED_BLOBS = {
    "LICENSE": {
        "git_blob_sha": "489b5ca4c18e415aa4debac90c8f2e9fd2521cb5",
        "bytes": 1501,
        "role": "license",
    },
    "USAD.ipynb": {
        "git_blob_sha": "dcb68b75b27073a89d7cb596dbe164555b47cb22",
        "bytes": 73031,
        "role": "author_training_and_data_contract",
    },
    "usad.py": {
        "git_blob_sha": "2d8a1d17df47e4a099b47709da2f317710a489ba",
        "bytes": 4149,
        "role": "author_model_and_training_code",
    },
    "utils.py": {
        "git_blob_sha": "3a8a28fee5a9d22e11c727cf0e1c454f23d2d495",
        "bytes": 2164,
        "role": "author_utility_code",
    },
}


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows: list[dict[str, object]] = []

    for filename, expected in EXPECTED_BLOBS.items():
        path = VENDOR / filename
        if path.is_file():
            actual_bytes = path.stat().st_size
            actual_blob = _git_blob_sha(path)
            verified = (
                actual_bytes == expected["bytes"]
                and actual_blob == expected["git_blob_sha"]
            )
            status = "VERIFIED_FIXED_COMMIT_BLOB" if verified else "HASH_MISMATCH"
            sha256 = sha256_file(path)
        else:
            actual_bytes = None
            actual_blob = None
            verified = False
            status = "MISSING"
            sha256 = None
        rows.append(
            {
                "artifact": filename,
                "role": expected["role"],
                "status": status,
                "verified": verified,
                "expected_bytes": expected["bytes"],
                "actual_bytes": actual_bytes,
                "expected_git_blob_sha": expected["git_blob_sha"],
                "actual_git_blob_sha": actual_blob,
                "sha256": sha256,
            }
        )

    notebook_path = VENDOR / "USAD.ipynb"
    notebook_text = (
        notebook_path.read_text(encoding="utf-8")
        if notebook_path.is_file()
        else ""
    )
    data_contract_observed = (
        "SWaT_Dataset_Normal_v1.csv" in notebook_text
        and "SWaT_Dataset_Attack_v0.csv" in notebook_text
        and "window_size=12" in notebook_text.replace(" ", "")
    )
    drive_links_observed = (
        "1rVJ5ry5GG-ZZi5yI4x9lICB8VhErXwCw" in notebook_text
        and "1iDYc0OEmidN712fquOBRFjln90SbpaE7" in notebook_text
    )

    swat_files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and "SWaT" in path.name
        and path.suffix.lower() in {".csv", ".parquet", ".npy", ".npz"}
    ]
    local_swat_data_available = bool(swat_files)

    stage277 = _read_json(TABLES / "stage277_baseline_formality_matrix.json")

    verified_count = sum(bool(row["verified"]) for row in rows)
    required_author_sources_verified = verified_count == len(EXPECTED_BLOBS)
    paper_faithful_reproduction_ready = (
        required_author_sources_verified and local_swat_data_available
    )
    if required_author_sources_verified and local_swat_data_available:
        status = "PASS_AUTHOR_SOURCE_AND_DATA_READY"
    elif required_author_sources_verified:
        status = "BOUNDARY_AUTHOR_SOURCE_VERIFIED_DATA_NOT_AVAILABLE"
    else:
        status = "BOUNDARY_PUBLIC_ICS_CPS_SOURCE_NOT_READY"

    payload = {
        "status": status,
        "protocol": (
            "Stage283 reads the fixed USAD commit metadata, local vendored "
            "files, and the author notebook only. It verifies Git blob hashes "
            "and records the missing SWaT data boundary. It does not train, "
            "evaluate, tune, or inspect any test labels."
        ),
        "repository": REPOSITORY,
        "commit": COMMIT,
        "verified_artifact_count": int(verified_count),
        "expected_artifact_count": len(EXPECTED_BLOBS),
        "required_author_sources_verified": required_author_sources_verified,
        "author_data_contract_observed": data_contract_observed,
        "author_drive_links_observed": drive_links_observed,
        "local_swat_data_available": local_swat_data_available,
        "paper_faithful_reproduction_ready": paper_faithful_reproduction_ready,
        "p0_3_status_before": stage277.get("p0_3_status"),
        "p0_3_status_after": "OPEN",
        "p0_3_checkbox_updated": False,
        "claim_ceiling": (
            "The fixed-commit author source and author-linked SWaT data are "
            "available for provenance and reproduction. This does not by "
            "itself complete P0-3: the same-contract comparison and the "
            "remaining baseline requirements must still be satisfied."
            if paper_faithful_reproduction_ready
            else (
                "The fixed-commit author source is available for provenance "
                "audit, but no paper-faithful USAD evaluation is possible "
                "until the author-linked SWaT data are obtained under their "
                "access terms."
            )
        ),
        "rows": rows,
        "missing_data": [
            {
                "dataset": "SWaT",
                "expected_files": [
                    "SWaT_Dataset_Normal_v1.csv",
                    "SWaT_Dataset_Attack_v0.csv",
                ],
                "source_declared_by_author_notebook": (
                    "Google Drive links recorded in USAD.ipynb"
                ),
                "local_files_found": [str(path) for path in swat_files],
            }
        ],
        "failed_probe_artifacts": [
            {
                "path": "vendor/usad-e25af45.tar.gz",
                "status": "TRUNCATED_DOWNLOAD_NOT_SOURCE_EVIDENCE",
                "sha256": (
                    sha256_file(VENDOR / "usad-e25af45.tar.gz")
                    if (VENDOR / "usad-e25af45.tar.gz").is_file()
                    else None
                ),
            },
            {
                "path": "vendor/README.md",
                "status": "ZERO_BYTE_DOWNLOAD_NOT_SOURCE_EVIDENCE",
            },
        ],
        "limitations": [
            "Verified author source files are not a trained baseline result.",
            "The author notebook points to SWaT CSVs; their local availability is recorded above.",
            "No proxy result may be relabeled as a paper-faithful USAD result.",
            "P0-3 remains open until the same-contract baseline comparison is complete.",
        ],
        "input_sha256": {
            "stage277_baseline_formality_matrix": sha256_file(
                TABLES / "stage277_baseline_formality_matrix.json"
            ),
            "usad_tree": sha256_file(VENDOR / "usad-tree.json"),
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }

    frame = pd.DataFrame(rows)
    frame.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Stage 283 公开 ICS/CPS 作者源码来源审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 固定提交：`{COMMIT}`",
        f"- 已验证作者文件：`{verified_count}/{len(EXPECTED_BLOBS)}`",
        f"- notebook 数据契约已观察：`{data_contract_observed}`",
        f"- 本地 SWaT 数据可用：`{local_swat_data_available}`",
        f"- 论文忠实复现就绪：`{paper_faithful_reproduction_ready}`",
        f"- P0-3：`{payload['p0_3_status_after']}`；复选框未更新。",
        "",
        "## 已验证文件",
        "",
    ]
    for row in rows:
        report_lines.append(
            f"- `{row['artifact']}`：`{row['status']}`；"
            f"Git blob `{row['actual_git_blob_sha']}`。"
        )
    report_lines += [
        "",
        "## 边界",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "verified_artifacts": verified_count,
                "local_swat_data_available": local_swat_data_available,
                "paper_faithful_reproduction_ready": (
                    paper_faithful_reproduction_ready
                ),
                "p0_3_status_after": payload["p0_3_status_after"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
