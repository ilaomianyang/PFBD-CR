from __future__ import annotations

"""Finalize the Stage310 package hashes before opening the access gate."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "confirmatory_package"
TABLES = ROOT / "tables"
DOWNLOAD_SCRIPT = (
    ROOT / "src" / "stage311_met_norway_disjoint_official_download.py"
)
ANALYSIS_SCRIPT = (
    ROOT / "src" / "stage312_met_norway_disjoint_locked_transfer.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not DOWNLOAD_SCRIPT.is_file():
        raise RuntimeError("Stage311 download script is missing")
    if not ANALYSIS_SCRIPT.is_file():
        raise RuntimeError("Stage312 analysis script is missing")

    analysis_path = PACKAGE_ROOT / "analysis_freeze.json"
    access_path = PACKAGE_ROOT / "access_gate.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    access = json.loads(access_path.read_text(encoding="utf-8"))
    finalized_at = datetime.now(timezone.utc).isoformat()

    analysis["download_script"] = (
        "src/stage311_met_norway_disjoint_official_download.py"
    )
    analysis["download_script_sha256"] = sha256_file(DOWNLOAD_SCRIPT)
    analysis["analysis_script_sha256"] = sha256_file(ANALYSIS_SCRIPT)
    analysis["analysis_script_present_at_freeze"] = True
    analysis["implementation_frozen_at_utc"] = finalized_at
    analysis_path.write_text(
        json.dumps(
            analysis,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    access["status"] = "ACCESS_GATE_OPENED_AFTER_FREEZE"
    access["gate_opened_utc"] = finalized_at
    access["download_script_sha256"] = sha256_file(DOWNLOAD_SCRIPT)
    access["analysis_script_sha256"] = sha256_file(ANALYSIS_SCRIPT)
    access_path.write_text(
        json.dumps(
            access,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = {
        "status": "PASS_DISJOINT_PACKAGE_IMPLEMENTATION_FROZEN",
        "download_script_sha256": sha256_file(DOWNLOAD_SCRIPT),
        "analysis_script_sha256": sha256_file(ANALYSIS_SCRIPT),
        "analysis_freeze_sha256": sha256_file(analysis_path),
        "access_gate_sha256": sha256_file(access_path),
        "gate_opened_utc": finalized_at,
    }
    (TABLES / "stage310a_disjoint_package_finalize.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
