"""Build the final submission manifest with SHA256 hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
RELEASE = ROOT / "release" / "TrustMeteo-QC_v1.0"
OUT = FINAL / "submission_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    files = sorted(
        path
        for path in FINAL.rglob("*")
        if path.is_file() and path != OUT
    )
    tree_digest = hashlib.sha256()
    if RELEASE.is_dir():
        for path in sorted(RELEASE.rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(RELEASE))
                tree_digest.update(relative.encode("utf-8"))
                tree_digest.update(b"\0")
                tree_digest.update(bytes.fromhex(sha256(path)))
    manifest = {
        "package": "TrustMeteo-QC final submission package",
        "date": "2026-10-09",
        "status": "FINAL_PACKAGE_MANIFEST",
        "code_release_tree_sha256": tree_digest.hexdigest() if RELEASE.is_dir() else None,
        "code_release_file_count": sum(
            1 for path in RELEASE.rglob("*") if path.is_file()
        )
        if RELEASE.is_dir()
        else 0,
        "files": [
            {
                "path": str(path.relative_to(FINAL)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
        "rules": [
            "V6 remains the primary frozen evidence layer.",
            "V7 and Stage466/467 remain exploratory.",
            "FMI/SMHI negative or mixed transfers remain visible.",
            "All attacks are controlled simulations, not production incidents.",
            "No real-production attack recall is claimed.",
        ],
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(OUT), "files": len(files)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
