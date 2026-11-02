from __future__ import annotations

"""Stage 273: release-package anonymization and portability audit.

The audit scans the release package for local absolute paths, user-name
tokens, obvious author identifiers, and cache/bytecode files. Findings are
reported as a table; this stage does not silently edit release files.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RELEASE = ROOT / "release" / "TrustMeteo-QC_v1.0"
PREFIX = "stage273_release_anonymization_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


PATTERNS = {
    "windows_absolute_path": re.compile(
        r"(?i)[A-Z]:\\[^\\\r\n\"' {}\[\]]+\\[^\\\r\n\"' {}\[\]]+"
    ),
    "user_name_token": re.compile(r"(?i)\b" + "ila" + "om" + r"\b"),
    "author_name_token": re.compile("范" + "明洋"),
    "email_like": re.compile(
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    ),
}

SCAN_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".txt",
    ".ps1",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".csv",
}


def _scan() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    self_outputs = {
        (TABLES / f"{PREFIX}.csv").resolve(),
        (TABLES / f"{PREFIX}.json").resolve(),
        (RELEASE / "tables" / f"{PREFIX}.csv").resolve(),
        (RELEASE / "tables" / f"{PREFIX}.json").resolve(),
    }
    for path in sorted(RELEASE.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() in self_outputs:
            continue
        if path.suffix.lower() not in SCAN_SUFFIXES:
            if path.suffix.lower() in {".pyc", ".pyo"}:
                rows.append(
                    {
                        "relative_path": str(path.relative_to(RELEASE)),
                        "check": "bytecode_or_cache",
                        "line": 0,
                        "match": path.suffix.lower(),
                        "severity": "P1_RELEASE_HYGIENE",
                    }
                )
            continue
        if "__pycache__" in path.parts:
            rows.append(
                {
                    "relative_path": str(path.relative_to(RELEASE)),
                    "check": "bytecode_or_cache",
                    "line": 0,
                    "match": "__pycache__",
                    "severity": "P1_RELEASE_HYGIENE",
                }
            )
            continue
        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for check, pattern in PATTERNS.items():
                match = pattern.search(line)
                if match:
                    rows.append(
                        {
                            "relative_path": str(
                                path.relative_to(RELEASE)
                            ),
                            "check": check,
                            "line": line_number,
                            "match": match.group(0),
                            "severity": (
                                "P0_IDENTITY_OR_LOCAL_PATH"
                                if check
                                in {"user_name_token", "author_name_token"}
                                else "P1_PORTABILITY"
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not RELEASE.is_dir():
        raise FileNotFoundError(RELEASE)
    findings = _scan()
    if findings.empty:
        findings = pd.DataFrame(
            columns=[
                "relative_path",
                "check",
                "line",
                "match",
                "severity",
            ]
        )
    findings.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    path_findings = findings.loc[
        findings["check"].eq("windows_absolute_path")
    ]
    identity_findings = findings.loc[
        findings["severity"].eq("P0_IDENTITY_OR_LOCAL_PATH")
    ]
    cache_findings = findings.loc[
        findings["check"].eq("bytecode_or_cache")
    ]
    status = (
        "PASS_RELEASE_ANONYMIZATION_AUDIT"
        if findings.empty
        else "REVIEW_REQUIRED_RELEASE_ANONYMIZATION"
    )
    payload = {
        "status": status,
        "protocol": (
            "Stage273 scans the release directory for local absolute paths, "
            "user/author identity tokens, email-like strings, and bytecode "
            "cache files. It does not edit release files."
        ),
        "release_path": str(RELEASE.relative_to(ROOT)),
        "finding_rows": int(len(findings)),
        "absolute_path_rows": int(len(path_findings)),
        "identity_rows": int(len(identity_findings)),
        "cache_rows": int(len(cache_findings)),
        "clean": bool(findings.empty),
        "findings": findings.to_dict("records"),
        "limitations": [
            "A clean scan is necessary but not sufficient for formal "
            "double-blind submission anonymization.",
            "The scan covers text files and cache filenames, not binary "
            "metadata.",
            "Release files must still be reviewed for proprietary or "
            "restricted raw data.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report = [
        "# 阶段273 Release 匿名化与可移植性审计",
        "",
        f"- 状态：`{status}`",
        f"- 发现问题数：`{payload['finding_rows']}`",
        f"- 本机绝对路径：`{payload['absolute_path_rows']}`",
        f"- 身份标识：`{payload['identity_rows']}`",
        f"- 缓存/字节码：`{payload['cache_rows']}`",
        "",
        "## 边界",
        "",
        "- 清洁扫描是匿名投稿的必要条件，不是充分条件。",
        "- 本阶段不自动修改 release 文件；发现问题后必须单独修复并重跑。",
        "- 受限原始数据、许可和二进制元数据仍需人工复核。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "finding_rows": payload["finding_rows"],
                "absolute_path_rows": payload["absolute_path_rows"],
                "identity_rows": payload["identity_rows"],
                "cache_rows": payload["cache_rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
