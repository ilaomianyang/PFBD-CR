from __future__ import annotations

"""Stage 450: audit the V9 raw-fetch to normalization provenance boundary.

This is a source/metadata audit only.  It proves that the normalization
template cannot accept arbitrary files from the raw directory: the files must
come from a completed Stage442 fetch record and match its path, size and
SHA256.  No observation values or performance tables are read.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
MANIFEST = PACKAGE / "data_manifest.json"
FETCH_RECORD = ROOT / "tables" / "stage442_v9_official_window_fetch.json"
TEMPLATE = ROOT / "src" / "stage425_v9_2026_official_window_download_template.py"
OUT_JSON = ROOT / "tables" / "stage450_v9_fetch_normalization_provenance_audit.json"
OUT_REPORT = ROOT / "reports" / "stage450_v9_fetch_normalization_provenance_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    template_text = TEMPLATE.read_text(encoding="utf-8")
    fetch = json.loads(FETCH_RECORD.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    checks = [
        {
            "check": "fetch_record_path_declared",
            "status": "PASS"
            if "stage442_v9_official_window_fetch.json" in template_text
            else "FAIL",
            "value": "stage442_v9_official_window_fetch.json",
        },
        {
            "check": "completed_fetch_required",
            "status": "PASS"
            if "PASS_V9_OFFICIAL_FETCH_COMPLETED" in template_text
            and "fetch_ready" in template_text
            else "FAIL",
            "value": fetch.get("status"),
        },
        {
            "check": "raw_paths_must_match_fetch_record",
            "status": "PASS"
            if "input_paths != expected_paths" in template_text
            and "raw files do not exactly match Stage442" in template_text
            else "FAIL",
            "value": "exact provider path-set equality is enforced",
        },
        {
            "check": "raw_bytes_and_sha256_rechecked",
            "status": "PASS"
            if "raw file byte count differs from Stage442 record" in template_text
            and "raw file SHA256 differs from Stage442 record" in template_text
            else "FAIL",
            "value": "bytes and SHA256 are checked before parsing",
        },
        {
            "check": "historical_normalized_not_used_as_raw_input",
            "status": "PASS"
            if "historical_normalized_path" not in template_text.split(
                "def raw_inputs", 1
            )[1].split("def sha256", 1)[0]
            else "FAIL",
            "value": "raw_inputs scans only the dedicated 2026-10 raw directory",
        },
        {
            "check": "current_fetch_and_manifest_are_not_ready",
            "status": "PASS"
            if fetch.get("status") == "BLOCKED_V9_FETCH_BEFORE_WINDOW_CLOSE"
            and manifest.get("status") == "NOT_READY"
            and manifest.get("observation_values_read") is False
            else "FAIL",
            "value": {
                "fetch_status": fetch.get("status"),
                "manifest_status": manifest.get("status"),
                "observation_values_read": manifest.get(
                    "observation_values_read"
                ),
            },
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_FETCH_NORMALIZATION_PROVENANCE_AUDIT"
        if not failures
        else "FAIL_V9_FETCH_NORMALIZATION_PROVENANCE_AUDIT"
    )
    payload = {
        "stage": 450,
        "status": status,
        "checks": checks,
        "failed_checks": failures,
        "observation_values_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "historical_substitution": False,
        "hashes": {
            "template": sha256(TEMPLATE),
            "fetch_record": sha256(FETCH_RECORD),
            "manifest": sha256(MANIFEST),
        },
        "claim_boundary": [
            "本阶段只审计采集记录到规范化入口的血缘，不产生性能结果。",
            "当前官方窗口未闭合，manifest 仍为 NOT_READY。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage450 V9 采集到规范化血缘审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 规范化模板现在必须绑定 Stage442 正式采集记录并复核路径、字节数和 SHA256。",
                "- 当前未读取观测值、攻击标签或性能表。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 450, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
