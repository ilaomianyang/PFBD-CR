from __future__ import annotations

"""Stage 434: audit cryptographic binding of the V9 ready manifest.

Source/contract-only audit. No official files are read and no test rows are
loaded while the V9 manifest remains NOT_READY.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
MANIFEST = PACKAGE / "data_manifest.json"
COMPAT = ROOT / "src" / "stage425_v9_compatibility.py"
OUT_JSON = ROOT / "tables" / "stage434_v9_manifest_hash_binding_audit.json"
OUT_REPORT = ROOT / "reports" / "stage434_v9_manifest_hash_binding_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    compat_text = COMPAT.read_text(encoding="utf-8")
    checks = [
        {
            "check": "sha256_function_present",
            "status": "PASS" if "def _sha256" in compat_text else "FAIL",
        },
        {
            "check": "raw_hash_verified_before_evaluation",
            "status": (
                "PASS"
                if '_sha256(raw_path).lower() != str(meta["raw_sha256"]).lower()'
                in compat_text
                else "FAIL"
            ),
        },
        {
            "check": "normalized_hash_verified_before_evaluation",
            "status": (
                "PASS"
                if '_sha256(normalized_path).lower() != str(' in compat_text
                else "FAIL"
            ),
        },
        {
            "check": "test_grid_revalidated_after_load",
            "status": (
                "PASS"
                if "test parquet timestamps are outside the grid" in compat_text
                and "test parquet row count" in compat_text
                and "test parquet station set mismatch" in compat_text
                else "FAIL"
            ),
        },
        {
            "check": "manifest_currently_not_ready",
            "status": (
                "PASS" if manifest.get("status") == "NOT_READY" else "FAIL"
            ),
            "value": manifest.get("status"),
        },
        {
            "check": "protocol_window_unchanged",
            "status": (
                "PASS"
                if protocol.get("window", {}).get("test_start_utc")
                == "2026-10-01T00:00:00Z"
                and protocol.get("window", {}).get("test_end_utc")
                == "2026-10-31T23:00:00Z"
                else "FAIL"
            ),
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_MANIFEST_HASH_BINDING_AUDIT"
        if not failures
        else "FAIL_V9_MANIFEST_HASH_BINDING_AUDIT"
    )
    payload = {
        "stage": 434,
        "status": status,
        "protocol_id": protocol.get("protocol_id"),
        "observation_values_read": False,
        "attack_labels_read": False,
        "old_test_rows_read": False,
        "performance_evaluated": False,
        "checks": checks,
        "failed_checks": failures,
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "manifest": sha256(MANIFEST),
            "compatibility": sha256(COMPAT),
        },
        "claim_boundary": [
            "本阶段只审计 manifest 与文件哈希绑定逻辑，不产生性能结果。",
            "当前官方窗口未就绪，必须保持 NOT_READY。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage434 V9 manifest 哈希绑定审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 未读取官方观测值、攻击标签或旧 test。",
                "- 当前清单必须保持 `NOT_READY`。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 434, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
