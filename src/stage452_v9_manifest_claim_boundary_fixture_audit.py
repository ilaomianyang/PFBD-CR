from __future__ import annotations

"""Stage 452: pure-memory tests for the V9 manifest claim boundary."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage425_v9_compatibility import (  # noqa: E402
    V9_NORMALIZED_DIR,
    V9_RAW_DIR,
    _require_under,
    validate_manifest_claim_metadata,
)

OUT_JSON = ROOT / "tables" / "stage452_v9_manifest_claim_boundary_fixture_audit.json"
OUT_REPORT = ROOT / "reports" / "stage452_v9_manifest_claim_boundary_fixture_audit.md"


def must_raise(callback) -> bool:
    try:
        callback()
    except RuntimeError:
        return True
    return False


def main() -> int:
    valid = {
        "status": "OFFICIAL_WINDOW_READY",
        "failures": [],
        "observation_values_read": True,
        "historical_substitution": False,
    }
    checks = [
        {
            "check": "valid_claim_metadata_accepts",
            "status": "PASS"
            if (validate_manifest_claim_metadata(valid) is None)
            else "FAIL",
            "value": valid,
        },
        {
            "check": "provider_failures_rejected",
            "status": "PASS"
            if must_raise(
                lambda: validate_manifest_claim_metadata(
                    {**valid, "failures": ["ECCC failed"]}
                )
            )
            else "FAIL",
            "value": ["ECCC failed"],
        },
        {
            "check": "historical_substitution_rejected",
            "status": "PASS"
            if must_raise(
                lambda: validate_manifest_claim_metadata(
                    {**valid, "historical_substitution": True}
                )
            )
            else "FAIL",
            "value": True,
        },
        {
            "check": "raw_path_scope_accepts",
            "status": "PASS"
            if (
                _require_under(
                    V9_RAW_DIR / "provider" / "file.csv", V9_RAW_DIR, "raw"
                )
                is None
            )
            else "FAIL",
            "value": str(V9_RAW_DIR),
        },
        {
            "check": "normalized_path_scope_accepts",
            "status": "PASS"
            if (
                _require_under(
                    V9_NORMALIZED_DIR / "provider.parquet",
                    V9_NORMALIZED_DIR,
                    "normalized",
                )
                is None
            )
            else "FAIL",
            "value": str(V9_NORMALIZED_DIR),
        },
        {
            "check": "path_escape_rejected",
            "status": "PASS"
            if must_raise(
                lambda: _require_under(
                    V9_RAW_DIR / ".." / "old_window.parquet",
                    V9_RAW_DIR,
                    "raw",
                )
            )
            else "FAIL",
            "value": "path traversal",
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_MANIFEST_CLAIM_BOUNDARY_FIXTURE_AUDIT"
        if not failures
        else "FAIL_V9_MANIFEST_CLAIM_BOUNDARY_FIXTURE_AUDIT"
    )
    payload = {
        "stage": 452,
        "status": status,
        "checks": checks,
        "failed_checks": failures,
        "synthetic_fixture_only": True,
        "official_observation_values_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "historical_substitution": False,
        "claim_boundary": [
            "夹具只验证 manifest 声明和路径范围门禁，不代表官方数据或性能结果。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage452 V9 manifest 声明边界夹具审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 纯内存验证 ready 声明、失败项、历史替代和路径越界门禁。",
                "- 夹具不读取观测值，不产生性能结论。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 452, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
