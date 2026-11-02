from __future__ import annotations

"""Stage 430: audit the V9 data-gate integrity before official data arrive.

This is metadata-only. It does not read observation rows, request network
content, generate attacks, fit detectors, or inspect old test outputs.
"""

import hashlib
import json
from pathlib import Path

from stage425_v9_compatibility import EXPECTED_GRID, EXPECTED_STATIONS


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
MANIFEST = PACKAGE / "data_manifest.json"
TEMPLATE = ROOT / "src" / "stage425_v9_2026_official_window_download_template.py"
COMPAT = ROOT / "src" / "stage425_v9_compatibility.py"
OUT_JSON = ROOT / "tables" / "stage430_v9_data_gate_integrity_audit.json"
OUT_REPORT = ROOT / "reports" / "stage430_v9_data_gate_integrity_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    template_text = TEMPLATE.read_text(encoding="utf-8")
    compatibility_text = COMPAT.read_text(encoding="utf-8")

    checks = [
        {
            "check": "protocol_window_frozen",
            "status": (
                "PASS"
                if protocol.get("window", {}).get("test_start_utc")
                == "2026-10-01T00:00:00Z"
                and protocol.get("window", {}).get("test_end_utc")
                == "2026-10-31T23:00:00Z"
                else "FAIL"
            ),
        },
        {
            "check": "frozen_station_sets_complete",
            "status": (
                "PASS"
                if {name: len(ids) for name, ids in EXPECTED_STATIONS.items()}
                == {
                    "MeteoSwiss": 12,
                    "NOAA ISD": 11,
                    "KNMI": 12,
                    "ECCC": 16,
                }
                else "FAIL"
            ),
            "value": {
                name: sorted(ids) for name, ids in EXPECTED_STATIONS.items()
            },
        },
        {
            "check": "complete_hourly_grid_contract",
            "status": "PASS" if len(EXPECTED_GRID) == 744 else "FAIL",
            "value": {
                "hours_per_station": len(EXPECTED_GRID),
                "expected_rows": {
                    name: len(ids) * len(EXPECTED_GRID)
                    for name, ids in EXPECTED_STATIONS.items()
                },
            },
        },
        {
            "check": "template_hard_gates_present",
            "status": (
                "PASS"
                if "station set mismatch" in template_text
                and "duplicate station/timestamp" in template_text
                and "reindex(EXPECTED_GRID)" in template_text
                else "FAIL"
            ),
        },
        {
            "check": "ready_manifest_requires_integrity_fields",
            "status": (
                "PASS"
                if "actual_station_ids" in compatibility_text
                and "normalized test row count" in compatibility_text
                and "raw_sha256" in compatibility_text
                else "FAIL"
            ),
        },
        {
            "check": "current_manifest_not_ready",
            "status": (
                "PASS" if manifest.get("status") == "NOT_READY" else "FAIL"
            ),
            "value": manifest.get("status"),
        },
        {
            "check": "historical_substitution_disabled",
            "status": (
                "PASS"
                if manifest.get("historical_substitution") is False
                else "FAIL"
            ),
        },
    ]
    failures = [item["check"] for item in checks if item["status"] != "PASS"]
    status = (
        "PASS_V9_DATA_GATE_INTEGRITY_AUDIT"
        if not failures
        else "FAIL_V9_DATA_GATE_INTEGRITY_AUDIT"
    )
    payload = {
        "stage": 430,
        "status": status,
        "protocol_id": protocol.get("protocol_id"),
        "observation_values_read": False,
        "http_response_bodies_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "checks": checks,
        "failed_checks": failures,
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "manifest": sha256(MANIFEST),
            "template": sha256(TEMPLATE),
            "compatibility": sha256(COMPAT),
        },
        "claim_boundary": [
            "本审计只核验 V9 数据完整性门禁，不产生检测性能结果。",
            "官方 2026-10 数据未就绪时，必须保持 NOT_READY。",
            "所有未来攻击仍是透明受控模拟攻击，不是真实生产攻击日志。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage430 V9 数据完整性门禁审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 只读取协议、清单和源码文本；未读取观测值、攻击标签或旧 test。",
                "- 当前官方窗口清单仍必须保持 `NOT_READY`。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": 430, "status": status, "failed_checks": failures}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
