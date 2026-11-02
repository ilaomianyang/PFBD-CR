from __future__ import annotations

"""Stage 465: freeze the supplemental V9 execution-gate dependencies.

Stage456 freezes the core evaluator bundle.  This supplemental audit freezes
the later orchestration and semantic-gate code without touching observations,
attack labels, or performance outputs.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
MANIFEST = PACKAGE / "data_manifest.json"
PROTOCOL = PACKAGE / "protocol_freeze.json"
OUT_JSON = ROOT / "tables" / "stage465_v9_execution_gate_dependency_freeze_audit.json"
OUT_REPORT = ROOT / "reports" / "stage465_v9_execution_gate_dependency_freeze_audit.md"

SOURCES = (
    "src/stage458_v9_postwindow_orchestrator.py",
    "src/stage461_v9_postwindow_runtime_gate_audit.py",
    "src/stage462_v9_final_audit_gate_fixture_audit.py",
    "src/stage463_v9_final_audit_freshness_fixture_audit.py",
    "src/stage464_v9_final_audit_semantic_consistency_fixture_audit.py",
    "tests/run_tests.py",
    "tests/test_quality_audits.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for relative in SOURCES:
        path = ROOT / relative
        rows.append(
            {
                "path": relative,
                "present": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    failures = [
        row["path"] for row in rows if row["present"] is not True or not row["sha256"]
    ]
    checks = [
        {
            "check": "all_execution_gate_sources_present_and_hashed",
            "status": "PASS" if not failures else "FAIL",
            "value": rows,
        },
        {
            "check": "protocol_identity_matches",
            "status": "PASS"
            if protocol.get("protocol_id")
            == "TMQC-SIMULATED-ATTACK-V9-OCT2026-20260925-v1.2"
            else "FAIL",
            "value": protocol.get("protocol_id"),
        },
        {
            "check": "manifest_remains_not_ready",
            "status": "PASS"
            if manifest.get("status") == "NOT_READY"
            else "FAIL",
            "value": manifest.get("status"),
        },
        {
            "check": "no_observation_or_performance_access_claimed",
            "status": "PASS"
            if manifest.get("observation_values_read") is False
            and manifest.get("historical_substitution") is False
            else "FAIL",
            "value": {
                "observation_values_read": manifest.get("observation_values_read"),
                "historical_substitution": manifest.get("historical_substitution"),
            },
        },
    ]
    failures.extend(row["check"] for row in checks if row["status"] != "PASS")
    status = (
        "PASS_V9_EXECUTION_GATE_DEPENDENCY_FREEZE_AUDIT"
        if not failures
        else "FAIL_V9_EXECUTION_GATE_DEPENDENCY_FREEZE_AUDIT"
    )
    result = {
        "stage": 465,
        "status": status,
        "freeze_id": "TMQC-V9-EXECUTION-GATE-BUNDLE-20260925-v1.0",
        "protocol_id": protocol.get("protocol_id"),
        "source_count": len(SOURCES),
        "sources": rows,
        "checks": checks,
        "failed_checks": failures,
        "official_observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "claim_boundary": [
            "本阶段只冻结 V9 执行编排和门禁源码，不产生任何性能结果。",
            "manifest 未就绪时不得运行评价，也不得用旧 test 替代。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage465 V9 执行门禁依赖冻结审计",
                "",
                f"- 状态：`{status}`",
                f"- 冻结 ID：`{result['freeze_id']}`",
                f"- 源码数量：`{len(SOURCES)}`",
                f"- 失败检查数：`{len(failures)}`",
                "- 本阶段只冻结执行门禁依赖，不读取观测值或性能结果。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 465, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
