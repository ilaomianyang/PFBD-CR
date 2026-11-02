from __future__ import annotations

"""Stage 446: verify that the V9 frozen protocol names the V9 entrypoints.

This is a metadata/reproducibility audit only.  It reads protocol, source and
manifest metadata, but never loads official observations, attack labels, or
performance tables.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
MANIFEST = PACKAGE / "data_manifest.json"
RUNNER = ROOT / "src" / "run_v9_locked_pipeline.py"
EVALUATOR = ROOT / "src" / "stage425_v9_locked_evaluation.py"
AUDITOR = ROOT / "src" / "stage426_v9_locked_audit.py"
OUT_JSON = ROOT / "tables" / "stage446_v9_protocol_entrypoint_lineage_audit.json"
OUT_REPORT = ROOT / "reports" / "stage446_v9_protocol_entrypoint_lineage_audit.md"

EXPECTED_EVALUATOR = "src/stage425_v9_locked_evaluation.py"
EXPECTED_AUDITOR = "src/stage426_v9_locked_audit.py"
EXPECTED_STATISTICAL_AUDITOR = "src/stage445_v9_statistical_acceptance_audit.py"
LEGACY_MARKERS = ("stage349_v6_locked_evaluation.py", "stage351_v6_station_bh_audit.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runner_text = RUNNER.read_text(encoding="utf-8")
    runner_evaluator_present = (
        str(EVALUATOR) in runner_text
        or EVALUATOR.name in runner_text
    )
    runner_auditor_present = (
        str(AUDITOR) in runner_text
        or AUDITOR.name in runner_text
    )
    runner_statistical_auditor_present = (
        str(ROOT / EXPECTED_STATISTICAL_AUDITOR) in runner_text
        or Path(EXPECTED_STATISTICAL_AUDITOR).name in runner_text
    )
    checks = [
        {
            "check": "protocol_revision_declared",
            "status": (
                "PASS"
                if protocol.get("protocol_id")
                == "TMQC-SIMULATED-ATTACK-V9-OCT2026-20260925-v1.2"
                and protocol.get("supersedes_protocol_id")
                == "TMQC-SIMULATED-ATTACK-V9-OCT2026-20260925-v1.1"
                else "FAIL"
            ),
            "value": {
                "protocol_id": protocol.get("protocol_id"),
                "supersedes_protocol_id": protocol.get("supersedes_protocol_id"),
            },
        },
        {
            "check": "protocol_evaluation_entrypoint_is_v9",
            "status": (
                "PASS"
                if protocol.get("evaluation_entrypoint") == EXPECTED_EVALUATOR
                else "FAIL"
            ),
            "value": protocol.get("evaluation_entrypoint"),
        },
        {
            "check": "protocol_audit_entrypoint_is_v9",
            "status": (
                "PASS" if protocol.get("audit_entrypoint") == EXPECTED_AUDITOR else "FAIL"
            ),
            "value": protocol.get("audit_entrypoint"),
        },
        {
            "check": "protocol_statistical_audit_entrypoint_is_v9",
            "status": (
                "PASS"
                if protocol.get("statistical_audit_entrypoint")
                == EXPECTED_STATISTICAL_AUDITOR
                else "FAIL"
            ),
            "value": protocol.get("statistical_audit_entrypoint"),
        },
        {
            "check": "declared_entrypoint_files_exist",
            "status": (
                "PASS"
                if EVALUATOR.exists()
                and AUDITOR.exists()
                and (ROOT / EXPECTED_STATISTICAL_AUDITOR).exists()
                else "FAIL"
            ),
            "value": {
                "evaluation": str(EVALUATOR),
                "audit": str(AUDITOR),
                "statistical_audit": str(ROOT / EXPECTED_STATISTICAL_AUDITOR),
            },
        },
        {
            "check": "runner_matches_protocol_and_has_no_legacy_fallback",
            "status": (
                "PASS"
                if runner_evaluator_present
                and runner_auditor_present
                and runner_statistical_auditor_present
                and not any(marker in runner_text for marker in LEGACY_MARKERS)
                else "FAIL"
            ),
            "value": {
                "expected_evaluator_in_runner": runner_evaluator_present,
                "expected_auditor_in_runner": runner_auditor_present,
                "expected_statistical_auditor_in_runner": runner_statistical_auditor_present,
                "legacy_markers_in_runner": [
                    marker for marker in LEGACY_MARKERS if marker in runner_text
                ],
            },
        },
        {
            "check": "manifest_remains_not_ready",
            "status": "PASS" if manifest.get("status") == "NOT_READY" else "FAIL",
            "value": manifest.get("status"),
        },
        {
            "check": "no_performance_evaluation_performed",
            "status": "PASS",
            "value": {
                "observation_values_read": False,
                "attack_labels_read": False,
                "performance_evaluated": False,
            },
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_PROTOCOL_ENTRYPOINT_LINEAGE_AUDIT"
        if not failures
        else "FAIL_V9_PROTOCOL_ENTRYPOINT_LINEAGE_AUDIT"
    )
    payload = {
        "stage": 446,
        "status": status,
        "protocol_id": protocol.get("protocol_id"),
        "checks": checks,
        "failed_checks": failures,
        "observation_values_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "runner": sha256(RUNNER),
            "evaluator": sha256(EVALUATOR),
            "auditor": sha256(AUDITOR),
            "statistical_auditor": sha256(
                ROOT / EXPECTED_STATISTICAL_AUDITOR
            ),
            "manifest": sha256(MANIFEST),
        },
        "claim_boundary": [
            "本阶段只修复并审计协议入口元数据，不产生性能结果。",
            "官方窗口未就绪，不能把 V9 解释为已完成或全面领先。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage446 V9 协议入口血缘审计",
                "",
                f"- 状态：`{status}`",
                f"- 协议：`{protocol.get('protocol_id')}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 本阶段仅修复执行入口元数据；未读取官方观测值、攻击标签或性能表。",
                "- V9 官方窗口仍未就绪，性能验收保持未完成。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 446, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
