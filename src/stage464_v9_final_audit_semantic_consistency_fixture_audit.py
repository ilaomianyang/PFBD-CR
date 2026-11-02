from __future__ import annotations

"""Stage 464: fixture audit for final-audit semantic consistency."""

import json
from pathlib import Path

import stage458_v9_postwindow_orchestrator as orchestrator


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tables" / "stage464_final_audit_semantic_fixture"
OUT_JSON = ROOT / "tables" / "stage464_v9_final_audit_semantic_consistency_fixture_audit.json"
OUT_REPORT = ROOT / "reports" / "stage464_v9_final_audit_semantic_consistency_fixture_audit.md"


def write_record(
    path: Path,
    *,
    status: str,
    performance: bool,
    failed_checks: list[str],
    check_status: str,
    manifest_status: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "performance_evaluated": performance,
                "failed_checks": failed_checks,
                "checks": [{"check": "fixture", "status": check_status}],
                "manifest_status": manifest_status,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    structural = FIXTURE_DIR / "stage426.json"
    statistical = FIXTURE_DIR / "stage445.json"
    original_structural = orchestrator.FINAL_AUDIT
    original_statistical = orchestrator.FINAL_STATISTICAL
    orchestrator.FINAL_AUDIT = structural
    orchestrator.FINAL_STATISTICAL = statistical
    cases: list[dict[str, object]] = []
    try:
        # A semantically consistent pair may pass.
        write_record(
            structural,
            status="PASS_V9_LOCKED_AUDIT",
            performance=True,
            failed_checks=[],
            check_status="PASS",
            manifest_status="OFFICIAL_WINDOW_READY",
        )
        write_record(
            statistical,
            status="PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT",
            performance=True,
            failed_checks=[],
            check_status="PASS",
            manifest_status="OFFICIAL_WINDOW_READY",
        )
        passed, details = orchestrator.final_audit_status()
        cases.append(
            {
                "case": "semantically_consistent_pass",
                "expected_pass": True,
                "observed_pass": passed,
                "details": details,
            }
        )

        # A PASS status with a non-empty failure list must be rejected.
        write_record(
            statistical,
            status="PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT",
            performance=True,
            failed_checks=["hidden_failure"],
            check_status="PASS",
            manifest_status="OFFICIAL_WINDOW_READY",
        )
        passed, details = orchestrator.final_audit_status()
        cases.append(
            {
                "case": "pass_with_failed_checks",
                "expected_pass": False,
                "observed_pass": passed,
                "details": details,
            }
        )

        # A PASS status with a failing check must be rejected.
        write_record(
            statistical,
            status="PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT",
            performance=True,
            failed_checks=[],
            check_status="FAIL",
            manifest_status="OFFICIAL_WINDOW_READY",
        )
        passed, details = orchestrator.final_audit_status()
        cases.append(
            {
                "case": "pass_with_failed_check_row",
                "expected_pass": False,
                "observed_pass": passed,
                "details": details,
            }
        )

        # A PASS status from a non-ready manifest must be rejected.
        write_record(
            statistical,
            status="PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT",
            performance=True,
            failed_checks=[],
            check_status="PASS",
            manifest_status="NOT_READY",
        )
        passed, details = orchestrator.final_audit_status()
        cases.append(
            {
                "case": "pass_with_not_ready_manifest",
                "expected_pass": False,
                "observed_pass": passed,
                "details": details,
            }
        )

        # A PASS status with a blocked check row must also be rejected.
        write_record(
            statistical,
            status="PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT",
            performance=True,
            failed_checks=[],
            check_status="BLOCKED",
            manifest_status="OFFICIAL_WINDOW_READY",
        )
        passed, details = orchestrator.final_audit_status()
        cases.append(
            {
                "case": "pass_with_blocked_check_row",
                "expected_pass": False,
                "observed_pass": passed,
                "details": details,
            }
        )
    finally:
        orchestrator.FINAL_AUDIT = original_structural
        orchestrator.FINAL_STATISTICAL = original_statistical

    failures = [
        item
        for item in cases
        if bool(item["expected_pass"]) != bool(item["observed_pass"])
    ]
    status = (
        "PASS_V9_FINAL_AUDIT_SEMANTIC_CONSISTENCY_FIXTURE_AUDIT"
        if not failures
        else "FAIL_V9_FINAL_AUDIT_SEMANTIC_CONSISTENCY_FIXTURE_AUDIT"
    )
    result = {
        "stage": 464,
        "status": status,
        "checks": cases,
        "failed_checks": failures,
        "synthetic_fixture_only": True,
        "official_observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "claim_boundary": [
            "本阶段只验证最终 PASS 状态的语义一致性，不产生性能结果。",
            "通过不代表 TrustMeteo-QC 在任何攻击场景或基线下领先。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage464 V9 最终审计语义一致性夹具审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查用例数：`{len(cases)}`；失败数：`{len(failures)}`",
                "- PASS 状态、失败列表、检查行和官方 manifest 状态必须相互一致；PASS 产物中不能含 BLOCKED 检查行。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 464, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
