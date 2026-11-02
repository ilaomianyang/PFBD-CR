from __future__ import annotations

"""Stage 462: fixture audit for the V9 final-audit completion gate.

The fixture exercises the post-window orchestrator's final status check with
isolated JSON records only.  It does not load observation rows or evaluate
performance.
"""

import json
from pathlib import Path

import stage458_v9_postwindow_orchestrator as orchestrator


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tables" / "stage462_final_audit_gate_fixture"
OUT_JSON = ROOT / "tables" / "stage462_v9_final_audit_gate_fixture_audit.json"
OUT_REPORT = ROOT / "reports" / "stage462_v9_final_audit_gate_fixture_audit.md"


def write_record(path: Path, status: str, performance: bool) -> None:
    path.write_text(
        json.dumps(
            {"status": status, "performance_evaluated": performance},
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
        # A missing artifact must block completion.
        structural.unlink(missing_ok=True)
        statistical.unlink(missing_ok=True)
        passed, details = orchestrator.final_audit_status()
        cases.append(
            {
                "case": "both_missing",
                "expected_pass": False,
                "observed_pass": passed,
                "details": details,
            }
        )

        # A blocked statistical artifact must block completion.
        write_record(structural, "PASS_V9_LOCKED_AUDIT", True)
        write_record(statistical, "BLOCKED_V9_STATISTICAL_AUDIT_NOT_READY", False)
        passed, details = orchestrator.final_audit_status()
        cases.append(
            {
                "case": "statistical_blocked",
                "expected_pass": False,
                "observed_pass": passed,
                "details": details,
            }
        )

        # A structural failure must block completion even if statistics pass.
        write_record(structural, "FAIL_V9_LOCKED_AUDIT", False)
        write_record(statistical, "PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT", True)
        passed, details = orchestrator.final_audit_status()
        cases.append(
            {
                "case": "structural_failed",
                "expected_pass": False,
                "observed_pass": passed,
                "details": details,
            }
        )

        # Only two explicit PASS artifacts with performance flags may complete.
        write_record(structural, "PASS_V9_LOCKED_AUDIT", True)
        write_record(statistical, "PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT", True)
        passed, details = orchestrator.final_audit_status()
        cases.append(
            {
                "case": "both_explicit_pass",
                "expected_pass": True,
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
        "PASS_V9_FINAL_AUDIT_GATE_FIXTURE_AUDIT"
        if not failures
        else "FAIL_V9_FINAL_AUDIT_GATE_FIXTURE_AUDIT"
    )
    result = {
        "stage": 462,
        "status": status,
        "checks": cases,
        "failed_checks": failures,
        "synthetic_fixture_only": True,
        "official_observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "claim_boundary": [
            "本阶段只验证最终审计状态门禁，不产生任何性能结果。",
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
                "# Stage462 V9 最终审计状态门禁夹具审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查用例数：`{len(cases)}`；失败数：`{len(failures)}`",
                "- 缺失、阻断或任一审计失败均不能被标记为完成。",
                "- 仅结构审计和统计验收均显式 PASS 时才允许完成。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 462, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
