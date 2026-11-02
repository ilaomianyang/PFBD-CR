from __future__ import annotations

"""Stage 463: fixture audit for stale final-audit artifact rejection."""

import json
import time
from pathlib import Path

import stage458_v9_postwindow_orchestrator as orchestrator


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tables" / "stage463_final_audit_freshness_fixture"
OUT_JSON = ROOT / "tables" / "stage463_v9_final_audit_freshness_fixture_audit.json"
OUT_REPORT = ROOT / "reports" / "stage463_v9_final_audit_freshness_fixture_audit.md"


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
        write_record(structural, "PASS_V9_LOCKED_AUDIT", True)
        write_record(statistical, "PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT", True)
        fresh_threshold = time.time_ns() - 1
        passed, details = orchestrator.final_audit_status(
            minimum_mtime_ns=fresh_threshold
        )
        cases.append(
            {
                "case": "both_fresh_and_explicit_pass",
                "expected_pass": True,
                "observed_pass": passed,
                "details": details,
            }
        )

        stale_threshold = time.time_ns() + 10_000_000_000
        passed, details = orchestrator.final_audit_status(
            minimum_mtime_ns=stale_threshold
        )
        cases.append(
            {
                "case": "both_explicit_pass_but_stale",
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
        "PASS_V9_FINAL_AUDIT_FRESHNESS_FIXTURE_AUDIT"
        if not failures
        else "FAIL_V9_FINAL_AUDIT_FRESHNESS_FIXTURE_AUDIT"
    )
    result = {
        "stage": 463,
        "status": status,
        "checks": cases,
        "failed_checks": failures,
        "synthetic_fixture_only": True,
        "official_observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "claim_boundary": [
            "本阶段只验证旧统计产物的新鲜度门禁，不产生性能结果。",
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
                "# Stage463 V9 最终审计产物新鲜度夹具审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查用例数：`{len(cases)}`；失败数：`{len(failures)}`",
                "- 旧的 PASS 状态即使内容正确，也不能替代本轮重新生成的审计产物。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 463, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
