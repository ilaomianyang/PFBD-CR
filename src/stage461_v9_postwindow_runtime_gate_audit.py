from __future__ import annotations

"""Stage 461: runtime audit for the V9 post-window safety gate.

This audit executes the frozen orchestrator before the official window closes
and verifies that it exits before invoking any child step.  It is a runtime
gate check only: it never reads observation values, attack labels, or old test
rows, and it never evaluates performance.
"""

import ast
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = ROOT / "src" / "stage458_v9_postwindow_orchestrator.py"
ARTIFACT = ROOT / "tables" / "stage458_v9_postwindow_orchestrator.json"
OUT_JSON = ROOT / "tables" / "stage461_v9_postwindow_runtime_gate_audit.json"
OUT_REPORT = ROOT / "reports" / "stage461_v9_postwindow_runtime_gate_audit.md"
WINDOW_CLOSE = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _source_gate_order(source: str) -> tuple[bool, int | None, int | None]:
    tree = ast.parse(source)
    gate_line: int | None = None
    subprocess_line: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            text = ast.unparse(node.test)
            if "checked_at < WINDOW_CLOSE" in text:
                gate_line = node.lineno
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr == "run"
            ):
                subprocess_line = node.lineno
    ordered = (
        gate_line is not None
        and subprocess_line is not None
        and gate_line < subprocess_line
    )
    return ordered, gate_line, subprocess_line


def main() -> int:
    checked_at = datetime.now(timezone.utc)
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    ordered, gate_line, subprocess_line = _source_gate_order(source)
    completed = subprocess.run(
        [sys.executable, "-u", str(ORCHESTRATOR)],
        cwd=str(ROOT.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    payload = {}
    if ARTIFACT.exists():
        payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    checks = [
        {
            "check": "window_is_still_before_close",
            "status": "PASS" if checked_at < WINDOW_CLOSE else "FAIL",
            "value": checked_at.isoformat(),
        },
        {
            "check": "source_gate_precedes_subprocess",
            "status": "PASS" if ordered else "FAIL",
            "value": {
                "gate_line": gate_line,
                "subprocess_line": subprocess_line,
            },
        },
        {
                "check": "source_has_final_audit_status_gate",
            "status": (
                "PASS"
                if "def final_audit_status(" in source
                and "final_audit_status(" in source
                and "PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT" in source
                and "FAIL_V9_POSTWINDOW_ORCHESTRATOR_FINAL_AUDIT" in source
                else "FAIL"
            ),
            "value": {
                "final_audit_function": "def final_audit_status(" in source,
                "final_audit_call": "final_audit_status(" in source,
                "statistical_pass_token": (
                    "PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT" in source
                ),
                "final_failure_token": (
                    "FAIL_V9_POSTWINDOW_ORCHESTRATOR_FINAL_AUDIT" in source
                ),
            },
        },
        {
            "check": "runtime_artifact_is_expected_block",
            "status": (
                "PASS"
                if payload.get("status")
                == "BLOCKED_V9_POSTWINDOW_ORCHESTRATOR_BEFORE_WINDOW_CLOSE"
                else "FAIL"
            ),
            "value": payload.get("status"),
        },
        {
            "check": "runtime_has_zero_child_steps",
            "status": "PASS" if payload.get("steps") == [] else "FAIL",
            "value": payload.get("steps"),
        },
        {
            "check": "runtime_did_not_evaluate_performance",
            "status": (
                "PASS"
                if payload.get("performance_evaluated") is False
                else "FAIL"
            ),
            "value": payload.get("performance_evaluated"),
        },
        {
            "check": "runtime_did_not_read_observations_or_old_test",
            "status": (
                "PASS"
                if payload.get("observation_values_read") is False
                and payload.get("old_test_rows_read") is False
                and payload.get("attack_labels_read") is False
                else "FAIL"
            ),
            "value": {
                "observation_values_read": payload.get("observation_values_read"),
                "old_test_rows_read": payload.get("old_test_rows_read"),
                "attack_labels_read": payload.get("attack_labels_read"),
            },
        },
    ]
    failed = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_POSTWINDOW_RUNTIME_GATE_AUDIT"
        if not failed
        else "FAIL_V9_POSTWINDOW_RUNTIME_GATE_AUDIT"
    )
    result = {
        "stage": 461,
        "status": status,
        "checked_at_utc": checked_at.isoformat(),
        "window_close_utc": WINDOW_CLOSE.isoformat().replace("+00:00", "Z"),
        "orchestrator_returncode": int(completed.returncode),
        "orchestrator_stdout": completed.stdout.strip(),
        "orchestrator_stderr": completed.stderr.strip(),
        "checks": checks,
        "failed_checks": failed,
        "official_observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "synthetic_fixture_only": False,
        "claim_boundary": [
            "本阶段只验证窗口前运行时门禁，不产生性能结果。",
            "通过不代表 TrustMeteo-QC 在任何攻击场景或基线下领先。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage461 V9 窗口后运行时门禁审计",
        "",
        f"- 状态：`{status}`",
        f"- 检查时间：`{checked_at.isoformat()}`",
        f"- 官方窗口闭合时间：`{WINDOW_CLOSE.isoformat().replace('+00:00', 'Z')}`",
        f"- 续跑器返回码：`{completed.returncode}`（窗口前预期为非零阻断）",
        f"- 失败检查数：`{len(failed)}`",
        "",
        "## 结论",
        "",
        "- 运行时门禁必须在任何子步骤之前阻断；本审计不读取观测值、不读取旧 test、不评价性能。",
        "- 该审计通过不代表任何性能优势或全面领先。",
        "",
        f"- JSON：`{OUT_JSON}`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": 461,
                "status": status,
                "failed_checks": failed,
                "performance_evaluated": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
