from __future__ import annotations

"""Stage 458: safe post-window V9 execution orchestrator.

Before the official October 2026 window closes this command performs no
network or observation access and records a hard blocked status.  After the
window closes it executes the frozen sequence:

    official fetch -> official normalization -> readiness gate
    -> evaluation -> structural audit -> statistical acceptance

Each step is captured in a separate log.  A non-zero step stops the sequence;
no later stage is allowed to interpret a partial package as a locked result.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
OUT_JSON = ROOT / "tables" / "stage458_v9_postwindow_orchestrator.json"
OUT_REPORT = ROOT / "reports" / "stage458_v9_postwindow_orchestrator.md"
OUT_LOG = ROOT / "reports" / "stage458_v9_postwindow_orchestrator.log"
FINAL_AUDIT = ROOT / "tables" / "stage426_v9_locked_audit.json"
FINAL_STATISTICAL = ROOT / "tables" / "stage445_v9_statistical_acceptance_audit.json"
WINDOW_CLOSE = datetime(2026, 11, 1, tzinfo=timezone.utc)


def command_for(script: str) -> list[str]:
    return [sys.executable, "-u", str(ROOT / "src" / script)]


def write_artifacts(payload: dict[str, object], log_text: str) -> None:
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_LOG.write_text(log_text, encoding="utf-8")
    steps = payload.get("steps", [])
    lines = [
        "# Stage458 V9 官方窗口后续跑编排器",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 检查时间：`{payload['checked_at_utc']}`",
        f"- 官方窗口闭合时间：`{payload['window_close_utc']}`",
        f"- 执行步骤数：`{len(steps)}`",
        "- 官方窗口闭合前不联网、不读取观测值、不执行性能评价。",
        "- 窗口闭合后严格执行：采集 → 规范化 → 数据门禁 → 评价 → 结构审计 → 统计验收。",
        "- 任一步骤失败都会停止，不会把部分结果标记为锁定完成。",
        "",
        "## 步骤",
        "",
    ]
    if steps:
        lines.extend(
            f"- `{step['script']}`：`{step['status']}`（returncode={step['returncode']}）"
            for step in steps
        )
    else:
        lines.append("- 尚未执行任何子步骤。")
    lines.extend(
        [
            "",
            "## 科学边界",
            "",
            "- 本编排器本身不产生性能结果。",
            "- 所有攻击仍为真实观测背景上的透明受控模拟攻击，不是真实生产攻击日志。",
            "- 不允许用旧 V6 test 或历史 normalized 文件替代 V9 官方窗口。",
            "",
            f"- JSON：`{OUT_JSON}`",
            f"- 日志：`{OUT_LOG}`",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_audit_status(
    *,
    minimum_mtime_ns: int | None = None,
) -> tuple[bool, dict[str, object]]:
    """Require both structural and statistical audits to explicitly pass."""
    details: dict[str, object] = {}
    for path, expected in (
        (FINAL_AUDIT, "PASS_V9_LOCKED_AUDIT"),
        (FINAL_STATISTICAL, "PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT"),
    ):
        if not path.exists():
            details[path.name] = {"status": "MISSING"}
            return False, details
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            details[path.name] = {"status": "UNREADABLE", "error": str(exc)}
            return False, details
        status = record.get("status")
        mtime_ns = path.stat().st_mtime_ns
        failed_checks = record.get("failed_checks", [])
        check_rows = record.get("checks", [])
        checks_clean = (
            isinstance(check_rows, list)
            and all(
                not isinstance(row, dict)
                or row.get("status") == "PASS"
                for row in check_rows
            )
        )
        details[path.name] = {
            "status": status,
            "expected": expected,
            "performance_evaluated": record.get("performance_evaluated"),
            "manifest_status": record.get("manifest_status"),
            "failed_checks": failed_checks,
            "checks_clean": checks_clean,
            "mtime_ns": mtime_ns,
        }
        if (
            status != expected
            or record.get("performance_evaluated") is not True
            or failed_checks != []
            or not checks_clean
            or record.get("manifest_status") != "OFFICIAL_WINDOW_READY"
        ):
            return False, details
        if minimum_mtime_ns is not None and mtime_ns < minimum_mtime_ns:
            details[path.name]["freshness"] = "STALE"
            details[path.name]["minimum_mtime_ns"] = minimum_mtime_ns
            return False, details
        details[path.name]["freshness"] = "FRESH"
    return True, details


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checked_at = datetime.now(timezone.utc)
    started_ns = time.time_ns()
    payload: dict[str, object] = {
        "stage": 458,
        "protocol_id": protocol.get("protocol_id"),
        "checked_at_utc": checked_at.isoformat(),
        "started_ns": started_ns,
        "window_close_utc": WINDOW_CLOSE.isoformat().replace("+00:00", "Z"),
        "observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "steps": [],
        "commands": [
            "stage442_v9_official_window_fetch.py",
            "stage425_v9_2026_official_window_download_template.py",
            "stage424_v9_execution_readiness_audit.py",
            "run_v9_locked_pipeline.py",
        ],
    }
    if checked_at < WINDOW_CLOSE:
        payload.update(
            {
                "status": "BLOCKED_V9_POSTWINDOW_ORCHESTRATOR_BEFORE_WINDOW_CLOSE",
                "blocking_reason": (
                    "当前时间早于 2026-11-01T00:00:00Z；未调用任何网络、"
                    "规范化或评价子程序。"
                ),
            }
        )
        write_artifacts(payload, f"{checked_at.isoformat()} BLOCKED before window close\n")
        print(
            json.dumps(
                {
                    "stage": 458,
                    "status": payload["status"],
                    "steps": 0,
                    "performance_evaluated": False,
                },
                ensure_ascii=False,
            )
        )
        return 2

    commands = [
        command_for("stage442_v9_official_window_fetch.py"),
        command_for("stage425_v9_2026_official_window_download_template.py"),
        command_for("stage424_v9_execution_readiness_audit.py"),
        command_for("run_v9_locked_pipeline.py"),
    ]
    log_parts: list[str] = []
    for command in commands:
        script = Path(command[-1]).name
        completed = subprocess.run(
            command,
            cwd=str(ROOT.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        output = f"{completed.stdout}\n{completed.stderr}".strip()
        log_parts.append(f"$ {' '.join(command)}\n{output}\n")
        step_status = "PASS" if completed.returncode == 0 else "FAIL"
        payload["steps"].append(
            {
                "script": script,
                "status": step_status,
                "returncode": int(completed.returncode),
            }
        )
        if completed.returncode != 0:
            payload.update(
                {
                    "status": "FAIL_V9_POSTWINDOW_ORCHESTRATOR",
                    "blocking_reason": f"{script} returned {completed.returncode}",
                }
            )
            payload["performance_evaluated"] = False
            write_artifacts(payload, "\n".join(log_parts))
            print(
                json.dumps(
                    {
                        "stage": 458,
                        "status": payload["status"],
                        "steps": len(payload["steps"]),
                        "performance_evaluated": False,
                    },
                    ensure_ascii=False,
                )
            )
            return 1

    audits_pass, audit_details = final_audit_status(
        minimum_mtime_ns=started_ns,
    )
    payload["final_audit_status"] = audit_details
    if not audits_pass:
        payload.update(
            {
                "status": "FAIL_V9_POSTWINDOW_ORCHESTRATOR_FINAL_AUDIT",
                "blocking_reason": (
                    "evaluation subprocesses returned zero, but the final "
                    "structural/statistical audit artifacts are not both PASS"
                ),
                "performance_evaluated": False,
            }
        )
        write_artifacts(payload, "\n".join(log_parts))
        print(
            json.dumps(
                {
                    "stage": 458,
                    "status": payload["status"],
                    "steps": len(payload["steps"]),
                    "performance_evaluated": False,
                },
                ensure_ascii=False,
            )
        )
        return 1

    payload.update(
        {
            "status": "COMPLETED_V9_POSTWINDOW_ORCHESTRATOR",
            "performance_evaluated": True,
        }
    )
    write_artifacts(payload, "\n".join(log_parts))
    print(
        json.dumps(
            {
                "stage": 458,
                "status": payload["status"],
                "steps": len(payload["steps"]),
                "performance_evaluated": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
