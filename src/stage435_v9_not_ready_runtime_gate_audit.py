from __future__ import annotations

"""Stage 435: runtime refusal audit for an unready V9 manifest.

The command intentionally runs the V9 entrypoint while the official-window
manifest is NOT_READY. It must fail before loading rows or fitting anything.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
MANIFEST = PACKAGE / "data_manifest.json"
RUNNER = ROOT / "src" / "stage425_v9_locked_evaluation.py"
OUT_JSON = ROOT / "tables" / "stage435_v9_not_ready_runtime_gate_audit.json"
OUT_REPORT = ROOT / "reports" / "stage435_v9_not_ready_runtime_gate_audit.md"
OUTPUTS = [
    ROOT / "tables" / "stage425_v9_locked_evaluation.json",
    ROOT / "tables" / "stage425_v9_locked_evaluation_detail.csv",
    ROOT / "tables" / "stage425_v9_locked_evaluation_scenario.csv",
    ROOT / "tables" / "stage425_v9_locked_evaluation_station.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    before = {
        str(path): path.stat().st_mtime_ns
        for path in OUTPUTS
        if path.exists()
    }
    completed = subprocess.run(
        [sys.executable, "-u", str(RUNNER)],
        cwd=str(ROOT.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    after = {
        str(path): path.stat().st_mtime_ns
        for path in OUTPUTS
        if path.exists()
    }
    changed_outputs = [
        path
        for path, mtime in after.items()
        if path in before and before[path] != mtime
    ]
    new_outputs = [path for path in after if path not in before]
    stdout_stderr = f"{completed.stdout}\n{completed.stderr}"
    checks = [
        {
            "check": "manifest_is_not_ready",
            "status": "PASS" if manifest.get("status") == "NOT_READY" else "FAIL",
            "value": manifest.get("status"),
        },
        {
            "check": "runner_refuses_before_evaluation",
            "status": (
                "PASS"
                if completed.returncode != 0
                and "official window is not ready" in stdout_stderr
                else "FAIL"
            ),
            "value": {
                "returncode": completed.returncode,
                "message_present": "official window is not ready" in stdout_stderr,
            },
        },
        {
            "check": "no_new_v9_outputs_created",
            "status": "PASS" if not new_outputs else "FAIL",
            "value": new_outputs,
        },
        {
            "check": "no_existing_v9_outputs_changed",
            "status": "PASS" if not changed_outputs else "FAIL",
            "value": changed_outputs,
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_NOT_READY_RUNTIME_GATE_AUDIT"
        if not failures
        else "FAIL_V9_NOT_READY_RUNTIME_GATE_AUDIT"
    )
    payload = {
        "stage": 435,
        "status": status,
        "manifest_status": manifest.get("status"),
        "runner_returncode": completed.returncode,
        "observation_values_read": False,
        "attack_labels_read": False,
        "old_test_rows_read": False,
        "performance_evaluated": False,
        "checks": checks,
        "failed_checks": failures,
        "hashes": {
            "manifest": sha256(MANIFEST),
            "runner": sha256(RUNNER),
        },
        "claim_boundary": [
            "本阶段只验证 NOT_READY 运行时门禁，不产生检测性能结果。",
            "官方窗口未就绪时不得使用旧年份或旧 test 替代。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage435 V9 NOT_READY 运行时门禁审计",
                "",
                f"- 状态：`{status}`",
                f"- 入口退出码：`{completed.returncode}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 入口必须在读取 test 行和训练前拒绝运行。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": 435,
                "status": status,
                "returncode": completed.returncode,
                "failed_checks": failures,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
