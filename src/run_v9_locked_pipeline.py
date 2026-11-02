from __future__ import annotations

"""Run the frozen V9 pipeline in the only permitted order.

The runner never falls back to V6/V8. With a NOT_READY manifest it records a
blocked run and exits without loading observations or invoking evaluation.
When data are ready it runs evaluation, structural audit, and the independent
statistical acceptance audit in that order.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
MANIFEST = PACKAGE / "data_manifest.json"
REPORT = ROOT / "reports" / "stage436_v9_locked_pipeline_run.json"
LOG = ROOT / "reports" / "stage436_v9_locked_pipeline_run.log"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    started = datetime.now(timezone.utc).isoformat()
    commands = [
        [
            sys.executable,
            "-u",
            str(ROOT / "src" / "stage425_v9_locked_evaluation.py"),
        ],
        [
            sys.executable,
            "-u",
            str(ROOT / "src" / "stage426_v9_locked_audit.py"),
        ],
        [
            sys.executable,
            "-u",
            str(ROOT / "src" / "stage445_v9_statistical_acceptance_audit.py"),
        ],
    ]
    payload: dict[str, object] = {
        "stage": 436,
        "started_at_utc": started,
        "manifest_status": manifest.get("status"),
        "observation_values_read": False,
        "attack_labels_read": False,
        "old_test_rows_read": False,
        "performance_evaluated": False,
        "commands": [command[2] for command in commands],
        "steps": [],
    }
    if manifest.get("status") != "OFFICIAL_WINDOW_READY":
        payload.update(
            {
                "status": "BLOCKED_V9_EXECUTION_NOT_READY",
                "blocking_reason": "data_manifest.status is not OFFICIAL_WINDOW_READY",
            }
        )
        REPORT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        LOG.write_text(
            f"{started} BLOCKED manifest_status={manifest.get('status')}\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "stage": 436,
                    "status": payload["status"],
                    "manifest_status": manifest.get("status"),
                },
                ensure_ascii=False,
            )
        )
        return 2

    log_lines = []
    for command in commands:
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
        log_lines.append(f"$ {' '.join(command)}\n{output}\n")
        payload["steps"].append(
            {
                "command": command[2],
                "returncode": completed.returncode,
            }
        )
        if completed.returncode != 0:
            payload["status"] = "FAIL_V9_LOCKED_PIPELINE"
            payload["blocking_reason"] = f"{command[2]} returned {completed.returncode}"
            payload["performance_evaluated"] = False
            break
    else:
        payload["status"] = "COMPLETED_V9_EVALUATION_AND_AUDIT"
        payload["performance_evaluated"] = True
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": 436,
                "status": payload["status"],
                "manifest_status": manifest.get("status"),
                "steps": payload["steps"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "COMPLETED_V9_EVALUATION_AND_AUDIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
