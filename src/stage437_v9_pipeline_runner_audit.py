from __future__ import annotations

"""Stage 437: audit the V9 continuation runner while data are not ready."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026" / "data_manifest.json"
)
RUNNER = ROOT / "src" / "run_v9_locked_pipeline.py"
REPORT = ROOT / "reports" / "stage436_v9_locked_pipeline_run.json"
OUT_JSON = ROOT / "tables" / "stage437_v9_pipeline_runner_audit.json"
OUT_REPORT = ROOT / "reports" / "stage437_v9_pipeline_runner_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    run = json.loads(REPORT.read_text(encoding="utf-8"))
    text = RUNNER.read_text(encoding="utf-8")
    checks = [
        {
            "check": "manifest_not_ready",
            "status": "PASS" if manifest.get("status") == "NOT_READY" else "FAIL",
        },
        {
            "check": "runner_records_blocked_status",
            "status": (
                "PASS"
                if run.get("status") == "BLOCKED_V9_EXECUTION_NOT_READY"
                and run.get("manifest_status") == "NOT_READY"
                else "FAIL"
            ),
            "value": run.get("status"),
        },
        {
            "check": "runner_has_no_fallback_entrypoint",
            "status": (
                "PASS"
                if "stage349_v6_locked_evaluation" not in text
                and "stage351_v6_station_bh_audit" not in text
                else "FAIL"
            ),
        },
        {
            "check": "runner_orders_evaluation_then_audit",
            "status": (
                "PASS"
                if text.index("stage425_v9_locked_evaluation.py")
                < text.index("stage426_v9_locked_audit.py")
                else "FAIL"
            ),
        },
        {
            "check": "runner_orders_statistical_acceptance_last",
            "status": (
                "PASS"
                if text.index("stage426_v9_locked_audit.py")
                < text.index("stage445_v9_statistical_acceptance_audit.py")
                else "FAIL"
            ),
        },
        {
            "check": "blocked_run_does_not_evaluate",
            "status": (
                "PASS"
                if run.get("performance_evaluated") is False
                and run.get("observation_values_read") is False
                and run.get("old_test_rows_read") is False
                else "FAIL"
            ),
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_PIPELINE_RUNNER_AUDIT"
        if not failures
        else "FAIL_V9_PIPELINE_RUNNER_AUDIT"
    )
    payload = {
        "stage": 437,
        "status": status,
        "checks": checks,
        "failed_checks": failures,
        "manifest_status": manifest.get("status"),
        "observation_values_read": False,
        "attack_labels_read": False,
        "old_test_rows_read": False,
        "performance_evaluated": False,
        "hashes": {
            "manifest": sha256(MANIFEST),
            "runner": sha256(RUNNER),
            "run_report": sha256(REPORT),
        },
        "claim_boundary": [
            "本阶段只审计续跑器行为，不产生检测性能结果。",
            "NOT_READY 时不得使用旧 test 或旧评价入口替代。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage437 V9 续跑器审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 当前清单为 NOT_READY，续跑器必须只记录阻断并退出。",
                "- 未读取观测值、攻击标签或旧 test。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": 437, "status": status, "failed_checks": failures}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
