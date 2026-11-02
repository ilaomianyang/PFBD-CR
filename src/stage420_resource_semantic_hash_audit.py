from __future__ import annotations

"""Stage 420: semantic and hash audit for Stage 419 resource evidence.

This is a read-only audit of the Stage 419 artifact and clean input-file
hashes.  It does not read attack logs or project test labels, fit a detector,
select a threshold, or calculate detection metrics.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = Path(__file__).resolve()
INPUT = TABLES / "stage419_testbed_resource_only_audit.json"
OUTPUT = TABLES / "stage420_resource_semantic_hash_audit.json"
REPORT = REPORTS / "stage420_resource_semantic_hash_audit.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    check(
        "status_and_counts",
        payload.get("status") == "PASS_TESTBED_RESOURCE_ONLY_AUDIT"
        and payload.get("seed_count") == 5
        and payload.get("run_count") == 5
        and payload.get("completed_count") == 5
        and payload.get("failure_count") == 0,
        {
            key: payload.get(key)
            for key in (
                "status",
                "seed_count",
                "run_count",
                "completed_count",
                "failure_count",
            )
        },
    )
    check(
        "no_attack_or_test_access",
        payload.get("attack_logs_read") is False
        and payload.get("test_labels_read") is False
        and all(
            row.get("attack_logs_read") is False
            and row.get("test_labels_read") is False
            for row in payload.get("runs", [])
        ),
        "all top-level and per-seed access flags are false",
    )
    check(
        "not_a_performance_result",
        payload.get("performance_status") == "NOT_EVALUATED"
        and payload.get("fit_seconds") == "NOT_APPLICABLE"
        and payload.get("parameter_count") == "NOT_APPLICABLE",
        {
            "performance_status": payload.get("performance_status"),
            "fit_seconds": payload.get("fit_seconds"),
            "parameter_count": payload.get("parameter_count"),
        },
    )

    runs = payload.get("runs", [])
    run_checks = []
    for row in runs:
        timing = row.get("batch_elapsed_ms", {})
        run_checks.append(
            row.get("status") == "PASS"
            and row.get("frames") == 120
            and all(
                finite(timing.get(name))
                for name in ("mean_ms", "p50_ms", "p95_ms", "max_ms")
            )
            and finite(row.get("peak_working_set_mb"))
            and Path(ROOT / row["input_path"]).is_file()
            and sha256_file(ROOT / row["input_path"]) == row.get("input_sha256")
        )
    check(
        "clean_input_hashes_and_resources",
        len(run_checks) == 5 and all(run_checks),
        {"run_count": len(run_checks), "all_pass": all(run_checks)},
    )

    forbidden_metric_keys = {
        "f1",
        "precision",
        "recall",
        "event_coverage",
        "average_precision",
        "roc_auc",
        "false_alarm_rate",
    }
    observed_keys = set(payload)
    observed_keys.update(
        key for row in runs for key in row
    )
    check(
        "no_detection_metric_claims",
        observed_keys.isdisjoint(forbidden_metric_keys),
        {
            "forbidden_metric_keys_found": sorted(
                observed_keys.intersection(forbidden_metric_keys)
            )
        },
    )
    boundary_text = " ".join(
        [payload.get("protocol", ""), *payload.get("claim_boundary", [])]
    )
    check(
        "claim_boundary_present",
        "不是主检测性能结果" in boundary_text
        and "不是生产环境" in boundary_text
        and "不能支持真实事故召回率" in boundary_text,
        "resource-only and non-production boundary statements present",
    )
    check(
        "code_hash_matches",
        payload.get("code_sha256") == sha256_file(
            ROOT / "src" / "stage419_testbed_resource_only_audit.py"
        ),
        {"recorded": payload.get("code_sha256")},
    )

    failures = [item for item in checks if item["status"] == "FAIL"]
    result = {
        "status": "PASS_STAGE419_RESOURCE_SEMANTIC_HASH_AUDIT"
        if not failures
        else "FAIL_STAGE419_RESOURCE_SEMANTIC_HASH_AUDIT",
        "stage": 420,
        "input_artifact": str(INPUT.relative_to(ROOT)),
        "checks": checks,
        "failure_count": len(failures),
        "source_sha256": sha256_file(
            ROOT / "src" / "stage419_testbed_resource_only_audit.py"
        ),
        "input_artifact_sha256": sha256_file(INPUT),
        "rule": (
            "只读核验 Stage419 资源证据的完整性和语义边界；"
            "不读取攻击日志或项目test，不训练、不调阈值、不计算检测性能。"
        ),
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage420 Stage419 资源证据语义与哈希审计",
        "",
        f"- 状态：`{result['status']}`",
        "- 范围：只读 Stage419 JSON、其 clean 输入文件哈希和 Stage419 源代码哈希。",
        "- 攻击日志、攻击标签和项目 test 未读取；不训练、不调阈值、不计算检测性能。",
        "",
        "| 检查 | 状态 |",
        "|---|---|",
    ]
    lines.extend(
        f"| `{item['check']}` | `{item['status']}` |" for item in checks
    )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 该审计只确认资源证据没有被误标为检测性能结果。",
            "- 它不能关闭 70% 场景、全部基线、未见组合泛化、FAR 漂移或全面领先门槛。",
            f"- JSON：`{OUTPUT}`",
            f"- 报告：`{REPORT}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "failure_count": result["failure_count"],
                "output": str(OUTPUT),
                "report": str(REPORT),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
