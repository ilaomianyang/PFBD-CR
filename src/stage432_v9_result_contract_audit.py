from __future__ import annotations

"""Stage 432: audit the V9 result-table contract before data access.

This metadata/source audit ensures the later locked run can produce all
pre-registered metrics and factor-level tables without inspecting test rows.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
RUNNER = ROOT / "src" / "stage425_v9_locked_evaluation.py"
BASE = ROOT / "src" / "stage340_v4_route_locked_evaluation.py"
AUDITOR = ROOT / "src" / "stage426_v9_locked_audit.py"
OUT_JSON = ROOT / "tables" / "stage432_v9_result_contract_audit.json"
OUT_REPORT = ROOT / "reports" / "stage432_v9_result_contract_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    runner_text = RUNNER.read_text(encoding="utf-8")
    base_text = BASE.read_text(encoding="utf-8")
    auditor_text = AUDITOR.read_text(encoding="utf-8")

    detail_metrics = {
        "f1",
        "precision",
        "recall",
        "row_recall",
        "average_precision",
        "false_alarm_rate",
        "event_coverage",
    }
    detail_tokens = {
        "source",
        "seed",
        "method",
        "event_metrics",
        "metrics",
        "OUT_DETAIL",
    }
    scenario_columns = {
        "source",
        "seed",
        "method",
        "scenario",
        "event_coverage",
        "row_recall",
        "events",
    }
    station_columns = {
        "source",
        "seed",
        "method",
        "station_id",
        "tp",
        "fp",
        "fn",
        "tn",
        "f1",
        "event_coverage",
        "events",
        "detected_events",
    }
    all_scenario_names = {
        str(row.get("name")) for row in protocol.get("attack_scenarios", [])
    }
    checks = [
        {
            "check": "v9_output_namespace_isolated",
            "status": (
                "PASS"
                if all(
                    token in runner_text
                    for token in (
                        "stage425_v9_locked_evaluation.json",
                        "stage425_v9_locked_evaluation_detail.csv",
                        "stage425_v9_locked_evaluation_scenario.csv",
                        "stage425_v9_locked_evaluation_station.csv",
                    )
                )
                else "FAIL"
            ),
        },
        {
            "check": "detail_metrics_contract_present",
            "status": (
                "PASS"
                if all(token in base_text for token in detail_tokens)
                and all(metric in base_text for metric in detail_metrics)
                else "FAIL"
            ),
            "value": sorted(detail_metrics),
        },
        {
            "check": "scenario_metrics_contract_present",
            "status": (
                "PASS"
                if all(
                    token in base_text
                    for token in (
                        "scenario_rows",
                        "event_coverage",
                        "row_recall",
                        "events",
                    )
                )
                else "FAIL"
            ),
            "value": sorted(scenario_columns),
        },
        {
            "check": "station_metrics_contract_present",
            "status": (
                "PASS"
                if all(
                    token in base_text
                    for token in (
                        "station_rows",
                        '"tp"',
                        '"fp"',
                        '"fn"',
                        '"tn"',
                        '"detected_events"',
                    )
                )
                else "FAIL"
            ),
            "value": sorted(station_columns),
        },
        {
            "check": "all_frozen_scenarios_audited",
            "status": (
                "PASS"
                if all_scenario_names
                and "scenario_names_complete" in auditor_text
                and "attack_scenario" in base_text
                else "FAIL"
            ),
            "value": sorted(all_scenario_names),
        },
        {
            "check": "required_reporting_levels_frozen",
            "status": (
                "PASS"
                if all(
                    level in str(protocol.get("acceptance_gates", {}).get("reporting", []))
                    for level in (
                        "pooled",
                        "station_equal",
                        "provider_equal",
                    )
                )
                else "FAIL"
            ),
            "value": protocol.get("acceptance_gates", {}).get("reporting", []),
        },
        {
            "check": "simulated_attack_boundary_present",
            "status": (
                "PASS"
                if "controlled simulated-attack claims" in str(
                    protocol.get("claim_boundary", [])
                )
                and "production incident" in str(protocol.get("claim_boundary", []))
                else "FAIL"
            ),
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_RESULT_CONTRACT_AUDIT"
        if not failures
        else "FAIL_V9_RESULT_CONTRACT_AUDIT"
    )
    payload = {
        "stage": 432,
        "status": status,
        "protocol_id": protocol.get("protocol_id"),
        "observation_values_read": False,
        "attack_labels_read": False,
        "old_test_rows_read": False,
        "performance_evaluated": False,
        "checks": checks,
        "failed_checks": failures,
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "runner": sha256(RUNNER),
            "base": sha256(BASE),
            "auditor": sha256(AUDITOR),
        },
        "claim_boundary": [
            "本阶段只审计结果表和统计契约，不产生检测性能结果。",
            "未来结果必须同时保留 pooled、站点等权和 provider 等权层级。",
            "所有攻击均为透明受控模拟攻击，不是真实生产攻击日志。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage432 V9 结果表契约审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 只读取协议和源码，不读取观测值、攻击标签或旧 test。",
                "- 本阶段不产生检测性能结果。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 432, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
