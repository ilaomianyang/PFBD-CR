from __future__ import annotations

"""Stage 431: audit V9 attack-factor coverage before data access.

Metadata-only audit. It reads only the frozen protocol and does not access
observation values, labels, old test outputs, or network responses.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "confirmatory_package"
    / "simulated_attack_v9_oct2026"
    / "protocol_freeze.json"
)
OUT_JSON = ROOT / "tables" / "stage431_v9_factor_coverage_audit.json"
OUT_REPORT = ROOT / "reports" / "stage431_v9_factor_coverage_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    scenarios = protocol.get("attack_scenarios", [])
    names = {str(row.get("name")) for row in scenarios}
    base_types = {str(row.get("base_type")) for row in scenarios}
    intensities = {str(row.get("factor_intensity")) for row in scenarios}
    durations = {str(row.get("factor_duration")) for row in scenarios}
    lengths = {str(row.get("factor_length")) for row in scenarios}
    coordinated = [
        row for row in scenarios if row.get("coordinated") is True
    ]
    unseen = [
        row for row in scenarios if row.get("unseen_combination") is True
    ]
    checks = [
        {
            "check": "scenario_count_exact",
            "status": "PASS" if len(scenarios) == 13 else "FAIL",
            "value": len(scenarios),
        },
        {
            "check": "attack_families_complete",
            "status": (
                "PASS"
                if base_types
                == {
                    "spike",
                    "offset",
                    "drift",
                    "time_shift",
                    "stuck",
                    "gaussian_noise",
                }
                else "FAIL"
            ),
            "value": sorted(base_types),
        },
        {
            "check": "intensity_factor_complete",
            "status": "PASS" if intensities == {"weak", "strong"} else "FAIL",
            "value": sorted(intensities),
        },
        {
            "check": "duration_factor_complete",
            "status": "PASS" if durations == {"short", "long"} else "FAIL",
            "value": sorted(durations),
        },
        {
            "check": "event_length_factor_complete",
            "status": "PASS" if lengths == {"short", "long"} else "FAIL",
            "value": sorted(lengths),
        },
        {
            "check": "unseen_coordinated_combination_exact",
            "status": (
                "PASS"
                if len(coordinated) == 1
                and len(unseen) == 1
                and coordinated[0].get("name") == "coordinated_drift_unseen"
                and unseen[0].get("station_count") == 2
                else "FAIL"
            ),
            "value": {
                "coordinated": [row.get("name") for row in coordinated],
                "unseen": [row.get("name") for row in unseen],
            },
        },
        {
            "check": "independent_test_seed_stream",
            "status": (
                "PASS"
                if protocol.get("valid_attack_seed_offset") != protocol.get(
                    "test_attack_seed_offset"
                )
                and "independent" in str(
                    protocol.get("attack_generation_rules", {}).get("test", "")
                )
                else "FAIL"
            ),
            "value": {
                "valid_attack_seed_offset": protocol.get("valid_attack_seed_offset"),
                "test_attack_seed_offset": protocol.get("test_attack_seed_offset"),
            },
        },
        {
            "check": "baseline_and_seed_freeze_complete",
            "status": (
                "PASS"
                if len(protocol.get("seeds", [])) == 5
                and len(protocol.get("method_freeze", {}).get("baselines", [])) == 11
                else "FAIL"
            ),
            "value": {
                "seed_count": len(protocol.get("seeds", [])),
                "baseline_count": len(
                    protocol.get("method_freeze", {}).get("baselines", [])
                ),
            },
        },
        {
            "check": "primary_metric_and_far_frozen",
            "status": (
                "PASS"
                if protocol.get("primary_endpoint") == "event_coverage"
                and float(protocol.get("primary_far")) == 0.0065
                else "FAIL"
            ),
            "value": {
                "primary_endpoint": protocol.get("primary_endpoint"),
                "primary_far": protocol.get("primary_far"),
            },
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_FACTOR_COVERAGE_AUDIT"
        if not failures
        else "FAIL_V9_FACTOR_COVERAGE_AUDIT"
    )
    payload = {
        "stage": 431,
        "status": status,
        "protocol_id": protocol.get("protocol_id"),
        "observation_values_read": False,
        "attack_labels_read": False,
        "old_test_rows_read": False,
        "performance_evaluated": False,
        "checks": checks,
        "failed_checks": failures,
        "protocol_sha256": sha256(PROTOCOL),
        "claim_boundary": [
            "本阶段只审计冻结攻击因子覆盖，不产生检测性能结果。",
            "攻击仍是透明受控模拟攻击，不是真实生产攻击日志。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage431 V9 攻击因子覆盖审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 只读取冻结协议，未读取观测值、攻击标签或旧 test。",
                "- 本阶段不产生性能结果。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 431, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
