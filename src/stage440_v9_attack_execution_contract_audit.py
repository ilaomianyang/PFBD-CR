from __future__ import annotations

"""Stage 440: audit that V9 attack execution reads the frozen protocol.

This source/protocol audit does not load observations, generate attack
labels, evaluate methods, or inspect any previous test output.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
GENERATOR = ROOT / "src" / "stage320_simulated_attack_confirmatory_evaluation.py"
RUNNER = ROOT / "src" / "stage425_v9_locked_evaluation.py"
COMPAT = ROOT / "src" / "stage425_v9_compatibility.py"
OUT_JSON = ROOT / "tables" / "stage440_v9_attack_execution_contract_audit.json"
OUT_REPORT = ROOT / "reports" / "stage440_v9_attack_execution_contract_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    generator_text = GENERATOR.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    compat_text = COMPAT.read_text(encoding="utf-8")
    scenarios = protocol.get("attack_scenarios", [])
    by_name = {str(row.get("name")): row for row in scenarios}
    expected_names = {
        "weak_spike_short",
        "strong_spike_long",
        "weak_offset_short",
        "strong_offset_long",
        "short_drift",
        "long_drift",
        "short_lag",
        "long_delay",
        "short_stuck",
        "long_stuck",
        "weak_noise_short",
        "strong_noise_long",
        "coordinated_drift_unseen",
    }
    checks = [
        {
            "check": "all_frozen_scenarios_have_lengths",
            "status": "PASS"
            if set(by_name) == expected_names
            and all(int(row.get("length", 0)) > 0 for row in scenarios)
            else "FAIL",
            "value": len(scenarios),
        },
        {
            "check": "intensity_duration_length_fields_are_consumed",
            "status": "PASS"
            if 'scenario["length"]' in generator_text
            and 'scenario["magnitude"]' in generator_text
            and 'scenario["start"]' in generator_text
            and 'scenario["end"]' in generator_text
            and 'scenario["lag_rows"]' in generator_text
            and 'scenario["sigma"]' in generator_text
            else "FAIL",
        },
        {
            "check": "time_shift_uses_frozen_lag_rows",
            "status": "PASS"
            if 'scenario["lag_rows"]' in generator_text
            and "source_row_position" in generator_text
            and "value replayed" in str(protocol.get("attack_generation_rules", {}).get("delay", ""))
            else "FAIL",
        },
        {
            "check": "coordinated_scenario_uses_two_stations_and_common_times",
            "status": "PASS"
            if by_name.get("coordinated_drift_unseen", {}).get("coordinated") is True
            and by_name.get("coordinated_drift_unseen", {}).get("station_count") == 2
            and "common_times" in generator_text
            and "pair = stations[:2]" in generator_text
            else "FAIL",
        },
        {
            "check": "unseen_combination_is_declared_and_not_selected_from_test",
            "status": "PASS"
            if by_name.get("coordinated_drift_unseen", {}).get("unseen_combination") is True
            and "test labels are evaluation-only" in str(
                protocol.get("attack_generation_rules", {}).get("test", "")
            )
            and "unseen_combination" in RUNNER.read_text(encoding="utf-8")
            else "FAIL",
        },
        {
            "check": "valid_and_test_seed_streams_are_distinct",
            "status": "PASS"
            if protocol.get("valid_attack_seed_offset")
            != protocol.get("test_attack_seed_offset")
            and "VALID_OFFSET" in runner_text
            and "TEST_OFFSET" in runner_text
            else "FAIL",
        },
        {
            "check": "v9_runner_overrides_scenario_mapping",
            "status": "PASS"
            if "base.simbase._scenario_base_type = scenario_type" in runner_text
            and "def scenario_type" in compat_text
            else "FAIL",
        },
        {
            "check": "no_legacy_test_fallback",
            "status": "PASS"
            if "old normalized files must not be substituted" in compat_text
            and "require_ready_data_manifest" in runner_text
            else "FAIL",
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_ATTACK_EXECUTION_CONTRACT_AUDIT"
        if not failures
        else "FAIL_V9_ATTACK_EXECUTION_CONTRACT_AUDIT"
    )
    payload = {
        "stage": 440,
        "status": status,
        "protocol_id": protocol.get("protocol_id"),
        "checks": checks,
        "failed_checks": failures,
        "observation_values_read": False,
        "attack_labels_read": False,
        "old_test_rows_read": False,
        "performance_evaluated": False,
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "generator": sha256(GENERATOR),
            "runner": sha256(RUNNER),
            "compatibility": sha256(COMPAT),
        },
        "claim_boundary": [
            "本阶段只审计攻击执行代码是否读取冻结协议，不产生性能结果。",
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
                "# Stage440 V9 攻击执行契约审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 仅读取冻结协议和源码，未读取观测值、攻击标签或旧 test。",
                "- 已核对强度、持续时间、事件长度、延迟、协调漂移和独立随机流。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 440, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
