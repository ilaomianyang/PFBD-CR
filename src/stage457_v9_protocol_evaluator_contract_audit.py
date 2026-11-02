from __future__ import annotations

"""Stage 457: audit the frozen V9 protocol against the executable evaluator.

This is a pre-window contract audit.  It reads protocol/source text only and
does not load official observations, historical test rows, labels, or result
tables.  Its purpose is to catch a method/scenario/endpoint mismatch before
the October 2026 official window is closed.
"""

import ast
import hashlib
import json
from pathlib import Path

from stage425_v9_compatibility import V9_SCENARIOS, validate_protocol


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
COMPAT = ROOT / "src" / "stage425_v9_compatibility.py"
EVALUATOR = ROOT / "src" / "stage425_v9_locked_evaluation.py"
BASE = ROOT / "src" / "stage340_v4_route_locked_evaluation.py"
AUDITOR = ROOT / "src" / "stage426_v9_locked_audit.py"
STATISTICS = ROOT / "src" / "stage445_v9_statistical_acceptance_audit.py"
RUNNER = ROOT / "src" / "run_v9_locked_pipeline.py"
OUT_JSON = ROOT / "tables" / "stage457_v9_protocol_evaluator_contract_audit.json"
OUT_REPORT = ROOT / "reports" / "stage457_v9_protocol_evaluator_contract_audit.md"

METHOD_RENAMES = {
    "route_knn_arima": "trust_arima",
    "one_class_svm": "one_class_svm_20k",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "BASELINES",
                    "CANDIDATES",
                    "PRIMARY_FAR",
                }:
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return values


def check(name: str, passed: bool, value: object) -> dict[str, object]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "value": value}


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    compat_text = COMPAT.read_text(encoding="utf-8")
    evaluator_text = EVALUATOR.read_text(encoding="utf-8")
    base_text = BASE.read_text(encoding="utf-8")
    auditor_text = AUDITOR.read_text(encoding="utf-8")
    statistics_text = STATISTICS.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    protocol_error: str | None = None
    try:
        validate_protocol(protocol)
    except Exception as exc:  # pragma: no cover - surfaced in the artifact
        protocol_error = f"{type(exc).__name__}: {exc}"

    frozen_baselines = [
        str(value) for value in protocol["method_freeze"]["baselines"]
    ]
    frozen_candidates = [
        "trust_only",
        "trust_arima",
        "trust_shape_union",
    ]
    base_constants = parse_constants(BASE)
    base_baselines = [
        METHOD_RENAMES.get(str(value), str(value))
        for value in base_constants.get("BASELINES", [])
    ]
    base_candidates = [
        METHOD_RENAMES.get(str(value), str(value))
        for value in base_constants.get("CANDIDATES", [])
    ]
    scenario_names = {
        str(item["name"]) for item in protocol.get("attack_scenarios", [])
    }
    expected_methods = set(frozen_candidates + frozen_baselines)

    checks = [
        check(
            "protocol_validation",
            protocol_error is None,
            protocol_error or protocol.get("protocol_id"),
        ),
        check(
            "frozen_scenario_set_matches_compatibility",
            scenario_names == set(V9_SCENARIOS) and len(scenario_names) == 13,
            sorted(scenario_names),
        ),
        check(
            "frozen_baselines_match_executable_baselines",
            frozen_baselines == base_baselines,
            {"protocol": frozen_baselines, "evaluator": base_baselines},
        ),
        check(
            "frozen_candidates_match_executable_candidates",
            frozen_candidates == base_candidates,
            {"protocol": frozen_candidates, "evaluator": base_candidates},
        ),
        check(
            "method_namespace_is_complete",
            expected_methods.issubset(
                set(frozen_candidates + frozen_baselines)
            )
            and len(expected_methods) == 14,
            {"method_count": len(expected_methods), "methods": sorted(expected_methods)},
        ),
        check(
            "v9_entrypoint_overrides_frozen_offsets_and_far",
            'base.VALID_OFFSET = int(protocol["valid_attack_seed_offset"])'
            in evaluator_text
            and 'base.TEST_OFFSET = int(protocol["test_attack_seed_offset"])'
            in evaluator_text
            and 'base.PRIMARY_FAR = float(protocol["primary_far"])' in evaluator_text,
            {
                "valid_offset_binding": "base.VALID_OFFSET" in evaluator_text,
                "test_offset_binding": "base.TEST_OFFSET" in evaluator_text,
                "far_binding": "base.PRIMARY_FAR" in evaluator_text,
            },
        ),
        check(
            "v9_entrypoint_binds_protocol_scenario_type",
            "base.simbase._scenario_base_type = scenario_type" in evaluator_text
            and "scenario_type" in compat_text,
            True,
        ),
        check(
            "scenario_namespace_renames_ocsvm_budget",
            'scenario["method"] = scenario["method"].replace(method_renames)'
            in evaluator_text
            and '"one_class_svm": "one_class_svm_20k"' in evaluator_text,
            {
                "scenario_replace_uses_method_renames": (
                    'scenario["method"] = scenario["method"].replace(method_renames)'
                    in evaluator_text
                ),
                "ocsvm_mapping": '"one_class_svm": "one_class_svm_20k"'
                in evaluator_text,
            },
        ),
        check(
            "v9_output_namespace_isolated",
            all(
                token in evaluator_text
                for token in (
                    "stage425_v9_locked_evaluation.json",
                    "stage425_v9_locked_evaluation_detail.csv",
                    "stage425_v9_locked_evaluation_scenario.csv",
                    "stage425_v9_locked_evaluation_station.csv",
                )
            )
            and 'base.OUT_JSON = ROOT / "tables" / "stage425_v9_locked_evaluation.json"'
            in evaluator_text
            and 'base.OUT_DETAIL = ROOT / "tables" / "stage425_v9_locked_evaluation_detail.csv"'
            in evaluator_text,
            True,
        ),
        check(
            "ready_manifest_gate_precedes_evaluation",
            evaluator_text.index("require_ready_data_manifest")
            < evaluator_text.index("base.main()")
            and "require_ready_data_manifest" in auditor_text,
            True,
        ),
        check(
            "statistics_gate_uses_frozen_primary_endpoint_and_far",
            "event_coverage" in statistics_text
            and "false_alarm_rate" in statistics_text
            and "benjamini_hochberg" in statistics_text
            and "far_drift" in statistics_text,
            True,
        ),
        check(
            "runner_order_is_evaluation_audit_statistics",
            runner_text.index("stage425_v9_locked_evaluation.py")
            < runner_text.index("stage426_v9_locked_audit.py")
            < runner_text.index("stage445_v9_statistical_acceptance_audit.py"),
            True,
        ),
        check(
            "provider_scope_is_frozen_to_four",
            int(protocol.get("provider_count", -1)) == 4
            and set(protocol.get("providers", {}))
            == {"MeteoSwiss", "NOAA ISD", "KNMI", "ECCC"},
            sorted(protocol.get("providers", {})),
        ),
        check(
            "no_historical_fallback_marker_in_v9_entrypoint",
            "stage349_v6_locked_evaluation.py" not in runner_text
            and "stage351_v6_station_bh_audit.py" not in runner_text,
            True,
        ),
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_PROTOCOL_EVALUATOR_CONTRACT_AUDIT"
        if not failures
        else "FAIL_V9_PROTOCOL_EVALUATOR_CONTRACT_AUDIT"
    )
    payload = {
        "stage": 457,
        "status": status,
        "protocol_id": protocol.get("protocol_id"),
        "checks": checks,
        "failed_checks": failures,
        "method_count": len(expected_methods),
        "scenario_count": len(scenario_names),
        "official_observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "claim_boundary": [
            "本阶段只审计 V9 协议与评价器契约，不产生性能结果。",
            "官方窗口未闭合前不得读取官方观测值或旧 test 替代。",
            "所有攻击仍是透明受控模拟攻击，不代表真实生产攻击日志。",
        ],
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "compatibility": sha256(COMPAT),
            "evaluator": sha256(EVALUATOR),
            "base": sha256(BASE),
            "auditor": sha256(AUDITOR),
            "statistics": sha256(STATISTICS),
            "runner": sha256(RUNNER),
        },
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage457 V9 协议与评价器契约审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                f"- 冻结方法数：`{len(expected_methods)}`；冻结场景数：`{len(scenario_names)}`",
                "- 未读取官方观测值、旧 test 行、攻击标签或性能表。",
                "- 本阶段不产生检测性能结果。",
                "",
                "## 失败检查",
                "",
                *([f"- `{item}`" for item in failures] or ["- 无。"]),
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
                "stage": 457,
                "status": status,
                "failed_checks": failures,
                "method_count": len(expected_methods),
                "scenario_count": len(scenario_names),
                "performance_evaluated": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
