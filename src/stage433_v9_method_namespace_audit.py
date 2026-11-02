from __future__ import annotations

"""Stage 433: audit the V9 method namespace translation.

The legacy evaluator uses ``route_knn_arima`` internally. V9 freezes the
public candidate name as ``trust_arima``; this audit ensures that translation
and the 14-method audit contract are present before any test data are read.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
RUNNER = ROOT / "src" / "stage425_v9_locked_evaluation.py"
AUDITOR = ROOT / "src" / "stage426_v9_locked_audit.py"
BASE = ROOT / "src" / "stage340_v4_route_locked_evaluation.py"
OUT_JSON = ROOT / "tables" / "stage433_v9_method_namespace_audit.json"
OUT_REPORT = ROOT / "reports" / "stage433_v9_method_namespace_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    runner_text = RUNNER.read_text(encoding="utf-8")
    auditor_text = AUDITOR.read_text(encoding="utf-8")
    base_text = BASE.read_text(encoding="utf-8")
    baselines = list(protocol["method_freeze"]["baselines"])
    expected = ["trust_only", "trust_arima", "trust_shape_union", *baselines]
    checks = [
        {
            "check": "frozen_candidate_names_present",
            "status": (
                "PASS"
                if all(name in (runner_text + base_text) for name in expected[:3])
                else "FAIL"
            ),
            "value": expected[:3],
        },
        {
            "check": "legacy_route_translation_present",
            "status": (
                "PASS"
                if "route_knn_arima" in runner_text
                and '"route_knn_arima": "trust_arima"' in runner_text
                else "FAIL"
            ),
        },
        {
            "check": "ocsvm_budget_translation_present",
            "status": (
                "PASS"
                if '"one_class_svm": "one_class_svm_20k"' in runner_text
                and "one_class_svm_20k" in expected
                else "FAIL"
            ),
        },
        {
            "check": "scenario_factor_enrichment_present",
            "status": (
                "PASS"
                if all(
                    field in runner_text
                    for field in (
                        "factor_intensity",
                        "factor_duration",
                        "factor_length",
                        "unseen_combination",
                    )
                )
                else "FAIL"
            ),
        },
        {
            "check": "auditor_uses_protocol_baselines",
            "status": (
                "PASS"
                if "protocol[\"method_freeze\"][\"baselines\"]" in auditor_text
                and "trust_arima" in auditor_text
                and "trust_shape_union" in auditor_text
                else "FAIL"
            ),
        },
        {
            "check": "expected_method_count_is_14",
            "status": "PASS" if len(expected) == 14 else "FAIL",
            "value": {"candidate_count": 3, "baseline_count": len(baselines)},
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_METHOD_NAMESPACE_AUDIT"
        if not failures
        else "FAIL_V9_METHOD_NAMESPACE_AUDIT"
    )
    payload = {
        "stage": 433,
        "status": status,
        "protocol_id": protocol.get("protocol_id"),
        "expected_methods": expected,
        "observation_values_read": False,
        "attack_labels_read": False,
        "old_test_rows_read": False,
        "performance_evaluated": False,
        "checks": checks,
        "failed_checks": failures,
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "runner": sha256(RUNNER),
            "auditor": sha256(AUDITOR),
            "base": sha256(BASE),
        },
        "claim_boundary": [
            "本阶段只审计方法命名空间和输出适配，不产生检测性能结果。",
            "所有攻击仍是透明受控模拟攻击，不是真实生产攻击日志。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage433 V9 方法命名空间审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                f"- 预期方法数：`{len(expected)}`（3 个 Trust 候选 + 11 个基线）",
                "- 未读取观测值、攻击标签或旧 test。",
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
            {"stage": 433, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
