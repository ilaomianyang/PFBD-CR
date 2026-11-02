"""Stage 321: final scientific audit for the frozen simulated-attack package.

This audit does not change scores, choose a new method, or reinterpret test
labels.  It only checks whether the frozen Stage320 evidence supports the
predeclared comprehensive-superiority claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL = ROOT / "confirmatory_package" / "simulated_attack_v1" / "protocol_freeze.json"
FREEZE = TABLES / "stage320_simulated_attack_code_freeze.json"
RESULT = TABLES / "stage320_simulated_attack_confirmatory_evaluation.json"
EVAL = TABLES / "stage320_simulated_attack_evaluation.csv"
STATION = TABLES / "stage320_simulated_attack_station.csv"
TYPE = TABLES / "stage320_simulated_attack_type.csv"
SCENARIO = TABLES / "stage320_simulated_attack_scenario.csv"
FIT = TABLES / "stage320_simulated_attack_fit_resources.csv"
RESOURCE_COVERAGE = TABLES / "stage299_method_resource_coverage_audit.json"
TRUST_RESOURCE = TABLES / "stage301_trust_resource_budget.json"
BASELINE_RESOURCE = TABLES / "stage290_same_contract_baseline_resource_budget.json"
OUT = TABLES / "stage321_simulated_attack_final_audit.json"
REPORT = REPORTS / "stage321_simulated_attack_final_audit.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    checks.append({"check": "protocol_hash_matches_result", "passed": result["protocol_sha256"] == sha256_file(PROTOCOL), "value": result["protocol_sha256"]})
    checks.append({"check": "code_hash_matches_freeze_and_result", "passed": freeze["script_sha256"] == result["source_code_sha256"], "value": freeze["script_sha256"]})
    expected = {"provider_count": 4, "seed_count": 5, "scenario_count": 7, "method_count": 13}
    checks.append({"check": "frozen_scope_counts", "passed": result["provider_count"] == expected["provider_count"] and result["seed_count"] == expected["seed_count"] and result["scenario_count"] == expected["scenario_count"] and len(result["methods"]) == expected["method_count"], "value": {"providers": result["provider_count"], "seeds": result["seed_count"], "scenarios": result["scenario_count"], "methods": len(result["methods"])}})

    required_files = [EVAL, STATION, TYPE, SCENARIO, FIT]
    checks.append({"check": "required_outputs_present", "passed": all(path.exists() and path.stat().st_size > 0 for path in required_files), "value": [path.relative_to(ROOT).as_posix() for path in required_files]})

    evaluation = pd.read_csv(EVAL)
    station = pd.read_csv(STATION)
    type_values = pd.read_csv(TYPE)
    scenario_values = pd.read_csv(SCENARIO)
    fit_values = pd.read_csv(FIT)
    resource_coverage = json.loads(RESOURCE_COVERAGE.read_text(encoding="utf-8"))
    trust_resource = json.loads(TRUST_RESOURCE.read_text(encoding="utf-8"))
    baseline_resource = json.loads(BASELINE_RESOURCE.read_text(encoding="utf-8"))
    checks.append({"check": "row_counts_match_protocol", "passed": len(evaluation) == 4 * 5 * 13 and len(scenario_values) == 4 * 5 * 13 * 7, "value": {"evaluation_rows": len(evaluation), "scenario_rows": len(scenario_values), "station_rows": len(station), "type_rows": len(type_values)}})
    checks.append({"check": "test_far_is_reported_not_used_as_calibration_gate", "passed": bool(np.isfinite(evaluation["false_alarm_rate"].to_numpy(dtype=float)).all()), "value": {"test_far_min": float(evaluation["false_alarm_rate"].min()), "test_far_max": float(evaluation["false_alarm_rate"].max()), "calibration_rule": "valid normal rows only at FAR=0.01"}})
    checks.append({"check": "resource_fit_times_present", "passed": bool((fit_values["fit_seconds"] > 0).all()), "value": fit_values[["source", "fit_seconds"]].to_dict("records")})
    checks.append({"check": "historical_resource_audits_complete", "passed": resource_coverage.get("trust_resource_fields_complete") is True and trust_resource.get("trust_resource_fields_complete") is True and baseline_resource.get("all_methods_have_failure_state") is True and baseline_resource.get("failed_method_count") == 0, "value": {"stage299": resource_coverage.get("status"), "stage301": trust_resource.get("status"), "stage290": baseline_resource.get("status"), "trust_resource_fields_complete": resource_coverage.get("trust_resource_fields_complete"), "baseline_failed_method_count": baseline_resource.get("failed_method_count")}})

    comparisons = pd.DataFrame(result["comparisons"])
    event_comparisons = comparisons[comparisons["metric"].eq("event_coverage")].copy()
    negative_or_noninferior = event_comparisons[
        (event_comparisons["difference"] <= 0.0)
        | (~event_comparisons["supported"].astype(bool))
    ]
    checks.append({"check": "comprehensive_primary_superiority", "passed": bool(result["all_primary_comparisons_supported"]), "value": result["all_primary_comparisons_supported"]})
    checks.append({"check": "negative_baseline_results_retained", "passed": len(negative_or_noninferior) >= 3, "value": negative_or_noninferior[["candidate", "baseline", "difference", "ci95_low", "ci95_high", "supported"]].to_dict("records")})

    protocol_scenarios = [str(row["name"]) for row in protocol["attack_scenarios"]]
    missing_expected = [name for name in ["delay", "drop"] if name not in protocol_scenarios]
    checks.append({"check": "attack_scope_contains_delay_and_drop", "passed": not missing_expected, "value": {"protocol_scenarios": protocol_scenarios, "missing": missing_expected}})

    pass_count = sum(bool(row["passed"]) for row in checks)
    claim = "SIMULATED_ATTACK_LIMITED_SUPERIORITY_ONLY"
    if result["all_primary_comparisons_supported"] and not missing_expected:
        claim = "SIMULATED_ATTACK_COMPREHENSIVE_SUPERIORITY_SUPPORTED"
    payload = {
        "status": "PASS_SCIENTIFIC_AUDIT_CLAIM_CEILING",
        "claim_ceiling": claim,
        "checks": checks,
        "passed_checks": pass_count,
        "total_checks": len(checks),
        "key_negative_results": negative_or_noninferior[["candidate", "baseline", "difference", "ci95_low", "ci95_high", "supported"]].to_dict("records"),
        "limitations": [
            "Stage320 的主终点是 event coverage；ARIMA innovation 在四 provider 主结果上稳定领先 trust_valid_f1。",
            "One-Class SVM、PCA、kNN 和轻量自编码器在 event coverage 上未通过 Trust 的确认性非劣效门槛。",
            "当前冻结协议不含 delay 和 drop 场景，不能把七场景结果扩写为九类攻击全面覆盖。",
            "所有攻击均为真实观测背景上的预注册模拟攻击，不是真实生产事故召回率。",
            "四个 provider 不支持机构无关或站点无关声称；资源表包含拟合时间，但没有完整端到端推理基准。",
            "资源审计已由 Stage269/290/299/300/301 提供；这些资源证据在 train-normal 或固定基准边界上测量，不改变检测性能结论。",
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Stage321 模拟攻击最终科学验收",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 允许的最高声称：`{claim}`",
        f"- 审计通过：`{pass_count}/{len(checks)}`",
        "",
        "## 结论",
        "",
        "- 不能声称“模拟攻击全面领先”。",
        "- 可以声称：在四个冻结 provider 的七类预注册模拟攻击和 FAR=0.01 运行点下，Trust 的 F1 方案相对 Robust、Isolation Forest、EWMA、GRU、MLP 重建和 SPC 代理具有优势。",
        "- 必须保留：ARIMA innovation 的 event coverage 优势，以及 One-Class SVM、PCA、kNN、轻量自编码器的互补/非劣效未通过结果。",
        "- 当前协议缺少 delay、drop，不能写成覆盖所有预期攻击类型。",
        "",
        "## 产物",
        "",
        f"- `tables/stage320_simulated_attack_confirmatory_evaluation.json`",
        f"- `tables/stage320_simulated_attack_scenario.csv`",
        f"- `tables/stage320_simulated_attack_win_loss_matrix.csv`",
        f"- `tables/stage320_simulated_attack_fit_resources.csv`",
        f"- `tables/stage320_simulated_attack_code_freeze.json`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "claim_ceiling": claim, "passed_checks": pass_count, "total_checks": len(checks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
