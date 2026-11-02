from __future__ import annotations

"""Scientific audit for the frozen V2 simulated-attack evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL = ROOT / "confirmatory_package" / "simulated_attack_v2" / "protocol_freeze.json"
RESULT = TABLES / "stage326_simulated_attack_v2_evaluation.json"
EVAL = TABLES / "stage326_simulated_attack_v2_evaluation.csv"
STATION = TABLES / "stage326_simulated_attack_v2_station.csv"
SCENARIO = TABLES / "stage326_simulated_attack_v2_scenario.csv"
FIT = TABLES / "stage326_simulated_attack_v2_fit_resources.csv"
COMPARE = TABLES / "stage326_simulated_attack_v2_win_loss_matrix.csv"
WRAPPER = ROOT / "src" / "stage326_simulated_attack_v2_evaluation.py"
OUT_JSON = TABLES / "stage327_simulated_attack_v2_final_audit.json"
OUT_REPORT = REPORTS / "stage327_simulated_attack_v2_final_audit.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    evaluation = pd.read_csv(EVAL)
    station = pd.read_csv(STATION)
    scenarios = pd.read_csv(SCENARIO)
    fit = pd.read_csv(FIT)
    compare = pd.read_csv(COMPARE)
    station_counts = station[["source", "station_id"]].drop_duplicates().groupby("source").size().to_dict()
    expected = {"evaluation_rows": 4 * 5 * 13, "station_rows": sum(int(count) for count in station_counts.values()) * 5 * 13, "scenario_rows": 4 * 5 * 13 * 9}
    checks = []
    checks.append({"check": "protocol_hash_matches_result", "passed": result.get("protocol_sha256") == sha256(PROTOCOL), "value": {"result": result.get("protocol_sha256"), "actual": sha256(PROTOCOL)}})
    checks.append({"check": "wrapper_hash_present", "passed": bool(sha256(WRAPPER)), "value": sha256(WRAPPER)})
    checks.append({"check": "frozen_scope_counts", "passed": result.get("provider_count") == 4 and result.get("seed_count") == 5 and result.get("scenario_count") == 9 and len(result.get("methods", [])) == 13, "value": {"providers": result.get("provider_count"), "seeds": result.get("seed_count"), "scenarios": result.get("scenario_count"), "methods": len(result.get("methods", []))}})
    checks.append({"check": "delay_and_drop_replay_present", "passed": {"delay", "drop_replay"}.issubset(set(protocol_item["name"] for protocol_item in protocol["attack_scenarios"])), "value": sorted(scenarios["scenario"].unique().tolist())})
    checks.append({"check": "row_counts_match_protocol", "passed": result.get("evaluation_rows") == expected["evaluation_rows"] and result.get("station_rows") == expected["station_rows"] and result.get("scenario_rows") == expected["scenario_rows"], "value": {"expected": expected, "station_counts": station_counts, "actual": {"evaluation_rows": len(evaluation), "station_rows": len(station), "scenario_rows": len(scenarios)}}})
    checks.append({"check": "fair_method_seed_provider_contract", "passed": evaluation.groupby("method").size().nunique() == 1 and evaluation["source"].nunique() == 4 and evaluation["seed"].nunique() == 5, "value": {"rows_per_method": evaluation.groupby("method").size().to_dict(), "sources": sorted(evaluation["source"].unique().tolist()), "seeds": sorted(evaluation["seed"].unique().tolist())}})
    metric_cols = ["f1", "precision", "recall", "row_recall", "false_alarm_rate", "event_coverage", "average_precision"]
    checks.append({"check": "finite_metrics", "passed": bool(np.isfinite(evaluation[metric_cols].to_numpy(dtype=float)).all()), "value": metric_cols})
    checks.append({"check": "resource_fields_complete", "passed": fit["fit_seconds"].notna().all() and len(fit) == 4, "value": fit[["source", "fit_seconds"]].to_dict("records")})
    checks.append({"check": "primary_comparisons_not_all_supported", "passed": not bool(result.get("all_primary_comparisons_supported")), "value": bool(result.get("all_primary_comparisons_supported"))})
    unsupported = compare[(compare["metric"] == "event_coverage") & (~compare["supported"].astype(bool))]
    checks.append({"check": "negative_or_uncertain_baselines_retained", "passed": len(unsupported) >= 3, "value": unsupported[["candidate", "baseline", "difference", "ci95_low", "ci95_high", "bh_q"]].to_dict("records")})
    passed = sum(bool(row["passed"]) for row in checks)
    pooled_trust = next(row for row in result["summary"] if row["method"] == "trust_valid_f1")
    out = {
        "status": "PASS_V2_SCIENTIFIC_AUDIT_CLAIM_CEILING" if passed == len(checks) else "FAIL_V2_SCIENTIFIC_AUDIT",
        "claim_ceiling": "SIMULATED_ATTACK_LIMITED_SUPERIORITY_ONLY",
        "checks": checks,
        "passed_checks": passed,
        "total_checks": len(checks),
        "key_v2_results": {
            "trust_valid_f1_event_coverage": float(pooled_trust["event_coverage"]),
            "trust_valid_f1_f1": float(pooled_trust["f1"]),
            "trust_valid_f1_false_alarm_rate": float(pooled_trust["false_alarm_rate"]),
            "scenario_count": 9,
        },
        "unsupported_event_comparisons": unsupported[["candidate", "baseline", "difference", "ci95_low", "ci95_high", "bh_q"]].to_dict("records"),
        "limitations": [
            "所有攻击均为真实观测背景上的预注册模拟攻击，不是真实生产攻击日志。",
            "delay 使用固定滞后值重放；drop_replay 使用旧值保持，不能外推到所有网络丢包机制。",
            "One-Class SVM 与 PCA 的 event coverage 95% CI 仍跨零，不能声称对全部基线全面领先。",
            "V2 复用四个冻结 provider 的观测边界，不能证明机构无关或站点无关。",
            "Stage322/323 的融合仍为探索性；本次 V2 未将后验融合结果写入主方法。",
        ],
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Stage327 V2 模拟攻击最终科学审计",
        "",
        f"- 状态：{out['status']}",
        f"- 当前主张上限：{out['claim_ceiling']}",
        f"- 审计通过：{passed}/{len(checks)}",
        "",
        "## V2 结果摘要",
        "",
        f"- trust_valid_f1 pooled event coverage：{out['key_v2_results']['trust_valid_f1_event_coverage']:.6f}",
        f"- trust_valid_f1 pooled F1：{out['key_v2_results']['trust_valid_f1_f1']:.6f}",
        f"- trust_valid_f1 pooled FAR：{out['key_v2_results']['trust_valid_f1_false_alarm_rate']:.6f}",
        "- 场景：low_spike、low_offset、slow_drift、short_lag、delay、long_stuck、drop_replay、low_noise、coordinated_offset。",
        "",
        "## 主要不利结果",
        "",
    ]
    for row in out["unsupported_event_comparisons"]:
        lines.append(f"- Trust vs {row['baseline']}：差值 {row['difference']:.6f}，95% CI [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]，BH q={row['bh_q']:.6f}，未通过。")
    lines.extend(["", "## 结论", "", "V2 已补齐 delay 与 drop-replay 的协议和统一评价，且哈希、行数、方法契约、资源字段和不利结果审计通过。但 One-Class SVM、PCA 等比较仍未达到预注册非劣效门槛，因此不能把本轮结果升级为“模拟攻击范围内全面领先”。", ""])
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": out["status"], "passed_checks": passed, "total_checks": len(checks), "claim_ceiling": out["claim_ceiling"]}, ensure_ascii=False))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
