from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "confirmatory_package" / "simulated_attack_v3" / "protocol_freeze.json"
V4_DIR = ROOT / "confirmatory_package" / "simulated_attack_v4"
V4 = V4_DIR / "protocol_freeze.json"
REPORT = ROOT / "reports" / "stage342_v4_protocol_freeze.md"
SRC_HASH_FILES = [ROOT / "src" / "stage340_v4_route_locked_evaluation.py"]
def sha256(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def main() -> int:
    out = copy.deepcopy(json.loads(V3.read_text(encoding="utf-8")))
    out["status"] = "SIMULATION_PROTOCOL_V4_FROZEN_BEFORE_PERFORMANCE_EVALUATION"
    out["protocol_id"] = "TMQC-SIMULATED-ATTACK-V4-20260922"
    out["frozen_at_utc"] = "2026-09-22T00:00:00Z"
    out["purpose"] = "真实气象观测背景上的预注册模拟攻击最终锁定流；V4 使用未见攻击参数组合与独立随机流，不代表真实生产攻击日志。"
    out["valid_attack_seed_offset"] = 970001
    out["test_attack_seed_offset"] = 980001
    out["attack_scenarios"] = [
        {"name": "low_spike", "base_type": "spike", "magnitude": 4.0, "length": 5},
        {"name": "low_offset", "base_type": "offset", "magnitude": 2.0, "length": 10},
        {"name": "slow_drift", "base_type": "drift", "start": 0.0, "end": 2.5, "length": 30},
        {"name": "short_lag", "base_type": "time_shift", "lag_rows": 3, "length": 5},
        {"name": "delay", "base_type": "time_shift", "lag_rows": 5, "length": 14},
        {"name": "long_stuck", "base_type": "stuck", "length": 20},
        {"name": "drop_replay", "base_type": "stuck", "length": 10, "description": "丢包后旧值保持不变的链路重放"},
        {"name": "low_noise", "base_type": "gaussian_noise", "sigma": 0.9, "length": 14},
        {"name": "coordinated_offset", "base_type": "offset", "magnitude": 2.0, "length": 10, "station_count": 2, "coordinated": True},
    ]
    out["method_freeze"]["v4_candidate_family"] = ["trust_only", "route_knn_arima", "trust_shape_union"]
    out["method_freeze"]["v4_selection_rule"] = "valid only; retain at least 95% of trust_only validation F1, maximize validation event coverage, tie by F1; test labels forbidden"
    out["claim_boundary"] = ["This package supports only simulated/controlled attack claims.", "V4 is a new attack-parameter and random-stream lock on the same four providers.", "All baseline wins and failures remain reportable."]
    out["source_code_sha256"] = "PENDING_AFTER_SCRIPT_WRITE"
    V4_DIR.mkdir(parents=True, exist_ok=True); V4.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text("\n".join(["# Stage342 V4 最终锁定协议冻结", "", "- 状态：SIMULATION_PROTOCOL_V4_FROZEN_BEFORE_PERFORMANCE_EVALUATION", f"- 协议：`{out['protocol_id']}`", "- 候选：trust_only、route_knn_arima、trust_shape_union。", "- V4 test 攻击参数和随机流在运行前冻结，test 标签只用于最终评价。", "- 所有攻击仍是模拟注入。", ""]), encoding="utf-8")
    print(json.dumps({"status": out["status"], "protocol_id": out["protocol_id"], "path": str(V4)}, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
