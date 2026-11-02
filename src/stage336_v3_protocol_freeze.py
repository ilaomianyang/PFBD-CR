from __future__ import annotations
"""Freeze the new V3 attack stream before any V3 test evaluation."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
V2 = ROOT / "confirmatory_package" / "simulated_attack_v2" / "protocol_freeze.json"
V3_DIR = ROOT / "confirmatory_package" / "simulated_attack_v3"
V3 = V3_DIR / "protocol_freeze.json"
REPORT = ROOT / "reports" / "stage336_v3_protocol_freeze.md"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    frozen = json.loads(V2.read_text(encoding="utf-8"))
    out = copy.deepcopy(frozen)
    out["status"] = "SIMULATION_PROTOCOL_V3_FROZEN_BEFORE_PERFORMANCE_EVALUATION"
    out["protocol_id"] = "TMQC-SIMULATED-ATTACK-V3-20260922"
    out["frozen_at_utc"] = "2026-09-22T00:00:00Z"
    out["purpose"] = "真实气象观测背景上的预注册模拟攻击独立锁定流；V3 使用未见参数组合和独立攻击随机流，不代表真实生产攻击日志。"
    out["valid_attack_seed_offset"] = 950001
    out["test_attack_seed_offset"] = 960001
    out["attack_scenarios"] = [
        {"name": "low_spike", "base_type": "spike", "magnitude": 3.5, "length": 3},
        {"name": "low_offset", "base_type": "offset", "magnitude": 1.5, "length": 8},
        {"name": "slow_drift", "base_type": "drift", "start": 0.0, "end": 2.0, "length": 36},
        {"name": "short_lag", "base_type": "time_shift", "lag_rows": 1, "length": 4},
        {"name": "delay", "base_type": "time_shift", "lag_rows": 6, "length": 16},
        {"name": "long_stuck", "base_type": "stuck", "length": 24},
        {"name": "drop_replay", "base_type": "stuck", "length": 12, "description": "丢包后旧值保持不变的链路重放"},
        {"name": "low_noise", "base_type": "gaussian_noise", "sigma": 0.75, "length": 16},
        {"name": "coordinated_offset", "base_type": "offset", "magnitude": 1.5, "length": 12, "station_count": 2, "coordinated": True},
    ]
    out["ensemble_freeze"] = {
        "candidate_family": {
            "trust_only": [["trust_valid_f1", 0.01]],
            "trust_arima": [["trust_valid_f1", 0.0075], ["arima_innovation", 0.0025]],
            "trust_shape_union": [["trust_valid_f1", 0.005], ["ewma_deviation", 0.0025], ["arima_innovation", 0.0025]],
        },
        "selection": "valid only; retain at least 95% of trust_valid_f1 validation F1, then maximize validation event coverage, then F1; test labels forbidden",
        "primary_far": 0.01,
    }
    out["claim_boundary"] = [
        "This package supports only simulated/controlled attack claims.",
        "V3 is a new attack-parameter and random-stream lock on the same four providers; it does not establish institution independence.",
        "All baseline wins and failures must remain in the final report.",
    ]
    out["source_code_sha256"] = sha256(SRC / "stage334_exploratory_budgeted_ensemble_search.py")
    V3_DIR.mkdir(parents=True, exist_ok=True)
    V3.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text("\n".join(["# Stage336 V3 独立锁定协议冻结", "", "- 状态：SIMULATION_PROTOCOL_V3_FROZEN_BEFORE_PERFORMANCE_EVALUATION", f"- 协议：`{out['protocol_id']}`", "- 仍使用四个冻结 provider，但改用未见攻击参数组合与独立随机流。", "- 候选族、预算、valid 选择规则和主 FAR 已在查看 V3 test 前冻结。", "- 所有结果仍属于模拟攻击评价，不能写成真实生产攻击召回率。", ""]), encoding="utf-8")
    print(json.dumps({"status": out["status"], "protocol_id": out["protocol_id"], "path": str(V3), "source_code_sha256": out["source_code_sha256"]}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
