from __future__ import annotations
import copy, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
V4=ROOT/"confirmatory_package"/"simulated_attack_v4"/"protocol_freeze.json"
V5_DIR=ROOT/"confirmatory_package"/"simulated_attack_v5"; V5=V5_DIR/"protocol_freeze.json"
REPORT=ROOT/"reports"/"stage345_v5_protocol_freeze.md"
def main():
    out=copy.deepcopy(json.loads(V4.read_text(encoding="utf-8"))); out["status"]="SIMULATION_PROTOCOL_V5_FROZEN_BEFORE_PERFORMANCE_EVALUATION"; out["protocol_id"]="TMQC-SIMULATED-ATTACK-V5-20260922"; out["frozen_at_utc"]="2026-09-22T00:00:00Z"; out["purpose"]="真实气象观测背景上的预注册模拟攻击安全运行点锁定流；V5 使用未见攻击参数组合、独立随机流和主 FAR=0.0075，不代表真实生产攻击日志。"; out["valid_attack_seed_offset"]=990001; out["test_attack_seed_offset"]=1000001; out["primary_far"]=0.0075
    out["attack_scenarios"]=[{"name":"low_spike","base_type":"spike","magnitude":4.5,"length":4},{"name":"low_offset","base_type":"offset","magnitude":1.8,"length":12},{"name":"slow_drift","base_type":"drift","start":0.0,"end":2.2,"length":28},{"name":"short_lag","base_type":"time_shift","lag_rows":2,"length":5},{"name":"delay","base_type":"time_shift","lag_rows":5,"length":12},{"name":"long_stuck","base_type":"stuck","length":18},{"name":"drop_replay","base_type":"stuck","length":9,"description":"丢包后旧值保持不变的链路重放"},{"name":"low_noise","base_type":"gaussian_noise","sigma":0.85,"length":12},{"name":"coordinated_offset","base_type":"offset","magnitude":1.8,"length":10,"station_count":2,"coordinated":True}]
    out["method_freeze"]["v5_primary_far"]=0.0075; out["method_freeze"]["v5_budget_rule"]="trust 50%, EWMA 25%, ARIMA 25% of primary FAR; valid-only candidate selection; test labels forbidden"; out["source_code_sha256"]="PENDING_AFTER_V5_SCRIPT_WRITE"
    V5_DIR.mkdir(parents=True, exist_ok=True)
    V5.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        "# Stage345 V5 安全运行点协议冻结",
        "",
        "- 状态：SIMULATION_PROTOCOL_V5_FROZEN_BEFORE_PERFORMANCE_EVALUATION",
        f"- 协议：`{out['protocol_id']}`",
        "- 主 FAR：0.0075；预算式融合按 50%/25%/25% 分配。",
        "- V5 使用未见攻击参数组合和独立随机流，test 标签只用于最终评价。",
    ]
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"status": out["status"], "protocol_id": out["protocol_id"], "path": str(V5)}, ensure_ascii=False))
    return 0
if __name__=="__main__": raise SystemExit(main())
