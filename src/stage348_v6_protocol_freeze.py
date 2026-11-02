from __future__ import annotations
import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; V5=ROOT/"confirmatory_package"/"simulated_attack_v5"/"protocol_freeze.json"; D=ROOT/"confirmatory_package"/"simulated_attack_v6"; P=D/"protocol_freeze.json"; R=ROOT/"reports"/"stage348_v6_protocol_freeze.md"
def main():
 out=copy.deepcopy(json.loads(V5.read_text(encoding="utf-8")))
 out["status"]="SIMULATION_PROTOCOL_V6_FROZEN_BEFORE_PERFORMANCE_EVALUATION"
 out["protocol_id"]="TMQC-SIMULATED-ATTACK-V6-20260922"
 out["purpose"]="真实气象观测背景上的预注册模拟攻击安全运行点最终锁定流；V6 使用未见攻击参数组合、独立随机流和主 FAR=0.0065，不代表真实生产攻击日志。"
 out["valid_attack_seed_offset"]=1010001; out["test_attack_seed_offset"]=1020001; out["primary_far"]=0.0065
 out["attack_scenarios"]=json.loads("[{\"name\":\"low_spike\",\"base_type\":\"spike\",\"magnitude\":4.2,\"length\":4},{\"name\":\"low_offset\",\"base_type\":\"offset\",\"magnitude\":1.6,\"length\":10},{\"name\":\"slow_drift\",\"base_type\":\"drift\",\"start\":0.0,\"end\":2.0,\"length\":32},{\"name\":\"short_lag\",\"base_type\":\"time_shift\",\"lag_rows\":2,\"length\":6},{\"name\":\"delay\",\"base_type\":\"time_shift\",\"lag_rows\":6,\"length\":14},{\"name\":\"long_stuck\",\"base_type\":\"stuck\",\"length\":20},{\"name\":\"drop_replay\",\"base_type\":\"stuck\",\"length\":10},{\"name\":\"low_noise\",\"base_type\":\"gaussian_noise\",\"sigma\":0.8,\"length\":14},{\"name\":\"coordinated_offset\",\"base_type\":\"offset\",\"magnitude\":1.6,\"length\":10,\"station_count\":2,\"coordinated\":true} ]")
 out["method_freeze"]["v6_primary_far"]=0.0065; out["method_freeze"]["v6_budget_rule"]="trust 50%, EWMA 25%, ARIMA 25% of primary FAR; valid-only selection; test labels forbidden"
 D.mkdir(parents=True,exist_ok=True); P.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
 R.write_text("\n".join(["# Stage348 V6 运行点冻结","","- 状态：SIMULATION_PROTOCOL_V6_FROZEN_BEFORE_PERFORMANCE_EVALUATION",f"- 协议：`{out['protocol_id']}`","- 主 FAR：0.0065；新攻击参数与随机流已冻结。"]),encoding="utf-8")
 print(json.dumps({"status":out["status"],"protocol_id":out["protocol_id"],"path":str(P)},ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
