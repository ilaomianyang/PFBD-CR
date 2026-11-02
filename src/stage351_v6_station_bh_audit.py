from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; TABLES=ROOT/"tables"; REPORTS=ROOT/"reports"
STATION=TABLES/"stage349_v6_locked_evaluation_station.csv"; DETAIL=TABLES/"stage349_v6_locked_evaluation_detail.csv"; SCENARIO=TABLES/"stage349_v6_locked_evaluation_scenario.csv"; PROTOCOL=ROOT/"confirmatory_package"/"simulated_attack_v6"/"protocol_freeze.json"; OUT_JSON=TABLES/"stage351_v6_station_bh_audit.json"; OUT_REPORT=REPORTS/"stage351_v6_station_bh_audit.md"
BASELINES=["robust_max_deviation","isolation_forest","one_class_svm","pca_reconstruction","mlp_autoencoder_reconstruction","knn_distance","ewma_deviation","arima_innovation","gru_predictor","lightweight_autoencoder","ics_cps_spc_proxy"]; CANDIDATES=["trust_only","route_knn_arima","trust_shape_union"]
def bh(p):
 p=np.asarray(p,dtype=float); out=np.full(len(p),np.nan); order=np.argsort(p); adj=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; out[order]=np.minimum(adj,1.0); return out
def bootstrap(diff_by_provider,seed,it=10000):
 rng=np.random.default_rng(seed); providers=sorted(diff_by_provider); draws=np.empty(it,float)
 for i in range(it):
  vals=[]
  for _ in providers:
   prov=providers[int(rng.integers(0,len(providers)))]; x=diff_by_provider[prov]; vals.append(float(np.mean(x[rng.integers(0,len(x),size=len(x))])))
  draws[i]=float(np.mean(vals))
 return float(draws.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975)),float(2*min(np.mean(draws<=0),np.mean(draws>=0)))
def main():
 protocol=json.loads(PROTOCOL.read_text(encoding="utf-8")); st=pd.read_csv(STATION); det=pd.read_csv(DETAIL); sc=pd.read_csv(SCENARIO)
 sm=sc.groupby(["scenario","method"],as_index=False).event_coverage.mean(); base=sm[sm.method.isin(BASELINES)].rename(columns={"method":"baseline","event_coverage":"baseline_event_coverage"}); cand=sm[sm.method.isin(CANDIDATES)].rename(columns={"method":"candidate","event_coverage":"candidate_event_coverage"}); mm=cand.merge(base,on="scenario",how="inner"); mm["win"]=mm.candidate_event_coverage>mm.baseline_event_coverage; scen_summary=pd.DataFrame([{"candidate":n,"scenario_win_share":float(mm[mm.candidate.eq(n)].win.mean()),"beaten_pairs":int(mm[mm.candidate.eq(n)].win.sum()),"total_pairs":int(len(mm[mm.candidate.eq(n)]))} for n in CANDIDATES]).sort_values("scenario_win_share",ascending=False); best=str(scen_summary.iloc[0]["candidate"])
 rows=[]
 for cand_name in CANDIDATES:
  left=st[st.method.eq(cand_name)]
  for base_name in BASELINES:
   right=st[st.method.eq(base_name)]
   a=left.merge(right,on=["source","seed","station_id"],suffixes=("_cand","_base"),validate="one_to_one")
   diffs=a.event_coverage_cand-a.event_coverage_base; by={}
   for provider,g in a.assign(diff=diffs).groupby("source"):
    by[str(provider)]=g["diff"].to_numpy(float)
   point,lo,hi,p=bootstrap(by,20260922+len(rows)); rows.append({"candidate":cand_name,"baseline":base_name,"event_coverage_difference_station":point,"ci95_low":lo,"ci95_high":hi,"p_two_sided":min(1.0,p),"station_rows":int(len(a)),"provider_count":int(a.source.nunique()),"candidate_f1_station_mean":float(a.f1_cand.mean()),"baseline_f1_station_mean":float(a.f1_base.mean())})
 pairs=pd.DataFrame(rows); pairs["bh_q"]=np.nan
 for cand_name in CANDIDATES:
  idx=pairs.candidate.eq(cand_name); pairs.loc[idx,"bh_q"]=bh(pairs.loc[idx,"p_two_sided"].to_numpy())
 selected=det[det.selected.astype(bool)]; sel={"f1":float(selected.f1.mean()),"event_coverage":float(selected.event_coverage.mean()),"false_alarm_rate":float(selected.false_alarm_rate.mean()),"runs":int(len(selected))}
 best_rows=pairs[pairs.candidate.eq(best)]; gate={"scenario_win_share_ge_70":bool(float(scen_summary.iloc[0]["scenario_win_share"])>=.70),"selected_mean_far_le_0.01":bool(sel["false_alarm_rate"]<=.01),"best_candidate_all_station_ci_low_positive":bool((best_rows.ci95_low>0).all()),"best_candidate_all_bh_q_lt_0.05":bool((best_rows.bh_q<.05).all())}
 payload={"status":"FINAL_V6_STATION_LEVEL_AUDIT","protocol_id":protocol["protocol_id"],"best_candidate":best,"selected_summary":sel,"scenario_summary":scen_summary.to_dict("records"),"gate":gate,"pair_summary":pairs.to_dict("records"),"limitations":["四个 provider 仍不支持机构无关声称。","所有攻击均为模拟注入，不是真实生产攻击。"]}
 OUT_JSON.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
 lines=["# Stage351 V6 站点级 bootstrap 与 BH 审计","",f"- 状态：`{payload['status']}`",f"- 协议：`{protocol['protocol_id']}`",f"- 最佳候选：`{best}`",f"- 选择运行 F1 / event coverage / FAR：`{sel['f1']:.6f}` / `{sel['event_coverage']:.6f}` / `{sel['false_alarm_rate']:.6f}`","",scen_summary.to_markdown(index=False),"","## Gate","",json.dumps(gate,ensure_ascii=False,indent=2),"","## 站点级比较","",pairs[pairs.candidate.eq(best)].to_markdown(index=False)]
 OUT_REPORT.write_text("\n".join(lines),encoding="utf-8")
 print(json.dumps({"status":payload["status"],"best_candidate":best,"selected_summary":sel,"gate":gate},ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
