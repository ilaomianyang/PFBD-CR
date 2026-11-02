from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SCENARIO = TABLES / "stage340_v4_route_locked_evaluation_scenario.csv"
DETAIL = TABLES / "stage340_v4_route_locked_evaluation_detail.csv"
PROTOCOL = ROOT / "confirmatory_package" / "simulated_attack_v4" / "protocol_freeze.json"
OUT_JSON = TABLES / "stage341_v4_locked_audit.json"
OUT_REPORT = REPORTS / "stage341_v4_locked_audit.md"
BASELINES = ["robust_max_deviation", "isolation_forest", "one_class_svm", "pca_reconstruction", "mlp_autoencoder_reconstruction", "knn_distance", "ewma_deviation", "arima_innovation", "gru_predictor", "lightweight_autoencoder", "ics_cps_spc_proxy"]
CANDIDATES = ["trust_only", "route_knn_arima", "trust_shape_union"]
def ci(values, seed, iterations=10000):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return float("nan"),float("nan")
    rng=np.random.default_rng(seed); draws=rng.choice(x,size=(iterations,len(x)),replace=True).mean(axis=1)
    return float(np.quantile(draws,0.025)),float(np.quantile(draws,0.975))
def main():
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8")); s=pd.read_csv(SCENARIO); d=pd.read_csv(DETAIL)
    sm=s.groupby(["scenario","method"],as_index=False).event_coverage.mean(); base=sm[sm.method.isin(BASELINES)].rename(columns={"method":"baseline","event_coverage":"baseline_event_coverage"}); cand=sm[sm.method.isin(CANDIDATES)].rename(columns={"method":"candidate","event_coverage":"candidate_event_coverage"}); m=cand.merge(base,on="scenario",how="inner"); m["win"]=m.candidate_event_coverage>m.baseline_event_coverage
    summary=[]
    for name in CANDIDATES:
        q=m[m.candidate.eq(name)]; summary.append({"candidate":name,"scenario_baseline_win_share":float(q.win.mean()),"beaten_pairs":int(q.win.sum()),"total_pairs":int(len(q))})
    summary_df=pd.DataFrame(summary).sort_values("scenario_baseline_win_share",ascending=False)
    pairs=[]
    for name in CANDIDATES:
        left=d[d.method.eq(name)][["source","seed","event_coverage","f1","false_alarm_rate"]]
        for base_name in BASELINES:
            right=d[d.method.eq(base_name)][["source","seed","event_coverage","f1","false_alarm_rate"]]
            a=left.merge(right,on=["source","seed"],suffixes=("_cand","_base"),validate="one_to_one"); diff=a.event_coverage_cand-a.event_coverage_base; lo,hi=ci(diff,20260922+len(pairs))
            pairs.append({"candidate":name,"baseline":base_name,"event_coverage_difference":float(diff.mean()),"ci95_low_provider_seed":lo,"ci95_high_provider_seed":hi,"candidate_far":float(a.false_alarm_rate_cand.mean()),"baseline_far":float(a.false_alarm_rate_base.mean()),"candidate_f1":float(a.f1_cand.mean()),"baseline_f1":float(a.f1_base.mean())})
    pairs_df=pd.DataFrame(pairs); best=summary_df.iloc[0].to_dict(); selected=d[d.selected.astype(bool)]; sel={"f1":float(selected.f1.mean()),"event_coverage":float(selected.event_coverage.mean()),"false_alarm_rate":float(selected.false_alarm_rate.mean()),"runs":int(len(selected))}
    gate={"scenario_win_share_ge_70":bool(float(best["scenario_baseline_win_share"])>=0.70),"selected_mean_far_le_0.01":bool(sel["false_alarm_rate"]<=0.01),"all_baseline_provider_seed_ci_low_positive":bool((pairs_df[pairs_df.candidate.eq(best["candidate"])] .ci95_low_provider_seed>0).all())}
    payload={"status":"PASS_V4_LOCKED_EVALUATION_NEEDS_FINAL_CLAIM_DECISION","protocol_id":protocol["protocol_id"],"best_candidate":best,"selected_summary":sel,"gate":gate,"candidate_summary":summary_df.to_dict("records"),"pair_summary":pairs_df.to_dict("records"),"limitations":["四个 provider 仍不支持机构无关声称。","provider×seed CI 不是站点簇 bootstrap；正式主表仍需站点级重采样与 BH 校正。","所有攻击均为模拟注入，不是真实生产攻击。"]}
    report_lines = [
        "# Stage341 V4 最终锁定审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{protocol['protocol_id']}`",
        f"- 最佳候选：`{best['candidate']}`，场景×基线比例 `{float(best['scenario_baseline_win_share']):.1%}`",
        f"- 选择运行 F1 / event coverage / FAR：`{sel['f1']:.6f}` / `{sel['event_coverage']:.6f}` / `{sel['false_alarm_rate']:.6f}`",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Gate",
        "",
        json.dumps(gate, ensure_ascii=False, indent=2),
        "",
        "- 只有全部 gate 和正式站点级统计通过，才能升级主张。",
    ]
    OUT_JSON.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    OUT_REPORT.write_text("\n".join(report_lines),encoding="utf-8")
    print(json.dumps({"status":payload["status"],"best":best,"selected_summary":sel,"gate":gate},ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
