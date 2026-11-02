from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL_PATH = ROOT / "confirmatory_package" / "simulated_attack_v4" / "protocol_freeze.json"
OUT_JSON = TABLES / "stage340_v4_route_locked_evaluation.json"
OUT_DETAIL = TABLES / "stage340_v4_route_locked_evaluation_detail.csv"
OUT_SCENARIO = TABLES / "stage340_v4_route_locked_evaluation_scenario.csv"
OUT_STATION = TABLES / "stage340_v4_route_locked_evaluation_station.csv"
OUT_REPORT = REPORTS / "stage340_v4_route_locked_evaluation.md"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
import stage320_simulated_attack_confirmatory_evaluation as simbase  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import detector_frame, fit_detectors, metrics, score_components  # noqa: E402
from stage208_dwd_persistent_repeat_protection import event_metrics, far_threshold  # noqa: E402
from stage217_swiss_controlled_attack_evaluation import fuse_scores, robust_standardize  # noqa: E402
from stage240_nonstuck_balanced_selection import PROTOCOLS, make_frame  # noqa: E402
from stage244_cross_provider_nonstuck_candidate_features import candidate_features, select_candidate_stage244  # noqa: E402
from stage248_cross_provider_fair_baseline_audit import feature_matrix, fit_baselines, score_baselines  # noqa: E402
from stage256_lightweight_strong_baselines import fit_lightweight_baselines, score_lightweight_baselines  # noqa: E402
from stage262_security_domain_baselines import fit_security_baselines, score_security_baselines  # noqa: E402
from stage326_simulated_attack_v2_evaluation import _internal_type  # noqa: E402
simbase._scenario_base_type = _internal_type
SEEDS = (3407, 6257, 7919, 104729, 20260715)
VALID_OFFSET = 970001
TEST_OFFSET = 980001
PRIMARY_FAR = 0.01
F1_FLOOR = 0.95
BASELINES = ["robust_max_deviation", "isolation_forest", "one_class_svm", "pca_reconstruction", "mlp_autoencoder_reconstruction", "knn_distance", "ewma_deviation", "arima_innovation", "gru_predictor", "lightweight_autoencoder", "ics_cps_spc_proxy"]
CANDIDATES = ["trust_only", "route_knn_arima", "trust_shape_union"]

def load_json(p: Path) -> dict: return json.loads(p.read_text(encoding="utf-8"))

def norm_pair(valid: np.ndarray, test: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(valid, dtype=float)[normal]; ref = ref[np.isfinite(ref)]
    med = float(np.median(ref)) if len(ref) else 0.0
    scale = max(1.4826 * float(np.median(np.abs(ref - med))) if len(ref) else 1.0, 1e-6)
    return np.nan_to_num((valid - med) / scale), np.nan_to_num((test - med) / scale)

def route(values: dict[str, np.ndarray], feats: dict[tuple[str, int], np.ndarray], ref_feats: dict[tuple[str, int], np.ndarray], normal: np.ndarray, test_mode: bool = False) -> np.ndarray:
    curvature = feats[("second_diff_zscore", 1)]
    drift = feats[("slope_6", 1)]
    noise = feats[("noise_ratio", 1)]
    ref_curv = ref_feats[("second_diff_zscore", 1)][normal]
    ref_drift = ref_feats[("slope_6", 1)][normal]
    ref_noise = ref_feats[("noise_ratio", 1)][normal]
    qc = float(np.quantile(ref_curv, 0.90)); qd = float(np.quantile(ref_drift, 0.90)); qn = float(np.quantile(ref_noise, 0.90))
    out = values["trust_valid_f1"].copy()
    noise_mask = noise >= qn
    drift_mask = (noise < qn) & (drift >= qd)
    curv_mask = (noise < qn) & (drift < qd) & (curvature >= qc)
    out[noise_mask] = values["trust_valid_f1"][noise_mask]
    out[drift_mask] = values["arima_innovation"][drift_mask]
    out[curv_mask] = values["knn_distance"][curv_mask]
    return out

def scenario_rows(frame: pd.DataFrame, pred: np.ndarray, source: str, seed: int, method: str) -> list[dict[str, object]]:
    attack = frame["is_attack"].astype(bool).to_numpy(); labels = frame["attack_scenario"].astype(str).to_numpy(); out=[]
    for scenario in sorted(set(labels[attack])):
        mask = attack & (labels == scenario); e = frame.loc[mask, ["attack_event_id"]].copy(); e["predicted"] = pred[mask]; g=e.groupby("attack_event_id", sort=True)["predicted"].any()
        out.append({"source":source,"seed":int(seed),"method":method,"scenario":scenario,"event_coverage":float(g.mean()) if len(g) else 0.0,"row_recall":float(pred[mask].mean()) if mask.any() else 0.0,"events":int(len(g))})
    return out

def station_rows(frame: pd.DataFrame, pred: np.ndarray, source: str, seed: int, method: str) -> list[dict[str, object]]:
    rows = []
    attack = frame["is_attack"].astype(bool).to_numpy()
    observed = frame["observation_present"].astype(bool).to_numpy()
    for station, part in frame.groupby("station_id", sort=True):
        idx = part.index.to_numpy(dtype=int)
        a = attack[idx] & observed[idx]
        p = pred[idx] & observed[idx]
        tp = int((p & a).sum()); fp = int((p & ~a).sum()); fn = int((~p & a).sum()); tn = int((~p & ~a & observed[idx]).sum())
        ev = part.loc[a, ["attack_event_id"]].copy(); ev["predicted"] = p[a]
        grouped = ev.groupby("attack_event_id", sort=True)["predicted"].any()
        denom = 2 * tp + fp + fn
        rows.append({"source": source, "seed": int(seed), "method": method, "station_id": str(station), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": float(2 * tp / denom) if denom else 0.0, "event_coverage": float(grouped.mean()) if len(grouped) else 0.0, "events": int(len(grouped)), "detected_events": int(grouped.sum())})
    return rows

def main() -> int:
    frozen = load_json(PROTOCOL_PATH); scenarios = {str(x["name"]):x for x in frozen["attack_scenarios"]}
    detail=[]; scen=[]; station_detail=[]; selection_counts={}
    for source, protocol_path in PROTOCOLS.items():
        protocol=load_json(protocol_path); frame=detector_frame(make_frame(source, protocol)); detectors=fit_detectors(frame, protocol)
        train_normal=(frame["split"].eq("train") & frame["observation_present"]).to_numpy(dtype=bool)
        comp=score_components(frame, detectors); fused,_=fuse_scores(frame, comp); base_z=robust_standardize(fused, train_normal).to_numpy(dtype=float); feats_base=candidate_features(frame, train_normal); matrix,_=feature_matrix(comp, feats_base, train_normal)
        old=fit_baselines(frame, matrix, train_normal); light,_=fit_lightweight_baselines(frame, matrix, train_normal); sec,_=fit_security_baselines(frame, matrix, train_normal)
        for seed in SEEDS:
            vb=simbase.make_simulated_blocks(frame, frozen, seed, "valid", VALID_OFFSET); tb=simbase.make_simulated_blocks(frame, frozen, seed, "test", TEST_OFFSET)
            valid=simbase.inject_simulated_attack(frame, vb, scenarios); test=simbase.inject_simulated_attack(frame, tb, scenarios)
            vc=score_components(valid, detectors); tc=score_components(test, detectors); vf,_=fuse_scores(valid, vc); tf,_=fuse_scores(test, tc)
            vz=robust_standardize(vf, train_normal).to_numpy(dtype=float); tz=robust_standardize(tf, train_normal).to_numpy(dtype=float); fv=candidate_features(valid, train_normal); ft=candidate_features(test, train_normal)
            vm=valid["split"].eq("valid").to_numpy(dtype=bool); tm=test["split"].eq("test").to_numpy(dtype=bool); vf_frame=valid.loc[vm].reset_index(drop=True); tf_frame=test.loc[tm].reset_index(drop=True); normal=(vf_frame["observation_present"] & ~vf_frame["is_attack"]).to_numpy(dtype=bool)
            vmatrix,_=feature_matrix(vc, fv, train_normal); tmatrix,_=feature_matrix(tc, ft, train_normal)
            vs={"trust_valid_f1":vz[vm]}; ts={"trust_valid_f1":tz[tm]}
            selected=select_candidate_stage244("valid_f1", vz[vm], {k:v[vm] for k,v in fv.items()}, vf_frame, normal, PRIMARY_FAR); key=(str(selected["feature"]), int(selected["window"]))
            vs["trust_valid_f1"]=(1.0-float(selected["weight"])) * vz[vm] + float(selected["weight"]) * fv[key][vm]; ts["trust_valid_f1"]=(1.0-float(selected["weight"])) * tz[tm] + float(selected["weight"]) * ft[key][tm]
            vs.update(score_baselines(old, vmatrix[vm])); ts.update(score_baselines(old, tmatrix[tm])); vs.update(score_lightweight_baselines(light, vmatrix[vm])); ts.update(score_lightweight_baselines(light, tmatrix[tm])); vs.update(score_security_baselines(sec, vf_frame, vmatrix[vm])); ts.update(score_security_baselines(sec, tf_frame, tmatrix[tm]))
            nv={}; nt={}
            for method in vs: nv[method],nt[method]=norm_pair(np.asarray(vs[method]), np.asarray(ts[method]), normal)
            nv["route_knn_arima"]=route(nv, {k:v[vm] for k,v in fv.items()}, {k:v[vm] for k,v in fv.items()}, normal); nt["route_knn_arima"]=route(nt, {k:v[tm] for k,v in ft.items()}, {k:v[vm] for k,v in fv.items()}, normal, True)
            # Budgeted shape union uses valid-only thresholds and is applied identically to test.
            union_valid=np.zeros(len(normal),dtype=bool); union_test=np.zeros(len(tf_frame),dtype=bool)
            for method,budget in (("trust_valid_f1",0.50 * PRIMARY_FAR),("ewma_deviation",0.25 * PRIMARY_FAR),("arima_innovation",0.25 * PRIMARY_FAR)):
                thr=far_threshold(nv[method], normal, budget); union_valid |= nv[method] >= thr; union_test |= nt[method] >= thr
            nv["trust_shape_union"]=np.maximum.reduce([nv["trust_valid_f1"],nv["ewma_deviation"],nv["arima_innovation"]]); nt["trust_shape_union"]=np.maximum.reduce([nt["trust_valid_f1"],nt["ewma_deviation"],nt["arima_innovation"]]); yv=vf_frame["is_attack"].astype(bool).to_numpy(); yt=tf_frame["is_attack"].astype(bool).to_numpy(); obs_v=vf_frame["observation_present"].to_numpy(dtype=bool); obs_t=tf_frame["observation_present"].to_numpy(dtype=bool)
            valid_preds={}; test_preds={}; valid_metrics={}
            for method in ["trust_only","route_knn_arima"]:
                score_method="trust_valid_f1" if method=="trust_only" else method; thr=far_threshold(nv[score_method], normal, PRIMARY_FAR); valid_preds[method]=obs_v & (nv[score_method]>=thr); test_preds[method]=obs_t & (nt[score_method]>=thr); valid_metrics[method]=metrics(yv,valid_preds[method],nv[score_method])
            valid_preds["trust_shape_union"]=union_valid & obs_v; test_preds["trust_shape_union"]=union_test & obs_t; valid_metrics["trust_shape_union"]=metrics(yv,valid_preds["trust_shape_union"],nv["trust_shape_union"])
            trust_f1=valid_metrics["trust_only"]["f1"]; eligible=[m for m in CANDIDATES if valid_metrics[m]["f1"]>=F1_FLOOR*trust_f1]; chosen=max(eligible,key=lambda m:(event_metrics(vf_frame,valid_preds[m])["event_coverage"],valid_metrics[m]["f1"])) if eligible else "trust_only"; selection_counts[chosen]=selection_counts.get(chosen,0)+1
            for method in CANDIDATES:
                pred=test_preds[method]; score=nt["trust_valid_f1"] if method=="trust_only" else nt[method]; met=metrics(yt,pred,score); ev=event_metrics(tf_frame,pred); detail.append({"source":source,"seed":int(seed),"method":method,"selected":bool(method==chosen),"valid_f1":float(valid_metrics[method]["f1"]),"valid_event_coverage":float(event_metrics(vf_frame,valid_preds[method])["event_coverage"]),**met,**ev}); scen.extend(scenario_rows(tf_frame,pred,source,int(seed),method)); station_detail.extend(station_rows(tf_frame,pred,source,int(seed),method))
            for method in BASELINES:
                thr=far_threshold(nv[method],normal,PRIMARY_FAR); pred=obs_t & (nt[method]>=thr); met=metrics(yt,pred,nt[method]); ev=event_metrics(tf_frame,pred); detail.append({"source":source,"seed":int(seed),"method":method,"selected":False,"valid_f1":float("nan"),"valid_event_coverage":float("nan"),**met,**ev}); scen.extend(scenario_rows(tf_frame,pred,source,int(seed),method)); station_detail.extend(station_rows(tf_frame,pred,source,int(seed),method))
    detail_df=pd.DataFrame(detail); scen_df=pd.DataFrame(scen); station_df=pd.DataFrame(station_detail); detail_df.to_csv(OUT_DETAIL,index=False); scen_df.to_csv(OUT_SCENARIO,index=False); station_df.to_csv(OUT_STATION,index=False)
    summary=detail_df.groupby("method",as_index=False)[["f1","event_coverage","row_recall","average_precision","false_alarm_rate"]].mean()
    payload={"status":"LOCKED_V4_EVALUATION_PENDING_AUDIT","protocol_id":frozen["protocol_id"],"selection_counts":selection_counts,"summary":summary.to_dict("records"),"limitations":["V4 test 运行前未查看；仍需独立审计判定。","所有攻击均为模拟注入，不是真实生产攻击。"]}
    OUT_JSON.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); OUT_REPORT.write_text("\n".join(["# Stage340 V4 路由锁定评价","",f"- 状态：`{payload['status']}`",f"- 协议：`{frozen['protocol_id']}`",f"- 选择计数：`{selection_counts}`","",summary.to_markdown(index=False),"","- 需要 Stage341 独立审计后才能决定是否通过70%门槛。"]),encoding="utf-8")
    print(json.dumps({"status":payload["status"],"selection_counts":selection_counts,"summary":payload["summary"]},ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
