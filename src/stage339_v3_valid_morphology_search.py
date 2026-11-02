from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL_PATH = ROOT / "confirmatory_package" / "simulated_attack_v3" / "protocol_freeze.json"
OUT_JSON = TABLES / "stage339_v3_valid_morphology_search.json"
OUT_REPORT = REPORTS / "stage339_v3_valid_morphology_search.md"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
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
VALID_OFFSET = 950001
PRIMARY_FAR = 0.01
F1_FLOOR = 0.95
GATES = [
    {"name": "trust_only", "curvature_method": "trust_valid_f1", "drift_method": "trust_valid_f1", "noise_method": "trust_valid_f1", "q_curvature": 0.90, "q_drift": 0.90, "q_noise": 0.90},
    {"name": "route_knn_pca", "curvature_method": "knn_distance", "drift_method": "pca_reconstruction", "noise_method": "trust_valid_f1", "q_curvature": 0.90, "q_drift": 0.90, "q_noise": 0.90},
    {"name": "route_ocsvm_pca", "curvature_method": "one_class_svm", "drift_method": "pca_reconstruction", "noise_method": "trust_valid_f1", "q_curvature": 0.90, "q_drift": 0.90, "q_noise": 0.90},
    {"name": "route_knn_arima", "curvature_method": "knn_distance", "drift_method": "arima_innovation", "noise_method": "trust_valid_f1", "q_curvature": 0.90, "q_drift": 0.90, "q_noise": 0.90},
    {"name": "route_knn_pca_lowq", "curvature_method": "knn_distance", "drift_method": "pca_reconstruction", "noise_method": "trust_valid_f1", "q_curvature": 0.80, "q_drift": 0.80, "q_noise": 0.85},
    {"name": "route_ocsvm_pca_lowq", "curvature_method": "one_class_svm", "drift_method": "pca_reconstruction", "noise_method": "trust_valid_f1", "q_curvature": 0.80, "q_drift": 0.80, "q_noise": 0.85},
    {"name": "route_knn_pca_highq", "curvature_method": "knn_distance", "drift_method": "pca_reconstruction", "noise_method": "trust_valid_f1", "q_curvature": 0.95, "q_drift": 0.95, "q_noise": 0.95},
]

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def norm_pair(valid: np.ndarray, test: np.ndarray, valid_normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(valid, dtype=float)[valid_normal]
    ref = ref[np.isfinite(ref)]
    med = float(np.median(ref)) if len(ref) else 0.0
    scale = max(1.4826 * float(np.median(np.abs(ref - med))) if len(ref) else 1.0, 1e-6)
    return (np.nan_to_num((valid - med) / scale), np.nan_to_num((test - med) / scale))

def routed_score(scores: dict[str, np.ndarray], features: dict[tuple[str, int], np.ndarray], valid_normal: np.ndarray, gate: dict[str, object]) -> np.ndarray:
    curvature = features[("second_diff_zscore", 1)]
    drift = features[("slope_6", 1)]
    noise = features[("noise_ratio", 1)]
    qc = float(np.quantile(curvature[valid_normal], float(gate["q_curvature"])))
    qd = float(np.quantile(drift[valid_normal], float(gate["q_drift"])))
    qn = float(np.quantile(noise[valid_normal], float(gate["q_noise"])))
    out = scores["trust_valid_f1"].copy()
    out[noise >= qn] = scores[str(gate["noise_method"])][noise >= qn]
    out[(noise < qn) & (drift >= qd)] = scores[str(gate["drift_method"])][(noise < qn) & (drift >= qd)]
    out[(noise < qn) & (drift < qd) & (curvature >= qc)] = scores[str(gate["curvature_method"])][(noise < qn) & (drift < qd) & (curvature >= qc)]
    return out

def main() -> int:
    frozen = load_json(PROTOCOL_PATH)
    scenario_by_name = {str(row["name"]): row for row in frozen["attack_scenarios"]}
    rows = []
    selection_counts = {}
    for source, protocol_path in PROTOCOLS.items():
        protocol = load_json(protocol_path)
        frame = detector_frame(make_frame(source, protocol))
        detectors = fit_detectors(frame, protocol)
        train_normal = (frame["split"].eq("train") & frame["observation_present"]).to_numpy(dtype=bool)
        components = score_components(frame, detectors)
        fused, _ = fuse_scores(frame, components)
        base_z = robust_standardize(fused, train_normal).to_numpy(dtype=float)
        features = candidate_features(frame, train_normal)
        matrix, _ = feature_matrix(components, features, train_normal)
        old = fit_baselines(frame, matrix, train_normal)
        light, _ = fit_lightweight_baselines(frame, matrix, train_normal)
        security, _ = fit_security_baselines(frame, matrix, train_normal)
        for seed in SEEDS:
            blocks = simbase.make_simulated_blocks(frame, frozen, seed, "valid", VALID_OFFSET)
            attacked = simbase.inject_simulated_attack(frame, blocks, scenario_by_name)
            comp = score_components(attacked, detectors)
            fused_attacked, _ = fuse_scores(attacked, comp)
            z = robust_standardize(fused_attacked, train_normal).to_numpy(dtype=float)
            feats = candidate_features(attacked, train_normal)
            mask = attacked["split"].eq("valid").to_numpy(dtype=bool)
            vf = attacked.loc[mask].reset_index(drop=True)
            normal = (vf["observation_present"] & ~vf["is_attack"]).to_numpy(dtype=bool)
            vmatrix, _ = feature_matrix(comp, feats, train_normal)
            vs = {"trust_valid_f1": z[mask]}
            selected = select_candidate_stage244("valid_f1", z[mask], {k: v[mask] for k, v in feats.items()}, vf, normal, PRIMARY_FAR)
            key = (str(selected["feature"]), int(selected["window"]))
            vs["trust_valid_f1"] = (1.0 - float(selected["weight"])) * z[mask] + float(selected["weight"]) * feats[key][mask]
            vs.update(score_baselines(old, vmatrix[mask]))
            vs.update(score_lightweight_baselines(light, vmatrix[mask]))
            vs.update(score_security_baselines(security, vf, vmatrix[mask]))
            normalized = {}
            for method, values in vs.items():
                normalized[method], _ = norm_pair(np.asarray(values), np.asarray(values), normal)
            y = vf["is_attack"].astype(bool).to_numpy()
            observed = vf["observation_present"].to_numpy(dtype=bool)
            trust_threshold = far_threshold(normalized["trust_valid_f1"], normal, PRIMARY_FAR)
            trust_pred = observed & (normalized["trust_valid_f1"] >= trust_threshold)
            trust_metric = metrics(y, trust_pred, normalized["trust_valid_f1"])
            best = None
            for gate in GATES:
                score = routed_score(normalized, {k: v[mask] for k, v in feats.items()}, normal, gate)
                threshold = far_threshold(score, normal, PRIMARY_FAR)
                pred = observed & (score >= threshold)
                met = metrics(y, pred, score)
                ev = event_metrics(vf, pred)
                row = {"gate": gate["name"], "valid_f1": float(met["f1"]), "valid_event_coverage": float(ev["event_coverage"]), "valid_far": float(met["false_alarm_rate"]), "source": source, "seed": int(seed)}
                if met["f1"] >= F1_FLOOR * trust_metric["f1"] and (best is None or (row["valid_event_coverage"], row["valid_f1"]) > (best["valid_event_coverage"], best["valid_f1"])):
                    best = row
            if best is None:
                best = {"gate": "trust_only", "valid_f1": float(trust_metric["f1"]), "valid_event_coverage": float(event_metrics(vf, trust_pred)["event_coverage"]), "valid_far": float(trust_metric["false_alarm_rate"]), "source": source, "seed": int(seed)}
            rows.append(best)
            selection_counts[best["gate"]] = selection_counts.get(best["gate"], 0) + 1
    detail = pd.DataFrame(rows)
    summary = detail.groupby("gate", as_index=False)[["valid_f1", "valid_event_coverage", "valid_far"]].mean().sort_values("valid_event_coverage", ascending=False)
    payload = {"status": "VALID_ONLY_EXPLORATORY_SEARCH", "protocol_id": frozen["protocol_id"], "selection_counts": selection_counts, "summary": summary.to_dict("records"), "limitations": ["只使用 V3 valid 标签；未读取 V3 test。", "该结果不能直接作为确认性主结果，需另行冻结 V4 test。"]}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_REPORT.write_text("\n".join(["# Stage339 V3 valid 形态路由探索", "", f"- 状态：`{payload['status']}`", f"- 选择计数：`{selection_counts}`", "", summary.to_markdown(index=False), "", "- 未读取 V3 test；任何新候选必须在新的 V4 锁定流确认。", ""]), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
