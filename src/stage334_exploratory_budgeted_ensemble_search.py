from __future__ import annotations

"""Stage 334: exploratory detector-budget search on the already viewed V2 test.

This stage is deliberately post-hoc and cannot update the V2 confirmatory claim.
It searches a small pre-declared family of FAR-budgeted unions using only valid
labels for selection, then uses the viewed V2 test only for diagnostics.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL_PATH = ROOT / "confirmatory_package" / "simulated_attack_v2" / "protocol_freeze.json"
OUT_JSON = TABLES / "stage334_exploratory_budgeted_ensemble_search.json"
OUT_DETAIL = TABLES / "stage334_exploratory_budgeted_ensemble_search_detail.csv"
OUT_SCENARIO = TABLES / "stage334_exploratory_budgeted_ensemble_search_scenario.csv"
OUT_REPORT = REPORTS / "stage334_exploratory_budgeted_ensemble_search.md"

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
VALID_OFFSET = 930001
TEST_OFFSET = 940001
PRIMARY_FAR = 0.01
SEEDS = (3407, 6257, 7919, 104729, 20260715)
F1_FLOOR = 0.95
VIEWED_TEST = True

# Fixed before reading this diagnostic; candidate names, members and budgets are not
# expanded after inspecting V2 test outputs.
UNION_CANDIDATES = {
    "trust_only": (("trust_valid_f1", 0.0100),),
    "trust_ocsvm": (("trust_valid_f1", 0.0075), ("one_class_svm", 0.0025)),
    "trust_pca": (("trust_valid_f1", 0.0075), ("pca_reconstruction", 0.0025)),
    "trust_knn": (("trust_valid_f1", 0.0075), ("knn_distance", 0.0025)),
    "trust_ewma": (("trust_valid_f1", 0.0075), ("ewma_deviation", 0.0025)),
    "trust_arima": (("trust_valid_f1", 0.0075), ("arima_innovation", 0.0025)),
    "trust_ocsvm_pca": (("trust_valid_f1", 0.0060), ("one_class_svm", 0.0020), ("pca_reconstruction", 0.0020)),
    "trust_ocsvm_pca_knn": (("trust_valid_f1", 0.0040), ("one_class_svm", 0.0020), ("pca_reconstruction", 0.0020), ("knn_distance", 0.0020)),
    "trust_shape_union": (("trust_valid_f1", 0.0050), ("ewma_deviation", 0.0025), ("arima_innovation", 0.0025)),
    "trust_heterogeneous_union": (("trust_valid_f1", 0.0040), ("one_class_svm", 0.0015), ("pca_reconstruction", 0.0015), ("knn_distance", 0.0015), ("ewma_deviation", 0.0015)),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def budgeted_union(scores: dict[str, np.ndarray], members: tuple[tuple[str, float], ...], normal: np.ndarray, observed: np.ndarray, calibration_scores: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, dict[str, float]]:
    pred = np.zeros(len(observed), dtype=bool)
    thresholds: dict[str, float] = {}
    calibration_scores = scores if calibration_scores is None else calibration_scores
    for method, budget in members:
        threshold = far_threshold(calibration_scores[method], normal, float(budget))
        thresholds[method] = float(threshold)
        pred |= scores[method] >= threshold
    return pred & observed, thresholds


def scenario_rows(frame: pd.DataFrame, pred: np.ndarray, source: str, seed: int, candidate: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    attack = frame["is_attack"].astype(bool).to_numpy()
    labels = frame["attack_scenario"].astype(str).to_numpy()
    for scenario in sorted(set(labels[attack])):
        mask = attack & (labels == scenario)
        ev = frame.loc[mask, ["attack_event_id"]].copy()
        ev["predicted"] = pred[mask]
        grouped = ev.groupby("attack_event_id", sort=True)["predicted"].any()
        rows.append({
            "source": source,
            "seed": int(seed),
            "candidate": candidate,
            "scenario": scenario,
            "event_coverage": float(grouped.mean()) if len(grouped) else 0.0,
            "row_recall": float(pred[mask].mean()) if mask.any() else 0.0,
            "events": int(len(grouped)),
        })
    return rows


def main() -> int:
    frozen = load_json(PROTOCOL_PATH)
    scenario_by_name = {str(row["name"]): row for row in frozen["attack_scenarios"]}
    detail: list[dict[str, object]] = []
    scenario_detail: list[dict[str, object]] = []
    selection_counts: dict[str, int] = {}
    for source, protocol_path in PROTOCOLS.items():
        protocol = load_json(protocol_path)
        frame = detector_frame(make_frame(source, protocol))
        detectors = fit_detectors(frame, protocol)
        train_normal = (frame["split"].eq("train") & frame["observation_present"]).to_numpy(dtype=bool)
        base_components = score_components(frame, detectors)
        base_fused, _ = fuse_scores(frame, base_components)
        base_z = robust_standardize(base_fused, train_normal).to_numpy(dtype=float)
        base_features = candidate_features(frame, train_normal)
        base_matrix, _ = feature_matrix(base_components, base_features, train_normal)
        fitted_old = fit_baselines(frame, base_matrix, train_normal)
        fitted_light, _ = fit_lightweight_baselines(frame, base_matrix, train_normal)
        fitted_security, _ = fit_security_baselines(frame, base_matrix, train_normal)
        for seed in SEEDS:
            valid_blocks = simbase.make_simulated_blocks(frame, frozen, seed, "valid", VALID_OFFSET)
            test_blocks = simbase.make_simulated_blocks(frame, frozen, seed, "test", TEST_OFFSET)
            valid = simbase.inject_simulated_attack(frame, valid_blocks, scenario_by_name)
            test = simbase.inject_simulated_attack(frame, test_blocks, scenario_by_name)
            valid_components = score_components(valid, detectors)
            test_components = score_components(test, detectors)
            valid_fused, _ = fuse_scores(valid, valid_components)
            test_fused, _ = fuse_scores(test, test_components)
            valid_base = robust_standardize(valid_fused, train_normal).to_numpy(dtype=float)
            test_base = robust_standardize(test_fused, train_normal).to_numpy(dtype=float)
            valid_features = candidate_features(valid, train_normal)
            test_features = candidate_features(test, train_normal)
            valid_mask = valid["split"].eq("valid").to_numpy(dtype=bool)
            test_mask = test["split"].eq("test").to_numpy(dtype=bool)
            valid_frame = valid.loc[valid_mask].reset_index(drop=True)
            test_frame = test.loc[test_mask].reset_index(drop=True)
            valid_normal = (valid_frame["observation_present"] & ~valid_frame["is_attack"]).to_numpy(dtype=bool)
            valid_matrix, _ = feature_matrix(valid_components, valid_features, train_normal)
            test_matrix, _ = feature_matrix(test_components, test_features, train_normal)
            valid_scores = {"trust_valid_f1": valid_base[valid_mask]}
            test_scores = {"trust_valid_f1": test_base[test_mask]}
            selected = select_candidate_stage244("valid_f1", valid_base[valid_mask], {k: v[valid_mask] for k, v in valid_features.items()}, valid_frame, valid_normal, PRIMARY_FAR)
            key = (str(selected["feature"]), int(selected["window"]))
            valid_scores["trust_valid_f1"] = (1.0 - float(selected["weight"])) * valid_base[valid_mask] + float(selected["weight"]) * valid_features[key][valid_mask]
            test_scores["trust_valid_f1"] = (1.0 - float(selected["weight"])) * test_base[test_mask] + float(selected["weight"]) * test_features[key][test_mask]
            valid_scores.update({k: v for k, v in score_baselines(fitted_old, valid_matrix[valid_mask]).items()})
            test_scores.update({k: v for k, v in score_baselines(fitted_old, test_matrix[test_mask]).items()})
            valid_scores.update({k: v for k, v in score_lightweight_baselines(fitted_light, valid_matrix[valid_mask]).items()})
            test_scores.update({k: v for k, v in score_lightweight_baselines(fitted_light, test_matrix[test_mask]).items()})
            valid_scores.update({k: v for k, v in score_security_baselines(fitted_security, valid_frame, valid_matrix[valid_mask]).items()})
            test_scores.update({k: v for k, v in score_security_baselines(fitted_security, test_frame, test_matrix[test_mask]).items()})
            y_valid = valid_frame["is_attack"].astype(bool).to_numpy()
            observed_valid = valid_frame["observation_present"].to_numpy(dtype=bool)
            y_test = test_frame["is_attack"].astype(bool).to_numpy()
            observed_test = test_frame["observation_present"].to_numpy(dtype=bool)
            trust_thr = far_threshold(valid_scores["trust_valid_f1"], valid_normal, PRIMARY_FAR)
            trust_pred_valid = observed_valid & (valid_scores["trust_valid_f1"] >= trust_thr)
            trust_valid_metric = metrics(y_valid, trust_pred_valid, valid_scores["trust_valid_f1"])
            candidate_rows = []
            for candidate, members in UNION_CANDIDATES.items():
                pred_valid, thresholds = budgeted_union(valid_scores, members, valid_normal, observed_valid)
                met = metrics(y_valid, pred_valid, np.maximum.reduce([valid_scores[m] for m, _ in members]))
                ev = event_metrics(valid_frame, pred_valid)
                candidate_rows.append({"candidate": candidate, "valid_f1": float(met["f1"]), "valid_event_coverage": float(ev["event_coverage"]), "thresholds": thresholds})
            eligible = [row for row in candidate_rows if row["valid_f1"] >= F1_FLOOR * float(trust_valid_metric["f1"])]
            chosen = max(eligible, key=lambda row: (row["valid_event_coverage"], row["valid_f1"], -len(UNION_CANDIDATES[row["candidate"]]))) if eligible else next(row for row in candidate_rows if row["candidate"] == "trust_only")
            selection_counts[chosen["candidate"]] = selection_counts.get(chosen["candidate"], 0) + 1
            for candidate_name in UNION_CANDIDATES:
                members = UNION_CANDIDATES[candidate_name]
                pred_test, thresholds = budgeted_union(test_scores, members, valid_normal, observed_test, calibration_scores=valid_scores)
                score_test = np.maximum.reduce([test_scores[m] for m, _ in members])
                met = metrics(y_test, pred_test, score_test)
                ev = event_metrics(test_frame, pred_test)
                detail.append({"source": source, "seed": int(seed), "candidate": candidate_name, "selected": bool(candidate_name == chosen["candidate"]), "valid_f1": float(next(x["valid_f1"] for x in candidate_rows if x["candidate"] == candidate_name)), "valid_event_coverage": float(next(x["valid_event_coverage"] for x in candidate_rows if x["candidate"] == candidate_name)), **met, **ev})
                scenario_detail.extend(scenario_rows(test_frame, pred_test, source, int(seed), candidate_name))
            for method in sorted(set(valid_scores) - {"trust_valid_f1"}):
                threshold = far_threshold(valid_scores[method], valid_normal, PRIMARY_FAR)
                pred_test = observed_test & (test_scores[method] >= threshold)
                met = metrics(y_test, pred_test, test_scores[method])
                ev = event_metrics(test_frame, pred_test)
                detail.append({"source": source, "seed": int(seed), "candidate": method, "selected": False, "valid_f1": float("nan"), "valid_event_coverage": float("nan"), **met, **ev})
                scenario_detail.extend(scenario_rows(test_frame, pred_test, source, int(seed), method))
    detail_df = pd.DataFrame(detail)
    scenario_df = pd.DataFrame(scenario_detail)
    detail_df.to_csv(OUT_DETAIL, index=False)
    scenario_df.to_csv(OUT_SCENARIO, index=False)
    main_baselines = ["robust_max_deviation", "isolation_forest", "one_class_svm", "pca_reconstruction", "knn_distance", "ewma_deviation", "arima_innovation", "gru_predictor", "mlp_autoencoder_reconstruction", "lightweight_autoencoder", "ics_cps_spc_proxy"]
    base_scenario = pd.read_csv(TABLES / "stage326_simulated_attack_v2_scenario.csv")
    base_scenario = base_scenario[base_scenario["method"].isin(main_baselines + ["trust_valid_f1"])].copy()
    def win_share(candidate_name: str) -> float:
        cand = scenario_df[scenario_df["candidate"].eq(candidate_name)][["source","seed","scenario","event_coverage"]]
        merged = cand.merge(base_scenario[base_scenario["method"].isin(main_baselines)], on=["source","seed","scenario"], suffixes=("_cand","_base"))
        return float((merged["event_coverage_cand"] > merged["event_coverage_base"]).mean()) if len(merged) else float("nan")
    summary = detail_df.groupby("candidate", as_index=False)[["f1","event_coverage","row_recall","average_precision","false_alarm_rate"]].mean()
    summary["scenario_baseline_win_share"] = summary["candidate"].map(win_share)
    status = "EXPLORATORY_POST_HOC_NOT_CONFIRMATORY" if VIEWED_TEST else "LOCKED_V3_EVALUATION_PENDING_AUDIT"
    limitations = (["V2 test 已被查看；所有结果仅为探索性诊断。", "任何超过70%的场景层胜出比例都必须在新的未接触锁定窗口复核后才能升级。"] if VIEWED_TEST else ["本次使用新的攻击参数与独立随机流；test 标签仅用于最终评价。", "是否通过主终点、FAR、CI 和场景门槛需由独立审计脚本判定。"]) + ["所有攻击仍是预注册模拟攻击，不是真实生产攻击日志。"]
    payload = {"status": status, "protocol_id": frozen["protocol_id"], "candidate_family": {k: list(v) for k, v in UNION_CANDIDATES.items()}, "selection_counts": selection_counts, "summary": summary.to_dict("records"), "limitations": limitations}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# Stage334 预算式多检测器融合评价", "", f"- 状态：`{payload['status']}`", f"- 选择计数：`{selection_counts}`", "", summary.to_markdown(index=False), ""]
    lines.extend(["- V2 test 已查看，本结果不能升级为确认性主结果。", "- 若为 V3 新锁定流，则须由独立审计脚本判断是否通过。"] if VIEWED_TEST else ["- 本次为新的锁定攻击流，test 仅评价。", "- 仍需独立审计和负结果检查后才能升级主张。"])
    lines.append("")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selection_counts": selection_counts, "summary": payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
