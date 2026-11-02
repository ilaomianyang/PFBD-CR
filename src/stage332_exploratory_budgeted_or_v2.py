from __future__ import annotations

"""Exploratory budgeted OR fusion for the V2 simulation package."""

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
OUT_JSON = TABLES / "stage332_exploratory_budgeted_or_v2.json"
OUT_DETAIL = TABLES / "stage332_exploratory_budgeted_or_v2_detail.csv"
OUT_REPORT = REPORTS / "stage332_exploratory_budgeted_or_v2.md"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import detector_frame, fit_detectors, metrics, score_components
from stage208_dwd_persistent_repeat_protection import event_metrics, far_threshold
from stage217_swiss_controlled_attack_evaluation import fuse_scores, robust_standardize
from stage240_nonstuck_balanced_selection import PROTOCOLS, make_frame
from stage244_cross_provider_nonstuck_candidate_features import candidate_features, select_candidate_stage244
from stage262_security_domain_baselines import _apply_reference, _ar_scores, fit_security_baselines
import stage320_simulated_attack_confirmatory_evaluation as simbase
from stage326_simulated_attack_v2_evaluation import _internal_type

simbase._scenario_base_type = _internal_type

VALID_OFFSET = 930001
TEST_OFFSET = 940001
PRIMARY_FAR = 0.01
SEEDS = (3407, 6257, 7919, 104729, 20260715)
F1_FLOOR_RATIO = 0.95
BUDGET_PAIRS = ((0.0075, 0.0025), (0.005, 0.005), (0.0025, 0.0075))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    median = float(np.median(ref))
    scale = max(1.4826 * float(np.median(np.abs(ref - median))), 1e-6)
    return np.nan_to_num((values - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> int:
    frozen = load_json(PROTOCOL_PATH)
    scenario_by_name = {str(row["name"]): row for row in frozen["attack_scenarios"]}
    rows = []
    for source, protocol_path in PROTOCOLS.items():
        protocol = load_json(protocol_path)
        frame = detector_frame(make_frame(source, protocol))
        detectors = fit_detectors(frame, protocol)
        train_normal = (frame["split"].eq("train") & frame["observation_present"]).to_numpy(dtype=bool)
        components = score_components(frame, detectors)
        fused, _ = fuse_scores(frame, components)
        base_z = robust_standardize(fused, train_normal).to_numpy(dtype=float)
        features = candidate_features(frame, train_normal)
        security_fit, _ = fit_security_baselines(frame, np.column_stack([base_z]), train_normal)
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
            selected = select_candidate_stage244("valid_f1", valid_base[valid_mask], {k: v[valid_mask] for k, v in valid_features.items()}, valid_frame, valid_normal, PRIMARY_FAR)
            key = (str(selected["feature"]), int(selected["window"]))
            trust_valid = (1.0 - float(selected["weight"])) * valid_base[valid_mask] + float(selected["weight"]) * valid_features[key][valid_mask]
            trust_test = (1.0 - float(selected["weight"])) * test_base[test_mask] + float(selected["weight"]) * test_features[key][test_mask]
            valid_ar_raw = _ar_scores(valid_frame, security_fit["ar_coefficients"])
            test_ar_raw = _ar_scores(test_frame, security_fit["ar_coefficients"])
            ar_center, ar_scale = security_fit["references"]["arima_innovation"]
            valid_ar = _apply_reference(valid_ar_raw, float(ar_center), float(ar_scale))
            test_ar = _apply_reference(test_ar_raw, float(ar_center), float(ar_scale))
            trust_ref = trust_valid[valid_normal]
            ar_ref = valid_ar[valid_normal]
            trust_valid_n = normalize(trust_valid, trust_ref)
            trust_test_n = normalize(trust_test, trust_ref)
            ar_valid_n = normalize(valid_ar, ar_ref)
            ar_test_n = normalize(test_ar, ar_ref)
            y_valid = valid_frame["is_attack"].astype(bool).to_numpy()
            y_test = test_frame["is_attack"].astype(bool).to_numpy()
            trust_valid_threshold = far_threshold(trust_valid_n, valid_normal, PRIMARY_FAR)
            trust_valid_pred = valid_frame["observation_present"].to_numpy(dtype=bool) & (trust_valid_n >= trust_valid_threshold)
            trust_valid_metric = metrics(y_valid, trust_valid_pred, trust_valid_n)
            candidates = []
            for trust_budget, ar_budget in BUDGET_PAIRS:
                trust_thr = far_threshold(trust_valid_n, valid_normal, trust_budget)
                ar_thr = far_threshold(ar_valid_n, valid_normal, ar_budget)
                pred_valid = valid_frame["observation_present"].to_numpy(dtype=bool) & ((trust_valid_n >= trust_thr) | (ar_valid_n >= ar_thr))
                met = metrics(y_valid, pred_valid, np.maximum(trust_valid_n, ar_valid_n))
                ev = event_metrics(valid_frame, pred_valid)
                candidates.append({"trust_budget": trust_budget, "arima_budget": ar_budget, "trust_threshold": float(trust_thr), "arima_threshold": float(ar_thr), "valid_f1": float(met["f1"]), "valid_event_coverage": float(ev["event_coverage"])})
            eligible = [row for row in candidates if row["valid_f1"] >= F1_FLOOR_RATIO * float(trust_valid_metric["f1"])]
            chosen = max(eligible, key=lambda row: (row["valid_event_coverage"], row["valid_f1"])) if eligible else candidates[0]
            fusion_pred_test = test_frame["observation_present"].to_numpy(dtype=bool) & ((trust_test_n >= chosen["trust_threshold"]) | (ar_test_n >= chosen["arima_threshold"]))
            fusion_score_test = np.maximum(trust_test_n, ar_test_n)
            fusion_met = metrics(y_test, fusion_pred_test, fusion_score_test)
            fusion_ev = event_metrics(test_frame, fusion_pred_test)
            rows.append({"source": source, "seed": seed, "method": "budgeted_or_exploratory", "trust_budget": chosen["trust_budget"], "arima_budget": chosen["arima_budget"], "valid_f1": chosen["valid_f1"], "valid_event_coverage": chosen["valid_event_coverage"], **fusion_met, **fusion_ev})
            trust_test_threshold = far_threshold(trust_valid_n, valid_normal, PRIMARY_FAR)
            trust_pred_test = test_frame["observation_present"].to_numpy(dtype=bool) & (trust_test_n >= trust_test_threshold)
            trust_met = metrics(y_test, trust_pred_test, trust_test_n)
            trust_ev = event_metrics(test_frame, trust_pred_test)
            rows.append({"source": source, "seed": seed, "method": "trust_reference", "trust_budget": PRIMARY_FAR, "arima_budget": 0.0, "valid_f1": float(trust_valid_metric["f1"]), "valid_event_coverage": float(event_metrics(valid_frame, trust_valid_pred)["event_coverage"]), **trust_met, **trust_ev})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DETAIL, index=False)
    summary = detail.groupby("method", as_index=False)[["f1", "event_coverage", "row_recall", "average_precision", "false_alarm_rate"]].mean()
    selection = detail[detail["method"].eq("budgeted_or_exploratory")].groupby(["trust_budget", "arima_budget"]).size().reset_index(name="count").to_dict("records")
    payload = {"status": "EXPLORATORY_POST_HOC_NOT_CONFIRMATORY", "protocol_id": frozen["protocol_id"], "budget_pairs": [list(pair) for pair in BUDGET_PAIRS], "f1_floor_ratio": F1_FLOOR_RATIO, "detail_rows": len(detail), "selection_counts": selection, "summary": summary.to_dict("records"), "limitations": ["V2 test 已被查看；本轮只能作为探索性优化。", "预算门控只用 valid 标签选择，test 仅评价。", "所有攻击仍是预注册模拟攻击，不是真实生产攻击日志。"]}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    OUT_REPORT.write_text("\n".join(["# Stage332 预算式 OR 融合探索", "", f"- 状态：{payload['status']}", f"- FAR 预算对：{payload['budget_pairs']}", f"- 选择计数：{selection}", "", summary.to_markdown(index=False), "", "- V2 test 已查看，本结果不能升级为确认性主结果。", ""]), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selection_counts": selection, "summary": payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
