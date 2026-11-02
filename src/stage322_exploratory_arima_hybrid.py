"""Stage 322: post-hoc exploratory Trust + ARIMA hybrid.

This is intentionally not a confirmatory result.  Stage320 test outputs have
already been inspected, so every result from this script is exploratory.  A
fixed weight grid is selected on valid only and evaluated on the same frozen
test package solely to diagnose whether a simple hybrid is worth a future
independent lock.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL_PATH = ROOT / "confirmatory_package" / "simulated_attack_v1" / "protocol_freeze.json"
OUT_JSON = TABLES / "stage322_exploratory_arima_hybrid.json"
OUT_DETAIL = TABLES / "stage322_exploratory_arima_hybrid_detail.csv"
OUT_REPORT = REPORTS / "stage322_exploratory_arima_hybrid.md"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import detector_frame, fit_detectors, metrics, score_components
from stage208_dwd_persistent_repeat_protection import event_metrics, far_threshold
from stage217_swiss_controlled_attack_evaluation import fuse_scores, robust_standardize
from stage240_nonstuck_balanced_selection import PROTOCOLS, make_frame
from stage244_cross_provider_nonstuck_candidate_features import candidate_features, select_candidate_stage244
from stage262_security_domain_baselines import _apply_reference, _ar_scores, fit_security_baselines
from stage320_simulated_attack_confirmatory_evaluation import (
    TEST_OFFSET,
    VALID_OFFSET,
    inject_simulated_attack,
    make_simulated_blocks,
)

WEIGHTS = tuple(float(x) for x in np.linspace(0.0, 1.0, 11))
SEEDS = (3407, 6257, 7919, 104729, 20260715)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_on_valid(values: np.ndarray, valid_normal: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ref = values[np.asarray(valid_normal, dtype=bool)]
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    median = float(np.median(ref))
    mad = max(1.4826 * float(np.median(np.abs(ref - median))), 1e-6)
    return np.nan_to_num((values - median) / mad, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> int:
    started = time.perf_counter()
    frozen = load_json(PROTOCOL_PATH)
    scenario_by_name = {str(row["name"]): row for row in frozen["attack_scenarios"]}
    rows: list[dict[str, object]] = []
    for source, protocol_path in PROTOCOLS.items():
        protocol = load_json(protocol_path)
        frame = detector_frame(make_frame(source, protocol))
        detectors = fit_detectors(frame, protocol)
        train_normal = (frame["split"].eq("train") & frame["observation_present"]).to_numpy(dtype=bool)
        components = score_components(frame, detectors)
        fused, _ = fuse_scores(frame, components)
        base_z = robust_standardize(fused, train_normal).to_numpy(dtype=float)
        features = candidate_features(frame, train_normal)
        # ARIMA is fitted only on train-normal rows, exactly as Stage262.
        security_fit, fit_info = fit_security_baselines(frame, np.column_stack([base_z]), train_normal)
        for seed in SEEDS:
            valid_blocks = make_simulated_blocks(frame, frozen, seed, "valid", VALID_OFFSET)
            test_blocks = make_simulated_blocks(frame, frozen, seed, "test", TEST_OFFSET)
            valid = inject_simulated_attack(frame, valid_blocks, scenario_by_name)
            test = inject_simulated_attack(frame, test_blocks, scenario_by_name)
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
            selected = select_candidate_stage244("valid_f1", valid_base[valid_mask], {key: value[valid_mask] for key, value in valid_features.items()}, valid_frame, valid_normal, 0.01)
            key = (str(selected["feature"]), int(selected["window"]))
            trust_valid = (1.0 - float(selected["weight"])) * valid_base[valid_mask] + float(selected["weight"]) * valid_features[key][valid_mask]
            trust_test = (1.0 - float(selected["weight"])) * test_base[test_mask] + float(selected["weight"]) * test_features[key][test_mask]
            valid_ar_raw = _ar_scores(valid_frame, security_fit["ar_coefficients"])
            test_ar_raw = _ar_scores(test_frame, security_fit["ar_coefficients"])
            ar_center, ar_scale = security_fit["references"]["arima_innovation"]
            valid_security = _apply_reference(valid_ar_raw, float(ar_center), float(ar_scale))
            test_security = _apply_reference(test_ar_raw, float(ar_center), float(ar_scale))
            trust_valid_n = normalize_on_valid(trust_valid, valid_normal)
            trust_test_n = normalize_on_valid(trust_test, valid_normal[:0] if False else np.zeros(len(test_frame), dtype=bool))
            # Test scaling uses the valid-normal reference from the valid score.
            trust_ref = trust_valid[np.asarray(valid_normal, dtype=bool)]
            trust_med = float(np.median(trust_ref)) if trust_ref.size else 0.0
            trust_scale = max(1.4826 * float(np.median(np.abs(trust_ref - trust_med))), 1e-6) if trust_ref.size else 1.0
            trust_test_n = np.nan_to_num((trust_test - trust_med) / trust_scale, nan=0.0, posinf=0.0, neginf=0.0)
            arima_valid_n = normalize_on_valid(valid_security, valid_normal)
            arima_ref = valid_security[np.asarray(valid_normal, dtype=bool)]
            arima_med = float(np.median(arima_ref)) if arima_ref.size else 0.0
            arima_scale = max(1.4826 * float(np.median(np.abs(arima_ref - arima_med))), 1e-6) if arima_ref.size else 1.0
            arima_test_n = np.nan_to_num((test_security - arima_med) / arima_scale, nan=0.0, posinf=0.0, neginf=0.0)
            y_valid = valid_frame["is_attack"].astype(bool).to_numpy()
            y_test = test_frame["is_attack"].astype(bool).to_numpy()
            best = None
            for trust_weight in WEIGHTS:
                score_valid = trust_weight * trust_valid_n + (1.0 - trust_weight) * arima_valid_n
                threshold = far_threshold(score_valid, valid_normal, 0.01)
                predicted = valid_frame["observation_present"].to_numpy(dtype=bool) & (score_valid >= threshold)
                row_metric = metrics(y_valid, predicted, score_valid)
                event_metric = event_metrics(valid_frame, predicted)
                key_order = (float(event_metric["event_coverage"]), float(row_metric["f1"]), float(trust_weight))
                if best is None or key_order > best["key_order"]:
                    best = {"key_order": key_order, "trust_weight": trust_weight, "threshold": threshold, "valid_f1": row_metric["f1"], "valid_event_coverage": event_metric["event_coverage"]}
            assert best is not None
            score_test = float(best["trust_weight"]) * trust_test_n + (1.0 - float(best["trust_weight"])) * arima_test_n
            predicted_test = test_frame["observation_present"].to_numpy(dtype=bool) & (score_test >= float(best["threshold"]))
            test_metrics = metrics(y_test, predicted_test, score_test)
            test_events = event_metrics(test_frame, predicted_test)
            arima_threshold = far_threshold(arima_valid_n, valid_normal, 0.01)
            arima_pred = test_frame["observation_present"].to_numpy(dtype=bool) & (arima_test_n >= arima_threshold)
            arima_metrics = metrics(y_test, arima_pred, arima_test_n)
            arima_events = event_metrics(test_frame, arima_pred)
            rows.extend([
                {"source": source, "seed": seed, "method": "trust_arima_hybrid_exploratory", "trust_weight": float(best["trust_weight"]), "valid_f1": float(best["valid_f1"]), "valid_event_coverage": float(best["valid_event_coverage"]), **test_metrics, **test_events, "selection_feature": selected["feature"], "selection_window": int(selected["window"]), "selection_weight": float(selected["weight"])}
                ,{"source": source, "seed": seed, "method": "arima_innovation_reference", "trust_weight": 0.0, "valid_f1": np.nan, "valid_event_coverage": np.nan, **arima_metrics, **arima_events, "selection_feature": "", "selection_window": 0, "selection_weight": 0.0}
            ])
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DETAIL, index=False)
    summary = detail.groupby("method", as_index=False)[["f1", "precision", "recall", "event_coverage", "row_recall", "average_precision", "false_alarm_rate"]].mean()
    payload = {"status": "EXPLORATORY_POST_HOC_NOT_CONFIRMATORY", "protocol_id": frozen["protocol_id"], "weights": list(WEIGHTS), "provider_count": 4, "seed_count": len(SEEDS), "detail_rows": len(detail), "summary": summary.to_dict("records"), "limitations": ["Stage320 test结果已先被查看，本阶段不得升级为确认性结论。", "只用valid选择融合权重，test只评价。", "没有新增独立时间窗或provider。", "所有攻击仍是预注册模拟攻击。"]}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    OUT_REPORT.write_text("\n".join(["# Stage322 Trust+ARIMA 探索性融合", "", "- 状态：`EXPLORATORY_POST_HOC_NOT_CONFIRMATORY`", f"- provider：4；种子：{len(SEEDS)}；权重网格：{list(WEIGHTS)}", "- 只使用 valid 选择权重，test 仅评价；由于 Stage320 test 已被查看，本结果不能进入确认性主表。", "", summary.to_markdown(index=False), "", "- 该分析只用于判断简单融合是否值得在未来独立锁定窗口上重新预注册。", f"- 运行耗时：{time.perf_counter() - started:.2f}s"
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "elapsed_seconds": time.perf_counter() - started}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
