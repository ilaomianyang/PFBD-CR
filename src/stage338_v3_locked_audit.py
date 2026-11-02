from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SCENARIO = TABLES / "stage337_v3_locked_evaluation_scenario.csv"
DETAIL = TABLES / "stage337_v3_locked_evaluation_detail.csv"
PROTOCOL = ROOT / "confirmatory_package" / "simulated_attack_v3" / "protocol_freeze.json"
OUT_JSON = TABLES / "stage338_v3_locked_audit.json"
OUT_REPORT = REPORTS / "stage338_v3_locked_audit.md"
BASELINES = ["robust_max_deviation", "isolation_forest", "one_class_svm", "pca_reconstruction", "mlp_autoencoder_reconstruction", "knn_distance", "ewma_deviation", "arima_innovation", "gru_predictor", "lightweight_autoencoder", "ics_cps_spc_proxy"]
CANDIDATES = ["trust_only", "trust_arima", "trust_shape_union", "trust_ocsvm", "trust_pca", "trust_knn", "trust_ewma", "trust_heterogeneous_union", "trust_ocsvm_pca", "trust_ocsvm_pca_knn"]

def bootstrap_ci(values: np.ndarray, seed: int = 20260923, iterations: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    scenario = pd.read_csv(SCENARIO)
    detail = pd.read_csv(DETAIL)
    scenario_mean = scenario.groupby(["scenario", "candidate"], as_index=False)["event_coverage"].mean()
    base = scenario_mean[scenario_mean["candidate"].isin(BASELINES)].rename(columns={"event_coverage": "baseline_event_coverage", "candidate": "baseline"})
    cand = scenario_mean[scenario_mean["candidate"].isin(CANDIDATES)].rename(columns={"event_coverage": "candidate_event_coverage", "candidate": "candidate"})
    merged = cand.merge(base, on="scenario", how="inner")
    merged["win"] = merged["candidate_event_coverage"] > merged["baseline_event_coverage"]
    summary = []
    for name in CANDIDATES:
        part = merged[merged["candidate"].eq(name)]
        summary.append({"candidate": name, "scenario_baseline_win_share": float(part["win"].mean()) if len(part) else float("nan"), "beaten_pairs": int(part["win"].sum()), "total_pairs": int(len(part))})
    summary_df = pd.DataFrame(summary).sort_values("scenario_baseline_win_share", ascending=False)
    pair_rows = []
    for name in CANDIDATES:
        left = detail[detail["candidate"].eq(name)][["source", "seed", "event_coverage", "f1", "false_alarm_rate"]]
        for baseline in BASELINES:
            right = detail[detail["candidate"].eq(baseline)][["source", "seed", "event_coverage", "f1", "false_alarm_rate"]]
            aligned = left.merge(right, on=["source", "seed"], suffixes=("_cand", "_base"), validate="one_to_one")
            diff = (aligned["event_coverage_cand"] - aligned["event_coverage_base"]).to_numpy(dtype=float)
            lo, hi = bootstrap_ci(diff, seed=20260923 + len(pair_rows))
            pair_rows.append({"candidate": name, "baseline": baseline, "event_coverage_difference": float(diff.mean()), "ci95_low_provider_seed": lo, "ci95_high_provider_seed": hi, "positive_provider_seed_share": float((diff > 0).mean()), "candidate_f1": float(aligned["f1_cand"].mean()), "baseline_f1": float(aligned["f1_base"].mean()), "candidate_far": float(aligned["false_alarm_rate_cand"].mean()), "baseline_far": float(aligned["false_alarm_rate_base"].mean())})
    pairs = pd.DataFrame(pair_rows)
    best = summary_df.iloc[0].to_dict()
    selected = detail[detail["selected"].astype(bool)]
    selected_summary = {"f1": float(selected["f1"].mean()), "event_coverage": float(selected["event_coverage"].mean()), "false_alarm_rate": float(selected["false_alarm_rate"].mean()), "selected_runs": int(len(selected))}
    gate = {"scenario_win_share_ge_70": bool(float(best["scenario_baseline_win_share"]) >= 0.70), "selected_mean_far_le_0.01": bool(selected_summary["false_alarm_rate"] <= 0.01), "all_pair_provider_seed_ci_positive": bool((pairs[pairs["candidate"].eq(best["candidate"])]
        ["ci95_low_provider_seed"] > 0).all())}
    payload = {"status": "PASS_V3_LOCKED_EVALUATION_NEEDS_METHOD_LEVEL_AUDIT", "protocol_id": protocol["protocol_id"], "scenario_definition": "per-scenario mean event coverage then candidate > baseline, matching Stage328", "best_candidate_by_scenario_share": best, "selected_summary": selected_summary, "gate": gate, "candidate_summary": summary_df.to_dict("records"), "pair_summary": pairs.to_dict("records"), "limitations": ["V3 使用四个冻结 provider；不支持机构无关或真实攻击普适声称。", "provider×seed CI 不是旧 V2 的站点簇 bootstrap；需要在最终主表中补充站点级重采样。", "如果任何基线在正式主终点上胜出，全面领先主张自动关闭。"]}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Stage338 V3 独立锁定审计", "", f"- 状态：`{payload['status']}`", f"- 协议：`{protocol['protocol_id']}`", f"- 最佳候选（按场景层比例）：`{best['candidate']}`，比例 `{float(best['scenario_baseline_win_share']):.1%}`", f"- 选择运行平均 F1 / event coverage / FAR：`{selected_summary['f1']:.6f}` / `{selected_summary['event_coverage']:.6f}` / `{selected_summary['false_alarm_rate']:.6f}`", "", summary_df.to_markdown(index=False), "", "## Gate", "", json.dumps(gate, ensure_ascii=False, indent=2), "", "## 证据边界", "", "- 本次 test 攻击流未在 V3 运行前查看。", "- 所有攻击仍是模拟注入，不是真实生产攻击。", "- 需要站点级 bootstrap 和 BH 校正后，才能决定是否升级主张。", ""]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "best": best, "selected_summary": selected_summary, "gate": gate}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
