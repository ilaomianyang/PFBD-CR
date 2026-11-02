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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from secure_recovery import add_recovery_candidates, choose_weights, combine_candidates  # noqa: E402
from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402
from stage44_recovery_confidence_gate import summarize  # noqa: E402

NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
GATES = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95]
MAX_VALID_CLEAN_HARM_RATE = 0.20
STRATEGIES = {
    "no_recovery": {"temporal": 0.0, "spatial": 0.0, "era5": 0.0},
    "legacy_selected": None,
    "temporal_only": {"temporal": 1.0, "spatial": 0.0, "era5": 0.0},
    "spatial_only": {"temporal": 0.0, "spatial": 1.0, "era5": 0.0},
}


def run_seed(source: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    frame = build_dataset(source, seed)
    scored, _ = score(frame)
    scored["observed_value"] = pd.to_numeric(scored["effective_observed_value"], errors="coerce")
    scored["is_anomaly"] = scored["combined_is_attack"].astype(bool)
    scored["era5_corrected_t2m_c"] = np.nan
    valid = scored["split"].eq("valid")
    test = scored["split"].eq("test")
    vf = scored.loc[valid].copy()
    tf = scored.loc[test].copy()
    valid_risk = pd.to_numeric(vf["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
    valid_alert = vf["predicted_attack"].astype(bool).to_numpy()
    alert_risk = valid_risk[valid_alert & np.isfinite(valid_risk)]
    if len(alert_risk) == 0:
        raise ValueError(f"no NOAA validation alerts for seed={seed}")
    test_risk = pd.to_numeric(tf["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
    test_alert = tf["predicted_attack"].astype(bool).to_numpy()
    rows: list[dict[str, object]] = []
    for gate_quantile in GATES:
        threshold = float(np.quantile(alert_risk, gate_quantile))
        gate_v = valid_alert & (valid_risk >= threshold)
        gate_t = test_alert & (test_risk >= threshold)
        vc = add_recovery_candidates(vf, pd.Series(gate_v, index=vf.index), vf["era5_corrected_t2m_c"])
        tc = add_recovery_candidates(tf, pd.Series(gate_t, index=tf.index), tf["era5_corrected_t2m_c"])
        legacy_weights = choose_weights(vc)
        weights_for_q = dict(STRATEGIES)
        weights_for_q["legacy_selected"] = legacy_weights
        for strategy, weights in weights_for_q.items():
            vr = combine_candidates(vc, weights)
            tr = combine_candidates(tc, weights)
            vm = summarize(vf, vr, gate_v)
            tm = summarize(tf, tr, gate_t)
            rows.append({
                "seed": int(seed),
                "gate_quantile": float(gate_quantile),
                "risk_threshold": threshold,
                "strategy": strategy,
                "weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                "valid_utility": vm["net_absolute_error_gain"],
                "test_utility": tm["net_absolute_error_gain"],
                "valid_attack_mae_after": vm["detected_attack_mae_after"],
                "test_attack_mae_after": tm["detected_attack_mae_after"],
                "valid_attack_improvement_rate": vm["detected_attack_improvement_rate"],
                "test_attack_improvement_rate": tm["detected_attack_improvement_rate"],
                "valid_attack_worsening_rate": vm["detected_attack_worsening_rate"],
                "test_attack_worsening_rate": tm["detected_attack_worsening_rate"],
                "valid_clean_harm_rate": vm["normal_harm_rate"],
                "test_clean_harm_rate": tm["normal_harm_rate"],
                "valid_gated_alert_rows": vm["gated_alert_rows"],
                "test_gated_alert_rows": tm["gated_alert_rows"],
                "valid_clean_changed_rows": vm["normal_changed_rows"],
                "test_clean_changed_rows": tm["normal_changed_rows"],
            })
    return rows


def main() -> int:
    if not NOAA.is_file():
        raise FileNotFoundError(NOAA)
    source = pd.read_parquet(NOAA)
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        rows.extend(run_seed(source, seed))
    detail = pd.DataFrame(rows)
    selected_unconstrained = []
    selected_constrained = []
    for seed in SEEDS:
        candidates = detail.loc[detail["seed"] == seed].to_dict(orient="records")
        best = max(candidates, key=lambda row: (float(row["valid_utility"]), -int(row["valid_gated_alert_rows"]), row["strategy"], -float(row["gate_quantile"])))
        feasible = [row for row in candidates if float(row["valid_clean_harm_rate"]) <= MAX_VALID_CLEAN_HARM_RATE]
        constrained = max(feasible or [row for row in candidates if row["strategy"] == "no_recovery"], key=lambda row: (float(row["valid_utility"]), -int(row["valid_gated_alert_rows"]), row["strategy"], -float(row["gate_quantile"])))
        best = dict(best)
        constrained = dict(constrained)
        best["selection_policy"] = "unconstrained_max_valid_utility"
        constrained["selection_policy"] = "valid_clean_harm_rate_le_0.20"
        selected_unconstrained.append(best)
        selected_constrained.append(constrained)
    selected = pd.DataFrame(selected_unconstrained + selected_constrained)

    def reference(strategy: str) -> pd.Series:
        return detail.loc[(detail["strategy"] == strategy) & (detail["gate_quantile"] == 0.0)].mean(numeric_only=True)

    unconstrained = selected.loc[selected["selection_policy"] == "unconstrained_max_valid_utility"]
    constrained = selected.loc[selected["selection_policy"] == "valid_clean_harm_rate_le_0.20"]
    legacy = reference("legacy_selected")
    temporal = reference("temporal_only")
    aggregate = pd.DataFrame([
        {"group": "unconstrained_selected_by_valid", "combinations": len(unconstrained), "test_utility_mean": float(unconstrained["test_utility"].mean()), "test_attack_mae_after_mean": float(unconstrained["test_attack_mae_after"].mean()), "test_attack_improvement_rate_mean": float(unconstrained["test_attack_improvement_rate"].mean()), "test_clean_harm_rate_mean": float(unconstrained["test_clean_harm_rate"].mean())},
        {"group": "safety_constrained_selected_by_valid", "combinations": len(constrained), "test_utility_mean": float(constrained["test_utility"].mean()), "test_attack_mae_after_mean": float(constrained["test_attack_mae_after"].mean()), "test_attack_improvement_rate_mean": float(constrained["test_attack_improvement_rate"].mean()), "test_clean_harm_rate_mean": float(constrained["test_clean_harm_rate"].mean())},
        {"group": "legacy_ungated_reference", "combinations": len(SEEDS), "test_utility_mean": float(legacy["test_utility"]), "test_attack_mae_after_mean": float(legacy["test_attack_mae_after"]), "test_attack_improvement_rate_mean": float(legacy["test_attack_improvement_rate"]), "test_clean_harm_rate_mean": float(legacy["test_clean_harm_rate"])},
        {"group": "temporal_ungated_reference", "combinations": len(SEEDS), "test_utility_mean": float(temporal["test_utility"]), "test_attack_mae_after_mean": float(temporal["test_attack_mae_after"]), "test_attack_improvement_rate_mean": float(temporal["test_attack_improvement_rate"]), "test_clean_harm_rate_mean": float(temporal["test_clean_harm_rate"])},
    ])
    safety_row = aggregate.loc[aggregate["group"] == "safety_constrained_selected_by_valid"].iloc[0]
    legacy_row = aggregate.loc[aggregate["group"] == "legacy_ungated_reference"].iloc[0]
    safety_uses_recovery = bool((constrained["strategy"] != "no_recovery").any())
    status = "CANDIDATE_GAIN_REQUIRES_ADDITIONAL_EXTERNAL_DATA" if safety_uses_recovery and safety_row["test_utility_mean"] > 0.0 and safety_row["test_utility_mean"] > legacy_row["test_utility_mean"] and safety_row["test_clean_harm_rate_mean"] <= legacy_row["test_clean_harm_rate_mean"] else "NEGATIVE_RESULT_RETAINED"
    selected_counts = selected.groupby(["selection_policy", "strategy"], as_index=False).size().rename(columns={"size": "selected_combinations"})
    payload = {
        "status": status,
        "protocol": "阶段130使用NOAA ISD官方归一化表和五个固定受控攻击种子；时间/空间恢复策略与风险门控分位数预先固定，仅用每个种子的valid utility选择；另以valid正常误报警改写恶化率≤20%执行安全约束；locked test只评价，不根据test重新选择恢复权重，不使用ERA5候选。",
        "source": str(NOAA),
        "seeds": SEEDS,
        "strategy_count": len(STRATEGIES),
        "gate_count": len(GATES),
        "detail_rows": len(detail),
        "selected_count": len(selected),
        "aggregate_count": len(aggregate),
        "max_valid_clean_harm_rate": MAX_VALID_CLEAN_HARM_RATE,
        "selected_strategy_counts": selected_counts.to_dict(orient="records"),
        "aggregate": aggregate.to_dict(orient="records"),
        "limitations": [
            "NOAA攻击为透明受控注入，不是真实攻击日志。",
            "NOAA质量标记和自然观测不能替代攻击真值。",
            "安全约束降低正常改写伤害时可能牺牲攻击恢复效果；结果不替代检测性能。",
            "仍需更多独立来源和真实运维故障日志复核。",
        ],
    }
    detail.to_csv(TABLES / "stage130_noaa_safety_constrained_recovery_detail.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(TABLES / "stage130_noaa_safety_constrained_recovery_selected.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage130_noaa_safety_constrained_recovery.csv", index=False, encoding="utf-8-sig")
    selected_counts.to_csv(TABLES / "stage130_noaa_safety_constrained_recovery_counts.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage130_noaa_safety_constrained_recovery.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段130 NOAA安全约束恢复复核", "", f"- 状态：**{status}**", f"- 五个种子；策略数=`{len(STRATEGIES)}`；门控数=`{len(GATES)}`；valid正常误报警改写恶化率约束=`≤{MAX_VALID_CLEAN_HARM_RATE:.2f}`", "", "| 组别 | 组合数 | test utility | test攻击MAE后 | test攻击改善率 | test正常改写恶化率 |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['group']} | {row['combinations']} | {row['test_utility_mean']:.4f} | {row['test_attack_mae_after_mean']:.4f} | {row['test_attack_improvement_rate_mean']:.4f} | {row['test_clean_harm_rate_mean']:.4f} |")
    lines += ["", "阶段130只用valid选择恢复策略，locked test只评价；NOAA攻击为受控注入，不能写成真实攻击恢复证据。"]
    (REPORTS / "stage130_noaa_safety_constrained_recovery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "detail_rows": len(detail), "selected_count": len(selected), "aggregate": aggregate.to_dict(orient="records")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
