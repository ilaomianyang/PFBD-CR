from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

SEEDS = [3407, 6257, 7919, 104729, 20260715]
GATES = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95]
STRATEGIES = {
    "no_recovery": {"temporal": 0.0, "spatial": 0.0, "era5": 0.0},
    "legacy_selected": None,
    "temporal_only": {"temporal": 1.0, "spatial": 0.0, "era5": 0.0},
    "spatial_only": {"temporal": 0.0, "spatial": 1.0, "era5": 0.0},
    "era5_only": {"temporal": 0.0, "spatial": 0.0, "era5": 1.0},
    "temporal_spatial_equal": {"temporal": 0.5, "spatial": 0.5, "era5": 0.0},
}
MAX_VALID_CLEAN_HARM_RATE = 0.20


def weighted_candidate(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    names = ["temporal_candidate", "spatial_candidate", "era5_candidate"]
    values = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    weight_array = np.asarray([weights.get("temporal", 0.0), weights.get("spatial", 0.0), weights.get("era5", 0.0)], dtype=float)
    finite = np.isfinite(values) & (weight_array[None, :] > 0)
    denominator = (finite * weight_array[None, :]).sum(axis=1)
    numerator = np.where(finite, values, 0.0) @ weight_array
    out = np.divide(numerator, denominator, out=np.full(len(frame), np.nan), where=denominator > 0)
    fallback = pd.to_numeric(frame["effective_observed_value"], errors="coerce").to_numpy(dtype=float)
    out[~np.isfinite(out)] = fallback[~np.isfinite(out)]
    return out


def metrics(frame: pd.DataFrame, candidate: np.ndarray, gate: np.ndarray) -> dict[str, float | int]:
    true = pd.to_numeric(frame["original_value"], errors="coerce").to_numpy(dtype=float)
    observed = pd.to_numeric(frame["effective_observed_value"], errors="coerce").to_numpy(dtype=float)
    attack = frame["combined_is_attack"].astype(bool).to_numpy()
    finite = np.isfinite(true) & np.isfinite(observed) & np.isfinite(candidate)
    recovered = observed.copy()
    recovered[gate & finite] = candidate[gate & finite]
    before = np.abs(observed - true)
    after = np.abs(recovered - true)
    attack_rows = finite & attack & gate
    clean_rows = finite & ~attack & gate
    changed = finite & (np.abs(recovered - observed) > 1e-9)
    attack_gain = before[attack_rows] - after[attack_rows]
    clean_harm = after[clean_rows] - before[clean_rows]
    return {
        "rows": int(len(frame)),
        "gated_alert_rows": int(gate.sum()),
        "detected_attack_rows": int(attack_rows.sum()),
        "detected_attack_mae_before": float(before[attack_rows].mean()) if attack_rows.any() else float("nan"),
        "detected_attack_mae_after": float(after[attack_rows].mean()) if attack_rows.any() else float("nan"),
        "detected_attack_improvement_rate": float((attack_gain > 1e-9).mean()) if attack_rows.any() else float("nan"),
        "detected_attack_worsening_rate": float((attack_gain < -1e-9).mean()) if attack_rows.any() else float("nan"),
        "clean_changed_rows": int((clean_rows & changed).sum()),
        "clean_harm_rate": float((clean_harm > 1e-9).mean()) if clean_rows.any() else 0.0,
        "attack_gain_sum": float(attack_gain.sum()) if attack_rows.any() else 0.0,
        "clean_harm_sum": float(np.maximum(clean_harm, 0.0).sum()) if clean_rows.any() else 0.0,
        "utility": float(attack_gain.sum() - np.maximum(clean_harm, 0.0).sum()) if attack_rows.any() or clean_rows.any() else 0.0,
    }


def main() -> int:
    test_path = TABLES / "stage124_leave_one_station_refit_multiseed_detail.parquet"
    valid_path = TABLES / "stage124_leave_one_station_refit_multiseed_valid_detail.parquet"
    source_path = TABLES / "stage124_leave_one_station_refit_multiseed.json"
    if not test_path.is_file() or not valid_path.is_file() or not source_path.is_file():
        raise FileNotFoundError("stage124 test/valid/source artifacts are required")
    test = pd.read_parquet(test_path)
    valid = pd.read_parquet(valid_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    legacy = {(int(row["seed"]), str(row["held_out_station"])): row.get("recovery_weights", {}) for row in source["results"]}

    details: list[dict[str, object]] = []
    selected_unconstrained: list[dict[str, object]] = []
    selected_constrained: list[dict[str, object]] = []
    for (seed, heldout), vf in valid.groupby(["seed", "held_out_station"], sort=True):
        tf = test[(test["seed"] == seed) & (test["held_out_station"] == heldout)].copy()
        vf = vf.copy()
        base_risk = pd.to_numeric(vf["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
        alert = vf["predicted_attack"].astype(bool).to_numpy()
        alert_risk = base_risk[alert & np.isfinite(base_risk)]
        if len(alert_risk) == 0:
            raise ValueError(f"no validation alerts for seed={seed}, held_out={heldout}")
        candidates_for_combo: list[dict[str, object]] = []
        for strategy, fixed_weights in STRATEGIES.items():
            weights = legacy[(int(seed), str(heldout))] if fixed_weights is None else fixed_weights
            valid_candidate = weighted_candidate(vf, weights)
            test_candidate = weighted_candidate(tf, weights)
            for gate_quantile in GATES:
                threshold = float(np.quantile(alert_risk, gate_quantile))
                valid_gate = alert & (base_risk >= threshold)
                test_risk = pd.to_numeric(tf["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
                test_gate = tf["predicted_attack"].astype(bool).to_numpy() & (test_risk >= threshold)
                vm = metrics(vf, valid_candidate, valid_gate)
                tm = metrics(tf, test_candidate, test_gate)
                row = {
                    "seed": int(seed),
                    "held_out_station": str(heldout),
                    "strategy": strategy,
                    "weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                    "gate_quantile": float(gate_quantile),
                    "risk_threshold": threshold,
                    "valid_utility": vm["utility"],
                    "test_utility": tm["utility"],
                    "valid_attack_gain_sum": vm["attack_gain_sum"],
                    "valid_clean_harm_sum": vm["clean_harm_sum"],
                    "test_attack_gain_sum": tm["attack_gain_sum"],
                    "test_clean_harm_sum": tm["clean_harm_sum"],
                    "valid_gated_alert_rows": vm["gated_alert_rows"],
                    "test_gated_alert_rows": tm["gated_alert_rows"],
                    "valid_attack_mae_before": vm["detected_attack_mae_before"],
                    "valid_attack_mae_after": vm["detected_attack_mae_after"],
                    "test_attack_mae_before": tm["detected_attack_mae_before"],
                    "test_attack_mae_after": tm["detected_attack_mae_after"],
                    "valid_attack_improvement_rate": vm["detected_attack_improvement_rate"],
                    "test_attack_improvement_rate": tm["detected_attack_improvement_rate"],
                    "valid_attack_worsening_rate": vm["detected_attack_worsening_rate"],
                    "test_attack_worsening_rate": tm["detected_attack_worsening_rate"],
                    "valid_clean_harm_rate": vm["clean_harm_rate"],
                    "test_clean_harm_rate": tm["clean_harm_rate"],
                    "valid_clean_changed_rows": vm["clean_changed_rows"],
                    "test_clean_changed_rows": tm["clean_changed_rows"],
                }
                details.append(row)
                candidates_for_combo.append(row)
        best = max(candidates_for_combo, key=lambda row: (float(row["valid_utility"]), -int(row["valid_gated_alert_rows"]), row["strategy"], -float(row["gate_quantile"])))
        feasible = [row for row in candidates_for_combo if float(row["valid_clean_harm_rate"]) <= MAX_VALID_CLEAN_HARM_RATE]
        constrained = max(feasible or [next(row for row in candidates_for_combo if row["strategy"] == "no_recovery")], key=lambda row: (float(row["valid_utility"]), -int(row["valid_gated_alert_rows"]), row["strategy"], -float(row["gate_quantile"])))
        best = dict(best)
        constrained = dict(constrained)
        best["selection_policy"] = "unconstrained_max_valid_utility"
        constrained["selection_policy"] = "valid_clean_harm_rate_le_0.20"
        selected_unconstrained.append(best)
        selected_constrained.append(constrained)

    detail = pd.DataFrame(details)
    selected_unconstrained_df = pd.DataFrame(selected_unconstrained)
    selected_constrained_df = pd.DataFrame(selected_constrained)
    selected_df = pd.concat([selected_unconstrained_df, selected_constrained_df], ignore_index=True)
    aggregate = pd.DataFrame([
        {"group": "unconstrained_selected_by_valid", "combinations": len(selected_unconstrained_df), "test_utility_mean": float(selected_unconstrained_df["test_utility"].mean()), "test_attack_mae_after_mean": float(selected_unconstrained_df["test_attack_mae_after"].mean()), "test_attack_improvement_rate_mean": float(selected_unconstrained_df["test_attack_improvement_rate"].mean()), "test_clean_harm_rate_mean": float(selected_unconstrained_df["test_clean_harm_rate"].mean())},
        {"group": "safety_constrained_selected_by_valid", "combinations": len(selected_constrained_df), "test_utility_mean": float(selected_constrained_df["test_utility"].mean()), "test_attack_mae_after_mean": float(selected_constrained_df["test_attack_mae_after"].mean()), "test_attack_improvement_rate_mean": float(selected_constrained_df["test_attack_improvement_rate"].mean()), "test_clean_harm_rate_mean": float(selected_constrained_df["test_clean_harm_rate"].mean())},
        {"group": "legacy_ungated_reference", "combinations": int((detail["strategy"] == "legacy_selected").sum() // len(GATES)), "test_utility_mean": float(detail.loc[(detail["strategy"] == "legacy_selected") & (detail["gate_quantile"] == 0.0), "test_utility"].mean()), "test_attack_mae_after_mean": float(detail.loc[(detail["strategy"] == "legacy_selected") & (detail["gate_quantile"] == 0.0), "test_attack_mae_after"].mean()), "test_attack_improvement_rate_mean": float(detail.loc[(detail["strategy"] == "legacy_selected") & (detail["gate_quantile"] == 0.0), "test_attack_improvement_rate"].mean()), "test_clean_harm_rate_mean": float(detail.loc[(detail["strategy"] == "legacy_selected") & (detail["gate_quantile"] == 0.0), "test_clean_harm_rate"].mean())},
        {"group": "temporal_ungated_reference", "combinations": int((detail["strategy"] == "temporal_only").sum() // len(GATES)), "test_utility_mean": float(detail.loc[(detail["strategy"] == "temporal_only") & (detail["gate_quantile"] == 0.0), "test_utility"].mean()), "test_attack_mae_after_mean": float(detail.loc[(detail["strategy"] == "temporal_only") & (detail["gate_quantile"] == 0.0), "test_attack_mae_after"].mean()), "test_attack_improvement_rate_mean": float(detail.loc[(detail["strategy"] == "temporal_only") & (detail["gate_quantile"] == 0.0), "test_attack_improvement_rate"].mean()), "test_clean_harm_rate_mean": float(detail.loc[(detail["strategy"] == "temporal_only") & (detail["gate_quantile"] == 0.0), "test_clean_harm_rate"].mean())},
    ])
    selected_counts = selected_df.groupby(["selection_policy", "strategy"], as_index=False).size().rename(columns={"size": "selected_combinations"})
    safety_test = float(aggregate.loc[aggregate["group"] == "safety_constrained_selected_by_valid", "test_utility_mean"].iloc[0])
    reference_test = float(aggregate.loc[aggregate["group"] == "legacy_ungated_reference", "test_utility_mean"].iloc[0])
    safety_harm = float(aggregate.loc[aggregate["group"] == "safety_constrained_selected_by_valid", "test_clean_harm_rate_mean"].iloc[0])
    reference_harm = float(aggregate.loc[aggregate["group"] == "legacy_ungated_reference", "test_clean_harm_rate_mean"].iloc[0])
    status = "CANDIDATE_GAIN_REQUIRES_CROSS_SITE_REVIEW" if safety_test > reference_test and safety_harm <= reference_harm else "NEGATIVE_RESULT_RETAINED"
    payload = {
        "status": status,
        "protocol": "阶段129使用阶段124每个种子×留出站点的源valid逐行明细，在预先固定的五种恢复策略和六个风险门控分位数中按攻击恢复增益减正常误报警改写伤害的valid utility选择一个组合；locked test只评价，检测预测不变，未报警行始终透传；不根据test重新选择恢复权重，不根据阶段128 test诊断重新选择恢复权重，也不读取阶段128 test诊断用于选择。",
        "source_test": str(test_path),
        "source_valid": str(valid_path),
        "strategy_count": len(STRATEGIES),
        "gate_count": len(GATES),
        "combination_count": len(selected_unconstrained_df),
        "detail_rows": len(detail),
        "selected_count": len(selected_df),
        "unconstrained_selected_count": len(selected_unconstrained_df),
        "safety_constrained_selected_count": len(selected_constrained_df),
        "aggregate_count": len(aggregate),
        "max_valid_clean_harm_rate": MAX_VALID_CLEAN_HARM_RATE,
        "strategies": list(STRATEGIES),
        "gate_quantiles": GATES,
        "selected": selected_df.to_dict(orient="records"),
        "selected_strategy_counts": selected_counts.to_dict(orient="records"),
        "aggregate": aggregate.to_dict(orient="records"),
        "limitations": [
            "阶段129仍是受控攻击恢复处置实验，不是真实故障日志。",
            "候选和门控只由valid utility选择；test结果不能反向调参。",
            "跨站点选择稳定性和外部NOAA迁移仍需独立复核。",
            "utility是预先固定的安全性权衡，不等同于检测F1或真实业务损失函数。",
        ],
    }
    detail.to_csv(TABLES / "stage129_valid_recovery_strategy_selection_detail.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage129_valid_recovery_strategy_selection_selected.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage129_valid_recovery_strategy_selection.csv", index=False, encoding="utf-8-sig")
    selected_counts.to_csv(TABLES / "stage129_valid_recovery_strategy_selection_counts.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage129_valid_recovery_strategy_selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段129 valid-only恢复策略与门控选择", "", f"- 状态：**{status}**", f"- 每个留出组合：策略数=`{len(STRATEGIES)}`；门控数=`{len(GATES)}`；valid正常误报警改写恶化率约束=`≤{MAX_VALID_CLEAN_HARM_RATE:.2f}`", "", "| 组别 | 组合数 | test utility | test攻击MAE后 | test攻击改善率 | test正常改写恶化率 |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['group']} | {row['combinations']} | {row['test_utility_mean']:.4f} | {row['test_attack_mae_after_mean']:.4f} | {row['test_attack_improvement_rate_mean']:.4f} | {row['test_clean_harm_rate_mean']:.4f} |")
    lines += ["", "阶段129只用valid选择候选与门控，locked test只评价；该结果不能替代阶段124主结果，也不能将受控攻击写成真实故障。"]
    (REPORTS / "stage129_valid_recovery_strategy_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "combinations": len(selected_df), "detail_rows": len(detail), "selected_strategy_counts": selected_counts.to_dict(orient="records")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
