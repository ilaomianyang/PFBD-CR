from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
GATES = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95]
MAX_VALID_CLEAN_HARM_RATE = 0.20
STRATEGIES = ["no_recovery", "causal_ffill", "causal_slope", "causal_blend"]


def causal_candidates(frame: pd.DataFrame, fallback: float, lower: float, upper: float) -> pd.DataFrame:
    out = frame.copy()
    out["causal_ffill"] = np.nan
    out["causal_slope"] = np.nan
    out["causal_blend"] = np.nan
    ordered = out.sort_values(["station_key", "timestamp_utc"]).reset_index().rename(columns={"index": "_original_index"})
    observed = pd.to_numeric(ordered["effective_observed_value"], errors="coerce").to_numpy(dtype=float)
    attacked = ordered["predicted_attack"].astype(bool).to_numpy()
    ffill = np.full(len(ordered), float(fallback), dtype=float)
    slope = np.full(len(ordered), float(fallback), dtype=float)
    for station, indices in ordered.groupby("station_key", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        last = None
        previous = None
        for position in positions:
            current = observed[position]
            if last is None:
                ffill[position] = fallback
                slope[position] = fallback
            else:
                ffill[position] = last
                estimate = last if previous is None else last + (last - previous)
                slope[position] = float(np.clip(estimate, lower, upper))
            if np.isfinite(current) and not attacked[position]:
                previous, last = last, current
    out.loc[ordered["_original_index"].to_numpy(), "causal_ffill"] = ffill
    out.loc[ordered["_original_index"].to_numpy(), "causal_slope"] = slope
    out["causal_blend"] = 0.5 * out["causal_ffill"] + 0.5 * out["causal_slope"]
    return out


def recover(frame: pd.DataFrame, strategy: str, gate: np.ndarray) -> np.ndarray:
    observed = pd.to_numeric(frame["effective_observed_value"], errors="coerce").to_numpy(dtype=float)
    result = observed.copy()
    if strategy == "no_recovery":
        return result
    candidate = pd.to_numeric(frame[strategy], errors="coerce").to_numpy(dtype=float)
    usable = gate & np.isfinite(candidate)
    result[usable] = candidate[usable]
    return result


def metrics(frame: pd.DataFrame, recovered: np.ndarray, gate: np.ndarray) -> dict[str, float | int]:
    true = pd.to_numeric(frame["original_value"], errors="coerce").to_numpy(dtype=float)
    observed = pd.to_numeric(frame["effective_observed_value"], errors="coerce").to_numpy(dtype=float)
    attack = frame["combined_is_attack"].astype(bool).to_numpy()
    finite = np.isfinite(true) & np.isfinite(observed) & np.isfinite(recovered)
    before = np.abs(observed - true)
    after = np.abs(recovered - true)
    attack_rows = finite & attack & gate
    clean_rows = finite & ~attack & gate
    attack_gain = before[attack_rows] - after[attack_rows]
    clean_harm = after[clean_rows] - before[clean_rows]
    return {
        "rows": int(len(frame)),
        "gated_alert_rows": int(gate.sum()),
        "detected_attack_rows": int(attack_rows.sum()),
        "attack_mae_before": float(before[attack_rows].mean()) if attack_rows.any() else float("nan"),
        "attack_mae_after": float(after[attack_rows].mean()) if attack_rows.any() else float("nan"),
        "attack_improvement_rate": float((attack_gain > 1e-9).mean()) if attack_rows.any() else float("nan"),
        "attack_worsening_rate": float((attack_gain < -1e-9).mean()) if attack_rows.any() else float("nan"),
        "clean_harm_rate": float((clean_harm > 1e-9).mean()) if clean_rows.any() else 0.0,
        "clean_changed_rows": int((clean_rows & (np.abs(recovered - observed) > 1e-9)).sum()),
        "attack_gain_sum": float(attack_gain.sum()) if attack_rows.any() else 0.0,
        "clean_harm_sum": float(np.maximum(clean_harm, 0.0).sum()) if clean_rows.any() else 0.0,
        "utility": float(attack_gain.sum() - np.maximum(clean_harm, 0.0).sum()) if attack_rows.any() or clean_rows.any() else 0.0,
    }


def main() -> int:
    test_path = TABLES / "stage124_leave_one_station_refit_multiseed_detail.parquet"
    valid_path = TABLES / "stage124_leave_one_station_refit_multiseed_valid_detail.parquet"
    if not test_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("stage124 test/valid detail artifacts are required")
    test = pd.read_parquet(test_path)
    valid = pd.read_parquet(valid_path)
    detail_rows: list[dict[str, object]] = []
    selected_unconstrained: list[dict[str, object]] = []
    selected_constrained: list[dict[str, object]] = []
    for (seed, heldout), vf_raw in valid.groupby(["seed", "held_out_station"], sort=True):
        tf_raw = test[(test["seed"] == seed) & (test["held_out_station"] == heldout)].copy()
        vf_raw = vf_raw.copy()
        source_observed = pd.to_numeric(vf_raw["effective_observed_value"], errors="coerce")
        source_clean = ~vf_raw["combined_is_attack"].astype(bool) & ~vf_raw["predicted_attack"].astype(bool)
        fallback = float(source_observed.loc[source_clean].median()) if source_clean.any() else float(source_observed.median())
        lower = float(source_observed.loc[source_clean].quantile(0.01)) if source_clean.any() else float(source_observed.quantile(0.01))
        upper = float(source_observed.loc[source_clean].quantile(0.99)) if source_clean.any() else float(source_observed.quantile(0.99))
        if not np.isfinite(fallback) or not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"non-finite causal calibration for seed={seed}, held_out={heldout}")
        vf = causal_candidates(vf_raw, fallback, lower, upper)
        tf = causal_candidates(tf_raw, fallback, lower, upper)
        risk_v = pd.to_numeric(vf["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
        alert_v = vf["predicted_attack"].astype(bool).to_numpy()
        alert_risk = risk_v[alert_v & np.isfinite(risk_v)]
        if len(alert_risk) == 0:
            raise ValueError(f"no validation alerts for seed={seed}, held_out={heldout}")
        risk_t = pd.to_numeric(tf["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
        alert_t = tf["predicted_attack"].astype(bool).to_numpy()
        candidates: list[dict[str, object]] = []
        for strategy in STRATEGIES:
            for gate_quantile in GATES:
                threshold = float(np.quantile(alert_risk, gate_quantile))
                gate_v = alert_v & (risk_v >= threshold)
                gate_t = alert_t & (risk_t >= threshold)
                vm = metrics(vf, recover(vf, strategy, gate_v), gate_v)
                tm = metrics(tf, recover(tf, strategy, gate_t), gate_t)
                row = {
                    "seed": int(seed), "held_out_station": str(heldout), "strategy": strategy,
                    "gate_quantile": float(gate_quantile), "risk_threshold": threshold,
                    "causal_fallback_value": fallback, "causal_lower_bound": lower, "causal_upper_bound": upper,
                    "valid_utility": vm["utility"], "test_utility": tm["utility"],
                    "valid_attack_mae_after": vm["attack_mae_after"], "test_attack_mae_after": tm["attack_mae_after"],
                    "valid_attack_improvement_rate": vm["attack_improvement_rate"], "test_attack_improvement_rate": tm["attack_improvement_rate"],
                    "valid_attack_worsening_rate": vm["attack_worsening_rate"], "test_attack_worsening_rate": tm["attack_worsening_rate"],
                    "valid_clean_harm_rate": vm["clean_harm_rate"], "test_clean_harm_rate": tm["clean_harm_rate"],
                    "valid_gated_alert_rows": vm["gated_alert_rows"], "test_gated_alert_rows": tm["gated_alert_rows"],
                }
                detail_rows.append(row)
                candidates.append(row)
        best = max(candidates, key=lambda row: (float(row["valid_utility"]), -int(row["valid_gated_alert_rows"]), row["strategy"], -float(row["gate_quantile"])))
        feasible = [row for row in candidates if float(row["valid_clean_harm_rate"]) <= MAX_VALID_CLEAN_HARM_RATE]
        constrained = max(feasible or [row for row in candidates if row["strategy"] == "no_recovery"], key=lambda row: (float(row["valid_utility"]), -int(row["valid_gated_alert_rows"]), row["strategy"], -float(row["gate_quantile"])))
        best = dict(best); constrained = dict(constrained)
        best["selection_policy"] = "unconstrained_max_valid_utility"
        constrained["selection_policy"] = "valid_clean_harm_rate_le_0.20"
        selected_unconstrained.append(best); selected_constrained.append(constrained)
    detail = pd.DataFrame(detail_rows)
    selected = pd.DataFrame(selected_unconstrained + selected_constrained)
    aggregate = pd.DataFrame([
        {"group": "unconstrained_selected_by_valid", "combinations": 35, "test_utility_mean": float(pd.DataFrame(selected_unconstrained)["test_utility"].mean()), "test_attack_mae_after_mean": float(pd.DataFrame(selected_unconstrained)["test_attack_mae_after"].mean()), "test_attack_improvement_rate_mean": float(pd.DataFrame(selected_unconstrained)["test_attack_improvement_rate"].mean()), "test_clean_harm_rate_mean": float(pd.DataFrame(selected_unconstrained)["test_clean_harm_rate"].mean())},
        {"group": "safety_constrained_selected_by_valid", "combinations": 35, "test_utility_mean": float(pd.DataFrame(selected_constrained)["test_utility"].mean()), "test_attack_mae_after_mean": float(pd.DataFrame(selected_constrained)["test_attack_mae_after"].mean()), "test_attack_improvement_rate_mean": float(pd.DataFrame(selected_constrained)["test_attack_improvement_rate"].mean()), "test_clean_harm_rate_mean": float(pd.DataFrame(selected_constrained)["test_clean_harm_rate"].mean())},
        {"group": "no_recovery_reference", "combinations": 35, "test_utility_mean": 0.0, "test_attack_mae_after_mean": float(detail.loc[(detail["strategy"] == "no_recovery") & (detail["gate_quantile"] == 0.0), "test_attack_mae_after"].mean()), "test_attack_improvement_rate_mean": 0.0, "test_clean_harm_rate_mean": 0.0},
    ])
    counts = selected.groupby(["selection_policy", "strategy"], as_index=False).size().rename(columns={"size": "selected_combinations"})
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "阶段134使用阶段124五种子×七站点valid/test明细，构造只看过去的causal_ffill、causal_slope和causal_blend候选；趋势外推在valid正常观测1%/99%分位数范围内裁剪，回退值只由valid未报警正常观测估计。策略和风险门控仅由valid utility选择，并列安全约束≤20%方案；test只评价，不使用未来观测、不根据test重新选择。",
        "strategy_count": len(STRATEGIES), "gate_count": len(GATES), "detail_rows": len(detail), "selected_count": len(selected), "aggregate_count": len(aggregate), "count_rows": len(counts),
        "selected_strategy_counts": counts.to_dict(orient="records"), "aggregate": aggregate.to_dict(orient="records"), "limitations": [
            "因果候选只使用过去信息，但仍是受控攻击恢复实验，不是真实在线故障日志。", "趋势裁剪边界由valid估计，不证明其对所有站点或季节最优。", "恢复处置不改变检测指标，不能替代检测性能比较。",
        ],
    }
    detail.to_csv(TABLES / "stage134_causal_recovery_optimization_detail.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(TABLES / "stage134_causal_recovery_optimization_selected.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage134_causal_recovery_optimization.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(TABLES / "stage134_causal_recovery_optimization_counts.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage134_causal_recovery_optimization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段134因果恢复候选优化", "", f"- 策略数：`{len(STRATEGIES)}`；门控数：`{len(GATES)}`；安全约束：`valid正常改写恶化率≤{MAX_VALID_CLEAN_HARM_RATE:.2f}`", "", "| 组别 | 组合数 | test utility | test攻击MAE后 | test攻击改善率 | test正常改写恶化率 |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['group']} | {row['combinations']} | {row['test_utility_mean']:.4f} | {row['test_attack_mae_after_mean']:.4f} | {row['test_attack_improvement_rate_mean']:.4f} | {row['test_clean_harm_rate_mean']:.4f} |")
    lines += ["", "阶段134所有候选只看过去信息；结果不能与阶段133离线双向插值直接混写。"]
    (REPORTS / "stage134_causal_recovery_optimization.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "selected_count": len(selected), "aggregate_count": len(aggregate), "counts": counts.to_dict(orient="records")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
