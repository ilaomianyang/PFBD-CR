from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from secure_recovery import add_recovery_candidates, choose_weights, combine_candidates
from stage44_recovery_confidence_gate import summarize

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
QUANTILES = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95]


def run_seed(seed: int) -> list[dict[str, object]]:
    path = TABLES / f"stage09b_local_robust_scores_seed{seed}.parquet"
    df = pd.read_parquet(path).copy()
    df["observed_value"] = pd.to_numeric(df["effective_observed_value"], errors="coerce")
    df["is_anomaly"] = df["combined_is_attack"].astype(bool)
    df["attack_pred_base"] = df["predicted_attack"].astype(bool)
    valid = df["split"].eq("valid")
    test = df["split"].eq("test")
    base_valid = df.loc[valid, "attack_pred_base"].to_numpy(dtype=bool)
    risk_valid = df.loc[valid, "robust_fused_risk"].to_numpy(dtype=float)
    risk_alert = risk_valid[base_valid]
    rows = []
    for q in QUANTILES:
        threshold = float(np.quantile(risk_alert, q))
        valid_gate = base_valid & (risk_valid >= threshold)
        risk_test = df.loc[test, "robust_fused_risk"].to_numpy(dtype=float)
        base_test = df.loc[test, "attack_pred_base"].to_numpy(dtype=bool)
        test_gate = base_test & (risk_test >= threshold)
        vf = df.loc[valid].copy()
        tf = df.loc[test].copy()
        vc = add_recovery_candidates(vf, pd.Series(valid_gate, index=vf.index), vf["era5_corrected_t2m_c"])
        weights = choose_weights(vc)
        vr = combine_candidates(vc, weights)
        tc = add_recovery_candidates(tf, pd.Series(test_gate, index=tf.index), tf["era5_corrected_t2m_c"])
        tr = combine_candidates(tc, weights)
        vm = summarize(vf, vr, valid_gate)
        tm = summarize(tf, tr, test_gate)
        rows.append({"seed": seed, "gate_quantile": q, "risk_threshold": threshold, "weights": weights, "valid": vm, "test": tm})
    return rows


def main() -> int:
    rows = []
    for seed in SEEDS:
        rows.extend(run_seed(seed))
    flat = []
    for r in rows:
        flat.append({"seed": r["seed"], "gate_quantile": r["gate_quantile"], "risk_threshold": r["risk_threshold"], **{f"valid_{k}": v for k, v in r["valid"].items()}, **{f"test_{k}": v for k, v in r["test"].items()}})
    detail = pd.DataFrame(flat)
    selected = []
    reference = []
    for seed, g in detail.groupby("seed", sort=True):
        selected.append(g.sort_values(["valid_net_absolute_error_gain", "valid_gated_alert_rows"], ascending=[False, True]).iloc[0].to_dict())
        reference.append(g[g["gate_quantile"] == 0.0].iloc[0].to_dict())
    selected_df = pd.DataFrame(selected)
    reference_df = pd.DataFrame(reference)
    metrics = ["test_detected_attack_mae_after", "test_detected_attack_improvement_rate", "test_detected_attack_worsening_rate", "test_normal_changed_rows", "test_normal_harm_rate", "test_net_absolute_error_gain"]
    aggregate = selected_df[metrics].agg(["mean", "std", "min", "max"]).to_dict()
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_CROSS_SITE_REVIEW", "protocol": "9B五种子；检测预测不变；恢复门控分位数和权重各自只由valid选择；test只评价；未报警行透传。", "seeds": SEEDS, "quantiles": QUANTILES, "detail": rows, "selected_by_valid": selected, "reference_ungated": reference, "aggregate_selected": aggregate}
    detail.to_csv(TABLES / "stage45_recovery_gate_multiseed_detail.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage45_recovery_gate_multiseed_selected.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage45_recovery_gate_multiseed.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段45恢复门控五种子验证", "", "检测预测固定为9B；门控和恢复权重只由各自valid选择，test只评价。", "", "| seed | valid选择门控 | test攻击恢复后MAE | 攻击改善率 | 攻击恶化率 | 正常改写行 | 正常伤害率 | 总误差净收益 |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in selected:
        lines.append(f"| {int(r['seed'])} | {r['gate_quantile']:.2f} | {r['test_detected_attack_mae_after']:.4f} | {r['test_detected_attack_improvement_rate']:.4f} | {r['test_detected_attack_worsening_rate']:.4f} | {int(r['test_normal_changed_rows'])} | {r['test_normal_harm_rate']:.4f} | {r['test_net_absolute_error_gain']:.2f} |")
    lines += ["", "聚合统计和完整门控候选见 `tables/stage45_recovery_gate_multiseed_detail.csv`。跨站点复核前不替换9B恢复结果。"]
    (REPORTS / "stage45_recovery_gate_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "seeds": len(SEEDS), "selected_gate_quantiles": selected_df["gate_quantile"].tolist(), "mean_net_gain": float(selected_df["test_net_absolute_error_gain"].mean())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
