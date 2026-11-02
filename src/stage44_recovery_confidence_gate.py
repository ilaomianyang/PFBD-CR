from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from secure_recovery import add_recovery_candidates, choose_weights, combine_candidates

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SCORES = TABLES / "stage09b_local_robust_scores.parquet"


def summarize(frame: pd.DataFrame, recovered: np.ndarray, gated_pred: np.ndarray) -> dict[str, float | int]:
    true = pd.to_numeric(frame["original_value"], errors="coerce").to_numpy(dtype=float)
    observed = pd.to_numeric(frame["effective_observed_value"], errors="coerce").to_numpy(dtype=float)
    label = frame["combined_is_attack"].astype(bool).to_numpy()
    finite = np.isfinite(true) & np.isfinite(observed) & np.isfinite(recovered)
    before = np.abs(true - observed)
    after = np.abs(true - recovered)
    improvement = before - after
    attack = finite & label & gated_pred
    normal = finite & (~label) & gated_pred
    changed = finite & (np.abs(recovered - observed) > 1e-9)
    return {
        "rows": int(len(frame)),
        "gated_alert_rows": int(gated_pred.sum()),
        "detected_attack_rows": int(attack.sum()),
        "missed_attack_rows": int((finite & label & ~gated_pred).sum()),
        "detected_attack_mae_before": float(before[attack].mean()) if attack.any() else float("nan"),
        "detected_attack_mae_after": float(after[attack].mean()) if attack.any() else float("nan"),
        "detected_attack_improvement_rate": float((improvement[attack] > 1e-9).mean()) if attack.any() else float("nan"),
        "detected_attack_worsening_rate": float((improvement[attack] < -1e-9).mean()) if attack.any() else float("nan"),
        "normal_changed_rows": int((normal & changed).sum()),
        "normal_harm_rate": float((after[normal] > before[normal] + 1e-9).mean()) if normal.any() else 0.0,
        "normal_mean_abs_change": float(np.abs(recovered[normal] - observed[normal]).mean()) if normal.any() else 0.0,
        "overall_mae_before": float(before[finite].mean()) if finite.any() else float("nan"),
        "overall_mae_after": float(after[finite].mean()) if finite.any() else float("nan"),
        "net_absolute_error_gain": float(improvement[finite].sum()) if finite.any() else float("nan"),
    }


def main() -> int:
    df = pd.read_parquet(SCORES).copy()
    df["observed_value"] = pd.to_numeric(df["effective_observed_value"], errors="coerce")
    df["is_anomaly"] = df["combined_is_attack"].astype(bool)
    df["attack_pred_base"] = df["predicted_attack"].astype(bool)
    train = df["split"].eq("train")
    valid = df["split"].eq("valid")
    test = df["split"].eq("test")
    valid_base = df.loc[valid, "attack_pred_base"].to_numpy(dtype=bool)
    valid_risk = df.loc[valid, "robust_fused_risk"].to_numpy(dtype=float)
    valid_pred_risk = valid_risk[valid_base]
    if len(valid_pred_risk) == 0:
        raise RuntimeError("no validation alerts available for recovery gate")
    quantiles = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95]
    rows = []
    details = []
    for q in quantiles:
        threshold = float(np.quantile(valid_pred_risk, q))
        valid_gate = valid_base & (df.loc[valid, "robust_fused_risk"].to_numpy(dtype=float) >= threshold)
        test_gate = df.loc[test, "attack_pred_base"].to_numpy(dtype=bool) & (df.loc[test, "robust_fused_risk"].to_numpy(dtype=float) >= threshold)
        valid_frame = df.loc[valid].copy()
        test_frame = df.loc[test].copy()
        valid_candidates = add_recovery_candidates(valid_frame, pd.Series(valid_gate, index=valid_frame.index), valid_frame["era5_corrected_t2m_c"])
        weights = choose_weights(valid_candidates)
        valid_recovered = combine_candidates(valid_candidates, weights)
        test_candidates = add_recovery_candidates(test_frame, pd.Series(test_gate, index=test_frame.index), test_frame["era5_corrected_t2m_c"])
        test_recovered = combine_candidates(test_candidates, weights)
        vm = summarize(valid_frame, valid_recovered, valid_gate)
        tm = summarize(test_frame, test_recovered, test_gate)
        rows.append({"gate_quantile": q, "risk_threshold": threshold, "weights": weights, "valid": vm, "test": tm})
        details.append({"gate_quantile": q, "risk_threshold": threshold, "split": "valid", **vm})
        details.append({"gate_quantile": q, "risk_threshold": threshold, "split": "test", **tm})
    # Select only on validation net absolute-error gain, with fewer gated alerts
    # as the tie-breaker; test is never consulted for selection.
    selected = max(rows, key=lambda r: (r["valid"]["net_absolute_error_gain"], -r["valid"]["gated_alert_rows"]))
    reference = rows[0]
    test_gain = selected["test"]["net_absolute_error_gain"] - reference["test"]["net_absolute_error_gain"]
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_MULTISEED" if test_gain > 0 and selected["test"]["normal_harm_rate"] <= reference["test"]["normal_harm_rate"] else "NEGATIVE_RESULT_RETAINED", "protocol": "检测预测固定为9B；恢复门控阈值和恢复权重只由valid选择；locked test只评价；未报警行始终透传。", "candidates": rows, "selected_by_valid": selected, "reference_ungated": reference}
    pd.DataFrame(details).to_csv(TABLES / "stage44_recovery_confidence_gate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage44_recovery_confidence_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段44恢复置信度门控实验", "", "检测结果固定为9B；仅对报警行增加恢复门控，门控与恢复权重由valid选择，test只评价。", "", "| 门控分位数 | 风险阈值 | test报警行 | 攻击恢复前→后MAE | 攻击改善率 | 攻击恶化率 | 正常改写行 | 正常伤害率 | 总误差净收益 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        t = r["test"]
        lines.append(f"| {r['gate_quantile']:.2f} | {r['risk_threshold']:.4f} | {t['gated_alert_rows']} | {t['detected_attack_mae_before']:.4f}→{t['detected_attack_mae_after']:.4f} | {t['detected_attack_improvement_rate']:.4f} | {t['detected_attack_worsening_rate']:.4f} | {t['normal_changed_rows']} | {t['normal_harm_rate']:.4f} | {t['net_absolute_error_gain']:.2f} |")
    lines += ["", f"- 状态：**{payload['status']}**", f"- valid选择门控分位数：`{selected['gate_quantile']:.2f}`。", "- 该实验不改变9B检测指标，只评价恢复处置的安全性与净效用。"]
    (REPORTS / "stage44_recovery_confidence_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected_gate_quantile": selected["gate_quantile"], "selected_test_net_gain": selected["test"]["net_absolute_error_gain"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
