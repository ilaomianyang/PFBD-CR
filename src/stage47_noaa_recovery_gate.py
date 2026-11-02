from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from secure_recovery import add_recovery_candidates, choose_weights, combine_candidates  # noqa: E402
from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402
from stage44_recovery_confidence_gate import summarize  # noqa: E402

TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
QUANTILES = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95]


def run_seed(source: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    frame = build_dataset(source, seed)
    scored, _ = score(frame)
    scored["observed_value"] = pd.to_numeric(scored["effective_observed_value"], errors="coerce")
    scored["is_anomaly"] = scored["combined_is_attack"].astype(bool)
    scored["era5_corrected_t2m_c"] = np.nan  # NOAA experiment intentionally uses no ERA5 candidate.
    valid = scored["split"].eq("valid")
    test = scored["split"].eq("test")
    base_v = scored.loc[valid, "predicted_attack"].astype(bool).to_numpy()
    risk_v = scored.loc[valid, "robust_fused_risk"].to_numpy(dtype=float)
    alert_risk = risk_v[base_v]
    risk_t = scored.loc[test, "robust_fused_risk"].to_numpy(dtype=float)
    base_t = scored.loc[test, "predicted_attack"].astype(bool).to_numpy()
    rows = []
    for q in QUANTILES:
        threshold = float(np.quantile(alert_risk, q))
        gate_v = base_v & (risk_v >= threshold)
        gate_t = base_t & (risk_t >= threshold)
        vf = scored.loc[valid].copy()
        tf = scored.loc[test].copy()
        vc = add_recovery_candidates(vf, pd.Series(gate_v, index=vf.index), vf["era5_corrected_t2m_c"])
        weights = choose_weights(vc)
        vr = combine_candidates(vc, weights)
        tc = add_recovery_candidates(tf, pd.Series(gate_t, index=tf.index), tf["era5_corrected_t2m_c"])
        tr = combine_candidates(tc, weights)
        rows.append({"seed": seed, "gate_quantile": q, "risk_threshold": threshold, "weights": weights, "valid": summarize(vf, vr, gate_v), "test": summarize(tf, tr, gate_t)})
    return rows


def main() -> int:
    source = pd.read_parquet(NOAA)
    rows = []
    for seed in SEEDS:
        rows.extend(run_seed(source, seed))
    selected = []
    reference = []
    for seed in SEEDS:
        candidates = [r for r in rows if r["seed"] == seed]
        selected.append(max(candidates, key=lambda r: (r["valid"]["net_absolute_error_gain"], -r["valid"]["gated_alert_rows"])))
        reference.append(next(r for r in candidates if r["gate_quantile"] == 0.0))
    s_df = pd.DataFrame([{"seed": r["seed"], "gate_quantile": r["gate_quantile"], "risk_threshold": r["risk_threshold"], **{f"valid_{k}": v for k, v in r["valid"].items()}, **{f"test_{k}": v for k, v in r["test"].items()}} for r in rows])
    sel_df = pd.DataFrame([{"seed": r["seed"], "gate_quantile": r["gate_quantile"], **{f"test_{k}": v for k, v in r["test"].items()}} for r in selected])
    ref_df = pd.DataFrame([{"seed": r["seed"], **{f"test_{k}": v for k, v in r["test"].items()}} for r in reference])
    mean_gain = float(sel_df["test_net_absolute_error_gain"].mean())
    mean_ref_gain = float(ref_df["test_net_absolute_error_gain"].mean())
    mean_harm = float(sel_df["test_normal_harm_rate"].mean())
    mean_ref_harm = float(ref_df["test_normal_harm_rate"].mean())
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_CROSS_SOURCE_REVIEW" if mean_gain > mean_ref_gain and mean_harm <= mean_ref_harm else "NEGATIVE_RESULT_RETAINED", "protocol": "NOAA三站点；固定2022年时间切分；5个受控攻击种子；恢复门控和时间/空间恢复权重只由valid选择；不使用ERA5候选；test只评价。", "seeds": SEEDS, "all_candidates": rows, "selected_by_valid": selected, "reference_ungated": reference, "summary": {"selected_mean_net_gain": mean_gain, "reference_mean_net_gain": mean_ref_gain, "selected_mean_normal_harm_rate": mean_harm, "reference_mean_normal_harm_rate": mean_ref_harm}}
    s_df.to_csv(TABLES / "stage47_noaa_recovery_gate_detail.csv", index=False, encoding="utf-8-sig")
    sel_df.to_csv(TABLES / "stage47_noaa_recovery_gate_selected.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage47_noaa_recovery_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段47 NOAA外部恢复门控验证", "", "NOAA实验只使用时间/空间候选，不把ERA5伪装成现场恢复值。", "", "| seed | valid选择门控 | test攻击恢复后MAE | 攻击恶化率 | 正常改写行 | 正常伤害率 | 总误差净收益 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in selected:
        t = r["test"]
        lines.append(f"| {r['seed']} | {r['gate_quantile']:.2f} | {t['detected_attack_mae_after']:.4f} | {t['detected_attack_worsening_rate']:.4f} | {int(t['normal_changed_rows'])} | {t['normal_harm_rate']:.4f} | {t['net_absolute_error_gain']:.2f} |")
    lines += ["", f"- 状态：**{payload['status']}**", f"- 门控平均净收益：`{mean_gain:.2f}`；未门控参考：`{mean_ref_gain:.2f}`。"]
    (REPORTS / "stage47_noaa_recovery_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected_mean_net_gain": mean_gain, "reference_mean_net_gain": mean_ref_gain}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
