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
TEST_DETAIL = TABLES / "stage22_leave_one_station_refit_detail.parquet"
VALID_DETAIL = TABLES / "stage22_leave_one_station_refit_valid_detail.parquet"
QUANTILES = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95]


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["observed_value"] = pd.to_numeric(out["effective_observed_value"], errors="coerce")
    out["is_anomaly"] = out["combined_is_attack"].astype(bool)
    return out


def evaluate_station(valid: pd.DataFrame, test: pd.DataFrame, station: str) -> list[dict[str, object]]:
    valid = prepare(valid)
    test = prepare(test)
    base_valid = valid["predicted_attack"].astype(bool).to_numpy()
    risk_valid = pd.to_numeric(valid["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
    alert_risk = risk_valid[base_valid]
    if len(alert_risk) == 0:
        raise RuntimeError(f"no validation alerts for {station}")
    rows = []
    risk_test = pd.to_numeric(test["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
    base_test = test["predicted_attack"].astype(bool).to_numpy()
    for q in QUANTILES:
        threshold = float(np.quantile(alert_risk, q))
        vg = base_valid & (risk_valid >= threshold)
        tg = base_test & (risk_test >= threshold)
        vc = add_recovery_candidates(valid, pd.Series(vg, index=valid.index), valid["era5_corrected_t2m_c"])
        weights = choose_weights(vc)
        vr = combine_candidates(vc, weights)
        tc = add_recovery_candidates(test, pd.Series(tg, index=test.index), test["era5_corrected_t2m_c"])
        tr = combine_candidates(tc, weights)
        vm = summarize(valid, vr, vg)
        tm = summarize(test, tr, tg)
        rows.append({"held_out_station": station, "gate_quantile": q, "risk_threshold": threshold, "weights": weights, "valid": vm, "test": tm})
    return rows


def main() -> int:
    test = pd.read_parquet(TEST_DETAIL)
    valid = pd.read_parquet(VALID_DETAIL)
    stations = sorted(test["held_out_station"].astype(str).unique().tolist())
    rows = []
    for station in stations:
        rows.extend(evaluate_station(valid[valid["held_out_station"].astype(str).eq(station)], test[test["held_out_station"].astype(str).eq(station)], station))
    selected = []
    reference = []
    for station, g in pd.DataFrame([{"held_out_station": r["held_out_station"], "gate_quantile": r["gate_quantile"], "valid_net_gain": r["valid"]["net_absolute_error_gain"], "valid_alerts": r["valid"]["gated_alert_rows"], "test": r["test"]} for r in rows]).groupby("held_out_station", sort=True):
        best_row = max((r for r in rows if r["held_out_station"] == station), key=lambda r: (r["valid"]["net_absolute_error_gain"], -r["valid"]["gated_alert_rows"]))
        selected.append(best_row)
        reference.append(next(r for r in rows if r["held_out_station"] == station and r["gate_quantile"] == 0.0))
    selected_test = pd.DataFrame([{"held_out_station": r["held_out_station"], **r["test"]} for r in selected])
    reference_test = pd.DataFrame([{"held_out_station": r["held_out_station"], **r["test"]} for r in reference])
    metrics = ["detected_attack_mae_after", "detected_attack_improvement_rate", "detected_attack_worsening_rate", "normal_changed_rows", "normal_harm_rate", "net_absolute_error_gain"]
    aggregate = {"selected": selected_test[metrics].agg(["mean", "std", "min", "max"]).to_dict(), "reference_ungated": reference_test[metrics].agg(["mean", "std", "min", "max"]).to_dict()}
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_CROSS_SITE_REVIEW", "protocol": "阶段22严格留一站点；每个留出站点的门控和恢复权重只由其余站点valid选择；留出站点test评价；未报警行透传。", "stations": stations, "all_candidates": rows, "selected_by_valid": selected, "reference_ungated": reference, "aggregate": aggregate}
    flat = []
    for r in rows:
        flat.append({"held_out_station": r["held_out_station"], "gate_quantile": r["gate_quantile"], "risk_threshold": r["risk_threshold"], **{f"valid_{k}": v for k, v in r["valid"].items()}, **{f"test_{k}": v for k, v in r["test"].items()}})
    pd.DataFrame(flat).to_csv(TABLES / "stage46_cross_station_recovery_gate_detail.csv", index=False, encoding="utf-8-sig")
    selected_test.to_csv(TABLES / "stage46_cross_station_recovery_gate_selected.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage46_cross_station_recovery_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段46严格留一站点恢复门控验证", "", "每个留出站点只使用其余站点的valid选择门控和恢复权重，留出站点test只评价。", "", "| 留出站点 | valid选择门控 | 攻击恢复后MAE | 攻击改善率 | 攻击恶化率 | 正常改写行 | 正常伤害率 | 总误差净收益 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in selected:
        t = r["test"]
        lines.append(f"| {r['held_out_station']} | {r['gate_quantile']:.2f} | {t['detected_attack_mae_after']:.4f} | {t['detected_attack_improvement_rate']:.4f} | {t['detected_attack_worsening_rate']:.4f} | {int(t['normal_changed_rows'])} | {t['normal_harm_rate']:.4f} | {t['net_absolute_error_gain']:.2f} |")
    lines += ["", "聚合结果和全部候选见 `tables/stage46_cross_station_recovery_gate_detail.csv`。在更多独立站点复核前，不替换主恢复策略。"]
    (REPORTS / "stage46_cross_station_recovery_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(stations), "mean_selected_net_gain": aggregate["selected"]["net_absolute_error_gain"]["mean"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
