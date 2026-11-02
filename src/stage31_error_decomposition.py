from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def mae_rows(frame: pd.DataFrame, value_col: str) -> float:
    true = pd.to_numeric(frame["original_value"], errors="coerce").to_numpy(dtype=float)
    value = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(true) & np.isfinite(value)
    return float(np.mean(np.abs(true[ok] - value[ok]))) if ok.any() else float("nan")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段31：逐站点和逐攻击类型误差分解")
    parser.add_argument("--cold", type=Path, default=TABLES / "stage22_leave_one_station_refit_detail.parquet")
    parser.add_argument("--calibrated", type=Path, default=TABLES / "stage23_unlabeled_site_calibration_detail.parquet")
    args = parser.parse_args()
    cold = pd.read_parquet(args.cold)
    cal = pd.read_parquet(args.calibrated)
    for frame in [cold, cal]:
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    station_rows = []
    for station, g in cold.groupby("held_out_station", sort=True):
        y = g["combined_is_attack"].astype(bool).to_numpy()
        p = g["predicted_attack"].astype(bool).to_numpy()
        attack = g[y]
        detected = g[y & p]
        missed = g[y & ~p]
        normal = g[~y]
        station_rows.append({
            "held_out_station": station,
            "test_rows": int(len(g)),
            "attack_rows": int(y.sum()),
            "detected_attack_rows": int((y & p).sum()),
            "missed_attack_rows": int((y & ~p).sum()),
            "false_positive_rows": int((~y & p).sum()),
            "recall": float((y & p).sum() / y.sum()) if y.sum() else float("nan"),
            "false_alarm_rate": float((~y & p).sum() / (~y).sum()) if (~y).sum() else float("nan"),
            "attack_mae_before": mae_rows(attack, "effective_observed_value"),
            "attack_mae_after_all": mae_rows(attack, "recovered_value"),
            "attack_mae_after_detected": mae_rows(detected, "recovered_value"),
            "attack_mae_before_missed": mae_rows(missed, "effective_observed_value"),
            "attack_mae_after_missed": mae_rows(missed, "recovered_value"),
            "normal_risk_p95": float(pd.to_numeric(normal["robust_fused_risk"], errors="coerce").quantile(0.95)),
            "normal_risk_p99": float(pd.to_numeric(normal["robust_fused_risk"], errors="coerce").quantile(0.99)),
        })
    station = pd.DataFrame(station_rows)
    type_rows = []
    for (held_out, attack_type), g in cold[cold["combined_is_attack"]].groupby(["held_out_station", "combined_attack_type"], sort=True):
        y = g["combined_is_attack"].astype(bool).to_numpy()
        p = g["predicted_attack"].astype(bool).to_numpy()
        type_rows.append({"held_out_station": held_out, "attack_type": attack_type, "rows": int(len(g)), "detected_rows": int((p & y).sum()), "recall": float((p & y).sum() / y.sum()), "mean_risk": float(pd.to_numeric(g["robust_fused_risk"], errors="coerce").mean()), "mean_missed_risk": float(pd.to_numeric(g.loc[~p, "robust_fused_risk"], errors="coerce").mean()) if (~p).any() else float("nan")})
    by_type = pd.DataFrame(type_rows)
    fp = cold[(~cold["combined_is_attack"].astype(bool)) & cold["predicted_attack"].astype(bool)].copy()
    fp["hour_utc"] = fp["timestamp_utc"].dt.hour
    fp["date_utc"] = fp["timestamp_utc"].dt.date.astype(str)
    fp_hour = fp.groupby(["held_out_station", "hour_utc"], sort=True).size().reset_index(name="false_positive_rows").sort_values("false_positive_rows", ascending=False)
    fp_day = fp.groupby(["held_out_station", "date_utc"], sort=True).size().reset_index(name="false_positive_rows").sort_values("false_positive_rows", ascending=False)
    station.to_csv(TABLES / "stage31_station_error_decomposition.csv", index=False, encoding="utf-8-sig")
    by_type.to_csv(TABLES / "stage31_attack_type_error_decomposition.csv", index=False, encoding="utf-8-sig")
    fp_hour.to_csv(TABLES / "stage31_false_positive_by_hour.csv", index=False, encoding="utf-8-sig")
    fp_day.to_csv(TABLES / "stage31_false_positive_by_day.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "仅使用阶段22/23锁定test明细做描述性误差分解，不据此重新选择阈值或修改test结果。", "station_results": station.to_dict(orient="records"), "attack_type_results": by_type.to_dict(orient="records"), "top_false_positive_hours": fp_hour.head(20).to_dict(orient="records"), "top_false_positive_days": fp_day.head(20).to_dict(orient="records"), "calibrated_reference": str(args.calibrated)}
    (TABLES / "stage31_error_decomposition.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段31逐站点和逐攻击类型误差分解", "", "该分析只描述已锁定 test 的误报、漏报和恢复收益，不重新调参。", "", "## 站点汇总", "", "| 站点 | 攻击行 | 检出 | 漏报 | 误报 | Recall | 误伤率 | MAE前 | MAE后 | 漏报行MAE前→后 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in station.iterrows():
        lines.append(f"| {r['held_out_station']} | {r['attack_rows']} | {r['detected_attack_rows']} | {r['missed_attack_rows']} | {r['false_positive_rows']} | {r['recall']:.4f} | {r['false_alarm_rate']:.4f} | {r['attack_mae_before']:.4f} | {r['attack_mae_after_all']:.4f} | {r['attack_mae_before_missed']:.4f}→{r['attack_mae_after_missed']:.4f} |")
    lines += ["", "## 最弱攻击类型（按留出站点）", "", "| 站点 | 攻击类型 | 行数 | Recall | 平均风险 | 漏报平均风险 |", "|---|---|---:|---:|---:|---:|"]
    for _, r in by_type.sort_values(["recall", "held_out_station"]).head(20).iterrows():
        lines.append(f"| {r['held_out_station']} | {r['attack_type']} | {r['rows']} | {r['recall']:.4f} | {r['mean_risk']:.4f} | {r['mean_missed_risk']:.4f} |")
    lines += ["", "## 误报集中时段（前10）", "", "| 站点 | UTC小时 | 误报行数 |", "|---|---:|---:|"]
    for _, r in fp_hour.head(10).iterrows():
        lines.append(f"| {r['held_out_station']} | {int(r['hour_utc']):02d} | {int(r['false_positive_rows'])} |")
    lines += ["", "完整逐类表、误报日期表和校准对照见 `tables/` 下的 stage31 文件。"]
    (REPORTS / "stage31_error_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(station), "attack_type_rows": len(by_type), "false_positive_rows": len(fp)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
