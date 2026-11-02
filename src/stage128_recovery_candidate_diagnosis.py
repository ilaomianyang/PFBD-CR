from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
CANDIDATES = ["temporal_candidate", "spatial_candidate", "era5_candidate", "recovered_value"]


def _candidate_error(frame: pd.DataFrame, candidate: str, mask: pd.Series) -> dict[str, float | int]:
    part = frame.loc[mask].copy()
    true = pd.to_numeric(part["original_value"], errors="coerce").to_numpy(dtype=float)
    estimate = pd.to_numeric(part[candidate], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(true) & np.isfinite(estimate)
    if not finite.any():
        return {"rows": int(len(part)), "finite_rows": 0, "mae": float("nan"), "rmse": float("nan"), "improvement_rate": float("nan"), "worsening_rate": float("nan")}
    error = np.abs(estimate[finite] - true[finite])
    baseline = np.abs(pd.to_numeric(part.loc[finite, "effective_observed_value"], errors="coerce").to_numpy(dtype=float) - true[finite])
    valid = np.isfinite(baseline)
    improvement = error[valid] < baseline[valid] - 1e-12
    worsening = error[valid] > baseline[valid] + 1e-12
    return {
        "rows": int(len(part)),
        "finite_rows": int(finite.sum()),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((estimate[finite] - true[finite]) ** 2))),
        "improvement_rate": float(improvement.mean()) if valid.any() else float("nan"),
        "worsening_rate": float(worsening.mean()) if valid.any() else float("nan"),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段128：五种子严格留一站点恢复候选诊断")
    args = parser.parse_args()
    detail_path = TABLES / "stage124_leave_one_station_refit_multiseed_detail.parquet"
    result_path = TABLES / "stage124_leave_one_station_refit_multiseed.json"
    if not detail_path.is_file() or not result_path.is_file():
        raise FileNotFoundError("stage124 detail/json missing")
    detail = pd.read_parquet(detail_path)
    payload124 = json.loads(result_path.read_text(encoding="utf-8"))
    weight_map = {(int(row["seed"]), str(row["held_out_station"])): row.get("recovery_weights", {}) for row in payload124["results"]}
    rows: list[dict[str, object]] = []
    for (seed, heldout), group in detail.groupby(["seed", "held_out_station"], sort=True):
        group = group.copy()
        attack_alert = group["combined_is_attack"].astype(bool) & group["predicted_attack"].astype(bool)
        clean_alert = ~group["combined_is_attack"].astype(bool) & group["predicted_attack"].astype(bool)
        for candidate in CANDIDATES:
            attack = _candidate_error(group, candidate, attack_alert)
            clean = _candidate_error(group, candidate, clean_alert)
            rows.append({
                "seed": int(seed),
                "held_out_station": str(heldout),
                "candidate": candidate,
                "selected_recovery_weights": json.dumps(weight_map.get((int(seed), str(heldout)), {}), ensure_ascii=False, sort_keys=True),
                "attack_alert_rows": int(attack_alert.sum()),
                "clean_alert_rows": int(clean_alert.sum()),
                "attack_mae": attack["mae"],
                "attack_rmse": attack["rmse"],
                "attack_improvement_rate": attack["improvement_rate"],
                "attack_worsening_rate": attack["worsening_rate"],
                "clean_mae": clean["mae"],
                "clean_rmse": clean["rmse"],
                "clean_improvement_rate": clean["improvement_rate"],
                "clean_worsening_rate": clean["worsening_rate"],
            })
    result = pd.DataFrame(rows)
    station_summary = result.groupby(["held_out_station", "candidate"], as_index=False).agg(
        seeds=("seed", "nunique"), attack_alert_rows_mean=("attack_alert_rows", "mean"), clean_alert_rows_mean=("clean_alert_rows", "mean"),
        attack_mae_mean=("attack_mae", "mean"), attack_rmse_mean=("attack_rmse", "mean"), attack_improvement_rate_mean=("attack_improvement_rate", "mean"), attack_worsening_rate_mean=("attack_worsening_rate", "mean"),
        clean_mae_mean=("clean_mae", "mean"), clean_rmse_mean=("clean_rmse", "mean"), clean_worsening_rate_mean=("clean_worsening_rate", "mean"),
    )
    candidate_summary = result.groupby("candidate", as_index=False).agg(
        combinations=("seed", "size"), attack_alert_rows_mean=("attack_alert_rows", "mean"), clean_alert_rows_mean=("clean_alert_rows", "mean"),
        attack_mae_mean=("attack_mae", "mean"), attack_rmse_mean=("attack_rmse", "mean"), attack_improvement_rate_mean=("attack_improvement_rate", "mean"), attack_worsening_rate_mean=("attack_worsening_rate", "mean"),
        clean_mae_mean=("clean_mae", "mean"), clean_rmse_mean=("clean_rmse", "mean"), clean_worsening_rate_mean=("clean_worsening_rate", "mean"),
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "读取阶段124五种子×七站点locked test逐行恢复候选；按已报警攻击行和误报警正常行分别计算当前加权恢复、temporal、spatial和ERA5候选相对原观测的MAE/RMSE、改善率和恶化率；不重新选择恢复权重、不改变检测预测，不将test诊断用于调参。",
        "source": str(detail_path),
        "result_count": len(result),
        "station_summary_count": len(station_summary),
        "candidate_summary_count": len(candidate_summary),
        "results": result.to_dict(orient="records"),
        "station_summary": station_summary.to_dict(orient="records"),
        "candidate_summary": candidate_summary.to_dict(orient="records"),
        "limitations": [
            "该阶段使用locked test做事后诊断，不可用于选择主模型或恢复权重。",
            "未报警行透传，诊断只针对报警后的处置行；正常误报警行的改写伤害单独报告。",
            "攻击为受控注入，候选误差不能直接代表真实故障修复效果。",
        ],
    }
    result.to_csv(TABLES / "stage128_recovery_candidate_diagnosis_detail.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(TABLES / "stage128_recovery_candidate_diagnosis_station_summary.csv", index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(TABLES / "stage128_recovery_candidate_diagnosis.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage128_recovery_candidate_diagnosis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段128恢复候选诊断", "",
        f"- 状态：**{payload['status']}**", f"- 组合数：`{len(result) // len(CANDIDATES)}`；候选数：`{len(CANDIDATES)}`", "",
        "| 候选 | 报警攻击行 | 误报警正常行 | 攻击MAE | 攻击改善率 | 攻击恶化率 | 正常误报警改写MAE | 正常改写恶化率 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_summary.to_dict(orient="records"):
        lines.append(f"| {row['candidate']} | {row['attack_alert_rows_mean']:.1f} | {row['clean_alert_rows_mean']:.1f} | {row['attack_mae_mean']:.4f} | {row['attack_improvement_rate_mean']:.4f} | {row['attack_worsening_rate_mean']:.4f} | {row['clean_mae_mean']:.4f} | {row['clean_worsening_rate_mean']:.4f} |")
    lines += ["", "该阶段是test事后诊断，不用于选择恢复权重；薄弱站点和恢复退化必须保留。"]
    (REPORTS / "stage128_recovery_candidate_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "results": len(result), "stations": len(station_summary["held_out_station"].unique()), "candidates": len(CANDIDATES)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
