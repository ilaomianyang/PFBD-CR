from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
CANDIDATES = ["offline_temporal_candidate", "causal_temporal_candidate"]


def errors(frame: pd.DataFrame, candidate: str, mask: pd.Series) -> dict[str, float | int]:
    part = frame.loc[mask]
    true = pd.to_numeric(part["original_value"], errors="coerce").to_numpy(dtype=float)
    estimate = pd.to_numeric(part[candidate], errors="coerce").to_numpy(dtype=float)
    observed = pd.to_numeric(part["effective_observed_value"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(true) & np.isfinite(estimate) & np.isfinite(observed)
    if not finite.any():
        return {"rows": int(len(part)), "finite_rows": 0, "mae": float("nan"), "rmse": float("nan"), "improvement_rate": float("nan"), "worsening_rate": float("nan")}
    err = np.abs(estimate[finite] - true[finite])
    baseline = np.abs(observed[finite] - true[finite])
    return {
        "rows": int(len(part)),
        "finite_rows": int(finite.sum()),
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((estimate[finite] - true[finite]) ** 2))),
        "improvement_rate": float((err < baseline - 1e-12).mean()),
        "worsening_rate": float((err > baseline + 1e-12).mean()),
    }


def causal_candidate(test: pd.DataFrame, fallback: float) -> np.ndarray:
    ordered = test.sort_values(["station_key", "timestamp_utc"]).copy()
    masked = pd.to_numeric(ordered["effective_observed_value"], errors="coerce").where(~ordered["predicted_attack"].astype(bool))
    result = masked.groupby(ordered["station_key"], sort=False).ffill().fillna(float(fallback))
    return pd.Series(result.to_numpy(dtype=float), index=ordered.index).reindex(test.index).to_numpy(dtype=float)


def main() -> int:
    test_path = TABLES / "stage124_leave_one_station_refit_multiseed_detail.parquet"
    valid_path = TABLES / "stage124_leave_one_station_refit_multiseed_valid_detail.parquet"
    if not test_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("stage124 test/valid detail artifacts are required")
    test = pd.read_parquet(test_path)
    valid = pd.read_parquet(valid_path)
    rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for (seed, heldout), group in test.groupby(["seed", "held_out_station"], sort=True):
        group = group.copy()
        source = valid.loc[valid["seed"].eq(seed) & valid["held_out_station"].eq(heldout)].copy()
        source_observed = pd.to_numeric(source["effective_observed_value"], errors="coerce")
        source_clean = source["combined_is_attack"].eq(False) & source["predicted_attack"].eq(False)
        fallback = float(source_observed.loc[source_clean].median()) if source_clean.any() else float(source_observed.median())
        if not np.isfinite(fallback):
            raise ValueError(f"non-finite causal fallback for seed={seed}, held_out={heldout}")
        group["offline_temporal_candidate"] = pd.to_numeric(group["temporal_candidate"], errors="coerce")
        group["causal_temporal_candidate"] = causal_candidate(group, fallback)
        attack_alert = group["combined_is_attack"].astype(bool) & group["predicted_attack"].astype(bool)
        clean_alert = ~group["combined_is_attack"].astype(bool) & group["predicted_attack"].astype(bool)
        for candidate in CANDIDATES:
            attack = errors(group, candidate, attack_alert)
            clean = errors(group, candidate, clean_alert)
            rows.append({
                "seed": int(seed),
                "held_out_station": str(heldout),
                "candidate": candidate,
                "causal_fallback_value": fallback,
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
        for station, station_group in group.groupby("station_key", sort=True):
            attack_alert_station = station_group["combined_is_attack"].astype(bool) & station_group["predicted_attack"].astype(bool)
            for candidate in CANDIDATES:
                attack = errors(station_group, candidate, attack_alert_station)
                station_rows.append({"seed": int(seed), "held_out_station": str(heldout), "station_key": str(station), "candidate": candidate, "attack_alert_rows": int(attack_alert_station.sum()), "attack_mae": attack["mae"], "attack_worsening_rate": attack["worsening_rate"]})
    result = pd.DataFrame(rows)
    station = pd.DataFrame(station_rows)
    candidate_summary = result.groupby("candidate", as_index=False).agg(
        combinations=("seed", "size"), attack_alert_rows_mean=("attack_alert_rows", "mean"), clean_alert_rows_mean=("clean_alert_rows", "mean"),
        attack_mae_mean=("attack_mae", "mean"), attack_rmse_mean=("attack_rmse", "mean"), attack_improvement_rate_mean=("attack_improvement_rate", "mean"), attack_worsening_rate_mean=("attack_worsening_rate", "mean"),
        clean_mae_mean=("clean_mae", "mean"), clean_rmse_mean=("clean_rmse", "mean"), clean_worsening_rate_mean=("clean_worsening_rate", "mean"),
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段133读取阶段124五种子×七站点locked test逐行结果，使用同一组合的valid未报警正常观测中位数作为因果回退值，构造只使用当前行之前未报警观测的causal temporal候选，与阶段124双向插值得到的offline temporal候选分层比较；检测预测固定，test只诊断，不重新选择权重或阈值。",
        "source_test": str(test_path),
        "source_valid": str(valid_path),
        "result_count": len(result),
        "station_detail_count": len(station),
        "candidate_summary_count": len(candidate_summary),
        "results": result.to_dict(orient="records"),
        "station_detail": station.to_dict(orient="records"),
        "candidate_summary": candidate_summary.to_dict(orient="records"),
        "limitations": [
            "causal候选以valid-only全局中位数回退，不代表所有在线部署的最优因果估计器。",
            "该阶段是恢复候选诊断，不改变检测指标，也不重新选择恢复权重。",
            "受控攻击不是真实故障日志；双向插值结果只能解释为离线处置。",
        ],
    }
    result.to_csv(TABLES / "stage133_causal_temporal_recovery_detail.csv", index=False, encoding="utf-8-sig")
    station.to_csv(TABLES / "stage133_causal_temporal_recovery_station_detail.csv", index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(TABLES / "stage133_causal_temporal_recovery.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage133_causal_temporal_recovery.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段133因果时间恢复候选审计", "", f"- 状态：**{payload['status']}**", "", "| 候选 | 报警攻击行 | 攻击MAE | 攻击改善率 | 攻击恶化率 | 正常误报警改写MAE | 正常改写恶化率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in candidate_summary.to_dict(orient="records"):
        lines.append(f"| {row['candidate']} | {row['attack_alert_rows_mean']:.1f} | {row['attack_mae_mean']:.4f} | {row['attack_improvement_rate_mean']:.4f} | {row['attack_worsening_rate_mean']:.4f} | {row['clean_mae_mean']:.4f} | {row['clean_worsening_rate_mean']:.4f} |")
    lines += ["", "阶段133只用于区分离线双向插值与只看过去的因果候选，不能把离线恢复结果写成在线能力。"]
    (REPORTS / "stage133_causal_temporal_recovery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "results": len(result), "station_detail": len(station), "candidates": len(candidate_summary)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
