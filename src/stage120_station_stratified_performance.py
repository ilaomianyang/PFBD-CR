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
SEEDS = [3407, 6257, 7919, 104729, 20260715]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from stage09_trust_pipeline import _event_metrics, _metrics  # noqa: E402


def main() -> int:
    station_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        score_path = TABLES / f"stage09b_local_robust_scores_seed{seed}.parquet"
        result_path = TABLES / f"stage09b_robust_fusion_seed{seed}.json"
        if not score_path.is_file() or not result_path.is_file():
            raise FileNotFoundError(f"missing stage9C artifact for seed={seed}")
        frame = pd.read_parquet(score_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        threshold = float(result["setting"]["threshold"])
        test = frame.loc[frame["split"].eq("test")].copy()
        test["predicted_attack"] = test["robust_fused_risk"].to_numpy(dtype=float) >= threshold
        y = test["combined_is_attack"].astype(bool).to_numpy()
        p = test["predicted_attack"].astype(bool).to_numpy()
        s = pd.to_numeric(test["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
        overall = _metrics(y, p, s)
        overall_event = _event_metrics(test, p)
        clean = ~y
        station_groups = [group for _, group in test.groupby("station_key", sort=True)]
        station_metrics = [_metrics(group["combined_is_attack"], group["predicted_attack"], group["robust_fused_risk"]) for group in station_groups]
        station_fars = [float(group.loc[~group["combined_is_attack"].astype(bool), "predicted_attack"].mean()) for group in station_groups if (~group["combined_is_attack"].astype(bool)).any()]
        seed_rows.append({"seed": seed, "stations": int(test["station_key"].nunique()), "rows": int(len(test)), "attack_rows": int(y.sum()), "macro_f1": float(np.mean([item["f1"] for item in station_metrics])), "macro_recall": float(np.mean([item["recall"] for item in station_metrics])), "macro_precision": float(np.mean([item["precision"] for item in station_metrics])), "macro_far": float(np.mean(station_fars)) if station_fars else float("nan"), "micro_f1": float(overall["f1"]), "micro_recall": float(overall["recall"]), "micro_auprc": float(overall["average_precision"]), "micro_far": float(p[clean].mean()), "event_recall": float(overall_event["event_recall"])})
        for station, part in test.groupby("station_key", sort=True):
            y_station = part["combined_is_attack"].astype(bool).to_numpy()
            p_station = part["predicted_attack"].astype(bool).to_numpy()
            score_station = pd.to_numeric(part["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
            metrics = _metrics(y_station, p_station, score_station)
            clean_station = ~y_station
            event = _event_metrics(part.reset_index(drop=True), p_station)
            station_rows.append({"seed": seed, "station_key": str(station), "rows": int(len(part)), "attack_rows": int(y_station.sum()), "clean_rows": int(clean_station.sum()), "f1": float(metrics["f1"]), "recall": float(metrics["recall"]), "precision": float(metrics["precision"]), "auprc": float(metrics["average_precision"]), "far": float(p_station[clean_station].mean()) if clean_station.any() else float("nan"), "event_recall": float(event["event_recall"]), "attack_events": int(event["events"]), "detected_events": int(event["detected_events"])})

    detail = pd.DataFrame(station_rows)
    seed_summary = pd.DataFrame(seed_rows)
    station_aggregate = detail.groupby("station_key", as_index=False).agg(seeds=("seed", "nunique"), rows_mean=("rows", "mean"), attack_rows_mean=("attack_rows", "mean"), f1_mean=("f1", "mean"), f1_std=("f1", "std"), recall_mean=("recall", "mean"), recall_std=("recall", "std"), precision_mean=("precision", "mean"), auprc_mean=("auprc", "mean"), far_mean=("far", "mean"), far_std=("far", "std"), event_recall_mean=("event_recall", "mean"), event_recall_std=("event_recall", "std"))
    required_stations = sorted(detail["station_key"].unique().tolist())
    if len(detail) != len(SEEDS) * len(required_stations) or (detail.groupby("station_key")["seed"].nunique() != len(SEEDS)).any():
        raise ValueError("incomplete seed×station coverage")

    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "读取阶段9C五个预先固定受控攻击种子的完整locked test逐行得分；按站点分别计算F1、Recall、Precision、AUPRC、正常误伤率和事件召回，同时报告站点宏平均、微平均和每站点跨种子波动；不重新训练、不重新选择模型、权重或阈值。该阶段用于检验性能是否被少数站点主导。", "source_scores": [str(TABLES / f"stage09b_local_robust_scores_seed{seed}.parquet") for seed in SEEDS], "seeds": SEEDS, "stations": required_stations, "station_detail": detail.to_dict(orient="records"), "station_aggregate": station_aggregate.to_dict(orient="records"), "seed_summary": seed_summary.to_dict(orient="records"), "limitations": ["站点数量为当前本地数据中的7个站点，不能替代更多独立站点。", "站点宏平均与微平均回答不同问题，不能挑选其一掩盖站点异质性。", "攻击仍为透明受控注入，不是真实网络攻击日志。"]}
    detail.to_csv(TABLES / "stage120_station_stratified_performance_detail.csv", index=False, encoding="utf-8-sig")
    station_aggregate.to_csv(TABLES / "stage120_station_stratified_performance.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(TABLES / "stage120_station_stratified_performance_seed_summary.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage120_station_stratified_performance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段120站点分层性能审计", "", "读取阶段9C五个种子的locked test逐行得分，检查模型性能是否由少数站点主导。", "", "| 站点 | 种子数 | F1均值±标准差 | Recall均值±标准差 | AUPRC均值 | 正常误伤率均值±标准差 | 事件召回均值±标准差 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in station_aggregate.to_dict(orient="records"):
        lines.append(f"| {row['station_key']} | {row['seeds']} | {row['f1_mean']:.4f}±{row['f1_std']:.4f} | {row['recall_mean']:.4f}±{row['recall_std']:.4f} | {row['auprc_mean']:.4f} | {row['far_mean']:.4f}±{row['far_std']:.4f} | {row['event_recall_mean']:.4f}±{row['event_recall_std']:.4f} |")
    lines += ["", "站点宏平均和微平均均已保存到seed_summary；站点差异不能被隐藏或解释为普适性证明。"]
    (REPORTS / "stage120_station_stratified_performance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "station_rows": len(detail), "stations": len(required_stations), "seeds": len(SEEDS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
