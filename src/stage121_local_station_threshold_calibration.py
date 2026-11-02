from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
Q_GRID = [0.90, 0.925, 0.95, 0.975, 0.99, 0.995]
SHRINK_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def evaluate(frame: pd.DataFrame, threshold_by_station: pd.Series, global_threshold: float, threshold: float | None = None) -> tuple[dict[str, float], pd.DataFrame]:
    data = frame.copy()
    if threshold is None:
        data["predicted_attack"] = data["robust_fused_risk"].to_numpy(dtype=float) >= data["station_key"].astype(str).map(threshold_by_station).fillna(global_threshold).to_numpy(dtype=float)
    else:
        data["predicted_attack"] = data["robust_fused_risk"].to_numpy(dtype=float) >= threshold
    station_rows: list[dict[str, float | str]] = []
    for station, part in data.groupby("station_key", sort=True):
        y = part["combined_is_attack"].astype(bool).to_numpy()
        p = part["predicted_attack"].astype(bool).to_numpy()
        attack = y
        clean = ~y
        tp = int((p & attack).sum())
        fp = int((p & clean).sum())
        fn = int((~p & attack).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        station_rows.append({"station_key": str(station), "rows": int(len(part)), "attack_rows": int(attack.sum()), "f1": f1, "recall": recall, "precision": precision, "far": float(p[clean].mean()) if clean.any() else np.nan})
    station = pd.DataFrame(station_rows)
    y = data["combined_is_attack"].astype(bool).to_numpy()
    p = data["predicted_attack"].astype(bool).to_numpy()
    clean = ~y
    tp = int((p & y).sum())
    fp = int((p & clean).sum())
    fn = int((~p & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {"micro_f1": f1, "micro_recall": recall, "micro_precision": precision, "micro_far": float(p[clean].mean()) if clean.any() else np.nan, "macro_f1": float(station["f1"].mean()), "macro_recall": float(station["recall"].mean()), "macro_precision": float(station["precision"].mean()), "macro_far": float(station["far"].mean()), "worst_station_recall": float(station["recall"].min()), "worst_station_f1": float(station["f1"].min())}
    return summary, station


def main() -> int:
    candidate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    selected_station_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        score_path = TABLES / f"stage09b_local_robust_scores_seed{seed}.parquet"
        result_path = TABLES / f"stage09b_robust_fusion_seed{seed}.json"
        frame = pd.read_parquet(score_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        train = frame.loc[frame["split"].eq("train")].copy()
        valid = frame.loc[frame["split"].eq("valid")].copy()
        test = frame.loc[frame["split"].eq("test")].copy()
        baseline_threshold = float(result["setting"]["threshold"])
        baseline_valid, _ = evaluate(valid, pd.Series(dtype=float), 0.0, baseline_threshold)
        baseline_test, baseline_station = evaluate(test, pd.Series(dtype=float), 0.0, baseline_threshold)
        baseline_rows.append({"seed": seed, "threshold": baseline_threshold, **{f"baseline_valid_{k}": v for k, v in baseline_valid.items()}, **{f"baseline_test_{k}": v for k, v in baseline_test.items()}, "baseline_test_worst_station_recall": float(baseline_station["recall"].min()), "baseline_test_worst_station_f1": float(baseline_station["f1"].min())})
        normal = train.loc[~train["combined_is_attack"].astype(bool)].copy()
        global_values = pd.to_numeric(normal["robust_fused_risk"], errors="coerce").dropna().to_numpy(dtype=float)
        candidates: list[dict[str, object]] = []
        for q in Q_GRID:
            global_q = float(np.quantile(global_values, q))
            station_q = normal.groupby(normal["station_key"].astype(str))["robust_fused_risk"].quantile(q)
            for shrink in SHRINK_GRID:
                thresholds = shrink * station_q + (1.0 - shrink) * global_q
                valid_summary, _ = evaluate(valid, thresholds, global_q)
                candidate = {"seed": seed, "q": q, "shrink": shrink, "global_threshold": global_q, "valid": valid_summary}
                candidates.append(candidate)
                candidate_rows.append({"seed": seed, "q": q, "shrink": shrink, "global_threshold": global_q, **{f"valid_{k}": v for k, v in valid_summary.items()}})
        best = max(candidates, key=lambda x: (x["valid"]["macro_f1"], -x["valid"]["macro_far"], x["shrink"]))
        q = float(best["q"])
        shrink = float(best["shrink"])
        global_q = float(best["global_threshold"])
        station_q = normal.groupby(normal["station_key"].astype(str))["robust_fused_risk"].quantile(q)
        thresholds = shrink * station_q + (1.0 - shrink) * global_q
        test_summary, station = evaluate(test, thresholds, global_q)
        selected = {"seed": seed, "q": q, "shrink": shrink, "global_threshold": global_q, **{f"valid_{k}": v for k, v in best["valid"].items()}, **{f"test_{k}": v for k, v in test_summary.items()}, "baseline_test_f1": baseline_test["micro_f1"], "baseline_test_macro_f1": baseline_test["macro_f1"], "baseline_test_recall": baseline_test["micro_recall"], "baseline_test_far": baseline_test["micro_far"]}
        selected_rows.append(selected)
        for row in station.to_dict(orient="records"):
            selected_station_rows.append({"seed": seed, "q": q, "shrink": shrink, **row})
        print(json.dumps({"seed": seed, "q": q, "shrink": shrink, "valid_macro_f1": best["valid"]["macro_f1"], "test_macro_f1": test_summary["macro_f1"], "test_worst_recall": test_summary["worst_station_recall"]}, ensure_ascii=False), flush=True)

    selected_df = pd.DataFrame(selected_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    station_df = pd.DataFrame(selected_station_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    summary = {metric: {"mean": float(selected_df[f"test_{metric}"].mean()), "std": float(selected_df[f"test_{metric}"].std(ddof=1)), "min": float(selected_df[f"test_{metric}"].min()), "max": float(selected_df[f"test_{metric}"].max())} for metric in ["micro_f1", "micro_recall", "micro_precision", "micro_far", "macro_f1", "macro_recall", "macro_precision", "macro_far", "worst_station_recall", "worst_station_f1"]}
    baseline_summary = {metric: {"mean": float(baseline_df[f"baseline_test_{metric}"].mean()), "std": float(baseline_df[f"baseline_test_{metric}"].std(ddof=1)), "min": float(baseline_df[f"baseline_test_{metric}"].min()), "max": float(baseline_df[f"baseline_test_{metric}"].max())} for metric in ["micro_f1", "micro_recall", "micro_far", "macro_f1", "worst_station_recall", "worst_station_f1"]}
    gains = {"micro_f1": summary["micro_f1"]["mean"] - baseline_summary["micro_f1"]["mean"], "macro_f1": summary["macro_f1"]["mean"] - baseline_summary["macro_f1"]["mean"], "worst_station_recall": summary["worst_station_recall"]["mean"] - baseline_summary["worst_station_recall"]["mean"]}
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_CROSS_STATION_REVIEW" if gains["macro_f1"] > 0 else "NEGATIVE_RESULT_RETAINED", "protocol": "读取阶段9C五个种子的locked test逐行风险；站点阈值仅由各自train正常样本的q分位数估计，q和站点向全局收缩比例只在valid按站点宏F1选择；test只评价。主模型、权重和攻击构造均不变。", "seeds": SEEDS, "q_grid": Q_GRID, "shrink_grid": SHRINK_GRID, "selected_by_valid": selected_rows, "candidate_count": len(candidate_rows), "summary": summary, "baseline_summary": baseline_summary, "gains_vs_baseline": gains, "station_detail": selected_station_rows, "limitations": ["当前本地仅7个站点，站点宏F1优化可能牺牲微平均或其他站点，必须并列报告。", "该候选只在本地9C协议验证，不能替代已有外部NOAA负结果。", "通过valid选择不等于跨站点冷启动泛化；不能根据test结果把候选固化为主模型。"]}
    candidate_df.to_csv(TABLES / "stage121_local_station_threshold_calibration_candidates.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage121_local_station_threshold_calibration_selected.csv", index=False, encoding="utf-8-sig")
    station_df.to_csv(TABLES / "stage121_local_station_threshold_calibration_station_detail.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage121_local_station_threshold_calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段121本地站点阈值校准", "", "站点阈值只由train正常样本估计，q和收缩比例只由valid站点宏F1选择，test只评价。", "", f"- 状态：**{payload['status']}**", f"- test微F1：候选 `{summary['micro_f1']['mean']:.4f}`；基线 `{baseline_summary['micro_f1']['mean']:.4f}`；差值 `{gains['micro_f1']:+.4f}`", f"- test宏F1：候选 `{summary['macro_f1']['mean']:.4f}`；基线 `{baseline_summary['macro_f1']['mean']:.4f}`；差值 `{gains['macro_f1']:+.4f}`", f"- 最弱站点Recall均值：候选 `{summary['worst_station_recall']['mean']:.4f}`；基线 `{baseline_summary['worst_station_recall']['mean']:.4f}`；差值 `{gains['worst_station_recall']:+.4f}`（均为逐站点最小Recall）", "", "| seed | q | 收缩比例 | test微F1 | test宏F1 | test微Recall | test微FAR | 最弱站点Recall |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in selected_rows:
        lines.append(f"| {row['seed']} | {row['q']:.3f} | {row['shrink']:.2f} | {row['test_micro_f1']:.4f} | {row['test_macro_f1']:.4f} | {row['test_micro_recall']:.4f} | {row['test_micro_far']:.4f} | {row['test_worst_station_recall']:.4f} |")
    lines += ["", "该候选不自动替换9B主模型；必须结合阶段120站点异质性、留一站点和外部NOAA结果解释。"]
    (REPORTS / "stage121_local_station_threshold_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "seeds": len(SEEDS), "candidates": len(candidate_df), "stations": int(station_df["station_key"].nunique())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
