from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
FAR_CAPS = [0.01, 0.02, 0.05, 0.10]
NORMAL_QUANTILES = np.unique(np.r_[np.linspace(0.50, 0.999, 80), 0.9995, 0.9999])


def evaluate(frame: pd.DataFrame, threshold: float, stations: list[str], include_auprc: bool = True) -> dict[str, float]:
    y = frame["combined_is_attack"].astype(bool).to_numpy()
    score = pd.to_numeric(frame["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
    pred = score >= float(threshold)
    clean = ~y
    tp = int((pred & y).sum())
    fp = int((pred & clean).sum())
    fn = int((~pred & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    station_f1: list[float] = []
    station_far: list[float] = []
    for station in stations:
        part = frame.loc[frame["station_key"].astype(str).eq(station)]
        if not len(part):
            continue
        m = evaluate(part, threshold, [], include_auprc=False) if len(stations) else None
        if m is not None:
            station_f1.append(m["micro_f1"])
            station_far.append(m["micro_far"])
    return {
        "micro_f1": float(f1),
        "micro_recall": float(recall),
        "micro_precision": float(precision),
        "micro_far": float(pred[clean].mean()) if clean.any() else float("nan"),
        "micro_auprc": float(average_precision_score(y, score)) if include_auprc and y.any() and clean.any() else float("nan"),
        "macro_f1": float(np.mean(station_f1)) if station_f1 else 0.0,
        "macro_far": float(np.mean(station_far)) if station_far else float("nan"),
    }


def choose_threshold(valid: pd.DataFrame, source_stations: list[str], far_cap: float) -> tuple[float, dict[str, float], int, int, bool]:
    normal = valid.loc[~valid["combined_is_attack"].astype(bool), "robust_fused_risk"]
    normal = pd.to_numeric(normal, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(normal) == 0:
        threshold = 0.0
        return threshold, evaluate(valid, threshold, source_stations), 0, 0, True
    thresholds = np.unique(np.quantile(normal, NORMAL_QUANTILES))
    feasible: list[tuple[float, dict[str, float], float]] = []
    for threshold in thresholds:
        metrics = evaluate(valid, float(threshold), source_stations, include_auprc=False)
        if np.isfinite(metrics["macro_far"]) and metrics["macro_far"] <= far_cap + 1e-12:
            feasible.append((float(threshold), metrics, float(threshold)))
    if feasible:
        threshold, metrics, _ = max(feasible, key=lambda item: (item[1]["macro_f1"], -item[1]["macro_far"], item[2]))
        return threshold, metrics, len(thresholds), len(feasible), False
    threshold = float(thresholds[-1])
    return threshold, evaluate(valid, threshold, source_stations, include_auprc=False), len(thresholds), 0, True


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段126：源站点valid低误报运行点与留出站点test评价")
    parser.add_argument("--far-caps", nargs="+", type=float, default=FAR_CAPS)
    args = parser.parse_args()
    far_caps = sorted(set(float(value) for value in args.far_caps))
    if not far_caps or any(value <= 0 or value > 1 for value in far_caps):
        raise ValueError("far caps must be in (0, 1]")
    valid_path = TABLES / "stage124_leave_one_station_refit_multiseed_valid_detail.parquet"
    test_path = TABLES / "stage124_leave_one_station_refit_multiseed_detail.parquet"
    summary_path = TABLES / "stage124_leave_one_station_refit_multiseed.csv"
    for path in [valid_path, test_path, summary_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    valid = pd.read_parquet(valid_path)
    test = pd.read_parquet(test_path)
    stage124 = pd.read_csv(summary_path)
    selected_rows: list[dict[str, object]] = []
    for _, baseline_row in stage124.iterrows():
        seed = int(baseline_row["seed"])
        heldout = str(baseline_row["held_out_station"])
        valid_group = valid.loc[(valid["seed"] == seed) & (valid["held_out_station"].astype(str) == heldout)].copy()
        test_group = test.loc[(test["seed"] == seed) & (test["held_out_station"].astype(str) == heldout) & (test["station_key"].astype(str) == heldout)].copy()
        source_stations = sorted(valid_group.loc[valid_group["station_key"].astype(str).ne(heldout), "station_key"].astype(str).unique().tolist())
        if valid_group.empty or test_group.empty or len(source_stations) == 0:
            raise ValueError(f"incomplete source/test group seed={seed}, heldout={heldout}")
        baseline_threshold = float(baseline_row["threshold"])
        baseline_test = evaluate(test_group, baseline_threshold, [heldout])
        for far_cap in far_caps:
            threshold, valid_metrics, candidate_count, feasible_count, fallback = choose_threshold(valid_group, source_stations, far_cap)
            test_metrics = evaluate(test_group, threshold, [heldout])
            selected_rows.append({
                "seed": seed,
                "held_out_station": heldout,
                "far_cap": far_cap,
                "source_station_count": len(source_stations),
                "candidate_threshold_count": candidate_count,
                "feasible_threshold_count": feasible_count,
                "fallback_to_highest_threshold": bool(fallback),
                "selected_threshold": float(threshold),
                "valid_macro_f1": valid_metrics["macro_f1"],
                "valid_macro_far": valid_metrics["macro_far"],
                "valid_micro_f1": valid_metrics["micro_f1"],
                "valid_micro_far": valid_metrics["micro_far"],
                "test_f1": test_metrics["micro_f1"],
                "test_recall": test_metrics["micro_recall"],
                "test_precision": test_metrics["micro_precision"],
                "test_auprc": test_metrics["micro_auprc"],
                "test_false_alarm_rate": test_metrics["micro_far"],
                "baseline_threshold": baseline_threshold,
                "baseline_test_f1": baseline_test["micro_f1"],
                "baseline_test_recall": baseline_test["micro_recall"],
                "baseline_test_false_alarm_rate": baseline_test["micro_far"],
                "test_f1_diff_vs_baseline": test_metrics["micro_f1"] - baseline_test["micro_f1"],
                "test_far_diff_vs_baseline": test_metrics["micro_far"] - baseline_test["micro_far"],
            })
    selected = pd.DataFrame(selected_rows)
    cap_summary = selected.groupby("far_cap", as_index=False).agg(
        combinations=("seed", "size"),
        fallback_count=("fallback_to_highest_threshold", "sum"),
        valid_macro_f1_mean=("valid_macro_f1", "mean"), valid_macro_far_mean=("valid_macro_far", "mean"),
        test_f1_mean=("test_f1", "mean"), test_recall_mean=("test_recall", "mean"), test_precision_mean=("test_precision", "mean"),
        test_far_mean=("test_false_alarm_rate", "mean"), baseline_test_f1_mean=("baseline_test_f1", "mean"), baseline_test_far_mean=("baseline_test_false_alarm_rate", "mean"),
        f1_diff_mean=("test_f1_diff_vs_baseline", "mean"), far_diff_mean=("test_far_diff_vs_baseline", "mean"),
    )
    station_summary = selected.groupby(["far_cap", "held_out_station"], as_index=False).agg(
        seeds=("seed", "nunique"), test_f1_mean=("test_f1", "mean"), test_recall_mean=("test_recall", "mean"),
        test_far_mean=("test_false_alarm_rate", "mean"), baseline_test_f1_mean=("baseline_test_f1", "mean"), baseline_test_far_mean=("baseline_test_false_alarm_rate", "mean"),
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段124模型仅由各源站点train拟合；对阶段124每个种子×留出站点，仅使用源站点valid逐行风险和标签，在预先固定的FAR上限1%/2%/5%/10%下按源站点宏F1选择阈值；留出站点locked test只评价。阶段124的F1最优阈值作为原基线并列报告，不使用test选择运行点。",
        "source_valid": str(valid_path),
        "source_test": str(test_path),
        "far_caps": far_caps,
        "candidate_quantiles": int(len(NORMAL_QUANTILES)),
        "selected_count": len(selected),
        "fallback_count": int(selected["fallback_to_highest_threshold"].sum()),
        "selected": selected.to_dict(orient="records"),
        "cap_summary": cap_summary.to_dict(orient="records"),
        "station_summary": station_summary.to_dict(orient="records"),
        "limitations": [
            "该阶段只改变阈值运行点，不重新拟合阶段124检测器、ERA5偏差、尺度或恢复权重。",
            "test AUPRC为阈值无关指标，原始阶段124 AUPRC仍作为主结果；本表不填充伪造的AUPRC。",
            "FAR约束是在源站点valid宏平均上选择，留出站点test FAR可能超出上限；这正是迁移风险的测量。",
            "攻击标签为受控注入，不能替代真实攻击数据。",
        ],
    }
    selected.to_csv(TABLES / "stage126_low_far_leave_one_station_operating_points_selected.csv", index=False, encoding="utf-8-sig")
    cap_summary.to_csv(TABLES / "stage126_low_far_leave_one_station_operating_points.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(TABLES / "stage126_low_far_leave_one_station_operating_points_station_summary.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage126_low_far_leave_one_station_operating_points.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段126低误报留一站点运行点", "",
        "阈值只在源站点valid按宏F1和FAR上限选择，留出站点locked test仅评价。", "",
        f"- 状态：**{payload['status']}**", f"- 组合数：`{len(selected)}`；回退到最高阈值的组合：`{payload['fallback_count']}`", "",
        "| valid FAR上限 | 组合数 | valid宏F1 | valid宏FAR | test F1 | test Recall | test FAR | 相对基线F1 | 相对基线FAR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cap_summary.to_dict(orient="records"):
        lines.append(f"| {row['far_cap']:.3f} | {row['combinations']} | {row['valid_macro_f1_mean']:.4f} | {row['valid_macro_far_mean']:.4f} | {row['test_f1_mean']:.4f} | {row['test_recall_mean']:.4f} | {row['test_far_mean']:.4f} | {row['f1_diff_mean']:+.4f} | {row['far_diff_mean']:+.4f} |")
    lines += ["", "该表是安全运行点分析，不替换阶段124的F1最优主结果；留出站点FAR偏离valid上限时应作为迁移风险报告。"]
    (REPORTS / "stage126_low_far_leave_one_station_operating_points.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected": len(selected), "caps": len(far_caps), "fallback": payload["fallback_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
