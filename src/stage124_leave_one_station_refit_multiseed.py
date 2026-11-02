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
DEFAULT_ERA5 = TABLES / "stage06_era5_station_reference.parquet"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from stage22_leave_one_station_refit import evaluate_holdout  # noqa: E402


def _summary_row(seed: int, result: dict[str, object]) -> dict[str, object]:
    metrics = result["test"]
    recovery = result["test_recovery"]
    event = result["test_event_metrics"]
    return {
        "seed": seed,
        "held_out_station": result["held_out_station"],
        "fit_train_rows": int(result["fit_train_rows"]),
        "fit_valid_rows": int(result["fit_valid_rows"]),
        "test_rows": int(result["held_out_test_rows"]),
        "threshold": float(result["setting"]["threshold"]),
        "valid_f1_other_stations": float(result["setting"]["valid_f1"]),
        "test_f1": float(metrics["f1"]),
        "test_recall": float(metrics["recall"]),
        "test_precision": float(metrics["precision"]),
        "test_auprc": float(metrics["average_precision"]),
        "test_false_alarm_rate": float(result["test_false_alarm_rate"]),
        "recovery_mae_before": float(recovery["mae_before"]),
        "recovery_mae_after": float(recovery["mae_after"]),
        "recovery_rmse_before": float(recovery["rmse_before"]),
        "recovery_rmse_after": float(recovery["rmse_after"]),
        "event_recall": float(event["event_recall"]),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段124：5种子×7站点严格留一站点重新拟合")
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    args = parser.parse_args()
    input_paths = {seed: TABLES / f"stage09c_input_seed{seed}.parquet" for seed in SEEDS}
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing seed inputs: " + ", ".join(missing))
    if not args.era5.is_file():
        raise FileNotFoundError(args.era5)

    rows: list[dict[str, object]] = []
    public_results: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    valid_detail_frames: list[pd.DataFrame] = []
    station_sets: dict[int, list[str]] = {}
    for seed in SEEDS:
        source = pd.read_parquet(input_paths[seed])
        stations = sorted(source["station_key"].astype(str).dropna().unique().tolist())
        station_sets[seed] = stations
        print(json.dumps({"seed": seed, "stations": len(stations), "input_rows": len(source)}, ensure_ascii=False), flush=True)
        for index, station in enumerate(stations, start=1):
            print(json.dumps({"seed": seed, "progress": f"{index}/{len(stations)}", "held_out_station": station}, ensure_ascii=False), flush=True)
            current = evaluate_holdout(source, args.era5, station)
            detail = current.pop("detail")
            valid_detail = current.pop("valid_detail")
            row = _summary_row(seed, current)
            rows.append(row)
            public_results.append({"seed": seed, **current})
            detail = detail.copy()
            detail.insert(0, "seed", seed)
            detail_frames.append(detail)
            valid_detail = valid_detail.copy()
            valid_detail.insert(0, "seed", seed)
            valid_detail_frames.append(valid_detail)

    station_names = sorted({station for stations in station_sets.values() for station in stations})
    if len(station_sets) != len(SEEDS) or any(station_sets[seed] != station_names for seed in SEEDS):
        raise ValueError(f"inconsistent station sets: {station_sets}")
    result = pd.DataFrame(rows).sort_values(["seed", "held_out_station"], kind="stable").reset_index(drop=True)
    detail = pd.concat(detail_frames, ignore_index=True)
    valid_detail = pd.concat(valid_detail_frames, ignore_index=True)
    station_summary = result.groupby("held_out_station", as_index=False).agg(
        seeds=("seed", "nunique"),
        test_f1_mean=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
        test_recall_mean=("test_recall", "mean"), test_recall_std=("test_recall", "std"),
        test_precision_mean=("test_precision", "mean"), test_auprc_mean=("test_auprc", "mean"),
        test_false_alarm_rate_mean=("test_false_alarm_rate", "mean"), test_false_alarm_rate_std=("test_false_alarm_rate", "std"),
        recovery_mae_before_mean=("recovery_mae_before", "mean"), recovery_mae_after_mean=("recovery_mae_after", "mean"),
        event_recall_mean=("event_recall", "mean"), event_recall_std=("event_recall", "std"),
    )
    seed_summary = result.groupby("seed", as_index=False).agg(
        stations=("held_out_station", "nunique"),
        test_f1_mean=("test_f1", "mean"), test_f1_std_across_stations=("test_f1", "std"),
        test_recall_mean=("test_recall", "mean"), test_precision_mean=("test_precision", "mean"),
        test_auprc_mean=("test_auprc", "mean"), test_false_alarm_rate_mean=("test_false_alarm_rate", "mean"),
        recovery_mae_before_mean=("recovery_mae_before", "mean"), recovery_mae_after_mean=("recovery_mae_after", "mean"),
        event_recall_mean=("event_recall", "mean"),
    )
    metrics = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate", "recovery_mae_before", "recovery_mae_after", "event_recall"]
    summary = {metric: {"mean": float(result[metric].mean()), "std": float(result[metric].std(ddof=1)), "min": float(result[metric].min()), "max": float(result[metric].max())} for metric in metrics}
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "对阶段9C五个预先固定攻击构造种子分别执行阶段22严格留一站点重新拟合；每个种子内逐一留出7个站点，留出站点不参与检测器、ERA5偏差、尺度、融合设置、阈值和恢复权重拟合，valid仅用于源站点设置选择，locked test仅评价。结果不跨种子重新选模型或参数。",
        "seeds": SEEDS,
        "stations": station_names,
        "source_inputs": {str(seed): str(path) for seed, path in input_paths.items()},
        "source_era5": str(args.era5),
        "result_count": len(result),
        "detail_rows": len(detail),
        "valid_detail_rows": len(valid_detail),
        "summary": summary,
        "results": public_results,
        "station_summary": station_summary.to_dict(orient="records"),
        "seed_summary": seed_summary.to_dict(orient="records"),
        "limitations": [
            "该阶段仍使用受控攻击标签，不能替代真实网络攻击日志。",
            "同一气象数据源上的五个攻击构造种子不等于五个独立地区或年份。",
            "跨种子均值和站点差异需配合独立bootstrap解释，不应只报告总体均值。",
        ],
    }
    result.to_csv(TABLES / "stage124_leave_one_station_refit_multiseed.csv", index=False, encoding="utf-8-sig")
    detail.to_parquet(TABLES / "stage124_leave_one_station_refit_multiseed_detail.parquet", index=False)
    valid_detail.to_parquet(TABLES / "stage124_leave_one_station_refit_multiseed_valid_detail.parquet", index=False)
    station_summary.to_csv(TABLES / "stage124_leave_one_station_refit_multiseed_station_summary.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(TABLES / "stage124_leave_one_station_refit_multiseed_seed_summary.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage124_leave_one_station_refit_multiseed.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段124五种子严格留一站点重新拟合", "",
        "每个攻击构造种子均独立执行7站点留出；留出站点不参与任何拟合或验证选择。", "",
        f"- 状态：**{payload['status']}**", f"- 组合数：`{len(result)}`（5种子×7站点）", f"- locked test逐行明细：`{len(detail)}`行；源站点valid逐行明细：`{len(valid_detail)}`行", "",
        "| 指标 | 均值 | 标准差 | 最小值 | 最大值 |", "|---|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        item = summary[metric]
        lines.append(f"| {metric} | {item['mean']:.4f} | {item['std']:.4f} | {item['min']:.4f} | {item['max']:.4f} |")
    lines += ["", "## 站点均值", "", "| 留出站点 | 种子数 | F1 | Recall | AUPRC | FAR | 恢复MAE前→后 | 事件召回 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in station_summary.to_dict(orient="records"):
        lines.append(f"| {row['held_out_station']} | {row['seeds']} | {row['test_f1_mean']:.4f} | {row['test_recall_mean']:.4f} | {row['test_auprc_mean']:.4f} | {row['test_false_alarm_rate_mean']:.4f} | {row['recovery_mae_before_mean']:.4f}→{row['recovery_mae_after_mean']:.4f} | {row['event_recall_mean']:.4f} |")
    lines += ["", "五种子结果应继续进行站点簇和种子级不确定性分析；该阶段不支持普适性或真实攻击结论。"]
    (REPORTS / "stage124_leave_one_station_refit_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "results": len(result), "detail_rows": len(detail), "valid_detail_rows": len(valid_detail), "seeds": len(SEEDS), "stations": len(station_names)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
