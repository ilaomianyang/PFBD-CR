from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stage22_leave_one_station_refit import DEFAULT_ERA5, DEFAULT_LOCAL, evaluate_holdout

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段23：无标签站点鲁棒校准")
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    args = parser.parse_args()
    source = pd.read_parquet(args.local)
    stations = sorted(source["station_key"].astype(str).unique().tolist())
    results = []
    detail_frames = []
    for i, station in enumerate(stations, 1):
        print(json.dumps({"progress": f"{i}/{len(stations)}", "held_out_station": station}, ensure_ascii=False), flush=True)
        current = evaluate_holdout(source, args.era5, station, calibrate_site=True)
        detail_frames.append(current.pop("detail"))
        results.append(current)

    rows = []
    for r in results:
        m = r["test"]
        c = r["site_calibration"]
        rows.append({
            "held_out_station": r["held_out_station"],
            "test_f1": m["f1"],
            "test_recall": m["recall"],
            "test_precision": m["precision"],
            "test_auprc": m["average_precision"],
            "test_false_alarm_rate": r["test_false_alarm_rate"],
            "recovery_mae_before": r["test_recovery"]["mae_before"],
            "recovery_mae_after": r["test_recovery"]["mae_after"],
            "event_recall": r["test_event_metrics"]["event_recall"],
            "source_threshold": c["source_threshold"],
            "site_threshold": c["site_threshold"],
            "site_train_risk_median": c["site_train_risk_median"],
            "site_train_risk_scale": c["site_train_risk_scale"],
        })
    result = pd.DataFrame(rows)
    detail = pd.concat(detail_frames, ignore_index=True)
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "stage23_unlabeled_site_calibration.csv", index=False, encoding="utf-8-sig")
    detail.to_parquet(out_dir / "stage23_unlabeled_site_calibration_detail.parquet", index=False)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "检测器、ERA5偏差、融合设置和源阈值只由其余站点拟合；留出站点仅使用无标签train风险的中位数/MAD做尺度迁移；test标签只用于最终评价。",
        "source_local": str(args.local),
        "source_era5": str(args.era5),
        "results": results,
        "summary": {
            "mean_test_f1": float(result["test_f1"].mean()),
            "std_test_f1": float(result["test_f1"].std(ddof=1)),
            "mean_test_recall": float(result["test_recall"].mean()),
            "mean_test_auprc": float(result["test_auprc"].mean()),
            "mean_test_false_alarm_rate": float(result["test_false_alarm_rate"].mean()),
            "mean_recovery_mae_before": float(result["recovery_mae_before"].mean()),
            "mean_recovery_mae_after": float(result["recovery_mae_after"].mean()),
            "mean_event_recall": float(result["event_recall"].mean()),
        },
    }
    (out_dir / "stage23_unlabeled_site_calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段23无标签站点鲁棒校准",
        "",
        "校准不读取留出站点的 train 标签，只用其历史风险分布的中位数/MAD，把其余站点验证集确定的阈值映射到该站点。该实验表示‘新站点有无标签历史数据’的部署条件，不是冷启动结果。",
        "",
        "| 留出站点 | F1 | Recall | AUPRC | 误伤率 | 阈值源→站点 | 恢复MAE前→后 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result.iterrows():
        lines.append(f"| {row['held_out_station']} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | {row['test_auprc']:.4f} | {row['test_false_alarm_rate']:.4f} | {row['source_threshold']:.4f}→{row['site_threshold']:.4f} | {row['recovery_mae_before']:.4f}→{row['recovery_mae_after']:.4f} |")
    s = payload["summary"]
    lines += [
        "",
        f"平均±标准差 F1：`{s['mean_test_f1']:.4f} ± {s['std_test_f1']:.4f}`",
        f"平均 Recall/AUPRC/误伤率：`{s['mean_test_recall']:.4f}` / `{s['mean_test_auprc']:.4f}` / `{s['mean_test_false_alarm_rate']:.4f}`",
        f"平均恢复 MAE：`{s['mean_recovery_mae_before']:.4f} → {s['mean_recovery_mae_after']:.4f}`；平均事件召回：`{s['mean_event_recall']:.4f}`",
        "",
        "结果必须与阶段22冷启动严格重拟合并列报告；若站点历史数据含较多异常，鲁棒校准仍可能失效，不能将其写成无条件跨域保证。",
    ]
    report = report_dir / "stage23_unlabeled_site_calibration.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report), "summary": payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
