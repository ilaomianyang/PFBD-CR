from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402


def main() -> int:
    df = pd.read_parquet(ROOT / "tables" / "stage09b_local_robust_scores.parquet")
    stations = sorted(df["station_key"].astype(str).unique().tolist())
    rows = []
    for station in stations:
        valid_other = df[(df["split"] == "valid") & (df["station_key"].astype(str) != station)]
        test_station = df[(df["split"] == "test") & (df["station_key"].astype(str) == station)]
        if test_station.empty or valid_other.empty:
            continue
        threshold, valid_f1 = _best_threshold(valid_other["combined_is_attack"].to_numpy(dtype=bool), valid_other["robust_fused_risk"].to_numpy(dtype=float))
        pred = test_station["robust_fused_risk"].to_numpy(dtype=float) >= threshold
        m = _metrics(test_station["combined_is_attack"].to_numpy(dtype=bool), pred, test_station["robust_fused_risk"].to_numpy(dtype=float))
        y = test_station["combined_is_attack"].to_numpy(dtype=bool)
        rows.append({"held_out_station": station, "valid_other_station_rows": len(valid_other), "test_rows": len(test_station), "threshold": threshold, "valid_f1_other_stations": valid_f1, "test_f1": m["f1"], "test_recall": m["recall"], "test_precision": m["precision"], "test_auprc": m["average_precision"], "test_false_alarm_rate": float(pred[~y].mean()) if (~y).any() else float("nan")})
    result = pd.DataFrame(rows)
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    result.to_csv(out_dir / "stage21_leave_one_station_out.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "每次留出一个站点；valid阈值只使用其他站点；test仅评价留出站点。风险分数本身沿用9B全局拟合，故该实验是阈值迁移审计，不等价于完全重新训练的跨站点泛化", "results": result.to_dict(orient="records"), "mean_test_f1": float(result["test_f1"].mean()), "mean_test_recall": float(result["test_recall"].mean()), "mean_test_false_alarm_rate": float(result["test_false_alarm_rate"].mean())}
    (out_dir / "stage21_leave_one_station_out.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段21留一站点阈值迁移", "", "每次留出一个站点，阈值只使用其他站点 valid 数据；该实验是阈值迁移审计，不等价于完全重新训练的跨站点泛化。", "", "| 留出站点 | test F1 | Recall | Precision | AUPRC | 正常误伤率 |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in result.iterrows():
        lines.append(f"| {r['held_out_station']} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_precision']:.4f} | {r['test_auprc']:.4f} | {r['test_false_alarm_rate']:.4f} |")
    lines += ["", f"平均 test F1/Recall/误伤率：`{payload['mean_test_f1']:.4f}` / `{payload['mean_test_recall']:.4f}` / `{payload['mean_test_false_alarm_rate']:.4f}`"]
    report = report_dir / "stage21_leave_one_station_out.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report), "mean_test_f1": payload["mean_test_f1"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
