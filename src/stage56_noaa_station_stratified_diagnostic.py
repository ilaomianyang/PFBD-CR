from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402


def main() -> int:
    source = pd.read_parquet(NOAA)
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        frame = build_dataset(source, seed)
        scored, result = score(frame)
        test = scored["split"].eq("test")
        threshold = float(result["setting"]["threshold"])
        score_values = scored["robust_fused_risk"].to_numpy(dtype=float)
        pred = score_values >= threshold
        for station, group in scored.loc[test].groupby("station_key", sort=True):
            idx = group.index.to_numpy()
            y = scored.loc[idx, "combined_is_attack"].astype(bool).to_numpy()
            p = pred[idx]
            s = score_values[idx]
            normal = ~y
            rows.append({
                "seed": seed,
                "station_key": str(station),
                "rows": int(len(idx)),
                "attack_rows": int(y.sum()),
                "test_f1": float(f1_score(y, p, zero_division=0)),
                "test_recall": float(recall_score(y, p, zero_division=0)),
                "test_precision": float(precision_score(y, p, zero_division=0)),
                "test_auprc": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan"),
                "test_far": float(p[normal].mean()) if normal.any() else float("nan"),
                "quality_flag_rate": float(pd.to_numeric(scored.loc[idx, "temperature_quality"], errors="coerce").fillna(9).ne(1).mean()),
                "missing_rate": float(scored.loc[idx, "observed_value"].isna().mean()),
            })
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("station_key", sort=True)[["test_f1", "test_recall", "test_precision", "test_auprc", "test_far", "quality_flag_rate", "missing_rate"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["station_key"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "12 NOAA站点、5种子；仅按站点分层统计同一受控攻击和固定时间切分，不重新调参。", "seeds": SEEDS, "detail": rows, "aggregate": aggregate.to_dict(orient="records")}
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage56_noaa_station_stratified_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage56_noaa_station_stratified.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage56_noaa_station_stratified.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段56 NOAA站点分层诊断", "", "按站点报告受控攻击结果与自然观测质量字段；不重新调参。", "", "| 站点 | F1 | Recall | Precision | AUPRC | FAR | 质量标记率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['station_key']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} | {row['quality_flag_rate_mean']:.4f} |")
    lines += ["", "该诊断用于识别站点异质性来源；站点差异不能通过删除弱站点处理，后续优化必须在预先定义的分层协议中复核。"]
    (REPORTS / "stage56_noaa_station_stratified.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": int(detail["station_key"].nunique()), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
