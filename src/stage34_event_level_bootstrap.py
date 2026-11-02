from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段34：事件级召回与bootstrap")
    parser.add_argument("--source", type=Path, default=TABLES / "stage09b_test_recovery.parquet")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    df = pd.read_parquet(args.source).copy()
    attack = df[df["combined_is_attack"].astype(bool)].copy()
    if attack.empty:
        raise ValueError("no attack rows")
    attack["predicted_attack"] = attack["predicted_attack"].astype(bool)
    attack["event_id"] = attack["combined_event_id"].astype(str)
    event_rows = []
    for event_id, g in attack.groupby("event_id", sort=True):
        pred = g["predicted_attack"].to_numpy(dtype=bool)
        risk = pd.to_numeric(g["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
        event_rows.append({
            "event_id": event_id,
            "station": str(g["station_key"].iloc[0]),
            "attack_type": str(g["combined_attack_type"].iloc[0]),
            "rows": int(len(g)),
            "detected_event": bool(pred.any()),
            "row_recall": float(pred.mean()),
            "mean_risk": float(np.nanmean(risk)),
            "max_risk": float(np.nanmax(risk)),
        })
    events = pd.DataFrame(event_rows)
    rng = np.random.default_rng(args.seed)
    n = len(events)
    boot = np.empty((args.iterations, 3), dtype=float)
    vals = events[["detected_event", "row_recall", "rows"]].to_numpy(dtype=float)
    for i in range(args.iterations):
        idx = rng.integers(0, n, size=n)
        sample = vals[idx]
        boot[i, 0] = float(sample[:, 0].mean())
        boot[i, 1] = float(np.average(sample[:, 1], weights=sample[:, 2]))
        boot[i, 2] = float(np.mean(sample[:, 2]))
    type_rows = []
    for attack_type, g in events.groupby("attack_type", sort=True):
        type_rows.append({
            "attack_type": attack_type,
            "events": int(len(g)),
            "rows": int(g["rows"].sum()),
            "detected_events": int(g["detected_event"].sum()),
            "event_recall": float(g["detected_event"].mean()),
            "mean_event_row_recall": float(g["row_recall"].mean()),
            "row_recall_weighted": float(np.average(g["row_recall"], weights=g["rows"])),
            "event_size_median": float(g["rows"].median()),
            "event_size_max": int(g["rows"].max()),
        })
    def ci(col: int) -> list[float]:
        return [float(np.quantile(boot[:, col], 0.025)), float(np.quantile(boot[:, col], 0.975))]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "只使用锁定test的事件标签和预测做描述性事件级统计；事件按combined_event_id聚合，不重新调参。bootstrap以事件为重采样单位。",
        "source": str(args.source),
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "events": int(n),
        "rows": int(len(attack)),
        "single_row_events": int((events["rows"] == 1).sum()),
        "multi_row_events": int((events["rows"] > 1).sum()),
        "max_event_rows": int(events["rows"].max()),
        "overall": {
            "event_recall": float(events["detected_event"].mean()),
            "event_recall_ci95": ci(0),
            "row_recall_weighted": float(np.average(events["row_recall"], weights=events["rows"])),
            "row_recall_weighted_ci95": ci(1),
            "mean_event_size": float(events["rows"].mean()),
            "mean_event_size_ci95": ci(2),
        },
        "by_type": type_rows,
    }
    events.to_csv(TABLES / "stage34_event_level_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(type_rows).to_csv(TABLES / "stage34_event_level_by_type.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage34_event_level_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段34事件级召回与bootstrap", "", "事件按 combined_event_id 聚合；bootstrap 以事件为单位，不重新调参。", "", "| 指标 | 数值 | CI95% |", "|---|---:|---|", f"| 事件数 | {n} | - |", f"| 单行事件/多行事件 | {payload['single_row_events']} / {payload['multi_row_events']} | - |", f"| 事件召回 | {payload['overall']['event_recall']:.4f} | [{payload['overall']['event_recall_ci95'][0]:.4f}, {payload['overall']['event_recall_ci95'][1]:.4f}] |", f"| 按行数加权行召回 | {payload['overall']['row_recall_weighted']:.4f} | [{payload['overall']['row_recall_weighted_ci95'][0]:.4f}, {payload['overall']['row_recall_weighted_ci95'][1]:.4f}] |", "", "## 按攻击类型", "", "| 攻击类型 | 事件数 | 行数 | 事件召回 | 事件内平均行召回 | 加权行召回 | 事件大小中位数 | 最大事件行数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in type_rows:
        lines.append(f"| {r['attack_type']} | {r['events']} | {r['rows']} | {r['event_recall']:.4f} | {r['mean_event_row_recall']:.4f} | {r['row_recall_weighted']:.4f} | {r['event_size_median']:.1f} | {r['event_size_max']} |")
    lines += ["", "当前测试集事件中绝大多数为单行事件，事件级与逐行召回高度相关；该粒度事实应在论文中明确。"]
    (REPORTS / "stage34_event_level_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "events": n, "event_recall": payload["overall"]["event_recall"], "ci95": payload["overall"]["event_recall_ci95"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
