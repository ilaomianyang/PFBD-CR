from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

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
    type_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        frame = build_dataset(source, seed)
        for mode, adaptive in [("global_physical_scale", False), ("station_adaptive_physical_scale", True)]:
            scored, result = score(frame, physical_station_adaptive=adaptive)
            rows.append({
                "seed": seed,
                "mode": mode,
                "valid_f1": result["valid"]["f1"],
                "test_f1": result["test"]["f1"],
                "test_recall": result["test"]["recall"],
                "test_precision": result["test"]["precision"],
                "test_auprc": result["test"]["average_precision"],
                "test_far": result["test_false_alarm_rate"],
                "threshold": result["setting"]["threshold"],
            })
            for item in result["attack_type_test"]:
                type_rows.append({"seed": seed, "mode": mode, **item})
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("mode", sort=False)[["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["mode"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "NOAA 12站点、5种子、同一受控攻击和时间切分；仅把Physical Rules的变化率尺度改为站点自适应，默认全局版本保留；valid选择，test评价。",
        "seeds": SEEDS,
        "detail": rows,
        "aggregate": aggregate.to_dict(orient="records"),
        "by_type": type_rows,
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage54_noaa_station_adaptive_physical_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage54_noaa_station_adaptive_physical.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(type_rows).to_csv(TABLES / "stage54_noaa_station_adaptive_physical_by_type.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage54_noaa_station_adaptive_physical.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段54 NOAA站点自适应物理规则尺度",
        "",
        "仅改变Physical Rules变化率尺度；global版本作为参考，站点自适应版本独立评估。",
        "",
        "| 版本 | test F1 | Recall | Precision | AUPRC | FAR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['mode']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} |")
    lines += ["", "若站点自适应版本改善，仍需在滚动季节和严格留一站点设置中复核，不能直接替换主模型。"]
    (REPORTS / "stage54_noaa_station_adaptive_physical.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "aggregate": payload["aggregate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
