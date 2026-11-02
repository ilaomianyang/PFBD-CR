from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402

TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]


def main() -> int:
    source = pd.read_parquet(NOAA)
    summaries = []
    type_rows = []
    integrity = []
    for seed in SEEDS:
        frame = build_dataset(source, seed)
        train_attack_rows = int(frame.loc[frame["split"].eq("train"), "combined_is_attack"].sum())
        integrity.append({"seed": seed, "train_attack_rows": train_attack_rows, "rows": int(len(frame)), "stations": int(frame["station_key"].nunique())})
        _, result = score(frame)
        result["seed"] = seed
        summaries.append({"seed": seed, "test_f1": result["test"]["f1"], "test_recall": result["test"]["recall"], "test_precision": result["test"]["precision"], "test_auprc": result["test"]["average_precision"], "test_false_alarm_rate": result["test_false_alarm_rate"], "valid_f1": result["valid"]["f1"]})
        for row in result["attack_type_test"]:
            type_rows.append({"seed": seed, **row})
    summary_df = pd.DataFrame(summaries)
    type_df = pd.DataFrame(type_rows)
    aggregate = {metric: {"mean": float(summary_df[metric].mean()), "std": float(summary_df[metric].std(ddof=1)), "min": float(summary_df[metric].min()), "max": float(summary_df[metric].max())} for metric in ["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]}
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA真实序列固定时间切分；5个独立攻击注入种子；train始终无攻击；每个种子独立valid选择和locked test评价。攻击为受控注入，不是NOAA真实攻击标签。", "seeds": SEEDS, "aggregate": aggregate, "per_seed": summaries, "by_type": type_rows, "integrity": integrity}
    summary_df.to_csv(TABLES / "stage40_noaa_controlled_multiseed.csv", index=False, encoding="utf-8-sig")
    type_df.to_csv(TABLES / "stage40_noaa_controlled_multiseed_by_type.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage40_noaa_controlled_multiseed.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    a = aggregate
    lines = ["# 阶段40 NOAA独立受控基准多种子验证", "", "攻击注入仅用于可审计的跨数据源协议验证，不代表真实网络攻击。", f"- 种子数：`{len(SEEDS)}`；训练集攻击行：`{sorted(set(x['train_attack_rows'] for x in integrity))}`", "", "## 汇总", "", "| 指标 | 均值 | 标准差 | 最小值 | 最大值 |", "|---|---:|---:|---:|---:|"]
    for metric, label in [("test_f1", "test F1"), ("test_recall", "test Recall"), ("test_precision", "test Precision"), ("test_auprc", "test AUPRC"), ("test_false_alarm_rate", "test FAR")]:
        x = a[metric]
        lines.append(f"| {label} | {x['mean']:.4f} | {x['std']:.4f} | {x['min']:.4f} | {x['max']:.4f} |")
    lines += ["", "## 每种子结果", "", "| seed | valid F1 | test F1 | Recall | Precision | AUPRC | FAR |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['seed']} | {r['valid_f1']:.4f} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_precision']:.4f} | {r['test_auprc']:.4f} | {r['test_false_alarm_rate']:.4f} |")
    lines += ["", "完整攻击类型和完整性表见 `tables/stage40_noaa_controlled_multiseed_by_type.csv` 与 JSON。"]
    (REPORTS / "stage40_noaa_controlled_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "seeds": len(SEEDS), "test_f1_mean": a["test_f1"]["mean"], "test_f1_std": a["test_f1"]["std"], "test_far_mean": a["test_false_alarm_rate"]["mean"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
