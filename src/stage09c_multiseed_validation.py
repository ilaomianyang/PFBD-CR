from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage05_attack_extensions import DEFAULT_DATA, build_extensions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段9C：不同受控攻击种子下的重复验证")
    parser.add_argument("--seeds", default="3407,6257,7919,104729,20260715")
    parser.add_argument("--noaa", type=Path, default=ROOT / "tables" / "stage06_noaa_isd_normalized.parquet")
    parser.add_argument("--era5", type=Path, default=ROOT / "tables" / "stage06_era5_station_reference.parquet")
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    base = pd.read_parquet(DEFAULT_DATA)
    rows = []
    for seed in seeds:
        tag = f"seed{seed}"
        input_path = ROOT / "tables" / f"stage09c_input_{tag}.parquet"
        extended, events = build_extensions(base, seed)
        extended.to_parquet(input_path, index=False)
        cmd = [sys.executable, str(SRC / "stage09b_robust_fusion.py"), "--local", str(input_path), "--noaa", str(args.noaa), "--era5", str(args.era5), "--tag", tag]
        result = subprocess.run(cmd, cwd=ROOT.parents[1], capture_output=True, text=True)
        result_path = ROOT / "tables" / f"stage09b_robust_fusion_{tag}.json"
        if result.returncode != 0 or not result_path.exists():
            raise RuntimeError(f"seed {seed} failed: {result.stdout[-1000:]}\n{result.stderr[-1000:]}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        test = payload["local_test"]
        rows.append({"seed": seed, "extension_events": int(len(events)), "test_f1": test["f1"], "test_recall": test["recall"], "test_precision": test["precision"], "test_auprc": test["average_precision"], "test_false_alarm_rate": payload["test_false_alarm_rate_on_clean_rows"], "event_recall": payload["test_event_metrics"]["event_recall"], "recovery_mae_after": payload["test_recovery"]["mae_after"], "returncode": result.returncode})
        print(json.dumps({"seed": seed, "status": "completed", "test_f1": test["f1"], "test_recall": test["recall"]}, ensure_ascii=False))
    detail = pd.DataFrame(rows)
    numeric = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate", "event_recall", "recovery_mae_after"]
    summary = {"seeds": seeds, "runs": int(len(detail)), "metrics": {col: {"mean": float(detail[col].mean()), "std": float(detail[col].std(ddof=1)) if len(detail) > 1 else 0.0, "min": float(detail[col].min()), "max": float(detail[col].max())} for col in numeric}}
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    detail.to_csv(out_dir / "stage09c_multiseed_detail.csv", index=False, encoding="utf-8-sig")
    (out_dir / "stage09c_multiseed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段9C多种子重复验证", "", "- 每个种子重新生成阶段5受控攻击扩展，并重新运行阶段9B。", "- 这些种子反映受控攻击构造不确定性，不应冒充独立真实网络入侵样本。", "", "| 指标 | 均值 | 标准差 | 最小值 | 最大值 |", "|---|---:|---:|---:|---:|"]
    labels = {"test_f1": "test F1", "test_recall": "test Recall", "test_precision": "test Precision", "test_auprc": "test AUPRC", "test_false_alarm_rate": "正常样本误伤率", "event_recall": "事件级召回", "recovery_mae_after": "恢复后 MAE"}
    for col in numeric:
        m = summary["metrics"][col]
        lines.append(f"| {labels[col]} | {m['mean']:.4f} | {m['std']:.4f} | {m['min']:.4f} | {m['max']:.4f} |")
    report = report_dir / "stage09c_multiseed_validation.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_PIPELINE_REVIEW_REQUIRED", "report": str(report), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
