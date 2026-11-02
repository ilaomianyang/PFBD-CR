from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def main() -> int:
    seeds = [3407, 6257, 7919, 104729, 20260715]
    rows = []
    for seed in seeds:
        tag = f"seed{seed}"
        local = ROOT / "tables" / f"stage09b_local_robust_scores_{tag}.parquet"
        cmd = [sys.executable, str(SRC / "stage12_hybrid_fusion.py"), "--local", str(local), "--tag", tag]
        result = subprocess.run(cmd, cwd=ROOT.parents[1], capture_output=True, text=True)
        path = ROOT / "tables" / f"stage12_hybrid_fusion_{tag}.json"
        if result.returncode != 0 or not path.exists():
            raise RuntimeError(f"seed {seed} failed: {result.stdout[-1000:]}\n{result.stderr[-1000:]}")
        data = json.loads(path.read_text(encoding="utf-8"))
        test = data["test"]
        rows.append({"seed": seed, "test_f1": test["f1"], "test_recall": test["recall"], "test_precision": test["precision"], "test_auprc": test["average_precision"], "test_false_alarm_rate": data["test_false_alarm_rate"], "event_recall": data["event_metrics"]["event_recall"], "recovery_mae_after": data["recovery"]["mae_after"]})
        print(json.dumps({"seed": seed, "status": "completed", "test_f1": test["f1"]}, ensure_ascii=False))
    detail = pd.DataFrame(rows)
    metrics = [c for c in detail.columns if c != "seed"]
    summary = {"seeds": seeds, "runs": len(detail), "metrics": {c: {"mean": float(detail[c].mean()), "std": float(detail[c].std(ddof=1)), "min": float(detail[c].min()), "max": float(detail[c].max())} for c in metrics}}
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    detail.to_csv(out_dir / "stage12_hybrid_multiseed_detail.csv", index=False, encoding="utf-8-sig")
    (out_dir / "stage12_hybrid_multiseed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段12混合模型多种子验证", "", "混合模型在阶段9C生成的五个受控攻击种子上重复运行。", "", "| 指标 | 均值 | 标准差 | 最小值 | 最大值 |", "|---|---:|---:|---:|---:|"]
    labels = {"test_f1": "test F1", "test_recall": "Recall", "test_precision": "Precision", "test_auprc": "AUPRC", "test_false_alarm_rate": "正常误伤率", "event_recall": "事件级召回", "recovery_mae_after": "恢复后 MAE"}
    for c in metrics:
        x = summary["metrics"][c]
        lines.append(f"| {labels.get(c, c)} | {x['mean']:.4f} | {x['std']:.4f} | {x['min']:.4f} | {x['max']:.4f} |")
    report = report_dir / "stage12_hybrid_multiseed.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_PIPELINE_REVIEW_REQUIRED", "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
