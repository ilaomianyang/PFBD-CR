from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BASE = Path(r"F:\数据集\Analysis\experiments\pc_mstf_ap_v4\parquet\v4_anomaly_injection_table.parquet")


def main() -> int:
    strengths = [0.5, 1.0, 1.5, 2.0]
    rows = []
    source = pd.read_parquet(BASE)
    for strength in strengths:
        tag = f"intensity{str(strength).replace('.', 'p')}"
        input_path = ROOT / "tables" / f"stage15_input_{tag}.parquet"
        # Import inside the script path so this command remains directly runnable.
        sys.path.insert(0, str(SRC))
        from stage05_attack_extensions import build_extensions
        extended, events = build_extensions(source, 20260911, strength)
        extended.to_parquet(input_path, index=False)
        cmd = [sys.executable, str(SRC / "stage09b_robust_fusion.py"), "--local", str(input_path), "--tag", tag]
        result = subprocess.run(cmd, cwd=ROOT.parents[1], capture_output=True, text=True)
        result_path = ROOT / "tables" / f"stage09b_robust_fusion_{tag}.json"
        if result.returncode != 0 or not result_path.exists():
            raise RuntimeError(f"strength {strength} failed: {result.stdout[-1000:]}\n{result.stderr[-1000:]}")
        data = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = data["local_test"]
        type_path = ROOT / "tables" / f"stage09b_attack_type_metrics_{tag}.csv"
        type_df = pd.read_csv(type_path)
        ext_types = type_df[type_df["attack_type"].isin(["coordinated_offset", "mixed_missing_attack", "replay"])]
        rows.append({"strength": strength, "extension_events": len(events), "test_f1": metrics["f1"], "test_recall": metrics["recall"], "test_auprc": metrics["average_precision"], "test_false_alarm_rate": data["test_false_alarm_rate_on_clean_rows"], "extension_mean_recall": float(ext_types["recall"].mean()), "coordinated_offset_recall": float(ext_types.loc[ext_types.attack_type.eq("coordinated_offset"), "recall"].iloc[0]), "mixed_missing_recall": float(ext_types.loc[ext_types.attack_type.eq("mixed_missing_attack"), "recall"].iloc[0]), "replay_recall": float(ext_types.loc[ext_types.attack_type.eq("replay"), "recall"].iloc[0])})
        print(json.dumps({"strength": strength, "status": "completed", "test_f1": metrics["f1"]}, ensure_ascii=False))
    result = pd.DataFrame(rows)
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    result.to_csv(out_dir / "stage15_attack_intensity_sweep.csv", index=False, encoding="utf-8-sig")
    (out_dir / "stage15_attack_intensity_sweep.json").write_text(json.dumps({"status": "PASS_PIPELINE_REVIEW_REQUIRED", "results": result.to_dict(orient="records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段15攻击强度敏感性", "", "仅对阶段5扩展攻击的偏移幅度进行缩放，时间切分、基础攻击和正常数据保持不变。", "", "| 强度 | test F1 | Recall | AUPRC | 正常误伤率 | 协同偏移 Recall | 混合缺失 Recall | 重放 Recall |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in result.iterrows():
        lines.append(f"| {r['strength']:.1f} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_auprc']:.4f} | {r['test_false_alarm_rate']:.4f} | {r['coordinated_offset_recall']:.4f} | {r['mixed_missing_recall']:.4f} | {r['replay_recall']:.4f} |")
    report = report_dir / "stage15_attack_intensity_sweep.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_PIPELINE_REVIEW_REQUIRED", "report": str(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
