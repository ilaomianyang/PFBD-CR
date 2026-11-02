from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_result(label: str, path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data["local_test"]
    recovery = data["test_recovery"]
    return {
        "variant": label,
        "test_f1": test["f1"],
        "test_recall": test["recall"],
        "test_precision": test["precision"],
        "test_auprc": test["average_precision"],
        "test_false_alarm_rate": data["test_false_alarm_rate_on_clean_rows"],
        "event_recall": data["test_event_metrics"]["event_recall"],
        "recovery_mae_after": recovery["mae_after"],
        "recovery_rmse_after": recovery["rmse_after"],
    }


def main() -> int:
    files = [
        ("完整模型", ROOT / "tables" / "stage09b_robust_fusion.json"),
        ("去结构化特征", ROOT / "tables" / "stage09b_robust_fusion_ablation_nostruct.json"),
        ("去ERA5背景", ROOT / "tables" / "stage09b_robust_fusion_ablation_noera5.json"),
    ]
    rows = [read_result(label, path) for label, path in files if path.exists()]
    detail = pd.DataFrame(rows)
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    detail.to_csv(out_dir / "stage09c_ablation_summary.csv", index=False, encoding="utf-8-sig")
    lines = ["# 阶段9C消融实验", "", "所有变体使用相同的本地受控攻击表；valid 仅用于各自选择阈值，test 只用于评价。", "", "| 变体 | test F1 | Recall | Precision | AUPRC | 正常误伤率 | 事件级召回 | 恢复后 MAE | 恢复后 RMSE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in detail.iterrows():
        lines.append(f"| {row['variant']} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | {row['test_precision']:.4f} | {row['test_auprc']:.4f} | {row['test_false_alarm_rate']:.4f} | {row['event_recall']:.4f} | {row['recovery_mae_after']:.4f} | {row['recovery_rmse_after']:.4f} |")
    lines += ["", "解释：消融结果用于说明模块贡献，不能据此挑选有利结果而删除不利结果。"]
    report = report_dir / "stage09c_ablation_summary.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_PIPELINE_REVIEW_REQUIRED", "report": str(report), "variants": int(len(detail))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
