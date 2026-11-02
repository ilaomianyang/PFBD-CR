from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = TABLES / "stage91_noaa_physical_component_repair_by_type.csv"


def main() -> int:
    detail = pd.read_csv(SOURCE)
    aggregate = detail.groupby(["variant", "attack_type"], sort=False).agg(rows=("rows", "sum"), detected_rows=("detected_rows", "sum"), recall_mean=("recall", "mean"), recall_std=("recall", "std"), recall_min=("recall", "min"), recall_max=("recall", "max")).reset_index()
    pivot = aggregate.pivot(index="attack_type", columns="variant", values="recall_mean").reset_index()
    if "legacy" in pivot and "component_zmax" in pivot:
        pivot["candidate_minus_legacy"] = pivot["component_zmax"] - pivot["legacy"]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "读取阶段91四折滚动、11站点、5种子、六类透明受控攻击的逐类型明细；不重新调参或重跑test。", "aggregate": aggregate.to_dict(orient="records"), "comparison": pivot.to_dict(orient="records"), "limitations": ["受控攻击不是真实攻击日志。", "逐类型Recall仅描述阶段91低误报候选与legacy的权衡，不替换主模型。"]}
    aggregate.to_csv(TABLES / "stage95_noaa_physical_component_attack_type.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage95_noaa_physical_component_attack_type.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段95 NOAA物理组件方案逐攻击类型诊断", "", "按阶段91结果汇总六类受控攻击的测试Recall，比较legacy与component_zmax。", "", "| 攻击类型 | legacy Recall | component_zmax Recall | 差值 |", "|---|---:|---:|---:|"]
    for row in pivot.to_dict(orient="records"):
        lines.append(f"| {row['attack_type']} | {row['legacy']:.4f} | {row['component_zmax']:.4f} | {row['candidate_minus_legacy']:+.4f} |")
    lines += ["", "component_zmax在所有攻击类型上Recall均明显低于legacy，因此只能作为低误报运行点候选，不能替换主检测方案。"]
    (REPORTS / "stage95_noaa_physical_component_attack_type.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "attack_types": int(len(pivot))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
