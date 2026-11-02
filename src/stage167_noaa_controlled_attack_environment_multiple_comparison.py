from __future__ import annotations

"""Stage 167: multiple-comparison correction for stage 166 interactions."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = TABLES / "stage166_noaa_controlled_attack_environment_interaction_detail.csv"
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]
BASELINES = ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]


def exact_sign_flip(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    observed_signed = float(values.mean())
    observed = abs(observed_signed)
    null = [abs(float((values * np.asarray(signs)).mean())) for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15)), observed_signed


def bh_adjust(values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(values, dtype=float))
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        idx = int(order[rank])
        running = min(running, float(values[idx]) * len(order) / (rank + 1))
        adjusted[idx] = running
    return adjusted.tolist()


def main() -> int:
    detail = pd.read_csv(SOURCE)
    rows: list[dict[str, object]] = []
    not_estimable: list[dict[str, object]] = []
    contexts = detail[["context_variable", "context", "attack_type"]].drop_duplicates().sort_values(["context_variable", "context", "attack_type"])
    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    keys = ["fold", "seed", "test_year", "context_variable", "context", "attack_type"]
    for baseline in BASELINES:
        other = detail[detail["method"].eq(baseline)]
        common = expanded.set_index(keys).join(other.set_index(keys), lsuffix="_expanded", rsuffix="_baseline", how="inner").reset_index()
        for metric in METRICS:
            common["diff"] = pd.to_numeric(common[f"{metric}_expanded"], errors="coerce") - pd.to_numeric(common[f"{metric}_baseline"], errors="coerce")
            for context_row in contexts.itertuples(index=False):
                variable, context, attack_type = str(context_row.context_variable), str(context_row.context), str(context_row.attack_type)
                group = common[common["context_variable"].eq(variable) & common["context"].eq(context) & common["attack_type"].eq(attack_type)].dropna(subset=["diff"])
                fold_values = group.groupby("fold", sort=True)["diff"].mean().dropna().to_numpy(dtype=float)
                if len(fold_values) < 2:
                    not_estimable.append({"comparison": f"expanded - {baseline}", "metric": metric, "context_variable": variable, "context": context, "attack_type": attack_type, "fold_count": int(len(fold_values)), "fold_differences": fold_values.tolist(), "reason": "fewer than two folds with estimable metric"})
                    continue
                p_value, mean_diff = exact_sign_flip(fold_values)
                rows.append({"comparison": f"expanded - {baseline}", "metric": metric, "context_variable": variable, "context": context, "attack_type": attack_type, "fold_count": int(len(fold_values)), "mean_fold_diff": mean_diff, "p_value_exact_sign_flip": p_value, "fold_differences": fold_values.tolist()})
    q_values = bh_adjust([float(row["p_value_exact_sign_flip"]) for row in rows])
    for row, q_value in zip(rows, q_values):
        row["bh_q_value"] = float(q_value)
    detail_out = pd.DataFrame(rows)
    not_estimable_out = pd.DataFrame(not_estimable)
    prefix = "stage167_noaa_controlled_attack_environment_multiple_comparison"
    detail_out.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    not_estimable_out.to_csv(TABLES / f"{prefix}_not_estimable.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段166攻击类型×环境状态交互的四折滚动、五种子、train正常样本拟合、valid阈值选择和test结果；先按fold汇总五种子，再对扩展方案相对四个基线的全部可估计交互假设执行精确sign-flip并统一做Benjamini–Hochberg校正。不重新训练、不重新选择候选、不根据校正后的test结果调参；攻击为透明受控注入，不是真实攻击日志；不可估计指标保留为空。",
        "source_artifact": SOURCE.name,
        "comparisons": [f"expanded - {baseline}" for baseline in BASELINES],
        "metrics": METRICS,
        "hypothesis_count": int(len(detail_out)),
        "not_estimable_count": int(len(not_estimable_out)),
        "detail": detail_out.to_dict(orient="records"),
        "not_estimable": not_estimable_out.to_dict(orient="records"),
        "q_below_05": int((detail_out["bh_q_value"] < 0.05).sum()) if not detail_out.empty else 0,
        "min_bh_q_value": float(detail_out["bh_q_value"].min()) if not detail_out.empty else None,
        "limitations": ["交互假设数量多且每个只有四个fold，BH校正后的结果功效有限。", "不可估计单元保留为空，不能填0。", "受控攻击不是真实攻击日志，交互结果不外推为真实入侵检测性能。"],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 阶段167 NOAA攻击类型×环境交互多重比较", "", "对阶段166的交互差异按fold执行精确sign-flip，并统一进行BH校正。", "", "| 比较 | 指标 | 状态变量 | 状态 | 攻击类型 | fold数 | 平均差 | 精确p | BH q |", "|---|---|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['context_variable']} | {row['context']} | {row['attack_type']} | {row['fold_count']} | {row['mean_fold_diff']:+.6f} | {row['p_value_exact_sign_flip']:.6f} | {row['bh_q_value']:.6f} |")
    lines += ["", f"可估计假设数：{len(rows)}；不可估计单元：{len(not_estimable_out)}；BH q<0.05：{int((detail_out['bh_q_value'] < 0.05).sum()) if not detail_out.empty else 0}。", "校正结果只用于透明报告，不根据test结果回写模型或参数。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "hypotheses": len(rows), "not_estimable": len(not_estimable_out), "q_below_05": payload["q_below_05"], "min_bh_q_value": payload["min_bh_q_value"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
