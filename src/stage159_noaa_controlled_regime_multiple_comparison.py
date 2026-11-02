from __future__ import annotations

"""Stage 159: multiple-comparison correction for stage-158 strata."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = TABLES / "stage158_noaa_controlled_regime_gap_performance_detail.csv"
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]


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
    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    default = detail[detail["method"].eq("TrustMeteo-QC default")]
    rows: list[dict[str, object]] = []
    not_estimable: list[dict[str, object]] = []
    for metric in METRICS:
        left = expanded.groupby(["fold", "context_variable", "context"], sort=True)[metric].mean().rename("expanded")
        right = default.groupby(["fold", "context_variable", "context"], sort=True)[metric].mean().rename("default")
        common = pd.concat([left, right], axis=1).dropna()
        for (variable, context), group in common.groupby(level=["context_variable", "context"], sort=True):
            diffs = (group["expanded"] - group["default"]).to_numpy(dtype=float)
            if len(diffs) < 2:
                not_estimable.append({"metric": metric, "context_variable": variable, "context": context, "fold_count": int(len(diffs)), "reason": "insufficient target-year folds with estimable metric"})
                continue
            p_value, mean_diff = exact_sign_flip(diffs)
            rows.append({"comparison": "expanded - default", "metric": metric, "context_variable": variable, "context": context, "fold_count": int(len(diffs)), "mean_fold_diff": mean_diff, "p_value_exact_sign_flip": p_value, "fold_differences": diffs.tolist()})
    q_values = bh_adjust([float(row["p_value_exact_sign_flip"]) for row in rows])
    for row, q_value in zip(rows, q_values):
        row["bh_q_value"] = float(q_value)
    detail_out = pd.DataFrame(rows)
    prefix = "stage159_noaa_controlled_regime_multiple_comparison"
    detail_out.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(not_estimable).to_csv(TABLES / f"{prefix}_not_estimable.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段158的四折滚动、五种子、valid阈值选择和环境分层test结果；先按滚动fold对五种子取均值，再对可估计的环境状态×指标差异执行精确sign-flip检验，并对全部假设统一做Benjamini–Hochberg校正。不重新训练、不重新选择候选、不根据校正后的test结果回写模型；攻击为透明受控注入，不是真实攻击日志。",
        "source_artifact": SOURCE.name,
        "metrics": METRICS,
        "hypothesis_count": int(len(detail_out)),
        "not_estimable_count": int(len(not_estimable)),
        "detail": detail_out.to_dict(orient="records"),
        "not_estimable": not_estimable,
        "q_below_05": int((detail_out["bh_q_value"] < 0.05).sum()) if not detail_out.empty else 0,
        "limitations": ["四个滚动fold的精确检验功效有限；BH校正用于防止分层结果选择性报告，不等于证明等效。", "无攻击正例的状态不估计AUPRC，保留为不可估计。", "受控攻击不是真实攻击日志，分层test不改变主模型选择。"],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 阶段159 NOAA受控攻击环境分层多重比较校正", "", "对阶段158可估计的环境状态×指标差异按滚动fold执行精确sign-flip检验，并统一做BH校正。", "", "| 指标 | 状态变量 | 状态 | fold数 | 平均差 | 精确p | BH q |", "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['context_variable']} | {row['context']} | {row['fold_count']} | {row['mean_fold_diff']:+.6f} | {row['p_value_exact_sign_flip']:.6f} | {row['bh_q_value']:.6f} |")
    lines += ["", f"可估计假设数：`{len(rows)}`；不可估计单元：`{len(not_estimable)}`；BH q<0.05：`{int((detail_out['bh_q_value'] < 0.05).sum()) if not detail_out.empty else 0}`。", "多重比较校正只用于透明报告，不根据校正后的test结果调参。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "hypotheses": len(rows), "not_estimable": len(not_estimable), "q_below_05": payload["q_below_05"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
