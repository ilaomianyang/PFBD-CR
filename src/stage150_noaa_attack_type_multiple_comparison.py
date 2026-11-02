from __future__ import annotations

"""Stage 150: exact fold sign-flip tests and BH correction for stage 149."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
EXPANDED = TABLES / "stage141_noaa_frozen_external_validation_by_attack_type.csv"
BASELINES = TABLES / "stage148_noaa_baseline_attack_type_detail_detail.csv"
METHOD_A = "TrustMeteo-QC frozen expanded"
METHODS_B = ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]


def exact_sign_flip(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null.append(abs(float((values * np.asarray(signs)).mean())))
    null_array = np.asarray(null, dtype=float)
    p_value = float(np.mean(null_array >= observed - 1e-15))
    return p_value, observed


def bh_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        idx = int(order[rank])
        running = min(running, float(p_values[idx]) * len(order) / (rank + 1))
        adjusted[idx] = running
    return adjusted.tolist()


def main() -> int:
    expanded = pd.read_csv(EXPANDED).rename(columns={"method": "source_method"})
    baselines = pd.read_csv(BASELINES).rename(columns={"method": "source_method"})
    expanded = expanded[expanded["source_method"].isin([METHOD_A, "TrustMeteo-QC default"])]
    baselines = baselines[baselines["source_method"].isin(["Robust max deviation", "Isolation Forest", "One-Class SVM", "TrustMeteo-QC"])]
    common = pd.concat([expanded[["fold", "seed", "source_method", "attack_type", "recall"]], baselines[["fold", "seed", "source_method", "attack_type", "recall"]]], ignore_index=True)
    rows: list[dict[str, object]] = []
    for method_b in METHODS_B:
        source_b = "TrustMeteo-QC" if method_b == "TrustMeteo-QC default" else method_b
        for attack_type in sorted(common["attack_type"].astype(str).unique()):
            a = common[(common["source_method"] == METHOD_A) & (common["attack_type"] == attack_type)].groupby("fold")["recall"].mean().rename("a")
            b = common[(common["source_method"] == source_b) & (common["attack_type"] == attack_type)].groupby("fold")["recall"].mean().rename("b")
            paired = pd.concat([a, b], axis=1).dropna()
            diffs = (paired["a"] - paired["b"]).to_numpy(dtype=float)
            p_value, observed = exact_sign_flip(diffs)
            rows.append({"comparison": f"{METHOD_A} - {method_b}", "attack_type": attack_type, "fold_count": len(diffs), "observed_mean_diff": observed if float(diffs.mean()) >= 0 else -observed, "p_value_exact_sign_flip": p_value, "fold_diffs": diffs.tolist()})
    p_values = [float(row["p_value_exact_sign_flip"]) for row in rows]
    q_values = bh_adjust(p_values)
    for row, q_value in zip(rows, q_values):
        row["bh_q_value"] = float(q_value)
    detail = pd.DataFrame(rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段149的四折、五种子类型级成对Recall差异；使用四折差异的精确sign-flip置换检验，并对24个模型比较×攻击类型假设做Benjamini–Hochberg校正；不重新训练、不重新选择候选、不根据test调参。攻击为透明受控注入，不是真实攻击日志。",
        "source_artifacts": [EXPANDED.name, BASELINES.name],
        "folds": sorted(common["fold"].astype(str).unique().tolist()),
        "comparisons": [METHOD_A + " - " + method for method in METHODS_B],
        "attack_types": sorted(common["attack_type"].astype(str).unique().tolist()),
        "hypothesis_count": len(detail),
        "detail": detail.to_dict(orient="records"),
        "limitations": ["只有四个滚动fold，精确sign-flip检验功效有限；q值不能替代更多独立年份或站点。", "多重比较校正只用于统计描述，不改变主模型选择。", "攻击为透明受控注入，不是真实攻击日志。"],
    }
    prefix = "stage150_noaa_attack_type_multiple_comparison"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段150 NOAA攻击类型多重比较校正", "", "对阶段149的24个类型级比较执行四折精确sign-flip检验，并做BH校正。", "", "| 比较 | 攻击类型 | 平均差 | 精确p | BH q |", "|---|---|---:|---:|---:|"]
    for row in detail.to_dict(orient="records"):
        lines.append(f"| {row['comparison']} | {row['attack_type']} | {row['observed_mean_diff']:+.4f} | {row['p_value_exact_sign_flip']:.4f} | {row['bh_q_value']:.4f} |")
    lines += ["", "四折样本量较小，校正结果用于透明报告，不支持把未显著差异写成等效。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "hypotheses": len(detail), "q_below_05": int((detail["bh_q_value"] < 0.05).sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
