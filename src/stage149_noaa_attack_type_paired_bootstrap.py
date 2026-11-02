from __future__ import annotations

"""Stage 149: paired fold bootstrap for attack-type recall comparisons."""

import hashlib
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


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_fold(values: pd.DataFrame, seed: int, repeats: int = 20000) -> tuple[float, float]:
    folds = values["fold"].drop_duplicates().tolist()
    grouped = {fold: values.loc[values["fold"].eq(fold), "diff"].to_numpy(dtype=float).mean() for fold in folds}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(folds, size=(repeats, len(folds)), replace=True)
    means = np.asarray([[grouped[fold] for fold in row] for row in sampled], dtype=float).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    expanded = pd.read_csv(EXPANDED)
    baselines = pd.read_csv(BASELINES)
    expanded = expanded[expanded["method"].isin([METHOD_A, "TrustMeteo-QC default"])].copy()
    baselines = baselines[baselines["method"].isin(["Robust max deviation", "Isolation Forest", "One-Class SVM", "TrustMeteo-QC"])].copy()
    baselines = baselines.rename(columns={"method": "source_method"})
    expanded = expanded.rename(columns={"method": "source_method"})
    common = pd.concat([expanded[["fold", "seed", "source_method", "attack_type", "recall"]], baselines[["fold", "seed", "source_method", "attack_type", "recall"]]], ignore_index=True)
    required_methods = {METHOD_A, "TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM", "TrustMeteo-QC"}
    if set(common["source_method"].unique()) != required_methods:
        raise ValueError("stage149 method coverage is incomplete")
    attacks = sorted(common["attack_type"].astype(str).unique().tolist())
    folds = sorted(common["fold"].astype(str).unique().tolist())
    rows: list[dict[str, object]] = []
    for method_b in METHODS_B:
        source_b = "TrustMeteo-QC" if method_b == "TrustMeteo-QC default" else method_b
        for attack_type in attacks:
            a = common[(common["source_method"] == METHOD_A) & (common["attack_type"] == attack_type)].groupby("fold", as_index=False)["recall"].mean().rename(columns={"recall": "recall_a"})
            b = common[(common["source_method"] == source_b) & (common["attack_type"] == attack_type)].groupby("fold", as_index=False)["recall"].mean().rename(columns={"recall": "recall_b"})
            paired = a.merge(b, on="fold", validate="one_to_one")
            paired["diff"] = paired["recall_a"] - paired["recall_b"]
            low, high = bootstrap_fold(paired, stable_seed("stage149", method_b, attack_type))
            rows.append({"comparison": f"{METHOD_A} - {method_b}", "attack_type": attack_type, "fold_count": len(paired), "mean_diff": float(paired["diff"].mean()), "fold_bootstrap_ci95_low": low, "fold_bootstrap_ci95_high": high, "positive_folds": int((paired["diff"] > 0).sum()), "negative_folds": int((paired["diff"] < 0).sum()), "expanded_zero_recall_folds": int((paired["recall_a"] <= 0).sum()), "baseline_zero_recall_folds": int((paired["recall_b"] <= 0).sum())})
    paired = pd.DataFrame(rows)
    detail = common.groupby(["source_method", "attack_type"], as_index=False).agg(fold_count=("fold", "nunique"), seed_count=("seed", "nunique"), recall_mean=("recall", "mean"), recall_min=("recall", "min"), recall_max=("recall", "max"))
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段141 expanded/default与阶段148四基线攻击类型级Recall；先按四个滚动fold对五个种子聚合，再做fold簇配对bootstrap；不重新训练、不重新选择候选、不根据test调参。攻击为透明受控注入，不是真实攻击日志。",
        "source_artifacts": [EXPANDED.name, BASELINES.name],
        "folds": folds,
        "seeds": sorted(common["seed"].unique().astype(int).tolist()),
        "attack_types": attacks,
        "comparisons": [METHOD_A + " - " + method for method in METHODS_B],
        "detail": detail.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "limitations": ["fold数量为4，bootstrap只能量化当前滚动年度集合的不确定性。", "阶段148基线重算与阶段141 expanded/default使用同一NOAA滚动受控注入协议，不能替代真实攻击日志。", "类型级Recall差异不等于总体F1或安全运行点优越性，必须同时报告FAR、AUPRC和年度/站点异质性。"],
    }
    prefix = "stage149_noaa_attack_type_paired_bootstrap"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段149 NOAA攻击类型级成对bootstrap", "", "先按四个滚动fold对五个种子聚合，再对expanded相对各基线的类型级Recall差异做fold簇bootstrap。", "", "| 比较 | 攻击类型 | 平均差 | CI95 | 正向折 | 负向折 | expanded零Recall折 | 基线零Recall折 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in paired.to_dict(orient="records"):
        lines.append(f"| {row['comparison']} | {row['attack_type']} | {row['mean_diff']:.4f} | [{row['fold_bootstrap_ci95_low']:.4f}, {row['fold_bootstrap_ci95_high']:.4f}] | {row['positive_folds']} | {row['negative_folds']} | {row['expanded_zero_recall_folds']} | {row['baseline_zero_recall_folds']} |")
    lines += ["", "该分析用于类型级比较和负结果保留，不支持全面领先或真实攻击能力声明。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail": len(detail), "paired": len(paired)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
