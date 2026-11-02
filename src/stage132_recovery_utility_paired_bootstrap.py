from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
BOOTSTRAPS = 10000
RANDOM_SEED = 13201
METRICS = ["test_utility", "test_attack_mae_after", "test_clean_harm_rate"]


def build_pairs(selected_path: Path, detail_path: Path, dataset: str, keys: list[str]) -> pd.DataFrame:
    selected = pd.read_csv(selected_path)
    detail = pd.read_csv(detail_path)
    reference = detail.loc[(detail["strategy"] == "legacy_selected") & (detail["gate_quantile"] == 0.0), keys + METRICS].copy()
    reference = reference.rename(columns={metric: f"reference_{metric}" for metric in METRICS})
    merged = selected.merge(reference, on=keys, how="inner", validate="many_to_one")
    if len(merged) != len(selected):
        raise ValueError(f"{dataset} reference merge mismatch: selected={len(selected)}, merged={len(merged)}")
    merged["dataset"] = dataset
    for metric in METRICS:
        merged[f"diff_{metric}"] = merged[metric] - merged[f"reference_{metric}"]
    return merged


def bootstrap(frame: pd.DataFrame, cluster: str, rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy, subset in frame.groupby("selection_policy", sort=True):
        for metric in METRICS:
            grouped = subset.groupby(cluster, as_index=False)[f"diff_{metric}"].mean()
            values = grouped[f"diff_{metric}"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or len(values) < 2:
                raise ValueError(f"invalid bootstrap values for {policy}/{metric}")
            samples = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
            low, high = np.quantile(samples, [0.025, 0.975])
            favorable = samples > 0 if metric == "test_utility" else samples < 0
            rows.append({
                "dataset": str(subset["dataset"].iloc[0]),
                "selection_policy": str(policy),
                "reference": "legacy_ungated_reference",
                "metric": metric,
                "cluster_level": cluster,
                "clusters": int(len(values)),
                "pairs": int(len(subset)),
                "observed_diff": float(values.mean()),
                "bootstrap_mean": float(samples.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "prob_diff_positive": float((samples > 0).mean()),
                "prob_diff_negative": float((samples < 0).mean()),
                "prob_favorable": float(favorable.mean()),
            })
    return rows


def main() -> int:
    local = build_pairs(
        TABLES / "stage129_valid_recovery_strategy_selection_selected.csv",
        TABLES / "stage129_valid_recovery_strategy_selection_detail.csv",
        "local_leave_one_station",
        ["seed", "held_out_station"],
    )
    noaa = build_pairs(
        TABLES / "stage130_noaa_safety_constrained_recovery_selected.csv",
        TABLES / "stage130_noaa_safety_constrained_recovery_detail.csv",
        "NOAA_external",
        ["seed"],
    )
    if "held_out_station" not in noaa.columns:
        noaa["held_out_station"] = ""
    pair_columns = [
        "dataset", "selection_policy", "seed", "held_out_station", "strategy", "gate_quantile",
        "test_utility", "reference_test_utility", "test_attack_mae_after", "reference_test_attack_mae_after",
        "test_clean_harm_rate", "reference_test_clean_harm_rate", "diff_test_utility",
        "diff_test_attack_mae_after", "diff_test_clean_harm_rate",
    ]
    pairs = pd.concat([local, noaa], ignore_index=True)[pair_columns]
    rng = np.random.default_rng(RANDOM_SEED)
    results = bootstrap(local, "held_out_station", rng) + bootstrap(noaa, "seed", rng)
    result = pd.DataFrame(results)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段132读取阶段129本地和阶段130 NOAA的valid-only已选择结果及legacy未门控参考；本地先按五个固定种子在七个留出站点内聚合，再以站点为簇执行10000次配对bootstrap；NOAA以五个固定种子为簇。test结果只用于差值评价，不根据test重新选择恢复策略或阈值，不重新训练。",
        "bootstraps": BOOTSTRAPS,
        "random_seed": RANDOM_SEED,
        "pair_rows": len(pairs),
        "result_rows": len(result),
        "results": results,
        "limitations": [
            "本地只有七个站点簇，NOAA只有五个攻击构造种子簇，区间不能外推为总体分布。",
            "NOAA种子共享同一自然观测底表，不等同于五个独立外部数据源。",
            "utility是恢复误差与正常改写伤害的工程权衡，不替代检测F1或真实业务损失。",
            "攻击为受控注入，不是真实运维故障日志。",
        ],
    }
    pairs.to_csv(TABLES / "stage132_recovery_utility_paired_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    result.to_csv(TABLES / "stage132_recovery_utility_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage132_recovery_utility_paired_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段132恢复效用配对簇级bootstrap", "", f"- bootstrap次数：`{BOOTSTRAPS}`", "", "| 数据集 | 选择策略 | 指标 | 簇数 | 差值 | 95% CI | 有利方向概率 |", "|---|---|---|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['dataset']} | {row['selection_policy']} | {row['metric']} | {row['clusters']} | {row['observed_diff']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['prob_favorable']:.4f} |")
    lines += ["", "差值均相对legacy未门控恢复；utility为正有利，攻击MAE和正常改写伤害为负有利。区间跨零时必须保留不确定性。"]
    (REPORTS / "stage132_recovery_utility_paired_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "pair_rows": len(pairs), "result_rows": len(result)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
