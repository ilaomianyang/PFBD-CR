from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
ALL_INPUT = TABLES / "stage102_noaa_trust_svm_valid_selection_all_candidates.csv"
SELECTED_INPUT = TABLES / "stage102_noaa_trust_svm_valid_selection_selected.csv"
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]
REFERENCES = ["svm_only", "trust_component_only"]
BOOTSTRAPS = 5000


def main() -> int:
    all_candidates = pd.read_csv(ALL_INPUT)
    selected = pd.read_csv(SELECTED_INPUT)
    keys = ["fold", "seed"]
    required = set(keys + ["method", *METRICS])
    if not required.issubset(all_candidates.columns) or not required.issubset(selected.columns):
        raise ValueError("stage102 files do not contain the required selection/stability columns")
    if len(selected) == 0 or selected[keys].drop_duplicates().shape[0] != len(selected):
        raise ValueError("selected table must contain exactly one row per fold-seed block")

    reference_tables: dict[str, pd.DataFrame] = {}
    for method in REFERENCES:
        ref = all_candidates.loc[all_candidates["method"].eq(method), keys + METRICS].copy()
        ref = ref.rename(columns={metric: f"{metric}_{method}" for metric in METRICS})
        reference_tables[method] = ref
    paired = selected[keys + ["method"] + METRICS].copy()
    for method, ref in reference_tables.items():
        paired = paired.merge(ref, on=keys, how="left", validate="one_to_one")
    if paired.isna().any().any():
        raise ValueError("missing paired reference rows in stage102 candidates")

    rng = np.random.default_rng(20260921)
    n = len(selected)
    methods = sorted(selected["method"].unique())
    selection_rows: list[dict[str, object]] = []
    for method in methods:
        indicator = (selected["method"].to_numpy() == method).astype(float)
        values = []
        for _ in range(BOOTSTRAPS):
            values.append(float(indicator[rng.integers(0, n, size=n)].mean()))
        arr = np.asarray(values)
        selection_rows.append({"method": method, "selected_count": int(indicator.sum()), "selection_fraction": float(indicator.mean()), "ci95_low": float(np.quantile(arr, 0.025)), "ci95_high": float(np.quantile(arr, 0.975))})

    difference_rows: list[dict[str, object]] = []
    for reference in REFERENCES:
        for metric in METRICS:
            diff = paired[metric].to_numpy(dtype=float) - paired[f"{metric}_{reference}"].to_numpy(dtype=float)
            boot = np.asarray([float(diff[rng.integers(0, n, size=n)].mean()) for _ in range(BOOTSTRAPS)])
            difference_rows.append({"reference": reference, "metric": metric, "paired_diff_mean": float(diff.mean()), "ci95_low": float(np.quantile(boot, 0.025)), "ci95_high": float(np.quantile(boot, 0.975)), "prob_selected_better": float((boot > 0).mean()) if metric != "test_far" else float((boot < 0).mean()), "direction": "higher_is_better" if metric != "test_far" else "lower_is_better"})

    fold_table = selected.groupby(["fold", "method"], sort=False).size().reset_index(name="selected_count")
    selection_df = pd.DataFrame(selection_rows)
    difference_df = pd.DataFrame(difference_rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "读取阶段102的20个valid选择块；不重新调参、不使用test选择；对选择频率和selected-vs-fixed baseline进行5000次折×种子块bootstrap。",
        "blocks": n,
        "bootstraps": BOOTSTRAPS,
        "selection_summary": selection_df.to_dict(orient="records"),
        "selection_by_fold": fold_table.to_dict(orient="records"),
        "paired_differences": difference_df.to_dict(orient="records"),
        "limitations": [
            "bootstrap单位是折×种子块，不是逐站点簇；站点簇证据见阶段100。",
            "selected候选由valid选择，但本阶段的test差值只用于事后估计，不重新选择模型。",
            "q=0.975是低误报运行点，不代表F1全局最优。",
            "攻击为透明受控注入，不是真实攻击日志。",
        ],
    }
    selection_df.to_csv(TABLES / "stage103_noaa_valid_selection_frequency.csv", index=False, encoding="utf-8-sig")
    fold_table.to_csv(TABLES / "stage103_noaa_valid_selection_by_fold.csv", index=False, encoding="utf-8-sig")
    difference_df.to_csv(TABLES / "stage103_noaa_valid_selection_paired_differences.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage103_noaa_valid_selection_stability.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段103 NOAA验证集选权重稳定性", "", "统计阶段102的20个valid选择块，并将selected候选与同一折×种子的固定SVM/Trust配对比较。", "", "## 选择频率", "", "| 方法 | 次数 | 比例 | 95% CI |", "|---|---:|---:|---|"]
    for row in selection_rows:
        lines.append(f"| {row['method']} | {row['selected_count']} | {row['selection_fraction']:.3f} | [{row['ci95_low']:.3f},{row['ci95_high']:.3f}] |")
    lines += ["", "## selected相对固定基线的配对差值", "", "| 参考 | 指标 | 差值 | 95% CI | selected更优概率 |", "|---|---|---:|---|---:|"]
    for row in difference_rows:
        lines.append(f"| {row['reference']} | {row['metric']} | {row['paired_diff_mean']:.4f} | [{row['ci95_low']:.4f},{row['ci95_high']:.4f}] | {row['prob_selected_better']:.4f} |")
    lines += ["", "选择频率按折展开见 `stage103_noaa_valid_selection_by_fold.csv`；本阶段不把事后test差值用于重新挑选模型。"]
    (REPORTS / "stage103_noaa_valid_selection_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "blocks": n, "methods": len(methods), "difference_rows": len(difference_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
