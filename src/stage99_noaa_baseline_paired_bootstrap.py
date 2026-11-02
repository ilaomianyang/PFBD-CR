from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
INPUT = TABLES / "stage98_noaa_low_far_baseline_comparison_detail.csv"
METHODS = ["Trust legacy", "Trust component_zmax", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]
Q_GRID = [0.90, 0.95, 0.975, 0.99, 0.995]
BOOTSTRAPS = 5000
SEED = 20260919


def _sample_block_indices(fold_indices: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Sample folds and then seeds within each fold using only integer indices."""
    sampled_fold_positions = rng.integers(0, len(fold_indices), size=len(fold_indices))
    sampled: list[np.ndarray] = []
    for fold_pos in sampled_fold_positions:
        block = fold_indices[int(fold_pos)]
        sampled.append(block[rng.integers(0, len(block), size=len(block))])
    return np.concatenate(sampled)


def main() -> int:
    detail = pd.read_csv(INPUT)
    required = {"fold", "seed", "method", "q", *METRICS}
    missing = required.difference(detail.columns)
    if missing:
        raise ValueError(f"stage98 detail missing columns: {sorted(missing)}")
    if set(detail["method"].unique()) != set(METHODS):
        raise ValueError(f"unexpected methods: {sorted(detail['method'].unique())}")

    rng = np.random.default_rng(SEED)
    summaries: list[dict[str, object]] = []
    for q in Q_GRID:
        q_frame = detail.loc[np.isclose(detail["q"].astype(float), q)].copy()
        block_count = q_frame[["fold", "seed"]].drop_duplicates().shape[0]
        if block_count < 4:
            raise ValueError(f"too few paired blocks for q={q}: {block_count}")
        wide = q_frame.pivot_table(index=["fold", "seed"], columns="method", values=METRICS)
        if wide.isna().any().any():
            raise ValueError(f"incomplete paired block table for q={q}")
        fold_indices = [np.flatnonzero(wide.index.get_level_values("fold").to_numpy() == fold) for fold in wide.index.get_level_values("fold").unique()]
        if any(len(indices) == 0 for indices in fold_indices):
            raise ValueError(f"empty fold in paired table for q={q}")
        rows: list[dict[str, object]] = []
        refs = ["Trust legacy", "Trust component_zmax"]
        for candidate in [m for m in METHODS if m not in refs]:
            for reference in refs:
                observed = {
                    metric: float((wide[(metric, candidate)] - wide[(metric, reference)]).mean())
                    for metric in METRICS
                }
                boot_values = {metric: [] for metric in METRICS}
                differences = {
                    metric: (wide[(metric, candidate)] - wide[(metric, reference)]).to_numpy(dtype=float)
                    for metric in METRICS
                }
                for _ in range(BOOTSTRAPS):
                    sampled = _sample_block_indices(fold_indices, rng)
                    for metric in METRICS:
                        boot_values[metric].append(float(differences[metric][sampled].mean()))
                for metric in METRICS:
                    arr = np.asarray(boot_values[metric], dtype=float)
                    row = {
                        "q": q,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "observed_diff_mean": observed[metric],
                        "bootstrap_diff_mean": float(arr.mean()),
                        "ci95_low": float(np.quantile(arr, 0.025)),
                        "ci95_high": float(np.quantile(arr, 0.975)),
                        "prob_candidate_better": float((arr > 0).mean()) if metric != "test_far" else float((arr < 0).mean()),
                        "direction": "higher_is_better" if metric != "test_far" else "lower_is_better",
                        "blocks": int(block_count),
                        "bootstraps": BOOTSTRAPS,
                    }
                    rows.append(row)
        summaries.extend(rows)

    out = pd.DataFrame(summaries)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段98已完成结果的层级配对bootstrap；先按4个滚动折有放回抽样，再在每个抽中折内按5个种子有放回抽样；不重新调参、不读取锁定测试标签以选择方法。",
        "source": str(INPUT),
        "q_grid": Q_GRID,
        "methods": METHODS,
        "metrics": METRICS,
        "blocks_per_q": int(detail[["fold", "seed"]].drop_duplicates().shape[0]),
        "bootstraps": BOOTSTRAPS,
        "random_seed": SEED,
        "results": out.to_dict(orient="records"),
        "limitations": [
            "阶段98明细按折×种子保存，没有逐站点混淆计数；本阶段不能替代站点簇bootstrap。",
            "q是预先固定的运行点网格；不能根据本bootstrap结果挑选主模型。",
            "攻击为透明受控注入，不是真实攻击日志。",
            "FAR差值采用候选减参考，FAR为负表示候选误报更低。",
        ],
    }
    out.to_csv(TABLES / "stage99_noaa_baseline_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage99_noaa_baseline_paired_bootstrap.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )

    lines = [
        "# 阶段99 NOAA基线层级配对bootstrap",
        "",
        "对阶段98的4个滚动折×5个种子进行层级配对bootstrap，比较各无监督基线相对Trust方案的指标差值。",
        "",
        "差值定义：候选方法减参考方法；FAR为负表示候选误报率更低。",
        "",
        "| q | 候选 | 参考 | 指标 | 观测差值 | 95% CI | 候选更优概率 |",
        "|---:|---|---|---|---:|---|---:|",
    ]
    for row in summaries:
        if abs(float(row["q"]) - 0.975) < 1e-9:
            lines.append(
                f"| {row['q']:.3f} | {row['candidate']} | {row['reference']} | {row['metric']} | "
                f"{row['observed_diff_mean']:.4f} | [{row['ci95_low']:.4f},{row['ci95_high']:.4f}] | "
                f"{row['prob_candidate_better']:.4f} |"
            )
    lines += [
        "",
        "阶段98的均值比较和本阶段置信区间必须与阶段86/97的站点簇bootstrap分开解释；本阶段只提供折×种子层面的配对不确定性。",
    ]
    (REPORTS / "stage99_noaa_baseline_paired_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(out), "q_values": len(Q_GRID)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
