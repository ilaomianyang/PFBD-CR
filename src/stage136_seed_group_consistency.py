from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
METRICS = [
    "test_f1",
    "test_recall",
    "test_precision",
    "test_auprc",
    "test_false_alarm_rate",
    "event_recall",
    "recovery_mae_after",
]


def exact_permutation_p(old: np.ndarray, new: np.ndarray) -> float:
    pooled = np.concatenate([old, new])
    n_old = len(old)
    observed = float(new.mean() - old.mean())
    diffs: list[float] = []
    for chosen in itertools.combinations(range(len(pooled)), n_old):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(chosen)] = True
        diffs.append(float(pooled[~mask].mean() - pooled[mask].mean()))
    diffs_arr = np.asarray(diffs, dtype=float)
    return float(np.mean(np.abs(diffs_arr) >= abs(observed) - 1e-15))


def bootstrap_diff(old: np.ndarray, new: np.ndarray, repeats: int = 10000, seed: int = 136136) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    old_idx = rng.integers(0, len(old), size=(repeats, len(old)))
    new_idx = rng.integers(0, len(new), size=(repeats, len(new)))
    diffs = new[new_idx].mean(axis=1) - old[old_idx].mean(axis=1)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def bh_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        index = int(order[rank])
        value = min(running, p_values[index] * len(p_values) / (rank + 1))
        adjusted[index] = value
        running = value
    return adjusted.tolist()


def main() -> int:
    old_path = TABLES / "stage09c_multiseed_detail.csv"
    new_path = TABLES / "stage135_independent_seed_replication_detail.csv"
    if not old_path.is_file() or not new_path.is_file():
        raise FileNotFoundError("stage09c and stage135 detail tables are required")
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    if len(old) != 5 or len(new) != 5:
        raise ValueError("stage136 requires exactly five original and five replication seeds")
    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for index, metric in enumerate(METRICS):
        old_values = old[metric].to_numpy(dtype=float)
        new_values = new[metric].to_numpy(dtype=float)
        if not np.isfinite(old_values).all() or not np.isfinite(new_values).all():
            raise ValueError(f"non-finite values in {metric}")
        ci_low, ci_high = bootstrap_diff(old_values, new_values, seed=136136 + index)
        p_value = exact_permutation_p(old_values, new_values)
        p_values.append(p_value)
        rows.append({
            "metric": metric,
            "original_mean": float(old_values.mean()),
            "replication_mean": float(new_values.mean()),
            "observed_diff_replication_minus_original": float(new_values.mean() - old_values.mean()),
            "original_std": float(old_values.std(ddof=1)),
            "replication_std": float(new_values.std(ddof=1)),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "exact_permutation_p": p_value,
        })
    adjusted = bh_adjust(p_values)
    for row, adjusted_p in zip(rows, adjusted):
        row["bh_adjusted_p"] = float(adjusted_p)
        row["ci95_includes_zero"] = bool(row["bootstrap_ci95_low"] <= 0 <= row["bootstrap_ci95_high"])
    detail = pd.DataFrame(rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段136比较阶段9C原五种子与阶段135五个预先冻结新种子；每项指标使用独立两组bootstrap 10000次和10个观测值上的精确置换检验，最后对七项p值做Benjamini–Hochberg校正。不根据test重新选择、不据此调参，也不回写locked test或主模型设置。",
        "original_seed_count": int(len(old)),
        "replication_seed_count": int(len(new)),
        "result_count": int(len(detail)),
        "metrics": detail.to_dict(orient="records"),
        "limitations": [
            "两组各只有5个种子，检验功效有限；CI跨零不能证明完全等价。",
            "两组来自同一气象数据源和同一受控攻击生成器，不能替代跨地区、跨年份或真实攻击验证。",
            "该阶段是稳定性审计，不是新的模型选择或性能优化步骤。",
        ],
    }
    detail.to_csv(TABLES / "stage136_seed_group_consistency_detail.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage136_seed_group_consistency.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段136原始种子与新复现种子一致性审计",
        "",
        "阶段136只做稳定性比较，不根据结果调参或回写锁定测试。",
        "",
        "| 指标 | 原始均值 | 新复现均值 | 差值 | bootstrap 95% CI | 精确置换 p | BH 校正 p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['original_mean']:.4f} | {row['replication_mean']:.4f} | {row['observed_diff_replication_minus_original']:+.4f} | [{row['bootstrap_ci95_low']:+.4f}, {row['bootstrap_ci95_high']:+.4f}] | {row['exact_permutation_p']:.4f} | {row['bh_adjusted_p']:.4f} |")
    lines += ["", "两组均只有5个种子；该审计用于发现明显种子组漂移，不能宣称统计等价或跨域普适。"]
    (REPORTS / "stage136_seed_group_consistency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "result_count": len(detail), "metrics": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
