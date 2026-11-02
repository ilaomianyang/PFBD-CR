from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
INPUT = TABLES / "stage09c_multiseed_detail.csv"
METRICS = [
    "test_f1",
    "test_recall",
    "test_precision",
    "test_auprc",
    "test_false_alarm_rate",
    "event_recall",
    "recovery_mae_after",
]
BOOTSTRAPS = 10000
SEED = 20260915


def main() -> int:
    detail = pd.read_csv(INPUT)
    required = {"seed", "returncode", *METRICS}
    if not required.issubset(detail.columns):
        raise ValueError(f"stage09c detail missing columns: {sorted(required.difference(detail.columns))}")
    if detail["seed"].nunique() != len(detail):
        raise ValueError("stage09c seeds must be unique")
    if not (detail["returncode"] == 0).all():
        raise ValueError("stage09c contains failed runs")
    values = detail[METRICS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("stage09c metrics contain non-finite values")

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        x = values[metric].to_numpy(dtype=float)
        sample_means = x[rng.integers(0, len(x), size=(BOOTSTRAPS, len(x)))].mean(axis=1)
        loo_means = np.array([(x.sum() - value) / (len(x) - 1) for value in x], dtype=float)
        rows.append(
            {
                "metric": metric,
                "seed_count": int(len(x)),
                "observed_mean": float(x.mean()),
                "observed_std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "observed_min": float(x.min()),
                "observed_max": float(x.max()),
                "bootstrap_mean_ci95_low": float(np.quantile(sample_means, 0.025)),
                "bootstrap_mean_ci95_high": float(np.quantile(sample_means, 0.975)),
                "bootstrap_mean_std": float(sample_means.std(ddof=1)),
                "leave_one_seed_min": float(loo_means.min()),
                "leave_one_seed_max": float(loo_means.max()),
                "bootstrap_samples": BOOTSTRAPS,
            }
        )

    out = pd.DataFrame(rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "读取阶段9C五个预先固定的受控攻击构造种子；不重新训练、不重新选阈值或模型；对每个种子的最终test指标做10000次种子级有放回bootstrap，并报告留一攻击种子均值范围。该分析只量化受控攻击构造随机性的指标波动。",
        "source": str(INPUT),
        "seeds": [int(x) for x in detail["seed"].tolist()],
        "metrics": METRICS,
        "bootstraps": BOOTSTRAPS,
        "seed": SEED,
        "results": out.to_dict(orient="records"),
        "limitations": [
            "只有5个受控攻击种子，bootstrap区间较粗，不能替代更多独立攻击生成机制或真实攻击日志。",
            "test指标仅被重新汇总，不用于模型、权重、阈值或运行点选择。",
            "种子级不确定性不代表站点、年份或攻击类型层级的不确定性；需结合阶段26、34、99、100和106。",
        ],
    }
    out.to_csv(TABLES / "stage117_multiseed_uncertainty.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage117_multiseed_uncertainty.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段117阶段9C受控攻击种子级不确定性",
        "",
        "只对阶段9C的五个预先固定攻击构造种子做bootstrap和留一法汇总，不改变任何模型或test选择。",
        "",
        "| 指标 | 种子数 | 均值 | 标准差 | 观察范围 | bootstrap均值95%区间 | 留一均值范围 |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    labels = {
        "test_f1": "test F1",
        "test_recall": "test Recall",
        "test_precision": "test Precision",
        "test_auprc": "test AUPRC",
        "test_false_alarm_rate": "正常误伤率",
        "event_recall": "事件召回",
        "recovery_mae_after": "恢复后 MAE",
    }
    for row in rows:
        lines.append(
            f"| {labels[row['metric']]} | {row['seed_count']} | {row['observed_mean']:.4f} | {row['observed_std']:.4f} | [{row['observed_min']:.4f},{row['observed_max']:.4f}] | [{row['bootstrap_mean_ci95_low']:.4f},{row['bootstrap_mean_ci95_high']:.4f}] | [{row['leave_one_seed_min']:.4f},{row['leave_one_seed_max']:.4f}] |"
        )
    lines += [
        "",
        "该阶段的区间只描述受控攻击种子随机性；不能解释为跨站点、跨年份或真实攻击泛化区间。",
    ]
    (REPORTS / "stage117_multiseed_uncertainty.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "metrics": len(rows), "seeds": len(detail), "bootstraps": BOOTSTRAPS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
