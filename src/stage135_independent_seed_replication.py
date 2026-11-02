from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
DEFAULT_DATA = Path(r"F:\数据集\Analysis\experiments\pc_mstf_ap_v4\parquet\v4_anomaly_injection_table.parquet")
DEFAULT_NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
DEFAULT_ERA5 = TABLES / "stage06_era5_station_reference.parquet"
# Frozen before execution; these seeds are not used by any preceding selection step.
REPLICATION_SEEDS = [271828, 314159, 161803, 8675309, 424242]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage05_attack_extensions import build_extensions  # noqa: E402


METRICS = [
    "test_f1",
    "test_recall",
    "test_precision",
    "test_auprc",
    "test_false_alarm_rate",
    "event_recall",
    "recovery_mae_after",
]


def bootstrap_mean(values: np.ndarray, seed: int = 135135, repeats: int = 10000) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap input must be finite and non-empty")
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[sample_idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def run_one(seed: int, source: pd.DataFrame, noaa: Path, era5: Path) -> tuple[dict[str, object], pd.DataFrame]:
    tag = f"stage135_seed{seed}"
    input_path = TABLES / f"stage135_input_seed{seed}.parquet"
    result_path = TABLES / f"stage09b_robust_fusion_{tag}.json"
    attack_path = TABLES / f"stage09b_attack_type_metrics_{tag}.csv"
    if not input_path.is_file():
        extended, events = build_extensions(source, seed)
        extended.to_parquet(input_path, index=False)
    else:
        extended = pd.read_parquet(input_path)
        events = extended.loc[extended["extension_is_attack"].astype(bool), ["extension_event_id"]].drop_duplicates()
    command = [
        sys.executable,
        str(SRC / "stage09b_robust_fusion.py"),
        "--local",
        str(input_path),
        "--noaa",
        str(noaa),
        "--era5",
        str(era5),
        "--tag",
        tag,
    ]
    if not result_path.is_file() or not attack_path.is_file():
        completed = subprocess.run(command, cwd=ROOT.parents[1], capture_output=True, text=True)
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(f"stage09b failed for seed {seed}:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    test = payload["local_test"]
    row = {
        "seed": int(seed),
        "extension_event_count": int(len(events)),
        "test_f1": float(test["f1"]),
        "test_recall": float(test["recall"]),
        "test_precision": float(test["precision"]),
        "test_auprc": float(test["average_precision"]),
        "test_false_alarm_rate": float(payload["test_false_alarm_rate_on_clean_rows"]),
        "event_recall": float(payload["test_event_metrics"]["event_recall"]),
        "recovery_mae_before": float(payload["test_recovery"]["mae_before"]),
        "recovery_mae_after": float(payload["test_recovery"]["mae_after"]),
        "alpha_base": float(payload["setting"]["alpha_base"]),
        "alpha_structural": float(payload["setting"]["alpha_structural"]),
        "window": int(payload["setting"]["window"]),
        "threshold": float(payload["setting"]["threshold"]),
        "valid_f1": float(payload["setting"]["valid_f1"]),
        "valid_false_alarm_rate": float(payload["setting"]["valid_false_alarm_rate"]),
    }
    attack_detail = pd.read_csv(attack_path)
    attack_detail.insert(0, "seed", int(seed))
    return row, attack_detail


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段135：冻结流程的新攻击构造种子独立复现")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--noaa", type=Path, default=DEFAULT_NOAA)
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    parser.add_argument("--seeds", default=",".join(map(str, REPLICATION_SEEDS)))
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if seeds != REPLICATION_SEEDS:
        raise ValueError(f"stage135 uses the pre-registered seed list only: {REPLICATION_SEEDS}")
    for path in [args.data, args.noaa, args.era5]:
        if not path.is_file():
            raise FileNotFoundError(path)
    source = pd.read_parquet(args.data)
    rows: list[dict[str, object]] = []
    attack_frames: list[pd.DataFrame] = []
    for seed in seeds:
        row, attack = run_one(seed, source, args.noaa, args.era5)
        rows.append(row)
        attack_frames.append(attack)
        print(json.dumps({"seed": seed, "status": "completed", "test_f1": row["test_f1"], "test_recall": row["test_recall"]}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    attack_detail = pd.concat(attack_frames, ignore_index=True)
    summary = {metric: bootstrap_mean(detail[metric].to_numpy(dtype=float), seed=135135 + i) for i, metric in enumerate(METRICS)}
    attack_numeric = ["f1", "recall", "precision", "average_precision"]
    grouped = attack_detail.groupby(["split", "attack_type"], sort=True)
    attack_aggregate = grouped.agg(
        **{f"{col}_{stat}": (col, stat) for col in attack_numeric for stat in ["mean", "std"]}
    ).reset_index()
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段135预先冻结五个新攻击构造种子，在不修改阶段9B流程、不根据test重新选择的条件下完整重跑；每个种子仍只由自身valid选择融合设置和阈值，test只评价。阶段135不覆盖阶段9C原五种子，也不把受控攻击写成真实入侵。",
        "seeds": seeds,
        "seed_count": len(seeds),
        "source_data": str(args.data),
        "source_noaa": str(args.noaa),
        "source_era5": str(args.era5),
        "result_count": len(detail),
        "attack_detail_rows": len(attack_detail),
        "attack_aggregate_rows": len(attack_aggregate),
        "metrics": summary,
        "setting_summary": {
            "alpha_base_values": sorted(detail["alpha_base"].unique().tolist()),
            "alpha_structural_values": sorted(detail["alpha_structural"].unique().tolist()),
            "window_values": sorted(detail["window"].unique().tolist()),
        },
        "limitations": [
            "新种子仍来自同一气象数据源和同一受控攻击生成器，不等于独立地区、年份或真实攻击日志。",
            "阶段135用于冻结流程的独立复现，不用于回写阶段9B的参数或锁定测试结论。",
            "攻击类型分层结果应与总体指标并列报告，不能用单一类型的增益替代综合性能。",
        ],
    }
    detail.to_csv(TABLES / "stage135_independent_seed_replication_detail.csv", index=False, encoding="utf-8-sig")
    attack_detail.to_csv(TABLES / "stage135_independent_seed_replication_attack_detail.csv", index=False, encoding="utf-8-sig")
    attack_aggregate.to_csv(TABLES / "stage135_independent_seed_replication_attack_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage135_independent_seed_replication.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段135新攻击种子独立复现",
        "",
        "该阶段使用预先冻结的新种子，不修改阶段9B流程，不读取阶段135 test做选择。结果仍属于同一数据源上的受控攻击复现。",
        "",
        "| 指标 | 均值 | 标准差 | 95% bootstrap CI | 范围 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "test_f1": "test F1",
        "test_recall": "test Recall",
        "test_precision": "test Precision",
        "test_auprc": "test AUPRC",
        "test_false_alarm_rate": "正常样本误伤率",
        "event_recall": "事件召回",
        "recovery_mae_after": "恢复后 MAE",
    }
    for metric in METRICS:
        item = summary[metric]
        lines.append(f"| {labels[metric]} | {item['mean']:.4f} | {item['std']:.4f} | [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] | {item['min']:.4f}–{item['max']:.4f} |")
    lines += ["", "攻击类型分层明细见 `stage135_independent_seed_replication_attack_detail.csv` 和聚合表。", "阶段135不能证明跨地区、跨年份或真实攻击普适性。"]
    (REPORTS / "stage135_independent_seed_replication.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "result_count": len(detail), "attack_detail_rows": len(attack_detail), "attack_aggregate_rows": len(attack_aggregate), "metrics": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
