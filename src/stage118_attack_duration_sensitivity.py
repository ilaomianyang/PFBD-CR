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
BASE_NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
BASE_ERA5 = TABLES / "stage06_era5_station_reference.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
DURATIONS = [4, 8, 12]
METRICS = [
    "test_f1",
    "test_recall",
    "test_precision",
    "test_auprc",
    "test_false_alarm_rate",
    "event_recall",
    "recovery_mae_after",
]


def truncate_extensions(frame: pd.DataFrame, duration: int) -> pd.DataFrame:
    if duration <= 0:
        raise ValueError("duration must be positive")
    out = frame.copy()
    event_ids = out.get("extension_event_id", pd.Series("", index=out.index)).fillna("").astype(str)
    for event_id in sorted(x for x in event_ids.unique() if x):
        event_rows = out.index[event_ids.eq(event_id)]
        event_part = out.loc[event_rows].sort_values(["station_key", "timestamp_utc"], kind="stable")
        keep: set[int] = set()
        for _, station_part in event_part.groupby("station_key", sort=True):
            keep.update(int(x) for x in station_part.index[:duration])
        reset = [int(x) for x in event_rows if int(x) not in keep]
        if reset:
            out.loc[reset, "extension_attack_type"] = "normal"
            out.loc[reset, "extension_is_attack"] = False
            out.loc[reset, "extension_delta"] = 0.0
            out.loc[reset, "extension_event_id"] = ""
            out.loc[reset, "extension_observed_value"] = out.loc[reset, "observed_value"].to_numpy(dtype=float)
    return out


def run_one(seed: int, duration: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_path = TABLES / f"stage09c_input_seed{seed}.parquet"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = pd.read_parquet(source_path)
    variant = truncate_extensions(source, duration)
    tag = f"stage118_dur{duration}_seed{seed}"
    input_path = TABLES / f"stage118_input_{tag}.parquet"
    variant.to_parquet(input_path, index=False)
    command = [
        sys.executable,
        str(SRC / "stage09b_robust_fusion.py"),
        "--local",
        str(input_path),
        "--noaa",
        str(BASE_NOAA),
        "--era5",
        str(BASE_ERA5),
        "--tag",
        tag,
    ]
    completed = subprocess.run(command, cwd=ROOT.parents[1], capture_output=True, text=True)
    result_path = TABLES / f"stage09b_robust_fusion_{tag}.json"
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"duration={duration}, seed={seed} failed:\n{completed.stdout[-1500:]}\n{completed.stderr[-1500:]}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    test = result["local_test"]
    row = {
        "duration": duration,
        "seed": seed,
        "extension_attack_rows": int(variant["extension_is_attack"].astype(bool).sum()),
        "extension_event_count": int(variant.loc[variant["extension_is_attack"].astype(bool), "extension_event_id"].nunique()),
        "test_f1": float(test["f1"]),
        "test_recall": float(test["recall"]),
        "test_precision": float(test["precision"]),
        "test_auprc": float(test["average_precision"]),
        "test_false_alarm_rate": float(result["test_false_alarm_rate_on_clean_rows"]),
        "event_recall": float(result["test_event_metrics"]["event_recall"]),
        "recovery_mae_after": float(result["test_recovery"]["mae_after"]),
        "returncode": int(completed.returncode),
    }
    type_path = TABLES / f"stage09b_attack_type_metrics_{tag}.csv"
    by_type: list[dict[str, object]] = []
    if type_path.is_file():
        type_detail = pd.read_csv(type_path)
        type_detail = type_detail.loc[type_detail["split"].eq("test")].copy()
        for item in type_detail.to_dict(orient="records"):
            by_type.append({"duration": duration, "seed": seed, "attack_type": item["attack_type"], "recall": float(item["recall"]), "rows": int(item.get("rows", 0))})
    return row, by_type


def main() -> int:
    details: list[dict[str, object]] = []
    by_type: list[dict[str, object]] = []
    for duration in DURATIONS:
        for seed in SEEDS:
            row, types = run_one(seed, duration)
            details.append(row)
            by_type.extend(types)
            print(json.dumps({"duration": duration, "seed": seed, "status": "completed", "test_f1": row["test_f1"], "attack_rows": row["extension_attack_rows"]}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(details)
    type_detail = pd.DataFrame(by_type)
    aggregate = detail.groupby("duration", as_index=False)[METRICS].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["duration"] + [f"{metric}_{stat}" for metric, stat in aggregate.columns.tolist()[1:]]
    type_aggregate = type_detail.groupby(["duration", "attack_type"], as_index=False).agg(
        seeds=("seed", "nunique"), rows_mean=("rows", "mean"), recall_mean=("recall", "mean"), recall_std=("recall", "std"), recall_min=("recall", "min"), recall_max=("recall", "max")
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "读取阶段9C五个预先固定受控攻击构造种子；对每个种子只将扩展攻击事件按站点截取为4、8、12个观测点，保持原始观测、训练边界和模型代码不变；每个变体重新执行阶段9B，valid仍按原规则选择，test只评价。该实验用于检验事件持续时间敏感性，不根据test结果调参。",
        "source_inputs": [str(TABLES / f"stage09c_input_seed{seed}.parquet") for seed in SEEDS],
        "seeds": SEEDS,
        "durations": DURATIONS,
        "detail": detail.to_dict(orient="records"),
        "duration_aggregate": aggregate.to_dict(orient="records"),
        "attack_type_aggregate": type_aggregate.to_dict(orient="records"),
        "limitations": [
            "持续时间只覆盖4、8、12个观测点，不能代表所有攻击持续时间。",
            "扩展攻击仍是透明受控注入，不是真实网络攻击日志。",
            "五个种子和固定攻击类型不等于跨站点、跨年份泛化证明。",
        ],
    }
    detail.to_csv(TABLES / "stage118_attack_duration_sensitivity_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage118_attack_duration_sensitivity.csv", index=False, encoding="utf-8-sig")
    type_aggregate.to_csv(TABLES / "stage118_attack_duration_sensitivity_by_type.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage118_attack_duration_sensitivity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 阶段118受控攻击事件持续时间敏感性",
        "",
        "对阶段9C五个固定种子，将扩展攻击事件按站点截取为4、8、12个观测点，并完整重跑阶段9B。",
        "",
        "| 时长 | F1均值±标准差 | Recall均值±标准差 | AUPRC均值±标准差 | 正常误伤率均值±标准差 | 事件召回均值±标准差 | 恢复后MAE均值±标准差 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['duration']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_false_alarm_rate_mean']:.4f}±{row['test_false_alarm_rate_std']:.4f} | {row['event_recall_mean']:.4f}±{row['event_recall_std']:.4f} | {row['recovery_mae_after_mean']:.4f}±{row['recovery_mae_after_std']:.4f} |")
    lines += ["", "该实验只支持事件持续时间敏感性，不支持真实攻击泛化或普适最优持续时间结论。"]
    (REPORTS / "stage118_attack_duration_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "runs": len(detail), "durations": DURATIONS, "seeds": SEEDS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
