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
SHIFTS = [-24, 0, 24]
METRICS = [
    "test_f1",
    "test_recall",
    "test_precision",
    "test_auprc",
    "test_false_alarm_rate",
    "event_recall",
    "recovery_mae_after",
]


def _event_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    event_ids = frame.get("extension_event_id", pd.Series("", index=frame.index)).fillna("").astype(str)
    records: list[dict[str, object]] = []
    for event_id in sorted(x for x in event_ids.unique() if x):
        event = frame.loc[event_ids.eq(event_id)].sort_values(["station_key", "timestamp_utc"], kind="stable")
        rows_by_station: list[dict[str, object]] = []
        for station, part in event.groupby("station_key", sort=True):
            part = part.sort_values("timestamp_utc", kind="stable")
            rows_by_station.append(
                {
                    "station_key": str(station),
                    "split": str(part["split"].iloc[0]),
                    "source_indices": [int(x) for x in part.index],
                    "values": pd.to_numeric(part["extension_observed_value"], errors="coerce").tolist(),
                    "deltas": pd.to_numeric(part["extension_delta"], errors="coerce").tolist(),
                    "attack_type": str(part["extension_attack_type"].iloc[0]),
                }
            )
        records.append({"event_id": event_id, "rows_by_station": rows_by_station})
    return records


def relocate_extensions(frame: pd.DataFrame, shift_rows: int) -> tuple[pd.DataFrame, dict[str, int]]:
    source = frame.copy()
    events = _event_records(source)
    out = source.copy()
    out["extension_attack_type"] = "normal"
    out["extension_is_attack"] = False
    out["extension_delta"] = 0.0
    out["extension_event_id"] = ""
    out["extension_observed_value"] = out["observed_value"].to_numpy(dtype=float)
    groups: dict[tuple[str, str], list[int]] = {}
    for (station, split), part in source.groupby(["station_key", "split"], sort=True):
        groups[(str(station), str(split))] = [int(x) for x in part.sort_values("timestamp_utc", kind="stable").index]
    claimed: set[int] = set()
    relocated = 0
    fallback = 0
    skipped = 0
    for event in events:
        candidate_targets: list[tuple[dict[str, object], list[int]]] = []
        valid = True
        for station_part in event["rows_by_station"]:
            source_indices = list(station_part["source_indices"])
            group_indices = groups[(station_part["station_key"], station_part["split"])]
            positions = [group_indices.index(index) for index in source_indices]
            targets = [group_indices[pos + shift_rows] if 0 <= pos + shift_rows < len(group_indices) else -1 for pos in positions]
            if any(index < 0 for index in targets) or any(index in claimed for index in targets):
                valid = False
                break
            candidate_targets.append((station_part, targets))
        if not valid:
            fallback += 1
            candidate_targets = []
            for station_part in event["rows_by_station"]:
                source_indices = list(station_part["source_indices"])
                if any(index in claimed for index in source_indices):
                    skipped += 1
                    candidate_targets = []
                    break
                candidate_targets.append((station_part, source_indices))
            if not candidate_targets:
                continue
        for station_part, targets in candidate_targets:
            values = station_part["values"]
            deltas = station_part["deltas"]
            for index, value, delta in zip(targets, values, deltas):
                out.loc[index, "extension_attack_type"] = station_part["attack_type"]
                out.loc[index, "extension_is_attack"] = True
                out.loc[index, "extension_delta"] = delta if np.isfinite(delta) else np.nan
                out.loc[index, "extension_event_id"] = event["event_id"]
                out.loc[index, "extension_observed_value"] = value
                claimed.add(int(index))
        if shift_rows != 0 and valid:
            relocated += 1
    return out, {"event_count": len(events), "relocated_events": relocated, "fallback_events": fallback, "skipped_events": skipped}


def run_one(seed: int, shift_rows: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_path = TABLES / f"stage09c_input_seed{seed}.parquet"
    source = pd.read_parquet(source_path)
    variant, placement = relocate_extensions(source, shift_rows)
    label = "baseline" if shift_rows == 0 else f"shiftm{abs(shift_rows)}" if shift_rows < 0 else f"shiftp{shift_rows}"
    tag = f"stage119_{label}_seed{seed}"
    input_path = TABLES / f"stage119_input_{tag}.parquet"
    variant.to_parquet(input_path, index=False)
    command = [sys.executable, str(SRC / "stage09b_robust_fusion.py"), "--local", str(input_path), "--noaa", str(BASE_NOAA), "--era5", str(BASE_ERA5), "--tag", tag]
    completed = subprocess.run(command, cwd=ROOT.parents[1], capture_output=True, text=True)
    result_path = TABLES / f"stage09b_robust_fusion_{tag}.json"
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"shift={shift_rows}, seed={seed} failed:\n{completed.stdout[-1500:]}\n{completed.stderr[-1500:]}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    test = result["local_test"]
    row = {"shift_rows": shift_rows, "seed": seed, **placement, "extension_attack_rows": int(variant["extension_is_attack"].astype(bool).sum()), "test_f1": float(test["f1"]), "test_recall": float(test["recall"]), "test_precision": float(test["precision"]), "test_auprc": float(test["average_precision"]), "test_false_alarm_rate": float(result["test_false_alarm_rate_on_clean_rows"]), "event_recall": float(result["test_event_metrics"]["event_recall"]), "recovery_mae_after": float(result["test_recovery"]["mae_after"]), "returncode": int(completed.returncode)}
    type_path = TABLES / f"stage09b_attack_type_metrics_{tag}.csv"
    types: list[dict[str, object]] = []
    if type_path.is_file():
        type_detail = pd.read_csv(type_path)
        for item in type_detail.loc[type_detail["split"].eq("test")].to_dict(orient="records"):
            types.append({"shift_rows": shift_rows, "seed": seed, "attack_type": item["attack_type"], "recall": float(item["recall"]), "rows": int(item.get("rows", 0))})
    return row, types


def main() -> int:
    details: list[dict[str, object]] = []
    by_type: list[dict[str, object]] = []
    for shift in SHIFTS:
        for seed in SEEDS:
            row, types = run_one(seed, shift)
            details.append(row)
            by_type.extend(types)
            print(json.dumps({"shift_rows": shift, "seed": seed, "status": "completed", "test_f1": row["test_f1"], "attack_rows": row["extension_attack_rows"]}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(details)
    type_detail = pd.DataFrame(by_type)
    aggregate = detail.groupby("shift_rows", as_index=False)[METRICS].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["shift_rows"] + [f"{metric}_{stat}" for metric, stat in aggregate.columns.tolist()[1:]]
    type_aggregate = type_detail.groupby(["shift_rows", "attack_type"], as_index=False).agg(seeds=("seed", "nunique"), rows_mean=("rows", "mean"), recall_mean=("recall", "mean"), recall_std=("recall", "std"), recall_min=("recall", "min"), recall_max=("recall", "max"))
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "读取阶段9C五个预先固定受控攻击构造种子；保持攻击类型、幅度、原始观测、训练边界和模型代码不变，仅在同一站点同一split内将扩展攻击事件平移−24、0、+24个观测行（数据采样间隔约10分钟）；每个变体重新执行阶段9B，valid仍按原规则选择，test只评价。该实验用于检验攻击时间位置敏感性，不根据test结果调参。", "source_inputs": [str(TABLES / f"stage09c_input_seed{seed}.parquet") for seed in SEEDS], "seeds": SEEDS, "shifts_rows": SHIFTS, "detail": detail.to_dict(orient="records"), "shift_aggregate": aggregate.to_dict(orient="records"), "attack_type_aggregate": type_aggregate.to_dict(orient="records"), "limitations": ["位置平移只覆盖±24个观测行，不能代表所有时间位置或跨站点位置。", "事件平移遇到边界或重叠时保留原位置并记录fallback，不能把所有事件都视为完全平移。", "扩展攻击仍是透明受控注入，不是真实网络攻击日志。"]}
    detail.to_csv(TABLES / "stage119_attack_position_sensitivity_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage119_attack_position_sensitivity.csv", index=False, encoding="utf-8-sig")
    type_aggregate.to_csv(TABLES / "stage119_attack_position_sensitivity_by_type.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage119_attack_position_sensitivity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段119受控攻击时间位置敏感性", "", "在同一站点同一split内将阶段9C扩展攻击事件平移−24、0、+24个观测行，采样间隔约10分钟；每个变体完整重跑阶段9B。", "", "| 平移行数 | F1均值±标准差 | Recall均值±标准差 | AUPRC均值±标准差 | 正常误伤率均值±标准差 | 事件召回均值±标准差 | 恢复后MAE均值±标准差 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['shift_rows']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_false_alarm_rate_mean']:.4f}±{row['test_false_alarm_rate_std']:.4f} | {row['event_recall_mean']:.4f}±{row['event_recall_std']:.4f} | {row['recovery_mae_after_mean']:.4f}±{row['recovery_mae_after_std']:.4f} |")
    lines += ["", "平移遇到边界或事件重叠时的fallback已逐行记录；该阶段只支持位置敏感性分析，不支持真实攻击泛化。"]
    (REPORTS / "stage119_attack_position_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "runs": len(detail), "shifts_rows": SHIFTS, "seeds": SEEDS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
