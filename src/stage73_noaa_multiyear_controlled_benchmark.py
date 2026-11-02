from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage70_noaa_isd_multiyear_normalized.parquet"
AUDIT = TABLES / "stage71_noaa_multiyear_quality_audit.json"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
COMPLETE_TYPES = ["spike", "offset", "drift", "time_shift", "stuck", "gaussian_noise"]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import choose_start, score  # noqa: E402


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def build_dataset(source: pd.DataFrame, seed: int, stations: set[str]) -> pd.DataFrame:
    raw = source[source["station_id"].astype(str).isin(stations)].copy()
    raw = raw.dropna(subset=["timestamp_utc", "temperature_c"])
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
    raw = raw.sort_values(["station_id", "timestamp_utc"], kind="stable").reset_index(drop=True)
    raw = raw.rename(columns={"station_id": "station_key", "temperature_c": "original_value"})
    raw["observed_value"] = raw["original_value"].astype(float)
    raw["effective_observed_value"] = raw["observed_value"]
    raw["combined_is_attack"] = False
    raw["combined_attack_type"] = "normal"
    raw["combined_event_id"] = raw.index.astype(str)
    raw["missing_observation"] = False
    raw["split"] = raw["year"].map({2021: "train", 2022: "valid", 2023: "test"})
    if raw["split"].isna().any():
        raise ValueError("unexpected year outside 2021-2023")
    for split in ["valid", "test"]:
        for station, group in raw[raw["split"].eq(split)].groupby("station_key", sort=True):
            idx = group.index.to_numpy()
            for type_index, attack_type in enumerate(COMPLETE_TYPES):
                length = 12 if attack_type == "drift" else 8
                start = choose_start(group, type_index, length)
                event_idx = idx[start:start + length]
                clean = raw.loc[event_idx, "original_value"].to_numpy(dtype=float)
                rng = np.random.default_rng(stable_seed(seed, split, station, attack_type))
                if attack_type == "spike":
                    values = clean + 5.0
                elif attack_type == "offset":
                    values = clean + 2.5
                elif attack_type == "drift":
                    values = clean + np.linspace(0.2, 2.4, length)
                elif attack_type == "time_shift":
                    prior = group["original_value"].to_numpy(dtype=float)
                    values = prior[start - 1:start - 1 + length]
                elif attack_type == "stuck":
                    values = np.repeat(clean[0], length)
                else:
                    values = clean + rng.normal(0.0, 1.0, size=length)
                raw.loc[event_idx, "observed_value"] = values
                raw.loc[event_idx, "effective_observed_value"] = values
                raw.loc[event_idx, "combined_is_attack"] = True
                raw.loc[event_idx, "combined_attack_type"] = attack_type
                raw.loc[event_idx, "combined_event_id"] = f"{split}_{station}_{attack_type}"
    return raw


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_multiyear_validation")}
    stations = {station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in [2021, 2022, 2023])}
    if len(stations) < 2:
        raise RuntimeError("fewer than two stations have all three eligible years")
    source = source[source["station_id"].astype(str).isin(stations) & source["year"].isin([2021, 2022, 2023])].copy()
    summaries: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    integrity: list[dict[str, object]] = []
    for seed in SEEDS:
        frame = build_dataset(source, seed, stations)
        train_attack_rows = int(frame.loc[frame["split"].eq("train"), "combined_is_attack"].sum())
        integrity.append({"seed": seed, "train_attack_rows": train_attack_rows, "rows": int(len(frame)), "stations": int(frame["station_key"].nunique()), "valid_attack_rows": int((frame["split"].eq("valid") & frame["combined_is_attack"]).sum()), "test_attack_rows": int((frame["split"].eq("test") & frame["combined_is_attack"]).sum())})
        scored, result = score(frame)
        summaries.append({"seed": seed, "valid_f1": result["valid"]["f1"], "test_f1": result["test"]["f1"], "test_recall": result["test"]["recall"], "test_precision": result["test"]["precision"], "test_auprc": result["test"]["average_precision"], "test_false_alarm_rate": result["test_false_alarm_rate"], "threshold": result["setting"]["threshold"]})
        test_mask = scored["split"].eq("test")
        for attack_type in COMPLETE_TYPES:
            mask = test_mask & scored["combined_attack_type"].eq(attack_type)
            pred = scored.loc[mask, "predicted_attack"].astype(bool).to_numpy()
            type_rows.append({"seed": seed, "attack_type": attack_type, "rows": int(mask.sum()), "detected_rows": int(pred.sum()), "recall": float(pred.mean()) if len(pred) else float("nan")})
    summary_df = pd.DataFrame(summaries)
    type_df = pd.DataFrame(type_rows)
    aggregate = {metric: {"mean": float(summary_df[metric].mean()), "std": float(summary_df[metric].std(ddof=1)), "min": float(summary_df[metric].min()), "max": float(summary_df[metric].max())} for metric in ["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]}
    type_aggregate = type_df.groupby("attack_type", sort=False)[["recall"]].agg(["mean", "std", "min", "max"]).reset_index()
    type_aggregate.columns = ["attack_type", "recall_mean", "recall_std", "recall_min", "recall_max"]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方三年度数据；2021训练、2022验证、2023锁定测试；仅11个三年均通过质量审计的站点；5个攻击种子；train始终无攻击；攻击为透明受控注入。", "stations": sorted(stations), "excluded_stations": sorted(set(source["station_id"].astype(str).unique()) - stations), "seeds": SEEDS, "aggregate": aggregate, "per_seed": summaries, "by_type": type_rows, "by_type_aggregate": type_aggregate.to_dict(orient="records"), "integrity": integrity}
    summary_df.to_csv(TABLES / "stage73_noaa_multiyear_controlled_benchmark.csv", index=False, encoding="utf-8-sig")
    type_df.to_csv(TABLES / "stage73_noaa_multiyear_controlled_by_type.csv", index=False, encoding="utf-8-sig")
    type_aggregate.to_csv(TABLES / "stage73_noaa_multiyear_controlled_by_type_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage73_noaa_multiyear_controlled_benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段73 NOAA多年度受控攻击基准", "", "2021训练、2022验证、2023锁定测试；只使用三年质量审计均通过的站点；攻击为透明受控注入。", "", f"- 站点：`{len(stations)}`；排除站点：`{len(payload['excluded_stations'])}`；种子：`{len(SEEDS)}`；训练攻击行：`{sorted(set(x['train_attack_rows'] for x in integrity))}`", f"- Test F1/Recall/Precision/AUPRC/FAR：`{aggregate['test_f1']['mean']:.4f}±{aggregate['test_f1']['std']:.4f}` / `{aggregate['test_recall']['mean']:.4f}` / `{aggregate['test_precision']['mean']:.4f}` / `{aggregate['test_auprc']['mean']:.4f}` / `{aggregate['test_false_alarm_rate']['mean']:.4f}`", "", "| 攻击类型 | Recall |", "|---|---:|"]
    for row in type_aggregate.to_dict(orient="records"):
        lines.append(f"| {row['attack_type']} | {row['recall_mean']:.4f}±{row['recall_std']:.4f} |")
    lines += ["", "该结果用于跨年份外部受控协议验证；受控攻击不是真实攻击日志，不能替代自然观测质量审计。"]
    (REPORTS / "stage73_noaa_multiyear_controlled_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(stations), "seeds": len(SEEDS), "aggregate": aggregate}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
