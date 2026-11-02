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
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"
AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [([2020], 2021, 2022), ([2020, 2021], 2022, 2023), ([2020, 2021, 2022], 2023, 2024), ([2020, 2021, 2022, 2023], 2024, 2025)]
COMPLETE_TYPES = ["spike", "offset", "drift", "time_shift", "stuck", "gaussian_noise"]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import choose_start, score  # noqa: E402


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def build_fold(source: pd.DataFrame, seed: int, train_years: list[int], valid_year: int, test_year: int, stations: set[str]) -> pd.DataFrame:
    all_years = [*train_years, valid_year, test_year]
    raw = source[source["station_id"].astype(str).isin(stations) & source["year"].isin(all_years)].copy()
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
    split_map = {year: "train" for year in train_years}
    split_map.update({valid_year: "valid", test_year: "test"})
    raw["split"] = raw["year"].map(split_map)
    if raw["split"].isna().any():
        raise ValueError("unexpected year in rolling fold")
    for split_name in ["valid", "test"]:
        for station, group in raw[raw["split"].eq(split_name)].groupby("station_key", sort=True):
            idx = group.index.to_numpy()
            for type_index, attack_type in enumerate(COMPLETE_TYPES):
                length = 12 if attack_type == "drift" else 8
                start = choose_start(group, type_index, length)
                event_idx = idx[start:start + length]
                clean = raw.loc[event_idx, "original_value"].to_numpy(dtype=float)
                rng = np.random.default_rng(stable_seed(seed, *train_years, valid_year, test_year, split_name, station, attack_type))
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
                raw.loc[event_idx, "combined_event_id"] = f"{split_name}_{station}_{attack_type}"
    return raw


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    if len(stations) < 2:
        raise RuntimeError("fewer than two stations have all extended years eligible")
    rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    integrity: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, result = score(frame)
            valid = scored["split"].eq("valid").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            pred = scored["predicted_attack"].astype(bool).to_numpy()
            integrity.append({"train_years": train_years, "valid_year": valid_year, "test_year": test_year, "seed": seed, "rows": int(len(frame)), "stations": int(frame["station_key"].nunique()), "train_attack_rows": int(frame.loc[frame["split"].eq("train"), "combined_is_attack"].sum()), "valid_attack_rows": int(frame.loc[frame["split"].eq("valid"), "combined_is_attack"].sum()), "test_attack_rows": int(frame.loc[frame["split"].eq("test"), "combined_is_attack"].sum())})
            rows.append({"train_years": "-".join(map(str, train_years)), "valid_year": valid_year, "test_year": test_year, "seed": seed, "valid_f1": result["valid"]["f1"], "test_f1": result["test"]["f1"], "test_recall": result["test"]["recall"], "test_precision": result["test"]["precision"], "test_auprc": result["test"]["average_precision"], "test_far": result["test_false_alarm_rate"], "threshold": result["setting"]["threshold"]})
            for attack_type in COMPLETE_TYPES:
                mask = test & scored["combined_attack_type"].eq(attack_type).to_numpy()
                part = pred[mask]
                type_rows.append({"train_years": "-".join(map(str, train_years)), "valid_year": valid_year, "test_year": test_year, "seed": seed, "attack_type": attack_type, "rows": int(mask.sum()), "detected_rows": int(part.sum()), "recall": float(part.mean()) if len(part) else float("nan")})
        print(json.dumps({"fold": [train_years, valid_year, test_year], "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)
    summary_df = pd.DataFrame(rows)
    type_df = pd.DataFrame(type_rows)
    fold_aggregate = summary_df.groupby(["train_years", "valid_year", "test_year"], sort=False).agg(seeds=("seed", "nunique"), test_f1_mean=("test_f1", "mean"), test_f1_std=("test_f1", "std"), test_recall_mean=("test_recall", "mean"), test_precision_mean=("test_precision", "mean"), test_auprc_mean=("test_auprc", "mean"), test_far_mean=("test_far", "mean")).reset_index()
    overall = {column: {"mean": float(summary_df[column].mean()), "std": float(summary_df[column].std(ddof=1)), "min": float(summary_df[column].min()), "max": float(summary_df[column].max())} for column in ["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]}
    type_aggregate = type_df.groupby(["test_year", "attack_type"], sort=False)["recall"].agg(["mean", "std", "min", "max"]).reset_index()
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020-2025扩展表；11个所有年度均通过质量审计的站点；滚动协议为累计历史训练→前一年验证→目标年测试，目标年依次为2022/2023/2024/2025；5个种子；train无攻击；valid/test攻击为透明受控注入。", "stations": stations, "folds": [{"train_years": train_years, "valid_year": valid_year, "test_year": test_year} for train_years, valid_year, test_year in FOLDS if set([*train_years, valid_year, test_year]).issubset(years)], "seeds": SEEDS, "overall": overall, "fold_aggregate": fold_aggregate.to_dict(orient="records"), "by_type": type_aggregate.to_dict(orient="records"), "integrity": integrity}
    summary_df.to_csv(TABLES / "stage84_noaa_rolling_multiyear_benchmark.csv", index=False, encoding="utf-8-sig")
    type_df.to_csv(TABLES / "stage84_noaa_rolling_multiyear_by_type.csv", index=False, encoding="utf-8-sig")
    fold_aggregate.to_csv(TABLES / "stage84_noaa_rolling_multiyear_by_fold.csv", index=False, encoding="utf-8-sig")
    type_aggregate.to_csv(TABLES / "stage84_noaa_rolling_multiyear_by_type_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage84_noaa_rolling_multiyear_benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段84 NOAA滚动多年度受控攻击基准", "", "使用累计历史训练→前一年验证→目标年测试，目标年为2022、2023、2024、2025；所有参数按每折valid选择，test只评价。", "", f"- 站点：`{len(stations)}`；折数：`{len(payload['folds'])}`；种子：`{len(SEEDS)}`", f"- 总体test F1：`{overall['test_f1']['mean']:.4f}±{overall['test_f1']['std']:.4f}`；Recall：`{overall['test_recall']['mean']:.4f}`；AUPRC：`{overall['test_auprc']['mean']:.4f}`；FAR：`{overall['test_far']['mean']:.4f}`", "", "| 训练年份→验证→测试 | F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in fold_aggregate.to_dict(orient="records"):
        lines.append(f"| {row['train_years']}→{row['valid_year']}→{row['test_year']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f} | {row['test_precision_mean']:.4f} | {row['test_auprc_mean']:.4f} | {row['test_far_mean']:.4f} |")
    lines += ["", "该阶段只支持时间外推下的稳定性判断；攻击由透明受控注入产生，不是自然网络攻击日志。"]
    (REPORTS / "stage84_noaa_rolling_multiyear_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(stations), "folds": len(payload["folds"]), "overall": overall}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
