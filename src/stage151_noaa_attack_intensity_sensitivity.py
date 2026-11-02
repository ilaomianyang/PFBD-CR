from __future__ import annotations

"""Stage 151: frozen external NOAA attack-intensity sensitivity.

The expanded TrustMeteo-QC candidate is fixed from the existing frozen
external protocol.  Only the threshold is estimated on each amplitude's
validation split; test labels are used solely for evaluation.
"""

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
FROZEN = TABLES / "stage141_noaa_frozen_external_validation_selected.csv"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
AMPLITUDES = [0.5, 1.0, 1.5, 2.0]
ATTACK_TYPES = ["spike", "offset", "drift", "time_shift", "stuck", "gaussian_noise"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import choose_start, score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import stable_seed  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components, candidate_scores  # noqa: E402


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def build_fold_scaled(source: pd.DataFrame, seed: int, train_years: list[int], valid_year: int, test_year: int, stations: set[str], amplitude: float) -> pd.DataFrame:
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
            for type_index, attack_type in enumerate(ATTACK_TYPES):
                length = 12 if attack_type == "drift" else 8
                start = choose_start(group, type_index, length)
                event_idx = idx[start:start + length]
                clean = raw.loc[event_idx, "original_value"].to_numpy(dtype=float)
                rng = np.random.default_rng(stable_seed(seed, *train_years, valid_year, test_year, split_name, station, attack_type))
                if attack_type == "spike":
                    values = clean + 5.0 * amplitude
                elif attack_type == "offset":
                    values = clean + 2.5 * amplitude
                elif attack_type == "drift":
                    values = clean + amplitude * np.linspace(0.2, 2.4, length)
                elif attack_type == "time_shift":
                    prior = group["original_value"].to_numpy(dtype=float)
                    values = prior[start - 1:start - 1 + length]
                elif attack_type == "stuck":
                    values = np.repeat(clean[0], length)
                else:
                    values = clean + rng.normal(0.0, amplitude, size=length)
                raw.loc[event_idx, "observed_value"] = values
                raw.loc[event_idx, "effective_observed_value"] = values
                raw.loc[event_idx, "combined_is_attack"] = True
                raw.loc[event_idx, "combined_attack_type"] = attack_type
                raw.loc[event_idx, "combined_event_id"] = f"{split_name}_{station}_{attack_type}"
    return raw


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    frozen = pd.read_csv(FROZEN)
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = {station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)}
    frozen = frozen[frozen["method"].eq("TrustMeteo-QC frozen expanded")].copy()
    if len(frozen) != len(FOLDS) * len(SEEDS):
        raise RuntimeError("frozen expanded settings are incomplete")
    settings = frozen.set_index(["fold", "seed"])
    rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        name = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            key = (name, seed)
            alpha = float(settings.loc[key, "alpha_base"])
            window = int(settings.loc[key, "window"])
            for amplitude in AMPLITUDES:
                scored, _ = score(build_fold_scaled(source, seed, train_years, valid_year, test_year, stations, amplitude))
                frame = add_components(scored)
                valid_score, test_score, threshold, valid_f1 = candidate_scores(frame, alpha, window)
                valid_mask = frame["split"].eq("valid").to_numpy()
                test_mask = frame["split"].eq("test").to_numpy()
                y_valid = frame.loc[valid_mask, "combined_is_attack"].astype(bool).to_numpy()
                y_test = frame.loc[test_mask, "combined_is_attack"].astype(bool).to_numpy()
                test_pred = test_score >= threshold
                metrics = _metrics(y_test, test_pred, test_score)
                rows.append({"fold": name, "seed": seed, "amplitude": amplitude, "alpha_base_frozen": alpha, "window_frozen": window, "threshold_valid": threshold, "valid_f1": valid_f1, "test_f1": float(metrics["f1"]), "test_recall": float(metrics["recall"]), "test_precision": float(metrics["precision"]), "test_auprc": float(metrics["average_precision"]), "test_false_alarm_rate": float(test_pred[~y_test].mean()) if (~y_test).any() else float("nan")})
                test_part = frame.loc[test_mask].copy()
                for attack_type in ATTACK_TYPES:
                    mask = test_part["combined_attack_type"].eq(attack_type).to_numpy()
                    type_rows.append({"fold": name, "seed": seed, "amplitude": amplitude, "attack_type": attack_type, "rows": int(mask.sum()), "detected_rows": int(test_pred[mask].sum()), "recall": float(test_pred[mask].mean()) if mask.any() else float("nan")})
        print(json.dumps({"fold": name, "amplitudes": len(AMPLITUDES), "seeds": len(SEEDS), "status": "completed"}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(rows)
    type_detail = pd.DataFrame(type_rows)
    aggregate = detail.groupby("amplitude", sort=True)[["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["amplitude"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    type_aggregate = type_detail.groupby(["amplitude", "attack_type"], sort=True)[["rows", "detected_rows", "recall"]].agg(["sum", "mean", "std"]).reset_index()
    type_aggregate.columns = ["amplitude", "attack_type"] + [f"{a}_{b}" for a, b in type_aggregate.columns.tolist()[2:]]
    paired_rows = []
    ref = detail[detail["amplitude"].eq(1.0)].set_index(["fold", "seed"])
    for amplitude in [0.5, 1.5, 2.0]:
        cur = detail[detail["amplitude"].eq(amplitude)].set_index(["fold", "seed"])
        joined = cur.join(ref, lsuffix="_current", rsuffix="_reference")
        for metric in ["test_f1", "test_recall", "test_auprc", "test_false_alarm_rate"]:
            diff = joined[f"{metric}_current"] - joined[f"{metric}_reference"]
            paired_rows.append({"comparison": f"amplitude_{amplitude:g} - amplitude_1.0", "metric": metric, "mean_diff": float(diff.mean()), "fold_seed_pairs": int(len(diff)), "min_diff": float(diff.min()), "max_diff": float(diff.max())})
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020–2025扩展表；11个质量合格站点；四折累计历史训练→前一年valid→目标年test；扩展候选alpha/window沿用阶段141每折每种子的冻结设置，攻击强度网格预先固定为0.5/1.0/1.5/2.0；每个强度只在valid估计阈值，test只评价；不根据test重新选择强度、候选或参数。攻击为透明受控注入，不是真实攻击日志。", "stations": sorted(stations), "folds": [fold_name(*fold) for fold in FOLDS], "seeds": SEEDS, "amplitudes": AMPLITUDES, "detail": detail.to_dict(orient="records"), "aggregate": aggregate.to_dict(orient="records"), "type_detail": type_detail.to_dict(orient="records"), "type_aggregate": type_aggregate.to_dict(orient="records"), "paired": paired_rows, "limitations": ["强度只改变预先定义的数值扰动幅度；time_shift和stuck不随幅度变化，需按类型解释。", "攻击仍为透明受控注入，不是真实网络攻击日志。", "四折和五种子用于不确定性描述，不支持把某一强度写成普适最优。"]}
    prefix = "stage151_noaa_attack_intensity_sensitivity"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    type_detail.to_csv(TABLES / f"{prefix}_type_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / f"{prefix}_aggregate.csv", index=False, encoding="utf-8-sig")
    type_aggregate.to_csv(TABLES / f"{prefix}_type_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段151 NOAA攻击强度敏感性", "", "扩展候选alpha/window沿用阶段141冻结设置；攻击幅度预先固定为0.5/1.0/1.5/2.0，每个强度只用valid估计阈值，test只评价。", "", "| 强度 | test F1 | Recall | Precision | AUPRC | FAR |", "|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['amplitude']:.1f} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_false_alarm_rate_mean']:.4f}±{row['test_false_alarm_rate_std']:.4f} |")
    lines += ["", "time_shift和stuck不随数值幅度变化；所有攻击均为透明受控注入，不能描述为真实攻击日志。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(detail), "type_rows": len(type_detail), "amplitudes": AMPLITUDES}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
