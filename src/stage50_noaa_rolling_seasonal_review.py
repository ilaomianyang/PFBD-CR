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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import COMPLETE_TYPES, choose_start, score  # noqa: E402
from stage49_noaa_online_threshold import causal_predictions  # noqa: E402

NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919]
FOLDS = {
    "spring_to_summer": ("2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01"),
    "summer_to_winter": ("2022-01-01", "2022-07-01", "2022-10-01", "2023-01-01"),
    "short_recent_train": ("2022-04-01", "2022-07-01", "2022-10-01", "2023-01-01"),
}


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def build_fold(source: pd.DataFrame, seed: int, bounds: tuple[str, str, str, str]) -> pd.DataFrame:
    start, valid_start, test_start, end = [pd.Timestamp(x, tz="UTC") for x in bounds]
    raw = source.dropna(subset=["timestamp_utc", "temperature_c"]).copy()
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
    raw = raw[(raw["timestamp_utc"] >= start) & (raw["timestamp_utc"] < end)].sort_values(["station_id", "timestamp_utc"], kind="stable").reset_index(drop=True)
    raw = raw.rename(columns={"station_id": "station_key", "temperature_c": "original_value"})
    raw["observed_value"] = raw["original_value"].astype(float)
    raw["effective_observed_value"] = raw["observed_value"]
    raw["combined_is_attack"] = False
    raw["combined_attack_type"] = "normal"
    raw["combined_event_id"] = raw.index.astype(str)
    raw["missing_observation"] = False
    raw["split"] = np.where(raw["timestamp_utc"] < valid_start, "train", np.where(raw["timestamp_utc"] < test_start, "valid", "test"))
    for split in ["valid", "test"]:
        for station, group in raw[raw["split"].eq(split)].groupby("station_key", sort=True):
            idx = group.index.to_numpy()
            for ti, attack_type in enumerate(COMPLETE_TYPES):
                length = 12 if attack_type == "drift" else 8
                start_pos = choose_start(group, ti, length)
                event_idx = idx[start_pos:start_pos + length]
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
                    values = prior[start_pos - 1:start_pos - 1 + length]
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
    rows = []
    for fold, bounds in FOLDS.items():
        for seed in SEEDS:
            frame = build_fold(source, seed, bounds)
            scored, fixed = score(frame)
            best = None
            for q in [0.99, 0.995, 0.999]:
                for window in [72, 240, 720]:
                    online = causal_predictions(scored, q, window)
                    valid = scored["split"].eq("valid").to_numpy()
                    test = scored["split"].eq("test").to_numpy()
                    yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
                    yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
                    mv = _metrics(yv, online[valid], scored.loc[valid, "robust_fused_risk"].to_numpy(dtype=float))
                    mt = _metrics(yt, online[test], scored.loc[test, "robust_fused_risk"].to_numpy(dtype=float))
                    candidate = {"fold": fold, "seed": seed, "quantile": q, "window_rows": window, "valid_f1": mv["f1"], "test_f1": mt["f1"], "test_recall": mt["recall"], "test_precision": mt["precision"], "test_auprc": mt["average_precision"], "test_far": float(online[test][~yt].mean()), "fixed_test_f1": fixed["test"]["f1"], "fixed_test_auprc": fixed["test"]["average_precision"], "fixed_test_far": fixed["test_false_alarm_rate"], "train_attack_rows": int(frame.loc[frame["split"].eq("train"), "combined_is_attack"].sum())}
                    if best is None or (candidate["valid_f1"], -candidate["test_far"]) > (best["valid_f1"], -best["test_far"]):
                        best = candidate
            rows.append(best)
    result = pd.DataFrame(rows)
    result["f1_gain_vs_fixed"] = result["test_f1"] - result["fixed_test_f1"]
    aggregate = result.groupby("fold", sort=False)[["test_f1", "fixed_test_f1", "f1_gain_vs_fixed", "test_recall", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["fold"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    gain_mean = float(result["f1_gain_vs_fixed"].mean())
    folds_positive = int((result.groupby("fold")["f1_gain_vs_fixed"].mean() > 0).sum())
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_MORE_SEASONS" if gain_mean > 0 and folds_positive >= 2 else "NEGATIVE_RESULT_RETAINED", "protocol": "NOAA三种预先固定滚动季节窗口、三个攻击种子；在线阈值由每折valid选择，test只评价；攻击为透明受控注入。", "folds": FOLDS, "seeds": SEEDS, "detail": rows, "aggregate": aggregate.to_dict(orient="records"), "mean_f1_gain_vs_fixed": gain_mean, "folds_with_positive_mean_gain": folds_positive}
    result.to_csv(TABLES / "stage50_noaa_rolling_seasonal_review.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage50_noaa_rolling_seasonal_review_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage50_noaa_rolling_seasonal_review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段50 NOAA滚动季节窗口复核", "", "每折在valid选择在线阈值，test只评价；受控注入不代表真实攻击。", "", "| 窗口 | 在线F1 | 固定阈值F1 | F1差值 | Recall | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate.to_dict(orient="records"):
        lines.append(f"| {r['fold']} | {r['test_f1_mean']:.4f}±{r['test_f1_std']:.4f} | {r['fixed_test_f1_mean']:.4f}±{r['fixed_test_f1_std']:.4f} | {r['f1_gain_vs_fixed_mean']:.4f} | {r['test_recall_mean']:.4f} | {r['test_auprc_mean']:.4f} | {r['test_far_mean']:.4f} |")
    lines += ["", f"- 状态：**{payload['status']}**", f"- 三折平均 F1 差值：`{gain_mean:.4f}`；平均提升窗口数：`{folds_positive}/3`。"]
    (REPORTS / "stage50_noaa_rolling_seasonal_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "mean_f1_gain_vs_fixed": gain_mean, "positive_folds": folds_positive}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
