from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402


def robust_scale(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 1.0
    med = float(values.median())
    mad = float((values - med).abs().median())
    return max(1.4826 * mad, 1e-6)


def fit_hierarchical(train: pd.DataFrame, shrink_k: float) -> dict[str, object]:
    normal = train.loc[~train["combined_is_attack"].astype(bool)].copy()
    risk = pd.to_numeric(normal["robust_fused_risk"], errors="coerce")
    normal = normal.loc[risk.notna()].copy()
    normal["_risk"] = risk.loc[normal.index].astype(float)
    global_loc = float(normal["_risk"].median())
    global_scale = robust_scale(normal["_risk"])

    def group_stats(key: pd.Series) -> dict[object, tuple[float, float, int]]:
        out: dict[object, tuple[float, float, int]] = {}
        for group, part in normal.groupby(key.loc[normal.index], sort=True):
            loc = float(part["_risk"].median())
            scale = robust_scale(part["_risk"])
            n = int(len(part))
            weight = n / (n + shrink_k)
            out[group] = (
                weight * loc + (1.0 - weight) * global_loc,
                weight * scale + (1.0 - weight) * global_scale,
                n,
            )
        return out

    station = group_stats(normal["station_key"].astype(str))
    hour = group_stats(normal["timestamp_utc"].dt.hour.astype(int))
    return {"global_loc": global_loc, "global_scale": global_scale, "station": station, "hour": hour}


def calibrated_score(frame: pd.DataFrame, calibration: dict[str, object], hour_weight: float) -> np.ndarray:
    global_loc = float(calibration["global_loc"])
    global_scale = float(calibration["global_scale"])
    station = calibration["station"]
    hour = calibration["hour"]
    risk = frame["robust_fused_risk"].to_numpy(dtype=float)
    stations = frame["station_key"].astype(str).tolist()
    hours = frame["timestamp_utc"].dt.hour.astype(int).tolist()
    station_loc = np.array([station.get(k, (global_loc, global_scale, 0))[0] for k in stations], dtype=float)
    station_scale = np.array([station.get(k, (global_loc, global_scale, 0))[1] for k in stations], dtype=float)
    hour_loc = np.array([hour.get(k, (global_loc, global_scale, 0))[0] for k in hours], dtype=float)
    hour_scale = np.array([hour.get(k, (global_loc, global_scale, 0))[1] for k in hours], dtype=float)
    station_z = (risk - station_loc) / np.maximum(station_scale, 1e-6)
    hour_z = (risk - hour_loc) / np.maximum(hour_scale, 1e-6)
    return (1.0 - hour_weight) * station_z + hour_weight * hour_z


def evaluate_seed(source: pd.DataFrame, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    frame = build_dataset(source, seed)
    scored, reference = score(frame)
    scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
    train = scored["split"].eq("train")
    valid = scored["split"].eq("valid")
    test = scored["split"].eq("test")
    yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    candidates: list[dict[str, object]] = []
    for shrink_k in [20.0, 50.0, 100.0, 200.0, 500.0]:
        calibration = fit_hierarchical(scored.loc[train], shrink_k)
        for hour_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            calibrated = calibrated_score(scored, calibration, hour_weight)
            train_values = calibrated[train.to_numpy()]
            for q in [0.90, 0.925, 0.95, 0.975, 0.99, 0.995]:
                threshold = float(np.nanquantile(train_values, q))
                pv = calibrated[valid.to_numpy()] >= threshold
                pt = calibrated[test.to_numpy()] >= threshold
                mv = _metrics(yv, pv, calibrated[valid.to_numpy()])
                mt = _metrics(yt, pt, calibrated[test.to_numpy()])
                candidates.append({
                    "seed": seed,
                    "shrink_k": shrink_k,
                    "hour_weight": hour_weight,
                    "q": q,
                    "threshold": threshold,
                    "valid_f1": mv["f1"],
                    "valid_far": float(pv[~yv].mean()),
                    "test_f1": mt["f1"],
                    "test_recall": mt["recall"],
                    "test_precision": mt["precision"],
                    "test_auprc": mt["average_precision"],
                    "test_far": float(pt[~yt].mean()),
                    "reference_test_f1": reference["test"]["f1"],
                    "reference_test_auprc": reference["test"]["average_precision"],
                    "reference_test_far": reference["test_false_alarm_rate"],
                })
    best = max(candidates, key=lambda x: (x["valid_f1"], -x["valid_far"]))
    return candidates, best


def main() -> int:
    source = pd.read_parquet(NOAA)
    rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for seed in SEEDS:
        candidates, best = evaluate_seed(source, seed)
        rows.extend(candidates)
        selected.append(best)
        print(json.dumps({"seed": seed, "selected": best, "candidate_count": len(candidates)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected)
    metrics = ["valid_f1", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far", "reference_test_f1", "reference_test_auprc", "reference_test_far"]
    summary = {metric: {"mean": float(selected_df[metric].mean()), "std": float(selected_df[metric].std(ddof=1)), "min": float(selected_df[metric].min()), "max": float(selected_df[metric].max())} for metric in metrics}
    f1_gain = summary["test_f1"]["mean"] - summary["reference_test_f1"]["mean"]
    auprc_gain = summary["test_auprc"]["mean"] - summary["reference_test_auprc"]["mean"]
    status = "CANDIDATE_GAIN_REQUIRES_CROSS_SEASON_REVIEW" if f1_gain > 0 and auprc_gain >= 0 else "NEGATIVE_RESULT_RETAINED"
    payload = {
        "status": status,
        "protocol": "NOAA 12站点、5种子；层级中位数/MAD仅由train正常样本估计；收缩系数、小时权重和分位阈值只在valid选择；test只评价；攻击为透明受控注入。",
        "seeds": SEEDS,
        "grid": {"shrink_k": [20.0, 50.0, 100.0, 200.0, 500.0], "hour_weight": [0.0, 0.25, 0.5, 0.75, 1.0], "q": [0.90, 0.925, 0.95, 0.975, 0.99, 0.995]},
        "selected_by_valid": selected,
        "summary": summary,
        "f1_gain_vs_reference": f1_gain,
        "auprc_gain_vs_reference": auprc_gain,
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage61_noaa_hierarchical_score_calibration_detail.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage61_noaa_hierarchical_score_calibration_selected.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage61_noaa_hierarchical_score_calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段61 NOAA层级风险分数校准多种子复核", "", "全局、站点和小时的风险中位数/MAD只由train正常样本估计；参数只由valid选择，test只评价。", "", f"- 状态：**{status}**", f"- 选中方案平均 test F1：`{summary['test_f1']['mean']:.4f}±{summary['test_f1']['std']:.4f}`；固定参考：`{summary['reference_test_f1']['mean']:.4f}`；差值：`{f1_gain:.4f}`", f"- AUPRC差值：`{auprc_gain:.4f}`；test FAR：`{summary['test_far']['mean']:.4f}±{summary['test_far']['std']:.4f}`", "", "| seed | shrink_k | 小时权重 | q | valid F1 | valid FAR | test F1 | Recall | Precision | AUPRC | FAR |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in selected:
        lines.append(f"| {row['seed']} | {row['shrink_k']:.0f} | {row['hour_weight']:.2f} | {row['q']:.3f} | {row['valid_f1']:.4f} | {row['valid_far']:.4f} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | {row['test_precision']:.4f} | {row['test_auprc']:.4f} | {row['test_far']:.4f} |")
    lines += ["", "该候选只有在独立季节和更多数据源上保持正向，才可考虑进入部署策略；不能自动替换9B主模型。"]
    (REPORTS / "stage61_noaa_hierarchical_score_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary, "f1_gain_vs_reference": f1_gain, "auprc_gain_vs_reference": auprc_gain}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
