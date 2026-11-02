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


def fit_empirical(train: pd.DataFrame) -> dict[str, object]:
    normal = train.loc[~train["combined_is_attack"].astype(bool)].copy()
    risk = pd.to_numeric(normal["robust_fused_risk"], errors="coerce")
    normal = normal.loc[risk.notna()].copy()
    normal["_risk"] = risk.loc[normal.index].astype(float)

    def arrays(key: pd.Series) -> dict[object, np.ndarray]:
        return {group: np.sort(part["_risk"].to_numpy(dtype=float)) for group, part in normal.groupby(key.loc[normal.index], sort=True)}

    return {
        "global": np.sort(normal["_risk"].to_numpy(dtype=float)),
        "station": arrays(normal["station_key"].astype(str)),
        "hour": arrays(normal["timestamp_utc"].dt.hour.astype(int)),
    }


def empirical_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(reference) == 0:
        return np.full(len(values), 0.5, dtype=float)
    ranks = np.searchsorted(reference, values, side="right")
    return ranks.astype(float) / float(len(reference))


def calibrated_percentile(frame: pd.DataFrame, calibration: dict[str, object], station_weight: float, hour_weight: float) -> np.ndarray:
    values = frame["robust_fused_risk"].to_numpy(dtype=float)
    global_ref = calibration["global"]
    station_refs = calibration["station"]
    hour_refs = calibration["hour"]
    global_pct = empirical_percentile(values, global_ref)
    station_pct = np.array([empirical_percentile(np.array([v]), station_refs.get(k, global_ref))[0] for v, k in zip(values, frame["station_key"].astype(str))], dtype=float)
    hour_pct = np.array([empirical_percentile(np.array([v]), hour_refs.get(k, global_ref))[0] for v, k in zip(values, frame["timestamp_utc"].dt.hour.astype(int))], dtype=float)
    global_weight = max(0.0, 1.0 - station_weight - hour_weight)
    return global_weight * global_pct + station_weight * station_pct + hour_weight * hour_pct


def evaluate_seed(source: pd.DataFrame, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    frame = build_dataset(source, seed)
    scored, reference = score(frame)
    scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
    train = scored["split"].eq("train")
    valid = scored["split"].eq("valid")
    test = scored["split"].eq("test")
    yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    calibration = fit_empirical(scored.loc[train])
    candidates: list[dict[str, object]] = []
    weights = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (0.0, 0.25), (0.0, 0.5), (0.0, 0.75), (0.0, 1.0), (0.25, 0.25), (0.5, 0.25), (0.25, 0.5)]
    for station_weight, hour_weight in weights:
        calibrated = calibrated_percentile(scored, calibration, station_weight, hour_weight)
        for q in [0.90, 0.925, 0.95, 0.975, 0.99, 0.995]:
            threshold = float(np.nanquantile(calibrated[train.to_numpy()], q))
            pv = calibrated[valid.to_numpy()] >= threshold
            pt = calibrated[test.to_numpy()] >= threshold
            mv = _metrics(yv, pv, calibrated[valid.to_numpy()])
            mt = _metrics(yt, pt, calibrated[test.to_numpy()])
            candidates.append({
                "seed": seed,
                "station_weight": station_weight,
                "hour_weight": hour_weight,
                "global_weight": max(0.0, 1.0 - station_weight - hour_weight),
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
                "reference_test_recall": reference["test"]["recall"],
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
    metrics = ["valid_f1", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far", "reference_test_f1", "reference_test_recall", "reference_test_auprc", "reference_test_far"]
    summary = {metric: {"mean": float(selected_df[metric].mean()), "std": float(selected_df[metric].std(ddof=1)), "min": float(selected_df[metric].min()), "max": float(selected_df[metric].max())} for metric in metrics}
    f1_gain = summary["test_f1"]["mean"] - summary["reference_test_f1"]["mean"]
    auprc_gain = summary["test_auprc"]["mean"] - summary["reference_test_auprc"]["mean"]
    status = "CANDIDATE_GAIN_REQUIRES_CROSS_SEASON_REVIEW" if f1_gain > 0 and auprc_gain >= 0 else "NEGATIVE_RESULT_RETAINED"
    payload = {"status": status, "protocol": "NOAA 12站点、5种子；经验分位数参考分布只由train正常样本估计；站点/小时/全局权重和阈值只在valid选择；test只评价；攻击为透明受控注入。", "seeds": SEEDS, "grid": {"weights": [[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0], [0.0, 0.25], [0.0, 0.5], [0.0, 0.75], [0.0, 1.0], [0.25, 0.25], [0.5, 0.25], [0.25, 0.5]], "q": [0.90, 0.925, 0.95, 0.975, 0.99, 0.995]}, "selected_by_valid": selected, "summary": summary, "f1_gain_vs_reference": f1_gain, "auprc_gain_vs_reference": auprc_gain}
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage64_noaa_empirical_percentile_calibration_detail.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage64_noaa_empirical_percentile_calibration_selected.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage64_noaa_empirical_percentile_calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段64 NOAA经验分位数校准多种子复核", "", "经验分位数参考分布只由train正常样本估计；权重和阈值只在valid选择，test只评价。", "", f"- 状态：**{status}**", f"- 选中方案平均 test F1：`{summary['test_f1']['mean']:.4f}±{summary['test_f1']['std']:.4f}`；固定参考：`{summary['reference_test_f1']['mean']:.4f}`；差值：`{f1_gain:.4f}`", f"- AUPRC差值：`{auprc_gain:.4f}`；test FAR：`{summary['test_far']['mean']:.4f}±{summary['test_far']['std']:.4f}`", "", "| seed | 站点权重 | 小时权重 | 全局权重 | q | valid F1 | test F1 | Recall | Precision | AUPRC | FAR |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in selected:
        lines.append(f"| {row['seed']} | {row['station_weight']:.2f} | {row['hour_weight']:.2f} | {row['global_weight']:.2f} | {row['q']:.3f} | {row['valid_f1']:.4f} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | {row['test_precision']:.4f} | {row['test_auprc']:.4f} | {row['test_far']:.4f} |")
    lines += ["", "若F1和AUPRC不能同时稳定优于固定参考，该候选只作为低误报/AUPRC诊断，不替换9B主模型。"]
    (REPORTS / "stage64_noaa_empirical_percentile_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary, "f1_gain_vs_reference": f1_gain, "auprc_gain_vs_reference": auprc_gain}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
