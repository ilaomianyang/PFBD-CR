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


def threshold_tables(train: pd.DataFrame, q: float) -> tuple[pd.Series, pd.Series, float]:
    normal = train[~train["combined_is_attack"].astype(bool)]
    station = normal.groupby(normal["station_key"].astype(str))["robust_fused_risk"].quantile(q)
    hour = normal.groupby(normal["timestamp_utc"].dt.hour)["robust_fused_risk"].quantile(q)
    global_threshold = float(normal["robust_fused_risk"].quantile(q))
    return station, hour, global_threshold


def apply_thresholds(frame: pd.DataFrame, station: pd.Series, hour: pd.Series, global_threshold: float, hour_weight: float) -> np.ndarray:
    station_threshold = frame["station_key"].astype(str).map(station).fillna(global_threshold).to_numpy(dtype=float)
    hour_threshold = frame["timestamp_utc"].dt.hour.map(hour).fillna(global_threshold).to_numpy(dtype=float)
    threshold = (1.0 - hour_weight) * station_threshold + hour_weight * hour_threshold
    return frame["robust_fused_risk"].to_numpy(dtype=float) >= threshold


def main() -> int:
    source = pd.read_parquet(NOAA)
    rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for seed in SEEDS:
        frame = build_dataset(source, seed)
        scored, reference = score(frame)
        scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
        train = scored["split"].eq("train")
        valid = scored["split"].eq("valid")
        test = scored["split"].eq("test")
        yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
        yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
        candidates: list[dict[str, object]] = []
        for q in [0.90, 0.925, 0.95, 0.975, 0.99, 0.995]:
            station, hour, global_threshold = threshold_tables(scored.loc[train], q)
            for hour_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
                pv = apply_thresholds(scored.loc[valid], station, hour, global_threshold, hour_weight)
                pt = apply_thresholds(scored.loc[test], station, hour, global_threshold, hour_weight)
                mv = _metrics(yv, pv, scored.loc[valid, "robust_fused_risk"].to_numpy(dtype=float))
                mt = _metrics(yt, pt, scored.loc[test, "robust_fused_risk"].to_numpy(dtype=float))
                candidates.append({"seed": seed, "q": q, "hour_weight": hour_weight, "valid_f1": mv["f1"], "valid_far": float(pv[~yv].mean()), "test_f1": mt["f1"], "test_recall": mt["recall"], "test_precision": mt["precision"], "test_auprc": mt["average_precision"], "test_far": float(pt[~yt].mean()), "reference_test_f1": reference["test"]["f1"], "reference_test_auprc": reference["test"]["average_precision"], "reference_test_far": reference["test_false_alarm_rate"]})
        best = max(candidates, key=lambda x: (x["valid_f1"], -x["valid_far"]))
        selected.append(best)
        rows.extend(candidates)
    detail = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected)
    summary = {metric: {"mean": float(selected_df[metric].mean()), "std": float(selected_df[metric].std(ddof=1)), "min": float(selected_df[metric].min()), "max": float(selected_df[metric].max())} for metric in ["valid_f1", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far", "reference_test_f1", "reference_test_auprc", "reference_test_far"]}
    f1_gain = summary["test_f1"]["mean"] - summary["reference_test_f1"]["mean"]
    auprc_gain = summary["test_auprc"]["mean"] - summary["reference_test_auprc"]["mean"]
    status = "CANDIDATE_GAIN_REQUIRES_CROSS_SEASON_REVIEW" if f1_gain > 0 and auprc_gain >= 0 else "NEGATIVE_RESULT_RETAINED"
    payload = {"status": status, "protocol": "NOAA 12站点、5种子；q和站点/小时收缩比例只在valid选择；阈值只由train正常样本估计；test只评价；攻击为透明受控注入。", "seeds": SEEDS, "grid": {"q": [0.90, 0.925, 0.95, 0.975, 0.99, 0.995], "hour_weight": [0.0, 0.25, 0.5, 0.75, 1.0]}, "selected_by_valid": selected, "summary": summary, "f1_gain_vs_reference": f1_gain, "auprc_gain_vs_reference": auprc_gain}
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage60_noaa_station_threshold_multiseed_detail.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage60_noaa_station_threshold_multiseed_selected.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage60_noaa_station_threshold_multiseed.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段60 NOAA站点阈值多种子复核", "", "阈值只由train正常样本估计，q和站点/小时收缩比例只在valid选择，test只评价。", "", f"- 状态：**{status}**", f"- 选中方案平均 test F1：`{summary['test_f1']['mean']:.4f}±{summary['test_f1']['std']:.4f}`；参考：`{summary['reference_test_f1']['mean']:.4f}`；F1差值：`{f1_gain:.4f}`", f"- AUPRC差值：`{auprc_gain:.4f}`；test FAR：`{summary['test_far']['mean']:.4f}±{summary['test_far']['std']:.4f}`", "", "| seed | q | 小时权重 | valid F1 | valid FAR | test F1 | Recall | Precision | AUPRC | FAR |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in selected:
        lines.append(f"| {row['seed']} | {row['q']:.3f} | {row['hour_weight']:.2f} | {row['valid_f1']:.4f} | {row['valid_far']:.4f} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | {row['test_precision']:.4f} | {row['test_auprc']:.4f} | {row['test_far']:.4f} |")
    lines += ["", "单种子候选只有在多种子和更多独立季节中保持正向，才可考虑进入部署策略；不能替换9B主模型。"]
    (REPORTS / "stage60_noaa_station_threshold_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary, "f1_gain_vs_reference": f1_gain, "auprc_gain_vs_reference": auprc_gain}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
