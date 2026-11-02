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

from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402
from stage64_noaa_empirical_percentile_calibration import calibrated_percentile, fit_empirical  # noqa: E402


def main() -> int:
    source = pd.read_parquet(NOAA)
    selected = pd.read_csv(TABLES / "stage64_noaa_empirical_percentile_calibration_selected.csv")
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        choice = selected.loc[selected["seed"].eq(seed)].iloc[0].to_dict()
        frame = build_dataset(source, seed)
        scored, _ = score(frame)
        scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
        train = scored["split"].eq("train")
        calibration = fit_empirical(scored.loc[train])
        calibrated = calibrated_percentile(scored, calibration, float(choice["station_weight"]), float(choice["hour_weight"]))
        test = scored["split"].eq("test")
        natural = test & ~scored["combined_is_attack"].astype(bool)
        quality_flagged = pd.to_numeric(scored["temperature_quality"], errors="coerce").fillna(9).ne(1)
        for group, mask in [("quality_flagged", natural & quality_flagged), ("quality_clean", natural & ~quality_flagged)]:
            values = calibrated[mask.to_numpy()]
            threshold = float(choice["threshold"])
            rows.append({"seed": seed, "group": group, "rows": int(mask.sum()), "mean_calibrated_percentile": float(np.mean(values)) if len(values) else float("nan"), "p95_calibrated_percentile": float(np.quantile(values, 0.95)) if len(values) else float("nan"), "alarm_rate": float(np.mean(values >= threshold)) if len(values) else float("nan"), "threshold": threshold})
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("group", sort=False)[["rows", "mean_calibrated_percentile", "p95_calibrated_percentile", "alarm_rate"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["group"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    flagged = aggregate.loc[aggregate["group"].eq("quality_flagged")].iloc[0]
    clean = aggregate.loc[aggregate["group"].eq("quality_clean")].iloc[0]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "使用阶段64每种子valid选定候选参数；只统计test中未注入的自然观测；NOAA温度质量标记仅作弱标签，不等同攻击真值。", "seeds": SEEDS, "detail": rows, "aggregate": aggregate.to_dict(orient="records"), "weak_label_contrast": {"alarm_rate_difference_flagged_minus_clean": float(flagged["alarm_rate_mean"] - clean["alarm_rate_mean"]), "mean_percentile_difference_flagged_minus_clean": float(flagged["mean_calibrated_percentile_mean"] - clean["mean_calibrated_percentile_mean"])}}
    detail.to_csv(TABLES / "stage67_noaa_candidate_quality_weak_label_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage67_noaa_candidate_quality_weak_label_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage67_noaa_candidate_quality_weak_label.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段67 NOAA候选质量标记弱标签审计", "", "只统计test中未注入的自然观测；质量标记不是攻击真值。", "", "| 分组 | 行数 | 平均经验分位数 | P95经验分位数 | 报警率 |", "|---|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['group']} | {row['rows_mean']:.1f}±{row['rows_std']:.1f} | {row['mean_calibrated_percentile_mean']:.4f}±{row['mean_calibrated_percentile_std']:.4f} | {row['p95_calibrated_percentile_mean']:.4f}±{row['p95_calibrated_percentile_std']:.4f} | {row['alarm_rate_mean']:.4f}±{row['alarm_rate_std']:.4f} |")
    lines += ["", f"质量可疑组与正常组报警率差值：`{payload['weak_label_contrast']['alarm_rate_difference_flagged_minus_clean']:.4f}`。", "该差值只能作为自然观测质量风险的弱标签对比，不能写成攻击检测召回率或真实入侵识别率。"]
    (REPORTS / "stage67_noaa_candidate_quality_weak_label.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "alarm_rate_difference": payload["weak_label_contrast"]["alarm_rate_difference_flagged_minus_clean"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
