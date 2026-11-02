from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
DEFAULT_SCORES = TABLES / "stage09b_local_robust_scores.parquet"

COMPONENTS = [
    "ERA5 Background",
    "Physical Rules",
    "Hampel MAD",
    "Kalman Innovation",
    "CUSUM",
    "Spatial Consistency",
    "base_fused_risk",
    "structural_risk",
    "robust_fused_risk",
]


def q(series: pd.Series, p: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(p)) if len(values) else float("nan")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段35：liupanshan_meteo误报来源诊断")
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    args = parser.parse_args()
    df = pd.read_parquet(args.scores).copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["is_attack"] = df["combined_is_attack"].astype(bool)
    df["is_fp"] = (~df["is_attack"]) & df["predicted_attack"].astype(bool)
    df["is_fn"] = df["is_attack"] & (~df["predicted_attack"].astype(bool))
    df["hour_utc"] = df["timestamp_utc"].dt.hour
    df["date_utc"] = df["timestamp_utc"].dt.date.astype(str)
    df["month_utc"] = df["timestamp_utc"].dt.month

    target = df[(df["station_key"] == "liupanshan_meteo") & (df["split"] == "test")].copy()
    if target.empty:
        raise SystemExit("liupanshan_meteo test rows not found")
    clean = target[~target["is_attack"]]
    fp = target[target["is_fp"]]
    clean_nonfp = clean[~clean["predicted_attack"]]

    # Compare detector components for false positives against clean, correctly
    # accepted rows; this is descriptive and never changes the locked result.
    comp_rows = []
    for component in COMPONENTS:
        comp_rows.append({
            "component": component,
            "clean_nonfp_mean": float(pd.to_numeric(clean_nonfp[component], errors="coerce").mean()),
            "clean_nonfp_p95": q(clean_nonfp[component], 0.95),
            "false_positive_mean": float(pd.to_numeric(fp[component], errors="coerce").mean()),
            "false_positive_p95": q(fp[component], 0.95),
            "mean_ratio_fp_to_clean": float(pd.to_numeric(fp[component], errors="coerce").mean() / max(pd.to_numeric(clean_nonfp[component], errors="coerce").mean(), 1e-12)),
        })
    components = pd.DataFrame(comp_rows)

    by_hour = target.groupby("hour_utc", as_index=False).agg(
        rows=("sample_id", "size"),
        clean_rows=("is_attack", lambda s: int((~s).sum())),
        false_positive_rows=("is_fp", "sum"),
        attack_rows=("is_attack", "sum"),
        mean_risk=("robust_fused_risk", "mean"),
    )
    by_hour["false_alarm_rate"] = by_hour["false_positive_rows"] / by_hour["clean_rows"].replace(0, np.nan)
    by_day = target.groupby("date_utc", as_index=False).agg(
        rows=("sample_id", "size"),
        clean_rows=("is_attack", lambda s: int((~s).sum())),
        false_positive_rows=("is_fp", "sum"),
        attack_rows=("is_attack", "sum"),
        mean_risk=("robust_fused_risk", "mean"),
    )
    by_day["false_alarm_rate"] = by_day["false_positive_rows"] / by_day["clean_rows"].replace(0, np.nan)

    # Station-level reference: compare normal test risk quantiles and ERA5 bias.
    station_rows = []
    normal_test = df[(df["split"] == "test") & (~df["is_attack"])].copy()
    for station, g in normal_test.groupby("station_key", sort=True):
        station_rows.append({
            "station_key": str(station),
            "rows": int(len(g)),
            "risk_mean": float(pd.to_numeric(g["robust_fused_risk"], errors="coerce").mean()),
            "risk_p95": q(g["robust_fused_risk"], 0.95),
            "risk_p99": q(g["robust_fused_risk"], 0.99),
            "era5_background_mean": float(pd.to_numeric(g["ERA5 Background"], errors="coerce").mean()),
            "era5_background_p95": q(g["ERA5 Background"], 0.95),
            "false_alarm_rate": float(g["predicted_attack"].astype(bool).mean()),
        })
    station_ref = pd.DataFrame(station_rows).sort_values("false_alarm_rate", ascending=False)

    summary = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "只使用锁定的stage09b test分数做描述性诊断；不重新选择阈值，不修改主模型结果。",
        "target_station": "liupanshan_meteo",
        "target_rows": int(len(target)),
        "clean_rows": int(len(clean)),
        "false_positive_rows": int(len(fp)),
        "attack_rows": int(target["is_attack"].sum()),
        "detected_attack_rows": int((target["is_attack"] & target["predicted_attack"].astype(bool)).sum()),
        "missed_attack_rows": int((target["is_fn"]).sum()),
        "false_alarm_rate": float(fp.shape[0] / max(clean.shape[0], 1)),
        "false_positive_top_hours": by_hour.sort_values("false_positive_rows", ascending=False).head(10).to_dict(orient="records"),
        "false_positive_top_days": by_day.sort_values("false_positive_rows", ascending=False).head(10).to_dict(orient="records"),
        "interpretation": [
            "若误报集中于少数小时/日期，优先验证时间分布偏移和站点校准。",
            "若ERA5 Background或Spatial Consistency在误报行显著升高，优先检查区域背景偏差或邻站不一致。",
            "本阶段不据诊断结果宣称改进，后续候选必须只在valid选择并在locked test复核。",
        ],
    }
    components.to_csv(TABLES / "stage35_liupanshan_component_diagnosis.csv", index=False, encoding="utf-8-sig")
    by_hour.to_csv(TABLES / "stage35_liupanshan_fp_by_hour.csv", index=False, encoding="utf-8-sig")
    by_day.to_csv(TABLES / "stage35_liupanshan_fp_by_day.csv", index=False, encoding="utf-8-sig")
    station_ref.to_csv(TABLES / "stage35_station_normal_risk_reference.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage35_liupanshan_fp_diagnosis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")

    lines = [
        "# 阶段35 liupanshan_meteo误报来源诊断",
        "",
        "该阶段只描述锁定test，不重新调参、不修改主模型。",
        "",
        "## 汇总",
        "",
        f"- test行数：{len(target)}；正常行：{len(clean)}；攻击行：{int(target['is_attack'].sum())}",
        f"- 误报行：{len(fp)}；误伤率：{summary['false_alarm_rate']:.4f}",
        f"- 检出攻击行：{summary['detected_attack_rows']}；漏报攻击行：{summary['missed_attack_rows']}",
        "",
        "## 风险分量比较",
        "",
        "| 分量 | 正常未报警均值 | 正常未报警P95 | 误报均值 | 误报P95 | 均值比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in components.iterrows():
        lines.append(f"| {r['component']} | {r['clean_nonfp_mean']:.4f} | {r['clean_nonfp_p95']:.4f} | {r['false_positive_mean']:.4f} | {r['false_positive_p95']:.4f} | {r['mean_ratio_fp_to_clean']:.3f} |")
    lines += ["", "## 误报集中小时（前10）", "", "| UTC小时 | 正常行 | 误报行 | 误伤率 |", "|---:|---:|---:|---:|"]
    for _, r in by_hour.sort_values("false_positive_rows", ascending=False).head(10).iterrows():
        lines.append(f"| {int(r['hour_utc']):02d} | {int(r['clean_rows'])} | {int(r['false_positive_rows'])} | {r['false_alarm_rate']:.4f} |")
    lines += ["", "完整分量、日期和站点参照表见 `tables/stage35_*.csv`。"]
    (REPORTS / "stage35_liupanshan_fp_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "target_rows": len(target), "false_positive_rows": len(fp), "false_alarm_rate": summary["false_alarm_rate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
