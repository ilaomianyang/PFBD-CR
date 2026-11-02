from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
DETAIL = TABLES / "stage93_noaa_natural_component_validation_station_detail.csv"
BOOTSTRAPS = 5000
RNG_SEED = 20260914


def ci(values: np.ndarray) -> list[float]:
    return [float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))]


def main() -> int:
    detail = pd.read_csv(DETAIL)
    detail["station_key"] = detail["station_key"].astype(str)
    stations = sorted(detail["station_key"].astype(str).unique())
    years = [int(value) for value in sorted(detail["target_year"].astype(int).unique())]
    qs = [float(value) for value in sorted(detail["q"].astype(float).unique())]
    rng = np.random.default_rng(RNG_SEED)
    summary_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for year in years:
        for q in qs:
            subset = detail[detail["target_year"].eq(year) & detail["q"].eq(q)].copy()
            pivot = subset.pivot(index="station_key", columns="method")
            pivot = pivot.reindex(stations)
            contrast_valid = pivot[[
                ("quality_flagged_alarm_rate", "legacy"),
                ("quality_clean_alarm_rate", "legacy"),
                ("quality_flagged_alarm_rate", "component_zmax"),
                ("quality_clean_alarm_rate", "component_zmax"),
            ]].notna().all(axis=1)
            contrast_stations = pivot.index[contrast_valid].astype(str).tolist()
            alarm_diff = []
            contrast_diff = []
            for b in range(BOOTSTRAPS):
                sampled = rng.choice(stations, size=len(stations), replace=True)
                part = pivot.loc[sampled]
                legacy_alarm = part[("alarm_rate", "legacy")].to_numpy(dtype=float)
                candidate_alarm = part[("alarm_rate", "component_zmax")].to_numpy(dtype=float)
                legacy_contrast = (part[("quality_flagged_alarm_rate", "legacy")] - part[("quality_clean_alarm_rate", "legacy")]).to_numpy(dtype=float)
                candidate_contrast = (part[("quality_flagged_alarm_rate", "component_zmax")] - part[("quality_clean_alarm_rate", "component_zmax")]).to_numpy(dtype=float)
                alarm_delta = float(np.nanmean(candidate_alarm) - np.nanmean(legacy_alarm))
                if contrast_stations:
                    sampled_contrast = rng.choice(contrast_stations, size=len(contrast_stations), replace=True)
                    contrast_part = pivot.loc[sampled_contrast]
                    legacy_contrast = (contrast_part[("quality_flagged_alarm_rate", "legacy")] - contrast_part[("quality_clean_alarm_rate", "legacy")]).to_numpy(dtype=float)
                    candidate_contrast = (contrast_part[("quality_flagged_alarm_rate", "component_zmax")] - contrast_part[("quality_clean_alarm_rate", "component_zmax")]).to_numpy(dtype=float)
                    contrast_delta = float(candidate_contrast.mean() - legacy_contrast.mean())
                else:
                    contrast_delta = float("nan")
                alarm_diff.append(alarm_delta)
                contrast_diff.append(contrast_delta)
                bootstrap_rows.append({"target_year": year, "q": q, "bootstrap": b, "alarm_rate_diff_candidate_minus_legacy": alarm_delta, "weak_label_contrast_diff_candidate_minus_legacy": contrast_delta})
            alarm_arr = np.asarray(alarm_diff, dtype=float)
            contrast_arr = np.asarray(contrast_diff, dtype=float)
            observed = subset.groupby("method").agg(alarm_rate=("alarm_rate", "mean"), quality_flagged_alarm_rate=("quality_flagged_alarm_rate", "mean"), quality_clean_alarm_rate=("quality_clean_alarm_rate", "mean")).to_dict(orient="index")
            finite_contrast = contrast_arr[np.isfinite(contrast_arr)]
            summary_rows.append({"target_year": year, "q": q, "alarm_stations": len(stations), "weak_label_contrast_stations": len(contrast_stations), "legacy_alarm_rate_mean": float(observed["legacy"]["alarm_rate"]), "candidate_alarm_rate_mean": float(observed["component_zmax"]["alarm_rate"]), "alarm_rate_diff_mean": float(alarm_arr.mean()), "alarm_rate_diff_ci95": ci(alarm_arr), "prob_candidate_alarm_lower": float((alarm_arr < 0).mean()), "legacy_weak_label_contrast": float(observed["legacy"]["quality_flagged_alarm_rate"] - observed["legacy"]["quality_clean_alarm_rate"]), "candidate_weak_label_contrast": float(observed["component_zmax"]["quality_flagged_alarm_rate"] - observed["component_zmax"]["quality_clean_alarm_rate"]), "weak_label_contrast_diff_mean": float(finite_contrast.mean()) if len(finite_contrast) else float("nan"), "weak_label_contrast_diff_ci95": ci(finite_contrast) if len(finite_contrast) else [float("nan"), float("nan")], "prob_candidate_contrast_higher": float((finite_contrast > 0).mean()) if len(finite_contrast) else float("nan")})
    summary_df = pd.DataFrame(summary_rows)
    boot_df = pd.DataFrame(bootstrap_rows)
    focus = summary_df[summary_df["q"].eq(0.995)]
    stable = bool((focus["alarm_rate_diff_ci95"].map(lambda x: x[1]) < 0).all()) if len(focus) else False
    status = "CANDIDATE_NATURAL_ALARM_STABLE_REQUIRES_MORE_YEARS" if stable else "NATURAL_ALARM_STABILITY_REVIEW"
    payload = {"status": status, "protocol": "使用阶段93自然观测站点明细；2021–2025目标年、q=0.99/0.995/0.999；11站点等权站点簇配对bootstrap 5000次；不使用目标年标签选择。", "stations": stations, "target_years": years, "q_grid": qs, "bootstraps": BOOTSTRAPS, "summary": summary_rows, "limitations": ["自然报警率不是攻击召回率。", "弱标签来自NOAA质量标记，不是真实攻击标签。", "站点等权bootstrap用于不确定性描述，不替换主模型。"]}
    summary_df.to_csv(TABLES / "stage94_noaa_natural_component_bootstrap.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(TABLES / "stage94_noaa_natural_component_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage94_noaa_natural_component_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段94 NOAA自然组件方案站点簇bootstrap", "", "对阶段93的自然报警率和质量弱标签差异进行11站点、5000次等权配对bootstrap。", "", f"- 状态：**{status}**", "", "| 目标年 | q | 自然报警率差值（候选−legacy） | 95% CI | 候选报警更低概率 | 弱标签对比差值 | 95% CI |", "|---:|---:|---:|---|---:|---:|---|"]
    for row in summary_rows:
        lines.append(f"| {row['target_year']} | {row['q']:.3f} | {row['alarm_rate_diff_mean']:.4f} | [{row['alarm_rate_diff_ci95'][0]:.4f},{row['alarm_rate_diff_ci95'][1]:.4f}] | {row['prob_candidate_alarm_lower']:.4f} | {row['weak_label_contrast_diff_mean']:.4f} | [{row['weak_label_contrast_diff_ci95'][0]:.4f},{row['weak_label_contrast_diff_ci95'][1]:.4f}] |")
    lines += ["", "该结果只说明自然观测报警尺度和弱标签区分的站点级稳定性；不能解释为真实攻击检测召回。"]
    (REPORTS / "stage94_noaa_natural_component_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "rows": len(summary_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
