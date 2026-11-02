from __future__ import annotations

"""Stage 145: prespecified station heterogeneity audit."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = TABLES / "stage141_noaa_frozen_external_validation_by_station.csv"
METHOD_A = "TrustMeteo-QC frozen expanded"
METHOD_B = "TrustMeteo-QC default"
METRICS = ["f1", "recall", "precision", "far"]


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_station(values: pd.DataFrame, metric: str, seed: int, repeats: int = 10000) -> tuple[float, float]:
    stations = values["station"].drop_duplicates().tolist()
    grouped = {station: values.loc[values["station"].eq(station), metric].to_numpy(dtype=float) for station in stations}
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        means[index] = np.concatenate([grouped[station] for station in sampled]).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    source = pd.read_csv(SOURCE)
    source["station"] = source["station"].astype(str)
    if set(source["method"].unique()) != {METHOD_A, METHOD_B} or source["station"].nunique() != 11:
        raise ValueError("stage141 station detail is incomplete")
    wide = source.pivot_table(index=["fold", "seed", "station"], columns="method", values=METRICS, aggfunc="first").reset_index()
    wide.columns = ["_".join(str(part) for part in column if str(part) != "") if isinstance(column, tuple) else str(column) for column in wide.columns]
    station_rows: list[dict[str, object]] = []
    for station, group in wide.groupby("station", sort=True):
        row: dict[str, object] = {"station": station, "cells": int(len(group))}
        for metric in METRICS:
            a = group[f"{metric}_{METHOD_A}"].to_numpy(dtype=float)
            b = group[f"{metric}_{METHOD_B}"].to_numpy(dtype=float)
            row[f"{metric}_expanded_mean"] = float(a.mean())
            row[f"{metric}_default_mean"] = float(b.mean())
            row[f"{metric}_diff"] = float((a - b).mean())
        station_rows.append(row)
    station_summary = pd.DataFrame(station_rows)
    station_summary["f1_rank"] = station_summary["f1_expanded_mean"].rank(method="min", ascending=False).astype(int)
    paired_rows: list[dict[str, object]] = []
    diff_frame = wide.copy()
    for metric in METRICS:
        diff_frame[f"diff_{metric}"] = diff_frame[f"{metric}_{METHOD_A}"] - diff_frame[f"{metric}_{METHOD_B}"]
        low, high = bootstrap_station(diff_frame.rename(columns={f"diff_{metric}": "value"}), "value", stable_seed("stage145", metric))
        values = diff_frame[f"diff_{metric}"].to_numpy(dtype=float)
        paired_rows.append({"comparison": "frozen expanded - default", "metric": metric, "n_cells": len(values), "station_count": int(diff_frame["station"].nunique()), "mean_diff": float(values.mean()), "station_bootstrap_ci95_low": low, "station_bootstrap_ci95_high": high, "stations_improved": int((station_summary[f"{metric}_diff"] > 0).sum()), "stations_worsened": int((station_summary[f"{metric}_diff"] < 0).sum())})
    paired = pd.DataFrame(paired_rows)
    extremes = pd.DataFrame([{"metric": metric, "best_station": str(station_summary.loc[station_summary[f"{metric}_diff"].idxmax(), "station"]), "best_diff": float(station_summary[f"{metric}_diff"].max()), "worst_station": str(station_summary.loc[station_summary[f"{metric}_diff"].idxmin(), "station"]), "worst_diff": float(station_summary[f"{metric}_diff"].min())} for metric in METRICS])
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段141 NOAA四折×五种子冻结外部test站点明细的站点异质性审计；不重新选择候选、不重新训练、不根据站点test调参；按11个站点做成对站点簇bootstrap。扩展候选与default均为透明受控注入协议，不是真实攻击日志。",
        "source_artifact": SOURCE.name,
        "methods": [METHOD_A, METHOD_B],
        "stations": sorted(source["station"].unique().tolist()),
        "folds": int(source["fold"].nunique()),
        "seeds": int(source["seed"].nunique()),
        "station_summary": station_summary.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "extremes": extremes.to_dict(orient="records"),
        "limitations": ["这是阶段141已有locked-test站点明细的预先定义分层分析，不增加独立站点或年份。", "站点数量为11，站点bootstrap只能量化当前站点集合的不确定性。", "FAR下降伴随Recall下降时必须按安全—召回权衡解释，不能写成全面提升。"],
    }
    prefix = "stage145_noaa_station_heterogeneity_audit"
    station_summary.to_csv(TABLES / f"{prefix}_station_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    extremes.to_csv(TABLES / f"{prefix}_extremes.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段145 NOAA站点异质性审计", "", "对阶段141冻结外部test站点明细做预先定义的站点分层和站点簇bootstrap，不重新选择候选。", "", "| 指标 | 平均差 | CI95 | 改善站点数 | 恶化站点数 |", "|---|---:|---:|---:|---:|"]
    for row in paired.to_dict(orient="records"):
        lines.append(f"| {row['metric']} | {row['mean_diff']:.4f} | [{row['station_bootstrap_ci95_low']:.4f}, {row['station_bootstrap_ci95_high']:.4f}] | {row['stations_improved']} | {row['stations_worsened']} |")
    lines += ["", "## 最好/最差站点", "", "| 指标 | 最好站点 | 差值 | 最差站点 | 差值 |", "|---|---|---:|---|---:|"]
    for row in extremes.to_dict(orient="records"):
        lines.append(f"| {row['metric']} | {row['best_station']} | {row['best_diff']:.4f} | {row['worst_station']} | {row['worst_diff']:.4f} |")
    lines += ["", "该分析支持站点异质性和安全—召回权衡解释，不支持跨地区普适性或真实攻击能力声明。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "stations": len(station_summary), "paired": len(paired), "extremes": len(extremes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
