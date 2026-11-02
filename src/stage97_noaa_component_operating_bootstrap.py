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
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"
AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
METHODS = ["legacy", "component_zmax"]
Q_POINTS = [0.90, 0.975]
BOOTSTRAPS = 5000

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage91_noaa_physical_component_repair import component_physical_scores  # noqa: E402
from stage92_noaa_physical_component_bootstrap import counts, metrics_from_counts, output_for_method  # noqa: E402


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    detail: list[dict[str, object]] = []
    station_counts = {str(q): {method: {station: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for station in stations} for method in METHODS} for q in Q_POINTS}
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            physical, _ = component_physical_scores(frame)
            train = scored["split"].eq("train").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            for method in METHODS:
                _, _, risk = output_for_method(scored, physical[method], method)
                for q in Q_POINTS:
                    threshold = float(np.nanquantile(risk[train & np.isfinite(risk)], q))
                    pred = risk[test] >= threshold
                    metrics = _metrics(yt, pred, risk[test])
                    far = float(pred[~yt].mean()) if (~yt).any() else float("nan")
                    detail.append({"fold": fold, "seed": seed, "method": method, "q": q, "threshold": threshold, "test_f1": float(metrics["f1"]), "test_recall": float(metrics["recall"]), "test_precision": float(metrics["precision"]), "test_auprc": float(metrics["average_precision"]), "test_far": far})
                    for station in stations:
                        mask = test & scored["station_key"].eq(station).to_numpy()
                        c = counts(scored.loc[mask, "combined_is_attack"].astype(bool).to_numpy(), pred[mask[test]])
                        for field in c:
                            station_counts[str(q)][method][station][field] += c[field]
            print(json.dumps({"fold": fold, "seed": seed, "methods": len(METHODS), "q_points": len(Q_POINTS)}, ensure_ascii=False), flush=True)
        print(json.dumps({"fold_completed": fold, "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)
    detail_df = pd.DataFrame(detail)
    aggregate = detail_df.groupby(["method", "q"], sort=False)[["test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["method", "q"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[2:]]
    rng = np.random.default_rng(20260915)
    boot_rows: list[dict[str, object]] = []
    boot_summary: list[dict[str, object]] = []
    for q in Q_POINTS:
        q_key = str(q)
        values = {"f1": [], "recall": [], "precision": [], "far": []}
        for b in range(BOOTSTRAPS):
            sampled = rng.choice(stations, size=len(stations), replace=True)
            totals = {method: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for method in METHODS}
            for station in sampled:
                for method in METHODS:
                    for field in totals[method]:
                        totals[method][field] += station_counts[q_key][method][str(station)][field]
            legacy = metrics_from_counts(totals["legacy"])
            candidate = metrics_from_counts(totals["component_zmax"])
            row = {"q": q, "bootstrap": b}
            for metric in ["f1", "recall", "precision", "far"]:
                diff = candidate[metric] - legacy[metric]
                row[f"{metric}_diff_candidate_minus_legacy"] = diff
                values[metric].append(diff)
            boot_rows.append(row)
        summary = {"q": q}
        for metric, arr in values.items():
            a = np.asarray(arr, dtype=float)
            summary[f"{metric}_diff_mean"] = float(a.mean())
            summary[f"{metric}_diff_ci95"] = [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
        summary["prob_f1_positive"] = float((np.asarray(values["f1"]) > 0).mean())
        summary["prob_far_lower"] = float((np.asarray(values["far"]) < 0).mean())
        boot_summary.append(summary)
    q975 = next(row for row in boot_summary if row["q"] == 0.975)
    status = "CANDIDATE_OPERATING_POINT_STABLE_REQUIRES_MORE_YEARS" if q975["f1_diff_ci95"][0] > 0 and q975["far_diff_ci95"][1] < 0 else "OPERATING_POINT_STABILITY_REVIEW"
    payload = {"status": status, "candidate_variant": "component_zmax", "protocol": "阶段96预先固定q=0.90与0.975运行点；四折滚动、11站点、5种子；alpha/window由valid选择，阈值仅由训练正常风险分位数确定；站点簇5000次配对bootstrap。", "stations": stations, "q_points": Q_POINTS, "bootstraps": BOOTSTRAPS, "detail": detail, "aggregate": aggregate.to_dict(orient="records"), "bootstrap_summary": boot_summary, "limitations": ["component_zmax运行点的主要收益是降低自然/正常误报，Recall仍显著低于legacy。", "运行点不是主模型替换方案。", "攻击为透明受控注入，不是真实攻击日志。"]}
    detail_df.to_csv(TABLES / "stage97_noaa_component_operating_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage97_noaa_component_operating_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(boot_summary).to_csv(TABLES / "stage97_noaa_component_operating_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(boot_rows).to_csv(TABLES / "stage97_noaa_component_operating_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage97_noaa_component_operating_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段97 NOAA物理组件运行点站点簇bootstrap", "", "固定q=0.90与q=0.975两个训练分位数运行点，对component_zmax与legacy进行站点簇配对bootstrap。", "", f"- 状态：**{status}**", "", "| q | F1差值 | 95% CI | Recall差值 | 95% CI | FAR差值 | 95% CI |", "|---:|---:|---|---:|---|---:|---|"]
    for row in boot_summary:
        lines.append(f"| {row['q']:.3f} | {row['f1_diff_mean']:.4f} | [{row['f1_diff_ci95'][0]:.4f},{row['f1_diff_ci95'][1]:.4f}] | {row['recall_diff_mean']:.4f} | [{row['recall_diff_ci95'][0]:.4f},{row['recall_diff_ci95'][1]:.4f}] | {row['far_diff_mean']:.4f} | [{row['far_diff_ci95'][0]:.4f},{row['far_diff_ci95'][1]:.4f}] |")
    lines += ["", "q=0.975可作为低误报安全运行点候选，但其Recall损失必须在论文中并列披露。"]
    (REPORTS / "stage97_noaa_component_operating_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": boot_summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
