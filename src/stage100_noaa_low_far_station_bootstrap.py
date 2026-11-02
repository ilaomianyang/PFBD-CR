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
METHODS = ["Trust legacy", "Trust component_zmax", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
REFERENCES = ["Trust legacy", "Trust component_zmax"]
Q = 0.975
BOOTSTRAPS = 5000

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage91_noaa_physical_component_repair import component_physical_scores  # noqa: E402
from stage92_noaa_physical_component_bootstrap import counts, metrics_from_counts, output_for_method  # noqa: E402
from stage98_noaa_low_far_baseline_comparison import add_structural, baseline_risks, score  # noqa: E402


def _sum_counts(rows: pd.DataFrame, method: str, stations: list[str]) -> dict[str, int]:
    selected = rows.loc[rows["method"].eq(method) & rows["station_id"].isin(stations)]
    return {field: int(selected[field].sum()) for field in ["tp", "fp", "fn", "tn"]}


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    station_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            physical, _ = component_physical_scores(frame)
            scored = add_structural(scored)
            risks = {
                "Trust legacy": output_for_method(scored, physical["legacy"], "legacy")[2],
                "Trust component_zmax": output_for_method(scored, physical["component_zmax"], "component_zmax")[2],
                **baseline_risks(scored),
            }
            train = scored["split"].eq("train").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            test_labels = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            test_stations = scored.loc[test, "station_key"].astype(str).to_numpy()
            for method in METHODS:
                risk = np.asarray(risks[method], dtype=float)
                threshold = float(np.nanquantile(risk[train & np.isfinite(risk)], Q))
                pred = risk[test] >= threshold
                total = counts(test_labels, pred)
                run_rows.append({"fold": fold, "seed": seed, "method": method, "q": Q, "threshold": threshold, **total, **metrics_from_counts(total)})
                for station in stations:
                    station_mask = test_stations == station
                    station_count = counts(test_labels[station_mask], pred[station_mask])
                    station_rows.append({"fold": fold, "seed": seed, "station_id": station, "method": method, "q": Q, "threshold": threshold, **station_count})
            print(json.dumps({"fold": fold, "seed": seed, "methods": len(METHODS), "q": Q}, ensure_ascii=False), flush=True)
        print(json.dumps({"fold_completed": fold, "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)

    station_detail = pd.DataFrame(station_rows)
    run_detail = pd.DataFrame(run_rows)
    expected_rows = len(FOLDS) * len(SEEDS) * len(stations) * len(METHODS)
    if len(station_detail) != expected_rows:
        raise ValueError(f"incomplete station detail: {len(station_detail)} != {expected_rows}")

    rng = np.random.default_rng(20260920)
    summary_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for candidate in [method for method in METHODS if method not in REFERENCES]:
        for reference in REFERENCES:
            values = {metric: [] for metric in ["f1", "recall", "precision", "far"]}
            observed_candidate = metrics_from_counts(_sum_counts(station_detail, candidate, stations))
            observed_reference = metrics_from_counts(_sum_counts(station_detail, reference, stations))
            for bootstrap in range(BOOTSTRAPS):
                sampled = [str(x) for x in rng.choice(stations, size=len(stations), replace=True)]
                candidate_metrics = metrics_from_counts(_sum_counts(station_detail, candidate, sampled))
                reference_metrics = metrics_from_counts(_sum_counts(station_detail, reference, sampled))
                row: dict[str, object] = {"bootstrap": bootstrap, "candidate": candidate, "reference": reference, "q": Q}
                for metric in values:
                    diff = candidate_metrics[metric] - reference_metrics[metric]
                    values[metric].append(diff)
                    row[f"{metric}_diff"] = diff
                bootstrap_rows.append(row)
            for metric, metric_values in values.items():
                arr = np.asarray(metric_values, dtype=float)
                observed_diff = observed_candidate[metric] - observed_reference[metric]
                summary_rows.append(
                    {
                        "candidate": candidate,
                        "reference": reference,
                        "q": Q,
                        "metric": metric,
                        "candidate_value": observed_candidate[metric],
                        "reference_value": observed_reference[metric],
                        "observed_diff": observed_diff,
                        "bootstrap_diff_mean": float(arr.mean()),
                        "ci95_low": float(np.quantile(arr, 0.025)),
                        "ci95_high": float(np.quantile(arr, 0.975)),
                        "prob_candidate_better": float((arr > 0).mean()) if metric != "far" else float((arr < 0).mean()),
                        "direction": "higher_is_better" if metric != "far" else "lower_is_better",
                    }
                )

    summary = pd.DataFrame(summary_rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "NOAA 2020–2025；4折滚动、11站点、5种子；固定训练正常风险分位数q=0.975；五种方法逐站点累计混淆计数；11站点簇5000次配对bootstrap。",
        "q": Q,
        "stations": stations,
        "folds": len(FOLDS),
        "seeds": SEEDS,
        "methods": METHODS,
        "bootstraps": BOOTSTRAPS,
        "run_detail": run_detail.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "limitations": [
            "q=0.975在实验前固定，本阶段不根据test重新挑选阈值。",
            "站点簇bootstrap保留站点内所有年度折和种子相关性，但站点数仍只有11个。",
            "AUPRC无法由聚合混淆计数重建，其折×种子不确定性见阶段99。",
            "攻击为透明受控注入，不是真实攻击日志。",
        ],
    }
    station_detail.to_parquet(TABLES / "stage100_noaa_low_far_station_counts.parquet", index=False)
    run_detail.to_csv(TABLES / "stage100_noaa_low_far_run_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "stage100_noaa_low_far_station_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bootstrap_rows).to_parquet(TABLES / "stage100_noaa_low_far_station_bootstrap_detail.parquet", index=False)
    (TABLES / "stage100_noaa_low_far_station_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")

    lines = [
        "# 阶段100 NOAA低误报运行点站点簇bootstrap",
        "",
        "固定q=0.975，对五种方法保存逐站点混淆计数，并执行11站点簇配对bootstrap。",
        "",
        "差值定义为候选减参考；FAR为负表示候选误报更低。",
        "",
        "| 候选 | 参考 | 指标 | 候选 | 参考 | 差值 | 95% CI | 候选更优概率 |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['candidate']} | {row['reference']} | {row['metric']} | {row['candidate_value']:.4f} | "
            f"{row['reference_value']:.4f} | {row['observed_diff']:.4f} | "
            f"[{row['ci95_low']:.4f},{row['ci95_high']:.4f}] | {row['prob_candidate_better']:.4f} |"
        )
    lines += ["", "本阶段为站点簇级证据；AUPRC的不确定性仍使用阶段99的折×种子层级结果。"]
    (REPORTS / "stage100_noaa_low_far_station_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "station_rows": len(station_detail), "summary_rows": len(summary)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
