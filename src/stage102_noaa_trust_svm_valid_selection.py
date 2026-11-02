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
Q = 0.975
VALID_FAR_CAP = 0.10
ALPHAS = [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 1.0]
METHODS = ["svm_only", "trust_component_only", *[f"trust_svm_mean_alpha_{alpha:.2f}" for alpha in ALPHAS], "trust_svm_max"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage91_noaa_physical_component_repair import component_physical_scores  # noqa: E402
from stage92_noaa_physical_component_bootstrap import output_for_method  # noqa: E402
from stage98_noaa_low_far_baseline_comparison import add_structural, baseline_risks, score  # noqa: E402
from stage101_noaa_trust_svm_complementarity import train_normal_z  # noqa: E402


def _metric_row(labels: np.ndarray, pred: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    metrics = _metrics(labels, pred, risk)
    return {
        "f1": float(metrics["f1"]),
        "recall": float(metrics["recall"]),
        "precision": float(metrics["precision"]),
        "auprc": float(metrics["average_precision"]),
        "far": float(pred[~labels].mean()) if (~labels).any() else float("nan"),
    }


def _select(rows: list[dict[str, object]]) -> dict[str, object]:
    feasible = [row for row in rows if float(row["valid_far"]) <= VALID_FAR_CAP]
    pool = feasible if feasible else rows
    if feasible:
        chosen = max(pool, key=lambda row: (float(row["valid_f1"]), float(row["valid_auprc"]), -float(row["valid_far"]), -float(row["threshold"])))
        reason = "max_valid_f1_subject_to_valid_far_cap"
    else:
        chosen = min(pool, key=lambda row: (float(row["valid_far"]), -float(row["valid_f1"]), -float(row["valid_auprc"])))
        reason = "no_feasible_candidate_min_valid_far_then_max_valid_f1"
    return {**chosen, "selection_reason": reason, "feasible_candidates": len(feasible)}


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    all_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            physical, _ = component_physical_scores(frame)
            scored = add_structural(scored)
            baseline = baseline_risks(scored)
            trust = output_for_method(scored, physical["component_zmax"], "component_zmax")[2]
            svm = baseline["One-Class SVM"]
            train = scored["split"].eq("train").to_numpy()
            valid = scored["split"].eq("valid").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            normal_train = train & ~scored["combined_is_attack"].astype(bool).to_numpy()
            trust_z = train_normal_z(np.asarray(trust, dtype=float), train, normal_train)
            svm_z = train_normal_z(np.asarray(svm, dtype=float), train, normal_train)
            risks: dict[str, np.ndarray] = {"svm_only": svm_z, "trust_component_only": trust_z, "trust_svm_max": np.maximum(trust_z, svm_z)}
            for alpha in ALPHAS:
                risks[f"trust_svm_mean_alpha_{alpha:.2f}"] = alpha * trust_z + (1.0 - alpha) * svm_z
            valid_labels = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
            test_labels = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            candidate_rows: list[dict[str, object]] = []
            for method in METHODS:
                risk = risks[method]
                threshold = float(np.nanquantile(risk[normal_train], Q))
                valid_metrics = _metric_row(valid_labels, risk[valid] >= threshold, risk[valid])
                test_metrics = _metric_row(test_labels, risk[test] >= threshold, risk[test])
                row = {"fold": fold, "seed": seed, "method": method, "alpha_trust": float(next((alpha for alpha in ALPHAS if method == f"trust_svm_mean_alpha_{alpha:.2f}"), np.nan)), "q": Q, "valid_far_cap": VALID_FAR_CAP, "threshold": threshold, **{f"valid_{key}": value for key, value in valid_metrics.items()}, **{f"test_{key}": value for key, value in test_metrics.items()}}
                candidate_rows.append(row)
                all_rows.append(row)
            selected_rows.append(_select(candidate_rows))
            print(json.dumps({"fold": fold, "seed": seed, "candidates": len(candidate_rows), "selected": selected_rows[-1]["method"], "selection_reason": selected_rows[-1]["selection_reason"]}, ensure_ascii=False), flush=True)
        print(json.dumps({"fold_completed": fold, "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(all_rows)
    selected = pd.DataFrame(selected_rows)
    aggregate_all = detail.groupby("method", sort=False)[["valid_f1", "valid_recall", "valid_precision", "valid_auprc", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate_all.columns = ["method"] + [f"{metric}_{stat}" for metric, stat in aggregate_all.columns.tolist()[1:]]
    aggregate_selected = selected.groupby("method", sort=False)[["valid_f1", "valid_recall", "valid_precision", "valid_auprc", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate_selected.columns = ["method"] + [f"{metric}_{stat}" for metric, stat in aggregate_selected.columns.tolist()[1:]]
    selection_counts = selected["method"].value_counts().rename_axis("method").reset_index(name="selected_count")
    selection_counts["selection_fraction"] = selection_counts["selected_count"] / len(selected)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "NOAA 2020–2025；四折滚动、11站点、5种子；Trust/SVM标准化只由train正常样本拟合；q=0.975固定；预先固定权重网格；每折每种子仅在valid上按FAR≤10%约束最大化F1选择；test只评价。",
        "q": Q,
        "valid_far_cap": VALID_FAR_CAP,
        "alphas": ALPHAS,
        "methods": METHODS,
        "stations": stations,
        "all_candidate_aggregate": aggregate_all.to_dict(orient="records"),
        "selected_aggregate": aggregate_selected.to_dict(orient="records"),
        "selection_counts": selection_counts.to_dict(orient="records"),
        "selected_detail": selected.to_dict(orient="records"),
        "limitations": [
            "FAR上限和权重网格在实验前固定；不得根据test最高项再次挑选。",
            "selected_aggregate只汇总valid选择后的test结果，all_candidate_aggregate用于完整披露候选。",
            "q=0.975是低误报运行点，不代表F1最优运行点。",
            "攻击为透明受控注入，不是真实攻击日志。",
        ],
    }
    detail.to_csv(TABLES / "stage102_noaa_trust_svm_valid_selection_all_candidates.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(TABLES / "stage102_noaa_trust_svm_valid_selection_selected.csv", index=False, encoding="utf-8-sig")
    aggregate_all.to_csv(TABLES / "stage102_noaa_trust_svm_valid_selection_all_aggregate.csv", index=False, encoding="utf-8-sig")
    aggregate_selected.to_csv(TABLES / "stage102_noaa_trust_svm_valid_selection_selected_aggregate.csv", index=False, encoding="utf-8-sig")
    selection_counts.to_csv(TABLES / "stage102_noaa_trust_svm_valid_selection_counts.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage102_noaa_trust_svm_valid_selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段102 NOAA Trust–SVM验证集选权重",
        "",
        "固定q=0.975；每个滚动折×种子只在valid上按 FAR≤10% 约束最大化F1，选择候选后评价test。",
        "",
        "## 选择频次",
        "",
        "| 方法 | 选择次数 | 选择比例 |",
        "|---|---:|---:|",
    ]
    for row in selection_counts.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['selected_count']} | {row['selection_fraction']:.3f} |")
    lines += ["", "## 选择后test汇总", "", "| 方法 | test F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate_selected.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} |")
    lines += ["", "完整候选test汇总见 `stage102_noaa_trust_svm_valid_selection_all_aggregate.csv`；不得省略未被valid选中的候选。"]
    (REPORTS / "stage102_noaa_trust_svm_valid_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "candidate_rows": len(detail), "selected_rows": len(selected)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
