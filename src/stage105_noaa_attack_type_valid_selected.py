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
ATTACK_TYPES = ["drift", "gaussian_noise", "offset", "spike", "stuck", "time_shift"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage91_noaa_physical_component_repair import component_physical_scores  # noqa: E402
from stage92_noaa_physical_component_bootstrap import output_for_method  # noqa: E402
from stage98_noaa_low_far_baseline_comparison import add_structural, baseline_risks, score  # noqa: E402
from stage101_noaa_trust_svm_complementarity import ALPHAS, train_normal_z  # noqa: E402
from stage102_noaa_trust_svm_valid_selection import _select  # noqa: E402


def candidate_risks(scored: pd.DataFrame, physical: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    baseline = baseline_risks(scored)
    trust = output_for_method(scored, physical["component_zmax"], "component_zmax")[2]
    svm = baseline["One-Class SVM"]
    train = scored["split"].eq("train").to_numpy()
    normal_train = train & ~scored["combined_is_attack"].astype(bool).to_numpy()
    trust_z = train_normal_z(np.asarray(trust, dtype=float), train, normal_train)
    svm_z = train_normal_z(np.asarray(svm, dtype=float), train, normal_train)
    risks: dict[str, np.ndarray] = {"svm_only": svm_z, "trust_component_only": trust_z}
    for alpha in ALPHAS:
        risks[f"trust_svm_mean_alpha_{alpha:.2f}"] = alpha * trust_z + (1.0 - alpha) * svm_z
    risks["trust_svm_max"] = np.maximum(trust_z, svm_z)
    return risks


def overall(labels: np.ndarray, pred: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    m = _metrics(labels, pred, risk)
    return {"f1": float(m["f1"]), "recall": float(m["recall"]), "precision": float(m["precision"]), "auprc": float(m["average_precision"]), "far": float(pred[~labels].mean()) if (~labels).any() else float("nan")}


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    type_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []

    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            physical, _ = component_physical_scores(frame)
            scored = add_structural(scored)
            risks = candidate_risks(scored, physical)
            train = scored["split"].eq("train").to_numpy()
            valid = scored["split"].eq("valid").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            normal_train = train & ~scored["combined_is_attack"].astype(bool).to_numpy()
            valid_labels = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
            test_labels = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            test_types = scored.loc[test, "combined_attack_type"].astype(str).to_numpy()
            candidate_rows: list[dict[str, object]] = []
            for method, risk in risks.items():
                threshold = float(np.nanquantile(risk[normal_train], Q))
                valid_metrics = overall(valid_labels, risk[valid] >= threshold, risk[valid])
                candidate_rows.append({"method": method, "threshold": threshold, "valid_f1": valid_metrics["f1"], "valid_auprc": valid_metrics["auprc"], "valid_far": valid_metrics["far"]})
            selected = _select(candidate_rows)
            selections.append({"fold": fold, "seed": seed, "selected_method": selected["method"], "selection_reason": selected["selection_reason"], "feasible_candidates": selected["feasible_candidates"]})
            methods = [("valid_selected", str(selected["method"])), ("svm_only", "svm_only"), ("trust_component_only", "trust_component_only")]
            for label, source_method in methods:
                risk = risks[source_method]
                threshold = float(np.nanquantile(risk[normal_train], Q))
                pred = risk[test] >= threshold
                metrics = overall(test_labels, pred, risk[test])
                overall_rows.append({"fold": fold, "seed": seed, "method": label, "source_method": source_method, "selected_method": selected["method"], "q": Q, **{f"test_{key}": value for key, value in metrics.items()}})
                for attack_type in ATTACK_TYPES:
                    attack_mask = test_types == attack_type
                    attack_pred = pred[attack_mask]
                    type_rows.append({"fold": fold, "seed": seed, "method": label, "source_method": source_method, "selected_method": selected["method"], "attack_type": attack_type, "rows": int(attack_mask.sum()), "detected_rows": int(attack_pred.sum()), "recall": float(attack_pred.mean()) if len(attack_pred) else float("nan")})
            print(json.dumps({"fold": fold, "seed": seed, "selected": selected["method"], "methods": len(methods), "attack_types": len(ATTACK_TYPES)}, ensure_ascii=False), flush=True)
        print(json.dumps({"fold_completed": fold, "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)

    type_df = pd.DataFrame(type_rows)
    overall_df = pd.DataFrame(overall_rows)
    selection_df = pd.DataFrame(selections)
    type_aggregate = type_df.groupby(["method", "attack_type"], sort=False)[["rows", "detected_rows", "recall"]].agg({"rows": "sum", "detected_rows": "sum", "recall": ["mean", "std", "min", "max"]}).reset_index()
    type_aggregate.columns = ["method", "attack_type", "rows", "detected_rows", "recall_mean", "recall_std", "recall_min", "recall_max"]
    overall_aggregate = overall_df.groupby("method", sort=False)[["test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    overall_aggregate.columns = ["method"] + [f"{metric}_{stat}" for metric, stat in overall_aggregate.columns.tolist()[1:]]
    pivot = type_aggregate.pivot(index="attack_type", columns="method", values="recall_mean").reset_index()
    if {"valid_selected", "svm_only", "trust_component_only"}.issubset(pivot.columns):
        pivot["selected_minus_svm"] = pivot["valid_selected"] - pivot["svm_only"]
        pivot["selected_minus_trust"] = pivot["valid_selected"] - pivot["trust_component_only"]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "NOAA 2020–2025；四折滚动、11站点、5种子；q=0.975固定；每折每种子仅在valid按FAR≤10%约束最大化F1选择Trust–SVM候选；test按六类透明受控攻击逐类统计Recall，并保留固定SVM/Trust参照。",
        "q": Q,
        "valid_far_cap": VALID_FAR_CAP,
        "attack_types": ATTACK_TYPES,
        "stations": stations,
        "selection": selection_df.to_dict(orient="records"),
        "selection_counts": selection_df["selected_method"].value_counts().rename_axis("method").reset_index(name="count").to_dict(orient="records"),
        "overall_aggregate": overall_aggregate.to_dict(orient="records"),
        "attack_type_aggregate": type_aggregate.to_dict(orient="records"),
        "attack_type_comparison": pivot.to_dict(orient="records"),
        "limitations": [
            "攻击类型表只报告攻击行Recall；Precision/F1/AUPRC/FAR统一在overall表报告，避免把攻击子集指标误当综合指标。",
            "valid_selected的候选由valid选择；test只评价，不按攻击类型test结果重新选权重。",
            "攻击为透明受控注入，不是真实攻击日志。",
        ],
    }
    type_df.to_csv(TABLES / "stage105_noaa_attack_type_valid_selected_detail.csv", index=False, encoding="utf-8-sig")
    overall_df.to_csv(TABLES / "stage105_noaa_attack_type_valid_selected_overall_detail.csv", index=False, encoding="utf-8-sig")
    selection_df.to_csv(TABLES / "stage105_noaa_attack_type_valid_selected_selection.csv", index=False, encoding="utf-8-sig")
    type_aggregate.to_csv(TABLES / "stage105_noaa_attack_type_valid_selected.csv", index=False, encoding="utf-8-sig")
    overall_aggregate.to_csv(TABLES / "stage105_noaa_attack_type_valid_selected_overall.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage105_noaa_attack_type_valid_selected.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段105 NOAA验证集选权重逐攻击类型诊断", "", "沿用阶段102的valid选权重规则，test按六类受控攻击统计Recall；综合F1、Recall、Precision、AUPRC、FAR另表报告。", "", "## 选择频次", "", "| 选择方法 | 次数 |", "|---|---:|"]
    for row in payload["selection_counts"]:
        lines.append(f"| {row['method']} | {row['count']} |")
    lines += ["", "## 攻击类型Recall", "", "| 攻击类型 | valid_selected | svm_only | trust_component_only | selected−SVM | selected−Trust |", "|---|---:|---:|---:|---:|---:|"]
    for row in payload["attack_type_comparison"]:
        lines.append(f"| {row['attack_type']} | {row.get('valid_selected', float('nan')):.4f} | {row.get('svm_only', float('nan')):.4f} | {row.get('trust_component_only', float('nan')):.4f} | {row.get('selected_minus_svm', float('nan')):+.4f} | {row.get('selected_minus_trust', float('nan')):+.4f} |")
    lines += ["", "## 综合指标", "", "| 方法 | F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in overall_aggregate.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} |")
    lines += ["", "攻击类型仍为受控注入；不能据此声称对真实攻击类型的普适检测能力。"]
    (REPORTS / "stage105_noaa_attack_type_valid_selected.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "type_rows": len(type_df), "overall_rows": len(overall_df), "selections": len(selection_df)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
