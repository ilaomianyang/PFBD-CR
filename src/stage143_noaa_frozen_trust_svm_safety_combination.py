from __future__ import annotations

"""Stage 143: pre-registered Trust/SVM safety combination on NOAA controlled data.

The frozen expanded Trust candidate and One-Class SVM are combined with a small,
fixed candidate set.  Each rolling fold selects at most one candidate on valid
under a fixed valid FAR cap; test is used only for evaluation.
"""

import hashlib
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
TRUST_ALPHA = 0.25
TRUST_WINDOW = 1
MEAN_ALPHAS = [0.25, 0.50, 0.75]
METRICS = ["f1", "recall", "precision", "auprc", "far"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage98_noaa_low_far_baseline_comparison import baseline_risks  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def robust_z(risk: np.ndarray, normal_train: np.ndarray) -> np.ndarray:
    values = np.asarray(risk, dtype=float)[normal_train]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite normal training risk")
    med = float(np.nanmedian(values))
    scale = float(max(1.4826 * np.nanmedian(np.abs(values - med)), 0.25))
    return np.maximum((np.asarray(risk, dtype=float) - med) / scale, 0.0)


def metric_row(labels: np.ndarray, pred: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    result = _metrics(labels, pred, risk)
    return {
        "f1": float(result["f1"]),
        "recall": float(result["recall"]),
        "precision": float(result["precision"]),
        "auprc": float(result["average_precision"]),
        "far": float(pred[~labels].mean()) if (~labels).any() else 0.0,
    }


def candidate_risks(scored: pd.DataFrame) -> dict[str, np.ndarray]:
    train = scored["split"].eq("train").to_numpy()
    normal_train = train & ~scored["combined_is_attack"].astype(bool).to_numpy()
    base = TRUST_ALPHA * scored["base_fused_risk"].to_numpy(dtype=float) + (1.0 - TRUST_ALPHA) * scored["structural_risk"].to_numpy(dtype=float)
    trust = smooth_risk(scored, base, TRUST_WINDOW)
    svm = baseline_risks(scored)["One-Class SVM"]
    trust_z = robust_z(trust, normal_train)
    svm_z = robust_z(svm, normal_train)
    risks = {
        "Trust frozen expanded": trust_z,
        "One-Class SVM": svm_z,
        "Trust-SVM mean alpha 0.25": 0.25 * trust_z + 0.75 * svm_z,
        "Trust-SVM mean alpha 0.50": 0.50 * trust_z + 0.50 * svm_z,
        "Trust-SVM mean alpha 0.75": 0.75 * trust_z + 0.25 * svm_z,
        "Trust-SVM max": np.maximum(trust_z, svm_z),
    }
    return risks


def evaluate_candidate(scored: pd.DataFrame, name: str, risk: np.ndarray) -> dict[str, object]:
    train = scored["split"].eq("train").to_numpy()
    valid = scored["split"].eq("valid").to_numpy()
    test = scored["split"].eq("test").to_numpy()
    normal_train = train & ~scored["combined_is_attack"].astype(bool).to_numpy()
    threshold = float(np.nanquantile(risk[normal_train], Q))
    y_valid = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    y_test = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    valid_risk = risk[valid]
    test_risk = risk[test]
    valid_metrics = metric_row(y_valid, valid_risk >= threshold, valid_risk)
    test_metrics = metric_row(y_test, test_risk >= threshold, test_risk)
    return {
        "method": name,
        "q": Q,
        "valid_far_cap": VALID_FAR_CAP,
        "threshold": threshold,
        **{f"valid_{key}": value for key, value in valid_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }


def choose_valid(rows: list[dict[str, object]]) -> dict[str, object]:
    feasible = [row for row in rows if float(row["valid_far"]) <= VALID_FAR_CAP]
    pool = feasible if feasible else rows
    chosen = max(pool, key=lambda row: (float(row["valid_f1"]), float(row["valid_auprc"]), -float(row["valid_far"]), -float(row["threshold"])))
    return {**chosen, "selection_reason": "max_valid_f1_then_auprc_under_far_cap" if feasible else "min_violation_fallback", "feasible_candidates": len(feasible)}


def bootstrap_cluster(values: pd.DataFrame, seed: int, repeats: int = 10000) -> tuple[float, float]:
    clusters = values["fold"].drop_duplicates().tolist()
    grouped = {fold: values.loc[values["fold"].eq(fold), "diff"].to_numpy(dtype=float) for fold in clusters}
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        means[index] = np.concatenate([grouped[fold] for fold in sampled]).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(value) for value in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    all_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            scored = add_components(scored)
            risks = candidate_risks(scored)
            candidates = [{"fold": fold, "seed": seed, **evaluate_candidate(scored, name, risk)} for name, risk in risks.items()]
            all_rows.extend(candidates)
            selected_rows.append({"fold": fold, "seed": seed, **choose_valid(candidates)})
            print(json.dumps({"fold": fold, "seed": seed, "selected": selected_rows[-1]["method"], "feasible": selected_rows[-1]["feasible_candidates"]}, ensure_ascii=False), flush=True)
        print(json.dumps({"fold_completed": fold, "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(all_rows)
    selected = pd.DataFrame(selected_rows)
    aggregate_all = detail.groupby("method", sort=False)[[f"valid_{m}" for m in METRICS] + [f"test_{m}" for m in METRICS]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate_all.columns = ["method"] + [f"{metric}_{stat}" for metric, stat in aggregate_all.columns.tolist()[1:]]
    aggregate_selected = selected.groupby("method", sort=False)[[f"valid_{m}" for m in METRICS] + [f"test_{m}" for m in METRICS]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate_selected.columns = ["method"] + [f"{metric}_{stat}" for metric, stat in aggregate_selected.columns.tolist()[1:]]
    counts = selected["method"].value_counts().rename_axis("method").reset_index(name="selected_count")
    counts["selection_fraction"] = counts["selected_count"] / len(selected)

    paired_rows: list[dict[str, object]] = []
    selected_index = selected.set_index(["fold", "seed"])
    for reference in ["Trust frozen expanded", "One-Class SVM"]:
        reference_index = detail.loc[detail["method"].eq(reference)].set_index(["fold", "seed"])
        common = selected_index.join(reference_index, lsuffix="_selected", rsuffix="_reference", how="inner")
        for metric in METRICS:
            diff = common[f"test_{metric}_selected"].to_numpy(dtype=float) - common[f"test_{metric}_reference"].to_numpy(dtype=float)
            pair = pd.DataFrame({"fold": common.index.get_level_values(0), "diff": diff})
            low, high = bootstrap_cluster(pair, stable_seed("stage143", reference, metric))
            paired_rows.append({"comparison": f"valid-selected - {reference}", "metric": metric, "n": len(diff), "mean_diff": float(diff.mean()), "bootstrap_fold_ci95_low": low, "bootstrap_fold_ci95_high": high})
    paired = pd.DataFrame(paired_rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "NOAA 2020–2025四折滚动、11个全年度合格站点、5种子；冻结Trust alpha=0.25/window=1与One-Class SVM先由train正常样本分别稳健标准化；候选集合和q=0.975、valid FAR≤10%约束预先固定；每折每种子仅由valid选择，test只评价；不根据test重新选择、不重新训练。攻击为透明受控注入，不是真实攻击日志。",
        "stations": stations,
        "folds": [fold_name(*fold) for fold in FOLDS if set([*fold[0], fold[1], fold[2]]).issubset(years)],
        "seeds": SEEDS,
        "q": Q,
        "valid_far_cap": VALID_FAR_CAP,
        "frozen_trust_alpha": TRUST_ALPHA,
        "frozen_trust_window": TRUST_WINDOW,
        "candidate_methods": list(risks.keys()),
        "detail": detail.to_dict(orient="records"),
        "selected": selected.to_dict(orient="records"),
        "aggregate_all": aggregate_all.to_dict(orient="records"),
        "aggregate_selected": aggregate_selected.to_dict(orient="records"),
        "selection_counts": counts.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "limitations": ["候选组合只在当前NOAA滚动受控协议中验证，不能宣称跨域普适。", "One-Class SVM若在F1/AUPRC上更强，必须保留该负结果。", "自然观测报警率和质量弱标签边界由阶段142单独报告，不能与本阶段受控攻击指标混写。"],
    }
    prefix = "stage143_noaa_frozen_trust_svm_safety_combination"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(TABLES / f"{prefix}_selected.csv", index=False, encoding="utf-8-sig")
    aggregate_all.to_csv(TABLES / f"{prefix}_aggregate_all.csv", index=False, encoding="utf-8-sig")
    aggregate_selected.to_csv(TABLES / f"{prefix}_aggregate_selected.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(TABLES / f"{prefix}_selection_counts.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段143 NOAA冻结 Trust–SVM 安全组合", "", "候选集合和选择规则在实验前固定；每个滚动折×种子仅由valid选择，test只评价。", "", "## 选择频次", "", "| 方法 | 次数 | 比例 |", "|---|---:|---:|"]
    for row in counts.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['selected_count']} | {row['selection_fraction']:.3f} |")
    lines += ["", "## valid选择后的test汇总", "", "| 方法 | F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate_selected.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} |")
    lines += ["", "## 成对差异", "", "| 比较 | 指标 | 平均差 | 折聚类CI95 |", "|---|---|---:|---:|"]
    for row in paired.to_dict(orient="records"):
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_diff']:.4f} | [{row['bootstrap_fold_ci95_low']:.4f}, {row['bootstrap_fold_ci95_high']:.4f}] |")
    lines += ["", "## 边界", "", "本阶段使用真实NOAA观测与透明受控注入，仅支持滚动外部协议下的候选组合评价；One-Class SVM的逐指标优势和所有不利结果必须保留。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail": len(detail), "selected": len(selected), "methods": len(risks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
