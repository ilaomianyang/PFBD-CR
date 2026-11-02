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
ALPHAS = [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 1.0]
METHODS = ["svm_only", "trust_component_only", *[f"trust_svm_mean_alpha_{alpha:.2f}" for alpha in ALPHAS], "trust_svm_max"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage91_noaa_physical_component_repair import component_physical_scores  # noqa: E402
from stage92_noaa_physical_component_bootstrap import output_for_method  # noqa: E402
from stage98_noaa_low_far_baseline_comparison import add_structural, baseline_risks, score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402


def train_normal_z(risk: np.ndarray, train: np.ndarray, normal_train: np.ndarray) -> np.ndarray:
    values = risk[normal_train & np.isfinite(risk)]
    if values.size == 0:
        raise ValueError("no finite normal train risk values")
    med = float(np.nanmedian(values))
    scale = float(max(1.4826 * np.nanmedian(np.abs(values - med)), 0.25))
    return np.maximum((risk - med) / scale, 0.0)


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    rows: list[dict[str, object]] = []

    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            physical, _ = component_physical_scores(frame)
            scored = add_structural(scored)
            base_risks = baseline_risks(scored)
            trust = output_for_method(scored, physical["component_zmax"], "component_zmax")[2]
            svm = base_risks["One-Class SVM"]
            train = scored["split"].eq("train").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            normal_train = train & ~scored["combined_is_attack"].astype(bool).to_numpy()
            trust_z = train_normal_z(np.asarray(trust, dtype=float), train, normal_train)
            svm_z = train_normal_z(np.asarray(svm, dtype=float), train, normal_train)
            risks: dict[str, np.ndarray] = {
                "svm_only": svm_z,
                "trust_component_only": trust_z,
                "trust_svm_max": np.maximum(trust_z, svm_z),
            }
            for alpha in ALPHAS:
                risks[f"trust_svm_mean_alpha_{alpha:.2f}"] = alpha * trust_z + (1.0 - alpha) * svm_z
            labels = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            for method in METHODS:
                risk = risks[method]
                threshold = float(np.nanquantile(risk[normal_train], Q))
                pred = risk[test] >= threshold
                m = _metrics(labels, pred, risk[test])
                rows.append({"fold": fold, "seed": seed, "method": method, "alpha_trust": float(next((alpha for alpha in ALPHAS if method == f"trust_svm_mean_alpha_{alpha:.2f}"), np.nan)), "q": Q, "threshold": threshold, "test_f1": float(m["f1"]), "test_recall": float(m["recall"]), "test_precision": float(m["precision"]), "test_auprc": float(m["average_precision"]), "test_far": float(pred[~labels].mean()) if (~labels).any() else float("nan")})
            print(json.dumps({"fold": fold, "seed": seed, "methods": len(METHODS), "q": Q}, ensure_ascii=False), flush=True)
        print(json.dumps({"fold_completed": fold, "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("method", sort=False)[["test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["method"] + [f"{metric}_{stat}" for metric, stat in aggregate.columns.tolist()[1:]]
    best = aggregate.sort_values("test_f1_mean", ascending=False).iloc[0].to_dict()
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "NOAA 2020–2025；四折滚动、11站点、5种子；Trust component_zmax与One-Class SVM风险先按train正常样本稳健标准化；预先固定Trust权重网格和max融合；q=0.975固定；test只评价。",
        "q": Q,
        "alphas": ALPHAS,
        "methods": METHODS,
        "stations": stations,
        "detail": detail.to_dict(orient="records"),
        "aggregate": aggregate.to_dict(orient="records"),
        "screening_best_by_test_f1_descriptive_only": best,
        "limitations": [
            "本阶段是互补性筛选，权重网格预先固定；不能把test最高项直接固化为主模型。",
            "所有权重和两个单模型都必须报告；FAR降低若伴随Recall下降不能称为全面改进。",
            "攻击为透明受控注入，不是真实攻击日志。",
        ],
    }
    detail.to_csv(TABLES / "stage101_noaa_trust_svm_complementarity_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage101_noaa_trust_svm_complementarity.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage101_noaa_trust_svm_complementarity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段101 NOAA Trust–SVM分数互补筛选",
        "",
        "固定q=0.975和预先列出的Trust权重网格，比较Trust component_zmax与One-Class SVM的标准化均值/最大值融合。",
        "",
        "| 方法 | test F1 | Recall | Precision | AUPRC | FAR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} |")
    lines += ["", "筛选表用于发现互补候选；不得根据test最高F1直接替换主模型，后续候选必须重新进行valid预注册选择和站点簇bootstrap。"]
    (REPORTS / "stage101_noaa_trust_svm_complementarity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(detail), "methods": len(METHODS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
