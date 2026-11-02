from __future__ import annotations

"""Stage 148: baseline attack-type detail for the frozen NOAA rolling protocol.

This reruns the exact stage-85 rolling baseline protocol only to recover
per-attack-type recalls for Robust, Isolation Forest, One-Class SVM and the
Trust legacy reference. It does not select a new model or tune on test.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _best_threshold  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage85_noaa_rolling_baseline_comparison import (  # noqa: E402
    AUDIT,
    FOLDS,
    NOAA,
    SCORE_COLS,
    SEEDS,
    add_structural,
    build_fold,
)

ATTACK_TYPES = ["spike", "offset", "drift", "time_shift", "stuck", "gaussian_noise"]
METHODS = ["Robust max deviation", "Isolation Forest", "One-Class SVM", "TrustMeteo-QC"]


def threshold_predictions(y_valid: np.ndarray, valid_score: np.ndarray, test_score: np.ndarray) -> tuple[np.ndarray, float, float]:
    threshold, valid_f1 = _best_threshold(y_valid, valid_score)
    return test_score >= threshold, float(threshold), float(valid_f1)


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    rows: list[dict[str, object]] = []
    integrity: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold_name = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, trust = score(frame)
            scored = add_structural(scored)
            train = scored["split"].eq("train").to_numpy()
            valid = scored["split"].eq("valid").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            normal_train = train & (~scored["combined_is_attack"].astype(bool).to_numpy())
            y_valid = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
            X = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            med = np.nanmedian(X[normal_train], axis=0)
            scale = np.maximum(1.4826 * np.nanmedian(np.abs(X[normal_train] - med), axis=0), 0.25)
            robust_score = np.maximum((X - med) / scale, 0.0).max(axis=1)
            iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=1, contamination="auto").fit(X[normal_train])
            iso_score = -iso.score_samples(X)
            svm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)).fit(X[normal_train])
            svm_score = -svm.decision_function(X).ravel()
            candidates: list[tuple[str, np.ndarray, np.ndarray]] = [
                ("Robust max deviation", robust_score[valid], robust_score[test]),
                ("Isolation Forest", iso_score[valid], iso_score[test]),
                ("One-Class SVM", svm_score[valid], svm_score[test]),
                ("TrustMeteo-QC", scored.loc[valid, "robust_fused_risk"].to_numpy(dtype=float), scored.loc[test, "robust_fused_risk"].to_numpy(dtype=float)),
            ]
            for method, valid_score, test_score in candidates:
                if method == "TrustMeteo-QC":
                    threshold = float(trust["setting"]["threshold"])
                    valid_f1 = float(trust["valid"]["f1"])
                    pred = test_score >= threshold
                else:
                    pred, threshold, valid_f1 = threshold_predictions(y_valid, valid_score, test_score)
                test_frame = scored.loc[test].copy()
                for attack_type in ATTACK_TYPES:
                    mask = test_frame["combined_attack_type"].astype(str).eq(attack_type).to_numpy()
                    type_pred = pred[mask]
                    rows.append({"fold": fold_name, "seed": int(seed), "method": method, "attack_type": attack_type, "rows": int(mask.sum()), "detected_rows": int(type_pred.sum()), "recall": float(type_pred.mean()) if len(type_pred) else float("nan"), "threshold": threshold, "valid_f1": valid_f1})
            integrity.append({"fold": fold_name, "seed": int(seed), "rows": int(len(frame)), "stations": int(frame["station_key"].nunique()), "train_attack_rows": int(frame.loc[frame["split"].eq("train"), "combined_is_attack"].sum()), "valid_attack_rows": int(frame.loc[frame["split"].eq("valid"), "combined_is_attack"].sum()), "test_attack_rows": int(frame.loc[frame["split"].eq("test"), "combined_is_attack"].sum())})
        print(json.dumps({"fold": fold_name, "methods": len(METHODS), "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(rows)
    aggregate = detail.groupby(["method", "attack_type"], sort=False)[["rows", "detected_rows", "recall"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["method", "attack_type"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[2:]]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "严格复用阶段85 NOAA官方2020–2025四折滚动、多年度、五种子和train正常样本拟合协议，仅补充四个基线的攻击类型级test Recall；每个方法阈值只由valid选择，test只评价，不重新选择候选、不根据test调参。攻击为透明受控注入，不是真实攻击日志。",
        "source_artifact": "stage85_noaa_rolling_baseline_comparison_detail.csv",
        "stations": stations,
        "folds": [f"{'-'.join(map(str, a))}->{b}->{c}" for a, b, c in FOLDS if set([*a, b, c]).issubset(years)],
        "seeds": SEEDS,
        "methods": METHODS,
        "attack_types": ATTACK_TYPES,
        "detail_rows": len(detail),
        "aggregate_rows": len(aggregate),
        "detail": detail.to_dict(orient="records"),
        "aggregate": aggregate.to_dict(orient="records"),
        "integrity": integrity,
        "limitations": ["本阶段补充的是阶段85同协议的类型级明细，不增加新的数据源或攻击生成器。", "攻击标签仍是透明受控注入，不是真实网络攻击日志。", "类型级Recall差异必须和总体F1/AUPRC/FAR以及阶段147的expanded/default交互审计一起解释。"],
    }
    prefix = "stage148_noaa_baseline_attack_type_detail"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / f"{prefix}_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段148 NOAA基线攻击类型级明细", "", "严格复用阶段85的四折滚动和五种子协议，补充Robust、Isolation Forest、One-Class SVM和Trust legacy的逐攻击类型Recall。", "", "| 方法 | 攻击类型 | 平均Recall | 标准差 | 最小 | 最大 |", "|---|---|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['attack_type']} | {row['recall_mean']:.4f} | {row['recall_std']:.4f} | {row['recall_min']:.4f} | {row['recall_max']:.4f} |")
    lines += ["", "阶段141没有保存One-Class SVM的类型级明细；本阶段按阶段85原协议重新计算，未使用test结果选择方法或参数。", "攻击为透明受控注入，不是真实攻击日志。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "aggregate_rows": len(aggregate)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
