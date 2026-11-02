from __future__ import annotations

"""Stage 152: station-level external baseline comparison.

The three baseline detectors are recomputed under the exact stage-85 rolling
protocol.  Trust default and frozen expanded station rows are read from the
already-frozen stage-141 artifact.  No candidate or threshold is selected
from test results in this stage.
"""

import hashlib
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
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"
AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
FROZEN_STATION = TABLES / "stage141_noaa_frozen_external_validation_by_station.csv"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
SCORE_COLS = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency", "structural_risk"]
METHODS = ["TrustMeteo-QC default", "TrustMeteo-QC frozen expanded", "Robust max deviation", "Isolation Forest", "One-Class SVM"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402
from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, structural_features, structural_risk_from_norm  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402


def add_structural(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    train = out["split"].eq("train")
    normal_train = train & (~out["combined_is_attack"].astype(bool))
    parts = []
    for split_name in ["train", "valid", "test"]:
        mask = out["split"].eq(split_name)
        parts.append(structural_features(out.loc[mask], STRUCTURAL_BASE).set_axis(out.index[mask]))
    features = pd.concat(parts).reindex(out.index).fillna(0.0)
    features["station_key"] = out["station_key"].astype(str)
    scaler = AdaptiveScaler(STRUCTURAL_BASE).fit(features.loc[train], normal_train.loc[train])
    out["structural_risk"] = structural_risk_from_norm(scaler.transform(features), enhanced=False)
    return out


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def station_rows(scored: pd.DataFrame, predictions: np.ndarray, method: str, fold: str, seed: int) -> list[dict[str, object]]:
    test = scored["split"].eq("test").to_numpy()
    part = scored.loc[test].copy()
    y = part["combined_is_attack"].astype(bool).to_numpy()
    rows = []
    for station in sorted(part["station_key"].astype(str).unique()):
        mask = part["station_key"].astype(str).eq(station).to_numpy()
        metrics = _metrics(y[mask], predictions[mask])
        rows.append({"fold": fold, "seed": seed, "method": method, "station": station, "rows": int(mask.sum()), "attack_rows": int(y[mask].sum()), "f1": float(metrics["f1"]), "recall": float(metrics["recall"]), "precision": float(metrics["precision"]), "far": float(predictions[mask][~y[mask]].mean()) if (~y[mask]).any() else float("nan")})
    return rows


def bootstrap_station(values: pd.DataFrame, metric: str, repeats: int = 20000, seed: int = 152) -> tuple[float, float]:
    stations = sorted(values["station"].astype(str).unique())
    grouped = values.groupby("station", sort=True)[metric].mean()
    rng = np.random.default_rng(seed)
    samples = rng.choice(stations, size=(repeats, len(stations)), replace=True)
    means = grouped.reindex(samples.ravel()).to_numpy().reshape(repeats, len(stations)).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    frozen = pd.read_csv(FROZEN_STATION)
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    station_rows_all = frozen[frozen["method"].isin(["TrustMeteo-QC default", "TrustMeteo-QC frozen expanded"])].copy()
    if len(station_rows_all) != len(FOLDS) * len(SEEDS) * len(stations) * 2:
        raise RuntimeError("stage141 station artifact is incomplete")
    rows: list[dict[str, object]] = station_rows_all.to_dict(orient="records")
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        name = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            scored = add_structural(scored)
            train = scored["split"].eq("train").to_numpy()
            valid = scored["split"].eq("valid").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            normal_train = train & (~scored["combined_is_attack"].astype(bool).to_numpy())
            yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
            X = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            med = np.nanmedian(X[normal_train], axis=0)
            scale = np.maximum(1.4826 * np.nanmedian(np.abs(X[normal_train] - med), axis=0), 0.25)
            robust = np.maximum((X - med) / scale, 0.0).max(axis=1)
            iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=1, contamination="auto").fit(X[normal_train])
            iso_score = -iso.score_samples(X)
            svm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)).fit(X[normal_train])
            svm_score = -svm.decision_function(X).ravel()
            for method, scores in [("Robust max deviation", robust), ("Isolation Forest", iso_score), ("One-Class SVM", svm_score)]:
                threshold, _ = _best_threshold(yv, scores[valid])
                rows.extend(station_rows(scored, scores[test] >= threshold, method, name, seed))
        print(json.dumps({"fold": name, "seeds": len(SEEDS), "methods": 3, "status": "completed"}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(rows)
    detail["station"] = detail["station"].astype(str)
    aggregate = detail.groupby(["method", "station"], sort=True)[["f1", "recall", "precision", "far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["method", "station"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[2:]]
    method_summary = detail.groupby("method", sort=True)[["f1", "recall", "precision", "far"]].mean().reset_index()
    paired = []
    expanded = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    for other in ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]:
        baseline = detail[detail["method"].eq(other)]
        left = expanded.groupby("station", sort=True)[["f1", "recall", "precision", "far"]].mean()
        right = baseline.groupby("station", sort=True)[["f1", "recall", "precision", "far"]].mean()
        for metric in ["f1", "recall", "precision", "far"]:
            diffs = (left[metric] - right[metric]).rename("diff").reset_index()
            low, high = bootstrap_station(diffs, "diff", seed=stable_seed("stage152", other, metric))
            paired.append({"comparison": f"expanded - {other}", "metric": metric, "station_count": len(diffs), "mean_station_diff": float(diffs["diff"].mean()), "station_bootstrap_ci95_low": low, "station_bootstrap_ci95_high": high, "stations_positive": int((diffs["diff"] > 0).sum()), "stations_negative": int((diffs["diff"] < 0).sum())})
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020–2025扩展表；11个质量合格站点；严格复用阶段85四折滚动、五种子和train正常样本拟合；Robust/Isolation Forest/One-Class SVM阈值只由valid选择，test只评价。Trust default/expanded站点明细复用阶段141冻结结果；按站点聚合后做11站点簇bootstrap；不重新选择候选、不根据test调参。攻击为透明受控注入，不是真实攻击日志。", "stations": stations, "folds": [fold_name(*fold) for fold in FOLDS], "seeds": SEEDS, "methods": METHODS, "detail": detail.to_dict(orient="records"), "station_aggregate": aggregate.to_dict(orient="records"), "method_summary": method_summary.to_dict(orient="records"), "paired": paired, "limitations": ["站点bootstrap只有11个站点簇，区间用于描述站点异质性，不等于跨地区普适性。", "Trust站点明细来自阶段141冻结外部结果，三种基线按阶段85同协议重算。", "攻击为透明受控注入，不是真实攻击日志；弱站点不能删除。"]}
    prefix = "stage152_noaa_baseline_station_comparison"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / f"{prefix}_station_aggregate.csv", index=False, encoding="utf-8-sig")
    method_summary.to_csv(TABLES / f"{prefix}_method_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired).to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段152 NOAA站点级基线比较", "", "Trust default/expanded复用阶段141冻结结果；Robust、Isolation Forest和One-Class SVM严格复用阶段85协议重算；按11站点聚合并做站点簇bootstrap。", "", "| 方法 | F1 | Recall | Precision | FAR |", "|---|---:|---:|---:|---:|"]
    for row in method_summary.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['f1']:.4f} | {row['recall']:.4f} | {row['precision']:.4f} | {row['far']:.4f} |")
    lines += ["", "## 扩展候选相对其他方法的站点簇差异", "", "| 比较 | 指标 | 平均站点差异 | CI95 | 正向站点 | 负向站点 |", "|---|---|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_station_diff']:+.4f} | [{row['station_bootstrap_ci95_low']:+.4f}, {row['station_bootstrap_ci95_high']:+.4f}] | {row['stations_positive']} | {row['stations_negative']} |")
    lines += ["", "站点簇数量有限，结果用于揭示异质性和总体指标是否被少数站点主导；攻击为透明受控注入。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "stations": len(stations), "methods": len(METHODS), "paired_rows": len(paired)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
