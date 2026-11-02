from __future__ import annotations

"""Stage 174: station and time concentration of low-FAR alarms.

This stage does not train or select a model. It replays the exact frozen
stage-172 scoring protocol, then records where each method's alarms fall.
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
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"
AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
FROZEN = TABLES / "stage141_noaa_frozen_external_validation_selected.csv"
STAGE172 = TABLES / "stage172_noaa_frozen_expanded_low_far_full_baseline.json"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [([2020], 2021, 2022), ([2020, 2021], 2022, 2023), ([2020, 2021, 2022], 2023, 2024), ([2020, 2021, 2022, 2023], 2024, 2025)]
Q_GRID = [0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
METHODS = ["TrustMeteo-QC frozen expanded", "TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components  # noqa: E402
from stage151_noaa_attack_intensity_sensitivity import build_fold_scaled  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage85_noaa_rolling_baseline_comparison import SCORE_COLS  # noqa: E402


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def share(values: pd.Series, fraction: float) -> float | None:
    total = float(values.sum())
    if total <= 0.0:
        return None
    ordered = np.sort(values.to_numpy(dtype=float))[::-1]
    count = max(1, int(np.ceil(len(ordered) * fraction)))
    return float(ordered[:count].sum() / total)


def concentration(rows: pd.DataFrame, entity: str) -> dict[str, object]:
    group_column = "station_id" if entity == "station" else entity
    counts = rows.groupby(group_column, sort=False)["alarm_count"].sum()
    total = float(counts.sum())
    if total <= 0.0:
        return {
            f"{entity}_count": int(len(counts)),
            f"{entity}_alarm_share_top10pct": None,
            f"{entity}_alarm_share_max": None,
            f"{entity}_alarm_gini": None,
            f"{entity}_alarm_hhi": None,
        }
    shares = counts / total
    ordered = np.sort(shares.to_numpy(dtype=float))
    n = len(ordered)
    gini = float((2.0 * np.dot(np.arange(1, n + 1), ordered) - (n + 1)) / n) if n else None
    return {
        f"{entity}_count": int(n),
        f"{entity}_alarm_share_top10pct": share(counts, 0.10),
        f"{entity}_alarm_share_max": float(shares.max()),
        f"{entity}_alarm_gini": gini,
        f"{entity}_alarm_hhi": float(np.square(shares).sum()),
    }


def metric_row(fold: str, seed: int, method: str, q: float, threshold: float, y: np.ndarray, pred: np.ndarray, risk: np.ndarray) -> dict[str, object]:
    metrics = _metrics(y, pred, risk)
    return {
        "fold": fold,
        "seed": int(seed),
        "method": method,
        "q": float(q),
        "threshold_train_normal": float(threshold),
        "test_rows": int(len(y)),
        "test_attack_rows": int(y.sum()),
        "test_normal_rows": int((~y).sum()),
        "test_alarm_count": int(pred.sum()),
        "test_true_positive": int((pred & y).sum()),
        "test_false_positive": int((pred & (~y)).sum()),
        "test_false_negative": int((~pred & y).sum()),
        "test_true_negative": int((~pred & (~y)).sum()),
        "test_f1": float(metrics["f1"]),
        "test_recall": float(metrics["recall"]),
        "test_precision": float(metrics["precision"]),
        "test_auprc": float(metrics["auprc"]) if "auprc" in metrics else float(metrics["average_precision"]),
        "test_false_alarm_rate": float(pred[~y].mean()) if (~y).any() else None,
    }


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    frozen = pd.read_csv(FROZEN)
    stage172 = json.loads(STAGE172.read_text(encoding="utf-8"))
    expected = {
        (str(row["fold"]), int(row["seed"]), str(row["method"]), round(float(row["q"]), 10)): float(row["test_f1"])
        for row in stage172["detail"]
    }
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = {station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)}
    frozen = frozen[frozen["method"].eq("TrustMeteo-QC frozen expanded")].copy().set_index(["fold", "seed"])
    rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    month_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            alpha = float(frozen.loc[(fold, seed), "alpha_base"])
            window = int(frozen.loc[(fold, seed), "window"])
            frame = build_fold_scaled(source, seed, train_years, valid_year, test_year, stations, 1.0)
            scored, _ = score(frame)
            scored = add_components(scored)
            train = scored["split"].eq("train").to_numpy()
            test_mask = scored["split"].eq("test").to_numpy()
            normal_train = train & (~scored["combined_is_attack"].astype(bool).to_numpy())
            y_test = scored.loc[test_mask, "combined_is_attack"].astype(bool).to_numpy()
            x = scored[SCORE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            med = np.nanmedian(x[normal_train], axis=0)
            scale = np.maximum(1.4826 * np.nanmedian(np.abs(x[normal_train] - med), axis=0), 0.25)
            robust = np.maximum((x - med) / scale, 0.0).max(axis=1)
            iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=1, contamination="auto").fit(x[normal_train])
            iso_risk = -iso.score_samples(x)
            svm = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)).fit(x[normal_train])
            svm_risk = -svm.decision_function(x).ravel()
            base = scored["base_fused_risk"].to_numpy(dtype=float)
            structural = scored["structural_risk"].to_numpy(dtype=float)
            expanded_raw = alpha * base + (1.0 - alpha) * structural
            expanded_risk = np.full(len(scored), np.nan, dtype=float)
            for split_name in ["train", "valid", "test"]:
                mask = scored["split"].eq(split_name).to_numpy()
                expanded_risk[mask] = smooth_risk(scored.loc[mask], expanded_raw[mask], window)
            risks = {
                "TrustMeteo-QC frozen expanded": expanded_risk,
                "TrustMeteo-QC default": scored["robust_fused_risk"].to_numpy(dtype=float),
                "Robust max deviation": robust,
                "Isolation Forest": iso_risk,
                "One-Class SVM": svm_risk,
            }
            test_frame = scored.loc[test_mask].copy()
            for method in METHODS:
                risk = risks[method]
                train_values = risk[normal_train & np.isfinite(risk)]
                for q in Q_GRID:
                    threshold = float(np.nanquantile(train_values, q))
                    pred = risk[test_mask] >= threshold
                    row = metric_row(fold, seed, method, q, threshold, y_test, pred, risk[test_mask])
                    rows.append(row)
                    key = (fold, seed, method, round(float(q), 10))
                    if key not in expected:
                        raise RuntimeError(f"missing stage172 comparison key: {key}")
                    if abs(float(row["test_f1"]) - expected[key]) > 1e-10:
                        raise RuntimeError(f"stage172 replay mismatch for {key}: {row['test_f1']} vs {expected[key]}")
                    alarms = pd.DataFrame({
                        "station_id": test_frame["station_key"].astype(str).to_numpy(),
                        "month": pd.to_datetime(test_frame["timestamp_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m").to_numpy(),
                        "alarm": pred.astype(int),
                        "attack": y_test.astype(int),
                    })
                    for station, group in alarms.groupby("station_id", sort=False):
                        station_rows.append({
                            "fold": fold,
                            "seed": int(seed),
                            "method": method,
                            "q": float(q),
                            "station_id": station,
                            "test_rows": int(len(group)),
                            "test_attack_rows": int(group["attack"].sum()),
                            "test_normal_rows": int((~group["attack"].astype(bool)).sum()),
                            "alarm_count": int(group["alarm"].sum()),
                            "true_positive": int(((group["alarm"] == 1) & (group["attack"] == 1)).sum()),
                            "false_positive": int(((group["alarm"] == 1) & (group["attack"] == 0)).sum()),
                            "false_negative": int(((group["alarm"] == 0) & (group["attack"] == 1)).sum()),
                            "true_negative": int(((group["alarm"] == 0) & (group["attack"] == 0)).sum()),
                        })
                    for month, group in alarms.groupby("month", sort=False):
                        month_rows.append({
                            "fold": fold,
                            "seed": int(seed),
                            "method": method,
                            "q": float(q),
                            "month": month,
                            "test_rows": int(len(group)),
                            "test_attack_rows": int(group["attack"].sum()),
                            "test_normal_rows": int((~group["attack"].astype(bool)).sum()),
                            "alarm_count": int(group["alarm"].sum()),
                            "true_positive": int(((group["alarm"] == 1) & (group["attack"] == 1)).sum()),
                            "false_positive": int(((group["alarm"] == 1) & (group["attack"] == 0)).sum()),
                        })
        print(json.dumps({"fold": fold, "seeds": len(SEEDS), "status": "completed"}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(rows)
    station_detail = pd.DataFrame(station_rows)
    month_detail = pd.DataFrame(month_rows)
    burden_rows: list[dict[str, object]] = []
    for (method, q), group in detail.groupby(["method", "q"], sort=False):
        station_part = station_detail[station_detail["method"].eq(method) & np.isclose(station_detail["q"].astype(float), float(q))]
        month_part = month_detail[month_detail["method"].eq(method) & np.isclose(month_detail["q"].astype(float), float(q))]
        row = {"method": method, "q": float(q), "fold_seed_cells": int(len(group)), "alarm_count": int(group["test_alarm_count"].sum()), "false_positive_count": int(group["test_false_positive"].sum()), "attack_count": int(group["test_attack_rows"].sum()), "test_rows": int(group["test_rows"].sum())}
        row.update(concentration(station_part, "station"))
        row.update(concentration(month_part, "month"))
        burden_rows.append(row)
    burden = pd.DataFrame(burden_rows)
    prefix = "stage174_noaa_low_far_alarm_burden_audit"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    station_detail.to_csv(TABLES / f"{prefix}_station_detail.csv", index=False, encoding="utf-8-sig")
    month_detail.to_csv(TABLES / f"{prefix}_month_detail.csv", index=False, encoding="utf-8-sig")
    burden.to_csv(TABLES / f"{prefix}_burden.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "复用阶段172的train/valid/test四折滚动、五种子、五种方法和六个预先固定q运行点；不重新训练、不重新选择阈值或方法。对冻结重放结果逐站点和逐月记录TP/FP/FN/TN与报警数，并计算报警集中度。攻击为透明受控注入，不是真实攻击日志；集中度只用于运行负担诊断，不能解释为真实入侵率。",
        "source_artifact": "stage172_noaa_frozen_expanded_low_far_full_baseline.json",
        "methods": METHODS,
        "q_grid": Q_GRID,
        "detail_rows": int(len(detail)),
        "station_detail_rows": int(len(station_detail)),
        "month_detail_rows": int(len(month_detail)),
        "burden_rows": int(len(burden)),
        "replay_mismatch_count": 0,
        "detail": detail.to_dict(orient="records"),
        "burden": burden.to_dict(orient="records"),
        "limitations": [
            "四个滚动fold和11个站点限制集中度结论；不同方法不能仅凭集中度排名。",
            "报警集中度是运行负担诊断，不等于检测召回或真实攻击率。",
            "NOAA攻击是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 阶段174 NOAA低误报报警负担与聚集性审计", "", "复用阶段172冻结协议，逐站点和逐月记录报警分布。", "", "| 方法 | q | 报警数 | FP数 | 站点前10%报警份额 | 最大站点份额 | 月份前10%报警份额 | 最大月份份额 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in burden.to_dict(orient="records"):
        def fmt(value: object) -> str:
            return "NA" if value is None or pd.isna(value) else f"{float(value):.4f}"
        lines.append(f"| {row['method']} | {row['q']:.4f} | {row['alarm_count']} | {row['false_positive_count']} | {fmt(row['station_alarm_share_top10pct'])} | {fmt(row['station_alarm_share_max'])} | {fmt(row['month_alarm_share_top10pct'])} | {fmt(row['month_alarm_share_max'])} |")
    lines += ["", f"明细={len(detail)}；站点明细={len(station_detail)}；月份明细={len(month_detail)}；冻结重放不一致=0。", "该阶段不改变模型或阈值；报警集中度不能单独作为模型优越性的证据。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "station_rows": len(station_detail), "month_rows": len(month_detail), "burden_rows": len(burden)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
