from __future__ import annotations

"""Stage 158: controlled NOAA attack performance by environment strata.

The attack labels here are transparent injections from the frozen external
protocol, not real attack logs.  This stage aligns controlled detection
performance with the natural-operation strata audited in stages 155/156.
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
FOLDS = [([2020], 2021, 2022), ([2020, 2021], 2022, 2023), ([2020, 2021, 2022], 2023, 2024), ([2020, 2021, 2022, 2023], 2024, 2025)]
SEEDS = [3407, 6257, 7919, 104729, 20260715]
METHODS = ["TrustMeteo-QC default", "TrustMeteo-QC frozen expanded"]
ALPHA = 0.25
WINDOW = 1

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402
from stage141_noaa_frozen_external_validation import add_components  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage155_noaa_natural_weather_regime_audit import local_regime  # noqa: E402
from stage156_noaa_natural_observation_gap_audit import gap_bucket  # noqa: E402


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def clean_context(scored: pd.DataFrame, source: pd.DataFrame, train_years: list[int]) -> pd.DataFrame:
    out = scored.copy()
    clean = source.copy()
    clean["station_key"] = clean["station_id"].astype(str)
    clean["timestamp_utc"] = pd.to_datetime(clean["timestamp_utc"], utc=True, errors="coerce")
    lookup = clean.drop_duplicates(["station_key", "timestamp_utc"]).set_index(["station_key", "timestamp_utc"])
    keys = pd.MultiIndex.from_frame(out[["station_key", "timestamp_utc"]].assign(station_key=out["station_key"].astype(str)))
    for source_col, target_col in [("temperature_c", "clean_temperature_c"), ("wind_speed_ms", "clean_wind_speed_ms")]:
        out[target_col] = pd.to_numeric(lookup[source_col].reindex(keys).to_numpy(), errors="coerce")
    train_mask = out["split"].eq("train")
    out["temperature_regime"] = local_regime(out.rename(columns={"clean_temperature_c": "temperature_c"}), "temperature_c", train_mask, {"low": "cold", "mid": "mild", "high": "hot"})
    out["gap_bucket"] = gap_bucket(out)
    return out


def finite_or_none(value: object) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def cluster_bootstrap(values: pd.DataFrame, metric: str, seed: int, repeats: int = 10000) -> tuple[float | None, float | None]:
    clusters = values["fold"].drop_duplicates().tolist()
    if len(clusters) < 2:
        return None, None
    by_cluster = {key: values.loc[values["fold"].eq(key), metric].to_numpy(dtype=float) for key in clusters}
    rng = np.random.default_rng(seed)
    sample = rng.choice(clusters, size=(repeats, len(clusters)), replace=True)
    means = np.array([np.concatenate([by_cluster[key] for key in row]).mean() for row in sample], dtype=float)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(value) for value in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    detail_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, default_result = score(frame)
            scored = add_components(scored)
            scored = clean_context(scored, source, train_years)
            test_mask = scored["split"].eq("test")
            y_test = scored.loc[test_mask, "combined_is_attack"].astype(bool).to_numpy()
            default_pred = scored.loc[test_mask, "predicted_attack"].astype(bool).to_numpy()
            default_score = scored.loc[test_mask, "robust_fused_risk"].to_numpy(dtype=float)
            base = scored["base_fused_risk"].to_numpy(dtype=float)
            structural = scored["structural_risk"].to_numpy(dtype=float)
            raw = ALPHA * base + (1.0 - ALPHA) * structural
            valid_mask = scored["split"].eq("valid")
            valid_score = smooth_risk(scored.loc[valid_mask], raw[valid_mask.to_numpy()], WINDOW)
            threshold, valid_f1 = _best_threshold(scored.loc[valid_mask, "combined_is_attack"].astype(bool).to_numpy(), valid_score)
            test_score = smooth_risk(scored.loc[test_mask], raw[test_mask.to_numpy()], WINDOW)
            expanded_pred = test_score >= threshold
            predictions = {"TrustMeteo-QC default": (default_pred, default_score), "TrustMeteo-QC frozen expanded": (expanded_pred, test_score)}
            test_context = scored.loc[test_mask].reset_index(drop=True)
            for method, (pred, risk) in predictions.items():
                for variable, labels in [("temperature_regime", ["cold", "mild", "hot", "missing"]), ("gap_bucket", ["regular_le_3h", "moderate_3_6h", "long_6_24h", "very_long_gt_24h"])]:
                    for label in labels:
                        mask = test_context[variable].eq(label).to_numpy()
                        if not mask.any():
                            continue
                        y = y_test[mask]
                        p = pred[mask]
                        m = _metrics(y, p, risk[mask])
                        detail_rows.append({"fold": fold, "seed": seed, "method": method, "test_year": test_year, "context_variable": variable, "context": label, "rows": int(mask.sum()), "attack_rows": int(y.sum()), "clean_rows": int((~y).sum()), "valid_f1": float(valid_f1) if method.endswith("expanded") else float(default_result["valid"]["f1"]), "test_f1": float(m["f1"]), "test_recall": float(m["recall"]), "test_precision": float(m["precision"]), "test_auprc": finite_or_none(m["average_precision"]), "test_false_alarm_rate": float(p[~y].mean()) if (~y).any() else None})
        print(json.dumps({"fold": fold, "seeds": len(SEEDS), "detail_rows": len(detail_rows)}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise ValueError("stage158 produced no stratified rows")
    paired_rows: list[dict[str, object]] = []
    left = detail[detail["method"].eq("TrustMeteo-QC frozen expanded")]
    right = detail[detail["method"].eq("TrustMeteo-QC default")]
    keys = ["fold", "seed", "test_year", "context_variable", "context"]
    common = left.set_index(keys).join(right.set_index(keys), lsuffix="_expanded", rsuffix="_default", how="inner").reset_index()
    for metric in ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]:
        common["diff"] = pd.to_numeric(common[f"{metric}_expanded"], errors="coerce") - pd.to_numeric(common[f"{metric}_default"], errors="coerce")
        for (variable, context), group in common.groupby(["context_variable", "context"], sort=True):
            group = group.dropna(subset=["diff"])
            if group.empty:
                continue
            low, high = cluster_bootstrap(group, "diff", stable_seed("stage158", metric, variable, context))
            if low is None or high is None:
                continue
            paired_rows.append({"comparison": "expanded - default", "metric": metric, "context_variable": variable, "context": context, "fold_seed_cells": int(len(group)), "mean_diff": float(group["diff"].mean()), "bootstrap_fold_ci95_low": low, "bootstrap_fold_ci95_high": high, "positive_cells": int((group["diff"] > 0).sum()), "negative_cells": int((group["diff"] < 0).sum())})

    prefix = "stage158_noaa_controlled_regime_gap_performance"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired_rows).to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020–2025扩展表和四折累计历史训练→valid→目标年test协议；攻击标签为透明受控注入，不是真实网络攻击日志。候选设置固定为阶段141/142冻结的default与alpha=0.25、window=1扩展方案；阈值只由每个fold的valid选择，test只按目标年气象状态和原始观测间隔分层评价，不根据分层test结果调参。NOAA质量字段仅作弱标签，不参与选择。", "stations": stations, "folds": [fold_name(*fold) for fold in FOLDS], "seeds": SEEDS, "methods": METHODS, "context_variables": {"temperature_regime": ["cold", "mild", "hot", "missing"], "gap_bucket": ["regular_le_3h", "moderate_3_6h", "long_6_24h", "very_long_gt_24h"]}, "detail_rows": int(len(detail)), "paired_rows": int(len(paired_rows)), "paired": paired_rows, "limitations": ["受控攻击不是真实攻击日志；状态和间隔分层只描述当前外部协议中的条件敏感性。", "部分长间隔状态样本较少，区间不支持普适结论。", "结果用于诊断环境条件下的召回—误报权衡，不回写冻结模型。"]}
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summary = detail.groupby(["method", "context_variable", "context"], sort=True).agg(cells=("fold", "size"), attack_rows=("attack_rows", "sum"), test_f1_mean=("test_f1", "mean"), test_recall_mean=("test_recall", "mean"), test_far_mean=("test_false_alarm_rate", "mean")).reset_index()
    lines = ["# 阶段158 NOAA受控攻击按气象状态和观测间隔分层", "", "攻击为透明受控注入；候选和阈值选择按冻结协议执行，目标年test只评价。", "", "| 方法 | 状态变量 | 状态 | 单元数 | 攻击行数 | F1均值 | Recall均值 | FAR均值 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['context_variable']} | {row['context']} | {int(row['cells'])} | {int(row['attack_rows'])} | {row['test_f1_mean']:.4f} | {row['test_recall_mean']:.4f} | {row['test_far_mean']:.4f} |")
    lines += ["", "## 扩展方案相对default的fold×seed成对差异", "", "| 指标 | 状态变量 | 状态 | 平均差 | CI95 | 正向单元 | 负向单元 |", "|---|---|---|---:|---:|---:|---:|"]
    for row in paired_rows:
        low = "NA" if row["bootstrap_fold_ci95_low"] is None else f"[{row['bootstrap_fold_ci95_low']:+.4f},{row['bootstrap_fold_ci95_high']:+.4f}]"
        lines.append(f"| {row['metric']} | {row['context_variable']} | {row['context']} | {row['mean_diff']:+.4f} | {low} | {row['positive_cells']} | {row['negative_cells']} |")
    lines += ["", "攻击标签为受控注入，不能写成真实入侵检测结果；分层结果不用于重新选择模型。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "detail_rows": len(detail), "paired_rows": len(paired_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
