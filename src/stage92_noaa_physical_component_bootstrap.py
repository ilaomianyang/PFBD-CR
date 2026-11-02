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
BOOTSTRAPS = 5000

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, best_robust_setting, smooth_risk, structural_features, structural_risk_from_norm  # noqa: E402
from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage91_noaa_physical_component_repair import component_physical_scores  # noqa: E402
from trust_fusion import TrustFusion  # noqa: E402


def counts(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {"tp": int(np.sum(y & pred)), "fp": int(np.sum((~y) & pred)), "fn": int(np.sum(y & (~pred))), "tn": int(np.sum((~y) & (~pred)))}


def metrics_from_counts(c: dict[str, int]) -> dict[str, float]:
    precision = c["tp"] / max(c["tp"] + c["fp"], 1)
    recall = c["tp"] / max(c["tp"] + c["fn"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"f1": f1, "recall": recall, "precision": precision, "far": c["fp"] / max(c["fp"] + c["tn"], 1)}


def output_for_method(scored: pd.DataFrame, physical: np.ndarray, method: str) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    out = scored.copy()
    out["Physical Rules"] = physical
    train = out["split"].eq("train")
    valid = out["split"].eq("valid")
    test = out["split"].eq("test")
    parts = []
    for split in ["train", "valid", "test"]:
        mask = out["split"].eq(split)
        parts.append(structural_features(out.loc[mask], STRUCTURAL_BASE).set_axis(out.index[mask]))
    features = pd.concat(parts).reindex(out.index).fillna(0.0)
    features["station_key"] = out["station_key"].astype(str)
    structural = structural_risk_from_norm(AdaptiveScaler(STRUCTURAL_BASE).fit(features.loc[train], train).transform(features), enhanced=False)
    fusion = TrustFusion(include_era5=False).fit(out.loc[train], out.loc[valid], out.loc[valid, "combined_is_attack"])
    base = fusion.transform(out)
    setting = best_robust_setting(out.loc[valid], base[valid.to_numpy()], structural[valid.to_numpy()])
    raw = float(setting["alpha_base"]) * base + float(setting["alpha_structural"]) * structural
    risk = np.zeros(len(out), dtype=float)
    pred = np.zeros(len(out), dtype=bool)
    for split in ["train", "valid", "test"]:
        mask = out["split"].eq(split)
        risk[mask.to_numpy()] = smooth_risk(out.loc[mask], raw[mask.to_numpy()], int(setting["window"]))
        pred[mask.to_numpy()] = risk[mask.to_numpy()] >= float(setting["threshold"])
    yv = out.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    yt = out.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    valid_m = _metrics(yv, pred[valid.to_numpy()], risk[valid.to_numpy()])
    test_m = _metrics(yt, pred[test.to_numpy()], risk[test.to_numpy()])
    valid_far = float(pred[valid.to_numpy()][~yv].mean()) if (~yv).any() else float("nan")
    test_far = float(pred[test.to_numpy()][~yt].mean()) if (~yt).any() else float("nan")
    result = {"method": method, "valid_f1": float(valid_m["f1"]), "valid_recall": float(valid_m["recall"]), "valid_auprc": float(valid_m["average_precision"]), "valid_far": valid_far, "test_f1": float(test_m["f1"]), "test_recall": float(test_m["recall"]), "test_precision": float(test_m["precision"]), "test_auprc": float(test_m["average_precision"]), "test_far": test_far, "threshold": float(setting["threshold"]), "alpha_base": float(setting["alpha_base"]), "alpha_structural": float(setting["alpha_structural"]), "window": int(setting["window"])}
    return result, pred, risk


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    detail: list[dict[str, object]] = []
    station_counts = {method: {station: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for station in stations} for method in METHODS}
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            physical, _ = component_physical_scores(frame)
            outputs = {}
            for method in METHODS:
                result, pred, risk = output_for_method(scored, physical[method], method)
                outputs[method] = (result, pred, risk)
                detail.append({"fold": fold, "seed": seed, **result})
                test = scored["split"].eq("test").to_numpy()
                for station in stations:
                    mask = test & scored["station_key"].eq(station).to_numpy()
                    c = counts(scored.loc[mask, "combined_is_attack"].astype(bool).to_numpy(), pred[mask])
                    for field in c:
                        station_counts[method][station][field] += c[field]
            print(json.dumps({"fold": fold, "seed": seed, "methods": len(METHODS)}, ensure_ascii=False), flush=True)
        print(json.dumps({"fold_completed": fold, "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)
    detail_df = pd.DataFrame(detail)
    aggregate = detail_df.groupby("method", sort=False)[["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["method"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    rng = np.random.default_rng(20260913)
    boot_rows = []
    for b in range(BOOTSTRAPS):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        totals = {method: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for method in METHODS}
        for station in sampled:
            for method in METHODS:
                for field in totals[method]:
                    totals[method][field] += station_counts[method][str(station)][field]
        legacy = metrics_from_counts(totals["legacy"])
        candidate = metrics_from_counts(totals["component_zmax"])
        boot_rows.append({"bootstrap": b, "f1_diff_candidate_minus_legacy": candidate["f1"] - legacy["f1"], "recall_diff_candidate_minus_legacy": candidate["recall"] - legacy["recall"], "precision_diff_candidate_minus_legacy": candidate["precision"] - legacy["precision"], "far_diff_candidate_minus_legacy": candidate["far"] - legacy["far"]})
    boot = pd.DataFrame(boot_rows)
    summary = {}
    for metric in ["f1", "recall", "precision", "far"]:
        col = f"{metric}_diff_candidate_minus_legacy"
        summary[f"{metric}_diff_mean"] = float(boot[col].mean())
        summary[f"{metric}_diff_ci95"] = [float(boot[col].quantile(0.025)), float(boot[col].quantile(0.975))]
    summary["prob_f1_positive"] = float((boot["f1_diff_candidate_minus_legacy"] > 0).mean())
    summary["prob_far_lower"] = float((boot["far_diff_candidate_minus_legacy"] < 0).mean())
    status = "CANDIDATE_LOW_FAR_STABLE_REQUIRES_MORE_YEARS" if summary["f1_diff_ci95"][0] > 0 and summary["far_diff_ci95"][1] < 0 else "CANDIDATE_LOW_FAR_NOT_SIGNIFICANT"
    payload = {"status": status, "candidate_variant": "component_zmax", "protocol": "阶段91预先固定component_zmax；NOAA官方2020–2025扩展表；11站点；四折累计历史训练→前一年验证→目标年测试；5种子；test逐站点统计；5000次站点簇配对bootstrap；阈值和融合设置仅由每折valid选择。", "stations": stations, "folds": [f"{'-'.join(map(str, a))}->{b}->{c}" for a, b, c in FOLDS], "seeds": SEEDS, "bootstraps": BOOTSTRAPS, "detail": detail, "aggregate": aggregate.to_dict(orient="records"), "station_counts": station_counts, "summary": summary, "limitations": ["component_zmax明显牺牲Recall，只能作为低误报候选。", "候选未替换9B主模型。", "攻击为透明受控注入，不是自然攻击日志。", "仍需自然观测跨年份复核，不能把报警率写成攻击召回率。"]}
    detail_df.to_csv(TABLES / "stage92_noaa_physical_component_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(TABLES / "stage92_noaa_physical_component_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage92_noaa_physical_component_bootstrap.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage92_noaa_physical_component_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    s = summary
    lines = ["# 阶段92 NOAA物理规则组件候选站点簇bootstrap", "", "固定阶段91预先指定的component_zmax，与legacy在相同滚动折/种子下进行站点簇级配对bootstrap。", "", f"- 状态：**{status}**", f"- F1差值：`{s['f1_diff_mean']:.4f}`，95% CI `[{s['f1_diff_ci95'][0]:.4f}, {s['f1_diff_ci95'][1]:.4f}]`；候选F1更高概率：`{s['prob_f1_positive']:.4f}`", f"- Recall差值：`{s['recall_diff_mean']:.4f}`，95% CI `[{s['recall_diff_ci95'][0]:.4f}, {s['recall_diff_ci95'][1]:.4f}]`", f"- Precision差值：`{s['precision_diff_mean']:.4f}`，95% CI `[{s['precision_diff_ci95'][0]:.4f}, {s['precision_diff_ci95'][1]:.4f}]`", f"- FAR差值：`{s['far_diff_mean']:.4f}`，95% CI `[{s['far_diff_ci95'][0]:.4f}, {s['far_diff_ci95'][1]:.4f}]`；候选FAR更低概率：`{s['prob_far_lower']:.4f}`", "", "| 方法 | test F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} |")
    lines += ["", "该候选的主要价值是降低误报，而不是提高主检测Recall；在更多自然年份复核前不得替换9B。"]
    (REPORTS / "stage92_noaa_physical_component_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
