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
SELECTED = TABLES / "stage88_noaa_physical_rule_cap_rolling_selected.csv"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [([2020], 2021, 2022), ([2020, 2021], 2022, 2023), ([2020, 2021, 2022], 2023, 2024), ([2020, 2021, 2022, 2023], 2024, 2025)]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, best_robust_setting, smooth_risk, structural_features, structural_risk_from_norm  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage88_noaa_physical_rule_cap_rolling import modified_base  # noqa: E402
from stage09_trust_pipeline import _metrics  # noqa: E402
from trust_fusion import TrustFusion  # noqa: E402


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


def counts(y: np.ndarray, p: np.ndarray) -> dict[str, int]:
    return {"tp": int(np.sum(y & p)), "fp": int(np.sum((~y) & p)), "fn": int(np.sum(y & (~p))), "tn": int(np.sum((~y) & (~p)))}


def metrics_from_counts(c: dict[str, int]) -> dict[str, float]:
    precision = c["tp"] / max(c["tp"] + c["fp"], 1)
    recall = c["tp"] / max(c["tp"] + c["fn"], 1)
    return {"f1": 2.0 * precision * recall / max(precision + recall, 1e-12), "recall": recall, "far": c["fp"] / max(c["fp"] + c["tn"], 1)}


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    selected = pd.read_csv(SELECTED)
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    by_method: dict[str, dict[str, dict[str, int]]] = {"capped": {}, "uncapped": {}}
    fold_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            scored = add_structural(scored)
            train = scored["split"].eq("train").to_numpy()
            valid = scored["split"].eq("valid").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
            yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            fusion = TrustFusion(include_era5=False).fit(scored.loc[train], scored.loc[valid], scored.loc[valid, "combined_is_attack"])
            ref_base = modified_base(fusion, scored, float("inf"))
            cap_row = selected[(selected["fold"].eq(fold)) & (selected["seed"].astype(int).eq(seed))]
            if len(cap_row) != 1:
                raise ValueError(f"missing selected cap for {fold}/{seed}")
            cap_value = cap_row.iloc[0]["physical_rule_cap"]
            cap = float("inf") if pd.isna(cap_value) else float(cap_value)
            cap_base = modified_base(fusion, scored, cap)
            structural = scored["structural_risk"].to_numpy(dtype=float)
            settings = {"uncapped": best_robust_setting(scored.loc[valid], ref_base[valid], structural[valid]), "capped": best_robust_setting(scored.loc[valid], cap_base[valid], structural[valid])}
            pred_map: dict[str, np.ndarray] = {}
            score_map: dict[str, np.ndarray] = {}
            for method, base in [("uncapped", ref_base), ("capped", cap_base)]:
                setting = settings[method]
                raw = float(setting["alpha_base"]) * base + float(setting["alpha_structural"]) * structural
                values = np.zeros(len(scored), dtype=float)
                values[valid] = smooth_risk(scored.loc[valid], raw[valid], int(setting["window"]))
                values[test] = smooth_risk(scored.loc[test], raw[test], int(setting["window"]))
                score_map[method] = values
                pred_map[method] = values >= float(setting["threshold"])
            ref_counts_fold = counts(yt, pred_map["uncapped"][test])
            cap_counts_fold = counts(yt, pred_map["capped"][test])
            fold_rows.append({"fold": fold, "seed": seed, "physical_rule_cap": None if not np.isfinite(cap) else cap, "uncapped_test_f1": metrics_from_counts(ref_counts_fold)["f1"], "capped_test_f1": metrics_from_counts(cap_counts_fold)["f1"], "uncapped_test_recall": metrics_from_counts(ref_counts_fold)["recall"], "capped_test_recall": metrics_from_counts(cap_counts_fold)["recall"], "uncapped_test_far": metrics_from_counts(ref_counts_fold)["far"], "capped_test_far": metrics_from_counts(cap_counts_fold)["far"], "uncapped_test_auprc": float(_metrics(yt, pred_map["uncapped"][test], score_map["uncapped"][test])["average_precision"]), "capped_test_auprc": float(_metrics(yt, pred_map["capped"][test], score_map["capped"][test])["average_precision"]), "uncapped_threshold": float(settings["uncapped"]["threshold"]), "capped_threshold": float(settings["capped"]["threshold"])})
            for method in ["uncapped", "capped"]:
                for station in stations:
                    mask = test & scored["station_key"].eq(station).to_numpy()
                    by_method[method][f"{fold}|{seed}|{station}"] = counts(scored.loc[mask, "combined_is_attack"].astype(bool).to_numpy(), pred_map[method][mask])
            selected_rows.append({"fold": fold, "seed": seed, "physical_rule_cap": None if not np.isfinite(cap) else cap, "uncapped_setting": settings["uncapped"], "capped_setting": settings["capped"]})
        print(json.dumps({"fold": fold, "seeds": len(SEEDS), "stations": len(stations)}, ensure_ascii=False), flush=True)

    station_counts = {method: {} for method in ["uncapped", "capped"]}
    for method in station_counts:
        for station in stations:
            total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            suffix = f"|{station}"
            for key, part in by_method[method].items():
                if key.endswith(suffix):
                    for field in total:
                        total[field] += part[field]
            station_counts[method][station] = total

    rng = np.random.default_rng(20260913)
    bootstrap_rows = []
    for b in range(5000):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        totals = {method: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for method in ["uncapped", "capped"]}
        for station in sampled:
            for method in totals:
                for field in totals[method]:
                    totals[method][field] += station_counts[method][str(station)][field]
        unc = metrics_from_counts(totals["uncapped"])
        cap = metrics_from_counts(totals["capped"])
        bootstrap_rows.append({"bootstrap": b, "f1_diff_capped_minus_uncapped": cap["f1"] - unc["f1"], "recall_diff_capped_minus_uncapped": cap["recall"] - unc["recall"], "far_diff_capped_minus_uncapped": cap["far"] - unc["far"]})
    boot = pd.DataFrame(bootstrap_rows)
    summary = {"f1_diff_mean": float(boot["f1_diff_capped_minus_uncapped"].mean()), "f1_diff_ci95": [float(boot["f1_diff_capped_minus_uncapped"].quantile(0.025)), float(boot["f1_diff_capped_minus_uncapped"].quantile(0.975))], "recall_diff_mean": float(boot["recall_diff_capped_minus_uncapped"].mean()), "recall_diff_ci95": [float(boot["recall_diff_capped_minus_uncapped"].quantile(0.025)), float(boot["recall_diff_capped_minus_uncapped"].quantile(0.975))], "far_diff_mean": float(boot["far_diff_capped_minus_uncapped"].mean()), "far_diff_ci95": [float(boot["far_diff_capped_minus_uncapped"].quantile(0.025)), float(boot["far_diff_capped_minus_uncapped"].quantile(0.975))], "prob_f1_positive": float((boot["f1_diff_capped_minus_uncapped"] > 0).mean()), "prob_far_lower": float((boot["far_diff_capped_minus_uncapped"] < 0).mean())}
    fold_df = pd.DataFrame(fold_rows)
    mean_gain = {"f1": float((fold_df["capped_test_f1"] - fold_df["uncapped_test_f1"]).mean()), "recall": float((fold_df["capped_test_recall"] - fold_df["uncapped_test_recall"]).mean()), "auprc": float((fold_df["capped_test_auprc"] - fold_df["uncapped_test_auprc"]).mean()), "far": float((fold_df["capped_test_far"] - fold_df["uncapped_test_far"]).mean())}
    status = "CANDIDATE_STABLE_REQUIRES_MORE_YEARS" if summary["f1_diff_ci95"][0] > 0 and summary["far_diff_ci95"][1] < 0 else "CANDIDATE_NOT_SIGNIFICANT"
    payload = {"status": status, "protocol": "阶段88选中上限；同一四折滚动协议、11站点、5种子；候选与无上限参考在每折/种子严格配对；站点簇5000次bootstrap；阈值和上限只由valid选择，test只评价。", "stations": stations, "seeds": SEEDS, "bootstraps": 5000, "fold_rows": fold_rows, "selected_settings": selected_rows, "station_counts": station_counts, "mean_gain_selected_vs_uncapped": mean_gain, "summary": summary}
    fold_df.to_csv(TABLES / "stage89_noaa_physical_rule_cap_bootstrap_fold_detail.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(TABLES / "stage89_noaa_physical_rule_cap_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage89_noaa_physical_rule_cap_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    s = summary
    lines = ["# 阶段89 NOAA物理规则上限站点簇级bootstrap", "", "阶段88 valid选中的Physical Rules上限候选与同一折/种子无上限参考进行站点簇级配对bootstrap。", "", f"- 状态：**{status}**", f"- F1差值：`{s['f1_diff_mean']:.4f}`，95% CI `[{s['f1_diff_ci95'][0]:.4f}, {s['f1_diff_ci95'][1]:.4f}]`；候选F1更高概率：`{s['prob_f1_positive']:.4f}`", f"- Recall差值：`{s['recall_diff_mean']:.4f}`，95% CI `[{s['recall_diff_ci95'][0]:.4f}, {s['recall_diff_ci95'][1]:.4f}]`", f"- FAR差值：`{s['far_diff_mean']:.4f}`，95% CI `[{s['far_diff_ci95'][0]:.4f}, {s['far_diff_ci95'][1]:.4f}]`；候选FAR更低概率：`{s['prob_far_lower']:.4f}`", "", "该结果决定是否值得增加更多年份；在统计稳定前不替换9B主模型。"]
    (REPORTS / "stage89_noaa_physical_rule_cap_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "mean_gain_selected_vs_uncapped": mean_gain, "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
