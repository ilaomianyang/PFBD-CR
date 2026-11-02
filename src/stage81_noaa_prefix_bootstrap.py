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
NOAA = TABLES / "stage70_noaa_isd_multiyear_normalized.parquet"
AUDIT = TABLES / "stage71_noaa_multiyear_quality_audit.json"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
BOOTSTRAPS = 5000
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage73_noaa_multiyear_controlled_benchmark import build_dataset  # noqa: E402
from stage79_noaa_prefix_adaptation import prefix_mask  # noqa: E402
from stage80_noaa_constrained_prefix_adaptation import blended_predictions, evaluate, select_point  # noqa: E402


def metric(y: np.ndarray, p: np.ndarray, s: np.ndarray) -> tuple[float, float, float, float]:
    m = _metrics(y, p, s)
    far = float(p[~y].mean()) if (~y).any() else float("nan")
    return float(m["f1"]), float(m["recall"]), float(m["average_precision"]), far


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_multiyear_validation")}
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in [2021, 2022, 2023])})
    source = source[source["station_id"].astype(str).isin(stations) & source["year"].isin([2021, 2022, 2023])].copy()
    by_seed_station: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    selected_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        frame = build_dataset(source, seed, set(stations))
        scored, fixed = score(frame)
        scored["year"] = pd.to_numeric(scored["year"], errors="coerce").astype(int)
        scored["station_key"] = scored["station_key"].astype(str)
        fixed_threshold = float(fixed["setting"]["threshold"])
        candidates: list[dict[str, object]] = []
        for days in [7, 30, 90]:
            for q in [0.95, 0.975, 0.99, 0.995, 0.999]:
                for shrinkage in [0.25, 0.5, 0.75, 1.0]:
                    pred, thresholds = blended_predictions(scored, 2022, days, q, shrinkage, fixed_threshold)
                    valid = evaluate(scored, pred, 2022, days)
                    candidates.append({"seed": seed, "prefix_days": days, "quantile": q, "shrinkage": shrinkage, "valid_f1": valid["f1"], "valid_recall": valid["recall"], "valid_far": valid["far"], "valid_auprc": valid["auprc"], "valid_prefix_attack_rows": valid["prefix_attack_rows"]})
        chosen = select_point(candidates, "max_f1")
        if chosen is None:
            raise RuntimeError("max_f1 candidate unexpectedly unavailable")
        days = int(chosen["prefix_days"])
        candidate_pred, candidate_thresholds = blended_predictions(scored, 2023, days, float(chosen["quantile"]), float(chosen["shrinkage"]), fixed_threshold)
        fixed_pred = scored["robust_fused_risk"].to_numpy(dtype=float) >= fixed_threshold
        seed_map: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for station in stations:
            test = scored["year"].eq(2023).to_numpy() & scored["station_key"].eq(station).to_numpy() & ~prefix_mask(scored, 2023, days)
            seed_map[station] = {"candidate": {"y": scored.loc[test, "combined_is_attack"].astype(bool).to_numpy(), "p": candidate_pred[test], "s": scored.loc[test, "robust_fused_risk"].to_numpy(dtype=float)}, "fixed": {"y": scored.loc[test, "combined_is_attack"].astype(bool).to_numpy(), "p": fixed_pred[test], "s": scored.loc[test, "robust_fused_risk"].to_numpy(dtype=float)}}
        by_seed_station[seed] = seed_map
        chosen_row = {**chosen, "test_fixed_threshold": fixed_threshold, "test_threshold_min": float(min(candidate_thresholds.values())), "test_threshold_median": float(np.median(list(candidate_thresholds.values()))), "test_threshold_max": float(max(candidate_thresholds.values()))}
        selected_rows.append(chosen_row)
        print(json.dumps({"seed": seed, "chosen": chosen_row}, ensure_ascii=False), flush=True)

    rng = np.random.default_rng(20260912)
    boot_rows: list[dict[str, object]] = []
    for b in range(BOOTSTRAPS):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        arrays = {"candidate": {"y": [], "p": [], "s": []}, "fixed": {"y": [], "p": [], "s": []}}
        for station in sampled:
            for seed in SEEDS:
                for model in ["candidate", "fixed"]:
                    for key in ["y", "p", "s"]:
                        arrays[model][key].append(by_seed_station[seed][station][model][key])
        cy, cp, cs = (np.concatenate(arrays["candidate"][key]) for key in ["y", "p", "s"])
        fy, fp, fs = (np.concatenate(arrays["fixed"][key]) for key in ["y", "p", "s"])
        cf1, crec, cap, cfar = metric(cy, cp, cs)
        ff1, frec, fap, ffar = metric(fy, fp, fs)
        boot_rows.append({"bootstrap": b, "candidate_f1": cf1, "fixed_f1": ff1, "f1_diff": cf1 - ff1, "candidate_recall": crec, "fixed_recall": frec, "recall_diff": crec - frec, "candidate_auprc": cap, "fixed_auprc": fap, "auprc_diff": cap - fap, "candidate_far": cfar, "fixed_far": ffar, "far_diff": cfar - ffar})
    boot = pd.DataFrame(boot_rows)
    def ci(column: str) -> list[float]:
        return [float(boot[column].quantile(0.025)), float(boot[column].quantile(0.975))]
    summary = {"f1_diff_mean": float(boot["f1_diff"].mean()), "f1_diff_ci95": ci("f1_diff"), "recall_diff_mean": float(boot["recall_diff"].mean()), "recall_diff_ci95": ci("recall_diff"), "auprc_diff_mean": float(boot["auprc_diff"].mean()), "auprc_diff_ci95": ci("auprc_diff"), "far_diff_mean": float(boot["far_diff"].mean()), "far_diff_ci95": ci("far_diff"), "prob_f1_positive": float((boot["f1_diff"] > 0).mean()), "prob_far_lower": float((boot["far_diff"] < 0).mean())}
    status = "CANDIDATE_NOT_SIGNIFICANT" if summary["f1_diff_ci95"][0] <= 0 or summary["prob_f1_positive"] < 0.95 else "CANDIDATE_GAIN_REQUIRES_MORE_YEARS"
    payload = {"status": status, "protocol": "2023 locked test；按11站点簇、5种子、5000次配对bootstrap；候选是2022 valid选择的max-F1前缀收缩方案；不使用2023标签调参。", "stations": stations, "seeds": SEEDS, "bootstraps": BOOTSTRAPS, "selected_by_valid": selected_rows, "summary": summary}
    boot.to_csv(TABLES / "stage81_noaa_prefix_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage81_noaa_prefix_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    s = summary
    lines = ["# 阶段81 NOAA前缀自适应站点簇级bootstrap", "", "以站点为重采样簇，比较阶段80验证期选择的max-F1前缀收缩候选与固定阈值在2023 locked test上的差异。", "", f"- 状态：**{status}**", f"- F1差值均值：`{s['f1_diff_mean']:.4f}`，95% CI `[{s['f1_diff_ci95'][0]:.4f}, {s['f1_diff_ci95'][1]:.4f}]`；候选F1更高概率：`{s['prob_f1_positive']:.4f}`", f"- Recall差值：`{s['recall_diff_mean']:.4f}`，95% CI `[{s['recall_diff_ci95'][0]:.4f}, {s['recall_diff_ci95'][1]:.4f}]`", f"- FAR差值：`{s['far_diff_mean']:.4f}`，95% CI `[{s['far_diff_ci95'][0]:.4f}, {s['far_diff_ci95'][1]:.4f}]`；候选FAR更低概率：`{s['prob_far_lower']:.4f}`", "", "该结果只用于判断候选是否值得增加年份复核，不覆盖9B主模型、不修改锁定测试数据。"]
    (REPORTS / "stage81_noaa_prefix_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
