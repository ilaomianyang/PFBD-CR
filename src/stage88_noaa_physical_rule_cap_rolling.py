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
FOLDS = [([2020], 2021, 2022), ([2020, 2021], 2022, 2023), ([2020, 2021, 2022], 2023, 2024), ([2020, 2021, 2022, 2023], 2024, 2025)]
CAPS = [float("inf"), 10.0, 5.0, 3.0, 2.0, 1.0]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, best_robust_setting, smooth_risk, structural_features, structural_risk_from_norm  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
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


def modified_base(fusion: TrustFusion, frame: pd.DataFrame, cap: float) -> np.ndarray:
    total = np.zeros(len(frame), dtype=float)
    for name in fusion.components:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        med, scale = fusion.location_scale[name]
        z = np.nan_to_num(np.maximum(values - med, 0.0) / scale, nan=0.0, posinf=1e6)
        if name == "Physical Rules" and np.isfinite(cap):
            z = np.minimum(z, cap)
        total += fusion.weights.get(name, 0.0) * z
    missing = frame.get("missing_observation", pd.Series(False, index=frame.index)).astype(bool).to_numpy()
    total[missing] += 1.0
    return total


def evaluate(yv: np.ndarray, yt: np.ndarray, rv: np.ndarray, rt: np.ndarray, threshold: float, cap: float, fold: str, seed: int, setting: dict[str, object]) -> dict[str, object]:
    mv = _metrics(yv, rv >= threshold, rv)
    mt = _metrics(yt, rt >= threshold, rt)
    return {"fold": fold, "seed": seed, "physical_rule_cap": None if not np.isfinite(cap) else cap, "valid_f1": float(mv["f1"]), "valid_recall": float(mv["recall"]), "valid_auprc": float(mv["average_precision"]), "valid_far": float((rv[~yv] >= threshold).mean()), "test_f1": float(mt["f1"]), "test_recall": float(mt["recall"]), "test_precision": float(mt["precision"]), "test_auprc": float(mt["average_precision"]), "test_far": float((rt[~yt] >= threshold).mean()), "threshold": float(threshold), "setting": setting}


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
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
            fold_candidates: list[dict[str, object]] = []
            for cap in CAPS:
                base = modified_base(fusion, scored, cap)
                structural = scored["structural_risk"].to_numpy(dtype=float)
                setting = best_robust_setting(scored.loc[valid], base[valid], structural[valid])
                raw = float(setting["alpha_base"]) * base + float(setting["alpha_structural"]) * structural
                rv = smooth_risk(scored.loc[valid], raw[valid], int(setting["window"]))
                rt = smooth_risk(scored.loc[test], raw[test], int(setting["window"]))
                row = evaluate(yv, yt, rv, rt, float(setting["threshold"]), cap, fold, seed, {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in setting.items()})
                rows.append(row)
                fold_candidates.append(row)
            chosen = max(fold_candidates, key=lambda x: (x["valid_f1"], -x["valid_far"], -(x["physical_rule_cap"] if x["physical_rule_cap"] is not None else 1e9)))
            selected.append(chosen)
        print(json.dumps({"fold": fold, "seeds": len(SEEDS), "caps": len(CAPS)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame([{k: v for k, v in row.items() if k != "setting"} for row in rows])
    selected_df = pd.DataFrame([{k: v for k, v in row.items() if k != "setting"} for row in selected])
    # Summarize every cap candidate; the selected-vs-reference comparison below
    # is paired by fold and seed and never selects using test metrics.
    aggregate = detail.groupby("physical_rule_cap", dropna=False, sort=False)[["valid_f1", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["physical_rule_cap"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    reference = detail[detail["physical_rule_cap"].isna()].copy()
    reference = reference[["fold", "seed", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].rename(columns={"test_f1": "reference_test_f1", "test_recall": "reference_test_recall", "test_precision": "reference_test_precision", "test_auprc": "reference_test_auprc", "test_far": "reference_test_far"})
    paired = selected_df.merge(reference, on=["fold", "seed"], how="left", validate="one_to_one")
    ref_f1 = float(paired["reference_test_f1"].mean())
    ref_ap = float(paired["reference_test_auprc"].mean())
    ref_far = float(paired["reference_test_far"].mean())
    selected_f1 = float(paired["test_f1"].mean())
    selected_ap = float(paired["test_auprc"].mean())
    selected_far = float(paired["test_far"].mean())
    candidate_gain = bool(selected_f1 > ref_f1 and selected_ap >= ref_ap and selected_far <= ref_far)
    status = "CANDIDATE_GAIN_REQUIRES_BOOTSTRAP" if candidate_gain else "NEGATIVE_RESULT_RETAINED"
    payload = {"status": status, "protocol": "NOAA官方2020-2025扩展表；11个全年度合格站点；四个累计历史训练→前一年验证→目标年测试滚动折；Physical Rules分量上限、融合设置和阈值只由每折valid选择；test只评价；攻击为透明受控注入。", "stations": stations, "folds": [f"{'-'.join(map(str, a))}->{b}->{c}" for a, b, c in FOLDS], "seeds": SEEDS, "caps": [None if not np.isfinite(x) else x for x in CAPS], "detail": rows, "selected_by_valid": selected, "aggregate_all_caps": aggregate.to_dict(orient="records"), "selected_vs_uncapped": {"selected_test_f1": selected_f1, "selected_test_auprc": selected_ap, "selected_test_far": selected_far, "reference_test_f1": ref_f1, "reference_test_auprc": ref_ap, "reference_test_far": ref_far, "f1_gain": selected_f1 - ref_f1, "auprc_gain": selected_ap - ref_ap, "far_gain": selected_far - ref_far}, "candidate_gain_before_bootstrap": candidate_gain}
    detail.to_csv(TABLES / "stage88_noaa_physical_rule_cap_rolling_detail.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage88_noaa_physical_rule_cap_rolling_selected.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage88_noaa_physical_rule_cap_rolling_summary.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage88_noaa_physical_rule_cap_rolling.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段88 NOAA滚动物理规则分量上限", "", "Physical Rules分量上限只由每折valid选择，重新计算融合、平滑和阈值；test只评价。", "", f"- 状态：**{status}**；固定无上限方案test F1：`{ref_f1:.4f}`；AUPRC：`{ref_ap:.4f}`；FAR：`{ref_far:.4f}`", "", "| cap | valid F1 | test F1 | Recall | Precision | AUPRC | FAR |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        cap = "∞" if pd.isna(row["physical_rule_cap"]) else f"{row['physical_rule_cap']:.1f}"
        lines.append(f"| {cap} | {row['valid_f1_mean']:.4f}±{row['valid_f1_std']:.4f} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f} | {row['test_precision_mean']:.4f} | {row['test_auprc_mean']:.4f} | {row['test_far_mean']:.4f} |")
    lines += ["", "只有候选在多种子、跨站点统计中同时改善主要指标，才可进入后续主模型讨论；否则保留为负结果。"]
    (REPORTS / "stage88_noaa_physical_rule_cap_rolling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "selected_vs_uncapped": payload["selected_vs_uncapped"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
