from __future__ import annotations

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
COMPLETE_TYPES = ["spike", "offset", "drift", "time_shift", "stuck", "gaussian_noise"]
VARIANTS = ["legacy", "component_zmax", "component_logrepeat_zmax", "component_logrepeat_zmax_cap8"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, best_robust_setting, structural_features, structural_risk_from_norm, smooth_risk  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from trust_fusion import TrustFusion, robust_location_scale  # noqa: E402


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def _split_components(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    part = frame.loc[frame["split"].eq(split), ["station_key", "timestamp_utc", "observed_value"]].copy()
    part["_value"] = pd.to_numeric(part["observed_value"], errors="coerce")
    part["_timestamp"] = pd.to_datetime(part["timestamp_utc"], utc=True, errors="coerce")
    part = part.sort_values(["station_key", "_timestamp"], kind="stable")
    parts: list[pd.DataFrame] = []
    for station, group in part.groupby("station_key", sort=False):
        values = group["_value"]
        diff = values.diff().abs()
        same = diff.fillna(np.inf).le(1e-8)
        run = same.groupby(group["station_key"], sort=False).cumsum().to_numpy(dtype=float)
        parts.append(pd.DataFrame({"index": group.index, "range": np.maximum.reduce([-40.0 - values.to_numpy(dtype=float), values.to_numpy(dtype=float) - 50.0, np.zeros(len(group))]) / 0.5, "rate_raw": diff.to_numpy(dtype=float), "repeat_raw": np.maximum(run - 2.0, 0.0)}, index=group.index))
    if not parts:
        return pd.DataFrame(index=frame.index, columns=["range", "rate_raw", "repeat_raw"], dtype=float)
    out = pd.concat(parts).drop(columns=["index"]).reindex(frame.index)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def component_physical_scores(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    train = frame["split"].eq("train") & ~frame["combined_is_attack"].astype(bool)
    by_split = {split: _split_components(frame, split) for split in ["train", "valid", "test"]}
    train_part = by_split["train"]
    finite_rates = train_part["rate_raw"]
    rate_scale = max(float(finite_rates.quantile(0.999)) if len(finite_rates) else 1.0, 0.25)
    for part in by_split.values():
        part["rate"] = part["rate_raw"] / rate_scale
    # Fit every physical sub-rule separately on train normals. This prevents a
    # long natural repeated run from inflating the scale of range/rate rules.
    stats: dict[str, tuple[float, float]] = {}
    for col in ["range", "rate", "repeat_raw"]:
        stats[col] = robust_location_scale(train_part.loc[train, col], floor=0.25)
    transformed: dict[str, np.ndarray] = {}
    for variant in VARIANTS[1:]:
        values = np.zeros(len(frame), dtype=float)
        for split in ["train", "valid", "test"]:
            part = by_split[split]
            repeat = np.log1p(part["repeat_raw"].to_numpy(dtype=float))
            if variant == "component_logrepeat_zmax_cap8":
                repeat_stats = robust_location_scale(np.log1p(train_part.loc[train, "repeat_raw"]), floor=0.25)
            else:
                repeat_stats = stats["repeat_raw"]
            components = [
                np.maximum(part["range"].to_numpy(dtype=float) - stats["range"][0], 0.0) / stats["range"][1],
                np.maximum(part["rate"].to_numpy(dtype=float) - stats["rate"][0], 0.0) / stats["rate"][1],
                np.maximum(repeat - repeat_stats[0], 0.0) / repeat_stats[1],
            ]
            z = np.maximum.reduce(components)
            if variant.endswith("cap8"):
                z = np.minimum(z, 8.0)
            values[part.index.to_numpy()] = z
        transformed[variant] = values
    raw_legacy = []
    legacy_values = np.zeros(len(frame), dtype=float)
    for split in ["train", "valid", "test"]:
        part = by_split[split]
        legacy_values[part.index.to_numpy()] = np.maximum.reduce([part["range"].to_numpy(dtype=float), part["rate"].to_numpy(dtype=float), part["repeat_raw"].to_numpy(dtype=float)])
    transformed["legacy"] = legacy_values
    diagnostics = {
        "rate_scale": rate_scale,
        "component_stats": {key: {"location": float(value[0]), "scale": float(value[1])} for key, value in stats.items()},
        "train_quantiles": {variant: {str(q): float(np.nanquantile(values[train.to_numpy()], q)) for q in [0.5, 0.95, 0.99, 0.999, 1.0]} for variant, values in transformed.items()},
        "train_saturation_at_legacy_caps": {str(cap): float(np.mean(transformed["legacy"][train.to_numpy()] >= cap)) for cap in [1.0, 2.0, 3.0, 5.0, 10.0]},
    }
    return transformed, diagnostics


def evaluate_variant(scored: pd.DataFrame, physical: np.ndarray, variant: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    out = scored.copy()
    out["Physical Rules"] = physical
    train = out["split"].eq("train")
    valid = out["split"].eq("valid")
    test = out["split"].eq("test")
    structural_parts = []
    for split in ["train", "valid", "test"]:
        mask = out["split"].eq(split)
        structural_parts.append(structural_features(out.loc[mask], STRUCTURAL_BASE).set_axis(out.index[mask]))
    features = pd.concat(structural_parts).reindex(out.index).fillna(0.0)
    features["station_key"] = out["station_key"].astype(str)
    scaler = AdaptiveScaler(STRUCTURAL_BASE).fit(features.loc[train], train)
    structural = structural_risk_from_norm(scaler.transform(features), enhanced=False)
    fusion = TrustFusion(include_era5=False).fit(out.loc[train], out.loc[valid], out.loc[valid, "combined_is_attack"])
    base = fusion.transform(out)
    setting = best_robust_setting(out.loc[valid], base[valid.to_numpy()], structural[valid.to_numpy()])
    raw = setting["alpha_base"] * base + setting["alpha_structural"] * structural
    risk = np.full(len(out), np.nan, dtype=float)
    pred = np.zeros(len(out), dtype=bool)
    for split in ["train", "valid", "test"]:
        mask = out["split"].eq(split)
        risk[mask.to_numpy()] = smooth_risk(out.loc[mask], raw[mask.to_numpy()], int(setting["window"]))
        pred[mask.to_numpy()] = risk[mask.to_numpy()] >= float(setting["threshold"])
    valid_y = out.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    test_y = out.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    valid_metrics = _metrics(valid_y, pred[valid.to_numpy()], risk[valid.to_numpy()])
    test_metrics = _metrics(test_y, pred[test.to_numpy()], risk[test.to_numpy()])
    type_rows = []
    for attack_type in COMPLETE_TYPES:
        mask = test & out["combined_attack_type"].eq(attack_type)
        yt = out.loc[mask, "combined_is_attack"].astype(bool).to_numpy()
        pp = pred[mask.to_numpy()]
        type_rows.append({"variant": variant, "attack_type": attack_type, "rows": int(mask.sum()), "detected_rows": int(pp.sum()), "recall": float(pp.mean()) if len(pp) else float("nan")})
    valid_pred = pred[valid.to_numpy()]
    test_pred = pred[test.to_numpy()]
    valid_far = float(valid_pred[~valid_y].mean()) if (~valid_y).any() else float("nan")
    test_far = float(test_pred[~test_y].mean()) if (~test_y).any() else float("nan")
    result = {"variant": variant, "valid_f1": float(valid_metrics["f1"]), "valid_recall": float(valid_metrics["recall"]), "valid_auprc": float(valid_metrics["average_precision"]), "valid_far": valid_far, "test_f1": float(test_metrics["f1"]), "test_recall": float(test_metrics["recall"]), "test_precision": float(test_metrics["precision"]), "test_auprc": float(test_metrics["average_precision"]), "test_far": test_far, "threshold": float(setting["threshold"]), "alpha_base": float(setting["alpha_base"]), "alpha_structural": float(setting["alpha_structural"]), "window": int(setting["window"]), "train_natural_alarm_rate": float(pred[train.to_numpy()].mean())}
    return result, type_rows


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    detail: list[dict[str, object]] = []
    by_type: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            physical, diag = component_physical_scores(frame)
            for variant in VARIANTS:
                result, type_rows = evaluate_variant(scored, physical[variant], variant)
                detail.append({"fold": fold, "train_years": "-".join(map(str, train_years)), "valid_year": valid_year, "test_year": test_year, "seed": seed, **result})
                by_type.extend([{**row, "fold": fold, "seed": seed, "test_year": test_year} for row in type_rows])
                diagnostics.append({"fold": fold, "seed": seed, "variant": variant, **diag})
            candidates = [x for x in detail if x["fold"] == fold and x["seed"] == seed]
            chosen = max(candidates, key=lambda x: (x["valid_f1"], -x["valid_far"], -x["threshold"]))
            selected.append({"fold": fold, "seed": seed, "selected_variant": chosen["variant"], "selected_valid_f1": chosen["valid_f1"], "selected_test_f1": chosen["test_f1"], "selected_test_recall": chosen["test_recall"], "selected_test_auprc": chosen["test_auprc"], "selected_test_far": chosen["test_far"]})
        print(json.dumps({"fold": fold, "variants": len(VARIANTS), "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)
    detail_df = pd.DataFrame(detail)
    type_df = pd.DataFrame(by_type)
    diagnostics_df = pd.DataFrame(diagnostics)
    selected_df = pd.DataFrame(selected)
    aggregate = detail_df.groupby("variant", sort=False)[["valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far", "train_natural_alarm_rate"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["variant"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    selected_summary = {key: {"mean": float(selected_df[key].mean()), "std": float(selected_df[key].std(ddof=1)), "min": float(selected_df[key].min()), "max": float(selected_df[key].max())} for key in ["selected_valid_f1", "selected_test_f1", "selected_test_recall", "selected_test_auprc", "selected_test_far"]}
    legacy = aggregate.loc[aggregate["variant"].eq("legacy")].iloc[0]
    candidate = aggregate.loc[aggregate["variant"].eq("component_zmax")].iloc[0]
    selected_test_f1_gain = selected_summary["selected_test_f1"]["mean"] - float(legacy["test_f1_mean"])
    candidate_gain = {"f1": float(candidate["test_f1_mean"] - legacy["test_f1_mean"]), "recall": float(candidate["test_recall_mean"] - legacy["test_recall_mean"]), "auprc": float(candidate["test_auprc_mean"] - legacy["test_auprc_mean"]), "far": float(candidate["test_far_mean"] - legacy["test_far_mean"])}
    candidate_status = "CANDIDATE_LOW_FAR_REQUIRES_BOOTSTRAP" if candidate_gain["far"] < 0 and candidate_gain["auprc"] > 0 else "NEGATIVE_RESULT_RETAINED"
    payload = {"status": candidate_status, "candidate_variant": "component_zmax", "protocol": "NOAA官方2020–2025扩展表；11个全年度质量合格站点；四个累计历史训练→前一年验证→目标年测试滚动折；5种子；六类透明受控攻击；组件修复方案预先固定，每折仅用valid选择运行设置，test只评价。", "stations": stations, "folds": [f"{'-'.join(map(str, a))}->{b}->{c}" for a, b, c in FOLDS], "seeds": SEEDS, "variants": VARIANTS, "detail": detail, "aggregate": aggregate.to_dict(orient="records"), "by_type": type_df.to_dict(orient="records"), "diagnostics": diagnostics, "selected_by_valid": selected, "selected_summary": selected_summary, "selected_test_f1_gain_vs_legacy": selected_test_f1_gain, "candidate_gain_vs_legacy": candidate_gain, "limitations": ["component_zmax尚未经过站点簇bootstrap；不得替换9B主模型。", "组件方案明显牺牲Recall，只能作为低误报运行点候选。", "所有攻击均为透明受控注入，不是自然攻击日志。", "自然观测报警率需结合阶段87/90解释，不能当作攻击召回率。"]}
    detail_df.to_csv(TABLES / "stage91_noaa_physical_component_repair_detail.csv", index=False, encoding="utf-8-sig")
    type_df.to_csv(TABLES / "stage91_noaa_physical_component_repair_by_type.csv", index=False, encoding="utf-8-sig")
    diagnostics_df.to_csv(TABLES / "stage91_noaa_physical_component_repair_diagnostics.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage91_noaa_physical_component_repair_selected.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage91_noaa_physical_component_repair.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage91_noaa_physical_component_repair.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段91 NOAA物理规则组件级修复基准", "", "将范围、变化率和重复段拆开进行训练期稳健归一化；四折滚动、五种子、六类受控攻击；每折运行设置只由valid选择。", "", f"- 状态：**{payload['status']}**", f"- 预先固定低误报候选 `component_zmax` 相对 legacy：F1 `{candidate_gain['f1']:+.4f}`；Recall `{candidate_gain['recall']:+.4f}`；AUPRC `{candidate_gain['auprc']:+.4f}`；FAR `{candidate_gain['far']:+.4f}`", f"- 按valid选择方案后 test F1 相对 legacy：`{selected_test_f1_gain:+.4f}`（验证集无法区分组件方案）", "", "| 方案 | test F1 | Recall | Precision | AUPRC | FAR | train自然报警率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['variant']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} | {row['train_natural_alarm_rate_mean']:.4f} |")
    lines += ["", "component_zmax只作为低误报运行点候选；它牺牲了大量Recall，必须经过站点簇配对bootstrap和自然观测复核后，才可讨论是否进入主模型。"]
    (REPORTS / "stage91_noaa_physical_component_repair.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "variants": len(VARIANTS), "folds": len(FOLDS), "seeds": len(SEEDS), "selected_test_f1_gain_vs_legacy": selected_test_f1_gain}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
