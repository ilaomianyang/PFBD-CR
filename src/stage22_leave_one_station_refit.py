from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors import (  # noqa: E402
    CUSUMDetector,
    HampelMADDetector,
    KalmanInnovationDetector,
    PhysicalRulesDetector,
    SpatialConsistencyDetector,
)
from secure_recovery import add_recovery_candidates, choose_weights, combine_candidates  # noqa: E402
from stage09_trust_pipeline import (  # noqa: E402
    _event_metrics,
    _metrics,
    _recovery_metrics,
)
from stage09b_robust_fusion import (  # noqa: E402
    AdaptiveScaler,
    STRUCTURAL_BASE,
    best_robust_setting,
    structural_features,
    structural_risk_from_norm,
    smooth_risk,
)
from trust_fusion import TrustFusion, robust_location_scale  # noqa: E402


DEFAULT_LOCAL = ROOT / "tables" / "stage05_controlled_attack_extensions.parquet"
DEFAULT_ERA5 = ROOT / "tables" / "stage06_era5_station_reference.parquet"


def add_era5_features(local: pd.DataFrame, era5_path: Path, fit_station_mask: pd.Series) -> pd.DataFrame:
    """Join ERA5 and estimate station bias only from non-held-out valid rows."""
    out = local.copy()
    out["_row_id"] = np.arange(len(out), dtype=int)
    era = pd.read_parquet(era5_path).copy()
    era["timestamp_utc"] = pd.to_datetime(era["timestamp_utc"], utc=True, errors="coerce")
    era["hour_utc"] = era["timestamp_utc"].dt.floor("h")
    era = era.dropna(subset=["station_key", "hour_utc"]).drop_duplicates(["station_key", "hour_utc"])
    out["hour_utc"] = out["timestamp_utc"].dt.floor("h")
    out = out.merge(era, on=["station_key", "hour_utc"], how="left", suffixes=("", "_era5"), sort=False)
    fit_station_mask = fit_station_mask.reindex(local.index, fill_value=False)
    valid_fit = out["split"].eq("valid") & out["_row_id"].map(fit_station_mask.to_dict()).fillna(False).astype(bool)
    valid_fit &= out["original_value"].notna() & out["era5_t2m_c"].notna()
    bias = (
        out.loc[valid_fit]
        .assign(_bias=lambda d: pd.to_numeric(d["original_value"], errors="coerce") - pd.to_numeric(d["era5_t2m_c"], errors="coerce"))
        .groupby("station_key", as_index=False)["_bias"]
        .median()
        .rename(columns={"_bias": "era5_median_bias_c"})
    )
    out = out.merge(bias, on="station_key", how="left", sort=False)
    out["era5_corrected_t2m_c"] = pd.to_numeric(out["era5_t2m_c"], errors="coerce") + out["era5_median_bias_c"].fillna(0.0)
    observed = pd.to_numeric(out["effective_observed_value"], errors="coerce")
    out["ERA5 Background"] = (out["era5_corrected_t2m_c"] - observed).abs()
    ordered = out.sort_values(["station_key", "hour_utc", "_row_id"], kind="stable")
    extreme = ordered.groupby("station_key", sort=False)["era5_t2m_c"].diff().abs().ge(2.0).fillna(False)
    out["era5_extreme_background"] = extreme.reindex(out.index).fillna(False).to_numpy(dtype=bool)
    out = out.sort_values("_row_id", kind="stable").drop(columns=["hour_utc", "_row_id"])
    out.index = local.index
    return out


def prepare_local_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Equivalent to stage09 preparation, accepting an already loaded table."""
    df = source.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "station_key", "split"]).sort_values(
        ["split", "station_key", "timestamp_utc", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    base_attack = df.get("is_anomaly", False).astype(bool)
    ext_attack = df.get("extension_is_attack", False).astype(bool)
    ext_value = pd.to_numeric(df.get("extension_observed_value"), errors="coerce")
    observed = pd.to_numeric(df["observed_value"], errors="coerce")
    df["effective_observed_value"] = observed.where(~ext_attack | ext_value.isna(), ext_value)
    df["combined_is_attack"] = base_attack | ext_attack
    ext_type = df.get("extension_attack_type", "normal").fillna("normal").astype(str)
    base_type = df.get("anomaly_type", "normal").fillna("normal").astype(str)
    df["combined_attack_type"] = np.where(ext_attack, ext_type, base_type)
    ext_event = df.get("extension_event_id", "").fillna("").astype(str)
    base_event = df.get("anomaly_record_id", "").fillna("").astype(str)
    df["combined_event_id"] = np.where(ext_attack & ext_event.ne(""), ext_event, base_event)
    df["missing_observation"] = df["effective_observed_value"].isna()
    return df


def score_by_split(detector, frame: pd.DataFrame, fit_station_mask: pd.Series) -> np.ndarray:
    """Score train/valid on fitting stations; score all test stations for spatial context."""
    scores = np.full(len(frame), np.nan, dtype=float)
    for split in ["train", "valid", "test"]:
        mask = frame["split"].eq(split)
        if split != "test":
            mask &= fit_station_mask
        if mask.any():
            scores[mask.to_numpy()] = detector.score(frame.loc[mask])
    return scores


def structural_by_split(frame: pd.DataFrame, fit_station_mask: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index, columns=STRUCTURAL_BASE, dtype=float)
    for split in ["train", "valid", "test"]:
        mask = frame["split"].eq(split)
        if split != "test":
            mask &= fit_station_mask
        if mask.any():
            part = structural_features(frame.loc[mask], STRUCTURAL_BASE)
            out.loc[mask, STRUCTURAL_BASE] = part.to_numpy()
    return out.fillna(0.0)


def evaluate_holdout(
    local_source: pd.DataFrame,
    era5_path: Path,
    held_out: str,
    calibrate_site: bool = False,
) -> dict[str, object]:
    local = prepare_local_frame(local_source)
    fit_station_mask = local["station_key"].astype(str).ne(str(held_out))
    local = add_era5_features(local, era5_path, fit_station_mask)
    detector_frame = local.copy()
    detector_frame["observed_value"] = detector_frame["effective_observed_value"]

    train_mask = local["split"].eq("train") & fit_station_mask
    valid_mask = local["split"].eq("valid") & fit_station_mask
    test_mask = local["split"].eq("test")
    normal_train = train_mask & ~local["combined_is_attack"]
    site_train_mask = local["split"].eq("train") & ~fit_station_mask
    detectors = [PhysicalRulesDetector(), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    for detector in detectors:
        detector.fit(detector_frame.loc[train_mask], normal_train.loc[train_mask])
        local[detector.name] = score_by_split(detector, detector_frame, fit_station_mask)
        if calibrate_site and site_train_mask.any():
            all_train = detector_frame[detector_frame["split"].eq("train")]
            all_train_scores = pd.Series(detector.score(all_train), index=all_train.index)
            local.loc[site_train_mask, detector.name] = all_train_scores.loc[site_train_mask].to_numpy()

    features = structural_by_split(detector_frame, fit_station_mask)
    if calibrate_site and site_train_mask.any():
        all_train = detector_frame[detector_frame["split"].eq("train")]
        all_train_features = structural_features(all_train, STRUCTURAL_BASE)
        features.loc[site_train_mask, STRUCTURAL_BASE] = all_train_features.loc[site_train_mask, STRUCTURAL_BASE].to_numpy()
    features["station_key"] = local["station_key"].astype(str)
    scaler = AdaptiveScaler(STRUCTURAL_BASE).fit(features.loc[train_mask], normal_train.loc[train_mask])
    structural_norm = scaler.transform(features)
    structural_risk = structural_risk_from_norm(structural_norm, enhanced=False)

    fusion = TrustFusion(include_era5=True).fit(local.loc[train_mask], local.loc[valid_mask], local.loc[valid_mask, "combined_is_attack"])
    base_risk = fusion.transform(local, extreme_background=local["era5_extreme_background"])
    setting = best_robust_setting(local.loc[valid_mask], base_risk[valid_mask.to_numpy()], structural_risk[valid_mask.to_numpy()])
    raw_risk = setting["alpha_base"] * base_risk + setting["alpha_structural"] * structural_risk
    robust_risk = np.full(len(local), np.nan, dtype=float)
    pred = np.zeros(len(local), dtype=bool)
    for split, mask in [("train", train_mask), ("valid", valid_mask), ("test", test_mask)]:
        if not mask.any():
            continue
        values = smooth_risk(local.loc[mask], raw_risk[mask.to_numpy()], int(setting["window"]))
        robust_risk[mask.to_numpy()] = values
        pred[mask.to_numpy()] = values >= float(setting["threshold"])
    calibration = None
    if calibrate_site and site_train_mask.any():
        site_train_risk = smooth_risk(local.loc[site_train_mask], raw_risk[site_train_mask.to_numpy()], int(setting["window"]))
        robust_risk[site_train_mask.to_numpy()] = site_train_risk
        global_median, global_scale = robust_location_scale(robust_risk[normal_train.to_numpy()], floor=0.25)
        site_median, site_scale = robust_location_scale(site_train_risk, floor=0.25)
        threshold_z = max((float(setting["threshold"]) - global_median) / global_scale, 0.0)
        site_threshold = site_median + threshold_z * site_scale
        held_test_mask = test_mask & ~fit_station_mask
        pred[held_test_mask.to_numpy()] = robust_risk[held_test_mask.to_numpy()] >= site_threshold
        calibration = {
            "method": "unlabeled_train_median_mad_transfer",
            "uses_held_out_labels": False,
            "global_train_normal_median": global_median,
            "global_train_normal_scale": global_scale,
            "source_threshold": float(setting["threshold"]),
            "source_threshold_z": threshold_z,
            "site_train_rows": int(site_train_mask.sum()),
            "site_train_risk_median": site_median,
            "site_train_risk_scale": site_scale,
            "site_threshold": site_threshold,
        }
    local["base_fused_risk"] = base_risk
    local["structural_risk"] = structural_risk
    local["robust_fused_risk"] = robust_risk
    local["predicted_attack"] = pred

    valid_recovery = local.loc[valid_mask].copy()
    valid_recovery["observed_value"] = valid_recovery["effective_observed_value"]
    valid_recovery["is_anomaly"] = valid_recovery["combined_is_attack"]
    valid_candidates = add_recovery_candidates(valid_recovery, valid_recovery["predicted_attack"], valid_recovery["era5_corrected_t2m_c"])
    recovery_weights = choose_weights(valid_candidates)
    test = local.loc[test_mask].copy()
    test["observed_value"] = test["effective_observed_value"]
    test["is_anomaly"] = test["combined_is_attack"]
    test_candidates = add_recovery_candidates(test, test["predicted_attack"], test["era5_corrected_t2m_c"])
    test["recovered_value"] = combine_candidates(test_candidates, recovery_weights)
    for candidate_name in ["temporal_candidate", "spatial_candidate", "era5_candidate"]:
        test[candidate_name] = test_candidates[candidate_name].to_numpy()
    held = test[test["station_key"].astype(str).eq(str(held_out))].copy().reset_index(drop=True)
    metrics = _metrics(held["combined_is_attack"], held["predicted_attack"], held["robust_fused_risk"])
    clean = held[~held["combined_is_attack"]]
    false_alarm = float(clean["predicted_attack"].mean()) if len(clean) else float("nan")
    rec = _recovery_metrics(held, held["recovered_value"].to_numpy())
    events = _event_metrics(held, held["predicted_attack"].to_numpy())
    detail = held[[
        "station_key", "timestamp_utc", "combined_attack_type", "combined_is_attack",
        "combined_event_id", "original_value", "effective_observed_value",
        "era5_corrected_t2m_c", "robust_fused_risk", "predicted_attack", "recovered_value",
        "temporal_candidate", "spatial_candidate", "era5_candidate",
    ]].copy()
    detail.insert(0, "held_out_station", str(held_out))
    valid_detail = valid_candidates[[
        "station_key", "timestamp_utc", "combined_attack_type", "combined_is_attack",
        "combined_event_id", "original_value", "effective_observed_value",
        "era5_corrected_t2m_c", "robust_fused_risk", "predicted_attack",
        "temporal_candidate", "spatial_candidate", "era5_candidate",
    ]].copy()
    valid_detail.insert(0, "held_out_station", str(held_out))
    return {
        "held_out_station": str(held_out),
        "fit_stations": sorted(local.loc[fit_station_mask, "station_key"].astype(str).unique().tolist()),
        "fit_train_rows": int(train_mask.sum()),
        "fit_valid_rows": int(valid_mask.sum()),
        "held_out_test_rows": int(len(held)),
        "setting": setting,
        "fusion_weights": fusion.weights,
        "recovery_weights": recovery_weights,
        "test": metrics,
        "test_false_alarm_rate": false_alarm,
        "test_event_metrics": events,
        "test_recovery": rec,
        "site_calibration": calibration,
        "detail": detail,
        "valid_detail": valid_detail,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段22：严格留一站点重新拟合审计")
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    args = parser.parse_args()
    source = pd.read_parquet(args.local)
    stations = sorted(source["station_key"].astype(str).unique().tolist())
    results = []
    detail_frames = []
    valid_detail_frames = []
    for i, station in enumerate(stations, start=1):
        print(json.dumps({"progress": f"{i}/{len(stations)}", "held_out_station": station}, ensure_ascii=False), flush=True)
        current = evaluate_holdout(source, args.era5, station)
        detail_frames.append(current.pop("detail"))
        valid_detail_frames.append(current.pop("valid_detail"))
        results.append(current)
    rows = []
    for r in results:
        m = r["test"]
        rows.append({
            "held_out_station": r["held_out_station"],
            "fit_train_rows": r["fit_train_rows"],
            "fit_valid_rows": r["fit_valid_rows"],
            "test_rows": r["held_out_test_rows"],
            "threshold": r["setting"]["threshold"],
            "valid_f1_other_stations": r["setting"]["valid_f1"],
            "test_f1": m["f1"],
            "test_recall": m["recall"],
            "test_precision": m["precision"],
            "test_auprc": m["average_precision"],
            "test_false_alarm_rate": r["test_false_alarm_rate"],
            "recovery_mae_before": r["test_recovery"]["mae_before"],
            "recovery_mae_after": r["test_recovery"]["mae_after"],
            "event_recall": r["test_event_metrics"]["event_recall"],
        })
    result = pd.DataFrame(rows)
    detail = pd.concat(detail_frames, ignore_index=True)
    valid_detail = pd.concat(valid_detail_frames, ignore_index=True)
    type_rows = []
    from sklearn.metrics import f1_score, precision_score, recall_score
    for (station, attack_type), group in detail.groupby(["held_out_station", "combined_attack_type"], sort=True):
        y = group["combined_is_attack"].astype(bool).to_numpy()
        p = group["predicted_attack"].astype(bool).to_numpy()
        if not y.any():
            continue
        type_rows.append({
            "held_out_station": station,
            "attack_type": attack_type,
            "rows": int(len(group)),
            "attack_rows": int(y.sum()),
            "detected_rows": int(p[y].sum()),
            "recall": float(recall_score(y, p, zero_division=0)),
            "precision": float(precision_score(y, p, zero_division=0)),
            "f1": float(f1_score(y, p, zero_division=0)),
            "mean_risk_attack": float(group.loc[group["combined_is_attack"], "robust_fused_risk"].mean()),
            "mean_risk_normal": float(group.loc[~group["combined_is_attack"], "robust_fused_risk"].mean()) if (~group["combined_is_attack"]).any() else float("nan"),
        })
    by_type = pd.DataFrame(type_rows)
    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "stage22_leave_one_station_refit.csv", index=False, encoding="utf-8-sig")
    detail.to_parquet(out_dir / "stage22_leave_one_station_refit_detail.parquet", index=False)
    valid_detail.to_parquet(out_dir / "stage22_leave_one_station_refit_valid_detail.parquet", index=False)
    by_type.to_csv(out_dir / "stage22_leave_one_station_refit_by_type.csv", index=False, encoding="utf-8-sig")
    public_results = [{k: v for k, v in item.items() if k != "detail"} for item in results]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "每个留出站点独立重新拟合：留出站点不参与检测器、ERA5偏差、尺度、融合设置、阈值和恢复权重；测试时允许使用同一时间戳的跨站点观测计算空间证据，不使用任何测试标签。",
        "source_local": str(args.local),
        "source_era5": str(args.era5),
        "results": public_results,
        "summary": {
            "mean_test_f1": float(result["test_f1"].mean()),
            "std_test_f1": float(result["test_f1"].std(ddof=1)),
            "mean_test_recall": float(result["test_recall"].mean()),
            "mean_test_auprc": float(result["test_auprc"].mean()),
            "mean_test_false_alarm_rate": float(result["test_false_alarm_rate"].mean()),
            "mean_recovery_mae_before": float(result["recovery_mae_before"].mean()),
            "mean_recovery_mae_after": float(result["recovery_mae_after"].mean()),
            "mean_event_recall": float(result["event_recall"].mean()),
        },
    }
    (out_dir / "stage22_leave_one_station_refit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段22严格留一站点重新拟合",
        "",
        "每个留出站点都重新拟合检测器、ERA5偏差、站点尺度、融合比例、阈值和恢复权重；留出站点不进入任何拟合或验证选择。测试时仅使用同一时间戳的跨站点观测计算空间证据，不使用测试标签。",
        "",
        "| 留出站点 | test F1 | Recall | Precision | AUPRC | 正常误伤率 | 恢复MAE前→后 | 事件召回 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result.iterrows():
        lines.append(f"| {row['held_out_station']} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | {row['test_precision']:.4f} | {row['test_auprc']:.4f} | {row['test_false_alarm_rate']:.4f} | {row['recovery_mae_before']:.4f}→{row['recovery_mae_after']:.4f} | {row['event_recall']:.4f} |")
    s = payload["summary"]
    lines += [
        "",
        f"平均±标准差 test F1：`{s['mean_test_f1']:.4f} ± {s['std_test_f1']:.4f}`",
        f"平均 Recall/AUPRC/误伤率：`{s['mean_test_recall']:.4f}` / `{s['mean_test_auprc']:.4f}` / `{s['mean_test_false_alarm_rate']:.4f}`",
        f"平均恢复 MAE：`{s['mean_recovery_mae_before']:.4f} → {s['mean_recovery_mae_after']:.4f}`；平均事件召回：`{s['mean_event_recall']:.4f}`",
        "",
        "该实验是严格跨站点重新拟合审计，结果不得表述为对所有站点或所有攻击类型的普适性证明；应与阶段21阈值迁移结果并列报告。",
        "",
        "## 站点级风险提示",
    ]
    for _, row in result.sort_values("test_false_alarm_rate", ascending=False).head(3).iterrows():
        lines.append(f"- `{row['held_out_station']}`：误伤率 `{row['test_false_alarm_rate']:.4f}`，F1 `{row['test_f1']:.4f}`，恢复 MAE `{row['recovery_mae_before']:.4f}→{row['recovery_mae_after']:.4f}`。")
    lines += [
        "",
        "按攻击类型的完整结果见 `stage22_leave_one_station_refit_by_type.csv`；逐行风险和恢复结果见 `stage22_leave_one_station_refit_detail.parquet`。",
    ]
    report = report_dir / "stage22_leave_one_station_refit.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report), "summary": payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
