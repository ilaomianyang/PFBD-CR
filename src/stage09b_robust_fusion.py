from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors import CUSUMDetector, HampelMADDetector, KalmanInnovationDetector, PhysicalRulesDetector, SpatialConsistencyDetector  # noqa: E402
from secure_recovery import add_recovery_candidates, choose_weights, combine_candidates  # noqa: E402
from stage09_trust_pipeline import _event_metrics, _metrics, _prepare_era5, _prepare_local, _recovery_metrics, _score_splits  # noqa: E402
from trust_fusion import TrustFusion, robust_location_scale  # noqa: E402

DEFAULT_LOCAL = ROOT / "tables" / "stage05_controlled_attack_extensions.parquet"
DEFAULT_NOAA = ROOT / "tables" / "stage06_noaa_isd_normalized.parquet"
DEFAULT_ERA5 = ROOT / "tables" / "stage06_era5_station_reference.parquet"
STRUCTURAL_BASE = ["stuck_run", "drift_delta", "lag_residual", "spatial_residual"]
STRUCTURAL_ENHANCED = STRUCTURAL_BASE + ["short_slope", "forecast_residual", "trend_residual"]


def structural_features(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or STRUCTURAL_BASE
    out = pd.DataFrame(index=frame.index)
    out[columns] = 0.0
    work = frame[["station_key", "timestamp_utc", "observed_value"]].copy()
    work["_value"] = pd.to_numeric(work["observed_value"], errors="coerce")
    work["_timestamp"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
    work = work.sort_values(["station_key", "_timestamp"], kind="stable")
    for _, group in work.groupby("station_key", sort=False):
        v = group["_value"]
        valid = v.notna()
        run = []
        current = 0
        previous = np.nan
        for value in v.to_numpy(dtype=float):
            if np.isfinite(value) and np.isfinite(previous) and abs(value - previous) <= 1e-8:
                current += 1
            elif np.isfinite(value):
                current = 1
            else:
                current = 0
            run.append(max(current - 3, 0.0))
            previous = value
        part = pd.DataFrame(index=group.index)
        part["stuck_run"] = run
        part["drift_delta"] = (v - v.shift(6)).abs() / 6.0
        part["lag_residual"] = (v - v.shift(144)).abs()
        part["spatial_residual"] = 0.0
        part["short_slope"] = (v - v.shift(12)).abs() / 12.0
        one_step_pred = v.shift(1) + (v.shift(1) - v.shift(2))
        part["forecast_residual"] = (v - one_step_pred).abs()
        part["trend_residual"] = (v - v.rolling(12, min_periods=4).median()).abs()
        out.loc[group.index, columns] = part.reindex(columns=columns, fill_value=0.0).to_numpy()
    if "timestamp_utc" in frame:
        vals = pd.to_numeric(frame["observed_value"], errors="coerce")
        med = frame.assign(_value=vals).groupby("timestamp_utc", sort=False)["_value"].transform("median")
        support = frame.assign(_value=vals).groupby("timestamp_utc", sort=False)["_value"].transform("count")
        out["spatial_residual"] = (vals - med).abs().where(support >= 2, 0.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def structural_risk_from_norm(norm: pd.DataFrame, enhanced: bool) -> np.ndarray:
    risk = (0.30 * norm["stuck_run"] + 0.25 * norm["drift_delta"] + 0.15 * norm["lag_residual"] + 0.15 * norm["spatial_residual"])
    if enhanced:
        risk = risk + 0.05 * norm["short_slope"] + 0.05 * norm["forecast_residual"] + 0.05 * norm["trend_residual"]
    return risk.to_numpy(dtype=float)


class AdaptiveScaler:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.global_stats: dict[str, tuple[float, float]] = {}
        self.station_stats: dict[str, dict[str, tuple[float, float]]] = {}

    def fit(self, frame: pd.DataFrame, normal_mask: pd.Series) -> "AdaptiveScaler":
        ref = frame.loc[normal_mask]
        for col in self.columns:
            self.global_stats[col] = robust_location_scale(ref[col], floor=0.25)
        for station, group in ref.groupby("station_key", sort=False):
            self.station_stats[str(station)] = {col: robust_location_scale(group[col], floor=0.25) for col in self.columns}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for col in self.columns:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            result = np.zeros(len(frame), dtype=float)
            stations = frame["station_key"].astype(str).to_numpy()
            for i, station in enumerate(stations):
                med, scale = self.station_stats.get(station, {}).get(col, self.global_stats[col])
                result[i] = max(values[i] - med, 0.0) / scale if np.isfinite(values[i]) else 0.0
            out[col] = np.nan_to_num(result, nan=0.0, posinf=1e6)
        return out


def smooth_risk(frame: pd.DataFrame, risk: np.ndarray, window: int) -> np.ndarray:
    out = pd.Series(np.asarray(risk, dtype=float), index=frame.index)
    if window <= 1:
        return out.to_numpy()
    temp = frame[["station_key", "timestamp_utc"]].copy()
    temp["_risk"] = out
    temp = temp.sort_values(["station_key", "timestamp_utc"], kind="stable")
    smoothed = temp.groupby("station_key", sort=False)["_risk"].rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
    return smoothed.reindex(frame.index).fillna(out).to_numpy(dtype=float)


def best_robust_setting(valid: pd.DataFrame, base: np.ndarray, structural: np.ndarray) -> dict[str, float | int]:
    y = valid["combined_is_attack"].to_numpy(dtype=bool)
    best: tuple[float, float, float, int] | None = None
    for alpha in [0.50, 0.65, 0.80, 0.90]:
        raw = alpha * base + (1.0 - alpha) * structural
        for window in [1, 3, 6]:
            score = smooth_risk(valid, raw, window)
            from stage09_trust_pipeline import _best_threshold
            threshold, f1 = _best_threshold(y, score)
            far = float(np.mean(score[~y] >= threshold)) if (~y).any() else 1.0
            candidate = (f1, -far, -threshold, window)
            if best is None or candidate > best:
                best = candidate
                selected = {"alpha_base": alpha, "alpha_structural": 1.0 - alpha, "window": window, "threshold": threshold, "valid_f1": f1, "valid_false_alarm_rate": far}
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段9B：站点自适应与结构化攻击鲁棒融合")
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--noaa", type=Path, default=DEFAULT_NOAA)
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    parser.add_argument("--tag", type=str, default="", help="结果文件标签，例如 seed3407")
    parser.add_argument("--no-structural", action="store_true", help="消融结构化特征")
    parser.add_argument("--no-era5", action="store_true", help="消融ERA5背景组件")
    parser.add_argument("--enhanced", action="store_true", help="启用增强结构化特征")
    args = parser.parse_args()

    local = _prepare_local(args.local)
    era_aligned, era_bias = _prepare_era5(local, args.era5)
    local["ERA5 Background"] = era_aligned["era5_background_score"].to_numpy()
    if args.no_era5:
        local["ERA5 Background"] = 0.0
    local["era5_extreme_background"] = era_aligned["era5_extreme_background"].to_numpy(dtype=bool)
    local["era5_corrected_t2m_c"] = era_aligned["era5_corrected_t2m_c"].to_numpy()
    detector_frame = local.copy()
    detector_frame["observed_value"] = detector_frame["effective_observed_value"]
    feature_parts = []
    structural_columns = STRUCTURAL_ENHANCED if args.enhanced else STRUCTURAL_BASE
    for split in ["train", "valid", "test"]:
        idx = local.index[local["split"].eq(split)]
        part = detector_frame.loc[idx].copy()
        feature_parts.append(structural_features(part, structural_columns).set_axis(idx))
    features = pd.concat(feature_parts).reindex(local.index).fillna(0.0)
    features["station_key"] = local["station_key"].astype(str)
    train = local["split"].eq("train")
    normal_train = train & ~local["combined_is_attack"]
    detectors = [PhysicalRulesDetector(), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    for detector in detectors:
        detector.fit(detector_frame.loc[train], normal_train.loc[train])
        local[detector.name] = _score_splits(detector, detector_frame)
    fusion = TrustFusion(include_era5=not args.no_era5).fit(local.loc[train], local.loc[local["split"].eq("valid")], local.loc[local["split"].eq("valid"), "combined_is_attack"])
    base_risk = fusion.transform(local, extreme_background=local["era5_extreme_background"])
    scaler = AdaptiveScaler(structural_columns).fit(features, normal_train)
    structural_norm = scaler.transform(features)
    structural_risk = structural_risk_from_norm(structural_norm, args.enhanced)
    if args.no_structural:
        structural_risk[:] = 0.0
    valid = local["split"].eq("valid")
    if args.no_structural:
        from stage09_trust_pipeline import _best_threshold
        y_valid = local.loc[valid, "combined_is_attack"].to_numpy(dtype=bool)
        base_valid = base_risk[valid.to_numpy()]
        threshold, valid_f1 = _best_threshold(y_valid, base_valid)
        setting = {"alpha_base": 1.0, "alpha_structural": 0.0, "window": 1, "threshold": threshold, "valid_f1": valid_f1, "valid_false_alarm_rate": float(np.mean(base_valid[~y_valid] >= threshold))}
    else:
        setting = best_robust_setting(local.loc[valid], base_risk[valid.to_numpy()], structural_risk[valid.to_numpy()])
    raw_risk = setting["alpha_base"] * base_risk + setting["alpha_structural"] * structural_risk
    robust_risk = np.full(len(local), np.nan, dtype=float)
    pred = np.zeros(len(local), dtype=bool)
    for split in ["train", "valid", "test"]:
        mask = local["split"].eq(split)
        robust_risk[mask.to_numpy()] = smooth_risk(local.loc[mask], raw_risk[mask.to_numpy()], int(setting["window"]))
        pred[mask.to_numpy()] = robust_risk[mask.to_numpy()] >= float(setting["threshold"])
    local["base_fused_risk"] = base_risk
    local["structural_risk"] = structural_risk
    local["robust_fused_risk"] = robust_risk
    local["trust_score"] = 1.0 / (1.0 + np.maximum(robust_risk, 0.0))
    local["predicted_attack"] = pred
    local["risk_level"] = pd.cut(local["robust_fused_risk"], bins=[-np.inf, 0.5, 1.5, 3.0, np.inf], labels=["低", "中", "高", "严重"], right=False).astype(str)
    local["suspicious_station"] = pred

    recovery_input = local.copy()
    recovery_input["observed_value"] = recovery_input["effective_observed_value"]
    recovery_input["is_anomaly"] = recovery_input["combined_is_attack"]
    valid_recovery = recovery_input.loc[valid].copy()
    valid_candidates = add_recovery_candidates(valid_recovery, local.loc[valid, "predicted_attack"], valid_recovery["era5_corrected_t2m_c"])
    recovery_weights = choose_weights(valid_candidates)
    test = local[local["split"].eq("test")].copy()
    test_input = recovery_input.loc[test.index].copy()
    test_candidates = add_recovery_candidates(test_input, test["predicted_attack"], test["era5_corrected_t2m_c"])
    test["recovered_value"] = combine_candidates(test_candidates, recovery_weights)

    valid_part = local.loc[valid]
    test_part = local[local["split"].eq("test")]
    valid_metrics = _metrics(valid_part["combined_is_attack"], valid_part["predicted_attack"], valid_part["robust_fused_risk"])
    test_metrics = _metrics(test_part["combined_is_attack"], test_part["predicted_attack"], test_part["robust_fused_risk"])
    by_type = []
    for split_name, part in [("valid", valid_part), ("test", test_part)]:
        for attack_type in sorted(part.loc[part["combined_is_attack"], "combined_attack_type"].unique()):
            y_type = part["combined_is_attack"].to_numpy(dtype=bool) & part["combined_attack_type"].eq(attack_type).to_numpy(dtype=bool)
            by_type.append({"split": split_name, "attack_type": attack_type, **_metrics(y_type, part["predicted_attack"], part["robust_fused_risk"])})
    clean_test = test_part[~test_part["combined_is_attack"]]
    false_alarm_rate = float(clean_test["predicted_attack"].mean()) if len(clean_test) else float("nan")
    recovery_metrics = _recovery_metrics(test, test["recovered_value"].to_numpy())
    event_metrics = _event_metrics(test_part, test_part["predicted_attack"].to_numpy())

    noaa = pd.read_parquet(args.noaa).rename(columns={"station_id": "station_key", "temperature_c": "observed_value"}).copy()
    noaa["timestamp_utc"] = pd.to_datetime(noaa["timestamp_utc"], utc=True, errors="coerce")
    noaa = noaa.dropna(subset=["timestamp_utc", "station_key", "observed_value"]).copy()
    noaa["split"] = "external"
    noaa_detector = noaa.copy()
    noaa_struct = structural_features(noaa_detector, structural_columns)
    noaa_struct["station_key"] = noaa["station_key"].astype(str)
    noaa_scores = pd.DataFrame(index=noaa.index)
    for detector in detectors:
        noaa_scores[detector.name] = detector.score(noaa_detector)
    noaa_scores["ERA5 Background"] = 0.0
    noaa_scores["missing_observation"] = False
    noaa_base = fusion.transform(noaa_scores, extreme_background=pd.Series(False, index=noaa.index))
    noaa_struct_norm = scaler.transform(noaa_struct)
    noaa_struct_risk = structural_risk_from_norm(noaa_struct_norm, args.enhanced)
    noaa_raw = setting["alpha_base"] * noaa_base + setting["alpha_structural"] * noaa_struct_risk
    noaa_robust = smooth_risk(noaa, noaa_raw, int(setting["window"]))
    noaa_rate = float(np.mean(noaa_robust >= float(setting["threshold"]))) if len(noaa_robust) else float("nan")
    cal = noaa["timestamp_utc"] < pd.Timestamp("2022-07-01", tz="UTC")
    ext_threshold = float(np.nanquantile(noaa_robust[cal.to_numpy()], 0.995)) if cal.any() else float("nan")
    hold = ~cal
    ext_hold_rate = float(np.mean(noaa_robust[hold.to_numpy()] >= ext_threshold)) if hold.any() else float("nan")

    out_dir, report_dir = ROOT / "tables", ROOT / "reports"
    out_dir.mkdir(exist_ok=True, parents=True); report_dir.mkdir(exist_ok=True, parents=True)
    suffix = f"_{args.tag}" if args.tag else ""
    local.to_parquet(out_dir / f"stage09b_local_robust_scores{suffix}.parquet", index=False)
    test.to_parquet(out_dir / f"stage09b_test_recovery{suffix}.parquet", index=False)
    pd.DataFrame(by_type).to_csv(out_dir / f"stage09b_attack_type_metrics{suffix}.csv", index=False, encoding="utf-8-sig")
    era_bias.to_csv(out_dir / f"stage09b_era5_bias{suffix}.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "train正常样本拟合；valid选择结构化融合设置和恢复权重；test只评价",
        "setting": setting,
        "base_fusion_weights": fusion.weights,
        "base_fusion_threshold": float(fusion.threshold),
        "recovery_weights": recovery_weights,
        "local_valid": valid_metrics,
        "local_test": test_metrics,
        "test_false_alarm_rate_on_clean_rows": false_alarm_rate,
        "test_event_metrics": event_metrics,
        "test_recovery": recovery_metrics,
        "noaa_external_alarm_rate_local_threshold": noaa_rate,
        "noaa_external_calibrated_threshold": ext_threshold,
        "noaa_external_holdout_alarm_rate": ext_hold_rate,
        "controlled_attack_note": "阶段5扩展攻击为受控实验标签，不代表真实网络入侵记录。",
    }
    (out_dir / f"stage09b_robust_fusion{suffix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段9B鲁棒融合与结构化攻击增强", "", f"- 状态：**{payload['status']}**", "- 训练集只使用正常样本拟合站点自适应尺度；验证集选择融合比例、连续窗口和阈值；测试集只评价。", "",
        "## 设置", "", f"- 基础/结构化风险比例：`{setting['alpha_base']:.2f}/{setting['alpha_structural']:.2f}`", f"- 连续窗口：`{int(setting['window'])}` 条记录", f"- 阈值：`{setting['threshold']:.6f}`", f"- 恢复权重：`{json.dumps(recovery_weights, ensure_ascii=False)}`", "",
        "## 结果", "", f"- valid F1/Recall/AUPRC：`{valid_metrics['f1']:.4f}` / `{valid_metrics['recall']:.4f}` / `{valid_metrics['average_precision']:.4f}`", f"- test F1/Recall/AUPRC：`{test_metrics['f1']:.4f}` / `{test_metrics['recall']:.4f}` / `{test_metrics['average_precision']:.4f}`", f"- test正常样本误伤率：`{false_alarm_rate:.6f}`", f"- test事件级召回：`{event_metrics['event_recall']:.4f}`", f"- 恢复前后 MAE：`{recovery_metrics['mae_before']:.4f}` → `{recovery_metrics['mae_after']:.4f}`", f"- NOAA本地阈值报警率：`{noaa_rate:.6f}`；外部校准后留出报警率：`{ext_hold_rate:.6f}`", "", "阶段9B结果用于与阶段9A基线比较；只有完成多种子和消融实验后，才能形成论文主结论。"
    ]
    report = report_dir / f"stage09b_robust_fusion{suffix}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report), "test": test_metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
