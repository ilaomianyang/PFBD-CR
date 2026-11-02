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

from detectors import (  # noqa: E402
    CUSUMDetector,
    HampelMADDetector,
    KalmanInnovationDetector,
    PhysicalRulesDetector,
    SpatialConsistencyDetector,
)
from secure_recovery import add_recovery_candidates, choose_weights, combine_candidates  # noqa: E402
from trust_fusion import TrustFusion  # noqa: E402

DEFAULT_LOCAL = ROOT / "tables" / "stage05_controlled_attack_extensions.parquet"
DEFAULT_NOAA = ROOT / "tables" / "stage06_noaa_isd_normalized.parquet"
DEFAULT_ERA5 = ROOT / "tables" / "stage06_era5_station_reference.parquet"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    out = {
        "rows": int(len(y_true)),
        "attack_rows": int(y_true.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if score is not None:
        score = np.asarray(score, dtype=float)
        finite = np.isfinite(score)
        if finite.any() and y_true[finite].sum() and (~y_true[finite]).sum():
            out["roc_auc"] = float(roc_auc_score(y_true[finite], score[finite]))
            out["average_precision"] = float(average_precision_score(y_true[finite], score[finite]))
        else:
            out["roc_auc"] = float("nan")
            out["average_precision"] = float("nan")
    return out


def _best_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import f1_score

    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    if not finite.any():
        return 1.0, 0.0
    s = score[finite]
    yy = y[finite]
    candidates = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 199)))
    if len(candidates) == 0:
        return 1.0, 0.0
    values = [(float(f1_score(yy, s >= t, zero_division=0)), float(t)) for t in candidates]
    best = max(values, key=lambda x: (x[0], -x[1]))
    return best[1], best[0]


def _prepare_local(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "station_key", "split"]).sort_values(["split", "station_key", "timestamp_utc", "sample_id"], kind="stable").reset_index(drop=True)
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


def _prepare_era5(local: pd.DataFrame, era5_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    era = pd.read_parquet(era5_path).copy()
    era["timestamp_utc"] = pd.to_datetime(era["timestamp_utc"], utc=True, errors="coerce")
    era["hour_utc"] = era["timestamp_utc"].dt.floor("h")
    era = era.dropna(subset=["station_key", "hour_utc"]).drop_duplicates(["station_key", "hour_utc"])
    ref = local[["station_key", "timestamp_utc", "split", "original_value"]].copy()
    ref["hour_utc"] = ref["timestamp_utc"].dt.floor("h")
    merged = ref.merge(era, on=["station_key", "hour_utc"], how="left", suffixes=("", "_era"))
    valid_clean = merged[merged["split"].eq("valid") & merged["original_value"].notna() & merged["era5_t2m_c"].notna()].copy()
    bias = valid_clean.assign(_bias=valid_clean["original_value"] - valid_clean["era5_t2m_c"]).groupby("station_key", as_index=False)["_bias"].median().rename(columns={"_bias": "era5_median_bias_c"})
    aligned = local[["station_key", "timestamp_utc"]].copy()
    aligned["_row_id"] = np.arange(len(aligned), dtype=int)
    aligned["hour_utc"] = aligned["timestamp_utc"].dt.floor("h")
    aligned = aligned.merge(era, on=["station_key", "hour_utc"], how="left", sort=False)
    aligned = aligned.merge(bias, on="station_key", how="left")
    aligned["era5_corrected_t2m_c"] = aligned["era5_t2m_c"] + aligned["era5_median_bias_c"].fillna(0.0)
    local_values = pd.to_numeric(local["effective_observed_value"], errors="coerce")
    aligned["era5_background_score"] = (pd.to_numeric(aligned["era5_corrected_t2m_c"], errors="coerce") - local_values.reindex(aligned["_row_id"]).to_numpy()).abs()
    ordered = aligned.sort_values(["station_key", "hour_utc", "_row_id"], kind="stable")
    extreme = ordered.groupby("station_key", sort=False)["era5_t2m_c"].diff().abs().ge(2.0).fillna(False)
    aligned["era5_extreme_background"] = extreme.reindex(aligned.index).fillna(False).to_numpy(dtype=bool)
    return aligned.sort_values("_row_id", kind="stable").drop(columns=["_row_id"]), bias


def _score_splits(detector, frame: pd.DataFrame) -> np.ndarray:
    scores = np.full(len(frame), np.nan, dtype=float)
    for split in ["train", "valid", "test"]:
        mask = frame["split"].eq(split).to_numpy()
        if mask.any():
            scores[mask] = detector.score(frame.loc[mask])
    return scores


def _event_metrics(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float | int]:
    event_rows = frame[frame["combined_is_attack"] & frame["combined_event_id"].ne("")].copy()
    if event_rows.empty:
        return {"events": 0, "detected_events": 0, "event_recall": float("nan")}
    event_rows["_pred"] = np.asarray(pred, dtype=bool)[event_rows.index]
    grouped = event_rows.groupby("combined_event_id", sort=True)["_pred"].any()
    return {"events": int(len(grouped)), "detected_events": int(grouped.sum()), "event_recall": float(grouped.mean())}


def _recovery_metrics(frame: pd.DataFrame, recovered: np.ndarray, observed_col: str = "effective_observed_value") -> dict[str, float | int]:
    true = pd.to_numeric(frame["original_value"], errors="coerce").to_numpy(dtype=float)
    observed = pd.to_numeric(frame[observed_col], errors="coerce").to_numpy(dtype=float)
    recovered = np.asarray(recovered, dtype=float)
    attacked = frame["combined_is_attack"].to_numpy(dtype=bool)
    finite = np.isfinite(true) & attacked
    if not finite.any():
        return {"attack_rows": 0, "mae_before": float("nan"), "rmse_before": float("nan"), "mae_after": float("nan"), "rmse_after": float("nan")}
    before = np.where(np.isfinite(observed), observed, np.nan)
    after = np.where(np.isfinite(recovered), recovered, observed)
    b = finite & np.isfinite(before)
    a = finite & np.isfinite(after)
    return {
        "attack_rows": int(finite.sum()),
        "mae_before": float(np.nanmean(np.abs(true[b] - before[b]))) if b.any() else float("nan"),
        "rmse_before": float(np.sqrt(np.nanmean((true[b] - before[b]) ** 2))) if b.any() else float("nan"),
        "mae_after": float(np.nanmean(np.abs(true[a] - after[a]))) if a.any() else float("nan"),
        "rmse_after": float(np.sqrt(np.nanmean((true[a] - after[a]) ** 2))) if a.any() else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段9：TrustMeteo-QC多证据融合与可信恢复流水线")
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--noaa", type=Path, default=DEFAULT_NOAA)
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    args = parser.parse_args()

    local = _prepare_local(args.local)
    era_aligned, era_bias = _prepare_era5(local, args.era5)
    local["ERA5 Background"] = era_aligned["era5_background_score"].to_numpy()
    local["era5_extreme_background"] = era_aligned["era5_extreme_background"].to_numpy(dtype=bool)
    local["era5_corrected_t2m_c"] = era_aligned["era5_corrected_t2m_c"].to_numpy()

    detector_frame = local.copy()
    detector_frame["observed_value"] = detector_frame["effective_observed_value"]
    detectors = [PhysicalRulesDetector(), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    train = local["split"].eq("train")
    normal_train = train & ~local["combined_is_attack"]
    for detector in detectors:
        detector.fit(detector_frame.loc[train], normal_train.loc[train])
        local[detector.name] = _score_splits(detector, detector_frame)

    fusion = TrustFusion(include_era5=True)
    valid = local["split"].eq("valid")
    fusion.fit(local.loc[train], local.loc[valid], local.loc[valid, "combined_is_attack"])
    local["fused_risk"] = fusion.transform(local, extreme_background=local["era5_extreme_background"])
    local["trust_score"] = fusion.trust_score(local["fused_risk"].to_numpy())
    local["predicted_attack"] = fusion.predict(local["fused_risk"].to_numpy())
    local["risk_level"] = pd.cut(local["fused_risk"], bins=[-np.inf, 0.5, 1.5, 3.0, np.inf], labels=["低", "中", "高", "严重"], right=False).astype(str)
    local["suspicious_station"] = local["predicted_attack"]

    recovery_input = local.copy()
    recovery_input["observed_value"] = recovery_input["effective_observed_value"]
    recovery_input["is_anomaly"] = recovery_input["combined_is_attack"]
    valid_recovery = recovery_input.loc[valid].copy()
    valid_candidates = add_recovery_candidates(valid_recovery, local.loc[valid, "predicted_attack"], valid_recovery["era5_corrected_t2m_c"])
    recovery_weights = choose_weights(valid_candidates)
    test = local[local["split"].eq("test")].copy()
    test_input = recovery_input.loc[test.index].copy()
    test_candidates = add_recovery_candidates(test_input, test["predicted_attack"], test["era5_corrected_t2m_c"])
    test_recovered = combine_candidates(test_candidates, recovery_weights)
    test["recovered_value"] = test_recovered

    local_valid = local.loc[valid]
    local_test = local[local["split"].eq("test")]
    valid_metrics = _metrics(local_valid["combined_is_attack"], local_valid["predicted_attack"], local_valid["fused_risk"])
    test_metrics = _metrics(local_test["combined_is_attack"], local_test["predicted_attack"], local_test["fused_risk"])
    by_type = []
    for split_name, part in [("valid", local_valid), ("test", local_test)]:
        attack_types = sorted(part.loc[part["combined_is_attack"], "combined_attack_type"].dropna().unique().tolist())
        for attack_type in attack_types:
            y_type = part["combined_is_attack"].to_numpy(dtype=bool) & part["combined_attack_type"].eq(attack_type).to_numpy(dtype=bool)
            by_type.append({"split": split_name, "attack_type": attack_type, **_metrics(y_type, part["predicted_attack"].to_numpy(), part["fused_risk"].to_numpy())})
    rec_metrics = _recovery_metrics(test, test["recovered_value"].to_numpy())
    normal_test = local_test[~local_test["combined_is_attack"]]
    false_alarm_rate = float(normal_test["predicted_attack"].mean()) if len(normal_test) else float("nan")
    event = _event_metrics(local_test, local_test["predicted_attack"].to_numpy())

    noaa = pd.read_parquet(args.noaa).copy()
    noaa = noaa.rename(columns={"station_id": "station_key", "temperature_c": "observed_value"})
    noaa["timestamp_utc"] = pd.to_datetime(noaa["timestamp_utc"], utc=True, errors="coerce")
    noaa = noaa.dropna(subset=["timestamp_utc", "observed_value", "station_key"]).copy()
    noaa["split"] = "external"
    noaa["missing_observation"] = False
    noaa_scores = pd.DataFrame(index=noaa.index)
    for detector in detectors:
        noaa_scores[detector.name] = detector.score(noaa)
    noaa_scores["ERA5 Background"] = 0.0
    noaa_scores["missing_observation"] = False
    noaa_fused = fusion.transform(noaa_scores, extreme_background=pd.Series(False, index=noaa.index))
    noaa_alarm_rate = float(np.mean(noaa_fused >= fusion.threshold)) if len(noaa_fused) else float("nan")
    noaa_calibration = noaa["timestamp_utc"] < pd.Timestamp("2022-07-01", tz="UTC")
    noaa_holdout = ~noaa_calibration
    if noaa_calibration.any():
        external_threshold = float(np.nanquantile(noaa_fused[noaa_calibration.to_numpy()], 0.995))
        external_calibration_alarm_rate = float(np.mean(noaa_fused[noaa_calibration.to_numpy()] >= external_threshold))
        external_holdout_alarm_rate = float(np.mean(noaa_fused[noaa_holdout.to_numpy()] >= external_threshold)) if noaa_holdout.any() else float("nan")
    else:
        external_threshold = float("nan")
        external_calibration_alarm_rate = float("nan")
        external_holdout_alarm_rate = float("nan")

    out_dir = ROOT / "tables"
    report_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    local.to_parquet(out_dir / "stage09_local_trust_scores.parquet", index=False)
    test.to_parquet(out_dir / "stage09_test_recovery.parquet", index=False)
    pd.DataFrame(by_type).to_csv(out_dir / "stage09_attack_type_metrics.csv", index=False, encoding="utf-8-sig")
    era_bias.to_csv(out_dir / "stage09_era5_bias.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "train正常样本拟合；valid选择融合权重、阈值和恢复权重；test仅评价；NOAA无攻击外部报警率",
        "source_local": str(args.local),
        "source_noaa": str(args.noaa),
        "source_era5": str(args.era5),
        "fusion_components": fusion.components,
        "fusion_weights": fusion.weights,
        "fusion_threshold": float(fusion.threshold),
        "recovery_weights": recovery_weights,
        "local_valid": valid_metrics,
        "local_test": test_metrics,
        "test_false_alarm_rate_on_clean_rows": false_alarm_rate,
        "test_event_metrics": event,
        "test_recovery": rec_metrics,
        "noaa_external_alarm_rate": noaa_alarm_rate,
        "noaa_external_calibrated_threshold": external_threshold,
        "noaa_external_calibration_alarm_rate": external_calibration_alarm_rate,
        "noaa_external_holdout_alarm_rate": external_holdout_alarm_rate,
        "era5_bias_stations": int(len(era_bias)),
        "era5_extreme_rows_local": int(local["era5_extreme_background"].sum()),
        "controlled_attack_note": "阶段5扩展攻击为受控实验标签，不代表真实网络入侵记录。",
    }
    (out_dir / "stage09_trust_pipeline.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")

    lines = [
        "# 阶段9多证据融合与可信恢复",
        "",
        f"- 状态：**{payload['status']}**",
        "- 协议：train 正常样本拟合；valid 选择融合权重、阈值和恢复权重；test 只评价。",
        "- 攻击标签：基础注入与阶段5扩展攻击合并；扩展攻击明确属于受控实验。",
        "",
        "## 融合设置",
        "",
        f"- 组件：{', '.join(fusion.components)}",
        f"- 权重：`{json.dumps(fusion.weights, ensure_ascii=False)}`",
        f"- valid阈值：`{fusion.threshold:.6f}`",
        f"- 恢复权重：`{json.dumps(recovery_weights, ensure_ascii=False)}`",
        "",
        "## 指标",
        "",
        f"- valid：F1=`{valid_metrics['f1']:.4f}`，Precision=`{valid_metrics['precision']:.4f}`，Recall=`{valid_metrics['recall']:.4f}`，AUPRC=`{valid_metrics['average_precision']:.4f}`",
        f"- test：F1=`{test_metrics['f1']:.4f}`，Precision=`{test_metrics['precision']:.4f}`，Recall=`{test_metrics['recall']:.4f}`，AUPRC=`{test_metrics['average_precision']:.4f}`",
        f"- test正常样本误伤率：`{false_alarm_rate:.6f}`",
        f"- test事件级召回：`{event['event_recall']:.4f}`（{event['detected_events']}/{event['events']}）",
        f"- test恢复前 MAE/RMSE：`{rec_metrics['mae_before']:.4f}` / `{rec_metrics['rmse_before']:.4f}`",
        f"- test恢复后 MAE/RMSE：`{rec_metrics['mae_after']:.4f}` / `{rec_metrics['rmse_after']:.4f}`",
        f"- NOAA无攻击外部报警率：`{noaa_alarm_rate:.6f}`（不是攻击召回率）",
        f"- NOAA前期0.995分位外部校准阈值：`{external_threshold:.6f}`；校准期报警率：`{external_calibration_alarm_rate:.6f}`；后期留出报警率：`{external_holdout_alarm_rate:.6f}`",
        f"- ERA5极端背景标记行数：`{int(local['era5_extreme_background'].sum())}`；站点偏差校准站点数：`{len(era_bias)}`",
        "",
        "## 输出文件",
        "",
        "- `stage09_local_trust_scores.parquet`：本地全量风险、标签、信任分数和站点标记。",
        "- `stage09_test_recovery.parquet`：测试期恢复候选与恢复结果。",
        "- `stage09_attack_type_metrics.csv`：按攻击类型和 split 的指标。",
        "- `stage09_trust_pipeline.json`：参数、指标和审计信息。",
        "",
        "阶段9流程已完成；是否达到论文性能标准仍需结合多种子、消融实验、统计置信区间和人工审计判断。",
    ]
    report = report_dir / "stage09_trust_pipeline.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report), "test": payload["local_test"], "noaa_alarm_rate": noaa_alarm_rate}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
