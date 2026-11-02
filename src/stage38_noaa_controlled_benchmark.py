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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors import CUSUMDetector, HampelMADDetector, KalmanInnovationDetector, PhysicalRulesDetector, SpatialConsistencyDetector  # noqa: E402
from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, best_robust_setting, structural_features, structural_risk_from_norm, smooth_risk  # noqa: E402
from stage09_trust_pipeline import _metrics  # noqa: E402
from trust_fusion import TrustFusion, robust_location_scale  # noqa: E402

NOAA = ROOT / "tables" / "stage06_noaa_isd_normalized.parquet"
COMPLETE_TYPES = ["spike", "offset", "drift", "time_shift", "stuck", "gaussian_noise"]


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def choose_start(group: pd.DataFrame, type_index: int, length: int) -> int:
    if len(group) < length + 4:
        raise ValueError("NOAA split too short for controlled attack")
    # Keep events separated and away from split boundaries.
    fractions = [0.12, 0.26, 0.40, 0.54, 0.68, 0.82]
    pos = int(round((len(group) - length - 2) * fractions[type_index])) + 1
    return max(1, min(pos, len(group) - length - 1))


def build_dataset(source: pd.DataFrame, seed: int) -> pd.DataFrame:
    raw = source.dropna(subset=["timestamp_utc", "temperature_c"]).copy()
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp_utc"]).sort_values(["station_id", "timestamp_utc"], kind="stable").reset_index(drop=True)
    raw = raw.rename(columns={"station_id": "station_key", "temperature_c": "original_value"})
    raw["observed_value"] = raw["original_value"].astype(float)
    raw["effective_observed_value"] = raw["observed_value"]
    raw["combined_is_attack"] = False
    raw["combined_attack_type"] = "normal"
    raw["combined_event_id"] = raw.index.astype(str)
    raw["missing_observation"] = False
    raw["split"] = np.where(raw["timestamp_utc"] < pd.Timestamp("2022-07-01", tz="UTC"), "train", np.where(raw["timestamp_utc"] < pd.Timestamp("2022-10-01", tz="UTC"), "valid", "test"))
    for split in ["valid", "test"]:
        for station, station_all in raw[raw["split"].eq(split)].groupby("station_key", sort=True):
            idx = station_all.index.to_numpy()
            for type_index, attack_type in enumerate(COMPLETE_TYPES):
                length = 8 if attack_type != "drift" else 12
                start = choose_start(station_all, type_index, length)
                event_idx = idx[start:start + length]
                event_id = f"noaa_{split}_{station}_{attack_type}"
                clean = raw.loc[event_idx, "original_value"].to_numpy(dtype=float)
                rng = np.random.default_rng(stable_seed(seed, split, station, attack_type))
                if attack_type == "spike":
                    values = clean + 5.0
                elif attack_type == "offset":
                    values = clean + 2.5
                elif attack_type == "drift":
                    values = clean + np.linspace(0.2, 2.4, length)
                elif attack_type == "time_shift":
                    prior = station_all["original_value"].to_numpy(dtype=float)
                    values = prior[start - 1:start - 1 + length]
                elif attack_type == "stuck":
                    values = np.repeat(clean[0], length)
                else:
                    values = clean + rng.normal(0.0, 1.0, size=length)
                raw.loc[event_idx, "observed_value"] = values
                raw.loc[event_idx, "effective_observed_value"] = values
                raw.loc[event_idx, "combined_is_attack"] = True
                raw.loc[event_idx, "combined_attack_type"] = attack_type
                raw.loc[event_idx, "combined_event_id"] = event_id
    return raw


def score(source: pd.DataFrame, physical_station_adaptive: bool = False) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = source.copy()
    detector_frame = frame.copy()
    train = frame["split"].eq("train")
    valid = frame["split"].eq("valid")
    test = frame["split"].eq("test")
    normal_train = train & (~frame["combined_is_attack"])
    detectors = [PhysicalRulesDetector(station_adaptive=physical_station_adaptive), HampelMADDetector(), KalmanInnovationDetector(), CUSUMDetector(), SpatialConsistencyDetector()]
    for detector in detectors:
        detector.fit(detector_frame.loc[train], normal_train.loc[train])
        col = np.full(len(frame), np.nan, dtype=float)
        for split in ["train", "valid", "test"]:
            mask = frame["split"].eq(split)
            if mask.any():
                col[mask.to_numpy()] = detector.score(detector_frame.loc[mask])
        frame[detector.name] = col
    parts = []
    for split_name in ["train", "valid", "test"]:
        mask = frame["split"].eq(split_name)
        part = structural_features(detector_frame.loc[mask], STRUCTURAL_BASE).set_axis(frame.index[mask])
        parts.append(part)
    features = pd.concat(parts).reindex(frame.index).fillna(0.0)
    features["station_key"] = frame["station_key"].astype(str)
    scaler = AdaptiveScaler(STRUCTURAL_BASE).fit(features.loc[train], normal_train.loc[train])
    structural = structural_risk_from_norm(scaler.transform(features), enhanced=False)
    fusion = TrustFusion(include_era5=False).fit(frame.loc[train], frame.loc[valid], frame.loc[valid, "combined_is_attack"])
    base = fusion.transform(frame)
    setting = best_robust_setting(frame.loc[valid], base[valid.to_numpy()], structural[valid.to_numpy()])
    raw = setting["alpha_base"] * base + setting["alpha_structural"] * structural
    scores = np.full(len(frame), np.nan, dtype=float)
    pred = np.zeros(len(frame), dtype=bool)
    for split_name in ["train", "valid", "test"]:
        mask = frame["split"].eq(split_name)
        values = smooth_risk(frame.loc[mask], raw[mask.to_numpy()], int(setting["window"]))
        scores[mask.to_numpy()] = values
        pred[mask.to_numpy()] = values >= float(setting["threshold"])
    frame["robust_fused_risk"] = scores
    frame["predicted_attack"] = pred
    yv = frame.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    yt = frame.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    mv = _metrics(yv, scores[valid.to_numpy()] >= float(setting["threshold"]), scores[valid.to_numpy()])
    mt = _metrics(yt, scores[test.to_numpy()] >= float(setting["threshold"]), scores[test.to_numpy()])
    by_type = []
    for attack_type in COMPLETE_TYPES:
        mask = test & frame["combined_attack_type"].eq(attack_type)
        y = frame.loc[mask, "combined_is_attack"].astype(bool).to_numpy()
        p = pred[mask.to_numpy()]
        by_type.append({"attack_type": attack_type, "rows": int(mask.sum()), "recall": float(p.mean()) if len(p) else float("nan"), "detected_rows": int(p.sum())})
    result = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA 12站点真实序列按2022-01至06训练、07至09验证、10至12锁定测试；攻击仅为可审计受控注入，不是NOAA真实攻击标签。", "seed": int(SEED), "setting": setting, "valid": mv, "test": mt, "test_false_alarm_rate": float((scores[test.to_numpy()][~yt] >= float(setting["threshold"])).mean()), "attack_type_test": by_type, "stations": sorted(frame["station_key"].astype(str).unique().tolist()), "rows": int(len(frame))}
    return frame, result


SEED = 20260912


def main() -> int:
    source = pd.read_parquet(NOAA)
    frame = build_dataset(source, SEED)
    scored, result = score(frame)
    TABLES.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(TABLES / "stage38_noaa_controlled_scores.parquet", index=False)
    (TABLES / "stage38_noaa_controlled_benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(result["attack_type_test"]).to_csv(TABLES / "stage38_noaa_controlled_by_type.csv", index=False, encoding="utf-8-sig")
    lines = ["# 阶段38 NOAA独立数据源受控攻击基准", "", "NOAA为外部真实观测；攻击标签由本阶段透明注入，仅用于独立协议验证，不代表真实网络入侵。", "", f"- 行数：`{result['rows']}`；站点：`{len(result['stations'])}`；时间切分：训练2022-01至06、验证2022-07至09、测试2022-10至12", f"- test F1/Recall/Precision/AUPRC：`{result['test']['f1']:.4f}` / `{result['test']['recall']:.4f}` / `{result['test']['precision']:.4f}` / `{result['test']['average_precision']:.4f}`", f"- test正常误伤率：`{result['test_false_alarm_rate']:.4f}`", "", "| 攻击类型 | 测试行数 | 检出行数 | Recall |", "|---|---:|---:|---:|"]
    for row in result["attack_type_test"]:
        lines.append(f"| {row['attack_type']} | {row['rows']} | {row['detected_rows']} | {row['recall']:.4f} |")
    lines += ["", "该结果只证明跨数据源协议可运行；不能替代真实攻击日志或官方质量标记。"]
    (REPORTS / "stage38_noaa_controlled_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": result["rows"], "test_f1": result["test"]["f1"], "test_auprc": result["test"]["average_precision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
