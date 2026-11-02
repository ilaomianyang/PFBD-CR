from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from secure_recovery import add_recovery_candidates  # noqa: E402
from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402
from stage44_recovery_confidence_gate import summarize  # noqa: E402

NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
QUANTILES = [0.0, 0.50, 0.70, 0.80, 0.90, 0.95]
WEIGHT_CANDIDATES = [
    {"temporal": 1.0, "spatial": 0.0, "multivariate": 0.0},
    {"temporal": 0.7, "spatial": 0.3, "multivariate": 0.0},
    {"temporal": 0.6, "spatial": 0.2, "multivariate": 0.2},
    {"temporal": 0.5, "spatial": 0.2, "multivariate": 0.3},
    {"temporal": 0.4, "spatial": 0.2, "multivariate": 0.4},
    {"temporal": 0.2, "spatial": 0.1, "multivariate": 0.7},
    {"temporal": 0.0, "spatial": 0.0, "multivariate": 1.0},
]


def design_matrix(frame: pd.DataFrame, stations: list[str], medians: dict[str, float]) -> np.ndarray:
    ts = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    hour = ts.dt.hour.to_numpy(dtype=float) + ts.dt.minute.to_numpy(dtype=float) / 60.0
    day = ts.dt.dayofyear.to_numpy(dtype=float)
    values = pd.DataFrame(index=frame.index)
    values["dewpoint_c"] = pd.to_numeric(frame["dewpoint_c"], errors="coerce").fillna(medians["dewpoint_c"])
    values["wind_speed_ms"] = pd.to_numeric(frame["wind_speed_ms"], errors="coerce").fillna(medians["wind_speed_ms"])
    values["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    values["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    values["day_sin"] = np.sin(2.0 * np.pi * day / 365.25)
    values["day_cos"] = np.cos(2.0 * np.pi * day / 365.25)
    station_series = frame["station_key"].astype(str)
    for station in stations:
        values[f"station_{station}"] = station_series.eq(station).astype(float)
    return values.to_numpy(dtype=float)


def fit_multivariate(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    train_normal = frame["split"].eq("train") & (~frame["combined_is_attack"].astype(bool))
    stations = sorted(frame.loc[train_normal, "station_key"].astype(str).unique().tolist())
    medians = {
        "dewpoint_c": float(pd.to_numeric(frame.loc[train_normal, "dewpoint_c"], errors="coerce").median()),
        "wind_speed_ms": float(pd.to_numeric(frame.loc[train_normal, "wind_speed_ms"], errors="coerce").median()),
    }
    x = design_matrix(frame, stations, medians)
    y = pd.to_numeric(frame["original_value"], errors="coerce").to_numpy(dtype=float)
    fit_mask = train_normal.to_numpy() & np.isfinite(y)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x[fit_mask], y[fit_mask])
    prediction = model.predict(x)
    clean_mae = {}
    for split in ["train", "valid", "test"]:
        mask = frame["split"].eq(split).to_numpy() & (~frame["combined_is_attack"].astype(bool).to_numpy()) & np.isfinite(y)
        clean_mae[split] = float(np.mean(np.abs(y[mask] - prediction[mask]))) if mask.any() else float("nan")
    info = {
        "model": "StandardScaler + Ridge(alpha=1.0)",
        "features": ["dewpoint_c", "wind_speed_ms", "hour_sin/cos", "day_of_year_sin/cos", "station_one_hot"],
        "fit_rows": int(fit_mask.sum()),
        "medians": medians,
        "clean_mae_by_split": clean_mae,
        "pressure_excluded_missing_rate": float(frame["pressure_hpa"].isna().mean()),
    }
    return prediction, info


def combine_extended(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    gate = frame["_attack_pred"].astype(bool).to_numpy()
    observed = pd.to_numeric(frame["observed_value"], errors="coerce").to_numpy(dtype=float)
    result = observed.copy()
    columns = [("temporal_candidate", "temporal"), ("spatial_candidate", "spatial"), ("multivariate_candidate", "multivariate")]
    for i in np.flatnonzero(gate):
        values, used = [], []
        for column, name in columns:
            value = frame.iloc[i][column]
            weight = float(weights.get(name, 0.0))
            if weight > 0 and pd.notna(value):
                values.append(float(value))
                used.append(weight)
        if used:
            result[i] = float(np.average(values, weights=used))
    return result


def choose_weights_extended(valid: pd.DataFrame) -> dict[str, float]:
    label = valid["is_anomaly"].astype(bool).to_numpy()
    true = pd.to_numeric(valid["original_value"], errors="coerce").to_numpy(dtype=float)
    best = (float("inf"), WEIGHT_CANDIDATES[0])
    for weights in WEIGHT_CANDIDATES:
        recovered = combine_extended(valid, weights)
        mask = label & np.isfinite(true) & np.isfinite(recovered)
        if mask.any():
            rmse = float(np.sqrt(np.mean((true[mask] - recovered[mask]) ** 2)))
            if rmse < best[0]:
                best = (rmse, weights)
    return best[1]


def run_seed(source: pd.DataFrame, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    frame = build_dataset(source, seed)
    scored, _ = score(frame)
    scored["observed_value"] = pd.to_numeric(scored["effective_observed_value"], errors="coerce")
    scored["is_anomaly"] = scored["combined_is_attack"].astype(bool)
    scored["era5_corrected_t2m_c"] = np.nan
    prediction, model_info = fit_multivariate(scored)
    scored["multivariate_candidate"] = prediction
    valid = scored["split"].eq("valid")
    test = scored["split"].eq("test")
    base_v = scored.loc[valid, "predicted_attack"].astype(bool).to_numpy()
    risk_v = scored.loc[valid, "robust_fused_risk"].to_numpy(dtype=float)
    alert_risk = risk_v[base_v]
    risk_t = scored.loc[test, "robust_fused_risk"].to_numpy(dtype=float)
    base_t = scored.loc[test, "predicted_attack"].astype(bool).to_numpy()
    rows = []
    for q in QUANTILES:
        threshold = float(np.quantile(alert_risk, q))
        gate_v = base_v & (risk_v >= threshold)
        gate_t = base_t & (risk_t >= threshold)
        vf = scored.loc[valid].copy()
        tf = scored.loc[test].copy()
        vc = add_recovery_candidates(vf, pd.Series(gate_v, index=vf.index), vf["era5_corrected_t2m_c"])
        vc["multivariate_candidate"] = vf["multivariate_candidate"].to_numpy(dtype=float)
        weights = choose_weights_extended(vc)
        vr = combine_extended(vc, weights)
        tc = add_recovery_candidates(tf, pd.Series(gate_t, index=tf.index), tf["era5_corrected_t2m_c"])
        tc["multivariate_candidate"] = tf["multivariate_candidate"].to_numpy(dtype=float)
        tr = combine_extended(tc, weights)
        rows.append({"seed": seed, "gate_quantile": q, "risk_threshold": threshold, "weights": weights, "valid": summarize(vf, vr, gate_v), "test": summarize(tf, tr, gate_t)})
    return rows, model_info


def main() -> int:
    source = pd.read_parquet(NOAA)
    rows, model_info = [], []
    for seed in SEEDS:
        current, info = run_seed(source, seed)
        rows.extend(current)
        model_info.append({"seed": seed, **info})
    selected = []
    for seed in SEEDS:
        candidates = [r for r in rows if r["seed"] == seed]
        selected.append(max(candidates, key=lambda r: (r["valid"]["net_absolute_error_gain"], -r["valid"]["gated_alert_rows"])))
    selected_df = pd.DataFrame([{"seed": r["seed"], "gate_quantile": r["gate_quantile"], "weights": json.dumps(r["weights"], ensure_ascii=False), **{f"test_{k}": v for k, v in r["test"].items()}} for r in selected])
    baseline = json.loads((TABLES / "stage47_noaa_recovery_gate.json").read_text(encoding="utf-8"))
    mean_gain = float(selected_df["test_net_absolute_error_gain"].mean())
    base_gain = float(baseline["summary"]["selected_mean_net_gain"])
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_ADDITIONAL_EXTERNAL_DATA" if mean_gain > 0 and mean_gain > base_gain else "NEGATIVE_RESULT_RETAINED", "protocol": "NOAA固定时间切分、5受控攻击种子；轻量多变量候选仅由train正常样本拟合；门控和恢复权重只由valid选择；test评价。", "seeds": SEEDS, "model_info": model_info, "all_candidates": rows, "selected_by_valid": selected, "comparison": {"multivariate_selected_mean_net_gain": mean_gain, "stage47_temporal_spatial_mean_net_gain": base_gain}}
    detail_rows = [{"seed": r["seed"], "gate_quantile": r["gate_quantile"], "risk_threshold": r["risk_threshold"], "weights": json.dumps(r["weights"], ensure_ascii=False), **{f"valid_{k}": v for k, v in r["valid"].items()}, **{f"test_{k}": v for k, v in r["test"].items()}} for r in rows]
    pd.DataFrame(detail_rows).to_csv(TABLES / "stage48_noaa_multivariate_recovery_detail.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage48_noaa_multivariate_recovery_selected.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage48_noaa_multivariate_recovery.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段48 NOAA轻量多变量恢复候选", "", "候选仅使用train正常样本拟合的露点、风速、周期时间和站点线性关系；气压因缺失率过高而排除。", "", "| seed | valid选择门控 | 恢复权重 | test攻击恢复后MAE | 攻击恶化率 | 正常改写行 | 总误差净收益 |", "|---:|---:|---|---:|---:|---:|---:|"]
    for r in selected:
        t = r["test"]
        lines.append(f"| {r['seed']} | {r['gate_quantile']:.2f} | `{json.dumps(r['weights'], ensure_ascii=False)}` | {t['detected_attack_mae_after']:.4f} | {t['detected_attack_worsening_rate']:.4f} | {int(t['normal_changed_rows'])} | {t['net_absolute_error_gain']:.2f} |")
    lines += ["", f"- 状态：**{payload['status']}**", f"- 多变量候选平均净收益：`{mean_gain:.2f}`；阶段47时间/空间候选：`{base_gain:.2f}`。", "- NOAA攻击仍是受控注入，不能写成真实攻击恢复结果。"]
    (REPORTS / "stage48_noaa_multivariate_recovery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "mean_net_gain": mean_gain, "baseline_mean_net_gain": base_gain}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
