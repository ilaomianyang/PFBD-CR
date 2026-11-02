from __future__ import annotations

import numpy as np
import pandas as pd


def add_recovery_candidates(df: pd.DataFrame, attack_pred: pd.Series, era5_corrected: pd.Series | None = None) -> pd.DataFrame:
    out = df.copy()
    out["_attack_pred"] = attack_pred.astype(bool).to_numpy()
    out["_masked_value"] = pd.to_numeric(out["observed_value"], errors="coerce").where(~out["_attack_pred"])
    out["temporal_candidate"] = np.nan
    out["spatial_candidate"] = np.nan
    ordered = out.sort_values(["station_key", "timestamp_utc"]).copy()
    for station, group in ordered.groupby("station_key", sort=False):
        s = group["_masked_value"].astype(float)
        interp = s.interpolate(limit_direction="both")
        out.loc[group.index, "temporal_candidate"] = interp.to_numpy()
    if "timestamp_utc" in out:
        med = out.groupby("timestamp_utc", sort=False)["_masked_value"].transform(lambda s: s.median() if s.notna().sum() else np.nan)
        out["spatial_candidate"] = med
    if era5_corrected is not None:
        out["era5_candidate"] = pd.to_numeric(era5_corrected, errors="coerce").to_numpy()
    else:
        out["era5_candidate"] = np.nan
    return out


def combine_candidates(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    names = [("temporal_candidate", weights.get("temporal", 0.0)), ("spatial_candidate", weights.get("spatial", 0.0)), ("era5_candidate", weights.get("era5", 0.0))]
    out = np.full(len(df), np.nan, dtype=float)
    attack_pred = df.get("_attack_pred", pd.Series(True, index=df.index)).astype(bool).to_numpy()
    for i, (_, row) in enumerate(df.iterrows()):
        # Recovery is an action only after the detector raises an alert.
        # Unflagged observations must pass through unchanged.
        if not attack_pred[i]:
            if pd.notna(row.get("observed_value")):
                out[i] = float(row["observed_value"])
            continue
        vals, ws = [], []
        for col, weight in names:
            value = row.get(col)
            if weight > 0 and pd.notna(value):
                vals.append(float(value)); ws.append(float(weight))
        if ws:
            out[i] = float(np.average(vals, weights=ws))
        elif pd.notna(row.get("observed_value")):
            out[i] = float(row["observed_value"])
    return out


def choose_weights(valid_df: pd.DataFrame) -> dict[str, float]:
    candidates = [
        {"temporal": 1.0, "spatial": 0.0, "era5": 0.0},
        {"temporal": 0.6, "spatial": 0.3, "era5": 0.1},
        {"temporal": 0.5, "spatial": 0.3, "era5": 0.2},
        {"temporal": 0.4, "spatial": 0.4, "era5": 0.2},
    ]
    attacked = valid_df["is_anomaly"].astype(bool)
    best = (float("inf"), candidates[0])
    for weights in candidates:
        pred = combine_candidates(valid_df, weights)
        true = pd.to_numeric(valid_df.loc[attacked, "original_value"], errors="coerce").to_numpy(dtype=float)
        est = pred[attacked.to_numpy()]
        finite = np.isfinite(true) & np.isfinite(est)
        if finite.any():
            rmse = float(np.sqrt(np.mean((true[finite] - est[finite]) ** 2)))
            if rmse < best[0]:
                best = (rmse, weights)
    return best[1]
