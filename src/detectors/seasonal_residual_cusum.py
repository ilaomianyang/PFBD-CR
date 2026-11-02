from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseDetector, normal_subset, numeric_series, robust_scale, sorted_frame


class SeasonalResidualCUSUMDetector(BaseDetector):
    """按站点和日内时刻去除季节基线后，对残差做双侧 CUSUM。"""

    name = "Seasonal Residual CUSUM"

    def __init__(self, drift_factor: float = 0.5, reset_factor: float = 5.0):
        self.drift_factor = float(drift_factor)
        self.reset_factor = float(reset_factor)
        self.baseline: dict[tuple[str, int], float] = {}
        self.station_baseline: dict[str, float] = {}
        self.station_scale: dict[str, float] = {}
        self.global_baseline = 0.0
        self.global_scale = 1.0

    @staticmethod
    def _slot(df: pd.DataFrame) -> pd.Series:
        ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        return ts.dt.hour * 60 + ts.dt.minute

    def fit(self, train_df: pd.DataFrame, normal_mask: pd.Series | None = None) -> "SeasonalResidualCUSUMDetector":
        ref = normal_subset(train_df, normal_mask).copy()
        ref["_value"] = numeric_series(ref, "observed_value")
        ref["_slot"] = self._slot(ref)
        self.global_baseline = float(ref["_value"].median()) if ref["_value"].notna().any() else 0.0
        self.global_scale = robust_scale(ref["_value"], floor=0.25)[1]
        self.station_baseline = ref.groupby("station_key", sort=False)["_value"].median().dropna().astype(float).to_dict()
        for station, group in ref.groupby("station_key", sort=False):
            self.station_scale[str(station)] = robust_scale(group["_value"], floor=0.25)[1]
        grouped = ref.dropna(subset=["_value", "_slot"]).groupby(["station_key", "_slot"], sort=False)["_value"].median()
        self.baseline = {(str(station), int(slot)): float(value) for (station, slot), value in grouped.items()}
        return self

    def _residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        ref = sorted_frame(df).copy()
        ref["_value"] = numeric_series(ref, "observed_value")
        ref["_slot"] = self._slot(ref)
        values = []
        scales = []
        for station, slot, value in zip(ref["station_key"].astype(str), ref["_slot"], ref["_value"]):
            base = self.baseline.get((station, int(slot)) if pd.notna(slot) else (station, -1), self.station_baseline.get(station, self.global_baseline))
            values.append(float(value - base) if np.isfinite(value) else np.nan)
            scales.append(self.station_scale.get(station, self.global_scale))
        ref["_residual"] = values
        ref["_scale"] = scales
        return ref

    def score(self, df: pd.DataFrame) -> np.ndarray:
        ref = self._residuals(df)
        out = pd.Series(0.0, index=ref.index, dtype=float)
        for _, group in ref.groupby("station_key", sort=False):
            pos = neg = 0.0
            scale = float(group["_scale"].median())
            k = self.drift_factor * scale
            h = self.reset_factor * scale
            for idx, residual in zip(group.index, group["_residual"]):
                if not np.isfinite(residual):
                    continue
                pos = max(0.0, pos + float(residual) - k)
                neg = max(0.0, neg - float(residual) - k)
                out.loc[idx] = max(pos, neg) / max(h, 1e-9)
                if pos > h or neg > h:
                    pos *= 0.5
                    neg *= 0.5
        return out.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
