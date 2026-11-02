from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseDetector, normal_subset, numeric_series, robust_scale


class SpatialConsistencyDetector(BaseDetector):
    name = "Spatial Consistency"

    def __init__(self, min_support: int = 2, scale_floor: float = 0.25):
        self.min_support = int(min_support)
        self.scale_floor = float(scale_floor)
        self.scale = 1.0

    @staticmethod
    def _residual(df: pd.DataFrame) -> pd.Series:
        values = numeric_series(df, "observed_value")
        if "timestamp_utc" not in df or "station_key" not in df:
            return pd.Series(0.0, index=df.index)
        temp = df[["timestamp_utc", "station_key"]].copy()
        temp["_value"] = values
        med = temp.groupby("timestamp_utc", sort=False)["_value"].transform(
            lambda s: s.median() if s.notna().sum() >= 2 else np.nan
        )
        support = temp.groupby("timestamp_utc", sort=False)["_value"].transform(lambda s: s.notna().sum() - 1)
        residual = (temp["_value"] - med).abs()
        residual = residual.where(support >= 1, 0.0)
        return residual

    def fit(self, train_df: pd.DataFrame, normal_mask: pd.Series | None = None) -> "SpatialConsistencyDetector":
        ref = normal_subset(train_df, normal_mask)
        residual = self._residual(ref)
        _, self.scale = robust_scale(residual, floor=self.scale_floor)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        residual = self._residual(df)
        return np.nan_to_num(residual.to_numpy(dtype=float) / self.scale, nan=0.0, posinf=1e6)
