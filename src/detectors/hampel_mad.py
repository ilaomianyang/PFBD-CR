from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseDetector, normal_subset, numeric_series, robust_scale, sorted_frame


class HampelMADDetector(BaseDetector):
    name = "Hampel MAD"

    def __init__(self, window: int = 7, scale_floor: float = 0.25):
        self.window = int(window)
        self.scale_floor = float(scale_floor)
        self.global_scale = 1.0

    def _residual(self, df: pd.DataFrame) -> pd.Series:
        ref = sorted_frame(df)
        values = numeric_series(ref, "observed_value")
        if "station_key" not in ref:
            med = values.rolling(self.window, center=True, min_periods=3).median()
            return (values - med).abs().reindex(df.index)
        ordered = ref.assign(_value=values)
        med = ordered.groupby("station_key", sort=False)["_value"].transform(
            lambda s: s.rolling(self.window, center=True, min_periods=3).median()
        )
        return (ordered["_value"] - med).abs().set_axis(ordered.index).reindex(df.index)

    def fit(self, train_df: pd.DataFrame, normal_mask: pd.Series | None = None) -> "HampelMADDetector":
        ref = normal_subset(train_df, normal_mask)
        residual = self._residual(ref)
        _, self.global_scale = robust_scale(residual, floor=self.scale_floor)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        residual = self._residual(df)
        return np.nan_to_num(residual.to_numpy(dtype=float) / self.global_scale, nan=0.0, posinf=1e6)
