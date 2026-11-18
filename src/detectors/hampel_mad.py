from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseDetector, normal_subset, numeric_series, robust_scale, sorted_frame


class HampelMADDetector(BaseDetector):
    """Hampel/MAD residual detector with a configurable window direction.

    ``center=True`` reproduces the frozen V6 configuration, whose rolling
    median is centred and therefore reads samples after the current step.
    ``center=False`` uses a backward-only window that ends at the current
    step, which keeps the residual free of future samples and matches the
    strictly causal streaming formulation.  The default stays ``True`` so the
    frozen V6 numbers remain reproducible.
    """

    name = "Hampel MAD"

    def __init__(self, window: int = 7, scale_floor: float = 0.25, center: bool = True):
        self.window = int(window)
        self.scale_floor = float(scale_floor)
        self.center = bool(center)
        self.global_scale = 1.0

    def _rolling_median(self, values: pd.Series) -> pd.Series:
        return values.rolling(self.window, center=self.center, min_periods=3).median()

    def _residual(self, df: pd.DataFrame) -> pd.Series:
        ref = sorted_frame(df)
        values = numeric_series(ref, "observed_value")
        if "station_key" not in ref:
            med = self._rolling_median(values)
            return (values - med).abs().reindex(df.index)
        ordered = ref.assign(_value=values)
        med = ordered.groupby("station_key", sort=False)["_value"].transform(self._rolling_median)
        return (ordered["_value"] - med).abs().set_axis(ordered.index).reindex(df.index)

    def fit(self, train_df: pd.DataFrame, normal_mask: pd.Series | None = None) -> "HampelMADDetector":
        ref = normal_subset(train_df, normal_mask)
        residual = self._residual(ref)
        _, self.global_scale = robust_scale(residual, floor=self.scale_floor)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        residual = self._residual(df)
        return np.nan_to_num(residual.to_numpy(dtype=float) / self.global_scale, nan=0.0, posinf=1e6)
