from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseDetector, normal_subset, numeric_series, robust_scale, sorted_frame


class CUSUMDetector(BaseDetector):
    name = "CUSUM"

    def __init__(self, drift_factor: float = 0.5, reset_factor: float = 4.0):
        self.drift_factor = float(drift_factor)
        self.reset_factor = float(reset_factor)
        self.center = 0.0
        self.scale = 1.0

    def fit(self, train_df: pd.DataFrame, normal_mask: pd.Series | None = None) -> "CUSUMDetector":
        ref = normal_subset(train_df, normal_mask)
        values = numeric_series(ref, "observed_value")
        self.center, self.scale = robust_scale(values, floor=0.25)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        ref = sorted_frame(df)
        values = numeric_series(ref, "observed_value")
        out = pd.Series(0.0, index=ref.index, dtype=float)
        groups = ref.groupby("station_key", sort=False) if "station_key" in ref else [("__global__", ref)]
        k = self.drift_factor * self.scale
        h = self.reset_factor * self.scale
        for _, group in groups:
            pos = neg = 0.0
            for idx, value in zip(group.index, pd.to_numeric(group["observed_value"], errors="coerce")):
                if not np.isfinite(value):
                    continue
                delta = float(value) - self.center
                pos = max(0.0, pos + delta - k)
                neg = max(0.0, neg - delta - k)
                out.loc[idx] = max(pos, neg) / max(h, 1e-9)
                if pos > h or neg > h:
                    pos *= 0.5
                    neg *= 0.5
        return out.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
