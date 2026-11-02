from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseDetector, normal_subset, numeric_series, robust_scale, sorted_frame


class PhysicalRulesDetector(BaseDetector):
    name = "Physical Rules"

    def __init__(self, valid_min: float = -40.0, valid_max: float = 50.0, quantile: float = 0.999, station_adaptive: bool = False):
        self.valid_min = float(valid_min)
        self.valid_max = float(valid_max)
        self.quantile = float(quantile)
        self.station_adaptive = bool(station_adaptive)
        self.rate_scale = 1.0
        self.station_rate_scales: dict[str, float] = {}
        self.stuck_tolerance = 1e-8

    def fit(self, train_df: pd.DataFrame, normal_mask: pd.Series | None = None) -> "PhysicalRulesDetector":
        ref = sorted_frame(normal_subset(train_df, normal_mask))
        values = numeric_series(ref, "observed_value")
        if "station_key" in ref:
            ordered = ref.assign(_value=values).sort_values(["station_key", "timestamp_utc"])
            diffs = ordered.groupby("station_key", sort=False)["_value"].diff().abs()
        else:
            diffs = values.diff().abs()
        finite = diffs[np.isfinite(diffs)]
        self.rate_scale = max(float(finite.quantile(self.quantile)) if len(finite) else 1.0, 0.25)
        self.station_rate_scales = {}
        if self.station_adaptive and "station_key" in ref:
            for station, group in ordered.groupby("station_key", sort=False):
                station_diffs = group["_value"].diff().abs()
                station_finite = station_diffs[np.isfinite(station_diffs)]
                self.station_rate_scales[str(station)] = max(
                    float(station_finite.quantile(self.quantile)) if len(station_finite) else self.rate_scale,
                    0.25,
                )
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        ref = sorted_frame(df)
        values = numeric_series(ref, "observed_value")
        range_violation = np.where(values < self.valid_min, self.valid_min - values, 0.0)
        range_violation = np.where(values > self.valid_max, values - self.valid_max, range_violation)
        if "station_key" in ref:
            ordered = ref.assign(_value=values).sort_values(["station_key", "timestamp_utc"])
            diff = ordered.groupby("station_key", sort=False)["_value"].diff().abs()
            diff = diff.reindex(ref.index)
        else:
            diff = values.diff().abs()
        if self.station_adaptive and "station_key" in ref:
            scales = ref["station_key"].astype(str).map(self.station_rate_scales).fillna(self.rate_scale).to_numpy(dtype=float)
            rate_score = np.nan_to_num(diff.to_numpy(dtype=float) / scales, nan=0.0, posinf=1e6)
        else:
            rate_score = np.nan_to_num(diff.to_numpy(dtype=float) / self.rate_scale, nan=0.0, posinf=1e6)
        # A long constant run is suspicious, but a single repeated value is not enough.
        repeated = np.zeros(len(ref), dtype=float)
        if "station_key" in ref:
            ordered = ref.assign(_value=values).sort_values(["station_key", "timestamp_utc"])
            same = ordered.groupby("station_key", sort=False)["_value"].diff().abs().fillna(np.inf).le(self.stuck_tolerance)
            run = same.groupby(ordered["station_key"], sort=False).cumsum().to_numpy(dtype=float)
            repeated = np.maximum(run - 2.0, 0.0)
            repeated = pd.Series(repeated, index=ordered.index).reindex(ref.index).fillna(0.0).to_numpy()
        score = np.maximum.reduce([np.nan_to_num(range_violation / 0.5, nan=0.0), rate_score, repeated])
        return pd.Series(score, index=ref.index).reindex(df.index).fillna(0.0).to_numpy(dtype=float)
