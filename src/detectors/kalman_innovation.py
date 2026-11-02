from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseDetector, normal_subset, numeric_series, robust_scale, sorted_frame


class KalmanInnovationDetector(BaseDetector):
    name = "Kalman Innovation"

    def __init__(self, process_scale_floor: float = 0.05, observation_scale_floor: float = 0.10):
        self.process_var = 1.0
        self.observation_var = 1.0
        self.process_scale_floor = float(process_scale_floor)
        self.observation_scale_floor = float(observation_scale_floor)
        self.initial_state: dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame, normal_mask: pd.Series | None = None) -> "KalmanInnovationDetector":
        ref = sorted_frame(normal_subset(train_df, normal_mask))
        values = numeric_series(ref, "observed_value")
        if "station_key" in ref:
            ordered = ref.assign(_value=values).sort_values(["station_key", "timestamp_utc"])
            diff = ordered.groupby("station_key", sort=False)["_value"].diff()
            self.initial_state = ordered.groupby("station_key", sort=False)["_value"].median().dropna().to_dict()
        else:
            diff = values.diff()
            self.initial_state = {"__global__": float(values.median())}
        _, scale = robust_scale(diff, floor=self.process_scale_floor)
        self.process_var = max(scale * scale, self.process_scale_floor**2)
        _, obs = robust_scale(values, floor=self.observation_scale_floor)
        # Observation noise is deliberately conservative; innovations remain interpretable.
        self.observation_var = max((0.5 * obs) ** 2, self.observation_scale_floor**2)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        ref = sorted_frame(df)
        values = numeric_series(ref, "observed_value")
        scores = pd.Series(0.0, index=ref.index, dtype=float)
        groups = ref.groupby("station_key", sort=False) if "station_key" in ref else [("__global__", ref)]
        for station, group in groups:
            state = float(self.initial_state.get(station, pd.to_numeric(group["observed_value"], errors="coerce").median()))
            variance = self.observation_var
            for idx, value in zip(group.index, pd.to_numeric(group["observed_value"], errors="coerce")):
                if not np.isfinite(value):
                    scores.loc[idx] = 0.0
                    continue
                pred_var = variance + self.process_var
                innovation = float(value) - state
                innovation_var = max(pred_var + self.observation_var, 1e-9)
                scores.loc[idx] = abs(innovation) / np.sqrt(innovation_var)
                gain = pred_var / innovation_var
                state = state + gain * innovation
                variance = max((1.0 - gain) * pred_var, 1e-9)
        return scores.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
