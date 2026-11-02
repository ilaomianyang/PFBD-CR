from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseDetector(ABC):
    """统一的轻量检测器接口。score 越大表示越可疑。"""

    name: str = "base"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, normal_mask: pd.Series | None = None) -> "BaseDetector":
        raise NotImplementedError

    @abstractmethod
    def score(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame, threshold: float) -> np.ndarray:
        scores = self.score(df)
        return np.asarray(scores >= float(threshold), dtype=bool)


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def normal_subset(df: pd.DataFrame, normal_mask: pd.Series | None) -> pd.DataFrame:
    if normal_mask is None:
        return df
    return df.loc[normal_mask.reindex(df.index, fill_value=False).astype(bool)]


def robust_scale(values: pd.Series | np.ndarray, floor: float = 1e-3) -> tuple[float, float]:
    arr = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, float(floor)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return median, max(1.4826 * mad, float(floor))


def sorted_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [c for c in ["station_key", "timestamp_utc"] if c in df]
    if not required:
        return df.copy()
    out = df.copy()
    out["_tmq_timestamp"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True)
    return out.sort_values(required[:-1] + ["_tmq_timestamp"] if len(required) == 2 else required).drop(columns=["_tmq_timestamp"])
