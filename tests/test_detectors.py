from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from detectors import (  # noqa: E402
    CUSUMDetector,
    HampelMADDetector,
    KalmanInnovationDetector,
    PhysicalRulesDetector,
    SpatialConsistencyDetector,
    SeasonalResidualCUSUMDetector,
)


def toy_frame() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    for t in range(12):
        ts = start + pd.Timedelta(hours=t)
        rows.extend([
            {"station_key": "a", "timestamp_utc": ts, "observed_value": 10.0 + 0.1 * t, "split": "train", "is_anomaly": False},
            {"station_key": "b", "timestamp_utc": ts, "observed_value": 10.2 + 0.1 * t, "split": "train", "is_anomaly": False},
        ])
    df = pd.DataFrame(rows)
    df.loc[(df["station_key"] == "a") & (df.index >= 10), "observed_value"] += 8.0
    df.loc[df.index >= 10, "split"] = "valid"
    df.loc[df.index >= 10, "is_anomaly"] = True
    return df


def test_all_detectors_fit_score_predict():
    df = toy_frame()
    train = df["split"].eq("train")
    normal = train & ~df["is_anomaly"]
    detectors = [
        PhysicalRulesDetector(),
        HampelMADDetector(),
        KalmanInnovationDetector(),
        CUSUMDetector(),
        SpatialConsistencyDetector(),
    ]
    for detector in detectors:
        detector.fit(df.loc[train], normal.loc[train])
        scores = detector.score(df)
        pred = detector.predict(df, 1.0)
        assert scores.shape == (len(df),)
        assert pred.shape == (len(df),)
        assert np.isfinite(scores).all()


def test_spatial_detector_responds_to_single_station_outlier():
    df = toy_frame()
    train = df["split"].eq("train")
    normal = train & ~df["is_anomaly"]
    detector = SpatialConsistencyDetector().fit(df.loc[train], normal.loc[train])
    scores = detector.score(df)
    attacked = (df["station_key"] == "a") & df["is_anomaly"]
    clean = (df["station_key"] == "b") & df["is_anomaly"]
    assert float(np.nanmean(scores[attacked])) > float(np.nanmean(scores[clean]))


def test_seasonal_residual_cusum_is_finite():
    df = toy_frame()
    train = df["split"].eq("train")
    detector = SeasonalResidualCUSUMDetector().fit(df.loc[train], train.loc[train])
    scores = detector.score(df)
    assert scores.shape == (len(df),)
    assert np.isfinite(scores).all()
