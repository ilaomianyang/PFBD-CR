from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from secure_recovery import add_recovery_candidates, choose_weights, combine_candidates  # noqa: E402
from trust_fusion import TrustFusion  # noqa: E402


def test_fusion_uses_validation_only_for_threshold_and_weights():
    train = pd.DataFrame({
        "combined_is_attack": [False, False, False, True],
        "Physical Rules": [0.0, 0.2, 0.1, 20.0],
        "Hampel MAD": [0.1, 0.2, 0.0, 10.0],
        "Kalman Innovation": [0.1, 0.2, 0.1, 8.0],
        "CUSUM": [0.0, 0.1, 0.2, 9.0],
        "Spatial Consistency": [0.1, 0.1, 0.2, 7.0],
        "ERA5 Background": [0.2, 0.1, 0.2, 12.0],
    })
    valid = pd.DataFrame({name: [0.1, 0.2, 8.0, 10.0] for name in TrustFusion.detector_names + ["ERA5 Background"]})
    labels = pd.Series([False, False, True, True])
    fusion = TrustFusion().fit(train, valid, labels)
    score = fusion.transform(valid)
    assert set(fusion.weights) == set(TrustFusion.detector_names + ["ERA5 Background"])
    assert np.isclose(sum(fusion.weights.values()), 1.0)
    assert fusion.predict(score).shape == (4,)
    assert score[2:].mean() > score[:2].mean()


def test_recovery_candidates_mask_predictions_and_select_valid_weights():
    df = pd.DataFrame({
        "station_key": ["a"] * 5,
        "timestamp_utc": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        "observed_value": [10.0, 11.0, 30.0, 13.0, 14.0],
        "original_value": [10.0, 11.0, 12.0, 13.0, 14.0],
        "is_anomaly": [False, False, True, False, False],
    })
    candidates = add_recovery_candidates(df, pd.Series([False, False, True, False, False]), pd.Series([10.0, 11.0, 12.0, 13.0, 14.0]))
    weights = choose_weights(candidates)
    recovered = combine_candidates(candidates, weights)
    assert np.isfinite(recovered).all()
    assert abs(recovered[2] - 12.0) < abs(df.loc[2, "observed_value"] - 12.0)
    # Recovery must not rewrite rows that were not flagged by the detector.
    for idx in [0, 1, 3, 4]:
        assert recovered[idx] == df.loc[idx, "observed_value"]
