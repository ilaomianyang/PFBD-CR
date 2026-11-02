from __future__ import annotations

import numpy as np
import pandas as pd


def robust_location_scale(values: pd.Series | np.ndarray, floor: float = 1e-6) -> tuple[float, float]:
    arr = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, float(floor)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, max(1.4826 * mad, float(floor))


class TrustFusion:
    """多证据风险融合；输入列的值越大表示越可疑。"""

    detector_names = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency"]

    def __init__(self, include_era5: bool = True):
        self.include_era5 = include_era5
        self.components: list[str] = []
        self.location_scale: dict[str, tuple[float, float]] = {}
        self.weights: dict[str, float] = {}
        self.threshold = 1.0
        self.extreme_factor = 1.0

    def fit(self, train_scores: pd.DataFrame, valid_scores: pd.DataFrame, valid_labels: pd.Series) -> "TrustFusion":
        candidate = list(self.detector_names)
        if self.include_era5 and "ERA5 Background" in train_scores:
            candidate.append("ERA5 Background")
        self.components = candidate
        label_col = "combined_is_attack" if "combined_is_attack" in train_scores else "is_anomaly"
        normal_train = ~train_scores[label_col].astype(bool) if label_col in train_scores else pd.Series(True, index=train_scores.index)
        for name in self.components:
            self.location_scale[name] = robust_location_scale(train_scores.loc[normal_train, name], floor=0.25)
        prevalence = float(valid_labels.mean())
        raw_weights = {}
        for name in self.components:
            score = pd.to_numeric(valid_scores[name], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(score)
            if finite.sum() and len(np.unique(valid_labels.to_numpy()[finite])) == 2:
                from sklearn.metrics import average_precision_score
                ap = float(average_precision_score(valid_labels.to_numpy()[finite], score[finite]))
            else:
                ap = prevalence
            raw_weights[name] = max(ap - prevalence, 0.01)
        total = sum(raw_weights.values())
        self.weights = {name: value / total for name, value in raw_weights.items()}
        valid_fused = self.transform(valid_scores)
        from sklearn.metrics import f1_score
        candidates = np.unique(np.quantile(valid_fused[np.isfinite(valid_fused)], np.linspace(0.02, 0.98, 193)))
        self.threshold = max(((f1_score(valid_labels, valid_fused >= t, zero_division=0), float(t)) for t in candidates), key=lambda x: x[0])[1]
        return self

    def transform(self, scores: pd.DataFrame, extreme_background: pd.Series | None = None) -> np.ndarray:
        if not self.components:
            raise RuntimeError("TrustFusion must be fitted before transform")
        total = np.zeros(len(scores), dtype=float)
        for name in self.components:
            values = pd.to_numeric(scores[name], errors="coerce").to_numpy(dtype=float)
            med, scale = self.location_scale[name]
            z = np.nan_to_num(np.maximum(values - med, 0.0) / scale, nan=0.0, posinf=1e6)
            total += self.weights.get(name, 0.0) * z
        missing = scores.get("missing_observation", pd.Series(False, index=scores.index)).astype(bool).to_numpy()
        total[missing] += 1.0
        if extreme_background is not None:
            # Widespread regional changes should be less likely to be treated as local tampering.
            total = np.where(extreme_background.to_numpy(dtype=bool), 0.75 * total, total)
        return total

    def predict(self, fused_score: np.ndarray) -> np.ndarray:
        return np.asarray(fused_score >= self.threshold, dtype=bool)

    def trust_score(self, fused_score: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.maximum(fused_score, 0.0))
