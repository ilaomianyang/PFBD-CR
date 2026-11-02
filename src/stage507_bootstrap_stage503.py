"""Provider-seed bootstrap for the Stage503 deep baselines.

This is a simple exploratory interval: resample the 20 provider-seed runs with
replacement.  It is NOT the V6 provider-first hierarchical bootstrap and must
not be reported as the confirmatory interval.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
METRICS = ["precision", "recall", "f1", "false_alarm_rate", "roc_auc",
           "average_precision", "event_coverage", "row_recall"]
MODELS = ["lstm_ae", "usad", "timesnet", "anomaly_transformer"]
LABELS = {
    "lstm_ae": "LSTM-AE",
    "usad": "USAD",
    "timesnet": "TimesNet",
    "anomaly_transformer": "Anomaly Transformer",
}


def main() -> int:
    detail = pd.read_csv(TABLES / "stage503_deep_baseline_detail.csv")
    test = detail[detail["split"].eq("test")]
    if len(test) != 80:
        raise SystemExit(f"expected 80 test rows, got {len(test)}")
    rng = np.random.default_rng(20261019)
    rows = []
    for model in MODELS:
        sub = test[test["method"].eq(model)][METRICS].to_numpy(float)
        n = len(sub)
        draws = rng.integers(0, n, size=(4000, n))
        for j, metric in enumerate(METRICS):
            values = sub[:, j][draws].mean(axis=1)
            rows.append({
                "model": model,
                "label": LABELS[model],
                "metric": metric,
                "mean": float(sub[:, j].mean()),
                "ci95_lo": float(np.quantile(values, 0.025)),
                "ci95_hi": float(np.quantile(values, 0.975)),
                "runs": n,
                "method": "provider_seed_bootstrap_4000",
                "status": "EXPLORATORY_POST_V6",
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage507_deep_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(out[out["metric"].isin(["f1", "event_coverage", "average_precision"])].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
