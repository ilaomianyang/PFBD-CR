"""Stage 480: station-cluster paired bootstrap correction for V6.

The frozen station table contains 42 unique stations, each repeated across
five seeds.  This stage first aggregates seeds within station, then resamples
providers and stations within providers.  It replaces the earlier practice of
treating 210 station-seed rows as independent station observations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

STATION = TABLES / "final_v6_station.csv"
OUT_CSV = TABLES / "final_v6_station_cluster_ci_corrected.csv"
OUT_JSON = REPORTS / "final_v6_station_cluster_ci_correction.json"
OUT_MD = REPORTS / "final_v6_station_cluster_ci_correction.md"

CANDIDATES = ["trust_shape_union", "trust_only"]
BASELINES = [
    "robust_max_deviation",
    "isolation_forest",
    "one_class_svm",
    "pca_reconstruction",
    "mlp_autoencoder_reconstruction",
    "knn_distance",
    "ewma_deviation",
    "arima_innovation",
    "gru_predictor",
    "lightweight_autoencoder",
    "ics_cps_spc_proxy",
]
METRICS = ["f1", "event_coverage", "precision", "recall", "false_alarm_rate"]
ITERATIONS = 10000
SEED = 20261009


def metric_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (
        float(2 * tp / (2 * tp + fp + fn))
        if (2 * tp + fp + fn)
        else 0.0
    )
    far = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
    }


def station_aggregate(station: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, source, station_id), group in station.groupby(
        ["method", "source", "station_id"], sort=True
    ):
        values = metric_from_counts(
            group["tp"].sum(),
            group["fp"].sum(),
            group["fn"].sum(),
            group["tn"].sum(),
        )
        events = int(group["events"].sum())
        detected = int(group["detected_events"].sum())
        rows.append(
            {
                "method": method,
                "source": source,
                "station_id": station_id,
                **values,
                "event_coverage": (
                    float(detected / events) if events else 0.0
                ),
                "events": events,
                "detected_events": detected,
            }
        )
    return pd.DataFrame(rows)


def paired_station_differences(
    station_metrics: pd.DataFrame,
    candidate: str,
    baseline: str,
) -> pd.DataFrame:
    left = station_metrics[station_metrics["method"].eq(candidate)].drop(
        columns=["method"]
    )
    right = station_metrics[station_metrics["method"].eq(baseline)].drop(
        columns=["method"]
    )
    paired = left.merge(
        right,
        on=["source", "station_id"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    for metric in METRICS:
        paired[f"{metric}_difference"] = (
            paired[f"{metric}_candidate"]
            - paired[f"{metric}_baseline"]
        )
    return paired


def hierarchical_bootstrap(
    paired: pd.DataFrame,
    metric: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, float, float]:
    providers = sorted(paired["source"].astype(str).unique())
    provider_arrays = {
        provider: paired.index[
            paired["source"].astype(str).eq(provider)
        ].to_numpy(dtype=int)
        for provider in providers
    }
    differences = paired[f"{metric}_difference"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_providers = rng.choice(
            providers,
            size=len(providers),
            replace=True,
        )
        selected: list[int] = []
        for provider in sampled_providers:
            rows = provider_arrays[str(provider)]
            selected.extend(
                rng.choice(rows, size=len(rows), replace=True).tolist()
            )
        draws[iteration] = float(differences[np.asarray(selected)].mean())
    point = float(differences.mean())
    low = float(np.quantile(draws, 0.025))
    high = float(np.quantile(draws, 0.975))
    below = int((draws < 0).sum())
    above = int((draws > 0).sum())
    p = float(
        (1.0 + 2.0 * min(below, above))
        / (1.0 + iterations)
    )
    return point, low, high, p


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values, dtype=float)
    n = len(values)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        index = order[rank]
        running = min(running, values[index] * n / (rank + 1))
        adjusted[index] = running
    return np.clip(adjusted, 0.0, 1.0)


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    station = pd.read_csv(STATION)
    station_metrics = station_aggregate(station)
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        for baseline in BASELINES:
            paired = paired_station_differences(
                station_metrics,
                candidate,
                baseline,
            )
            for metric in METRICS:
                point, low, high, p = hierarchical_bootstrap(
                    paired,
                    metric,
                    ITERATIONS,
                    SEED + len(rows),
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "difference": point,
                        "ci95_low": low,
                        "ci95_high": high,
                        "p_two_sided": p,
                        "provider_count": int(
                            paired["source"].nunique()
                        ),
                        "station_count": int(len(paired)),
                        "iterations": ITERATIONS,
                        "resampling": (
                            "providers_with_replacement_then_stations_"
                            "within_provider_with_replacement"
                        ),
                    }
                )
    corrected = pd.DataFrame(rows)
    corrected["bh_q"] = np.nan
    for (candidate, metric), group in corrected.groupby(
        ["candidate", "metric"], sort=True
    ):
        corrected.loc[group.index, "bh_q"] = benjamini_hochberg(
            group["p_two_sided"].to_numpy(dtype=float)
        )
    corrected.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    union = corrected[
        corrected["candidate"].eq("trust_shape_union")
        & corrected["metric"].isin(["f1", "event_coverage"])
    ]
    payload = {
        "status": "PASS_V6_STATION_CLUSTER_CI_CORRECTION",
        "source": str(STATION.relative_to(ROOT)),
        "unique_stations": int(
            station[["source", "station_id"]].drop_duplicates().shape[0]
        ),
        "station_seed_rows": int(len(station)),
        "iterations": ITERATIONS,
        "resampling_unit": (
            "provider then station within provider; both with replacement"
        ),
        "comparison_count": int(len(corrected)),
        "union_primary_examples": union.to_dict("records"),
        "rules": [
            "Seeds are aggregated within station before resampling.",
            "Providers are treated as four observed clusters, not a random "
            "population of institutions.",
            "BH is applied within each candidate-metric family.",
            "Event coverage, F1, precision, recall, and FAR are reported "
            "separately.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# V6 station-cluster CI correction",
        "",
        f"- Status: `{payload['status']}`",
        f"- Unique stations: `{payload['unique_stations']}`; "
        f"station-seed rows: `{payload['station_seed_rows']}`",
        f"- Iterations: `{ITERATIONS}`",
        f"- Resampling: `{payload['resampling_unit']}`",
        "",
        "## Trust union primary comparisons",
        "",
        union.to_markdown(index=False),
        "",
        "## Rules",
        "",
        "- Seeds are averaged within station before bootstrap.",
        "- Provider and station clusters are resampled with replacement.",
        "- BH correction is within candidate and metric families.",
        "- The four providers are not treated as an institutional population.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "unique_stations": payload["unique_stations"],
                "station_seed_rows": payload["station_seed_rows"],
                "comparisons": payload["comparison_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
