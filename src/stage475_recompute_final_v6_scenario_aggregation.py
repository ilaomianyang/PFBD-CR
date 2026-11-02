"""Recompute final V6 scenario aggregation with the frozen Stage353 rule.

The frozen win-share definition uses the mean of provider-seed event coverage,
not an event-count-weighted coverage. This script only re-aggregates the
already saved per-run scenario metrics; it does not rerun models or attacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
SCENARIO = TABLES / "final_v6_scenario_metrics.csv"
AGG = TABLES / "final_v6_scenario_aggregate.csv"
WINS = TABLES / "final_v6_scenario_wins.csv"
LENGTHS = TABLES / "final_v6_event_lengths.csv"
REBUILD = TABLES / "final_v6_rebuild.json"

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


def main() -> int:
    frame = pd.read_csv(SCENARIO)
    rows: list[dict[str, object]] = []
    for (scenario, method), group in frame.groupby(["scenario", "method"], sort=True):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())
        events = int(group["events"].sum())
        detected_events = int(group["detected_events"].sum())
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "runs": int(len(group)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                "row_recall": float(group["row_recall"].mean()),
                "f1": (
                    float(2 * tp / (2 * tp + fp + fn))
                    if (2 * tp + fp + fn)
                    else 0.0
                ),
                "false_alarm_rate": (
                    float(fp / (fp + tn)) if (fp + tn) else 0.0
                ),
                "roc_auc": float(group["roc_auc"].mean()),
                "average_precision": float(group["average_precision"].mean()),
                "events": events,
                "detected_events": detected_events,
                "event_coverage": float(group["event_coverage"].mean()),
                "event_coverage_weighted": (
                    float(detected_events / events) if events else 0.0
                ),
                "mean_event_length": float(group["mean_event_length"].mean()),
                "min_event_length": int(group["min_event_length"].min()),
                "max_event_length": int(group["max_event_length"].max()),
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(AGG, index=False, encoding="utf-8-sig")

    wins_rows: list[dict[str, object]] = []
    for scenario, group in aggregate.groupby("scenario", sort=True):
        candidate = group[group["method"].eq("trust_shape_union")]
        if candidate.empty:
            continue
        candidate_value = float(candidate["event_coverage"].iloc[0])
        wins = 0
        for baseline in BASELINES:
            baseline_row = group[group["method"].eq(baseline)]
            if baseline_row.empty:
                continue
            if candidate_value > float(baseline_row["event_coverage"].iloc[0]):
                wins += 1
        wins_rows.append(
            {
                "scenario": scenario,
                "candidate": "trust_shape_union",
                "candidate_event_coverage": candidate_value,
                "wins": wins,
                "total_baselines": len(BASELINES),
                "win_share": wins / len(BASELINES),
                "passes_single_scenario_70pct": wins / len(BASELINES) >= 0.70,
            }
        )
    wins_frame = pd.DataFrame(wins_rows)
    wins_frame.to_csv(WINS, index=False, encoding="utf-8-sig")
    event_lengths = (
        aggregate.groupby(["scenario", "method"], as_index=False)[
            ["mean_event_length", "events"]
        ]
        .mean()
        .sort_values(["scenario", "method"])
    )
    event_lengths.to_csv(LENGTHS, index=False, encoding="utf-8-sig")

    payload = json.loads(REBUILD.read_text(encoding="utf-8"))
    payload["scenario_win_share"] = (
        float(wins_frame["wins"].sum())
        / float(wins_frame["total_baselines"].sum())
    )
    payload["scenario_win_count"] = int(wins_frame["wins"].sum())
    payload["scenario_win_denominator"] = int(wins_frame["total_baselines"].sum())
    payload["scenario_aggregation_rule"] = (
        "Frozen Stage353 rule: provider-seed mean event coverage. "
        "event_coverage_weighted is retained as a secondary descriptive column."
    )
    REBUILD.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS_RECOMPUTED_FINAL_V6_SCENARIO_AGGREGATION",
                "wins": payload["scenario_win_count"],
                "total": payload["scenario_win_denominator"],
                "share": payload["scenario_win_share"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
