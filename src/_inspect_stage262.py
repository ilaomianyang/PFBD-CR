from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"


def main() -> None:
    payload = json.loads(
        (TABLES / "stage262_security_domain_baselines.json").read_text(
            encoding="utf-8"
        )
    )
    print(payload["status"])
    print(
        "providers",
        payload["provider_count"],
        "stations",
        payload["station_count"],
        "seeds",
        payload["seed_count"],
        "methods",
        payload["method_count"],
    )
    print(
        "rows",
        payload["evaluation_rows"],
        payload["station_rows"],
        payload["type_rows"],
    )
    print(
        json.dumps(
            payload["provider_equal"],
            ensure_ascii=False,
            indent=2,
        )
    )
    evaluation = pd.read_csv(
        TABLES / "stage262_security_domain_baselines_evaluation.csv"
    )
    print(
        evaluation.groupby("method")[
            [
                "f1",
                "recall",
                "precision",
                "false_alarm_rate",
                "average_precision",
                "event_coverage",
                "row_recall",
            ]
        ]
        .mean()
        .round(4)
        .to_string()
    )
    comparisons = pd.read_csv(
        TABLES / "stage262_security_domain_baselines_comparisons.csv"
    )
    print(
        comparisons[
            [
                "candidate",
                "baseline",
                "f1_difference",
                "f1_ci95_low",
                "f1_ci95_high",
                "event_coverage_difference",
                "event_ci95_low",
                "event_ci95_high",
                "f1_bootstrap_bh_q",
                "event_coverage_bootstrap_bh_q",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    type_aggregate = pd.read_csv(
        TABLES / "stage262_security_domain_baselines_type_aggregate.csv"
    )
    print(
        type_aggregate.groupby(["method", "attack_type"])[
            [
                "attack_rows",
                "detected_rows",
                "row_recall",
                "event_coverage",
            ]
        ]
        .mean()
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
