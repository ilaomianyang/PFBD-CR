"""Stage 501: exploratory V6 event-hit sensitivity audit.

This is a descriptive re-analysis of the already inspected V6 test split.
It does not change predictions, thresholds, model selection, or the frozen
submission package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
INPUT = FINAL / "tables" / "stage490_union_scores.parquet"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
EVENT_OUT = TABLES / "stage501_v6_event_detection_sensitivity.csv"
SUMMARY_OUT = TABLES / "stage501_v6_event_detection_sensitivity_summary.csv"
ALARM_OUT = TABLES / "stage501_v6_alarm_burden.csv"
ALARM_SUMMARY_OUT = TABLES / "stage501_v6_alarm_burden_summary.csv"
JSON_OUT = REPORTS / "stage501_v6_event_detection_sensitivity.json"
MD_OUT = REPORTS / "stage501_v6_event_detection_sensitivity.md"

METHOD = "trust_shape_union"
HIT_FRACTIONS = (0.01, 0.10, 0.25, 0.50, 1.00)
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20261011
REQUIRED_COLUMNS = {
    "source",
    "seed",
    "split",
    "station_id",
    "attack_scenario",
    "attack_event_id",
    "is_attack",
    "observation_present",
    "union_prediction",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_event_rows(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"input is missing required columns: {sorted(missing)}")

    attack = frame.loc[frame["is_attack"].astype(bool)].copy()
    attack = attack.loc[
        attack["attack_event_id"].astype(str).str.len().gt(0)
        & attack["attack_scenario"].astype(str).ne("normal")
    ]
    if attack.empty:
        raise ValueError("test split contains no labeled attack rows")

    attack["_predicted"] = attack["union_prediction"].astype(bool)
    keys = [
        "source",
        "seed",
        "station_id",
        "attack_scenario",
        "attack_event_id",
    ]
    events = (
        attack.groupby(keys, sort=True, dropna=False)
        .agg(
            attack_rows=("_predicted", "size"),
            detected_rows=("_predicted", "sum"),
        )
        .reset_index()
    )
    events["hit_fraction"] = events["detected_rows"] / events["attack_rows"]
    for fraction in HIT_FRACTIONS:
        label = f"coverage_ge_{int(fraction * 100):03d}pct"
        events[label] = (
            events["detected_rows"]
            >= np.ceil(fraction * events["attack_rows"]).astype(int)
        )
    return events


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        f"coverage_ge_{int(fraction * 100):03d}pct"
        for fraction in HIT_FRACTIONS
    ]
    grouped = (
        events.groupby(["source", "attack_scenario"], sort=True)
        .agg(
            events=("attack_rows", "size"),
            attack_rows=("attack_rows", "sum"),
            detected_attack_rows=("detected_rows", "sum"),
            **{
                metric: (metric, "mean")
                for metric in metric_columns
            },
        )
        .reset_index()
    )
    return grouped


def summarize_alarm_burden(frame: pd.DataFrame) -> pd.DataFrame:
    normal = frame.loc[
        ~frame["is_attack"].astype(bool)
        & frame["observation_present"].astype(bool)
    ].copy()
    normal["_false_alarm"] = normal["union_prediction"].astype(bool)
    grouped = (
        normal.groupby(["source", "seed", "station_id"], sort=True)
        .agg(
            normal_observations=("_false_alarm", "size"),
            false_alarm_observations=("_false_alarm", "sum"),
        )
        .reset_index()
    )
    grouped["false_alarms_per_10000_normal_rows"] = (
        10000
        * grouped["false_alarm_observations"]
        / grouped["normal_observations"]
    )
    return grouped


def summarize_alarm_burden_by_provider(alarms: pd.DataFrame) -> pd.DataFrame:
    station = (
        alarms.groupby(["source", "station_id"], sort=True)
        .agg(
            normal_observations=("normal_observations", "sum"),
            false_alarm_observations=("false_alarm_observations", "sum"),
        )
        .reset_index()
    )
    station["station_rate_per_10000"] = (
        10000
        * station["false_alarm_observations"]
        / station["normal_observations"]
    )
    rows: list[dict[str, object]] = []
    for provider, group in station.groupby("source", sort=True):
        fp = int(group["false_alarm_observations"].sum())
        normal = int(group["normal_observations"].sum())
        rows.append(
            {
                "source": str(provider),
                "stations": int(len(group)),
                "false_alarm_observations": fp,
                "normal_observations": normal,
                "pooled_rate_per_10000": 10000 * fp / normal if normal else 0.0,
                "station_equal_rate_per_10000": float(
                    group["station_rate_per_10000"].mean()
                ),
            }
        )
    all_fp = int(station["false_alarm_observations"].sum())
    all_normal = int(station["normal_observations"].sum())
    rows.append(
        {
            "source": "ALL_POOLED",
            "stations": int(len(station)),
            "false_alarm_observations": all_fp,
            "normal_observations": all_normal,
            "pooled_rate_per_10000": (
                10000 * all_fp / all_normal if all_normal else 0.0
            ),
            "station_equal_rate_per_10000": float(
                station["station_rate_per_10000"].mean()
            ),
        }
    )
    return pd.DataFrame(rows)


def hierarchical_station_bootstrap(
    events: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Provider -> station bootstrap of the event-weighted coverage rates."""
    metric_columns = [
        f"coverage_ge_{int(fraction * 100):03d}pct"
        for fraction in HIT_FRACTIONS
    ]
    station = (
        events.groupby(["source", "station_id"], sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    providers = sorted(station["source"].astype(str).unique())
    if len(providers) < 2:
        raise ValueError("at least two providers are required for this audit")
    by_provider = {
        provider: station.loc[station["source"].eq(provider), metric_columns]
        .to_numpy(dtype=float)
        for provider in providers
    }
    rng = np.random.default_rng(seed)
    draws = np.empty((iterations, len(metric_columns)), dtype=float)
    for iteration in range(iterations):
        sampled_providers = rng.choice(providers, size=len(providers), replace=True)
        provider_means = []
        for provider in sampled_providers:
            rows = by_provider[str(provider)]
            indices = rng.integers(0, len(rows), size=len(rows))
            provider_means.append(rows[indices].mean(axis=0))
        draws[iteration] = np.mean(provider_means, axis=0)

    rows: list[dict[str, object]] = []
    for index, metric in enumerate(metric_columns):
        point = float(station[metric].groupby(station["source"]).mean().mean())
        rows.append(
            {
                "metric": metric,
                "provider_station_equal_estimate": point,
                "ci95_low": float(np.quantile(draws[:, index], 0.025)),
                "ci95_high": float(np.quantile(draws[:, index], 0.975)),
                "iterations": iterations,
                "bootstrap_seed": seed,
                "providers": len(providers),
                "stations": int(
                    station[["source", "station_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def run() -> dict[str, object]:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    input_hash = sha256_file(INPUT)
    data = pd.read_parquet(INPUT)
    test = data.loc[data["split"].astype(str).eq("test")].copy()
    if test.empty:
        raise ValueError("no test rows found in the frozen score file")
    if test["source"].nunique() != 4:
        raise ValueError("expected all four frozen providers in the V6 test split")

    events = derive_event_rows(test)
    summary = summarize_events(events)
    alarms = summarize_alarm_burden(test)
    alarm_summary = summarize_alarm_burden_by_provider(alarms)
    intervals = hierarchical_station_bootstrap(events)

    events.to_csv(EVENT_OUT, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    alarms.to_csv(ALARM_OUT, index=False, encoding="utf-8-sig")
    alarm_summary.to_csv(ALARM_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    payload: dict[str, object] = {
        "stage": 501,
        "status": "EXPLORATORY_V6_EVENT_SENSITIVITY_COMPLETE",
        "evidence_class": "exploratory_reanalysis_of_previously_inspected_test",
        "method": METHOD,
        "input": str(INPUT),
        "input_sha256": input_hash,
        "test_rows": int(len(test)),
        "providers": sorted(test["source"].astype(str).unique().tolist()),
        "stations": int(test["station_id"].nunique()),
        "seeds": sorted(int(value) for value in test["seed"].unique()),
        "attack_events": int(len(events)),
        "hit_fraction_thresholds": list(HIT_FRACTIONS),
        "event_grouping": [
            "provider",
            "seed",
            "station_id",
            "attack_scenario",
            "attack_event_id",
        ],
        "primary_existing_definition": "at least one detected attack-labeled row per event",
        "alarm_burden_definition": (
            "false-positive alarmed observations per 10,000 rows with "
            "is_attack=false and observation_present=true"
        ),
        "bootstrap": {
            "unit": "provider -> station",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "intervals": intervals.to_dict(orient="records"),
        },
        "alarm_burden_summary": alarm_summary.to_dict(orient="records"),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "outputs": [
            str(EVENT_OUT),
            str(SUMMARY_OUT),
            str(ALARM_OUT),
            str(ALARM_SUMMARY_OUT),
        ],
        "claim_boundary": [
            "This is not a new independent or confirmatory test.",
            "No model, threshold, attack label, or prediction was changed.",
            "This audit evaluates the frozen trust_shape_union, not the Stage500 exploratory candidate.",
            "The 1% hit-fraction endpoint is equivalent to any-hit event coverage.",
            "Alarmed observations are not false-alarm episodes or user-facing incidents.",
            "V6 uses a centered-window batch component and is not strictly causal.",
            "The current V9 runner does not include the Stage500 candidate and must be amended before V9 data access.",
        ],
    }
    JSON_OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    top = "\n".join(
        [
            "| Event hit criterion | Provider-station equal coverage | 95% CI |",
            "|---|---:|---:|",
            *[
                f"| {int(float(row.metric.rsplit('_', 1)[-1].replace('pct', '')))}% | "
                f"{row.provider_station_equal_estimate:.4f} | "
                f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}] |"
                for row in intervals.itertuples(index=False)
            ],
        ]
    )
    alarm_table = "\n".join(
        [
            "| Provider | Stations | False-alarm rows | Normal rows | Pooled per 10,000 | Station-equal per 10,000 |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {row.source} | {int(row.stations)} | "
                f"{int(row.false_alarm_observations):,} | "
                f"{int(row.normal_observations):,} | "
                f"{row.pooled_rate_per_10000:.2f} | "
                f"{row.station_equal_rate_per_10000:.2f} |"
                for row in alarm_summary.itertuples(index=False)
            ],
        ]
    )
    report = "\n".join(
        [
            "# Stage501 V6 Event-Detection Sensitivity",
            "",
            f"- Status: `{payload['status']}`",
            "- Evidence class: exploratory re-analysis; the V6 test was already inspected.",
            f"- Frozen method: `{METHOD}`; no model or threshold changes.",
            f"- Test rows: `{len(test):,}`; providers: `{test['source'].nunique()}`; "
            f"stations: `{test['station_id'].nunique()}`; events: `{len(events):,}`.",
            "- Event hit criteria: at least 1%, 10%, 25%, 50%, or 100% of "
            "attack-labeled rows in that event detected.",
            "- Event bootstrap: provider -> station, 10,000 iterations.",
            "",
            "## Event Coverage",
            "",
            top,
            "",
            "Scenario/provider point estimates are in "
            "`tables/stage501_v6_event_detection_sensitivity_summary.csv`.",
            "",
            "## Alarm Burden",
            "",
            alarm_table,
            "",
            "Rates are alarmed observations per 10,000 observed normal rows, not "
            "alarm episodes. Per station-seed detail is in "
            "`tables/stage501_v6_alarm_burden.csv`.",
            "",
            "## Boundary",
            "",
            "- This is not independent confirmation and cannot replace V9.",
            "- This analysis evaluates the frozen `trust_shape_union`, not the later Stage500 exploratory candidate.",
            "- Predictions, thresholds, attack labels, and model selection were not changed.",
            "- The existing any-hit endpoint is shown as the 1% criterion; stricter "
            "criteria are sensitivity analyses, not replacements selected for better results.",
            "- V6 batch/centered-window results are not strictly causal online evidence.",
            "- The current V9 protocol/runner freezes an older candidate family; amend and re-freeze it before accessing V9 observations.",
            "",
            f"- Input SHA256: `{input_hash}`",
        ]
    )
    MD_OUT.write_text(report + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
