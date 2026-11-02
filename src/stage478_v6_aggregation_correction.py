"""Stage 478: correct V6 aggregation estimands without rerunning the model.

The frozen V6 detail table stores row metrics but not the underlying counts.
This stage reconstructs TP/FP/FN/TN from the saved metric definitions, then
reports count-pooled, provider-equal, station-equal, and provider-seed macro
estimands separately.  It does not change predictions, thresholds, attacks,
or the original frozen tables.
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

DETAIL = TABLES / "final_v6_detail.csv"
STATION = TABLES / "final_v6_station.csv"
ORIGINAL_AGG = TABLES / "final_v6_aggregation_levels.csv"

OUT_DETAIL_COUNTS = TABLES / "final_v6_detail_counts_reconstructed.csv"
OUT_AGG = TABLES / "final_v6_aggregation_levels_corrected.csv"
OUT_JSON = REPORTS / "final_v6_aggregation_correction.json"
OUT_MD = REPORTS / "final_v6_aggregation_correction.md"


def metric_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
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


def reconstruct_detail_counts(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in detail.to_dict("records"):
        attack_rows = int(row["attack_rows"])
        normal_rows = int(row["rows"]) - attack_rows
        recall = float(row["recall"])
        far = float(row["false_alarm_rate"])
        tp = int(round(recall * attack_rows))
        fp = int(round(far * normal_rows))
        fn = attack_rows - tp
        tn = normal_rows - fp
        reconstructed = metric_from_counts(tp, fp, fn, tn)
        for metric in ("precision", "recall", "f1", "false_alarm_rate"):
            if abs(reconstructed[metric] - float(row[metric])) > 1e-10:
                raise ValueError(
                    f"count reconstruction mismatch for "
                    f"{row['source']}/{row['seed']}/{row['method']}/{metric}"
                )
        rows.append(
            {
                "source": str(row["source"]),
                "seed": int(row["seed"]),
                "method": str(row["method"]),
                "selected": bool(row["selected"]),
                "rows": int(row["rows"]),
                "attack_rows": attack_rows,
                "normal_rows": normal_rows,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                **reconstructed,
                "roc_auc": float(row["roc_auc"]),
                "average_precision": float(row["average_precision"]),
                "events": int(row["events"]),
                "detected_events": int(row["detected_events"]),
                "event_coverage": float(row["event_coverage"]),
                "row_recall": float(row["row_recall"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_counts(group: pd.DataFrame, aggregation: str) -> dict[str, object]:
    tp = int(group["tp"].sum())
    fp = int(group["fp"].sum())
    fn = int(group["fn"].sum())
    tn = int(group["tn"].sum())
    attack_rows = (
        int(group["attack_rows"].sum())
        if "attack_rows" in group
        else tp + fn
    )
    normal_rows = (
        int(group["normal_rows"].sum())
        if "normal_rows" in group
        else fp + tn
    )
    events = int(group["events"].sum())
    detected = int(group["detected_events"].sum())
    values = metric_from_counts(tp, fp, fn, tn)
    return {
        "aggregation": aggregation,
        "method": str(group["method"].iloc[0]),
        "providers": int(group["source"].nunique()),
        "stations": int(group["station_id"].nunique())
        if "station_id" in group
        else None,
        "runs": int(len(group)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "attack_rows": attack_rows,
        "normal_rows": normal_rows,
        "events": events,
        "detected_events": detected,
        **values,
        "event_coverage": (
            float(detected / events) if events else 0.0
        ),
        "average_precision": np.nan,
        "roc_auc": np.nan,
        "estimand": "count_pooled",
        "notes": (
            "TP/FP/FN/TN reconstructed from saved metric definitions; "
            "event coverage is event-count weighted. AP/ROC are not "
            "recomputable from counts."
        ),
    }


def summarize_macro(group: pd.DataFrame) -> dict[str, object]:
    return {
        "aggregation": "provider_seed_macro",
        "method": str(group["method"].iloc[0]),
        "providers": int(group["source"].nunique()),
        "stations": None,
        "runs": int(len(group)),
        "tp": int(group["tp"].sum()),
        "fp": int(group["fp"].sum()),
        "fn": int(group["fn"].sum()),
        "tn": int(group["tn"].sum()),
        "attack_rows": int(group["attack_rows"].sum()),
        "normal_rows": int(group["normal_rows"].sum()),
        "events": int(group["events"].sum()),
        "detected_events": int(group["detected_events"].sum()),
        "precision": float(group["precision"].mean()),
        "recall": float(group["recall"].mean()),
        "f1": float(group["f1"].mean()),
        "false_alarm_rate": float(group["false_alarm_rate"].mean()),
        "event_coverage": float(group["event_coverage"].mean()),
        "average_precision": float(group["average_precision"].mean()),
        "roc_auc": float(group["roc_auc"].mean()),
        "estimand": "macro_provider_seed",
        "notes": (
            "Mean of the 20 provider-seed metrics; retained for comparison "
            "with the frozen Stage349 summary. It is not a count-pooled "
            "estimate and is not a provider-population estimate."
        ),
    }


def build_aggregation_levels(
    detail_counts: pd.DataFrame,
    station: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in detail_counts.groupby("method", sort=True):
        rows.append(summarize_macro(group))
        rows.append(summarize_counts(group, "pooled_counts"))

        provider_rows: list[dict[str, object]] = []
        for _, provider_group in group.groupby("source", sort=True):
            provider_rows.append(
                summarize_counts(provider_group, "provider_counts")
            )
        provider_frame = pd.DataFrame(provider_rows)
        rows.append(
            {
                "aggregation": "provider_equal_counts",
                "method": method,
                "providers": int(group["source"].nunique()),
                "stations": None,
                "runs": int(group["runs"].sum())
                if "runs" in group
                else int(len(group)),
                "tp": int(provider_frame["tp"].sum()),
                "fp": int(provider_frame["fp"].sum()),
                "fn": int(provider_frame["fn"].sum()),
                "tn": int(provider_frame["tn"].sum()),
                "attack_rows": int(provider_frame["attack_rows"].sum()),
                "normal_rows": int(provider_frame["normal_rows"].sum()),
                "events": int(provider_frame["events"].sum()),
                "detected_events": int(
                    provider_frame["detected_events"].sum()
                ),
                "precision": float(provider_frame["precision"].mean()),
                "recall": float(provider_frame["recall"].mean()),
                "f1": float(provider_frame["f1"].mean()),
                "false_alarm_rate": float(
                    provider_frame["false_alarm_rate"].mean()
                ),
                "event_coverage": float(
                    provider_frame["event_coverage"].mean()
                ),
                "average_precision": np.nan,
                "roc_auc": np.nan,
                "estimand": "provider_equal_count_metrics",
                "notes": (
                    "Each provider is first summarized from its own counts; "
                    "providers are then equally weighted. AP/ROC are not "
                    "recomputable from counts."
                ),
            }
        )

    station_rows: list[dict[str, object]] = []
    for (method, source, station_id), group in station.groupby(
        ["method", "source", "station_id"], sort=True
    ):
        summary = summarize_counts(
            group.assign(
                source=source,
                station_id=station_id,
                method=method,
            ),
            "station_counts",
        )
        station_rows.append(summary)
    station_frame = pd.DataFrame(station_rows)
    for method, group in station_frame.groupby("method", sort=True):
        rows.append(
            {
                "aggregation": "station_equal_counts",
                "method": method,
                "providers": int(
                    station[
                        station["method"].eq(method)
                    ]["source"].nunique()
                ),
                "stations": int(len(group)),
                "runs": int(group["runs"].sum()),
                "tp": int(group["tp"].sum()),
                "fp": int(group["fp"].sum()),
                "fn": int(group["fn"].sum()),
                "tn": int(group["tn"].sum()),
                "attack_rows": int(group["attack_rows"].sum()),
                "normal_rows": int(group["normal_rows"].sum()),
                "events": int(group["events"].sum()),
                "detected_events": int(group["detected_events"].sum()),
                "precision": float(group["precision"].mean()),
                "recall": float(group["recall"].mean()),
                "f1": float(group["f1"].mean()),
                "false_alarm_rate": float(
                    group["false_alarm_rate"].mean()
                ),
                "event_coverage": float(group["event_coverage"].mean()),
                "average_precision": np.nan,
                "roc_auc": np.nan,
                "estimand": "station_equal_count_metrics",
                "notes": (
                    "Each station is first summarized across seeds from its "
                    "own counts; stations are then equally weighted. AP/ROC "
                    "are not available in the frozen station table."
                ),
            }
        )

    columns = [
        "aggregation",
        "method",
        "estimand",
        "providers",
        "stations",
        "runs",
        "tp",
        "fp",
        "fn",
        "tn",
        "attack_rows",
        "normal_rows",
        "events",
        "detected_events",
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "event_coverage",
        "average_precision",
        "roc_auc",
        "notes",
    ]
    return pd.DataFrame(rows)[columns].sort_values(
        ["method", "aggregation"]
    )


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(DETAIL)
    station = pd.read_csv(STATION)
    detail_counts = reconstruct_detail_counts(detail)
    aggregation = build_aggregation_levels(detail_counts, station)
    detail_counts.to_csv(OUT_DETAIL_COUNTS, index=False, encoding="utf-8-sig")
    aggregation.to_csv(OUT_AGG, index=False, encoding="utf-8-sig")

    old = pd.read_csv(ORIGINAL_AGG)
    old_pooled = (
        old[old["aggregation"].eq("pooled")]
        .set_index("method")["f1"]
        .to_dict()
    )
    corrected_pooled = (
        aggregation[aggregation["aggregation"].eq("pooled_counts")]
        .set_index("method")["f1"]
        .to_dict()
    )
    comparison = []
    for method in sorted(set(old_pooled) & set(corrected_pooled)):
        comparison.append(
            {
                "method": method,
                "old_label_pooled_f1": float(old_pooled[method]),
                "corrected_count_pooled_f1": float(
                    corrected_pooled[method]
                ),
                "difference": float(
                    corrected_pooled[method] - old_pooled[method]
                ),
            }
        )
    payload = {
        "status": "PASS_V6_AGGREGATION_ESTIMAND_CORRECTION",
        "source": "final_v6_detail.csv and final_v6_station.csv",
        "methods": int(detail_counts["method"].nunique()),
        "runs": int(len(detail_counts)),
        "providers": int(detail_counts["source"].nunique()),
        "stations": int(
            station[["source", "station_id"]].drop_duplicates().shape[0]
        ),
        "corrected_estimands": [
            "provider_seed_macro",
            "pooled_counts",
            "provider_equal_counts",
            "station_equal_counts",
        ],
        "old_pooled_vs_corrected_counts": comparison,
        "rules": [
            "The old pooled label was a provider-seed macro mean.",
            "Count-pooled metrics use reconstructed TP/FP/FN/TN from saved metric definitions.",
            "AP and ROC are left unavailable for count-pooled levels.",
            "No model, threshold, attack, or prediction was changed.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# V6 aggregation estimand correction",
        "",
        f"- Status: `{payload['status']}`",
        f"- Methods: `{payload['methods']}`; runs: `{payload['runs']}`; "
        f"providers: `{payload['providers']}`; unique stations: "
        f"`{payload['stations']}`",
        "",
        "## Old pooled label versus corrected count-pooled F1",
        "",
        pd.DataFrame(comparison).to_markdown(index=False),
        "",
        "## Rules",
        "",
        "- `provider_seed_macro` is the old mean-of-runs summary.",
        "- `pooled_counts` is the count-pooled summary.",
        "- `provider_equal_counts` and `station_equal_counts` weight "
        "providers and stations equally after within-unit count aggregation.",
        "- AP/ROC are not available for count-pooled levels.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "methods": payload["methods"],
                "runs": payload["runs"],
                "comparison": comparison,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
