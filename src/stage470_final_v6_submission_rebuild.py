"""Deterministic V6 rebuild for the final submission package.

This script re-runs the frozen V6 evaluation and writes a self-contained
scenario-level table with full row metrics, event metrics, event lengths,
station-level details, and pooled/provider-equal/station-equal summaries.

It does not change the frozen protocol, thresholds, model selection rule,
attack generator, or test labels. The V6 test was already inspected during
exploratory work, so this remains a reproducibility rebuild rather than a new
confirmatory experiment.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage340_v4_route_locked_evaluation as base  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    detector_frame,
    fit_detectors,
    metrics,
    score_components,
)
from stage208_dwd_persistent_repeat_protection import (  # noqa: E402
    event_metrics,
    far_threshold,
)
from stage217_swiss_controlled_attack_evaluation import (  # noqa: E402
    fuse_scores,
    robust_standardize,
)
from stage240_nonstuck_balanced_selection import (  # noqa: E402
    PROTOCOLS,
    make_frame,
)
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_features,
    select_candidate_stage244,
)
from stage248_cross_provider_fair_baseline_audit import (  # noqa: E402
    feature_matrix,
    fit_baselines,
    score_baselines,
)
from stage256_lightweight_strong_baselines import (  # noqa: E402
    fit_lightweight_baselines,
    score_lightweight_baselines,
)
from stage262_security_domain_baselines import (  # noqa: E402
    fit_security_baselines,
    score_security_baselines,
)

FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"
PROTOCOL_PATH = ROOT / "confirmatory_package" / "simulated_attack_v6" / "protocol_freeze.json"
VALID_OFFSET = 1010001
TEST_OFFSET = 1020001
PRIMARY_FAR = 0.0065
F1_FLOOR = 0.95

OUT_DETAIL = TABLES / "final_v6_detail.csv"
OUT_SCENARIO = TABLES / "final_v6_scenario_metrics.csv"
OUT_STATION = TABLES / "final_v6_station.csv"
OUT_SUMMARY = TABLES / "final_v6_method_summary.csv"
OUT_AGGREGATION = TABLES / "final_v6_aggregation_levels.csv"
OUT_SCENARIO_AGG = TABLES / "final_v6_scenario_aggregate.csv"
OUT_SCENARIO_WINS = TABLES / "final_v6_scenario_wins.csv"
OUT_EVENT_LENGTH = TABLES / "final_v6_event_lengths.csv"
OUT_SELECTION = TABLES / "final_v6_selection.csv"
OUT_JSON = TABLES / "final_v6_rebuild.json"
OUT_REPORT = REPORTS / "final_v6_rebuild.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_rows_full(
    frame: pd.DataFrame,
    pred: np.ndarray,
    score: np.ndarray,
    source: str,
    seed: int,
    method: str,
) -> list[dict[str, object]]:
    attack = frame["is_attack"].astype(bool).to_numpy()
    labels = frame["attack_scenario"].astype(str).to_numpy()
    predicted = np.asarray(pred, dtype=bool)
    values = np.asarray(score, dtype=float)
    rows: list[dict[str, object]] = []
    for scenario in sorted(set(labels[attack])):
        mask = attack & (labels == scenario)
        # Evaluate this scenario against normal rows only. Rows belonging to
        # other attack scenarios are excluded so they are not counted as false
        # alarms for this scenario.
        evaluation_mask = (~attack) | mask
        row_metrics = metrics(
            mask[evaluation_mask],
            predicted[evaluation_mask],
            values[evaluation_mask],
        )
        tp = int((predicted[evaluation_mask] & mask[evaluation_mask]).sum())
        fp = int((predicted[evaluation_mask] & ~mask[evaluation_mask]).sum())
        fn = int((~predicted[evaluation_mask] & mask[evaluation_mask]).sum())
        tn = int((~predicted[evaluation_mask] & ~mask[evaluation_mask]).sum())
        event_rows = frame.loc[mask, ["attack_event_id"]].copy()
        event_rows["_pred"] = predicted[mask]
        grouped = event_rows.groupby("attack_event_id", sort=True)["_pred"].any()
        lengths = event_rows.groupby("attack_event_id", sort=True).size()
        rows.append(
            {
                "source": source,
                "seed": int(seed),
                "method": method,
                "scenario": scenario,
                "rows": int(row_metrics["rows"]),
                "attack_rows": int(row_metrics["attack_rows"]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(row_metrics["precision"]),
                "recall": float(row_metrics["recall"]),
                "row_recall": float(row_metrics["recall"]),
                "f1": float(row_metrics["f1"]),
                "false_alarm_rate": float(row_metrics["false_alarm_rate"]),
                "roc_auc": float(row_metrics["roc_auc"]),
                "average_precision": float(row_metrics["average_precision"]),
                "events": int(len(grouped)),
                "detected_events": int(grouped.sum()),
                "event_coverage": float(grouped.mean()) if len(grouped) else 0.0,
                "mean_event_length": float(lengths.mean()) if len(lengths) else 0.0,
                "min_event_length": int(lengths.min()) if len(lengths) else 0,
                "max_event_length": int(lengths.max()) if len(lengths) else 0,
            }
        )
    return rows


def station_rows_full(
    frame: pd.DataFrame,
    pred: np.ndarray,
    source: str,
    seed: int,
    method: str,
) -> list[dict[str, object]]:
    attack = frame["is_attack"].astype(bool).to_numpy()
    observed = frame["observation_present"].astype(bool).to_numpy()
    predicted = np.asarray(pred, dtype=bool)
    rows: list[dict[str, object]] = []
    for station, part in frame.groupby("station_id", sort=True):
        idx = part.index.to_numpy(dtype=int)
        y = attack[idx] & observed[idx]
        p = predicted[idx] & observed[idx]
        tp = int((p & y).sum())
        fp = int((p & ~y).sum())
        fn = int((~p & y).sum())
        tn = int((~p & ~y & observed[idx]).sum())
        event_rows = part.loc[y, ["attack_event_id"]].copy()
        event_rows["_pred"] = p[y]
        grouped = event_rows.groupby("attack_event_id", sort=True)["_pred"].any()
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        far = float(fp / (fp + tn)) if (fp + tn) else 0.0
        rows.append(
            {
                "source": source,
                "seed": int(seed),
                "method": method,
                "station_id": str(station),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "row_recall": recall,
                "f1": f1,
                "false_alarm_rate": far,
                "event_coverage": float(grouped.mean()) if len(grouped) else 0.0,
                "events": int(len(grouped)),
                "detected_events": int(grouped.sum()),
            }
        )
    return rows


def aggregate_levels(
    detail: pd.DataFrame,
    station: pd.DataFrame,
    baselines: list[str],
) -> pd.DataFrame:
    metrics_columns = [
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "roc_auc",
        "average_precision",
        "event_coverage",
        "row_recall",
    ]
    pooled = (
        detail.groupby("method", as_index=False)[metrics_columns]
        .mean()
        .assign(aggregation="pooled")
    )
    provider = (
        detail.groupby(["method", "source"], as_index=False)[metrics_columns]
        .mean()
        .groupby("method", as_index=False)[metrics_columns]
        .mean()
        .assign(aggregation="provider_equal")
    )
    station_metrics = [
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "event_coverage",
        "row_recall",
    ]
    station_equal = (
        station.groupby("method", as_index=False)[station_metrics]
        .mean()
        .assign(aggregation="station_equal")
    )
    for column in ["roc_auc", "average_precision"]:
        station_equal[column] = np.nan
    combined = pd.concat([pooled, provider, station_equal], ignore_index=True, sort=False)
    ordered = [
        "aggregation",
        "method",
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "roc_auc",
        "average_precision",
        "event_coverage",
        "row_recall",
    ]
    return combined[ordered]


def main() -> int:
    started = time.perf_counter()
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    frozen = load_json(PROTOCOL_PATH)
    scenarios = {str(row["name"]): row for row in frozen["attack_scenarios"]}
    detail_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    selection_counts: dict[str, int] = {}

    for source, protocol_path in PROTOCOLS.items():
        print(json.dumps({"event": "provider_start", "source": source}), flush=True)
        protocol = load_json(protocol_path)
        frame = detector_frame(make_frame(source, protocol))
        detectors = fit_detectors(frame, protocol)
        train_normal = (
            frame["split"].eq("train") & frame["observation_present"]
        ).to_numpy(dtype=bool)
        components = score_components(frame, detectors)
        fused, _ = fuse_scores(frame, components)
        base_z = robust_standardize(fused, train_normal).to_numpy(dtype=float)
        base_features = candidate_features(frame, train_normal)
        matrix, _ = feature_matrix(components, base_features, train_normal)
        old = fit_baselines(frame, matrix, train_normal)
        light, _ = fit_lightweight_baselines(frame, matrix, train_normal)
        security, _ = fit_security_baselines(frame, matrix, train_normal)

        for seed in base.SEEDS:
            valid_blocks = base.simbase.make_simulated_blocks(
                frame, frozen, seed, "valid", VALID_OFFSET
            )
            test_blocks = base.simbase.make_simulated_blocks(
                frame, frozen, seed, "test", TEST_OFFSET
            )
            valid = base.simbase.inject_simulated_attack(frame, valid_blocks, scenarios)
            test = base.simbase.inject_simulated_attack(frame, test_blocks, scenarios)
            valid_components = score_components(valid, detectors)
            test_components = score_components(test, detectors)
            valid_fused, _ = fuse_scores(valid, valid_components)
            test_fused, _ = fuse_scores(test, test_components)
            valid_z = robust_standardize(valid_fused, train_normal).to_numpy(dtype=float)
            test_z = robust_standardize(test_fused, train_normal).to_numpy(dtype=float)
            valid_features = candidate_features(valid, train_normal)
            test_features = candidate_features(test, train_normal)
            valid_matrix, _ = feature_matrix(valid_components, valid_features, train_normal)
            test_matrix, _ = feature_matrix(test_components, test_features, train_normal)
            valid_mask = valid["split"].eq("valid").to_numpy(dtype=bool)
            test_mask = test["split"].eq("test").to_numpy(dtype=bool)
            valid_frame = valid.loc[valid_mask].reset_index(drop=True)
            test_frame = test.loc[test_mask].reset_index(drop=True)
            normal = (
                valid_frame["observation_present"] & ~valid_frame["is_attack"]
            ).to_numpy(dtype=bool)
            valid_values = {"trust_valid_f1": valid_z[valid_mask]}
            test_values = {"trust_valid_f1": test_z[test_mask]}
            selected = select_candidate_stage244(
                "valid_f1",
                valid_z[valid_mask],
                {key: value[valid_mask] for key, value in valid_features.items()},
                valid_frame,
                normal,
                PRIMARY_FAR,
            )
            key = (str(selected["feature"]), int(selected["window"]))
            valid_values["trust_valid_f1"] = (
                (1.0 - float(selected["weight"])) * valid_z[valid_mask]
                + float(selected["weight"]) * valid_features[key][valid_mask]
            )
            test_values["trust_valid_f1"] = (
                (1.0 - float(selected["weight"])) * test_z[test_mask]
                + float(selected["weight"]) * test_features[key][test_mask]
            )
            valid_values.update(score_baselines(old, valid_matrix[valid_mask]))
            test_values.update(score_baselines(old, test_matrix[test_mask]))
            valid_values.update(
                score_lightweight_baselines(light, valid_matrix[valid_mask])
            )
            test_values.update(
                score_lightweight_baselines(light, test_matrix[test_mask])
            )
            valid_values.update(
                score_security_baselines(security, valid_frame, valid_matrix[valid_mask])
            )
            test_values.update(
                score_security_baselines(security, test_frame, test_matrix[test_mask])
            )
            normalized_valid: dict[str, np.ndarray] = {}
            normalized_test: dict[str, np.ndarray] = {}
            for method in valid_values:
                normalized_valid[method], normalized_test[method] = base.norm_pair(
                    np.asarray(valid_values[method]),
                    np.asarray(test_values[method]),
                    normal,
                )
            normalized_valid["route_knn_arima"] = base.route(
                normalized_valid,
                {key: value[valid_mask] for key, value in valid_features.items()},
                {key: value[valid_mask] for key, value in valid_features.items()},
                normal,
            )
            normalized_test["route_knn_arima"] = base.route(
                normalized_test,
                {key: value[test_mask] for key, value in test_features.items()},
                {key: value[valid_mask] for key, value in valid_features.items()},
                normal,
                True,
            )
            union_valid = np.zeros(len(normal), dtype=bool)
            union_test = np.zeros(len(test_frame), dtype=bool)
            for method, budget in (
                ("trust_valid_f1", 0.50 * PRIMARY_FAR),
                ("ewma_deviation", 0.25 * PRIMARY_FAR),
                ("arima_innovation", 0.25 * PRIMARY_FAR),
            ):
                threshold = far_threshold(normalized_valid[method], normal, budget)
                union_valid |= normalized_valid[method] >= threshold
                union_test |= normalized_test[method] >= threshold
            normalized_valid["trust_shape_union"] = np.maximum.reduce(
                [
                    normalized_valid["trust_valid_f1"],
                    normalized_valid["ewma_deviation"],
                    normalized_valid["arima_innovation"],
                ]
            )
            normalized_test["trust_shape_union"] = np.maximum.reduce(
                [
                    normalized_test["trust_valid_f1"],
                    normalized_test["ewma_deviation"],
                    normalized_test["arima_innovation"],
                ]
            )
            y_valid = valid_frame["is_attack"].astype(bool).to_numpy()
            y_test = test_frame["is_attack"].astype(bool).to_numpy()
            observed_valid = valid_frame["observation_present"].to_numpy(dtype=bool)
            observed_test = test_frame["observation_present"].to_numpy(dtype=bool)
            valid_predictions: dict[str, np.ndarray] = {}
            test_predictions: dict[str, np.ndarray] = {}
            valid_metrics: dict[str, dict[str, float]] = {}
            for method in ["trust_only", "route_knn_arima"]:
                score_method = "trust_valid_f1" if method == "trust_only" else method
                threshold = far_threshold(normalized_valid[score_method], normal, PRIMARY_FAR)
                valid_predictions[method] = observed_valid & (
                    normalized_valid[score_method] >= threshold
                )
                test_predictions[method] = observed_test & (
                    normalized_test[score_method] >= threshold
                )
                valid_metrics[method] = metrics(
                    y_valid,
                    valid_predictions[method],
                    normalized_valid[score_method],
                )
            valid_predictions["trust_shape_union"] = union_valid & observed_valid
            test_predictions["trust_shape_union"] = union_test & observed_test
            valid_metrics["trust_shape_union"] = metrics(
                y_valid,
                valid_predictions["trust_shape_union"],
                normalized_valid["trust_shape_union"],
            )
            trust_f1 = float(valid_metrics["trust_only"]["f1"])
            eligible = [
                method
                for method in base.CANDIDATES
                if valid_metrics[method]["f1"] >= F1_FLOOR * trust_f1
            ]
            chosen = (
                max(
                    eligible,
                    key=lambda method: (
                        event_metrics(valid_frame, valid_predictions[method])[
                            "event_coverage"
                        ],
                        valid_metrics[method]["f1"],
                    ),
                )
                if eligible
                else "trust_only"
            )
            selection_counts[chosen] = selection_counts.get(chosen, 0) + 1
            selection_rows.append(
                {
                    "source": source,
                    "seed": int(seed),
                    "chosen": chosen,
                    "trust_only_valid_f1": float(valid_metrics["trust_only"]["f1"]),
                    "route_valid_f1": float(valid_metrics["route_knn_arima"]["f1"]),
                    "union_valid_f1": float(valid_metrics["trust_shape_union"]["f1"]),
                    "trust_only_valid_event_coverage": float(
                        event_metrics(valid_frame, valid_predictions["trust_only"])[
                            "event_coverage"
                        ]
                    ),
                    "route_valid_event_coverage": float(
                        event_metrics(valid_frame, valid_predictions["route_knn_arima"])[
                            "event_coverage"
                        ]
                    ),
                    "union_valid_event_coverage": float(
                        event_metrics(valid_frame, valid_predictions["trust_shape_union"])[
                            "event_coverage"
                        ]
                    ),
                }
            )
            for method in base.CANDIDATES:
                predicted = test_predictions[method]
                score = (
                    normalized_test["trust_valid_f1"]
                    if method == "trust_only"
                    else normalized_test[method]
                )
                row_metrics = metrics(y_test, predicted, score)
                event_values = event_metrics(test_frame, predicted)
                detail_rows.append(
                    {
                        "source": source,
                        "seed": int(seed),
                        "method": method,
                        "selected": bool(method == chosen),
                        "threshold": float(
                            far_threshold(
                                normalized_valid[
                                    "trust_valid_f1"
                                    if method == "trust_only"
                                    else method
                                ],
                                normal,
                                PRIMARY_FAR,
                            )
                        ),
                        **row_metrics,
                        **event_values,
                    }
                )
                scenario_rows.extend(
                    scenario_rows_full(
                        test_frame,
                        predicted,
                        score,
                        source,
                        int(seed),
                        method,
                    )
                )
                station_rows.extend(
                    station_rows_full(
                        test_frame,
                        predicted,
                        source,
                        int(seed),
                        method,
                    )
                )
            for method in base.BASELINES:
                threshold = far_threshold(normalized_valid[method], normal, PRIMARY_FAR)
                predicted = observed_test & (normalized_test[method] >= threshold)
                row_metrics = metrics(y_test, predicted, normalized_test[method])
                event_values = event_metrics(test_frame, predicted)
                detail_rows.append(
                    {
                        "source": source,
                        "seed": int(seed),
                        "method": method,
                        "selected": False,
                        "threshold": float(threshold),
                        **row_metrics,
                        **event_values,
                    }
                )
                scenario_rows.extend(
                    scenario_rows_full(
                        test_frame,
                        predicted,
                        normalized_test[method],
                        source,
                        int(seed),
                        method,
                    )
                )
                station_rows.extend(
                    station_rows_full(
                        test_frame,
                        predicted,
                        source,
                        int(seed),
                        method,
                    )
                )
            print(
                json.dumps(
                    {
                        "event": "seed_complete",
                        "source": source,
                        "seed": int(seed),
                    }
                ),
                flush=True,
            )

    detail = pd.DataFrame(detail_rows)
    scenario = pd.DataFrame(scenario_rows)
    station = pd.DataFrame(station_rows)
    selection = pd.DataFrame(selection_rows)
    metric_columns = [
        "precision",
        "recall",
        "f1",
        "false_alarm_rate",
        "roc_auc",
        "average_precision",
        "event_coverage",
        "row_recall",
    ]
    summary = (
        detail.groupby("method", as_index=False)[metric_columns]
        .mean()
        .sort_values("method")
    )
    aggregation = aggregate_levels(detail, station, base.BASELINES)
    scenario_group_rows: list[dict[str, object]] = []
    for (scenario_name, method), group in scenario.groupby(
        ["scenario", "method"], sort=True
    ):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())
        events = int(group["events"].sum())
        detected_events = int(group["detected_events"].sum())
        event_coverage_mean = float(group["event_coverage"].mean())
        event_coverage_weighted = (
            float(detected_events / events) if events else 0.0
        )
        row_recall_mean = float(group["row_recall"].mean())
        scenario_group_rows.append(
            {
                "scenario": scenario_name,
                "method": method,
                "runs": int(len(group)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                "row_recall": row_recall_mean,
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
                "event_coverage": event_coverage_mean,
                "event_coverage_weighted": event_coverage_weighted,
                "mean_event_length": float(group["mean_event_length"].mean()),
                "min_event_length": int(group["min_event_length"].min()),
                "max_event_length": int(group["max_event_length"].max()),
            }
        )
    scenario_aggregate = pd.DataFrame(scenario_group_rows)
    scenario_wins: list[dict[str, object]] = []
    for scenario_name, group in scenario_aggregate.groupby("scenario", sort=True):
        candidate = group[group["method"].eq("trust_shape_union")]
        if candidate.empty:
            continue
        candidate_value = float(candidate["event_coverage"].iloc[0])
        wins = 0
        for baseline in base.BASELINES:
            baseline_row = group[group["method"].eq(baseline)]
            if baseline_row.empty:
                continue
            if candidate_value > float(baseline_row["event_coverage"].iloc[0]):
                wins += 1
        scenario_wins.append(
            {
                "scenario": scenario_name,
                "candidate": "trust_shape_union",
                "candidate_event_coverage": candidate_value,
                "wins": wins,
                "total_baselines": len(base.BASELINES),
                "win_share": wins / len(base.BASELINES),
                "passes_single_scenario_70pct": wins / len(base.BASELINES) >= 0.70,
            }
        )
    scenario_wins_df = pd.DataFrame(scenario_wins)
    event_lengths = (
        scenario_aggregate.groupby(["scenario", "method"], as_index=False)[
            ["mean_event_length", "events"]
        ]
        .mean()
        .sort_values(["scenario", "method"])
    )

    detail.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
    scenario.to_csv(OUT_SCENARIO, index=False, encoding="utf-8-sig")
    station.to_csv(OUT_STATION, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    aggregation.to_csv(OUT_AGGREGATION, index=False, encoding="utf-8-sig")
    scenario_aggregate.to_csv(OUT_SCENARIO_AGG, index=False, encoding="utf-8-sig")
    scenario_wins_df.to_csv(OUT_SCENARIO_WINS, index=False, encoding="utf-8-sig")
    event_lengths.to_csv(OUT_EVENT_LENGTH, index=False, encoding="utf-8-sig")
    selection.to_csv(OUT_SELECTION, index=False, encoding="utf-8-sig")

    payload = {
        "status": "PASS_V6_SUBMISSION_REBUILD_DETERMINISTIC",
        "protocol_id": frozen["protocol_id"],
        "protocol_sha256": base.simbase.sha256_file(PROTOCOL_PATH)
        if hasattr(base.simbase, "sha256_file")
        else "",
        "providers": list(PROTOCOLS),
        "seeds": list(base.SEEDS),
        "scenario_count": len(scenarios),
        "method_count": int(len(summary)),
        "selection_counts": selection_counts,
        "selected_run_summary": detail.loc[detail["selected"], metric_columns].mean().to_dict(),
        "candidate_summary": summary[
            summary["method"].eq("trust_shape_union")
        ].to_dict("records"),
        "scenario_win_share": (
            float(scenario_wins_df["wins"].sum())
            / float(scenario_wins_df["total_baselines"].sum())
            if not scenario_wins_df.empty
            else 0.0
        ),
        "limitations": [
            "V6 test has already been inspected; this is a reproducibility rebuild, not a new confirmatory test.",
            "All attacks are transparent simulated injections on real observation backgrounds.",
            "The four providers do not support provider-independent or station-independent claims.",
            "Per-scenario metrics use that scenario's attack rows as positives and all non-attack test rows as negatives.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Final V6 submission rebuild",
        "",
        f"- Status: `{payload['status']}`",
        f"- Protocol: `{payload['protocol_id']}`",
        f"- Providers: {', '.join(payload['providers'])}",
        f"- Seeds: {len(payload['seeds'])}; scenarios: {payload['scenario_count']}; methods: {payload['method_count']}",
        f"- Selection counts: `{selection_counts}`",
        f"- Selected-run F1: `{payload['selected_run_summary']['f1']:.6f}`",
        f"- Selected-run event coverage: `{payload['selected_run_summary']['event_coverage']:.6f}`",
        f"- Selected-run FAR: `{payload['selected_run_summary']['false_alarm_rate']:.6f}`",
        "",
        "## Scenario win share",
        "",
        scenario_wins_df.to_markdown(index=False),
        "",
        "## Boundary",
        "",
        "- This rebuild does not change model selection, thresholds, attack generation, or labels.",
        "- V6 test has already been inspected; the result remains bounded and is not upgraded to a new confirmatory claim.",
        "- The full scenario metrics are point estimates; the main station-cluster uncertainty audit remains in Stage351.",
        "",
        f"- Elapsed seconds: `{time.perf_counter() - started:.2f}`",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection_counts": selection_counts,
                "selected_run_summary": payload["selected_run_summary"],
                "scenario_win_share": payload["scenario_win_share"],
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
