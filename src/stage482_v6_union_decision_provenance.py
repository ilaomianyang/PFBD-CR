"""Stage 482: persist the actual V6 union decision rule.

V6 uses an OR of three component thresholds with a 50/25/25 FAR budget.  The
frozen detail table stores only a scalar max-score threshold, which is not the
executed rule.  This stage recomputes and saves the three thresholds and the
resulting test operating point without changing the frozen predictions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
)
from stage262_security_domain_baselines import (  # noqa: E402
    _apply_reference,
    _ar_scores,
    _ewma_scores,
    _fit_ar_coefficients,
    _fit_reference,
)

FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"
PROTOCOL_PATH = (
    ROOT
    / "confirmatory_package"
    / "simulated_attack_v6"
    / "protocol_freeze.json"
)
OUT_CSV = TABLES / "final_v6_union_decision_provenance.csv"
OUT_JSON = REPORTS / "final_v6_union_decision_provenance.json"
OUT_MD = REPORTS / "final_v6_union_decision_provenance.md"
DETAIL = TABLES / "final_v6_detail.csv"
OUT_SCORES = TABLES / "stage490_union_scores.parquet"

VALID_OFFSET = 1010001
TEST_OFFSET = 1020001
PRIMARY_FAR = 0.0065
COMPONENTS = [
    ("trust_valid_f1", 0.50),
    ("ewma_deviation", 0.25),
    ("arima_innovation", 0.25),
]


def main() -> int:
    started = time.perf_counter()
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    scenarios = {
        str(row["name"]): row for row in frozen["attack_scenarios"]
    }
    rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    for source, protocol_path in PROTOCOLS.items():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        frame = detector_frame(make_frame(source, protocol))
        detectors = fit_detectors(frame, protocol)
        train_normal = (
            frame["split"].eq("train") & frame["observation_present"]
        ).to_numpy(dtype=bool)
        components = score_components(frame, detectors)
        fused, _ = fuse_scores(frame, components)
        base_z = robust_standardize(fused, train_normal).to_numpy(
            dtype=float
        )
        base_features = candidate_features(frame, train_normal)
        matrix, _ = feature_matrix(
            components,
            base_features,
            train_normal,
        )
        ar_coefficients = _fit_ar_coefficients(frame, train_normal)
        raw_train = {
            "ewma_deviation": _ewma_scores(frame),
            "arima_innovation": _ar_scores(frame, ar_coefficients),
        }
        security_references = {
            method: _fit_reference(values, train_normal)
            for method, values in raw_train.items()
        }
        for seed in base.SEEDS:
            valid_blocks = base.simbase.make_simulated_blocks(
                frame,
                frozen,
                seed,
                "valid",
                VALID_OFFSET,
            )
            test_blocks = base.simbase.make_simulated_blocks(
                frame,
                frozen,
                seed,
                "test",
                TEST_OFFSET,
            )
            valid = base.simbase.inject_simulated_attack(
                frame,
                valid_blocks,
                scenarios,
            )
            test = base.simbase.inject_simulated_attack(
                frame,
                test_blocks,
                scenarios,
            )
            valid_components = score_components(valid, detectors)
            test_components = score_components(test, detectors)
            valid_fused, _ = fuse_scores(valid, valid_components)
            test_fused, _ = fuse_scores(test, test_components)
            valid_z = robust_standardize(
                valid_fused,
                train_normal,
            ).to_numpy(dtype=float)
            test_z = robust_standardize(
                test_fused,
                train_normal,
            ).to_numpy(dtype=float)
            valid_features = candidate_features(valid, train_normal)
            test_features = candidate_features(test, train_normal)
            valid_matrix, _ = feature_matrix(
                valid_components,
                valid_features,
                train_normal,
            )
            test_matrix, _ = feature_matrix(
                test_components,
                test_features,
                train_normal,
            )
            valid_mask = valid["split"].eq("valid").to_numpy(dtype=bool)
            test_mask = test["split"].eq("test").to_numpy(dtype=bool)
            valid_frame = valid.loc[valid_mask].reset_index(drop=True)
            test_frame = test.loc[test_mask].reset_index(drop=True)
            normal = (
                valid_frame["observation_present"]
                & ~valid_frame["is_attack"]
            ).to_numpy(dtype=bool)
            selected = select_candidate_stage244(
                "valid_f1",
                valid_z[valid_mask],
                {
                    key: values[valid_mask]
                    for key, values in valid_features.items()
                },
                valid_frame,
                normal,
                PRIMARY_FAR,
            )
            selected_key = (
                str(selected["feature"]),
                int(selected["window"]),
            )
            selected_weight = float(selected["weight"])
            valid_trust = (
                (1.0 - selected_weight) * valid_z[valid_mask]
                + selected_weight
                * valid_features[selected_key][valid_mask]
            )
            test_trust = (
                (1.0 - selected_weight) * test_z[test_mask]
                + selected_weight
                * test_features[selected_key][test_mask]
            )

            def security_scores(
                current_frame: pd.DataFrame,
            ) -> dict[str, np.ndarray]:
                raw = {
                    "ewma_deviation": _ewma_scores(current_frame),
                    "arima_innovation": _ar_scores(
                        current_frame,
                        ar_coefficients,
                    ),
                }
                return {
                    method: _apply_reference(
                        values,
                        float(security_references[method][0]),
                        float(security_references[method][1]),
                    )
                    for method, values in raw.items()
                }

            valid_scores = {
                "trust_valid_f1": valid_trust,
                **security_scores(valid_frame),
            }
            test_scores = {
                "trust_valid_f1": test_trust,
                **security_scores(test_frame),
            }
            normalized_valid: dict[str, np.ndarray] = {}
            normalized_test: dict[str, np.ndarray] = {}
            for method in ["trust_valid_f1", "ewma_deviation", "arima_innovation"]:
                normalized_valid[method], normalized_test[method] = (
                    base.norm_pair(
                        np.asarray(valid_scores[method]),
                        np.asarray(test_scores[method]),
                        normal,
                    )
                )
            union_valid = np.zeros(len(valid_frame), dtype=bool)
            union_test = np.zeros(len(test_frame), dtype=bool)
            component_rows: list[dict[str, object]] = []
            for method, budget_fraction in COMPONENTS:
                budget = float(PRIMARY_FAR) * float(budget_fraction)
                threshold = float(
                    far_threshold(
                        normalized_valid[method],
                        normal,
                        budget,
                    )
                )
                valid_alarm = normalized_valid[method] >= threshold
                test_alarm = normalized_test[method] >= threshold
                union_valid |= valid_alarm
                union_test |= test_alarm
                component_rows.append(
                    {
                        "component": method,
                        "budget_fraction": float(budget_fraction),
                        "valid_far_budget": budget,
                        "threshold": threshold,
                        "valid_measured_far": float(
                            valid_alarm[normal].mean()
                            if normal.any()
                            else 0.0
                        ),
                        "test_measured_far": float(
                            test_alarm[
                                test_frame["observation_present"].to_numpy(
                                    dtype=bool
                                )
                                & ~test_frame["is_attack"].to_numpy(
                                    dtype=bool
                                )
                            ].mean()
                            if (
                                test_frame["observation_present"]
                                & ~test_frame["is_attack"]
                            ).any()
                            else 0.0
                        ),
                    }
                )
            observed_valid = valid_frame["observation_present"].to_numpy(
                dtype=bool
            )
            observed_test = test_frame["observation_present"].to_numpy(
                dtype=bool
            )
            y_valid = valid_frame["is_attack"].astype(bool).to_numpy()
            y_test = test_frame["is_attack"].astype(bool).to_numpy()
            union_valid = union_valid & observed_valid
            union_test = union_test & observed_test
            valid_metrics = metrics(
                y_valid,
                union_valid,
                normalized_valid["trust_valid_f1"],
            )
            test_metrics = metrics(
                y_test,
                union_test,
                normalized_test["trust_valid_f1"],
            )
            test_event = event_metrics(test_frame, union_test)
            union_shape_valid = np.maximum.reduce(
                [
                    normalized_valid["trust_valid_f1"],
                    normalized_valid["ewma_deviation"],
                    normalized_valid["arima_innovation"],
                ]
            )
            union_shape_test = np.maximum.reduce(
                [
                    normalized_test["trust_valid_f1"],
                    normalized_test["ewma_deviation"],
                    normalized_test["arima_innovation"],
                ]
            )
            score_frames.append(
                pd.DataFrame(
                    {
                        "source": source,
                        "seed": int(seed),
                        "split": "valid",
                        "row_index": np.arange(len(valid_frame)),
                        "station_id": valid_frame["station_id"].astype(str),
                        "attack_scenario": valid_frame[
                            "attack_scenario"
                        ].astype(str),
                        "attack_event_id": valid_frame[
                            "attack_event_id"
                        ].astype(str),
                        "is_attack": valid_frame["is_attack"].astype(bool),
                        "observation_present": valid_frame[
                            "observation_present"
                        ].astype(bool),
                        "union_score": np.asarray(
                            union_shape_valid,
                            dtype=float,
                        ),
                        "union_prediction": np.asarray(
                            union_valid,
                            dtype=bool,
                        ),
                    }
                )
            )
            score_frames.append(
                pd.DataFrame(
                    {
                        "source": source,
                        "seed": int(seed),
                        "split": "test",
                        "row_index": np.arange(len(test_frame)),
                        "station_id": test_frame["station_id"].astype(str),
                        "attack_scenario": test_frame[
                            "attack_scenario"
                        ].astype(str),
                        "attack_event_id": test_frame[
                            "attack_event_id"
                        ].astype(str),
                        "is_attack": test_frame["is_attack"].astype(bool),
                        "observation_present": test_frame[
                            "observation_present"
                        ].astype(bool),
                        "union_score": np.asarray(
                            union_shape_test,
                            dtype=float,
                        ),
                        "union_prediction": np.asarray(
                            union_test,
                            dtype=bool,
                        ),
                    }
                )
            )
            for component in component_rows:
                rows.append(
                    {
                        "source": source,
                        "seed": int(seed),
                        **component,
                    }
                )
            rows.append(
                {
                    "source": source,
                    "seed": int(seed),
                    "component": "union_or",
                    "budget_fraction": 1.0,
                    "valid_far_budget": float(PRIMARY_FAR),
                    "threshold": None,
                    "valid_measured_far": float(
                        union_valid[observed_valid & ~y_valid].mean()
                        if (observed_valid & ~y_valid).any()
                        else 0.0
                    ),
                    "test_measured_far": float(
                        test_metrics["false_alarm_rate"]
                    ),
                    "test_precision": float(test_metrics["precision"]),
                    "test_recall": float(test_metrics["recall"]),
                    "test_f1": float(test_metrics["f1"]),
                    "test_event_coverage": float(
                        test_event["event_coverage"]
                    ),
                }
            )
    provenance = pd.DataFrame(rows)
    provenance.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    pd.concat(score_frames, ignore_index=True).to_parquet(
        OUT_SCORES,
        index=False,
    )
    union_rows = provenance[provenance["component"].eq("union_or")]
    expected = pd.read_csv(DETAIL)
    expected_union = expected[expected["method"].eq("trust_shape_union")]
    expected_f1 = float(expected_union["f1"].mean())
    expected_coverage = float(expected_union["event_coverage"].mean())
    observed_f1 = float(union_rows["test_f1"].mean())
    observed_coverage = float(
        union_rows["test_event_coverage"].mean()
    )
    reproduction_ok = (
        abs(observed_f1 - expected_f1) <= 1e-8
        and abs(observed_coverage - expected_coverage) <= 1e-8
    )
    payload = {
        "status": (
            "PASS_V6_UNION_DECISION_PROVENANCE"
            if reproduction_ok
            else "FAIL_V6_UNION_DECISION_PROVENANCE_REPRODUCTION_MISMATCH"
        ),
        "protocol_id": frozen["protocol_id"],
        "primary_far_budget": float(PRIMARY_FAR),
        "rule": (
            "OR of trust_valid_f1, ewma_deviation, and arima_innovation "
            "thresholds with 50/25/25 percent of the primary FAR budget."
        ),
        "rows": int(len(provenance)),
        "provider_seed_runs": int(len(union_rows)),
        "union_test_far_mean": float(
            union_rows["test_measured_far"].mean()
        ),
        "union_test_f1_mean": observed_f1,
        "union_test_event_coverage_mean": observed_coverage,
        "expected_frozen_f1": expected_f1,
        "expected_frozen_event_coverage": expected_coverage,
        "reproduction_ok": reproduction_ok,
        "component_threshold_summary": (
            provenance[~provenance["component"].eq("union_or")]
            .groupby("component", as_index=False)["threshold"]
            .agg(["min", "mean", "max"])
            .reset_index()
            .to_dict("records")
        ),
        "limitations": [
            "This stage recomputes the saved decision rule; it does not "
            "replace the frozen detail table.",
            "The scalar threshold in the old detail table is not the OR rule.",
            "The V6 test was already inspected; this remains a "
            "reproducibility/provenance analysis.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# V6 union decision-rule provenance",
        "",
        f"- Status: `{payload['status']}`",
        f"- Protocol: `{payload['protocol_id']}`",
        f"- Rule: {payload['rule']}",
        f"- Provider-seed runs: `{payload['provider_seed_runs']}`",
        f"- Union test FAR mean: `{payload['union_test_far_mean']:.6f}`",
        f"- Union test F1 mean: `{payload['union_test_f1_mean']:.6f}`",
        f"- Union test event coverage mean: "
        f"`{payload['union_test_event_coverage_mean']:.6f}`",
        "",
        "## Component threshold summary",
        "",
        pd.DataFrame(payload["component_threshold_summary"]).to_markdown(
            index=False
        ),
        "",
        "## Boundary",
        "",
        "- The executed rule is the component-threshold OR, not the old "
        "scalar max-score threshold.",
        "- This is a provenance reconstruction on the already inspected V6 "
        "test; it is not a new confirmatory test.",
        "",
        f"- Elapsed seconds: `{time.perf_counter() - started:.2f}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "runs": payload["provider_seed_runs"],
                "far": payload["union_test_far_mean"],
                "f1": payload["union_test_f1_mean"],
                "coverage": payload["union_test_event_coverage_mean"],
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
