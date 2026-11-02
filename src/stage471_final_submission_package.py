"""Assemble the final TrustMeteo-QC submission tables from frozen artifacts.

This stage is read-only with respect to experimental evidence. It copies and
normalizes already audited numbers into the final package, adds explicit
implementation/evidence labels, and records the negative boundaries that must
remain visible in the manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FINAL = ROOT / "submission_final_20261009"
FINAL_TABLES = FINAL / "tables"
FINAL_REPORTS = FINAL / "reports"


def load_json(name: str) -> dict[str, Any]:
    path = TABLES / name
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_operating_points() -> list[dict[str, Any]]:
    payload = load_json("stage42_multiseed_operating_curve.json")
    rows: list[dict[str, Any]] = []
    for row in payload["aggregate"]:
        rows.append(
            {
                "scope": "historical_stage42_multiseed_sensitivity",
                "operating_point": row["operating_point"],
                "target_valid_far": row.get("target_valid_far"),
                "test_f1_mean": row["test_f1_mean"],
                "test_f1_std": row["test_f1_std"],
                "test_recall_mean": row["test_recall_mean"],
                "test_precision_mean": row["test_precision_mean"],
                "test_auprc_mean": row["test_auprc_mean"],
                "test_far_mean": row["test_far_mean"],
                "note": (
                    "Historical multi-seed sensitivity; it does not replace the "
                    "V6 primary operating point and is not a test-selected point."
                ),
            }
        )
    return rows


def build_resource_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    resource_payload = load_json("stage290_same_contract_baseline_resource_budget.json")
    trust_payload = load_json("stage301_trust_resource_budget.json")
    testbed_payload = load_json("stage419_testbed_resource_only_audit.json")
    online_payload = load_json("stage255_online_causal_testbed.json")

    baseline_rows: list[dict[str, Any]] = []
    for row in resource_payload["methods"]:
        latency = row.get("score_latency") or {}
        baseline_rows.append(
            {
                "method": row["method"],
                "family": row["family"],
                "implementation": row["implementation"],
                "fit_rows": row.get("fit_rows"),
                "parameter_count": row.get("parameter_count"),
                "fit_seconds": row.get("fit_seconds"),
                "peak_working_set_mb": row.get("peak_working_set_mb"),
                "score_mean_ms": latency.get("mean_ms"),
                "score_p95_ms": latency.get("p95_ms"),
                "status": row.get("status"),
                "failure_state": row.get("failure_state"),
                "scope": "same_contract_train_normal_matrix_only",
            }
        )

    trust_measurement = trust_payload["measurement"]
    selection_latencies = [
        row["end_to_end_latency"]["mean_ms"] for row in trust_measurement["selection"]
    ]
    single_station_latencies = [
        row["single_station_latency"]["mean_ms"]
        for row in trust_measurement["selection"]
    ]
    trust_rows = [
        {
            "method": "trust_frozen_rule",
            "source": trust_payload["source"],
            "fit_rows": trust_measurement["fit_rows"],
            "rows_scored": trust_measurement["rows_scored"],
            "feature_count": trust_measurement["feature_count"],
            "configuration_parameter_count": trust_measurement[
                "configuration_parameter_count"
            ],
            "configuration_parameter_scope": trust_measurement[
                "configuration_parameter_scope"
            ],
            "mean_end_to_end_latency_ms": sum(selection_latencies)
            / len(selection_latencies),
            "mean_single_station_latency_ms": sum(single_station_latencies)
            / len(single_station_latencies),
            "note": "Configuration parameter count is not a neural-network parameter count.",
        }
    ]

    testbed_rows: list[dict[str, Any]] = []
    for row in testbed_payload["runs"]:
        testbed_rows.append(
            {
                "seed": row["seed"],
                "frames": row["frames"],
                "alerts": row["alerts"],
                "batch_mean_ms": row["batch_elapsed_ms"]["mean_ms"],
                "batch_p95_ms": row["batch_elapsed_ms"]["p95_ms"],
                "peak_working_set_mb": row["peak_working_set_mb"],
                "attack_logs_read": row["attack_logs_read"],
                "test_labels_read": row["test_labels_read"],
            }
        )
    testbed_rows.append(
        {
            "seed": "aggregate",
            "frames": "see_runs",
            "alerts": "see_runs",
            "batch_mean_ms": sum(row["batch_mean_ms"] for row in testbed_rows)
            / len(testbed_rows),
            "batch_p95_ms": sum(row["batch_p95_ms"] for row in testbed_rows)
            / len(testbed_rows),
            "peak_working_set_mb": max(
                row["peak_working_set_mb"] for row in testbed_rows
            ),
            "attack_logs_read": False,
            "test_labels_read": False,
        }
    )
    return baseline_rows, trust_rows, testbed_rows


def build_external_transfer() -> list[dict[str, Any]]:
    fmi = load_json("stage259_fmi_locked_transfer.json")
    smhi = load_json("stage367_smhi_station_holdout_locked_transfer.json")
    smhi_factor = load_json("stage373_smhi_factor_holdout_audit.json")
    smhi_third = load_json("stage381_smhi_third_station_audit.json")

    rows: list[dict[str, Any]] = []
    fmi_rows = pd.DataFrame(fmi["provider_equal"])
    trust_fmi = fmi_rows[fmi_rows["method"].eq("trust_valid_f1")].iloc[0]
    best_fmi = fmi_rows.sort_values("f1_mean", ascending=False).iloc[0]
    rows.append(
        {
            "source": "FMI",
            "scope": "locked external transfer",
            "status": fmi["status"],
            "trust_f1": float(trust_fmi["f1_mean"]),
            "best_baseline": str(best_fmi["method"]),
            "best_baseline_f1": float(best_fmi["f1_mean"]),
            "trust_event_coverage": float(trust_fmi["event_coverage_mean"]),
            "best_baseline_event_coverage": float(best_fmi["event_coverage_mean"]),
            "gate_or_boundary": (
                "One-Class SVM is stronger; FMI does not support provider-independent superiority."
            ),
        }
    )
    smhi_rows = pd.DataFrame(smhi["provider_equal"])
    trust_smhi = smhi_rows[smhi_rows["method"].eq("trust_valid_f1")].iloc[0]
    best_smhi = smhi_rows.sort_values("f1_mean", ascending=False).iloc[0]
    rows.append(
        {
            "source": "SMHI",
            "scope": "station holdout locked transfer",
            "status": smhi["status"],
            "trust_f1": float(trust_smhi["f1_mean"]),
            "best_baseline": str(best_smhi["method"]),
            "best_baseline_f1": float(best_smhi["f1_mean"]),
            "trust_event_coverage": float(trust_smhi["event_coverage_mean"]),
            "best_baseline_event_coverage": float(best_smhi["event_coverage_mean"]),
            "gate_or_boundary": (
                f"transfer_gate_status={smhi['transfer_gate_status']}; "
                f"primary_pass_count={smhi.get('primary_pass_count')}/"
                f"{smhi.get('primary_comparison_count')}"
            ),
        }
    )
    rows.append(
        {
            "source": "SMHI",
            "scope": "factor holdout exploratory audit",
            "status": smhi_factor["status"],
            "trust_f1": None,
            "best_baseline": None,
            "best_baseline_f1": None,
            "trust_event_coverage": None,
            "best_baseline_event_coverage": None,
            "gate_or_boundary": (
                f"scenario_win_share={smhi_factor['scenario_win_share']['share']:.4f}; "
                f"unseen_combination_event_coverage={smhi_factor['checks'][-2]['value']}"
            ),
        }
    )
    rows.append(
        {
            "source": "SMHI",
            "scope": "third-station holdout exploratory audit",
            "status": smhi_third["status"],
            "trust_f1": None,
            "best_baseline": None,
            "best_baseline_f1": None,
            "trust_event_coverage": None,
            "best_baseline_event_coverage": None,
            "gate_or_boundary": (
                f"scenario_win_share={smhi_third['scenario_win_share']['share']:.4f}; "
                f"unseen_combination_event_coverage="
                f"{next(item['value'] for item in smhi_third['checks'] if item['check'] == 'unseen_combination_event_coverage')}"
            ),
        }
    )
    return rows


def build_online_offline() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_json("stage261_causal_offline_recovery_contrast.json")
    online = load_json("stage255_online_causal_testbed.json")
    recovery_rows = payload["aggregate"]
    detection_rows = online["aggregate"]
    return recovery_rows, detection_rows


def build_method_boundary() -> list[dict[str, Any]]:
    return [
        {
            "method": "trust_only",
            "role": "proposed candidate",
            "implementation_status": "frozen implementation",
            "causal_online": True,
            "training_boundary": "train-normal fit; valid calibration; test evaluation only",
            "notes": "Trust evidence score at the frozen operating point.",
        },
        {
            "method": "route_knn_arima",
            "role": "proposed candidate",
            "implementation_status": "frozen implementation",
            "causal_online": True,
            "training_boundary": "valid-only route selection; test evaluation only",
            "notes": "Validation-only route among Trust, kNN, and ARIMA innovation.",
        },
        {
            "method": "trust_shape_union",
            "role": "proposed candidate",
            "implementation_status": "frozen implementation",
            "causal_online": True,
            "training_boundary": "valid-only union budget; test evaluation only",
            "notes": "Union of Trust, EWMA, and ARIMA evidence with 50/25/25 FAR budget.",
        },
        {
            "method": "robust_max_deviation",
            "role": "baseline",
            "implementation_status": "formal deterministic implementation",
            "causal_online": True,
            "training_boundary": "train-normal fit; valid threshold; test evaluation only",
            "notes": "Maximum positive robust deviation.",
        },
        {
            "method": "isolation_forest",
            "role": "baseline",
            "implementation_status": "formal sklearn implementation",
            "causal_online": True,
            "training_boundary": "train-normal fit; valid threshold; test evaluation only",
            "notes": "sklearn IsolationForest.",
        },
        {
            "method": "one_class_svm",
            "role": "baseline",
            "implementation_status": "formal sklearn implementation",
            "causal_online": True,
            "training_boundary": "20k train-normal rows for V6; 50k/100k are sensitivity only",
            "notes": "RBF One-Class SVM; complementary on FMI and some scenarios.",
        },
        {
            "method": "pca_reconstruction",
            "role": "baseline",
            "implementation_status": "formal sklearn implementation",
            "causal_online": False,
            "training_boundary": "100k train-normal rows; valid threshold; test evaluation only",
            "notes": "Reconstruction baseline.",
        },
        {
            "method": "mlp_autoencoder_reconstruction",
            "role": "baseline",
            "implementation_status": "formal sklearn implementation",
            "causal_online": False,
            "training_boundary": "100k train-normal rows; valid threshold; test evaluation only",
            "notes": "MLPRegressor reconstruction baseline.",
        },
        {
            "method": "knn_distance",
            "role": "baseline",
            "implementation_status": "formal sklearn implementation",
            "causal_online": False,
            "training_boundary": "10k train-normal core set; valid threshold; test evaluation only",
            "notes": "Nearest-neighbor distance baseline.",
        },
        {
            "method": "ewma_deviation",
            "role": "baseline",
            "implementation_status": "transparent statistical implementation",
            "causal_online": True,
            "training_boundary": "train-normal fit; valid threshold; test evaluation only",
            "notes": "Station-wise EWMA deviation; measured FAR can exceed 1%.",
        },
        {
            "method": "arima_innovation",
            "role": "baseline",
            "implementation_status": "transparent AR(4) proxy",
            "causal_online": True,
            "training_boundary": "train-normal fit; valid threshold; test evaluation only",
            "notes": "Not a full ARIMA reproduction.",
        },
        {
            "method": "gru_predictor",
            "role": "baseline",
            "implementation_status": "transparent reservoir proxy",
            "causal_online": True,
            "training_boundary": "fixed random reservoir plus linear readout",
            "notes": "Not an end-to-end trained GRU.",
        },
        {
            "method": "lightweight_autoencoder",
            "role": "baseline",
            "implementation_status": "transparent PCA autoencoder proxy",
            "causal_online": False,
            "training_boundary": "100k train-normal rows; valid threshold; test evaluation only",
            "notes": "Not a deep autoencoder.",
        },
        {
            "method": "ics_cps_spc_proxy",
            "role": "baseline",
            "implementation_status": "transparent SPC proxy",
            "causal_online": True,
            "training_boundary": "train-normal fit; valid threshold; test evaluation only",
            "notes": "Not a bitwise reproduction of a public ICS/CPS method.",
        },
    ]


def build_attack_parameters() -> list[dict[str, Any]]:
    protocol = json.loads(
        (
            ROOT
            / "confirmatory_package"
            / "simulated_attack_v6"
            / "protocol_freeze.json"
        ).read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    for scenario in protocol["attack_scenarios"]:
        rows.append(
            {
                "scenario": scenario["name"],
                "base_type": scenario["base_type"],
                "magnitude": scenario.get("magnitude"),
                "sigma": scenario.get("sigma"),
                "length_rows": scenario.get("length"),
                "lag_rows": scenario.get("lag_rows"),
                "start": scenario.get("start"),
                "end": scenario.get("end"),
                "coordinated": bool(scenario.get("coordinated", False)),
                "station_count": scenario.get("station_count"),
                "label_source": "frozen transparent generator",
            }
        )
    return rows


def build_evidence_hierarchy() -> list[dict[str, Any]]:
    return [
        {
            "evidence_layer": "V6 frozen four-provider simulated attack",
            "status": "primary frozen result",
            "role": "main result",
            "promotion_boundary": "V6 test was inspected during exploratory work; this package does not create a new confirmatory test.",
        },
        {
            "evidence_layer": "V7 factor stress test",
            "status": "exploratory",
            "role": "sensitivity only",
            "promotion_boundary": "77.6% scenario-by-baseline win share; not a replacement for V6.",
        },
        {
            "evidence_layer": "Stage246/248 historical package",
            "status": "historical",
            "role": "prior frozen analysis",
            "promotion_boundary": "Must not be presented as the current V6 primary endpoint.",
        },
        {
            "evidence_layer": "Stage466/467 majority-leadership audit",
            "status": "exploratory",
            "role": "goal audit only",
            "promotion_boundary": "V6 test feedback is already known; no confirmatory promotion.",
        },
        {
            "evidence_layer": "FMI/SMHI locked transfers",
            "status": "external negative/mixed",
            "role": "external validity",
            "promotion_boundary": "Do not claim provider-independent or station-independent superiority.",
        },
        {
            "evidence_layer": "Authorized loopback testbed",
            "status": "runtime evidence",
            "role": "deployment boundary",
            "promotion_boundary": "Not a production incident log and not a real attack recall estimate.",
        },
        {
            "evidence_layer": "Natural quality flags",
            "status": "weak labels",
            "role": "operational association",
            "promotion_boundary": "Not attack ground truth.",
        },
        {
            "evidence_layer": "V9 future-window package",
            "status": "NOT_READY / NOT_EVALUATED",
            "role": "not used",
            "promotion_boundary": "No performance claim is made from V9.",
        },
    ]


def build_negative_boundaries() -> list[dict[str, Any]]:
    return [
        {
            "boundary": "One-Class SVM",
            "evidence": "FMI F1 0.2156 vs Trust 0.1836; competitive on several providers",
            "required_wording": "complementary baseline, not universal dominance",
        },
        {
            "boundary": "FMI external transfer",
            "evidence": "Trust is not the strongest method on FMI",
            "required_wording": "negative external transfer",
        },
        {
            "boundary": "SMHI external transfer",
            "evidence": "transfer gate not passed; only 8/22 primary comparisons passed",
            "required_wording": "mixed or failed transfer",
        },
        {
            "boundary": "coordinated_offset",
            "evidence": "1/11 scenario-by-baseline wins, 9.1%",
            "required_wording": "retained failure boundary",
        },
        {
            "boundary": "slow_drift",
            "evidence": "3/11 scenario-by-baseline wins, 27.3%",
            "required_wording": "retained failure boundary",
        },
        {
            "boundary": "non-stuck instability",
            "evidence": "historical cross-zero uncertainty and ECCC-dependent F1 reversal",
            "required_wording": "exploratory and not a universal claim",
        },
        {
            "boundary": "EWMA FAR",
            "evidence": "V6 measured FAR 0.0106008, above the 0.01 acceptance gate",
            "required_wording": "constraint deviation retained",
        },
        {
            "boundary": "online vs offline recovery",
            "evidence": "online MAE 2.0706 vs offline 1.5177",
            "required_wording": "information sets differ; not equivalent",
        },
        {
            "boundary": "real production attacks",
            "evidence": "no independently governed production incident log",
            "required_wording": "controlled simulations only; no production recall claim",
        },
        {
            "boundary": "provider/station independence",
            "evidence": "four primary providers plus FMI/SMHI transfer checks",
            "required_wording": "do not claim provider-independent or station-independent generalization",
        },
    ]


def build_scenario_ci() -> list[dict[str, Any]]:
    path = FINAL_TABLES / "final_v6_scenario_metrics.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    metrics = [
        "precision",
        "row_recall",
        "f1",
        "event_coverage",
        "average_precision",
        "false_alarm_rate",
    ]
    rng = np.random.default_rng(20261009)
    rows: list[dict[str, Any]] = []
    for (scenario, method), group in frame.groupby(["scenario", "method"], sort=True):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if values.size == 0:
                continue
            draws = rng.choice(
                values,
                size=(10000, values.size),
                replace=True,
            ).mean(axis=1)
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "resampling_unit": "provider_seed",
                    "iterations": 10000,
                    "note": (
                        "Scenario-level provider-seed bootstrap; the main "
                        "station-cluster audit remains Stage351."
                    ),
                }
            )
    return rows


def main() -> int:
    FINAL_TABLES.mkdir(parents=True, exist_ok=True)
    FINAL_REPORTS.mkdir(parents=True, exist_ok=True)
    operating = build_operating_points()
    baseline_resources, trust_resources, testbed_resources = build_resource_tables()
    external = build_external_transfer()
    recovery, online_detection = build_online_offline()
    method_boundary = build_method_boundary()
    attack_parameters = build_attack_parameters()
    evidence_hierarchy = build_evidence_hierarchy()
    negative_boundaries = build_negative_boundaries()
    scenario_ci = build_scenario_ci()

    write_csv(FINAL_TABLES / "final_operating_points.csv", operating)
    write_csv(FINAL_TABLES / "final_baseline_resources.csv", baseline_resources)
    write_csv(FINAL_TABLES / "final_trust_resources.csv", trust_resources)
    write_csv(FINAL_TABLES / "final_testbed_resources.csv", testbed_resources)
    write_csv(FINAL_TABLES / "final_external_transfer.csv", external)
    write_csv(FINAL_TABLES / "final_online_offline_recovery.csv", recovery)
    write_csv(FINAL_TABLES / "final_online_detection.csv", online_detection)
    write_csv(FINAL_TABLES / "final_method_implementation_boundary.csv", method_boundary)
    write_csv(FINAL_TABLES / "final_attack_parameter_freeze.csv", attack_parameters)
    write_csv(FINAL_TABLES / "final_evidence_hierarchy.csv", evidence_hierarchy)
    write_csv(FINAL_TABLES / "final_negative_boundaries.csv", negative_boundaries)
    write_csv(FINAL_TABLES / "final_v6_scenario_ci.csv", scenario_ci)

    package_payload = {
        "status": "PASS_FINAL_SUBMISSION_TABLES_ASSEMBLED",
        "package_root": str(FINAL.relative_to(ROOT)),
        "tables": [
            "final_operating_points.csv",
            "final_baseline_resources.csv",
            "final_trust_resources.csv",
            "final_testbed_resources.csv",
            "final_external_transfer.csv",
            "final_online_offline_recovery.csv",
            "final_online_detection.csv",
            "final_method_implementation_boundary.csv",
            "final_attack_parameter_freeze.csv",
            "final_evidence_hierarchy.csv",
            "final_negative_boundaries.csv",
            "final_v6_scenario_ci.csv",
        ],
        "rules": [
            "No experimental metric is invented or interpolated.",
            "Historical operating-point sensitivity is labeled as historical and does not replace V6.",
            "FMI/SMHI negative and mixed results remain visible.",
            "Testbed and simulated attacks are never described as production incidents.",
            "Method implementation boundaries distinguish formal implementations from transparent proxies.",
        ],
    }
    (FINAL_TABLES / "final_submission_tables.json").write_text(
        json.dumps(package_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(package_payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
