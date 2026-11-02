from __future__ import annotations

"""Stage 230: synthesize the frozen cross-provider evidence.

This stage is read-only with respect to every source artifact.  It does
not train a detector, recalibrate a threshold, or select a provider,
station, seed, attack type, operating point, or metric.  It combines the
already frozen DWD, MeteoSwiss, NOAA, KNMI, and ECCC results into one
auditable source-level table and one transfer-arm table.

The synthesis is intentionally conservative.  It reports that absolute
score thresholds failed when transferred across providers and that, after
the valid-attack protocol correction, target-domain valid local selection
outperformed DWD feature/window/weight transfer with target-domain valid
recalibration in all four targets.  It keeps the exact provider and
station-count limits visible.  All attacks are transparent controlled
injections, not real attack logs.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage230_cross_provider_evidence_synthesis"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage218_swiss_negative_result_uncertainty import sha256_file  # noqa: E402

STAGE203 = TABLES / "stage203_independent_external_evidence_feasibility.json"
STAGE204 = TABLES / "stage204_dwd_protocol_preregistration.json"
STAGE205 = TABLES / "stage205_dwd_official_download.json"
STAGE206 = TABLES / "stage206_dwd_normalized.json"
STAGE207 = TABLES / "stage207_dwd_controlled_attack.json"
STAGE208 = TABLES / "stage208_dwd_persistent_repeat_protection.json"
STAGE209 = TABLES / "stage209_dwd_station_cluster_uncertainty.json"
STAGE210 = TABLES / "stage210_dwd_seed_randomized_replication.json"
STAGE211 = TABLES / "stage211_dwd_replication_station_uncertainty.json"
STAGE212 = TABLES / "stage212_dwd_operating_curve.json"
STAGE213 = TABLES / "stage213_dwd_external_evidence_closure.json"
STAGE214 = TABLES / "stage214_swiss_protocol_preregistration.json"
STAGE215 = TABLES / "stage215_swiss_official_download.json"
STAGE216 = TABLES / "stage216_swiss_normalized.json"
STAGE216A = TABLES / "stage216a_swiss_protocol_correction.json"
STAGE217 = TABLES / "stage217_swiss_controlled_attack.json"
STAGE218 = TABLES / "stage218_swiss_negative_result_uncertainty.json"
STAGE219 = TABLES / "stage219_swiss_dwd_protection_transfer.json"
STAGE220 = TABLES / "stage220_swiss_dwd_transfer_uncertainty.json"
STAGE221 = TABLES / "stage221_swiss_dwd_transfer_operating_curve.json"
STAGE222 = TABLES / "stage222_noaa_dwd_protection_transfer.json"
STAGE223 = TABLES / "stage223_noaa_transfer_uncertainty.json"
STAGE224 = TABLES / "stage224_knmi_protocol_preregistration.json"
STAGE225 = TABLES / "stage225_knmi_official_download.json"
STAGE226 = TABLES / "stage226_knmi_normalized.json"
STAGE226A = TABLES / "stage226a_knmi_missingness_audit.json"
STAGE227 = TABLES / "stage227_knmi_controlled_attack.json"
STAGE228 = TABLES / "stage228_knmi_dwd_protection_transfer.json"
STAGE229 = TABLES / "stage229_knmi_transfer_uncertainty.json"
STAGE234 = TABLES / "stage234_eccc_protocol_preregistration.json"
STAGE235 = TABLES / "stage235_eccc_official_download.json"
STAGE236 = TABLES / "stage236_eccc_normalized.json"
STAGE237 = TABLES / "stage237_eccc_controlled_attack.json"
STAGE238 = TABLES / "stage238_eccc_dwd_protection_transfer.json"
STAGE239 = TABLES / "stage239_eccc_transfer_uncertainty.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_arm(rows: list[dict], arm: str) -> dict:
    matches = [row for row in rows if str(row.get("arm", "")) == arm]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one arm {arm!r}, found {len(matches)}")
    return matches[0]


def find_selection_rule(
    rows: list[dict], selection_rule: str, max_far: float
) -> dict:
    matches = [
        row
        for row in rows
        if str(row.get("selection_rule", "")) == selection_rule
        and float(row.get("max_far", -1.0)) == float(max_far)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one aggregate row for "
            f"{selection_rule!r} at FAR={max_far}, found {len(matches)}"
        )
    return matches[0]


def metric_from_summary(rows: list[dict], metric: str) -> dict:
    matches = [row for row in rows if str(row.get("metric", "")) == metric]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one summary row for {metric!r}, found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    required = [
        STAGE203,
        STAGE204,
        STAGE205,
        STAGE206,
        STAGE207,
        STAGE208,
        STAGE209,
        STAGE210,
        STAGE211,
        STAGE212,
        STAGE213,
        STAGE214,
        STAGE215,
        STAGE216,
        STAGE216A,
        STAGE217,
        STAGE218,
        STAGE219,
        STAGE220,
        STAGE221,
        STAGE222,
        STAGE223,
        STAGE224,
        STAGE225,
        STAGE226,
        STAGE226A,
        STAGE227,
        STAGE228,
        STAGE229,
        STAGE234,
        STAGE235,
        STAGE236,
        STAGE237,
        STAGE238,
        STAGE239,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing frozen artifacts for Stage230: " + ", ".join(missing)
        )

    stage203 = load_json(STAGE203)
    stage204 = load_json(STAGE204)
    stage205 = load_json(STAGE205)
    stage206 = load_json(STAGE206)
    stage207 = load_json(STAGE207)
    stage208 = load_json(STAGE208)
    stage209 = load_json(STAGE209)
    stage210 = load_json(STAGE210)
    stage211 = load_json(STAGE211)
    stage212 = load_json(STAGE212)
    stage213 = load_json(STAGE213)
    stage214 = load_json(STAGE214)
    stage215 = load_json(STAGE215)
    stage216 = load_json(STAGE216)
    stage216a = load_json(STAGE216A)
    stage217 = load_json(STAGE217)
    stage218 = load_json(STAGE218)
    stage219 = load_json(STAGE219)
    stage220 = load_json(STAGE220)
    stage221 = load_json(STAGE221)
    stage222 = load_json(STAGE222)
    stage223 = load_json(STAGE223)
    stage224 = load_json(STAGE224)
    stage225 = load_json(STAGE225)
    stage226 = load_json(STAGE226)
    stage226a = load_json(STAGE226A)
    stage227 = load_json(STAGE227)
    stage228 = load_json(STAGE228)
    stage229 = load_json(STAGE229)
    stage234 = load_json(STAGE234)
    stage235 = load_json(STAGE235)
    stage236 = load_json(STAGE236)
    stage237 = load_json(STAGE237)
    stage238 = load_json(STAGE238)
    stage239 = load_json(STAGE239)

    source_rows: list[dict[str, object]] = []

    dwd_aggregate = find_selection_rule(
        stage208["aggregate"], "valid_f1", 0.01
    )
    dwd_replication = stage210["summary"]
    dwd_station_ci = metric_from_summary(stage209["bootstrap"], "f1")
    dwd_replication_ci = metric_from_summary(stage211["bootstrap"], "f1")
    dwd_primary_curve = [
        row
        for row in stage212["curve"]
        if str(row.get("metric", "")) == "f1" and bool(row.get("is_primary"))
    ]
    if len(dwd_primary_curve) != 2:
        raise RuntimeError("expected two primary DWD F1 curve rows")
    source_rows.append(
        {
            "source": "DWD",
            "provider": stage204["provider"],
            "role": "primary_cross_provider",
            "station_count": int(stage204["station_count"]),
            "years": ",".join(str(x) for x in stage204["years"]),
            "primary_far": float(stage204["primary_far"]),
            "primary_endpoint": str(stage204["primary_endpoint"]),
            "cluster_unit": str(stage204["cluster_unit"]),
            "download_status": str(stage205["status"]),
            "normalization_status": str(stage206["status"]),
            "performance_status": str(stage207["status"]),
            "transfer_status": "not_applicable_primary_source",
            "direct_threshold_transfer": "not_applicable_primary_source",
            "recalibrated_transfer_f1": float(dwd_aggregate["test_f1"]),
            "recalibrated_transfer_event_coverage": float(
                dwd_aggregate["test_event_coverage"]
            ),
            "replication_f1_mean": float(
                dwd_replication["candidate_test_f1_mean"]
            ),
            "f1_cluster_ci_low": float(dwd_station_ci["difference_ci95"][0]),
            "f1_cluster_ci_high": float(dwd_station_ci["difference_ci95"][1]),
            "f1_replication_ci_low": float(
                dwd_replication_ci["difference_ci95"][0]
            ),
            "f1_replication_ci_high": float(
                dwd_replication_ci["difference_ci95"][1]
            ),
            "source_limit": (
                "Only one provider; 12 stations; controlled injections only; "
                "event coverage is not uniformly positive at low-FAR points."
            ),
        }
    )

    swiss_aggregate = stage219["aggregate"]
    swiss_direct = find_arm(swiss_aggregate, "dwd_threshold_transfer")
    swiss_recal = find_arm(swiss_aggregate, "swiss_valid_recalibrated")
    swiss_selected = find_arm(swiss_aggregate, "swiss_valid_selected")
    swiss_f1 = stage220["primary_f1"]
    swiss_event = stage220["primary_event_coverage"]
    source_rows.append(
        {
            "source": "MeteoSwiss",
            "provider": stage214["provider"],
            "role": "second_cross_provider_transfer_target",
            "station_count": int(stage220["station_count"]),
            "years": ",".join(str(x) for x in stage214["years"]),
            "primary_far": float(stage214["primary_far"]),
            "primary_endpoint": str(stage214["primary_endpoint"]),
            "cluster_unit": str(stage214["cluster_unit"]),
            "download_status": str(stage215["status"]),
            "normalization_status": str(stage216["status"]),
            "performance_status": str(stage217["status"]),
            "transfer_status": str(stage219["status"]),
            "direct_threshold_transfer": "failed_f1_0_event_coverage_0",
            "recalibrated_transfer_f1": float(swiss_recal["f1_mean"]),
            "recalibrated_transfer_event_coverage": float(
                swiss_recal["event_coverage_mean"]
            ),
            "local_valid_selected_f1": float(swiss_selected["f1_mean"]),
            "local_valid_selected_event_coverage": float(
                swiss_selected["event_coverage_mean"]
            ),
            "direct_threshold_transfer_f1": float(swiss_direct["f1_mean"]),
            "direct_threshold_transfer_event_coverage": float(
                swiss_direct["event_coverage_mean"]
            ),
            "f1_cluster_ci_low": float(swiss_f1["ci95_low"]),
            "f1_cluster_ci_high": float(swiss_f1["ci95_high"]),
            "f1_positive_stations": int(swiss_f1["positive_stations"]),
            "f1_negative_stations": int(swiss_f1["negative_stations"]),
            "event_cluster_ci_low": float(swiss_event["ci95_low"]),
            "event_cluster_ci_high": float(swiss_event["ci95_high"]),
            "event_positive_stations": int(swiss_event["positive_stations"]),
            "event_negative_stations": int(swiss_event["negative_stations"]),
            "source_limit": (
                "Only 11 MeteoSwiss stations; direct DWD threshold transfer "
                "failed; target-domain valid recalibration is required; "
                "controlled injections only."
            ),
        }
    )

    noaa_aggregate = stage222["aggregate"]
    noaa_direct = find_arm(noaa_aggregate, "dwd_threshold_transfer")
    noaa_recal = find_arm(noaa_aggregate, "noaa_valid_recalibrated")
    noaa_selected = find_arm(noaa_aggregate, "noaa_valid_selected")
    noaa_f1 = stage223["primary_f1"]
    noaa_event = stage223["primary_event_coverage"]
    source_rows.append(
        {
            "source": "NOAA ISD",
            "provider": stage222["source"],
            "role": "third_cross_provider_transfer_target",
            "station_count": int(stage222["station_count"]),
            "years": "2020-2024",
            "primary_far": float(stage222["primary_far"]),
            "primary_endpoint": str(stage222["primary_endpoint"]),
            "cluster_unit": str(stage222["cluster_unit"]),
            "download_status": "official_local_archive",
            "normalization_status": "normalized_and_quality_audited",
            "performance_status": str(stage222["status"]),
            "transfer_status": str(stage222["status"]),
            "direct_threshold_transfer": "failed_f1_0_event_coverage_0",
            "recalibrated_transfer_f1": float(noaa_recal["f1_mean"]),
            "recalibrated_transfer_event_coverage": float(
                noaa_recal["event_coverage_mean"]
            ),
            "local_valid_selected_f1": float(noaa_selected["f1_mean"]),
            "local_valid_selected_event_coverage": float(
                noaa_selected["event_coverage_mean"]
            ),
            "direct_threshold_transfer_f1": float(noaa_direct["f1_mean"]),
            "direct_threshold_transfer_event_coverage": float(
                noaa_direct["event_coverage_mean"]
            ),
            "f1_cluster_ci_low": float(noaa_f1["ci95_low"]),
            "f1_cluster_ci_high": float(noaa_f1["ci95_high"]),
            "f1_positive_stations": int(noaa_f1["positive_stations"]),
            "f1_negative_stations": int(noaa_f1["negative_stations"]),
            "event_cluster_ci_low": float(noaa_event["ci95_low"]),
            "event_cluster_ci_high": float(noaa_event["ci95_high"]),
            "event_positive_stations": int(noaa_event["positive_stations"]),
            "event_negative_stations": int(noaa_event["negative_stations"]),
            "source_limit": (
                "Only 11 NOAA stations; DWD absolute threshold transfer "
                "failed; F1 interval crosses zero and event coverage is "
                "supported by one station with ten ties; controlled injections only."
            ),
        }
    )

    knmi_aggregate = stage228["aggregate"]
    knmi_direct = find_arm(knmi_aggregate, "dwd_threshold_transfer")
    knmi_recal = find_arm(knmi_aggregate, "knmi_valid_recalibrated")
    knmi_selected = find_arm(knmi_aggregate, "knmi_valid_selected")
    knmi_f1 = stage229["primary_f1"]
    knmi_event = stage229["primary_event_coverage"]
    source_rows.append(
        {
            "source": "KNMI",
            "provider": stage224["provider"],
            "role": "fourth_cross_provider_transfer_target",
            "station_count": int(stage229["station_count"]),
            "years": ",".join(str(x) for x in stage224["years"]),
            "primary_far": float(stage224["primary_far"]),
            "primary_endpoint": str(stage224["primary_endpoint"]),
            "cluster_unit": str(stage224["cluster_unit"]),
            "download_status": str(stage225["status"]),
            "normalization_status": str(stage226["status"]),
            "performance_status": str(stage227["status"]),
            "transfer_status": str(stage228["status"]),
            "direct_threshold_transfer": "failed_f1_0_event_coverage_0",
            "recalibrated_transfer_f1": float(knmi_recal["f1_mean"]),
            "recalibrated_transfer_event_coverage": float(
                knmi_recal["event_coverage_mean"]
            ),
            "local_valid_selected_f1": float(knmi_selected["f1_mean"]),
            "local_valid_selected_event_coverage": float(
                knmi_selected["event_coverage_mean"]
            ),
            "direct_threshold_transfer_f1": float(knmi_direct["f1_mean"]),
            "direct_threshold_transfer_event_coverage": float(
                knmi_direct["event_coverage_mean"]
            ),
            "f1_cluster_ci_low": float(knmi_f1["ci95_low"]),
            "f1_cluster_ci_high": float(knmi_f1["ci95_high"]),
            "f1_positive_stations": int(knmi_f1["positive_stations"]),
            "f1_negative_stations": int(knmi_f1["negative_stations"]),
            "event_cluster_ci_low": float(knmi_event["ci95_low"]),
            "event_cluster_ci_high": float(knmi_event["ci95_high"]),
            "event_positive_stations": int(knmi_event["positive_stations"]),
            "event_negative_stations": int(knmi_event["negative_stations"]),
            "source_limit": (
                "Only 6 KNMI stations after train/valid missingness "
                "screening; DWD absolute threshold transfer failed; F1 exact "
                "sign-flip p=0.03125 and BH q=0.0625 are close to but do not "
                "fall below 0.05; controlled injections only."
            ),
        }
    )

    eccc_aggregate = stage238["aggregate"]
    eccc_direct = find_arm(eccc_aggregate, "dwd_threshold_transfer")
    eccc_recal = find_arm(eccc_aggregate, "eccc_valid_recalibrated")
    eccc_selected = find_arm(eccc_aggregate, "eccc_valid_selected")
    eccc_f1 = stage239["primary_f1"]
    eccc_event = stage239["primary_event_coverage"]
    source_rows.append(
        {
            "source": "ECCC",
            "provider": stage234["provider"],
            "role": "fifth_cross_provider_transfer_target",
            "station_count": int(stage239["station_count"]),
            "years": ",".join(str(x) for x in stage234["years"]),
            "primary_far": float(stage234["primary_far"]),
            "primary_endpoint": str(stage234["primary_endpoint"]),
            "cluster_unit": str(stage234["cluster_unit"]),
            "download_status": str(stage235["status"]),
            "normalization_status": str(stage236["status"]),
            "performance_status": str(stage237["status"]),
            "transfer_status": str(stage238["status"]),
            "direct_threshold_transfer": "failed_f1_0_event_coverage_0",
            "recalibrated_transfer_f1": float(eccc_recal["f1_mean"]),
            "recalibrated_transfer_event_coverage": float(
                eccc_recal["event_coverage_mean"]
            ),
            "local_valid_selected_f1": float(eccc_selected["f1_mean"]),
            "local_valid_selected_event_coverage": float(
                eccc_selected["event_coverage_mean"]
            ),
            "direct_threshold_transfer_f1": float(eccc_direct["f1_mean"]),
            "direct_threshold_transfer_event_coverage": float(
                eccc_direct["event_coverage_mean"]
            ),
            "f1_cluster_ci_low": float(eccc_f1["ci95_low"]),
            "f1_cluster_ci_high": float(eccc_f1["ci95_high"]),
            "f1_positive_stations": int(eccc_f1["positive_stations"]),
            "f1_negative_stations": int(eccc_f1["negative_stations"]),
            "event_cluster_ci_low": float(eccc_event["ci95_low"]),
            "event_cluster_ci_high": float(eccc_event["ci95_high"]),
            "event_positive_stations": int(
                eccc_event["positive_stations"]
            ),
            "event_negative_stations": int(
                eccc_event["negative_stations"]
            ),
            "source_limit": (
                "Only 14 ECCC stations after the pre-registered coverage "
                "rule; DWD absolute threshold transfer failed; controlled "
                "injections only."
            ),
        }
    )

    transfer_rows: list[dict[str, object]] = []
    transfer_specs = [
        {
            "target_source": "MeteoSwiss",
            "target_provider": stage214["provider"],
            "stage": 219,
            "station_count": int(stage220["station_count"]),
            "aggregate": stage219["aggregate"],
            "direct_arm": "dwd_threshold_transfer",
            "recalibrated_arm": "swiss_valid_recalibrated",
            "local_arm": "swiss_valid_selected",
            "uncertainty": stage220,
            "f1_key": "primary_f1",
            "event_key": "primary_event_coverage",
        },
        {
            "target_source": "NOAA ISD",
            "target_provider": stage222["source"],
            "stage": 222,
            "station_count": int(stage222["station_count"]),
            "aggregate": stage222["aggregate"],
            "direct_arm": "dwd_threshold_transfer",
            "recalibrated_arm": "noaa_valid_recalibrated",
            "local_arm": "noaa_valid_selected",
            "uncertainty": stage223,
            "f1_key": "primary_f1",
            "event_key": "primary_event_coverage",
        },
        {
            "target_source": "KNMI",
            "target_provider": stage224["provider"],
            "stage": 228,
            "station_count": int(stage229["station_count"]),
            "aggregate": stage228["aggregate"],
            "direct_arm": "dwd_threshold_transfer",
            "recalibrated_arm": "knmi_valid_recalibrated",
            "local_arm": "knmi_valid_selected",
            "uncertainty": stage229,
            "f1_key": "primary_f1",
            "event_key": "primary_event_coverage",
        },
        {
            "target_source": "ECCC",
            "target_provider": stage234["provider"],
            "stage": 238,
            "station_count": int(stage239["station_count"]),
            "aggregate": stage238["aggregate"],
            "direct_arm": "dwd_threshold_transfer",
            "recalibrated_arm": "eccc_valid_recalibrated",
            "local_arm": "eccc_valid_selected",
            "uncertainty": stage239,
            "f1_key": "primary_f1",
            "event_key": "primary_event_coverage",
        },
    ]
    for spec in transfer_specs:
        direct = find_arm(spec["aggregate"], spec["direct_arm"])
        recalibrated = find_arm(spec["aggregate"], spec["recalibrated_arm"])
        local = find_arm(spec["aggregate"], spec["local_arm"])
        uncertainty = spec["uncertainty"]
        f1 = uncertainty[spec["f1_key"]]
        event = uncertainty[spec["event_key"]]
        transfer_rows.append(
            {
                "target_source": spec["target_source"],
                "target_provider": spec["target_provider"],
                "transfer_source": "Deutscher Wetterdienst (DWD)",
                "stage": int(spec["stage"]),
                "station_count": int(spec["station_count"]),
                "primary_far": float(uncertainty["primary_far"]),
                "direct_threshold_transfer_f1": float(direct["f1_mean"]),
                "direct_threshold_transfer_event_coverage": float(
                    direct["event_coverage_mean"]
                ),
                "target_valid_recalibrated_f1": float(recalibrated["f1_mean"]),
                "target_valid_recalibrated_event_coverage": float(
                    recalibrated["event_coverage_mean"]
                ),
                "target_valid_selected_f1": float(local["f1_mean"]),
                "target_valid_selected_event_coverage": float(
                    local["event_coverage_mean"]
                ),
                "selected_minus_recalibrated_f1": float(f1["mean_difference"]),
                "selected_minus_recalibrated_f1_ci95_low": float(
                    f1["ci95_low"]
                ),
                "selected_minus_recalibrated_f1_ci95_high": float(
                    f1["ci95_high"]
                ),
                "selected_minus_recalibrated_f1_p": float(f1["sign_flip_p"]),
                "selected_minus_recalibrated_f1_bh_q": float(f1["bh_q"]),
                "selected_minus_recalibrated_event_coverage": float(
                    event["mean_difference"]
                ),
                "selected_minus_recalibrated_event_coverage_ci95_low": float(
                    event["ci95_low"]
                ),
                "selected_minus_recalibrated_event_coverage_ci95_high": float(
                    event["ci95_high"]
                ),
                "selected_minus_recalibrated_event_coverage_p": float(
                    event["sign_flip_p"]
                ),
                "f1_positive_stations": int(f1["positive_stations"]),
                "f1_negative_stations": int(f1["negative_stations"]),
                "f1_zero_stations": int(f1["zero_stations"]),
                "event_positive_stations": int(event["positive_stations"]),
                "event_negative_stations": int(event["negative_stations"]),
                "event_zero_stations": int(event["zero_stations"]),
                "interpretation": (
                    "Absolute DWD threshold transfer failed; after the valid-"
                    "attack protocol correction, target-domain valid local "
                    "selection outperformed feature/window/weight transfer "
                    "with target-domain valid recalibration."
                ),
                "boundary": (
                    "Only this provider's available stations; controlled "
                    "injections only; do not generalize to multi-provider "
                    "superiority or real attack detection."
                ),
            }
        )

    direct_failures = [
        row
        for row in transfer_rows
        if float(row["direct_threshold_transfer_f1"]) == 0.0
        and float(row["direct_threshold_transfer_event_coverage"]) == 0.0
    ]
    selected_advantage_f1 = [
        row
        for row in transfer_rows
        if float(row["selected_minus_recalibrated_f1"]) > 0.0
    ]
    selected_advantage_event = [
        row
        for row in transfer_rows
        if float(row["selected_minus_recalibrated_event_coverage"]) > 0.0
    ]
    f1_significant = [
        row
        for row in transfer_rows
        if float(row["selected_minus_recalibrated_f1_p"]) < 0.05
        and float(row["selected_minus_recalibrated_f1"]) > 0.0
    ]
    event_significant = [
        row
        for row in transfer_rows
        if float(row["selected_minus_recalibrated_event_coverage_p"]) < 0.05
        and float(row["selected_minus_recalibrated_event_coverage"]) > 0.0
    ]

    checks: list[dict[str, object]] = [
        {
            "check": "source_count",
            "status": "PASS" if len(source_rows) == 5 else "FAIL",
            "value": len(source_rows),
        },
        {
            "check": "frozen_primary_far_consistency",
            "status": "PASS"
            if all(float(row["primary_far"]) == 0.01 for row in source_rows)
            else "FAIL",
            "value": [float(row["primary_far"]) for row in source_rows],
        },
        {
            "check": "frozen_primary_endpoint_consistency",
            "status": "PASS"
            if all(
                str(row["primary_endpoint"]) == "event_coverage"
                for row in source_rows
            )
            else "FAIL",
            "value": [str(row["primary_endpoint"]) for row in source_rows],
        },
        {
            "check": "station_cluster_unit_consistency",
            "status": "PASS"
            if all(str(row["cluster_unit"]) == "station" for row in source_rows)
            else "FAIL",
            "value": [str(row["cluster_unit"]) for row in source_rows],
        },
        {
            "check": "absolute_threshold_transfer_failures_retained",
            "status": "PASS" if len(direct_failures) == 4 else "FAIL",
            "value": [
                row["target_source"] for row in direct_failures
            ],
        },
        {
            "check": "target_valid_selected_f1_advantage_sources",
            "status": "PASS" if len(selected_advantage_f1) == 4 else "FAIL",
            "value": [
                row["target_source"] for row in selected_advantage_f1
            ],
        },
        {
            "check": "target_valid_selected_event_advantage_sources",
            "status": "PASS" if len(selected_advantage_event) == 4 else "FAIL",
            "value": [row["target_source"] for row in selected_advantage_event],
        },
        {
            "check": "f1_advantage_significance_reported",
            "status": "PASS" if len(f1_significant) == len(transfer_rows) else "FAIL",
            "value": [row["target_source"] for row in f1_significant],
        },
        {
            "check": "event_significance_not_uniform",
            "status": "PASS" if len(event_significant) < len(transfer_rows) else "FAIL",
            "value": [row["target_source"] for row in event_significant],
        },
        {
            "check": "dwd_single_provider_boundary_retained",
            "status": "PASS"
            if bool(stage213.get("single_provider_only"))
            and "不能扩展为多机构泛化声明" in str(stage213.get("protocol", ""))
            else "FAIL",
            "value": stage213.get("single_provider_only"),
        },
        {
            "check": "knmi_six_station_boundary_retained",
            "status": "PASS"
            if int(stage229.get("station_count", 0)) == 6
            and float(stage229["primary_f1"]["sign_flip_p"]) == 0.03125
            and float(stage229["primary_f1"]["bh_q"]) == 0.0625
            else "FAIL",
            "value": {
                "station_count": stage229.get("station_count"),
                "f1_p": stage229["primary_f1"]["sign_flip_p"],
                "f1_bh_q": stage229["primary_f1"]["bh_q"],
            },
        },
        {
            "check": "eccc_fourteen_station_boundary_retained",
            "status": "PASS"
            if int(stage239.get("station_count", 0)) == 14
            and int(stage236.get("retained_station_count", 0)) == 14
            and int(stage236.get("excluded_station_count", 0)) == 2
            else "FAIL",
            "value": {
                "station_count": stage239.get("station_count"),
                "retained_station_count": stage236.get(
                    "retained_station_count"
                ),
                "excluded_station_count": stage236.get(
                    "excluded_station_count"
                ),
            },
        },
        {
            "check": "controlled_injection_boundary_retained",
            "status": "PASS"
            if all(
                "controlled" in str(row["boundary"]).lower()
                for row in transfer_rows
            )
            else "FAIL",
            "value": True,
        },
    ]
    failed_checks = [row for row in checks if row["status"] == "FAIL"]
    status = (
        "PASS_PIPELINE_REVIEW_REQUIRED"
        if not failed_checks
        else "CROSS_PROVIDER_EVIDENCE_REVIEW_REQUIRED"
    )

    payload = {
        "status": status,
        "protocol": (
            "阶段230只读取阶段203-239已冻结的DWD、MeteoSwiss、NOAA、"
            "KNMI和ECCC协议、下载、标准化、受控攻击、迁移和不确定性产物，"
            "生成来源级表和迁移臂表。不重新训练、不重新选择特征、"
            "窗口、权重、阈值、站点、种子、攻击类型或运行点，"
            "不使用目标域test标签做任何选择。"
            "绝对DWD阈值跨机构直接迁移的零结果必须保留；"
            "修正valid攻击注入后，DWD选出的连续重复特征族、窗口和"
            "权重即使配合目标域valid正常样本重校准阈值，仍未超过目标域"
            "valid本地选择；目标域valid本地选择优于重校准迁移，"
            "不能把重校准写成迁移优势；该负结果必须保留。"
            "所有攻击都是透明受控注入，不是真实攻击日志；"
            "来源站点数和可用性限制必须随结果一并报告。"
        ),
        "source_count": len(source_rows),
        "transfer_target_count": len(transfer_rows),
        "direct_threshold_transfer_failure_count": len(direct_failures),
        "target_valid_selected_f1_advantage_count": len(
            selected_advantage_f1
        ),
        "target_valid_selected_event_advantage_count": len(
            selected_advantage_event
        ),
        "f1_significant_source_count": len(f1_significant),
        "event_significant_source_count": len(event_significant),
        "source_rows": source_rows,
        "transfer_rows": transfer_rows,
        "checks": checks,
        "failed_checks": failed_checks,
        "limitations": [
            "跨机构来源包括DWD、MeteoSwiss、NOAA ISD、KNMI和ECCC，但各来源的站点数、传感器网络、处理链和攻击块不同。",
            "DWD绝对分数阈值在MeteoSwiss、NOAA、KNMI和ECCC上直接迁移均失败，不能把绝对阈值写成跨机构通用。",
            "修正valid攻击注入后，目标域valid本地选择在四个迁移目标上均优于DWD特征、窗口和权重配合目标域valid重校准；不能把重校准写成迁移优势。",
            "MeteoSwiss为11站、NOAA为11站、KNMI为6站、ECCC为14站；站点簇区间只描述当前站点集合的不确定性。",
            "KNMI的F1精确检验p=0.03125、BH q=0.0625，接近但未低于0.05，不能写成全面或普适F1领先。",
            "所有攻击都是透明受控注入，不是真实攻击日志，不能解释为真实入侵率。",
            "DWD、MeteoSwiss、NOAA、KNMI和ECCC的跨来源结果不能写成多机构普适领先；跨来源方向一致性不等于所有站点、攻击类型或运行点均占优。",
            "来源级表用于证据综合，不替代各阶段的原始JSON、CSV、Parquet和逐行明细。",
        ],
        "artifact_sha256": {
            "stage203": sha256_file(STAGE203),
            "stage213": sha256_file(STAGE213),
            "stage219": sha256_file(STAGE219),
            "stage220": sha256_file(STAGE220),
            "stage221": sha256_file(STAGE221),
            "stage222": sha256_file(STAGE222),
            "stage223": sha256_file(STAGE223),
            "stage227": sha256_file(STAGE227),
            "stage228": sha256_file(STAGE228),
            "stage229": sha256_file(STAGE229),
            "stage234": sha256_file(STAGE234),
            "stage235": sha256_file(STAGE235),
            "stage236": sha256_file(STAGE236),
            "stage237": sha256_file(STAGE237),
            "stage238": sha256_file(STAGE238),
            "stage239": sha256_file(STAGE239),
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(source_rows).to_csv(
        TABLES / f"{PREFIX}_sources.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(transfer_rows).to_csv(
        TABLES / f"{PREFIX}_transfers.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(checks).to_csv(
        TABLES / f"{PREFIX}_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段230 跨机构外部证据综合",
        "",
        f"- 状态：**{status}**",
        f"- 来源数：`{len(source_rows)}`；迁移目标数：`{len(transfer_rows)}`",
        f"- 绝对阈值直接迁移失败：`{len(direct_failures)}/{len(transfer_rows)}`",
        f"- 目标域valid本地选择F1优势：`{len(selected_advantage_f1)}/{len(transfer_rows)}`",
        f"- 目标域valid本地选择事件覆盖优势：`{len(selected_advantage_event)}/{len(transfer_rows)}`",
        f"- F1本地选择优势精确检验显著来源：`{len(f1_significant)}/{len(transfer_rows)}`",
        f"- 事件覆盖本地选择优势精确检验显著来源：`{len(event_significant)}/{len(transfer_rows)}`",
        "",
        "## 来源级综合",
        "",
        "| 来源 | 机构 | 角色 | 站点 | 主FAR | 主终点 | 直接阈值迁移 | 重校准F1 | 重校准事件覆盖 | 边界 |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in source_rows:
        direct = (
            "不适用"
            if str(row.get("direct_threshold_transfer", "")).startswith(
                "not_applicable"
            )
            else f"F1={float(row['direct_threshold_transfer_f1']):.6f}; "
            f"事件覆盖={float(row['direct_threshold_transfer_event_coverage']):.6f}"
        )
        lines.append(
            f"| {row['source']} | {row['provider']} | {row['role']} | "
            f"{int(row['station_count'])} | {float(row['primary_far']):.6f} | "
            f"{row['primary_endpoint']} | {direct} | "
            f"{float(row['recalibrated_transfer_f1']):.6f} | "
            f"{float(row['recalibrated_transfer_event_coverage']):.6f} | "
            f"{row['source_limit']} |"
        )
    lines += [
        "",
        "## 跨机构迁移臂",
        "",
        "| 目标来源 | 站点 | 直接阈值F1 | 直接阈值事件覆盖 | 重校准F1 | 重校准事件覆盖 | 本地valid F1 | 本地−重校准F1 | 95%区间 | 事件覆盖差值 | 95%区间 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in transfer_rows:
        lines.append(
            f"| {row['target_source']} | {int(row['station_count'])} | "
            f"{float(row['direct_threshold_transfer_f1']):.6f} | "
            f"{float(row['direct_threshold_transfer_event_coverage']):.6f} | "
            f"{float(row['target_valid_recalibrated_f1']):.6f} | "
            f"{float(row['target_valid_recalibrated_event_coverage']):.6f} | "
            f"{float(row['target_valid_selected_f1']):.6f} | "
            f"{float(row['selected_minus_recalibrated_f1']):+.6f} | "
            f"[{float(row['selected_minus_recalibrated_f1_ci95_low']):+.6f}, "
            f"{float(row['selected_minus_recalibrated_f1_ci95_high']):+.6f}] | "
            f"{float(row['selected_minus_recalibrated_event_coverage']):+.6f} | "
            f"[{float(row['selected_minus_recalibrated_event_coverage_ci95_low']):+.6f}, "
            f"{float(row['selected_minus_recalibrated_event_coverage_ci95_high']):+.6f}] |"
        )
    lines += [
        "",
        "## 审计检查",
        "",
        "| 检查 | 状态 | 值 |",
        "|---|---|---|",
    ]
    for row in checks:
        lines.append(
            f"| {row['check']} | {row['status']} | {row['value']} |"
        )
    lines += [
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
        "",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "source_count": len(source_rows),
                "transfer_target_count": len(transfer_rows),
                "direct_threshold_transfer_failure_count": len(
                    direct_failures
                ),
                "f1_significant_source_count": len(f1_significant),
                "event_significant_source_count": len(event_significant),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if not failed_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
