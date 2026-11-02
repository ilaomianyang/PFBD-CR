from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
EVIDENCE = TABLES / "stage29_evidence_matrix.json"
SELF_NAMES = {
    "stage112_full_artifact_manifest.json",
    "stage112_full_artifact_manifest.csv",
    "stage112_full_artifact_manifest.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_status(path: Path) -> str:
    if path.suffix.lower() != ".json":
        return ""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "UNREADABLE_JSON"
    return str(value.get("status", "")) if isinstance(value, dict) else "NON_OBJECT_JSON"


def main() -> int:
    if not EVIDENCE.exists():
        raise FileNotFoundError(EVIDENCE)
    evidence_payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    evidence_rows = evidence_payload.get("rows", [])
    if not isinstance(evidence_rows, list):
        raise ValueError("stage29 evidence matrix rows must be a list")
    declared_evidence_items = evidence_payload.get("counts", {}).get("evidence_items")

    candidate_files = [
        path
        for base, patterns in (
            (TABLES, ("stage*.json", "stage*.csv", "stage*.parquet")),
            (REPORTS, ("stage*.md",)),
            (REPORTS / "figures", ("stage*.png", "stage*.pdf")),
            (ROOT / "src", ("stage*.py", "trust_fusion.py")),
            (ROOT / "tests", ("*.py",)),
        )
        for pattern in patterns
        for path in sorted(base.glob(pattern))
        if path.name not in SELF_NAMES and path.is_file()
    ]
    records: list[dict[str, object]] = []
    for path in candidate_files:
        records.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "artifact_name": path.name,
                "kind": path.suffix.lower().lstrip("."),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "json_status": json_status(path),
            }
        )

    indexed = {str(row["artifact_name"]): row for row in records}
    missing: list[str] = []
    resolved: list[dict[str, object]] = []
    for row in evidence_rows:
        artifact = str(row.get("artifact", ""))
        table_path = TABLES / artifact
        report_path = REPORTS / artifact
        exists = table_path.is_file() or report_path.is_file()
        if not exists:
            missing.append(artifact)
        resolved.append(
            {
                "evidence_id": row.get("evidence_id", ""),
                "artifact": artifact,
                "exists": exists,
                "indexed": artifact in indexed,
            }
        )

    duplicate_evidence_ids = sorted(
        pd.Series([str(row.get("evidence_id", "")) for row in evidence_rows]).loc[lambda s: s.duplicated()].unique().tolist()
    )
    required_pairs = [
        (TABLES / "stage109_noaa_quality_weak_label_bootstrap.json", TABLES / "stage109_noaa_quality_weak_label_bootstrap.csv"),
        (TABLES / "stage110_noaa_quality_hierarchical_bootstrap.json", TABLES / "stage110_noaa_quality_hierarchical_bootstrap.csv"),
        (TABLES / "stage111_noaa_quality_count_sensitivity.json", TABLES / "stage111_noaa_quality_count_sensitivity.csv"),
        (TABLES / "stage218_swiss_negative_result_uncertainty.json", TABLES / "stage218_swiss_negative_result_uncertainty_attack_type.csv"),
        (TABLES / "stage219_swiss_dwd_protection_transfer.json", TABLES / "stage219_swiss_dwd_protection_transfer_station_event_detail.csv"),
        (TABLES / "stage220_swiss_dwd_transfer_uncertainty.json", TABLES / "stage220_swiss_dwd_transfer_uncertainty_metrics.csv"),
        (TABLES / "stage221_swiss_dwd_transfer_operating_curve.json", TABLES / "stage221_swiss_dwd_transfer_operating_curve_curve.csv"),
        (TABLES / "stage222_noaa_dwd_protection_transfer.json", TABLES / "stage222_noaa_dwd_protection_transfer_station_event_detail.csv"),
        (TABLES / "stage223_noaa_transfer_uncertainty.json", TABLES / "stage223_noaa_transfer_uncertainty_metrics.csv"),
        (TABLES / "stage224_knmi_protocol_preregistration.json", TABLES / "stage224_knmi_protocol_preregistration_contract.csv"),
        (TABLES / "stage225_knmi_official_download.json", TABLES / "stage225_knmi_official_download_detail.csv"),
        (TABLES / "stage226_knmi_normalized.json", TABLES / "stage226_knmi_normalized_archive_audit.csv"),
        (TABLES / "stage226a_knmi_missingness_audit.json", TABLES / "stage226a_knmi_missingness_audit_split.csv"),
        (TABLES / "stage227_knmi_controlled_attack.json", TABLES / "stage227_knmi_controlled_attack_evaluation.csv"),
        (TABLES / "stage228_knmi_dwd_protection_transfer.json", TABLES / "stage228_knmi_dwd_protection_transfer_station_event_detail.csv"),
        (TABLES / "stage229_knmi_transfer_uncertainty.json", TABLES / "stage229_knmi_transfer_uncertainty_metrics.csv"),
        (TABLES / "stage230_cross_provider_evidence_synthesis.json", TABLES / "stage230_cross_provider_evidence_synthesis_sources.csv"),
        (TABLES / "stage231_cross_provider_robustness.json", TABLES / "stage231_cross_provider_robustness_checks.csv"),
        (TABLES / "stage232_attack_type_robustness.json", TABLES / "stage232_attack_type_robustness_checks.csv"),
        (TABLES / "stage233_valid_attack_correction_audit.json", TABLES / "stage233_valid_attack_correction_audit_targets.csv"),
        (TABLES / "stage234_eccc_protocol_preregistration.json", TABLES / "stage234_eccc_protocol_preregistration_contract.csv"),
        (TABLES / "stage235_eccc_official_download.json", TABLES / "stage235_eccc_official_download_station_year.csv"),
        (TABLES / "stage236_eccc_normalized.json", TABLES / "stage236_eccc_normalized_archive_audit.csv"),
        (TABLES / "stage237_eccc_controlled_attack.json", TABLES / "stage237_eccc_controlled_attack_evaluation.csv"),
        (TABLES / "stage238_eccc_dwd_protection_transfer.json", TABLES / "stage238_eccc_dwd_protection_transfer_station_detail.csv"),
        (TABLES / "stage239_eccc_transfer_uncertainty.json", TABLES / "stage239_eccc_transfer_uncertainty_paired_station.csv"),
        (TABLES / "stage240_nonstuck_balanced_selection.json", TABLES / "stage240_nonstuck_balanced_selection_evaluation.csv"),
        (TABLES / "stage240_nonstuck_balanced_selection.json", TABLES / "stage240_nonstuck_balanced_selection_provider_station.csv"),
        (TABLES / "stage242_multisource_transfer_matrix.json", TABLES / "stage242_multisource_transfer_matrix_evaluation.csv"),
        (TABLES / "stage242_multisource_transfer_matrix.json", TABLES / "stage242_multisource_transfer_matrix_station_detail.csv"),
        (TABLES / "stage242_multisource_transfer_matrix.json", TABLES / "stage242_multisource_transfer_matrix_local_reproduction.csv"),
        (TABLES / "stage243_knmi_negative_case_decomposition.json", TABLES / "stage243_knmi_negative_case_decomposition_station_detail.csv"),
        (TABLES / "stage243_knmi_negative_case_decomposition.json", TABLES / "stage243_knmi_negative_case_decomposition_seed_detail.csv"),
        (TABLES / "stage243_knmi_negative_case_decomposition.json", TABLES / "stage243_knmi_negative_case_decomposition_metric_detail.csv"),
        (TABLES / "stage243_knmi_negative_case_decomposition.json", TABLES / "stage243_knmi_negative_case_decomposition_summary.csv"),
        (TABLES / "stage244_cross_provider_nonstuck_candidate_features.json", TABLES / "stage244_cross_provider_nonstuck_candidate_features_evaluation.csv"),
        (TABLES / "stage244_cross_provider_nonstuck_candidate_features.json", TABLES / "stage244_cross_provider_nonstuck_candidate_features_station_detail.csv"),
        (TABLES / "stage244_cross_provider_nonstuck_candidate_features.json", TABLES / "stage244_cross_provider_nonstuck_candidate_features_type_detail.csv"),
        (TABLES / "stage244_cross_provider_nonstuck_candidate_features.json", TABLES / "stage244_cross_provider_nonstuck_candidate_features_selection.csv"),
        (TABLES / "stage244_cross_provider_nonstuck_candidate_features.json", TABLES / "stage244_cross_provider_nonstuck_candidate_features_evaluation_aggregate.csv"),
        (TABLES / "stage244_cross_provider_nonstuck_candidate_features.json", TABLES / "stage244_cross_provider_nonstuck_candidate_features_provider_station.csv"),
        (TABLES / "stage244_cross_provider_nonstuck_candidate_features.json", TABLES / "stage244_cross_provider_nonstuck_candidate_features_leave_one_provider_out.csv"),
        (TABLES / "stage244_cross_provider_nonstuck_candidate_features.json", TABLES / "stage244_cross_provider_nonstuck_candidate_features_nonstuck_type_detail.csv"),
        (TABLES / "stage244_cross_provider_nonstuck_candidate_features.json", TABLES / "stage244_cross_provider_nonstuck_candidate_features_nonstuck_type_equal.csv"),
        (TABLES / "stage244_cross_provider_nonstuck_candidate_features.json", TABLES / "stage244_cross_provider_nonstuck_candidate_features_selection_counts.csv"),
        (TABLES / "stage245_cross_provider_nonstuck_seed_replication.json", TABLES / "stage245_cross_provider_nonstuck_seed_replication_evaluation.csv"),
        (TABLES / "stage245_cross_provider_nonstuck_seed_replication.json", TABLES / "stage245_cross_provider_nonstuck_seed_replication_station_detail.csv"),
        (TABLES / "stage245_cross_provider_nonstuck_seed_replication.json", TABLES / "stage245_cross_provider_nonstuck_seed_replication_type_detail.csv"),
        (TABLES / "stage245_cross_provider_nonstuck_seed_replication.json", TABLES / "stage245_cross_provider_nonstuck_seed_replication_selection.csv"),
        (TABLES / "stage245_cross_provider_nonstuck_seed_replication.json", TABLES / "stage245_cross_provider_nonstuck_seed_replication_evaluation_aggregate.csv"),
        (TABLES / "stage245_cross_provider_nonstuck_seed_replication.json", TABLES / "stage245_cross_provider_nonstuck_seed_replication_provider_station.csv"),
        (TABLES / "stage245_cross_provider_nonstuck_seed_replication.json", TABLES / "stage245_cross_provider_nonstuck_seed_replication_leave_one_provider_out.csv"),
        (TABLES / "stage245_cross_provider_nonstuck_seed_replication.json", TABLES / "stage245_cross_provider_nonstuck_seed_replication_nonstuck_type_detail.csv"),
        (TABLES / "stage245_cross_provider_nonstuck_seed_replication.json", TABLES / "stage245_cross_provider_nonstuck_seed_replication_nonstuck_type_equal.csv"),
        (TABLES / "stage246_frozen_submission_package.json", TABLES / "stage246_frozen_submission_package_table1_primary_summary.csv"),
        (TABLES / "stage246_frozen_submission_package.json", TABLES / "stage246_frozen_submission_package_table2_cross_provider_transfer.csv"),
        (TABLES / "stage246_frozen_submission_package.json", TABLES / "stage246_frozen_submission_package_table3_attack_type.csv"),
        (TABLES / "stage246_frozen_submission_package.json", TABLES / "stage246_frozen_submission_package_table4_nonstuck_replication.csv"),
        (TABLES / "stage246_frozen_submission_package.json", TABLES / "stage246_frozen_submission_package_table5_recovery_safety.csv"),
        (TABLES / "stage246_frozen_submission_package.json", TABLES / "stage246_frozen_submission_package_table6_provider_balanced_nonstuck.csv"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure1_transfer.png"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure1_transfer.pdf"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure2_attack_type.png"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure2_attack_type.pdf"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure3_nonstuck.png"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure3_nonstuck.pdf"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure4_recovery.png"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure4_recovery.pdf"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure5_provider_balanced_nonstuck.png"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure5_provider_balanced_nonstuck.pdf"),
        (TABLES / "stage247_provider_balanced_nonstuck_audit.json", TABLES / "stage247_provider_balanced_nonstuck_audit_provider_equal.csv"),
        (TABLES / "stage247_provider_balanced_nonstuck_audit.json", TABLES / "stage247_provider_balanced_nonstuck_audit_leave_one_provider_out.csv"),
        (TABLES / "stage247_provider_balanced_nonstuck_audit.json", TABLES / "stage247_provider_balanced_nonstuck_audit_provider_rank.csv"),
        (TABLES / "stage248_cross_provider_fair_baseline_audit.json", TABLES / "stage248_cross_provider_fair_baseline_audit_evaluation.csv"),
        (TABLES / "stage248_cross_provider_fair_baseline_audit.json", TABLES / "stage248_cross_provider_fair_baseline_audit_evaluation_aggregate.csv"),
        (TABLES / "stage248_cross_provider_fair_baseline_audit.json", TABLES / "stage248_cross_provider_fair_baseline_audit_station_detail.csv"),
        (TABLES / "stage248_cross_provider_fair_baseline_audit.json", TABLES / "stage248_cross_provider_fair_baseline_audit_type_detail.csv"),
        (TABLES / "stage248_cross_provider_fair_baseline_audit.json", TABLES / "stage248_cross_provider_fair_baseline_audit_type_aggregate.csv"),
        (TABLES / "stage248_cross_provider_fair_baseline_audit.json", TABLES / "stage248_cross_provider_fair_baseline_audit_selection.csv"),
        (TABLES / "stage248_cross_provider_fair_baseline_audit.json", TABLES / "stage248_cross_provider_fair_baseline_audit_provider_equal.csv"),
        (TABLES / "stage248_cross_provider_fair_baseline_audit.json", TABLES / "stage248_cross_provider_fair_baseline_audit_comparisons.csv"),
        (TABLES / "stage248_cross_provider_fair_baseline_audit.json", TABLES / "stage248_cross_provider_fair_baseline_audit_provider_comparison.csv"),
        (TABLES / "stage248_cross_provider_fair_baseline_audit.json", TABLES / "stage248_cross_provider_fair_baseline_audit_paired_win.csv"),
        (TABLES / "stage246_frozen_submission_package.json", TABLES / "stage246_frozen_submission_package_table7_fair_baseline.csv"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure6_fair_baseline.png"),
        (TABLES / "stage246_frozen_submission_package.json", REPORTS / "figures" / "stage246_frozen_submission_package_figure6_fair_baseline.pdf"),
        (TABLES / "stage249_jisa_submission_gate.json", TABLES / "stage249_jisa_submission_gate_requirements.csv"),
        (TABLES / "stage250_jisa_risk_closure_audit.json", TABLES / "stage250_jisa_risk_closure_audit_risks.csv"),
        (TABLES / "stage251_jisa_stable_submission_objective.json", TABLES / "stage251_jisa_stable_submission_objective_risk_ceilings.csv"),
        (TABLES / "stage251_jisa_stable_submission_objective.json", TABLES / "stage251_jisa_stable_submission_objective_workstreams.csv"),
        (TABLES / "stage252_authorized_testbed_analysis.json", TABLES / "stage252_authorized_testbed_analysis_runs.csv"),
        (TABLES / "stage253_residual_risk_optimization_plan.json", TABLES / "stage253_residual_risk_optimization_plan_rows.csv"),
        (TABLES / "stage254_ocsvm_training_scale_sensitivity.json", TABLES / "stage254_ocsvm_training_scale_sensitivity_detail.csv"),
        (TABLES / "stage254_ocsvm_training_scale_sensitivity.json", TABLES / "stage254_ocsvm_training_scale_sensitivity_aggregate.csv"),
        (TABLES / "stage254_ocsvm_training_scale_sensitivity.json", TABLES / "stage254_ocsvm_training_scale_sensitivity_provider_equal.csv"),
        (TABLES / "stage254_ocsvm_training_scale_sensitivity.json", TABLES / "stage254_ocsvm_training_scale_sensitivity_fits.csv"),
        (TABLES / "stage255_online_causal_testbed.json", TABLES / "stage255_online_causal_testbed_runs.csv"),
        (TABLES / "stage255_online_causal_testbed.json", TABLES / "stage255_online_causal_testbed_recovery.csv"),
        (TABLES / "stage255_online_causal_testbed.json", TABLES / "stage255_online_causal_testbed_aggregate.csv"),
        (TABLES / "stage256_lightweight_strong_baselines.json", TABLES / "stage256_lightweight_strong_baselines_evaluation.csv"),
        (TABLES / "stage256_lightweight_strong_baselines.json", TABLES / "stage256_lightweight_strong_baselines_evaluation_aggregate.csv"),
        (TABLES / "stage256_lightweight_strong_baselines.json", TABLES / "stage256_lightweight_strong_baselines_provider_equal.csv"),
        (TABLES / "stage256_lightweight_strong_baselines.json", TABLES / "stage256_lightweight_strong_baselines_comparisons.csv"),
        (TABLES / "stage256_lightweight_strong_baselines.json", TABLES / "stage256_lightweight_strong_baselines_leave_one_provider_out.csv"),
        (TABLES / "stage256_lightweight_strong_baselines.json", TABLES / "stage256_lightweight_strong_baselines_selection.csv"),
        (TABLES / "stage256_lightweight_strong_baselines.json", TABLES / "stage256_lightweight_strong_baselines_station_detail.csv"),
        (TABLES / "stage256_lightweight_strong_baselines.json", TABLES / "stage256_lightweight_strong_baselines_type_detail.csv"),
        (TABLES / "stage256_lightweight_strong_baselines.json", TABLES / "stage256_lightweight_strong_baselines_type_aggregate.csv"),
        (TABLES / "stage257_fmi_protocol_preregistration.json", TABLES / "stage257_fmi_protocol_preregistration_stations.csv"),
        (TABLES / "stage257_fmi_protocol_preregistration.json", TABLES / "stage257_fmi_protocol_preregistration_station_metadata.xml"),
        (TABLES / "stage258_fmi_official_download.json", TABLES / "stage258_fmi_official_download_detail.csv"),
        (TABLES / "stage258_fmi_official_download.json", TABLES / "stage258_fmi_official_download_station_audit.csv"),
        (TABLES / "stage258_fmi_official_download.json", TABLES / "stage258_fmi_official_download_normalized.parquet"),
        (TABLES / "stage259_fmi_locked_transfer.json", TABLES / "stage259_fmi_locked_transfer_evaluation.csv"),
        (TABLES / "stage259_fmi_locked_transfer.json", TABLES / "stage259_fmi_locked_transfer_provider_equal.csv"),
        (TABLES / "stage259_fmi_locked_transfer.json", TABLES / "stage259_fmi_locked_transfer_comparisons.csv"),
        (TABLES / "stage259_fmi_locked_transfer.json", TABLES / "stage259_fmi_locked_transfer_selection.csv"),
        (TABLES / "stage259_fmi_locked_transfer.json", TABLES / "stage259_fmi_locked_transfer_station_detail.csv"),
        (TABLES / "stage259_fmi_locked_transfer.json", TABLES / "stage259_fmi_locked_transfer_type_detail.csv"),
        (TABLES / "stage259_fmi_locked_transfer.json", TABLES / "stage259_fmi_locked_transfer_leave_one_provider_out.csv"),
        (TABLES / "stage260_reviewer_risk_closure_audit.json", TABLES / "stage260_reviewer_risk_closure_audit.csv"),
        (TABLES / "stage261_causal_offline_recovery_contrast.json", TABLES / "stage261_causal_offline_recovery_contrast_detail.csv"),
        (TABLES / "stage261_causal_offline_recovery_contrast.json", TABLES / "stage261_causal_offline_recovery_contrast_aggregate.csv"),
        (TABLES / "stage262_security_domain_baselines.json", TABLES / "stage262_security_domain_baselines_evaluation.csv"),
        (TABLES / "stage262_security_domain_baselines.json", TABLES / "stage262_security_domain_baselines_evaluation_aggregate.csv"),
        (TABLES / "stage262_security_domain_baselines.json", TABLES / "stage262_security_domain_baselines_station_detail.csv"),
        (TABLES / "stage262_security_domain_baselines.json", TABLES / "stage262_security_domain_baselines_type_detail.csv"),
        (TABLES / "stage262_security_domain_baselines.json", TABLES / "stage262_security_domain_baselines_type_aggregate.csv"),
        (TABLES / "stage262_security_domain_baselines.json", TABLES / "stage262_security_domain_baselines_selection.csv"),
        (TABLES / "stage262_security_domain_baselines.json", TABLES / "stage262_security_domain_baselines_provider_equal.csv"),
        (TABLES / "stage262_security_domain_baselines.json", TABLES / "stage262_security_domain_baselines_comparisons.csv"),
        (TABLES / "stage262_security_domain_baselines.json", TABLES / "stage262_security_domain_baselines_leave_one_provider_out.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_pooled.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_station_equal.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_provider_equal.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_station_direction.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_attack_type.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_power_mde.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_event_definition_audit.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_low_far_and_f1_optimum.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_comparisons.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_leave_one_provider_out.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_bootstrap_structure.csv"),
        (TABLES / "stage263_unified_statistical_report.json", TABLES / "stage263_unified_statistical_report_pooled_cross_source.csv"),
        (
            TABLES / "stage275_external_preregistration_lineage_audit.json",
            TABLES / "stage275_external_preregistration_lineage_audit_providers.csv",
        ),
        (
            TABLES / "stage276_release_reproduction_input_audit.json",
            TABLES / "stage276_release_reproduction_input_audit.csv",
        ),
        (
            TABLES / "stage277_baseline_formality_matrix.json",
            TABLES / "stage277_baseline_formality_matrix.csv",
        ),
        (
            TABLES / "stage278_confirmatory_test_package_contract.json",
            TABLES / "stage278_confirmatory_test_package_contract_requirements.csv",
        ),
        (
            TABLES / "stage279_metric_schema_audit.json",
            TABLES / "stage279_metric_schema_audit_fields.csv",
        ),
        (
            TABLES / "stage279_metric_schema_audit.json",
            TABLES / "stage279_metric_schema_audit_missing.csv",
        ),
        (
            TABLES / "stage280_nonstuck_stability_audit.json",
            TABLES / "stage280_nonstuck_stability_audit_provider_type.csv",
        ),
        (
            TABLES / "stage280_nonstuck_stability_audit.json",
            TABLES / "stage280_nonstuck_stability_audit_seed_direction.csv",
        ),
        (
            TABLES / "stage280_nonstuck_stability_audit.json",
            TABLES / "stage280_nonstuck_stability_audit_type_summary.csv",
        ),
        (
            TABLES / "stage280_nonstuck_stability_audit.json",
            TABLES / "stage280_nonstuck_stability_audit_leave_one_provider.csv",
        ),
        (
            TABLES / "stage281_submission_blocker_register.json",
            TABLES / "stage281_submission_blocker_register.csv",
        ),
        (
            TABLES / "stage282_clean_environment_probe_audit.json",
            TABLES / "stage282_clean_environment_probe_audit.csv",
        ),
        (
            TABLES / "stage283_public_ics_cps_source_audit.json",
            TABLES / "stage283_public_ics_cps_source_audit.csv",
        ),
        (
            TABLES / "stage284_swat_author_data_download.json",
            TABLES / "stage284_swat_author_data_download.csv",
        ),
        (
            TABLES / "stage285_swat_usad_protocol_preregistration.json",
            TABLES / "stage287_swat_usad_evidence_upgrade_audit.json",
        ),
        (
            TABLES / "stage286_swat_usad_author_evaluation.json",
            TABLES / "stage286_swat_usad_author_evaluation_test_metrics.csv",
        ),
        (
            TABLES / "stage287_swat_usad_evidence_upgrade_audit.json",
            TABLES / "stage287_swat_usad_evidence_upgrade_audit_checks.csv",
        ),
        (
            TABLES / "stage288_swat_usad_resource_audit.json",
            TABLES / "stage288_swat_usad_resource_audit_methods.csv",
        ),
        (
            TABLES / "stage290_same_contract_baseline_resource_budget.json",
            TABLES / "stage290_same_contract_baseline_resource_budget.csv",
        ),
        (
            TABLES / "stage291_unified_win_loss_matrix.json",
            TABLES / "stage291_unified_win_loss_matrix.csv",
        ),
        (
            TABLES / "stage291_unified_win_loss_matrix.json",
            TABLES / "stage291_unified_win_loss_matrix_ledger.csv",
        ),
        (
            TABLES / "stage291_unified_win_loss_matrix.json",
            TABLES / "stage291_unified_win_loss_matrix_provider_equal.csv",
        ),
        (
            TABLES / "stage293_confirmatory_package_freeze_audit.json",
            TABLES / "stage293_confirmatory_package_freeze_audit_checks.csv",
        ),
        (
            TABLES / "stage293_confirmatory_package_freeze_audit.json",
            TABLES
            / "stage293_confirmatory_package_freeze_audit_freeze_files.csv",
        ),
        (
            TABLES / "stage294_japan_isd_protocol_preregistration.json",
            TABLES
            / "stage294_japan_isd_protocol_preregistration_stations.csv",
        ),
        (
            TABLES / "stage294_japan_isd_protocol_preregistration.json",
            TABLES
            / "stage294_japan_isd_protocol_preregistration_metadata_source.json",
        ),
        (
            TABLES / "stage296_japan_isd_official_download.json",
            TABLES / "stage296_japan_isd_official_download_detail.csv",
        ),
        (
            TABLES / "stage296_japan_isd_official_download.json",
            TABLES / "stage296_japan_isd_official_download_normalized.parquet",
        ),
        (
            TABLES / "stage296_japan_isd_official_download.json",
            TABLES / "stage296_japan_isd_official_download_station_audit.csv",
        ),
        (
            TABLES / "stage297_japan_isd_locked_transfer.json",
            TABLES / "stage297_japan_isd_locked_transfer_evaluation.csv",
        ),
        (
            TABLES / "stage297_japan_isd_locked_transfer.json",
            TABLES / "stage297_japan_isd_locked_transfer_provider_equal.csv",
        ),
        (
            TABLES / "stage297_japan_isd_locked_transfer.json",
            TABLES / "stage297_japan_isd_locked_transfer_comparisons.csv",
        ),
        (
            TABLES / "stage297_japan_isd_locked_transfer.json",
            TABLES / "stage297_japan_isd_locked_transfer_selection.csv",
        ),
        (
            TABLES / "stage298_japan_transfer_failure_decomposition.json",
            TABLES / "stage298_japan_transfer_failure_decomposition_comparisons.csv",
        ),
        (
            TABLES / "stage298_japan_transfer_failure_decomposition.json",
            TABLES / "stage298_japan_transfer_failure_decomposition_failed_comparisons.csv",
        ),
        (
            TABLES / "stage298_japan_transfer_failure_decomposition.json",
            TABLES / "stage298_japan_transfer_failure_decomposition_failed_seed_directions.csv",
        ),
        (
            TABLES / "stage298_japan_transfer_failure_decomposition.json",
            TABLES / "stage298_japan_transfer_failure_decomposition_method_summary.csv",
        ),
        (
            TABLES / "stage302_japan_ocsvm_budget_grid.json",
            TABLES / "stage302_japan_ocsvm_budget_grid_fit_rows.csv",
        ),
        (
            TABLES / "stage302_japan_ocsvm_budget_grid.json",
            TABLES / "stage302_japan_ocsvm_budget_grid_evaluation.csv",
        ),
        (
            TABLES / "stage302_japan_ocsvm_budget_grid.json",
            TABLES / "stage302_japan_ocsvm_budget_grid_aggregate.csv",
        ),
        (
            TABLES / "stage302_japan_ocsvm_budget_grid.json",
            TABLES / "stage302_japan_ocsvm_budget_grid_comparisons.csv",
        ),
        (
            TABLES / "stage299_method_resource_coverage_audit.json",
            TABLES / "stage299_method_resource_coverage_audit_methods.csv",
        ),
        (
            TABLES / "stage299_method_resource_coverage_audit.json",
            TABLES / "stage299_method_resource_coverage_audit_coverage.csv",
        ),
        (
            TABLES / "stage300_method_resource_evidence_upgrade.json",
            TABLES / "stage300_method_resource_evidence_upgrade_trust.csv",
        ),
        (
            TABLES / "stage300_method_resource_evidence_upgrade.json",
            TABLES / "stage300_method_resource_evidence_upgrade_testbed.csv",
        ),
        (
            TABLES / "stage303_independent_provider_source_feasibility.json",
            TABLES / "stage303_independent_provider_source_feasibility.csv",
        ),
        (
            TABLES / "stage304_met_norway_protocol_preregistration.json",
            TABLES
            / "stage304_met_norway_protocol_preregistration_stations.csv",
        ),
        (
            TABLES / "stage304_met_norway_protocol_preregistration.json",
            TABLES
            / "stage304_met_norway_protocol_preregistration_metadata_source.json",
        ),
        (
            TABLES / "stage306_met_norway_official_download.json",
            TABLES / "stage306_met_norway_official_download_detail.csv",
        ),
        (
            TABLES / "stage306_met_norway_official_download.json",
            TABLES
            / "stage306_met_norway_official_download_normalized.parquet",
        ),
        (
            TABLES / "stage306_met_norway_official_download.json",
            TABLES / "stage306_met_norway_official_download_station_audit.csv",
        ),
        (
            TABLES / "stage307_met_norway_locked_transfer.json",
            TABLES / "stage307_met_norway_locked_transfer_evaluation.csv",
        ),
        (
            TABLES / "stage307_met_norway_locked_transfer.json",
            TABLES / "stage307_met_norway_locked_transfer_provider_equal.csv",
        ),
        (
            TABLES / "stage307_met_norway_locked_transfer.json",
            TABLES / "stage307_met_norway_locked_transfer_comparisons.csv",
        ),
        (
            TABLES / "stage307_met_norway_locked_transfer.json",
            TABLES / "stage307_met_norway_locked_transfer_selection.csv",
        ),
        (
            TABLES / "stage308_met_norway_evidence_boundary_audit.json",
            TABLES / "stage308_met_norway_evidence_boundary_audit_checks.csv",
        ),
        (
            TABLES / "stage308_met_norway_evidence_boundary_audit.json",
            TABLES
            / "stage308_met_norway_evidence_boundary_audit_adverse_comparisons.csv",
        ),
        (
            TABLES / "stage313_cross_locked_test_win_loss_matrix.json",
            TABLES / "stage313_cross_locked_test_win_loss_matrix.csv",
        ),
        (
            TABLES / "stage313_cross_locked_test_win_loss_matrix.json",
            TABLES / "stage313_cross_locked_test_win_loss_matrix_ledger.csv",
        ),
        (
            TABLES / "stage313_cross_locked_test_win_loss_matrix.json",
            TABLES
            / "stage313_cross_locked_test_win_loss_matrix_provider_equal.csv",
        ),
        (
            TABLES / "stage313_cross_locked_test_win_loss_matrix.json",
            TABLES
            / "stage313_cross_locked_test_win_loss_matrix_direction_stability.csv",
        ),
        (
            TABLES / "stage313_cross_locked_test_win_loss_matrix.json",
            TABLES
            / "stage313_cross_locked_test_win_loss_matrix_nonstuck_type.csv",
        ),
        (
            TABLES / "stage313_cross_locked_test_win_loss_matrix.json",
            TABLES
            / "stage313_cross_locked_test_win_loss_matrix_nonstuck_cross_network.csv",
        ),
        (
            TABLES / "stage314_claim_ceiling_and_submission_scope_audit.json",
            TABLES
            / "stage314_claim_ceiling_and_submission_scope_audit_claim_matrix.csv",
        ),
        (
            TABLES / "stage314_claim_ceiling_and_submission_scope_audit.json",
            TABLES
            / "stage314_claim_ceiling_and_submission_scope_audit_p0_blockers.csv",
        ),
        (
            TABLES / "stage314_claim_ceiling_and_submission_scope_audit.json",
            TABLES
            / "stage314_claim_ceiling_and_submission_scope_audit_wording_map.csv",
        ),
        (
            TABLES / "stage314_claim_ceiling_and_submission_scope_audit.json",
            TABLES
            / "stage314_claim_ceiling_and_submission_scope_audit_forbidden_claims.csv",
        ),
        (
            TABLES / "stage315_noaa_2025_temporal_data_audit.json",
            TABLES
            / "stage315_noaa_2025_temporal_data_audit_normalized.parquet",
        ),
        (
            TABLES / "stage315_noaa_2025_temporal_data_audit.json",
            TABLES
            / "stage315_noaa_2025_temporal_data_audit_station_year.csv",
        ),
        (
            TABLES / "stage315_noaa_2025_temporal_locked_transfer.json",
            TABLES
            / "stage315_noaa_2025_temporal_locked_transfer_evaluation.csv",
        ),
        (
            TABLES / "stage315_noaa_2025_temporal_locked_transfer.json",
            TABLES
            / "stage315_noaa_2025_temporal_locked_transfer_provider_equal.csv",
        ),
        (
            TABLES / "stage315_noaa_2025_temporal_locked_transfer.json",
            TABLES
            / "stage315_noaa_2025_temporal_locked_transfer_comparisons.csv",
        ),
        (
            TABLES / "stage315_noaa_2025_temporal_locked_transfer.json",
            TABLES
            / "stage315_noaa_2025_temporal_locked_transfer_selection.csv",
        ),
        (
            TABLES / "stage315_noaa_2025_temporal_locked_transfer.json",
            TABLES
            / "stage315_noaa_2025_temporal_locked_transfer_station_detail.csv",
        ),
        (
            TABLES / "stage315_noaa_2025_temporal_locked_transfer.json",
            TABLES
            / "stage315_noaa_2025_temporal_locked_transfer_type_detail.csv",
        ),
        (
            TABLES / "stage316_noaa_2025_failure_decomposition.json",
            TABLES
            / "stage316_noaa_2025_failure_decomposition_failed_comparisons.csv",
        ),
        (
            TABLES / "stage316_noaa_2025_failure_decomposition.json",
            TABLES
            / "stage316_noaa_2025_failure_decomposition_attack_type_summary.csv",
        ),
        (
            TABLES / "stage316_noaa_2025_failure_decomposition.json",
            TABLES
            / "stage316_noaa_2025_failure_decomposition_station_summary.csv",
        ),
        (
            TABLES / "stage316_noaa_2025_failure_decomposition.json",
            TABLES
            / "stage316_noaa_2025_failure_decomposition_nonstuck_comparison.csv",
        ),
    ]
    pair_failures = [str(path) for pair in required_pairs for path in pair if not path.is_file() or path.stat().st_size == 0]
    failures = sorted(set(missing + duplicate_evidence_ids + pair_failures))
    if declared_evidence_items is not None and int(declared_evidence_items) != len(evidence_rows):
        failures.append(f"stage29 declared evidence_items={declared_evidence_items} but rows={len(evidence_rows)}")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED" if not failures else "FAIL",
        "protocol": "扫描TrustMeteo-QC的结果表、报告、src和tests源码，记录大小、SHA256和JSON状态；逐条检查阶段29证据矩阵引用的产物存在性和证据ID唯一性；阶段109–111、阶段218–239以及阶段262–263要求JSON/CSV成对存在。阶段112自身结果产物不纳入哈希清单，避免自引用循环。",
        "evidence_matrix": str(EVIDENCE),
        "evidence_rows": len(evidence_rows),
        "declared_evidence_items": declared_evidence_items,
        "artifact_files": len(records),
        "missing_evidence_artifacts": sorted(set(missing)),
        "duplicate_evidence_ids": duplicate_evidence_ids,
        "required_pair_failures": pair_failures,
        "failures": failures,
        "artifacts": records,
        "evidence_resolution": resolved,
        "limitations": [
            "清单证明文件存在性和可追溯哈希，不证明实验性能或统计结论。",
            "阶段112自身不纳入哈希清单；重新生成清单后应保存新的JSON/CSV/报告版本。",
        ],
    }
    (TABLES / "stage112_full_artifact_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(TABLES / "stage112_full_artifact_manifest.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 阶段112 TrustMeteo-QC全产物可追溯清单",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 证据矩阵条目：`{len(evidence_rows)}`",
        f"- 纳入清单的产物文件：`{len(records)}`",
        f"- 缺失证据产物：`{len(set(missing))}`",
        f"- 重复证据ID：`{len(duplicate_evidence_ids)}`",
        "",
        "该清单记录结果表、报告和源码文件的相对路径、字节数、SHA256和JSON状态；阶段112自身结果产物不纳入清单以避免哈希自引用。",
        "",
        "| 相对路径 | 类型 | 字节数 | SHA256 | JSON状态 |",
        "|---|---|---:|---|---|",
    ]
    lines.extend(
        f"| {item['relative_path']} | {item['kind']} | {item['bytes']} | `{item['sha256']}` | {item['json_status']} |"
        for item in records
    )
    if failures:
        lines += ["", "## 失败项", ""]
        lines.extend(f"- {item}" for item in failures)
    lines += ["", "该清单用于复核和归档，不替代性能、统计或真实攻击证据。"]
    (REPORTS / "stage112_full_artifact_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "evidence_rows": len(evidence_rows), "artifact_files": len(records), "failures": len(failures)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
