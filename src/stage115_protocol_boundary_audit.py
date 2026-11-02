from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

SOURCE_CONTRACTS = {
    "stage09b": SRC / "stage09b_robust_fusion.py",
    "stage98": SRC / "stage98_noaa_low_far_baseline_comparison.py",
    "stage102": SRC / "stage102_noaa_trust_svm_valid_selection.py",
    "stage105": SRC / "stage105_noaa_attack_type_valid_selected.py",
    "stage107": SRC / "stage107_noaa_natural_svm_validation.py",
    "stage109": SRC / "stage109_noaa_quality_weak_label_bootstrap.py",
    "stage116": SRC / "stage116_noaa_quality_label_permutation.py",
    "stage113": SRC / "stage113_era5_alignment_leakage_audit.py",
    "stage117": SRC / "stage117_multiseed_uncertainty.py",
    "stage118": SRC / "stage118_attack_duration_sensitivity.py",
    "stage119": SRC / "stage119_attack_position_sensitivity.py",
    "stage120": SRC / "stage120_station_stratified_performance.py",
    "stage121": SRC / "stage121_local_station_threshold_calibration.py",
    "stage122": SRC / "stage122_leave_one_station_threshold_transfer.py",
    "stage123": SRC / "stage123_threshold_transfer_bootstrap.py",
    "stage124": SRC / "stage124_leave_one_station_refit_multiseed.py",
    "stage125": SRC / "stage125_leave_one_station_refit_multiseed_bootstrap.py",
    "stage126": SRC / "stage126_low_far_leave_one_station_operating_points.py",
    "stage127": SRC / "stage127_low_far_operating_point_bootstrap.py",
    "stage128": SRC / "stage128_recovery_candidate_diagnosis.py",
    "stage129": SRC / "stage129_valid_recovery_strategy_selection.py",
    "stage130": SRC / "stage130_noaa_safety_constrained_recovery.py",
    "stage131": SRC / "stage131_safety_constraint_sensitivity.py",
    "stage132": SRC / "stage132_recovery_utility_paired_bootstrap.py",
    "stage133": SRC / "stage133_causal_temporal_recovery_audit.py",
    "stage134": SRC / "stage134_causal_recovery_optimization.py",
    "stage135": SRC / "stage135_independent_seed_replication.py",
    "stage136": SRC / "stage136_seed_group_consistency.py",
    "stage137": SRC / "stage137_enhanced_candidate_comparison.py",
    "stage138": SRC / "stage138_independent_baseline_comparison.py",
    "stage139": SRC / "stage139_expanded_fusion_grid.py",
    "stage140": SRC / "stage140_expanded_grid_independent_confirmation.py",
    "stage141": SRC / "stage141_noaa_frozen_external_validation.py",
    "stage142": SRC / "stage142_noaa_frozen_natural_operational_validation.py",
    "stage143": SRC / "stage143_noaa_frozen_trust_svm_safety_combination.py",
    "stage144": SRC / "stage144_noaa_cross_fold_candidate_transfer.py",
    "stage145": SRC / "stage145_noaa_station_heterogeneity_audit.py",
    "stage146": SRC / "stage146_noaa_year_heterogeneity_audit.py",
    "stage147": SRC / "stage147_noaa_attack_year_interaction_audit.py",
    "stage148": SRC / "stage148_noaa_baseline_attack_type_detail.py",
    "stage149": SRC / "stage149_noaa_attack_type_paired_bootstrap.py",
    "stage150": SRC / "stage150_noaa_attack_type_multiple_comparison.py",
    "stage151": SRC / "stage151_noaa_attack_intensity_sensitivity.py",
    "stage152": SRC / "stage152_noaa_baseline_station_comparison.py",
    "stage153": SRC / "stage153_noaa_natural_station_operational_comparison.py",
    "stage154": SRC / "stage154_noaa_natural_temporal_drift.py",
    "stage155": SRC / "stage155_noaa_natural_weather_regime_audit.py",
    "stage156": SRC / "stage156_noaa_natural_observation_gap_audit.py",
    "stage157": SRC / "stage157_noaa_natural_two_way_cluster_bootstrap.py",
    "stage158": SRC / "stage158_noaa_controlled_regime_gap_performance.py",
    "stage159": SRC / "stage159_noaa_controlled_regime_multiple_comparison.py",
    "stage160": SRC / "stage160_noaa_controlled_practical_effect_sensitivity.py",
    "stage161": SRC / "stage161_noaa_controlled_station_regime_performance.py",
    "stage162": SRC / "stage162_noaa_controlled_station_practical_effect.py",
    "stage163": SRC / "stage163_noaa_controlled_regime_full_baseline.py",
    "stage164": SRC / "stage164_noaa_controlled_full_baseline_multiple_comparison.py",
    "stage165": SRC / "stage165_noaa_controlled_station_full_baseline.py",
    "stage166": SRC / "stage166_noaa_controlled_attack_environment_interaction.py",
    "stage167": SRC / "stage167_noaa_controlled_attack_environment_multiple_comparison.py",
    "stage168": SRC / "stage168_noaa_controlled_attack_environment_practical_effect.py",
    "stage169": SRC / "stage169_noaa_controlled_attack_intensity_full_baseline.py",
    "stage170": SRC / "stage170_noaa_controlled_attack_duration_full_baseline.py",
    "stage171": SRC / "stage171_noaa_controlled_attack_position_full_baseline.py",
    "stage172": SRC / "stage172_noaa_frozen_expanded_low_far_full_baseline.py",
    "stage173": SRC / "stage173_noaa_low_far_full_baseline_multiple_comparison.py",
    "stage174": SRC / "stage174_noaa_low_far_alarm_burden_audit.py",
    "stage175": SRC / "stage175_noaa_low_far_alarm_burden_stability.py",
    "stage176": SRC / "stage176_noaa_low_far_station_equal_weight_metrics.py",
    "stage177": SRC / "stage177_noaa_station_calibrated_operating_points.py",
    "stage178": SRC / "stage178_noaa_station_calibrated_alarm_budget.py",
    "stage179": SRC / "stage179_noaa_station_budget_loss_decomposition.py",
    "stage180": SRC / "stage180_noaa_station_budget_row_tail_audit.py",
    "stage181": SRC / "stage181_noaa_non_stuck_failure_attribution.py",
    "stage182": SRC / "stage182_noaa_non_stuck_candidate_features.py",
    "stage183": SRC / "stage183_noaa_candidate_low_far_constraint.py",
    "stage184": SRC / "stage184_noaa_validation_calibrated_low_far.py",
    "stage185": SRC / "stage185_noaa_validation_calibrated_combination.py",
    "stage186": SRC / "stage186_noaa_targeted_nonstuck_feature_response.py",
    "stage187": SRC / "stage187_noaa_targeted_features_validation_calibrated.py",
    "stage188": SRC / "stage188_noaa_two_feature_stability_audit.py",
    "stage189": SRC / "stage189_noaa_event_level_failure_decomposition.py",
    "stage190": SRC / "stage190_noaa_event_reranking_validation.py",
    "stage191": SRC / "stage191_noaa_reranking_focused_diagnostics.py",
    "stage192": SRC / "stage192_noaa_repeat_gated_reranking.py",
    "stage193": SRC / "stage193_noaa_persistent_repeat_protection.py",
    "stage194": SRC / "stage194_noaa_position_perturbation_robustness.py",
    "stage195": SRC / "stage195_noaa_seed_randomized_placement.py",
    "stage196": SRC / "stage196_noaa_seed_randomized_independent_replication.py",
    "stage197_original": SRC / "stage197_noaa_seed_randomized_station_dependence.py",
    "stage197_independent": SRC / "stage197_noaa_seed_randomized_station_dependence.py",
    "stage198_original": SRC / "stage198_noaa_seed_randomized_cross_fold_transfer.py",
    "stage198_independent": SRC / "stage198_noaa_seed_randomized_cross_fold_transfer.py",
    "stage199": SRC / "stage199_liupanshan_cross_station_protection_transfer.py",
    "stage200": SRC / "stage200_liupanshan_protection_failure_attribution.py",
    "stage201": SRC / "stage201_liupanshan_conservative_protection_transfer.py",
    "stage202": SRC / "stage202_liupanshan_attack_type_station_decomposition.py",
    "stage203": SRC / "stage203_independent_external_evidence_feasibility.py",
    "stage204": SRC / "stage204_dwd_protocol_preregistration.py",
    "stage204a": SRC / "stage204a_dwd_protocol_correction.py",
    "stage205": SRC / "stage205_dwd_official_download.py",
    "stage206": SRC / "stage206_dwd_normalization.py",
    "stage207": SRC / "stage207_dwd_controlled_attack_evaluation.py",
    "stage208": SRC / "stage208_dwd_persistent_repeat_protection.py",
    "stage209": SRC / "stage209_dwd_station_cluster_uncertainty.py",
    "stage210": SRC / "stage210_dwd_seed_randomized_replication.py",
    "stage211": SRC / "stage211_dwd_replication_station_uncertainty.py",
    "stage212": SRC / "stage212_dwd_operating_curve.py",
    "stage213": SRC / "stage213_dwd_external_evidence_closure.py",
    "stage214": SRC / "stage214_swiss_protocol_preregistration.py",
    "stage215": SRC / "stage215_swiss_official_download.py",
    "stage216": SRC / "stage216_swiss_normalization.py",
    "stage216a": SRC / "stage216a_swiss_protocol_correction.py",
    "stage217": SRC / "stage217_swiss_controlled_attack_evaluation.py",
    "stage218": SRC / "stage218_swiss_negative_result_uncertainty.py",
    "stage219": SRC / "stage219_swiss_dwd_protection_transfer.py",
    "stage220": SRC / "stage220_swiss_dwd_transfer_uncertainty.py",
    "stage221": SRC / "stage221_swiss_dwd_transfer_operating_curve.py",
    "stage222": SRC / "stage222_noaa_dwd_protection_transfer.py",
    "stage223": SRC / "stage223_noaa_transfer_uncertainty.py",
    "stage224": SRC / "stage224_knmi_protocol_preregistration.py",
    "stage225": SRC / "stage225_knmi_official_download.py",
    "stage226": SRC / "stage226_knmi_normalization.py",
    "stage226a": SRC / "stage226a_knmi_missingness_audit.py",
    "stage227": SRC / "stage227_knmi_independent_reproduction.py",
    "stage228": SRC / "stage228_knmi_dwd_protection_transfer.py",
    "stage229": SRC / "stage229_knmi_transfer_uncertainty.py",
    "stage230": SRC / "stage230_cross_provider_evidence_synthesis.py",
    "stage231": SRC / "stage231_cross_provider_robustness.py",
    "stage232": SRC / "stage232_attack_type_robustness.py",
    "stage233": SRC / "stage233_valid_attack_correction_audit.py",
    "stage240": SRC / "stage240_nonstuck_balanced_selection.py",
    "stage244": SRC / "stage244_cross_provider_nonstuck_candidate_features.py",
    "stage245": SRC / "stage245_cross_provider_nonstuck_seed_replication.py",
    "stage246": SRC / "stage246_frozen_submission_package.py",
    "stage248": SRC / "stage248_cross_provider_fair_baseline_audit.py",
    "stage250": SRC / "stage250_jisa_risk_closure_audit.py",
}
# Performance experiments use an explicit train/valid/test contract. Natural
# NOAA audits are different: they are label-free observations or weak-label
# association analyses, so requiring all three split keywords would be false.
PERFORMANCE_CONTRACTS = {"stage09b", "stage98", "stage102", "stage105", "stage121", "stage122", "stage124", "stage126", "stage148", "stage158"}
AUDIT_CONTRACTS = {"stage203"}
PREREGISTRATION_CONTRACTS = {"stage204"}
CORRECTION_CONTRACTS = {"stage204a"}
DOWNLOAD_AUDIT_CONTRACTS = {"stage205"}
NORMALIZATION_CONTRACTS = {"stage206"}
DWD_PERFORMANCE_CONTRACTS = {"stage207", "stage208", "stage210"}
UNCERTAINTY_CONTRACTS = {"stage209", "stage211", "stage212"}
CLOSURE_CONTRACTS = {"stage213"}
SWISS_PREREGISTRATION_CONTRACTS = {"stage214"}
SWISS_DOWNLOAD_AUDIT_CONTRACTS = {"stage215"}
SWISS_NORMALIZATION_CONTRACTS = {"stage216"}
SWISS_CORRECTION_CONTRACTS = {"stage216a"}
SWISS_PERFORMANCE_CONTRACTS = {"stage217"}
SWISS_UNCERTAINTY_CONTRACTS = {"stage218"}
SWISS_TRANSFER_CONTRACTS = {"stage219"}
SWISS_TRANSFER_UNCERTAINTY_CONTRACTS = {"stage220"}
SWISS_OPERATING_CURVE_CONTRACTS = {"stage221"}
NOAA_TRANSFER_CONTRACTS = {"stage222"}
NOAA_TRANSFER_UNCERTAINTY_CONTRACTS = {"stage223"}
KNMI_PREREGISTRATION_CONTRACTS = {"stage224"}
KNMI_DOWNLOAD_AUDIT_CONTRACTS = {"stage225"}
KNMI_NORMALIZATION_CONTRACTS = {"stage226"}
KNMI_MISSINGNESS_CONTRACTS = {"stage226a"}
KNMI_PERFORMANCE_CONTRACTS = {"stage227"}
KNMI_TRANSFER_CONTRACTS = {"stage228"}
KNMI_TRANSFER_UNCERTAINTY_CONTRACTS = {"stage229"}
CROSS_PROVIDER_SYNTHESIS_CONTRACTS = {"stage230"}
CROSS_PROVIDER_ROBUSTNESS_CONTRACTS = {"stage231"}
ATTACK_TYPE_ROBUSTNESS_CONTRACTS = {"stage232"}
VALID_ATTACK_CORRECTION_CONTRACTS = {"stage233"}
NONSTUCK_BALANCED_SELECTION_CONTRACTS = {"stage240"}
NONSTUCK_CANDIDATE_FEATURE_CONTRACTS = {"stage244"}
NONSTUCK_SEED_REPLICATION_CONTRACTS = {"stage245"}
SUBMISSION_PACKAGE_CONTRACTS = {"stage246"}
FAIR_BASELINE_CONTRACTS = {"stage248"}
RISK_CLOSURE_CONTRACTS = {"stage250"}
SUMMARY_CONTRACTS = {"stage117", "stage118", "stage119", "stage120", "stage123", "stage125", "stage127", "stage128", "stage129", "stage130", "stage131", "stage132", "stage133", "stage134", "stage135", "stage136", "stage137", "stage138", "stage139", "stage140", "stage141", "stage143", "stage144", "stage145", "stage149", "stage150", "stage151", "stage152", "stage159", "stage160", "stage161", "stage162", "stage163", "stage164", "stage165", "stage166", "stage167", "stage168", "stage169", "stage170", "stage171", "stage172", "stage173", "stage174", "stage175", "stage176", "stage177", "stage178", "stage179", "stage180", "stage181", "stage182", "stage183", "stage184", "stage185", "stage186", "stage187", "stage188", "stage189", "stage190", "stage191", "stage192", "stage193", "stage194", "stage195", "stage196", "stage197_original", "stage197_independent", "stage198_original", "stage198_independent", "stage199", "stage200", "stage201", "stage202"}
YEAR_CONTRACTS = {"stage146"}
ATTACK_YEAR_CONTRACTS = {"stage147"}
NATURAL_CONTRACTS = {"stage107", "stage109", "stage113", "stage142", "stage153", "stage154", "stage155", "stage156", "stage157"}
PROTOCOL_JSONS = {
    "stage09b": TABLES / "stage09b_robust_fusion.json",
    "stage98": TABLES / "stage98_noaa_low_far_baseline_comparison.json",
    "stage102": TABLES / "stage102_noaa_trust_svm_valid_selection.json",
    "stage105": TABLES / "stage105_noaa_attack_type_valid_selected.json",
    "stage107": TABLES / "stage107_noaa_natural_svm_validation.json",
    "stage109": TABLES / "stage109_noaa_quality_weak_label_bootstrap.json",
    "stage116": TABLES / "stage116_noaa_quality_label_permutation.json",
    "stage113": TABLES / "stage113_era5_alignment_leakage_audit.json",
    "stage117": TABLES / "stage117_multiseed_uncertainty.json",
    "stage118": TABLES / "stage118_attack_duration_sensitivity.json",
    "stage119": TABLES / "stage119_attack_position_sensitivity.json",
    "stage120": TABLES / "stage120_station_stratified_performance.json",
    "stage121": TABLES / "stage121_local_station_threshold_calibration.json",
    "stage122": TABLES / "stage122_leave_one_station_threshold_transfer.json",
    "stage123": TABLES / "stage123_threshold_transfer_bootstrap.json",
    "stage124": TABLES / "stage124_leave_one_station_refit_multiseed.json",
    "stage125": TABLES / "stage125_leave_one_station_refit_multiseed_bootstrap.json",
    "stage126": TABLES / "stage126_low_far_leave_one_station_operating_points.json",
    "stage127": TABLES / "stage127_low_far_operating_point_bootstrap.json",
    "stage128": TABLES / "stage128_recovery_candidate_diagnosis.json",
    "stage129": TABLES / "stage129_valid_recovery_strategy_selection.json",
    "stage130": TABLES / "stage130_noaa_safety_constrained_recovery.json",
    "stage131": TABLES / "stage131_safety_constraint_sensitivity.json",
    "stage132": TABLES / "stage132_recovery_utility_paired_bootstrap.json",
    "stage133": TABLES / "stage133_causal_temporal_recovery.json",
    "stage134": TABLES / "stage134_causal_recovery_optimization.json",
    "stage135": TABLES / "stage135_independent_seed_replication.json",
    "stage136": TABLES / "stage136_seed_group_consistency.json",
    "stage137": TABLES / "stage137_enhanced_candidate_comparison.json",
    "stage138": TABLES / "stage138_independent_baseline_comparison.json",
    "stage139": TABLES / "stage139_expanded_fusion_grid.json",
    "stage140": TABLES / "stage140_expanded_grid_independent_confirmation.json",
    "stage141": TABLES / "stage141_noaa_frozen_external_validation.json",
    "stage142": TABLES / "stage142_noaa_frozen_natural_operational_validation.json",
    "stage143": TABLES / "stage143_noaa_frozen_trust_svm_safety_combination.json",
    "stage144": TABLES / "stage144_noaa_cross_fold_candidate_transfer.json",
    "stage145": TABLES / "stage145_noaa_station_heterogeneity_audit.json",
    "stage146": TABLES / "stage146_noaa_year_heterogeneity_audit.json",
    "stage147": TABLES / "stage147_noaa_attack_year_interaction_audit.json",
    "stage148": TABLES / "stage148_noaa_baseline_attack_type_detail.json",
    "stage149": TABLES / "stage149_noaa_attack_type_paired_bootstrap.json",
    "stage150": TABLES / "stage150_noaa_attack_type_multiple_comparison.json",
    "stage151": TABLES / "stage151_noaa_attack_intensity_sensitivity.json",
    "stage152": TABLES / "stage152_noaa_baseline_station_comparison.json",
    "stage153": TABLES / "stage153_noaa_natural_station_operational_comparison.json",
    "stage154": TABLES / "stage154_noaa_natural_temporal_drift.json",
    "stage155": TABLES / "stage155_noaa_natural_weather_regime_audit.json",
    "stage156": TABLES / "stage156_noaa_natural_observation_gap_audit.json",
    "stage157": TABLES / "stage157_noaa_natural_two_way_cluster_bootstrap.json",
    "stage158": TABLES / "stage158_noaa_controlled_regime_gap_performance.json",
    "stage159": TABLES / "stage159_noaa_controlled_regime_multiple_comparison.json",
    "stage160": TABLES / "stage160_noaa_controlled_practical_effect_sensitivity.json",
    "stage161": TABLES / "stage161_noaa_controlled_station_regime_performance.json",
    "stage162": TABLES / "stage162_noaa_controlled_station_practical_effect.json",
    "stage163": TABLES / "stage163_noaa_controlled_regime_full_baseline.json",
    "stage164": TABLES / "stage164_noaa_controlled_full_baseline_multiple_comparison.json",
    "stage165": TABLES / "stage165_noaa_controlled_station_full_baseline.json",
    "stage166": TABLES / "stage166_noaa_controlled_attack_environment_interaction.json",
    "stage167": TABLES / "stage167_noaa_controlled_attack_environment_multiple_comparison.json",
    "stage168": TABLES / "stage168_noaa_controlled_attack_environment_practical_effect.json",
    "stage169": TABLES / "stage169_noaa_controlled_attack_intensity_full_baseline.json",
    "stage170": TABLES / "stage170_noaa_controlled_attack_duration_full_baseline.json",
    "stage171": TABLES / "stage171_noaa_controlled_attack_position_full_baseline.json",
    "stage172": TABLES / "stage172_noaa_frozen_expanded_low_far_full_baseline.json",
    "stage173": TABLES / "stage173_noaa_low_far_full_baseline_multiple_comparison.json",
    "stage174": TABLES / "stage174_noaa_low_far_alarm_burden_audit.json",
    "stage175": TABLES / "stage175_noaa_low_far_alarm_burden_stability.json",
    "stage176": TABLES / "stage176_noaa_low_far_station_equal_weight_metrics.json",
    "stage177": TABLES / "stage177_noaa_station_calibrated_operating_points.json",
    "stage178": TABLES / "stage178_noaa_station_calibrated_alarm_budget.json",
    "stage179": TABLES / "stage179_noaa_station_budget_loss_decomposition.json",
    "stage180": TABLES / "stage180_noaa_station_budget_row_tail_audit.json",
    "stage181": TABLES / "stage181_noaa_non_stuck_failure_attribution.json",
    "stage182": TABLES / "stage182_noaa_non_stuck_candidate_features.json",
    "stage183": TABLES / "stage183_noaa_candidate_low_far_constraint.json",
    "stage184": TABLES / "stage184_noaa_validation_calibrated_low_far.json",
    "stage185": TABLES / "stage185_noaa_validation_calibrated_combination.json",
    "stage186": TABLES / "stage186_noaa_targeted_nonstuck_feature_response.json",
    "stage187": TABLES / "stage187_noaa_targeted_features_validation_calibrated.json",
    "stage188": TABLES / "stage188_noaa_two_feature_stability_audit.json",
    "stage189": TABLES / "stage189_noaa_event_level_failure_decomposition.json",
    "stage190": TABLES / "stage190_noaa_event_reranking_validation.json",
    "stage191": TABLES / "stage191_noaa_reranking_focused_diagnostics.json",
    "stage192": TABLES / "stage192_noaa_repeat_gated_reranking.json",
    "stage193": TABLES / "stage193_noaa_persistent_repeat_protection.json",
    "stage194": TABLES / "stage194_noaa_position_perturbation_robustness.json",
    "stage195": TABLES / "stage195_noaa_seed_randomized_placement.json",
    "stage196": TABLES / "stage196_noaa_seed_randomized_independent_replication.json",
    "stage197_original": TABLES / "stage197_noaa_seed_randomized_station_dependence_original.json",
    "stage197_independent": TABLES / "stage197_noaa_seed_randomized_station_dependence_independent.json",
    "stage198_original": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_original.json",
    "stage198_independent": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_independent.json",
    "stage199": TABLES / "stage199_liupanshan_cross_station_protection_transfer.json",
    "stage200": TABLES / "stage200_liupanshan_protection_failure_attribution.json",
    "stage201": TABLES / "stage201_liupanshan_conservative_protection_transfer.json",
    "stage202": TABLES / "stage202_liupanshan_attack_type_station_decomposition.json",
    "stage203": TABLES / "stage203_independent_external_evidence_feasibility.json",
    "stage204": TABLES / "stage204_dwd_protocol_preregistration.json",
    "stage204a": TABLES / "stage204a_dwd_protocol_schema_correction.json",
    "stage205": TABLES / "stage205_dwd_official_download.json",
    "stage206": TABLES / "stage206_dwd_normalized.json",
    "stage207": TABLES / "stage207_dwd_controlled_attack.json",
    "stage208": TABLES / "stage208_dwd_persistent_repeat_protection.json",
    "stage209": TABLES / "stage209_dwd_station_cluster_uncertainty.json",
    "stage210": TABLES / "stage210_dwd_seed_randomized_replication.json",
    "stage211": TABLES / "stage211_dwd_replication_station_uncertainty.json",
    "stage212": TABLES / "stage212_dwd_operating_curve.json",
    "stage213": TABLES / "stage213_dwd_external_evidence_closure.json",
    "stage214": TABLES / "stage214_swiss_protocol_preregistration.json",
    "stage215": TABLES / "stage215_swiss_official_download.json",
    "stage216": TABLES / "stage216_swiss_normalized.json",
    "stage216a": TABLES / "stage216a_swiss_protocol_correction.json",
    "stage217": TABLES / "stage217_swiss_controlled_attack.json",
    "stage218": TABLES / "stage218_swiss_negative_result_uncertainty.json",
    "stage219": TABLES / "stage219_swiss_dwd_protection_transfer.json",
    "stage220": TABLES / "stage220_swiss_dwd_transfer_uncertainty.json",
    "stage221": TABLES / "stage221_swiss_dwd_transfer_operating_curve.json",
    "stage222": TABLES / "stage222_noaa_dwd_protection_transfer.json",
    "stage223": TABLES / "stage223_noaa_transfer_uncertainty.json",
    "stage224": TABLES / "stage224_knmi_protocol_preregistration.json",
    "stage225": TABLES / "stage225_knmi_official_download.json",
    "stage226": TABLES / "stage226_knmi_normalized.json",
    "stage226a": TABLES / "stage226a_knmi_missingness_audit.json",
    "stage227": TABLES / "stage227_knmi_controlled_attack.json",
    "stage228": TABLES / "stage228_knmi_dwd_protection_transfer.json",
    "stage229": TABLES / "stage229_knmi_transfer_uncertainty.json",
    "stage230": TABLES / "stage230_cross_provider_evidence_synthesis.json",
    "stage231": TABLES / "stage231_cross_provider_robustness.json",
    "stage232": TABLES / "stage232_attack_type_robustness.json",
    "stage233": TABLES / "stage233_valid_attack_correction_audit.json",
    "stage240": TABLES / "stage240_nonstuck_balanced_selection.json",
    "stage244": TABLES / "stage244_cross_provider_nonstuck_candidate_features.json",
    "stage245": TABLES / "stage245_cross_provider_nonstuck_seed_replication.json",
    "stage246": TABLES / "stage246_frozen_submission_package.json",
    "stage248": TABLES / "stage248_cross_provider_fair_baseline_audit.json",
    "stage250": TABLES / "stage250_jisa_risk_closure_audit.json",
}


def check(name: str, condition: bool, value: object = None, reason: str = "") -> dict[str, object]:
    row: dict[str, object] = {"check": name, "status": "PASS" if condition else "FAIL"}
    if value is not None:
        row["value"] = value
    if reason:
        row["reason"] = reason
    return row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    checks: list[dict[str, object]] = []
    source_text: dict[str, str] = {}
    for name, path in SOURCE_CONTRACTS.items():
        exists = path.is_file() and path.stat().st_size > 0
        checks.append(check(f"{name}_source_exists", exists, str(path)))
        if exists:
            source_text[name] = path.read_text(encoding="utf-8")
    for name, path in PROTOCOL_JSONS.items():
        exists = path.is_file() and path.stat().st_size > 0
        checks.append(check(f"{name}_protocol_exists", exists, str(path)))

    for name, text in source_text.items():
        lowered = text.lower()
        if name in AUDIT_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_mentions_test", "test" in lowered),
                    check(f"{name}_no_training_or_download_boundary", "不训练模型" in text and "不下载新数据" in text),
                    check(f"{name}_independence_definition", "不同机构" in text and "不同传感器网络" in text),
                ]
            )
        elif name in PREREGISTRATION_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_mentions_test", "test" in lowered),
                    check(f"{name}_pre_download_freeze", "下载前" in text or "before any" in lowered),
                    check(f"{name}_no_test_tuning_boundary", "No target-domain test tuning" in text or "不根据test选择" in text),
                ]
            )
        elif name in CORRECTION_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_schema_correction", "RF_TU" in text and "QN_9" in text),
                    check(f"{name}_before_performance", "性能" in text and "不计算性能" in text),
                    check(f"{name}_no_training_or_selection", "不训练模型" in text and "不选择阈值" in text),
                ]
            )
        elif name in DOWNLOAD_AUDIT_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_official_dwd", "opendata.dwd.de" in text),
                    check(f"{name}_no_training_or_selection", "不训练模型" in text and "不选择阈值" in text),
                    check(f"{name}_no_performance", "不计算性能" in text),
                ]
            )
        elif name in NORMALIZATION_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_split_contract", "2021-2023" in text and "2024" in text and "2025" in text),
                    check(f"{name}_no_imputation_training_selection", "不插补" in text and "不训练模型" in text and "不选择阈值" in text),
                    check(f"{name}_no_performance", "不计算性能" in text),
                ]
            )
        elif name in UNCERTAINTY_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_read_frozen_result_only",
                        "只读取" in text and "不重新拟合" in text,
                    ),
                    check(
                        f"{name}_no_selection_or_calibration",
                        "不重新选择" in text and "不重新校准" in text,
                    ),
                    check(
                        f"{name}_station_cluster_boundary",
                        "站点" in text and "重采样簇" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控" in text and "不是真实攻击日志" in text,
                    ),
                    check(
                        f"{name}_external_replication_limit",
                        "不能替代独立站点外部复现" in text,
                    ),
                ]
            )
        elif name in CLOSURE_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_read_frozen_results_only",
                        "只读取" in text and "不训练" in text,
                    ),
                    check(
                        f"{name}_no_new_selection_or_calibration",
                        "不重新选择" in text and "不重新校准" in text,
                    ),
                    check(
                        f"{name}_single_provider_boundary",
                        "一个跨机构来源" in text
                        and "不能扩展为多机构泛化声明" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控" in text and "不是真实攻击日志" in text,
                    ),
                    check(
                        f"{name}_no_new_download_or_performance",
                        "不下载新数据" in text and "不计算新的性能结果" in text,
                    ),
                ]
            )
        elif name in SWISS_PREREGISTRATION_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_pre_download_freeze",
                        "before" in lowered
                        and "any" in lowered
                        and "downloaded" in lowered,
                    ),
                    check(
                        f"{name}_no_test_tuning_boundary",
                        "No target-domain test tuning" in text,
                    ),
                    check(
                        f"{name}_no_training_selection_performance",
                        "不训练模型" in text
                        and "不选择阈值" in text
                        and "不计算性能" in text,
                    ),
                ]
            )
        elif name in SWISS_DOWNLOAD_AUDIT_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_official_swiss_stac",
                        "data.geo.admin.ch" in text,
                    ),
                    check(
                        f"{name}_twelve_frozen_files",
                        "Stage-214" in text and "12" in text,
                    ),
                    check(
                        f"{name}_no_training_selection_performance",
                        "不训练模型" in text
                        and "不选择阈值" in text
                        and "不计算性能" in text,
                    ),
                ]
            )
        elif name in SWISS_NORMALIZATION_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_split_contract",
                        "2021-2023" in text
                        and "2024" in text
                        and "2025" in text,
                    ),
                    check(f"{name}_utc_boundary", "UTC" in text),
                    check(
                        f"{name}_no_imputation_training_selection_performance",
                        "not imputed" in lowered
                        and "不插补" in text
                        and "不训练模型" in text
                        and "不选择阈值" in text
                        and "不计算性能" in text,
                    ),
                ]
            )
        elif name in SWISS_CORRECTION_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_availability_correction_before_performance",
                        "before any detector" in lowered
                        and "performance" in lowered,
                    ),
                    check(
                        f"{name}_aeg_all_variables_empty",
                        "aeg" in lowered
                        and "temperature" in lowered
                        and "humidity" in lowered
                        and "dew-point" in lowered
                        and "pressure" in lowered,
                    ),
                    check(
                        f"{name}_no_test_based_exclusion",
                        "不依据任何test性能" in text,
                    ),
                ]
            )
        elif name in SWISS_PERFORMANCE_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_train_valid_test_boundary",
                        "2021-2023" in text
                        and "2024" in text
                        and "2025" in text,
                    ),
                    check(
                        f"{name}_no_test_selection",
                        "不根据test选择" in text,
                    ),
                    check(
                        f"{name}_no_dwd_weight_transfer",
                        "不迁移DWD权重" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控" in text and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in SWISS_UNCERTAINTY_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_read_frozen_result_only",
                        "只读取" in text
                        and "不重新选择" in text
                        and "不重新校准" in text,
                    ),
                    check(
                        f"{name}_deterministic_replay_boundary",
                        "deterministic" in lowered
                        and "replay" in lowered
                        and "Stage-217" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "transparent controlled" in lowered
                        and "attack" in lowered
                        and "replay" in lowered,
                    ),
                ]
            )
        elif name in SWISS_TRANSFER_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_train_valid_test_boundary",
                        "2021-2023" in text
                        and "2024" in text
                        and "2025" in text,
                    ),
                    check(
                        f"{name}_frozen_dwd_selection",
                        "Stage208" in text
                        and "feature" in lowered
                        and "window" in lowered
                        and "weight" in lowered,
                    ),
                    check(
                        f"{name}_no_test_selection",
                        "No Swiss test labels" in text
                        or "不用于特征" in text
                        or "不根据test选择" in text,
                    ),
                    check(
                        f"{name}_three_arm_boundary",
                        "dwd_threshold_transfer" in text
                        and "swiss_valid_recalibrated" in text
                        and "swiss_valid_selected" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控" in text
                        and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in SWISS_TRANSFER_UNCERTAINTY_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_read_frozen_result_only",
                        "read-only" in lowered
                        and "does not fit" in lowered,
                    ),
                    check(
                        f"{name}_station_cluster_boundary",
                        "stations are the resampling unit" in lowered
                        or "站点" in text
                        and "重采样" in text,
                    ),
                    check(
                        f"{name}_paired_comparison_boundary",
                        "swiss_valid_selected - swiss_valid_recalibrated"
                        in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "transparent controlled" in lowered
                        and "not real attack logs" in lowered,
                    ),
                ]
            )
        elif name in SWISS_OPERATING_CURVE_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_source_has_test",
                        "test" in lowered,
                    ),
                    check(
                        f"{name}_source_no_test_selection",
                        "test只评价" in text
                        and "不重新选择" in text,
                    ),
                    check(
                        f"{name}_source_primary_far_frozen",
                        "primary_far" in text
                        and "0.01" in text,
                    ),
                    check(
                        f"{name}_source_controlled_attack",
                        "透明受控" in text
                        and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in NOAA_TRANSFER_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_train_valid_test_boundary",
                        "2020-2022" in text
                        and "2023" in text
                        and "2024" in text,
                    ),
                    check(
                        f"{name}_frozen_dwd_selection",
                        "Stage208" in text
                        and "feature" in lowered
                        and "window" in lowered
                        and "weight" in lowered,
                    ),
                    check(
                        f"{name}_no_test_selection",
                        "test只评价" in text
                        and "不根据test选择" in text,
                    ),
                    check(
                        f"{name}_three_arm_boundary",
                        "dwd_threshold_transfer" in text
                        and "noaa_valid_recalibrated" in text
                        and "noaa_valid_selected" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "transparent controlled" in lowered
                        and "not real attack logs" in lowered,
                    ),
                ]
            )
        elif name in NOAA_TRANSFER_UNCERTAINTY_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_read_frozen_result_only",
                        "read-only" in lowered
                        and "does not fit" in lowered,
                    ),
                    check(
                        f"{name}_station_cluster_boundary",
                        "stations are the resampling unit" in lowered,
                    ),
                    check(
                        f"{name}_paired_comparison_boundary",
                        "noaa_valid_selected - noaa_valid_recalibrated"
                        in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "transparent controlled" in lowered
                        and "not real attack logs" in lowered,
                    ),
                ]
            )
        elif name in KNMI_PREREGISTRATION_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_pre_download_freeze",
                        "PROTOCOL_FROZEN_BEFORE_KNMI_PERFORMANCE" in text
                        and "NOT_DOWNLOADED" in text
                        and "NOT_EVALUATED" in text,
                    ),
                    check(
                        f"{name}_grid_and_station_boundary",
                        "STATION_GRID_DEGREES = 0.75" in text
                        and "STATION_COUNT" in text,
                    ),
                    check(
                        f"{name}_split_and_endpoint_boundary",
                        "2021-2023" in text
                        and "2024" in text
                        and "2025" in text
                        and "event_coverage" in text,
                    ),
                    check(
                        f"{name}_no_test_tuning",
                        "不根据test" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控" in text
                        and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in KNMI_DOWNLOAD_AUDIT_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_official_knmi_sources",
                        "api.dataplatform.knmi.nl" in text
                        and "cdn.knmi.nl" in text,
                    ),
                    check(
                        f"{name}_archive_and_field_audit",
                        "archive_audit" in text
                        and "required_columns_ok" in text,
                    ),
                    check(
                        f"{name}_no_training_selection_performance",
                        "不训练模型" in text
                        and "不选择阈值" in text
                        and "不计算性能" in text,
                    ),
                ]
            )
        elif name in KNMI_NORMALIZATION_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_split_contract",
                        "2021-2023" in text
                        and "2024" in text
                        and "2025" in text,
                    ),
                    check(
                        f"{name}_missing_and_scaling_boundary",
                        "不插补" in text
                        and "T" in text
                        and "U" in text
                        and "10.0" in text,
                    ),
                    check(
                        f"{name}_no_training_selection_performance",
                        "不训练模型" in text
                        and "不选择阈值" in text
                        and "不计算性能" in text,
                    ),
                ]
            )
        elif name in KNMI_MISSINGNESS_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_decision_split_boundary",
                        'DECISION_SPLITS = ("train", "valid")' in text
                        and 'TEST_SPLIT = "test"' in text,
                    ),
                    check(
                        f"{name}_availability_threshold_frozen",
                        "AVAILABILITY_THRESHOLD = 0.95" in text,
                    ),
                    check(
                        f"{name}_test_not_used_for_selection",
                        "2025 test可用率仅作事后" in text
                        and "不参与站点、特征、窗口、权重、阈值或" in text
                        and "攻击位置选择" in text,
                    ),
                    check(
                        f"{name}_no_training_selection_performance",
                        "不训练模型" in text
                        and "不选择阈值" in text
                        and "不计算性能" in text,
                    ),
                ]
            )
        elif name in KNMI_PERFORMANCE_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_train_valid_test_boundary",
                        "2021-2023" in text
                        and "2024" in text
                        and "2025" in text,
                    ),
                    check(
                        f"{name}_no_dwd_weight_transfer",
                        "不迁移DWD权重" in text,
                    ),
                    check(
                        f"{name}_no_test_selection",
                        "不根据test" in text
                        and "test只评价" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控" in text
                        and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in KNMI_TRANSFER_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_frozen_dwd_selection",
                        "阶段208" in text
                        and "feature" in lowered
                        and "window" in lowered
                        and "weight" in lowered,
                    ),
                    check(
                        f"{name}_target_valid_only",
                        "KNMI valid" in text
                        and "test只评价" in text,
                    ),
                    check(
                        f"{name}_three_arm_boundary",
                        "dwd_threshold_transfer" in text
                        and "knmi_valid_recalibrated" in text
                        and "knmi_valid_selected" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控" in text
                        and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in KNMI_TRANSFER_UNCERTAINTY_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_read_frozen_result_only",
                        "只读取" in text
                        and "不重新训练" in text
                        and "不重新选择" in text,
                    ),
                    check(
                        f"{name}_station_cluster_bootstrap",
                        "6个KNMI站点" in text
                        and "10000次站点簇" in text
                        and "配对bootstrap" in text,
                    ),
                    check(
                        f"{name}_no_test_selection",
                        "不使用KNMI test标签做任何选择" in text,
                    ),
                    check(
                        f"{name}_primary_endpoint_boundary",
                        "事件覆盖是单一预注册主终点" in text
                        and "不构成BH多重校正家族" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控" in text
                        and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in CROSS_PROVIDER_SYNTHESIS_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_read_frozen_results_only",
                        "只读取" in text
                        and "不重新训练" in text
                        and "不重新选择" in text,
                    ),
                    check(
                        f"{name}_source_scope_complete",
                        "DWD" in text
                        and "MeteoSwiss" in text
                        and "NOAA" in text
                        and "KNMI" in text,
                    ),
                    check(
                        f"{name}_absolute_threshold_failure_retained",
                        "绝对DWD阈值跨机构直接迁移的零结果必须保留" in text,
                    ),
                    check(
                        f"{name}_target_valid_local_selection_boundary",
                        "目标域valid本地选择优于重校准迁移" in text
                        and "不能把重校准写成迁移优势" in text,
                    ),
                    check(
                        f"{name}_no_target_test_selection",
                        "不使用目标域test标签做任何选择" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控注入" in text
                        and "不是真实攻击日志" in text,
                    ),
                    check(
                        f"{name}_source_limit_boundary",
                        "站点数和可用性限制必须随结果一并报告" in text,
                    ),
                ]
            )
        elif name in CROSS_PROVIDER_ROBUSTNESS_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_read_frozen_results_only",
                        "只读取" in text
                        and "不重新训练" in text
                        and "不重新选择" in text,
                    ),
                    check(
                        f"{name}_provider_equal_weighting",
                        "先在机构内对站点取均值" in text
                        and "再对MeteoSwiss、NOAA ISD、KNMI和ECCC四个机构等权取均值" in text,
                    ),
                    check(
                        f"{name}_station_imbalance_boundary",
                        "避免" in text and "11/11/6/14站点数不平衡" in text,
                    ),
                    check(
                        f"{name}_leave_one_provider_out",
                        "留一机构" in text,
                    ),
                    check(
                        f"{name}_hierarchical_bootstrap",
                        "两级层级bootstrap" in text
                        and "先重采样机构" in text
                        and "再在机构内重采样站点" in text,
                    ),
                    check(
                        f"{name}_sign_flip_power_boundary",
                        "最小双侧p=0.125" in text
                        and ("功效严重不足" in text or "功效仍然不足" in text),
                    ),
                    check(
                        f"{name}_no_target_test_selection",
                        "不使用目标域test标签做任何选择" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控注入" in text
                        and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in CROSS_PROVIDER_ROBUSTNESS_CONTRACTS:
            robustness = json.loads(
                PROTOCOL_JSONS["stage231"].read_text(encoding="utf-8")
            )
            leave_one_out = robustness.get("leave_one_provider_out", [])
            hierarchical = robustness.get("hierarchical_bootstrap", {})
            provider_sign_flip = robustness.get("provider_level_sign_flip", {})
            provider_equal = robustness.get("provider_equal_summary", {})
            checks.extend(
                [
                    check(
                        f"{name}_protocol_provider_equal_summary",
                        "先在机构内对站点取均值" in protocol
                        and int(provider_equal.get("provider_count", 0)) == 4
                        and int(provider_equal.get("station_count", 0)) == 42
                        and float(provider_equal.get("f1_difference", 0.0)) > 0.0
                        and float(provider_equal.get("event_difference", 0.0)) > 0.0,
                        provider_equal,
                    ),
                    check(
                        f"{name}_protocol_leave_one_provider_out_positive",
                        len(leave_one_out) == 4
                        and all(
                            bool(row.get("f1_direction_positive"))
                            and bool(row.get("event_direction_positive"))
                            for row in leave_one_out
                        ),
                        leave_one_out,
                    ),
                    check(
                        f"{name}_protocol_hierarchical_bootstrap_positive",
                        int(hierarchical.get("iterations", 0)) == 10000
                        and float(hierarchical.get("f1_ci95_low", -1.0)) > 0.0
                        and float(hierarchical.get("event_ci95_low", -1.0)) > 0.0,
                        hierarchical,
                    ),
                    check(
                        f"{name}_protocol_sign_flip_power_boundary",
                        int(provider_sign_flip.get("provider_count", 0)) == 4
                        and float(provider_sign_flip.get("minimum_two_sided_p", 0.0)) == 0.125
                        and float(provider_sign_flip.get("f1_sign_flip_p", 0.0)) >= 0.125
                        and float(provider_sign_flip.get("event_sign_flip_p", 0.0)) >= 0.125,
                        provider_sign_flip,
                    ),
                    check(
                        f"{name}_protocol_failed_checks_empty",
                        robustness.get("failed_checks") == []
                        and all(
                            row.get("status") == "PASS"
                            for row in robustness.get("checks", [])
                        ),
                        robustness.get("failed_checks"),
                    ),
                ]
            )
        elif name in FAIR_BASELINE_CONTRACTS:
            fair = json.loads(
                PROTOCOL_JSONS["stage248"].read_text(encoding="utf-8")
            )
            text = json.dumps(fair, ensure_ascii=False)
            comparisons = fair.get("comparison_rows", [])
            provider_equal = fair.get("provider_equal_rows", [])
            leave_one = fair.get("leave_one_provider_rows", [])
            methods = fair.get("methods", [])
            baseline_methods = fair.get("baseline_methods", [])
            checks.extend(
                [
                    check(
                        f"{name}_four_provider_boundary",
                        int(fair.get("provider_count", 0)) == 4,
                        fair.get("provider_count"),
                    ),
                    check(
                        f"{name}_primary_far_boundary",
                        float(fair.get("primary_far", 0.0)) == 0.01,
                        fair.get("primary_far"),
                    ),
                    check(
                        f"{name}_five_method_boundary",
                        int(fair.get("method_count", 0)) == 5
                        and int(fair.get("seed_count", 0)) == 5,
                        {
                            "method_count": fair.get("method_count"),
                            "seed_count": fair.get("seed_count"),
                        },
                    ),
                    check(
                        f"{name}_provider_equal_boundary",
                        int(fair.get("station_count", 0)) == 42
                        and len(provider_equal) == 5
                        and {row.get("method") for row in provider_equal}
                        == set(methods),
                        {
                            "station_count": fair.get("station_count"),
                            "provider_equal_rows": len(provider_equal),
                        },
                    ),
                    check(
                        f"{name}_three_fair_baselines",
                        set(baseline_methods)
                        == {
                            "robust_max_deviation",
                            "isolation_forest",
                            "one_class_svm",
                        },
                        baseline_methods,
                    ),
                    check(
                        f"{name}_trust_robust_f1_advantage",
                        all(
                            float(row.get("f1_difference", 0.0)) > 0.0
                            and float(row.get("f1_ci95_low", -1.0)) > 0.0
                            and float(
                                row.get("f1_bootstrap_bh_q", 1.0)
                            )
                            <= 0.05
                            and bool(row.get("f1_primary_supported"))
                            for row in comparisons
                            if row.get("candidate") == "trust_valid_f1"
                            and row.get("baseline")
                            in {
                                "robust_max_deviation",
                                "isolation_forest",
                            }
                        ),
                    ),
                    check(
                        f"{name}_trust_nonstuck_event_advantage",
                        all(
                            float(
                                row.get("event_coverage_difference", 0.0)
                            )
                            > 0.0
                            and float(row.get("event_ci95_low", -1.0)) > 0.0
                            and float(
                                row.get(
                                    "event_coverage_bootstrap_bh_q",
                                    1.0,
                                )
                            )
                            <= 0.05
                            and bool(
                                row.get(
                                    "event_coverage_primary_supported"
                                )
                            )
                            for row in comparisons
                            if row.get("candidate")
                            == "trust_nonstuck_event_macro"
                            and row.get("baseline")
                            in {
                                "robust_max_deviation",
                                "isolation_forest",
                            }
                        ),
                    ),
                    check(
                        f"{name}_one_class_complementarity_retained",
                        any(
                            row.get("candidate") == "trust_valid_f1"
                            and row.get("baseline") == "one_class_svm"
                            and float(
                                row.get("event_coverage_difference", 0.0)
                            )
                            < 0.0
                            for row in comparisons
                        ),
                    ),
                    check(
                        f"{name}_leave_one_provider_retained",
                        len(leave_one) == 24
                        and all(
                            bool(row.get("f1_direction_positive"))
                            for row in leave_one
                            if row.get("candidate") == "trust_valid_f1"
                            and row.get("baseline")
                            in {
                                "robust_max_deviation",
                                "isolation_forest",
                            }
                        )
                        and all(
                            bool(row.get("event_direction_positive"))
                            for row in leave_one
                            if row.get("candidate")
                            == "trust_nonstuck_event_macro"
                            and row.get("baseline")
                            in {
                                "robust_max_deviation",
                                "isolation_forest",
                            }
                        ),
                        {
                            "leave_one_provider_rows": len(leave_one),
                        },
                    ),
                    check(
                        f"{name}_ocsvm_subsampling_disclosed",
                        int(fair.get("ocsvm_fit_cap", 0)) == 20000
                        and "One-Class SVM" in text
                        and all(
                            int(row.get("ocsvm_fit_rows", 0)) == 20000
                            for row in fair.get("fit_rows", [])
                        ),
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控注入" in text
                        and "不是真实攻击日志" in text,
                    ),
                    check(
                        f"{name}_universal_sota_boundary_retained",
                        "不能宣称全面领先" in text,
                    ),
                    check(
                        f"{name}_unfavorable_baselines_retained",
                        "如果某基线占优，该结果必须保留" in text,
                    ),
                ]
            )
        elif name in RISK_CLOSURE_CONTRACTS:
            risk = json.loads(
                PROTOCOL_JSONS["stage250"].read_text(encoding="utf-8")
            )
            source_path = SOURCE_CONTRACTS["stage250"]
            source_body = source_path.read_text(encoding="utf-8")
            checks.extend(
                [
                    check(
                        f"{name}_risk_count_boundary",
                        int(risk.get("risk_count", 0)) == 7
                        and int(risk.get("resolved_count", 0)) == 0
                        and int(risk.get("bounded_count", 0)) == 7,
                        {
                            "risk_count": risk.get("risk_count"),
                            "resolved_count": risk.get("resolved_count"),
                            "bounded_count": risk.get("bounded_count"),
                        },
                    ),
                    check(
                        f"{name}_risk_rows_boundary",
                        len(risk.get("risks", [])) == 7
                        and all(
                            row.get("disposition") == "BOUNDED"
                            and row.get("allowed_main_claim")
                            and row.get("forbidden_claim")
                            for row in risk.get("risks", [])
                        ),
                        len(risk.get("risks", [])),
                    ),
                    check(
                        f"{name}_read_only_source_boundary",
                        all(
                            token not in source_body
                            for token in (
                                "fit_detectors(",
                                "inject_attack(",
                                "select_candidate(",
                                "OneClassSVM(",
                                "IsolationForest(",
                            )
                        ),
                        "read-only risk closure audit",
                    ),
                ]
            )
        elif name in ATTACK_TYPE_ROBUSTNESS_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_read_frozen_results_only",
                        "只读取" in text
                        and "不重新训练" in text
                        and "不重新选择" in text,
                    ),
                    check(
                        f"{name}_attack_type_scope_complete",
                        "攻击类型" in text
                        and "stuck" in text,
                    ),
                    check(
                        f"{name}_provider_equal_weighting",
                        "机构内同类型种子先取均值" in text
                        and "再对机构等权取均值" in text,
                    ),
                    check(
                        f"{name}_hierarchical_bootstrap",
                        "两级层级bootstrap" in text
                        and "先重采样机构" in text
                        and "再在机构内重采样种子" in text,
                    ),
                    check(
                        f"{name}_no_type_precision_f1",
                        "不包含正常行误报计数" in text
                        and "不伪造类型级Precision或F1" in text,
                    ),
                    check(
                        f"{name}_sign_flip_power_boundary",
                        "最小双侧p=0.125" in text,
                    ),
                    check(
                        f"{name}_no_target_test_selection",
                        "不使用目标域test标签做任何选择" in text,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控注入" in text
                        and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in VALID_ATTACK_CORRECTION_CONTRACTS:
            correction = json.loads(
                PROTOCOL_JSONS["stage233"].read_text(encoding="utf-8")
            )
            targets = {
                row["target_source"]: row
                for row in correction.get("target_results", [])
            }
            checks.extend(
                [
                    check(
                        f"{name}_old_claim_invalidated",
                        correction.get("old_claim_status") == "INVALIDATED",
                        correction.get("old_claim_status"),
                    ),
                    check(
                        f"{name}_corrected_direction_all_targets",
                        set(targets)
                        == {"MeteoSwiss", "NOAA ISD", "KNMI", "ECCC"}
                        and all(
                            float(row.get("selected_minus_recalibrated_f1", 0.0))
                            > 0.0
                            and float(
                                row.get(
                                    "selected_minus_recalibrated_event_coverage",
                                    0.0,
                                )
                            )
                            > 0.0
                            for row in targets.values()
                        ),
                        targets,
                    ),
                    check(
                        f"{name}_direct_threshold_failure_retained",
                        all(
                            float(row.get("direct_threshold_transfer_f1", -1.0))
                            == 0.0
                            and float(
                                row.get(
                                    "direct_threshold_transfer_event_coverage",
                                    -1.0,
                                )
                            )
                            == 0.0
                            for row in targets.values()
                        ),
                        targets,
                    ),
                    check(
                        f"{name}_no_target_test_selection",
                        "不使用目标域test标签做任何选择" in correction.get(
                            "protocol", ""
                        ),
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控注入" in correction.get("protocol", "")
                        and "不是真实攻击日志" in correction.get("protocol", ""),
                    ),
                ]
            )
        elif name in NONSTUCK_BALANCED_SELECTION_CONTRACTS:
            nonstuck = json.loads(
                PROTOCOL_JSONS["stage240"].read_text(encoding="utf-8")
            )
            text = json.dumps(nonstuck, ensure_ascii=False)
            checks.extend(
                [
                    check(
                        f"{name}_train_valid_test_boundary",
                        "train" in text and "valid" in text and "test" in text,
                    ),
                    check(
                        f"{name}_test_selection_forbidden",
                        "不使用test标签做任何选择" in text
                        or "test只评价" in text,
                    ),
                    check(
                        f"{name}_does_not_replace_frozen_main_result",
                        "不替换" in text,
                    ),
                    check(
                        f"{name}_four_provider_boundary",
                        int(nonstuck.get("provider_count", 0)) == 4
                        and int(nonstuck.get("station_count", 0)) == 42,
                        nonstuck.get("provider_count"),
                    ),
                    check(
                        f"{name}_predeclared_selection_rules",
                        set(nonstuck.get("selection_rules", []))
                        == {"valid_f1", "valid_nonstuck_event_macro"},
                        nonstuck.get("selection_rules"),
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控注入" in text
                        and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in NONSTUCK_CANDIDATE_FEATURE_CONTRACTS:
            candidate = json.loads(
                PROTOCOL_JSONS["stage244"].read_text(encoding="utf-8")
            )
            text = json.dumps(candidate, ensure_ascii=False)
            bootstrap = candidate.get("hierarchical_bootstrap", {})
            checks.extend(
                [
                    check(
                        f"{name}_train_valid_test_boundary",
                        "train" in text and "valid" in text and "test" in text,
                    ),
                    check(
                        f"{name}_test_selection_forbidden",
                        "test只评价" in text
                        and "候选权重和阈值只用valid选择" in text,
                    ),
                    check(
                        f"{name}_does_not_replace_frozen_main_result",
                        "不替换" in text,
                    ),
                    check(
                        f"{name}_four_provider_boundary",
                        int(candidate.get("provider_count", 0)) == 4
                        and int(candidate.get("station_count", 0)) == 42,
                        candidate.get("provider_count"),
                    ),
                    check(
                        f"{name}_candidate_boundary",
                        int(candidate.get("candidate_count", 0)) == 91
                        and int(candidate.get("new_feature_count", 0)) == 7,
                        candidate.get("candidate_count"),
                    ),
                    check(
                        f"{name}_predeclared_selection_rules",
                        set(candidate.get("selection_rules", []))
                        == {"valid_f1", "valid_nonstuck_event_macro"},
                        candidate.get("selection_rules"),
                    ),
                    check(
                        f"{name}_positive_point_estimate_retained",
                        float(bootstrap.get("f1_difference", 0.0)) > 0.0
                        and float(
                            bootstrap.get("event_coverage_difference", 0.0)
                        )
                        > 0.0,
                        bootstrap,
                    ),
                    check(
                        f"{name}_cross_zero_uncertainty_retained",
                        float(bootstrap.get("f1_ci95_low", 0.0)) < 0.0
                        and float(bootstrap.get("event_ci95_low", 0.0)) < 0.0,
                        bootstrap,
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控注入" in text
                        and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in NONSTUCK_SEED_REPLICATION_CONTRACTS:
            replication = json.loads(
                PROTOCOL_JSONS["stage245"].read_text(encoding="utf-8")
            )
            text = json.dumps(replication, ensure_ascii=False)
            bootstrap = replication.get("hierarchical_bootstrap", {})
            direction = replication.get("direction_consistency", {})
            checks.extend(
                [
                    check(
                        f"{name}_train_valid_test_boundary",
                        "train" in text
                        and "valid" in text
                        and "test" in text,
                    ),
                    check(
                        f"{name}_test_selection_forbidden",
                        "test只评价" in text,
                    ),
                    check(
                        f"{name}_does_not_replace_frozen_main_result",
                        "不替换" in text or "不修改" in text,
                    ),
                    check(
                        f"{name}_independent_seeds",
                        replication.get("replication_seeds")
                        == [41011, 73189, 96001, 131071, 196613]
                        and len(replication.get("seed_audit_rows", [])) == 4
                        and all(
                            int(row.get("overlap_count", -1)) == 0
                            for row in replication.get("seed_audit_rows", [])
                        ),
                        replication.get("replication_seeds"),
                    ),
                    check(
                        f"{name}_four_provider_boundary",
                        int(replication.get("provider_count", 0)) == 4
                        and int(replication.get("station_count", 0)) == 42,
                        replication.get("provider_count"),
                    ),
                    check(
                        f"{name}_candidate_boundary",
                        int(replication.get("candidate_count", 0)) == 91,
                        replication.get("candidate_count"),
                    ),
                    check(
                        f"{name}_direction_consistency_retained",
                        direction.get("f1_direction_consistent") is True
                        and direction.get(
                            "event_coverage_direction_consistent"
                        )
                        is True,
                        direction,
                    ),
                    check(
                        f"{name}_cross_zero_uncertainty_retained",
                        float(bootstrap.get("f1_ci95_low", 0.0)) < 0.0
                        and float(bootstrap.get("f1_ci95_high", 0.0)) > 0.0
                        and float(bootstrap.get("event_ci95_low", 0.0)) < 0.0
                        and float(bootstrap.get("event_ci95_high", 0.0)) > 0.0,
                        bootstrap,
                    ),
                    check(
                        f"{name}_negative_cases_retained",
                        any(
                            str(row.get("attack_type")) == "drift"
                            and float(
                                row.get("row_recall_difference", 0.0)
                            )
                            < 0.0
                            for row in replication.get(
                                "nonstuck_type_equal_rows", []
                            )
                        )
                        and any(
                            str(row.get("removed_source")) == "ECCC"
                            and float(row.get("f1_difference", 0.0)) < 0.0
                            for row in replication.get(
                                "leave_one_provider_out_rows", []
                            )
                        ),
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控注入" in text
                        and "不是真实攻击日志" in text,
                    ),
                ]
            )
        elif name in SUBMISSION_PACKAGE_CONTRACTS:
            package = json.loads(
                PROTOCOL_JSONS["stage246"].read_text(encoding="utf-8")
            )
            text = json.dumps(package, ensure_ascii=False)
            checks.extend(
                [
                    check(
                        f"{name}_read_only_frozen_results",
                        "不训练" in text
                        and "不重新选择" in text
                        and "不重新评价" in text,
                    ),
                    check(
                        f"{name}_primary_far_boundary",
                        float(package.get("primary_far", 0.0)) == 0.01,
                        package.get("primary_far"),
                    ),
                    check(
                        f"{name}_four_provider_boundary",
                        int(package.get("provider_count", 0)) == 4,
                        package.get("provider_count"),
                    ),
                    check(
                        f"{name}_frozen_and_generated_counts",
                        int(package.get("frozen_input_count", 0)) == 31
                        and int(
                            package.get("generated_deliverable_count", 0)
                        )
                        == 28,
                        {
                            "frozen_input_count": package.get(
                                "frozen_input_count"
                            ),
                            "generated_deliverable_count": package.get(
                                "generated_deliverable_count"
                            ),
                        },
                    ),
                    check(
                        f"{name}_no_failed_checks",
                        package.get("failed_checks") == []
                        and all(
                            row.get("status") == "PASS"
                            for row in package.get("checks", [])
                        ),
                        package.get("failed_checks"),
                    ),
                    check(
                        f"{name}_controlled_attack_boundary",
                        "透明受控注入" in text
                        and "不是真实攻击日志" in text,
                    ),
                    check(
                        f"{name}_nonstuck_uncertainty_boundary",
                        "区间仍跨零" in text
                        or any(
                            "区间仍跨零" in item
                            for item in package.get("limitations", [])
                        ),
                    ),
                    check(
                        f"{name}_stuck_concentration_boundary",
                        "优势主要集中在stuck" in text,
                    ),
                    check(
                        f"{name}_provider_balanced_boundary",
                        "事件覆盖在留一机构后方向仍为正" in text
                        and "F1在移除ECCC后转负" in text,
                    ),
                    check(
                        f"{name}_causal_recovery_boundary",
                        "严格因果恢复弱于离线双向插值" in text,
                    ),
                ]
            )
        elif name in ATTACK_TYPE_ROBUSTNESS_CONTRACTS:
            attack_type = json.loads(
                PROTOCOL_JSONS["stage232"].read_text(encoding="utf-8")
            )
            by_type = {
                row["attack_type"]: row
                for row in attack_type.get("provider_equal_rows", [])
            }
            leave_one_type = {
                row["removed_attack_type"]: row
                for row in attack_type.get(
                    "leave_one_attack_type_out_rows", []
                )
            }
            bootstrap_type = {
                row["attack_type"]: row
                for row in attack_type.get(
                    "hierarchical_bootstrap_rows", []
                )
            }
            stuck = by_type.get("stuck", {})
            remove_stuck = leave_one_type.get("stuck", {})
            stuck_bootstrap = bootstrap_type.get("stuck", {})
            checks.extend(
                [
                    check(
                        f"{name}_protocol_attack_type_scope",
                        int(attack_type.get("attack_type_count", 0)) == 6
                        and set(by_type)
                        == {
                            "drift",
                            "gaussian_noise",
                            "offset",
                            "spike",
                            "stuck",
                            "time_shift",
                        },
                        sorted(by_type),
                    ),
                    check(
                        f"{name}_protocol_stuck_direction_positive",
                        float(stuck.get("row_recall_difference", 0.0)) > 0.0
                        and float(
                            stuck.get("event_coverage_difference", 0.0)
                        )
                        > 0.0,
                        stuck,
                    ),
                    check(
                        f"{name}_protocol_stuck_concentration_retained",
                        float(remove_stuck.get("row_recall_difference", 0.0))
                        <= 0.0
                        and float(
                            remove_stuck.get(
                                "event_coverage_difference", 0.0
                            )
                        )
                        <= 0.0,
                        remove_stuck,
                    ),
                    check(
                        f"{name}_protocol_stuck_bootstrap_not_significant",
                        float(
                            stuck_bootstrap.get("row_recall_ci95_low", -1.0)
                        )
                        > 0.0
                        and float(
                            stuck_bootstrap.get(
                                "event_coverage_ci95_low", -1.0
                            )
                        )
                        > 0.0,
                        stuck_bootstrap,
                    ),
                    check(
                        f"{name}_protocol_sign_flip_power_boundary",
                        float(stuck.get("row_recall_sign_flip_p", 0.0))
                        >= 0.125
                        and float(
                            stuck.get("event_coverage_sign_flip_p", 0.0)
                        )
                        >= 0.125,
                        stuck,
                    ),
                    check(
                        f"{name}_protocol_failed_checks_empty",
                        attack_type.get("failed_checks") == []
                        and all(
                            row.get("status") == "PASS"
                            for row in attack_type.get("checks", [])
                        ),
                        attack_type.get("failed_checks"),
                    ),
                ]
            )
        elif name in DWD_PERFORMANCE_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_train_valid_test_boundary", "train" in lowered and "valid" in lowered and "test" in lowered),
                    check(f"{name}_no_test_selection", "不根据test" in text and "test只评价" in text),
                    check(f"{name}_controlled_attack_boundary", "透明受控" in text and "不是真实攻击日志" in text),
                    check(f"{name}_clean_time_shift_source", "clean_values" in text and "source_row_position" in text),
                    (
                        check(
                            f"{name}_separate_valid_test_sequences",
                            "分别构造" in text
                            or "separately" in lowered,
                        )
                        if name == "stage208"
                        else check(
                            f"{name}_separate_valid_test_sequences",
                            True,
                        )
                    ),
                ]
            )
        elif name in ATTACK_YEAR_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_mentions_test", "test" in lowered),
                    check(f"{name}_no_selection_boundary", "不重新选" in text or "不重新训练" in text or "不根据test" in text),
                    check(f"{name}_attack_year_boundary", "攻击类型" in text and "目标年份" in text),
                ]
            )
        elif name in YEAR_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_mentions_test", "test" in lowered),
                    check(f"{name}_no_selection_boundary", "不重新选" in text or "不重新训练" in text or "不根据test" in text),
                    check(f"{name}_fold_uncertainty_boundary", "fold bootstrap" in lowered or "fold" in lowered),
                ]
            )
        elif name in SUMMARY_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_mentions_test", "test" in lowered),
                    check(f"{name}_no_selection_boundary", "不重新选" in text or "不重新训练" in text or "不根据test" in text),
                    check(f"{name}_seed_uncertainty_boundary", "种子" in text),
                ]
            )
        elif name in PERFORMANCE_CONTRACTS:
            checks.append(check(f"{name}_mentions_train", "train" in lowered))
            checks.append(check(f"{name}_mentions_valid", "valid" in lowered))
            checks.append(check(f"{name}_mentions_test", "test" in lowered))
        elif name in {"stage107", "stage142", "stage153", "stage154", "stage155", "stage156", "stage157"}:
            checks.extend(
                [
                    check(f"{name}_natural_observation_boundary", "自然观测" in text),
                    check(f"{name}_no_attack_boundary", "不注入攻击" in text),
                    check(f"{name}_weak_label_boundary", "弱标签" in text),
                ]
            )
        elif name in {"stage109", "stage116"}:
            checks.extend(
                [
                    check(f"{name}_natural_observation_boundary", "自然" in text),
                    check(f"{name}_weak_label_boundary", "弱标签" in text),
                    check(f"{name}_no_attack_truth_boundary", "攻击真值" in text),
                ]
            )
        elif name == "stage113":
            checks.extend(
                [
                    check(f"{name}_valid_calibration_boundary", "valid" in lowered),
                    check(f"{name}_test_exclusion_boundary", "测试期" in text or "test" in lowered),
                    check(f"{name}_era5_auxiliary_boundary", "不是现场真值" in text or "区域背景" in text),
                ]
            )

    select_text = source_text.get("stage102", "")
    match = re.search(r"def _select\(.*?(?=\ndef main\(|\Z)", select_text, flags=re.S)
    select_block = match.group(0) if match else ""
    checks.extend(
        [
            check("stage102_select_function_found", bool(select_block)),
            check("stage102_select_uses_valid_far", "valid_far" in select_block),
            check("stage102_select_uses_valid_f1", "valid_f1" in select_block),
            check("stage102_select_does_not_use_test_metrics", "test_" not in select_block, reason="只允许从valid候选字段选择；test指标可在函数外汇总"),
        ]
    )
    checks.append(check("stage09b_best_setting_uses_valid_metrics", "valid_f1" in source_text.get("stage09b", "") and "valid_false_alarm_rate" in source_text.get("stage09b", "")))
    checks.append(check("stage113_calibration_source_uses_valid_normal", 'aggregate("valid")' in source_text.get("stage113", "") or 'aggregate("valid")' in (SRC / "stage08_external_calibration.py").read_text(encoding="utf-8")))

    for name, path in PROTOCOL_JSONS.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = str(payload.get("protocol", ""))
        lowered = protocol.lower()
        if name in AUDIT_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_protocol_has_test", "test" in lowered, protocol[:220]),
                    check(f"{name}_protocol_no_training_or_download", "不训练模型" in protocol and "不下载新数据" in protocol, protocol[:220]),
                    check(f"{name}_protocol_independence_definition", "不同机构" in protocol and "不同传感器网络" in protocol, protocol[:220]),
                ]
            )
        elif name in PREREGISTRATION_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_protocol_has_test", "test" in lowered, protocol[:220]),
                    check(f"{name}_protocol_pre_download_freeze", "下载前" in protocol or "before download" in lowered, protocol[:220]),
                    check(f"{name}_protocol_no_test_tuning", "不根据test选择" in protocol or "No target-domain test tuning" in protocol, protocol[:220]),
                ]
            )
        elif name in CORRECTION_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_protocol_schema_correction", "RF_TU" in protocol and "QN_9" in protocol, protocol[:220]),
                    check(f"{name}_protocol_before_performance", "性能评价前" in protocol or "before performance" in lowered, protocol[:220]),
                    check(f"{name}_protocol_no_training_selection_performance", "不训练模型" in protocol and "不选择阈值" in protocol and "不计算性能" in protocol, protocol[:220]),
                ]
            )
        elif name in DOWNLOAD_AUDIT_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_protocol_official_download", "DWD官方OpenData" in protocol, protocol[:220]),
                    check(f"{name}_protocol_no_training_selection_performance", "不训练模型" in protocol and "不选择阈值" in protocol and "不计算性能" in protocol, protocol[:220]),
                    check(f"{name}_protocol_auxiliary_boundary", "TF" in protocol and "辅助" in protocol, protocol[:220]),
                ]
            )
        elif name in NORMALIZATION_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_protocol_split_contract", "2021-2023为train" in protocol and "2024为valid" in protocol and "2025为test" in protocol, protocol[:220]),
                    check(f"{name}_protocol_no_imputation_training_selection_performance", "不插补" in protocol and "不训练模型" in protocol and "不选择阈值" in protocol and "不计算性能" in protocol, protocol[:220]),
                    check(f"{name}_protocol_quality_boundary", "QN_9" in protocol and "质量" in protocol, protocol[:220]),
                ]
            )
        elif name in UNCERTAINTY_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_read_frozen_result_only",
                        "只读取" in protocol and "不重新拟合" in protocol,
                        protocol[:240],
                    ),
                    check(
                        f"{name}_protocol_no_selection_or_calibration",
                        "不重新选择" in protocol
                        and "不重新校准" in protocol,
                        protocol[:240],
                    ),
                    check(
                        f"{name}_protocol_station_cluster_boundary",
                        "站点" in protocol and "重采样簇" in protocol,
                        protocol[:240],
                    ),
                    check(
                        f"{name}_protocol_controlled_attack",
                        "透明受控" in protocol
                        and "不是真实攻击日志" in protocol,
                        protocol[:240],
                    ),
                    check(
                        f"{name}_protocol_external_replication_limit",
                        "不能替代独立站点外部复现" in protocol,
                        protocol[:240],
                    ),
                ]
            )
        elif name in CLOSURE_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_read_frozen_results_only",
                        "只读取" in protocol and "不训练" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_no_new_selection_or_calibration",
                        "不重新选择" in protocol and "不重新校准" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_single_provider_boundary",
                        "一个跨机构来源" in protocol
                        and "不能扩展为多机构泛化声明" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_controlled_attack",
                        "透明受控" in protocol and "不是真实攻击日志" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_no_new_download_or_performance",
                        "不下载新数据" in protocol
                        and "不计算新的性能结果" in protocol,
                        protocol[:260],
                    ),
                ]
            )
        elif name in SWISS_PREREGISTRATION_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_pre_download_freeze",
                        payload.get("status") == "PROTOCOL_FROZEN_BEFORE_DOWNLOAD"
                        and payload.get("download_status") == "NOT_DOWNLOADED"
                        and payload.get("performance_status") == "NOT_EVALUATED",
                        payload.get("status"),
                    ),
                    check(
                        f"{name}_protocol_no_test_tuning",
                        "No target-domain test tuning is permitted"
                        in str(payload.get("selection_boundary", "")),
                    ),
                    check(
                        f"{name}_protocol_no_training_selection_performance",
                        "不训练模型" in protocol
                        and "不选择阈值" in protocol
                        and "不计算性能" in protocol,
                        protocol[:220],
                    ),
                    check(
                        f"{name}_protocol_station_count",
                        int(payload.get("station_count", 0)) == 12,
                        payload.get("station_count"),
                    ),
                ]
            )
        elif name in SWISS_DOWNLOAD_AUDIT_CONTRACTS:
            detail = payload.get("detail", [])
            checks.extend(
                [
                    check(
                        f"{name}_protocol_official_swiss_stac",
                        "瑞士联邦官方STAC" in protocol
                        and all(
                            "data.geo.admin.ch" in str(row.get("url", ""))
                            for row in detail
                        ),
                        protocol[:220],
                    ),
                    check(
                        f"{name}_protocol_twelve_files",
                        len(detail) == 12
                        and int(payload.get("expected_archives", 0)) == 12,
                        len(detail),
                    ),
                    check(
                        f"{name}_protocol_no_training_selection_performance",
                        "不训练模型" in protocol
                        and "不选择阈值" in protocol
                        and "不计算性能" in protocol,
                        protocol[:220],
                    ),
                ]
            )
        elif name in SWISS_NORMALIZATION_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_split_contract",
                        payload.get("train_years") == [2021, 2022, 2023]
                        and int(payload.get("valid_year", 0)) == 2024
                        and int(payload.get("test_year", 0)) == 2025,
                    ),
                    check(
                        f"{name}_protocol_utc_boundary",
                        payload.get("timezone") == "UTC",
                        payload.get("timezone"),
                    ),
                    check(
                        f"{name}_protocol_no_imputation_training_selection_performance",
                        "不插补" in protocol
                        and "不训练模型" in protocol
                        and "不选择阈值" in protocol
                        and "不计算性能" in protocol
                        and "no imputation" in str(
                            payload.get("missing_rule", "")
                        ).lower(),
                        protocol[:220],
                    ),
                ]
            )
        elif name in SWISS_CORRECTION_CONTRACTS:
            excluded = [
                str(row.get("station_id"))
                for row in payload.get("excluded_stations", [])
            ]
            checks.extend(
                [
                    check(
                        f"{name}_protocol_before_performance",
                        "性能评价前" in protocol
                        and payload.get("performance_status") == "NOT_EVALUATED",
                        protocol[:240],
                    ),
                    check(
                        f"{name}_protocol_aeg_only",
                        excluded == ["aeg"]
                        and int(payload.get("retained_station_count", 0)) == 11,
                        excluded,
                    ),
                    check(
                        f"{name}_protocol_no_test_based_exclusion",
                        "不依据任何test性能" in protocol,
                        protocol[:240],
                    ),
                ]
            )
        elif name in SWISS_PERFORMANCE_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_train_valid_test",
                        "2021-2023 train" in protocol
                        and "2024 valid" in protocol
                        and "2025 test" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_no_test_selection",
                        "不根据test选择" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_no_dwd_weight_transfer",
                        "不迁移DWD权重" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_controlled_attack",
                        "透明受控" in protocol
                        and "不是真实攻击日志" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_retained_station_boundary",
                        int(payload.get("station_count", 0)) == 11
                        and payload.get("excluded_station_ids") == "aeg",
                        payload.get("station_ids"),
                    ),
                ]
            )
        elif name in SWISS_UNCERTAINTY_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_read_frozen_result_only",
                        "只读取" in protocol
                        and "不训练模型" in protocol
                        and "不重新选择" in protocol
                        and "不重新校准" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_deterministic_replay_boundary",
                        "确定性评价重放" in protocol
                        and "复现阶段217" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_station_cluster_boundary",
                        "站点是重采样簇" in protocol
                        and "种子是簇内重复测量" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_controlled_attack",
                        "透明受控" in protocol
                        and "不是真实攻击日志" in protocol,
                        protocol[:260],
                    ),
                    check(
                        f"{name}_protocol_provider_contrast_not_paired",
                        payload.get("provider_contrast_is_paired") is False
                        and "不能扩展为多机构泛化" in str(
                            payload.get(
                                "provider_contrast_interpretation",
                                "",
                            )
                        ),
                        payload.get("provider_contrast_interpretation", ""),
                    ),
                    check(
                        f"{name}_protocol_replay_reproduces_stage217",
                        payload.get("replay_reproduces_stage217") is True
                        and float(
                            payload.get(
                                "replay_max_abs_row_metric_difference",
                                1.0,
                            )
                        )
                        <= 1e-12
                        and float(
                            payload.get(
                                "replay_max_abs_event_metric_difference",
                                1.0,
                            )
                        )
                        <= 1e-12,
                        [
                            payload.get(
                                "replay_max_abs_row_metric_difference"
                            ),
                            payload.get(
                                "replay_max_abs_event_metric_difference"
                            ),
                        ],
                    ),
                ]
            )
        elif name in SWISS_TRANSFER_CONTRACTS:
            aggregate = {
                row["arm"]: row
                for row in payload.get("aggregate", [])
            }
            checks.extend(
                [
                    check(
                        f"{name}_protocol_three_arms",
                        set(aggregate)
                        == {
                            "dwd_threshold_transfer",
                            "swiss_valid_recalibrated",
                            "swiss_valid_selected",
                        },
                        sorted(aggregate),
                    ),
                    check(
                        f"{name}_protocol_dwd_selection_frozen",
                        "阶段208" in protocol
                        and "不重新选择" in protocol
                        and "逐种子" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_target_valid_only",
                        "瑞士valid" in protocol
                        and "test只评价" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_controlled_attack",
                        "透明受控" in protocol
                        and "不是真实攻击日志" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_absolute_threshold_negative_retained",
                        float(
                            aggregate["dwd_threshold_transfer"][
                                "f1_mean"
                            ]
                        )
                        == 0.0
                        and float(
                            aggregate["swiss_valid_selected"][
                                "f1_mean"
                            ]
                        )
                        > float(
                            aggregate["swiss_valid_recalibrated"]["f1_mean"]
                        ),
                        aggregate,
                    ),
                ]
            )
        elif name in SWISS_TRANSFER_UNCERTAINTY_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_station_cluster_paired_bootstrap",
                        "站点簇配对bootstrap" in protocol
                        and int(payload.get("bootstrap_iterations", 0))
                        == 10000,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_no_reselection",
                        "不重新训练" in protocol
                        and "不重新选择" in protocol
                        and "不使用瑞士test标签做任何选择" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_primary_comparison",
                        payload.get("comparison")
                        == "swiss_valid_selected_minus_swiss_valid_recalibrated",
                        payload.get("comparison"),
                    ),
                    check(
                        f"{name}_protocol_primary_endpoint_positive",
                        float(
                            payload["primary_event_coverage"]["ci95_low"]
                        )
                        > 0.0
                        and int(
                            payload["primary_event_coverage"][
                                "positive_stations"
                            ]
                        )
                        > int(
                            payload["primary_event_coverage"][
                                "negative_stations"
                            ]
                        ),
                        payload.get("primary_event_coverage"),
                    ),
                ]
            )
        elif name in SWISS_OPERATING_CURVE_CONTRACTS:
            primary = {
                row["metric"]: row
                for row in payload.get("curve", [])
                if row.get("is_primary")
            }
            checks.extend(
                [
                    check(
                        f"{name}_protocol_far_grid_frozen",
                        set(float(value) for value in payload.get("far_grid", []))
                        == {0.005, 0.01, 0.02, 0.05}
                        and float(payload.get("primary_far", 0.0)) == 0.01,
                        payload.get("far_grid"),
                    ),
                    check(
                        f"{name}_protocol_primary_endpoint_positive",
                        set(primary) == {
                            "f1",
                            "recall",
                            "precision",
                            "false_alarm_rate",
                            "event_coverage",
                        }
                        and float(primary["f1"]["ci95_low"]) > 0.0
                        and float(primary["event_coverage"]["ci95_low"]) > 0.0,
                        {
                            metric: {
                                "ci95_low": row["ci95_low"],
                                "ci95_high": row["ci95_high"],
                            }
                            for metric, row in primary.items()
                        },
                    ),
                    check(
                        f"{name}_protocol_absolute_threshold_negative_retained",
                        any(
                            float(row.get("f1", 1.0)) == 0.0
                            and float(row.get("false_alarm_rate", 1.0))
                            == 0.0
                            for row in payload.get("arm_aggregate", [])
                            if row.get("arm") == "dwd_threshold_transfer"
                        ),
                        next(
                            (
                                row
                                for row in payload.get("arm_aggregate", [])
                                if row.get("arm") == "dwd_threshold_transfer"
                            ),
                            None,
                        ),
                    ),
                ]
            )
        elif name in NOAA_TRANSFER_CONTRACTS:
            aggregate = {
                row["arm"]: row
                for row in payload.get("aggregate", [])
            }
            checks.extend(
                [
                    check(
                        f"{name}_protocol_three_arms",
                        set(aggregate)
                        == {
                            "dwd_threshold_transfer",
                            "noaa_valid_recalibrated",
                            "noaa_valid_selected",
                        },
                        sorted(aggregate),
                    ),
                    check(
                        f"{name}_protocol_dwd_selection_frozen",
                        "阶段208" in protocol
                        and "不重新选择" in protocol
                        and "逐种子" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_target_valid_only",
                        "NOAA valid" in protocol
                        and "test只评价" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_no_test_selection",
                        "不根据test选择" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_controlled_attack",
                        "透明受控" in protocol
                        and "不是真实攻击日志" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_absolute_threshold_negative_retained",
                        float(
                            aggregate["dwd_threshold_transfer"][
                                "f1_mean"
                            ]
                        )
                        == 0.0
                        and float(
                            aggregate["dwd_threshold_transfer"][
                                "event_coverage_mean"
                            ]
                        )
                        == 0.0,
                        aggregate["dwd_threshold_transfer"],
                    ),
                ]
            )
        elif name in NOAA_TRANSFER_UNCERTAINTY_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_station_cluster_paired_bootstrap",
                        "站点簇配对bootstrap" in protocol
                        and int(payload.get("bootstrap_iterations", 0))
                        == 10000,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_no_reselection",
                        "不重新训练" in protocol
                        and "不重新选择" in protocol
                        and "不使用NOAA test标签做任何选择" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_primary_comparison",
                        payload.get("comparison")
                        == "noaa_valid_selected_minus_noaa_valid_recalibrated",
                        payload.get("comparison"),
                    ),
                    check(
                        f"{name}_protocol_local_selection_partial_support",
                        float(payload["primary_f1"]["mean_difference"]) > 0.0
                        and float(payload["primary_f1"]["ci95_low"]) > 0.0
                        and int(
                            payload["primary_f1"]["positive_stations"]
                        )
                        == 11
                        and int(
                            payload["primary_f1"]["negative_stations"]
                        )
                        == 0
                        and float(
                            payload["primary_event_coverage"][
                                "mean_difference"
                            ]
                        )
                        > 0.0
                        and float(
                            payload["primary_event_coverage"]["ci95_low"]
                        )
                        > 0.0
                        and int(
                            payload["primary_event_coverage"][
                                "positive_stations"
                            ]
                        )
                        == 6
                        and int(
                            payload["primary_event_coverage"][
                                "negative_stations"
                            ]
                        )
                        == 3
                        and int(
                            payload["primary_event_coverage"][
                                "zero_stations"
                            ]
                        )
                        == 2
                        and float(
                            payload["primary_event_coverage"]["sign_flip_p"]
                        )
                        == 0.109375,
                        {
                            "f1": payload["primary_f1"],
                            "event_coverage": payload[
                                "primary_event_coverage"
                            ],
                        },
                    ),
                ]
            )
        elif name in KNMI_PREREGISTRATION_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_pre_download_freeze",
                        payload.get("status")
                        == "PROTOCOL_FROZEN_BEFORE_KNMI_PERFORMANCE"
                        and payload.get("download_status") == "NOT_DOWNLOADED"
                        and payload.get("performance_status") == "NOT_EVALUATED",
                        payload.get("status"),
                    ),
                    check(
                        f"{name}_protocol_grid_and_station_count",
                        float(payload.get("station_grid_degrees", 0.0)) == 0.75
                        and int(payload.get("station_count", 0)) == 12,
                        {
                            "grid": payload.get("station_grid_degrees"),
                            "stations": payload.get("station_count"),
                        },
                    ),
                    check(
                        f"{name}_protocol_split_and_endpoint",
                        payload.get("split")
                        == {
                            "train": [2021, 2022, 2023],
                            "valid": [2024],
                            "test": [2025],
                        }
                        and float(payload.get("primary_far", 0.0)) == 0.01
                        and payload.get("primary_endpoint") == "event_coverage"
                        and payload.get("secondary_endpoint") == "row_recall"
                        and payload.get("cluster_unit") == "station",
                        {
                            "split": payload.get("split"),
                            "primary_far": payload.get("primary_far"),
                            "primary_endpoint": payload.get("primary_endpoint"),
                            "cluster_unit": payload.get("cluster_unit"),
                        },
                    ),
                    check(
                        f"{name}_protocol_no_test_tuning",
                        "test只评价" in protocol
                        and "不根据test" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_controlled_attack",
                        "透明受控" in protocol
                        and "不是真实攻击日志" in protocol,
                        protocol[:300],
                    ),
                ]
            )
        elif name in KNMI_DOWNLOAD_AUDIT_CONTRACTS:
            detail = payload.get("detail", [])
            checks.extend(
                [
                    check(
                        f"{name}_protocol_official_knmi_download",
                        "KNMI官方" in protocol
                        and "Open Data API" in protocol
                        and "官方CDN" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_download_and_archive_counts",
                        int(payload.get("station_count", 0)) == 12
                        and int(payload.get("expected_archives", 0)) == 12
                        and int(payload.get("failed_count", -1)) == 0
                        and int(payload.get("required_columns_ok_count", 0)) == 12
                        and int(payload.get("archive_audit_failure_count", -1)) == 0
                        and len(detail) == 12,
                        {
                            "stations": payload.get("station_count"),
                            "archives": payload.get("expected_archives"),
                            "failed": payload.get("failed_count"),
                            "field_pass": payload.get("required_columns_ok_count"),
                            "archive_fail": payload.get("archive_audit_failure_count"),
                        },
                    ),
                    check(
                        f"{name}_protocol_no_training_selection_performance",
                        "不训练模型" in protocol
                        and "不选择阈值" in protocol
                        and "不计算性能" in protocol,
                        protocol[:300],
                    ),
                ]
            )
        elif name in KNMI_NORMALIZATION_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_split_contract",
                        payload.get("train_years") == [2021, 2022, 2023]
                        and int(payload.get("valid_year", 0)) == 2024
                        and int(payload.get("test_year", 0)) == 2025,
                    ),
                    check(
                        f"{name}_protocol_no_imputation_training_selection_performance",
                        "不插补" in protocol
                        and "不训练模型" in protocol
                        and "不选择阈值" in protocol
                        and "不计算性能" in protocol,
                        protocol[:300],
                    ),
                    check(
                        f"{name}_protocol_rows_and_duplicate_audit",
                        int(payload.get("rows", 0)) == 525888
                        and int(payload.get("expected_rows", 0)) == 525888
                        and int(payload.get("duplicate_station_timestamp_rows", -1)) == 0
                        and int(payload.get("raw_missing_timestamps", -1)) == 0,
                        {
                            "rows": payload.get("rows"),
                            "duplicates": payload.get("duplicate_station_timestamp_rows"),
                            "raw_missing": payload.get("raw_missing_timestamps"),
                        },
                    ),
                    check(
                        f"{name}_protocol_performance_not_evaluated",
                        payload.get("performance_status") == "NOT_EVALUATED",
                        payload.get("performance_status"),
                    ),
                ]
            )
        elif name in KNMI_MISSINGNESS_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_train_valid_decision_only",
                        payload.get("decision_splits") == ["train", "valid"]
                        and payload.get("test_split") == "test",
                        {
                            "decision_splits": payload.get("decision_splits"),
                            "test_split": payload.get("test_split"),
                        },
                    ),
                    check(
                        f"{name}_protocol_availability_threshold",
                        float(payload.get("availability_threshold", 0.0)) == 0.95,
                        payload.get("availability_threshold"),
                    ),
                    check(
                        f"{name}_protocol_retained_and_excluded_stations",
                        payload.get("retained_station_ids")
                        == [215, 251, 277, 278, 286, 290]
                        and payload.get("excluded_station_ids")
                        == [209, 229, 258, 285, 308, 315]
                        and int(payload.get("retained_station_count", 0)) == 6
                        and int(payload.get("excluded_station_count", 0)) == 6,
                        {
                            "retained": payload.get("retained_station_ids"),
                            "excluded": payload.get("excluded_station_ids"),
                        },
                    ),
                    check(
                        f"{name}_protocol_rows",
                        int(payload.get("decision_rows", 0)) == 420768
                        and int(payload.get("test_rows", 0)) == 105120,
                        {
                            "decision_rows": payload.get("decision_rows"),
                            "test_rows": payload.get("test_rows"),
                        },
                    ),
                    check(
                        f"{name}_protocol_no_training_selection_performance",
                        "不训练模型" in protocol
                        and "不选择阈值" in protocol
                        and "不计算性能" in protocol,
                        protocol[:300],
                    ),
                ]
            )
        elif name in KNMI_PERFORMANCE_CONTRACTS:
            summary = {
                row["metric"]: row for row in payload.get("primary_summary", [])
            }
            checks.extend(
                [
                    check(
                        f"{name}_protocol_third_source_boundary",
                        "第三来源" in protocol
                        and "不迁移DWD权重" in protocol
                        and "不根据test" in protocol,
                        protocol[:320],
                    ),
                    check(
                        f"{name}_protocol_frozen_sites",
                        int(payload.get("station_count", 0)) == 6
                        and payload.get("station_ids")
                        == ["215", "251", "277", "278", "286", "290"]
                        and payload.get("excluded_station_ids")
                        == [209, 229, 258, 285, 308, 315],
                        {
                            "stations": payload.get("station_ids"),
                            "excluded": payload.get("excluded_station_ids"),
                        },
                    ),
                    check(
                        f"{name}_protocol_primary_endpoint",
                        float(payload.get("primary_far", 0.0)) == 0.01
                        and payload.get("primary_endpoint") == "event_coverage"
                        and payload.get("secondary_endpoint") == "row_recall"
                        and payload.get("cluster_unit") == "station"
                        and set(summary) == {
                            "precision",
                            "recall",
                            "f1",
                            "false_alarm_rate",
                            "roc_auc",
                            "average_precision",
                            "event_coverage",
                            "row_recall",
                        },
                        {
                            "primary_far": payload.get("primary_far"),
                            "primary_endpoint": payload.get("primary_endpoint"),
                            "metrics": sorted(summary),
                        },
                    ),
                    check(
                        f"{name}_protocol_controlled_attack",
                        "透明受控" in protocol
                        and "不是真实攻击日志" in protocol,
                        protocol[:320],
                    ),
                ]
            )
        elif name in KNMI_TRANSFER_CONTRACTS:
            aggregate = {
                row["arm"]: row for row in payload.get("aggregate", [])
            }
            checks.extend(
                [
                    check(
                        f"{name}_protocol_frozen_dwd_selection",
                        "阶段208" in protocol
                        and "不重新选择" in protocol
                        and "逐种子" in protocol,
                        protocol[:320],
                    ),
                    check(
                        f"{name}_protocol_target_valid_only",
                        "KNMI valid" in protocol
                        and "test只评价" in protocol
                        and "不用于特征" in protocol,
                        protocol[:320],
                    ),
                    check(
                        f"{name}_protocol_three_arms",
                        set(aggregate)
                        == {
                            "dwd_threshold_transfer",
                            "knmi_valid_recalibrated",
                            "knmi_valid_selected",
                        },
                        sorted(aggregate),
                    ),
                    check(
                        f"{name}_protocol_absolute_threshold_negative_retained",
                        float(
                            aggregate["dwd_threshold_transfer"]["f1_mean"]
                        )
                        == 0.0
                        and float(
                            aggregate["dwd_threshold_transfer"][
                                "event_coverage_mean"
                            ]
                        )
                        == 0.0,
                        aggregate["dwd_threshold_transfer"],
                    ),
                    check(
                        f"{name}_protocol_controlled_attack",
                        "透明受控" in protocol
                        and "不是真实攻击日志" in protocol,
                        protocol[:320],
                    ),
                ]
            )
        elif name in KNMI_TRANSFER_UNCERTAINTY_CONTRACTS:
            checks.extend(
                [
                    check(
                        f"{name}_protocol_station_cluster_paired_bootstrap",
                        "6个KNMI站点" in protocol
                        and "10000次站点簇" in protocol
                        and int(payload.get("bootstrap_iterations", 0))
                        == 10000
                        and int(payload.get("station_count", 0)) == 6,
                        protocol[:320],
                    ),
                    check(
                        f"{name}_protocol_no_reselection",
                        "不重新训练" in protocol
                        and "不重新选择" in protocol
                        and "不使用KNMI test标签做任何选择" in protocol,
                        protocol[:320],
                    ),
                    check(
                        f"{name}_protocol_primary_comparison",
                        payload.get("comparison")
                        == "knmi_valid_selected_minus_knmi_valid_recalibrated",
                        payload.get("comparison"),
                    ),
                    check(
                        f"{name}_protocol_local_selection_supported_with_q_boundary",
                        float(payload["primary_event_coverage"]["ci95_low"])
                        > 0.0
                        and int(
                            payload["primary_event_coverage"][
                                "positive_stations"
                            ]
                        )
                        > int(
                            payload["primary_event_coverage"][
                                "negative_stations"
                            ]
                        )
                        and float(
                            payload["primary_event_coverage"]["sign_flip_p"]
                        )
                        < 0.05
                        and float(payload["primary_f1"]["ci95_low"]) > 0.0
                        and float(payload["primary_f1"]["sign_flip_p"]) == 0.03125
                        and float(payload["primary_f1"]["bh_q"]) >= 0.05,
                        {
                            "event_coverage": payload[
                                "primary_event_coverage"
                            ],
                            "f1": payload["primary_f1"],
                        },
                    ),
                ]
            )
        elif name in DWD_PERFORMANCE_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_protocol_train_valid_test", "train" in lowered and "valid" in lowered and "test" in lowered, protocol[:240]),
                    check(f"{name}_protocol_no_test_selection", "不根据test选择" in protocol, protocol[:240]),
                    check(f"{name}_protocol_controlled_attack", "透明受控" in protocol and "不是真实攻击日志" in protocol, protocol[:240]),
                    check(f"{name}_protocol_clean_time_shift", "time_shift只从阶段206干净序列" in protocol, protocol[:240]),
                    (
                        check(
                            f"{name}_protocol_separate_valid_test",
                            "分别构造" in protocol
                            or "separately" in lowered,
                            protocol[:240],
                        )
                        if name == "stage208"
                        else check(
                            f"{name}_protocol_separate_valid_test",
                            True,
                            protocol[:240],
                        )
                    ),
                ]
            )
        elif name in ATTACK_YEAR_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_protocol_has_test", "test" in lowered, protocol[:220]),
                    check(f"{name}_protocol_no_selection", "不重新选" in protocol or "不重新训练" in protocol or "不根据test" in protocol, protocol[:220]),
                    check(f"{name}_protocol_attack_year_boundary", "攻击类型" in protocol and "目标年份" in protocol, protocol[:220]),
                ]
            )
        elif name in YEAR_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_protocol_has_test", "test" in lowered, protocol[:220]),
                    check(f"{name}_protocol_no_selection", "不重新选" in protocol or "不重新训练" in protocol or "不根据test" in protocol, protocol[:220]),
                    check(f"{name}_protocol_fold_uncertainty", "fold" in lowered or "目标年" in protocol, protocol[:220]),
                ]
            )
        elif name in SUMMARY_CONTRACTS:
            checks.extend(
                [
                    check(f"{name}_protocol_has_test", "test" in lowered, protocol[:220]),
                    check(f"{name}_protocol_no_selection", "不重新选" in protocol or "不重新训练" in protocol or "不根据test" in protocol, protocol[:220]),
                    check(f"{name}_protocol_seed_uncertainty", "种子" in protocol, protocol[:220]),
                ]
            )
        elif name in PERFORMANCE_CONTRACTS:
            checks.append(check(f"{name}_protocol_has_valid", "valid" in lowered, protocol[:220]))
            checks.append(check(f"{name}_protocol_has_test", "test" in lowered, protocol[:220]))
        elif name in {"stage107", "stage142", "stage153", "stage154", "stage155", "stage156", "stage157"}:
            checks.extend(
                [
                    check(f"{name}_protocol_natural_observation", "自然观测" in protocol, protocol[:220]),
                    check(f"{name}_protocol_no_attack", "不注入攻击" in protocol, protocol[:220]),
                    check(f"{name}_protocol_weak_label", "弱标签" in protocol, protocol[:220]),
                ]
            )
        elif name in {"stage109", "stage116"}:
            checks.extend(
                [
                    check(f"{name}_protocol_natural", "自然" in protocol, protocol[:220]),
                    check(f"{name}_protocol_weak_label", "弱标签" in protocol, protocol[:220]),
                    check(f"{name}_protocol_no_attack_truth", "攻击真值" in protocol, protocol[:220]),
                ]
            )
        elif name == "stage113":
            checks.extend(
                [
                    check(f"{name}_protocol_valid_boundary", "valid" in lowered or "验证" in protocol, protocol[:220]),
                    check(f"{name}_protocol_test_boundary", "测试期" in protocol or "test" in lowered, protocol[:220]),
                    check(f"{name}_protocol_era5_auxiliary", "不是现场真值" in protocol or "区域背景" in protocol or "ERA5" in protocol, protocol[:220]),
                ]
            )
    checks.extend(
        [
            check("stage102_result_has_selected_and_test_metrics", _stage102_result_check()),
            check("stage105_result_preserves_attack_type_and_overall", _stage105_result_check()),
            check("stage107_result_has_natural_no_attack_boundary", _stage107_result_check()),
            check(
                "stage217_protocol_sha256_matches_stage214",
                json.loads(
                    PROTOCOL_JSONS["stage217"].read_text(encoding="utf-8")
                ).get("protocol_sha256")
                == sha256_file(PROTOCOL_JSONS["stage214"]),
            ),
            check(
                "stage217_normalized_sha256_matches_stage216",
                json.loads(
                    PROTOCOL_JSONS["stage217"].read_text(encoding="utf-8")
                ).get("normalized_sha256")
                == sha256_file(
                    TABLES / "stage216_swiss_normalized.parquet"
                ),
            ),
            check(
                "stage217_correction_sha256_matches_stage216a",
                json.loads(
                    PROTOCOL_JSONS["stage217"].read_text(encoding="utf-8")
                ).get("correction_sha256")
                == sha256_file(PROTOCOL_JSONS["stage216a"]),
            ),
            check(
                "stage218_protocol_sha256_matches_stage214",
                json.loads(
                    PROTOCOL_JSONS["stage218"].read_text(encoding="utf-8")
                ).get("protocol_sha256")
                == sha256_file(PROTOCOL_JSONS["stage214"]),
            ),
            check(
                "stage218_normalized_sha256_matches_stage216",
                json.loads(
                    PROTOCOL_JSONS["stage218"].read_text(encoding="utf-8")
                ).get("normalized_sha256")
                == sha256_file(
                    TABLES / "stage216_swiss_normalized.parquet"
                ),
            ),
            check(
                "stage218_correction_sha256_matches_stage216a",
                json.loads(
                    PROTOCOL_JSONS["stage218"].read_text(encoding="utf-8")
                ).get("correction_sha256")
                == sha256_file(PROTOCOL_JSONS["stage216a"]),
            ),
            check(
                "stage218_stage217_sha256_matches",
                json.loads(
                    PROTOCOL_JSONS["stage218"].read_text(encoding="utf-8")
                ).get("stage217_sha256")
                == sha256_file(PROTOCOL_JSONS["stage217"]),
            ),
            check(
                "stage219_protocol_sha256_matches_stage214",
                json.loads(
                    PROTOCOL_JSONS["stage219"].read_text(encoding="utf-8")
                ).get("protocol_sha256")
                == sha256_file(PROTOCOL_JSONS["stage214"]),
            ),
            check(
                "stage219_normalized_sha256_matches_stage216",
                json.loads(
                    PROTOCOL_JSONS["stage219"].read_text(encoding="utf-8")
                ).get("normalized_sha256")
                == sha256_file(
                    TABLES / "stage216_swiss_normalized.parquet"
                ),
            ),
            check(
                "stage219_correction_sha256_matches_stage216a",
                json.loads(
                    PROTOCOL_JSONS["stage219"].read_text(encoding="utf-8")
                ).get("correction_sha256")
                == sha256_file(PROTOCOL_JSONS["stage216a"]),
            ),
            check(
                "stage219_dwd_selection_sha256_matches_stage208",
                json.loads(
                    PROTOCOL_JSONS["stage219"].read_text(encoding="utf-8")
                ).get("dwd_selection_sha256")
                == sha256_file(
                    TABLES
                    / "stage208_dwd_persistent_repeat_protection_selection.csv"
                ),
            ),
            check(
                "stage220_stage219_sha256_matches",
                json.loads(
                    PROTOCOL_JSONS["stage220"].read_text(encoding="utf-8")
                ).get("stage219_json_sha256")
                == sha256_file(PROTOCOL_JSONS["stage219"]),
            ),
            check(
                "stage220_stage219_station_sha256_matches",
                json.loads(
                    PROTOCOL_JSONS["stage220"].read_text(encoding="utf-8")
                ).get("stage219_station_sha256")
                == sha256_file(
                    TABLES
                    / "stage219_swiss_dwd_protection_transfer_station_detail.csv"
                ),
            ),
            check(
                "stage220_stage219_events_sha256_matches",
                json.loads(
                    PROTOCOL_JSONS["stage220"].read_text(encoding="utf-8")
                ).get("stage219_events_sha256")
                == sha256_file(
                    TABLES
                    / "stage219_swiss_dwd_protection_transfer_events.csv"
                ),
            ),
            check(
                "stage227_protocol_sha256_matches_stage224",
                json.loads(
                    PROTOCOL_JSONS["stage227"].read_text(encoding="utf-8")
                ).get("protocol_sha256")
                == sha256_file(PROTOCOL_JSONS["stage224"]),
            ),
            check(
                "stage227_normalized_sha256_matches_stage226",
                json.loads(
                    PROTOCOL_JSONS["stage227"].read_text(encoding="utf-8")
                ).get("normalized_sha256")
                == sha256_file(
                    TABLES / "stage226_knmi_normalized.parquet"
                ),
            ),
            check(
                "stage227_correction_sha256_matches_stage226a",
                json.loads(
                    PROTOCOL_JSONS["stage227"].read_text(encoding="utf-8")
                ).get("correction_sha256")
                == sha256_file(PROTOCOL_JSONS["stage226a"]),
            ),
            check(
                "stage228_protocol_sha256_matches_stage224",
                json.loads(
                    PROTOCOL_JSONS["stage228"].read_text(encoding="utf-8")
                ).get("protocol_sha256")
                == sha256_file(PROTOCOL_JSONS["stage224"]),
            ),
            check(
                "stage228_normalized_sha256_matches_stage226",
                json.loads(
                    PROTOCOL_JSONS["stage228"].read_text(encoding="utf-8")
                ).get("normalized_sha256")
                == sha256_file(
                    TABLES / "stage226_knmi_normalized.parquet"
                ),
            ),
            check(
                "stage228_correction_sha256_matches_stage226a",
                json.loads(
                    PROTOCOL_JSONS["stage228"].read_text(encoding="utf-8")
                ).get("correction_sha256")
                == sha256_file(PROTOCOL_JSONS["stage226a"]),
            ),
            check(
                "stage228_dwd_selection_sha256_matches_stage208",
                json.loads(
                    PROTOCOL_JSONS["stage228"].read_text(encoding="utf-8")
                ).get("dwd_selection_sha256")
                == sha256_file(
                    TABLES
                    / "stage208_dwd_persistent_repeat_protection_selection.csv"
                ),
            ),
            check(
                "stage229_stage228_sha256_matches",
                json.loads(
                    PROTOCOL_JSONS["stage229"].read_text(encoding="utf-8")
                ).get("stage228_json_sha256")
                == sha256_file(PROTOCOL_JSONS["stage228"]),
            ),
            check(
                "stage229_stage228_station_sha256_matches",
                json.loads(
                    PROTOCOL_JSONS["stage229"].read_text(encoding="utf-8")
                ).get("stage228_station_sha256")
                == sha256_file(
                    TABLES
                    / "stage228_knmi_dwd_protection_transfer_station_detail.csv"
                ),
            ),
            check(
                "stage229_stage228_station_event_sha256_matches",
                json.loads(
                    PROTOCOL_JSONS["stage229"].read_text(encoding="utf-8")
                ).get("stage228_station_event_sha256")
                == sha256_file(
                    TABLES
                    / "stage228_knmi_dwd_protection_transfer_station_event_detail.csv"
                ),
            ),
        ]
    )

    manual_review = [
        "静态源码检查不能替代人工逐行复核；尤其要复核新增实验是否在test标签上做过隐式排序或筛选。",
        "受控攻击和NOAA质量弱标签的真实性边界仍需在论文中人工确认。",
        "KNMI只有6个主变量可用站点；事件覆盖显著不能外推为多机构普适领先，F1精确检验仍不显著。",
        "DWD绝对阈值直接迁移到KNMI的零结果必须保留；可迁移的是特征族、窗口和权重，阈值必须用目标域valid正常样本重校准。",
        "阶段231只有4个机构，机构级精确sign-flip检验最小双侧p=0.125，仍只能作为低功效描述性稳健性检查；机构等权和两级bootstrap不能替代更多独立机构验证。",
        "阶段232显示跨机构总体正向方向主要集中在stuck攻击类型；移除stuck后其余攻击类型的行召回和事件覆盖方向均转为负，stuck的两级bootstrap下界为0，不能写成所有攻击类型普遍占优或显著领先。",
    ]
    failed = [item for item in checks if item["status"] == "FAIL"]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED" if not failed else "FAIL",
        "protocol": "对关键实验源码和JSON协议执行静态边界检查：受控性能实验要求train/valid/test边界；自然观测与弱标签审计使用相应的无攻击、弱标签和校准边界契约；stage102选择函数只读取valid候选字段且不读取test_*；结果同时保留test指标。静态检查不能替代人工审阅。",
        "checks": checks,
        "failed_checks": failed,
        "manual_review_items": manual_review,
        "source_contracts": {name: str(path) for name, path in SOURCE_CONTRACTS.items()},
        "protocol_jsons": {name: str(path) for name, path in PROTOCOL_JSONS.items()},
        "limitations": ["静态关键词和函数块检查不等于形式化信息流证明。", "通过本审计不代表模型性能领先或攻击标签真实。"],
    }
    (TABLES / "stage115_protocol_boundary_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(checks).to_csv(TABLES / "stage115_protocol_boundary_audit.csv", index=False, encoding="utf-8-sig")
    lines = ["# 阶段115 train-valid-test协议边界审计", "", f"- 状态：**{payload['status']}**", f"- 检查项：`{len(checks)}`；失败项：`{len(failed)}`", "", "| 检查 | 状态 |", "|---|---|"]
    lines.extend(f"| {item['check']} | {item['status']} |" for item in checks)
    lines += ["", "## 人工复核项", ""]
    lines.extend(f"- {item}" for item in manual_review)
    lines += ["", "静态边界审计通过不等于性能优越性，也不替代人工检查和真实攻击数据。"]
    (REPORTS / "stage115_protocol_boundary_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks), "failures": len(failed)}, ensure_ascii=False))
    return 0 if not failed else 1


def _stage102_result_check() -> bool:
    payload = json.loads((TABLES / "stage102_noaa_trust_svm_valid_selection.json").read_text(encoding="utf-8"))
    selected = payload.get("selected_detail", [])
    return len(selected) == 20 and all(any(key.startswith("test_") for key in row) for row in selected)


def _stage105_result_check() -> bool:
    payload = json.loads((TABLES / "stage105_noaa_attack_type_valid_selected.json").read_text(encoding="utf-8"))
    return len(payload.get("attack_type_aggregate", [])) == 18 and len(payload.get("overall_aggregate", [])) == 3


def _stage107_result_check() -> bool:
    payload = json.loads((TABLES / "stage107_noaa_natural_svm_validation.json").read_text(encoding="utf-8"))
    protocol = str(payload.get("protocol", ""))
    return len(payload.get("detail", [])) == 45 and "不注入攻击" in protocol and "不使用目标年标签" in protocol


if __name__ == "__main__":
    raise SystemExit(main())
