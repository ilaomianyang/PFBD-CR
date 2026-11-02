from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

JSON_FILES = {
    "stage99": TABLES / "stage99_noaa_baseline_paired_bootstrap.json",
    "stage103": TABLES / "stage103_noaa_valid_selection_stability.json",
    "stage105": TABLES / "stage105_noaa_attack_type_valid_selected.json",
    "stage106": TABLES / "stage106_noaa_attack_type_paired_bootstrap.json",
    "stage107": TABLES / "stage107_noaa_natural_svm_validation.json",
    "stage108": TABLES / "stage108_noaa_natural_alarm_station_bootstrap.json",
    "stage109": TABLES / "stage109_noaa_quality_weak_label_bootstrap.json",
    "stage110": TABLES / "stage110_noaa_quality_hierarchical_bootstrap.json",
    "stage111": TABLES / "stage111_noaa_quality_count_sensitivity.json",
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
    "stage234": TABLES / "stage234_eccc_protocol_preregistration.json",
    "stage235": TABLES / "stage235_eccc_official_download.json",
    "stage236": TABLES / "stage236_eccc_normalized.json",
    "stage237": TABLES / "stage237_eccc_controlled_attack.json",
    "stage238": TABLES / "stage238_eccc_dwd_protection_transfer.json",
    "stage239": TABLES / "stage239_eccc_transfer_uncertainty.json",
    "stage240": TABLES / "stage240_nonstuck_balanced_selection.json",
    "stage241": TABLES / "stage241_transfer_heterogeneity_audit.json",
    "stage242": TABLES / "stage242_multisource_transfer_matrix.json",
    "stage243": TABLES / "stage243_knmi_negative_case_decomposition.json",
    "stage244": TABLES / "stage244_cross_provider_nonstuck_candidate_features.json",
    "stage245": TABLES / "stage245_cross_provider_nonstuck_seed_replication.json",
    "stage246": TABLES / "stage246_frozen_submission_package.json",
    "stage247": TABLES / "stage247_provider_balanced_nonstuck_audit.json",
    "stage248": TABLES / "stage248_cross_provider_fair_baseline_audit.json",
    "stage249": TABLES / "stage249_jisa_submission_gate.json",
    "stage250": TABLES / "stage250_jisa_risk_closure_audit.json",
    "stage251": TABLES / "stage251_jisa_stable_submission_objective.json",
    "stage252": TABLES / "stage252_authorized_testbed_analysis.json",
    "stage253": TABLES / "stage253_residual_risk_optimization_plan.json",
    "stage116": TABLES / "stage116_noaa_quality_label_permutation.json",
}
TABULAR_FILES = {
    "stage98": TABLES / "stage98_noaa_low_far_baseline_comparison_detail.csv",
    "stage100": TABLES / "stage100_noaa_low_far_station_counts.parquet",
    "stage101": TABLES / "stage101_noaa_trust_svm_complementarity_detail.csv",
    "stage102_all": TABLES / "stage102_noaa_trust_svm_valid_selection_all_candidates.csv",
    "stage102_selected": TABLES / "stage102_noaa_trust_svm_valid_selection_selected.csv",
    "stage105_type": TABLES / "stage105_noaa_attack_type_valid_selected_detail.csv",
    "stage105_overall": TABLES / "stage105_noaa_attack_type_valid_selected_overall_detail.csv",
    "stage107_detail": TABLES / "stage107_noaa_natural_svm_validation_detail.csv",
    "stage107_station": TABLES / "stage107_noaa_natural_svm_validation_station_detail.csv",
    "stage109": TABLES / "stage109_noaa_quality_weak_label_bootstrap.csv",
    "stage110": TABLES / "stage110_noaa_quality_hierarchical_bootstrap.csv",
    "stage111": TABLES / "stage111_noaa_quality_count_sensitivity.csv",
    "stage113": TABLES / "stage113_era5_alignment_leakage_audit.csv",
    "stage117": TABLES / "stage117_multiseed_uncertainty.csv",
    "stage118_detail": TABLES / "stage118_attack_duration_sensitivity_detail.csv",
    "stage118": TABLES / "stage118_attack_duration_sensitivity.csv",
    "stage118_by_type": TABLES / "stage118_attack_duration_sensitivity_by_type.csv",
    "stage119_detail": TABLES / "stage119_attack_position_sensitivity_detail.csv",
    "stage119": TABLES / "stage119_attack_position_sensitivity.csv",
    "stage119_by_type": TABLES / "stage119_attack_position_sensitivity_by_type.csv",
    "stage120_detail": TABLES / "stage120_station_stratified_performance_detail.csv",
    "stage120": TABLES / "stage120_station_stratified_performance.csv",
    "stage120_seed": TABLES / "stage120_station_stratified_performance_seed_summary.csv",
    "stage121_candidates": TABLES / "stage121_local_station_threshold_calibration_candidates.csv",
    "stage121_selected": TABLES / "stage121_local_station_threshold_calibration_selected.csv",
    "stage121_station": TABLES / "stage121_local_station_threshold_calibration_station_detail.csv",
    "stage122_candidates": TABLES / "stage122_leave_one_station_threshold_transfer_candidates.csv",
    "stage122_selected": TABLES / "stage122_leave_one_station_threshold_transfer_selected.csv",
    "stage122_station": TABLES / "stage122_leave_one_station_threshold_transfer_station_summary.csv",
    "stage122_seed": TABLES / "stage122_leave_one_station_threshold_transfer_seed_summary.csv",
    "stage123": TABLES / "stage123_threshold_transfer_bootstrap.csv",
    "stage124": TABLES / "stage124_leave_one_station_refit_multiseed.csv",
    "stage124_detail": TABLES / "stage124_leave_one_station_refit_multiseed_detail.parquet",
    "stage124_valid_detail": TABLES / "stage124_leave_one_station_refit_multiseed_valid_detail.parquet",
    "stage124_station": TABLES / "stage124_leave_one_station_refit_multiseed_station_summary.csv",
    "stage124_seed": TABLES / "stage124_leave_one_station_refit_multiseed_seed_summary.csv",
    "stage125": TABLES / "stage125_leave_one_station_refit_multiseed_bootstrap.csv",
    "stage126_selected": TABLES / "stage126_low_far_leave_one_station_operating_points_selected.csv",
    "stage126": TABLES / "stage126_low_far_leave_one_station_operating_points.csv",
    "stage126_station": TABLES / "stage126_low_far_leave_one_station_operating_points_station_summary.csv",
    "stage127": TABLES / "stage127_low_far_operating_point_bootstrap.csv",
    "stage128_detail": TABLES / "stage128_recovery_candidate_diagnosis_detail.csv",
    "stage128_station": TABLES / "stage128_recovery_candidate_diagnosis_station_summary.csv",
    "stage128": TABLES / "stage128_recovery_candidate_diagnosis.csv",
    "stage129_detail": TABLES / "stage129_valid_recovery_strategy_selection_detail.csv",
    "stage129_selected": TABLES / "stage129_valid_recovery_strategy_selection_selected.csv",
    "stage129": TABLES / "stage129_valid_recovery_strategy_selection.csv",
    "stage129_counts": TABLES / "stage129_valid_recovery_strategy_selection_counts.csv",
    "stage130_detail": TABLES / "stage130_noaa_safety_constrained_recovery_detail.csv",
    "stage130_selected": TABLES / "stage130_noaa_safety_constrained_recovery_selected.csv",
    "stage130": TABLES / "stage130_noaa_safety_constrained_recovery.csv",
    "stage130_counts": TABLES / "stage130_noaa_safety_constrained_recovery_counts.csv",
    "stage131_detail": TABLES / "stage131_safety_constraint_sensitivity_detail.csv",
    "stage131": TABLES / "stage131_safety_constraint_sensitivity.csv",
    "stage131_counts": TABLES / "stage131_safety_constraint_sensitivity_counts.csv",
    "stage132_detail": TABLES / "stage132_recovery_utility_paired_bootstrap_detail.csv",
    "stage132": TABLES / "stage132_recovery_utility_paired_bootstrap.csv",
    "stage133_detail": TABLES / "stage133_causal_temporal_recovery_detail.csv",
    "stage133_station": TABLES / "stage133_causal_temporal_recovery_station_detail.csv",
    "stage133": TABLES / "stage133_causal_temporal_recovery.csv",
    "stage134_detail": TABLES / "stage134_causal_recovery_optimization_detail.csv",
    "stage134_selected": TABLES / "stage134_causal_recovery_optimization_selected.csv",
    "stage134": TABLES / "stage134_causal_recovery_optimization.csv",
    "stage134_counts": TABLES / "stage134_causal_recovery_optimization_counts.csv",
    "stage135_detail": TABLES / "stage135_independent_seed_replication_detail.csv",
    "stage135_attack_detail": TABLES / "stage135_independent_seed_replication_attack_detail.csv",
    "stage135_attack_aggregate": TABLES / "stage135_independent_seed_replication_attack_aggregate.csv",
    "stage136_detail": TABLES / "stage136_seed_group_consistency_detail.csv",
    "stage137_detail": TABLES / "stage137_enhanced_candidate_comparison_detail.csv",
    "stage137_attack_detail": TABLES / "stage137_enhanced_candidate_comparison_attack_detail.csv",
    "stage137_attack_aggregate": TABLES / "stage137_enhanced_candidate_comparison_attack_aggregate.csv",
    "stage138_detail": TABLES / "stage138_independent_baseline_comparison_detail.csv",
    "stage138_aggregate": TABLES / "stage138_independent_baseline_comparison_aggregate.csv",
    "stage138_paired": TABLES / "stage138_independent_baseline_comparison_paired.csv",
    "stage139_detail": TABLES / "stage139_expanded_fusion_grid_detail.csv",
    "stage139_selected": TABLES / "stage139_expanded_fusion_grid_selected.csv",
    "stage139_paired": TABLES / "stage139_expanded_fusion_grid_paired.csv",
    "stage139_aggregate": TABLES / "stage139_expanded_fusion_grid_aggregate.csv",
    "stage139_metric_paired": TABLES / "stage139_expanded_fusion_grid_metric_paired.csv",
    "stage140_detail": TABLES / "stage140_expanded_grid_independent_confirmation_detail.csv",
    "stage140_selected": TABLES / "stage140_expanded_grid_independent_confirmation_selected.csv",
    "stage140_paired": TABLES / "stage140_expanded_grid_independent_confirmation_paired.csv",
    "stage140_aggregate": TABLES / "stage140_expanded_grid_independent_confirmation_aggregate.csv",
    "stage140_metric_paired": TABLES / "stage140_expanded_grid_independent_confirmation_metric_paired.csv",
    "stage141_detail": TABLES / "stage141_noaa_frozen_external_validation_detail.csv",
    "stage141_selected": TABLES / "stage141_noaa_frozen_external_validation_selected.csv",
    "stage141_paired": TABLES / "stage141_noaa_frozen_external_validation_paired.csv",
    "stage141_fold": TABLES / "stage141_noaa_frozen_external_validation_by_fold.csv",
    "stage141_attack": TABLES / "stage141_noaa_frozen_external_validation_by_attack_type.csv",
    "stage141_station": TABLES / "stage141_noaa_frozen_external_validation_by_station.csv",
    "stage141_low_far": TABLES / "stage141_noaa_frozen_external_validation_low_far.csv",
    "stage142_detail": TABLES / "stage142_noaa_frozen_natural_operational_validation_detail.csv",
    "stage142_station": TABLES / "stage142_noaa_frozen_natural_operational_validation_station_detail.csv",
    "stage142_aggregate": TABLES / "stage142_noaa_frozen_natural_operational_validation.csv",
    "stage142_station_summary": TABLES / "stage142_noaa_frozen_natural_operational_validation_station_summary.csv",
    "stage142_paired": TABLES / "stage142_noaa_frozen_natural_operational_validation_weak_label_paired.csv",
    "stage143_detail": TABLES / "stage143_noaa_frozen_trust_svm_safety_combination_detail.csv",
    "stage143_selected": TABLES / "stage143_noaa_frozen_trust_svm_safety_combination_selected.csv",
    "stage143_aggregate_all": TABLES / "stage143_noaa_frozen_trust_svm_safety_combination_aggregate_all.csv",
    "stage143_aggregate_selected": TABLES / "stage143_noaa_frozen_trust_svm_safety_combination_aggregate_selected.csv",
    "stage143_selection_counts": TABLES / "stage143_noaa_frozen_trust_svm_safety_combination_selection_counts.csv",
    "stage143_paired": TABLES / "stage143_noaa_frozen_trust_svm_safety_combination_paired.csv",
    "stage144_aggregate": TABLES / "stage144_noaa_cross_fold_candidate_transfer.csv",
    "stage144_detail": TABLES / "stage144_noaa_cross_fold_candidate_transfer_detail.csv",
    "stage144_selection_counts": TABLES / "stage144_noaa_cross_fold_candidate_transfer_selection_counts.csv",
    "stage144_paired": TABLES / "stage144_noaa_cross_fold_candidate_transfer_paired.csv",
    "stage145_station": TABLES / "stage145_noaa_station_heterogeneity_audit_station_summary.csv",
    "stage145_paired": TABLES / "stage145_noaa_station_heterogeneity_audit_paired.csv",
    "stage145_extremes": TABLES / "stage145_noaa_station_heterogeneity_audit_extremes.csv",
    "stage146_year": TABLES / "stage146_noaa_year_heterogeneity_audit_year_summary.csv",
    "stage146_paired": TABLES / "stage146_noaa_year_heterogeneity_audit_paired.csv",
    "stage147_detail": TABLES / "stage147_noaa_attack_year_interaction_audit_detail.csv",
    "stage147_paired": TABLES / "stage147_noaa_attack_year_interaction_audit_paired.csv",
    "stage148_detail": TABLES / "stage148_noaa_baseline_attack_type_detail_detail.csv",
    "stage148_aggregate": TABLES / "stage148_noaa_baseline_attack_type_detail_aggregate.csv",
    "stage149_detail": TABLES / "stage149_noaa_attack_type_paired_bootstrap_detail.csv",
    "stage149_paired": TABLES / "stage149_noaa_attack_type_paired_bootstrap_paired.csv",
    "stage150_detail": TABLES / "stage150_noaa_attack_type_multiple_comparison_detail.csv",
    "stage151_detail": TABLES / "stage151_noaa_attack_intensity_sensitivity_detail.csv",
    "stage151_type_detail": TABLES / "stage151_noaa_attack_intensity_sensitivity_type_detail.csv",
    "stage151_aggregate": TABLES / "stage151_noaa_attack_intensity_sensitivity_aggregate.csv",
    "stage151_type_aggregate": TABLES / "stage151_noaa_attack_intensity_sensitivity_type_aggregate.csv",
    "stage152_detail": TABLES / "stage152_noaa_baseline_station_comparison_detail.csv",
    "stage152_station_aggregate": TABLES / "stage152_noaa_baseline_station_comparison_station_aggregate.csv",
    "stage152_method_summary": TABLES / "stage152_noaa_baseline_station_comparison_method_summary.csv",
    "stage152_paired": TABLES / "stage152_noaa_baseline_station_comparison_paired.csv",
    "stage153_detail": TABLES / "stage153_noaa_natural_station_operational_comparison_detail.csv",
    "stage153_summary": TABLES / "stage153_noaa_natural_station_operational_comparison_summary.csv",
    "stage153_contrast_summary": TABLES / "stage153_noaa_natural_station_operational_comparison_contrast_summary.csv",
    "stage153_paired": TABLES / "stage153_noaa_natural_station_operational_comparison_paired.csv",
    "stage154_summary": TABLES / "stage154_noaa_natural_temporal_drift_summary.csv",
    "stage154_paired": TABLES / "stage154_noaa_natural_temporal_drift_paired.csv",
    "stage155_detail": TABLES / "stage155_noaa_natural_weather_regime_audit_detail.csv",
    "stage155_station": TABLES / "stage155_noaa_natural_weather_regime_audit_station_detail.csv",
    "stage155_paired": TABLES / "stage155_noaa_natural_weather_regime_audit_paired.csv",
    "stage156_detail": TABLES / "stage156_noaa_natural_observation_gap_audit_detail.csv",
    "stage156_station": TABLES / "stage156_noaa_natural_observation_gap_audit_station_detail.csv",
    "stage156_paired": TABLES / "stage156_noaa_natural_observation_gap_audit_paired.csv",
    "stage157": TABLES / "stage157_noaa_natural_two_way_cluster_bootstrap.csv",
    "stage158_detail": TABLES / "stage158_noaa_controlled_regime_gap_performance_detail.csv",
    "stage158_paired": TABLES / "stage158_noaa_controlled_regime_gap_performance_paired.csv",
    "stage159_detail": TABLES / "stage159_noaa_controlled_regime_multiple_comparison_detail.csv",
    "stage159_not_estimable": TABLES / "stage159_noaa_controlled_regime_multiple_comparison_not_estimable.csv",
    "stage160_detail": TABLES / "stage160_noaa_controlled_practical_effect_sensitivity_detail.csv",
    "stage160_not_estimable": TABLES / "stage160_noaa_controlled_practical_effect_sensitivity_not_estimable.csv",
    "stage161_detail": TABLES / "stage161_noaa_controlled_station_regime_performance_detail.csv",
    "stage161_station": TABLES / "stage161_noaa_controlled_station_regime_performance_station_summary.csv",
    "stage162_station_detail": TABLES / "stage162_noaa_controlled_station_practical_effect_station_detail.csv",
    "stage162_summary": TABLES / "stage162_noaa_controlled_station_practical_effect_summary.csv",
    "stage162_not_estimable": TABLES / "stage162_noaa_controlled_station_practical_effect_not_estimable.csv",
    "stage162_weak": TABLES / "stage162_noaa_controlled_station_practical_effect_weak_stations.csv",
    "stage163_detail": TABLES / "stage163_noaa_controlled_regime_full_baseline_detail.csv",
    "stage163_summary": TABLES / "stage163_noaa_controlled_regime_full_baseline_summary.csv",
    "stage163_paired": TABLES / "stage163_noaa_controlled_regime_full_baseline_paired.csv",
    "stage163_not_estimable": TABLES / "stage163_noaa_controlled_regime_full_baseline_not_estimable.csv",
    "stage164_detail": TABLES / "stage164_noaa_controlled_full_baseline_multiple_comparison_detail.csv",
    "stage164_not_estimable": TABLES / "stage164_noaa_controlled_full_baseline_multiple_comparison_not_estimable.csv",
    "stage165_detail": TABLES / "stage165_noaa_controlled_station_full_baseline_detail.csv",
    "stage165_station_summary": TABLES / "stage165_noaa_controlled_station_full_baseline_station_summary.csv",
    "stage165_paired": TABLES / "stage165_noaa_controlled_station_full_baseline_paired.csv",
    "stage165_not_estimable": TABLES / "stage165_noaa_controlled_station_full_baseline_not_estimable.csv",
    "stage166_detail": TABLES / "stage166_noaa_controlled_attack_environment_interaction_detail.csv",
    "stage166_summary": TABLES / "stage166_noaa_controlled_attack_environment_interaction_summary.csv",
    "stage166_paired": TABLES / "stage166_noaa_controlled_attack_environment_interaction_paired.csv",
    "stage166_not_estimable": TABLES / "stage166_noaa_controlled_attack_environment_interaction_not_estimable.csv",
    "stage167_detail": TABLES / "stage167_noaa_controlled_attack_environment_multiple_comparison_detail.csv",
    "stage167_not_estimable": TABLES / "stage167_noaa_controlled_attack_environment_multiple_comparison_not_estimable.csv",
    "stage168_detail": TABLES / "stage168_noaa_controlled_attack_environment_practical_effect_detail.csv",
    "stage168_not_estimable": TABLES / "stage168_noaa_controlled_attack_environment_practical_effect_not_estimable.csv",
    "stage169_detail": TABLES / "stage169_noaa_controlled_attack_intensity_full_baseline_detail.csv",
    "stage169_summary": TABLES / "stage169_noaa_controlled_attack_intensity_full_baseline_summary.csv",
    "stage169_type_detail": TABLES / "stage169_noaa_controlled_attack_intensity_full_baseline_type_detail.csv",
    "stage169_type_summary": TABLES / "stage169_noaa_controlled_attack_intensity_full_baseline_type_summary.csv",
    "stage169_paired": TABLES / "stage169_noaa_controlled_attack_intensity_full_baseline_paired.csv",
    "stage169_not_estimable": TABLES / "stage169_noaa_controlled_attack_intensity_full_baseline_not_estimable.csv",
    "stage170_detail": TABLES / "stage170_noaa_controlled_attack_duration_full_baseline_detail.csv",
    "stage170_summary": TABLES / "stage170_noaa_controlled_attack_duration_full_baseline_summary.csv",
    "stage170_type_detail": TABLES / "stage170_noaa_controlled_attack_duration_full_baseline_type_detail.csv",
    "stage170_type_summary": TABLES / "stage170_noaa_controlled_attack_duration_full_baseline_type_summary.csv",
    "stage170_paired": TABLES / "stage170_noaa_controlled_attack_duration_full_baseline_paired.csv",
    "stage170_not_estimable": TABLES / "stage170_noaa_controlled_attack_duration_full_baseline_not_estimable.csv",
    "stage171_detail": TABLES / "stage171_noaa_controlled_attack_position_full_baseline_detail.csv",
    "stage171_summary": TABLES / "stage171_noaa_controlled_attack_position_full_baseline_summary.csv",
    "stage171_type_detail": TABLES / "stage171_noaa_controlled_attack_position_full_baseline_type_detail.csv",
    "stage171_type_summary": TABLES / "stage171_noaa_controlled_attack_position_full_baseline_type_summary.csv",
    "stage171_paired": TABLES / "stage171_noaa_controlled_attack_position_full_baseline_paired.csv",
    "stage171_not_estimable": TABLES / "stage171_noaa_controlled_attack_position_full_baseline_not_estimable.csv",
    "stage172_detail": TABLES / "stage172_noaa_frozen_expanded_low_far_full_baseline_detail.csv",
    "stage172_summary": TABLES / "stage172_noaa_frozen_expanded_low_far_full_baseline_summary.csv",
    "stage172_paired": TABLES / "stage172_noaa_frozen_expanded_low_far_full_baseline_paired.csv",
    "stage172_not_estimable": TABLES / "stage172_noaa_frozen_expanded_low_far_full_baseline_not_estimable.csv",
    "stage173_detail": TABLES / "stage173_noaa_low_far_full_baseline_multiple_comparison_detail.csv",
    "stage173_not_estimable": TABLES / "stage173_noaa_low_far_full_baseline_multiple_comparison_not_estimable.csv",
    "stage174_detail": TABLES / "stage174_noaa_low_far_alarm_burden_audit_detail.csv",
    "stage174_station_detail": TABLES / "stage174_noaa_low_far_alarm_burden_audit_station_detail.csv",
    "stage174_month_detail": TABLES / "stage174_noaa_low_far_alarm_burden_audit_month_detail.csv",
    "stage174_burden": TABLES / "stage174_noaa_low_far_alarm_burden_audit_burden.csv",
    "stage175_detail": TABLES / "stage175_noaa_low_far_alarm_burden_stability_detail.csv",
    "stage175_comparisons": TABLES / "stage175_noaa_low_far_alarm_burden_stability_comparisons.csv",
    "stage176_detail": TABLES / "stage176_noaa_low_far_station_equal_weight_metrics_detail.csv",
    "stage176_station_detail": TABLES / "stage176_noaa_low_far_station_equal_weight_metrics_station_detail.csv",
    "stage176_comparisons": TABLES / "stage176_noaa_low_far_station_equal_weight_metrics_comparisons.csv",
    "stage177_detail": TABLES / "stage177_noaa_station_calibrated_operating_points_detail.csv",
    "stage177_station_detail": TABLES / "stage177_noaa_station_calibrated_operating_points_station_detail.csv",
    "stage177_baseline_comparisons": TABLES / "stage177_noaa_station_calibrated_operating_points_baseline_comparisons.csv",
    "stage177_calibration_comparisons": TABLES / "stage177_noaa_station_calibrated_operating_points_calibration_comparisons.csv",
    "stage178_detail": TABLES / "stage178_noaa_station_calibrated_alarm_budget_detail.csv",
    "stage178_station_detail": TABLES / "stage178_noaa_station_calibrated_alarm_budget_station_detail.csv",
    "stage178_baseline_comparisons": TABLES / "stage178_noaa_station_calibrated_alarm_budget_baseline_comparisons.csv",
    "stage178_calibration_comparisons": TABLES / "stage178_noaa_station_calibrated_alarm_budget_calibration_comparisons.csv",
    "stage179_detail": TABLES / "stage179_noaa_station_budget_loss_decomposition_detail.csv",
    "stage179_comparisons": TABLES / "stage179_noaa_station_budget_loss_decomposition_comparisons.csv",
    "stage180_summary": TABLES / "stage180_noaa_station_budget_row_tail_audit_summary.csv",
    "stage180_station_detail": TABLES / "stage180_noaa_station_budget_row_tail_audit_station_detail.csv",
    "stage180_station_type_detail": TABLES / "stage180_noaa_station_budget_row_tail_audit_station_type_detail.csv",
    "stage180_event_detail": TABLES / "stage180_noaa_station_budget_row_tail_audit_event_detail.csv",
    "stage180_event_aggregate": TABLES / "stage180_noaa_station_budget_row_tail_audit_event_aggregate.csv",
    "stage181_component_detail": TABLES / "stage181_noaa_non_stuck_failure_attribution_component_detail.csv",
    "stage181_type_aggregate": TABLES / "stage181_noaa_non_stuck_failure_attribution_type_aggregate.csv",
    "stage182_type_detail": TABLES / "stage182_noaa_non_stuck_candidate_features_type_detail.csv",
    "stage182_type_aggregate": TABLES / "stage182_noaa_non_stuck_candidate_features_type_aggregate.csv",
    "stage183_detail": TABLES / "stage183_noaa_candidate_low_far_constraint_detail.csv",
    "stage183_aggregate": TABLES / "stage183_noaa_candidate_low_far_constraint_aggregate.csv",
    "stage183_type_detail": TABLES / "stage183_noaa_candidate_low_far_constraint_type_detail.csv",
    "stage183_type_aggregate": TABLES / "stage183_noaa_candidate_low_far_constraint_type_aggregate.csv",
    "stage184_detail": TABLES / "stage184_noaa_validation_calibrated_low_far_detail.csv",
    "stage184_aggregate": TABLES / "stage184_noaa_validation_calibrated_low_far_aggregate.csv",
    "stage184_type_detail": TABLES / "stage184_noaa_validation_calibrated_low_far_type_detail.csv",
    "stage184_type_aggregate": TABLES / "stage184_noaa_validation_calibrated_low_far_type_aggregate.csv",
    "stage184_paired_vs_baseline": TABLES / "stage184_noaa_validation_calibrated_low_far_paired_vs_baseline.csv",
    "stage184_paired_vs_baseline_aggregate": TABLES / "stage184_noaa_validation_calibrated_low_far_paired_vs_baseline_aggregate.csv",
    "stage185_detail": TABLES / "stage185_noaa_validation_calibrated_combination_detail.csv",
    "stage185_type_detail": TABLES / "stage185_noaa_validation_calibrated_combination_type_detail.csv",
    "stage185_weight_selection": TABLES / "stage185_noaa_validation_calibrated_combination_weight_selection.csv",
    "stage185_weight_aggregate": TABLES / "stage185_noaa_validation_calibrated_combination_weight_aggregate.csv",
    "stage185_aggregate": TABLES / "stage185_noaa_validation_calibrated_combination_aggregate.csv",
    "stage185_paired_vs_baseline": TABLES / "stage185_noaa_validation_calibrated_combination_paired_vs_baseline.csv",
    "stage186_detail": TABLES / "stage186_noaa_targeted_nonstuck_feature_response_detail.csv",
    "stage186_aggregate": TABLES / "stage186_noaa_targeted_nonstuck_feature_response_aggregate.csv",
    "stage186_correlation": TABLES / "stage186_noaa_targeted_nonstuck_feature_response_correlation.csv",
    "stage186_correlation_aggregate": TABLES / "stage186_noaa_targeted_nonstuck_feature_response_correlation_aggregate.csv",
    "stage187_detail": TABLES / "stage187_noaa_targeted_features_validation_calibrated_detail.csv",
    "stage187_aggregate": TABLES / "stage187_noaa_targeted_features_validation_calibrated_aggregate.csv",
    "stage187_type_detail": TABLES / "stage187_noaa_targeted_features_validation_calibrated_type_detail.csv",
    "stage187_type_aggregate": TABLES / "stage187_noaa_targeted_features_validation_calibrated_type_aggregate.csv",
    "stage187_paired_vs_baseline": TABLES / "stage187_noaa_targeted_features_validation_calibrated_paired_vs_baseline.csv",
    "stage187_paired_vs_baseline_aggregate": TABLES / "stage187_noaa_targeted_features_validation_calibrated_paired_vs_baseline_aggregate.csv",
    "stage188_detail": TABLES / "stage188_noaa_two_feature_stability_audit_detail.csv",
    "stage188_type_detail": TABLES / "stage188_noaa_two_feature_stability_audit_type_detail.csv",
    "stage188_weight_selection": TABLES / "stage188_noaa_two_feature_stability_audit_weight_selection.csv",
    "stage188_weight_aggregate": TABLES / "stage188_noaa_two_feature_stability_audit_weight_aggregate.csv",
    "stage188_aggregate": TABLES / "stage188_noaa_two_feature_stability_audit_aggregate.csv",
    "stage188_paired_vs_baseline": TABLES / "stage188_noaa_two_feature_stability_audit_paired_vs_baseline.csv",
    "stage188_stability": TABLES / "stage188_noaa_two_feature_stability_audit_stability.csv",
    "stage189_detail": TABLES / "stage189_noaa_event_level_failure_decomposition_detail.csv",
    "stage189_overall_aggregate": TABLES / "stage189_noaa_event_level_failure_decomposition_overall_aggregate.csv",
    "stage189_type_aggregate": TABLES / "stage189_noaa_event_level_failure_decomposition_type_aggregate.csv",
    "stage189_length_aggregate": TABLES / "stage189_noaa_event_level_failure_decomposition_length_aggregate.csv",
    "stage189_risk_aggregate": TABLES / "stage189_noaa_event_level_failure_decomposition_risk_aggregate.csv",
    "stage189_station_aggregate": TABLES / "stage189_noaa_event_level_failure_decomposition_station_aggregate.csv",
    "stage190_detail": TABLES / "stage190_noaa_event_reranking_validation_detail.csv",
    "stage190_type_detail": TABLES / "stage190_noaa_event_reranking_validation_type_detail.csv",
    "stage190_selection": TABLES / "stage190_noaa_event_reranking_validation_selection.csv",
    "stage190_selection_aggregate": TABLES / "stage190_noaa_event_reranking_validation_selection_aggregate.csv",
    "stage190_aggregate": TABLES / "stage190_noaa_event_reranking_validation_aggregate.csv",
    "stage190_paired_vs_baseline": TABLES / "stage190_noaa_event_reranking_validation_paired_vs_baseline.csv",
    "stage190_stability": TABLES / "stage190_noaa_event_reranking_validation_stability.csv",
    "stage191_detail": TABLES / "stage191_noaa_reranking_focused_diagnostics_detail.csv",
    "stage191_type_aggregate": TABLES / "stage191_noaa_reranking_focused_diagnostics_type_aggregate.csv",
    "stage191_settings": TABLES / "stage191_noaa_reranking_focused_diagnostics_settings.csv",
    "stage191_consensus": TABLES / "stage191_noaa_reranking_focused_diagnostics_consensus.csv",
    "stage191_fold_stability": TABLES / "stage191_noaa_reranking_focused_diagnostics_fold_stability.csv",
    "stage191_overall_aggregate": TABLES / "stage191_noaa_reranking_focused_diagnostics_overall_aggregate.csv",
    "stage192_detail": TABLES / "stage192_noaa_repeat_gated_reranking_detail.csv",
    "stage192_type_detail": TABLES / "stage192_noaa_repeat_gated_reranking_type_detail.csv",
    "stage192_selection": TABLES / "stage192_noaa_repeat_gated_reranking_selection.csv",
    "stage192_selection_aggregate": TABLES / "stage192_noaa_repeat_gated_reranking_selection_aggregate.csv",
    "stage192_aggregate": TABLES / "stage192_noaa_repeat_gated_reranking_aggregate.csv",
    "stage192_paired_vs_baseline": TABLES / "stage192_noaa_repeat_gated_reranking_paired_vs_baseline.csv",
    "stage192_stability": TABLES / "stage192_noaa_repeat_gated_reranking_stability.csv",
    "stage193_detail": TABLES / "stage193_noaa_persistent_repeat_protection_detail.csv",
    "stage193_type_detail": TABLES / "stage193_noaa_persistent_repeat_protection_type_detail.csv",
    "stage193_selection": TABLES / "stage193_noaa_persistent_repeat_protection_selection.csv",
    "stage193_selection_aggregate": TABLES / "stage193_noaa_persistent_repeat_protection_selection_aggregate.csv",
    "stage193_aggregate": TABLES / "stage193_noaa_persistent_repeat_protection_aggregate.csv",
    "stage193_paired_vs_baseline": TABLES / "stage193_noaa_persistent_repeat_protection_paired_vs_baseline.csv",
    "stage193_stability": TABLES / "stage193_noaa_persistent_repeat_protection_stability.csv",
    "stage194_detail": TABLES / "stage194_noaa_position_perturbation_robustness_detail.csv",
    "stage194_aggregate": TABLES / "stage194_noaa_position_perturbation_robustness_aggregate.csv",
    "stage194_paired_vs_baseline": TABLES / "stage194_noaa_position_perturbation_robustness_paired_vs_baseline.csv",
    "stage194_placement_redundancy": TABLES / "stage194_noaa_position_perturbation_robustness_placement_redundancy.csv",
    "stage195_detail": TABLES / "stage195_noaa_seed_randomized_placement_detail.csv",
    "stage195_placement_detail": TABLES / "stage195_noaa_seed_randomized_placement_placement_detail.csv",
    "stage195_placement_summary": TABLES / "stage195_noaa_seed_randomized_placement_placement_summary.csv",
    "stage195_aggregate": TABLES / "stage195_noaa_seed_randomized_placement_aggregate.csv",
    "stage195_paired_vs_baseline": TABLES / "stage195_noaa_seed_randomized_placement_paired_vs_baseline.csv",
    "stage196_detail": TABLES / "stage196_noaa_seed_randomized_independent_replication_detail.csv",
    "stage196_placement_detail": TABLES / "stage196_noaa_seed_randomized_independent_replication_placement_detail.csv",
    "stage196_placement_summary": TABLES / "stage196_noaa_seed_randomized_independent_replication_placement_summary.csv",
    "stage196_aggregate": TABLES / "stage196_noaa_seed_randomized_independent_replication_aggregate.csv",
    "stage196_paired_vs_baseline": TABLES / "stage196_noaa_seed_randomized_independent_replication_paired_vs_baseline.csv",
    "stage197_original_detail": TABLES / "stage197_noaa_seed_randomized_station_dependence_original_detail.csv",
    "stage197_original_station_detail": TABLES / "stage197_noaa_seed_randomized_station_dependence_original_station_detail.csv",
    "stage197_original_station_summary": TABLES / "stage197_noaa_seed_randomized_station_dependence_original_station_summary.csv",
    "stage197_original_loso": TABLES / "stage197_noaa_seed_randomized_station_dependence_original_loso.csv",
    "stage197_independent_detail": TABLES / "stage197_noaa_seed_randomized_station_dependence_independent_detail.csv",
    "stage197_independent_station_detail": TABLES / "stage197_noaa_seed_randomized_station_dependence_independent_station_detail.csv",
    "stage197_independent_station_summary": TABLES / "stage197_noaa_seed_randomized_station_dependence_independent_station_summary.csv",
    "stage197_independent_loso": TABLES / "stage197_noaa_seed_randomized_station_dependence_independent_loso.csv",
    "stage198_original_detail": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_original_detail.csv",
    "stage198_original_selections": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_original_selections.csv",
    "stage198_original_selection_counts": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_original_selection_counts.csv",
    "stage198_original_aggregate": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_original_aggregate.csv",
    "stage198_original_paired_vs_baseline": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_original_paired_vs_baseline.csv",
    "stage198_independent_detail": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_independent_detail.csv",
    "stage198_independent_selections": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_independent_selections.csv",
    "stage198_independent_selection_counts": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_independent_selection_counts.csv",
    "stage198_independent_aggregate": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_independent_aggregate.csv",
    "stage198_independent_paired_vs_baseline": TABLES / "stage198_noaa_seed_randomized_cross_fold_transfer_independent_paired_vs_baseline.csv",
    "stage199_detail": TABLES / "stage199_liupanshan_cross_station_protection_transfer_detail.csv",
    "stage199_selections": TABLES / "stage199_liupanshan_cross_station_protection_transfer_selections.csv",
    "stage199_station_summary": TABLES / "stage199_liupanshan_cross_station_protection_transfer_station_summary.csv",
    "stage199_aggregate": TABLES / "stage199_liupanshan_cross_station_protection_transfer_aggregate.csv",
    "stage199_paired_vs_baseline": TABLES / "stage199_liupanshan_cross_station_protection_transfer_paired_vs_baseline.csv",
    "stage200_detail": TABLES / "stage200_liupanshan_protection_failure_attribution_detail.csv",
    "stage200_aggregate": TABLES / "stage200_liupanshan_protection_failure_attribution_aggregate.csv",
    "stage200_paired": TABLES / "stage200_liupanshan_protection_failure_attribution_paired.csv",
    "stage201_detail": TABLES / "stage201_liupanshan_conservative_protection_transfer_detail.csv",
    "stage201_aggregate": TABLES / "stage201_liupanshan_conservative_protection_transfer_aggregate.csv",
    "stage201_paired": TABLES / "stage201_liupanshan_conservative_protection_transfer_paired.csv",
    "stage204_stations": TABLES / "stage204_dwd_protocol_preregistration_stations.csv",
    "stage205_detail": TABLES / "stage205_dwd_official_download_detail.csv",
    "stage206_station_audit": TABLES / "stage206_dwd_normalized_station_audit.csv",
    "stage206_archive_audit": TABLES / "stage206_dwd_normalized_archive_audit.csv",
    "stage207_evaluation": TABLES / "stage207_dwd_controlled_attack_evaluation.csv",
    "stage207_events": TABLES / "stage207_dwd_controlled_attack_events.csv",
    "stage207_type_detail": TABLES / "stage207_dwd_controlled_attack_type_detail.csv",
    "stage207_type_aggregate": TABLES / "stage207_dwd_controlled_attack_type_aggregate.csv",
    "stage208_evaluation": TABLES / "stage208_dwd_persistent_repeat_protection_evaluation.csv",
    "stage208_events": TABLES / "stage208_dwd_persistent_repeat_protection_events.csv",
    "stage208_selection": TABLES / "stage208_dwd_persistent_repeat_protection_selection.csv",
    "stage208_selection_counts": TABLES / "stage208_dwd_persistent_repeat_protection_selection_counts.csv",
    "stage208_type_detail": TABLES / "stage208_dwd_persistent_repeat_protection_type_detail.csv",
    "stage208_type_aggregate": TABLES / "stage208_dwd_persistent_repeat_protection_type_aggregate.csv",
    "stage208_paired": TABLES / "stage208_dwd_persistent_repeat_protection_paired_vs_stage207.csv",
    "stage209": TABLES / "stage209_dwd_station_cluster_uncertainty.csv",
    "stage209_station": TABLES / "stage209_dwd_station_cluster_uncertainty_station.csv",
    "stage209_paired_station": TABLES / "stage209_dwd_station_cluster_uncertainty_paired_station.csv",
    "stage210_evaluation": TABLES / "stage210_dwd_seed_randomized_replication_evaluation.csv",
    "stage210_events": TABLES / "stage210_dwd_seed_randomized_replication_events.csv",
    "stage210_selection": TABLES / "stage210_dwd_seed_randomized_replication_selection.csv",
    "stage210_type_detail": TABLES / "stage210_dwd_seed_randomized_replication_type_detail.csv",
    "stage210_station_detail": TABLES / "stage210_dwd_seed_randomized_replication_station_detail.csv",
    "stage210_paired": TABLES / "stage210_dwd_seed_randomized_replication_paired.csv",
    "stage211": TABLES / "stage211_dwd_replication_station_uncertainty.csv",
    "stage211_station": TABLES / "stage211_dwd_replication_station_uncertainty_station.csv",
    "stage211_paired_station": TABLES / "stage211_dwd_replication_station_uncertainty_paired_station.csv",
    "stage212": TABLES / "stage212_dwd_operating_curve.csv",
    "stage212_seed_direction": TABLES / "stage212_dwd_operating_curve_seed_direction.csv",
    "stage213_checks": TABLES / "stage213_dwd_external_evidence_closure_checks.csv",
    "stage214_stations": TABLES / "stage214_swiss_protocol_preregistration_stations.csv",
    "stage215_detail": TABLES / "stage215_swiss_official_download_detail.csv",
    "stage216_normalized": TABLES / "stage216_swiss_normalized.parquet",
    "stage216_station_audit": TABLES / "stage216_swiss_normalized_station_audit.csv",
    "stage216_archive_audit": TABLES / "stage216_swiss_normalized_archive_audit.csv",
    "stage217_evaluation": TABLES / "stage217_swiss_controlled_attack_evaluation.csv",
    "stage217_events": TABLES / "stage217_swiss_controlled_attack_events.csv",
    "stage217_type_detail": TABLES / "stage217_swiss_controlled_attack_type_detail.csv",
    "stage217_type_aggregate": TABLES / "stage217_swiss_controlled_attack_type_aggregate.csv",
    "stage217_station_detail": TABLES / "stage217_swiss_controlled_attack_station_detail.csv",
    "stage217_summary": TABLES / "stage217_swiss_controlled_attack_summary.csv",
    "stage218_attack_type": TABLES / "stage218_swiss_negative_result_uncertainty_attack_type.csv",
    "stage218_station": TABLES / "stage218_swiss_negative_result_uncertainty_station.csv",
    "stage218_sign_flip": TABLES / "stage218_swiss_negative_result_uncertainty_sign_flip.csv",
    "stage218_provider_contrast": TABLES / "stage218_swiss_negative_result_uncertainty_provider_contrast.csv",
    "stage219_evaluation": TABLES / "stage219_swiss_dwd_protection_transfer_evaluation.csv",
    "stage219_events": TABLES / "stage219_swiss_dwd_protection_transfer_events.csv",
    "stage219_type_detail": TABLES / "stage219_swiss_dwd_protection_transfer_type_detail.csv",
    "stage219_type_aggregate": TABLES / "stage219_swiss_dwd_protection_transfer_type_aggregate.csv",
    "stage219_station_detail": TABLES / "stage219_swiss_dwd_protection_transfer_station_detail.csv",
    "stage219_station_event": TABLES / "stage219_swiss_dwd_protection_transfer_station_event_detail.csv",
    "stage219_selection": TABLES / "stage219_swiss_dwd_protection_transfer_selection.csv",
    "stage219_paired": TABLES / "stage219_swiss_dwd_protection_transfer_paired.csv",
    "stage220_metrics": TABLES / "stage220_swiss_dwd_transfer_uncertainty_metrics.csv",
    "stage220_station_bootstrap": TABLES / "stage220_swiss_dwd_transfer_uncertainty_station_bootstrap.csv",
    "stage220_paired_station": TABLES / "stage220_swiss_dwd_transfer_uncertainty_paired_station.csv",
    "stage221_curve": TABLES / "stage221_swiss_dwd_transfer_operating_curve_curve.csv",
    "stage221_arm_aggregate": TABLES / "stage221_swiss_dwd_transfer_operating_curve_arm_aggregate.csv",
    "stage221_evaluation": TABLES / "stage221_swiss_dwd_transfer_operating_curve_evaluation.csv",
    "stage221_station_detail": TABLES / "stage221_swiss_dwd_transfer_operating_curve_station_detail.csv",
    "stage221_station_event": TABLES / "stage221_swiss_dwd_transfer_operating_curve_station_event_detail.csv",
    "stage222_evaluation": TABLES / "stage222_noaa_dwd_protection_transfer_evaluation.csv",
    "stage222_events": TABLES / "stage222_noaa_dwd_protection_transfer_events.csv",
    "stage222_type_detail": TABLES / "stage222_noaa_dwd_protection_transfer_type_detail.csv",
    "stage222_type_aggregate": TABLES / "stage222_noaa_dwd_protection_transfer_type_aggregate.csv",
    "stage222_station_detail": TABLES / "stage222_noaa_dwd_protection_transfer_station_detail.csv",
    "stage222_station_event": TABLES / "stage222_noaa_dwd_protection_transfer_station_event_detail.csv",
    "stage222_selection": TABLES / "stage222_noaa_dwd_protection_transfer_selection.csv",
    "stage223_metrics": TABLES / "stage223_noaa_transfer_uncertainty_metrics.csv",
    "stage223_paired_station": TABLES / "stage223_noaa_transfer_uncertainty_paired_station.csv",
    "stage224_contract": TABLES / "stage224_knmi_protocol_preregistration_contract.csv",
    "stage225_detail": TABLES / "stage225_knmi_official_download_detail.csv",
    "stage225_stations": TABLES / "stage225_knmi_stations.csv",
    "stage226_normalized": TABLES / "stage226_knmi_normalized.parquet",
    "stage226_station_audit": TABLES / "stage226_knmi_normalized_station_audit.csv",
    "stage226_archive_audit": TABLES / "stage226_knmi_normalized_archive_audit.csv",
    "stage226a_year": TABLES / "stage226a_knmi_missingness_audit_year.csv",
    "stage226a_station": TABLES / "stage226a_knmi_missingness_audit_station.csv",
    "stage226a_split": TABLES / "stage226a_knmi_missingness_audit_split.csv",
    "stage227_evaluation": TABLES / "stage227_knmi_controlled_attack_evaluation.csv",
    "stage227_events": TABLES / "stage227_knmi_controlled_attack_events.csv",
    "stage227_station_detail": TABLES / "stage227_knmi_controlled_attack_station_detail.csv",
    "stage227_summary": TABLES / "stage227_knmi_controlled_attack_summary.csv",
    "stage227_type_detail": TABLES / "stage227_knmi_controlled_attack_type_detail.csv",
    "stage227_type_aggregate": TABLES / "stage227_knmi_controlled_attack_type_aggregate.csv",
    "stage228_evaluation": TABLES / "stage228_knmi_dwd_protection_transfer_evaluation.csv",
    "stage228_events": TABLES / "stage228_knmi_dwd_protection_transfer_events.csv",
    "stage228_paired": TABLES / "stage228_knmi_dwd_protection_transfer_paired.csv",
    "stage228_selection": TABLES / "stage228_knmi_dwd_protection_transfer_selection.csv",
    "stage228_station_detail": TABLES / "stage228_knmi_dwd_protection_transfer_station_detail.csv",
    "stage228_station_event": TABLES / "stage228_knmi_dwd_protection_transfer_station_event_detail.csv",
    "stage228_type_detail": TABLES / "stage228_knmi_dwd_protection_transfer_type_detail.csv",
    "stage228_type_aggregate": TABLES / "stage228_knmi_dwd_protection_transfer_type_aggregate.csv",
    "stage229_metrics": TABLES / "stage229_knmi_transfer_uncertainty_metrics.csv",
    "stage229_paired_station": TABLES / "stage229_knmi_transfer_uncertainty_paired_station.csv",
    "stage230_sources": TABLES / "stage230_cross_provider_evidence_synthesis_sources.csv",
    "stage230_transfers": TABLES / "stage230_cross_provider_evidence_synthesis_transfers.csv",
    "stage230_checks": TABLES / "stage230_cross_provider_evidence_synthesis_checks.csv",
    "stage231_sources": TABLES / "stage231_cross_provider_robustness_sources.csv",
    "stage231_leave_one_provider_out": TABLES / "stage231_cross_provider_robustness_leave_one_provider_out.csv",
    "stage231_hierarchical_bootstrap": TABLES / "stage231_cross_provider_robustness_hierarchical_bootstrap.csv",
    "stage231_station_effects": TABLES / "stage231_cross_provider_robustness_station_effects.csv",
    "stage231_checks": TABLES / "stage231_cross_provider_robustness_checks.csv",
    "stage232_sources": TABLES / "stage232_attack_type_robustness_sources.csv",
    "stage232_provider_equal_by_type": TABLES / "stage232_attack_type_robustness_provider_equal_by_type.csv",
    "stage232_leave_one_provider_out_by_type": TABLES / "stage232_attack_type_robustness_leave_one_provider_out_by_type.csv",
    "stage232_leave_one_attack_type_out": TABLES / "stage232_attack_type_robustness_leave_one_attack_type_out.csv",
    "stage232_hierarchical_bootstrap_by_type": TABLES / "stage232_attack_type_robustness_hierarchical_bootstrap_by_type.csv",
    "stage232_checks": TABLES / "stage232_attack_type_robustness_checks.csv",
    "stage233_targets": TABLES / "stage233_valid_attack_correction_audit_targets.csv",
    "stage233_checks": TABLES / "stage233_valid_attack_correction_audit_checks.csv",
    "stage240_evaluation": TABLES / "stage240_nonstuck_balanced_selection_evaluation.csv",
    "stage240_station_detail": TABLES / "stage240_nonstuck_balanced_selection_station_detail.csv",
    "stage240_type_detail": TABLES / "stage240_nonstuck_balanced_selection_type_detail.csv",
    "stage240_selection": TABLES / "stage240_nonstuck_balanced_selection_selection.csv",
    "stage240_evaluation_aggregate": TABLES / "stage240_nonstuck_balanced_selection_evaluation_aggregate.csv",
    "stage240_provider_equal": TABLES / "stage240_nonstuck_balanced_selection_provider_equal.csv",
    "stage240_provider_station": TABLES / "stage240_nonstuck_balanced_selection_provider_station.csv",
    "stage240_leave_one_provider_out": TABLES / "stage240_nonstuck_balanced_selection_leave_one_provider_out.csv",
    "stage240_nonstuck_type_detail": TABLES / "stage240_nonstuck_balanced_selection_nonstuck_type_detail.csv",
    "stage240_nonstuck_type_equal": TABLES / "stage240_nonstuck_balanced_selection_nonstuck_type_equal.csv",
    "stage240_selection_counts": TABLES / "stage240_nonstuck_balanced_selection_selection_counts.csv",
    "stage242_evaluation": TABLES / "stage242_multisource_transfer_matrix_evaluation.csv",
    "stage242_station_detail": TABLES / "stage242_multisource_transfer_matrix_station_detail.csv",
    "stage242_local_reproduction": TABLES / "stage242_multisource_transfer_matrix_local_reproduction.csv",
    "stage243_station_detail": TABLES / "stage243_knmi_negative_case_decomposition_station_detail.csv",
    "stage243_seed_detail": TABLES / "stage243_knmi_negative_case_decomposition_seed_detail.csv",
    "stage243_metric_detail": TABLES / "stage243_knmi_negative_case_decomposition_metric_detail.csv",
    "stage243_summary": TABLES / "stage243_knmi_negative_case_decomposition_summary.csv",
    "stage244_evaluation": TABLES / "stage244_cross_provider_nonstuck_candidate_features_evaluation.csv",
    "stage244_station_detail": TABLES / "stage244_cross_provider_nonstuck_candidate_features_station_detail.csv",
    "stage244_type_detail": TABLES / "stage244_cross_provider_nonstuck_candidate_features_type_detail.csv",
    "stage244_selection": TABLES / "stage244_cross_provider_nonstuck_candidate_features_selection.csv",
    "stage244_evaluation_aggregate": TABLES / "stage244_cross_provider_nonstuck_candidate_features_evaluation_aggregate.csv",
    "stage244_provider_station": TABLES / "stage244_cross_provider_nonstuck_candidate_features_provider_station.csv",
    "stage244_leave_one_provider_out": TABLES / "stage244_cross_provider_nonstuck_candidate_features_leave_one_provider_out.csv",
    "stage244_nonstuck_type_detail": TABLES / "stage244_cross_provider_nonstuck_candidate_features_nonstuck_type_detail.csv",
    "stage244_nonstuck_type_equal": TABLES / "stage244_cross_provider_nonstuck_candidate_features_nonstuck_type_equal.csv",
    "stage244_selection_counts": TABLES / "stage244_cross_provider_nonstuck_candidate_features_selection_counts.csv",
    "stage245_evaluation": TABLES / "stage245_cross_provider_nonstuck_seed_replication_evaluation.csv",
    "stage245_station_detail": TABLES / "stage245_cross_provider_nonstuck_seed_replication_station_detail.csv",
    "stage245_type_detail": TABLES / "stage245_cross_provider_nonstuck_seed_replication_type_detail.csv",
    "stage245_selection": TABLES / "stage245_cross_provider_nonstuck_seed_replication_selection.csv",
    "stage245_evaluation_aggregate": TABLES / "stage245_cross_provider_nonstuck_seed_replication_evaluation_aggregate.csv",
    "stage245_provider_station": TABLES / "stage245_cross_provider_nonstuck_seed_replication_provider_station.csv",
    "stage245_leave_one_provider_out": TABLES / "stage245_cross_provider_nonstuck_seed_replication_leave_one_provider_out.csv",
    "stage245_nonstuck_type_detail": TABLES / "stage245_cross_provider_nonstuck_seed_replication_nonstuck_type_detail.csv",
    "stage245_nonstuck_type_equal": TABLES / "stage245_cross_provider_nonstuck_seed_replication_nonstuck_type_equal.csv",
    "stage246_table1": TABLES / "stage246_frozen_submission_package_table1_primary_summary.csv",
    "stage246_table2": TABLES / "stage246_frozen_submission_package_table2_cross_provider_transfer.csv",
    "stage246_table3": TABLES / "stage246_frozen_submission_package_table3_attack_type.csv",
    "stage246_table4": TABLES / "stage246_frozen_submission_package_table4_nonstuck_replication.csv",
    "stage246_table5": TABLES / "stage246_frozen_submission_package_table5_recovery_safety.csv",
    "stage246_table6": TABLES / "stage246_frozen_submission_package_table6_provider_balanced_nonstuck.csv",
    "stage246_table7": TABLES / "stage246_frozen_submission_package_table7_fair_baseline.csv",
    "stage246_table8": TABLES / "stage246_frozen_submission_package_table8_natural_validation.csv",
    "stage249_requirements": TABLES / "stage249_jisa_submission_gate_requirements.csv",
    "stage250_risks": TABLES / "stage250_jisa_risk_closure_audit_risks.csv",
    "stage251_risk_ceilings": TABLES / "stage251_jisa_stable_submission_objective_risk_ceilings.csv",
    "stage251_workstreams": TABLES / "stage251_jisa_stable_submission_objective_workstreams.csv",
    "stage252_runs": TABLES / "stage252_authorized_testbed_analysis_runs.csv",
    "stage253_rows": TABLES / "stage253_residual_risk_optimization_plan_rows.csv",
    "stage116_yearly": TABLES / "stage116_noaa_quality_label_permutation_yearly.csv",
    "stage116_aggregate": TABLES / "stage116_noaa_quality_label_permutation_aggregate.csv",
}
EXPECTED_ROWS = {
    "stage98": 500,
    "stage100": 1100,
    "stage101": 240,
    "stage102_all": 240,
    "stage102_selected": 20,
    "stage105_type": 360,
    "stage105_overall": 60,
    "stage107_detail": 45,
    "stage107_station": 495,
    "stage109": 45,
    "stage110": 6,
    "stage111": 45,
    "stage113": 3,
    "stage117": 7,
    "stage118_detail": 15,
    "stage118": 3,
    "stage118_by_type": 27,
    "stage119_detail": 15,
    "stage119": 3,
    "stage119_by_type": 27,
    "stage120_detail": 35,
    "stage120": 7,
    "stage120_seed": 5,
    "stage121_candidates": 150,
    "stage121_selected": 5,
    "stage121_station": 35,
    "stage122_candidates": 210,
    "stage122_selected": 35,
    "stage122_station": 7,
    "stage122_seed": 5,
    "stage123": 12,
    "stage124": 35,
    "stage124_detail": 31290,
    "stage124_valid_detail": 187320,
    "stage124_station": 7,
    "stage124_seed": 5,
    "stage125": 16,
    "stage126_selected": 140,
    "stage126": 4,
    "stage126_station": 28,
    "stage127": 12,
    "stage128_detail": 140,
    "stage128_station": 28,
    "stage128": 4,
    "stage129_detail": 1260,
    "stage129_selected": 70,
    "stage129": 4,
    "stage129_counts": 3,
    "stage130_detail": 120,
    "stage130_selected": 10,
    "stage130": 4,
    "stage130_counts": 2,
    "stage131_detail": 280,
    "stage131": 14,
    "stage131_counts": 27,
    "stage132_detail": 80,
    "stage132": 12,
    "stage133_detail": 70,
    "stage133_station": 70,
    "stage133": 2,
    "stage134_detail": 840,
    "stage134_selected": 70,
    "stage134": 3,
    "stage134_counts": 6,
    "stage135_detail": 5,
    "stage135_attack_detail": 90,
    "stage135_attack_aggregate": 18,
    "stage136_detail": 7,
    "stage137_detail": 5,
    "stage137_attack_detail": 180,
    "stage137_attack_aggregate": 36,
    "stage138_detail": 20,
    "stage138_aggregate": 4,
    "stage138_paired": 15,
    "stage139_detail": 140,
    "stage139_selected": 5,
    "stage139_paired": 5,
    "stage139_aggregate": 2,
    "stage139_metric_paired": 5,
    "stage140_detail": 140,
    "stage140_selected": 5,
    "stage140_paired": 5,
    "stage140_aggregate": 2,
    "stage140_metric_paired": 5,
    "stage141_detail": 100,
    "stage141_selected": 20,
    "stage141_paired": 20,
    "stage141_fold": 20,
    "stage141_attack": 240,
    "stage141_station": 440,
    "stage141_low_far": 8,
    "stage142_detail": 75,
    "stage142_station": 825,
    "stage142_aggregate": 75,
    "stage142_station_summary": 75,
    "stage142_paired": 6,
    "stage143_detail": 120,
    "stage143_selected": 20,
    "stage143_aggregate_all": 6,
    "stage143_aggregate_selected": 3,
    "stage143_selection_counts": 3,
    "stage143_paired": 10,
    "stage144_aggregate": 4,
    "stage144_detail": 20,
    "stage144_selection_counts": 2,
    "stage144_paired": 5,
    "stage145_station": 11,
    "stage145_paired": 4,
    "stage145_extremes": 4,
    "stage146_year": 4,
    "stage146_paired": 10,
    "stage147_detail": 24,
    "stage147_paired": 6,
    "stage148_detail": 480,
    "stage148_aggregate": 24,
    "stage149_detail": 36,
    "stage149_paired": 24,
    "stage150_detail": 24,
    "stage151_detail": 80,
    "stage151_type_detail": 480,
    "stage151_aggregate": 4,
    "stage151_type_aggregate": 24,
    "stage152_detail": 1100,
    "stage152_station_aggregate": 55,
    "stage152_method_summary": 5,
    "stage152_paired": 16,
    "stage153_detail": 825,
    "stage153_summary": 75,
    "stage153_contrast_summary": 75,
    "stage153_paired": 12,
    "stage154_summary": 15,
    "stage154_paired": 8,
    "stage155_detail": 600,
    "stage155_station": 5700,
    "stage155_paired": 96,
    "stage156_detail": 285,
    "stage156_station": 3000,
    "stage156_paired": 48,
    "stage157": 144,
    "stage158_detail": 270,
    "stage158_paired": 33,
    "stage159_detail": 33,
    "stage159_not_estimable": 1,
    "stage160_detail": 33,
    "stage160_not_estimable": 2,
    "stage161_detail": 2900,
    "stage161_station": 154,
    "stage162_station_detail": 364,
    "stage162_summary": 34,
    "stage162_not_estimable": 1,
    "stage162_weak": 34,
    "stage163_detail": 675,
    "stage163_summary": 35,
    "stage163_paired": 132,
    "stage163_not_estimable": 8,
    "stage164_detail": 132,
    "stage164_not_estimable": 8,
    "stage165_detail": 7250,
    "stage165_station_summary": 385,
    "stage165_paired": 136,
    "stage165_not_estimable": 4,
    "stage166_detail": 2525,
    "stage166_summary": 145,
    "stage166_paired": 540,
    "stage166_not_estimable": 40,
    "stage167_detail": 540,
    "stage167_not_estimable": 40,
    "stage168_detail": 540,
    "stage168_not_estimable": 40,
    "stage169_detail": 400,
    "stage169_summary": 20,
    "stage169_type_detail": 2400,
    "stage169_type_summary": 120,
    "stage169_paired": 80,
    "stage169_not_estimable": 0,
    "stage170_detail": 300,
    "stage170_summary": 15,
    "stage170_type_detail": 1800,
    "stage170_type_summary": 90,
    "stage170_paired": 60,
    "stage170_not_estimable": 0,
    "stage171_detail": 300,
    "stage171_summary": 15,
    "stage171_type_detail": 1800,
    "stage171_type_summary": 90,
    "stage171_paired": 60,
    "stage171_not_estimable": 0,
    "stage172_detail": 600,
    "stage172_summary": 30,
    "stage172_paired": 120,
    "stage172_not_estimable": 0,
    "stage173_detail": 120,
    "stage173_not_estimable": 0,
    "stage174_detail": 600,
    "stage174_burden": 30,
    "stage175_detail": 1200,
    "stage175_comparisons": 144,
    "stage176_detail": 600,
    "stage176_station_detail": 6600,
    "stage176_comparisons": 192,
    "stage177_detail": 600,
    "stage177_station_detail": 6600,
    "stage177_baseline_comparisons": 96,
    "stage177_calibration_comparisons": 24,
    "stage178_detail": 1800,
    "stage178_station_detail": 19800,
    "stage178_baseline_comparisons": 288,
    "stage178_calibration_comparisons": 72,
    "stage179_detail": 1800,
    "stage179_comparisons": 360,
    "stage180_summary": 120,
    "stage180_station_detail": 1320,
    "stage180_station_type_detail": 7920,
    "stage180_event_detail": 7920,
    "stage180_event_aggregate": 36,
    "stage181_component_detail": 72,
    "stage181_type_aggregate": 36,
    "stage182_type_detail": 360,
    "stage182_type_aggregate": 18,
    "stage183_detail": 240,
    "stage183_aggregate": 36,
    "stage183_type_detail": 1440,
    "stage183_type_aggregate": 216,
    "stage184_detail": 1200,
    "stage184_aggregate": 60,
    "stage184_type_detail": 7200,
    "stage184_type_aggregate": 360,
    "stage184_paired_vs_baseline": 4800,
    "stage184_paired_vs_baseline_aggregate": 240,
    "stage185_detail": 240,
    "stage185_type_detail": 1440,
    "stage185_weight_selection": 240,
    "stage185_weight_aggregate": 62,
    "stage185_aggregate": 12,
    "stage185_paired_vs_baseline": 84,
    "stage186_detail": 1920,
    "stage186_aggregate": 96,
    "stage186_correlation": 1280,
    "stage186_correlation_aggregate": 64,
    "stage187_detail": 3200,
    "stage187_aggregate": 160,
    "stage187_type_detail": 19200,
    "stage187_type_aggregate": 960,
    "stage187_paired_vs_baseline": 17920,
    "stage187_paired_vs_baseline_aggregate": 896,
    "stage188_detail": 240,
    "stage188_type_detail": 1440,
    "stage188_weight_selection": 240,
    "stage188_weight_aggregate": 35,
    "stage188_aggregate": 12,
    "stage188_paired_vs_baseline": 84,
    "stage188_stability": 12,
    "stage189_detail": 7920,
    "stage189_overall_aggregate": 6,
    "stage189_type_aggregate": 36,
    "stage189_length_aggregate": 12,
    "stage189_risk_aggregate": 54,
    "stage189_station_aggregate": 66,
    "stage190_detail": 240,
    "stage190_type_detail": 1440,
    "stage190_selection": 240,
    "stage190_selection_aggregate": 47,
    "stage190_aggregate": 12,
    "stage190_paired_vs_baseline": 84,
    "stage190_stability": 12,
    "stage191_detail": 720,
    "stage191_type_aggregate": 36,
    "stage191_settings": 21,
    "stage191_consensus": 6,
    "stage191_fold_stability": 24,
    "stage191_overall_aggregate": 6,
    "stage192_detail": 240,
    "stage192_type_detail": 1440,
    "stage192_selection": 240,
    "stage192_selection_aggregate": 39,
    "stage192_aggregate": 12,
    "stage192_paired_vs_baseline": 84,
    "stage192_stability": 12,
    "stage193_detail": 240,
    "stage193_type_detail": 1440,
    "stage193_selection": 240,
    "stage193_selection_aggregate": 50,
    "stage193_aggregate": 12,
    "stage193_paired_vs_baseline": 84,
    "stage193_stability": 12,
    "stage194_detail": 240,
    "stage194_aggregate": 60,
    "stage194_paired_vs_baseline": 84,
    "stage194_placement_redundancy": 24,
    "stage195_detail": 240,
    "stage195_placement_detail": 2640,
    "stage195_placement_summary": 48,
    "stage195_aggregate": 12,
    "stage195_paired_vs_baseline": 84,
    "stage196_detail": 240,
    "stage196_placement_detail": 2640,
    "stage196_placement_summary": 48,
    "stage196_aggregate": 12,
    "stage196_paired_vs_baseline": 84,
    "stage197_original_detail": 240,
    "stage197_original_station_detail": 2640,
    "stage197_original_station_summary": 132,
    "stage197_original_loso": 2640,
    "stage197_independent_detail": 240,
    "stage197_independent_station_detail": 2640,
    "stage197_independent_station_summary": 132,
    "stage197_independent_loso": 2640,
    "stage198_original_detail": 240,
    "stage198_original_selections": 48,
    "stage198_original_selection_counts": 26,
    "stage198_original_aggregate": 12,
    "stage198_original_paired_vs_baseline": 84,
    "stage198_independent_detail": 240,
    "stage198_independent_selections": 48,
    "stage198_independent_selection_counts": 21,
    "stage198_independent_aggregate": 12,
    "stage198_independent_paired_vs_baseline": 84,
    "stage199_detail": 420,
    "stage199_selections": 420,
    "stage199_station_summary": 84,
    "stage199_aggregate": 12,
    "stage199_paired_vs_baseline": 84,
    "stage200_detail": 140,
    "stage200_aggregate": 4,
    "stage200_paired": 20,
    "stage201_detail": 420,
    "stage201_aggregate": 12,
    "stage201_paired": 72,
    "stage202_detail": 700,
    "stage202_station": 140,
    "stage202_tests": 24,
    "stage203_inventory": 4,
    "stage203_requirements": 5,
    "stage203_candidates": 4,
    "stage204_stations": 12,
    "stage205_detail": 24,
    "stage206_station_audit": 12,
    "stage206_archive_audit": 12,
    "stage207_evaluation": 20,
    "stage207_events": 20,
    "stage207_type_detail": 30,
    "stage207_type_aggregate": 6,
    "stage214_stations": 12,
    "stage215_detail": 12,
    "stage216_normalized": 525888,
    "stage216_station_audit": 12,
    "stage216_archive_audit": 12,
    "stage217_evaluation": 20,
    "stage217_events": 20,
    "stage217_type_detail": 30,
    "stage217_type_aggregate": 6,
    "stage217_station_detail": 55,
    "stage217_summary": 8,
    "stage218_attack_type": 6,
    "stage218_station": 11,
    "stage218_sign_flip": 6,
    "stage218_provider_contrast": 7,
    "stage219_evaluation": 15,
    "stage219_events": 15,
    "stage219_type_detail": 90,
    "stage219_type_aggregate": 18,
    "stage219_station_detail": 165,
    "stage219_station_event": 165,
    "stage219_selection": 15,
    "stage219_paired": 21,
    "stage220_metrics": 4,
    "stage220_station_bootstrap": 3,
    "stage220_paired_station": 11,
    "stage221_curve": 20,
    "stage221_arm_aggregate": 12,
    "stage221_evaluation": 60,
    "stage221_station_detail": 660,
    "stage221_station_event": 660,
    "stage222_evaluation": 15,
    "stage222_events": 15,
    "stage222_type_detail": 90,
    "stage222_type_aggregate": 18,
    "stage222_station_detail": 165,
    "stage222_station_event": 165,
    "stage222_selection": 15,
    "stage223_metrics": 4,
    "stage223_paired_station": 11,
    "stage224_contract": 1,
    "stage225_detail": 12,
    "stage225_stations": 12,
    "stage226_normalized": 525888,
    "stage226_station_audit": 12,
    "stage226_archive_audit": 12,
    "stage226a_year": 60,
    "stage226a_station": 12,
    "stage226a_split": 36,
    "stage227_evaluation": 20,
    "stage227_events": 20,
    "stage227_station_detail": 30,
    "stage227_summary": 8,
    "stage227_type_detail": 30,
    "stage227_type_aggregate": 6,
    "stage228_evaluation": 15,
    "stage228_events": 15,
    "stage228_paired": 14,
    "stage228_selection": 15,
    "stage228_station_detail": 90,
    "stage228_station_event": 90,
    "stage228_type_detail": 90,
    "stage228_type_aggregate": 18,
    "stage229_metrics": 4,
    "stage229_paired_station": 6,
    "stage230_sources": 5,
    "stage230_transfers": 4,
    "stage230_checks": 13,
    "stage231_sources": 4,
    "stage231_leave_one_provider_out": 4,
    "stage231_hierarchical_bootstrap": 1,
    "stage231_station_effects": 42,
    "stage231_checks": 10,
    "stage232_sources": 24,
    "stage232_provider_equal_by_type": 6,
    "stage232_leave_one_provider_out_by_type": 24,
    "stage232_leave_one_attack_type_out": 6,
    "stage232_hierarchical_bootstrap_by_type": 6,
    "stage232_checks": 11,
    "stage233_targets": 4,
    "stage233_checks": 5,
    "stage240_evaluation": 60,
    "stage240_station_detail": 630,
    "stage240_type_detail": 360,
    "stage240_selection": 40,
    "stage240_evaluation_aggregate": 12,
    "stage240_provider_equal": 12,
    "stage240_provider_station": 4,
    "stage240_leave_one_provider_out": 4,
    "stage240_nonstuck_type_detail": 20,
    "stage240_nonstuck_type_equal": 5,
    "stage240_selection_counts": 22,
    "stage241_station_detail": 42,
    "stage241_type_summary": 6,
    "stage241_leave_one_target_out": 4,
    "stage242_pairwise": 20,
    "stage242_target_summary": 5,
    "stage242_source_summary": 5,
    "stage242_transfer_manifest": 100,
    "stage242_evaluation": 125,
    "stage242_station_detail": 1350,
    "stage242_local_reproduction": 5,
    "stage243_station_detail": 26,
    "stage243_seed_detail": 10,
    "stage243_metric_detail": 12,
    "stage243_summary": 2,
    "stage244_evaluation": 60,
    "stage244_station_detail": 630,
    "stage244_type_detail": 360,
    "stage244_selection": 40,
    "stage244_evaluation_aggregate": 12,
    "stage244_provider_station": 4,
    "stage244_leave_one_provider_out": 4,
    "stage244_nonstuck_type_detail": 20,
    "stage244_nonstuck_type_equal": 5,
    "stage244_selection_counts": 23,
    "stage245_evaluation": 60,
    "stage245_station_detail": 630,
    "stage245_type_detail": 360,
    "stage245_selection": 40,
    "stage245_evaluation_aggregate": 12,
    "stage245_provider_station": 4,
    "stage245_leave_one_provider_out": 4,
    "stage245_nonstuck_type_detail": 20,
    "stage245_nonstuck_type_equal": 5,
    "stage246_table1": 5,
    "stage246_table2": 4,
    "stage246_table3": 6,
    "stage246_table4": 10,
    "stage246_table5": 3,
    "stage246_table6": 10,
    "stage246_table7": 6,
    "stage246_table8": 6,
    "stage249_requirements": 10,
    "stage250_risks": 7,
    "stage251_risk_ceilings": 7,
    "stage251_workstreams": 7,
    "stage252_runs": 35,
    "stage253_rows": 7,
    "stage116_yearly": 45,
    "stage116_aggregate": 9,
}
ALLOWED_ALL_MISSING_COLUMNS = {
    "stage230_sources": {
        "replication_f1_mean",
        "f1_replication_ci_low",
        "f1_replication_ci_high",
        "local_valid_selected_f1",
        "local_valid_selected_event_coverage",
        "direct_threshold_transfer_f1",
        "direct_threshold_transfer_event_coverage",
        "f1_positive_stations",
        "f1_negative_stations",
        "event_cluster_ci_low",
        "event_cluster_ci_high",
        "event_positive_stations",
        "event_negative_stations",
    },
}


def is_range_metric(key: str) -> bool:
    key = key.lower()
    if key.endswith("_sum") or key.endswith("sum"):
        return False
    tokens = ("f1", "recall", "precision", "coverage", "auprc", "average_precision", "false_alarm_rate", "alarm_rate", "prob_")
    exclusions = (
        "diff",
        "gain",
        "delta",
        "change",
        "ci",
        "std",
        "variance",
        "rank",
        "minus",
        "years",
        "count",
        "cells",
        "folds",
        "slope",
        "pairs",
        "rows",
        "stations",
        "providers",
        "items",
        "stuck",
        "non_stuck",
    )
    return any(token in key for token in tokens) and not any(token in key for token in exclusions)


def is_probability_key(key: str) -> bool:
    key = key.lower()
    return key.startswith("prob_") or "probability" in key


def walk_json(value: Any, path: str, checks: list[dict[str, object]], parent: Any = None) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            walk_json(child, f"{path}.{key}", checks, value)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            walk_json(child, f"{path}[{index}]", checks, value)
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        return
    number = float(value)
    key = path.rsplit(".", 1)[-1].split("[", 1)[0]
    optional_missing = (
        key == "quality_flagged_alarm_rate" and isinstance(parent, dict) and int(parent.get("quality_flagged_rows", 1)) == 0
    ) or (
        key == "quality_clean_alarm_rate" and isinstance(parent, dict) and int(parent.get("quality_clean_rows", 1)) == 0
    )
    if not math.isfinite(number):
        checks.append({"check": "json_numeric_finite", "status": "REVIEW" if optional_missing else "FAIL", "path": path, "value": repr(value), "reason": "no flagged rows" if optional_missing else "non-finite value"})
        if optional_missing:
            return
    if is_probability_key(key) or key.lower() == "q" or is_range_metric(key):
        checks.append({"check": "json_metric_in_unit_interval", "status": "PASS" if 0.0 <= number <= 1.0 else "FAIL", "path": path, "value": number})
    if key.lower().endswith("ci95"):
        pass


def check_ci_order(payload: Any, path: str, checks: list[dict[str, object]]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, list) and len(value) == 2 and "ci95" in key.lower():
                try:
                    low, high = float(value[0]), float(value[1])
                    checks.append({"check": "json_ci95_order", "status": "PASS" if low <= high else "FAIL", "path": f"{path}.{key}", "value": [low, high]})
                except (TypeError, ValueError):
                    checks.append({"check": "json_ci95_numeric", "status": "FAIL", "path": f"{path}.{key}", "value": repr(value)})
            check_ci_order(value, f"{path}.{key}", checks)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            check_ci_order(value, f"{path}[{index}]", checks)


def check_frame(frame: pd.DataFrame, name: str, checks: list[dict[str, object]]) -> None:
    expected = EXPECTED_ROWS.get(name)
    # An explicitly zero-row not-estimable table is a valid outcome when all
    # metrics are estimable; do not fabricate a placeholder row just to satisfy
    # a generic non-empty check.
    nonempty_ok = len(frame) > 0 or expected == 0
    checks.append({"check": f"{name}_nonempty", "status": "PASS" if nonempty_ok else "FAIL", "rows": int(len(frame))})
    if expected is not None:
        checks.append({"check": f"{name}_expected_rows", "status": "PASS" if len(frame) == expected else "FAIL", "rows": int(len(frame)), "expected": expected})
    for column in frame.columns:
        series = frame[column]
        if name == "stage205_detail" and column in {
            "downloaded_at_utc",
            "error",
            "audit_error",
        }:
            # These are provenance and failure-message fields, not numeric
            # measurements.  Missing values are valid for offline replay.
            continue
        if name == "stage215_detail" and column in {
            "error",
            "audit_error",
        }:
            # Empty failure-message fields are valid when every download and
            # CSV-structure audit succeeds.
            continue
        if name == "stage225_detail" and column in {
            "downloaded_at_utc",
            "error",
            "audit_error",
        }:
            # KNMI archives may already be present locally, so the download
            # timestamp and failure-message fields are legitimately empty.
            continue
        if not pd.api.types.is_numeric_dtype(series):
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        allowed_missing = pd.Series(False, index=frame.index)
        if column in ALLOWED_ALL_MISSING_COLUMNS.get(name, set()):
            allowed_missing = numeric.isna()
        if name == "stage216_normalized" and column in {
            "temperature_c",
            "humidity_pct",
            "dew_point_c",
            "pressure_hpa",
        }:
            # Stage216 retains official empty CSV fields as missing values.
            # The JSON audit records their counts separately; missing
            # measurements are not non-finite numeric corruption.
            allowed_missing = numeric.isna()
        elif name == "stage226_normalized" and column in {
            "temperature_c",
            "humidity_pct",
            "dew_point_c",
            "pressure_hpa",
        }:
            # KNMI empty fields are retained as missing by protocol; the
            # normalization JSON records their counts separately.
            allowed_missing = numeric.isna()
        elif column == "quality_flagged_alarm_rate" and "quality_flagged_rows" in frame.columns:
            allowed_missing = numeric.isna() & (pd.to_numeric(frame["quality_flagged_rows"], errors="coerce") == 0)
        elif column == "quality_clean_alarm_rate" and "quality_clean_rows" in frame.columns:
            allowed_missing = numeric.isna() & (pd.to_numeric(frame["quality_clean_rows"], errors="coerce") == 0)
        elif column == "alpha_trust" and "method" in frame.columns:
            allowed_missing = numeric.isna() & frame["method"].astype(str).isin({"svm_only", "trust_only", "trust_component_only", "trust_svm_max"})
        elif column == "spatial_candidate" and "temporal_candidate" in frame.columns:
            # No unmasked neighbour at a timestamp is a valid spatial fallback;
            # another candidate must remain available so recovery is defined.
            allowed_missing = numeric.isna() & pd.to_numeric(frame["temporal_candidate"], errors="coerce").notna()
        elif name.startswith("stage141_") and column in {"alpha_base", "alpha_structural", "window", "valid_false_alarm_rate"}:
            # These fields are not applicable to the reused standalone baselines;
            # retain the missingness instead of fabricating parameter values.
            allowed_missing = numeric.isna()
        elif name in {"stage158_detail", "stage161_detail", "stage163_detail", "stage165_detail", "stage166_detail"} and column == "test_auprc" and "attack_rows" in frame.columns:
            allowed_missing = numeric.isna() & (pd.to_numeric(frame["attack_rows"], errors="coerce") == 0)
        elif name == "stage163_summary" and column == "test_auprc_mean" and "attack_rows" in frame.columns:
            allowed_missing = numeric.isna() & (pd.to_numeric(frame["attack_rows"], errors="coerce") == 0)
        elif name in {"stage161_detail", "stage165_detail", "stage166_detail"} and column == "test_false_alarm_rate" and "clean_rows" in frame.columns:
            allowed_missing = numeric.isna() & (pd.to_numeric(frame["clean_rows"], errors="coerce") == 0)
        elif name == "stage165_detail" and column == "threshold" and "method" in frame.columns:
            allowed_missing = numeric.isna() & frame["method"].astype(str).isin({"TrustMeteo-QC default", "TrustMeteo-QC frozen expanded"})
        elif name == "stage166_detail" and column == "threshold":
            allowed_missing = numeric.isna()
        elif name == "stage175_detail" and "entity" in frame.columns:
            entity = frame["entity"].astype(str)
            zero_alarms = pd.to_numeric(frame["alarm_count"], errors="coerce").eq(0)
            if column.startswith("station_"):
                allowed_missing = numeric.isna() & (entity.eq("month") | zero_alarms)
            elif column.startswith("month_"):
                allowed_missing = numeric.isna() & (entity.eq("station") | zero_alarms)
        elif name == "stage176_detail":
            zero_alarms = pd.to_numeric(frame["alarm_count"], errors="coerce").eq(0)
            if column in {"pooled_precision", "macro_precision", "pooled_minus_macro_macro_precision"}:
                allowed_missing = numeric.isna() & zero_alarms
        elif name == "stage176_station_detail":
            zero_alarms = pd.to_numeric(frame["alarm_count"], errors="coerce").eq(0)
            if column == "precision":
                allowed_missing = numeric.isna() & zero_alarms
        elif name == "stage177_detail":
            zero_global = pd.to_numeric(frame["global_alarm_count"], errors="coerce").eq(0)
            zero_station = pd.to_numeric(frame["station_alarm_count"], errors="coerce").eq(0)
            if column in {"global_precision", "global_macro_precision"}:
                allowed_missing = numeric.isna() & zero_global
            elif column in {"station_precision", "station_macro_precision"}:
                allowed_missing = numeric.isna() & zero_station
        elif name == "stage177_station_detail":
            zero_alarms = pd.to_numeric(frame["global_alarm_count"], errors="coerce").eq(0) & pd.to_numeric(frame["station_alarm_count"], errors="coerce").eq(0)
            if column in {"global_precision", "station_precision"}:
                allowed_missing = numeric.isna() & zero_alarms
        elif name == "stage178_detail":
            if column in {
                "global_budget_precision",
                "global_budget_macro_precision",
                "station_budget_precision",
                "station_budget_macro_precision",
            }:
                allowed_missing = numeric.isna()
        elif name == "stage178_station_detail":
            if column in {"precision"}:
                allowed_missing = numeric.isna()
        elif name == "stage180_station_type_detail":
            if column in {
                "mean_risk_percentile_selected",
                "mean_risk_percentile_unselected",
            }:
                allowed_missing = numeric.isna()
        elif name.startswith("stage189_"):
            # Event-level suppression ratios are undefined when the
            # denominator is zero (for example, no uncovered events or no
            # station-budget suppression at q=0.999).  Keep these values
            # missing instead of fabricating zeros.
            denominator_column = {
                "ranking_failure_share_of_uncovered": "uncovered_events",
                "station_budget_suppression_share_of_uncovered": "uncovered_events",
                "zero_station_budget_share_of_suppression": "station_budget_suppression_events",
                "local_displacement_share_of_suppression": "station_budget_suppression_events",
            }.get(str(column))
            if denominator_column is not None and denominator_column in frame.columns:
                allowed_missing = numeric.isna() & pd.to_numeric(
                    frame[denominator_column], errors="coerce"
                ).eq(0)
            elif name == "stage189_detail" and column in {
                "station_budget_threshold_risk",
                "station_budget_threshold_percentile",
            }:
                allowed_missing = numeric.isna() & pd.to_numeric(
                    frame["station_total_selected_rows"], errors="coerce"
                ).eq(0)
        elif name == "stage193_stability" and column == "mean_gate_quantile":
            # The mean gate quantile is undefined when no fold/seed in the
            # stratum adopted persistent-repeat protection at all.  Keep it
            # missing instead of fabricating zero.
            allowed_missing = numeric.isna() & pd.to_numeric(
                frame["protected_pairs"], errors="coerce"
            ).eq(0)
        elif name == "stage246_table1" and "evidence" in frame.columns:
            # Stage246 table 1 mixes one benchmark row with transfer and
            # difference rows.  Absolute row recall, FAR and baseline values
            # are not applicable to difference-only rows; keep that
            # not-applicable missingness instead of fabricating zeros.
            if column in {
                "row_recall",
                "false_alarm_rate",
                "baseline_f1",
                "baseline_event_coverage",
            }:
                allowed_missing = numeric.isna() & ~frame["evidence"].astype(str).eq(
                    "DWD controlled benchmark"
                )
        invalid = (~np.isfinite(numeric.to_numpy(dtype=float))) & (~allowed_missing.to_numpy())
        checks.append({"check": f"{name}_{column}_finite", "status": "PASS" if not bool(invalid.any()) else "FAIL", "allowed_missing": int(allowed_missing.sum())})
        if is_probability_key(str(column)) or str(column).lower() == "q" or is_range_metric(str(column)):
            valid = numeric.loc[~allowed_missing].dropna()
            checks.append({"check": f"{name}_{column}_unit_interval", "status": "PASS" if bool(valid.between(0, 1).all()) else "FAIL", "min": float(valid.min()) if len(valid) else None, "max": float(valid.max()) if len(valid) else None})


def main() -> int:
    checks: list[dict[str, object]] = []
    for name, path in {**JSON_FILES, **TABULAR_FILES}.items():
        checks.append({"check": f"{name}_exists", "status": "PASS" if path.is_file() and path.stat().st_size > 0 else "FAIL", "path": str(path)})
    if any(item["status"] == "FAIL" for item in checks):
        payload = {"status": "FAIL", "checks": checks}
        (TABLES / "stage114_recent_semantic_sanity_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(json.dumps(payload, ensure_ascii=False))

    for name, path in JSON_FILES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        allowed_statuses = {"PASS_PIPELINE_REVIEW_REQUIRED"}
        if name in {"stage121", "stage122"}:
            allowed_statuses |= {"CANDIDATE_GAIN_REQUIRES_CROSS_STATION_REVIEW", "NEGATIVE_RESULT_RETAINED"}
        if name == "stage199":
            allowed_statuses |= {"NEGATIVE_RESULT_RETAINED"}
        if name == "stage200":
            allowed_statuses |= {"DIAGNOSTIC_ONLY_ORACLE_THRESHOLD"}
        if name == "stage201":
            allowed_statuses |= {"NEGATIVE_RESULT_RETAINED"}
        if name == "stage202":
            allowed_statuses |= {"NEGATIVE_RESULT_RETAINED"}
        if name == "stage203":
            allowed_statuses |= {"EVIDENCE_GAP_RETAINED"}
        if name == "stage204":
            allowed_statuses |= {"PROTOCOL_FROZEN_BEFORE_DOWNLOAD"}
        if name == "stage204a":
            allowed_statuses |= {
                "PROTOCOL_SCHEMA_CORRECTED_BEFORE_PERFORMANCE"
            }
        if name == "stage205":
            allowed_statuses |= {"PASS_PIPELINE_REVIEW_REQUIRED"}
        if name == "stage206":
            allowed_statuses |= {"PASS_PIPELINE_REVIEW_REQUIRED"}
        if name == "stage207":
            allowed_statuses |= {"PASS_PIPELINE_REVIEW_REQUIRED"}
        if name == "stage208":
            allowed_statuses |= {"PASS_PIPELINE_REVIEW_REQUIRED"}
        if name == "stage209":
            allowed_statuses |= {"PASS_PIPELINE_REVIEW_REQUIRED"}
        if name == "stage210":
            allowed_statuses |= {"PASS_PIPELINE_REVIEW_REQUIRED"}
        if name == "stage211":
            allowed_statuses |= {"PASS_PIPELINE_REVIEW_REQUIRED"}
        if name == "stage212":
            allowed_statuses |= {"PASS_PIPELINE_REVIEW_REQUIRED"}
        if name == "stage213":
            allowed_statuses |= {"EXTERNAL_EVIDENCE_CLOSURE_SINGLE_PROVIDER"}
        if name == "stage214":
            allowed_statuses |= {"PROTOCOL_FROZEN_BEFORE_DOWNLOAD"}
        if name == "stage216a":
            allowed_statuses |= {
                "PROTOCOL_AVAILABILITY_CORRECTED_BEFORE_PERFORMANCE"
            }
        if name == "stage224":
            allowed_statuses |= {"PROTOCOL_FROZEN_BEFORE_KNMI_PERFORMANCE"}
        if name == "stage234":
            allowed_statuses |= {"PROTOCOL_FROZEN_BEFORE_ECCC_PERFORMANCE"}
        if name == "stage235":
            allowed_statuses |= {"REVIEW_REQUIRED"}
        if name == "stage226a":
            allowed_statuses |= {"MISSINGNESS_LOCALIZED"}
        if name == "stage129":
            allowed_statuses |= {"CANDIDATE_GAIN_REQUIRES_CROSS_SITE_REVIEW", "NEGATIVE_RESULT_RETAINED"}
        if name == "stage130":
            allowed_statuses |= {"CANDIDATE_GAIN_REQUIRES_ADDITIONAL_EXTERNAL_DATA", "NEGATIVE_RESULT_RETAINED"}
        if name == "stage122":
            allowed_statuses.add("CANDIDATE_TRANSFER_REQUIRES_REVIEW")
        if name == "stage232":
            allowed_statuses.add("ATTACK_TYPE_ROBUSTNESS_REVIEW_REQUIRED")
        if name == "stage247":
            allowed_statuses.add("BOUNDARY_EVIDENCE_RETAINED")
        if name == "stage249":
            allowed_statuses.add("PASS_JISA_SUBMISSION_GATE")
        if name == "stage250":
            allowed_statuses.add("PASS_RISK_CLOSURE_AUDIT")
        if name == "stage251":
            allowed_statuses.add("PASS_STABLE_SUBMISSION_OBJECTIVE")
        if name == "stage252":
            allowed_statuses.add("PASS_AUTHORIZED_TESTBED_ANALYSIS")
        if name == "stage253":
            allowed_statuses.add("PASS_RESIDUAL_RISK_OPTIMIZATION_PLAN")
        checks.append({"check": f"{name}_status", "status": "PASS" if payload.get("status") in allowed_statuses else "FAIL", "value": payload.get("status")})
        walk_json(payload, name, checks)
        check_ci_order(payload, name, checks)
    for name, path in TABULAR_FILES.items():
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        check_frame(frame, name, checks)

    expected_json_rows = {"stage99": 150, "stage103": 20, "stage106": 12, "stage107": 45, "stage108": 30, "stage109": 45, "stage110": 6, "stage111": 45, "stage116": 54, "stage117": 7, "stage118": 45, "stage119": 45, "stage120": 47, "stage121": 190, "stage169": 400, "stage170": 300, "stage171": 300, "stage172": 600, "stage173": 120, "stage174": 600, "stage175": 1344, "stage176": 7392, "stage177": 7320, "stage178": 21960, "stage179": 2160, "stage180": 4488306, "stage181": 4471938, "stage182": 2235933, "stage183": 1932, "stage184": 4705440, "stage185": 2078, "stage186": 3360, "stage187": 12553216, "stage188": 2063, "stage189": 8094, "stage190": 2075, "stage191": 813, "stage192": 2067, "stage193": 2078, "stage194": 408, "stage195": 3024, "stage196": 3024, "stage197_original": 5652, "stage197_independent": 5652, "stage198_original": 410, "stage198_independent": 405, "stage199": 1020, "stage200": 164, "stage201": 504, "stage202": 864, "stage203": 13, "stage204a": 16, "stage205": 24, "stage206": 12}
    expected_json_rows.update({"stage173": 120})
    expected_json_rows.update({"stage122": 257, "stage123": 12, "stage124": 47, "stage125": 16, "stage126": 172, "stage127": 12, "stage128": 172, "stage129": 1337, "stage130": 136, "stage131": 321, "stage132": 92, "stage133": 142, "stage134": 919, "stage135": 113, "stage136": 7, "stage137": 221, "stage138": 39, "stage139": 152, "stage140": 152, "stage141": 107, "stage142": 1056, "stage143": 162, "stage144": 31, "stage145": 19, "stage146": 14, "stage147": 30, "stage148": 524, "stage149": 60, "stage150": 24, "stage151": 600, "stage152": 1176, "stage153": 987, "stage154": 23, "stage155": 6396, "stage156": 3333, "stage157": 144, "stage158": 303, "stage159": 34, "stage160": 35, "stage161": 3054, "stage162": 399})
    expected_json_rows.update({"stage208": 972})
    expected_json_rows.update({"stage209": 6})
    expected_json_rows.update({"stage210": 2646})
    expected_json_rows.update({"stage211": 6})
    expected_json_rows.update({"stage212": 48})
    expected_json_rows.update({"stage213": 5})
    expected_json_rows.update({"stage214": 12, "stage215": 12, "stage216": 12, "stage216a": 1, "stage217": 144, "stage218": 35, "stage219": 47, "stage220": 12, "stage221": 60, "stage222": 401, "stage223": 6, "stage224": 4, "stage225": 12, "stage226": 12, "stage226a": 6, "stage227": 119, "stage228": 40, "stage229": 6, "stage230": 23, "stage231": 64, "stage232": 57, "stage233": 29, "stage234": 23, "stage235": 80, "stage236": 16, "stage237": 159, "stage238": 40, "stage239": 6, "stage240": 1137, "stage241": 14, "stage242": 35, "stage243": 24, "stage244": 1126, "stage245": 1103, "stage246": 69, "stage247": 25, "stage248": 1929, "stage249": 26, "stage250": 26, "stage251": 37, "stage252": 151, "stage253": 68})
    for name, expected in expected_json_rows.items():
        payload = json.loads(JSON_FILES[name].read_text(encoding="utf-8"))
        if name == "stage105":
            actual = len(payload.get("type_detail", payload.get("detail", []))) + len(payload.get("overall_detail", []))
        elif name == "stage116":
            actual = len(payload.get("yearly_results", [])) + len(payload.get("aggregate_results", []))
        elif name == "stage117":
            actual = len(payload.get("results", []))
        elif name == "stage118":
            actual = len(payload.get("detail", [])) + len(payload.get("duration_aggregate", [])) + len(payload.get("attack_type_aggregate", []))
        elif name == "stage119":
            actual = len(payload.get("detail", [])) + len(payload.get("shift_aggregate", [])) + len(payload.get("attack_type_aggregate", []))
        elif name == "stage120":
            actual = len(payload.get("station_detail", [])) + len(payload.get("station_aggregate", [])) + len(payload.get("seed_summary", []))
        elif name == "stage121":
            actual = len(payload.get("selected_by_valid", [])) + len(payload.get("station_detail", [])) + int(payload.get("candidate_count", 0))
        elif name == "stage122":
            actual = int(payload.get("candidate_count", 0)) + int(payload.get("selected_count", 0)) + len(payload.get("station_summary", [])) + len(payload.get("seed_summary", []))
        elif name == "stage123":
            actual = len(payload.get("results", []))
        elif name == "stage124":
            actual = len(payload.get("results", [])) + len(payload.get("station_summary", [])) + len(payload.get("seed_summary", []))
        elif name == "stage125":
            actual = len(payload.get("results", []))
        elif name == "stage126":
            actual = len(payload.get("selected", [])) + len(payload.get("cap_summary", [])) + len(payload.get("station_summary", []))
        elif name == "stage127":
            actual = len(payload.get("results", []))
        elif name == "stage128":
            actual = int(payload.get("result_count", 0)) + int(payload.get("station_summary_count", 0)) + int(payload.get("candidate_summary_count", 0))
        elif name == "stage129":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("selected_count", 0)) + int(payload.get("aggregate_count", 0)) + len(payload.get("selected_strategy_counts", []))
        elif name == "stage130":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("selected_count", 0)) + int(payload.get("aggregate_count", 0)) + len(payload.get("selected_strategy_counts", []))
        elif name == "stage131":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("aggregate_rows", 0)) + int(payload.get("count_rows", 0))
        elif name == "stage132":
            actual = int(payload.get("pair_rows", 0)) + int(payload.get("result_rows", 0))
        elif name == "stage133":
            actual = int(payload.get("result_count", 0)) + int(payload.get("station_detail_count", 0)) + int(payload.get("candidate_summary_count", 0))
        elif name == "stage134":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("selected_count", 0)) + int(payload.get("aggregate_count", 0)) + int(payload.get("count_rows", 0))
        elif name == "stage135":
            actual = int(payload.get("result_count", 0)) + int(payload.get("attack_detail_rows", 0)) + int(payload.get("attack_aggregate_rows", 0))
        elif name == "stage136":
            actual = int(payload.get("result_count", 0))
        elif name == "stage137":
            actual = int(payload.get("result_count", 0)) + int(payload.get("attack_detail_rows", 0)) + int(payload.get("attack_aggregate_rows", 0))
        elif name == "stage138":
            actual = int(payload.get("result_count", 0)) + int(payload.get("aggregate_count", 0)) + int(payload.get("paired_count", 0))
        elif name == "stage139":
            actual = int(payload.get("candidate_count", 0)) + int(payload.get("selected_count", 0)) + int(payload.get("aggregate_count", 0)) + int(payload.get("paired_count", 0))
        elif name == "stage140":
            actual = int(payload.get("candidate_count", 0)) + int(payload.get("selected_count", 0)) + int(payload.get("aggregate_count", 0)) + int(payload.get("paired_count", 0))
        elif name == "stage141":
            actual = len(payload.get("aggregate", [])) + len(payload.get("fold_aggregate", [])) + len(payload.get("paired", [])) + len(payload.get("low_far_aggregate", [])) + len(payload.get("attack_type_aggregate", [])) + len(payload.get("station_aggregate", [])) + len(payload.get("integrity", []))
        elif name == "stage142":
            actual = len(payload.get("detail", [])) + len(payload.get("station_detail", [])) + len(payload.get("aggregate", [])) + len(payload.get("station_summary", [])) + len(payload.get("weak_label_paired", []))
        elif name == "stage143":
            actual = len(payload.get("detail", [])) + len(payload.get("selected", [])) + len(payload.get("aggregate_all", [])) + len(payload.get("aggregate_selected", [])) + len(payload.get("selection_counts", [])) + len(payload.get("paired", []))
        elif name == "stage144":
            actual = len(payload.get("transfer", [])) + len(payload.get("transfer_detail", [])) + len(payload.get("selection_counts", [])) + len(payload.get("paired", []))
        elif name == "stage145":
            actual = len(payload.get("station_summary", [])) + len(payload.get("paired", [])) + len(payload.get("extremes", []))
        elif name == "stage146":
            actual = len(payload.get("year_summary", [])) + len(payload.get("paired", []))
        elif name == "stage147":
            actual = len(payload.get("detail", [])) + len(payload.get("paired", []))
        elif name == "stage148":
            actual = len(payload.get("detail", [])) + len(payload.get("aggregate", [])) + len(payload.get("integrity", []))
        elif name == "stage149":
            actual = len(payload.get("detail", [])) + len(payload.get("paired", []))
        elif name == "stage151":
            actual = len(payload.get("detail", [])) + len(payload.get("aggregate", [])) + len(payload.get("type_detail", [])) + len(payload.get("type_aggregate", [])) + len(payload.get("paired", []))
        elif name == "stage152":
            actual = len(payload.get("detail", [])) + len(payload.get("station_aggregate", [])) + len(payload.get("method_summary", [])) + len(payload.get("paired", []))
        elif name == "stage153":
            actual = int(payload.get("detail_rows", 0)) + len(payload.get("summary", [])) + len(payload.get("contrast_summary", [])) + len(payload.get("paired", []))
        elif name == "stage154":
            actual = len(payload.get("summary", [])) + len(payload.get("paired", []))
        elif name == "stage155":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("station_detail_rows", 0)) + len(payload.get("paired", []))
        elif name == "stage156":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("station_detail_rows", 0)) + len(payload.get("paired", []))
        elif name == "stage157":
            actual = int(payload.get("row_count", 0))
        elif name == "stage158":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("paired_rows", 0))
        elif name == "stage159":
            actual = int(payload.get("hypothesis_count", 0)) + int(payload.get("not_estimable_count", 0))
        elif name == "stage160":
            actual = int(payload.get("hypothesis_count", 0)) + int(payload.get("not_estimable_count", 0))
        elif name == "stage161":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("station_summary_rows", 0))
        elif name == "stage162":
            actual = int(payload.get("station_detail_rows", 0)) + int(payload.get("summary_rows", 0)) + int(payload.get("not_estimable_count", 0))
        elif name == "stage163":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("summary_rows", 0)) + int(payload.get("paired_rows", 0)) + int(payload.get("not_estimable_count", 0))
        elif name == "stage164":
            actual = int(payload.get("hypothesis_count", 0)) + int(payload.get("not_estimable_count", 0))
        elif name == "stage165":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("station_summary_rows", 0)) + int(payload.get("paired_rows", 0)) + int(payload.get("not_estimable_count", 0))
        elif name == "stage166":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("summary_rows", 0)) + int(payload.get("paired_rows", 0)) + int(payload.get("not_estimable_count", 0))
        elif name == "stage167":
            actual = int(payload.get("hypothesis_count", 0)) + int(payload.get("not_estimable_count", 0))
        elif name == "stage168":
            actual = int(payload.get("hypothesis_count", 0)) + int(payload.get("not_estimable_count", 0))
        elif name == "stage169":
            actual = int(payload.get("detail_rows", 0))
        elif name == "stage175":
            actual = int(payload.get("detail_rows", 0)) + int(payload.get("comparison_rows", 0))
        elif name == "stage176":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("station_detail_rows", 0))
                + int(payload.get("comparison_rows", 0))
            )
        elif name == "stage177":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("station_detail_rows", 0))
                + int(payload.get("baseline_comparison_rows", 0))
                + int(payload.get("calibration_comparison_rows", 0))
            )
        elif name == "stage178":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("station_detail_rows", 0))
                + int(payload.get("baseline_comparison_rows", 0))
                + int(payload.get("calibration_comparison_rows", 0))
            )
        elif name == "stage179":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("comparison_rows", 0))
            )
        elif name == "stage180":
            actual = (
                int(payload.get("row_detail_rows", 0))
                + int(payload.get("station_detail_rows", 0))
                + int(payload.get("station_type_detail_rows", 0))
                + int(payload.get("event_detail_rows", 0))
                + int(payload.get("event_aggregate_rows", 0))
            )
        elif name == "stage181":
            actual = (
                int(payload.get("row_detail_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("type_aggregate_rows", 0))
                + int(payload.get("component_detail_rows", 0))
            )
        elif name == "stage182":
            actual = (
                int(payload.get("row_detail_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("type_aggregate_rows", 0))
            )
        elif name == "stage183":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("type_aggregate_rows", 0))
            )
        elif name == "stage184":
            actual = (
                int(payload.get("score_detail_rows", 0))
                + int(payload.get("detail_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("type_aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
                + int(payload.get("paired_aggregate_rows", 0))
            )
        elif name == "stage185":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("weight_selection_rows", 0))
                + int(payload.get("weight_aggregate_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
            )
        elif name == "stage186":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("correlation_rows", 0))
                + int(payload.get("correlation_aggregate_rows", 0))
            )
        elif name == "stage187":
            actual = (
                int(payload.get("score_detail_rows", 0))
                + int(payload.get("detail_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("type_aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
                + int(payload.get("paired_aggregate_rows", 0))
            )
        elif name == "stage188":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("weight_selection_rows", 0))
                + int(payload.get("weight_aggregate_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
                + int(payload.get("stability_rows", 0))
            )
        elif name == "stage189":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("overall_aggregate_rows", 0))
                + int(payload.get("type_aggregate_rows", 0))
                + int(payload.get("length_aggregate_rows", 0))
                + int(payload.get("risk_aggregate_rows", 0))
                + int(payload.get("station_aggregate_rows", 0))
            )
        elif name == "stage190":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("selection_rows", 0))
                + int(payload.get("selection_aggregate_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
                + int(payload.get("stability_rows", 0))
            )
        elif name == "stage191":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("type_aggregate_rows", 0))
                + int(payload.get("settings_rows", 0))
                + int(payload.get("consensus_rows", 0))
                + int(payload.get("fold_stability_rows", 0))
                + int(payload.get("overall_aggregate_rows", 0))
            )
        elif name == "stage192":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("selection_rows", 0))
                + int(payload.get("selection_aggregate_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
                + int(payload.get("stability_rows", 0))
            )
        elif name == "stage193":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("selection_rows", 0))
                + int(payload.get("selection_aggregate_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
                + int(payload.get("stability_rows", 0))
            )
        elif name == "stage194":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
                + int(payload.get("placement_redundancy_rows", 0))
            )
        elif name == "stage195":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("placement_detail_rows", 0))
                + int(payload.get("placement_summary_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
            )
        elif name == "stage196":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("placement_detail_rows", 0))
                + int(payload.get("placement_summary_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
            )
        elif name.startswith("stage197_"):
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("station_detail_rows", 0))
                + int(payload.get("station_summary_rows", 0))
                + int(payload.get("loso_rows", 0))
            )
        elif name.startswith("stage198_"):
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("selection_rows", 0))
                + int(payload.get("selection_count_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
            )
        elif name == "stage199":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("selection_rows", 0))
                + int(payload.get("station_summary_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
            )
        elif name == "stage200":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
            )
        elif name == "stage201":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
            )
        elif name == "stage202":
            actual = (
                int(payload.get("detail_rows", 0))
                + int(payload.get("station_rows", 0))
                + int(payload.get("test_rows", 0))
            )
        elif name == "stage203":
            actual = (
                int(payload.get("inventory_rows", 0))
                + int(payload.get("requirement_rows", 0))
                + int(payload.get("candidate_rows", 0))
            )
        elif name == "stage204a":
            actual = len(payload.get("frozen_items_unchanged", []))
        elif name == "stage206":
            actual = len(payload.get("archive_audit", []))
        elif name == "stage207":
            actual = (
                int(payload.get("evaluation_rows", 0))
                + int(payload.get("event_rows", 0))
                + int(payload.get("type_rows", 0))
                + len(payload.get("type_aggregate", []))
            )
        elif name == "stage208":
            actual = (
                int(payload.get("evaluation_rows", 0))
                + int(payload.get("event_rows", 0))
                + int(payload.get("selection_rows", 0))
                + int(payload.get("selection_count_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("type_aggregate_rows", 0))
                + int(payload.get("paired_rows", 0))
            )
        elif name == "stage209":
            actual = len(payload.get("bootstrap", []))
        elif name == "stage210":
            actual = (
                int(payload.get("evaluation_rows", 0))
                + int(payload.get("event_rows", 0))
                + int(payload.get("selection_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("station_detail_rows", 0))
                + len(payload.get("paired", []))
            )
        elif name == "stage211":
            actual = len(payload.get("bootstrap", []))
        elif name == "stage212":
            actual = int(payload.get("curve_rows", 0))
        elif name == "stage213":
            actual = int(payload.get("requirements_total", 0))
        elif name == "stage214":
            actual = len(payload.get("station_ids", []))
        elif name == "stage215":
            actual = len(payload.get("detail", []))
        elif name == "stage216":
            actual = len(payload.get("archive_audit", []))
        elif name == "stage216a":
            actual = len(payload.get("excluded_stations", []))
        elif name == "stage217":
            actual = (
                int(payload.get("evaluation_rows", 0))
                + int(payload.get("event_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("station_detail_rows", 0))
                + len(payload.get("primary_summary", []))
                + len(payload.get("type_aggregate", []))
                + len(payload.get("attack_manifest", []))
            )
        elif name == "stage218":
            actual = (
                len(payload.get("primary_station_event_coverage", []))
                + len(payload.get("seed_summary", []))
                + len(payload.get("attack_type_station_cluster", []))
                + len(payload.get("attack_type_sign_flip", []))
                + len(payload.get("provider_descriptive_contrast", []))
            )
        elif name == "stage219":
            actual = (
                len(payload.get("aggregate", []))
                + len(payload.get("type_aggregate", []))
                + len(payload.get("paired", []))
                + len(payload.get("transfer_manifest", []))
            )
        elif name == "stage220":
            actual = (
                len(payload.get("metric_uncertainty", []))
                + len(payload.get("station_bootstrap", []))
                + len(payload.get("event_aggregate", []))
                + 2
            )
        elif name == "stage221":
            actual = int(payload.get("evaluation_rows", 0))
        elif name == "stage222":
            actual = (
                len(payload.get("aggregate", []))
                + len(payload.get("type_aggregate", []))
                + len(payload.get("transfer_manifest", []))
                + int(payload.get("evaluation_rows", 0))
                + int(payload.get("event_rows", 0))
                + int(payload.get("station_detail_rows", 0))
                + int(payload.get("station_event_detail_rows", 0))
                + int(payload.get("selection_rows", 0))
            )
        elif name == "stage223":
            actual = (
                len(payload.get("metric_uncertainty", []))
                + 2
            )
        elif name == "stage224":
            actual = len(payload.get("limitations", []))
        elif name == "stage225":
            actual = len(payload.get("detail", []))
        elif name == "stage226":
            actual = len(payload.get("archive_audit", []))
        elif name == "stage226a":
            actual = len(payload.get("excluded_station_ids", []))
        elif name == "stage227":
            actual = (
                int(payload.get("evaluation_rows", 0))
                + int(payload.get("event_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("station_detail_rows", 0))
                + len(payload.get("primary_summary", []))
                + len(payload.get("type_aggregate", []))
                + len(payload.get("attack_manifest", []))
            )
        elif name == "stage228":
            actual = (
                len(payload.get("aggregate", []))
                + len(payload.get("type_aggregate", []))
                + len(payload.get("paired", []))
                + len(payload.get("transfer_manifest", []))
            )
        elif name == "stage229":
            actual = len(payload.get("metric_uncertainty", [])) + 2
        elif name == "stage230":
            actual = (
                len(payload.get("source_rows", []))
                + len(payload.get("transfer_rows", []))
                + len(payload.get("checks", []))
                + 1
            )
        elif name == "stage231":
            actual = (
                len(payload.get("source_rows", []))
                + len(payload.get("leave_one_provider_out", []))
                + len(payload.get("station_effects", []))
                + len(payload.get("checks", []))
                + 4
            )
        elif name == "stage232":
            actual = (
                len(payload.get("provider_equal_rows", []))
                + len(payload.get("leave_one_provider_out_rows", []))
                + len(payload.get("leave_one_attack_type_out_rows", []))
                + len(payload.get("hierarchical_bootstrap_rows", []))
                + len(payload.get("checks", []))
                + 4
            )
        elif name == "stage233":
            actual = (
                len(payload.get("target_results", []))
                + len(payload.get("checks", []))
                + len(payload.get("limitations", []))
                + len(payload.get("affected_stages", []))
                + 4
            )
        elif name == "stage234":
            actual = (
                len(payload.get("station_ids", []))
                + len(payload.get("contract", []))
                + len(payload.get("limitations", []))
                + 3
            )
        elif name == "stage235":
            actual = len(payload.get("station_year_detail", []))
        elif name == "stage236":
            actual = len(payload.get("archive_audit", []))
        elif name == "stage237":
            actual = (
                int(payload.get("evaluation_rows", 0))
                + int(payload.get("event_rows", 0))
                + int(payload.get("type_detail_rows", 0))
                + int(payload.get("station_detail_rows", 0))
                + len(payload.get("primary_summary", []))
                + len(payload.get("type_aggregate", []))
                + len(payload.get("attack_manifest", []))
            )
        elif name == "stage238":
            actual = (
                len(payload.get("aggregate", []))
                + len(payload.get("type_aggregate", []))
                + len(payload.get("paired", []))
                + len(payload.get("transfer_manifest", []))
            )
        elif name == "stage239":
            actual = len(payload.get("metric_uncertainty", [])) + 2
        elif name == "stage240":
            actual = (
                int(payload.get("evaluation_rows", 0))
                + int(payload.get("station_rows", 0))
                + int(payload.get("type_rows", 0))
                + int(payload.get("selection_rows", 0))
                + len(payload.get("provider_equal_rows", []))
                + len(payload.get("provider_station_rows", []))
                + len(payload.get("leave_one_provider_out_rows", []))
                + len(payload.get("nonstuck_type_equal_rows", []))
                + len(payload.get("selection_count_rows", []))
            )
        elif name == "stage241":
            actual = (
                len(payload.get("source_rows", []))
                + len(payload.get("leave_one_target_rows", []))
                + len(payload.get("type_rows", []))
            )
        elif name == "stage242":
            actual = (
                len(payload.get("pairwise_rows", []))
                + len(payload.get("target_summary_rows", []))
                + len(payload.get("source_summary_rows", []))
                + len(payload.get("local_arm_reproduction", []))
            )
        elif name == "stage243":
            actual = (
                len(payload.get("summary_rows", []))
                + len(payload.get("seed_rows", []))
                + len(payload.get("metric_rows", []))
            )
        elif name == "stage244":
            actual = (
                int(payload.get("evaluation_rows", 0))
                + int(payload.get("station_rows", 0))
                + int(payload.get("type_rows", 0))
                + int(payload.get("selection_rows", 0))
                + len(payload.get("provider_station_rows", []))
                + len(payload.get("leave_one_provider_out_rows", []))
                + len(payload.get("nonstuck_type_equal_rows", []))
                + len(payload.get("selection_count_rows", []))
            )
        elif name == "stage245":
            actual = (
                int(payload.get("evaluation_rows", 0))
                + int(payload.get("station_rows", 0))
                + int(payload.get("type_rows", 0))
                + int(payload.get("selection_rows", 0))
                + len(payload.get("provider_station_rows", []))
                + len(payload.get("leave_one_provider_out_rows", []))
                + len(payload.get("nonstuck_type_equal_rows", []))
            )
        elif name == "stage246":
            actual = (
                int(payload.get("frozen_input_count", 0))
                + int(payload.get("generated_deliverable_count", 0))
                + len(payload.get("checks", []))
            )
        elif name == "stage247":
            actual = (
                int(payload.get("source_rows", 0))
                + len(payload.get("provider_equal_rows", []))
                + len(payload.get("leave_one_provider_out_rows", []))
                + len(payload.get("provider_rank_rows", []))
                + len(payload.get("checks", []))
            )
        elif name == "stage248":
            actual = (
                int(payload.get("provider_count", 0))
                + int(payload.get("station_count", 0))
                + int(payload.get("seed_count", 0))
                + int(payload.get("method_count", 0))
                + len(payload.get("methods", []))
                + len(payload.get("baseline_methods", []))
                + len(payload.get("fit_rows", []))
                + int(payload.get("evaluation_rows", 0))
                + int(payload.get("station_rows", 0))
                + int(payload.get("type_rows", 0))
                + int(payload.get("selection_rows", 0))
                + len(payload.get("provider_equal_rows", []))
                + len(payload.get("comparison_rows", []))
                + len(payload.get("paired_win_rows", []))
                + len(payload.get("provider_comparison_rows", []))
                + len(payload.get("leave_one_provider_rows", []))
                + len(payload.get("limitations", []))
            )
        elif name == "stage249":
            actual = (
                int(payload.get("requirements_total", 0))
                + len(payload.get("claim_boundaries", []))
                + len(payload.get("input_hashes", {}))
            )
        elif name == "stage250":
            actual = (
                int(payload.get("risk_count", 0))
                + int(payload.get("resolved_count", 0))
                + int(payload.get("bounded_count", 0))
                + len(payload.get("risks", []))
                + len(payload.get("frozen_input_hashes", {}))
            )
        elif name == "stage251":
            actual = (
                int(payload.get("risk_ceiling_count", 0))
                + int(payload.get("resolved_count", 0))
                + int(payload.get("bounded_count", 0))
                + len(payload.get("risk_ceilings", []))
                + int(payload.get("workstream_count", 0))
                + len(payload.get("workstreams", []))
                + len(payload.get("frozen_input_hashes", {}))
            )
        elif name == "stage252":
            actual = (
                int(payload.get("run_count", 0))
                + int(payload.get("clean_run_count", 0))
                + int(payload.get("attack_run_count", 0))
                + int(payload.get("attack_type_count", 0))
                + int(payload.get("seed_count", 0))
                + int(payload.get("failure_count", 0))
                + len(payload.get("summary_rows", []))
                + len(payload.get("input_hashes", {}))
            )
        elif name == "stage253":
            actual = (
                int(payload.get("row_count", 0))
                + int(payload.get("can_optimize_now_count", 0))
                + int(payload.get("limited_count", 0))
                + int(payload.get("resolved_now_count", 0))
                + int(payload.get("testbed_run_count", 0))
                + int(payload.get("testbed_attack_type_count", 0))
                + int(payload.get("testbed_seed_count", 0))
                + len(payload.get("rows", []))
                + len(payload.get("input_hashes", {}))
            )
        else:
            actual = len(payload.get("results", payload.get("detail", [])))
            if name == "stage103":
                actual = payload.get("blocks", actual)
        checks.append({"check": f"{name}_structured_row_count", "status": "PASS" if actual == expected else "FAIL", "actual": actual, "expected": expected})
    stage105 = json.loads(JSON_FILES["stage105"].read_text(encoding="utf-8"))
    checks.append({"check": "stage105_attack_type_aggregate_rows", "status": "PASS" if len(stage105.get("attack_type_aggregate", [])) == 18 else "FAIL", "actual": len(stage105.get("attack_type_aggregate", [])), "expected": 18})
    checks.append({"check": "stage105_overall_aggregate_rows", "status": "PASS" if len(stage105.get("overall_aggregate", [])) == 3 else "FAIL", "actual": len(stage105.get("overall_aggregate", [])), "expected": 3})
    stage240 = json.loads(JSON_FILES["stage240"].read_text(encoding="utf-8"))
    stage240_contract = [
        len(stage240.get("provider_equal_rows", [])) == 12,
        len(stage240.get("provider_station_rows", [])) == 4,
        len(stage240.get("leave_one_provider_out_rows", [])) == 4,
        len(stage240.get("nonstuck_type_equal_rows", [])) == 5,
        set(stage240.get("selection_rules", []))
        == {"valid_f1", "valid_nonstuck_event_macro"},
        int(stage240.get("provider_count", 0)) == 4,
        int(stage240.get("station_count", 0)) == 42,
        int(stage240.get("candidate_count", 0)) == 42,
        float(stage240["hierarchical_bootstrap"]["f1_ci95_low"])
        <= float(stage240["hierarchical_bootstrap"]["f1_ci95_high"]),
        float(stage240["hierarchical_bootstrap"]["event_ci95_low"])
        <= float(stage240["hierarchical_bootstrap"]["event_ci95_high"]),
    ]
    checks.append(
        {
            "check": "stage240_nonstuck_selection_contract",
            "status": "PASS" if all(stage240_contract) else "FAIL",
            "value": {
                "provider_equal_rows": len(
                    stage240.get("provider_equal_rows", [])
                ),
                "provider_station_rows": len(
                    stage240.get("provider_station_rows", [])
                ),
                "leave_one_provider_rows": len(
                    stage240.get("leave_one_provider_out_rows", [])
                ),
                "nonstuck_type_equal_rows": len(
                    stage240.get("nonstuck_type_equal_rows", [])
                ),
                "selection_rules": stage240.get("selection_rules", []),
            },
        }
    )
    stage241 = json.loads(JSON_FILES["stage241"].read_text(encoding="utf-8"))
    stage241_contract = [
        int(stage241.get("target_count", 0)) == 4,
        int(stage241.get("station_count", 0)) == 42,
        len(stage241.get("source_rows", [])) == 4,
        len(stage241.get("leave_one_target_rows", [])) == 4,
        len(stage241.get("type_rows", [])) == 6,
        float(stage241.get("provider_equal_f1_difference", -1.0)) > 0.0,
        float(stage241.get("nonstuck_row_recall_difference", 1.0)) < 0.0,
        float(
            stage241.get("nonstuck_event_coverage_difference", 1.0)
        )
        < 0.0,
    ]
    checks.append(
        {
            "check": "stage241_transfer_heterogeneity_contract",
            "status": "PASS" if all(stage241_contract) else "FAIL",
            "value": {
                "target_count": stage241.get("target_count"),
                "station_count": stage241.get("station_count"),
                "provider_equal_f1_difference": stage241.get(
                    "provider_equal_f1_difference"
                ),
                "nonstuck_row_recall_difference": stage241.get(
                    "nonstuck_row_recall_difference"
                ),
                "nonstuck_event_coverage_difference": stage241.get(
                    "nonstuck_event_coverage_difference"
                ),
            },
        }
    )
    stage242 = json.loads(JSON_FILES["stage242"].read_text(encoding="utf-8"))
    stage242_contract = [
        int(stage242.get("provider_count", 0)) == 5,
        int(stage242.get("pairwise_count", 0)) == 20,
        int(stage242.get("pairwise_station_count", 0)) == 216,
        int(stage242.get("pairwise_local_positive", 0)) == 18,
        int(stage242.get("negative_pair_count", 0)) == 2,
        len(stage242.get("negative_pairs", [])) == 2,
        not bool(stage242.get("all_targets_local_positive", True)),
        not bool(stage242.get("all_sources_local_positive", True)),
        int(stage242.get("local_arm_exact_provider_count", 0)) == 4,
        sum(
            bool(row.get("exact_match"))
            for row in stage242.get("local_arm_reproduction", [])
            if row.get("exact_upstream_replay_expected")
        )
        == 4,
        sum(
            not bool(row.get("exact_upstream_replay_expected"))
            for row in stage242.get("local_arm_reproduction", [])
        )
        == 1,
        not any(
            "target_sign_flip_p" in row
            for row in stage242.get("pairwise_rows", [])
        ),
    ]
    checks.append(
        {
            "check": "stage242_multisource_transfer_contract",
            "status": "PASS" if all(stage242_contract) else "FAIL",
            "value": {
                "provider_count": stage242.get("provider_count"),
                "pairwise_count": stage242.get("pairwise_count"),
                "pairwise_station_count": stage242.get(
                    "pairwise_station_count"
                ),
                "pairwise_local_positive": stage242.get(
                    "pairwise_local_positive"
                ),
                "negative_pair_count": stage242.get("negative_pair_count"),
                "local_arm_exact_provider_count": stage242.get(
                    "local_arm_exact_provider_count"
                ),
            },
        }
    )
    stage242_tables = {
        "evaluation": (
            TABULAR_FILES["stage242_evaluation"],
            EXPECTED_ROWS["stage242_evaluation"],
        ),
        "station_detail": (
            TABULAR_FILES["stage242_station_detail"],
            EXPECTED_ROWS["stage242_station_detail"],
        ),
        "local_reproduction": (
            TABULAR_FILES["stage242_local_reproduction"],
            EXPECTED_ROWS["stage242_local_reproduction"],
        ),
    }
    for label, (path, expected) in stage242_tables.items():
        actual = len(pd.read_csv(path))
        checks.append(
            {
                "check": f"stage242_{label}_table_rows",
                "status": "PASS" if actual == expected else "FAIL",
                "actual": actual,
                "expected": expected,
            }
        )

    stage243 = json.loads(JSON_FILES["stage243"].read_text(encoding="utf-8"))
    stage243_contract = [
        int(stage243.get("pair_count", 0)) == 2,
        len(stage243.get("summary_rows", [])) == 2,
        len(stage243.get("seed_rows", [])) == 10,
        len(stage243.get("metric_rows", [])) == 12,
        all(row.get("status") == "PASS" for row in stage243.get("checks", [])),
    ]
    checks.append(
        {
            "check": "stage243_knmi_negative_case_contract",
            "status": "PASS" if all(stage243_contract) else "FAIL",
            "value": {
                "pair_count": stage243.get("pair_count"),
                "summary_rows": len(stage243.get("summary_rows", [])),
                "seed_rows": len(stage243.get("seed_rows", [])),
                "metric_rows": len(stage243.get("metric_rows", [])),
            },
        }
    )
    stage243_tables = {
        "station_detail": (
            TABULAR_FILES["stage243_station_detail"],
            EXPECTED_ROWS["stage243_station_detail"],
        ),
        "seed_detail": (
            TABULAR_FILES["stage243_seed_detail"],
            EXPECTED_ROWS["stage243_seed_detail"],
        ),
        "metric_detail": (
            TABULAR_FILES["stage243_metric_detail"],
            EXPECTED_ROWS["stage243_metric_detail"],
        ),
        "summary": (
            TABULAR_FILES["stage243_summary"],
            EXPECTED_ROWS["stage243_summary"],
        ),
    }
    for label, (path, expected) in stage243_tables.items():
        actual = len(pd.read_csv(path))
        checks.append(
            {
                "check": f"stage243_{label}_table_rows",
                "status": "PASS" if actual == expected else "FAIL",
                "actual": actual,
                "expected": expected,
            }
        )

    stage244 = json.loads(JSON_FILES["stage244"].read_text(encoding="utf-8"))
    bootstrap = stage244.get("hierarchical_bootstrap", {})
    stage244_contract = [
        stage244.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED",
        int(stage244.get("provider_count", 0)) == 4,
        int(stage244.get("station_count", 0)) == 42,
        int(stage244.get("candidate_count", 0)) == 91,
        int(stage244.get("new_feature_count", 0)) == 7,
        set(stage244.get("selection_rules", []))
        == {"valid_f1", "valid_nonstuck_event_macro"},
        int(bootstrap.get("provider_count", 0)) == 4,
        int(bootstrap.get("station_rows", 0)) == 210,
        int(bootstrap.get("iterations", 0)) == 10000,
        float(bootstrap.get("f1_difference", 0.0)) > 0.0,
        float(bootstrap.get("event_coverage_difference", 0.0)) > 0.0,
        float(bootstrap.get("f1_ci95_low", 0.0)) < 0.0,
        float(bootstrap.get("event_ci95_low", 0.0)) < 0.0,
    ]
    checks.append(
        {
            "check": "stage244_cross_provider_nonstuck_contract",
            "status": "PASS" if all(stage244_contract) else "FAIL",
            "value": {
                "provider_count": stage244.get("provider_count"),
                "station_count": stage244.get("station_count"),
                "candidate_count": stage244.get("candidate_count"),
                "new_feature_count": stage244.get("new_feature_count"),
                "hierarchical_bootstrap": bootstrap,
            },
        }
    )
    stage244_tables = {
        "evaluation": (
            TABULAR_FILES["stage244_evaluation"],
            EXPECTED_ROWS["stage244_evaluation"],
        ),
        "station_detail": (
            TABULAR_FILES["stage244_station_detail"],
            EXPECTED_ROWS["stage244_station_detail"],
        ),
        "type_detail": (
            TABULAR_FILES["stage244_type_detail"],
            EXPECTED_ROWS["stage244_type_detail"],
        ),
        "selection": (
            TABULAR_FILES["stage244_selection"],
            EXPECTED_ROWS["stage244_selection"],
        ),
        "evaluation_aggregate": (
            TABULAR_FILES["stage244_evaluation_aggregate"],
            EXPECTED_ROWS["stage244_evaluation_aggregate"],
        ),
        "provider_station": (
            TABULAR_FILES["stage244_provider_station"],
            EXPECTED_ROWS["stage244_provider_station"],
        ),
        "leave_one_provider_out": (
            TABULAR_FILES["stage244_leave_one_provider_out"],
            EXPECTED_ROWS["stage244_leave_one_provider_out"],
        ),
        "nonstuck_type_detail": (
            TABULAR_FILES["stage244_nonstuck_type_detail"],
            EXPECTED_ROWS["stage244_nonstuck_type_detail"],
        ),
        "nonstuck_type_equal": (
            TABULAR_FILES["stage244_nonstuck_type_equal"],
            EXPECTED_ROWS["stage244_nonstuck_type_equal"],
        ),
        "selection_counts": (
            TABULAR_FILES["stage244_selection_counts"],
            EXPECTED_ROWS["stage244_selection_counts"],
        ),
    }
    for label, (path, expected) in stage244_tables.items():
        actual = len(pd.read_csv(path))
        checks.append(
            {
                "check": f"stage244_{label}_table_rows",
                "status": "PASS" if actual == expected else "FAIL",
                "actual": actual,
                "expected": expected,
            }
        )

    failed = [item for item in checks if item["status"] == "FAIL"]
    result = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED" if not failed else "FAIL",
        "protocol": "对阶段98–123近期结果执行语义合理性审计：指标有限且在[0,1]、概率和q在[0,1]、数值有限、置信区间下界不高于上界、表格行数符合协议、JSON状态正确；差值/增益/标准差等允许为负或超出[0,1]，不强行截断。",
        "checks": checks,
        "failed_checks": failed,
        "limitations": ["语义审计只能发现明显数值和结构异常，不能证明方法优越性。", "差值和置信区间不被误判为[0,1]指标；仍需回到原始定义解释。"],
    }
    (TABLES / "stage114_recent_semantic_sanity_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = pd.DataFrame([{"status": result["status"], "checks": len(checks), "failures": len(failed)}])
    summary.to_csv(TABLES / "stage114_recent_semantic_sanity_audit.csv", index=False, encoding="utf-8-sig")
    lines = ["# 阶段114近期结果语义合理性审计", "", f"- 状态：**{result['status']}**", f"- 检查项：`{len(checks)}`；失败项：`{len(failed)}`", "", "| 检查 | 状态 |", "|---|---|"]
    lines.extend(f"| {item['check']} | {item['status']} |" for item in checks)
    lines += ["", "差值、增益、标准差和置信区间允许出现负值；只有真正的概率/比例指标强制检查[0,1]。"]
    (REPORTS / "stage114_recent_semantic_sanity_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failures": len(failed)}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
