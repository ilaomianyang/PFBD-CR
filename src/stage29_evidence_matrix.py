from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def read_json(name: str) -> dict:
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


def add(rows: list[dict[str, object]], evidence_id: str, stage: str, artifact: str, status: str, policy: str, metrics: str, limits: str) -> None:
    rows.append({"evidence_id": evidence_id, "stage": stage, "artifact": artifact, "status": status, "test_policy": policy, "primary_metrics": metrics, "limitations": limits})


def main() -> int:
    rows: list[dict[str, object]] = []
    b = read_json("stage09b_robust_fusion.json")
    c = read_json("stage09c_multiseed_summary.json")
    d = read_json("stage09d_operating_point.json")
    h = read_json("stage12_hybrid_fusion.json")
    hm = read_json("stage12_hybrid_multiseed_summary.json")
    s22 = read_json("stage22_leave_one_station_refit.json")
    s23 = read_json("stage23_unlabeled_site_calibration.json")
    s24 = read_json("stage24_lag6_feature_experiment.json")
    s26 = read_json("stage26_station_cluster_bootstrap.json")
    s27 = read_json("stage27_temporal_rolling_holdout.json")
    s28 = read_json("stage28_operating_curve.json")
    s30 = read_json("stage30_recovery_integrity_audit.json")
    s32 = read_json("stage32_recovery_utility_audit.json")
    s33 = read_json("stage33_station_hour_threshold.json")
    s34 = read_json("stage34_event_level_bootstrap.json")
    s35 = read_json("stage35_liupanshan_fp_diagnosis.json")
    s36 = read_json("stage36_hampel_cap_experiment.json")
    s37 = read_json("stage37_station_component_calibration.json")
    s38 = read_json("stage38_noaa_controlled_benchmark.json")
    s39 = read_json("stage39_noaa_physical_rule_cap.json")
    s40 = read_json("stage40_noaa_controlled_multiseed.json")
    s41 = read_json("stage41_noaa_baseline_comparison.json")
    s42 = read_json("stage42_multiseed_operating_curve.json")
    s43 = read_json("stage43_noaa_station_hour_threshold.json")
    s44 = read_json("stage44_recovery_confidence_gate.json")
    s45 = read_json("stage45_recovery_gate_multiseed.json")
    s46 = read_json("stage46_cross_station_recovery_gate.json")
    s47 = read_json("stage47_noaa_recovery_gate.json")
    s48 = read_json("stage48_noaa_multivariate_recovery.json")
    s49 = read_json("stage49_noaa_online_threshold.json")
    s50 = read_json("stage50_noaa_rolling_seasonal_review.json")
    s52 = read_json("stage52_noaa_official_download.json")
    s53 = read_json("stage53_noaa_station_calibration.json")
    s54 = read_json("stage54_noaa_station_adaptive_physical.json")
    s55 = read_json("stage55_noaa_climatology_residual.json")
    s56 = read_json("stage56_noaa_station_stratified.json")
    s57 = read_json("stage57_noaa_leave_one_station.json")
    s58 = read_json("stage58_noaa_data_quality_audit.json")
    s59 = read_json("stage59_noaa_station_local_fit.json")
    s60 = read_json("stage60_noaa_station_threshold_multiseed.json")
    s61 = read_json("stage61_noaa_hierarchical_score_calibration.json")
    s62 = read_json("stage62_noaa_hierarchical_rolling_review.json")
    s63 = read_json("stage63_noaa_hierarchical_attack_type.json")
    s64 = read_json("stage64_noaa_empirical_percentile_calibration.json")
    s65 = read_json("stage65_noaa_two_stage_gate_multiseed.json")
    s66 = read_json("stage66_noaa_candidate_baseline_comparison.json")
    s67 = read_json("stage67_noaa_candidate_quality_weak_label.json")
    s68 = read_json("stage68_noaa_quality_station_stratified.json")
    s70 = read_json("stage70_noaa_multiyear_ingest.json")
    s71 = read_json("stage71_noaa_multiyear_quality_audit.json")
    s72 = read_json("stage72_noaa_multiyear_natural_validation.json")
    s73 = read_json("stage73_noaa_multiyear_controlled_benchmark.json")
    s74 = read_json("stage74_noaa_multiyear_calibration_benchmark.json")
    s75 = read_json("stage75_noaa_multiyear_calibration_attack_type.json")
    s76 = read_json("stage76_noaa_multiyear_baseline_comparison.json")
    s77 = read_json("stage77_noaa_multiyear_cluster_bootstrap.json")
    s78 = read_json("stage78_noaa_multiyear_drift_diagnostic.json")
    s79 = read_json("stage79_noaa_prefix_adaptation.json")
    s80 = read_json("stage80_noaa_constrained_prefix_adaptation.json")
    s81 = read_json("stage81_noaa_prefix_bootstrap.json")
    s82 = read_json("stage82_noaa_extended_ingest.json")
    s83 = read_json("stage83_noaa_extended_quality_audit.json")
    s84 = read_json("stage84_noaa_rolling_multiyear_benchmark.json")
    s85 = read_json("stage85_noaa_rolling_baseline_comparison.json")
    s86 = read_json("stage86_noaa_rolling_station_bootstrap.json")
    s87 = read_json("stage87_noaa_extended_natural_validation.json")
    s88 = read_json("stage88_noaa_physical_rule_cap_rolling.json")
    s89 = read_json("stage89_noaa_physical_rule_cap_bootstrap.json")
    s90 = read_json("stage90_noaa_natural_physical_cap.json")
    s91 = read_json("stage91_noaa_physical_component_repair.json")
    s92 = read_json("stage92_noaa_physical_component_bootstrap.json")
    s93 = read_json("stage93_noaa_natural_component_validation.json")
    s94 = read_json("stage94_noaa_natural_component_bootstrap.json")
    s95 = read_json("stage95_noaa_physical_component_attack_type.json")
    s96 = read_json("stage96_noaa_physical_component_operating_curve.json")
    s97 = read_json("stage97_noaa_component_operating_bootstrap.json")
    s98 = read_json("stage98_noaa_low_far_baseline_comparison.json")
    s99 = read_json("stage99_noaa_baseline_paired_bootstrap.json")
    s100 = read_json("stage100_noaa_low_far_station_bootstrap.json")
    s101 = read_json("stage101_noaa_trust_svm_complementarity.json")
    s102 = read_json("stage102_noaa_trust_svm_valid_selection.json")
    s103 = read_json("stage103_noaa_valid_selection_stability.json")
    s104 = read_json("stage104_recent_artifact_integrity.json")
    s105 = read_json("stage105_noaa_attack_type_valid_selected.json")
    s106 = read_json("stage106_noaa_attack_type_paired_bootstrap.json")
    s107 = read_json("stage107_noaa_natural_svm_validation.json")
    s108 = read_json("stage108_noaa_natural_alarm_station_bootstrap.json")
    s109 = read_json("stage109_noaa_quality_weak_label_bootstrap.json")
    s110 = read_json("stage110_noaa_quality_hierarchical_bootstrap.json")
    s111 = read_json("stage111_noaa_quality_count_sensitivity.json")
    s112 = read_json("stage112_full_artifact_manifest.json")
    s113 = read_json("stage113_era5_alignment_leakage_audit.json")
    s114 = read_json("stage114_recent_semantic_sanity_audit.json")
    s115 = read_json("stage115_protocol_boundary_audit.json")
    s116 = read_json("stage116_noaa_quality_label_permutation.json")
    s117 = read_json("stage117_multiseed_uncertainty.json")
    s118 = read_json("stage118_attack_duration_sensitivity.json")
    s119 = read_json("stage119_attack_position_sensitivity.json")
    s120 = read_json("stage120_station_stratified_performance.json")
    s121 = read_json("stage121_local_station_threshold_calibration.json")
    s122 = read_json("stage122_leave_one_station_threshold_transfer.json")
    s123 = read_json("stage123_threshold_transfer_bootstrap.json")
    s124 = read_json("stage124_leave_one_station_refit_multiseed.json")
    s125 = read_json("stage125_leave_one_station_refit_multiseed_bootstrap.json")
    s126 = read_json("stage126_low_far_leave_one_station_operating_points.json")
    s127 = read_json("stage127_low_far_operating_point_bootstrap.json")
    s128 = read_json("stage128_recovery_candidate_diagnosis.json")
    s129 = read_json("stage129_valid_recovery_strategy_selection.json")
    s130 = read_json("stage130_noaa_safety_constrained_recovery.json")
    s131 = read_json("stage131_safety_constraint_sensitivity.json")
    s132 = read_json("stage132_recovery_utility_paired_bootstrap.json")
    s133 = read_json("stage133_causal_temporal_recovery.json")
    s134 = read_json("stage134_causal_recovery_optimization.json")
    s135 = read_json("stage135_independent_seed_replication.json")
    s136 = read_json("stage136_seed_group_consistency.json")
    s137 = read_json("stage137_enhanced_candidate_comparison.json")
    s138 = read_json("stage138_independent_baseline_comparison.json")
    s139 = read_json("stage139_expanded_fusion_grid.json")
    s140 = read_json("stage140_expanded_grid_independent_confirmation.json")
    s141 = read_json("stage141_noaa_frozen_external_validation.json")
    s142 = read_json("stage142_noaa_frozen_natural_operational_validation.json")
    s143 = read_json("stage143_noaa_frozen_trust_svm_safety_combination.json")
    s144 = read_json("stage144_noaa_cross_fold_candidate_transfer.json")
    s145 = read_json("stage145_noaa_station_heterogeneity_audit.json")
    s146 = read_json("stage146_noaa_year_heterogeneity_audit.json")
    s147 = read_json("stage147_noaa_attack_year_interaction_audit.json")
    s148 = read_json("stage148_noaa_baseline_attack_type_detail.json")
    s149 = read_json("stage149_noaa_attack_type_paired_bootstrap.json")
    s150 = read_json("stage150_noaa_attack_type_multiple_comparison.json")
    s151 = read_json("stage151_noaa_attack_intensity_sensitivity.json")
    s152 = read_json("stage152_noaa_baseline_station_comparison.json")
    s153 = read_json("stage153_noaa_natural_station_operational_comparison.json")
    s154 = read_json("stage154_noaa_natural_temporal_drift.json")
    s155 = read_json("stage155_noaa_natural_weather_regime_audit.json")
    s156 = read_json("stage156_noaa_natural_observation_gap_audit.json")
    s157 = read_json("stage157_noaa_natural_two_way_cluster_bootstrap.json")
    s158 = read_json("stage158_noaa_controlled_regime_gap_performance.json")
    s159 = read_json("stage159_noaa_controlled_regime_multiple_comparison.json")
    s160 = read_json("stage160_noaa_controlled_practical_effect_sensitivity.json")
    s161 = read_json("stage161_noaa_controlled_station_regime_performance.json")
    s162 = read_json("stage162_noaa_controlled_station_practical_effect.json")
    s163 = read_json("stage163_noaa_controlled_regime_full_baseline.json")
    s164 = read_json("stage164_noaa_controlled_full_baseline_multiple_comparison.json")
    s165 = read_json("stage165_noaa_controlled_station_full_baseline.json")
    s166 = read_json("stage166_noaa_controlled_attack_environment_interaction.json")
    s167 = read_json("stage167_noaa_controlled_attack_environment_multiple_comparison.json")
    s168 = read_json("stage168_noaa_controlled_attack_environment_practical_effect.json")
    s169 = read_json("stage169_noaa_controlled_attack_intensity_full_baseline.json")
    s170 = read_json("stage170_noaa_controlled_attack_duration_full_baseline.json")
    s171 = read_json("stage171_noaa_controlled_attack_position_full_baseline.json")
    s172 = read_json("stage172_noaa_frozen_expanded_low_far_full_baseline.json")
    s173 = read_json("stage173_noaa_low_far_full_baseline_multiple_comparison.json")
    s174 = read_json("stage174_noaa_low_far_alarm_burden_audit.json")
    s175 = read_json("stage175_noaa_low_far_alarm_burden_stability.json")
    s176 = read_json("stage176_noaa_low_far_station_equal_weight_metrics.json")
    s177 = read_json("stage177_noaa_station_calibrated_operating_points.json")
    s178 = read_json("stage178_noaa_station_calibrated_alarm_budget.json")
    s179 = read_json("stage179_noaa_station_budget_loss_decomposition.json")
    s180 = read_json("stage180_noaa_station_budget_row_tail_audit.json")
    s181 = read_json("stage181_noaa_non_stuck_failure_attribution.json")
    s182 = read_json("stage182_noaa_non_stuck_candidate_features.json")
    s183 = read_json("stage183_noaa_candidate_low_far_constraint.json")
    s184 = read_json("stage184_noaa_validation_calibrated_low_far.json")
    s185 = read_json("stage185_noaa_validation_calibrated_combination.json")
    s186 = read_json("stage186_noaa_targeted_nonstuck_feature_response.json")
    s187 = read_json("stage187_noaa_targeted_features_validation_calibrated.json")
    s188 = read_json("stage188_noaa_two_feature_stability_audit.json")
    s189 = read_json("stage189_noaa_event_level_failure_decomposition.json")
    s190 = read_json("stage190_noaa_event_reranking_validation.json")
    s191 = read_json("stage191_noaa_reranking_focused_diagnostics.json")
    s192 = read_json("stage192_noaa_repeat_gated_reranking.json")
    s193 = read_json("stage193_noaa_persistent_repeat_protection.json")
    s194 = read_json("stage194_noaa_position_perturbation_robustness.json")
    s195 = read_json("stage195_noaa_seed_randomized_placement.json")
    s196 = read_json("stage196_noaa_seed_randomized_independent_replication.json")
    s197o = read_json("stage197_noaa_seed_randomized_station_dependence_original.json")
    s197i = read_json("stage197_noaa_seed_randomized_station_dependence_independent.json")
    s198o = read_json("stage198_noaa_seed_randomized_cross_fold_transfer_original.json")
    s198i = read_json("stage198_noaa_seed_randomized_cross_fold_transfer_independent.json")
    s199 = read_json("stage199_liupanshan_cross_station_protection_transfer.json")
    s200 = read_json("stage200_liupanshan_protection_failure_attribution.json")
    s201 = read_json("stage201_liupanshan_conservative_protection_transfer.json")
    s202 = read_json("stage202_liupanshan_attack_type_station_decomposition.json")
    s203 = read_json("stage203_independent_external_evidence_feasibility.json")
    s204 = read_json("stage204_dwd_protocol_preregistration.json")
    s204a = read_json("stage204a_dwd_protocol_schema_correction.json")
    s205 = read_json("stage205_dwd_official_download.json")
    s206 = read_json("stage206_dwd_normalized.json")
    s207 = read_json("stage207_dwd_controlled_attack.json")
    s208 = read_json("stage208_dwd_persistent_repeat_protection.json")
    s209 = read_json("stage209_dwd_station_cluster_uncertainty.json")
    s210 = read_json("stage210_dwd_seed_randomized_replication.json")
    s211 = read_json("stage211_dwd_replication_station_uncertainty.json")
    s212 = read_json("stage212_dwd_operating_curve.json")
    s213 = read_json("stage213_dwd_external_evidence_closure.json")
    add(rows, "stage09b_core", "9B", "stage09b_robust_fusion.json", b["status"], "train拟合；valid选择；locked test评价", f"F1={b['local_test']['f1']:.4f}; Recall={b['local_test']['recall']:.4f}; AUPRC={b['local_test']['average_precision']:.4f}; FAR={b['test_false_alarm_rate_on_clean_rows']:.4f}", "本地受控攻击；不代表真实入侵")
    add(rows, "stage09c_multiseed", "9C", "stage09c_multiseed_summary.json", "PASS_PIPELINE_REVIEW_REQUIRED", "5个受控攻击种子；每次valid选择；test评价", f"F1={c['metrics']['test_f1']['mean']:.4f}±{c['metrics']['test_f1']['std']:.4f}", "种子重复不等于5组真实攻击数据")
    add(rows, "stage09d_low_far", "9D", "stage09d_operating_point.json", d["status"], "train正常分位数；valid约束FAR；test评价", f"F1={d['test']['f1']:.4f}; Recall={d['test']['recall']:.4f}; FAR={d['test_false_alarm_rate']:.4f}", "安全运行点牺牲Recall")
    add(rows, "stage11_baselines", "11", "stage11_baseline_benchmark.json", "PASS_PIPELINE_REVIEW_REQUIRED", "各基线按同一split和valid阈值", "Isolation Forest F1=0.5360; Logistic supervised F1=0.5576", "监督基线是有标签上界，不与无监督方法同类")
    add(rows, "stage12_hybrid", "12", "stage12_hybrid_multiseed_summary.json", "PASS_PIPELINE_REVIEW_REQUIRED", "独立基线证据加入；valid选择；test评价", f"F1={hm['metrics']['test_f1']['mean']:.4f}±{hm['metrics']['test_f1']['std']:.4f}", "混合模型不代表全面优于所有基线")
    add(rows, "stage13_unseen_attack", "13", "stage13_unseen_attack_generalization.json", "PASS_PIPELINE_REVIEW_REQUIRED", "逐类留出valid攻击类型；test评价", "平均留出类型Recall=0.4402", "time_shift/offset等未见类型较弱")
    add(rows, "stage14_supervised_upper", "14", "stage14_supervised_upper_bounds.json", "PASS_PIPELINE_REVIEW_REQUIRED", "监督标签仅作上界", "监督逻辑回归与HGB上界", "不能与无监督主方法直接宣称公平领先")
    add(rows, "stage15_intensity", "15", "stage15_attack_intensity_sweep.json", "PASS_PIPELINE_REVIEW_REQUIRED", "固定协议下强度0.5/1/1.5/2.0", "F1随强度上升；协同偏移对强度敏感", "受控攻击强度不是真实攻击分布")
    add(rows, "stage16_negative", "16", "stage16_seasonal_cusum_validation.json", "PASS_PIPELINE_REVIEW_REQUIRED", "valid选择；test评价", "季节CUSUM test F1=0.5144", "未形成稳定主模型增益")
    add(rows, "stage17_empirical", "17", "stage17_empirical_attack_profiles.parquet", "PASS_PIPELINE_REVIEW_REQUIRED", "NOAA差分经验分布构造受控攻击", "经验分布攻击结果已保存", "仍是受控注入，不是真实攻击日志")
    add(rows, "stage18_statistics", "18", "stage18_statistical_comparison.json", "PASS_PIPELINE_REVIEW_REQUIRED", "2000次配对bootstrap", "AUPRC优于Isolation Forest；F1未优于两基线", "结论限定于当前本地test")
    add(rows, "stage19_noaa_quality", "19", "stage19_noaa_quality_audit.json", "PASS_PIPELINE_REVIEW_REQUIRED", "NOAA官方质量标记弱标签审计", "可疑14行风险更高", "不能计算攻击召回率")
    add(rows, "stage20_noaa_temporal", "20", "stage20_noaa_temporal_holdout.json", "PASS_PIPELINE_REVIEW_REQUIRED", "NOAA前半年校准、后半年留出", "后半年报警率=0.1023%", "无攻击自然观测；不是召回率")
    add(rows, "stage21_threshold_transfer", "21", "stage21_leave_one_station_out.json", "PASS_PIPELINE_REVIEW_REQUIRED", "留出站点；仅迁移阈值", "平均F1=0.5186", "风险模型仍为全站点拟合")
    add(rows, "stage22_strict_refit", "22", "stage22_leave_one_station_refit.json", s22["status"], "留出站点不参与任何拟合；test仅评价", f"F1={s22['summary']['mean_test_f1']:.4f}±{s22['summary']['std_test_f1']:.4f}; AUPRC={s22['summary']['mean_test_auprc']:.4f}; FAR={s22['summary']['mean_test_false_alarm_rate']:.4f}", "仅7站点；弱站点和弱攻击必须保留")
    add(rows, "stage23_unlabeled_calibration", "23", "stage23_unlabeled_site_calibration.json", s23["status"], "其余站点拟合；留出站点只用无标签train MAD校准", f"F1={s23['summary']['mean_test_f1']:.4f}; Recall={s23['summary']['mean_test_recall']:.4f}; FAR={s23['summary']['mean_test_false_alarm_rate']:.4f}", "低误报改善伴随Recall/事件召回下降")
    add(rows, "stage24_negative", "24", "stage24_lag6_feature_experiment.json", s24["status"], "valid选择lag-6特征；test评价", f"F1={s24['test']['f1']:.4f}; AUPRC={s24['test']['average_precision']:.4f}", "未优于9B；作为负结果保留")
    add(rows, "stage25_integrity", "25", "stage25_data_integrity_audit.json", read_json("stage25_data_integrity_audit.json")["status"], "独立数据审计", "无重复、无跨split重叠、扩展攻击未进train", "只证明数据完整，不证明性能")
    add(rows, "stage26_cluster_bootstrap", "26", "stage26_station_cluster_bootstrap.json", s26["status"], "7站点簇级5000次配对bootstrap", "校准F1差值CI不跨0；Recall差值CI接近跨0", "站点数量仍较少")
    add(rows, "stage27_temporal_holdout", "27", "stage27_temporal_rolling_holdout.json", s27["status"], "三个固定滚动时间窗口；valid选择；test评价", f"F1={s27['summary']['mean_test_f1']:.4f}±{s27['summary']['std_test_f1']:.4f}; AUPRC={s27['summary']['mean_test_auprc']:.4f}", "时间窗口造成性能和FAR波动")
    add(rows, "stage28_operating_curve", "28", "stage28_operating_curve.json", s28["status"], "valid正常分位数运行点；test评价", "0.5% valid FAR点：test FAR=0.384%，Precision=0.9403", "运行点存在Recall/FAR权衡")
    add(rows, "stage30_recovery_integrity", "30", "stage30_recovery_integrity_audit.json", s30["status"], "逐个检查恢复明细", "10个恢复产物均无未报警行被改写", "误报行仍可能被恢复动作改写，需单独解释")
    add(rows, "stage32_recovery_utility", "32", "stage32_recovery_utility_audit.json", s32["status"], "锁定test逐行统计检测后改善/恶化与误报伤害", "9B检测攻击MAE 3.4011→0.8935；改善率85.5%", "误报恢复可能伤害正常观测，低误报点需单独报告")
    add(rows, "stage33_station_hour_threshold", "33", "stage33_station_hour_threshold.json", s33["status"], "train按站点×小时估计阈值；valid选收缩；test评价", f"F1={s33['test']['f1']:.4f}; FAR={s33['test_false_alarm_rate']:.4f}; Precision={s33['test']['precision']:.4f}", "降低误伤但降低Recall，作为负结果")
    add(rows, "stage34_event_level_bootstrap", "34", "stage34_event_level_bootstrap.json", s34["status"], "锁定test按combined_event_id聚合；5000次事件级bootstrap；不重新调参", f"事件召回={s34['overall']['event_recall']:.4f} [{s34['overall']['event_recall_ci95'][0]:.4f},{s34['overall']['event_recall_ci95'][1]:.4f}]; 加权行召回={s34['overall']['row_recall_weighted']:.4f}", "测试事件以单行事件为主；事件级结果不能替代攻击类型和逐行结果")
    add(rows, "stage35_liupanshan_fp_diagnosis", "35", "stage35_liupanshan_fp_diagnosis.json", s35["status"], "锁定test分量、小时、日期和站点参照描述；不重新调参", f"liupanshan误伤率={s35['false_alarm_rate']:.4f}; 误报行={s35['false_positive_rows']}; Hampel误报/正常均值比约11.5x", "诊断性分析，不构成性能改进")
    add(rows, "stage36_hampel_cap", "36", "stage36_hampel_cap_experiment.json", s36["status"], "valid选择Hampel分量上限和融合设置；locked test评价", f"最优候选test F1={s36['selected_by_valid']['test_f1']:.4f}; liupanshan FAR={s36['selected_by_valid']['liupanshan_false_alarm_rate']:.4f}", "简单分量截断降低局部误报但未提升整体指标，保留负结果")
    add(rows, "stage37_station_component_calibration", "37", "stage37_station_component_calibration.json", s37["status"], "train正常样本估计站点分量尺度；valid选择；locked test评价", f"候选最优test F1={s37['selected_by_valid']['test']['f1']:.4f}; AUPRC={s37['selected_by_valid']['test']['average_precision']:.4f}", "站点分量归一化未优于9B，保留负结果")
    add(rows, "stage38_noaa_controlled_benchmark", "38", "stage38_noaa_controlled_benchmark.json", s38["status"], "NOAA三站点按时间留出；train拟合、valid选择、locked test评价；攻击为透明受控注入", f"test F1={s38['test']['f1']:.4f}; Recall={s38['test']['recall']:.4f}; AUPRC={s38['test']['average_precision']:.4f}; FAR={s38['test_false_alarm_rate']:.4f}", "NOAA无真实攻击真值；外部受控注入结果不能冒充真实攻击检测")
    add(rows, "stage39_noaa_physical_rule_cap", "39", "stage39_noaa_physical_rule_cap.json", s39["status"], "NOAA外部受控测试；valid选择物理规则分量上限；test评价", f"候选与参考test F1均为{s39['selected_by_valid']['test_f1']:.4f}; FAR均为{s39['selected_by_valid']['test_false_alarm_rate']:.4f}", "季节域偏移未被单一分量截断解决，保留负结果")
    add(rows, "stage40_noaa_controlled_multiseed", "40", "stage40_noaa_controlled_multiseed.json", s40["status"], "NOAA固定时间切分；5个受控攻击种子；每次valid选择、locked test评价", f"test F1={s40['aggregate']['test_f1']['mean']:.4f}±{s40['aggregate']['test_f1']['std']:.4f}; FAR={s40['aggregate']['test_false_alarm_rate']['mean']:.4f}±{s40['aggregate']['test_false_alarm_rate']['std']:.4f}", "外部攻击仍为受控注入；季节迁移导致性能波动")
    add(rows, "stage41_noaa_baseline_comparison", "41", "stage41_noaa_baseline_comparison.json", s41["status"], "NOAA同一时间切分/攻击/5种子；无监督基线只用train正常样本拟合", "TrustMeteo F1均值略高；AUPRC低于基线；FAR仍偏高", "不能宣称外部全面领先；仅支持有限的F1相对优势")
    op1 = next(x for x in s42["aggregate"] if x["operating_point"] == "valid_FAR_1.0%")
    add(rows, "stage42_multiseed_operating_curve", "42", "stage42_multiseed_operating_curve.json", s42["status"], "9B五种子；各自valid正常分位数或valid F1选点；locked test评价", f"valid FAR 1%点test FAR={op1['test_far_mean']:.4f}±{op1['test_far_std']:.4f}; Precision={op1['test_precision_mean']:.4f}±{op1['test_precision_std']:.4f}; Recall={op1['test_recall_mean']:.4f}±{op1['test_recall_std']:.4f}", "低误报运行点牺牲召回；必须同时报告F1最优点")
    add(rows, "stage43_noaa_station_hour_threshold", "43", "stage43_noaa_station_hour_threshold.json", s43["status"], "NOAA受控外部测试；train正常样本按站点×小时估计阈值；valid选择；test评价", f"候选test F1={s43['selected_by_valid']['test_f1']:.4f}; FAR={s43['selected_by_valid']['test_false_alarm_rate']:.4f}", "降低外部误报但显著牺牲F1/Recall，保留负结果")
    add(rows, "stage44_recovery_confidence_gate", "44", "stage44_recovery_confidence_gate.json", s44["status"], "9B检测预测固定；valid选择恢复风险门控和恢复权重；locked test评价", f"valid选择门控后test净误差收益={s44['selected_by_valid']['test']['net_absolute_error_gain']:.2f}; 攻击恶化率={s44['selected_by_valid']['test']['detected_attack_worsening_rate']:.4f}", "只优化恢复处置，不改变检测指标；需多种子和跨站点复核")
    add(rows, "stage45_recovery_gate_multiseed", "45", "stage45_recovery_gate_multiseed.json", s45["status"], "9B五种子；各自valid选择恢复门控和权重；test评价", f"五种子平均总误差净收益={s45['aggregate_selected']['test_net_absolute_error_gain']['mean']:.2f}; 攻击恢复恶化率需结合逐种子表解释", "尚未完成留一站点恢复复核，不能替换主恢复策略")
    add(rows, "stage46_cross_station_recovery_gate", "46", "stage46_cross_station_recovery_gate.json", s46["status"], "严格留一站点；其余站点valid选择门控和恢复权重；留出站点test评价", f"7站点平均总误差净收益={s46['aggregate']['selected']['net_absolute_error_gain']['mean']:.2f}; 6/7站点选择0.5门控分位数", "部分站点仍选择不门控；需更多独立站点后才能固化为主策略")
    add(rows, "stage47_noaa_recovery_gate", "47", "stage47_noaa_recovery_gate.json", s47["status"], "NOAA三站点固定时间切分；5种子；门控和时间/空间恢复权重只由valid选择；test评价", f"门控平均净收益={s47['summary']['selected_mean_net_gain']:.2f}; 未门控参考={s47['summary']['reference_mean_net_gain']:.2f}", "相对未门控有改善但仍为负净收益，不能跨数据源固化")
    add(rows, "stage48_noaa_multivariate_recovery", "48", "stage48_noaa_multivariate_recovery.json", s48["status"], "NOAA固定时间切分；5种子；多变量候选仅由train正常样本拟合；valid选门控/权重；test评价", f"多变量平均净收益={s48['comparison']['multivariate_selected_mean_net_gain']:.2f}; 时间/空间基线={s48['comparison']['stage47_temporal_spatial_mean_net_gain']:.2f}", "相对基线有方向性改善但仍为负，不能跨域固化")
    add(rows, "stage49_noaa_online_threshold", "49", "stage49_noaa_online_threshold.json", s49["status"], "NOAA固定时间切分；因果历史风险分位数；q和窗口仅由valid选择；test评价", f"在线阈值test F1={s49['comparison']['online_test_f1_mean']:.4f}; 固定阈值={s49['comparison']['fixed_test_f1_mean']:.4f}", "单一季节切分上的候选，需滚动季节复核")
    add(rows, "stage50_noaa_rolling_seasonal_review", "50", "stage50_noaa_rolling_seasonal_review.json", s50["status"], "三个预先固定滚动季节窗口；每折valid选在线阈值；test评价", f"平均F1提升={s50['mean_f1_gain_vs_fixed']:.4f}; 正向窗口={s50['folds_with_positive_mean_gain']}/3", "仍是NOAA受控注入；需要更多独立季节/数据源后才能固化")
    s53_hour = next(x for x in s53["aggregate"] if x["mode"] == "station_hour_mad" and x["operating_point"] == "far1")
    s54_adaptive = next(x for x in s54["aggregate"] if x["mode"] == "station_adaptive_physical_scale")
    add(rows, "stage51_noaa_station_inventory", "51", "stage51_noaa_station_inventory.json", "PASS", "仅按本地官方isd-history筛选；不下载、不修改实验结果", "符合2022年覆盖站点=12；新增候选=9", "清单不是性能证据；候选必须先下载和审计")
    add(rows, "stage52_noaa_official_download", "52", "stage52_noaa_official_download.json", s52["status"], "仅NOAA/NCEI官方URL；保存下载时间、字节数和SHA256；字段初检", f"下载={s52['downloaded_count']}; 字段通过={s52['audit_columns_ok_count']}/{s52['candidate_count']}", "仅证明数据来源和格式，不证明检测性能")
    add(rows, "stage53_noaa_station_calibration", "53", "stage53_noaa_station_calibration.json", s53["status"], "12站点；校准只用train正常样本；valid选阈值；test评价", f"station×hour FAR1%点：F1={s53_hour['test_f1_mean']:.4f}; Recall={s53_hour['test_recall_mean']:.4f}; FAR={s53_hour['test_far_mean']:.4f}; AUPRC={s53_hour['test_auprc_mean']:.4f}", "降低误报但召回不足；需独立季节复核，不替换主模型")
    add(rows, "stage54_noaa_station_adaptive_physical", "54", "stage54_noaa_station_adaptive_physical.json", s54["status"], "12站点、5种子；仅变化率尺度站点自适应；valid选择；test评价", f"自适应F1={s54_adaptive['test_f1_mean']:.4f}; AUPRC={s54_adaptive['test_auprc_mean']:.4f}; FAR={s54_adaptive['test_far_mean']:.4f}", "与全局尺度完全相同，作为无增益优化保留")
    s55_res = next(x for x in s55["aggregate"] if x["mode"] == "station_hour_residual")
    add(rows, "stage55_noaa_climatology_residual", "55", "stage55_noaa_climatology_residual.json", s55["status"], "12站点、5种子；站点—小时基线只用train正常观测；valid选择；test评价", f"残差F1={s55_res['test_f1_mean']:.4f}; AUPRC={s55_res['test_auprc_mean']:.4f}; FAR={s55_res['test_far_mean']:.4f}", "AUPRC有方向性改善但F1和FAR仍不足")
    add(rows, "stage56_noaa_station_stratified", "56", "stage56_noaa_station_stratified.json", s56["status"], "12站点、5种子；固定阈值；按站点分层统计；不重新调参", "识别高FAR站点和站点异质性来源", "诊断性证据，不得删除弱站点")
    add(rows, "stage57_noaa_leave_one_station", "57", "stage57_noaa_leave_one_station.json", s57["status"], "12站点、3种子；每次留出站点完全不参与train/valid拟合；test只评价留出站点", f"总体F1={s57['summary']['test_f1']['mean']:.4f}±{s57['summary']['test_f1']['std']:.4f}; AUPRC={s57['summary']['test_auprc']['mean']:.4f}; FAR={s57['summary']['test_far']['mean']:.4f}", "外部冷启动泛化失败；不能宣称跨站点普适")
    add(rows, "stage58_noaa_data_quality_audit", "58", "stage58_noaa_data_quality_audit.json", s58["status"], "阶段6标准化表逐站检查时间唯一性、名义采样间隔覆盖、缺失、质量标记和坐标稳定性", f"站点={s58['stations']}; 重复行={s58['summary']['duplicate_rows']}; 最小名义覆盖率={s58['summary']['min_nominal_coverage']:.4f}", "数据质量通过不等于攻击检测性能通过")
    add(rows, "stage59_noaa_station_local_fit", "59", "stage59_noaa_station_local_fit.json", s59["status"], "12站点、5种子；时间检测器按本站train正常数据拟合；valid统一选阈值；test只评价", f"本地拟合test F1={s59['aggregate']['test_f1']['mean']:.4f}; FAR={s59['aggregate']['test_far']['mean']:.4f}; AUPRC={s59['aggregate']['test_auprc']['mean']:.4f}", "统一阈值下近乎全量报警，作为负结果保留")
    add(rows, "stage60_noaa_station_threshold_multiseed", "60", "stage60_noaa_station_threshold_multiseed.json", s60["status"], "12站点、5种子；q和站点/小时收缩比例只在valid选择；阈值由train正常样本估计；test只评价", f"选中方案test F1={s60['summary']['test_f1']['mean']:.4f}±{s60['summary']['test_f1']['std']:.4f}; AUPRC={s60['summary']['test_auprc']['mean']:.4f}; FAR={s60['summary']['test_far']['mean']:.4f}; 相对固定参考F1差值={s60['f1_gain_vs_reference']:.4f}", "未优于固定融合参考，作为负结果保留；外部攻击仍为透明受控注入")
    add(rows, "stage61_noaa_hierarchical_score_calibration", "61", "stage61_noaa_hierarchical_score_calibration.json", s61["status"], "12站点、5种子；全局/站点/小时中位数和MAD只由train正常样本估计；收缩系数、小时权重和分位阈值只在valid选择；test只评价", f"层级校准test F1={s61['summary']['test_f1']['mean']:.4f}±{s61['summary']['test_f1']['std']:.4f}; AUPRC={s61['summary']['test_auprc']['mean']:.4f}; FAR={s61['summary']['test_far']['mean']:.4f}; F1差值={s61['f1_gain_vs_reference']:.4f}; AUPRC差值={s61['auprc_gain_vs_reference']:.4f}", "F1/Recall未超过固定参考，但AUPRC和FAR改善；仅作为低误报候选，需跨季节复核")
    add(rows, "stage62_noaa_hierarchical_rolling_review", "62", "stage62_noaa_hierarchical_rolling_review.json", s62["status"], "三个预先固定滚动季节窗口、五个攻击种子；每折train拟合层级校准，valid选参数，test只评价", f"跨窗口平均F1差值={s62['mean_f1_gain_vs_fixed']:.4f}; AUPRC差值={s62['mean_auprc_gain_vs_fixed']:.4f}; F1正向窗口={s62['folds_with_positive_mean_f1_gain']}/{len(s62['folds'])}", "仅1/3窗口F1正向；AUPRC改善伴随召回下降，作为低误报候选和负结果保留")
    add(rows, "stage63_noaa_hierarchical_attack_type", "63", "stage63_noaa_hierarchical_attack_type.json", s63["status"], "使用阶段62每折每种子valid选定参数；不重新使用test标签；按六类受控攻击统计test Recall和正常FAR", "spike/offset/drift/time_shift/stuck/gaussian_noise逐类Recall与固定参考对照已保存", "层级标准化在所有攻击类型上Recall均明显低于固定参考；只作为诊断性负结果")
    add(rows, "stage64_noaa_empirical_percentile_calibration", "64", "stage64_noaa_empirical_percentile_calibration.json", s64["status"], "12站点、5种子；经验分位数参考分布只由train正常样本估计；站点/小时/全局权重和阈值只在valid选择；test只评价", f"经验分位数test F1={s64['summary']['test_f1']['mean']:.4f}±{s64['summary']['test_f1']['std']:.4f}; AUPRC={s64['summary']['test_auprc']['mean']:.4f}; FAR={s64['summary']['test_far']['mean']:.4f}; F1差值={s64['f1_gain_vs_reference']:.4f}; AUPRC差值={s64['auprc_gain_vs_reference']:.4f}", "AUPRC/FAR改善但F1和Recall未改善，作为负结果保留")
    add(rows, "stage65_noaa_two_stage_gate_multiseed", "65", "stage65_noaa_two_stage_gate_multiseed.json", s65["status"], "12站点、5种子；原始风险和站点经验分位数只由train正常样本估计；门控模式和两个分位阈值只在valid选择；test只评价", f"两阶段门控test F1={s65['summary']['test_f1']['mean']:.4f}±{s65['summary']['test_f1']['std']:.4f}; AUPRC={s65['summary']['test_auprc']['mean']:.4f}; FAR={s65['summary']['test_far']['mean']:.4f}; F1差值={s65['f1_gain_vs_reference']:.4f}", "最优门控退化为站点分位数单独门控，与阶段64等价；未恢复Recall，作为负结果保留")
    add(rows, "stage66_noaa_candidate_baseline_comparison", "66", "stage66_noaa_candidate_baseline_comparison.json", s66["status"], "同一NOAA 12站点、同一时间切分和五种子；候选来自阶段64 valid选择结果；不重新调参；仅做公平外部汇总", "经验分位数候选F1/AUPRC/FAR与Robust、Isolation Forest、One-Class SVM、固定TrustMeteo-QC的均值、标准差和配对差值已保存", "候选AUPRC/FAR改善但F1低于固定参考和One-Class SVM；不能宣称全面领先")
    add(rows, "stage67_noaa_candidate_quality_weak_label", "67", "stage67_noaa_candidate_quality_weak_label.json", s67["status"], "使用阶段64每种子valid选定参数；只统计test中未注入的自然观测；NOAA温度质量标记仅作弱标签", f"质量可疑组与正常组报警率差值={s67['weak_label_contrast']['alarm_rate_difference_flagged_minus_clean']:.4f}; 分组风险和报警率按种子汇总已保存", "质量标记不是攻击真值；只能支持自然观测质量风险区分，不能写成攻击召回率")
    add(rows, "stage68_noaa_quality_station_stratified", "68", "stage68_noaa_quality_station_stratified.json", s68["status"], "使用阶段64每种子valid选定参数；只统计test中未注入自然观测；按12站点分层比较质量可疑/正常报警率", f"质量可疑组报警率高于正常组的站点={s68['stations_with_positive_alarm_rate_difference']}/{s68['station_count']}", "仅5/12站点方向一致，部分可疑样本极少；弱标签结论受站点异质性和样本量限制")
    add(rows, "stage70_noaa_multiyear_ingest", "70", "stage70_noaa_multiyear_ingest.json", s70["status"], "递归接入NOAA官方2021、2022、2023文件；保留year字段；不修改2022阶段6基线", f"文件={s70['file_count']}; 站点={s70['stations']}; 年份={s70['years']}; 行数={s70['rows']}; 重复键={s70['duplicate_station_timestamp_rows']}", "数据接入审计不等于检测性能通过")
    add(rows, "stage71_noaa_multiyear_quality_audit", "71", "stage71_noaa_multiyear_quality_audit.json", s71["status"], "逐年逐站审计重复时间戳、名义覆盖、缺失、质量标记和坐标；不合格站点-年份排除", f"合格站点-年份={s71['eligible_station_years']}; 排除={len(s71['excluded_station_years'])}; 排除项含2021-57014099999长缺口", "排除项必须披露；不能通过放宽覆盖率阈值进入性能实验")
    add(rows, "stage72_noaa_multiyear_natural_validation", "72", "stage72_noaa_multiyear_natural_validation.json", s72["status"], "2021训练、2022无标签验证、2023无标签测试；训练分位数设阈值；不注入攻击、不使用测试标签", "三年度自然报警率、质量可疑/正常分组报警率和q=0.99/0.995/0.999结果已保存", "自然报警率不是攻击召回率；仅支持时间稳定性和弱标签审计")
    add(rows, "stage73_noaa_multiyear_controlled_benchmark", "73", "stage73_noaa_multiyear_controlled_benchmark.json", s73["status"], "2021训练、2022验证、2023锁定测试；11个三年质量合格站点；5个受控攻击种子；train无攻击", f"跨年份固定方案test F1={s73['aggregate']['test_f1']['mean']:.4f}; Recall={s73['aggregate']['test_recall']['mean']:.4f}; FAR={s73['aggregate']['test_false_alarm_rate']['mean']:.4f}", "攻击Recall接近1但FAR约99%，说明训练年份阈值严重失配；受控攻击不是真实攻击日志")
    add(rows, "stage74_noaa_multiyear_calibration_benchmark", "74", "stage74_noaa_multiyear_calibration_benchmark.json", s74["status"], "2021训练、2022验证校准、2023锁定测试；11个三年质量合格站点；5个种子；test只评价", f"校准test F1={s74['summary']['test_f1']['mean']:.4f}; Recall={s74['summary']['test_recall']['mean']:.4f}; FAR={s74['summary']['test_far']['mean']:.4f}; 相对固定FAR差值={s74['f1_gain_vs_fixed']:.4f}", "FAR显著下降但Recall大幅下降，不能替换9B主模型；需作为安全运行点/负结果报告")
    add(rows, "stage75_noaa_multiyear_calibration_attack_type", "75", "stage75_noaa_multiyear_calibration_attack_type.json", s75["status"], "使用阶段74由2022 valid选择的参数；只统计2023 locked test逐类Recall；不重新调参", "六类攻击逐类固定Recall、校准Recall和变化已保存", "所有攻击类型Recall均下降；用于解释跨年份校准的FAR/Recall权衡")
    add(rows, "stage76_noaa_multiyear_baseline_comparison", "76", "stage76_noaa_multiyear_baseline_comparison.json", s76["status"], "同一2021→2022→2023协议、11个三年质量合格站点、5个种子；无监督基线只用train正常样本拟合，valid选阈值，test评价", "Robust、Isolation Forest、One-Class SVM和TrustMeteo-QC的跨年份F1/Recall/AUPRC/FAR公平表已保存", "One-Class SVM和Isolation Forest优于TrustMeteo-QC；跨年份固定融合不具备外部优势")
    add(rows, "stage77_noaa_multiyear_cluster_bootstrap", "77", "stage77_noaa_multiyear_cluster_bootstrap.json", s77["status"], "2023 locked test；11站点簇、5种子、5000次配对bootstrap；阈值仅由valid选择", f"SVM−Trust F1差值={s77['summary']['f1_diff_mean']:.4f}，95% CI=[{s77['summary']['f1_diff_ci95'][0]:.4f},{s77['summary']['f1_diff_ci95'][1]:.4f}]; FAR差值={s77['summary']['far_diff_mean']:.4f}，CI=[{s77['summary']['far_diff_ci95'][0]:.4f},{s77['summary']['far_diff_ci95'][1]:.4f}]", "只支持当前跨年份协议下的站点簇级差异；不证明所有数据源或攻击类型普适")
    add(rows, "stage78_noaa_multiyear_drift_diagnostic", "78", "stage78_noaa_multiyear_drift_diagnostic.json", s78["status"], "冻结评分器；排除透明受控注入行；按11站点、三年度、5种子配对量化自然观测漂移", f"特征={len(s78['features'])}; 聚合组合={s78['aggregate_rows']}; 最大PSI P95={s78['severe_pairs'][0]['psi_p95']:.4f} ({s78['severe_pairs'][0]['feature']})", "PSI对离散分数敏感；只用于诊断和预注册自适应，不等于攻击检测性能")
    add(rows, "stage79_noaa_prefix_adaptation_negative", "79", "stage79_noaa_prefix_adaptation.json", s79["status"], "2022目标年份无标签前缀选择prefix/q；冻结后应用于2023前缀后locked test；不使用2023标签调参", f"自适应F1={s79['summary']['test_f1']['mean']:.4f}; Recall={s79['summary']['test_recall']['mean']:.4f}; FAR={s79['summary']['test_far']['mean']:.4f}; 相对固定F1差值={s79['gains_vs_fixed']['f1']:.4f}", "FAR下降但F1/Recall显著下降，不能替换固定方案；作为负结果保留")
    add(rows, "stage80_noaa_constrained_prefix_adaptation", "80", "stage80_noaa_constrained_prefix_adaptation.json", s80["status"], "2022验证期预定义max-F1、Recall≥0.5和FAR≤0.10运行点；候选为目标年前缀分位数与固定阈值收缩；2023只评价", f"max-F1运行点test F1={next(x['test_f1_mean'] for x in s80['summary'] if x['operating_point']=='max_f1'):.4f}; FAR={next(x['test_far_mean'] for x in s80['summary'] if x['operating_point']=='max_f1'):.4f}; Recall={next(x['test_recall_mean'] for x in s80['summary'] if x['operating_point']=='max_f1'):.4f}", "max-F1仅小幅增益且Recall下降；低FAR运行点检测能力不足；不能直接替换主模型")
    add(rows, "stage81_noaa_prefix_bootstrap", "81", "stage81_noaa_prefix_bootstrap.json", s81["status"], "2023 locked test；11站点簇、5种子、5000次配对bootstrap；候选参数只由2022 valid选择", f"F1差值={s81['summary']['f1_diff_mean']:.4f}; 95% CI=[{s81['summary']['f1_diff_ci95'][0]:.4f},{s81['summary']['f1_diff_ci95'][1]:.4f}]; FAR差值={s81['summary']['far_diff_mean']:.4f}; F1正向概率={s81['summary']['prob_f1_positive']:.4f}", "F1差异CI跨零且Recall显著下降；仅支持低误报运行点候选，不证明主模型全面改进")
    add(rows, "stage82_noaa_extended_ingest", "82", "stage82_noaa_extended_ingest.json", s82["status"], "独立接入2020–2025 NOAA/NCEI官方文件；保留2022原始文件；不覆盖阶段70旧表", f"文件={s82['file_count']}; 站点={s82['stations']}; 年份={s82['years']}; 行数={s82['rows']}; 重复键={s82['duplicate_station_timestamp_rows']}", "标准化接入不等于质量和性能通过；后续仅使用阶段83合格站点-年份")
    add(rows, "stage83_noaa_extended_quality_audit", "83", "stage83_noaa_extended_quality_audit.json", s83["status"], "逐年逐站审计覆盖率、缺失、重复、坐标稳定性；不合格站点-年份排除", f"可用站点-年份={s83['eligible_station_years']}; 排除={len(s83['excluded_station_years'])}; 年份={s83['years']}", "57014099999的2020/2021因长缺口排除；不能通过放宽覆盖率纳入")
    add(rows, "stage84_noaa_rolling_multiyear_benchmark", "84", "stage84_noaa_rolling_multiyear_benchmark.json", s84["status"], "11个全年度合格站点；累计历史训练→前一年验证→目标年测试；2022–2025四折；5种子；test只评价", f"总体test F1={s84['overall']['test_f1']['mean']:.4f}±{s84['overall']['test_f1']['std']:.4f}; Recall={s84['overall']['test_recall']['mean']:.4f}; AUPRC={s84['overall']['test_auprc']['mean']:.4f}; FAR={s84['overall']['test_far']['mean']:.4f}", "跨年份表现随折次变化明显；攻击为透明受控注入，不是真实网络攻击日志")
    add(rows, "stage85_noaa_rolling_baseline_comparison", "85", "stage85_noaa_rolling_baseline_comparison.json", s85["status"], "与阶段84完全相同滚动折、站点、种子和攻击；无监督基线只用train正常样本拟合、valid选阈值", "One-Class SVM/Isolation Forest/Robust max deviation/TrustMeteo-QC的总体及逐折F1、Recall、AUPRC、FAR已保存", "One-Class SVM和Isolation Forest总体优于TrustMeteo-QC；必须报告负结果，不能宣称跨年份全面领先")
    add(rows, "stage86_noaa_rolling_station_bootstrap", "86", "stage86_noaa_rolling_station_bootstrap.json", s86["status"], "阶段85相同四个滚动折；11站点簇、5种子、5000次配对bootstrap；阈值仅由valid选择", f"SVM−Trust F1差值={s86['summary_vs_trust']['One-Class SVM']['f1_diff_mean']:.4f}; 95% CI=[{s86['summary_vs_trust']['One-Class SVM']['f1_diff_ci95'][0]:.4f},{s86['summary_vs_trust']['One-Class SVM']['f1_diff_ci95'][1]:.4f}]; IF−Trust F1差值={s86['summary_vs_trust']['Isolation Forest']['f1_diff_mean']:.4f}", "站点簇级差异支持当前扩展协议下基线更强；不外推为所有数据源或真实攻击普遍结论")
    add(rows, "stage87_noaa_extended_natural_validation", "87", "stage87_noaa_extended_natural_validation.json", s87["status"], "2020–2025扩展表；11个全年度合格站点；历史年份训练、目标年份自然观测测试；不注入攻击、不使用目标年标签", "2021–2025自然报警率、质量可疑/正常弱标签差异和站点分层结果已保存", "自然报警率不是攻击召回率；2022–2025报警率为零需结合分数尺度诊断，不能写成性能全面稳定")
    add(rows, "stage88_noaa_physical_rule_cap_rolling", "88", "stage88_noaa_physical_rule_cap_rolling.json", s88["status"], "四折累计历史训练→前一年验证→目标年测试；Physical Rules上限只由valid选择；test只评价", f"选中候选F1差值={s88['selected_vs_uncapped']['f1_gain']:.4f}; AUPRC差值={s88['selected_vs_uncapped']['auprc_gain']:.4f}; FAR差值={s88['selected_vs_uncapped']['far_gain']:.4f}", "候选会牺牲部分Recall；尚未经过跨更多年份验证，不能替换9B")
    add(rows, "stage89_noaa_physical_rule_cap_bootstrap", "89", "stage89_noaa_physical_rule_cap_bootstrap.json", s89["status"], "阶段88同一折/种子；11站点簇、5000次配对bootstrap；候选与无上限参考严格配对", f"F1差值={s89['summary']['f1_diff_mean']:.4f}; 95% CI=[{s89['summary']['f1_diff_ci95'][0]:.4f},{s89['summary']['f1_diff_ci95'][1]:.4f}]; FAR差值={s89['summary']['far_diff_mean']:.4f}; Recall差值={s89['summary']['recall_diff_mean']:.4f}", "支持低误报运行点候选，但Recall显著下降；需更多年份和自然观测复核")
    add(rows, "stage90_noaa_natural_physical_cap", "90", "stage90_noaa_natural_physical_cap.json", s90["status"], "2021–2025自然观测；历史年份训练、目标年份测试；预先列出Physical Rules上限网格；不使用目标年标签、不注入攻击", "自然报警率、质量可疑/正常弱标签差异、站点分层和物理分量位置/尺度已保存", "cap=1/2/3/5/10在当前实现中结果相同，未证明能修复自然观测零报警现象；自然报警率不等于攻击召回率")
    add(rows, "stage91_noaa_physical_component_repair", "91", "stage91_noaa_physical_component_repair.json", s91["status"], "四折滚动、11站点、5种子、六类透明受控攻击；组件方案预先固定；运行设置仅由valid选择；test只评价", f"component_zmax相对legacy：F1差值={s91['candidate_gain_vs_legacy']['f1']:.4f}; Recall差值={s91['candidate_gain_vs_legacy']['recall']:.4f}; AUPRC差值={s91['candidate_gain_vs_legacy']['auprc']:.4f}; FAR差值={s91['candidate_gain_vs_legacy']['far']:.4f}", "仅表现为低误报候选且Recall显著下降；验证集无法区分方案；尚未经过站点簇bootstrap，不得替换9B")
    add(rows, "stage92_noaa_physical_component_bootstrap", "92", "stage92_noaa_physical_component_bootstrap.json", s92["status"], "固定stage91 component_zmax；四折滚动、11站点、5种子；站点簇5000次配对bootstrap；阈值和融合设置仅由valid选择", f"F1差值={s92['summary']['f1_diff_mean']:.4f}; 95% CI=[{s92['summary']['f1_diff_ci95'][0]:.4f},{s92['summary']['f1_diff_ci95'][1]:.4f}]; Recall差值={s92['summary']['recall_diff_mean']:.4f}; FAR差值={s92['summary']['far_diff_mean']:.4f}", "低误报差异具有站点簇级稳定性，但Recall显著下降；仍需更多自然年份，不能替换9B主模型")
    add(rows, "stage93_noaa_natural_component_validation", "93", "stage93_noaa_natural_component_validation.json", s93["status"], "2021–2025自然观测；历史年份训练、目标年份测试；q网格预先固定；不注入攻击；质量标记仅作弱标签", "component_zmax自然报警率、质量可疑/正常分组报警率和站点分层结果已保存", "自然报警率不是攻击召回率；仍需站点簇不确定性分析，不能据此宣称普适")
    add(rows, "stage94_noaa_natural_component_bootstrap", "94", "stage94_noaa_natural_component_bootstrap.json", s94["status"], "阶段93站点明细；2021–2025、q=0.99/0.995/0.999；11站点5000次等权配对bootstrap；不使用目标年标签", "component_zmax相对legacy的自然报警率差异及质量弱标签对比差异的站点簇区间已保存", "自然报警率不是攻击召回率；弱标签子组样本不均衡，结果用于稳定性诊断，不替换主模型")
    add(rows, "stage95_noaa_physical_component_attack_type", "95", "stage95_noaa_physical_component_attack_type.json", s95["status"], "读取阶段91逐类型test明细；不重新调参或重跑test", "六类攻击逐类Recall及component_zmax−legacy差值已保存", "component_zmax在所有攻击类型上Recall均明显下降，只能作为低误报运行点候选")
    add(rows, "stage96_noaa_physical_component_operating_curve", "96", "stage96_noaa_physical_component_operating_curve.json", s96["status"], "四折滚动、11站点、5种子；融合设置由valid选择；阈值只由训练正常风险分位数q网格确定", "legacy与component_zmax在q=0.90/0.95/0.975/0.99/0.995/0.999下的F1、Recall、Precision、AUPRC、FAR曲线已保存", "运行曲线只描述安全运行权衡，不能根据test挑点替换9B")
    add(rows, "stage97_noaa_component_operating_bootstrap", "97", "stage97_noaa_component_operating_bootstrap.json", s97["status"], "固定q=0.90与0.975；四折滚动、11站点、5种子；站点簇5000次配对bootstrap；阈值只由训练正常风险分位数确定", f"q=0.975 F1差值={next(x['f1_diff_mean'] for x in s97['bootstrap_summary'] if x['q']==0.975):.4f}; 95% CI=[{next(x['f1_diff_ci95'] for x in s97['bootstrap_summary'] if x['q']==0.975)[0]:.4f},{next(x['f1_diff_ci95'] for x in s97['bootstrap_summary'] if x['q']==0.975)[1]:.4f}]; Recall差值={next(x['recall_diff_mean'] for x in s97['bootstrap_summary'] if x['q']==0.975):.4f}; FAR差值={next(x['far_diff_mean'] for x in s97['bootstrap_summary'] if x['q']==0.975):.4f}", "低误报运行点差异稳定，但Recall显著下降；只能作为安全运行点，不能替换9B主模型")
    add(rows, "stage98_noaa_low_far_baseline_comparison", "98", "stage98_noaa_low_far_baseline_comparison.json", s98["status"], "2020–2025 NOAA；四折滚动、11站点、5种子；Trust与Robust/Isolation Forest/One-Class SVM使用相同训练正常分位数q网格；阈值只由训练正常样本确定", "q=0.975下同时保存Trust legacy、component_zmax、Robust max deviation、Isolation Forest、One-Class SVM的F1/Recall/Precision/AUPRC/FAR及均值±标准差", "公平低误报比较显示One-Class SVM/Isolation Forest在部分折中点更有竞争力；component_zmax仅改善FAR，不能宣称全面领先或替换9B")
    s99_q = [x for x in s99["results"] if x["q"] == 0.975 and x["reference"] == "Trust component_zmax" and x["candidate"] == "One-Class SVM"]
    s99_f1 = next(x for x in s99_q if x["metric"] == "test_f1")
    s99_far = next(x for x in s99_q if x["metric"] == "test_far")
    add(rows, "stage99_noaa_baseline_paired_bootstrap", "99", "stage99_noaa_baseline_paired_bootstrap.json", s99["status"], "基于阶段98四折×五种子；按折层级和种子层级有放回配对bootstrap 5000次；q网格预先固定；不重新调参", f"q=0.975时SVM−component_zmax F1差值={s99_f1['observed_diff_mean']:.4f}; 95% CI=[{s99_f1['ci95_low']:.4f},{s99_f1['ci95_high']:.4f}]; FAR差值={s99_far['observed_diff_mean']:.4f}; 候选F1更优概率={s99_f1['prob_candidate_better']:.4f}", "这是折×种子层面的不确定性，不替代逐站点bootstrap；SVM综合指标更强但不能据此宣称所有场景普适")
    s100_f1 = next(x for x in s100["summary"] if x["candidate"] == "One-Class SVM" and x["reference"] == "Trust component_zmax" and x["metric"] == "f1")
    s100_far = next(x for x in s100["summary"] if x["candidate"] == "One-Class SVM" and x["reference"] == "Trust component_zmax" and x["metric"] == "far")
    s100_recall = next(x for x in s100["summary"] if x["candidate"] == "One-Class SVM" and x["reference"] == "Trust component_zmax" and x["metric"] == "recall")
    add(rows, "stage100_noaa_low_far_station_bootstrap", "100", "stage100_noaa_low_far_station_bootstrap.json", s100["status"], "NOAA 2020–2025；四折滚动、11站点、5种子；固定q=0.975；逐站点混淆计数；11站点簇5000次配对bootstrap", f"SVM−component_zmax：F1差值={s100_f1['observed_diff']:.4f}; 95% CI=[{s100_f1['ci95_low']:.4f},{s100_f1['ci95_high']:.4f}]; FAR差值={s100_far['observed_diff']:.4f}; Recall差值={s100_recall['observed_diff']:.4f}", "站点簇级证据支持SVM在q=0.975综合指标和FAR上的优势，但Recall显著下降；AUPRC不由计数重建，见阶段99")
    add(rows, "stage101_noaa_trust_svm_complementarity", "101", "stage101_noaa_trust_svm_complementarity.json", s101["status"], "NOAA 2020–2025；四折滚动、11站点、5种子；Trust/SVM先由train正常样本稳健标准化；预先固定均值权重和max网格；q=0.975固定", "12个候选同时保存F1、Recall、Precision、AUPRC、FAR；alpha=0.15略增Recall/AUPRC但F1下降，alpha=0.35 Recall最高但FAR和F1权衡恶化", "仅为互补性筛选，不得按test最高项固化权重；正式候选仍需valid选择和独立站点簇bootstrap")
    selected_counts = {row["method"]: row["selected_count"] for row in s102["selection_counts"]}
    add(rows, "stage102_noaa_trust_svm_valid_selection", "102", "stage102_noaa_trust_svm_valid_selection.json", s102["status"], "NOAA 2020–2025；四折滚动、11站点、5种子；q=0.975固定；每折每种子仅在valid按FAR≤10%约束最大化F1选权重；test只评价", f"20次选择中alpha=0.35占{selected_counts.get('trust_svm_mean_alpha_0.35', 0)}次、alpha=0.50占{selected_counts.get('trust_svm_mean_alpha_0.50', 0)}次、SVM单模型占{selected_counts.get('svm_only', 0)}次；选择后test完整指标已保存", "权重随时间折变化，不能固化为单一主模型；q=0.975是低误报运行点，不代表全局F1最优")
    s103_svm_f1 = next(x for x in s103["paired_differences"] if x["reference"] == "svm_only" and x["metric"] == "test_f1")
    s103_svm_recall = next(x for x in s103["paired_differences"] if x["reference"] == "svm_only" and x["metric"] == "test_recall")
    s103_svm_far = next(x for x in s103["paired_differences"] if x["reference"] == "svm_only" and x["metric"] == "test_far")
    add(rows, "stage103_noaa_valid_selection_stability", "103", "stage103_noaa_valid_selection_stability.json", s103["status"], "阶段102的20个折×种子选择块；选择频率和selected相对固定基线的配对bootstrap 5000次；不重新选模型", f"selected−SVM：F1差值={s103_svm_f1['paired_diff_mean']:.4f}; 95% CI=[{s103_svm_f1['ci95_low']:.4f},{s103_svm_f1['ci95_high']:.4f}]; Recall差值={s103_svm_recall['paired_diff_mean']:.4f}; FAR差值={s103_svm_far['paired_diff_mean']:.4f}", "融合相对SVM提高Recall但显著降低F1/Precision/AUPRC并提高FAR；仅支持召回增强方向，不形成全面优势")
    add(rows, "stage104_recent_artifact_integrity", "104", "stage104_recent_artifact_integrity.json", s104["status"], "独立检查阶段98–111产物存在性、行数、唯一键、指标完整性、候选集合一致性和源表SHA256", f"检查项={len(s104['checks'])}; 失败项={len(s104.get('failed_checks', []))}; 源表SHA256={s104['source_sha256']}", "完整性审计不能证明性能优越性；性能结论仍需回到阶段98–111原始结果")
    s105_selected = next(x for x in s105["overall_aggregate"] if x["method"] == "valid_selected")
    add(rows, "stage105_noaa_attack_type_valid_selected", "105", "stage105_noaa_attack_type_valid_selected.json", s105["status"], "NOAA 2020–2025；四折滚动、11站点、5种子；q=0.975；每折每种子仅由valid按FAR≤10%选权重；test按六类受控攻击统计Recall并保留固定SVM/Trust参照", f"valid_selected总体test F1={s105_selected['test_f1_mean']:.4f}±{s105_selected['test_f1_std']:.4f}; Recall={s105_selected['test_recall_mean']:.4f}; AUPRC={s105_selected['test_auprc_mean']:.4f}; FAR={s105_selected['test_far_mean']:.4f}; 六类攻击Recall明细已保存", "相对SVM多数攻击类型Recall提高但总体F1/AUPRC/FAR不占优；攻击为透明受控注入，不是真实攻击日志")
    s106_drift = next(x for x in s106["results"] if x["attack_type"] == "drift" and x["reference"] == "svm_only")
    s106_stuck = next(x for x in s106["results"] if x["attack_type"] == "stuck" and x["reference"] == "svm_only")
    add(rows, "stage106_noaa_attack_type_paired_bootstrap", "106", "stage106_noaa_attack_type_paired_bootstrap.json", s106["status"], "阶段105六类攻击Recall；四折×五种子层级配对bootstrap 5000次；不重新选权重", f"selected−SVM：drift差值={s106_drift['observed_diff']:.4f}; 95% CI=[{s106_drift['ci95_low']:.4f},{s106_drift['ci95_high']:.4f}]; stuck差值={s106_stuck['observed_diff']:.4f}; 95% CI=[{s106_stuck['ci95_low']:.4f},{s106_stuck['ci95_high']:.4f}]", "攻击类型增益具有选择性；stuck不占优，其他部分类型CI跨零；不能宣称攻击类型普适")
    add(rows, "stage107_noaa_natural_svm_validation", "107", "stage107_noaa_natural_svm_validation.json", s107["status"], "NOAA 2020–2025；历史年份训练、目标年份自然观测测试；Trust legacy/component_zmax/SVM；q=0.99/0.995/0.999；不注入攻击、不使用目标年标签", "5个目标年份、45条年度明细和495条站点明细已保存自然报警率与质量弱标签对比", "自然报警率不是攻击召回率；质量标记是弱标签，不能当攻击真值")
    s108_key = next(x for x in s108["results"] if x["target_year"] == 2023 and abs(x["q"] - 0.995) < 1e-9 and x["reference"] == "Trust component_zmax")
    add(rows, "stage108_noaa_natural_alarm_station_bootstrap", "108", "stage108_noaa_natural_alarm_station_bootstrap.json", s108["status"], "阶段107站点自然报警率；按目标年/q/站点5000次等权配对bootstrap；不使用攻击标签、不重新选阈值", f"2023 q=0.995时SVM−component自然报警率差值={s108_key['observed_diff']:.4f}; 95% CI=[{s108_key['ci95_low']:.4f},{s108_key['ci95_high']:.4f}]; SVM更低概率={s108_key['prob_candidate_lower']:.4f}", "仅支持自然报警率的站点稳定性，不支持攻击召回或真实攻击普适性")
    s109_key = next(x for x in s109["results"] if x["target_year"] == 2023 and abs(x["q"] - 0.995) < 1e-9 and x["method"] == "One-Class SVM")
    add(rows, "stage109_noaa_quality_weak_label_bootstrap", "109", "stage109_noaa_quality_weak_label_bootstrap.json", s109["status"], "阶段107质量可疑/正常自然记录；按目标年/q/方法做5000次等权站点bootstrap；不使用攻击标签、不重新选阈值", f"2023 q=0.995 One-Class SVM质量可疑−正常报警率差异={s109_key['contrast_mean']:+.4f}; 95% CI=[{s109_key['ci95_low']:+.4f},{s109_key['ci95_high']:+.4f}]; 差异为正概率={s109_key['prob_contrast_positive']:.4f}", "NOAA质量标记仅为弱标签；差异不能解释为攻击召回或真实攻击检测")
    s110_key = next(x for x in s110["results"] if abs(x["q"] - 0.995) < 1e-9 and x["reference"] == "Trust component_zmax")
    add(rows, "stage110_noaa_quality_hierarchical_bootstrap", "110", "stage110_noaa_quality_hierarchical_bootstrap.json", s110["status"], "阶段107质量弱标签对比；年份一级、站点二级；5000次层级bootstrap；不使用攻击标签、不重新选阈值", f"q=0.995 SVM−component质量对比差异={s110_key['observed_diff_in_contrast']:+.4f}; 95% CI=[{s110_key['ci95_low']:+.4f},{s110_key['ci95_high']:+.4f}]; SVM更高概率={s110_key['prob_candidate_more_quality_aligned']:.4f}", "层级结果只支持当前年份/站点集合的弱标签稳定性；质量标记不是攻击真值")
    s111_key = next(x for x in s111["results"] if abs(x["q"] - 0.995) < 1e-9 and x["method"] == "One-Class SVM" and x["min_flagged_rows"] == 5)
    add(rows, "stage111_noaa_quality_count_sensitivity", "111", "stage111_noaa_quality_count_sensitivity.json", s111["status"], "阶段107质量弱标签；预先固定可疑行数阈值1/3/5/10/20；年份一级、站点二级bootstrap并列行数加权；不使用攻击标签、不重新选阈值", f"q=0.995 SVM、最少5条可疑记录：站点等权差异={s111_key['station_equal_contrast']:+.4f}; 95% CI=[{s111_key['station_equal_ci95_low']:+.4f},{s111_key['station_equal_ci95_high']:+.4f}]; 行数加权差异={s111_key['row_weighted_contrast']:+.4f}", "小样本敏感性不是攻击真值；站点等权和行数加权必须并列报告")
    add(rows, "stage112_full_artifact_manifest", "112", "stage112_full_artifact_manifest.json", s112["status"], "扫描所有stage JSON/CSV/Parquet/报告；记录大小、SHA256和JSON状态；核对证据矩阵引用产物、证据ID唯一性及阶段109–111成对文件", f"证据矩阵条目={s112['evidence_rows']}; 纳入清单文件={s112['artifact_files']}; 缺失={len(s112['missing_evidence_artifacts'])}; 重复证据ID={len(s112['duplicate_evidence_ids'])}", "清单只证明可追溯性和文件完整性，不证明性能、统计显著性或真实攻击能力")
    add(rows, "stage113_era5_alignment_leakage_audit", "113", "stage113_era5_alignment_leakage_audit.json", s113["status"], "阶段07/08 ERA5对齐与偏差订正；检查站点—小时唯一性、有限值、网格距离、自然/受控小时分层，并静态核对valid正常样本拟合边界", f"检查项={len(s113['checks'])}; 失败项={sum(not value for value in s113['checks'].values())}; 对齐行数={s113['alignment_rows']}; 订正明细={s113['bias_rows']}; 订正后自然小时MAE={s113['metrics'][1]['corrected_mae_c']:.4f}°C", "ERA5是区域背景而非现场真值；对齐和订正审计不等于攻击检测或恢复性能")
    add(rows, "stage114_recent_semantic_sanity_audit", "114", "stage114_recent_semantic_sanity_audit.json", s114["status"], "阶段98–113近期JSON和表格；检查比例/概率范围、有限值、置信区间顺序、预期行数和状态；允许差值/增益为负及无分母时的弱标签缺失值", f"检查项={len(s114['checks'])}; 失败项={len(s114['failed_checks'])}; 比例与结构检查全部通过", "语义审计只能发现明显数值/结构异常，不能证明性能优越性；质量弱标签缺失值需结合分母解释")
    add(rows, "stage115_protocol_boundary_audit", "115", "stage115_protocol_boundary_audit.json", s115["status"], "静态核对受控性能实验的train/valid/test边界，以及自然观测、弱标签和ERA5辅助变量的专用边界契约", f"检查项={len(s115['checks'])}; 失败项={len(s115['failed_checks'])}; 协议边界检查通过", "静态关键词和函数块检查不能替代人工信息流复核；通过不代表性能领先或标签真实")
    s116_key = next(x for x in s116["aggregate_results"] if abs(x["q"] - 0.995) < 1e-9 and x["method"] == "Trust component_zmax")
    add(rows, "stage116_noaa_quality_label_permutation", "116", "stage116_noaa_quality_label_permutation.json", s116["status"], "在站点×年份内固定报警总数和质量分组大小，10000次条件随机化；逐年45项和跨年9项分别做Benjamini–Hochberg校正", f"q=0.995 component_zmax观察差异={s116_key['observed_year_equal_contrast']:+.4f}; 单侧p={s116_key['permutation_p_one_sided']:.5f}; BH q={s116_key['bh_q_one_sided_9']:.5f}; 9项中校正后显著数={sum(x['bh_q_one_sided_9'] < 0.05 for x in s116['aggregate_results'])}", "质量字段仅作弱标签；条件随机化支持统计关联而非攻击真值、因果关系或真实攻击召回")
    s117_f1 = next(x for x in s117["results"] if x["metric"] == "test_f1")
    add(rows, "stage117_multiseed_uncertainty", "117", "stage117_multiseed_uncertainty.json", s117["status"], "读取阶段9C五个预先固定受控攻击种子；10000次种子级bootstrap和留一攻击种子汇总；不重新选模型、权重或阈值", f"test F1均值={s117_f1['observed_mean']:.4f}; bootstrap均值95%区间=[{s117_f1['bootstrap_mean_ci95_low']:.4f},{s117_f1['bootstrap_mean_ci95_high']:.4f}]; 留一均值范围=[{s117_f1['leave_one_seed_min']:.4f},{s117_f1['leave_one_seed_max']:.4f}]", "仅5个受控攻击种子；区间不代表跨站点、跨年份或真实攻击泛化")
    s118_d4 = next(x for x in s118["duration_aggregate"] if x["duration"] == 4)
    s118_d12 = next(x for x in s118["duration_aggregate"] if x["duration"] == 12)
    add(rows, "stage118_attack_duration_sensitivity", "118", "stage118_attack_duration_sensitivity.json", s118["status"], "固定阶段9C五个种子；扩展攻击事件按站点截取4/8/12个观测点；每个变体重新训练和valid选择，test只评价", f"F1均值：时长4={s118_d4['test_f1_mean']:.4f}、时长12={s118_d12['test_f1_mean']:.4f}; Recall：{s118_d4['test_recall_mean']:.4f}→{s118_d12['test_recall_mean']:.4f}; 正常误伤率：{s118_d4['test_false_alarm_rate_mean']:.4f}→{s118_d12['test_false_alarm_rate_mean']:.4f}", "事件持续时间只覆盖4/8/12点；结果显示F1、Recall与误伤率存在权衡，不支持普适最优持续时间")
    s119_m24 = next(x for x in s119["shift_aggregate"] if x["shift_rows"] == -24)
    s119_0 = next(x for x in s119["shift_aggregate"] if x["shift_rows"] == 0)
    s119_p24 = next(x for x in s119["shift_aggregate"] if x["shift_rows"] == 24)
    add(rows, "stage119_attack_position_sensitivity", "119", "stage119_attack_position_sensitivity.json", s119["status"], "固定阶段9C五个种子；同一站点同一split内将扩展攻击事件平移−24/0/+24个观测行；每个变体重新训练和valid选择，test只评价", f"F1均值：−24={s119_m24['test_f1_mean']:.4f}、0={s119_0['test_f1_mean']:.4f}、+24={s119_p24['test_f1_mean']:.4f}; Recall：{s119_m24['test_recall_mean']:.4f}/{s119_0['test_recall_mean']:.4f}/{s119_p24['test_recall_mean']:.4f}; fallback事件记录在明细表", "位置只覆盖±24观测行；结果对时间位置敏感，固定注入位置不能代表普适攻击分布")
    s120_weak = min(s120["station_aggregate"], key=lambda x: x["recall_mean"])
    s120_strong = max(s120["station_aggregate"], key=lambda x: x["recall_mean"])
    s120_seed = next(x for x in s120["seed_summary"] if x["seed"] == 3407)
    add(rows, "stage120_station_stratified_performance", "120", "stage120_station_stratified_performance.json", s120["status"], "读取阶段9C五个种子的locked test逐行得分；按站点报告F1、Recall、Precision、AUPRC、FAR和事件召回，并同时给出宏/微平均；不重新训练或选参", f"7站点×5种子={len(s120['station_detail'])}; 站点Recall均值范围={s120_weak['recall_mean']:.4f}–{s120_strong['recall_mean']:.4f}; seed3407宏F1={s120_seed['macro_f1']:.4f}、微F1={s120_seed['micro_f1']:.4f}", "站点异质性明显时不能只报告微平均；当前仅7个站点，攻击仍为受控注入")
    s121_gain = s121["gains_vs_baseline"]
    add(rows, "stage121_local_station_threshold_calibration", "121", "stage121_local_station_threshold_calibration.json", s121["status"], "站点阈值由各自train正常样本估计，q和收缩比例只在valid按站点宏F1选择，locked test只评价；不改变主模型、权重或攻击构造", f"候选数={s121['candidate_count']}; test微F1差值={s121_gain['micro_f1']:+.4f}; 宏F1差值={s121_gain['macro_f1']:+.4f}; 最弱站点Recall差值={s121_gain['worst_station_recall']:+.4f}", "增益很小且伴随Recall/FAR权衡；当前仅本地7站点，必须继续留一站点和外部NOAA复核，不能替换9B主模型")
    s122_strict = s122["comparison"]["strict_leave_one_station"]
    s122_adapted = s122["comparison"]["target_history_adapted"]
    add(rows, "stage122_leave_one_station_threshold_transfer", "122", "stage122_leave_one_station_threshold_transfer.json", s122["status"], "每个种子逐一留出站点；q仅由源站点train/valid选择；严格迁移不使用留出站点train，目标历史无标签适配单独报告，test只评价", f"严格迁移F1={s122_strict['f1']:.4f}、Recall={s122_strict['recall']:.4f}、FAR={s122_strict['far']:.4f}; 历史适配F1={s122_adapted['f1']:.4f}、Recall={s122_adapted['recall']:.4f}、FAR={s122_adapted['far']:.4f}", "严格迁移未优于基线；适配收益跨站点方向不一致且不是完整检测器冷启动，不能合并为普适提升")
    s123_strict_f1 = next(x for x in s123["results"] if x["method"] == "strict" and x["reference"] == "baseline" and x["metric"] == "f1")
    s123_adapted_f1 = next(x for x in s123["results"] if x["method"] == "adapted" and x["reference"] == "baseline" and x["metric"] == "f1")
    s123_strict_source_f1 = next(x for x in s123["results"] if x["method"] == "strict" and x["reference"] == "source_baseline" and x["metric"] == "f1")
    s123_adapted_source_f1 = next(x for x in s123["results"] if x["method"] == "adapted" and x["reference"] == "source_baseline" and x["metric"] == "f1")
    add(rows, "stage123_threshold_transfer_bootstrap", "123", "stage123_threshold_transfer_bootstrap.json", s123["status"], "阶段122每个留出站点×种子的locked test结果先按站点聚合，再对7个站点簇做10000次配对bootstrap；不重新选q或训练；同时相对全站点和源站点valid-only基线", f"严格迁移相对全站点基线F1差值={s123_strict_f1['observed_diff']:+.4f}; CI=[{s123_strict_f1['ci95_low']:+.4f},{s123_strict_f1['ci95_high']:+.4f}]; 适配相对全站点基线={s123_adapted_f1['observed_diff']:+.4f}; 源站点valid-only基线下严格迁移={s123_strict_source_f1['observed_diff']:+.4f}、适配={s123_adapted_source_f1['observed_diff']:+.4f}", "仅7个站点簇且区间跨零；适配只能作为候选，不支持稳定跨站点提升")
    s124_f1 = s124["summary"]["test_f1"]
    s124_far = s124["summary"]["test_false_alarm_rate"]
    s125_seed_f1 = next(x for x in s125["results"] if x["cluster_level"] == "seed" and x["metric"] == "test_f1")
    s125_station_f1 = next(x for x in s125["results"] if x["cluster_level"] == "station" and x["metric"] == "test_f1")
    add(rows, "stage124_leave_one_station_refit_multiseed", "124", "stage124_leave_one_station_refit_multiseed.json", s124["status"], "阶段9C五个固定攻击构造种子分别执行阶段22严格留一站点重拟合；每个留出站点不参与拟合或valid选择，test只评价", f"35个组合；F1均值={s124_f1['mean']:.4f}、标准差={s124_f1['std']:.4f}; Recall均值={s124['summary']['test_recall']['mean']:.4f}; FAR均值={s124_far['mean']:.4f}; 恢复MAE={s124['summary']['recovery_mae_before']['mean']:.4f}→{s124['summary']['recovery_mae_after']['mean']:.4f}", "站点差异和误伤率波动很大；受控攻击种子不等于独立地区/年份，不能宣称普适性")
    add(rows, "stage125_leave_one_station_refit_multiseed_bootstrap", "125", "stage125_leave_one_station_refit_multiseed_bootstrap.json", s125["status"], "阶段124结果先按5个种子或7个留出站点聚合，再分别做10000次均值bootstrap；不重新训练或选择参数", f"F1种子簇95% CI=[{s125_seed_f1['ci95_low']:.4f},{s125_seed_f1['ci95_high']:.4f}]; F1站点簇95% CI=[{s125_station_f1['ci95_low']:.4f},{s125_station_f1['ci95_high']:.4f}]", "簇数很少且区间只描述当前种子/站点集合不确定性；高误伤站点和恢复退化必须并列报告")
    s126_5 = next(x for x in s126["cap_summary"] if abs(x["far_cap"] - 0.05) < 1e-12)
    add(rows, "stage126_low_far_leave_one_station_operating_points", "126", "stage126_low_far_leave_one_station_operating_points.json", s126["status"], "仅用源站点valid风险和标签按宏F1及FAR上限1%/2%/5%/10%选阈值，留出站点locked test只评价；阶段124 F1最优阈值作为基线", f"5%运行点test F1={s126_5['test_f1_mean']:.4f}; Recall={s126_5['test_recall_mean']:.4f}; FAR={s126_5['test_far_mean']:.4f}; 相对基线F1={s126_5['f1_diff_mean']:+.4f}; FAR={s126_5['far_diff_mean']:+.4f}", "低误报运行点伴随Recall权衡，留出站点FAR可能超出valid上限；不替换F1最优主结果")
    s127_5_f1 = next(x for x in s127["results"] if abs(x["far_cap"] - 0.05) < 1e-12 and x["metric"] == "f1")
    s127_5_recall = next(x for x in s127["results"] if abs(x["far_cap"] - 0.05) < 1e-12 and x["metric"] == "recall")
    add(rows, "stage127_low_far_operating_point_bootstrap", "127", "stage127_low_far_operating_point_bootstrap.json", s127["status"], "阶段126每个FAR运行点的5种子×7站点locked test结果先按站点聚合，再做10000次站点簇配对bootstrap；不重新选阈值", f"5%运行点F1差值={s127_5_f1['observed_diff']:+.4f}; 95% CI=[{s127_5_f1['ci95_low']:+.4f},{s127_5_f1['ci95_high']:+.4f}]; Recall差值={s127_5_recall['observed_diff']:+.4f}; CI=[{s127_5_recall['ci95_low']:+.4f},{s127_5_recall['ci95_high']:+.4f}]", "F1区间跨零且Recall下降；该证据支持安全运行点/召回权衡，不支持全面提升")
    s128_temporal = next(x for x in s128["candidate_summary"] if x["candidate"] == "temporal_candidate")
    s128_recovered = next(x for x in s128["candidate_summary"] if x["candidate"] == "recovered_value")
    add(rows, "stage128_recovery_candidate_diagnosis", "128", "stage128_recovery_candidate_diagnosis.json", s128["status"], "只读取阶段124 locked test逐行结果做事后候选诊断；不重新选择恢复权重、不改变检测预测；按报警攻击行和误报警正常行分层", f"时间候选攻击MAE={s128_temporal['attack_mae_mean']:.4f}、改善率={s128_temporal['attack_improvement_rate_mean']:.4f}; 当前恢复攻击MAE={s128_recovered['attack_mae_mean']:.4f}; 当前恢复正常误报警改写恶化率={s128_recovered['clean_worsening_rate_mean']:.4f}", "test事后诊断不能用于调参；候选效果不等同于真实故障修复；误报警改写伤害必须保留")
    s129_unconstrained = next(x for x in s129["aggregate"] if x["group"] == "unconstrained_selected_by_valid")
    s129_safety = next(x for x in s129["aggregate"] if x["group"] == "safety_constrained_selected_by_valid")
    add(rows, "stage129_valid_recovery_strategy_selection", "129", "stage129_valid_recovery_strategy_selection.json", s129["status"], "候选策略和风险门控仅由每个种子×留出站点的源valid utility选择；安全约束方案另以valid正常误报警改写恶化率≤20%筛选；locked test只评价", f"无约束选择test utility={s129_unconstrained['test_utility_mean']:.4f}、正常改写恶化率={s129_unconstrained['test_clean_harm_rate_mean']:.4f}; 安全约束选择utility={s129_safety['test_utility_mean']:.4f}、攻击MAE后={s129_safety['test_attack_mae_after_mean']:.4f}、正常改写恶化率={s129_safety['test_clean_harm_rate_mean']:.4f}", "安全约束显著降低误报警改写伤害但牺牲攻击恢复效用；受控攻击处置结果需跨站点和外部数据复核")
    s130_safety = next(x for x in s130["aggregate"] if x["group"] == "safety_constrained_selected_by_valid")
    s130_legacy = next(x for x in s130["aggregate"] if x["group"] == "legacy_ungated_reference")
    add(rows, "stage130_noaa_safety_constrained_recovery", "130", "stage130_noaa_safety_constrained_recovery.json", s130["status"], "NOAA五个固定受控攻击种子；候选和门控仅由valid选择，并以valid正常误报警改写恶化率≤20%执行安全约束；locked test只评价；不使用ERA5候选", f"安全约束选择test utility={s130_safety['test_utility_mean']:.4f}、攻击MAE后={s130_safety['test_attack_mae_after_mean']:.4f}、正常改写恶化率={s130_safety['test_clean_harm_rate_mean']:.4f}; legacy未门控utility={s130_legacy['test_utility_mean']:.4f}", "五个种子均选择no_recovery，外部结果保留为负结果；攻击为受控注入，不能替代真实运维故障日志")
    s131_local = next(x for x in s131["aggregate"] if x["dataset"] == "local_leave_one_station" and abs(x["max_valid_clean_harm_rate"] - 0.20) < 1e-12)
    s131_noaa = next(x for x in s131["aggregate"] if x["dataset"] == "NOAA_external" and abs(x["max_valid_clean_harm_rate"] - 0.20) < 1e-12)
    add(rows, "stage131_safety_constraint_sensitivity", "131", "stage131_safety_constraint_sensitivity.json", s131["status"], "复用阶段129本地和阶段130 NOAA全候选valid/test明细；预先固定0/5/10/20/30/50/100%正常误报警改写恶化率上限；valid重新选择，test只评价，不重新训练", f"20%上限本地test utility={s131_local['test_utility_mean']:.4f}、正常改写恶化率={s131_local['test_clean_harm_rate_mean']:.4f}; NOAA test utility={s131_noaa['test_utility_mean']:.4f}、选择no_recovery={s131_noaa['selected_no_recovery']}/{s131_noaa['combinations']}", "敏感性分析复用已有受控候选，不增加独立攻击样本；不证明20%或其他阈值普适最优")
    s132_local_safe = next(x for x in s132["results"] if x["dataset"] == "local_leave_one_station" and x["selection_policy"] == "valid_clean_harm_rate_le_0.20" and x["metric"] == "test_clean_harm_rate")
    s132_local_mae = next(x for x in s132["results"] if x["dataset"] == "local_leave_one_station" and x["selection_policy"] == "valid_clean_harm_rate_le_0.20" and x["metric"] == "test_attack_mae_after")
    s132_noaa_safe = next(x for x in s132["results"] if x["dataset"] == "NOAA_external" and x["selection_policy"] == "valid_clean_harm_rate_le_0.20" and x["metric"] == "test_utility")
    add(rows, "stage132_recovery_utility_paired_bootstrap", "132", "stage132_recovery_utility_paired_bootstrap.json", s132["status"], "阶段129/130已选择结果相对legacy未门控参考；本地按7个留出站点簇、NOAA按5个固定种子簇执行10000次配对bootstrap；test只评价差值，不重新选择策略", f"本地安全约束正常改写伤害差值={s132_local_safe['observed_diff']:+.4f}、95%CI=[{s132_local_safe['ci95_low']:+.4f},{s132_local_safe['ci95_high']:+.4f}]; 本地攻击MAE差值={s132_local_mae['observed_diff']:+.4f}、CI=[{s132_local_mae['ci95_low']:+.4f},{s132_local_mae['ci95_high']:+.4f}]; NOAA安全方案utility差值={s132_noaa_safe['observed_diff']:+.4f}", "本地站点簇和NOAA种子簇数量有限；NOAA相对legacy的优势主要来自no_recovery，不能写成恢复算法增益")
    s133_offline = next(x for x in s133["candidate_summary"] if x["candidate"] == "offline_temporal_candidate")
    s133_causal = next(x for x in s133["candidate_summary"] if x["candidate"] == "causal_temporal_candidate")
    add(rows, "stage133_causal_temporal_recovery_audit", "133", "stage133_causal_temporal_recovery.json", s133["status"], "阶段124五种子×七站点locked test；使用同一组合valid-only未报警中位数作为因果回退值；双向插值与只看过去候选仅做test事后诊断，不重新选择检测或恢复权重", f"离线时间候选攻击MAE={s133_offline['attack_mae_mean']:.4f}、改善率={s133_offline['attack_improvement_rate_mean']:.4f}; 因果候选攻击MAE={s133_causal['attack_mae_mean']:.4f}、改善率={s133_causal['attack_improvement_rate_mean']:.4f}; 因果正常改写恶化率={s133_causal['clean_worsening_rate_mean']:.4f}", "双向插值只能解释为离线修复；因果候选性能下降且仍有误报警改写风险，不能宣称在线能力")
    s134_safe = next(x for x in s134["aggregate"] if x["group"] == "safety_constrained_selected_by_valid")
    s134_unconstrained = next(x for x in s134["aggregate"] if x["group"] == "unconstrained_selected_by_valid")
    add(rows, "stage134_causal_recovery_optimization", "134", "stage134_causal_recovery_optimization.json", s134["status"], "阶段124五种子×七站点；causal_ffill/causal_slope/causal_blend只使用过去未报警观测，趋势边界和回退值由valid估计；候选/门控仅由valid选择，test只评价", f"安全约束test utility={s134_safe['test_utility_mean']:.4f}、攻击MAE后={s134_safe['test_attack_mae_after_mean']:.4f}、正常改写恶化率={s134_safe['test_clean_harm_rate_mean']:.4f}; 无约束攻击MAE后={s134_unconstrained['test_attack_mae_after_mean']:.4f}、正常改写恶化率={s134_unconstrained['test_clean_harm_rate_mean']:.4f}", "因果方案仍弱于离线双向插值；无约束方案误报警改写风险高，不能宣称在线全面领先")
    s135_f1 = s135["metrics"]["test_f1"]
    s135_recall = s135["metrics"]["test_recall"]
    add(rows, "stage135_independent_seed_replication", "135", "stage135_independent_seed_replication.json", s135["status"], "预先冻结五个新攻击构造种子；沿用阶段9B完整流程，每个种子仅由自身valid选择融合设置和阈值，test只评价，不根据test重新选择", f"新种子test F1={s135_f1['mean']:.4f}±{s135_f1['std']:.4f}; 95% bootstrap CI=[{s135_f1['ci95_low']:.4f},{s135_f1['ci95_high']:.4f}]; Recall={s135_recall['mean']:.4f}±{s135_recall['std']:.4f}; 正常误伤率={s135['metrics']['test_false_alarm_rate']['mean']:.4f}; 恢复后MAE={s135['metrics']['recovery_mae_after']['mean']:.4f}", "仍是同一气象数据源和同一受控攻击生成器；支持流程复现和种子不确定性，不等于跨地区/年份或真实攻击普适性")
    s136_f1 = next(x for x in s136["metrics"] if x["metric"] == "test_f1")
    s136_far = next(x for x in s136["metrics"] if x["metric"] == "test_false_alarm_rate")
    add(rows, "stage136_seed_group_consistency", "136", "stage136_seed_group_consistency.json", s136["status"], "比较阶段9C原五种子与阶段135新五种子；独立两组bootstrap和精确置换检验，七项p值做BH校正；不据此调参或回写锁定测试", f"F1新−原差值={s136_f1['observed_diff_replication_minus_original']:+.4f}; bootstrap CI=[{s136_f1['bootstrap_ci95_low']:+.4f},{s136_f1['bootstrap_ci95_high']:+.4f}]; BH p={s136_f1['bh_adjusted_p']:.4f}; FAR差值={s136_far['observed_diff_replication_minus_original']:+.4f}", "两组各5个种子导致检验功效有限；CI跨零只能说明未发现明显漂移，不能证明统计等价或跨域普适")
    s137_f1 = next(x for x in s137["comparisons"] if x["metric"] == "test_f1")
    s137_auprc = next(x for x in s137["comparisons"] if x["metric"] == "test_auprc")
    s137_far = next(x for x in s137["comparisons"] if x["metric"] == "test_false_alarm_rate")
    add(rows, "stage137_enhanced_candidate_comparison", "137", "stage137_enhanced_candidate_comparison.json", s137["status"], "阶段135五个新种子上成对比较预先存在的default与--enhanced候选；两种方案各自仅由valid选择，locked test只比较，不根据test调参", f"enhanced−default F1差值={s137_f1['observed_diff']:+.4f}; bootstrap CI=[{s137_f1['bootstrap_ci95_low']:+.4f},{s137_f1['bootstrap_ci95_high']:+.4f}]; AUPRC差值={s137_auprc['observed_diff']:+.4f}; FAR差值={s137_far['observed_diff']:+.4f}", "增强候选在新种子上未形成综合增益并提高误报，作为负结果保留，不替换default主模型")
    s138_rows = {x["method"]: x for x in s138["aggregate"]}
    add(rows, "stage138_independent_baseline_comparison", "138", "stage138_independent_baseline_comparison.json", s138["status"], "阶段135五个新种子；Trust、Robust max deviation、Isolation Forest和One-Class SVM使用相同输入与split；基线只用train正常样本拟合、valid选阈值、test评价", f"Trust F1={s138_rows['Stage09B robust fusion']['test_f1_mean']:.4f}; Isolation Forest F1={s138_rows['Isolation Forest']['test_f1_mean']:.4f}; SVM FAR={s138_rows['One-Class SVM']['test_false_alarm_rate_mean']:.4f}; Trust AUPRC={s138_rows['Stage09B robust fusion']['test_auprc_mean']:.4f}", "Isolation Forest在F1/Recall上更强，SVM在Precision/FAR上更强；Trust仅在部分指标占优，必须保留逐指标权衡")
    s139_f1 = next(x for x in s139["paired"] if x["metric"] == "test_f1")
    s139_auprc = next(x for x in s139["paired"] if x["metric"] == "test_auprc")
    s139_far = next(x for x in s139["paired"] if x["metric"] == "test_false_alarm_rate")
    add(rows, "stage139_expanded_fusion_grid", "139", "stage139_expanded_fusion_grid.json", s139["status"], "在阶段135五个新种子上扩展预先固定的alpha/window网格；每个种子仅由valid最大化F1，test只评价，不根据test重新选择；default为冻结参考", f"扩展−default F1差值={s139_f1['expanded_minus_default_mean']:+.4f}; bootstrap CI=[{s139_f1['bootstrap_ci95_low']:+.4f},{s139_f1['bootstrap_ci95_high']:+.4f}]; AUPRC差值={s139_auprc['expanded_minus_default_mean']:+.4f}; FAR差值={s139_far['expanded_minus_default_mean']:+.4f}", "扩展网格提高F1/AUPRC但提高FAR；这是同一数据源上的候选信号，仍需独立种子或外部数据确认，不能直接替换default")
    s140_f1 = next(x for x in s140["paired"] if x["metric"] == "test_f1")
    s140_auprc = next(x for x in s140["paired"] if x["metric"] == "test_auprc")
    s140_far = next(x for x in s140["paired"] if x["metric"] == "test_false_alarm_rate")
    add(rows, "stage140_expanded_grid_independent_confirmation", "140", "stage140_expanded_grid_independent_confirmation.json", s140["status"], "阶段139完成后冻结alpha/window网格和valid选择规则，在第三组五个新种子上仅运行一次；test只评价，不根据test重新选择或扩大网格", f"冻结扩展−default F1差值={s140_f1['expanded_minus_default_mean']:+.4f}; bootstrap CI=[{s140_f1['bootstrap_ci95_low']:+.4f},{s140_f1['bootstrap_ci95_high']:+.4f}]; AUPRC差值={s140_auprc['expanded_minus_default_mean']:+.4f}; FAR差值={s140_far['expanded_minus_default_mean']:+.4f}", "第三组种子支持候选增益且FAR未明显变化，但仍来自同一数据源和受控攻击生成器，需外部数据验证")
    s141_rows = {x["method"]: x for x in s141["aggregate"]}
    s141_f1 = next(x for x in s141["paired"] if x["comparison"] == "expanded - TrustMeteo-QC default" and x["metric"] == "test_f1")
    s141_auprc = next(x for x in s141["paired"] if x["comparison"] == "expanded - TrustMeteo-QC default" and x["metric"] == "test_auprc")
    s141_far = next(x for x in s141["paired"] if x["comparison"] == "expanded - TrustMeteo-QC default" and x["metric"] == "test_false_alarm_rate")
    add(rows, "stage141_noaa_frozen_external_validation", "141", "stage141_noaa_frozen_external_validation.json", s141["status"], "NOAA四折滚动、多年度、五种子；扩展alpha/window在外部运行前冻结，每折仅由valid选择，test只评价；Robust/Isolation Forest/One-Class SVM复用同协议阶段85结果", f"扩展候选test F1={s141_rows['TrustMeteo-QC frozen expanded']['test_f1_mean']:.4f}、AUPRC={s141_rows['TrustMeteo-QC frozen expanded']['test_auprc_mean']:.4f}、FAR={s141_rows['TrustMeteo-QC frozen expanded']['test_false_alarm_rate_mean']:.4f}; 相对default F1差值={s141_f1['mean_diff']:+.4f} CI=[{s141_f1['bootstrap_cluster_ci95_low']:+.4f},{s141_f1['bootstrap_cluster_ci95_high']:+.4f}]; AUPRC差值={s141_auprc['mean_diff']:+.4f}; FAR差值={s141_far['mean_diff']:+.4f}; SVM F1={s141_rows['One-Class SVM']['test_f1_mean']:.4f}", "NOAA观测为真实外部气象数据但攻击为透明受控注入；扩展候选提高F1/Precision并降低FAR，却仍低于One-Class SVM的F1/AUPRC，不能宣称全面领先或真实攻击能力")

    s142_rows = {x["method"]: x for x in s142["aggregate"] if int(x["target_year"]) == 2025 and abs(float(x["q"]) - 0.995) < 1e-9}
    s142_exp = s142_rows["TrustMeteo-QC frozen expanded"]
    s142_svm = s142_rows["One-Class SVM"]
    s142_quality = next(x for x in s142["weak_label_paired"] if x["comparison"] == "expanded - One-Class SVM" and x["metric"] == "quality_clean_alarm_rate")
    qlow = s142_quality.get("bootstrap_station_ci95_low")
    qhigh = s142_quality.get("bootstrap_station_ci95_high")
    quality_ci = "NA" if qlow is None or qhigh is None or pd.isna(qlow) or pd.isna(qhigh) else f"[{qlow:.4f},{qhigh:.4f}]"
    add(rows, "stage142_noaa_frozen_natural_operational_validation", "142", "stage142_noaa_frozen_natural_operational_validation.json", s142["status"], "NOAA 2020–2025自然观测；每个目标年仅用历史正常观测估计尺度；不注入攻击、不使用攻击真值；阶段141 alpha=0.25/window=1冻结规则只做无标签运行投影；质量字段仅作弱标签", f"2025 q=.995扩展候选自然报警率={s142_exp['target_alarm_rate']:.4f}、质量正常报警率={s142_exp['quality_clean_alarm_rate']:.4f}; 同运行点SVM自然报警率={s142_svm['target_alarm_rate']:.4f}; 扩展−SVM质量正常报警率差值={s142_quality['mean_diff']:+.4f}、站点bootstrap CI={quality_ci}", "自然报警率不是攻击召回率；质量标记不是攻击真值；扩展候选自然运行投影不能替代受控性能或真实攻击日志，部分质量弱标签对比因无可疑行而不可估计")
    s143_f1 = next(x for x in s143["paired"] if x["comparison"] == "valid-selected - One-Class SVM" and x["metric"] == "f1")
    s143_recall = next(x for x in s143["paired"] if x["comparison"] == "valid-selected - One-Class SVM" and x["metric"] == "recall")
    s143_far = next(x for x in s143["paired"] if x["comparison"] == "valid-selected - One-Class SVM" and x["metric"] == "far")
    s143_counts = {x["method"]: x["selected_count"] for x in s143["selection_counts"]}
    add(rows, "stage143_noaa_frozen_trust_svm_safety_combination", "143", "stage143_noaa_frozen_trust_svm_safety_combination.json", s143["status"], "NOAA 2020–2025四折滚动、五种子；冻结Trust/SVM候选和q=0.975、valid FAR≤10%约束；每折仅由valid选择，test只评价，不根据test重新选择", f"valid-selected相对SVM F1差值={s143_f1['mean_diff']:+.4f} CI=[{s143_f1['bootstrap_fold_ci95_low']:+.4f},{s143_f1['bootstrap_fold_ci95_high']:+.4f}]; Recall差值={s143_recall['mean_diff']:+.4f}; FAR差值={s143_far['mean_diff']:+.4f}; 选择次数alpha=.25={s143_counts.get('Trust-SVM mean alpha 0.25', 0)}、SVM={s143_counts.get('One-Class SVM', 0)}", "组合只在当前NOAA滚动受控注入协议中验证；SVM仍在Recall等指标占优或组合区间跨零时必须保留负结果，不能宣称全面领先")
    s144_f1 = next(x for x in s144["paired"] if x["metric"] == "f1")
    s144_recall = next(x for x in s144["paired"] if x["metric"] == "recall")
    s144_far = next(x for x in s144["paired"] if x["metric"] == "far")
    s144_counts = {x["method"]: x["fold_count"] for x in s144["selection_counts"]}
    add(rows, "stage144_noaa_cross_fold_candidate_transfer", "144", "stage144_noaa_cross_fold_candidate_transfer.json", s144["status"], "每次留出一个滚动折；仅用其余三折valid选择固定候选；留出折test只评价，不使用留出折test做选择、不根据test重新选择", f"跨折valid-selected相对SVM F1差值={s144_f1['mean_diff']:+.4f} CI=[{s144_f1['bootstrap_fold_ci95_low']:+.4f},{s144_f1['bootstrap_fold_ci95_high']:+.4f}]; Recall差值={s144_recall['mean_diff']:+.4f}; FAR差值={s144_far['mean_diff']:+.4f}; alpha=.25跨折选择={s144_counts.get('Trust-SVM mean alpha 0.25', 0)}/{len(s144['folds'])}", "跨折迁移只显示低误报和Precision权衡信号，F1区间跨零且Recall下降；不能宣称组合普适领先")
    s145_f1 = next(x for x in s145["paired"] if x["metric"] == "f1")
    s145_recall = next(x for x in s145["paired"] if x["metric"] == "recall")
    s145_far = next(x for x in s145["paired"] if x["metric"] == "far")
    add(rows, "stage145_noaa_station_heterogeneity_audit", "145", "stage145_noaa_station_heterogeneity_audit.json", s145["status"], "复用阶段141冻结外部test站点明细；按11站点分层并做站点簇bootstrap；不重新选择候选、不重新训练、不根据站点test调参", f"扩展−default F1差值={s145_f1['mean_diff']:+.4f} CI=[{s145_f1['station_bootstrap_ci95_low']:+.4f},{s145_f1['station_bootstrap_ci95_high']:+.4f}]; Recall差值={s145_recall['mean_diff']:+.4f} CI=[{s145_recall['station_bootstrap_ci95_low']:+.4f},{s145_recall['station_bootstrap_ci95_high']:+.4f}]; FAR差值={s145_far['mean_diff']:+.4f} CI=[{s145_far['station_bootstrap_ci95_low']:+.4f},{s145_far['station_bootstrap_ci95_high']:+.4f}]", "站点异质性审计显示F1/Precision和FAR改善伴随多数站点Recall下降；57036099999为弱站点，不能宣称跨地区普适或全面提升")
    s146_f1 = next(x for x in s146["paired"] if x["comparison"] == "expanded - default" and x["metric"] == "f1")
    s146_recall = next(x for x in s146["paired"] if x["comparison"] == "expanded - default" and x["metric"] == "recall")
    s146_auprc = next(x for x in s146["paired"] if x["comparison"] == "expanded - default" and x["metric"] == "auprc")
    s146_svm_f1 = next(x for x in s146["paired"] if x["comparison"] == "expanded - One-Class SVM" and x["metric"] == "f1")
    add(rows, "stage146_noaa_year_heterogeneity_audit", "146", "stage146_noaa_year_heterogeneity_audit.json", s146["status"], "复用阶段141冻结by-fold结果；按2022–2025四个目标年份分层并做fold bootstrap；不重新训练、不重新选择候选、不根据目标年test调参", f"扩展−default F1差值={s146_f1['mean_diff']:+.4f} CI=[{s146_f1['fold_bootstrap_ci95_low']:+.4f},{s146_f1['fold_bootstrap_ci95_high']:+.4f}]；AUPRC差值={s146_auprc['mean_diff']:+.4f}；Recall正向年份={s146_recall['positive_years']}/4；扩展−SVM F1差值={s146_svm_f1['mean_diff']:+.4f} CI=[{s146_svm_f1['fold_bootstrap_ci95_low']:+.4f},{s146_svm_f1['fold_bootstrap_ci95_high']:+.4f}]", "只有四个目标年份；扩展候选相对default的Recall/FAR存在年度波动，且四年F1均低于One-Class SVM，不能宣称跨年份普适领先")
    s147_spike = next(x for x in s147["paired"] if x["attack_type"] == "spike")
    s147_noise = next(x for x in s147["paired"] if x["attack_type"] == "gaussian_noise")
    s147_stuck = next(x for x in s147["paired"] if x["attack_type"] == "stuck")
    add(rows, "stage147_noaa_attack_year_interaction_audit", "147", "stage147_noaa_attack_year_interaction_audit.json", s147["status"], "复用阶段141冻结攻击类型明细；按六类攻击×四个目标年份分层并做年度bootstrap；不重新训练、不重新选择候选、不根据test结果调参", f"spike Recall差值={s147_spike['mean_diff']:+.4f} CI=[{s147_spike['year_bootstrap_ci95_low']:+.4f},{s147_spike['year_bootstrap_ci95_high']:+.4f}]；gaussian_noise差值={s147_noise['mean_diff']:+.4f}；stuck差值={s147_stuck['mean_diff']:+.4f}；扩展Recall≤0单元={len(s147['zero_or_lower_recall_cells'])}", "类型级结果显示spike/gaussian_noise等攻击Recall相对default下降，且11个年度×类型单元扩展Recall为0；阶段141没有SVM类型明细，不能补造逐类型比较")
    svm_types = {x["attack_type"]: x["recall_mean"] for x in s148["aggregate"] if x["method"] == "One-Class SVM"}
    add(rows, "stage148_noaa_baseline_attack_type_detail", "148", "stage148_noaa_baseline_attack_type_detail.json", s148["status"], "严格复用阶段85四折滚动、五种子和train正常样本拟合；Robust/IF/SVM/Trust legacy阈值只由valid选择，test只评价", f"SVM类型级Recall：stuck={svm_types['stuck']:.4f}、spike={svm_types['spike']:.4f}、offset={svm_types['offset']:.4f}、drift={svm_types['drift']:.4f}、time_shift={svm_types['time_shift']:.4f}、gaussian_noise={svm_types['gaussian_noise']:.4f}", "补齐的是阶段85同协议类型级明细，不增加独立数据源；攻击为透明受控注入，类型级低Recall必须与总体指标并列报告")
    s149_svm_spike = next(x for x in s149["paired"] if x["comparison"].endswith("One-Class SVM") and x["attack_type"] == "spike")
    s149_svm_noise = next(x for x in s149["paired"] if x["comparison"].endswith("One-Class SVM") and x["attack_type"] == "gaussian_noise")
    s149_if_stuck = next(x for x in s149["paired"] if x["comparison"].endswith("Isolation Forest") and x["attack_type"] == "stuck")
    add(rows, "stage149_noaa_attack_type_paired_bootstrap", "149", "stage149_noaa_attack_type_paired_bootstrap.json", s149["status"], "复用阶段141/148类型级Recall；先按四个滚动fold聚合五个种子，再做fold簇成对bootstrap；不重新训练、不重新选择候选、不根据test调参", f"expanded−SVM spike差值={s149_svm_spike['mean_diff']:+.4f} CI=[{s149_svm_spike['fold_bootstrap_ci95_low']:+.4f},{s149_svm_spike['fold_bootstrap_ci95_high']:+.4f}]；expanded−SVM gaussian_noise={s149_svm_noise['mean_diff']:+.4f}；expanded−IF stuck={s149_if_stuck['mean_diff']:+.4f} CI=[{s149_if_stuck['fold_bootstrap_ci95_low']:+.4f},{s149_if_stuck['fold_bootstrap_ci95_high']:+.4f}]", "类型级比较显示expanded对多数攻击类型不占优，stuck优势需与其他类型负结果和总体FAR/AUPRC并列解释")
    s150_q = [float(x["bh_q_value"]) for x in s150["detail"]]
    add(rows, "stage150_noaa_attack_type_multiple_comparison", "150", "stage150_noaa_attack_type_multiple_comparison.json", s150["status"], "复用阶段149四折、五种子类型级Recall差异；精确sign-flip检验；24个假设做BH校正；不重新训练、不重新选择候选、不根据test调参", f"假设数={s150['hypothesis_count']}；BH校正后q<0.05={sum(q < 0.05 for q in s150_q)}；最小q={min(s150_q):.4f}", "四折统计功效有限；未显著不等于等效；多重校正只用于透明报告，不改变主模型选择")
    s151_agg = {float(x["amplitude"]): x for x in s151["aggregate"]}
    s151_type = [x for x in s151["type_aggregate"] if float(x["amplitude"]) == 0.5]
    weak_types = [str(x["attack_type"]) for x in s151_type if float(x["recall_mean"]) < 0.01]
    add(rows, "stage151_noaa_attack_intensity_sensitivity", "151", "stage151_noaa_attack_intensity_sensitivity.json", s151["status"], "复用阶段141冻结扩展候选alpha/window；预先固定攻击强度0.5/1.0/1.5/2.0；每个强度只由valid估计阈值，test只评价，不根据test选强度或参数", f"F1={s151_agg[0.5]['test_f1_mean']:.4f}→{s151_agg[2.0]['test_f1_mean']:.4f}; Recall={s151_agg[0.5]['test_recall_mean']:.4f}→{s151_agg[2.0]['test_recall_mean']:.4f}; FAR={s151_agg[0.5]['test_false_alarm_rate_mean']:.4f}→{s151_agg[2.0]['test_false_alarm_rate_mean']:.4f}; 强度0.5低Recall类型={','.join(weak_types)}", "强度增强带来Recall/F1提升同时增加FAR；time_shift/stuck不随数值幅度变化；低强度gaussian_noise、drift等弱项必须保留，攻击仍为透明受控注入")
    s152_svm_f1 = next(x for x in s152["paired"] if x["comparison"].endswith("One-Class SVM") and x["metric"] == "f1")
    s152_default_recall = next(x for x in s152["paired"] if x["comparison"].endswith("TrustMeteo-QC default") and x["metric"] == "recall")
    s152_default_far = next(x for x in s152["paired"] if x["comparison"].endswith("TrustMeteo-QC default") and x["metric"] == "far")
    add(rows, "stage152_noaa_baseline_station_comparison", "152", "stage152_noaa_baseline_station_comparison.json", s152["status"], "严格复用阶段85四折、五种子和valid阈值；Trust站点明细复用阶段141冻结结果；按11站点聚合后做站点簇bootstrap；不重新选择候选、不根据test调参", f"expanded−default Recall={s152_default_recall['mean_station_diff']:+.4f} CI=[{s152_default_recall['station_bootstrap_ci95_low']:+.4f},{s152_default_recall['station_bootstrap_ci95_high']:+.4f}]；expanded−default FAR={s152_default_far['mean_station_diff']:+.4f}；expanded−SVM F1={s152_svm_f1['mean_station_diff']:+.4f} CI=[{s152_svm_f1['station_bootstrap_ci95_low']:+.4f},{s152_svm_f1['station_bootstrap_ci95_high']:+.4f}]", "扩展候选在站点层面主要改善Precision/FAR并牺牲Recall；相对SVM的F1区间跨零，不能宣称全面优于基线；弱站点必须保留")
    s153_svm_alarm = next(x for x in s153["paired"] if x["comparison"].endswith("One-Class SVM") and x["metric"] == "alarm_rate")
    s153_svm_flagged = next(x for x in s153["paired"] if x["comparison"].endswith("One-Class SVM") and x["metric"] == "quality_flagged_alarm_rate")
    add(rows, "stage153_noaa_natural_station_operational_comparison", "153", "stage153_noaa_natural_station_operational_comparison.json", s153["status"], "复用阶段142冻结自然观测站点明细；不注入攻击、不使用目标年攻击标签、不根据目标年标签选q；按11站点、5个目标年份和3个q运行点做站点簇bootstrap", f"expanded−SVM自然报警率站点差异={s153_svm_alarm['overall_mean_station_diff']:+.6f} CI=[{s153_svm_alarm['overall_station_bootstrap_ci95_low']:+.6f},{s153_svm_alarm['overall_station_bootstrap_ci95_high']:+.6f}]；质量标记报警率差异={s153_svm_flagged['overall_mean_station_diff']:+.6f}", "自然报警率不是攻击召回率；NOAA质量字段只是弱标签；站点簇数量有限，结果仅支持运行安全和弱标签关联描述")
    s154_expanded = next(x for x in s154["summary"] if x["method"] == "TrustMeteo-QC frozen expanded" and float(x["q"]) == 0.995)
    s154_default = next(x for x in s154["summary"] if x["method"] == "TrustMeteo-QC default" and float(x["q"]) == 0.995)
    s154_svm = next(x for x in s154["paired"] if x["comparison"].endswith("One-Class SVM") and x["metric"] == "alarm_rate")
    add(rows, "stage154_noaa_natural_temporal_drift", "154", "stage154_noaa_natural_temporal_drift.json", s154["status"], "复用阶段142冻结自然观测；5个目标年份、11站点、3个q运行点；估计报警率年度斜率并做站点bootstrap；不注入攻击、不使用攻击标签、不根据目标年调参", f"q=.995扩展年度斜率={s154_expanded['alarm_rate_year_slope']:+.8f} CI=[{s154_expanded['alarm_rate_slope_ci95_low']:+.8f},{s154_expanded['alarm_rate_slope_ci95_high']:+.8f}]；default斜率={s154_default['alarm_rate_year_slope']:+.8f}；扩展−SVM斜率差={s154_svm['mean_station_q_slope_difference']:+.8f} CI=[{s154_svm['bootstrap_ci95_low']:+.8f},{s154_svm['bootstrap_ci95_high']:+.8f}]", "只有5个目标年份；斜率用于自然运行漂移诊断，不证明长期稳定性或攻击召回")
    s155_cold_svm = next(x for x in s155["paired"] if x["comparison"].endswith("One-Class SVM") and float(x["q"]) == 0.995 and x["regime_variable"] == "temperature" and x["regime"] == "cold")
    s155_strong_svm = next(x for x in s155["paired"] if x["comparison"].endswith("One-Class SVM") and float(x["q"]) == 0.995 and x["regime_variable"] == "wind_speed" and x["regime"] == "strong")
    s155_mild_default = next(x for x in s155["paired"] if x["comparison"].endswith("TrustMeteo-QC default") and float(x["q"]) == 0.995 and x["regime_variable"] == "temperature" and x["regime"] == "mild")
    add(rows, "stage155_noaa_natural_weather_regime", "155", "stage155_noaa_natural_weather_regime_audit.json", s155["status"], "复用阶段142冻结自然观测；状态边界和风险阈值只由历史训练年估计；5个目标年份、11站点、温度/风速状态和3个固定q运行点；目标年不注入攻击、不根据状态结果调参", f"q=.995扩展−SVM冷态报警率差={s155_cold_svm['mean_station_diff']:+.6f} CI=[{s155_cold_svm['station_bootstrap_ci95_low']:+.6f},{s155_cold_svm['station_bootstrap_ci95_high']:+.6f}]；强风态差={s155_strong_svm['mean_station_diff']:+.6f} CI=[{s155_strong_svm['station_bootstrap_ci95_low']:+.6f},{s155_strong_svm['station_bootstrap_ci95_high']:+.6f}]；扩展−default温和态差={s155_mild_default['mean_station_diff']:+.6f}", "自然报警率不是攻击召回率；状态边界是站点本地训练年分位数而非物理异常阈值；default自然报警更低的结果必须保留")
    s156_gap_svm = next(x for x in s156["paired"] if x["comparison"].endswith("One-Class SVM") and float(x["q"]) == 0.995 and x["gap_bucket"] == "long_6_24h")
    s156_regular_default = next(x for x in s156["paired"] if x["comparison"].endswith("TrustMeteo-QC default") and float(x["q"]) == 0.995 and x["gap_bucket"] == "regular_le_3h")
    add(rows, "stage156_noaa_natural_observation_gap", "156", "stage156_noaa_natural_observation_gap_audit.json", s156["status"], "复用阶段142冻结自然观测；按固定时间戳间隔分层；5个目标年份、11站点和3个固定q运行点；阈值只由历史训练年估计；不注入攻击、不使用目标年攻击标签、不根据目标年报警结果调参", f"q=.995长间隔(6–24h)扩展−SVM报警率差={s156_gap_svm['mean_station_diff']:+.6f} CI=[{s156_gap_svm['station_bootstrap_ci95_low']:+.6f},{s156_gap_svm['station_bootstrap_ci95_high']:+.6f}]；规则间隔扩展−default差={s156_regular_default['mean_station_diff']:+.6f} CI=[{s156_regular_default['station_bootstrap_ci95_low']:+.6f},{s156_regular_default['station_bootstrap_ci95_high']:+.6f}]", "观测间隔分层只描述自然运行安全；长间隔样本较少且不能解释为攻击召回或通信中断因果")
    s157_temp = next(x for x in s157["detail"] if x["source_stage"] == "stage155" and x["comparison"].endswith("One-Class SVM") and float(x["q"]) == 0.995 and x["regime_variable"] == "temperature" and x["regime"] == "cold")
    s157_gap = next(x for x in s157["detail"] if x["source_stage"] == "stage156" and x["comparison"].endswith("One-Class SVM") and float(x["q"]) == 0.995 and x["gap_bucket"] == "long_6_24h")
    add(rows, "stage157_noaa_natural_two_way_cluster_bootstrap", "157", "stage157_noaa_natural_two_way_cluster_bootstrap.json", s157["status"], "复用阶段155/156站点明细；不重新训练、不重新选择q；同时重采样站点簇和目标年份簇，检验自然报警差异区间对聚类单位的敏感性", f"温度冷态扩展−SVM双层CI=[{s157_temp['two_way_bootstrap_ci95_low']:+.6f},{s157_temp['two_way_bootstrap_ci95_high']:+.6f}]；6–24h长间隔扩展−SVM双层CI=[{s157_gap['two_way_bootstrap_ci95_low']:+.6f},{s157_gap['two_way_bootstrap_ci95_high']:+.6f}]", "双层bootstrap不是新增独立样本；长间隔区间跨零时必须保留不确定性，不能宣称普遍优于基线")
    s158_hot_f1 = next(x for x in s158["paired"] if x["metric"] == "test_f1" and x["context_variable"] == "temperature_regime" and x["context"] == "hot")
    s158_cold_f1 = next(x for x in s158["paired"] if x["metric"] == "test_f1" and x["context_variable"] == "temperature_regime" and x["context"] == "cold")
    s158_regular_far = next(x for x in s158["paired"] if x["metric"] == "test_false_alarm_rate" and x["context_variable"] == "gap_bucket" and x["context"] == "regular_le_3h")
    add(rows, "stage158_noaa_controlled_regime_gap_performance", "158", "stage158_noaa_controlled_regime_gap_performance.json", s158["status"], "NOAA四折累计历史训练→valid→目标年test；攻击为透明受控注入；default与冻结扩展方案按每折valid选阈值，test只按原始温度状态和观测间隔分层评价", f"高温态扩展−default F1差={s158_hot_f1['mean_diff']:+.4f} CI=[{s158_hot_f1['bootstrap_fold_ci95_low']:+.4f},{s158_hot_f1['bootstrap_fold_ci95_high']:+.4f}]；寒冷态F1差={s158_cold_f1['mean_diff']:+.4f} CI=[{s158_cold_f1['bootstrap_fold_ci95_low']:+.4f},{s158_cold_f1['bootstrap_fold_ci95_high']:+.4f}]；规则间隔FAR差={s158_regular_far['mean_diff']:+.4f}", "受控攻击不是真实攻击日志；分层结果显示环境条件敏感性，不能据此宣称全面领先")
    s159_best = min(s159["detail"], key=lambda x: float(x["bh_q_value"]))
    add(rows, "stage159_noaa_controlled_regime_multiple_comparison", "159", "stage159_noaa_controlled_regime_multiple_comparison.json", s159["status"], "复用阶段158四折五种子分层结果；按fold先取种子均值，再对33个可估计状态×指标假设执行精确sign-flip和BH校正；不重新训练、不重新选择候选", f"假设数={s159['hypothesis_count']}；不可估计={s159['not_estimable_count']}；q<0.05={s159['q_below_05']}；最小BH q={s159_best['bh_q_value']:.4f}（{s159_best['metric']} / {s159_best['context_variable']}={s159_best['context']}）", "四个fold功效有限；多重校正结果用于防止选择性报告，不支持把未显著写成等效")
    s160_hot = next(x for x in s160["detail"] if x["metric"] == "test_f1" and x["context_variable"] == "temperature_regime" and x["context"] == "hot")
    s160_cold = next(x for x in s160["detail"] if x["metric"] == "test_f1" and x["context_variable"] == "temperature_regime" and x["context"] == "cold")
    s160_long_far = next(x for x in s160["detail"] if x["metric"] == "test_false_alarm_rate" and x["context_variable"] == "gap_bucket" and x["context"] == "long_6_24h")
    add(rows, "stage160_noaa_controlled_practical_effect_sensitivity", "160", "stage160_noaa_controlled_practical_effect_sensitivity.json", s160["status"], "复用阶段158四折五种子；预先固定F1/Precision/AUPRC=0.02、Recall=0.05、FAR=0.01实际效应边界，按fold计算越界情况和bootstrap区间；不重新训练、不依据结果调参", f"高温F1差={s160_hot['mean_fold_diff']:+.4f}，四fold均超过0.02；寒冷F1差={s160_cold['mean_fold_diff']:+.4f}，四fold均在0.02内；6–24h FAR差={s160_long_far['mean_fold_diff']:+.4f}，四fold均超过0.01下降边界", "实际效应边界是分析约定，不替代显著性；FAR下降不等于召回提升；不可估计AUPRC保留为空")
    s162_hot = next(x for x in s162["summary"] if x["metric"] == "test_f1" and x["context_variable"] == "temperature_regime" and x["context"] == "hot")
    add(rows, "stage161_noaa_controlled_station_regime_performance", "161", "stage161_noaa_controlled_station_regime_performance.json", s161["status"], "复用阶段158四折滚动、五种子和valid阈值；test只评价；在站点×温度状态/观测间隔层面保存冻结扩展与default的F1、Recall、Precision、AUPRC、FAR", f"站点层明细={s161['detail_rows']}；站点汇总={s161['station_summary_rows']}；站点覆盖={len(s161['stations'])}", "站点层用于异质性诊断，不改变冻结模型；受控攻击不是真实攻击日志")
    add(rows, "stage162_noaa_controlled_station_practical_effect", "162", "stage162_noaa_controlled_station_practical_effect.json", s162["status"], "复用阶段161站点层test结果；预先固定F1/Precision/AUPRC=0.02、Recall=0.05、FAR=0.01实际效应边界；站点bootstrap并列正向、负向和弱站点，不重新训练或调参", f"可估计单元={s162['summary_rows']}；不可估计={s162['not_estimable_count']}；高温F1站点平均差={s162_hot['mean_station_diff']:+.4f}，正向站点={s162_hot['positive_effect_station_count']}，负向站点={s162_hot['negative_effect_station_count']}", "站点数量有限；弱站点不能删除；实际效应边界不替代多重比较或真实攻击验证")
    s163_hot_svm = next(x for x in s163["paired"] if x["comparison"] == "expanded - One-Class SVM" and x["metric"] == "test_f1" and x["context_variable"] == "temperature_regime" and x["context"] == "hot")
    s163_cold_svm = next(x for x in s163["paired"] if x["comparison"] == "expanded - One-Class SVM" and x["metric"] == "test_f1" and x["context_variable"] == "temperature_regime" and x["context"] == "cold")
    add(rows, "stage163_noaa_controlled_regime_full_baseline", "163", "stage163_noaa_controlled_regime_full_baseline.json", s163["status"], "NOAA四折五种子；五种方法同一train/valid/test边界；独立基线只用train正常样本拟合、valid选阈值，test按温度状态和观测间隔分层", f"分层明细={s163['detail_rows']}；成对比较={s163['paired_rows']}；高温F1扩展−SVM={s163_hot_svm['mean_diff']:+.4f} CI=[{s163_hot_svm['bootstrap_fold_ci95_low']:+.4f},{s163_hot_svm['bootstrap_fold_ci95_high']:+.4f}]；寒冷F1扩展−SVM={s163_cold_svm['mean_diff']:+.4f} CI=[{s163_cold_svm['bootstrap_fold_ci95_low']:+.4f},{s163_cold_svm['bootstrap_fold_ci95_high']:+.4f}]", "扩展方案不能宣称全面优于所有基线；寒冷和规则间隔下SVM等基线可能占优；受控攻击不是真实攻击日志")
    add(rows, "stage164_noaa_controlled_full_baseline_multiple_comparison", "164", "stage164_noaa_controlled_full_baseline_multiple_comparison.json", s164["status"], "复用阶段163五种方法分层结果；按fold汇总五种子，对扩展方案相对四个基线的132个可估计假设执行精确sign-flip和BH校正，不重新训练或调参", f"可估计假设={s164['hypothesis_count']}；不可估计={s164['not_estimable_count']}；BH q<0.05={s164['q_below_05']}；最小BH q={s164['min_bh_q_value']:.4f}", "多重校正后没有显著环境分层假设，不能把高温结果写成普遍统计显著优势")
    s165_hot_svm = next(x for x in s165["paired"] if x["comparison"] == "expanded - One-Class SVM" and x["metric"] == "test_f1" and x["context_variable"] == "temperature_regime" and x["context"] == "hot")
    add(rows, "stage165_noaa_controlled_station_full_baseline", "165", "stage165_noaa_controlled_station_full_baseline.json", s165["status"], "复用阶段161站点层Trust结果并补齐Robust、Isolation Forest和One-Class SVM；四折五种子；各独立基线只用train正常样本拟合、valid选阈值，test只评价；站点bootstrap检查优势是否由少数站点贡献", f"站点明细={s165['detail_rows']}；站点汇总={s165['station_summary_rows']}；可估计成对单元={s165['paired_rows']}；高温F1扩展−SVM站点平均差={s165_hot_svm['mean_station_diff']:+.4f} CI=[{s165_hot_svm['station_bootstrap_ci95_low']:+.4f},{s165_hot_svm['station_bootstrap_ci95_high']:+.4f}]，正向站点={s165_hot_svm['positive_station_count']}，负向站点={s165_hot_svm['negative_station_count']}", "站点层不可估计单元较多时必须保留为空；弱站点和基线优势不能删除；受控攻击不是真实攻击日志")
    s166_stuck_svm = next(x for x in s166["paired"] if x["comparison"] == "expanded - One-Class SVM" and x["metric"] == "test_f1" and x["context_variable"] == "temperature_regime" and x["context"] == "hot" and x["attack_type"] == "stuck")
    s166_drift_svm = next(x for x in s166["paired"] if x["comparison"] == "expanded - One-Class SVM" and x["metric"] == "test_f1" and x["context_variable"] == "temperature_regime" and x["context"] == "hot" and x["attack_type"] == "drift")
    add(rows, "stage166_noaa_controlled_attack_environment_interaction", "166", "stage166_noaa_controlled_attack_environment_interaction.json", s166["status"], "NOAA四折五种子；五种方法同一train/valid/test边界；每个攻击类型×环境状态保留全部正常行作为误报分母，test只评价", f"交互明细={s166['detail_rows']}；成对比较={s166['paired_rows']}；高温stuck F1扩展−SVM={s166_stuck_svm['mean_diff']:+.4f} CI=[{s166_stuck_svm['bootstrap_fold_ci95_low']:+.4f},{s166_stuck_svm['bootstrap_fold_ci95_high']:+.4f}]；高温drift F1扩展−SVM={s166_drift_svm['mean_diff']:+.4f} CI=[{s166_drift_svm['bootstrap_fold_ci95_low']:+.4f},{s166_drift_svm['bootstrap_fold_ci95_high']:+.4f}]", "攻击类型交互显示选择性优势和负结果，不能汇总成全面领先；受控攻击不是真实攻击日志")
    add(rows, "stage167_noaa_controlled_attack_environment_multiple_comparison", "167", "stage167_noaa_controlled_attack_environment_multiple_comparison.json", s167["status"], "复用阶段166交互结果；按fold汇总五种子，对540个攻击类型×环境×指标×基线假设统一执行精确sign-flip和Benjamini–Hochberg校正，不重新训练或调参", f"可估计假设={s167['hypothesis_count']}；不可估计={s167['not_estimable_count']}；BH q<0.05={s167['q_below_05']}；最小BH q={s167['min_bh_q_value']:.4f}", "交互多重校正后没有显著假设；四fold功效有限，不能把选择性信号写成普遍统计显著优势")
    counts168 = s168.get("conclusion_counts", {})
    add(rows, "stage168_noaa_controlled_attack_environment_practical_effect", "168", "stage168_noaa_controlled_attack_environment_practical_effect.json", s168["status"], "复用阶段166交互结果；预先固定F1/Precision/AUPRC=0.02、Recall=0.05、FAR=0.01实际效应边界，按fold汇总五种子并计算bootstrap区间；不重新训练、不根据结果调参", f"可估计假设={s168['hypothesis_count']}；不可估计={s168['not_estimable_count']}；全fold实际增益={counts168.get('all_folds_practical_gain', 0)}；全fold实际损失={counts168.get('all_folds_practical_loss', 0)}；混合/小效应={counts168.get('mixed_or_small', 0)}", "实际效应边界是预先固定的工程解释约定，不替代显著性；受控攻击不是真实攻击日志；弱攻击类型和负结果必须保留")
    s169_f1 = s169.get("summary", [])
    expanded_10 = next((x for x in s169_f1 if x.get("method") == "TrustMeteo-QC frozen expanded" and float(x.get("amplitude", -1)) == 1.0), None)
    svm_10 = next((x for x in s169_f1 if x.get("method") == "One-Class SVM" and float(x.get("amplitude", -1)) == 1.0), None)
    add(rows, "stage169_noaa_controlled_attack_intensity_full_baseline", "169", "stage169_noaa_controlled_attack_intensity_full_baseline.json", s169["status"], "NOAA四折五种子；攻击强度0.5/1.0/1.5/2.0预先固定；五种方法使用相同数据、攻击位置和train/valid/test边界，独立基线只用train正常样本拟合、valid选阈值、test评价", f"明细={s169['detail_rows']}；类型明细={s169['type_detail_rows']}；成对比较={s169['paired_rows']}；强度1.0扩展F1={float(expanded_10['test_f1_mean']):.4f}；SVM F1={float(svm_10['test_f1_mean']):.4f}", "扩展方案不能宣称强度全范围优于SVM；攻击为透明受控注入，不是真实攻击日志；负结果和方法差异必须并列报告")
    s170_f1 = s170.get("summary", [])
    expanded_8 = next((x for x in s170_f1 if x.get("method") == "TrustMeteo-QC frozen expanded" and int(x.get("duration", -1)) == 8), None)
    svm_8 = next((x for x in s170_f1 if x.get("method") == "One-Class SVM" and int(x.get("duration", -1)) == 8), None)
    expanded_4 = next((x for x in s170_f1 if x.get("method") == "TrustMeteo-QC frozen expanded" and int(x.get("duration", -1)) == 4), None)
    add(rows, "stage170_noaa_controlled_attack_duration_full_baseline", "170", "stage170_noaa_controlled_attack_duration_full_baseline.json", s170["status"], "NOAA四折五种子；固定攻击类型和位置，仅将事件截取为4/8/12个观测点；五种方法使用相同数据和train/valid/test边界，独立基线只用train正常样本拟合、valid选阈值、test评价", f"明细={s170['detail_rows']}；类型明细={s170['type_detail_rows']}；成对比较={s170['paired_rows']}；时长4扩展F1={float(expanded_4['test_f1_mean']):.4f}；时长8扩展F1={float(expanded_8['test_f1_mean']):.4f}；时长8 SVM F1={float(svm_8['test_f1_mean']):.4f}", "短事件下扩展方案弱于多个基线；时长8/12下仍低于SVM；持续时间敏感性必须作为限制报告，不能宣称全时长领先")
    s171_f1 = s171.get("summary", [])
    expanded_0 = next((x for x in s171_f1 if x.get("method") == "TrustMeteo-QC frozen expanded" and int(x.get("shift_rows", 999)) == 0), None)
    expanded_m24 = next((x for x in s171_f1 if x.get("method") == "TrustMeteo-QC frozen expanded" and int(x.get("shift_rows", 999)) == -24), None)
    svm_0 = next((x for x in s171_f1 if x.get("method") == "One-Class SVM" and int(x.get("shift_rows", 999)) == 0), None)
    add(rows, "stage171_noaa_controlled_attack_position_full_baseline", "171", "stage171_noaa_controlled_attack_position_full_baseline.json", s171["status"], "NOAA四折五种子；攻击类型、幅度和持续时间固定，仅在同站点同split内平移-24/0/+24个观测行；五种方法使用相同数据和train/valid/test边界，独立基线只用train正常样本拟合、valid选阈值、test评价", f"明细={s171['detail_rows']}；类型明细={s171['type_detail_rows']}；成对比较={s171['paired_rows']}；原位扩展F1={float(expanded_0['test_f1_mean']):.4f}；-24行扩展F1={float(expanded_m24['test_f1_mean']):.4f}；原位SVM F1={float(svm_0['test_f1_mean']):.4f}", "位置变化会改变相对差异；扩展方案在三种位置均低于SVM，不能宣称位置无关或全面领先；攻击为透明受控注入")
    s172_summary = s172.get("summary", [])
    s172_expanded_995 = next(x for x in s172_summary if x.get("method") == "TrustMeteo-QC frozen expanded" and abs(float(x.get("q", -1)) - 0.995) < 1e-12)
    s172_svm_995 = next(x for x in s172_summary if x.get("method") == "One-Class SVM" and abs(float(x.get("q", -1)) - 0.995) < 1e-12)
    s172_expanded_999 = next(x for x in s172_summary if x.get("method") == "TrustMeteo-QC frozen expanded" and abs(float(x.get("q", -1)) - 0.999) < 1e-12)
    s172_svm_999 = next(x for x in s172_summary if x.get("method") == "One-Class SVM" and abs(float(x.get("q", -1)) - 0.999) < 1e-12)
    add(rows, "stage172_noaa_frozen_expanded_low_far_full_baseline", "172", "stage172_noaa_frozen_expanded_low_far_full_baseline.json", s172["status"], "NOAA 2020–2025四折滚动、五种子；攻击强度固定1.0；q=0.95/0.975/0.99/0.995/0.9975/0.999预先固定；扩展、default、Robust、Isolation Forest、One-Class SVM使用相同边界，独立基线只用train正常样本拟合、阈值由train分位数确定、test只评价", f"明细={s172['detail_rows']}；汇总={s172['summary_rows']}；成对比较={s172['paired_rows']}；q=.995扩展F1={float(s172_expanded_995['test_f1_mean']):.4f}、SVM F1={float(s172_svm_995['test_f1_mean']):.4f}；q=.999扩展F1={float(s172_expanded_999['test_f1_mean']):.4f}、SVM F1={float(s172_svm_999['test_f1_mean']):.4f}", "低误报点扩展方案未优于One-Class SVM，q=.995也低于Isolation Forest；仅相对default和Robust在部分指标有小幅改善；攻击为透明受控注入，不是真实攻击日志")
    add(rows, "stage173_noaa_low_far_full_baseline_multiple_comparison", "173", "stage173_noaa_low_far_full_baseline_multiple_comparison.json", s173["status"], "复用阶段172四折五种子和六个预先固定q运行点；按fold汇总种子后，对扩展相对四个基线的q×指标差异执行精确sign-flip并统一BH校正；不重新训练、不根据test调参", f"可估计假设={s173['hypothesis_count']}；不可估计={s173['not_estimable_count']}；BH q<0.05={s173['q_below_05']}；最小BH q={s173['min_bh_q_value']:.4f}", "四折统计功效有限；校正后无显著全面优势，不能选择性报告单个q或指标；攻击为透明受控注入，不是真实攻击日志")
    burden_995 = next(x for x in s174["burden"] if x.get("method") == "TrustMeteo-QC frozen expanded" and abs(float(x.get("q", -1)) - 0.995) < 1e-12)
    add(rows, "stage174_noaa_low_far_alarm_burden_audit", "174", "stage174_noaa_low_far_alarm_burden_audit.json", s174["status"], "复用阶段172四折五种子、五种方法和六个预先固定q运行点；不重新训练、不重新选择阈值；逐站点和逐月记录TP/FP/FN/TN与报警数，并计算报警集中度", f"明细={s174['detail_rows']}；站点明细={s174['station_detail_rows']}；月份明细={s174['month_detail_rows']}；q=0.995扩展报警数={burden_995['alarm_count']}；站点前10%报警份额={burden_995['station_alarm_share_top10pct']:.4f}；冻结重放不一致={s174['replay_mismatch_count']}", "四个滚动fold和11个站点限制集中度结论；报警集中度是运行负担诊断，不等于检测召回或真实攻击率；NOAA攻击为透明受控注入，不是真实攻击日志")
    add(rows, "stage175_noaa_low_far_alarm_burden_stability", "175", "stage175_noaa_low_far_alarm_burden_stability.json", s175["status"], "复用阶段174的train/valid/test逐站点和逐月报警计数；按fold×种子重算集中度，扩展方案相对四个基线按fold汇总并执行精确sign-flip与BH校正；不重新训练、不重新选择q或阈值", f"明细={s175['detail_rows']}；成对比较={s175['comparison_rows']}；BH q<0.05={s175['bh_q_below_05']}；最小BH q={s175['min_bh_q_value']:.4f}；扩展q=0.995站点前10%份额均值={s175['expanded_q995_station_top10pct_mean']:.4f}", "四个滚动fold功效有限；BH校正后无显著差异不能写成方法等效；集中度只描述运行负担，不是召回或真实攻击率；NOAA攻击为透明受控注入")
    add(rows, "stage176_noaa_low_far_station_equal_weight_metrics", "176", "stage176_noaa_low_far_station_equal_weight_metrics.json", s176["status"], "复用阶段174站点级TP/FP/FN/TN；按fold×种子计算行汇总pooled与站点等权macro指标，并计算误报超配比；扩展方案相对四个基线执行精确sign-flip与BH校正；不重新训练、不重新选择q或阈值", f"明细={s176['detail_rows']}；站点明细={s176['station_detail_rows']}；成对比较={s176['comparison_rows']}；BH q<0.05={s176['bh_q_below_05']}；最小BH q={s176['min_bh_q_value']:.4f}；扩展q=0.995 pooled F1={s176['expanded_q995_pooled_f1_mean']:.4f}、macro F1={s176['expanded_q995_macro_f1_mean']:.4f}", "macro指标只改变站点权重，不创造独立样本；误报超配只描述运行负担；BH校正后无显著差异；NOAA攻击为透明受控注入")
    add(rows, "stage177_noaa_station_calibrated_operating_points", "177", "stage177_noaa_station_calibrated_operating_points.json", s177["status"], "复用阶段172冻结风险分数、四折、五种子、五方法和六个预先固定q；每个站点只用本站train正常样本的风险分位数确定阈值，test只评价；全局阈值与站点校准阈值并列比较，并执行精确sign-flip与BH校正；不重新训练、不重新选择模型或q", f"明细={s177['detail_rows']}；站点明细={s177['station_detail_rows']}；基线比较={s177['baseline_comparison_rows']}；校准比较={s177['calibration_comparison_rows']}；全局重放不一致={s177['global_replay_mismatch_count']}；BH q<0.05={s177['baseline_bh_q_below_05'] + s177['calibration_bh_q_below_05']}；最小BH q={min(s177['min_baseline_bh_q_value'], s177['min_calibration_bh_q_value']):.4f}；扩展q=0.995站点F1={s177['expanded_q995_station_f1_mean']:.4f}、全局F1={s177['expanded_q995_global_f1_mean']:.4f}", "站点校准只改变运行点，不改变风险分数；每站正常样本量不同，分位数目标不等于每站相同报警数；四折功效有限，BH校正后无显著差异不等于方法等效；NOAA攻击为透明受控注入，不是真实攻击日志")
    rule_995 = s178["expanded_q995_by_allocation_rule"]
    add(rows, "stage178_noaa_station_calibrated_alarm_budget", "178", "stage178_noaa_station_calibrated_alarm_budget.json", s178["status"], "复用阶段172冻结风险分数、四折、五种子、五方法和六个预先固定预算分位；全局对照和站点分配都精确取相同总报警预算，站点预算分别按站点规模、站点等量和站点平均风险三种预先固定规则做最大余数分配，只使用站点大小和无标签风险；不重新训练、不重新选择模型或q；差异按fold执行精确sign-flip与BH校正", f"明细={s178['detail_rows']}；站点明细={s178['station_detail_rows']}；基线比较={s178['baseline_comparison_rows']}；校准比较={s178['calibration_comparison_rows']}；BH q<0.05={s178['baseline_bh_q_below_05'] + s178['calibration_bh_q_below_05']}；扩展q=0.995 F1：size={rule_995['size']['station_budget_f1_mean']:.4f}、uniform={rule_995['uniform']['station_budget_f1_mean']:.4f}、risk={rule_995['risk']['station_budget_f1_mean']:.4f}；对应全局预算F1={rule_995['size']['global_budget_f1_mean']:.4f}/{rule_995['uniform']['global_budget_f1_mean']:.4f}/{rule_995['risk']['global_budget_f1_mean']:.4f}", "等预算比较只改变报警在站点间的分配，不改变总报警数；size、uniform、risk是预先固定的公平性检验，不是业务最优分配；四折功效有限，BH校正后无显著差异不等于方法等效；NOAA攻击为透明受控注入，不是真实攻击日志")
    s179_rule = s179["expanded_q995_by_allocation_rule"]
    add(rows, "stage179_noaa_station_budget_loss_decomposition", "179", "stage179_noaa_station_budget_loss_decomposition.json", s179["status"], "复用阶段178逐站点报警计数；不重新训练、不重算风险、不重新选择模型、阈值、q或分配规则；将固定预算下的Recall损失分解为oracle上限、固定站点分配上限和站内排序后的实际值；oracle使用test标签仅作事后不可部署诊断", f"明细={s179['detail_rows']}；比较={s179['comparison_rows']}；q=0.995实际Recall：size={s179_rule['size']['actual_recall_mean']:.4f}、uniform={s179_rule['uniform']['actual_recall_mean']:.4f}、risk={s179_rule['risk']['actual_recall_mean']:.4f}；分配上限Recall：size={s179_rule['size']['allocation_ceiling_recall_mean']:.4f}、uniform={s179_rule['uniform']['allocation_ceiling_recall_mean']:.4f}、risk={s179_rule['risk']['allocation_ceiling_recall_mean']:.4f}；分配损失TP：size={s179_rule['size']['allocation_loss_count_mean']:.2f}、uniform={s179_rule['uniform']['allocation_loss_count_mean']:.2f}、risk={s179_rule['risk']['allocation_loss_count_mean']:.2f}", "oracle和固定分配上限都使用test攻击计数，仅用于解释损失来源，不能用于部署或选择规则；四折功效有限，BH校正后无显著差异不等于方法等效；NOAA攻击为透明受控注入，不是真实攻击日志")
    s180_summary = pd.DataFrame(s180["summary"])
    s180_q995 = (
        s180_summary[s180_summary["q"].eq(0.995)]
        .groupby("allocation_rule", sort=True)["attack_recall"]
        .mean()
        .to_dict()
    )
    s180_events = {
        (row["allocation_rule"], row["attack_type"]): row
        for row in s180["event_aggregate"]
        if abs(float(row["q"]) - 0.995) < 1e-12
    }
    add(rows, "stage180_noaa_station_budget_row_tail_audit", "180", "stage180_noaa_station_budget_row_tail_audit.json", s180["status"], "复用阶段178冻结扩展评分设置和同一受控注入；只保留预先固定的q=0.995和q=0.999低误报运行点，保存逐行test风险、风险分位、三种站点预算选择、攻击类型和事件ID；模型、alpha、window、q和分配规则均不根据test选择；oracle式计数仅作诊断，不部署", f"逐行={s180['row_detail_rows']}；站点={s180['station_detail_rows']}；站点×类型={s180['station_type_detail_rows']}；事件={s180['event_detail_rows']}；事件聚合={s180['event_aggregate_rows']}；q=0.995站点预算Recall均值：size={s180_q995['size']:.4f}、uniform={s180_q995['uniform']:.4f}、risk={s180_q995['risk']:.4f}；事件覆盖：stuck size={s180_events[('size', 'stuck')]['event_coverage']:.4f}、uniform={s180_events[('uniform', 'stuck')]['event_coverage']:.4f}、risk={s180_events[('risk', 'stuck')]['event_coverage']:.4f}；非stuck类型覆盖接近0", "低误报运行点主要依赖stuck事件，不能外推为所有攻击类型普遍低误报检测能力；逐行证据仍来自透明受控注入，不代表真实攻击日志；oracle式计数和风险分位只用于解释漏检位置，不能部署")
    s181_components = {
        row["component"]: row
        for row in s181["component_detail"]
        if abs(float(row["q"]) - 0.995) < 1e-12
        and row["attack_type"] == "stuck"
    }
    add(rows, "stage181_noaa_non_stuck_failure_attribution", "181", "stage181_noaa_non_stuck_failure_attribution.json", s181["status"], "复用阶段178冻结扩展评分设置和同一透明受控注入；不重新训练、不重新选择alpha、window、q或阈值；把逐行漏检归因到现有结构风险分量和基础融合分量，只作诊断", f"逐行={s181['row_detail_rows']}；类型明细={s181['type_detail_rows']}；类型聚合={s181['type_aggregate_rows']}；分量明细={s181['component_detail_rows']}；stuck structural_risk攻击均值={s181_components['structural_risk']['attack_mean']:.4f}；非stuck分量响应弱", "该阶段说明低误报主要依赖stuck；非stuck漏检不能仅靠调融合权重解决；NOAA攻击为透明受控注入")
    s182_agg = pd.DataFrame(s182["type_aggregate"])
    s182_diff = s182_agg[s182_agg["feature"].eq("difference_burst")]
    add(rows, "stage182_noaa_non_stuck_candidate_features", "182", "stage182_noaa_non_stuck_candidate_features.json", s182["status"], "复用冻结扩展评分和同一受控注入；新增局部残差波动、相邻差分突变、滚动方差跃迁三类因果候选特征；权重和阈值只在valid选择，test只评价；候选不替换冻结主模型", f"逐行={s182['row_detail_rows']}；类型聚合={s182['type_aggregate_rows']}；difference_burst pooled recall：stuck={s182_diff.loc[s182_diff['attack_type'].eq('stuck'),'pooled_recall'].iloc[0]:.4f}、spike={s182_diff.loc[s182_diff['attack_type'].eq('spike'),'pooled_recall'].iloc[0]:.4f}、drift={s182_diff.loc[s182_diff['attack_type'].eq('drift'),'pooled_recall'].iloc[0]:.4f}", "候选特征带来非stuck信号，但valid选择的max-F1运行点FAR约1.6%-2.3%，不能直接替换冻结主模型；NOAA攻击为透明受控注入")
    s183_agg = pd.DataFrame(s183["type_aggregate"])
    s183_diag = s183_agg[
        s183_agg["feature"].eq("difference_burst")
        & s183_agg["max_far"].eq(0.05)
        & s183_agg["weight"].eq(0.2)
        & s183_agg["attack_type"].eq("spike")
    ].iloc[0]
    add(rows, "stage183_noaa_candidate_low_far_constraint", "183", "stage183_noaa_candidate_low_far_constraint.json", s183["status"], "复用阶段182逐行候选分数；在test正常行上按预先固定FAR上限做分位诊断；不重新训练、不根据test选择特征或权重；仅用于诊断候选上限，不可部署", f"运行点={s183['detail_rows']}；类型聚合={s183['type_aggregate_rows']}；difference_burst 5%FAR/weight0.2 spike pooled recall={s183_diag['pooled_attack_recall']:.4f}、事件覆盖={s183_diag['mean_event_coverage']:.4f}", "FAR阈值来自test正常行分位，只能作为上限诊断，不能部署或写成模型提升；NOAA攻击为透明受控注入")
    s184_agg = pd.DataFrame(s184["aggregate"])
    s184_pair = pd.DataFrame(s184["paired_aggregate"])
    s184_20 = s184_agg[
        s184_agg["feature"].eq("difference_burst")
        & s184_agg["max_far"].eq(0.02)
        & s184_agg["weight"].eq(0.2)
    ].iloc[0]
    s184_pair_20 = s184_pair[
        s184_pair["feature"].eq("difference_burst")
        & s184_pair["max_far"].eq(0.02)
        & s184_pair["weight"].eq(0.2)
        & s184_pair["metric"].eq("test_f1")
    ].iloc[0]
    add(rows, "stage184_noaa_validation_calibrated_low_far", "184", "stage184_noaa_validation_calibrated_low_far.json", s184["status"], "复用阶段182候选特征和阶段141冻结alpha/window；每个fold/seed只在valid正常样本上按预先固定FAR上限校准阈值，test只评价；不重新训练、不替换冻结主模型", f"分数明细={s184['score_detail_rows']}；运行点={s184['detail_rows']}；类型聚合={s184['type_aggregate_rows']}；difference_burst 2%FAR/weight0.2 pooled test recall={s184_20['pooled_test_recall']:.4f}、test FAR={s184_20['pooled_test_false_alarm_rate']:.4f}；相对冻结基线test F1平均差={s184_pair_20['mean_diff']:+.4f}", "验证集校准后增益明显缩小且折种子波动存在；候选证据不足以替换主模型；NOAA攻击为透明受控注入，不是真实攻击日志")
    s185_agg = pd.DataFrame(s185["aggregate"])
    s185_pair = pd.DataFrame(s185["paired"])
    s185_20 = s185_agg[
        s185_agg["selection_rule"].eq("valid_f1")
        & s185_agg["max_far"].eq(0.02)
    ].iloc[0]
    s185_pair_20 = s185_pair[
        s185_pair["selection_rule"].eq("valid_f1")
        & s185_pair["max_far"].eq(0.02)
        & s185_pair["metric"].eq("test_f1")
    ].iloc[0]
    add(rows, "stage185_noaa_validation_calibrated_combination", "185", "stage185_noaa_validation_calibrated_combination.json", s185["status"], "复用阶段184 valid/test候选分数；三类候选特征按非负权重组合，总权重不超过0.5；每个fold/seed只在valid上按预先声明的三种规则选择权重，并在固定FAR上限下用valid正常样本校准阈值；test只评价，不重新训练、不替换冻结主模型", f"组合数={s185['weight_combination_count']}；运行点={s185['detail_rows']}；类型明细={s185['type_detail_rows']}；valid_f1规则2%FAR test F1={s185_20['test_f1_mean']:.4f}、Recall={s185_20['test_recall_mean']:.4f}、非stuck事件覆盖宏平均={s185_20['test_nonstuck_event_macro_mean']:.4f}；相对冻结基线F1平均差={s185_pair_20['mean_diff']:+.4f}", "2%FAR下组合平均增益存在，但1%FAR下F1下降，权重选择分散，折种子波动大；候选组合仍不能替换冻结主模型；NOAA攻击为透明受控注入")
    s186_agg = pd.DataFrame(s186["aggregate"])
    s186_test = s186_agg[s186_agg["split"].eq("test")]
    s186_noise_spike = s186_test[
        s186_test["feature"].eq("noise_ratio")
        & s186_test["attack_type"].eq("spike")
    ].iloc[0]
    s186_repeat_stuck = s186_test[
        s186_test["feature"].eq("repeat_ratio")
        & s186_test["attack_type"].eq("stuck")
    ].iloc[0]
    add(rows, "stage186_noaa_targeted_nonstuck_feature_response", "186", "stage186_noaa_targeted_nonstuck_feature_response.json", s186["status"], "复用阶段178冻结评分协议和同一透明受控注入；新增8个针对非stuck类型设计的因果特征；诊断阈值只由train正常样本95%分位固定，valid和test只报告特征响应与事件覆盖；不重新训练、不根据test选择特征、权重或阈值", f"明细={s186['detail_rows']}；聚合={s186['aggregate_rows']}；相关矩阵={s186['correlation_rows']}；noise_ratio spike超过p95比例={s186_noise_spike['attack_above_p95_rate']:.4f}、事件覆盖={s186_noise_spike['mean_event_coverage']:.4f}；repeat_ratio stuck事件覆盖={s186_repeat_stuck['mean_event_coverage']:.4f}", "drift/offset/gaussian_noise等类型的新增特征响应较弱；特征响应强不等于低误报检测增益；仍需独立valid校准审计；NOAA攻击为透明受控注入，不是真实攻击日志")
    s187_agg = pd.DataFrame(s187["aggregate"])
    s187_pair = pd.DataFrame(s187["paired_aggregate"])
    s187_f1_repeat = s187_pair[
        s187_pair["feature"].eq("repeat_ratio")
        & s187_pair["max_far"].eq(0.02)
        & s187_pair["weight"].eq(0.1)
        & s187_pair["metric"].eq("test_f1")
    ].iloc[0]
    s187_nonstuck_noise = s187_pair[
        s187_pair["feature"].eq("noise_ratio")
        & s187_pair["max_far"].eq(0.02)
        & s187_pair["weight"].eq(0.5)
        & s187_pair["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    add(rows, "stage187_noaa_targeted_features_validation_calibrated", "187", "stage187_noaa_targeted_features_validation_calibrated.json", s187["status"], "复用阶段186八个定向因果特征和阶段141冻结alpha/window；每个fold/seed只在train正常样本上标准化，只在valid正常样本上按预先固定的FAR上限校准阈值；test只评价，不重新训练、不根据test选择特征、权重或阈值", f"分数明细={s187['score_detail_rows']}；运行点={s187['detail_rows']}；类型聚合={s187['type_aggregate_rows']}；repeat_ratio 2%FAR/weight0.1 test F1平均差={s187_f1_repeat['mean_diff']:+.4f}；noise_ratio 2%FAR/weight0.5非stuck事件覆盖平均差={s187_nonstuck_noise['mean_diff']:+.4f}", "repeat_ratio的2%FAR增益稳定但幅度很小；noise_ratio提高非stuck事件覆盖的同时抬高FAR；1%FAR下多数特征无稳定增益；候选不替换冻结主模型；NOAA攻击为透明受控注入，不是真实攻击日志")
    s188_agg = pd.DataFrame(s188["aggregate"])
    s188_pair = pd.DataFrame(s188["paired"])
    s188_f1_2 = s188_pair[
        s188_pair["selection_rule"].eq("valid_f1")
        & s188_pair["max_far"].eq(0.02)
        & s188_pair["metric"].eq("test_f1")
    ].iloc[0]
    s188_nonstuck_2 = s188_pair[
        s188_pair["selection_rule"].eq("valid_event_macro")
        & s188_pair["max_far"].eq(0.02)
        & s188_pair["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    add(rows, "stage188_noaa_two_feature_stability_audit", "188", "stage188_noaa_two_feature_stability_audit.json", s188["status"], "复用阶段187 valid/test候选分数，仅组合repeat_ratio和noise_ratio；非负权重总权重不超过0.5；每个fold/seed只在valid上按预先声明的三种规则选择权重，并在固定FAR上限下用valid正常样本校准阈值；test只评价", f"组合数={s188['weight_combination_count']}；运行点={s188['detail_rows']}；类型明细={s188['type_detail_rows']}；valid_f1规则2%FAR test F1平均差={s188_f1_2['mean_diff']:+.4f}（正向={s188_f1_2['positive_pairs']}/20）；valid_event_macro规则2%FAR非stuck事件覆盖平均差={s188_nonstuck_2['mean_diff']:+.4f}（正向={s188_nonstuck_2['positive_pairs']}/20，负向={s188_nonstuck_2['negative_pairs']}/20）", "两特征不互补：权重选择几乎总将repeat_ratio置0；组合在1%/2%FAR下拖低整体F1；5%FAR下F1小幅提升但运行点偏高；组合不替换冻结主模型；NOAA攻击为透明受控注入，不是真实攻击日志")
    s189_agg = pd.DataFrame(s189["overall_aggregate"])
    s189_q995 = s189_agg[
        s189_agg["q"].eq(0.995)
        & s189_agg["allocation_rule"].eq("size")
    ].iloc[0]
    s189_q999 = s189_agg[
        s189_agg["q"].eq(0.999)
        & s189_agg["allocation_rule"].eq("size")
    ].iloc[0]
    add(rows, "stage189_noaa_event_level_failure_decomposition", "189", "stage189_noaa_event_level_failure_decomposition.json", s189["status"], "复用阶段180冻结扩展风险的逐行test明细和事件明细；不重新训练、不改变模型、alpha、window、q、阈值或分配规则，不根据test选择任何规则；事件按ranking_failure和station_budget_suppression互斥分解", f"事件={s189['detail_rows']}；q=.995 size事件覆盖={s189_q995['event_coverage']:.4f}、排序失败率={s189_q995['ranking_failure_rate']:.4f}、站点预算抑制率={s189_q995['station_budget_suppression_rate']:.4f}；q=.999 size事件覆盖={s189_q999['event_coverage']:.4f}、排序失败率={s189_q999['ranking_failure_rate']:.4f}、站点预算抑制率={s189_q999['station_budget_suppression_rate']:.4f}", "低误报漏检主要由全局预算内事件排序失败造成，不是站点分配失败；事件级覆盖不能替代逐行F1/Recall/FAR；零站点预算和站内置换是诊断子类；NOAA攻击为透明受控注入，不是真实攻击日志")
    s190_agg = pd.DataFrame(s190["aggregate"])
    s190_pair = pd.DataFrame(s190["paired"])
    s190_f1_1 = s190_pair[
        s190_pair["selection_rule"].eq("valid_f1")
        & s190_pair["max_far"].eq(0.01)
        & s190_pair["metric"].eq("test_f1")
    ].iloc[0]
    s190_nonstuck_1 = s190_pair[
        s190_pair["selection_rule"].eq("valid_f1")
        & s190_pair["max_far"].eq(0.01)
        & s190_pair["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    add(rows, "stage190_noaa_event_reranking_validation", "190", "stage190_noaa_event_reranking_validation.json", s190["status"], "复用阶段187的noise_ratio valid/test候选分数，不重新训练；按站点和时间构造因果滚动最大值或均值，窗口预先固定为2/3/6；每个fold/seed只在valid上按valid_f1、全类型事件覆盖宏平均、非stuck事件覆盖宏平均三种规则选择窗口、聚合和权重，并在固定FAR上限下只用valid正常样本校准阈值；test只评价", f"运行点={s190['detail_rows']}；类型明细={s190['type_detail_rows']}；valid_f1规则1%FAR test F1平均差={s190_f1_1['mean_diff']:+.4f}（正向={s190_f1_1['positive_pairs']}/20，负向={s190_f1_1['negative_pairs']}/20）；非stuck事件覆盖平均差={s190_nonstuck_1['mean_diff']:+.4f}（正向={s190_nonstuck_1['positive_pairs']}/20，负向={s190_nonstuck_1['negative_pairs']}/20）", "事件级重排候选不替换冻结主模型；滚动聚合可能伤害stuck或短事件；1%FAR下F1平均差为正但折种子正负各半；NOAA攻击为透明受控注入，不是真实攻击日志")
    s191_consensus = pd.DataFrame(s191["consensus"])
    s191_type = pd.DataFrame(s191["type_aggregate"])
    s191_overall = pd.DataFrame(s191["overall_aggregate"])
    s191_f1_consensus = s191_consensus[
        s191_consensus["selection_rule"].eq("valid_f1")
        & s191_consensus["max_far"].eq(0.01)
    ].iloc[0]
    s191_spike = s191_type[
        s191_type["selection_rule"].eq("valid_f1")
        & s191_type["max_far"].eq(0.01)
        & s191_type["attack_type"].eq("spike")
    ].iloc[0]
    s191_stuck = s191_type[
        s191_type["selection_rule"].eq("valid_f1")
        & s191_type["max_far"].eq(0.01)
        & s191_type["attack_type"].eq("stuck")
    ].iloc[0]
    s191_overall_f1 = s191_overall[
        s191_overall["selection_rule"].eq("valid_f1")
        & s191_overall["max_far"].eq(0.01)
    ].iloc[0]
    add(rows, "stage191_noaa_reranking_focused_diagnostics", "191", "stage191_noaa_reranking_focused_diagnostics.json", s191["status"], "复用阶段190的valid-only选择和test评价结果，不重新训练、不重新选择窗口、聚合、权重或阈值；基线阈值只由阶段187 valid正常样本按固定FAR上限重算；只诊断1%/2%FAR下的配置集中度、攻击类型增益和stuck代价；test只评价", f"valid_f1 1%FAR共识配置=窗口{s191_f1_consensus['consensus_window']}/{s191_f1_consensus['consensus_aggregation']}/候选权重{s191_f1_consensus['consensus_candidate_weight']:.1f}，占比={s191_f1_consensus['consensus_share']:.2f}，不同配置={s191_f1_consensus['distinct_settings']}；spike pooled Recall差={s191_spike['pooled_delta_attack_recall']:+.4f}；stuck pooled Recall差={s191_stuck['pooled_delta_attack_recall']:+.4f}；F1正/负={int(s191_overall_f1['positive_f1_pairs'])}/{int(s191_overall_f1['negative_f1_pairs'])}", "收益主要来自spike，drift/offset幅度较小，time_shift仍为负；stuck逐行召回下降但事件覆盖不变；折种子配对不是独立样本；本阶段只解释阶段190，不替换冻结主模型；NOAA攻击为透明受控注入，不是真实攻击日志")
    s192_pair = pd.DataFrame(s192["paired"])
    s192_f1_2 = s192_pair[
        s192_pair["selection_rule"].eq("valid_f1")
        & s192_pair["max_far"].eq(0.02)
        & s192_pair["metric"].eq("test_f1")
    ].iloc[0]
    s192_nonstuck_2 = s192_pair[
        s192_pair["selection_rule"].eq("valid_f1")
        & s192_pair["max_far"].eq(0.02)
        & s192_pair["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    add(rows, "stage192_noaa_repeat_gated_reranking", "192", "stage192_noaa_repeat_gated_reranking.json", s192["status"], "复用阶段187的noise_ratio和repeat_ratio候选分数，不重新训练；repeat_ratio门控阈值只由valid正常样本按预先固定分位0.5/0.75/0.9/0.95校准；每个fold/seed只在valid上选择门控分位、窗口、聚合和权重，并在固定FAR上限下只用valid正常样本校准阈值；test只评价", f"运行点={s192['detail_rows']}；类型明细={s192['type_detail_rows']}；valid_f1规则2%FAR test F1平均差={s192_f1_2['mean_diff']:+.4f}（正向={s192_f1_2['positive_pairs']}/20）；非stuck事件覆盖平均差={s192_nonstuck_2['mean_diff']:+.4f}（正向={s192_nonstuck_2['positive_pairs']}/20，负向={s192_nonstuck_2['negative_pairs']}/20）", "负结果：1%FAR下20/20折种子选择候选权重0，门控退化为基线；2%FAR下保住stuck行级召回，但spike增益明显缩小，非stuck事件覆盖多数折种子下降；门控候选不替换冻结主模型；NOAA攻击为透明受控注入，不是真实攻击日志")
    s193_pair = pd.DataFrame(s193["paired"])
    s193_f1_1 = s193_pair[
        s193_pair["selection_rule"].eq("valid_f1")
        & s193_pair["max_far"].eq(0.01)
        & s193_pair["metric"].eq("test_f1")
    ].iloc[0]
    s193_nonstuck_1 = s193_pair[
        s193_pair["selection_rule"].eq("valid_f1")
        & s193_pair["max_far"].eq(0.01)
        & s193_pair["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    s193_stability = pd.DataFrame(s193["stability"])
    s193_protected_1 = s193_stability[
        s193_stability["selection_rule"].eq("valid_f1")
        & s193_stability["max_far"].eq(0.01)
    ].iloc[0]
    add(rows, "stage193_noaa_persistent_repeat_protection", "193", "stage193_noaa_persistent_repeat_protection.json", s193["status"], "复用阶段187的noise_ratio和repeat_ratio候选分数，不重新训练；repeat_ratio门控阈值只由valid正常样本按预先固定分位0.75/0.9/0.95校准；连续重复长度预先固定为2/3；候选集合同时包含不门控选项，因此严格覆盖阶段190的重排空间；每个fold/seed只在valid选择门控分位、连续长度、窗口、聚合和权重，并在固定FAR上限下只用valid正常样本校准阈值；test只评价", f"运行点={s193['detail_rows']}；类型明细={s193['type_detail_rows']}；valid_f1规则1%FAR test F1平均差={s193_f1_1['mean_diff']:+.4f}（正向={s193_f1_1['positive_pairs']}/20，负向={s193_f1_1['negative_pairs']}/20）；非stuck事件覆盖平均差={s193_nonstuck_1['mean_diff']:+.4f}（正向={s193_nonstuck_1['positive_pairs']}/20，负向={s193_nonstuck_1['negative_pairs']}/20）；采用保护折种子={int(s193_protected_1['protected_pairs'])}/20", "连续重复保护在valid上被部分折种子采用，1%FAR下保住stuck行级召回并保留非stuck事件覆盖增益；但门控候选仍不替换冻结主模型，跨数据源普适性未证明；连续重复保护可能误伤真实长时段突刺；NOAA攻击为透明受控注入，不是真实攻击日志")
    s194_pair = pd.DataFrame(s194["paired"])
    s194_f1_1 = s194_pair[
        s194_pair["selection_rule"].eq("valid_f1")
        & s194_pair["max_far"].eq(0.01)
        & s194_pair["metric"].eq("test_f1")
    ].iloc[0]
    s194_nonstuck_1 = s194_pair[
        s194_pair["selection_rule"].eq("valid_f1")
        & s194_pair["max_far"].eq(0.01)
        & s194_pair["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    s194_primary = pd.DataFrame(s194["aggregate"])
    s194_primary = s194_primary[
        s194_primary["selection_rule"].eq("valid_f1")
        & s194_primary["max_far"].eq(0.01)
    ]
    add(rows, "stage194_noaa_position_perturbation_robustness", "194", "stage194_noaa_position_perturbation_robustness.json", s194["status"], "复用阶段193的连续重复保护候选空间；攻击位置在预先固定的对称偏移网格-0.06/-0.03/0/+0.03/+0.06上重建整条流程；每个fold/偏移只在valid选择门控分位、连续长度、窗口、聚合和权重，并在固定FAR上限下只用valid正常样本校准阈值；test只评价；不修改也不替换冻结协议与冻结主模型", f"重建单元={s194['detail_rows']}；位置偏移数={len(s194['position_shifts'])}；valid_f1规则1%FAR跨位置 test F1平均差={s194_f1_1['mean_diff']:+.4f}（正向={s194_f1_1['positive_folds']}/20，负向={s194_f1_1['negative_folds']}/20）；非stuck事件覆盖平均差={s194_nonstuck_1['mean_diff']:+.4f}（正向={s194_nonstuck_1['positive_folds']}/20，负向={s194_nonstuck_1['negative_folds']}/20）；逐位置F1平均差范围={s194_primary['test_f1_diff_mean'].min():+.4f}至{s194_primary['test_f1_diff_mean'].max():+.4f}", "位置扰动下总体方向仍为正，但+0.03偏移处F1平均差为负，说明候选并非对所有位置都稳健；现有受控协议中choose_start不依赖种子，同一折内五种子攻击位置完全相同，位置冗余审计已记录该局限；位置敏感性只覆盖当前NOAA站点、年份与受控注入生成器；NOAA攻击为透明受控注入，不是真实攻击日志")
    s195_pair = pd.DataFrame(s195["paired"])
    s195_f1_1 = s195_pair[
        s195_pair["selection_rule"].eq("valid_f1")
        & s195_pair["max_far"].eq(0.01)
        & s195_pair["metric"].eq("test_f1")
    ].iloc[0]
    s195_nonstuck_1 = s195_pair[
        s195_pair["selection_rule"].eq("valid_f1")
        & s195_pair["max_far"].eq(0.01)
        & s195_pair["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    s195_aggregate = pd.DataFrame(s195["aggregate"])
    s195_primary = s195_aggregate[
        s195_aggregate["selection_rule"].eq("valid_f1")
        & s195_aggregate["max_far"].eq(0.01)
    ].iloc[0]
    add(rows, "stage195_noaa_seed_randomized_placement", "195", "stage195_noaa_seed_randomized_placement.json", s195["status"], "每个fold/seed把每个split按攻击类型划分成预先固定的连续区块，每个攻击在自己的区块内按种子均匀随机选择起点；复用阶段193的连续重复保护候选空间；每个fold/seed只在valid选择门控分位、连续长度、窗口、聚合和权重，并在固定FAR上限下只用valid正常样本校准阈值；test只评价；不修改也不替换冻结协议与冻结主模型", f"评估行={s195['detail_rows']}；位置明细={s195['placement_detail_rows']}；位置摘要={s195['placement_summary_rows']}；valid_f1规则1%FAR test F1平均差={s195_f1_1['mean_diff']:+.4f}（正向={s195_f1_1['positive_pairs']}/20，负向={s195_f1_1['negative_pairs']}/20）；非stuck事件覆盖平均差={s195_nonstuck_1['mean_diff']:+.4f}（正向={s195_nonstuck_1['positive_pairs']}/20，负向={s195_nonstuck_1['negative_pairs']}/20）；采用保护折种子={int(s195_primary['protected_pairs'])}/20", "种子随机化后阶段193候选在1%FAR下方向仍为正，但仍有负向折种子，且位置随机化只覆盖当前NOAA站点、年份与受控注入生成器；候选不替换冻结主模型；NOAA攻击为透明受控注入，不是真实攻击日志")
    s196_pair = pd.DataFrame(s196["paired"])
    s196_f1_1 = s196_pair[
        s196_pair["selection_rule"].eq("valid_f1")
        & s196_pair["max_far"].eq(0.01)
        & s196_pair["metric"].eq("test_f1")
    ].iloc[0]
    s196_nonstuck_1 = s196_pair[
        s196_pair["selection_rule"].eq("valid_f1")
        & s196_pair["max_far"].eq(0.01)
        & s196_pair["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    s196_aggregate = pd.DataFrame(s196["aggregate"])
    s196_primary = s196_aggregate[
        s196_aggregate["selection_rule"].eq("valid_f1")
        & s196_aggregate["max_far"].eq(0.01)
    ].iloc[0]
    add(rows, "stage196_noaa_seed_randomized_independent_replication", "196", "stage196_noaa_seed_randomized_independent_replication.json", s196["status"], "在预先冻结的一组全新种子上重复阶段195的种子随机化攻击位置协议；每个fold/seed只在valid选择门控分位、连续长度、窗口、聚合和权重，并在固定FAR上限下只用valid正常样本校准阈值；test只评价；不修改也不替换冻结协议与冻结主模型", f"独立种子={len(s196['replication_seeds'])}；评估行={s196['detail_rows']}；位置明细={s196['placement_detail_rows']}；valid_f1规则1%FAR test F1平均差={s196_f1_1['mean_diff']:+.4f}（正向={s196_f1_1['positive_pairs']}/20，负向={s196_f1_1['negative_pairs']}/20）；非stuck事件覆盖平均差={s196_nonstuck_1['mean_diff']:+.4f}（正向={s196_nonstuck_1['positive_pairs']}/20，负向={s196_nonstuck_1['negative_pairs']}/20）；采用保护折种子={int(s196_primary['protected_pairs'])}/20", "新种子复现支持阶段195的方向性结论，但新种子仍来自同一NOAA数据源、年份和受控注入生成器；2%FAR下F1平均差接近零，说明运行点依赖仍存在；候选不替换冻结主模型；NOAA攻击为透明受控注入，不是真实攻击日志")
    add(rows, "stage197_noaa_seed_randomized_station_dependence", "197", "stage197_noaa_seed_randomized_station_dependence_independent.json", s197i["status"], "复用阶段195/196的逐行结果做站点依赖审计；逐站点和留一站点指标只用于诊断；每个fold/seed仍只在valid选择门控分位、连续长度、窗口、聚合和权重，并在固定FAR上限下只用valid正常样本校准阈值；test只评价；不根据test选择", f"原种子队列1%FAR F1平均差为正站点={s197o['primary_f1_positive_stations']}/{s197o['primary_station_count']}、负向={s197o['primary_f1_negative_stations']}；独立种子队列正向={s197i['primary_f1_positive_stations']}/{s197i['primary_station_count']}、负向={s197i['primary_f1_negative_stations']}；独立队列留一站点F1正向={s197i['loso_f1_positive_rows']}/{s197i['loso_rows']}、负向={s197i['loso_f1_negative_rows']}/{s197i['loso_rows']}；独立队列留一站点非stuck覆盖正向={s197i['loso_nonstuck_positive_rows']}/{s197i['loso_rows']}、负向={s197i['loso_nonstuck_negative_rows']}/{s197i['loso_rows']}", "站点依赖审计显示F1增益不是只由单个站点支撑，但仍有少数站点和留一行F1为负；站点级F1在小样本站点上波动较大，留一站点不能替代跨数据源外部验证；候选不替换冻结主模型；NOAA攻击为透明受控注入，不是真实攻击日志")
    s198o_pair = pd.DataFrame(s198o["paired"])
    s198i_pair = pd.DataFrame(s198i["paired"])
    s198o_f1_1 = s198o_pair[
        s198o_pair["selection_rule"].eq("valid_f1")
        & s198o_pair["max_far"].eq(0.01)
        & s198o_pair["metric"].eq("test_f1")
    ].iloc[0]
    s198i_f1_1 = s198i_pair[
        s198i_pair["selection_rule"].eq("valid_f1")
        & s198i_pair["max_far"].eq(0.01)
        & s198i_pair["metric"].eq("test_f1")
    ].iloc[0]
    s198o_nonstuck_1 = s198o_pair[
        s198o_pair["selection_rule"].eq("valid_f1")
        & s198o_pair["max_far"].eq(0.01)
        & s198o_pair["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    s198i_nonstuck_1 = s198i_pair[
        s198i_pair["selection_rule"].eq("valid_f1")
        & s198i_pair["max_far"].eq(0.01)
        & s198i_pair["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    add(rows, "stage198_noaa_seed_randomized_cross_fold_transfer", "198", "stage198_noaa_seed_randomized_cross_fold_transfer_independent.json", s198i["status"], "对每个留出折只在其余三个折的valid结果上选择一个固定门控分位、连续长度、窗口、聚合和权重；留出折自身的valid正常样本只用于按固定FAR上限校准阈值；留出折test只评价；不根据test选择；不修改也不替换冻结协议与冻结主模型", f"原种子队列1%FAR test F1平均差={s198o_f1_1['mean_diff']:+.4f}（正向={s198o_f1_1['positive_pairs']}/20，负向={s198o_f1_1['negative_pairs']}/20）；非stuck事件覆盖平均差={s198o_nonstuck_1['mean_diff']:+.4f}（正向={s198o_nonstuck_1['positive_pairs']}/20，负向={s198o_nonstuck_1['negative_pairs']}/20）；独立种子队列F1平均差={s198i_f1_1['mean_diff']:+.4f}（正向={s198i_f1_1['positive_pairs']}/20，负向={s198i_f1_1['negative_pairs']}/20）；非stuck事件覆盖平均差={s198i_nonstuck_1['mean_diff']:+.4f}（正向={s198i_nonstuck_1['positive_pairs']}/20，负向={s198i_nonstuck_1['negative_pairs']}/20）；两队列均为240条折种子评价、48条固定选择", "跨折迁移减少折内选择自由，两队列在1%FAR下方向一致，但F1仍有3/40个轻微负向折种子，2%FAR下部分规则不再占优；跨折只覆盖当前NOAA站点、年份和受控注入生成器，不能替代独立数据源验证；候选不替换冻结主模型；NOAA攻击为透明受控注入，不是真实攻击日志")
    s199_aggregate = pd.DataFrame(s199["aggregate"])
    s199_primary = s199_aggregate[
        s199_aggregate["selection_rule"].eq("valid_f1")
        & s199_aggregate["max_far"].eq(0.01)
    ].iloc[0]
    s199_paired = pd.DataFrame(s199["paired"])
    s199_f1_1 = s199_paired[
        s199_paired["selection_rule"].eq("valid_f1")
        & s199_paired["max_far"].eq(0.01)
        & s199_paired["metric"].eq("test_f1")
    ].iloc[0]
    s199_nonstuck_1 = s199_paired[
        s199_paired["selection_rule"].eq("valid_f1")
        & s199_paired["max_far"].eq(0.01)
        & s199_paired["metric"].eq("test_nonstuck_event_macro")
    ].iloc[0]
    add(rows, "stage199_liupanshan_cross_station_protection_transfer", "199", "stage199_liupanshan_cross_station_protection_transfer.json", s199["status"], "对每个攻击构造种子和留出站点，只在其余六个源站点的valid结果上选择一个固定保护配置；留出站点不参与特征标准化、配置选择或阈值校准；留出站点test只评价；不根据test选择；不修改也不替换冻结主模型", f"组合数={s199['detail_rows']}（5种子×7站点×12规则/FAR）；1%FAR valid_f1 test F1平均差={s199_f1_1['mean_diff']:+.6f}（正向={s199_f1_1['positive_pairs']}/35，负向={s199_f1_1['negative_pairs']}/35）；非stuck事件覆盖平均差={s199_nonstuck_1['mean_diff']:+.6f}（正向={s199_nonstuck_1['positive_pairs']}/35，负向={s199_nonstuck_1['negative_pairs']}/35）；采用保护={int(s199_primary['protected_pairs'])}/35", "这是保留的负结果：连续重复保护配置从源站点迁移到留出站点后，1%FAR下F1和非stuck覆盖平均均略为负，不能宣称跨站点稳定增益；六盘山仍是单一时段、单一地区的受控攻击数据，五个攻击构造种子不等于五个独立地区或年份；攻击为透明受控注入，不是真实攻击日志")
    add(rows, "stage200_liupanshan_protection_failure_attribution", "200", "stage200_liupanshan_protection_failure_attribution.json", s200["status"], "复用阶段199冻结配置和冻结分数，只比较源站阈值与使用留出站点test正常标签的oracle阈值；oracle只做失败归因，不参与模型、配置、阈值或规则选择", f"1%FAR valid_f1：源站迁移相对基线F1平均差={s200['primary_source_minus_baseline_f1']:+.6f}；oracle阈值相对源站阈值F1平均增益={s200['primary_oracle_gain_f1']:+.6f}（正向={s200['primary_oracle_gain_f1_positive_pairs']}/35，负向={s200['primary_oracle_gain_f1_negative_pairs']}/35）；非stuck平均增益={s200['primary_oracle_gain_nonstuck']:+.6f}（正向={s200['primary_oracle_gain_nonstuck_positive_pairs']}/35，负向={s200['primary_oracle_gain_nonstuck_negative_pairs']}/35）；平均阈值偏移={s200['mean_threshold_shift']:+.6f}", "oracle阈值使用test正常标签，只证明阈值偏移不是主要失败来源，不能作为可部署方法或论文性能结果；非stuck覆盖在oracle下仍为负，保护配置跨站点本身不成立；六盘山是单一时段、单一地区的受控攻击数据，攻击为透明受控注入，不是真实攻击日志")
    add(rows, "stage201_liupanshan_conservative_protection_transfer", "201", "stage201_liupanshan_conservative_protection_transfer.json", s201["status"], "预先固定的保守迁移规则：仅当保护候选在六个源站valid上相对无门控基线同时满足F1增益和非stuck覆盖不退化时才启用保护，否则退回冻结基线；配置选择只用源站valid，留出站点test只评价", f"组合数={s201['detail_rows']}（5种子×7站点×12规则/FAR）；1%FAR valid_f1采用保护={s201['primary_protected_pairs']}/35；F1平均差={s201['primary_f1_diff']:+.6f}（正向={s201['primary_f1_positive_pairs']}/35，负向={s201['primary_f1_negative_pairs']}/35）；非stuck事件覆盖平均差={s201['primary_nonstuck_diff']:+.6f}（正向={s201['primary_nonstuck_positive_pairs']}/35，负向={s201['primary_nonstuck_negative_pairs']}/35）", "这是第二个保留的负结果：保守规则提高了源站valid准入条件并退回部分基线，但跨站点非stuck事件覆盖仍略为负，不能证明跨站点稳定增益；六盘山是单一时段、单一地区的受控攻击数据，五个攻击构造种子不等于五个独立地区或年份；攻击为透明受控注入，不是真实攻击日志")
    s202_primary = [
        row
        for row in s202["tests"]
        if float(row["max_far"]) == float(s202["primary_far"])
    ]
    s202_stuck = next(
        row for row in s202_primary if row["attack_type"] == "stuck"
    )
    add(rows, "stage202_liupanshan_attack_type_station_decomposition", "202", "stage202_liupanshan_attack_type_station_decomposition.json", s202["status"], "复用阶段199 valid_f1冻结配置和固定FAR阈值；每个FAR×攻击类型在留出站点内先平均五个固定种子，再做站点级精确sign-flip检验，并对24个主假设做BH校正；不重新选择候选、阈值、特征、站点、攻击类型或事件；留出站点test只评价", f"分解单元={s202['test_rows']}（4 FAR×6攻击类型）；1%FAR下事件覆盖差最大的标准类型为stuck：{s202_stuck['event_coverage_diff_mean']:+.6f}（正向站点={int(s202_stuck['event_coverage_positive_stations'])}/{int(s202_stuck['stations'])}，站点级p={s202_stuck['event_coverage_sign_flip_p']:.6f}，BH q={s202_stuck['event_coverage_bh_q']:.6f}）；1%FAR稳定正事件覆盖单元={s202['primary_event_stable_positive_cells']}/6，稳定正行召回单元={s202['primary_row_recall_stable_positive_cells']}/6", "这是第三个保留的负结果：没有攻击类型在站点聚类和BH校正后稳定为正，stuck的正向均值不能写成显著或可迁移增益；六盘山冻结test中的受控攻击事件当前均为单行事件，因此事件覆盖与行召回数值完全一致；六盘山是单一时段、单一地区的受控攻击数据，攻击为透明受控注入，不是真实攻击日志")
    add(rows, "stage203_independent_external_evidence_feasibility", "203", "stage203_independent_external_evidence_feasibility.json", s203["status"], "只审计现有外部来源的独立性、攻击标签状态和跨来源可用性；不训练模型、不选择阈值、不下载新数据、不计算新性能结果；要求不同机构、不同传感器网络、不同处理链的现场观测配合预先冻结的透明攻击协议", f"现有来源={s203['inventory_rows']}；最低要求={s203['requirement_rows']}；候选官方来源={s203['candidate_rows']}；满足跨机构/跨传感器网络独立外部验证的来源={s203['usable_independent_external_sources']}；独立外部验证已完成={s203['independent_external_validation_completed']}", "这是保留的证据缺口：NOAA ISD 2020-2025的12站和2020-2025年份划分仍是同源内部留出；六盘山7站是主实验内部站点留出；ERA5是不同来源的辅助再分析背景，不是独立现场传感器真值；候选MeteoSwiss/KNMI/DWD尚未下载验证，不能写成已完成外部验证")
    add(rows, "stage204_dwd_protocol_preregistration", "204", "stage204_dwd_protocol_preregistration.json", s204["status"], "在下载任何DWD观测前冻结跨机构外部验证协议：来源、站点选择、年份、变量、质量码、透明攻击、种子、FAR、主终点、聚类单位和多重校正；不训练模型、不选择阈值、不计算性能", f"候选站点池={s204['eligible_station_count']}；冻结站点={s204['station_count']}；年份={s204['years']}；主FAR={s204['primary_far']}；下载状态={s204['download_status']}；性能状态={s204['performance_status']}", "这是预注册证据，不是性能结果；DWD尚未下载和验证，不能写成已完成跨机构外部验证；攻击仍须是透明受控注入，不能描述为真实攻击日志")
    add(rows, "stage204a_dwd_protocol_schema_correction", "204A", "stage204a_dwd_protocol_schema_correction.json", s204a["status"], "在计算任何DWD性能之前，依据官方归档元数据更正产品字段记录；不改变站点、年份、划分、攻击、FAR、终点、聚类或多重校正；不训练模型、不选择阈值、不计算性能", f"主湿度变量={s204a['corrected_primary_humidity']['field']}；质量码={s204a['corrected_primary_humidity']['quality_field']}；辅助TF归档={s204a['tf_archives_retained_as_auxiliary']}；性能状态={s204a['performance_status']}", "这是下载后、性能评价前的字段更正记录，不是性能结果；TF归档仅辅助保留，不能替代TU中的RF_TU；DWD攻击仍须是透明受控注入")
    add(rows, "stage205_dwd_official_download", "205", "stage205_dwd_official_download.json", s205["status"], "只从DWD官方OpenData历史目录下载阶段204冻结的12个站点、TU主产品和TF辅助归档；保留URL、时间、字节数、SHA256、成员、字段、时间范围和站号；不训练模型、不选择阈值、不计算性能", f"预期归档={s205['expected_archives']}；主产品={s205['primary_archives']}；辅助归档={s205['auxiliary_archives']}；已存在={s205['already_present_count']}；失败={s205['failed_count']}；字段通过={s205['required_columns_ok_count']}", "这是可追溯下载与归档结构审计，不是模型性能；TF归档不进入主分析；DWD质量码不是攻击标签；攻击仍是透明受控注入")
    add(rows, "stage206_dwd_normalized", "206", "stage206_dwd_normalized.json", s206["status"], "只标准化阶段204冻结的12个DWD站点和阶段205下载的TU主产品；按2021-2023/2024/2025切分；9990.0转缺失并保留QN_9；不插补、不训练模型、不选择阈值、不计算性能", f"站点={s206['station_count']}；行数={s206['rows']}；缺测行={s206['missing_observation_rows']}；重复={s206['duplicate_station_timestamp_rows']}；质量接受={s206['quality_accepted_rows']}；质量拒绝(观测)={s206['quality_rejected_observed_rows']}；性能状态={s206['performance_status']}", "这是标准化数据审计，不是模型性能；499个原始断档被显式保留为缺测，没有插补；DWD质量码不是攻击标签；攻击仍须透明受控注入")
    s207_stuck = next(
        row
        for row in s207["type_aggregate"]
        if row["attack_type"] == "stuck"
    )
    add(rows, "stage207_dwd_controlled_attack_negative_result", "207", "stage207_dwd_controlled_attack.json", s207["status"], "严格复用阶段204冻结协议和阶段207运行前固定的攻击参数；三个无监督分量只在train正常段拟合，valid正常段按FAR网格校准，2025 test只评价；攻击块按站点真实全表索引非重叠注入，time_shift只读取阶段206干净序列；不根据test选择组件、权重、阈值、站点或攻击位置", f"主FAR={s207['primary_far']}；评价行={s207['evaluation_rows']}；事件行={s207['event_rows']}；平均事件覆盖={sum(x['event_coverage'] for x in s207['primary_event_metrics'])/len(s207['primary_event_metrics']):.6f}；平均行召回={sum(x['row_recall'] for x in s207['primary_event_metrics'])/len(s207['primary_event_metrics']):.6f}；stuck事件覆盖={s207_stuck['event_coverage']:.6f}；stuck行召回={s207_stuck['row_recall']:.6f}", "这是DWD跨机构外部验证的第一份修正后负结果，不能写成全面领先；修正前的站内索引误用会把攻击行写到其它站点并虚高事件覆盖，修正后的数字才是可追溯结果；stuck在五个种子上事件覆盖为0，其余多数类型也偏弱；攻击是透明受控注入，不是真实攻击日志；DWD质量码不是攻击标签")
    s208_primary = pd.DataFrame(s208["primary_aggregate"])
    s208_paired = pd.DataFrame(s208["primary_paired"])
    s208_f1 = s208_primary[
        s208_primary["selection_rule"].eq("valid_f1")
    ].iloc[0]
    s208_f1_pair = s208_paired[
        s208_paired["selection_rule"].eq("valid_f1")
        & s208_paired["metric"].eq("test_f1")
    ].iloc[0]
    s208_event_pair = s208_paired[
        s208_paired["selection_rule"].eq("valid_f1")
        & s208_paired["metric"].eq("test_event_coverage")
    ].iloc[0]
    s208_stuck = next(
        row
        for row in s208["primary_type_aggregate"]
        if row["selection_rule"] == "valid_f1"
        and row["attack_type"] == "stuck"
    )
    add(rows, "stage208_dwd_persistent_repeat_protection", "208", "stage208_dwd_persistent_repeat_protection.json", s208["status"], "保留阶段207冻结基线；valid和test分别构造透明受控攻击序列，避免valid攻击通过顺序检测器污染test；repeat_ratio和repeat_run的窗口与权重只在valid选择，test只评价；三个分量仍只在train正常段拟合，valid正常段按固定FAR校准阈值；不根据test选择特征、权重、窗口、阈值、站点或攻击位置", f"主FAR={s208['primary_far']}；候选={s208['candidate_count']}；评价行={s208['evaluation_rows']}；事件行={s208['event_rows']}；valid_f1规则test F1={s208_f1['test_f1']:.6f}（相对阶段207 {s208_f1_pair['mean_diff']:+.6f}，正向{s208_f1_pair['positive_pairs']}/{s208_f1_pair['pairs']}）；事件覆盖={s208_f1['test_event_coverage']:.6f}（相对阶段207 {s208_event_pair['mean_diff']:+.6f}，正向{s208_event_pair['positive_pairs']}/{s208_event_pair['pairs']}）；stuck事件覆盖={s208_stuck['event_coverage']:.6f}", "这是DWD外部验证上的正向候选，但仍是透明受控注入，不是真实攻击日志；改善集中在stuck，其他攻击类型仍有低行召回；valid选择不能替代外部独立复现；阶段207修正后的负结果必须并列保留，不能选择性隐藏")
    s209_f1 = next(
        row
        for row in s209["bootstrap"]
        if row["metric"] == "f1"
    )
    s209_event = next(
        row
        for row in s209["bootstrap"]
        if row["metric"] == "event_coverage"
    )
    s209_f1_paired = next(
        row
        for row in s209["paired_station"]
        if row["metric"] == "f1"
    )
    add(rows, "stage209_dwd_station_cluster_uncertainty", "209", "stage209_dwd_station_cluster_uncertainty.json", s209["status"], "只读取阶段208固定valid_f1规则、主FAR和test站点级明细；不重新拟合、不重新选择、不重新校准阈值；以12个DWD站点为重采样簇，五个种子作为站点内重复测量，对候选与阶段207做配对bootstrap", f"站点={len(s209['stations'])}；种子={len(s209['seeds'])}；bootstrap={s209['iterations']}；F1差值={s209_f1['difference_mean']:+.6f}，CI95=[{s209_f1['difference_ci95'][0]:+.6f},{s209_f1['difference_ci95'][1]:+.6f}]，候选更高概率={s209_f1['candidate_greater_probability']:.6f}；事件覆盖差值={s209_event['difference_mean']:+.6f}，CI95=[{s209_event['difference_ci95'][0]:+.6f},{s209_event['difference_ci95'][1]:+.6f}]；F1正向站点={s209_f1_paired['positive_stations']}/{s209_f1_paired['stations']}", "只有12个DWD站点，不能替代独立站点外部复现；五个种子不是五套独立真实攻击数据；所有攻击都是透明受控注入；该区间不允许用于test回看改选")
    s210_f1 = next(
        row
        for row in s210["paired"]
        if row["metric"] == "test_f1"
    )
    s210_event = next(
        row
        for row in s210["paired"]
        if row["metric"] == "event_coverage"
    )
    add(rows, "stage210_dwd_seed_randomized_replication", "210", "stage210_dwd_seed_randomized_replication.json", s210["status"], "保持阶段208的候选特征族、窗口、权重网格、valid_f1规则和train/valid/test边界不变；预先声明五个不与原种子重叠的新攻击块种子；特征、窗口、权重和阈值仍只在2024 valid选择，2025 test只评价", f"复现种子={s210['replication_seeds']}；F1候选={s210['summary']['candidate_test_f1_mean']:.6f}，基线={s210['summary']['baseline_test_f1_mean']:.6f}，差值={s210_f1['mean_diff']:+.6f}，正向={s210_f1['positive_pairs']}/{s210_f1['pairs']}；事件覆盖候选={s210['summary']['candidate_event_coverage_mean']:.6f}，基线={s210['summary']['baseline_event_coverage_mean']:.6f}，差值={s210_event['mean_diff']:+.6f}，正向={s210_event['positive_pairs']}/{s210_event['pairs']}；行召回差值={next(row['mean_diff'] for row in s210['paired'] if row['metric'] == 'row_recall'):+.6f}", "新种子只改变攻击块位置，不是新的机构、站点或真实攻击数据；事件覆盖在复现中为4/5种子正向，不能写成完全稳定；所有攻击都是透明受控注入")
    s211_f1 = next(
        row
        for row in s211["bootstrap"]
        if row["metric"] == "f1"
    )
    s211_event = next(
        row
        for row in s211["bootstrap"]
        if row["metric"] == "event_coverage"
    )
    s211_f1_paired = next(
        row
        for row in s211["paired_station"]
        if row["metric"] == "f1"
    )
    s211_event_paired = next(
        row
        for row in s211["paired_station"]
        if row["metric"] == "event_coverage"
    )
    add(rows, "stage211_dwd_replication_station_uncertainty", "211", "stage211_dwd_replication_station_uncertainty.json", s211["status"], "只读取阶段210固定valid_f1规则、主FAR和test站点级明细；不重新拟合、不重新选择、不重新校准阈值；以12个DWD站点为重采样簇，五个新种子作为站点内重复测量，对候选与阶段207做配对bootstrap", f"F1差值={s211_f1['difference_mean']:+.6f}，CI95=[{s211_f1['difference_ci95'][0]:+.6f},{s211_f1['difference_ci95'][1]:+.6f}]，候选更高概率={s211_f1['candidate_greater_probability']:.6f}，F1正向站点={s211_f1_paired['positive_stations']}/{s211_f1_paired['stations']}；事件覆盖差值={s211_event['difference_mean']:+.6f}，CI95=[{s211_event['difference_ci95'][0]:+.6f},{s211_event['difference_ci95'][1]:+.6f}]，候选更高概率={s211_event['candidate_greater_probability']:.6f}，事件覆盖正向站点={s211_event_paired['positive_stations']}/{s211_event_paired['stations']}", "只有12个DWD站点；新种子不是新的机构、站点或真实攻击数据；事件覆盖在站点方向为7正2负3零，不能宣称完全稳定；不能替代独立站点外部复现")
    s212_primary = [
        row
        for row in s212["curve"]
        if row["metric"] == "f1" and row["is_primary"]
    ]
    s212_low = [
        row
        for row in s212["curve"]
        if row["metric"] == "f1" and float(row["max_far"]) == 0.005
    ]
    s212_primary_text = "; ".join(
        "{label}={value:+.6f} CI95=[{low:+.6f},{high:+.6f}]".format(
            label=row["source_label"],
            value=row["difference_mean"],
            low=row["difference_ci95"][0],
            high=row["difference_ci95"][1],
        )
        for row in s212_primary
    )
    s212_low_text = "; ".join(
        "{label}={value:+.6f} CI95=[{low:+.6f},{high:+.6f}]".format(
            label=row["source_label"],
            value=row["difference_mean"],
            low=row["difference_ci95"][0],
            high=row["difference_ci95"][1],
        )
        for row in s212_low
    )
    add(rows, "stage212_dwd_operating_curve", "212", "stage212_dwd_operating_curve.json", s212["status"], "只汇总阶段208和阶段210已冻结的FAR网格结果，并读取对应站点级明细做站点簇配对bootstrap；不训练、不重新选择、不重新校准阈值；主运行点仍是预先冻结的FAR=0.01，其余点仅作敏感性曲线", f"主FAR两套种子的F1差值分别为{s212_primary_text}；低误报FAR=0.005两套种子的F1差值分别为{s212_low_text}", "FAR=0.005/0.02/0.05只是冻结网格上的敏感性分析，不是test选出的最优运行点；F1在各点5/5种子正向，但事件覆盖在FAR=0.005两套种子分别为1/5和2/5正向，不能宣称所有运行点事件覆盖都占优")
    s213_supported = [
        row
        for row in s213["checks"]
        if str(row["status"]).startswith("SUPPORTED")
    ]
    add(rows, "stage213_dwd_external_evidence_closure", "213", "stage213_dwd_external_evidence_closure.json", s213["status"], "只读取阶段203五项最低要求和阶段204-212已冻结的DWD协议、下载、标准化、评价、复现和不确定性产物；逐项核验来源独立性、透明受控攻击、无目标域调参、主运行点与不确定性、可复现访问；不训练、不重新选择、不重新校准阈值，不下载新数据", f"最低要求支持={len(s213_supported)}/{s213['requirements_total']}；跨机构来源={s213['source_provider']}；主FAR={s213['primary_far']}；主终点={s213['primary_endpoint']}；聚类单位={s213['cluster_unit']}；站点数={s213['station_count']}", "只有一个跨机构来源DWD，不能扩展为多机构泛化声明；所有攻击都是透明受控注入，不是真实攻击日志；DWD质量码QN_9不是攻击标签；事件覆盖在FAR=0.005等敏感性点不占优")
    s214 = read_json("stage214_swiss_protocol_preregistration.json")
    s215 = read_json("stage215_swiss_official_download.json")
    s216 = read_json("stage216_swiss_normalized.json")
    s216a = read_json("stage216a_swiss_protocol_correction.json")
    s217 = read_json("stage217_swiss_controlled_attack.json")
    s218 = read_json("stage218_swiss_negative_result_uncertainty.json")
    s219 = read_json("stage219_swiss_dwd_protection_transfer.json")
    s220 = read_json("stage220_swiss_dwd_transfer_uncertainty.json")
    s221 = read_json("stage221_swiss_dwd_transfer_operating_curve.json")
    s222 = read_json("stage222_noaa_dwd_protection_transfer.json")
    s223 = read_json("stage223_noaa_transfer_uncertainty.json")
    s224 = read_json("stage224_knmi_protocol_preregistration.json")
    s225 = read_json("stage225_knmi_official_download.json")
    s226 = read_json("stage226_knmi_normalized.json")
    s226a = read_json("stage226a_knmi_missingness_audit.json")
    s227 = read_json("stage227_knmi_controlled_attack.json")
    s228 = read_json("stage228_knmi_dwd_protection_transfer.json")
    s229 = read_json("stage229_knmi_transfer_uncertainty.json")
    s230 = read_json("stage230_cross_provider_evidence_synthesis.json")
    s231 = read_json("stage231_cross_provider_robustness.json")
    s232 = read_json("stage232_attack_type_robustness.json")
    s233 = read_json("stage233_valid_attack_correction_audit.json")
    s240 = read_json("stage240_nonstuck_balanced_selection.json")
    s241 = read_json("stage241_transfer_heterogeneity_audit.json")
    s242 = read_json("stage242_multisource_transfer_matrix.json")
    s243 = read_json("stage243_knmi_negative_case_decomposition.json")
    s244 = read_json("stage244_cross_provider_nonstuck_candidate_features.json")
    s245 = read_json("stage245_cross_provider_nonstuck_seed_replication.json")
    s246 = read_json("stage246_frozen_submission_package.json")
    s247 = read_json("stage247_provider_balanced_nonstuck_audit.json")
    s248 = read_json("stage248_cross_provider_fair_baseline_audit.json")
    s275 = read_json("stage275_external_preregistration_lineage_audit.json")
    s276 = read_json("stage276_release_reproduction_input_audit.json")
    s277 = read_json("stage277_baseline_formality_matrix.json")
    s278 = read_json("stage278_confirmatory_test_package_contract.json")
    s279 = read_json("stage279_metric_schema_audit.json")
    s280 = read_json("stage280_nonstuck_stability_audit.json")
    s281 = read_json("stage281_submission_blocker_register.json")
    s282 = read_json("stage282_clean_environment_probe_audit.json")
    s283 = read_json("stage283_public_ics_cps_source_audit.json")
    s284 = read_json("stage284_swat_author_data_download.json")
    s285 = read_json("stage285_swat_usad_protocol_preregistration.json")
    s286 = read_json("stage286_swat_usad_author_evaluation.json")
    s287 = read_json("stage287_swat_usad_evidence_upgrade_audit.json")
    s288 = read_json("stage288_swat_usad_resource_audit.json")
    s289 = read_json("stage289_final_baseline_set_freeze.json")
    s290 = read_json("stage290_same_contract_baseline_resource_budget.json")
    s291 = read_json("stage291_unified_win_loss_matrix.json")
    s292 = read_json(
        "stage292_ocsvm_non_inferiority_boundary_audit.json"
    )
    s293 = read_json("stage293_confirmatory_package_freeze_audit.json")
    s294 = read_json(
        "stage294_japan_isd_protocol_preregistration.json"
    )
    s295 = read_json(
        "stage295_japan_isd_locked_evaluation_contract.json"
    )
    s296 = read_json("stage296_japan_isd_official_download.json")
    s297 = read_json("stage297_japan_isd_locked_transfer.json")
    s298 = read_json(
        "stage298_japan_transfer_failure_decomposition.json"
    )
    s299 = read_json("stage299_method_resource_coverage_audit.json")
    s300 = read_json("stage300_method_resource_evidence_upgrade.json")
    s301 = read_json("stage301_trust_resource_budget.json")
    s302 = read_json("stage302_japan_ocsvm_budget_grid.json")
    s303 = read_json(
        "stage303_independent_provider_source_feasibility.json"
    )
    s309 = read_json(
        "stage309_public_ics_cps_source_feasibility.json"
    )
    s304 = read_json(
        "stage304_met_norway_protocol_preregistration.json"
    )
    s305 = read_json(
        "stage305_met_norway_locked_evaluation_contract.json"
    )
    s306 = read_json("stage306_met_norway_official_download.json")
    s307 = read_json("stage307_met_norway_locked_transfer.json")
    s308 = read_json(
        "stage308_met_norway_evidence_boundary_audit.json"
    )
    s310 = read_json(
        "stage310_disjoint_met_norway_confirmatory_package.json"
    )
    s310a = read_json(
        "stage310a_disjoint_package_finalize.json"
    )
    s311 = read_json(
        "stage311_met_norway_disjoint_download.json"
    )
    s312 = read_json(
        "stage312_met_norway_disjoint_locked_transfer.json"
    )
    s313 = read_json(
        "stage313_cross_locked_test_win_loss_matrix.json"
    )
    s314 = read_json(
        "stage314_claim_ceiling_and_submission_scope_audit.json"
    )
    s315 = read_json(
        "stage315_noaa_2025_temporal_locked_transfer.json"
    )
    s316 = read_json(
        "stage316_noaa_2025_failure_decomposition.json"
    )
    s217_summary = {
        row["metric"]: row for row in s217["primary_summary"]
    }
    s217_stuck = next(
        row
        for row in s217["type_aggregate"]
        if row["attack_type"] == "stuck"
    )
    s217_spike = next(
        row
        for row in s217["type_aggregate"]
        if row["attack_type"] == "spike"
    )
    add(rows, "stage214_swiss_protocol_preregistration", "214", "stage214_swiss_protocol_preregistration.json", s214["status"], "在下载任何MeteoSwiss观测前冻结第二跨机构外部验证协议；只读取瑞士联邦官方STAC站点元数据和资产URL；固定站点、年份、变量、缺失规则、透明攻击、种子、FAR、主终点、聚类单位和多重校正；不训练模型、不选择阈值、不下载观测CSV、不计算性能", f"官方站点总数={s214['all_station_count']}；冻结站点={s214['station_count']}；年份={s214['years']}；主FAR={s214['primary_far']}；主终点={s214['primary_endpoint']}；下载状态={s214['download_status']}；性能状态={s214['performance_status']}", "这是预注册证据，不是性能结果；MeteoSwiss尚未下载和验证，不能写成已完成第二来源外部验证；攻击仍须是透明受控注入，不是真实攻击日志")
    add(rows, "stage215_swiss_official_download", "215", "stage215_swiss_official_download.json", s215["status"], "只从瑞士联邦官方STAC下载阶段214冻结的12个MeteoSwiss站点小时历史CSV；保留URL、下载时间、字节数、SHA256、字段、行数、站号和时间范围；不训练模型、不选择阈值、不计算性能", f"预期文件={s215['expected_archives']}；下载={s215['downloaded_count']}；已存在={s215['already_present_count']}；失败={s215['failed_count']}；字段通过={s215['required_columns_ok_count']}；审计失败={s215['audit_failure_count']}", "这是可追溯下载与CSV结构审计，不是模型性能；MeteoSwiss小时CSV没有与DWD QN_9等价的逐行质量码；空字段不是攻击标签；攻击仍是透明受控注入")
    add(rows, "stage216_swiss_normalized", "216", "stage216_swiss_normalized.json", s216["status"], "只标准化阶段214冻结的12个MeteoSwiss站点和阶段215下载的官方小时CSV；reference_timestamp按官方文档声明的UTC解析；按2021-2023/2024/2025切分；空字段转缺失并保留；不插补、不训练模型、不选择阈值、不计算性能", f"站点={s216['station_count']}；行数={s216['rows']}；缺测行={s216['missing_observation_rows']}；重复={s216['duplicate_station_timestamp_rows']}；温度缺失={s216['temperature_missing_rows']}；湿度缺失={s216['humidity_missing_rows']}；性能状态={s216['performance_status']}", "这是标准化数据审计，不是模型性能；官方文档确认时间戳为UTC、缺失值为空字段；MeteoSwiss没有与DWD QN_9等价的逐行质量码；空字段不是攻击标签")
    add(rows, "stage216a_swiss_protocol_correction", "216A", "stage216a_swiss_protocol_correction.json", s216a["status"], "在性能评价前记录MeteoSwiss下载后的可用性修正：预先冻结的12个站点中，aeg在2021-2025所有小时温度、湿度、露点和气压均为空，因此排除；其余11个站点保留；年份、划分、攻击、种子、FAR、终点、聚类和多重校正保持不变；不训练模型、不选择阈值、不计算性能", f"原冻结站点={s216a['original_station_count']}；排除={s216a['excluded_stations'][0]['station_id']}；保留站点={s216a['retained_station_count']}；性能状态={s216a['performance_status']}", "这是下载后、性能评价前的可用性修正，不是性能结果；排除aeg依据完整协议年份全空，不依据任何test性能；所有后续攻击仍是透明受控注入")
    add(rows, "stage217_swiss_controlled_attack_not_reproduced", "217", "stage217_swiss_controlled_attack.json", s217["status"], "在阶段216标准化的MeteoSwiss第二来源上独立复现：三个无监督分量只在2021-2023 train正常段重新拟合；2024 valid正常段按冻结FAR网格校准；2025 test只评价；使用阶段216A保留的11个站点；不迁移DWD权重，不根据test选择任何组件、阈值、站点或攻击位置", f"站点={s217['station_count']}；主FAR={s217['primary_far']}；F1={s217_summary['f1']['mean']:.6f}；事件覆盖={s217_summary['event_coverage']['mean']:.6f}；行召回={s217_summary['row_recall']['mean']:.6f}；spike事件覆盖={s217_spike['event_coverage']:.6f}；stuck事件覆盖={s217_stuck['event_coverage']:.6f}", "这是第二来源未复现DWD优势的保留负结果，不能写成多机构泛化领先；MeteoSwiss没有与DWD QN_9等价的逐行质量码；空字段只表示缺失，不是攻击标签；所有攻击都是透明受控注入，不是真实攻击日志")

    s218_event = s218["event_coverage_station_cluster_bootstrap"]
    s218_sign = {
        row["attack_type"]: row
        for row in s218["attack_type_sign_flip"]
    }
    add(rows, "stage218_swiss_negative_result_uncertainty", "218", "stage218_swiss_negative_result_uncertainty.json", s218["status"], "只读取阶段214冻结协议、阶段216标准化数据、阶段216A可用性修正和阶段217冻结结果；在冻结主FAR=0.01上做一次确定性评价重放，仅恢复事件覆盖的站点级不确定性；不训练模型、不重新选择特征或权重、不重新校准阈值、不选择站点或攻击位置；站点是重采样簇，种子是簇内重复测量", f"站点={s218['station_count']}；主FAR={s218['primary_far']}；事件覆盖={s218_event['mean']:.6f}，站点簇CI95=[{s218_event['ci95_low']:.6f},{s218_event['ci95_high']:.6f}]；stuck均值={s218_sign['stuck']['mean_difference']:.6f}，BH q={s218_sign['stuck']['bh_q_value']:.6f}；spike均值={s218_sign['spike']['mean_difference']:.6f}，BH q={s218_sign['spike']['bh_q_value']:.6f}；重放最大行级差={s218['replay_max_abs_row_metric_difference']:.3e}", "这是对阶段217负结果的只读不确定性审计，不是新的模型选择；站点级事件覆盖由确定性重放恢复，重放必须逐种子复现阶段217主结果；DWD与MeteoSwiss的描述性对照不是配对比较，不能扩展为多机构泛化领先；所有攻击都是透明受控注入，不是真实攻击日志")
    s219_aggregate = {
        row["arm"]: row for row in s219["aggregate"]
    }
    add(rows, "stage219_swiss_dwd_protection_transfer", "219", "stage219_swiss_dwd_protection_transfer.json", s219["status"], "逐种子读取阶段208在DWD上由valid_f1选出的特征、窗口和权重，原样迁移到MeteoSwiss；三臂分别为直接迁移DWD绝对阈值、保留DWD特征/窗口/权重但仅在瑞士valid正常样本重校准阈值、以及瑞士valid本地选择；valid和test使用分离的确定性攻击序列，2025 test只评价，不根据test选择特征、窗口、权重或阈值", f"主FAR={s219['primary_far']}；DWD阈值直接迁移F1={s219_aggregate['dwd_threshold_transfer']['f1_mean']:.6f}、事件覆盖={s219_aggregate['dwd_threshold_transfer']['event_coverage_mean']:.6f}；保留特征/权重并在瑞士valid重校准F1={s219_aggregate['swiss_valid_recalibrated']['f1_mean']:.6f}、事件覆盖={s219_aggregate['swiss_valid_recalibrated']['event_coverage_mean']:.6f}、FAR={s219_aggregate['swiss_valid_recalibrated']['false_alarm_rate_mean']:.6f}；瑞士本地选择F1={s219_aggregate['swiss_valid_selected']['f1_mean']:.6f}、事件覆盖={s219_aggregate['swiss_valid_selected']['event_coverage_mean']:.6f}；本地选择减重校准迁移F1={s219['swiss_valid_selected_minus_swiss_valid_recalibrated_f1']:+.6f}、事件覆盖={s219['swiss_valid_selected_minus_swiss_valid_recalibrated_event_coverage']:+.6f}", "DWD绝对阈值直接迁移在瑞士尺度上完全失效，说明绝对分数阈值不可跨机构复用；保留DWD特征/窗口/权重并重校准阈值仍低于目标域valid本地选择，因此不能写成重校准迁移占优；MeteoSwiss只有11站且与DWD站点、传感器、处理链和攻击块不同，不能写成多机构普适领先；所有攻击都是透明受控注入，不是真实攻击日志")
    s220_event = s220["primary_event_coverage"]
    s220_f1 = s220["primary_f1"]
    add(rows, "stage220_swiss_dwd_transfer_uncertainty", "220", "stage220_swiss_dwd_transfer_uncertainty.json", s220["status"], "只读取阶段219冻结的三臂逐站点test明细和逐种子事件指标，按11个MeteoSwiss站点做10000次站点簇配对bootstrap，并对F1、Recall、Precision、FAR做精确sign-flip检验和BH校正；不重新训练、不重新选择特征、窗口、权重或阈值，不使用瑞士test标签做任何选择；主比较为瑞士valid本地选择减重校准迁移", f"主比较=swiss_valid_selected减swiss_valid_recalibrated；F1差值={s220_f1['mean_difference']:+.6f}，CI95=[{s220_f1['ci95_low']:+.6f},{s220_f1['ci95_high']:+.6f}]，正向站点={int(s220_f1['positive_stations'])}/{int(s220_f1['stations'])}，BH q={s220_f1['bh_q']:.6f}；事件覆盖差值={s220_event['mean_difference']:+.6f}，CI95=[{s220_event['ci95_low']:+.6f},{s220_event['ci95_high']:+.6f}]，正向站点={int(s220_event['positive_stations'])}/{int(s220_event['stations'])}，BH q={s220_event['bh_q']:.6f}；FAR差值={next(row['mean_difference'] for row in s220['metric_uncertainty'] if row['metric'] == 'false_alarm_rate'):+.6f}", "只有11个MeteoSwiss站点，站点簇区间只描述当前站点集合的不确定性；DWD绝对阈值直接迁移仍为保留负结果；瑞士valid重校准不代表无目标域标签部署，而且重校准迁移仍低于目标域valid本地选择；所有攻击都是透明受控注入，不是真实攻击日志")
    s221_primary = {
        row["metric"]: row
        for row in s221["curve"]
        if row["is_primary"]
    }
    s221_f1 = s221_primary["f1"]
    s221_event = s221_primary["event_coverage"]
    add(rows, "stage221_swiss_dwd_transfer_operating_curve", "221", "stage221_swiss_dwd_transfer_operating_curve.json", s221["status"], "复用阶段219冻结的DWD特征、窗口、权重、MeteoSwiss数据切分、透明受控攻击、检测器和候选网格，valid和test使用分离的确定性攻击序列，只把主FAR=0.01扩展到阶段214预先冻结的0.005/0.01/0.02/0.05运行曲线；主比较为目标域valid本地选择减重校准迁移；test只评价，不重新选择、不重新校准主运行点", f"主FAR={s221['primary_far']}；F1差值={s221_f1['mean_difference']:+.6f}，CI95=[{s221_f1['ci95_low']:+.6f},{s221_f1['ci95_high']:+.6f}]，正向站点={int(s221_f1['positive_stations'])}/{int(s221_f1['stations'])}，BH q={s221_f1['bh_q']:.6f}；事件覆盖差值={s221_event['mean_difference']:+.6f}，CI95=[{s221_event['ci95_low']:+.6f},{s221_event['ci95_high']:+.6f}]，正向站点={int(s221_event['positive_stations'])}/{int(s221_event['stations'])}，BH q={s221_event['bh_q']:.6f}", "主FAR=0.01上本地选择方向稳定，但FAR=0.02的F1方向略转负、FAR=0.05区间跨零，不能写成所有运行点全面领先；DWD绝对阈值直接迁移失败必须保留；只有11个MeteoSwiss站点，不能写成多机构普适领先；所有攻击都是透明受控注入，不是真实攻击日志")
    s222_aggregate = {
        row["arm"]: row for row in s222["aggregate"]
    }
    add(rows, "stage222_noaa_dwd_protection_transfer", "222", "stage222_noaa_dwd_protection_transfer.json", s222["status"], "在NOAA/NCEI Global Hourly ISD第三来源上执行阶段208冻结的DWD保护候选迁移；使用阶段83质量审计后2020-2024可用的11个站点，2020-2022 train、2023 valid、2024 test；DWD的feature、window、weight逐种子从阶段208冻结selection读取；valid和test使用分离的确定性攻击序列；三臂为DWD绝对阈值直接迁移、保留DWD特征/窗口/权重并在NOAA valid正常样本重校准阈值、以及NOAA valid本地选择；test只评价，不根据test选择特征、窗口、权重、阈值、站点或攻击位置", f"主FAR={s222['primary_far']}；DWD阈值直接迁移F1={s222_aggregate['dwd_threshold_transfer']['f1_mean']:.6f}、事件覆盖={s222_aggregate['dwd_threshold_transfer']['event_coverage_mean']:.6f}；NOAA valid重校准F1={s222_aggregate['noaa_valid_recalibrated']['f1_mean']:.6f}、事件覆盖={s222_aggregate['noaa_valid_recalibrated']['event_coverage_mean']:.6f}、FAR={s222_aggregate['noaa_valid_recalibrated']['false_alarm_rate_mean']:.6f}；NOAA valid本地选择F1={s222_aggregate['noaa_valid_selected']['f1_mean']:.6f}、事件覆盖={s222_aggregate['noaa_valid_selected']['event_coverage_mean']:.6f}", "这是第三来源的冻结迁移结果，只能写成部分支持或中性，不能扩展为所有机构普适领先；DWD绝对阈值直接迁移在NOAA上仍为保留负结果；NOAA只有11站且站点网络、传感器和处理链与DWD、MeteoSwiss不同；2025本地文件不完整，因此test固定为完整的2024年；所有攻击都是透明受控注入，不是真实攻击日志")
    s223_f1 = s223["primary_f1"]
    s223_event = s223["primary_event_coverage"]
    add(rows, "stage223_noaa_transfer_uncertainty", "223", "stage223_noaa_transfer_uncertainty.json", s223["status"], "只读取阶段222冻结的NOAA第三来源三臂逐站点test明细和逐种子事件指标，按11个NOAA站点做10000次站点簇配对bootstrap，并对F1、Recall、Precision、FAR做精确sign-flip检验和BH校正；事件覆盖是单一预注册主终点，其bh_q字段直接记录精确sign-flip p值；不重新训练、不重新选择特征、窗口、权重或阈值，不使用NOAA test标签做任何选择", f"主比较=noaa_valid_selected减noaa_valid_recalibrated；F1差值={s223_f1['mean_difference']:+.6f}，CI95=[{s223_f1['ci95_low']:+.6f},{s223_f1['ci95_high']:+.6f}]，正向/负向/持平站点={int(s223_f1['positive_stations'])}/{int(s223_f1['negative_stations'])}/{int(s223_f1['zero_stations'])}，BH q={s223_f1['bh_q']:.6f}；事件覆盖差值={s223_event['mean_difference']:+.6f}，CI95=[{s223_event['ci95_low']:+.6f},{s223_event['ci95_high']:+.6f}]，正向/负向/持平站点={int(s223_event['positive_stations'])}/{int(s223_event['negative_stations'])}/{int(s223_event['zero_stations'])}，精确p={s223_event['sign_flip_p']:.6f}", "F1区间跨零且事件覆盖只有1个站点改善、10个持平，因此这是第三来源部分支持或中性结果，不能写成显著领先；只有11个NOAA站点，站点簇区间只描述当前站点集合的不确定性；DWD绝对阈值直接迁移仍为保留负结果；所有攻击都是透明受控注入，不是真实攻击日志")

    add(rows, "stage224_knmi_protocol_preregistration", "224", "stage224_knmi_protocol_preregistration.json", s224["status"], "在下载任何KNMI小时观测前冻结第三跨机构外部验证协议；只使用官方目录和预声明0.75度站点网格；固定年份、变量、缺失规则、透明攻击、种子、FAR、主终点、聚类单位和多重校正；不训练模型、不选择阈值、不下载观测、不计算性能", f"官方站点池={s224['station_count']}；年份={s224['years']}；主FAR={s224['primary_far']}；主终点={s224['primary_endpoint']}；下载状态={s224['download_status']}；性能状态={s224['performance_status']}", "这是预注册证据，不是性能结果；KNMI尚未下载和验证，不能写成已完成第三来源外部验证；攻击仍是透明受控注入，不是真实攻击日志")
    add(rows, "stage225_knmi_official_download", "225", "stage225_knmi_official_download.json", s225["status"], "只从KNMI官方Open Data API和官方CDN下载阶段224冻结的12个站点小时归档；保留URL、时间、字节数、SHA256、成员、字段、行数、站号和时间范围；不训练模型、不选择阈值、不计算性能", f"官方目录站点={s225['official_catalogue_station_count']}；冻结站点={s225['station_count']}；预期归档={s225['expected_archives']}；已存在={s225['already_present_count']}；失败={s225['failed_count']}；字段通过={s225['required_columns_ok_count']}；审计失败={s225['archive_audit_failure_count']}", "这是可追溯下载与归档结构审计，不是模型性能；KNMI小时序列可能因站址迁移和观测方式变化而不均质，不能用于趋势分析；空字段不是攻击标签；攻击仍是透明受控注入")
    add(rows, "stage226_knmi_normalized", "226", "stage226_knmi_normalized.json", s226["status"], "只标准化阶段225下载的12个KNMI官方站点小时归档；2021-2023为train，2024为valid，2025为test；HH按官方文件头解释为小时区间起点；T/TD/P除以10，U保持百分比；空字段转缺失并保留；不插补、不训练模型、不选择阈值、不计算性能", f"站点={s226['station_count']}；行数={s226['rows']}；缺测行={s226['missing_observation_rows']}；重复={s226['duplicate_station_timestamp_rows']}；温度缺失={s226['temperature_missing_rows']}；湿度缺失={s226['humidity_missing_rows']}；性能状态={s226['performance_status']}", "这是标准化数据审计，不是模型性能；263112行观测缺失显式保留；KNMI没有与DWD QN_9等价的逐行数值质量码；空字段不是攻击标签")
    add(rows, "stage226a_knmi_missingness_audit", "226A", "stage226a_knmi_missingness_audit.json", s226a["status"], "只定位KNMI冻结12站数据中的缺失分布；站点保留或排除只依据train+valid可用率，阈值0.95；test可用率单独报告且不参与任何选择；不训练模型、不选择阈值、不计算性能", f"决策划分={'+'.join(s226a['decision_splits'])}；test划分={s226a['test_split']}；阈值={s226a['availability_threshold']}；保留站点={','.join(map(str, s226a['retained_station_ids']))}；排除站点={','.join(map(str, s226a['excluded_station_ids']))}；决策行={s226a['decision_rows']}；test行={s226a['test_rows']}", "这是下载后、性能评价前的缺失可用性更正，不是性能结果；排除6站依据train+valid主变量缺失，不依据任何test性能；test可用率仅用于披露；所有后续攻击仍是透明受控注入")
    s227_summary = {row["metric"]: row for row in s227["primary_summary"]}
    s227_stuck = next(row for row in s227["type_aggregate"] if row["attack_type"] == "stuck")
    add(rows, "stage227_knmi_controlled_attack_not_reproduced", "227", "stage227_knmi_controlled_attack.json", s227["status"], "在阶段226标准化的KNMI第三来源上独立复现：三个无监督分量只在2021-2023 train正常段重新拟合；2024 valid正常段按冻结FAR网格校准；2025 test只评价；使用阶段226A保留的6个站点；不迁移DWD权重，不根据test选择任何组件、阈值、站点或攻击位置", f"站点={s227['station_count']}；主FAR={s227['primary_far']}；F1={s227_summary['f1']['mean']:.6f}；事件覆盖={s227_summary['event_coverage']['mean']:.6f}；行召回={s227_summary['row_recall']['mean']:.6f}；stuck事件覆盖={s227_stuck['event_coverage']:.6f}", "这是第三来源独立复现的中等/负结果，不能写成多机构泛化领先；KNMI没有与DWD QN_9等价的逐行质量码；空字段只表示缺失，不是攻击标签；所有攻击都是透明受控注入，不是真实攻击日志")
    s228_aggregate = {row["arm"]: row for row in s228["aggregate"]}
    add(rows, "stage228_knmi_dwd_protection_transfer", "228", "stage228_knmi_dwd_protection_transfer.json", s228["status"], "逐种子读取阶段208在DWD上由valid_f1选出的特征、窗口和权重，原样迁移到KNMI；valid和test使用分离的确定性攻击序列；三臂分别为直接迁移DWD绝对阈值、保留DWD特征/窗口/权重但仅在KNMI valid正常样本重校准阈值、以及KNMI valid本地选择；2025 test只评价，不根据test选择特征、窗口、权重或阈值", f"主FAR={s228['primary_far']}；DWD阈值直接迁移F1={s228_aggregate['dwd_threshold_transfer']['f1_mean']:.6f}、事件覆盖={s228_aggregate['dwd_threshold_transfer']['event_coverage_mean']:.6f}；KNMI valid重校准F1={s228_aggregate['knmi_valid_recalibrated']['f1_mean']:.6f}、事件覆盖={s228_aggregate['knmi_valid_recalibrated']['event_coverage_mean']:.6f}、FAR={s228_aggregate['knmi_valid_recalibrated']['false_alarm_rate_mean']:.6f}；KNMI本地选择F1={s228_aggregate['knmi_valid_selected']['f1_mean']:.6f}、事件覆盖={s228_aggregate['knmi_valid_selected']['event_coverage_mean']:.6f}", "DWD绝对阈值直接迁移在KNMI上完全失效，说明绝对分数阈值不可跨机构复用；保留DWD特征/窗口/权重并重校准阈值仍低于目标域valid本地选择，不能写成重校准迁移占优；KNMI只有6站且与DWD站点、传感器、处理链和攻击块不同，不能写成多机构普适领先；所有攻击都是透明受控注入，不是真实攻击日志")
    s229_f1 = s229["primary_f1"]
    s229_event = s229["primary_event_coverage"]
    add(rows, "stage229_knmi_transfer_uncertainty", "229", "stage229_knmi_transfer_uncertainty.json", s229["status"], "只读取阶段228冻结的KNMI第三来源三臂逐站点test明细和逐种子事件指标，按6个KNMI站点做10000次站点簇配对bootstrap，并对F1、Recall、Precision、FAR做精确sign-flip检验和BH校正；事件覆盖是单一预注册主终点，其bh_q字段直接记录精确sign-flip p值；不重新训练、不重新选择特征、窗口、权重或阈值，不使用KNMI test标签做任何选择", f"主比较=knmi_valid_selected减knmi_valid_recalibrated；F1差值={s229_f1['mean_difference']:+.6f}，CI95=[{s229_f1['ci95_low']:+.6f},{s229_f1['ci95_high']:+.6f}]，正向/负向/持平站点={int(s229_f1['positive_stations'])}/{int(s229_f1['negative_stations'])}/{int(s229_f1['zero_stations'])}，精确p={s229_f1['sign_flip_p']:.6f}，BH q={s229_f1['bh_q']:.6f}；事件覆盖差值={s229_event['mean_difference']:+.6f}，CI95=[{s229_event['ci95_low']:+.6f},{s229_event['ci95_high']:+.6f}]，正向/负向/持平站点={int(s229_event['positive_stations'])}/{int(s229_event['negative_stations'])}/{int(s229_event['zero_stations'])}，精确p={s229_event['sign_flip_p']:.6f}", "预注册主终点event coverage得到站点簇支持，但F1精确检验不显著，不能宣称全面或普适领先；只有6个KNMI站点，站点簇区间只描述当前站点集合的不确定性；DWD绝对阈值直接迁移仍为保留负结果；所有攻击都是透明受控注入，不是真实攻击日志")
    add(rows, "stage230_cross_provider_evidence_synthesis", "230", "stage230_cross_provider_evidence_synthesis.json", s230["status"], "只读取阶段203-239已冻结的DWD、MeteoSwiss、NOAA、KNMI和ECCC协议、下载、标准化、受控攻击、迁移和不确定性产物，生成来源级表和迁移臂表；不重新训练、不重新选择特征、窗口、权重、阈值、站点、种子、攻击类型或运行点，不使用目标域test标签做任何选择", f"来源数={s230['source_count']}；迁移目标数={s230['transfer_target_count']}；绝对阈值直接迁移失败={s230['direct_threshold_transfer_failure_count']}；目标域valid本地选择F1优势={s230['target_valid_selected_f1_advantage_count']}；事件覆盖正向={s230['target_valid_selected_event_advantage_count']}；F1精确检验显著来源={s230['f1_significant_source_count']}；事件覆盖精确检验显著来源={s230['event_significant_source_count']}", "跨机构来源站点数、传感器网络、处理链和攻击块不同；DWD绝对分数阈值在MeteoSwiss、NOAA、KNMI和ECCC上直接迁移均失败；特征、窗口和权重可迁移但必须在目标域valid正常样本上重校准阈值；KNMI的F1精确检验不显著；ECCC的14站来自预注册可用性规则；所有攻击都是透明受控注入，不是真实攻击日志")
    add(rows, "stage231_cross_provider_robustness", "231", "stage231_cross_provider_robustness.json", s231["status"], "只读取阶段219/220、222/223、228/229和238/239已冻结的逐站点test明细，重建目标域valid本地选择减目标域valid重校准的站点级F1和事件覆盖差值；先在机构内对站点取均值，再对四个机构等权取均值，并执行留一机构和固定种子两级层级bootstrap；不重新训练、不重新选择特征、窗口、权重、阈值、站点、种子、攻击类型或运行点，不使用目标域test标签做任何选择", f"机构数={s231['provider_equal_summary']['provider_count']}；站点数={s231['station_count']}；机构等权F1差值={s231['provider_equal_summary']['f1_difference']:+.6f}；机构等权事件覆盖差值={s231['provider_equal_summary']['event_difference']:+.6f}；留一机构F1方向正向={sum(bool(row['f1_direction_positive']) for row in s231['leave_one_provider_out'])}/{len(s231['leave_one_provider_out'])}；留一机构事件覆盖方向正向={sum(bool(row['event_direction_positive']) for row in s231['leave_one_provider_out'])}/{len(s231['leave_one_provider_out'])}；两级bootstrap F1 CI95=[{s231['hierarchical_bootstrap']['f1_ci95_low']:+.6f},{s231['hierarchical_bootstrap']['f1_ci95_high']:+.6f}]；事件覆盖CI95=[{s231['hierarchical_bootstrap']['event_ci95_low']:+.6f},{s231['hierarchical_bootstrap']['event_ci95_high']:+.6f}]", "只有四个跨机构迁移目标，机构级精确sign-flip检验最小双侧p=0.125，功效仍然不足；机构等权和两级bootstrap只能说明当前四个机构方向不由单一机构拉动，不能写成多机构普适领先；NOAA和KNMI的F1显著性仍有限；所有攻击都是透明受控注入，不是真实攻击日志")
    stuck232 = next(row for row in s232["provider_equal_rows"] if row["attack_type"] == "stuck")
    remove_stuck232 = next(row for row in s232["leave_one_attack_type_out_rows"] if row["removed_attack_type"] == "stuck")
    add(rows, "stage232_attack_type_robustness", "232", "stage232_attack_type_robustness.json", s232["status"], "只读取阶段219、222、228和238已冻结的逐种子攻击类型明细和类型汇总，重建目标域valid本地选择减目标域valid重校准的逐类型行召回和事件覆盖差值；机构内同类型种子先取均值，再对四个机构等权取均值，并执行留一机构、留一攻击类型和固定种子两级层级bootstrap；不重新训练、不重新选择特征、窗口、权重、阈值、站点、种子、攻击类型或运行点，不使用目标域test标签做任何选择；类型级明细不含正常行误报计数，因此不构造类型级Precision或F1", f"攻击类型数={s232['attack_type_count']}；stuck机构等权行召回差值={stuck232['row_recall_difference']:+.6f}；stuck事件覆盖差值={stuck232['event_coverage_difference']:+.6f}；stuck留一机构方向正向={sum(bool(row['row_recall_direction_positive']) and bool(row['event_coverage_direction_positive']) for row in s232['leave_one_provider_out_rows'] if row['attack_type'] == 'stuck')}/4；移除stuck后行召回差值={remove_stuck232['row_recall_difference']:+.6f}；事件覆盖差值={remove_stuck232['event_coverage_difference']:+.6f}", "跨机构总体正向方向主要集中在stuck；移除stuck后其余攻击类型的行召回和事件覆盖方向均转为负，不能写成所有攻击类型普遍占优；只有四个迁移目标，机构级精确sign-flip检验最小双侧p=0.125，功效仍然不足；stuck两级bootstrap下界为正，但仍不能替代独立机构显著性或真实攻击验证；类型级明细不包含正常行误报计数，不能构造类型级Precision或F1；所有攻击都是透明受控注入，不是真实攻击日志")
    add(rows, "stage233_valid_attack_correction_audit", "233", "stage233_valid_attack_correction_audit.json", s233["status"], "只读取阶段219-239已冻结产物，记录旧valid参考臂未注入攻击导致的结论失效，以及修正valid/test分离攻击后方向反转；不训练、不重选、不重校准、不使用目标域test标签做任何选择", f"旧结论状态={s233['old_claim_status']}；受影响阶段={','.join(s233['affected_stages'])}；修正后MeteoSwiss、NOAA ISD、KNMI和ECCC的F1与事件覆盖差值均为正；DWD绝对阈值直接迁移仍为0", "旧方向“目标域valid重校准迁移优于本地选择”已失效，不得再用于论文、表格、图或后续汇总；修正方向为目标域valid本地选择优于重校准迁移；只有四个迁移机构，且优势主要集中在stuck，不能写成多机构或全攻击类型普适领先")
    s240_boot = s240["hierarchical_bootstrap"]
    add(rows, "stage240_nonstuck_balanced_selection", "240", "stage240_nonstuck_balanced_selection.json", s240["status"], "复用四个目标域冻结协议、标准化数据、split、受控攻击生成器和42个候选；每个seed在目标域valid上分别按valid_f1和valid_nonstuck_event_macro选择；非stuck规则优先最大化valid非stuck事件覆盖宏平均，再按valid F1、总事件宏平均、stuck事件覆盖、权重/窗口/特征稳定排序；test只评价，不替换冻结主结果", f"机构数={s240['provider_count']}；站点数={s240['station_count']}；候选数={s240['candidate_count']}；非stuck规则减valid重校准的机构等权F1差值={s240['provider_equal_difference']['f1_difference']:+.6f}，CI95=[{s240_boot['f1_ci95_low']:+.6f},{s240_boot['f1_ci95_high']:+.6f}]；事件覆盖差值={s240['provider_equal_difference']['event_coverage_difference']:+.6f}，CI95=[{s240_boot['event_ci95_low']:+.6f},{s240_boot['event_ci95_high']:+.6f}]；机构级精确p(F1)={s240_boot['provider_sign_flip_f1_p']:.6f}，p(事件覆盖)={s240_boot['provider_sign_flip_event_p']:.6f}", "这是非stuck平衡的保留负结果：一旦valid选择优先非stuck事件覆盖，跨机构F1和事件覆盖优势基本消失，区间跨零；该阶段不替换冻结主结果，不能把总体优势改写成非stuck普适优势；只有四个迁移机构，机构级精确检验功效有限；所有攻击都是透明受控注入，不是真实攻击日志")
    s241_stuck = next(
        row for row in s241["type_rows"] if row["attack_type"] == "stuck"
    )
    add(
        rows,
        "stage241_transfer_heterogeneity_audit",
        "241",
        "stage241_transfer_heterogeneity_audit.json",
        s241["status"],
        "只读取阶段219/222/228/238已冻结的逐站点和攻击类型test明细，重建目标域valid本地选择减重校准迁移的站点级F1差值；按目标机构计算站点方向、留一目标机构后的机构等权方向和攻击类型方向；不训练、不重新选择特征/窗口/权重/阈值/站点/种子/攻击类型，不使用目标域test标签做任何选择",
        f"目标机构数={s241['target_count']}；站点数={s241['station_count']}；机构等权F1差值={s241['provider_equal_f1_difference']:+.6f}；42/42站点F1方向为正；留一目标机构方向正向=4/4；stuck行召回差值={s241_stuck['row_recall_difference']:+.6f}、事件覆盖差值={s241_stuck['event_coverage_difference']:+.6f}；非stuck行召回差值={s241['nonstuck_row_recall_difference']:+.6f}、事件覆盖差值={s241['nonstuck_event_coverage_difference']:+.6f}",
        "该阶段确认当前正向方向不是单个站点或单个目标机构驱动，但非stuck方向仍为负，优势主要集中在stuck；只有四个目标机构，机构级精确检验功效有限；所有攻击都是透明受控注入，不是真实攻击日志",
    )
    add(
        rows,
        "stage242_multisource_transfer_matrix",
        "242",
        "stage242_multisource_transfer_matrix.json",
        s242["status"],
        "复用五个机构已冻结的源域valid_f1特征、窗口和权重，构建20个有向跨机构迁移对；目标域只用valid正常样本重校准阈值，test只评价；目标域本地选择使用同一冻结候选网格作为参照；不训练新检测器、不根据test选择任何参数",
        f"机构数={s242['provider_count']}；有向迁移对={s242['pairwise_count']}；站点-对={s242['pairwise_station_count']}；目标域本地选择F1方向为正={s242['pairwise_local_positive']}/20；反向或持平={s242['negative_pair_count']}/20；Stage240本地臂逐行精确复现={s242['local_arm_exact_provider_count']}/4；反例如DWD←KNMI和ECCC←KNMI",
        "多源扩展没有支持全部正向：20个有向对中18个支持目标域本地选择，2个不支持；反向或持平集中在KNMI作为来源域；MeteoSwiss、NOAA ISD、KNMI和ECCC的本地臂逐行精确复现Stage240，DWD为同规则但不同冻结运行点的参照；20个有向对不是20个独立机构样本，机构级依赖仍然存在；单个迁移对不报告机构级sign-flip p值；所有攻击都是透明受控注入，不是真实攻击日志",
    )
    s243_dwd = next(
        row for row in s243["summary_rows"] if row["target"] == "DWD"
    )
    s243_eccc = next(
        row for row in s243["summary_rows"] if row["target"] == "ECCC"
    )
    add(
        rows,
        "stage243_knmi_negative_case_decomposition",
        "243",
        "stage243_knmi_negative_case_decomposition.json",
        s243["status"],
        "只读取阶段242冻结的逐站点和逐种子明细，分解DWD←KNMI和ECCC←KNMI两个反例；不训练、不重新选择、不重校准阈值、不使用test标签做选择",
        f"反例数={s243['pair_count']}；DWD←KNMI平均F1差值={s243_dwd['mean_f1_difference']:+.6f}，站点级行为完全一致={s243_dwd['allclose_station_behavior']}；ECCC←KNMI平均F1差值={s243_eccc['mean_f1_difference']:+.6f}，负方向种子={s243_eccc['seed_direction_negative']}/5，正/负/零站点={s243_eccc['positive_stations']}/{s243_eccc['negative_stations']}/{s243_eccc['zero_stations']}；ECCC反例中recall略升、precision和FAR恶化，ROC-AUC与平均精度一致下降",
        "DWD←KNMI是运行点等价持平而非性能退化，不能写成多源全面正向；ECCC←KNMI是明确的跨种子反例，必须保留；该分解只解释两个冻结反例，不能外推为所有KNMI来源迁移的普遍规律；所有攻击都是透明受控注入，不是真实攻击日志",
    )
    s244_boot = s244["hierarchical_bootstrap"]
    add(
        rows,
        "stage244_cross_provider_nonstuck_candidate_features",
        "244",
        "stage244_cross_provider_nonstuck_candidate_features.json",
        s244["status"],
        "复用四个目标域冻结协议、标准化数据、split、受控攻击生成器和候选空间；每个seed只在目标域valid上按valid_f1或valid_nonstuck_event_macro选择候选；test只评价；不训练新检测器、不使用目标域test标签做选择、不替换冻结主结果",
        f"机构数={s244['provider_count']}；站点数={s244['station_count']}；评估明细={s244['evaluation_rows']}；候选选择明细={s244['selection_rows']}；provider等权F1差值={s244_boot['f1_difference']:+.6f}，CI95=[{s244_boot['f1_ci95_low']:+.6f},{s244_boot['f1_ci95_high']:+.6f}]；事件覆盖差值={s244_boot['event_coverage_difference']:+.6f}，CI95=[{s244_boot['event_ci95_low']:+.6f},{s244_boot['event_ci95_high']:+.6f}]；机构级精确p(F1)={s244_boot['provider_sign_flip_f1_p']:.6f}，p(事件覆盖)={s244_boot['provider_sign_flip_event_p']:.6f}",
        "该阶段是冻结候选上的非stuck扩展方向证据而非普遍显著优势：provider等权bootstrap区间仍跨零，机构级精确检验功效有限；不得把方向一致改写成所有非stuck类型或机构显著领先；所有攻击都是透明受控注入，不是真实攻击日志",
    )
    s245_boot = s245["hierarchical_bootstrap"]
    add(
        rows,
        "stage245_cross_provider_nonstuck_seed_replication",
        "245",
        "stage245_cross_provider_nonstuck_seed_replication.json",
        s245["status"],
        "在预先冻结的五个独立种子上重复阶段244的非stuck扩展协议；每个seed只在目标域valid上选择；test只评价；不重新选择候选、窗口、特征、权重或阈值，不替换冻结主结果",
        f"独立种子数={len(s245['replication_seeds'])}；机构数={s245['provider_count']}；站点数={s245['station_count']}；评估明细={s245['evaluation_rows']}；选择明细={s245['selection_rows']}；provider等权F1差值={s245_boot['f1_difference']:+.6f}，CI95=[{s245_boot['f1_ci95_low']:+.6f},{s245_boot['f1_ci95_high']:+.6f}]；事件覆盖差值={s245_boot['event_coverage_difference']:+.6f}，CI95=[{s245_boot['event_ci95_low']:+.6f},{s245_boot['event_ci95_high']:+.6f}]；机构级精确p(F1)={s245_boot['provider_sign_flip_f1_p']:.6f}，p(事件覆盖)={s245_boot['provider_sign_flip_event_p']:.6f}",
        "独立种子复现支持方向性结论，但provider等权bootstrap区间仍跨零；移除ECCC会反转F1方向，drift行召回仍为负，优势不能解释为所有机构或所有攻击类型显著领先；所有攻击都是透明受控注入，不是真实攻击日志",
    )
    add(
        rows,
        "stage246_frozen_submission_package",
        "246",
        "stage246_frozen_submission_package.json",
        s246["status"],
        "只读取已经冻结的检测、迁移、非stuck复现和因果恢复结果，生成论文主表、主图和SHA256投稿包清单；不训练、不重新选择特征/窗口/权重/阈值、不重新评价任何test集、不替换任何原始证据",
        f"冻结输入={s246['frozen_input_count']}；生成交付物={s246['generated_deliverable_count']}；主FAR={s246['primary_far']}；DWD F1={s246['key_results']['dwd_f1']:.6f}、事件覆盖={s246['key_results']['dwd_event_coverage']:.6f}、行召回={s246['key_results']['dwd_row_recall']:.6f}；四机构等权迁移F1差值={s246['key_results']['cross_provider_f1_difference']:+.6f}、事件覆盖差值={s246['key_results']['cross_provider_event_difference']:+.6f}；非stuck独立种子F1差值={s246['key_results']['stage245_f1_difference']:+.6f}、事件覆盖差值={s246['key_results']['stage245_event_difference']:+.6f}；机构等权事件覆盖差值={s246['key_results']['provider_balanced_stage245_event_difference']:+.6f}；移除ECCC后F1差值={s246['key_results']['provider_balanced_stage245_remove_eccc_f1_difference']:+.6f}",
        "冻结投稿包证明输入可追溯、交付物完整和边界已披露，不证明性能或统计显著；非stuck独立种子区间仍跨零；机构等权时事件覆盖留一方向仍正，但移除ECCC后F1转负；优势主要集中在stuck，严格因果恢复弱于离线双向恢复；只有四个跨机构目标；所有攻击都是透明受控注入，不是真实攻击日志",
    )
    s247_checks = {row["check"]: row["status"] for row in s247["checks"]}
    s247_lopo = {
        row["removed_source"]: row
        for row in s247["leave_one_provider_out_rows"]
        if row["stage"] == "stage245"
    }
    add(
        rows,
        "stage247_provider_balanced_nonstuck_audit",
        "247",
        "stage247_provider_balanced_nonstuck_audit.json",
        s247["status"],
        "只读取阶段244和阶段245已经冻结的provider级结果，重建机构等权点估计和留一机构方向；不训练、不重新选择特征、窗口、权重或阈值，不使用test标签做任何选择",
        f"来源行={s247['source_rows']}；stage244 F1差值={s247['provider_equal_rows'][0]['f1_difference']:+.6f}、事件覆盖差值={s247['provider_equal_rows'][0]['event_coverage_difference']:+.6f}；stage245 F1差值={s247['provider_equal_rows'][1]['f1_difference']:+.6f}、事件覆盖差值={s247['provider_equal_rows'][1]['event_coverage_difference']:+.6f}；留一ECCC的stage245 F1差值={s247_lopo['ECCC']['f1_difference']:+.6f}、事件覆盖差值={s247_lopo['ECCC']['event_coverage_difference']:+.6f}；留一机构事件覆盖方向全正={s247_checks['all_leave_one_event_positive']}；留一机构F1方向全正={s247_checks['all_leave_one_f1_positive']}",
        "该审计确认非stuck事件覆盖的机构等权方向在留一机构后仍为正，但F1在移除ECCC后转负；因此不能宣称F1优势不由单一机构驱动，也不能把方向性复制写成普遍显著优势；该阶段只审计冻结结果，不产生新的检测性能；所有攻击都是透明受控注入，不是真实攻击日志",
    )

    s248_valid_robust = next(
        row
        for row in s248["comparison_rows"]
        if row["candidate"] == "trust_valid_f1"
        and row["baseline"] == "robust_max_deviation"
    )
    s248_valid_isolation = next(
        row
        for row in s248["comparison_rows"]
        if row["candidate"] == "trust_valid_f1"
        and row["baseline"] == "isolation_forest"
    )
    s248_nonstuck_robust = next(
        row
        for row in s248["comparison_rows"]
        if row["candidate"] == "trust_nonstuck_event_macro"
        and row["baseline"] == "robust_max_deviation"
    )
    s248_nonstuck_isolation = next(
        row
        for row in s248["comparison_rows"]
        if row["candidate"] == "trust_nonstuck_event_macro"
        and row["baseline"] == "isolation_forest"
    )
    s248_valid_svm = next(
        row
        for row in s248["comparison_rows"]
        if row["candidate"] == "trust_valid_f1"
        and row["baseline"] == "one_class_svm"
    )
    add(
        rows,
        "stage248_cross_provider_fair_baseline_audit",
        "248",
        "stage248_cross_provider_fair_baseline_audit.json",
        s248["status"],
        "在四个外部目标机构上补齐低FAR公平基线比较；所有方法共享相同train/valid/test边界、透明受控攻击序列和valid正常样本FAR=0.01阈值，test只评价；Trust候选按valid_f1或valid_nonstuck_event_macro选择，Robust max deviation、Isolation Forest和One-Class SVM只在train正常行拟合；One-Class SVM因训练规模使用确定性子采样并披露；不替换冻结主结果",
        f"机构数={s248['provider_count']}；站点数={s248['station_count']}；方法数={s248['method_count']}；Trust(valid_f1)对Robust max deviation的F1差值={s248_valid_robust['f1_difference']:+.6f}，CI95=[{s248_valid_robust['f1_ci95_low']:+.6f},{s248_valid_robust['f1_ci95_high']:+.6f}]；对Isolation Forest的F1差值={s248_valid_isolation['f1_difference']:+.6f}，CI95=[{s248_valid_isolation['f1_ci95_low']:+.6f},{s248_valid_isolation['f1_ci95_high']:+.6f}]；Trust(非stuck)对Robust事件覆盖差值={s248_nonstuck_robust['event_coverage_difference']:+.6f}，CI95=[{s248_nonstuck_robust['event_ci95_low']:+.6f},{s248_nonstuck_robust['event_ci95_high']:+.6f}]；对Isolation Forest差值={s248_nonstuck_isolation['event_coverage_difference']:+.6f}，CI95=[{s248_nonstuck_isolation['event_ci95_low']:+.6f},{s248_nonstuck_isolation['event_ci95_high']:+.6f}]；对One-Class SVM的F1差值={s248_valid_svm['f1_difference']:+.6f}，CI95=[{s248_valid_svm['f1_ci95_low']:+.6f},{s248_valid_svm['f1_ci95_high']:+.6f}]",
        "该公平基线审计是补充证据，不替换冻结主结果；One-Class SVM因计算预算在四个机构上均使用20000行train-normal确定性子采样，必须与全训练集结果分开解释；Trust(valid_f1)对Robust和Isolation Forest的F1优势区间不跨零，但One-Class SVM在F1和事件覆盖上呈互补关系，因此不能宣称全面领先；只有四个迁移目标，机构级精确检验功效有限；所有攻击都是透明受控注入，不是真实攻击日志。",
    )

    s249 = read_json("stage249_jisa_submission_gate.json")
    add(
        rows,
        "stage249_jisa_submission_gate",
        "249",
        "stage249_jisa_submission_gate.json",
        s249["status"],
        "只读取阶段246冻结投稿包、阶段248公平基线审计、阶段29证据矩阵和阶段10/112/114/115审计结果，逐条检查JISA投稿目标要求；不训练、不重新选择、不重新评价任何test集",
        f"要求通过={s249['requirements_passed']}/{s249['requirements_total']}；失败={s249['requirements_failed']}；主FAR={s249['primary_far']}；目标期刊={s249['target_journal']}",
        "该门审计只证明冻结材料逐条具备目标所要求的证据入口和解释边界，不新增性能证据；所有攻击均为透明受控注入，不是真实攻击日志；非stuck F1移除ECCC后转负、事件覆盖留一方向为正、One-Class SVM为互补边界、严格因果恢复弱于离线双向插值等边界必须随论文共同提交。",
    )

    s250 = read_json("stage250_jisa_risk_closure_audit.json")
    add(
        rows,
        "stage250_jisa_risk_closure_audit",
        "250",
        "stage250_jisa_risk_closure_audit.json",
        s250["status"],
        "只读取阶段244/245/246/248/249冻结证据，将剩余风险映射为处置状态、允许主声称和禁止声称；不训练、不选阈值、不选特征、不重新评价test集",
        f"风险数={s250['risk_count']}；已解决={s250['resolved_count']}；保留边界={s250['bounded_count']}；风险ID=" + ",".join(row["risk_id"] for row in s250["risks"]),
        "该审计把剩余风险固化为边界而不是性能证据；所有7项风险当前均标记为BOUNDED，任何未通过独立验证的结论不得写入论文主优势；所有攻击均为透明受控注入，不是真实攻击日志。",
    )

    s251 = read_json("stage251_jisa_stable_submission_objective.json")
    add(
        rows,
        "stage251_jisa_stable_submission_objective",
        "251",
        "stage251_jisa_stable_submission_objective.json",
        s251["status"],
        "把稳投稿目标转成可审计契约：逐项给出证据可达到的最高状态、升级门槛、论文动作和1-6投入顺序；不训练、不选阈值、不选特征、不重新评价test集",
        f"风险上限项={s251['risk_ceiling_count']}；当前可升级为RESOLVED={s251['resolved_count']}；必须保留BOUNDED={s251['bounded_count']}；工作流={s251['workstream_count']}",
        "该契约明确当前版本可稳投的范围，也明确真实攻击日志、新独立机构、One-Class SVM全量预算和全新锁定测试四项外部门槛未满足；未满足门槛的结论不能升级为确认性主声称。",
    )

    s252 = read_json("stage252_authorized_testbed_analysis.json")
    add(
        rows,
        "stage252_authorized_testbed_analysis",
        "252",
        "stage252_authorized_testbed_analysis.json",
        s252["status"],
        "在隔离loopback TCP链路上执行真实运行时攻击；保存发送/接收日志、哈希和运行摘要；不训练检测器、不重新评价冻结test",
        f"运行数={s252['run_count']}；干净运行={s252['clean_run_count']}；攻击运行={s252['attack_run_count']}；攻击类型={s252['attack_type_count']}；种子={s252['seed_count']}；失败={s252['failure_count']}",
        "测试床攻击是真实运行时事件，但环境为授权隔离测试床，不是生产事故日志；结果最多把外部效度风险降为PARTIALLY_CLOSED，不能覆盖冻结主终点或声称全球真实攻击普适性。",
    )

    s253 = read_json("stage253_residual_risk_optimization_plan.json")
    add(
        rows,
        "stage253_residual_risk_optimization_plan",
        "253",
        "stage253_residual_risk_optimization_plan.json",
        s253["status"],
        "把剩余七项风险拆成可推进、有限推进和必须新增数据/锁定的优化上限；不训练、不选阈值、不重新评价冻结test",
        f"风险项={s253['row_count']}；可实质推进={s253['can_optimize_now_count']}；有限推进={s253['limited_count']}；当前可RESOLVED={s253['resolved_now_count']}；测试床运行={s253['testbed_run_count']}",
        "该计划明确哪些剩余风险可通过真实测试床、基线规模敏感性或因果延迟实验推进，哪些必须依赖新机构、新锁定测试或独立生产日志；所有新结果仍需预注册并保持旧主表冻结。",
    )

    s254 = read_json("stage254_ocsvm_training_scale_sensitivity.json")
    s254_equal = {
        int(row["requested_rows"]): row
        for row in s254["provider_equal_rows"]
    }
    add(
        rows,
        "stage254_ocsvm_training_scale_sensitivity",
        "254",
        "stage254_ocsvm_training_scale_sensitivity.json",
        s254["status"],
        "复用Stage248冻结provider、split、透明攻击、检测器拟合、特征构造和valid正常样本FAR=0.01阈值；只改One-Class SVM训练规模，预注册20k/50k/100k并报告全部档位；test只评价，不替换冻结主结果",
        f"机构数={s254['provider_count']}；拟合行={s254['fit_row_count']}；评价行={s254['detail_rows']}；20k F1均值={s254_equal[20000]['f1_mean']:.6f}、事件覆盖均值={s254_equal[20000]['event_coverage_mean']:.6f}；50k F1均值={s254_equal[50000]['f1_mean']:.6f}、事件覆盖均值={s254_equal[50000]['event_coverage_mean']:.6f}；100k F1均值={s254_equal[100000]['f1_mean']:.6f}、事件覆盖均值={s254_equal[100000]['event_coverage_mean']:.6f}；100k实际拟合行范围={s254_equal[100000]['fit_rows_min']}-{s254_equal[100000]['fit_rows_max']}",
        "该敏感性说明20k训练截断不是One-Class SVM互补结论的唯一原因，但没有形成Trust全面领先；NOAA ISD在100k请求下实际只有91717个训练正常行且仍接近失效；所有攻击都是透明受控注入，不是真实攻击日志。",
    )

    s255 = read_json("stage255_online_causal_testbed.json")
    s256 = read_json("stage256_lightweight_strong_baselines.json")
    s257 = read_json("stage257_fmi_protocol_preregistration.json")
    s258 = read_json("stage258_fmi_official_download.json")
    s259 = read_json("stage259_fmi_locked_transfer.json")
    s259_by_method = {row["method"]: row for row in s259["provider_equal"]}
    s259_trust = s259_by_method["trust_valid_f1"]
    s259_best = max(s259["provider_equal"], key=lambda row: row["f1_mean"])
    s260 = read_json("stage260_reviewer_risk_closure_audit.json")
    s261 = read_json("stage261_causal_offline_recovery_contrast.json")
    s262 = read_json("stage262_security_domain_baselines.json")
    s263 = read_json("stage263_unified_statistical_report.json")
    s269 = read_json("stage269_deployment_complexity_audit.json")
    add(
        rows,
        "stage255_online_causal_testbed",
        "255",
        "stage255_online_causal_testbed.json",
        s255["status"],
        "只读取阶段252授权测试床的发送/接收日志；执行严格因果逐帧与事件级在线检测、延迟、吞吐和恢复测量；不训练、不选阈值、不重新评价冻结test",
        f"运行数={s255['run_count']}；逐帧F1均值={s255['f1_mean']:.4f}；事件F1均值={s255['event_f1_mean']:.4f}；正常FAR均值={s255['false_alarm_rate_mean']:.4f}；平均首报延迟={s255['detection_latency_mean_seconds']:.4f}s；平均吞吐={s255['throughput_mean_frames_per_second']:.2f}帧/s",
        "该阶段用于在线因果延迟、吞吐和恢复审计，不替代Stage248主检测比较；drop运行在发送前已重编号，接收侧没有可用序号缺口，drop在线识别保持负结果；测试床不是生产事故日志，在线恢复不能等同于离线双向插值。",
    )
    add(rows, "stage256_lightweight_strong_baselines", "256", "stage256_lightweight_strong_baselines.json", s256["status"], "复用Stage248冻结边界；PCA重建、MLP自编码器和kNN仅用train正常行拟合，valid校准，test只评价", f"机构数={s256['provider_count']}；站点数={s256['station_count']}；种子数={s256['seed_count']}；方法数={s256['method_count']}；新增轻量基线={','.join(s256['new_baseline_methods'])}", "仅为探索性补充，不替换Stage246/248确认性主结果；如新基线占优必须原样保留；受控攻击不是真实攻击日志。")
    add(
        rows,
        "stage275_external_preregistration_lineage_audit",
        "275",
        "stage275_external_preregistration_lineage_audit.json",
        s275["status"],
        "只读审计FMI与SMHI的协议冻结、官方下载、评价契约、锁定迁移和test使用血缘；不下载新数据、不训练、不调参、不改变预测",
        f"外部来源={s275['provider_count']}；独立来源={s275['independent_provider_count']}；完整下载前冻结={s275['frozen_before_full_download_count']}；严格零观测访问={s275['strict_zero_observation_access_before_freeze']}；FMI时间链={s275['fmi']['time_chain_status']}；SMHI时间链={s275['smhi']['time_chain_status']}；是否构成新确认性测试包={s275['is_a_new_confirmatory_test_package']}；是否可升级机构无关主张={s275['can_promote_provider_independence']}；是否可升级真实生产主张={s275['can_promote_real_production_claim']}",
        "FMI缺少内部UTC下载时间，SMHI存在已披露的冻结前20 KB头部探测；两者性能门槛均未通过，且test只用于评价并不能单独证明机构或站点无关；受控注入不是真实生产事故标签。",
    )
    add(
        rows,
        "stage276_release_reproduction_input_audit",
        "276",
        "stage276_release_reproduction_input_audit.json",
        s276["status"],
        "只读审计release重建入口直接依赖的冻结输入边界；不复制原始数据、不训练、不调参、不接触test观测",
        f"直接冻结输入={s276['input_count']}；工作区存在={s276['workspace_present_count']}；release存在={s276['release_present_count']}；release表文件={s276['release_table_file_count']}；代码级复现={s276['code_level_reproduction_verified']}；数据级复现={s276['data_level_reproduction_verified']}",
        "首版清单只证明当前重建入口的直接输入边界，不自动等于完整依赖闭包；只有从release根目录实际跑通重建、审计和回归后，才可升级数据级复现结论。",
    )
    add(
        rows,
        "stage277_baseline_formality_matrix",
        "277",
        "stage277_baseline_formality_matrix.json",
        s277["status"],
        "只读审计冻结基线的实现来源、证据层级、公开代码与公开数据可获得性；不训练、不校准、不改变任何预测",
        f"方法数={s277['method_count']}；正式实现数={s277['formal_implementation_count']}；透明代理数={s277['transparent_proxy_count']}；阻断项数={s277['blocked_count']}；公开数据可用数={s277['public_dataset_available_count']}；可无新数据升级数={s277['can_upgrade_without_new_data_count']}；P0-3状态={s277['p0_3_status']}",
        "透明代理不能写成公开ICS/CPS方法的论文忠实复现；本地实现不能冒充作者代码；公开USAD作者代码与SWaT已经正式评价，但公开端点不等同于TrustMeteo-QC合同，P0-3仍未通过；本阶段不把探索性结果升级为确认性结果。",
    )
    add(
        rows,
        "stage278_confirmatory_test_package_contract",
        "278",
        "stage278_confirmatory_test_package_contract.json",
        s278["status"],
        "只生成未来独立确认性测试包的协议契约；不下载数据、不查看标签、不训练、不校准、不生成性能数值",
        f"要求项={s278['requirement_count']}；已满足={s278['satisfied_count']}；尚未冻结={s278['not_yet_frozen_count']}；是否构成新确认性测试包={s278['is_a_new_confirmatory_test_package']}；是否下载数据={s278['data_actually_downloaded']}；是否查看test标签={s278['test_labels_inspected']}；是否冻结分析={s278['analysis_frozen']}",
        "该阶段只是协议骨架，不是新的独立数据包，也不产生任何性能或优越性结论；P0-6仍未通过。",
    )
    add(
        rows,
        "stage279_metric_schema_audit",
        "279",
        "stage279_metric_schema_audit.json",
        s279["status"],
        "只读审计冻结主结果表是否同时暴露F1、精确率、召回率、行召回、事件覆盖、FAR、AUPRC及事件分母；不训练、不校准、不改变任何预测",
        f"审计表数={s279['artifact_count']}；必需字段检查={s279['required_field_check_count']}；已存在={s279['present_field_check_count']}；缺失={s279['missing_field_check_count']}；可选字段存在={s279['optional_field_present_count']}",
        "字段存在只证明报告模式完整，不证明指标定义跨表完全等价，也不建立任何性能优势；缺失字段不得自动回填。",
    )
    add(
        rows,
        "stage280_nonstuck_stability_audit",
        "280",
        "stage280_nonstuck_stability_audit.json",
        s280["status"],
        "只读审计非stuck的攻击类型事件数、5种子方向、逐机构类型方向和留一机构方向；不训练、不重新选择特征、窗口、权重或阈值",
        f"类型数={len(s280['attack_types'])}；机构-类型单元={s280['provider_type_rows']}；种子方向行={s280['seed_direction_rows']}；留一机构行={s280['leave_one_provider_rows']}；失败检查={','.join(s280['failed_checks'])}",
        "事件数和种子覆盖通过不等于效果稳定；当前仍有类型-机构事件覆盖或行召回为零，P0-4未通过，主贡献应限定为stuck/重复值/冻结状态异常。",
    )
    add(
        rows,
        "stage281_submission_blocker_register",
        "281",
        "stage281_submission_blocker_register.json",
        s281["status"],
        "只读汇总当前无法由现有证据关闭的提交阻断项，明确全面领先、真实生产领先和干净环境复现的声称上限",
        f"阻断项={s281['blocker_count']}；P0开放={s281['p0_open_count']}；P1开放={s281['p1_open_count']}；允许全面领先={s281['universal_superiority_allowed']}；允许生产领先={s281['production_superiority_allowed']}；允许干净环境复现={s281['clean_environment_reproduction_allowed']}",
        "该登记只约束声称，不构成方法失败证据；关闭任何阻断项都必须依赖新的独立数据、预注册实验或有实际执行记录的干净环境复现。",
    )
    add(
        rows,
        "stage282_clean_environment_probe_audit",
        "282",
        "stage282_clean_environment_probe_audit.json",
        s282["status"],
        "只读记录干净环境复现的宿主权限、离线缓存和公开来源访问边界；不安装、不训练、不接触旧test",
        f"探测项={s282['probe_count']}；通过={s282['pass_count']}；阻断={s282['blocked_count']}；部分={s282['partial_count']}；干净安装完成={s282['clean_install_completed']}；release冻结输入级复现保留={s282['release_data_level_reproduction_preserved']}",
        "该审计证明当前主机不能据此声称干净环境端到端复现，也不解决公开ICS/CPS正式实现缺口；未通过项继续保持OPEN。",
    )
    add(
        rows,
        "stage283_public_ics_cps_source_audit",
        "283",
        "stage283_public_ics_cps_source_audit.json",
        s283["status"],
        (
            "只读核验固定提交USAD作者源码、许可证和notebook数据契约；"
            "不训练、不评价、不调参、不接触test标签"
        ),
        (
            f"验证作者文件={s283['verified_artifact_count']}/"
            f"{s283['expected_artifact_count']}；"
            f"作者数据契约已观察={s283['author_data_contract_observed']}；"
            f"本地SWaT数据可用={s283['local_swat_data_available']}；"
            f"论文忠实复现就绪={s283['paper_faithful_reproduction_ready']}；"
            f"P0-3状态={s283['p0_3_status_after']}"
        ),
        (
            "作者源码可追溯不等于基线已复现；notebook指向的SWaT数据"
            "缺失，不能把透明代理改名为USAD；P0-3保持OPEN。"
        ),
    )
    add(
        rows,
        "stage284_swat_author_data_download",
        "284",
        "stage284_swat_author_data_download.json",
        s284["status"],
        (
            "只下载并审计作者notebook声明的两个SWaT Google Drive文件；"
            "拒绝HTML确认页，记录SHA256和CSV结构；不训练、不评价"
        ),
        (
            f"下载文件={sum(row['status'] == 'DOWNLOADED_AND_CSV_AUDITED' for row in s284['rows'])}/"
            f"{len(s284['rows'])}；论文忠实复现就绪="
            f"{s284['paper_faithful_reproduction_ready']}；P0-3="
            f"{s284['p0_3_status_after']}"
        ),
        (
            "数据来源和哈希可追溯不等于性能结果；后续必须使用冻结协议"
            "完成作者代码训练和评价，且不能把SWaT公开基准写成生产事故日志。"
        ),
    )
    add(
        rows,
        "stage285_swat_usad_protocol_preregistration",
        "285",
        "stage285_swat_usad_protocol_preregistration.json",
        s285["status"],
        (
            "在首次USAD训练和查看test指标前冻结作者提交、数据哈希、"
            "normal 80/20窗口切分、窗口长度、作者损失、Adam、batch、"
            "epoch、valid-normal FAR=0.01阈值和事件/行级评价契约"
        ),
        (
            f"协议={s285['protocol_id']}；输入={len(s285['inputs'])}；"
            f"窗口={s285['window']['length']}；epochs="
            f"{s285['model']['epochs']}；FAR="
            f"{s285['threshold']['primary_far']}；test标签用于阈值="
            f"{s285['threshold']['test_labels_used_for_threshold']}"
        ),
        (
            "作者notebook使用test标签选择ROC等误报阈值，本协议明确替换为"
            "valid-normal FAR=0.01阈值以避免泄漏；该修改限制为论文忠实"
            "复现的协议适配，不能声称逐行等同作者notebook输出。"
        ),
    )
    add(
        rows,
        "stage286_swat_usad_author_evaluation",
        "286",
        "stage286_swat_usad_author_evaluation.json",
        s286["status"],
        (
            "使用固定提交作者usad.py、作者链接SWaT数据和Stage285冻结协议"
            "完成100 epoch正式评价；训练和阈值固定后才读取attack test"
        ),
        (
            f"F1={s286['test']['f1']:.4f}；precision="
            f"{s286['test']['precision']:.4f}；recall="
            f"{s286['test']['recall']:.4f}；FAR="
            f"{s286['test']['false_alarm_rate']:.4f}；event coverage="
            f"{s286['test']['event_coverage']:.4f}；AUPRC="
            f"{s286['test']['auprc']:.4f}"
        ),
        (
            "USAD结果不利且原样保留；这是公开SWaT基准评价，不是生产"
            "事故召回率，也不是与TrustMeteo-QC同数据同端点的直接对比。"
        ),
    )
    add(
        rows,
        "stage287_swat_usad_evidence_upgrade_audit",
        "287",
        "stage287_swat_usad_evidence_upgrade_audit.json",
        s287["status"],
        (
            "只读审计Stage283-286的来源、数据、协议、哈希、训练配方、"
            "无test阈值泄漏和指标完整性；不训练、不调参、不改变预测"
        ),
        (
            "公开ICS/CPS作者代码正式评价子项="
            f"{s287['public_ics_cps_formal_implementation_subitem']}；"
            f"审计通过={s287['checks_passed']}/"
            f"{s287['checks_total_excluding_same_contract']}；P0-3="
            f"{s287['p0_3_after']}；同契约Trust对比="
            f"{s287['same_contract_comparison']}"
        ),
        (
            "该结果关闭公开ICS/CPS作者代码与数据缺口，但不关闭P0-3整体："
            "最终基线集合、统一重跑、资源预算、全胜负表和Trust对所有"
            "主要基线的预注册门槛仍未完成；全面领先声称继续禁止。"
        ),
    )
    add(
        rows,
        "stage288_swat_usad_resource_audit",
        "288",
        "stage288_swat_usad_resource_audit.json",
        s288["status"],
        (
            "只读取作者链接SWaT normal文件，按Stage285冻结配方重训并"
            "测量参数量、训练时间、系统/CUDA峰值内存和valid-normal"
            "推理延迟；不读取attack test文件或test标签，不改变检测指标"
        ),
        (
            f"参数量={s288['model']['parameter_count']}；训练窗口="
            f"{s288['training']['train_windows']}；epoch="
            f"{s288['training']['epochs']}；训练秒="
            f"{s288['training']['wall_seconds']:.3f}；峰值工作集MB="
            f"{s288['training']['peak_working_set_mb']:.3f}；CUDA峰值"
            f"分配MB={s288['training']['cuda_peak_allocated_mb']:.3f}；"
            f"吞吐行/秒="
            f"{s288['valid_normal_inference']['throughput_rows_per_second']:.3f}"
        ),
        (
            "该阶段是资源测量，不替换Stage286检测性能，也不是"
            "TrustMeteo-QC同数据同端点比较；公开SWaT基准不能写成生产"
            "事故日志。"
        ),
    )
    add(
        rows,
        "stage289_final_baseline_set_freeze",
        "289",
        "stage289_final_baseline_set_freeze.json",
        s289["status"],
        (
            "只读冻结最终基线集合，明确同契约候选、未来新锁定test候选、"
            "透明代理和独立公开基准的边界；不训练、不校准、不接触test标签、"
            "不改变任何预测"
        ),
        (
            f"协议={s289['protocol_id']}；方法总数={s289['method_count']}；"
            f"同契约候选={s289['same_contract_candidate_count']}；"
            f"未来锁定test候选="
            f"{s289['future_locked_test_candidate_count']}；"
            f"独立公开基准="
            f"{s289['independent_public_benchmark_count']}；"
            f"探索性方法={s289['exploratory_only_count']}；"
            f"冻结后可增删={s289['can_add_or_remove_after_freeze']}；"
            f"同契约全胜负矩阵="
            f"{s289['same_contract_win_loss_matrix_available']}；"
            f"P0-3={s289['p0_3_status']}"
        ),
        (
            "冻结集合不等于完成同契约统一重跑，也不关闭P0-3；"
            "Stage256/262仍是探索性，ICS/CPS SPC仍是透明代理，"
            "USAD/SWaT是不同端点的独立公开基准，不能并入同契约胜负矩阵。"
        ),
    )
    add(
        rows,
        "stage290_same_contract_baseline_resource_budget",
        "290",
        "stage290_same_contract_baseline_resource_budget.json",
        s290["status"],
        (
            "只读重建冻结train-normal特征矩阵，在隔离子进程中按冻结配置"
            "拟合11个同契约最终基线并测量资源；不生成攻击、不读取valid/test"
            "标签、不计算检测指标、不替换任何冻结结果"
        ),
        (
            f"协议={s290['protocol_id']}；数据源={s290['source']}；"
            f"方法数={s290['method_count']}；"
            f"完成={s290['completed_method_count']}；"
            f"失败={s290['failed_method_count']}；"
            f"全部有失败状态="
            f"{s290['all_methods_have_failure_state']}"
        ),
        (
            "资源证据只说明当前主机和固定配置的拟合/评分成本，"
            "不改变任何检测性能或领先结论；One-Class SVM仍使用"
            "Stage248冻结预算，Stage274全量预算证据保持独立。"
        ),
    )
    add(
        rows,
        "stage291_unified_win_loss_matrix",
        "291",
        "stage291_unified_win_loss_matrix.json",
        s291["status"],
        (
            "只读汇总Stage248冻结候选层和Stage256/262探索层的全部"
            "候选-基线比较，生成统一胜负账本；不训练、不校准、不读取"
            "标签、不改变预测，也不创建新的锁定test"
        ),
        (
            f"来源阶段={s291['source_count']}；比较行="
            f"{s291['comparison_row_count']}；冻结层="
            f"{s291['frozen_comparison_row_count']}；探索层="
            f"{s291['exploratory_comparison_row_count']}；基线="
            f"{s291['baseline_count']}；现有证据账本可用="
            f"{s291['existing_evidence_win_loss_ledger_available']}；"
            f"统一新锁定test胜负矩阵="
            f"{s291['same_contract_win_loss_matrix_available']}；P0-3="
            f"{s291['p0_3_status']}"
        ),
        (
            "现有证据账本完整不等于新的锁定test重跑；Stage256/262仍为"
            "探索性，One-Class SVM等不利结果必须保留，不能据此升级"
            "全面领先结论。"
        ),
    )
    add(
        rows,
        "stage292_ocsvm_non_inferiority_boundary_audit",
        "292",
        "stage292_ocsvm_non_inferiority_boundary_audit.json",
        s292["status"],
        (
            "只读审计One-Class SVM非劣效声称边界：检查是否存在预注册"
            "非劣效界值、新的独立锁定test、同一锁定test上的统一重跑"
            "和确认性通过；不训练、不调参、不接触新test"
        ),
        (
            f"协议={s292['protocol_id']}；预注册界值="
            f"{s292['pre_registered_margin_available']}；新锁定test="
            f"{s292['new_locked_test_available']}；同test重跑="
            f"{s292['same_contract_new_locked_test_rerun']}；"
            f"要求训练档位={s292['required_ocsvm_fit_budgets']}；"
            f"实际档位={s292['observed_ocsvm_fit_budgets']}；"
            f"两候选全档位通过={s292['japan_primary_budget_pass']}；"
            f"完全通过候选={s292['fully_passing_candidates']}；"
            f"非劣效通过={s292['non_inferiority_pass_available']}；"
            f"状态={s292['non_inferiority_status']}"
        ),
        (
            "日本冻结test的20k/50k/100k档位已由Stage302补齐；"
            "trust_valid_f1三档双终点通过，但"
            "trust_nonstuck_event_macro在50k/100k的F1非劣未通过，"
            "因此P0-2仍未满足；One-Class SVM继续作为互补基线。"
        ),
    )
    add(
        rows,
        "stage302_japan_ocsvm_budget_grid",
        "302",
        "stage302_japan_ocsvm_budget_grid.json",
        s302["status"],
        (
            "复用Stage294/295/296冻结日本数据协议、攻击、种子、"
            "FAR阈值和评价代码，只改变One-Class SVM训练档位，"
            "补齐20k/50k/100k网格并分别与Stage297的冻结20k结果"
            "及同网格档位比较；不修改Trust或主表"
        ),
        (
            f"协议={s302['protocol_id']}；站点={s302['station_count']}；"
            f"种子={s302['seed_count']}；要求档位="
            f"{s302['required_budgets']}；实际档位="
            f"{s302['observed_budgets']}；网格完整="
            f"{s302['budget_grid_complete']}；评价行="
            f"{s302['evaluation_rows']}；比较数="
            f"{s302['comparison_count']}；全部较小档位非劣="
            f"{s302['all_smaller_budgets_non_inferior']}"
        ),
        (
            "这是同一NOAA provider下的新站点网络锁定迁移，不是"
            "独立机构；攻击为透明受控注入。20k档位两个候选均通过，"
            "但50k/100k档位上trust_nonstuck_event_macro的F1非劣"
            "失败，不利结果保留，不能升级全面领先。"
        ),
    )
    add(
        rows,
        "stage293_confirmatory_package_freeze_audit",
        "293",
        "stage293_confirmatory_package_freeze_audit.json",
        s293["status"],
        (
            "只读审计未来独立确认性测试包的协议、数据、分析、访问门"
            "和test清单冻结就绪状态；不下载数据、不读取标签、不训练、"
            "不调参、不产生性能结果"
        ),
        (
            f"协议={s293['protocol_id']}；包目录存在="
            f"{s293['package_present']}；新机构已选择="
            f"{s293['new_provider_selected']}；通过检查="
            f"{s293['passed_check_count']}/{s293['check_count']}；"
            f"构成新确认性测试包="
            f"{s293['is_a_new_confirmatory_test_package']}；"
            f"是否查看test标签={s293['test_labels_inspected']}；"
            f"声称上限={s293['claim_ceiling']}"
        ),
        (
            "缺少确认性包是证据缺口，不是模型失败；本阶段不能创造"
            "独立性，旧test不能通过改名或复制成为新的确认性测试，"
            "P0-6继续保持OPEN。"
        ),
    )
    s297_arima = next(
        row
        for row in s297["primary_comparison"]
        if row["baseline"] == "arima_innovation"
    )
    s297_gru = next(
        row
        for row in s297["primary_comparison"]
        if row["baseline"] == "gru_predictor"
    )
    add(
        rows,
        "stage294_japan_isd_protocol_preregistration",
        "294",
        "stage294_japan_isd_protocol_preregistration.json",
        s294["status"],
        (
            "在下载任何日本站点小时观测前冻结新的NOAA ISD日本站点"
            "网络、站点选择、年份划分、变量、官方质量码、缺失规则、"
            "透明受控攻击、种子、FAR、主终点和聚类单位；不读取观测"
            "值、不训练模型、不选择阈值、不计算性能"
        ),
        (
            f"协议={s294['protocol_id']}；网络={s294['network_id']}；"
            f"站点={s294['station_count']}；train={s294['train_years']}；"
            f"valid={s294['valid_year']}；test={s294['test_year']}；"
            f"质量码={s294['quality_accepted_code']}；"
            f"下载前未访问观测="
            f"{s294['strict_no_observation_access_before_freeze']}；"
            f"下载状态={s294['download_status']}；"
            f"性能状态={s294['performance_status']}"
        ),
        (
            "这是下载前协议冻结，不是性能结果；日本网络与既有六盘山"
            "NOAA站点空间不相交，但仍属于同一NOAA provider，不能"
            "据此证明机构独立；攻击是透明受控注入，不是真实攻击日志。"
        ),
    )
    add(
        rows,
        "stage295_japan_isd_locked_evaluation_contract",
        "295",
        "stage295_japan_isd_locked_evaluation_contract.json",
        s295["status"],
        (
            "在下载日本观测前冻结13种方法的统一评价契约、模型配置、"
            "train/valid/test规则、主FAR、主终点、站点簇bootstrap、"
            "BH多重校正和-0.02非劣效界值；不训练、不校准阈值、"
            "不读取观测或标签"
        ),
        (
            f"契约={s295['contract_id']}；方法数={len(s295['methods'])}；"
            f"主FAR={s295['primary_far']}；主终点="
            f"{s295['primary_endpoint']}；聚类单位="
            f"{s295['cluster_unit']}；网络层级="
            f"{s295['network_level']}；非劣效界值="
            f"{s295['non_inferiority_margin']}；"
            f"bootstrap={s295['bootstrap']['iterations']}"
        ),
        (
            "单一NOAA日本网络最多提供有界的新站点网络迁移证据；"
            "该契约明确不能解决provider独立、站点独立或真实生产"
            "事故召回率问题。"
        ),
    )
    add(
        rows,
        "stage296_japan_isd_official_download",
        "296",
        "stage296_japan_isd_official_download.json",
        s296["status"],
        (
            "只从NOAA NCEI官方global-hourly路径下载Stage294冻结的"
            "12个日本站点、5年共60个归档并规范化；按官方逐字段质量码"
            "和覆盖率规则审计；不训练模型、不校准阈值、不注入攻击、"
            "不计算性能"
        ),
        (
            f"预期归档={s296['expected_archives']}；"
            f"下载={s296['downloaded_count']}；"
            f"缓存={s296['cached_count']}；失败={s296['failed_count']}；"
            f"规范化行={s296['normalized_rows']}；"
            f"保留站点={s296['retained_station_count']}/"
            f"{s296['station_count']}；"
            f"性能状态={s296['performance_status']}"
        ),
        (
            "这是可追溯官方下载与覆盖率审计，不是检测性能；"
            "2025 test为部分年度，截至2025-08-24；日本网络与"
            "既有NOAA站点空间不相交但仍属同一provider。"
        ),
    )
    add(
        rows,
        "stage297_japan_isd_locked_transfer",
        "297",
        "stage297_japan_isd_locked_transfer.json",
        s297["status"],
        (
            "在Stage294协议、Stage295评价契约和Stage296下载审计之后，"
            "在日本NOAA ISD新站点网络上对13种方法执行同一锁定迁移；"
            "所有方法使用相同train/valid/test、透明受控攻击、种子、"
            "FAR=0.01和主终点；test只评价，不调参"
        ),
        (
            f"迁移门槛={s297['transfer_gate_status']}；"
            f"网络={s297['network_count']}；provider="
            f"{s297['provider_count']}；站点={s297['station_count']}；"
            f"种子={s297['seed_count']}；方法={s297['method_count']}；"
            f"主比较通过={s297['primary_pass_count']}/"
            f"{s297['primary_comparison_count']}；"
            f"全部基线支持={s297['all_baselines_supported']}；"
            f"相对ARIMA事件覆盖差="
            f"{s297_arima['event_coverage_difference']:+.6f}"
            f"(CI=[{s297_arima['event_ci95_low']:+.6f},"
            f"{s297_arima['event_ci95_high']:+.6f}])；"
            f"相对GRU事件覆盖差="
            f"{s297_gru['event_coverage_difference']:+.6f}"
            f"(CI=[{s297_gru['event_ci95_low']:+.6f},"
            f"{s297_gru['event_ci95_high']:+.6f}])"
        ),
        (
            "22个预注册主比较中18个通过，但相对ARIMA和GRU的事件"
            "覆盖为不利结果，因此all_baselines_supported=false且"
            "transfer_gate_status=NOT_PASSED；日本网络仍属同一NOAA"
            "provider，不能关闭P0-1、P0-2、P0-3、P0-4或P0-6；"
            "所有攻击都是透明受控注入。"
        ),
    )
    add(
        rows,
        "stage298_japan_transfer_failure_decomposition",
        "298",
        "stage298_japan_transfer_failure_decomposition.json",
        s298["status"],
        (
            "只读取Stage297冻结比较、评价和机构等权表，把日本锁定"
            "迁移未通过拆成比较级、种子级和失败原因；不训练、"
            "不调阈值、不生成攻击、不读取新test"
        ),
        (
            f"比较={s298['comparison_count']}；"
            f"通过={s298['passed_comparison_count']}；"
            f"未通过={s298['failed_comparison_count']}；"
            f"源门槛={s298['source_transfer_gate_status']}；"
            f"站点级明细可用="
            f"{s298['station_level_detail_available']}；"
            f"攻击类型级明细可用="
            f"{s298['attack_type_level_detail_available']}"
        ),
        (
            "这是只读失败分解，不是新实验；日本网络仍属NOAA，"
            "不能关闭provider独立性；Stage297未归档站点级和攻击"
            "类型级明细，不能通过选择性重分析补齐；ARIMA显著负向"
            "和GRU方向负向结果必须保留；所有攻击是透明受控注入。"
        ),
    )
    add(
        rows,
        "stage299_method_resource_coverage_audit",
        "299",
        "stage299_method_resource_coverage_audit.json",
        s299["status"],
        (
            "只对账Stage269、Stage290、Stage288、Stage255和Stage272"
            "已有的方法级资源字段；不训练、不调阈值、不读取valid/test、"
            "不跨合同合并证据"
        ),
        (
            f"方法={s299['method_count']}；"
            f"同契约基线={s299['same_contract_method_count']}；"
            f"同契约资源预算完整="
            f"{s299['same_contract_resource_budget_complete']}；"
            f"全部方法资源字段完整="
            f"{s299['all_method_resource_evidence_complete']}；"
            f"Trust资源字段完整="
            f"{s299['trust_resource_fields_complete']}；"
            f"USAD资源字段完整="
            f"{s299['usad_resource_fields_complete']}；"
            f"测试床资源字段完整="
            f"{s299['testbed_resource_fields_complete']}"
        ),
        (
            "Stage290的完整资源预算只适用于11个同契约基线，不适用于"
            "Trust候选、USAD或授权测试床；Trust参数量和拟合/内存/"
            "延迟仍为UNAVAILABLE，模型结构文字不能当作参数量；"
            "USAD是不同端点的公开SWaT基准；缺失值不估算、不填补。"
        ),
    )
    add(
        rows,
        "stage300_method_resource_evidence_upgrade",
        "300",
        "stage300_method_resource_evidence_upgrade.json",
        s300["status"],
        (
            f"Trust冻结选择配置={s300['trust_selection_rows']}；"
            f"Trust选择参数字段计数="
            f"{s300['trust_selection_parameter_count']}；"
            f"Trust配置复杂度参数计数="
            f"{s300['trust_configuration_parameter_count']}；"
            f"Trust资源字段完整={s300['trust_resource_fields_complete']}；"
            f"测试床资源字段完整="
            f"{s300['testbed_resource_fields_complete']}；"
            f"全部方法资源字段完整="
            f"{s300['all_method_resource_evidence_complete']}"
        ),
        (
            "配置参数量不是神经网络参数量；吞吐倒数得到的平均延迟不是"
            "逐帧延迟分布；Trust拟合时间、内存、端到端延迟及测试床"
            "参数量、fit行数、拟合时间、峰值内存仍为UNAVAILABLE；"
            "本阶段不训练、不调阈值、不读取valid/test，也不关闭任何P0。"
        ),
        (
            "配置参数量不是神经网络参数量；吞吐倒数得到的平均延迟不是"
            "逐帧延迟分布；Trust拟合时间、内存、端到端延迟及测试床"
            "参数量、fit行数、拟合时间、峰值内存仍为UNAVAILABLE；"
            "本阶段不训练、不调阈值、不读取valid/test，也不关闭任何P0。"
        ),
    )
    s301_measurement = s301["measurement"]
    add(
        rows,
        "stage301_trust_resource_budget",
        "301",
        "stage301_trust_resource_budget.json",
        s301["status"],
        (
            "在隔离子进程中只重建MeteoSwiss冻结train-normal特征矩阵，"
            "实测冻结Trust统计分量的拟合、评分延迟、单站点吞吐和峰值"
            "工作集；不生成攻击、不读取valid/test、不校准阈值、不计算"
            "检测指标、不替换冻结结果"
        ),
        (
            f"拟合行数={s301_measurement['fit_rows']}；"
            f"评分行数={s301_measurement['rows_scored']}；"
            f"分量拟合秒={s301_measurement['component_fit_seconds']}；"
            f"峰值工作集MB="
            f"{s301_measurement['peak_working_set_mb']}；"
            f"全量评分均值ms="
            f"{s301_measurement['score_latency']['mean_ms']}；"
            f"单站点吞吐行每秒="
            f"{s301_measurement['single_station_throughput_rows_per_second']}；"
            f"配置复杂度参数计数="
            f"{s301_measurement['configuration_parameter_count']}"
        ),
        (
            "资源证据只说明当前主机上的部署成本，不改变任何准确性或"
            "领先结论；配置复杂度参数计数不是神经网络参数量；拟合、"
            "内存和延迟只在MeteoSwiss train-normal矩阵上测量，不是"
            "四机构聚合；目标域valid重校准未在本阶段重跑；本阶段不"
            "关闭任何P0声称门。"
        ),
    )
    add(rows, "stage257_fmi_protocol_preregistration", "257", "stage257_fmi_protocol_preregistration.json", s257["status"], "下载前冻结FMI第五机构站点、年份、变量、覆盖率、攻击类型、种子、FAR和主终点；test不得用于选择", f"独立机构={s257['provider']}；候选站点={s257['station_count']}；年份={','.join(map(str, s257['years']))}；种子数={len(s257['seeds'])}；主终点={s257['primary_endpoint']}", "这是协议预注册，不是性能结果；后续攻击仍为透明受控注入。")
    add(rows, "stage258_fmi_official_download", "258", "stage258_fmi_official_download.json", s258["status"], "按Stage257冻结协议下载FMI官方逐小时观测并规范化，只做来源、覆盖率和哈希审计，不训练、不调参", f"下载档案={s258['downloaded_count']}/{s258['expected_archives']}；失败={s258['failed_count']}；规范化行={s258['normalized_rows']}；保留站点={s258['retained_station_count']}", "本阶段不提供检测性能；覆盖率不足站点必须保留在审计并排除。")
    add(rows, "stage259_fmi_locked_transfer", "259", "stage259_fmi_locked_transfer.json", s259["status"], "在Stage257冻结、Stage258审计后的FMI test上锁定迁移评价；Trust与六个基线同协议比较，test只评价", f"站点={s259['station_count']}；种子={s259['seed_count']}；方法={s259['method_count']}；Trust F1={s259_trust['f1_mean']:.4f}；最佳F1={s259_best['f1_mean']:.4f}({s259_best['method']})；Trust排名={sorted(s259['provider_equal'], key=lambda row: row['f1_mean'], reverse=True).index(s259_trust)+1}", "FMI上One-Class SVM胜出，不能宣称Trust跨机构全面领先；结果属于受控注入，不能写成真实事故召回率。")
    add(rows, "stage260_reviewer_risk_closure_audit", "260", "stage260_reviewer_risk_closure_audit.json", s260["status"], "只读取冻结主结果与新增外部/测试床证据，逐项审计声称上限，不训练、不重新选择旧test", f"风险项={s260['risk_count']}；部分关闭={s260['partially_closed_count']}；有界保留={s260['bounded_count']}；完全关闭={s260['resolved_count']}", "审计明确生产事故普适性、机构无关、在线等同离线和真实攻击召回率仍被禁止；不能把风险审计当作性能增益。")
    add(rows, "stage261_causal_offline_recovery_contrast", "261", "stage261_causal_offline_recovery_contrast.json", s261["status"], "复用Stage252/255日志，在相同事件比较只看过去的在线因果恢复与使用未来帧的离线双向插值；不训练、不调参", f"运行={s261['run_count']}；在线MAE={s261['online_mae_mean']:.4f}；离线MAE={s261['offline_mae_mean']:.4f}；因果惩罚={s261['causal_penalty_mae_mean']:.4f}", "在线与离线信息集不同且本阶段未出现相等结果；只能报告安全—效用权衡，禁止声称在线等同离线。")
    add(
        rows,
        "stage262_security_domain_baselines",
        "262",
        "stage262_security_domain_baselines.json",
        s262["status"],
        "复用Stage248冻结的机构、split、透明受控攻击和FAR=0.01终点；新增EWMA、ARIMA innovation、轻量GRU reservoir、轻量自编码器和ICS/CPS SPC proxy；所有拟合只用train正常行，阈值只用valid正常行，test只评价",
        f"机构数={s262['provider_count']}；站点数={s262['station_count']}；种子数={s262['seed_count']}；方法数={s262['method_count']}；评价行={s262['evaluation_rows']}；站点明细行={s262['station_rows']}；攻击类型明细行={s262['type_rows']}；新增基线={','.join(s262['new_baseline_methods'])}",
        "探索性安全基线补充，不替换Stage246/248确认性主结果；GRU是轻量reservoir代理，ICS/CPS是透明SPC proxy，不是原模型逐位复现；所有攻击仍是透明受控注入；如新基线占优必须原样保留。",
    )
    add(
        rows,
        "stage269_deployment_complexity_audit",
        "269",
        "stage269_deployment_complexity_audit.json",
        s269["status"],
        (
            "只读整合冻结方法规模、拟合预算和部署信息；"
            "用冻结Stage256配置重建Isolation Forest、PCA、MLP自编码器"
            "和kNN的资源指标，不改变split、阈值、预测或检测结论。"
        ),
        (
            f"方法数={s269['method_count']}；"
            f"冻结主结果方法数={s269['frozen_main_method_count']}；"
            f"实测内存方法数={s269['measured_memory_method_count']}；"
            f"实测延迟方法数={s269['measured_latency_method_count']}；"
            "PCA/MLP/kNN独立子进程="
            f"{s269['reconstruction_benchmark']['pca']['isolated_subprocess']}/"
            f"{s269['reconstruction_benchmark']['mlp_autoencoder']['isolated_subprocess']}/"
            f"{s269['reconstruction_benchmark']['knn']['isolated_subprocess']}"
        ),
        (
            "资源实测只说明当前主机和固定配置的部署成本，不改变任何"
            "准确性或领先结论；Isolation Forest峰值仍是本进程上限，"
            "One-Class SVM峰值仍是Stage274共享进程累计值，不能逐机构"
            "归因；未实测字段继续保持UNAVAILABLE。"
        ),
    )
    add(
        rows,
        "stage263_unified_statistical_report",
        "263",
        "stage263_unified_statistical_report.json",
        s263["status"],
        "只读取Stage248、Stage256和Stage262已落盘评价，不训练、不重新选择阈值、不修改test预测；pooled、站点等权、机构等权分层报告",
        f"来源阶段={s263['stage_count']}；pooled行={s263['pooled_rows']}；站点等权行={s263['station_equal_rows']}；机构等权行={s263['provider_equal_rows']}；逐站点方向行={s263['station_direction_rows']}；攻击类型行={s263['attack_type_rows']}；功效/MDE行={s263['power_mde_rows']}；事件定义审计行={s263['event_definition_audit_rows']}；低误报/F1最优点行={s263['low_far_and_f1_optimum_rows']}；比较行={s263['comparison_rows']}；留一机构行={s263['leave_one_provider_rows']}",
        "该阶段只统一统计口径和分层报告，不产生新性能证据；Stage256/262保持探索性且不能替换Stage246/248；事件覆盖以每个(station_id, attack_type, split, seed)攻击块为分母并与行召回并列；机构数量有限，MDE不能支持机构无关声称；不利结果必须保留。",
    )

    add(
        rows,
        "stage303_independent_provider_source_feasibility",
        "303",
        "stage303_independent_provider_source_feasibility.json",
        s303["status"],
        (
            "只读审计六个未使用官方机构候选的许可、小时能力、批量归档"
            "能力和机器访问条件；不下载观测、不冻结站点、不训练模型、"
            "不校准阈值、不计算性能"
        ),
        (
            f"候选数={s303['candidate_count']}；"
            f"阻断候选数={s303['blocked_candidate_count']}；"
            f"可升级候选数={s303['promotable_candidate_count']}"
        ),
        (
            "官方页面可达不等于可重复批量下载，也不等于新机构性能证据；"
            "本阶段不能关闭P0-1、P0-5、P0-6或全面领先门。"
        ),
    )
    add(
        rows,
        "stage309_public_ics_cps_source_feasibility",
        "309",
        "stage309_public_ics_cps_source_feasibility.json",
        s309["status"],
        (
            "只读审计SWaT、WADI、BATADAL、HAI、SMD和SMAP/MSL的"
            "官方入口、数据性质、标签层级、许可边界、机器访问和能否"
            "进入同契约评价；不下载数据、不读取标签、不训练模型、"
            "不校准阈值、不计算性能"
        ),
        (
            f"候选数={s309['candidate_count']}；"
            f"阻断候选={s309['blocked_candidate_count']}；"
            f"测试床候选={s309['testbed_candidate_count']}；"
            f"真实生产事故日志={s309['production_incident_log_count']}；"
            f"同契约可用={s309['same_contract_ready_count']}；"
            f"P0-3={s309['p0_3_status_after']}；"
            f"P0-5={s309['p0_5_status_after']}"
        ),
        (
            "公开测试床或不同域异常基准不等于真实生产事故日志，也不能"
            "并入天气站点同契约胜负矩阵；官方入口可达不等于许可和"
            "机器访问已确认；本阶段不关闭P0-3、P0-5或任何全面领先门。"
        ),
    )
    add(
        rows,
        "stage304_met_norway_protocol_preregistration",
        "304",
        "stage304_met_norway_protocol_preregistration.json",
        s304["status"],
        (
            "在下载任何MET Norway Frost站点小时观测前冻结独立机构协议："
            "12站点、2021-2025切分、质量码、缺失规则、透明受控攻击、"
            "种子、FAR、主终点、聚类和多重校正；只读取官方元数据，"
            "不读取观测值、不训练、不计算性能"
        ),
        (
            f"协议={s304['protocol_id']}；修订="
            f"{s304.get('protocol_revision', '')}；网络="
            f"{s304['network_id']}；站点={s304['station_count']}；"
            f"下载前未访问观测="
            f"{s304['strict_no_observation_access_before_freeze']}；"
            f"下载状态={s304['download_status']}；性能状态="
            f"{s304['performance_status']}"
        ),
        (
            "这是下载前协议冻结，不是性能结果；MET Norway是独立机构"
            "但仍只有一个provider网络，不能证明站点无关；所有攻击仍是"
            "透明受控注入，不是真实攻击日志。"
        ),
    )
    add(
        rows,
        "stage305_met_norway_locked_evaluation_contract",
        "305",
        "stage305_met_norway_locked_evaluation_contract.json",
        s305["status"],
        (
            "在下载MET Norway观测前冻结统一评价契约、13个逻辑方法、"
            "One-Class SVM 20k/50k/100k预算、train/valid/test规则、"
            "FAR=0.01、event coverage主终点、站点簇bootstrap、BH校正"
            "和-0.02非劣效界值；不训练、不校准阈值、不读取标签"
        ),
        (
            f"契约={s305['contract_id']}；方法数={len(s305['methods'])}；"
            f"主FAR={s305['primary_far']}；主终点="
            f"{s305['primary_endpoint']}；聚类单位="
            f"{s305['cluster_unit']}；网络层级="
            f"{s305['network_level']}；非劣效界值="
            f"{s305['non_inferiority_margin']}；bootstrap="
            f"{s305['bootstrap']['iterations']}"
        ),
        (
            "单一MET Norway网络最多提供有界的新provider迁移证据；"
            "不能解决站点独立、真实生产事故召回率或全面领先问题。"
        ),
    )
    add(
        rows,
        "stage306_met_norway_official_download",
        "306",
        "stage306_met_norway_official_download.json",
        s306["status"],
        (
            "只从MET Norway Frost官方observations端点下载Stage304冻结"
            "的12站点2021-2025小时温度并规范化；按官方质量码0和覆盖率"
            "门槛审计；不训练模型、不校准阈值、不注入攻击、不计算性能"
        ),
        (
            f"预期请求={s306['expected_requests']}；下载="
            f"{s306['downloaded_count']}；缓存={s306['cached_count']}；"
            f"失败={s306['failed_count']}；规范化行="
            f"{s306['normalized_rows']}；保留站点="
            f"{s306['retained_station_count']}/{s306['station_count']}；"
            f"性能状态={s306['performance_status']}"
        ),
        (
            "这是可追溯官方下载与覆盖率审计，不是检测性能；2025 test"
            "为部分年度，截至2025-08-24；独立机构不等于站点无关或"
            "真实攻击召回率。"
        ),
    )
    add(
        rows,
        "stage307_met_norway_locked_transfer",
        "307",
        "stage307_met_norway_locked_transfer.json",
        s307["status"],
        (
            "在Stage304协议、Stage305契约和Stage306下载审计之后，在"
            "MET Norway新站点网络上执行15个方法的锁定迁移；所有方法"
            "使用相同train/valid/test、透明受控攻击、5种子、FAR=0.01"
            "和event coverage主终点；test只评价"
        ),
        (
            f"迁移门槛={s307['transfer_gate_status']}；provider="
            f"{s307['provider_count']}；站点={s307['station_count']}；"
            f"种子={s307['seed_count']}；方法={s307['method_count']}；"
            f"主比较通过={s307['primary_pass_count']}/"
            f"{s307['primary_comparison_count']}；全部基线支持="
            f"{s307['all_baselines_supported']}"
        ),
        (
            "修正后只有8/26主比较通过，存在18个不利比较，因此"
            "transfer_gate_status=NOT_PASSED；该修正发生在首次接触"
            "test之后，只能作为探索性不利证据，不能关闭P0-1、P0-2、"
            "P0-3、P0-4或P0-6；所有攻击仍是透明受控注入，不是真实"
            "生产事故。"
        ),
    )
    add(
        rows,
        "stage308_met_norway_evidence_boundary_audit",
        "308",
        "stage308_met_norway_evidence_boundary_audit.json",
        s308["status"],
        (
            "只读审计Stage304-307冻结产物，保留第一次Stage307错误统计"
            "的26/26假阳性，记录修正后的8/26不利结果、哈希、修改时序"
            "和声称边界；不下载、不训练、不校准、不注入攻击、不计算新指标"
        ),
        (
            f"检查通过={s308['passed_check_count']}/"
            f"{s308['check_count']}；第一次无效结果="
            f"{s308['first_pass']['primary_pass_count']}/"
            f"{s308['first_pass']['primary_comparison_count']}；"
            f"修正后={s308['corrected_pass']['primary_pass_count']}/"
            f"{s308['corrected_pass']['primary_comparison_count']}；"
            f"不利比较={s308['corrected_pass']['adverse_comparison_count']}"
        ),
        (
            "第一次26/26因配对bootstrap未按candidate/baseline过滤而无效；"
            "修正版是在首次接触test后修改代码得到，不能作为确认性测试包；"
            "MET Norway仍只有一个provider网络，不能证明站点无关；本阶段"
            "不关闭任何P0，也不支持全面领先、真实攻击召回率或生产普适性"
            "声称。"
        ),
    )
    add(
        rows,
        "stage310_disjoint_met_norway_confirmatory_package",
        "310",
        "stage310_disjoint_met_norway_confirmatory_package.json",
        s310["status"],
        (
            "在下载任何新站网观测前冻结与Stage304完全不重叠的"
            "MET Norway站点网络，并记录协议、数据、分析和访问门哈希；"
            "只读取官方元数据，不读取观测值、不训练、不计算性能"
        ),
        (
            f"provider是否为新机构={s310['provider_is_new']}；"
            f"与Stage304站点不相交="
            f"{s310['network_is_disjoint_from_stage304']}；"
            f"站点={s310['station_count']}；"
            f"可用不相交站点池={s310['eligible_disjoint_station_count']}；"
            f"读取观测值={s310['observation_values_read']}；"
            f"声称上限={s310['claim_ceiling']}"
        ),
        (
            "这是同一provider内的不相交站网冻结，不是新独立机构；"
            "只能支持测试隔离和站网稳健性，不能关闭P0-1、P0-5、"
            "P0-6或全面领先门。"
        ),
    )
    add(
        rows,
        "stage310a_disjoint_package_finalize",
        "310A",
        "stage310a_disjoint_package_finalize.json",
        s310a["status"],
        (
            "在访问门打开前冻结Stage311下载实现和Stage312分析实现的"
            "SHA256；只记录实现哈希，不读取观测、不训练、不计算性能"
        ),
        (
            f"下载脚本哈希={s310a['download_script_sha256'][:12]}…；"
            f"分析脚本哈希={s310a['analysis_script_sha256'][:12]}…；"
            f"门打开时间={s310a['gate_opened_utc']}"
        ),
        (
            "实现冻结证明脚本版本可追溯，不证明科学结论；Stage312"
            "运行前未接触test标签。"
        ),
    )
    add(
        rows,
        "stage311_met_norway_disjoint_download",
        "311",
        "stage311_met_norway_disjoint_download.json",
        s311["status"],
        (
            "只从MET Norway Frost官方observations端点下载Stage310冻结"
            "的12个不相交站点2021-2025小时温度；按每站每年预期小时网格"
            "审计覆盖率，官方质量码0；不训练、不校准、不注入攻击、"
            "不计算性能"
        ),
        (
            f"预期请求={s311['expected_requests']}；缓存="
            f"{s311['cached_count']}；失败={s311['failed_count']}；"
            f"规范化行={s311['normalized_rows']}；保留站点="
            f"{s311['retained_station_count']}/{s311['station_count']}；"
            f"排除站点={','.join(s311['excluded_station_ids'])}；"
            f"覆盖率下限={s311['coverage_min']}"
        ),
        (
            "SN77490和SN89985因整年或测试期覆盖不达标被排除，官方"
            "404原样保留；覆盖分母是预期逐小时网格而非API返回行数；"
            "同一provider站网不能证明机构无关。"
        ),
    )
    add(
        rows,
        "stage312_met_norway_disjoint_locked_transfer",
        "312",
        "stage312_met_norway_disjoint_locked_transfer.json",
        s312["status"],
        (
            "在Stage310协议、Stage311下载审计和实现冻结之后，在10个"
            "不相交MET Norway站点上执行15个方法的锁定迁移；所有方法"
            "共享train/valid/test、透明受控攻击、5种子、FAR=0.01和"
            "event coverage主终点；test只评价"
        ),
        (
            f"迁移门槛={s312['transfer_gate_status']}；provider是否"
            f"为新机构={s312['provider_is_new']}；站点="
            f"{s312['station_count']}；种子={s312['seed_count']}；"
            f"方法={s312['method_count']}；主比较通过="
            f"{s312['primary_pass_count']}/"
            f"{s312['primary_comparison_count']}；全部基线支持="
            f"{s312['all_baselines_supported']}"
        ),
        (
            "主比较仅6/26通过；One-Class SVM三个预算在F1和event "
            "coverage上均显著优于两个Trust候选，重建基线也多数胜出；"
            "该不利结果必须保留，station_network_dependence为"
            "BOUNDED_BY_ADVERSE_DISJOINT_NETWORK；不关闭任何P0。"
        ),
    )
    add(
        rows,
        "stage313_cross_locked_test_win_loss_matrix",
        "313",
        "stage313_cross_locked_test_win_loss_matrix.json",
        s313["status"],
        (
            "只读统一Stage297、Stage307和Stage312三个已冻结锁定迁移的"
            "比较表与现有类型明细；不训练、不重算预测、不调阈值、不新增"
            "provider，也不把同provider的不同站网计为新机构"
        ),
        (
            f"来源={s313['source_count']}；锁定测试="
            f"{s313['locked_test_count']}；比较行="
            f"{s313['comparison_row_count']}；归一化主终点通过="
            f"{s313['normalized_primary_pass_count']}；未通过="
            f"{s313['normalized_primary_fail_count']}；全部基线在每个"
            f"锁定测试通过="
            f"{s313['all_baselines_supported_on_every_locked_test']}；"
            f"非stuck零事件覆盖单元="
            f"{s313['nonstuck_zero_event_cell_count']}；新增provider="
            f"{s313['new_provider_added']}"
        ),
        (
            "42/74条归一化主终点未通过，非stuck仍有1个零事件覆盖和"
            "1个零行召回单元；日本无类型明细，MET Norway两个网络属于"
            "同一provider，不能关闭P0-1至P0-4，也不支持全面领先。"
        ),
    )
    add(
        rows,
        "stage314_claim_ceiling_and_submission_scope_audit",
        "314",
        "stage314_claim_ceiling_and_submission_scope_audit.json",
        s314["status"],
        (
            "只读固化四个冻结主provider、停止新增provider的作者决定、"
            "P0阻断项、允许/禁止声称和论文措辞映射；不训练、不调参、"
            "不读取test标签、不新增性能结果"
        ),
        (
            f"主provider数={s314['provider_scope']['frozen_main_provider_count']}；"
            f"P0阻断={s314['p0_open_count']}；允许声称="
            f"{s314['allowed_claim_count']}；禁止声称="
            f"{s314['forbidden_claim_count']}；全面领先允许="
            f"{s314['universal_superiority_allowed']}；生产领先允许="
            f"{s314['production_superiority_allowed']}"
        ),
        (
            "该阶段只固化声称边界，不关闭P0-1至P0-6；四个主provider固定"
            "为ECCC、KNMI、MeteoSwiss、NOAA ISD；FMI/SMHI保持单机构"
            "锁定外部验证；同provider不同站网不计为新机构；工程PASS、"
            "哈希完整和干净环境复现均不能升级为性能或生产领先声称。"
        ),
    )
    add(
        rows,
        "stage315_noaa_2025_temporal_locked_transfer",
        "315",
        "stage315_noaa_2025_temporal_locked_transfer.json",
        s315["status"],
        (
            "在预先冻结的NOAA ISD 2025时间窗上执行同provider时间锁定"
            "迁移；复用Stage307的检测器、特征、阈值、事件和统计契约；"
            "2024旧test年份完全排除；不训练新变体、不调阈值、不新增"
            "provider"
        ),
        (
            f"provider新增={s315['provider_is_new']}；站网不相交于"
            f"Stage304={s315['network_is_disjoint_from_stage304']}；"
            f"站点={s315['station_count']}；种子={s315['seed_count']}；"
            f"方法={s315['method_count']}；主终点通过="
            f"{s315['primary_pass_count']}/{s315['primary_comparison_count']}；"
            f"全部基线支持={s315['all_baselines_supported']}；"
            f"迁移门={s315['transfer_gate_status']}"
        ),
        (
            "NOAA 2025仍是同一provider和同一11站网络，不能关闭P0-1或"
            "P0-6；16/26主比较通过但transfer_gate_status=NOT_PASSED，"
            "event coverage对ARIMA、GRU、EWMA和SPC proxy不利，非stuck"
            "F1对6个基线不利；所有攻击仍是透明受控注入，不能写成真实"
            "生产事故召回率。"
        ),
    )
    add(
        rows,
        "stage316_noaa_2025_failure_decomposition",
        "316",
        "stage316_noaa_2025_failure_decomposition.json",
        s316["status"],
        (
            "只读分解Stage315 NOAA 2025时间锁定结果；不训练、不调参、"
            "不新增test、不改变预测；逐条保留失败比较、攻击类型和站点"
            "层面的不利结果"
        ),
        (
            f"来源={s316['source_stage']}；主终点通过="
            f"{s316['primary_pass_count']}/{s316['primary_comparison_count']}；"
            f"失败比较={s316['primary_fail_count']}；全面支持="
            f"{s316['all_baselines_supported']}；P0-2="
            f"{s316['p0_2_status_after']}；P0-3="
            f"{s316['p0_3_status_after']}；P0-4="
            f"{s316['p0_4_status_after']}"
        ),
        (
            "该分解是诊断证据，不能把失败的迁移门升级为确认性结论；"
            "NOAA仍为同一provider，受控注入不等于生产日志，P0-2、"
            "P0-3和P0-4继续保持OPEN。"
        ),
    )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage29_evidence_matrix.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "purpose": "所有实验的唯一证据索引；不替代原始结果文件", "rows": rows, "counts": {"evidence_items": len(rows), "negative_results": int(sum("negative" in str(x["evidence_id"]) for x in rows)), "external_audits": 3, "cross_site_audits": 7}}
    (TABLES / "stage29_evidence_matrix.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段29实验证据矩阵", "", "该矩阵只索引原始实验产物，不覆盖、不删除不利结果；正式分析必须回到各阶段 JSON/CSV/Parquet。", "", "| 证据 | 阶段 | 状态 | test策略 | 主要指标 | 限制 |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['evidence_id']} | {r['stage']} | {r['status']} | {r['test_policy']} | {r['primary_metrics']} | {r['limitations']} |")
    lines += ["", "矩阵包含受控攻击、NOAA自然观测弱标签、ERA5辅助背景、跨站点冷启动/校准、滚动时间留出、负结果和运行曲线；这些证据的统计含义不同，不能合并成单一‘全面领先’结论。NOAA质量标记始终只作为弱标签。"]
    (REPORTS / "stage29_evidence_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "evidence_items": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
