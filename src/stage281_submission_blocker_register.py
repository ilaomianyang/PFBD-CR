from __future__ import annotations

"""Stage 281: consolidated submission blocker register.

This is a read-only register of unresolved scientific and engineering
blockers. It does not train, tune, or reinterpret any result. Its purpose is
to make the claim ceiling machine-readable so downstream documents cannot
promote a bounded result into a universal-superiority claim.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage281_submission_blocker_register"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


def _load(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stage259 = _load("stage259_fmi_locked_transfer")
    stage267 = _load("stage267_smhi_locked_transfer")
    stage274 = _load("stage274_ocsvm_full_scale_sensitivity")
    stage277 = _load("stage277_baseline_formality_matrix")
    stage278 = _load("stage278_confirmatory_test_package_contract")
    stage280 = _load("stage280_nonstuck_stability_audit")
    stage276 = _load("stage276_release_reproduction_input_audit")
    stage283 = _load("stage283_public_ics_cps_source_audit")
    stage287 = _load("stage287_swat_usad_evidence_upgrade_audit")
    stage289 = _load("stage289_final_baseline_set_freeze")
    stage290 = _load("stage290_same_contract_baseline_resource_budget")
    stage291 = _load("stage291_unified_win_loss_matrix")
    stage292 = _load("stage292_ocsvm_non_inferiority_boundary_audit")
    stage293 = _load("stage293_confirmatory_package_freeze_audit")
    stage294 = _load("stage294_japan_isd_protocol_preregistration")
    stage295 = _load("stage295_japan_isd_locked_evaluation_contract")
    stage296 = _load("stage296_japan_isd_official_download")
    stage297 = _load("stage297_japan_isd_locked_transfer")
    stage298 = _load("stage298_japan_transfer_failure_decomposition")
    stage302 = _load("stage302_japan_ocsvm_budget_grid")
    stage309 = _load("stage309_public_ics_cps_source_feasibility")
    stage310 = _load(
        "stage310_disjoint_met_norway_confirmatory_package"
    )
    stage310a = _load("stage310a_disjoint_package_finalize")
    stage311 = _load("stage311_met_norway_disjoint_download")
    stage312 = _load("stage312_met_norway_disjoint_locked_transfer")
    stage313 = _load("stage313_cross_locked_test_win_loss_matrix")
    stage315 = _load("stage315_noaa_2025_temporal_locked_transfer")
    stage316 = _load("stage316_noaa_2025_failure_decomposition")
    stage282 = _load("stage282_clean_environment_probe_audit")

    rows = [
        {
            "blocker_id": "P0-1_external_provider_gate",
            "category": "external_validity",
            "status": "OPEN",
            "evidence": (
                "FMI/SMHI are independently locked but do not demonstrate "
                "non-inferiority against every major baseline. The frozen "
                "Japanese NOAA station network also failed its transfer gate: "
                "18/22 primary comparisons passed, but event coverage was "
                "adverse for ARIMA innovation and the GRU predictor. The "
                "Japanese network is spatially disjoint from the earlier NOAA "
                "network but remains under the same NOAA provider. The "
                "Stage310-312 disjoint MET Norway station network is also "
                "not a new provider and failed its locked transfer gate "
                "with only 6/26 primary comparisons passing. The author has "
                "frozen the external scope at the four main providers and "
                "will not add another provider; the universal claim remains "
                "unavailable. The NOAA 2025 temporal holdout reuses the same "
                "provider and the same 11-station network: 16/26 primary "
                "comparisons passed, but the transfer gate is NOT_PASSED and "
                "event coverage is adverse against ARIMA innovation, the GRU "
                "predictor, EWMA deviation, and the SPC proxy."
            ),
            "claim_ceiling": (
                "cross-provider transfer evidence, not institution-agnostic "
                "universal superiority"
            ),
            "required_action": (
                "not pursued under the frozen four-provider scope; keep the "
                "claim bounded unless independently governed new evidence "
                "later becomes available"
            ),
            "source_stage": "stage259_fmi_locked_transfer;"
            "stage267_smhi_locked_transfer;"
            "stage294_japan_isd_protocol_preregistration;"
            "stage297_japan_isd_locked_transfer;"
            "stage310_disjoint_met_norway_confirmatory_package;"
            "stage311_met_norway_disjoint_download;"
            "stage312_met_norway_disjoint_locked_transfer;"
            "stage315_noaa_2025_temporal_locked_transfer;"
            "stage316_noaa_2025_failure_decomposition",
            "source_hash": sha256_file(
                TABLES / "stage297_japan_isd_locked_transfer.json"
            ),
        },
        {
            "blocker_id": "P0-2_ocsvm_non_inferiority",
            "category": "baseline_competition",
            "status": "OPEN",
            "evidence": (
                "The frozen Japan locked test completed the pre-registered "
                "20k/50k/100k One-Class SVM budget grid in Stage302. "
                "trust_valid_f1 passed both endpoints at all three budgets, "
                "but trust_nonstuck_event_macro failed F1 non-inferiority "
                "at 50k and 100k while passing event coverage. The contract "
                "requires both primary Trust candidates and all budgets to "
                "pass, so the pre-registered non-inferiority gate is not "
                "met. On the Stage312 disjoint MET Norway network, all "
                "three One-Class SVM budgets significantly outperformed "
                "both Trust candidates on F1 and event coverage."
            ),
            "claim_ceiling": "complementary-baseline claim only",
            "required_action": (
                "a new pre-registered confirmatory test in which both "
                "primary Trust candidates pass F1 and event coverage "
                "non-inferiority at every required One-Class SVM budget"
            ),
            "source_stage": (
                "stage274_ocsvm_full_scale_sensitivity;"
                "stage292_ocsvm_non_inferiority_boundary_audit;"
                "stage302_japan_ocsvm_budget_grid;"
                "stage312_met_norway_disjoint_locked_transfer"
            ),
            "source_hash": sha256_file(
                TABLES / "stage302_japan_ocsvm_budget_grid.json"
            ),
        },
        {
            "blocker_id": "P0-3_public_ics_cps_baseline",
            "category": "baseline_formality",
            "status": "OPEN",
            "evidence": (
                "The local ICS/CPS comparator is a transparent SPC proxy; "
                "fixed-commit USAD author source and author-linked SWaT data "
                "have now been formally evaluated under a frozen protocol, "
                "but the public SWaT endpoint is not the TrustMeteo-QC "
                "contract. The final same-contract baseline set is frozen, "
                "resource budgets are complete, and an existing-evidence "
                "win/loss ledger is available, but no new locked test has "
                "rerun all final baselines and the One-Class SVM "
                "non-inferiority gate remains unpassed. The Japanese locked "
                "transfer did rerun 13 methods under one contract, but "
                "all_baselines_supported=false. Stage312 reran 15 methods "
                "under one contract on the disjoint MET Norway network, "
                "but only 6/26 primary comparisons passed. Stage309 audited six "
                "additional public ICS/CPS candidates and found no source "
                "that is both same-contract ready and a production incident "
                "log. Stage315 reran 15 methods under the frozen NOAA 2025 "
                "temporal contract; only 16/26 primary comparisons passed, "
                "all_baselines_supported=false, and the transfer gate is "
                "NOT_PASSED."
            ),
            "claim_ceiling": (
                "do not describe the proxy as a paper-faithful public ICS/CPS "
                "reproduction"
            ),
            "required_action": (
                "run all final baselines on a new locked test with the same "
                "split, FAR, event, and evaluation contract, and pass the "
                "pre-registered advantage or non-inferiority gates"
            ),
            "source_stage": (
                "stage277_baseline_formality_matrix;"
                "stage287_swat_usad_evidence_upgrade_audit;"
                "stage289_final_baseline_set_freeze;"
                "stage290_same_contract_baseline_resource_budget;"
                "stage291_unified_win_loss_matrix;"
                "stage295_japan_isd_locked_evaluation_contract;"
                "stage297_japan_isd_locked_transfer;"
                "stage312_met_norway_disjoint_locked_transfer;"
                "stage315_noaa_2025_temporal_locked_transfer;"
                "stage316_noaa_2025_failure_decomposition;"
                "stage309_public_ics_cps_source_feasibility"
            ),
            "source_hash": sha256_file(
                TABLES / "stage297_japan_isd_locked_transfer.json"
            ),
        },
        {
            "blocker_id": "P0-4_nonstuck_stability",
            "category": "attack_type_stability",
            "status": "OPEN",
            "evidence": (
                "Provider-type event coverage and row recall still contain "
                "zero-valued cells; seed-level point directions do not remove "
                "them. On the Japanese locked transfer, event coverage was "
                "lower than ARIMA innovation and the GRU predictor. On the "
                "Stage312 disjoint MET Norway network, both Trust candidates "
                "also lost event coverage to multiple baselines, including "
                "One-Class SVM, ARIMA innovation, and reconstruction models. "
                "On the NOAA 2025 temporal holdout, trust_nonstuck_event_macro "
                "lost F1 to ARIMA innovation, EWMA deviation, the GRU "
                "predictor, the SPC proxy, PCA reconstruction, and the "
                "lightweight autoencoder; trust_valid_f1 also lost event "
                "coverage to ARIMA innovation, EWMA deviation, the GRU "
                "predictor, and the SPC proxy."
            ),
            "claim_ceiling": (
                "limit the primary contribution to stuck/repeated/frozen "
                "states unless a new locked test passes"
            ),
            "required_action": (
                "pre-register non-stuck on a new independent locked test and "
                "pass provider-type and leave-one-provider gates"
            ),
            "source_stage": (
                "stage280_nonstuck_stability_audit;"
                "stage297_japan_isd_locked_transfer;"
                "stage312_met_norway_disjoint_locked_transfer;"
                "stage315_noaa_2025_temporal_locked_transfer;"
                "stage316_noaa_2025_failure_decomposition"
            ),
            "source_hash": sha256_file(
                TABLES / "stage297_japan_isd_locked_transfer.json"
            ),
        },
        {
            "blocker_id": "P0-5_real_production_labels",
            "category": "real_world_evidence",
            "status": "OPEN",
            "evidence": (
                "The project contains controlled injections, an authorized "
                "testbed, and weak natural labels, but no authorized production "
                "incident log with independent event labels. Stage309 audited "
                "SWaT, WADI, BATADAL, HAI, SMD, and SMAP/MSL; all are testbed, "
                "simulation, or different-domain benchmarks rather than "
                "production incident logs."
            ),
            "claim_ceiling": (
                "no production attack recall or production-superiority claim"
            ),
            "required_action": (
                "obtain an authorized, auditable production or public ICS/CPS "
                "incident log and freeze its evaluation protocol before use"
            ),
            "source_stage": (
                "stage278_confirmatory_test_package_contract;"
                "stage309_public_ics_cps_source_feasibility"
            ),
            "source_hash": sha256_file(
                TABLES / "stage309_public_ics_cps_source_feasibility.json"
            ),
        },
        {
            "blocker_id": "P0-6_new_confirmatory_test",
            "category": "test_isolation",
            "status": "OPEN",
            "evidence": (
                "Stage278 is a contract template only. Stage293 finds a "
                "hash-frozen package with package_present=true and "
                "analysis_frozen=true, but new_provider_selected=false and "
                "is_a_new_confirmatory_test_package=false. The Stage310-312 "
                "package is hash-frozen and implementation-frozen before "
                "test access, but its provider_is_new=false, so it cannot "
                "satisfy the new-provider requirement. Stage315 is a new "
                "2025 temporal holdout on the same NOAA provider and the same "
                "11-station network, so it is robustness evidence rather than "
                "a new independent confirmatory package."
            ),
            "claim_ceiling": (
                "existing tests remain confirmatory only for their frozen "
                "versions; new variants are exploratory"
            ),
            "required_action": (
                "not pursued under the frozen four-provider scope; new "
                "variants remain exploratory unless a genuinely new locked "
                "test is later authorized"
            ),
            "source_stage": (
                "stage278_confirmatory_test_package_contract;"
                "stage293_confirmatory_package_freeze_audit;"
                "stage294_japan_isd_protocol_preregistration;"
                "stage295_japan_isd_locked_evaluation_contract;"
                "stage296_japan_isd_official_download;"
                "stage297_japan_isd_locked_transfer;"
                "stage310_disjoint_met_norway_confirmatory_package;"
                "stage310a_disjoint_package_finalize;"
                "stage311_met_norway_disjoint_download;"
                "stage312_met_norway_disjoint_locked_transfer;"
                "stage315_noaa_2025_temporal_locked_transfer;"
                "stage316_noaa_2025_failure_decomposition"
            ),
            "source_hash": sha256_file(
                TABLES / "stage297_japan_isd_locked_transfer.json"
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    if frame["blocker_id"].duplicated().any():
        raise RuntimeError("duplicate submission blocker ids")
    if not frame["status"].eq("OPEN").all():
        raise RuntimeError("unexpected blocker status")

    payload = {
        "status": "OPEN_SUBMISSION_BLOCKERS_RECORDED",
        "protocol": (
            "Stage281 only reads existing audit summaries and records open "
            "submission blockers. It does not train, tune, alter predictions, "
            "or upgrade any claim."
        ),
        "blocker_count": int(len(frame)),
        "open_count": int(frame["status"].eq("OPEN").sum()),
        "p0_open_count": int(
            frame["blocker_id"].str.startswith("P0-").sum()
        ),
        "p1_open_count": int(
            frame["blocker_id"].str.startswith("P1_").sum()
        ),
        "universal_superiority_allowed": False,
        "production_superiority_allowed": False,
        "clean_environment_reproduction_allowed": bool(
            stage282.get("clean_install_completed")
        ),
        "claim_ceiling": (
            "bounded evidence under controlled and authorized conditions; "
            "no universal, institution-agnostic, production, or clean-install "
            "superiority claim"
        ),
        "rows": frame.to_dict("records"),
        "supporting_values": {
            "fmi_transfer_status": stage259.get("status"),
            "smhi_transfer_status": stage267.get("status"),
            "ocsvm_status": stage274.get("status"),
            "ocsvm_non_inferiority_status": stage292.get(
                "non_inferiority_status"
            ),
            "ocsvm_budget_grid_status": stage302.get("status"),
            "ocsvm_budget_grid_complete": stage302.get(
                "budget_grid_complete"
            ),
            "ocsvm_observed_budgets": stage302.get("observed_budgets"),
            "ocsvm_all_smaller_budgets_non_inferior": stage302.get(
                "all_smaller_budgets_non_inferior"
            ),
            "ocsvm_candidate_budget_summary": stage292.get(
                "candidate_budget_summary"
            ),
            "ocsvm_pre_registered_margin_available": stage292.get(
                "pre_registered_margin_available"
            ),
            "ocsvm_new_locked_test_available": stage292.get(
                "new_locked_test_available"
            ),
            "p0_3_status": stage277.get("p0_3_status"),
            "public_ics_cps_author_source_verified": stage283.get(
                "required_author_sources_verified"
            ),
            "public_ics_cps_data_available": bool(
                stage287.get("paper_faithful_public_baseline_available")
            ),
            "public_ics_cps_formal_evaluation": stage287.get(
                "public_ics_cps_formal_implementation_subitem"
            ),
            "public_ics_cps_same_contract_comparison": stage287.get(
                "same_contract_comparison"
            ),
            "final_baseline_set_frozen": stage289.get(
                "final_set_frozen"
            ),
            "same_contract_resource_budget_complete": (
                stage290.get("completed_method_count")
                == stage290.get("method_count")
                and stage290.get("failed_method_count") == 0
            ),
            "existing_evidence_win_loss_ledger_available": stage291.get(
                "existing_evidence_win_loss_ledger_available"
            ),
            "same_contract_win_loss_matrix_available": stage291.get(
                "same_contract_win_loss_matrix_available"
            ),
            "same_contract_win_loss_matrix_is_new_locked_test": (
                stage291.get(
                    "same_contract_win_loss_matrix_is_new_locked_test"
                )
            ),
            "new_confirmatory_test": stage278.get(
                "is_a_new_confirmatory_test_package"
            ),
            "confirmatory_package_present": stage293.get(
                "package_present"
            ),
            "confirmatory_package_new_provider_selected": stage293.get(
                "new_provider_selected"
            ),
            "confirmatory_package_is_new_test": stage293.get(
                "is_a_new_confirmatory_test_package"
            ),
            "confirmatory_package_test_labels_inspected": stage293.get(
                "test_labels_inspected"
            ),
            "confirmatory_package_claim_ceiling": stage293.get(
                "claim_ceiling"
            ),
            "nonstuck_status": stage280.get("status"),
            "release_data_level_reproduction": stage276.get(
                "data_level_reproduction_verified"
            ),
            "clean_environment_reproduction_completed": stage282.get(
                "clean_install_completed"
            ),
            "clean_environment_isolated_python": stage282.get(
                "isolated_python"
            ),
            "japan_network_protocol_status": stage294.get("status"),
            "japan_network_provider_independent": stage294.get(
                "independent_source"
            ),
            "japan_network_station_count": stage294.get("station_count"),
            "japan_network_quality_accepted_code": stage294.get(
                "quality_accepted_code"
            ),
            "japan_network_no_observation_access_before_freeze": (
                stage294.get("strict_no_observation_access_before_freeze")
            ),
            "japan_evaluation_method_count": len(
                stage295.get("methods", [])
            ),
            "japan_evaluation_primary_far": stage295.get("primary_far"),
            "japan_evaluation_primary_endpoint": stage295.get(
                "primary_endpoint"
            ),
            "japan_evaluation_non_inferiority_margin": stage295.get(
                "non_inferiority_margin"
            ),
            "japan_download_expected_archives": stage296.get(
                "expected_archives"
            ),
            "japan_download_success_count": int(
                stage296.get("downloaded_count", 0)
            )
            + int(stage296.get("cached_count", 0)),
            "japan_download_failed_count": stage296.get("failed_count"),
            "japan_normalized_rows": stage296.get("normalized_rows"),
            "japan_retained_station_count": stage296.get(
                "retained_station_count"
            ),
            "japan_transfer_gate_status": stage297.get(
                "transfer_gate_status"
            ),
            "japan_transfer_primary_pass_count": stage297.get(
                "primary_pass_count"
            ),
            "japan_transfer_primary_comparison_count": stage297.get(
                "primary_comparison_count"
            ),
            "japan_transfer_all_baselines_supported": stage297.get(
                "all_baselines_supported"
            ),
            "japan_transfer_failed_comparison_count": stage298.get(
                "failed_comparison_count"
            ),
            "japan_transfer_failed_comparisons": stage298.get(
                "failed_comparisons"
            ),
            "japan_transfer_station_level_detail_available": (
                stage298.get("station_level_detail_available")
            ),
            "japan_transfer_attack_type_level_detail_available": (
                stage298.get("attack_type_level_detail_available")
            ),
            "public_ics_cps_candidate_count": stage309.get(
                "candidate_count"
            ),
            "public_ics_cps_blocked_candidate_count": stage309.get(
                "blocked_candidate_count"
            ),
            "public_ics_cps_testbed_candidate_count": stage309.get(
                "testbed_candidate_count"
            ),
            "public_ics_cps_production_incident_log_count": stage309.get(
                "production_incident_log_count"
            ),
            "public_ics_cps_same_contract_ready_count": stage309.get(
                "same_contract_ready_count"
            ),
            "public_ics_cps_p0_3_status_after": stage309.get(
                "p0_3_status_after"
            ),
            "public_ics_cps_p0_5_status_after": stage309.get(
                "p0_5_status_after"
            ),
            "disjoint_met_norway_provider_is_new": stage310.get(
                "provider_is_new"
            ),
            "disjoint_met_norway_network_is_disjoint_from_stage304": (
                stage310.get("network_is_disjoint_from_stage304")
            ),
            "disjoint_met_norway_download_script_sha256": stage310a.get(
                "download_script_sha256"
            ),
            "disjoint_met_norway_analysis_script_sha256": stage310a.get(
                "analysis_script_sha256"
            ),
            "disjoint_met_norway_retained_station_count": stage311.get(
                "retained_station_count"
            ),
            "disjoint_met_norway_excluded_station_ids": stage311.get(
                "excluded_station_ids"
            ),
            "disjoint_met_norway_transfer_gate_status": stage312.get(
                "transfer_gate_status"
            ),
            "disjoint_met_norway_primary_pass_count": stage312.get(
                "primary_pass_count"
            ),
            "disjoint_met_norway_primary_comparison_count": stage312.get(
                "primary_comparison_count"
            ),
            "disjoint_met_norway_all_baselines_supported": stage312.get(
                "all_baselines_supported"
            ),
            "disjoint_met_norway_station_network_dependence": (
                stage312.get("promotion_status", {}).get(
                    "station_network_dependence"
                )
            ),
            "cross_locked_test_comparison_row_count": stage313.get(
                "comparison_row_count"
            ),
            "cross_locked_test_normalized_primary_pass_count": (
                stage313.get("normalized_primary_pass_count")
            ),
            "cross_locked_test_normalized_primary_fail_count": (
                stage313.get("normalized_primary_fail_count")
            ),
            "cross_locked_test_all_baselines_supported": stage313.get(
                "all_baselines_supported_on_every_locked_test"
            ),
            "cross_locked_test_nonstuck_zero_event_cell_count": (
                stage313.get("nonstuck_zero_event_cell_count")
            ),
            "cross_locked_test_new_provider_added": stage313.get(
                "new_provider_added"
            ),
            "cross_locked_test_same_provider_networks_counted_as_new": (
                stage313.get(
                    "same_provider_networks_counted_as_new_provider"
                )
            ),
            "noaa_2025_provider_is_new": stage315.get("provider_is_new"),
            "noaa_2025_network_is_disjoint_from_stage304": stage315.get(
                "network_is_disjoint_from_stage304"
            ),
            "noaa_2025_transfer_gate_status": stage315.get(
                "transfer_gate_status"
            ),
            "noaa_2025_primary_pass_count": stage315.get(
                "primary_pass_count"
            ),
            "noaa_2025_primary_comparison_count": stage315.get(
                "primary_comparison_count"
            ),
            "noaa_2025_all_baselines_supported": stage315.get(
                "all_baselines_supported"
            ),
            "noaa_2025_failure_comparison_count": stage316.get(
                "primary_fail_count"
            ),
            "noaa_2025_p0_2_status_after": stage316.get(
                "p0_2_status_after"
            ),
            "noaa_2025_p0_3_status_after": stage316.get(
                "p0_3_status_after"
            ),
            "noaa_2025_p0_4_status_after": stage316.get(
                "p0_4_status_after"
            ),
        },
        "limitations": [
            "An OPEN blocker is not evidence that the method failed; it is "
            "evidence that the stated universal claim is not yet supported.",
            "Removing a blocker requires new evidence, not renaming or "
            "reformatting an existing result.",
            "Controlled, testbed, and weak-label evidence are not production "
            "incident labels.",
        ],
        "input_sha256": {
            name: sha256_file(TABLES / f"{name}.json")
            for name in [
                "stage259_fmi_locked_transfer",
                "stage267_smhi_locked_transfer",
                "stage274_ocsvm_full_scale_sensitivity",
                "stage277_baseline_formality_matrix",
                "stage278_confirmatory_test_package_contract",
                "stage280_nonstuck_stability_audit",
                "stage276_release_reproduction_input_audit",
                "stage283_public_ics_cps_source_audit",
                "stage287_swat_usad_evidence_upgrade_audit",
                "stage289_final_baseline_set_freeze",
                "stage290_same_contract_baseline_resource_budget",
                "stage291_unified_win_loss_matrix",
                "stage292_ocsvm_non_inferiority_boundary_audit",
                "stage293_confirmatory_package_freeze_audit",
                "stage294_japan_isd_protocol_preregistration",
                "stage295_japan_isd_locked_evaluation_contract",
                "stage296_japan_isd_official_download",
                "stage297_japan_isd_locked_transfer",
                "stage298_japan_transfer_failure_decomposition",
                "stage302_japan_ocsvm_budget_grid",
                "stage309_public_ics_cps_source_feasibility",
                "stage310_disjoint_met_norway_confirmatory_package",
                "stage310a_disjoint_package_finalize",
                "stage311_met_norway_disjoint_download",
                "stage312_met_norway_disjoint_locked_transfer",
                "stage282_clean_environment_probe_audit",
                "stage313_cross_locked_test_win_loss_matrix",
                "stage315_noaa_2025_temporal_locked_transfer",
                "stage316_noaa_2025_failure_decomposition",
            ]
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    frame.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage 281 提交阻断登记",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 阻断项：`{payload['blocker_count']}`（P0 `{payload['p0_open_count']}`，P1 `{payload['p1_open_count']}`）",
        f"- 是否允许全面领先：`{payload['universal_superiority_allowed']}`",
        f"- 是否允许真实生产领先：`{payload['production_superiority_allowed']}`",
        f"- 是否允许声称干净环境复现：`{payload['clean_environment_reproduction_allowed']}`",
        "",
        "## 当前声称上限",
        "",
        f"- {payload['claim_ceiling']}",
        "",
        "## 阻断项",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['blocker_id']}`（{row['category']}）：{row['evidence']} "
            f"所需证据：{row['required_action']}"
        )
    lines += [
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "blockers": payload["blocker_count"],
                "p0_open": payload["p0_open_count"],
                "p1_open": payload["p1_open_count"],
                "universal_superiority_allowed": payload[
                    "universal_superiority_allowed"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
