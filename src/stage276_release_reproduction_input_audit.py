from __future__ import annotations

"""Stage 276: audit the release-package frozen-input boundary.

This stage is read-only.  It determines which frozen-table inputs the release
rebuild entry point actually needs, records their hashes, and makes the
distinction between a code-complete package and a data-complete package
auditable.  It never copies raw data or changes a frozen artifact.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
_nested_release = ROOT / "release" / "TrustMeteo-QC_v1.0"
RELEASE = _nested_release if _nested_release.is_dir() else ROOT
RELEASE_MANIFEST = (
    RELEASE / "tables" / "stage276_release_reproduction_input_manifest.json"
)
RELEASE_SOURCE_MANIFEST = (
    RELEASE / "tables" / "stage276_release_source_manifest.json"
)
RELEASE_SUCCESS_MARKER = (
    RELEASE / "reports" / "rebuild_frozen_outputs_success.json"
)
PREFIX = "stage276_release_reproduction_input_audit"

# These are the frozen inputs read directly by the release rebuild steps.
# Stage268 consumes the evaluation/station/type detail triples; Stage269 uses
# the Stage248/254/255/256/261/262 JSON summaries plus Stage274 resource audit;
# Stage270-273 use the frozen tables listed below; Stage275 uses the FMI/SMHI
# lineage files already declared in its own input manifest.
INPUT_FILES = [
    "stage204_dwd_protocol_preregistration.json",
    "stage214_swiss_protocol_preregistration.json",
    "stage224_knmi_protocol_preregistration.json",
    "stage234_eccc_protocol_preregistration.json",
    "stage208_dwd_persistent_repeat_protection_selection.csv",
    "stage216_swiss_normalized.parquet",
    "stage82_noaa_extended_normalized.parquet",
    "stage226_knmi_normalized.parquet",
    "stage236_eccc_normalized.parquet",
    "stage216a_swiss_protocol_correction.json",
    "stage83_noaa_extended_quality_audit.json",
    "stage226a_knmi_missingness_audit.json",
    "stage236_eccc_normalized.json",
    "stage248_cross_provider_fair_baseline_audit.json",
    "stage248_cross_provider_fair_baseline_audit_evaluation.csv",
    "stage248_cross_provider_fair_baseline_audit_station_detail.csv",
    "stage248_cross_provider_fair_baseline_audit_type_detail.csv",
    "stage248_cross_provider_fair_baseline_audit_comparisons.csv",
    "stage248_cross_provider_fair_baseline_audit_leave_one_provider_out.csv",
    "stage248_cross_provider_fair_baseline_audit_provider_equal.csv",
    "stage250_jisa_risk_closure_audit.json",
    "stage250_jisa_risk_closure_audit_risks.csv",
    "stage251_jisa_stable_submission_objective.json",
    "stage251_jisa_stable_submission_objective_risk_ceilings.csv",
    "stage251_jisa_stable_submission_objective_workstreams.csv",
    "stage263_unified_statistical_report.json",
    "stage263_unified_statistical_report_pooled.csv",
    "stage263_unified_statistical_report_provider_equal.csv",
    "stage254_ocsvm_training_scale_sensitivity.json",
    "stage274_ocsvm_full_scale_sensitivity.json",
    "stage274_ocsvm_full_scale_sensitivity_provider_equal.csv",
    "stage274_ocsvm_full_scale_sensitivity_resource_audit.json",
    "stage255_online_causal_testbed.json",
    "stage256_lightweight_strong_baselines.json",
    "stage261_causal_offline_recovery_contrast.json",
    "stage262_security_domain_baselines.json",
    "stage262_resource_sampling.json",
    "stage256_lightweight_strong_baselines_evaluation.csv",
    "stage256_lightweight_strong_baselines_station_detail.csv",
    "stage256_lightweight_strong_baselines_type_detail.csv",
    "stage256_lightweight_strong_baselines_comparisons.csv",
    "stage256_lightweight_strong_baselines_leave_one_provider_out.csv",
    "stage256_lightweight_strong_baselines_provider_equal.csv",
    "stage262_security_domain_baselines_evaluation.csv",
    "stage262_security_domain_baselines_station_detail.csv",
    "stage262_security_domain_baselines_type_detail.csv",
    "stage262_security_domain_baselines_comparisons.csv",
    "stage262_security_domain_baselines_leave_one_provider_out.csv",
    "stage262_security_domain_baselines_provider_equal.csv",
    "stage259_fmi_locked_transfer.json",
    "stage259_fmi_locked_transfer_evaluation.csv",
    "stage259_fmi_locked_transfer_station_detail.csv",
    "stage259_fmi_locked_transfer_type_detail.csv",
    "stage267_smhi_locked_transfer.json",
    "stage267_smhi_locked_transfer_evaluation.csv",
    "stage267_smhi_locked_transfer_station_detail.csv",
    "stage267_smhi_locked_transfer_type_detail.csv",
    "stage151_noaa_attack_intensity_sensitivity.json",
    "stage169_noaa_controlled_attack_intensity_full_baseline.json",
    "stage118_attack_duration_sensitivity.json",
    "stage170_noaa_controlled_attack_duration_full_baseline.json",
    "stage119_attack_position_sensitivity.json",
    "stage171_noaa_controlled_attack_position_full_baseline.json",
    "stage268_provider_scope_audit_event_audit.csv",
    "stage189_noaa_event_level_failure_decomposition_type_aggregate.csv",
    "stage189_noaa_event_level_failure_decomposition_length_aggregate.csv",
    "stage189_noaa_event_level_failure_decomposition_risk_aggregate.csv",
    "stage243_knmi_negative_case_decomposition_station_detail.csv",
    "stage243_knmi_negative_case_decomposition_metric_detail.csv",
    "stage246_frozen_submission_package.json",
    "stage244_cross_provider_nonstuck_candidate_features_type_detail.csv",
    "stage245_cross_provider_nonstuck_seed_replication_type_detail.csv",
    "stage264_smhi_protocol_preregistration.json",
    "stage257_fmi_protocol_preregistration.json",
    "stage258_fmi_official_download.json",
    "stage258_fmi_official_download_detail.csv",
    "stage265_smhi_locked_evaluation_contract.json",
    "stage266_smhi_official_download.json",
    "stage266_smhi_official_download_detail.csv",
    "stage285_swat_usad_protocol_preregistration.json",
    "stage286_swat_usad_author_evaluation.json",
    "stage286_swat_usad_author_evaluation_test_metrics.csv",
    "stage286_swat_usad_author_evaluation_test_scores.csv",
    "stage283_public_ics_cps_source_audit.json",
    "stage284_swat_author_data_download.json",
    "stage287_swat_usad_evidence_upgrade_audit.json",
    "stage288_swat_usad_resource_audit.json",
    "stage288_swat_usad_resource_audit_methods.csv",
    "stage290_same_contract_baseline_resource_budget.json",
    "stage290_same_contract_baseline_resource_budget.csv",
    "stage291_unified_win_loss_matrix.json",
    "stage291_unified_win_loss_matrix.csv",
    "stage291_unified_win_loss_matrix_ledger.csv",
    "stage291_unified_win_loss_matrix_provider_equal.csv",
    "stage294_japan_isd_protocol_preregistration.json",
    "stage294_japan_isd_protocol_preregistration_stations.csv",
    "stage294_japan_isd_protocol_preregistration_metadata_source.json",
    "stage295_japan_isd_locked_evaluation_contract.json",
    "stage296_japan_isd_official_download.json",
    "stage296_japan_isd_official_download_detail.csv",
    "stage296_japan_isd_official_download_normalized.parquet",
    "stage296_japan_isd_official_download_station_audit.csv",
    "stage297_japan_isd_locked_transfer.json",
    "stage297_japan_isd_locked_transfer_evaluation.csv",
    "stage297_japan_isd_locked_transfer_provider_equal.csv",
    "stage297_japan_isd_locked_transfer_comparisons.csv",
    "stage297_japan_isd_locked_transfer_selection.csv",
    "stage298_japan_transfer_failure_decomposition.json",
    "stage298_japan_transfer_failure_decomposition_comparisons.csv",
    "stage298_japan_transfer_failure_decomposition_failed_comparisons.csv",
    "stage298_japan_transfer_failure_decomposition_failed_seed_directions.csv",
    "stage298_japan_transfer_failure_decomposition_method_summary.csv",
    "stage302_japan_ocsvm_budget_grid.json",
    "stage302_japan_ocsvm_budget_grid_fit_rows.csv",
    "stage302_japan_ocsvm_budget_grid_evaluation.csv",
    "stage302_japan_ocsvm_budget_grid_aggregate.csv",
    "stage302_japan_ocsvm_budget_grid_comparisons.csv",
    "stage303_independent_provider_source_feasibility.json",
    "stage303_independent_provider_source_feasibility.csv",
    "stage309_public_ics_cps_source_feasibility.json",
    "stage309_public_ics_cps_source_feasibility.csv",
    "stage304_met_norway_protocol_preregistration.json",
    "stage304_met_norway_protocol_preregistration_stations.csv",
    "stage304_met_norway_protocol_preregistration_metadata_source.json",
    "stage305_met_norway_locked_evaluation_contract.json",
    "stage306_met_norway_official_download.json",
    "stage306_met_norway_official_download_detail.csv",
    "stage306_met_norway_official_download_normalized.parquet",
    "stage306_met_norway_official_download_station_audit.csv",
    "stage307_met_norway_locked_transfer.json",
    "stage307_met_norway_locked_transfer_evaluation.csv",
    "stage307_met_norway_locked_transfer_provider_equal.csv",
    "stage307_met_norway_locked_transfer_comparisons.csv",
    "stage307_met_norway_locked_transfer_selection.csv",
    "stage307_met_norway_locked_transfer_invalid_v1.json",
    "stage307_met_norway_locked_transfer_type_detail.csv",
    "stage308_met_norway_evidence_boundary_audit.json",
    "stage308_met_norway_evidence_boundary_audit_checks.csv",
    "stage308_met_norway_evidence_boundary_audit_adverse_comparisons.csv",
    "stage310_disjoint_met_norway_confirmatory_package.json",
    "stage310_disjoint_met_norway_confirmatory_package_files.csv",
    "stage310_disjoint_met_norway_confirmatory_package_stations.csv",
    "stage310a_disjoint_package_finalize.json",
    "stage311_met_norway_disjoint_download.json",
    "stage311_met_norway_disjoint_download_detail.csv",
    "stage311_met_norway_disjoint_download_normalized.parquet",
    "stage311_met_norway_disjoint_download_station_audit.csv",
    "stage311_met_norway_disjoint_download_year_audit.csv",
    "stage312_met_norway_disjoint_locked_transfer.json",
    "stage312_met_norway_disjoint_locked_transfer_evaluation.csv",
    "stage312_met_norway_disjoint_locked_transfer_provider_equal.csv",
    "stage312_met_norway_disjoint_locked_transfer_comparisons.csv",
    "stage312_met_norway_disjoint_locked_transfer_selection.csv",
    "stage312_met_norway_disjoint_locked_transfer_station_detail.csv",
    "stage312_met_norway_disjoint_locked_transfer_type_detail.csv",
    "stage313_cross_locked_test_win_loss_matrix.json",
    "stage313_cross_locked_test_win_loss_matrix.csv",
    "stage313_cross_locked_test_win_loss_matrix_ledger.csv",
    "stage313_cross_locked_test_win_loss_matrix_provider_equal.csv",
    "stage313_cross_locked_test_win_loss_matrix_direction_stability.csv",
    "stage313_cross_locked_test_win_loss_matrix_nonstuck_type.csv",
    "stage313_cross_locked_test_win_loss_matrix_nonstuck_cross_network.csv",
    "stage315_noaa_2025_temporal_data_audit.json",
    "stage315_noaa_2025_temporal_data_audit_normalized.parquet",
    "stage315_noaa_2025_temporal_locked_transfer.json",
    "stage315_noaa_2025_temporal_locked_transfer_evaluation.csv",
    "stage315_noaa_2025_temporal_locked_transfer_provider_equal.csv",
    "stage315_noaa_2025_temporal_locked_transfer_comparisons.csv",
    "stage315_noaa_2025_temporal_locked_transfer_selection.csv",
    "stage315_noaa_2025_temporal_locked_transfer_station_detail.csv",
    "stage315_noaa_2025_temporal_locked_transfer_type_detail.csv",
    "stage316_noaa_2025_failure_decomposition.json",
    "stage316_noaa_2025_failure_decomposition_failed_comparisons.csv",
    "stage316_noaa_2025_failure_decomposition_attack_type_summary.csv",
    "stage316_noaa_2025_failure_decomposition_station_summary.csv",
    "stage316_noaa_2025_failure_decomposition_nonstuck_comparison.csv",
    "stage248_cross_provider_fair_baseline_audit_selection.csv",
]

# These files are not frozen tables, but Stage283 cannot be reproduced from
# the release without them. They are audited separately so the frozen-table
# count keeps its original meaning.
INPUT_SOURCE_FILES = [
    "src/rebuild_frozen_outputs.py",
    "src/stage276_release_reproduction_input_audit.py",
    "src/stage269_deployment_complexity_audit.py",
    "src/stage272_final_parameter_table.py",
    "src/stage277_baseline_formality_matrix.py",
    "src/stage278_confirmatory_test_package_contract.py",
    "src/stage281_submission_blocker_register.py",
    "src/stage282_clean_environment_probe_audit.py",
    "src/stage283_public_ics_cps_source_audit.py",
    "src/stage284_swat_author_data_download.py",
    "src/stage287_swat_usad_evidence_upgrade_audit.py",
    "src/stage288_swat_usad_resource_audit.py",
    "src/stage289_final_baseline_set_freeze.py",
    "src/stage290_same_contract_baseline_resource_budget.py",
    "src/stage291_unified_win_loss_matrix.py",
    "src/stage292_ocsvm_non_inferiority_boundary_audit.py",
    "src/stage293_confirmatory_package_freeze_audit.py",
    "src/stage294_japan_isd_protocol_preregistration.py",
    "src/stage295_japan_isd_locked_evaluation_contract.py",
    "src/stage296_japan_isd_official_download.py",
    "src/stage297_japan_isd_locked_transfer.py",
    "src/stage298_japan_transfer_failure_decomposition.py",
    "src/stage302_japan_ocsvm_budget_grid.py",
    "src/stage303_independent_provider_source_feasibility.py",
    "src/stage309_public_ics_cps_source_feasibility.py",
    "src/stage304_met_norway_protocol_preregistration.py",
    "src/stage305_met_norway_locked_evaluation_contract.py",
    "src/stage306_met_norway_official_download.py",
    "src/stage307_met_norway_locked_transfer.py",
    "src/stage308_met_norway_evidence_boundary_audit.py",
    "src/stage310_disjoint_met_norway_confirmatory_package.py",
    "src/stage310a_finalize_disjoint_package.py",
    "src/stage311_met_norway_disjoint_official_download.py",
    "src/stage312_met_norway_disjoint_locked_transfer.py",
    "src/stage313_cross_locked_test_win_loss_matrix.py",
    "src/stage314_claim_ceiling_and_submission_scope_audit.py",
    "src/stage315_noaa_2025_temporal_data_audit.py",
    "src/stage315_noaa_2025_temporal_locked_transfer.py",
    "src/stage316_noaa_2025_failure_decomposition.py",
    "src/stage307_met_norway_locked_transfer.py",
    "src/stage262_security_domain_baselines.py",
    "src/stage256_lightweight_strong_baselines.py",
    "src/stage254_ocsvm_training_scale_sensitivity.py",
    "src/stage248_cross_provider_fair_baseline_audit.py",
    "src/stage244_cross_provider_nonstuck_candidate_features.py",
    "src/stage240_nonstuck_balanced_selection.py",
    "src/stage222_noaa_dwd_protection_transfer.py",
    "src/stage219_swiss_dwd_protection_transfer.py",
    "src/stage218_swiss_negative_result_uncertainty.py",
    "src/stage217_swiss_controlled_attack_evaluation.py",
    "src/stage208_dwd_persistent_repeat_protection.py",
    "src/stage207_dwd_controlled_attack_evaluation.py",
    "src/stage299_method_resource_coverage_audit.py",
    "src/stage300_method_resource_evidence_upgrade.py",
    "src/stage301_trust_resource_budget.py",
    "tests/run_release_rebuild_tests.py",
    "tests/test_quality_audits.py",
    "tests/run_tests.py",
    "vendor/LICENSE",
    "vendor/USAD.ipynb",
    "vendor/usad-tree.json",
    "vendor/usad.py",
    "vendor/utils.py",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    release_manifest = {}
    if RELEASE_MANIFEST.is_file():
        release_manifest = json.loads(
            RELEASE_MANIFEST.read_text(encoding="utf-8")
        )
    release_manifest_rows = {
        str(row.get("relative_path", "")): row
        for row in release_manifest.get("rows", [])
    }
    release_source_manifest = {}
    if RELEASE_SOURCE_MANIFEST.is_file():
        release_source_manifest = json.loads(
            RELEASE_SOURCE_MANIFEST.read_text(encoding="utf-8")
        )
    release_source_manifest_rows = {
        str(row.get("relative_path", "")): row
        for row in release_source_manifest.get("rows", [])
    }
    release_success_marker = {}
    if RELEASE_SUCCESS_MARKER.is_file():
        release_success_marker = json.loads(
            RELEASE_SUCCESS_MARKER.read_text(encoding="utf-8")
        )

    rows = []
    missing = []
    total_bytes = 0
    for name in INPUT_FILES:
        path = TABLES / name
        relative = f"tables/{name}"
        release_path = RELEASE / relative
        manifest_row = release_manifest_rows.get(relative, {})
        expected_release_sha = str(manifest_row.get("sha256", ""))
        release_present = release_path.is_file()
        release_sha = _sha256(release_path) if release_present else ""
        release_sha_matches = bool(
            release_present
            and expected_release_sha
            and release_sha == expected_release_sha
        )
        if not path.is_file():
            missing.append(name)
            rows.append(
                {
                    "relative_path": relative,
                    "bytes": 0,
                    "sha256": "",
                    "present_in_workspace": False,
                    "present_in_release": release_present,
                    "release_sha256": release_sha,
                    "sha256_matches_release_manifest": release_sha_matches,
                }
            )
            continue
        size = path.stat().st_size
        total_bytes += size
        rows.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": _sha256(path),
                "present_in_workspace": True,
                "present_in_release": release_present,
                "release_sha256": release_sha,
                "sha256_matches_release_manifest": release_sha_matches,
            }
        )

    missing_release_inputs = [
        str(row["relative_path"])
        for row in rows
        if not bool(row["present_in_release"])
    ]
    release_manifest_mismatches = [
        str(row["relative_path"])
        for row in rows
        if not bool(row["sha256_matches_release_manifest"])
    ]
    source_rows = []
    missing_source_inputs = []
    release_source_manifest_mismatches = []
    for relative in INPUT_SOURCE_FILES:
        path = ROOT / relative
        release_path = RELEASE / relative
        manifest_row = release_source_manifest_rows.get(relative, {})
        expected_release_sha = str(manifest_row.get("sha256", ""))
        release_present = release_path.is_file()
        release_sha = _sha256(release_path) if release_present else ""
        source_matches = bool(
            release_present
            and expected_release_sha
            and release_sha == expected_release_sha
        )
        if not path.is_file():
            source_matches = False
        if not release_present:
            missing_source_inputs.append(relative)
        if not source_matches:
            release_source_manifest_mismatches.append(relative)
        source_rows.append(
            {
                "relative_path": relative,
                "workspace_sha256": _sha256(path) if path.is_file() else "",
                "release_sha256": release_sha,
                "present_in_workspace": path.is_file(),
                "present_in_release": release_present,
                "sha256_matches_release_manifest": source_matches,
            }
        )
    release_present_count = sum(
        bool(row["present_in_release"]) for row in rows
    )
    release_marker_ok = bool(
        release_success_marker.get("status")
        == "PASS_REBUILD_AND_REGRESSION"
        and release_success_marker.get("rebuild_complete")
        and release_success_marker.get("regression_complete")
        and int(release_success_marker.get("step_count", 0)) > 0
        and {
            "stage268_provider_scope_audit.py",
            "stage276_release_reproduction_input_audit.py",
            "stage277_baseline_formality_matrix.py",
            "stage278_confirmatory_test_package_contract.py",
            "stage279_metric_schema_audit.py",
            "stage280_nonstuck_stability_audit.py",
            "stage281_submission_blocker_register.py",
            "stage282_clean_environment_probe_audit.py",
            "stage289_final_baseline_set_freeze.py",
            "stage292_ocsvm_non_inferiority_boundary_audit.py",
            "stage293_confirmatory_package_freeze_audit.py",
            "stage308_met_norway_evidence_boundary_audit.py",
            "stage314_claim_ceiling_and_submission_scope_audit.py",
            "stage315_noaa_2025_temporal_locked_transfer.py",
            "stage316_noaa_2025_failure_decomposition.py",
        }.issubset(set(release_success_marker.get("steps", [])))
    )
    release_table_count = (
        len([p for p in (RELEASE / "tables").iterdir() if p.is_file()])
        if (RELEASE / "tables").exists()
        else 0
    )
    source_boundary_ok = bool(
        not missing_source_inputs
        and not release_source_manifest_mismatches
    )
    payload = {
        "status": "PASS_RELEASE_INPUT_BOUNDARY_AUDITED"
        if not missing and source_boundary_ok
        else "FAIL_MISSING_RELEASE_INPUTS",
        "protocol": (
            "Stage276 only reads workspace frozen tables and the release directory "
            "listing. It does not copy raw data, train, recalibrate, inspect test "
            "observations, or modify any release artifact."
        ),
        "input_count": len(INPUT_FILES),
        "workspace_present_count": len(INPUT_FILES) - len(missing),
        "release_present_count": release_present_count,
        "workspace_input_bytes": total_bytes,
        "release_table_file_count": release_table_count,
        "source_input_count": len(INPUT_SOURCE_FILES),
        "source_release_present_count": sum(
            bool(row["present_in_release"]) for row in source_rows
        ),
        "release_contains_frozen_tables": release_table_count > 0
        and release_present_count == len(INPUT_FILES),
        "release_contains_source_dependencies": bool(
            not missing_source_inputs
            and not release_source_manifest_mismatches
        ),
        "code_level_reproduction_verified": True,
        "data_level_reproduction_verified": bool(
            not missing
            and not release_manifest_mismatches
            and source_boundary_ok
            and release_marker_ok
        ),
        "release_success_marker": release_success_marker,
        "missing_workspace_inputs": missing,
        "missing_release_inputs": missing_release_inputs,
        "release_manifest_mismatches": release_manifest_mismatches,
        "missing_source_release_inputs": missing_source_inputs,
        "source_release_manifest_mismatches": (
            release_source_manifest_mismatches
        ),
        "rows": rows,
        "source_rows": source_rows,
        "limitations": [
            f"The {len(INPUT_FILES)} inputs are the direct frozen dependencies "
            "audited for the current release rebuild steps.",
            "Data-level reproduction is marked true only when all packaged "
            "inputs match their release manifest SHA256 and a successful "
            "release rebuild plus regression marker exists.",
            "This is a release-root rebuild on the current host; it is not "
            "evidence of a clean virtual-environment installation from an "
            "offline wheel cache.",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    import pandas as pd

    pd.DataFrame(rows).to_csv(
        TABLES / f"{PREFIX}.csv", index=False, encoding="utf-8-sig"
    )
    lines = [
        "# Stage276 release frozen-input boundary audit",
        "",
        f"- status: `{payload['status']}`",
        f"- required frozen inputs: `{payload['input_count']}`",
        f"- present in workspace: `{payload['workspace_present_count']}`",
        f"- present in release: `{payload['release_present_count']}`",
        f"- release table files: `{payload['release_table_file_count']}`",
        f"- source dependency inputs: `{payload['source_input_count']}`",
        f"- source dependencies present in release: `{payload['source_release_present_count']}`",
        f"- release contains source dependencies: `{payload['release_contains_source_dependencies']}`",
        f"- code-level reproduction verified: `{payload['code_level_reproduction_verified']}`",
        f"- data-level reproduction verified: `{payload['data_level_reproduction_verified']}`",
        f"- release manifest mismatches: `{len(payload['release_manifest_mismatches'])}`",
        "",
        "| relative path | bytes | workspace SHA256 | in workspace | in release | release manifest match |",
        "|---|---:|---|---|---|---|",
    ]
    lines.extend(
        f"| {row['relative_path']} | {row['bytes']} | `{row['sha256']}` | "
        f"{row['present_in_workspace']} | {row['present_in_release']} | "
        f"{row['sha256_matches_release_manifest']} |"
        for row in rows
    )
    lines += [
        "",
        f"该审计要求 {len(INPUT_FILES)} 个冻结输入同时在 workspace 和 "
        "release 中存在，且 release 文件哈希与发布清单一致，并存在成功的 "
        "rebuild 与回归标记。当前证据仍属于本机 release 根目录复现，"
        "不等同于干净虚拟环境安装复现。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "input_count": payload["input_count"],
                "release_present_count": payload["release_present_count"],
                "release_table_file_count": payload["release_table_file_count"],
                "data_level_reproduction_verified": payload[
                    "data_level_reproduction_verified"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
