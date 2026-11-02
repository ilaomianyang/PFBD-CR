"""Stage 485: semantic checks for the post-review correction layer."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"
OUT_JSON = REPORTS / "post_review_semantic_audit.json"
OUT_MD = REPORTS / "post_review_semantic_audit.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    checks: list[dict[str, object]] = []
    required_tables = [
        "final_v6_aggregation_levels_corrected.csv",
        "final_v6_scenario_ci_corrected.csv",
        "final_v6_station_cluster_ci_corrected.csv",
        "final_v6_union_decision_provenance.csv",
        "final_method_implementation_boundary_corrected.csv",
        "final_baseline_resources_corrected.csv",
        "final_trust_resources_corrected.csv",
        "final_testbed_resources_corrected.csv",
    ]
    for name in required_tables:
        checks.append(
            {
                "check": f"table_exists:{name}",
                "status": (TABLES / name).is_file(),
            }
        )
    reports = [
        "final_v6_aggregation_correction.json",
        "final_v6_scenario_ci_correction.json",
        "final_v6_station_cluster_ci_correction.json",
        "final_v6_information_set_audit.json",
        "final_v6_union_decision_provenance.json",
        "final_resource_boundary_correction.json",
    ]
    for name in reports:
        path = REPORTS / name
        status = False
        if path.is_file():
            status = str(load_json(path).get("status", "")).startswith(
                "PASS_"
            )
        checks.append(
            {
                "check": f"report_status:{name}",
                "status": status,
            }
        )
    manuscript = (
        FINAL / "manuscript_final_20261009.md"
    ).read_text(encoding="utf-8")
    supplement = (
        FINAL / "supplementary_final_20261009.md"
    ).read_text(encoding="utf-8")
    cover = (
        FINAL / "submission_materials_final_20261009.md"
    ).read_text(encoding="utf-8")
    checks.extend(
        [
            {
                "check": "manuscript_batch_information_set",
                "status": (
                    "batch evaluation" in manuscript
                    and "centered Hampel" in manuscript
                ),
            },
            {
                "check": "manuscript_corrected_tables",
                "status": (
                    "final_v6_aggregation_levels_corrected.csv"
                    in manuscript
                    and "final_v6_scenario_ci_corrected.csv"
                    in manuscript
                ),
            },
            {
                "check": "supplement_test_count_current",
                "status": "290 passing tests" in supplement,
            },
            {
                "check": "cover_letter_no_unconfirmed_author_claim",
                "status": (
                    "All authors have approved" not in cover
                    and "does not assert those confirmations"
                    in cover
                ),
            },
            {
                "check": "old_rebuild_report_current_win_count",
                "status": (
                    "73/99" in (
                        REPORTS / "final_v6_rebuild.md"
                    ).read_text(encoding="utf-8")
                    and "authoritative frozen Stage353 rule uses"
                    in (
                        REPORTS / "final_v6_rebuild.md"
                    ).read_text(encoding="utf-8")
                ),
            },
        ]
    )
    stage486 = (
        REPORTS / "stage486_formal_lagged_mlp_baseline.json"
    )
    stage487 = REPORTS / "stage487_formal_baseline_merge.json"
    stage488 = REPORTS / "stage488_extended_formal_baseline_gate.json"
    checks.extend(
        [
            {
                "check": "stage486_formal_baseline_status",
                "status": (
                    stage486.is_file()
                    and load_json(stage486).get("status")
                    == "PASS_EXPLORATORY_FORMAL_LAGGED_MLP_BASELINE"
                ),
            },
            {
                "check": "stage487_formal_baseline_merge_status",
                "status": (
                    stage487.is_file()
                    and load_json(stage487).get("status")
                    == "PASS_EXPLORATORY_FORMAL_BASELINE_MERGE"
                ),
            },
            {
                "check": "stage488_extended_gate_retains_failure",
                "status": (
                    stage488.is_file()
                    and load_json(stage488).get("status")
                    == "EXPLORATORY_EXTENDED_FORMAL_BASELINE_GATE"
                    and load_json(stage488).get(
                        "extended_gate_passes_70pct"
                    )
                    is False
                ),
            },
            {
                "check": "stage489_validation_selected_hybrid_status",
                "status": (
                    (
                        REPORTS
                        / "stage489_validation_selected_hybrid.json"
                    ).is_file()
                    and load_json(
                        REPORTS
                        / "stage489_validation_selected_hybrid.json"
                    ).get("status")
                    == "PASS_EXPLORATORY_VALIDATION_SELECTED_HYBRID"
                ),
            },
            {
                "check": "stage490_score_fusion_status",
                "status": (
                    (
                        REPORTS / "stage490_score_fusion.json"
                    ).is_file()
                    and load_json(
                        REPORTS / "stage490_score_fusion.json"
                    ).get("status")
                    == "PASS_EXPLORATORY_VALIDATION_SCORE_FUSION"
                ),
            },
            {
                "check": "stage491_constrained_fusion_status",
                "status": (
                    (
                        REPORTS / "stage491_constrained_fusion.json"
                    ).is_file()
                    and load_json(
                        REPORTS / "stage491_constrained_fusion.json"
                    ).get("status")
                    == "PASS_EXPLORATORY_CONSTRAINED_FUSION"
                ),
            },
            {
                "check": "stage492_constraint_margin_sweep_status",
                "status": (
                    (
                        REPORTS
                        / "stage492_constraint_margin_sweep.json"
                    ).is_file()
                    and load_json(
                        REPORTS
                        / "stage492_constraint_margin_sweep.json"
                    ).get("status")
                    == "PASS_EXPLORATORY_CONSTRAINT_MARGIN_SWEEP"
                ),
            },
            {
                "check": "stage493_constrained_fusion_stability_status",
                "status": (
                    (
                        REPORTS
                        / "stage493_constrained_fusion_stability.json"
                    ).is_file()
                    and load_json(
                        REPORTS
                        / "stage493_constrained_fusion_stability.json"
                    ).get("status")
                    == "PASS_EXPLORATORY_CONSTRAINED_FUSION_STABILITY"
                ),
            },
            {
                "check": "stage494_provider_heterogeneity_status",
                "status": (
                    (
                        REPORTS
                        / "stage494_provider_heterogeneity.json"
                    ).is_file()
                    and load_json(
                        REPORTS
                        / "stage494_provider_heterogeneity.json"
                    ).get("status")
                    == "PASS_EXPLORATORY_PROVIDER_HETEROGENEITY_AUDIT"
                ),
            },
            {
                "check": "stage495_provider_adaptive_fusion_status",
                "status": (
                    (
                        REPORTS
                        / "stage495_provider_adaptive_fusion.json"
                    ).is_file()
                    and load_json(
                        REPORTS
                        / "stage495_provider_adaptive_fusion.json"
                    ).get("status")
                    == "PARTIAL_EXPLORATORY_PROVIDER_ADAPTIVE_FUSION_UNION_FALLBACK"
                ),
            },
            {
                "check": "stage496_far_safety_calibration_status",
                "status": (
                    (
                        REPORTS
                        / "stage496_far_safety_calibration.json"
                    ).is_file()
                    and load_json(
                        REPORTS
                        / "stage496_far_safety_calibration.json"
                    ).get("status")
                    == "PASS_EXPLORATORY_FAR_SAFETY_CALIBRATION_AUDIT"
                ),
            },
            {
                "check": "stage497_noaa_station_calibration_status",
                "status": (
                    (
                        REPORTS
                        / "stage497_noaa_station_calibration.json"
                    ).is_file()
                    and load_json(
                        REPORTS
                        / "stage497_noaa_station_calibration.json"
                    ).get("status")
                    == "PASS_EXPLORATORY_NOAA_STATION_CALIBRATION_FALLBACK"
                ),
            },
            {
                "check": "stage498_knmi_far_transfer_status",
                "status": (
                    (
                        REPORTS
                        / "stage498_knmi_far_transfer_correction.json"
                    ).is_file()
                    and load_json(
                        REPORTS
                        / "stage498_knmi_far_transfer_correction.json"
                    ).get("status")
                    == "PASS_EXPLORATORY_KNMI_FAR_TRANSFER_CORRECTION"
                ),
            },
            {
                "check": "stage499_noaa_representation_status",
                "status": (
                    (
                        REPORTS
                        / "stage499_noaa_representation_v2.json"
                    ).is_file()
                    and load_json(
                        REPORTS
                        / "stage499_noaa_representation_v2.json"
                    ).get("status")
                    == "PASS_EXPLORATORY_NOAA_REPRESENTATION_V2_IMPROVED"
                ),
            },
            {
                "check": "stage500_final_freeze_status",
                "status": (
                    (
                        REPORTS
                        / "stage500_final_experiment_freeze.json"
                    ).is_file()
                    and load_json(
                        REPORTS
                        / "stage500_final_experiment_freeze.json"
                    ).get("status")
                    == "EXPLORATORY_FINAL_CANDIDATE_FROZEN_WITH_BOUNDARIES"
                ),
            },
        ]
    )
    failed = [item for item in checks if not item["status"]]
    payload = {
        "status": (
            "PASS_POST_REVIEW_SEMANTIC_AUDIT"
            if not failed
            else "FAIL_POST_REVIEW_SEMANTIC_AUDIT"
        ),
        "check_count": len(checks),
        "failed_checks": failed,
        "checks": checks,
        "scope": (
            "Semantic/file-presence audit for the post-review correction "
            "layer; it does not replace the statistical reports."
        ),
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Post-review semantic audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Checks: `{payload['check_count']}`",
        f"- Failed: `{len(failed)}`",
        "",
    ]
    for item in checks:
        lines.append(
            f"- [{'x' if item['status'] else ' '}] `{item['check']}`"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "checks": payload["check_count"],
                "failed": len(failed),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
