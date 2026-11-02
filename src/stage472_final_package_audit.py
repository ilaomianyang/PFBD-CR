"""Final package audit for the TrustMeteo-QC submission.

The audit is intentionally textual and structural. It does not rerun any
experiment and does not upgrade exploratory evidence. It verifies that the
authoritative package contains the required tables, claim boundaries, and
manuscript statements.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
MANUSCRIPT = FINAL / "manuscript_final_20261009.md"
SUPPLEMENT = FINAL / "supplementary_final_20261009.md"
SUBMISSION = FINAL / "submission_materials_final_20261009.md"
MANIFEST = FINAL / "submission_manifest.json"
OUT_JSON = FINAL / "reports" / "final_package_audit.json"
OUT_MD = FINAL / "reports" / "final_package_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    status: bool,
    severity: str,
    message: str,
    evidence: Any = None,
) -> None:
    checks.append(
        {
            "id": check_id,
            "status": "PASS" if status else "FAIL",
            "severity": severity,
            "message": message,
            "evidence": evidence,
        }
    )


def main() -> int:
    checks: list[dict[str, Any]] = []
    manuscript = load_text(MANUSCRIPT)
    supplement = load_text(SUPPLEMENT)
    submission = load_text(SUBMISSION)
    manuscript_norm = re.sub(r"-\s+", "-", re.sub(r"\s+", " ", manuscript.lower()))
    supplement_norm = re.sub(r"-\s+", "-", re.sub(r"\s+", " ", supplement.lower()))

    add_check(
        checks,
        "authoritative_manuscript_exists",
        MANUSCRIPT.is_file(),
        "P0",
        "The final folder has one authoritative manuscript.",
    )
    add_check(
        checks,
        "authoritative_supplement_exists",
        SUPPLEMENT.is_file(),
        "P0",
        "The final folder has one authoritative supplement.",
    )
    add_check(
        checks,
        "submission_materials_exist",
        SUBMISSION.is_file(),
        "P0",
        "The final folder has submission materials.",
    )
    add_check(
        checks,
        "manifest_exists",
        MANIFEST.is_file(),
        "P0",
        "The final folder has a manifest.",
    )
    add_check(
        checks,
        "primary_candidate_named",
        "trust_shape_union" in manuscript_norm,
        "P0",
        "The manuscript names the V6 primary candidate.",
    )
    add_check(
        checks,
        "global_win_share_defined",
        "73/99" in manuscript_norm and "73.7%" in manuscript_norm,
        "P0",
        "The manuscript defines the global 73/99 scenario-by-baseline result.",
    )
    add_check(
        checks,
        "scenario_failures_retained",
        "coordinated_offset" in manuscript_norm and "slow_drift" in manuscript_norm,
        "P0",
        "The manuscript retains the two scenario-level failure boundaries.",
    )
    add_check(
        checks,
        "far_terms_separated",
        all(
            token.lower() in manuscript_norm
            for token in [
                "FAR = 0.0065",
                "0.008470",
                "0.0106008",
            ]
        ),
        "P0",
        "The manuscript separates the V6 budget reference, measured FAR, and EWMA constraint deviation.",
    )
    add_check(
        checks,
        "provider_scope_separated",
        "eccc, knmi, meteoswiss, and noaa isd" in manuscript_norm
        and "dwd" in manuscript_norm
        and "historical" in manuscript_norm,
        "P0",
        "The four-provider V6 scope and historical DWD boundary are separated.",
    )
    add_check(
        checks,
        "ocsvm_budget_reconciled",
        "20k" in manuscript_norm
        and "50k" in manuscript_norm
        and "100k" in manuscript_norm
        and "sensitivity" in manuscript_norm,
        "P0",
        "One-Class SVM confirmatory and sensitivity budgets are reconciled.",
    )
    add_check(
        checks,
        "external_negative_results_retained",
        "fmi" in manuscript_norm
        and "smhi" in manuscript_norm
        and ("not the strongest" in manuscript_norm or "negative" in manuscript_norm),
        "P0",
        "External FMI/SMHI negative or mixed evidence is retained.",
    )
    add_check(
        checks,
        "online_offline_boundary_retained",
        "online" in manuscript_norm
        and "offline" in manuscript_norm
        and ("not equivalent" in manuscript_norm or "weaker" in manuscript_norm),
        "P0",
        "Online causal recovery and offline interpolation are separated.",
    )
    add_check(
        checks,
        "simulation_boundary_retained",
        "simulated" in manuscript_norm
        and "not production" in manuscript_norm
        and "real-production attack recall" in manuscript_norm,
        "P0",
        "The manuscript keeps the simulated-attack and production-recall boundary.",
    )
    add_check(
        checks,
        "implementation_boundaries_present",
        "transparent proxy" in supplement_norm
        and "not an end-to-end trained gru" in supplement_norm
        and "not a bitwise reproduction" in supplement_norm,
        "P1",
        "The supplement labels proxy and formal implementations.",
    )
    add_check(
        checks,
        "scenario_metrics_present",
        "final_v6_scenario_metrics.csv" in supplement_norm
        and "precision" in supplement_norm
        and "average_precision" in supplement_norm,
        "P1",
        "The supplement points to full scenario-level metrics.",
    )
    add_check(
        checks,
        "aggregation_levels_present",
        "final_v6_aggregation_levels.csv" in supplement_norm
        and "provider_equal" in supplement_norm
        and "station_equal" in supplement_norm,
        "P1",
        "The supplement points to pooled, provider-equal, and station-equal summaries.",
    )
    add_check(
        checks,
        "event_lengths_present",
        "final_v6_event_lengths.csv" in supplement_norm,
        "P1",
        "The supplement points to event-length evidence.",
    )
    add_check(
        checks,
        "scenario_ci_present",
        "final_v6_scenario_ci.csv" in supplement_norm,
        "P1",
        "The supplement points to scenario-level 95% intervals.",
    )
    add_check(
        checks,
        "resource_evidence_present",
        "final_baseline_resources.csv" in supplement_norm
        and "final_trust_resources.csv" in supplement_norm
        and "final_testbed_resources.csv" in supplement_norm,
        "P1",
        "The supplement points to resource and testbed evidence.",
    )
    add_check(
        checks,
        "operating_point_sensitivity_present",
        "final_operating_points.csv" in supplement_norm
        and "historical" in supplement_norm,
        "P1",
        "The supplement reports operating-point sensitivity with a historical boundary.",
    )
    add_check(
        checks,
        "evidence_hierarchy_present",
        "final_evidence_hierarchy.csv" in supplement_norm
        and "exploratory" in supplement_norm
        and "historical" in supplement_norm,
        "P0",
        "The supplement separates primary, exploratory, historical, and external evidence.",
    )
    add_check(
        checks,
        "negative_boundary_table_present",
        "final_negative_boundaries.csv" in supplement_norm,
        "P0",
        "The supplement points to the negative-boundary table.",
    )
    figure_dir = FINAL / "figures"
    add_check(
        checks,
        "final_figures_present",
        (figure_dir / "figure1_v6_scenario_performance.pdf").is_file()
        and (figure_dir / "figure2_v6_aggregation_levels.pdf").is_file()
        and (figure_dir / "figure3_external_transfer_boundary.pdf").is_file()
        and (figure_dir / "figure4_online_offline_recovery.pdf").is_file(),
        "P1",
        "The final figures are present in editable and raster formats.",
    )
    add_check(
        checks,
        "no_universal_claim",
        "do not establish universal attack coverage" in manuscript_norm
        and "provider-independent superiority" in manuscript_norm
        and "station-independent performance" in manuscript_norm
        and "not as a universal replacement" in manuscript_norm,
        "P0",
        "The final manuscript does not claim universal, provider-independent, or station-independent superiority.",
    )

    findings = [item for item in checks if item["status"] == "FAIL"]
    payload = {
        "status": "PASS_FINAL_PACKAGE_AUDIT" if not findings else "FAIL_FINAL_PACKAGE_AUDIT",
        "package_root": str(FINAL.relative_to(ROOT)),
        "checks": checks,
        "finding_count": len(findings),
        "hashes": {
            "manuscript_sha256": sha256(MANUSCRIPT) if MANUSCRIPT.is_file() else None,
            "supplement_sha256": sha256(SUPPLEMENT) if SUPPLEMENT.is_file() else None,
            "submission_materials_sha256": sha256(SUBMISSION)
            if SUBMISSION.is_file()
            else None,
        },
        "rule": "Structural and claim-boundary audit only; no experimental metric is recomputed.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Final package audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Findings: `{payload['finding_count']}`",
        "",
        "| Check | Status | Severity | Message |",
        "|---|---|---|---|",
    ]
    for item in checks:
        lines.append(
            f"| `{item['id']}` | `{item['status']}` | `{item['severity']}` | {item['message']} |"
        )
    lines.extend(
        [
            "",
            "## Rule",
            "",
            payload["rule"],
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "findings": len(findings)}, ensure_ascii=False))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
