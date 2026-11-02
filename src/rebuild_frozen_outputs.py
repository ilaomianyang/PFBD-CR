from __future__ import annotations

"""Single entry point for rebuilding frozen post-analysis artifacts.

The command intentionally rebuilds only the read-only audits and figures that
can be deterministically reconstructed from frozen tables. It does not retrain
models, reselect thresholds, or touch any raw or test prediction. The existing
Stage246/248 confirmatory tables remain the frozen source of truth.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"

STEPS = [
    ("stage268_provider_scope_audit.py", "provider scope and event denominators"),
    ("stage269_deployment_complexity_audit.py", "deployment complexity and resource evidence"),
    ("stage270_attack_factor_unified_table.py", "unified attack-factor evidence table"),
    ("stage271_failure_case_figures.py", "failure, false-alarm, and miss figures"),
    ("stage272_final_parameter_table.py", "final parameter and method inventory"),
    ("stage273_release_anonymization_audit.py", "release anonymization audit"),
    (
        "stage275_external_preregistration_lineage_audit.py",
        "external preregistration lineage audit",
    ),
    (
        "stage276_release_reproduction_input_audit.py",
        "release reproduction input boundary audit",
    ),
    (
        "stage277_baseline_formality_matrix.py",
        "baseline formality and public-source gap audit",
    ),
    (
        "stage278_confirmatory_test_package_contract.py",
        "future independent confirmatory-test package contract",
    ),
    (
        "stage279_metric_schema_audit.py",
        "required metric and event-denominator schema audit",
    ),
    (
        "stage280_nonstuck_stability_audit.py",
        "non-stuck type and seed stability audit",
    ),
    (
        "stage289_final_baseline_set_freeze.py",
        "final baseline-set scope freeze",
    ),
    (
        "stage291_unified_win_loss_matrix.py",
        "unified existing baseline win-loss ledger",
    ),
    (
        "stage292_ocsvm_non_inferiority_boundary_audit.py",
        "One-Class SVM non-inferiority boundary audit",
    ),
    (
        "stage293_confirmatory_package_freeze_audit.py",
        "future confirmatory-package freeze readiness audit",
    ),
    (
        "stage299_method_resource_coverage_audit.py",
        "method-level resource-evidence coverage audit",
    ),
    (
        "stage300_method_resource_evidence_upgrade.py",
        "method resource evidence upgrade audit",
    ),
    (
        "stage301_trust_resource_budget.py",
        "isolated train-only Trust resource budget",
    ),
    (
        "stage309_public_ics_cps_source_feasibility.py",
        "public ICS/CPS source feasibility audit",
    ),
    (
        "stage308_met_norway_evidence_boundary_audit.py",
        "MET Norway evidence-boundary audit",
    ),
    (
        "stage312_met_norway_disjoint_locked_transfer.py",
        "disjoint MET Norway locked-transfer result audit",
    ),
    (
        "stage313_cross_locked_test_win_loss_matrix.py",
        "cross-locked-test win/loss and non-stuck stability audit",
    ),
    (
        "stage315_noaa_2025_temporal_locked_transfer.py",
        "NOAA 2025 temporal locked-transfer result audit",
    ),
    (
        "stage316_noaa_2025_failure_decomposition.py",
        "NOAA 2025 failure decomposition audit",
    ),
    (
        "stage282_clean_environment_probe_audit.py",
        "clean-install reproduction boundary probe",
    ),
    (
        "stage281_submission_blocker_register.py",
        "open submission blocker and claim-ceiling register",
    ),
    (
        "stage314_claim_ceiling_and_submission_scope_audit.py",
        "claim ceiling and submission scope audit",
    ),
]

FINAL_CLAIM_CHAIN = [
    "stage276_release_reproduction_input_audit.py",
    "stage282_clean_environment_probe_audit.py",
    "stage281_submission_blocker_register.py",
    "stage314_claim_ceiling_and_submission_scope_audit.py",
]


def _run(script: Path, extra_args: list[str] | None = None) -> int:
    print(f"RUN {script.relative_to(ROOT)}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", "-u", str(script)]
    command.extend(list(extra_args or []))
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{script.relative_to(ROOT)} failed with exit code "
            f"{completed.returncode}"
        )
    return completed.returncode


def _clean_generated_caches() -> int:
    targets = [
        path for path in ROOT.rglob("__pycache__") if path.is_dir()
    ]
    for path in targets:
        resolved = path.resolve()
        if ROOT.resolve() not in resolved.parents:
            raise RuntimeError(f"unsafe cache path: {resolved}")
    for path in targets:
        shutil.rmtree(path)
    return len(targets)


def _write_success_marker(regression_complete: bool) -> None:
    executed_steps = [item[0] for item in STEPS]
    marker = {
        "status": "PASS_REBUILD_AND_REGRESSION"
        if regression_complete
        else "PASS_REBUILD_WITHOUT_REGRESSION",
        "rebuild_complete": True,
        "regression_complete": regression_complete,
        "step_count": len(executed_steps),
        "steps": executed_steps,
        "final_claim_chain": list(FINAL_CLAIM_CHAIN),
    }
    marker_path = ROOT / "reports" / "rebuild_frozen_outputs_success.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_final_claim_chain() -> None:
    descriptions = {
        "stage276_release_reproduction_input_audit.py": (
            "release reproduction input boundary audit"
        ),
        "stage282_clean_environment_probe_audit.py": (
            "clean-environment reproduction probe audit"
        ),
        "stage281_submission_blocker_register.py": (
            "open submission blocker and claim-ceiling register"
        ),
        "stage314_claim_ceiling_and_submission_scope_audit.py": (
            "claim ceiling and submission scope audit"
        ),
    }
    for relative in FINAL_CLAIM_CHAIN:
        print(f"STEP {descriptions[relative]}", flush=True)
        _run(SRC / relative)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Rebuild artifacts without running the regression suite.",
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    removed_caches = _clean_generated_caches()
    if removed_caches:
        print(f"CLEAN generated __pycache__ dirs: {removed_caches}", flush=True)
    for relative, description in STEPS:
        if relative in {
            "stage276_release_reproduction_input_audit.py",
            "stage282_clean_environment_probe_audit.py",
            "stage281_submission_blocker_register.py",
            "stage314_claim_ceiling_and_submission_scope_audit.py",
        }:
            continue
        print(f"STEP {description}", flush=True)
        _run(SRC / relative)
    print("STEP release reproduction input boundary audit", flush=True)
    _run(SRC / "stage276_release_reproduction_input_audit.py")
    _run_final_claim_chain()
    release_tests = TESTS / "run_release_rebuild_tests.py"
    regression_script = (
        release_tests if release_tests.is_file() else TESTS / "run_tests.py"
    )
    _write_success_marker(False)
    if not args.skip_tests:
        print("STEP regression suite", flush=True)
        _run(
            regression_script,
            (
                ["--skip-reproduction-input-test"]
                if release_tests.is_file()
                else []
            ),
        )
    _write_success_marker(not args.skip_tests)
    _run_final_claim_chain()
    if not args.skip_tests:
        print("STEP final regression suite", flush=True)
        _run(regression_script)
    _write_success_marker(not args.skip_tests)
    print("REBUILD COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
