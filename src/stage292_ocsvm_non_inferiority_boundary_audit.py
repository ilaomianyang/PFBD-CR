from __future__ import annotations

"""Stage 292: audit the One-Class SVM non-inferiority claim boundary.

This is a read-only audit. It does not train, tune, inspect labels, or
create a new locked test. It separates the existing exploratory
One-Class SVM evidence from the requirements needed for a confirmatory
non-inferiority claim.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage292_ocsvm_non_inferiority_boundary_audit"
PROTOCOL_ID = "TMQC-OCSVM-NON-INFERIORITY-BOUNDARY-20260921-v1.0"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


def _load_json(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stage248 = _load_json("stage248_cross_provider_fair_baseline_audit")
    stage274 = _load_json("stage274_ocsvm_full_scale_sensitivity")
    stage278 = _load_json("stage278_confirmatory_test_package_contract")
    stage291 = _load_json("stage291_unified_win_loss_matrix")
    stage295 = _load_json(
        "stage295_japan_isd_locked_evaluation_contract"
    )
    stage297 = _load_json("stage297_japan_isd_locked_transfer")
    stage302 = _load_json("stage302_japan_ocsvm_budget_grid")

    comparisons = _load_csv(
        "stage248_cross_provider_fair_baseline_audit_comparisons.csv"
    )
    provider_equal = _load_csv(
        "stage248_cross_provider_fair_baseline_audit_provider_equal.csv"
    )
    stage274_provider = _load_csv(
        "stage274_ocsvm_full_scale_sensitivity_provider_equal.csv"
    )
    stage291_ledger = _load_csv(
        "stage291_unified_win_loss_matrix_ledger.csv"
    )
    japan_comparisons = _load_csv(
        "stage297_japan_isd_locked_transfer_comparisons.csv"
    )
    japan_evaluation = _load_csv(
        "stage297_japan_isd_locked_transfer_evaluation.csv"
    )
    stage302_comparisons = _load_csv(
        "stage302_japan_ocsvm_budget_grid_comparisons.csv"
    )

    ocsvm_rows = comparisons.loc[
        comparisons["baseline"].astype(str).eq("one_class_svm")
    ].copy()
    if ocsvm_rows.empty:
        raise RuntimeError("Stage248 has no One-Class SVM comparisons")
    required_comparison_columns = {
        "candidate",
        "baseline",
        "f1_difference",
        "f1_ci95_low",
        "event_coverage_difference",
        "event_ci95_low",
        "f1_bootstrap_bh_q",
        "event_coverage_bootstrap_bh_q",
    }
    missing = sorted(
        required_comparison_columns - set(ocsvm_rows.columns)
    )
    if missing:
        raise RuntimeError(
            f"Stage248 One-Class SVM comparison columns missing: {missing}"
        )

    stage248_ocsvm = provider_equal.loc[
        provider_equal["method"].astype(str).eq("one_class_svm")
    ]
    trust_rows = provider_equal.loc[
        provider_equal["method"].astype(str).isin(
            ["trust_valid_f1", "trust_nonstuck_event_macro"]
        )
    ].copy()
    if stage248_ocsvm.empty or trust_rows.empty:
        raise RuntimeError("Stage248 provider-equal rows are incomplete")

    if len(stage274_provider) != 1:
        raise RuntimeError(
            "Stage274 provider-equal table must contain one One-Class SVM row"
        )
    stage274_ocsvm = stage274_provider.copy()
    stage274_ocsvm.insert(0, "method", "one_class_svm")

    stage291_ocsvm = stage291_ledger.loc[
        stage291_ledger["baseline"].astype(str).eq("one_class_svm")
    ].copy()
    if stage291_ocsvm.empty:
        raise RuntimeError("Stage291 One-Class SVM ledger rows missing")

    japan_margin = stage295.get("non_inferiority_margin")
    pre_registered_margin_available = japan_margin is not None
    pre_registered_margin = (
        float(japan_margin) if pre_registered_margin_available else None
    )
    pre_registered_margin_source = (
        "stage295_japan_isd_locked_evaluation_contract"
        if pre_registered_margin_available
        else "NOT_FOUND_IN_FROZEN_ARTIFACTS"
    )
    japan_is_locked_transfer = bool(
        stage297.get("status") == "COMPLETED_JAPAN_ISD_LOCKED_TRANSFER"
        and stage297.get("contract_id") == stage295.get("contract_id")
    )
    new_locked_test_available = bool(
        stage278.get("is_a_new_confirmatory_test_package")
        or japan_is_locked_transfer
    )
    same_contract_new_locked_test = bool(
        stage291.get("same_contract_win_loss_matrix_is_new_locked_test")
        or japan_is_locked_transfer
    )

    required_budgets = [
        int(value)
        for value in stage295.get("method_contract", {})
        .get("one_class_svm", {})
        .get("reported_fit_budgets", [])
    ]
    japan_ocsvm_evaluation = japan_evaluation.loc[
        japan_evaluation["method"].astype(str).eq("one_class_svm")
    ].copy()
    if stage302.get("status") != (
        "PASS_JAPAN_OCSVM_BUDGET_GRID_COMPLETE"
    ):
        raise RuntimeError("Stage302 budget grid is not complete")
    japan_ocsvm_fit_budgets = sorted(
        int(value) for value in stage302.get("observed_budgets", [])
    )
    if stage302.get("required_budgets") != required_budgets:
        raise RuntimeError("Stage302 required budgets changed")
    required_budgets_covered = bool(
        required_budgets
        and set(required_budgets).issubset(set(japan_ocsvm_fit_budgets))
    )
    japan_ocsvm_rows = stage302_comparisons.loc[
        stage302_comparisons["baseline"].astype(str).str.startswith(
            "one_class_svm_"
        )
        & stage302_comparisons["candidate"].astype(str).isin(
            ["trust_valid_f1", "trust_nonstuck_event_macro"]
        )
    ].copy()
    japan_ocsvm_rows["f1_non_inferior_bool"] = (
        japan_ocsvm_rows["f1_non_inferior"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    japan_ocsvm_rows["event_non_inferior_bool"] = (
        japan_ocsvm_rows["event_coverage_non_inferior"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    japan_ocsvm_rows["both_endpoints_non_inferior"] = (
        japan_ocsvm_rows["f1_non_inferior_bool"]
        & japan_ocsvm_rows["event_non_inferior_bool"]
    )
    japan_ocsvm_f1_pass = bool(
        not japan_ocsvm_rows.empty
        and japan_ocsvm_rows["f1_non_inferior_bool"].any()
    )
    japan_ocsvm_event_pass = bool(
        not japan_ocsvm_rows.empty
        and japan_ocsvm_rows["event_non_inferior_bool"].any()
    )
    japan_ocsvm_any_candidate_pass = bool(
        not japan_ocsvm_rows.empty
        and japan_ocsvm_rows["both_endpoints_non_inferior"].any()
    )
    candidate_budget_summary: list[dict[str, object]] = []
    for candidate in (
        "trust_valid_f1",
        "trust_nonstuck_event_macro",
    ):
        rows = japan_ocsvm_rows.loc[
            japan_ocsvm_rows["candidate"].astype(str).eq(candidate)
        ]
        candidate_budget_summary.append(
            {
                "candidate": candidate,
                "budget_count": int(rows["requested_rows"].nunique()),
                "all_required_budgets_executed": bool(
                    set(required_budgets)
                    == {
                        int(value)
                        for value in rows["requested_rows"]
                        .dropna()
                        .astype(int)
                        .tolist()
                    }
                ),
                "f1_non_inferior_all_budgets": bool(
                    not rows.empty
                    and rows["f1_non_inferior_bool"].all()
                ),
                "event_coverage_non_inferior_all_budgets": bool(
                    not rows.empty
                    and rows["event_non_inferior_bool"].all()
                ),
                "both_endpoints_non_inferior_all_budgets": bool(
                    not rows.empty
                    and rows["both_endpoints_non_inferior"].all()
                ),
                "f1_failed_budgets": sorted(
                    int(value)
                    for value in rows.loc[
                        ~rows["f1_non_inferior_bool"],
                        "requested_rows",
                    ]
                    .dropna()
                    .astype(int)
                    .tolist()
                ),
                "event_coverage_failed_budgets": sorted(
                    int(value)
                    for value in rows.loc[
                        ~rows["event_non_inferior_bool"],
                        "requested_rows",
                    ]
                    .dropna()
                    .astype(int)
                    .tolist()
                ),
            }
        )
    required_candidate_count = 2
    fully_passing_candidates = [
        str(row["candidate"])
        for row in candidate_budget_summary
        if bool(row["both_endpoints_non_inferior_all_budgets"])
    ]
    japan_ocsvm_primary_budget_pass = bool(
        required_budgets_covered
        and len(fully_passing_candidates) == required_candidate_count
    )
    non_inferiority_pass_available = bool(
        japan_ocsvm_primary_budget_pass
    )
    non_inferiority_status = (
        "PASSED_ALL_REQUIRED_BUDGETS_ALL_REQUIRED_CANDIDATES"
        if non_inferiority_pass_available
        else (
            "PARTIAL_REQUIRED_BUDGETS_COMPLETE_REQUIRED_CANDIDATE_FAILURE"
            if required_budgets_covered
            else "NOT_PASSED_REQUIRED_BUDGETS_INCOMPLETE"
        )
    )
    claim_ceiling = (
        "One-Class SVM remains a complementary baseline; "
        "no confirmatory non-inferiority claim is supported."
        if not non_inferiority_pass_available
        else (
            "Pre-registered non-inferiority is supported on the frozen "
            "Japan locked transfer for every required candidate and "
            "One-Class SVM budget."
        )
    )

    checks = [
        {
            "check_id": "pre_registered_margin_present",
            "passed": pre_registered_margin_available,
            "detail": (
                "Frozen Japan evaluation contract contains a "
                f"pre-registered margin of {pre_registered_margin}."
                if pre_registered_margin_available
                else "No frozen pre-registered One-Class SVM "
                "non-inferiority margin was found in the audited artifacts."
            ),
        },
        {
            "check_id": "new_locked_test_present",
            "passed": new_locked_test_available,
            "detail": (
                "A frozen Japan locked-transfer evaluation is available."
                if japan_is_locked_transfer
                else (
                    f"Stage278 is_a_new_confirmatory_test_package="
                    f"{new_locked_test_available}."
                )
            ),
        },
        {
            "check_id": "same_contract_new_locked_test_rerun",
            "passed": same_contract_new_locked_test,
            "detail": (
                "Stage297 evaluated One-Class SVM and both Trust candidates "
                "on the same frozen Japan locked test."
                if japan_is_locked_transfer
                else (
                    "Stage291 same_contract_win_loss_matrix_is_new_locked_test="
                    f"{same_contract_new_locked_test}."
                )
            ),
        },
        {
            "check_id": "required_ocsvm_fit_budgets_covered",
            "passed": required_budgets_covered,
            "detail": (
                f"Required budgets={required_budgets}; observed budgets="
                f"{japan_ocsvm_fit_budgets}."
            ),
        },
        {
            "check_id": "non_inferiority_primary_budget_pass",
            "passed": japan_ocsvm_primary_budget_pass,
            "detail": (
                "All required Trust candidates passed F1 and event "
                "coverage non-inferiority at every required One-Class SVM "
                "budget."
                if japan_ocsvm_primary_budget_pass
                else "At least one required Trust candidate did not pass "
                "both pre-registered endpoints at every required budget."
            ),
        },
        {
            "check_id": "non_inferiority_pass_available",
            "passed": non_inferiority_pass_available,
            "detail": (
                "All pre-registered One-Class SVM budgets and both "
                "required Trust candidates passed."
                if non_inferiority_pass_available
                else "The required budget grid is complete, but not every "
                "required Trust candidate passed both endpoints at every "
                "budget."
                if required_budgets_covered
                else "The required One-Class SVM budget grid is incomplete."
            ),
        },
    ]
    passed_check_count = int(sum(row["passed"] for row in checks))
    if not japan_is_locked_transfer:
        raise RuntimeError("frozen Japan locked transfer is missing")

    payload = {
        "status": "PASS_OCSVM_NON_INFERIORITY_BOUNDARY_AUDITED",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "Stage292 only reads frozen and exploratory One-Class SVM "
            "artifacts. It does not train, calibrate, inspect labels, alter "
            "predictions, or promote exploratory evidence into a "
            "confirmatory non-inferiority result."
        ),
        "baseline": "one_class_svm",
        "pre_registered_margin_available": pre_registered_margin_available,
        "pre_registered_margin": pre_registered_margin,
        "pre_registered_margin_source": pre_registered_margin_source,
        "new_locked_test_available": new_locked_test_available,
        "same_contract_new_locked_test_rerun": (
            same_contract_new_locked_test
        ),
        "required_ocsvm_fit_budgets": required_budgets,
        "observed_ocsvm_fit_budgets": japan_ocsvm_fit_budgets,
        "required_ocsvm_fit_budgets_covered": required_budgets_covered,
        "japan_primary_budget_f1_pass": japan_ocsvm_f1_pass,
        "japan_primary_budget_event_coverage_non_inferior": (
            japan_ocsvm_event_pass
        ),
        "japan_primary_budget_candidate_pass": (
            japan_ocsvm_any_candidate_pass
        ),
        "japan_primary_budget_candidate_results": (
            japan_ocsvm_rows[
                [
                    "candidate",
                    "baseline",
                    "requested_rows",
                    "f1_difference",
                    "f1_ci95_low",
                    "f1_non_inferior_bool",
                    "event_coverage_difference",
                    "event_ci95_low",
                    "event_non_inferior_bool",
                    "both_endpoints_non_inferior",
                ]
            ].to_dict("records")
        ),
        "candidate_budget_summary": candidate_budget_summary,
        "fully_passing_candidates": fully_passing_candidates,
        "required_candidate_count": required_candidate_count,
        "japan_primary_budget_pass": japan_ocsvm_primary_budget_pass,
        "non_inferiority_pass_available": non_inferiority_pass_available,
        "non_inferiority_status": non_inferiority_status,
        "passed_check_count": passed_check_count,
        "check_count": len(checks),
        "checks": checks,
        "claim_ceiling": claim_ceiling,
        "frozen_provider_equal": {
            "stage248_one_class_svm": stage248_ocsvm.iloc[0].to_dict(),
            "stage248_trust_candidates": trust_rows.to_dict("records"),
            "stage274_one_class_svm": stage274_ocsvm.iloc[0].to_dict(),
        },
        "existing_comparison_rows": ocsvm_rows.to_dict("records"),
        "unified_ledger_rows": stage291_ocsvm.to_dict("records"),
        "source_status": {
            "stage248": stage248.get("status"),
            "stage274": stage274.get("status"),
            "stage278": stage278.get("status"),
            "stage291": stage291.get("status"),
            "stage295_japan_isd_locked_evaluation_contract": stage295.get(
                "status"
            ),
            "stage297_japan_isd_locked_transfer": stage297.get("status"),
            "stage302_japan_ocsvm_budget_grid": stage302.get("status"),
        },
        "input_sha256": {
            "stage248_cross_provider_fair_baseline_audit": sha256_file(
                TABLES / "stage248_cross_provider_fair_baseline_audit.json"
            ),
            "stage248_comparisons": sha256_file(
                TABLES
                / "stage248_cross_provider_fair_baseline_audit_comparisons.csv"
            ),
            "stage248_provider_equal": sha256_file(
                TABLES
                / "stage248_cross_provider_fair_baseline_audit_provider_equal.csv"
            ),
            "stage274_ocsvm_full_scale_sensitivity": sha256_file(
                TABLES / "stage274_ocsvm_full_scale_sensitivity.json"
            ),
            "stage274_provider_equal": sha256_file(
                TABLES
                / "stage274_ocsvm_full_scale_sensitivity_provider_equal.csv"
            ),
            "stage278_confirmatory_test_package_contract": sha256_file(
                TABLES / "stage278_confirmatory_test_package_contract.json"
            ),
            "stage291_unified_win_loss_matrix": sha256_file(
                TABLES / "stage291_unified_win_loss_matrix.json"
            ),
            "stage291_ledger": sha256_file(
                TABLES / "stage291_unified_win_loss_matrix_ledger.csv"
            ),
            "stage295_japan_isd_locked_evaluation_contract": sha256_file(
                TABLES
                / "stage295_japan_isd_locked_evaluation_contract.json"
            ),
            "stage297_japan_isd_locked_transfer": sha256_file(
                TABLES / "stage297_japan_isd_locked_transfer.json"
            ),
            "stage297_japan_isd_locked_transfer_comparisons": sha256_file(
                TABLES
                / "stage297_japan_isd_locked_transfer_comparisons.csv"
            ),
            "stage297_japan_isd_locked_transfer_evaluation": sha256_file(
                TABLES
                / "stage297_japan_isd_locked_transfer_evaluation.csv"
            ),
            "stage302_japan_ocsvm_budget_grid": sha256_file(
                TABLES / "stage302_japan_ocsvm_budget_grid.json"
            ),
            "stage302_japan_ocsvm_budget_grid_comparisons": sha256_file(
                TABLES
                / "stage302_japan_ocsvm_budget_grid_comparisons.csv"
            ),
        },
        "limitations": [
            "Stage297 used the 20k One-Class SVM budget despite its "
            "lightweight_fit_info.fit_rows=100000 field; Stage302 is the "
            "authoritative execution of the 20k/50k/100k budget grid.",
            "trust_valid_f1 passed both endpoints at all three budgets, "
            "but trust_nonstuck_event_macro failed F1 non-inferiority at "
            "the 50k and 100k budgets.",
            "The contract requires both primary Trust candidates and all "
            "three budgets to pass; P0-2 therefore remains open and "
            "One-Class SVM remains a complementary baseline.",
            "Japan is a new NOAA station network under the same provider, "
            "so the result does not establish provider or station "
            "independence.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(checks).to_csv(
        TABLES / f"{PREFIX}_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ocsvm_rows.to_csv(
        TABLES / f"{PREFIX}_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = [
        "# Stage 292 One-Class SVM 非劣效边界审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 预注册非劣效界值可用：`{pre_registered_margin_available}`",
        f"- 新锁定 test 可用：`{new_locked_test_available}`",
        f"- 同一新锁定 test 重跑：`{same_contract_new_locked_test}`",
        f"- 要求的 One-Class SVM 训练档位：`{required_budgets}`",
        f"- 实际观察到的训练档位：`{japan_ocsvm_fit_budgets}`",
        f"- 训练档位完整：`{required_budgets_covered}`",
        f"- 两个候选在所有档位均双终点通过："
        f"`{japan_ocsvm_primary_budget_pass}`",
        f"- 完全通过的候选：`{fully_passing_candidates}`",
        f"- 非劣效通过：`{non_inferiority_pass_available}`",
        f"- 结论：`{non_inferiority_status}`",
        "",
        "## 检查",
        "",
        "| 检查 | 通过 | 说明 |",
        "|---|---|---|",
    ]
    for row in checks:
        report.append(
            f"| {row['check_id']} | {row['passed']} | {row['detail']} |"
        )
    report += [
        "",
        "## 现有证据",
        "",
        "| 来源 | 候选 | F1 | 事件覆盖 | 行召回 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in trust_rows.to_dict("records"):
        report.append(
            f"| Stage248 provider-equal | {row['method']} | "
            f"{float(row['f1_mean']):.6f} | "
            f"{float(row['event_coverage_mean']):.6f} | "
            f"{float(row['row_recall_mean']):.6f} |"
        )
    one = stage248_ocsvm.iloc[0]
    report.append(
        f"| Stage248 provider-equal | one_class_svm | "
        f"{float(one['f1_mean']):.6f} | "
        f"{float(one['event_coverage_mean']):.6f} | "
        f"{float(one['row_recall_mean']):.6f} |"
    )
    full = stage274_ocsvm.iloc[0]
    report.append(
        f"| Stage274 provider-equal | one_class_svm | "
        f"{float(full['f1_mean']):.6f} | "
        f"{float(full['event_coverage_mean']):.6f} | "
        f"{float(full['row_recall_mean']):.6f} |"
    )
    report += [
        "",
        "## 边界",
        "",
        "- 本阶段只读现有结果，不训练、不调参、不接触新的 test。",
        "- Stage302 已在同一冻结日本 test 上完成 20k/50k/100k 档位。",
        "- `trust_valid_f1` 三档双终点均通过。",
        "- `trust_nonstuck_event_macro` 在 50k/100k 的 F1 非劣未通过，"
        "事件覆盖通过。",
        "- 预注册契约要求两个 Trust 候选和三个训练档位均通过，因此"
        "P0-2 仍未通过。",
        "- One-Class SVM 继续作为互补基线保留。",
        "- 不能据此升级为全面领先或机构无关结论。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": payload["status"],
                "pre_registered_margin_available": (
                    pre_registered_margin_available
                ),
                "new_locked_test_available": new_locked_test_available,
                "required_ocsvm_fit_budgets": required_budgets,
                "observed_ocsvm_fit_budgets": japan_ocsvm_fit_budgets,
                "required_ocsvm_fit_budgets_covered": (
                    required_budgets_covered
                ),
                "japan_primary_budget_pass": (
                    japan_ocsvm_primary_budget_pass
                ),
                "non_inferiority_status": non_inferiority_status,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
