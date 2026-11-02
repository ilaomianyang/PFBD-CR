from __future__ import annotations

"""Stage 287: audit whether the SWaT/USAD result can upgrade P0-3.

This stage is read-only. It does not train, tune, recalibrate, or change any
prediction. It checks the provenance, frozen protocol, input hashes, and
reported metrics from Stages 283-286, then records the exact evidence level
that the result may support.
"""

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage287_swat_usad_evidence_upgrade_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import sha256_file  # noqa: E402


def _load(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stage283 = _load("stage283_public_ics_cps_source_audit")
    stage284 = _load("stage284_swat_author_data_download")
    stage285 = _load("stage285_swat_usad_protocol_preregistration")
    stage286 = _load("stage286_swat_usad_author_evaluation")

    required_metrics = {
        "f1",
        "precision",
        "recall",
        "row_recall",
        "event_coverage",
        "false_alarm_rate",
        "auprc",
    }
    actual_metrics = set(stage286["test"])
    frozen_input_hash_map = {
        "raw/public_ics_cps/SWaT_Dataset_Normal_v1.csv": "normal",
        "raw/public_ics_cps/SWaT_Dataset_Attack_v0.csv": "attack",
        "vendor/usad.py": "usad_py",
        "vendor/utils.py": "utils_py",
    }
    protocol_input_hashes_match = all(
        stage286["input_sha256"].get(frozen_input_hash_map[name])
        == metadata["sha256"]
        for name, metadata in stage285["inputs"].items()
        if name in frozen_input_hash_map
    )
    checks = {
        "author_source_verified": bool(
            stage283["required_author_sources_verified"]
            and stage283["verified_artifact_count"] == 4
        ),
        "author_data_downloaded_and_hashed": bool(
            stage284["paper_faithful_reproduction_ready"]
            and all(
                row["status"] == "DOWNLOADED_AND_CSV_AUDITED"
                and int(row["bytes"]) > 0
                and len(str(row["sha256"])) == 64
                for row in stage284["rows"]
            )
        ),
        "protocol_frozen_before_training": bool(
            stage285["status"] == "FROZEN_PUBLIC_SWAT_USAD_PROTOCOL"
            and stage285["threshold"]["test_labels_used_for_threshold"]
            is False
        ),
        "author_training_recipe_used": bool(
            stage286["epochs"] == 100
            and stage286["window_length"] == 12
            and stage286["source_commit"] == stage285["source_commit"]
        ),
        "frozen_input_hash_verification_passed": bool(
            protocol_input_hashes_match
            and stage286["frozen_input_hash_verification"]
            == "all_protocol_inputs_matched"
        ),
        "threshold_calibrated_on_valid_normal_only": bool(
            abs(float(stage286["valid_false_alarm_rate"]) - 0.01)
            <= 0.001
        ),
        "required_event_and_row_metrics_reported": required_metrics.issubset(
            actual_metrics
        ),
        "same_data_contract_as_trustmeteo_qc": False,
    }
    checks_passed = sum(
        value for name, value in checks.items()
        if name != "same_data_contract_as_trustmeteo_qc"
    )
    p0_3_subitem_passed = bool(
        checks_passed == len(checks) - 1
        and not checks["same_data_contract_as_trustmeteo_qc"]
    )
    payload = {
        "status": (
            "PASS_PUBLIC_ICS_CPS_AUTHOR_BASELINE_EVALUATED_"
            "SAME_CONTRACT_COMPARISON_STILL_OPEN"
        ),
        "protocol": (
            "Read-only audit of the fixed-commit USAD source, author-linked "
            "SWaT data, Stage285 frozen protocol, and Stage286 evaluation. "
            "No training, tuning, threshold change, or test-label inspection "
            "occurs in this stage."
        ),
        "p0_3_before": "IN_PROGRESS",
        "p0_3_after": "IN_PROGRESS",
        "public_ics_cps_formal_implementation_subitem": (
            "PASSED"
            if p0_3_subitem_passed
            else "NOT_PASSED"
        ),
        "same_contract_comparison": "NOT_AVAILABLE",
        "paper_faithful_public_baseline_available": bool(
            p0_3_subitem_passed
        ),
        "checks": checks,
        "checks_passed": int(checks_passed),
        "checks_total_excluding_same_contract": len(checks) - 1,
        "usad_test": {
            key: stage286["test"][key]
            for key in sorted(required_metrics)
        },
        "favorable_result": False,
        "favorable_result_reason": (
            "USAD is retained as a completed author-code public ICS/CPS "
            "baseline, but this stage does not compare it to TrustMeteo-QC "
            "because the data, split, and endpoint contracts differ."
        ),
        "claim_ceiling": (
            "The result supports the claim that a paper-faithful USAD "
            "author-code evaluation on author-linked SWaT data was completed "
            "under a frozen no-test-threshold protocol. It does not support "
            "a claim that TrustMeteo-QC outperforms USAD on the same data, "
            "nor a production-attack or universal-superiority claim."
        ),
        "remaining_p0_3_requirements": [
            "Freeze the final baseline set before any further test access.",
            "Run every final baseline under one common data and endpoint contract.",
            "Report training budget, parameter count, runtime, memory, and failure state for every baseline.",
            "Place all wins and losses in the main or formal supplementary table.",
            "Evaluate TrustMeteo-QC against all major baselines under the pre-registered non-inferiority or superiority rule.",
        ],
        "input_sha256": {
            name: sha256_file(TABLES / f"{name}.json")
            for name in [
                "stage283_public_ics_cps_source_audit",
                "stage284_swat_author_data_download",
                "stage285_swat_usad_protocol_preregistration",
                "stage286_swat_usad_author_evaluation",
            ]
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "check": name,
                "passed": value,
                "evidence": (
                    "Stage283-286 frozen artifacts"
                    if value
                    else "not satisfied by the available contract"
                ),
            }
            for name, value in checks.items()
        ]
    ).to_csv(
        TABLES / f"{PREFIX}_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage 287 SWaT/USAD 证据升级审计",
        "",
        f"- 状态：`{payload['status']}`",
        "- 公开 ICS/CPS 作者代码正式评价子项："
        f"`{payload['public_ics_cps_formal_implementation_subitem']}`",
        f"- P0-3：`{payload['p0_3_after']}`（未勾选整体复选框）",
        f"- 同契约 Trust 对比：`{payload['same_contract_comparison']}`",
        f"- 审计通过：`{payload['checks_passed']}/"
        f"{payload['checks_total_excluding_same_contract']}`",
        "",
        "## USAD 公开 SWaT 结果",
        "",
    ]
    for key, value in payload["usad_test"].items():
        lines.append(f"- `{key}`：`{value}`")
    lines += [
        "",
        "## 边界",
        "",
        f"- {payload['claim_ceiling']}",
        "- USAD 的 `F1=0.2744`、`FAR=0.6226`、`event coverage=0.7714` "
        "按原样保留，不因不利而删除或改写。",
        "",
        "## 尚未满足的 P0-3 项",
        "",
        *[f"- {item}" for item in payload["remaining_p0_3_requirements"]],
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "public_ics_cps_subitem": (
                    payload[
                        "public_ics_cps_formal_implementation_subitem"
                    ]
                ),
                "p0_3_after": payload["p0_3_after"],
                "checks": (
                    f"{payload['checks_passed']}/"
                    f"{payload['checks_total_excluding_same_contract']}"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
