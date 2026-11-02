from __future__ import annotations

"""Stage 289: freeze the final baseline set without creating new results.

This stage is a read-only provenance and scope audit. It freezes which
baselines may enter the same-contract comparison, which baselines must remain
exploratory until a new locked test exists, and which public benchmark is an
independent endpoint that must never be merged into the TrustMeteo-QC
win/loss matrix.

The stage does not train, calibrate, inspect labels, or change any prediction.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage289_final_baseline_set_freeze"
PROTOCOL_ID = "TMQC-FINAL-BASELINE-SET-20260921-v1.0"

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


def _row(
    method: str,
    display_name: str,
    source_stage: str,
    family: str,
    frozen_layer: str,
    same_contract_as_trust: bool,
    independent_public_endpoint: bool,
    exploratory_only: bool,
    future_locked_test_candidate: bool,
    evidence_level: str,
    status: str,
    required_action: str,
    limitation: str,
) -> dict[str, object]:
    return {
        "method": method,
        "display_name": display_name,
        "source_stage": source_stage,
        "family": family,
        "frozen_layer": frozen_layer,
        "same_contract_as_trust": same_contract_as_trust,
        "independent_public_endpoint": independent_public_endpoint,
        "exploratory_only": exploratory_only,
        "future_locked_test_candidate": future_locked_test_candidate,
        "evidence_level": evidence_level,
        "status": status,
        "required_action": required_action,
        "limitation": limitation,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stage246 = _load("stage246_frozen_submission_package")
    stage248 = _load("stage248_cross_provider_fair_baseline_audit")
    stage256 = _load("stage256_lightweight_strong_baselines")
    stage262 = _load("stage262_security_domain_baselines")
    stage277 = _load("stage277_baseline_formality_matrix")
    stage287 = _load("stage287_swat_usad_evidence_upgrade_audit")
    stage288 = _load("stage288_swat_usad_resource_audit")

    if str(stage277.get("p0_3_status")) != "NOT_PASSED":
        raise RuntimeError("Stage277 P0-3 status changed unexpectedly")
    if str(stage287.get("same_contract_comparison")) != "NOT_AVAILABLE":
        raise RuntimeError("USAD same-contract comparison unexpectedly changed")
    if not bool(stage287.get("paper_faithful_public_baseline_available")):
        raise RuntimeError("public ICS/CPS baseline availability changed")
    rows: list[dict[str, object]] = []
    for method in stage248.get("baseline_methods", []):
        rows.append(
            _row(
                method=str(method),
                display_name=str(method),
                source_stage="stage248_cross_provider_fair_baseline_audit",
                family="same_contract_classical_baseline",
                frozen_layer="same_contract_main_candidate",
                same_contract_as_trust=True,
                independent_public_endpoint=False,
                exploratory_only=False,
                future_locked_test_candidate=True,
                evidence_level="frozen_confirmatory_local_implementation",
                status="FROZEN_SAME_CONTRACT_CANDIDATE",
                required_action=(
                    "Use the frozen Stage248 split, attack stream, FAR=0.01 "
                    "threshold rule, event definition, and evaluation code in "
                    "the new locked test."
                ),
                limitation=(
                    "Local implementation under the frozen Stage248 contract; "
                    "not represented as author code."
                ),
            )
        )

    for method in stage256.get("new_baseline_methods", []):
        rows.append(
            _row(
                method=str(method),
                display_name=str(method),
                source_stage="stage256_lightweight_strong_baselines",
                family="lightweight_reconstruction_baseline",
                frozen_layer="future_locked_test_candidate",
                same_contract_as_trust=True,
                independent_public_endpoint=False,
                exploratory_only=True,
                future_locked_test_candidate=True,
                evidence_level="exploratory_local_implementation",
                status="EXPLORATORY_UNTIL_NEW_LOCKED_TEST",
                required_action=(
                    "Include in the frozen baseline set for a future locked "
                    "test; do not promote the existing exploratory result."
                ),
                limitation=(
                    "The Stage256 result is exploratory and cannot replace "
                    "the frozen Stage246/248 main result."
                ),
            )
        )

    for method in stage262.get("new_baseline_methods", []):
        is_proxy = str(method) == "ics_cps_spc_proxy"
        rows.append(
            _row(
                method=str(method),
                display_name=str(method),
                source_stage="stage262_security_domain_baselines",
                family=(
                    "transparent_ics_cps_proxy"
                    if is_proxy
                    else "security_or_time_series_baseline"
                ),
                frozen_layer="future_locked_test_candidate",
                same_contract_as_trust=True,
                independent_public_endpoint=False,
                exploratory_only=True,
                future_locked_test_candidate=True,
                evidence_level=(
                    "transparent_proxy_not_paper_faithful"
                    if is_proxy
                    else "exploratory_local_implementation"
                ),
                status=(
                    "FROZEN_AS_TRANSPARENT_PROXY_ONLY"
                    if is_proxy
                    else "EXPLORATORY_UNTIL_NEW_LOCKED_TEST"
                ),
                required_action=(
                    "Keep the proxy label and evaluate it under the same new "
                    "locked-test contract; it cannot be renamed as a "
                    "paper-faithful public ICS/CPS method."
                    if is_proxy
                    else (
                        "Include in the frozen baseline set for a future "
                        "locked test; do not promote the existing exploratory "
                        "result."
                    )
                ),
                limitation=(
                    "Transparent SPC proxy; not a paper-faithful public "
                    "ICS/CPS reproduction."
                    if is_proxy
                    else (
                        "Local exploratory implementation; no author-code or "
                        "paper-faithful reproduction claim is made."
                    )
                ),
            )
        )

    rows.append(
        _row(
            method="usad_author_code",
            display_name="USAD author code on public SWaT",
            source_stage=(
                "stage287_swat_usad_evidence_upgrade_audit;"
                "stage288_swat_usad_resource_audit"
            ),
            family="public_ics_cps_anomaly_detection",
            frozen_layer="independent_public_benchmark",
            same_contract_as_trust=False,
            independent_public_endpoint=True,
            exploratory_only=False,
            future_locked_test_candidate=False,
            evidence_level="paper_faithful_author_code_public_benchmark",
            status="FORMAL_PUBLIC_BASELINE_DIFFERENT_CONTRACT",
            required_action=(
                "Report as an independent public SWaT benchmark. Do not merge "
                "it into the TrustMeteo-QC same-endpoint win/loss matrix."
            ),
            limitation=(
                "Author-linked SWaT is not the TrustMeteo-QC contract and is "
                "not a production incident log. Stage288 measures resources "
                "only and does not replace Stage286 metrics."
            ),
        )
    )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("final baseline set is empty")
    if frame["method"].duplicated().any():
        duplicated = frame.loc[
            frame["method"].duplicated(), "method"
        ].tolist()
        raise RuntimeError(f"duplicate methods: {duplicated}")

    same_contract = frame[frame["same_contract_as_trust"].astype(bool)]
    independent = frame[frame["independent_public_endpoint"].astype(bool)]
    exploratory = frame[frame["exploratory_only"].astype(bool)]
    future = frame[frame["future_locked_test_candidate"].astype(bool)]
    formal = frame[
        frame["status"].isin(
            {
                "FROZEN_SAME_CONTRACT_CANDIDATE",
                "FORMAL_PUBLIC_BASELINE_DIFFERENT_CONTRACT",
            }
        )
    ]
    proxy = frame[frame["family"].eq("transparent_ics_cps_proxy")]

    if len(same_contract) != 11:
        raise RuntimeError(
            f"expected 11 same-contract candidates, got {len(same_contract)}"
        )
    if len(independent) != 1:
        raise RuntimeError(
            f"expected 1 independent public endpoint, got {len(independent)}"
        )
    if len(exploratory) != 8:
        raise RuntimeError(
            f"expected 8 exploratory candidates, got {len(exploratory)}"
        )
    if len(proxy) != 1:
        raise RuntimeError(f"expected 1 proxy, got {len(proxy)}")
    if bool(independent["same_contract_as_trust"].any()):
        raise RuntimeError("independent public endpoint entered main contract")

    frame.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_FINAL_BASELINE_SET_FROZEN_WITH_SCOPE_BOUNDARIES",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "Stage289 freezes the final baseline set from existing frozen "
            "audits only. It does not train, calibrate, inspect labels, alter "
            "predictions, or promote an exploratory result."
        ),
        "main_endpoint_source": (
            "stage248_cross_provider_fair_baseline_audit"
        ),
        "main_primary_far": float(stage248.get("primary_far")),
        "main_primary_endpoint": str(stage248.get("primary_endpoint")),
        "method_count": int(len(frame)),
        "same_contract_candidate_count": int(len(same_contract)),
        "independent_public_benchmark_count": int(len(independent)),
        "exploratory_only_count": int(len(exploratory)),
        "future_locked_test_candidate_count": int(len(future)),
        "formal_frozen_record_count": int(len(formal)),
        "transparent_proxy_count": int(len(proxy)),
        "final_set_frozen": True,
        "can_add_or_remove_after_freeze": False,
        "same_contract_win_loss_matrix_available": False,
        "p0_3_status": "IN_PROGRESS",
        "claim_ceiling": (
            "final baseline set frozen; same-contract new locked-test "
            "comparison, all-baseline resource completion, and pre-registered "
            "non-inferiority/advantage gates remain open"
        ),
        "layers": {
            "same_contract_main_candidates": same_contract[
                "method"
            ].tolist(),
            "future_locked_test_candidates": future["method"].tolist(),
            "independent_public_benchmark": independent["method"].tolist(),
            "transparent_proxy": proxy["method"].tolist(),
        },
        "supporting_values": {
            "stage246_status": stage246.get("status"),
            "stage248_method_count": stage248.get("method_count"),
            "stage256_new_baseline_count": len(
                stage256.get("new_baseline_methods", [])
            ),
            "stage262_new_baseline_count": len(
                stage262.get("new_baseline_methods", [])
            ),
            "stage277_method_count": stage277.get("method_count"),
            "stage277_p0_3_status": stage277.get("p0_3_status"),
            "stage287_same_contract_comparison": stage287.get(
                "same_contract_comparison"
            ),
            "stage288_resource_status": stage288.get("status"),
        },
        "input_sha256": {
            name: sha256_file(TABLES / f"{name}.json")
            for name in [
                "stage246_frozen_submission_package",
                "stage248_cross_provider_fair_baseline_audit",
                "stage256_lightweight_strong_baselines",
                "stage262_security_domain_baselines",
                "stage277_baseline_formality_matrix",
                "stage287_swat_usad_evidence_upgrade_audit",
                "stage288_swat_usad_resource_audit",
            ]
        },
        "limitations": [
            "Freezing the set does not create a same-contract comparison and "
            "does not close P0-3.",
            "Exploratory Stage256/262 results remain exploratory until they "
            "are evaluated on a new locked test.",
            "The ICS/CPS SPC proxy remains a transparent proxy.",
            "USAD on public SWaT is an independent public benchmark with a "
            "different endpoint and is not a production incident log.",
            "No baseline failure, unfavorable result, or missing measurement "
            "is removed by this stage.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Stage 289 最终基线集合冻结",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{payload['protocol_id']}`",
        f"- 方法总数：`{payload['method_count']}`",
        f"- 同契约候选：`{payload['same_contract_candidate_count']}`",
        f"- 独立公开基准：`{payload['independent_public_benchmark_count']}`",
        f"- 探索性方法：`{payload['exploratory_only_count']}`",
        f"- 未来锁定test候选：`{payload['future_locked_test_candidate_count']}`",
        f"- 冻结后是否允许增删：`{payload['can_add_or_remove_after_freeze']}`",
        f"- 同契约全胜负矩阵是否可用：`{payload['same_contract_win_loss_matrix_available']}`",
        f"- P0-3 状态：`{payload['p0_3_status']}`",
        "",
        "## 边界",
        "",
        "- 冻结集合不等于完成统一重跑，也不等于 P0-3 通过。",
        "- Stage256/262 的现有结果仍是探索性，不能因进入候选集合而升级。",
        "- ICS/CPS SPC 仍是透明代理，不能改名成论文忠实公开方法。",
        "- USAD/SWaT 是不同端点的独立公开基准，不能并入同契约胜负矩阵。",
        "- 本阶段不删除任何不利结果、缺失测量或失败状态。",
        "",
        "## 方法分层",
        "",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"- `{row['method']}`：`{row['frozen_layer']}`；"
            f"same_contract={row['same_contract_as_trust']}；"
            f"status={row['status']}"
        )
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "methods": payload["method_count"],
                "same_contract": payload[
                    "same_contract_candidate_count"
                ],
                "independent_public": payload[
                    "independent_public_benchmark_count"
                ],
                "exploratory": payload["exploratory_only_count"],
                "p0_3_status": payload["p0_3_status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
