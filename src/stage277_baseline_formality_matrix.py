from __future__ import annotations

"""Stage 277: freeze the evidence level and provenance of every baseline.

This stage is a read-only provenance audit. It separates a paper-faithful
implementation or author code from a transparent proxy, a local
reimplementation, and a missing baseline. It does not train, recalibrate,
or change any prediction.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage277_baseline_formality_matrix"

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


def _record(
    method: str,
    family: str,
    source_stage: str,
    evidence_level: str,
    formal_implementation: bool,
    author_code_available: bool,
    public_dataset_required: bool,
    public_dataset_available: bool,
    same_endpoint_as_trust: bool,
    can_upgrade_without_new_data: bool,
    status: str,
    limitation: str,
) -> dict[str, object]:
    return {
        "method": method,
        "family": family,
        "source_stage": source_stage,
        "evidence_level": evidence_level,
        "formal_implementation": formal_implementation,
        "author_code_available": author_code_available,
        "public_dataset_required": public_dataset_required,
        "public_dataset_available": public_dataset_available,
        "same_endpoint_as_trust": same_endpoint_as_trust,
        "can_upgrade_without_new_data": can_upgrade_without_new_data,
        "status": status,
        "limitation": limitation,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stage248 = _load("stage248_cross_provider_fair_baseline_audit")
    stage256 = _load("stage256_lightweight_strong_baselines")
    stage262 = _load("stage262_security_domain_baselines")
    stage246 = _load("stage246_frozen_submission_package")
    stage287 = _load("stage287_swat_usad_evidence_upgrade_audit")

    records: list[dict[str, object]] = []

    for method in stage248.get("baseline_methods", []):
        records.append(
            _record(
                method=str(method),
                family="classical_or_cross_provider_baseline",
                source_stage="stage248_cross_provider_fair_baseline_audit",
                evidence_level="frozen_confirmatory_local_implementation",
                formal_implementation=True,
                author_code_available=False,
                public_dataset_required=False,
                public_dataset_available=False,
                same_endpoint_as_trust=True,
                can_upgrade_without_new_data=True,
                status="FROZEN_CONFIRMATORY_BASELINE",
                limitation=(
                    "Local implementation under the frozen Stage248 contract; "
                    "not represented as author code unless separately proven."
                ),
            )
        )

    for method in stage256.get("new_baseline_methods", []):
        records.append(
            _record(
                method=str(method),
                family="lightweight_strong_baseline",
                source_stage="stage256_lightweight_strong_baselines",
                evidence_level="exploratory_local_implementation",
                formal_implementation=True,
                author_code_available=False,
                public_dataset_required=False,
                public_dataset_available=False,
                same_endpoint_as_trust=True,
                can_upgrade_without_new_data=False,
                status="EXPLORATORY_ONLY",
                limitation=(
                    "Local implementation evaluated on the exploratory Stage256 "
                    "contract; this does not upgrade the frozen Stage246/248 main result."
                ),
            )
        )

    for method in stage262.get("new_baseline_methods", []):
        is_proxy = str(method) == "ics_cps_spc_proxy"
        records.append(
            _record(
                method=str(method),
                family=(
                    "public_ics_cps_required"
                    if is_proxy
                    else "security_or_time_series_baseline"
                ),
                source_stage="stage262_security_domain_baselines",
                evidence_level=(
                    "transparent_proxy_not_paper_faithful"
                    if is_proxy
                    else "exploratory_local_implementation"
                ),
                formal_implementation=not is_proxy,
                author_code_available=False,
                public_dataset_required=is_proxy,
                public_dataset_available=False,
                same_endpoint_as_trust=True,
                can_upgrade_without_new_data=False,
                status=(
                    "BLOCKED_PUBLIC_ICS_CPS_SOURCE_AND_DATA_REQUIRED"
                    if is_proxy
                    else "EXPLORATORY_ONLY"
                ),
                limitation=(
                    "This is a transparent SPC proxy. It must not be called a "
                    "paper-faithful reproduction of a public ICS/CPS baseline."
                    if is_proxy
                    else (
                        "Local exploratory implementation; no author-code or "
                        "paper-faithful reproduction claim is made."
                    )
                ),
            )
        )

    if (
        stage287[
            "public_ics_cps_formal_implementation_subitem"
        ]
        == "PASSED"
    ):
        records.append(
            _record(
                method="usad_author_code",
                family="public_ics_cps_anomaly_detection",
                source_stage="stage287_swat_usad_evidence_upgrade_audit",
                evidence_level=(
                    "paper_faithful_author_code_public_benchmark"
                ),
                formal_implementation=True,
                author_code_available=True,
                public_dataset_required=True,
                public_dataset_available=True,
                same_endpoint_as_trust=False,
                can_upgrade_without_new_data=False,
                status="FORMAL_PUBLIC_BASELINE_DIFFERENT_CONTRACT",
                limitation=(
                    "Fixed-commit author code and author-linked SWaT data "
                    "were evaluated under a frozen protocol. The public SWaT "
                    "endpoint is not the same as the TrustMeteo-QC contract, "
                    "so this record does not upgrade the main comparison."
                ),
            )
        )

    methods = pd.DataFrame(records)
    if methods.empty:
        raise RuntimeError("baseline formality matrix is empty")
    if methods["method"].duplicated().any():
        duplicated = methods.loc[
            methods["method"].duplicated(), "method"
        ].tolist()
        raise RuntimeError(f"duplicate methods: {duplicated}")

    proxy = methods[methods["evidence_level"].eq("transparent_proxy_not_paper_faithful")]
    formal = methods[methods["formal_implementation"].astype(bool)]
    blocked = methods[methods["status"].str.startswith("BLOCKED_")]
    dataset_available = methods["public_dataset_available"].astype(bool)
    upgradeable = methods["can_upgrade_without_new_data"].astype(bool)

    methods.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_BASELINE_FORMALITY_AUDIT_WITH_BOUNDED_GAPS",
        "protocol": (
            "Stage277 reads frozen Stage246/248/256/262 provenance and "
            "method inventories only. It does not train, recalibrate, or "
            "change any prediction."
        ),
        "method_count": int(len(methods)),
        "formal_implementation_count": int(len(formal)),
        "transparent_proxy_count": int(len(proxy)),
        "blocked_count": int(len(blocked)),
        "public_dataset_available_count": int(dataset_available.sum()),
        "can_upgrade_without_new_data_count": int(upgradeable.sum()),
        "p0_3_status": (
            "NOT_PASSED"
            if not blocked.empty
            else "PASSED"
        ),
        "paper_faithful_public_baseline_available": bool(
            methods["author_code_available"].astype(bool).any()
        ),
        "input_sha256": {
            name: sha256_file(TABLES / f"{name}.json")
            for name in [
                "stage246_frozen_submission_package",
                "stage248_cross_provider_fair_baseline_audit",
                "stage256_lightweight_strong_baselines",
                "stage262_security_domain_baselines",
                "stage287_swat_usad_evidence_upgrade_audit",
            ]
        },
        "limitations": [
            "A transparent proxy is not a paper-faithful public baseline.",
            "A local implementation is not author code unless provenance says so.",
            "The public ICS/CPS author baseline is present and formally evaluated, but its endpoint differs from TrustMeteo-QC.",
            "This audit does not upgrade exploratory results into the frozen confirmatory result.",
            "No failure result is removed or rewritten by this stage.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Stage 277 基线正式性矩阵",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 方法数：`{payload['method_count']}`",
        f"- 正式实现数：`{payload['formal_implementation_count']}`",
        f"- 透明代理数：`{payload['transparent_proxy_count']}`",
        f"- 阻断项数：`{payload['blocked_count']}`",
        f"- 当前可升级的方法数：`{payload['can_upgrade_without_new_data_count']}`",
        f"- P0-3 状态：`{payload['p0_3_status']}`",
        "",
        "## 边界",
        "",
        "- 透明代理不能写成公开 ICS/CPS 方法的论文忠实复现。",
        "- 本地实现不能冒充作者代码。",
        "- 本审计不训练、不校准阈值、不改变预测或主结果。",
        "- 本审计不把探索性结果升级为确认性结果。",
        "",
    ]
    for row in methods.to_dict("records"):
        lines.append(
            f"- `{row['method']}`：`{row['evidence_level']}`；"
            f"formal={row['formal_implementation']}；"
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
                "formal": payload["formal_implementation_count"],
                "proxy": payload["transparent_proxy_count"],
                "blocked": payload["blocked_count"],
                "p0_3_status": payload["p0_3_status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
