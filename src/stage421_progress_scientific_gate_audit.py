from __future__ import annotations

"""Stage 421: read-only audit of progress checkboxes against evidence gates."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROGRESS = ROOT / "进度表.md"
INPUTS = {
    "stage246": TABLES / "stage246_frozen_submission_package.json",
    "stage267": TABLES / "stage267_smhi_locked_transfer.json",
    "stage352": TABLES / "stage352_submission_consistency_audit_v6_sync.json",
    "stage406": TABLES / "stage406_fmi_sep2026_factor_audit.json",
    "stage407": TABLES / "stage407_met_norway_sep2026_factor_audit.json",
    "stage417": TABLES / "stage417_cross_temporal_factor_failure_decomposition.json",
    "stage418": TABLES / "stage418_unseen_window_feasibility_audit.json",
    "stage419": TABLES / "stage419_testbed_resource_only_audit.json",
    "stage420": TABLES / "stage420_resource_semantic_hash_audit.json",
}
OUTPUT = TABLES / "stage421_progress_scientific_gate_audit.json"
REPORT = REPORTS / "stage421_progress_scientific_gate_audit.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkbox(text: str, phrase: str) -> str | None:
    match = re.search(rf"- \[([ xX])\] [^\n]*{re.escape(phrase)}", text)
    return match.group(1).lower() if match else None


def main() -> int:
    progress = PROGRESS.read_text(encoding="utf-8")
    payloads = {name: load(path) for name, path in INPUTS.items()}
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}
        )

    stage246 = payloads["stage246"]
    stage352 = payloads["stage352"]
    add(
        "frozen_provider_scope",
        stage246.get("provider_count") == 4
        and stage352.get("v6_scope", {}).get("provider_count") == 4
        and set(stage352.get("v6_scope", {}).get("providers", []))
        == {"ECCC", "KNMI", "MeteoSwiss", "NOAA ISD"},
        {
            "stage246_provider_count": stage246.get("provider_count"),
            "stage352_scope": stage352.get("v6_scope", {}).get("providers"),
        },
    )
    add(
        "v6_consistency_pass",
        stage352.get("status") == "PASS"
        and stage352.get("finding_count") == 0
        and stage352.get("v6_scope", {}).get("best_candidate")
        == "trust_shape_union",
        {
            "stage352_status": stage352.get("status"),
            "finding_count": stage352.get("finding_count"),
            "best_candidate": stage352.get("v6_scope", {}).get("best_candidate"),
        },
    )
    stage267 = payloads["stage267"]
    add(
        "smhi_transfer_failure_retained",
        stage267.get("transfer_gate_status") == "NOT_PASSED"
        and (
            checkbox(progress, "Trust 在新增机构上的机构等权主终点")
            in {"", None}
            or checkbox(progress, "Trust 在新增机构上的机构等权主终点") == " "
        ),
        {
            "transfer_gate_status": stage267.get("transfer_gate_status"),
            "checkbox_state": checkbox(progress, "Trust 在新增机构上的机构等权主终点"),
        },
    )
    stage406 = payloads["stage406"]
    stage407 = payloads["stage407"]
    fmi_share = stage406.get("scenario_win_share", {}).get("share")
    met_share = stage407.get("scenario_win_share", {}).get("share")
    add(
        "temporal_70pct_failures_retained",
        fmi_share == 0.6060606060606061
        and met_share == 0.5050505050505051
        and checkbox(progress, "FMI 场景×基线达到 `70%`") == " "
        and checkbox(progress, "MET Norway 场景×基线达到 `70%`") == " ",
        {
            "fmi_win_share": fmi_share,
            "met_norway_win_share": met_share,
        },
    )
    stage417 = payloads["stage417"]
    add(
        "unseen_combination_instability_retained",
        stage417.get("unseen_positive_rows") == 8
        and stage417.get("unseen_zero_rows") == 22
        and stage417.get("unseen_negative_rows") == 14,
        {
            "positive": stage417.get("unseen_positive_rows"),
            "zero": stage417.get("unseen_zero_rows"),
            "negative": stage417.get("unseen_negative_rows"),
        },
    )
    stage418 = payloads["stage418"]
    add(
        "network_block_not_promoted",
        stage418.get("status") == "BLOCKED_NETWORK_UNVERIFIED"
        and stage418.get("performance_status") == "NOT_EVALUATED"
        and stage418.get("observation_values_read") is False,
        {
            "status": stage418.get("status"),
            "performance_status": stage418.get("performance_status"),
        },
    )
    stage419 = payloads["stage419"]
    stage420 = payloads["stage420"]
    add(
        "resource_only_stages_marked_complete",
        stage419.get("status") == "PASS_TESTBED_RESOURCE_ONLY_AUDIT"
        and stage420.get("status") == "PASS_STAGE419_RESOURCE_SEMANTIC_HASH_AUDIT"
        and stage419.get("performance_status") == "NOT_EVALUATED"
        and stage420.get("failure_count") == 0,
        {
            "stage419_status": stage419.get("status"),
            "stage420_status": stage420.get("status"),
        },
    )

    failures = [item for item in checks if item["status"] == "FAIL"]
    result = {
        "status": "PASS_STAGE421_PROGRESS_SCIENTIFIC_GATE_AUDIT"
        if not failures
        else "FAIL_STAGE421_PROGRESS_SCIENTIFIC_GATE_AUDIT",
        "stage": 421,
        "progress_path": str(PROGRESS.relative_to(ROOT)),
        "checks": checks,
        "failure_count": len(failures),
        "input_hashes": {
            name: sha256_file(path) for name, path in INPUTS.items()
        },
        "rule": (
            "只读对齐进度表与既有证据门槛；不重跑、不调参、不访问旧test标签，"
            "不把工程通过升级为科学性能通过。"
        ),
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage421 进度表科学闸门一致性审计",
        "",
        f"- 状态：`{result['status']}`",
        "- 规则：只读对齐进度表和既有产物，不重跑、不调参、不访问旧 test 标签。",
        "",
        "| 检查 | 状态 |",
        "|---|---|",
    ]
    lines.extend(
        f"| `{item['check']}` | `{item['status']}` |" for item in checks
    )
    lines.extend(
        [
            "",
            "## 科学边界",
            "",
            "- Stage267 的 SMHI 迁移失败、Stage406/407 的时间窗低于 70%、Stage417 的未见组合不稳定均被保留。",
            "- Stage418 网络不可验证不被解释为数据不存在或性能失败。",
            "- Stage419/420 仅为资源与语义审计，不是检测性能结果。",
            "- 本审计不会关闭全面领先、全部基线或真实生产攻击门槛。",
            f"- JSON：`{OUTPUT}`",
            f"- 报告：`{REPORT}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "failure_count": result["failure_count"],
                "output": str(OUTPUT),
                "report": str(REPORT),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
