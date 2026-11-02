from __future__ import annotations

"""Stage 293: audit the operational readiness of a future confirmatory package.

Stage 278 defines the requirements for a new confirmatory test package.  This
stage adds a machine-checkable freeze ledger for the package directory.  It is
deliberately read-only: it does not download data, inspect labels, train a
model, calibrate a threshold, or create a performance claim.

The audit is useful only when a genuinely new provider package exists.  Until
then it records the missing package and keeps P0-6 open.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage293_confirmatory_package_freeze_audit"
PACKAGE_ROOT = ROOT / "confirmatory_package"
PROTOCOL_PREFIX = "stage278_confirmatory_test_package_contract"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _file_evidence(path: Path) -> dict[str, object]:
    return {
        "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha256(path) if path.is_file() else "",
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    contract = _json(TABLES / f"{PROTOCOL_PREFIX}.json")
    protocol_freeze = _json(PACKAGE_ROOT / "protocol_freeze.json")
    data_freeze = _json(PACKAGE_ROOT / "data_freeze.json")
    analysis_freeze = _json(PACKAGE_ROOT / "analysis_freeze.json")
    access_gate = _json(PACKAGE_ROOT / "access_gate.json")
    test_manifest = _json(PACKAGE_ROOT / "test_manifest.json")

    freeze_files = {
        "protocol_freeze": _file_evidence(
            PACKAGE_ROOT / "protocol_freeze.json"
        ),
        "data_freeze": _file_evidence(PACKAGE_ROOT / "data_freeze.json"),
        "analysis_freeze": _file_evidence(
            PACKAGE_ROOT / "analysis_freeze.json"
        ),
        "access_gate": _file_evidence(PACKAGE_ROOT / "access_gate.json"),
        "test_manifest": _file_evidence(
            PACKAGE_ROOT / "test_manifest.json"
        ),
    }

    checks = [
        {
            "check_id": "stage278_contract_available",
            "passed": bool(contract),
            "detail": (
                "Stage278 confirmatory contract is present."
                if contract
                else "Stage278 confirmatory contract is missing."
            ),
        },
        {
            "check_id": "protocol_freeze_present",
            "passed": bool(protocol_freeze),
            "detail": (
                "Protocol freeze file is present."
                if protocol_freeze
                else "Protocol freeze file is missing."
            ),
        },
        {
            "check_id": "data_freeze_present",
            "passed": bool(data_freeze),
            "detail": (
                "Data freeze file is present."
                if data_freeze
                else "Data freeze file is missing."
            ),
        },
        {
            "check_id": "analysis_freeze_present",
            "passed": bool(analysis_freeze),
            "detail": (
                "Analysis freeze file is present."
                if analysis_freeze
                else "Analysis freeze file is missing."
            ),
        },
        {
            "check_id": "access_gate_closed_before_outcomes",
            "passed": bool(
                access_gate
                and access_gate.get("labels_inspected") is False
                and access_gate.get("outcome_summaries_inspected") is False
                and access_gate.get("protocol_frozen") is True
                and access_gate.get("data_frozen") is True
                and access_gate.get("analysis_frozen") is True
            ),
            "detail": (
                "Access gate proves labels and outcome summaries were not "
                "inspected before all three freezes."
                if access_gate
                else "Access gate file is missing."
            ),
        },
        {
            "check_id": "test_manifest_hashes_present",
            "passed": bool(
                test_manifest
                and test_manifest.get("rows")
                and all(
                    str(row.get("sha256", "")).strip()
                    for row in test_manifest.get("rows", [])
                )
            ),
            "detail": (
                "Test manifest contains hash-bound files."
                if test_manifest
                else "Test manifest is missing."
            ),
        },
        {
            "check_id": "new_provider_selected",
            "passed": bool(
                protocol_freeze.get("provider_id")
                and protocol_freeze.get("provider_is_new") is True
            ),
            "detail": (
                "A new provider is selected."
                if protocol_freeze
                else "No new provider selection is frozen."
            ),
        },
        {
            "check_id": "confirmatory_package_complete",
            "passed": False,
            "detail": (
                "A complete package requires a new provider, three freezes, "
                "a closed access gate, hash-bound data, and a frozen analysis "
                "script before test outcomes are inspected."
            ),
        },
    ]
    checks[-1]["passed"] = bool(
        all(row["passed"] for row in checks[:-1])
        and access_gate.get("labels_inspected") is False
        and access_gate.get("outcome_summaries_inspected") is False
    )

    passed_count = int(sum(bool(row["passed"]) for row in checks))
    package_present = PACKAGE_ROOT.is_dir()
    new_provider_selected = bool(
        protocol_freeze.get("provider_id")
        and protocol_freeze.get("provider_is_new") is True
    )
    status = (
        "PASS_NEW_CONFIRMATORY_PACKAGE_FROZEN"
        if checks[-1]["passed"]
        else "BLOCKED_NO_FROZEN_CONFIRMATORY_PACKAGE"
    )
    payload = {
        "status": status,
        "protocol_id": "TMQC-CONFIRMATORY-PACKAGE-FREEZE-20260921-v1.0",
        "protocol": (
            "Stage293 audits the presence and hash-bound state of a future "
            "independent confirmatory package. It never downloads data, "
            "inspects labels, trains a method, calibrates a threshold, or "
            "creates a performance result."
        ),
        "protocol_version": "TMQC-CONFIRMATORY-PACKAGE-FREEZE-20260921-v1.0",
        "audit_frozen_date": "2026-09-21",
        "package_root_relative": str(
            PACKAGE_ROOT.relative_to(ROOT)
        ).replace("\\", "/"),
        "package_present": package_present,
        "new_provider_selected": new_provider_selected,
        "freeze_file_count": int(len(freeze_files)),
        "freeze_files": freeze_files,
        "check_count": len(checks),
        "passed_check_count": passed_count,
        "is_a_new_confirmatory_test_package": checks[-1]["passed"],
        "data_actually_downloaded": bool(
            data_freeze.get("raw_files_downloaded") is True
        ),
        "test_labels_inspected": bool(
            access_gate.get("labels_inspected") is True
        ),
        "analysis_frozen": bool(
            analysis_freeze.get("analysis_frozen") is True
        ),
        "claim_ceiling": (
            "NO_CONFIRMATORY_CLAIM"
            if not checks[-1]["passed"]
            else "NEW_CONFIRMATORY_TEST_PACKAGE_FROZEN"
        ),
        "checks": checks,
        "source_status": {
            "stage278": contract.get("status"),
            "stage278_is_new_confirmatory_test_package": contract.get(
                "is_a_new_confirmatory_test_package"
            ),
        },
        "input_sha256": {
            "stage278_contract": (
                _sha256(TABLES / f"{PROTOCOL_PREFIX}.json")
                if (TABLES / f"{PROTOCOL_PREFIX}.json").is_file()
                else ""
            ),
        },
        "limitations": [
            "A missing package is an evidence gap, not a model failure.",
            "The audit cannot create independence by renaming an old test.",
            "No result from an old test may be promoted by this stage.",
        ],
        "code_sha256": _sha256(Path(__file__).resolve()),
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(checks).to_csv(
        TABLES / f"{PREFIX}_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {"freeze_file": key, **value}
            for key, value in freeze_files.items()
        ]
    ).to_csv(
        TABLES / f"{PREFIX}_freeze_files.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = [
        "# Stage 293 新确认性测试包冻结就绪审计",
        "",
        f"- 状态：`{status}`",
        f"- 包目录存在：`{package_present}`",
        f"- 新机构已选择：`{new_provider_selected}`",
        f"- 通过检查：`{passed_count}/{len(checks)}`",
        f"- 构成新确认性测试包：`{checks[-1]['passed']}`",
        f"- 是否查看 test 标签：`{payload['test_labels_inspected']}`",
        f"- 声称上限：`{payload['claim_ceiling']}`",
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
        "## 边界",
        "",
        "- 本阶段不下载新数据、不训练、不调参、不读取 test 标签。",
        "- 没有真正的新机构和新数据时，P0-6 必须保持未通过。",
        "- 旧 test 不能通过改名或复制成为新的确认性测试。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "package_present": package_present,
                "new_provider_selected": new_provider_selected,
                "passed_check_count": passed_count,
                "check_count": len(checks),
                "is_a_new_confirmatory_test_package": checks[-1]["passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
