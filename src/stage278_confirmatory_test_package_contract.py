from __future__ import annotations

"""Stage 278: contract for a future independent confirmatory test package.

This stage does not download data, inspect labels, train a model, calibrate a
threshold, or create a performance claim. It records the minimum protocol,
selection, split, endpoint, baseline, analysis, and access-control contract
that must be frozen before a new external provider can become confirmatory.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage278_confirmatory_test_package_contract"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requirement(
    requirement_id: str,
    category: str,
    requirement: str,
    evidence_required: str,
) -> dict[str, object]:
    return {
        "requirement_id": requirement_id,
        "category": category,
        "requirement": requirement,
        "evidence_required": evidence_required,
        "currently_satisfied": False,
        "status": "NOT_YET_FROZEN",
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    requirements = [
        _requirement(
            "SELECTION_PROVIDER",
            "selection",
            "Select an independent national network or provider that was "
            "not used for any previous model, feature, threshold, or attack "
            "selection decision.",
            "Provider identity, official landing page, access route, and a "
            "signed selection timestamp.",
        ),
        _requirement(
            "SELECTION_STATIONS",
            "selection",
            "Freeze station identifiers, coordinates, sensor families, and "
            "inclusion rules before downloading observations.",
            "Station manifest with stable IDs and a SHA256-bound timestamp.",
        ),
        _requirement(
            "SELECTION_PERIOD",
            "selection",
            "Freeze calendar period, train/valid/test boundaries, timezone "
            "convention, and missingness handling.",
            "Split manifest with no test-label summary available at freeze time.",
        ),
        _requirement(
            "DATA_LICENSE",
            "data",
            "Record redistribution rights, attribution, retention, and "
            "privacy or security constraints for the provider.",
            "License text hash and governance approval identifier.",
        ),
        _requirement(
            "DATA_RAW_HASH",
            "data",
            "Freeze the complete raw download manifest before any test "
            "observation is analyzed.",
            "Raw file list, byte sizes, SHA256 values, and download timestamps.",
        ),
        _requirement(
            "DATA_NORMALIZATION",
            "data",
            "Freeze normalization, units, quality flags, resampling, and "
            "missing-value policy.",
            "Code and configuration SHA256 plus a coverage audit.",
        ),
        _requirement(
            "METHOD_FEATURES",
            "method",
            "Freeze feature definitions, causal windows, station recalibration "
            "rules, and whether each method can run online.",
            "Feature contract and code SHA256.",
        ),
        _requirement(
            "METHOD_THRESHOLD",
            "method",
            "Freeze threshold selection on train-normal and valid-normal rows "
            "only, including the primary FAR.",
            "Calibration protocol and code SHA256.",
        ),
        _requirement(
            "METHOD_ATTACKS",
            "method",
            "Freeze controlled attack types, amplitudes, durations, positions, "
            "seeds, and the separation from natural weak labels.",
            "Attack-generation code and parameter hash.",
        ),
        _requirement(
            "BASELINE_SET",
            "baseline",
            "Freeze the complete baseline set and provenance, including "
            "author-code, paper-faithful, local, and transparent-proxy labels.",
            "Baseline provenance matrix and code/data availability statement.",
        ),
        _requirement(
            "BASELINE_BUDGET",
            "baseline",
            "Freeze equal or explicitly justified training budgets and all "
            "resource measurements for every baseline.",
            "Budget table with fit rows, time, memory, seed, and failures.",
        ),
        _requirement(
            "ENDPOINT_PRIMARY",
            "statistics",
            "Freeze the primary endpoint before test access; report pooled, "
            "station-equal, and provider-equal summaries together.",
            "Analysis protocol with estimator definitions.",
        ),
        _requirement(
            "ENDPOINT_MULTIPLICITY",
            "statistics",
            "Freeze the non-inferiority margin, confidence level, multiplicity "
            "correction, and sensitivity analyses.",
            "Statistical analysis plan with code SHA256.",
        ),
        _requirement(
            "ENDPOINT_EVENTS",
            "statistics",
            "Freeze the event definition, denominator, length strata, row "
            "recall, precision, FAR, F1, AUPRC, and event coverage reporting.",
            "Event contract and denominator audit.",
        ),
        _requirement(
            "ACCESS_FREEZE",
            "access",
            "Keep test labels and outcome summaries inaccessible until the "
            "protocol, data, method, baseline, and analysis hashes are frozen.",
            "Three timestamps: protocol freeze, data freeze, analysis freeze.",
        ),
        _requirement(
            "ACCESS_NO_RETURN",
            "access",
            "Any post-freeze variant informed by the new test is exploratory "
            "and cannot replace the frozen confirmatory result.",
            "Versioned analysis registry and change log.",
        ),
        _requirement(
            "REPORTING_NEGATIVE",
            "reporting",
            "Publish all baseline wins, failures, missing values, and evidence "
            "limitations with the confirmatory result.",
            "Result manifest that retains negative and non-estimable outcomes.",
        ),
        _requirement(
            "REPORTING_LAYERS",
            "reporting",
            "Keep controlled injections, authorized testbed attacks, natural "
            "weak labels, and real production incidents in separate tables.",
            "Evidence-layer labels attached to every result row.",
        ),
    ]
    frame = pd.DataFrame(requirements)
    if frame.empty or frame["requirement_id"].duplicated().any():
        raise RuntimeError("invalid confirmatory contract requirements")
    frame.to_csv(
        TABLES / f"{PREFIX}_requirements.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "status": "PROTOCOL_TEMPLATE_ONLY_NOT_A_CONFIRMATORY_TEST",
        "protocol": (
            "This stage only freezes a checklist for a future independent "
            "confirmatory test package. It does not download or inspect data, "
            "train a method, calibrate a threshold, or produce performance "
            "numbers."
        ),
        "protocol_version": "TMQC-CONFIRMATORY-CONTRACT-20260921-v1.0",
        "protocol_frozen_date": "2026-09-21",
        "requirement_count": int(len(frame)),
        "satisfied_count": int(frame["currently_satisfied"].sum()),
        "not_yet_frozen_count": int(
            frame["status"].eq("NOT_YET_FROZEN").sum()
        ),
        "is_a_new_confirmatory_test_package": False,
        "data_actually_downloaded": False,
        "test_labels_inspected": False,
        "analysis_frozen": False,
        "claim_ceiling": "NO_CONFIRMATORY_CLAIM",
        "requirements": requirements,
        "limitations": [
            "This is a protocol skeleton, not a frozen data package.",
            "No new independent provider is present in this stage.",
            "No performance result or superiority claim is produced.",
            "The existing old test remains historical and cannot be upgraded "
            "to a new confirmatory test by this stage.",
        ],
        "code_sha256": _sha256(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage 278 新确认性测试包契约模板",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 要求项：`{payload['requirement_count']}`",
        f"- 已满足：`{payload['satisfied_count']}`",
        f"- 尚未冻结：`{payload['not_yet_frozen_count']}`",
        "- 是否构成新确认性测试包：`False`",
        "- 是否下载新数据：`False`",
        "- 是否查看test标签：`False`",
        "- 分析是否冻结：`False`",
        "",
        "## 边界",
        "",
        "- 本阶段只是未来独立测试包的协议骨架。",
        "- 未选择新的独立机构，未生成任何性能数值。",
        "- 旧test不能被本阶段升级为新的确认性测试。",
        "",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "requirement_count": payload["requirement_count"],
                "satisfied_count": payload["satisfied_count"],
                "is_a_new_confirmatory_test_package": payload[
                    "is_a_new_confirmatory_test_package"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
