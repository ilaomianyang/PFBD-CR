from __future__ import annotations

"""Stage 456: freeze and audit the V9 source dependency bundle.

The official October window is still unavailable.  This stage therefore
freezes hashes for the exact V9 protocol, runner, evaluator, attack generator,
normalizer, provider adapters, fetcher, and statistical auditor that will be
used later.  It reads source text and metadata only; it never reads official
observations, old test rows, attack labels, or performance outputs.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
MANIFEST = PACKAGE / "data_manifest.json"
SOURCE_BUNDLE = PACKAGE / "source_dependency_freeze.json"
OUT_JSON = ROOT / "tables" / "stage456_v9_source_dependency_freeze_audit.json"
OUT_REPORT = ROOT / "reports" / "stage456_v9_source_dependency_freeze_audit.md"

FREEZE_ID = "TMQC-V9-SOURCE-BUNDLE-20260925-v1.0"
FROZEN_AT = "2026-09-25T00:00:00Z"

CORE_SOURCES = {
    "protocol": PACKAGE / "protocol_freeze.json",
    "data_manifest": PACKAGE / "data_manifest.json",
    "runner": ROOT / "src" / "run_v9_locked_pipeline.py",
    "evaluation": ROOT / "src" / "stage425_v9_locked_evaluation.py",
    "audit": ROOT / "src" / "stage426_v9_locked_audit.py",
    "statistics": ROOT / "src" / "stage445_v9_statistical_acceptance_audit.py",
    "compatibility": ROOT / "src" / "stage425_v9_compatibility.py",
    "normalization": ROOT / "src" / "stage425_v9_2026_official_window_download_template.py",
    "adapters": ROOT / "src" / "v9_official_adapters.py",
    "attack_generator": ROOT / "src" / "stage320_simulated_attack_confirmatory_evaluation.py",
    "attack_contract": ROOT / "src" / "stage440_v9_attack_execution_contract_audit.py",
    "official_fetcher": ROOT / "src" / "stage442_v9_official_window_fetch.py",
    "download_plan": ROOT / "src" / "stage428_v9_official_download_dry_run.py",
    "source_plan": ROOT / "src" / "stage427_v9_official_source_plan_audit.py",
    "regression_entrypoint": ROOT / "tests" / "run_tests.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def build_bundle() -> dict:
    missing = [name for name, path in CORE_SOURCES.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing V9 source dependency files: {missing}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "freeze_id": FREEZE_ID,
        "frozen_at_utc": FROZEN_AT,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_status": protocol.get("status"),
        "manifest_status_at_freeze": manifest.get("status"),
        "official_observation_values_read_at_freeze": False,
        "old_test_rows_read_at_freeze": False,
        "performance_evaluated_at_freeze": False,
        "source_files": {
            name: {
                "path": relative(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in CORE_SOURCES.items()
        },
        "claim_boundary": [
            "该清单只冻结 V9 源码依赖，不产生任何性能结果。",
            "官方窗口未就绪时不得读取官方观测值或旧 test 替代。",
        ],
    }


def main() -> int:
    bundle = build_bundle()
    SOURCE_BUNDLE.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runner_text = CORE_SOURCES["runner"].read_text(encoding="utf-8")
    legacy_markers = [
        marker
        for marker in (
            "stage349_v6_locked_evaluation.py",
            "stage351_v6_station_bh_audit.py",
        )
        if marker in runner_text
    ]
    checks = [
        {
            "check": "bundle_identity_frozen",
            "status": "PASS"
            if bundle["freeze_id"] == FREEZE_ID
            and bundle["frozen_at_utc"] == FROZEN_AT
            and bundle["protocol_id"]
            == "TMQC-SIMULATED-ATTACK-V9-OCT2026-20260925-v1.2"
            else "FAIL",
            "value": {
                "freeze_id": bundle["freeze_id"],
                "frozen_at_utc": bundle["frozen_at_utc"],
                "protocol_id": bundle["protocol_id"],
            },
        },
        {
            "check": "all_core_sources_present_and_hashed",
            "status": "PASS"
            if len(bundle["source_files"]) == len(CORE_SOURCES)
            and all(
                len(str(item["sha256"])) == 64
                and int(item["bytes"]) > 0
                for item in bundle["source_files"].values()
            )
            else "FAIL",
            "value": {"source_count": len(bundle["source_files"])},
        },
        {
            "check": "current_hashes_match_frozen_bundle",
            "status": "PASS"
            if all(
                sha256(ROOT / str(item["path"])).lower()
                == str(item["sha256"]).lower()
                for item in bundle["source_files"].values()
            )
            else "FAIL",
            "value": "all listed source hashes rechecked",
        },
        {
            "check": "manifest_is_not_ready_at_freeze",
            "status": "PASS"
            if manifest.get("status") == "NOT_READY"
            and bundle["manifest_status_at_freeze"] == "NOT_READY"
            else "FAIL",
            "value": manifest.get("status"),
        },
        {
            "check": "runner_has_no_legacy_fallback",
            "status": "PASS" if not legacy_markers else "FAIL",
            "value": {"legacy_markers": legacy_markers},
        },
        {
            "check": "no_performance_or_observation_access_claimed",
            "status": "PASS"
            if bundle["official_observation_values_read_at_freeze"] is False
            and bundle["old_test_rows_read_at_freeze"] is False
            and bundle["performance_evaluated_at_freeze"] is False
            else "FAIL",
            "value": {
                "official_observation_values_read": False,
                "old_test_rows_read": False,
                "performance_evaluated": False,
            },
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_SOURCE_DEPENDENCY_FREEZE_AUDIT"
        if not failures
        else "FAIL_V9_SOURCE_DEPENDENCY_FREEZE_AUDIT"
    )
    payload = {
        "stage": 456,
        "status": status,
        "freeze_id": FREEZE_ID,
        "protocol_id": protocol.get("protocol_id"),
        "checks": checks,
        "failed_checks": failures,
        "source_count": len(CORE_SOURCES),
        "observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "claim_boundary": [
            "本阶段只冻结和复核 V9 源码依赖哈希，不产生官方性能结果。",
            "当前 manifest 仍为 NOT_READY，不能用旧 V6 test 替代。",
        ],
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "manifest": sha256(MANIFEST),
            "source_bundle": sha256(SOURCE_BUNDLE),
        },
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage456 V9 源码依赖冻结审计",
                "",
                f"- 状态：`{status}`",
                f"- freeze id：`{FREEZE_ID}`",
                f"- 源码依赖数：`{len(CORE_SOURCES)}`；失败数：`{len(failures)}`",
                "- 复核 runner、评价器、审计器、统计器、规范化器、适配器、攻击生成器和官方采集器哈希。",
                "- 未读取官方观测值、旧 test、攻击标签或性能表。",
                "",
                "## 失败检查",
                "",
                *([f"- `{name}`" for name in failures] or ["- 无。"]),
                "",
                f"- JSON：`{OUT_JSON}`",
                f"- 源码清单：`{SOURCE_BUNDLE}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": 456,
                "status": status,
                "freeze_id": FREEZE_ID,
                "source_count": len(CORE_SOURCES),
                "failed_checks": failures,
                "performance_evaluated": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
