from __future__ import annotations

"""Stage 449: audit the V9 package immediately before the official window.

This is a read-only integrity audit.  It verifies that the frozen protocol,
entrypoints, attack/statistical contracts and NOT_READY data manifest are
still aligned.  It deliberately refuses to create a performance result and
does not inspect observation values, attack labels or historical test rows.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
MANIFEST = PACKAGE / "data_manifest.json"
RAW_DIR = PACKAGE / "raw_official_2026_10"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
OUT_JSON = TABLES / "stage449_v9_prewindow_freeze_integrity_audit.json"
OUT_REPORT = REPORTS / "stage449_v9_prewindow_freeze_integrity_audit.md"

EXPECTED_PROTOCOL_ID = "TMQC-SIMULATED-ATTACK-V9-OCT2026-20260925-v1.2"
EXPECTED_PROVIDERS = {"MeteoSwiss", "NOAA ISD", "KNMI", "ECCC"}
EXPECTED_SCENARIOS = {
    "weak_spike_short",
    "strong_spike_long",
    "weak_offset_short",
    "strong_offset_long",
    "short_drift",
    "long_drift",
    "short_lag",
    "long_delay",
    "short_stuck",
    "long_stuck",
    "weak_noise_short",
    "strong_noise_long",
    "coordinated_drift_unseen",
}
EXPECTED_BASELINES = {
    "robust_max_deviation",
    "isolation_forest",
    "one_class_svm_20k",
    "pca_reconstruction",
    "mlp_autoencoder_reconstruction",
    "knn_distance",
    "ewma_deviation",
    "arima_innovation",
    "gru_predictor",
    "lightweight_autoencoder",
    "ics_cps_spc_proxy",
}

CORE_FILES = {
    "protocol": PROTOCOL,
    "manifest": MANIFEST,
    "runner": ROOT / "src" / "run_v9_locked_pipeline.py",
    "evaluation": ROOT / "src" / "stage425_v9_locked_evaluation.py",
    "audit": ROOT / "src" / "stage426_v9_locked_audit.py",
    "statistics": ROOT / "src" / "stage445_v9_statistical_acceptance_audit.py",
    "official_fetcher": ROOT / "src" / "stage442_v9_official_window_fetch.py",
    "normalization_template": ROOT / "src" / "stage425_v9_2026_official_window_download_template.py",
    "compatibility": ROOT / "src" / "stage425_v9_compatibility.py",
    "adapters": ROOT / "src" / "v9_official_adapters.py",
    "attack_contract": ROOT / "src" / "stage440_v9_attack_execution_contract_audit.py",
}

PERFORMANCE_OUTPUTS = (
    TABLES / "stage425_v9_locked_evaluation_detail.csv",
    TABLES / "stage425_v9_locked_evaluation_scenario.csv",
    TABLES / "stage425_v9_locked_evaluation_station.csv",
    TABLES / "stage425_v9_locked_evaluation.json",
    TABLES / "stage426_v9_locked_audit.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runner_text = CORE_FILES["runner"].read_text(encoding="utf-8")

    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, value: object, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "value": value,
                "detail": detail,
            }
        )

    add(
        "protocol_identity_frozen",
        protocol.get("protocol_id") == EXPECTED_PROTOCOL_ID
        and protocol.get("status")
        == "CONFIRMATORY_PROTOCOL_FROZEN_PENDING_NEW_OFFICIAL_WINDOW",
        {
            "protocol_id": protocol.get("protocol_id"),
            "status": protocol.get("status"),
        },
        "V9 协议必须仍是 2026-10 官方窗口的 v1.2 冻结版本。",
    )
    add(
        "provider_scope_unchanged",
        protocol.get("provider_count") == 4
        and set(protocol.get("providers", {})) == EXPECTED_PROVIDERS,
        sorted(protocol.get("providers", {})),
        "不得在窗口前新增、删除或替换 provider。",
    )
    add(
        "scenario_scope_unchanged",
        len(protocol.get("attack_scenarios", [])) == 13
        and {
            str(item.get("name")) for item in protocol.get("attack_scenarios", [])
        }
        == EXPECTED_SCENARIOS,
        sorted(
            str(item.get("name"))
            for item in protocol.get("attack_scenarios", [])
        ),
        "13 个攻击场景及未见组合必须保持不变。",
    )
    baselines = set(protocol.get("method_freeze", {}).get("baselines", []))
    add(
        "baseline_scope_unchanged",
        baselines == EXPECTED_BASELINES
        and set(protocol.get("method_freeze", {}).get("trust_candidates", []))
        == {"trust_valid_f1", "trust_nonstuck_event_macro"},
        {
            "baselines": sorted(baselines),
            "trust_candidates": protocol.get("method_freeze", {}).get(
                "trust_candidates", []
            ),
        },
        "候选、11 个主基线和 OCSVM 敏感性档位必须保持冻结。",
    )
    add(
        "statistical_contract_unchanged",
        protocol.get("primary_far") == 0.0065
        and protocol.get("primary_endpoint") == "event_coverage"
        and protocol.get("bootstrap", {}).get("iterations") == 10000
        and protocol.get("window", {}).get("test_start_utc")
        == "2026-10-01T00:00:00Z"
        and protocol.get("window", {}).get("test_end_utc")
        == "2026-10-31T23:00:00Z",
        {
            "primary_far": protocol.get("primary_far"),
            "primary_endpoint": protocol.get("primary_endpoint"),
            "bootstrap_iterations": protocol.get("bootstrap", {}).get(
                "iterations"
            ),
            "window": protocol.get("window"),
        },
        "主终点、FAR、bootstrap 和官方窗口不得被事后修改。",
    )
    add(
        "entrypoints_present_and_runner_aligned",
        all(path.exists() for path in CORE_FILES.values())
        and "stage425_v9_locked_evaluation.py" in runner_text
        and "stage426_v9_locked_audit.py" in runner_text
        and "stage445_v9_statistical_acceptance_audit.py" in runner_text
        and "stage349_v6_locked_evaluation.py" not in runner_text
        and "stage351_v6_station_bh_audit.py" not in runner_text,
        {name: str(path) for name, path in CORE_FILES.items()},
        "续跑器必须只调用 V9 评价、结构审计和统计验收，且无 V6 fallback。",
    )
    add(
        "manifest_still_not_ready",
        manifest.get("status") == "NOT_READY"
        and manifest.get("observation_values_read") is False
        and manifest.get("historical_substitution") is False,
        {
            "status": manifest.get("status"),
            "observation_values_read": manifest.get("observation_values_read"),
            "historical_substitution": manifest.get("historical_substitution"),
            "providers": sorted(manifest.get("providers", {})),
        },
        "官方窗口未结束时必须保持 NOT_READY，不得以旧数据替代。",
    )
    raw_files = (
        sorted(str(path.relative_to(PACKAGE)) for path in RAW_DIR.rglob("*") if path.is_file())
        if RAW_DIR.exists()
        else []
    )
    add(
        "no_official_window_files_present",
        not raw_files and not any(path.exists() for path in PERFORMANCE_OUTPUTS),
        {
            "raw_official_file_count": len(raw_files),
            "performance_outputs_present": [
                str(path.relative_to(ROOT)) for path in PERFORMANCE_OUTPUTS if path.exists()
            ],
        },
        "窗口前不得生成官方观测文件或 V9 性能明细。",
    )
    add(
        "claim_boundary_and_no_performance_read",
        "controlled simulated-attack" in " ".join(
            str(value) for value in protocol.get("claim_boundary", [])
        ).lower()
        and manifest.get("failures"),
        {
            "claim_boundary": protocol.get("claim_boundary"),
            "manifest_failures": manifest.get("failures", []),
            "observation_values_read": False,
            "attack_labels_read": False,
            "performance_evaluated": False,
        },
        "当前产物只能支持受控模拟攻击准备状态，不能产生性能结论。",
    )

    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_PREWINDOW_FREEZE_INTEGRITY_AUDIT"
        if not failures
        else "FAIL_V9_PREWINDOW_FREEZE_INTEGRITY_AUDIT"
    )
    payload = {
        "stage": 449,
        "status": status,
        "protocol_id": protocol.get("protocol_id"),
        "checks": checks,
        "failed_checks": failures,
        "observation_values_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "historical_substitution": False,
        "hashes": {
            name: sha256(path) if path.exists() else None
            for name, path in CORE_FILES.items()
        },
        "claim_boundary": [
            "本阶段只验证 V9 冻结包的血缘和未就绪门禁，不产生性能结果。",
            "官方 2026-10 数据未到位前，不得运行评价或用旧窗口替代。",
            "所有攻击仍是透明受控模拟攻击，不是真实生产攻击日志。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage449 V9 窗口前冻结完整性审计",
                "",
                f"- 状态：`{status}`",
                f"- 协议：`{protocol.get('protocol_id')}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 本阶段只读取协议、manifest、源码和目录存在性；未读取观测值、攻击标签或旧 test 明细。",
                "- 官方 2026-10 数据仍未就绪，因此没有性能结果，也没有全面领先结论。",
                "",
                "## 失败检查",
                "",
                *([f"- `{name}`" for name in failures] or ["- 无。"]),
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": 449,
                "status": status,
                "failed_checks": failures,
                "observation_values_read": False,
                "performance_evaluated": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
