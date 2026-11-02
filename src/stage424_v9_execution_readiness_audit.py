from __future__ import annotations

"""Stage 424: audit whether the V9 protocol can be executed safely.

This stage is deliberately read-only with respect to observation values,
attack labels, model fitting, threshold calibration, and test evaluation.  It
checks the frozen V9 contract and the legacy evaluation entrypoint for
compatibility hazards before any new official window is accessed.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
OUT_JSON = TABLES / "stage424_v9_execution_readiness_audit.json"
OUT_REPORT = REPORTS / "stage424_v9_execution_readiness_audit.md"

LEGACY_ENTRYPOINT = SRC / "stage349_v6_locked_evaluation.py"
LEGACY_BASE = SRC / "stage340_v4_route_locked_evaluation.py"
SCENARIO_HELPER = SRC / "stage326_simulated_attack_v2_evaluation.py"
V9_COMPATIBILITY = SRC / "stage425_v9_compatibility.py"
V9_RUNNER = SRC / "stage425_v9_locked_evaluation.py"
V9_DOWNLOAD_TEMPLATE = SRC / "stage425_v9_2026_official_window_download_template.py"
V9_AUDITOR = SRC / "stage426_v9_locked_audit.py"
V9_SOURCE_PLAN = SRC / "stage427_v9_official_source_plan_audit.py"
V9_DATA_MANIFEST = PACKAGE / "data_manifest.json"
DOWNLOAD_SCRIPTS = {
    "MeteoSwiss": [
        SRC / "stage215_swiss_official_download.py",
        SRC / "stage216_swiss_normalized_archive_audit.py",
    ],
    "NOAA ISD": [
        SRC / "stage82_noaa_extended_ingest.py",
        SRC / "stage376_noaa_2026_download_audit.py",
    ],
    "KNMI": [
        SRC / "stage225_knmi_official_download.py",
        SRC / "stage226_knmi_normalization_and_missingness_audit.py",
    ],
    "ECCC": [
        SRC / "stage235_eccc_official_download.py",
        SRC / "stage236_eccc_normalized_archive_audit.py",
    ],
}

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
REQUIRED_SCENARIO_FIELDS = {
    "name",
    "base_type",
    "length",
    "factor_intensity",
    "factor_duration",
    "factor_length",
}
REQUIRED_PROVIDER_FIELDS = {
    "normalized",
    "protocol",
    "split",
    "normalized_sha256",
    "protocol_sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(value: str) -> Path:
    """Resolve protocol-relative paths without reading parquet contents."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def check(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    value: object,
    detail: str,
) -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "value": value,
            "detail": detail,
        }
    )


def main() -> int:
    checks: list[dict[str, object]] = []
    protocol_exists = PROTOCOL.exists()
    check(
        checks,
        "v9_protocol_present",
        protocol_exists,
        str(PROTOCOL),
        "冻结协议文件必须存在。",
    )
    if not protocol_exists:
        raise FileNotFoundError(PROTOCOL)
    payload = load_json(PROTOCOL)

    check(
        checks,
        "protocol_status_frozen",
        payload.get("status")
        == "CONFIRMATORY_PROTOCOL_FROZEN_PENDING_NEW_OFFICIAL_WINDOW",
        payload.get("status"),
        "协议必须仍处于冻结但未评价状态。",
    )
    check(
        checks,
        "provider_scope_exact",
        set(payload.get("providers", {})) == EXPECTED_PROVIDERS
        and payload.get("provider_count") == len(EXPECTED_PROVIDERS),
        sorted(payload.get("providers", {})),
        "V9 只允许四个冻结主 provider。",
    )
    check(
        checks,
        "scenario_scope_exact",
        {str(x.get("name")) for x in payload.get("attack_scenarios", [])}
        == EXPECTED_SCENARIOS
        and len(payload.get("attack_scenarios", [])) == 13,
        sorted(str(x.get("name")) for x in payload.get("attack_scenarios", [])),
        "13 个场景必须完整且不能静默增删。",
    )

    scenarios = {
        str(item.get("name")): item for item in payload.get("attack_scenarios", [])
    }
    missing_scenario_fields = {
        name: sorted(REQUIRED_SCENARIO_FIELDS - set(spec))
        for name, spec in scenarios.items()
        if REQUIRED_SCENARIO_FIELDS - set(spec)
    }
    check(
        checks,
        "scenario_fields_complete",
        not missing_scenario_fields,
        missing_scenario_fields,
        "每个场景必须有类型、强度、持续时间和长度因子。",
    )
    lag_ok = (
        scenarios.get("short_lag", {}).get("base_type") == "time_shift"
        and scenarios.get("short_lag", {}).get("lag_rows") == 1
        and scenarios.get("long_delay", {}).get("base_type") == "time_shift"
        and scenarios.get("long_delay", {}).get("lag_rows") == 8
    )
    check(
        checks,
        "scenario_specific_lag_values",
        lag_ok,
        {
            "short_lag": scenarios.get("short_lag", {}).get("lag_rows"),
            "long_delay": scenarios.get("long_delay", {}).get("lag_rows"),
        },
        "延迟必须读取各自场景的 lag_rows，不能用固定延迟。",
    )
    coordinated = scenarios.get("coordinated_drift_unseen", {})
    check(
        checks,
        "unseen_coordinated_drift_declared",
        coordinated.get("coordinated") is True
        and coordinated.get("unseen_combination") is True
        and coordinated.get("station_count") == 2,
        coordinated,
        "未见组合必须是双站点协调漂移且未参与模型选择。",
    )

    provider_checks: list[dict[str, object]] = []
    unresolved_download_templates: list[str] = []
    for provider, meta in payload.get("providers", {}).items():
        missing_fields = sorted(REQUIRED_PROVIDER_FIELDS - set(meta))
        normalized = resolve_repo_path(str(meta.get("normalized", "")))
        provider_protocol = resolve_repo_path(str(meta.get("protocol", "")))
        normalized_present = normalized.exists()
        protocol_present = provider_protocol.exists()
        normalized_hash_ok = (
            normalized_present
            and sha256(normalized).lower() == str(meta.get("normalized_sha256", "")).lower()
        )
        protocol_hash_ok = (
            protocol_present
            and sha256(provider_protocol).lower()
            == str(meta.get("protocol_sha256", "")).lower()
        )
        source_scripts = [path for path in DOWNLOAD_SCRIPTS.get(provider, []) if path.exists()]
        has_future_window_template = any(
            "2026" in path.name or "temporal" in path.name.lower()
            for path in source_scripts
        )
        if not has_future_window_template:
            unresolved_download_templates.append(provider)
        provider_checks.append(
            {
                "provider": provider,
                "missing_fields": missing_fields,
                "normalized_path": str(normalized),
                "protocol_path": str(provider_protocol),
                "normalized_present": normalized_present,
                "protocol_present": protocol_present,
                "normalized_hash_matches_protocol": normalized_hash_ok,
                "protocol_hash_matches_protocol": protocol_hash_ok,
                "download_scripts_present": [str(path) for path in source_scripts],
                "future_window_download_template_present": has_future_window_template,
            }
        )
    check(
        checks,
        "provider_metadata_and_hashes_audited",
        len(provider_checks) == 4
        and all(
            not row["missing_fields"]
            and row["normalized_present"]
            and row["protocol_present"]
            and row["normalized_hash_matches_protocol"]
            and row["protocol_hash_matches_protocol"]
            for row in provider_checks
        ),
        provider_checks,
        "只核对文件存在性与 SHA256，不读取 parquet 观测值。",
    )

    legacy_text = LEGACY_ENTRYPOINT.read_text(encoding="utf-8") if LEGACY_ENTRYPOINT.exists() else ""
    base_text = LEGACY_BASE.read_text(encoding="utf-8") if LEGACY_BASE.exists() else ""
    helper_text = SCENARIO_HELPER.read_text(encoding="utf-8") if SCENARIO_HELPER.exists() else ""
    legacy_bound_to_v6 = "simulated_attack_v6" in legacy_text
    legacy_output_v6 = "stage349_v6_locked_evaluation" in legacy_text
    v9_compatibility_present = V9_COMPATIBILITY.exists()
    v9_runner_present = V9_RUNNER.exists()
    v9_download_template_present = V9_DOWNLOAD_TEMPLATE.exists()
    v9_accepts_names = (
        v9_compatibility_present
        and all(name in V9_COMPATIBILITY.read_text(encoding="utf-8") for name in EXPECTED_SCENARIOS)
    )
    check(
        checks,
        "legacy_entrypoint_isolated_from_v9",
        legacy_bound_to_v6 and legacy_output_v6,
        {
            "legacy_bound_to_v6": legacy_bound_to_v6,
            "legacy_output_v6": legacy_output_v6,
        },
        "旧入口必须被识别为 V6 入口，不能直接拿来跑 V9。",
    )
    check(
        checks,
        "v9_scenario_names_supported_by_legacy_generator",
        v9_accepts_names and v9_runner_present,
        {
            "dedicated_v9_compatibility_present": v9_compatibility_present,
            "dedicated_v9_runner_present": v9_runner_present,
            "dedicated_v9_name_coverage": v9_accepts_names,
            "legacy_generator_name_coverage": all(
                name in helper_text for name in EXPECTED_SCENARIOS
            ),
            "legacy_base_name_coverage": all(
                name in base_text for name in EXPECTED_SCENARIOS
            ),
        },
        "V9 场景名必须有专用生成器映射和隔离入口；旧映射不完整不构成通过理由。",
    )
    check(
        checks,
        "v9_download_template_available",
        v9_download_template_present,
        {
            "dedicated_template_present": v9_download_template_present,
            "historical_provider_template_gaps": unresolved_download_templates,
        },
        "官方新窗口必须有专用双边界过滤和覆盖率审计入口；没有时只能保持 NOT_EVALUATED。",
    )
    check(
        checks,
        "v9_audit_entrypoint_isolated",
        V9_AUDITOR.exists(),
        str(V9_AUDITOR),
        "V9 结果必须使用独立审计器，不能回落到 V6/V8 审计器。",
    )
    check(
        checks,
        "v9_official_source_plan_present",
        V9_SOURCE_PLAN.exists(),
        str(V9_SOURCE_PLAN),
        "四个 provider 必须有独立的官方 URL/字段计划。",
    )
    check(
        checks,
        "v9_official_window_data_manifest_ready",
        V9_DATA_MANIFEST.exists()
        and json.loads(V9_DATA_MANIFEST.read_text(encoding="utf-8")).get("status")
        == "OFFICIAL_WINDOW_READY",
        (
            json.loads(V9_DATA_MANIFEST.read_text(encoding="utf-8"))
            if V9_DATA_MANIFEST.exists()
            else {"status": "MISSING"}
        ),
        "只有新官方窗口四 provider 数据门禁通过后才能运行性能评价。",
    )

    required_protocol_fields = {
        "primary_far": payload.get("primary_far") == 0.0065,
        "primary_endpoint": payload.get("primary_endpoint") == "event_coverage",
        "bootstrap": payload.get("bootstrap", {}).get("iterations") == 10000,
        "test_window": payload.get("window", {}).get("test_start_utc")
        == "2026-10-01T00:00:00Z"
        and payload.get("window", {}).get("test_end_utc")
        == "2026-10-31T23:00:00Z",
        "test_evaluation_only": "test labels are evaluation-only"
        in str(payload.get("attack_generation_rules", {}).get("test", "")),
    }
    check(
        checks,
        "statistical_contract_frozen",
        all(required_protocol_fields.values()),
        required_protocol_fields,
        "主 FAR、主终点、bootstrap、测试窗和 test-only 边界必须冻结。",
    )

    hash_map = {
        "protocol_sha256": sha256(PROTOCOL),
        "legacy_entrypoint_sha256": sha256(LEGACY_ENTRYPOINT)
        if LEGACY_ENTRYPOINT.exists()
        else None,
        "legacy_base_sha256": sha256(LEGACY_BASE) if LEGACY_BASE.exists() else None,
        "scenario_helper_sha256": sha256(SCENARIO_HELPER)
        if SCENARIO_HELPER.exists()
        else None,
        "v9_compatibility_sha256": sha256(V9_COMPATIBILITY)
        if V9_COMPATIBILITY.exists()
        else None,
        "v9_runner_sha256": sha256(V9_RUNNER) if V9_RUNNER.exists() else None,
        "v9_download_template_sha256": sha256(V9_DOWNLOAD_TEMPLATE)
        if V9_DOWNLOAD_TEMPLATE.exists()
        else None,
        "v9_auditor_sha256": sha256(V9_AUDITOR) if V9_AUDITOR.exists() else None,
        "v9_source_plan_sha256": sha256(V9_SOURCE_PLAN)
        if V9_SOURCE_PLAN.exists()
        else None,
    }
    failure_names = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_EXECUTION_READY"
        if not failure_names
        else "BLOCKED_V9_EXECUTION_NOT_READY"
    )
    result = {
        "stage": 424,
        "status": status,
        "protocol_id": payload.get("protocol_id"),
        "observation_values_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "old_test_rows_read": False,
        "checks": checks,
        "failed_checks": failure_names,
        "provider_checks": provider_checks,
        "unresolved_download_templates": unresolved_download_templates,
        "hashes": hash_map,
        "blocking_rules": [
            "未通过时不得运行性能评价，不得用旧年份或旧 test 替代。",
            "旧 V6 入口不能直接执行 V9。",
            "V9 场景必须由专用、可审计的生成器支持。",
            "官方新窗口不可用时保持 NOT_EVALUATED。",
        ],
        "claim_boundary": [
            "本阶段只审计执行就绪性，不产生检测性能结果。",
            "所有未来攻击仍是透明受控模拟攻击，不是真实生产攻击日志。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage424 V9 执行就绪与协议兼容性审计",
        "",
        f"- 状态：`{status}`",
        f"- 协议：`{payload.get('protocol_id')}`",
        "- 本阶段只读协议、源码和文件哈希；未读取观测值、攻击标签或旧 test 明细。",
        f"- 失败检查数：`{len(failure_names)}`",
        "",
        "## 失败/阻断检查",
        "",
    ]
    if failure_names:
        lines.extend(f"- `{name}`" for name in failure_names)
    else:
        lines.append("- 无。")
    lines += [
        "",
        "## 科学边界",
        "",
        "- 本阶段不是性能实验，不能关闭 70% 场景、未见组合泛化或全面领先门槛。",
        "- 若官方 2026-10 数据不可访问，必须保持 `NOT_EVALUATED`，不得替换旧年份。",
        "- 所有攻击均为透明受控模拟攻击，不是真实生产攻击日志。",
        "",
        f"- JSON：`{OUT_JSON}`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": 424,
                "status": status,
                "failed_checks": failure_names,
                "observation_values_read": False,
                "performance_evaluated": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
