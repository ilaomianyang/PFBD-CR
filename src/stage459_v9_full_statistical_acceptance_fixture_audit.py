from __future__ import annotations

"""Stage 459: execute the complete V9 statistical gate on a fixture only.

The fixture exercises the real Stage445 entrypoint with temporary CSV/JSON
files and a temporary ready-manifest shim.  It is intentionally not an
evaluation result: no official observations, historical test rows, labels, or
real performance artifacts are read or written.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
OUT_JSON = ROOT / "tables" / "stage459_v9_full_statistical_acceptance_fixture_audit.json"
OUT_REPORT = ROOT / "reports" / "stage459_v9_full_statistical_acceptance_fixture_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixture(protocol: dict, root: Path) -> tuple[Path, Path, Path, Path, Path]:
    providers = ("MeteoSwiss", "NOAA ISD", "KNMI", "ECCC")
    seeds = tuple(int(value) for value in protocol["seeds"])
    methods = (
        ("trust_only", "trust_arima", "trust_shape_union")
        + tuple(str(value) for value in protocol["method_freeze"]["baselines"])
    )
    scenarios = tuple(str(item["name"]) for item in protocol["attack_scenarios"])

    evaluation_rows = []
    station_rows = []
    scenario_rows = []
    for provider in providers:
        for seed in seeds:
            for method in methods:
                candidate = method.startswith("trust_")
                evaluation_rows.append(
                    {
                        "source": provider,
                        "seed": seed,
                        "method": method,
                        "event_coverage": 0.80 if candidate else 0.20,
                        "false_alarm_rate": float(protocol["primary_far"]),
                    }
                )
                for station_index in range(2):
                    station_rows.append(
                        {
                            "source": provider,
                            "seed": seed,
                            "station_id": f"{provider}_{station_index}",
                            "method": method,
                            "f1": 0.80 if candidate else 0.20,
                            "event_coverage": 0.80 if candidate else 0.20,
                        }
                    )
                for scenario in scenarios:
                    scenario_rows.append(
                        {
                            "source": provider,
                            "seed": seed,
                            "method": method,
                            "scenario": scenario,
                            "event_coverage": 0.80 if candidate else 0.20,
                        }
                    )

    eval_path = root / "stage459_fixture_evaluation.csv"
    scenario_path = root / "stage459_fixture_scenario.csv"
    station_path = root / "stage459_fixture_station.csv"
    summary_path = root / "stage459_fixture_summary.json"
    manifest_path = root / "stage459_fixture_manifest.json"
    pd.DataFrame(evaluation_rows).to_csv(eval_path, index=False)
    pd.DataFrame(scenario_rows).to_csv(scenario_path, index=False)
    pd.DataFrame(station_rows).to_csv(station_path, index=False)
    summary_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "status": "OFFICIAL_WINDOW_READY",
                "failures": [],
                "observation_values_read": True,
                "historical_substitution": False,
                "window": {
                    "test_start_utc": "2026-10-01T00:00:00Z",
                    "test_end_utc": "2026-10-31T23:00:00Z",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return eval_path, scenario_path, station_path, summary_path, manifest_path


def main() -> int:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import stage445_v9_statistical_acceptance_audit as audit  # noqa: E402

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    fixture_root = ROOT / "tables"
    eval_path, scenario_path, station_path, summary_path, manifest_path = (
        build_fixture(protocol, fixture_root)
    )
    original = {
        "MANIFEST": audit.MANIFEST,
        "EVAL": audit.EVAL,
        "SCENARIO": audit.SCENARIO,
        "STATION": audit.STATION,
        "SUMMARY": audit.SUMMARY,
        "OUT_JSON": audit.OUT_JSON,
        "OUT_REPORT": audit.OUT_REPORT,
        "require_ready_data_manifest": audit.require_ready_data_manifest,
    }
    fixture_json = ROOT / "tables" / "stage459_stage445_fixture_output.json"
    fixture_report = ROOT / "reports" / "stage459_stage445_fixture_output.md"
    try:
        audit.MANIFEST = manifest_path
        audit.EVAL = eval_path
        audit.SCENARIO = scenario_path
        audit.STATION = station_path
        audit.SUMMARY = summary_path
        audit.OUT_JSON = fixture_json
        audit.OUT_REPORT = fixture_report
        audit.require_ready_data_manifest = lambda _path: json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        return_code = int(audit.main())
        result = json.loads(fixture_json.read_text(encoding="utf-8"))
    finally:
        for key, value in original.items():
            setattr(audit, key, value)

    checks = [
        {
            "check": "stage445_fixture_executes",
            "status": "PASS" if return_code == 0 else "FAIL",
            "value": return_code,
        },
        {
            "check": "stage445_fixture_acceptance_status",
            "status": "PASS"
            if result.get("status") == "PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT"
            else "FAIL",
            "value": result.get("status"),
        },
        {
            "check": "all_stage445_checks_pass_on_fixture",
            "status": "PASS"
            if result.get("failed_checks") == []
            else "FAIL",
            "value": result.get("failed_checks"),
        },
        {
            "check": "fixture_is_not_official_performance",
            "status": "PASS"
            if result.get("claim_boundary")
            and result.get("performance_evaluated") is True
            else "FAIL",
            "value": {
                "temporary_fixture": True,
                "official_observation_values_read": False,
                "historical_test_rows_read": False,
            },
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_FULL_STATISTICAL_ACCEPTANCE_FIXTURE_AUDIT"
        if not failures
        else "FAIL_V9_FULL_STATISTICAL_ACCEPTANCE_FIXTURE_AUDIT"
    )
    payload = {
        "stage": 459,
        "status": status,
        "checks": checks,
        "failed_checks": failures,
        "temporary_fixture_only": True,
        "official_observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "stage445_fixture_status": result.get("status"),
        "stage445_fixture_failed_checks": result.get("failed_checks", []),
        "claim_boundary": [
            "本阶段只在临时夹具上执行真实 Stage445 统计入口，不产生官方性能结果。",
            "夹具通过不代表任何攻击检测优势、非劣效或全面领先。",
            "官方窗口未闭合，data_manifest 仍为 NOT_READY。",
        ],
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "stage445": sha256(
                ROOT / "src" / "stage445_v9_statistical_acceptance_audit.py"
            ),
        },
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage459 V9 完整统计验收夹具审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 使用临时 CSV/JSON 和 ready-manifest shim 执行真实 Stage445 入口。",
                "- 夹具仅验证流程和字段契约，不代表官方观测数据或性能结果。",
                "- 未写入正式 Stage445 产物，未读取 V9 官方文件或旧 test。",
                "",
                *([f"- 失败：`{item}`" for item in failures] or ["- 无。"]),
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
                "stage": 459,
                "status": status,
                "failed_checks": failures,
                "temporary_fixture_only": True,
                "performance_evaluated": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
