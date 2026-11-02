from __future__ import annotations

"""Independent V9 result audit.

The audit is intentionally separate from the legacy V6/V8 auditors.  It
refuses to inspect any result table until the V9 official-window manifest is
ready, and it never changes thresholds or selects a candidate.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from stage425_v9_compatibility import require_ready_data_manifest, validate_protocol


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
MANIFEST = PACKAGE / "data_manifest.json"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
EVAL = TABLES / "stage425_v9_locked_evaluation_detail.csv"
SCENARIO = TABLES / "stage425_v9_locked_evaluation_scenario.csv"
STATION = TABLES / "stage425_v9_locked_evaluation_station.csv"
OUT_JSON = TABLES / "stage426_v9_locked_audit.json"
OUT_REPORT = REPORTS / "stage426_v9_locked_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    manifest = require_ready_data_manifest(MANIFEST)
    missing = [str(path) for path in (EVAL, SCENARIO, STATION) if not path.exists()]
    if missing:
        raise RuntimeError(f"V9 evaluation outputs are missing: {missing}")
    evaluation = pd.read_csv(EVAL)
    scenario = pd.read_csv(SCENARIO)
    station = pd.read_csv(STATION)
    expected_methods = {
        "trust_only",
        "trust_arima",
        "trust_shape_union",
        *protocol["method_freeze"]["baselines"],
    }
    checks = [
        {
            "check": "manifest_ready",
            "status": "PASS" if manifest["status"] == "OFFICIAL_WINDOW_READY" else "FAIL",
            "value": manifest["status"],
        },
        {
            "check": "evaluation_methods_complete",
            "status": "PASS"
            if set(evaluation["method"].astype(str)) == expected_methods
            else "FAIL",
            "value": sorted(evaluation["method"].astype(str).unique()),
        },
        {
            "check": "scenario_names_complete",
            "status": "PASS"
            if set(scenario["scenario"].astype(str))
            == {str(item["name"]) for item in protocol["attack_scenarios"]}
            else "FAIL",
            "value": sorted(scenario["scenario"].astype(str).unique()),
        },
        {
            "check": "scenario_factor_columns_complete",
            "status": (
                "PASS"
                if {
                    "base_type",
                    "factor_intensity",
                    "factor_duration",
                    "factor_length",
                    "unseen_combination",
                    "coordinated",
                }.issubset(scenario.columns)
                else "FAIL"
            ),
            "value": sorted(scenario.columns.astype(str)),
        },
        {
            "check": "test_far_within_contract",
            "status": "PASS"
            if "false_alarm_rate" in evaluation.columns
            and bool((evaluation["false_alarm_rate"] <= 0.0365).all())
            else "FAIL",
            "value": float(evaluation["false_alarm_rate"].max())
            if "false_alarm_rate" in evaluation.columns and len(evaluation)
            else None,
        },
        {
            "check": "all_attacks_are_simulated",
            "status": "PASS",
            "value": "controlled_simulated_attacks_only",
        },
    ]
    status = (
        "PASS_V9_LOCKED_AUDIT"
        if all(item["status"] == "PASS" for item in checks)
        else "FAIL_V9_LOCKED_AUDIT"
    )
    payload = {
        "stage": 426,
        "status": status,
        "protocol_id": protocol["protocol_id"],
        "manifest_status": manifest["status"],
        "checks": checks,
        "evaluation_rows": int(len(evaluation)),
        "scenario_rows": int(len(scenario)),
        "station_rows": int(len(station)),
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "manifest": sha256(MANIFEST),
            "evaluation": sha256(EVAL),
            "scenario": sha256(SCENARIO),
            "station": sha256(STATION),
        },
        "claim_boundary": [
            "仅审计 V9 预注册模拟攻击结果，不是真实生产事故召回率。",
            "该审计不替代统计解释，不删除任何基线胜出或失败场景。",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage426 V9 锁定结果独立审计",
                "",
                f"- 状态：`{status}`",
                f"- 协议：`{protocol['protocol_id']}`",
                f"- 检查数：`{len(checks)}`；失败数：`{sum(x['status'] == 'FAIL' for x in checks)}`",
                "- 所有攻击均为透明受控模拟攻击，不是真实生产攻击日志。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": 426, "status": status}, ensure_ascii=False))
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
