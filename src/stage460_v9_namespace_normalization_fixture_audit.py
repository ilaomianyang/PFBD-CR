from __future__ import annotations

"""Stage 460: executable fixture audit for V9 output namespace normalization."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
OUT_JSON = ROOT / "tables" / "stage460_v9_namespace_normalization_fixture_audit.json"
OUT_REPORT = ROOT / "reports" / "stage460_v9_namespace_normalization_fixture_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import stage425_v9_locked_evaluation as evaluator  # noqa: E402

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    detail_path = ROOT / "tables" / "stage460_fixture_detail.csv"
    station_path = ROOT / "tables" / "stage460_fixture_station.csv"
    scenario_path = ROOT / "tables" / "stage460_fixture_scenario.csv"
    summary_path = ROOT / "tables" / "stage460_fixture_summary.json"
    report_path = ROOT / "reports" / "stage460_fixture_report.md"

    pd.DataFrame(
        [
            {"method": "route_knn_arima", "value": 1},
            {"method": "one_class_svm", "value": 2},
            {"method": "trust_only", "value": 3},
        ]
    ).to_csv(detail_path, index=False)
    pd.DataFrame(
        [
            {"method": "route_knn_arima", "value": 1},
            {"method": "one_class_svm", "value": 2},
            {"method": "trust_only", "value": 3},
        ]
    ).to_csv(station_path, index=False)
    pd.DataFrame(
        [
            {"method": "route_knn_arima", "scenario": "strong_spike_long"},
            {"method": "one_class_svm", "scenario": "strong_spike_long"},
            {"method": "trust_only", "scenario": "strong_spike_long"},
        ]
    ).to_csv(scenario_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "summary": [
                    {"method": "route_knn_arima"},
                    {"method": "one_class_svm"},
                ],
                "comparisons": [
                    {
                        "candidate": "route_knn_arima",
                        "baseline": "one_class_svm",
                    }
                ],
                "selection_counts": {
                    "route_knn_arima": 1,
                    "one_class_svm": 1,
                    "trust_only": 1,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text("route_knn_arima one_class_svm\n", encoding="utf-8")

    original = {
        "V9_DETAIL": evaluator.V9_DETAIL,
        "V9_STATION": evaluator.V9_STATION,
        "V9_SCENARIO": evaluator.V9_SCENARIO,
        "V9_JSON": evaluator.V9_JSON,
        "V9_REPORT": evaluator.V9_REPORT,
    }
    try:
        evaluator.V9_DETAIL = detail_path
        evaluator.V9_STATION = station_path
        evaluator.V9_SCENARIO = scenario_path
        evaluator.V9_JSON = summary_path
        evaluator.V9_REPORT = report_path
        evaluator._normalize_v9_outputs(protocol)
    finally:
        for key, value in original.items():
            setattr(evaluator, key, value)

    detail = pd.read_csv(detail_path)
    station = pd.read_csv(station_path)
    scenario = pd.read_csv(scenario_path)
    payload_json = json.loads(summary_path.read_text(encoding="utf-8"))
    report_text = report_path.read_text(encoding="utf-8")

    expected = {"trust_only", "trust_arima", "one_class_svm_20k"}
    checks = [
        {
            "check": "detail_namespace_normalized",
            "status": "PASS" if set(detail["method"]) == expected else "FAIL",
            "value": sorted(detail["method"].astype(str).unique()),
        },
        {
            "check": "station_namespace_normalized",
            "status": "PASS" if set(station["method"]) == expected else "FAIL",
            "value": sorted(station["method"].astype(str).unique()),
        },
        {
            "check": "scenario_namespace_normalized",
            "status": "PASS" if set(scenario["method"]) == expected else "FAIL",
            "value": sorted(scenario["method"].astype(str).unique()),
        },
        {
            "check": "json_namespace_normalized",
            "status": "PASS"
            if {
                str(row["method"]) for row in payload_json["summary"]
            }
            == {"trust_arima", "one_class_svm_20k"}
            and payload_json["comparisons"][0]["candidate"] == "trust_arima"
            and payload_json["comparisons"][0]["baseline"] == "one_class_svm_20k"
            and set(payload_json["selection_counts"]) == expected
            else "FAIL",
            "value": payload_json,
        },
        {
            "check": "report_namespace_normalized",
            "status": "PASS"
            if "route_knn_arima" not in report_text
            and "one_class_svm" not in report_text.split()
            and "trust_arima" in report_text
            and "one_class_svm_20k" in report_text
            else "FAIL",
            "value": report_text.strip(),
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_NAMESPACE_NORMALIZATION_FIXTURE_AUDIT"
        if not failures
        else "FAIL_V9_NAMESPACE_NORMALIZATION_FIXTURE_AUDIT"
    )
    payload = {
        "stage": 460,
        "status": status,
        "checks": checks,
        "failed_checks": failures,
        "synthetic_fixture_only": True,
        "official_observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "claim_boundary": [
            "本阶段只执行输出命名空间转换夹具，不读取官方观测值或旧 test。",
            "夹具通过不代表任何性能优势或全面领先。",
        ],
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "evaluator": sha256(
                ROOT / "src" / "stage425_v9_locked_evaluation.py"
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
                "# Stage460 V9 输出命名空间转换夹具审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 直接调用 `_normalize_v9_outputs()` 验证四类输出。",
                "- 使用项目内明确命名的临时夹具，不代表官方性能结果。",
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
                "stage": 460,
                "status": status,
                "failed_checks": failures,
                "synthetic_fixture_only": True,
                "performance_evaluated": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
