from __future__ import annotations

"""Stage 233: audit the valid-attack correction and direction reversal.

This stage is read-only.  It records that the earlier transfer audits
used a valid reference arm that did not contain an injected attack
signal, and that correcting the valid/test attack construction reversed
the comparison direction for MeteoSwiss, NOAA ISD, and KNMI.

It does not retrain, reselect, recalibrate, or use target-domain test
labels for any choice.  The audit exists so downstream paper, table, and
figure generation cannot silently reuse the invalidated direction.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import sha256_file  # noqa: E402

PREFIX = "stage233_valid_attack_correction_audit"

SOURCES = {
    "stage219": TABLES / "stage219_swiss_dwd_protection_transfer.json",
    "stage220": TABLES / "stage220_swiss_dwd_transfer_uncertainty.json",
    "stage221": TABLES / "stage221_swiss_dwd_transfer_operating_curve.json",
    "stage222": TABLES / "stage222_noaa_dwd_protection_transfer.json",
    "stage223": TABLES / "stage223_noaa_transfer_uncertainty.json",
    "stage228": TABLES / "stage228_knmi_dwd_protection_transfer.json",
    "stage229": TABLES / "stage229_knmi_transfer_uncertainty.json",
    "stage230": TABLES / "stage230_cross_provider_evidence_synthesis.json",
    "stage231": TABLES / "stage231_cross_provider_robustness.json",
    "stage232": TABLES / "stage232_attack_type_robustness.json",
    "stage238": TABLES / "stage238_eccc_dwd_protection_transfer.json",
    "stage239": TABLES / "stage239_eccc_transfer_uncertainty.json",
}

SOURCE_SCRIPTS = {
    "stage219": SRC / "stage219_swiss_dwd_protection_transfer.py",
    "stage220": SRC / "stage220_swiss_dwd_transfer_uncertainty.py",
    "stage221": SRC / "stage221_swiss_dwd_transfer_operating_curve.py",
    "stage222": SRC / "stage222_noaa_dwd_protection_transfer.py",
    "stage223": SRC / "stage223_noaa_transfer_uncertainty.py",
    "stage228": SRC / "stage228_knmi_dwd_protection_transfer.py",
    "stage229": SRC / "stage229_knmi_transfer_uncertainty.py",
    "stage230": SRC / "stage230_cross_provider_evidence_synthesis.py",
    "stage231": SRC / "stage231_cross_provider_robustness.py",
    "stage232": SRC / "stage232_attack_type_robustness.py",
    "stage238": SRC / "stage238_eccc_dwd_protection_transfer.py",
    "stage239": SRC / "stage239_eccc_transfer_uncertainty.py",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    payloads = {name: load_json(path) for name, path in SOURCES.items()}

    stage219 = payloads["stage219"]
    stage219_aggregate = {
        row["arm"]: row for row in stage219["aggregate"]
    }
    stage222 = payloads["stage222"]
    stage222_aggregate = {
        row["arm"]: row for row in stage222["aggregate"]
    }
    stage228 = payloads["stage228"]
    stage228_aggregate = {
        row["arm"]: row for row in stage228["aggregate"]
    }
    stage238 = payloads["stage238"]
    stage238_aggregate = {
        row["arm"]: row for row in stage238["aggregate"]
    }

    target_rows = [
        {
            "target_source": "MeteoSwiss",
            "station_count": int(payloads["stage220"]["station_count"]),
            "comparison": payloads["stage220"]["comparison"],
            "selected_minus_recalibrated_f1": float(
                payloads["stage220"]["primary_f1"]["mean_difference"]
            ),
            "selected_minus_recalibrated_f1_ci95_low": float(
                payloads["stage220"]["primary_f1"]["ci95_low"]
            ),
            "selected_minus_recalibrated_event_coverage": float(
                payloads["stage220"]["primary_event_coverage"][
                    "mean_difference"
                ]
            ),
            "selected_minus_recalibrated_event_ci95_low": float(
                payloads["stage220"]["primary_event_coverage"]["ci95_low"]
            ),
            "direct_threshold_transfer_f1": float(
                stage219_aggregate["dwd_threshold_transfer"]["f1_mean"]
            ),
            "direct_threshold_transfer_event_coverage": float(
                stage219_aggregate["dwd_threshold_transfer"][
                    "event_coverage_mean"
                ]
            ),
        },
        {
            "target_source": "NOAA ISD",
            "station_count": int(payloads["stage223"]["station_count"]),
            "comparison": payloads["stage223"]["comparison"],
            "selected_minus_recalibrated_f1": float(
                payloads["stage223"]["primary_f1"]["mean_difference"]
            ),
            "selected_minus_recalibrated_f1_ci95_low": float(
                payloads["stage223"]["primary_f1"]["ci95_low"]
            ),
            "selected_minus_recalibrated_event_coverage": float(
                payloads["stage223"]["primary_event_coverage"][
                    "mean_difference"
                ]
            ),
            "selected_minus_recalibrated_event_ci95_low": float(
                payloads["stage223"]["primary_event_coverage"]["ci95_low"]
            ),
            "direct_threshold_transfer_f1": float(
                stage222_aggregate["dwd_threshold_transfer"]["f1_mean"]
            ),
            "direct_threshold_transfer_event_coverage": float(
                stage222_aggregate["dwd_threshold_transfer"][
                    "event_coverage_mean"
                ]
            ),
        },
        {
            "target_source": "KNMI",
            "station_count": int(payloads["stage229"]["station_count"]),
            "comparison": payloads["stage229"]["comparison"],
            "selected_minus_recalibrated_f1": float(
                payloads["stage229"]["primary_f1"]["mean_difference"]
            ),
            "selected_minus_recalibrated_f1_ci95_low": float(
                payloads["stage229"]["primary_f1"]["ci95_low"]
            ),
            "selected_minus_recalibrated_event_coverage": float(
                payloads["stage229"]["primary_event_coverage"][
                    "mean_difference"
                ]
            ),
            "selected_minus_recalibrated_event_ci95_low": float(
                payloads["stage229"]["primary_event_coverage"]["ci95_low"]
            ),
            "direct_threshold_transfer_f1": float(
                stage228_aggregate["dwd_threshold_transfer"]["f1_mean"]
            ),
            "direct_threshold_transfer_event_coverage": float(
                stage228_aggregate["dwd_threshold_transfer"][
                    "event_coverage_mean"
                ]
            ),
        },
        {
            "target_source": "ECCC",
            "station_count": int(payloads["stage239"]["station_count"]),
            "comparison": payloads["stage239"]["comparison"],
            "selected_minus_recalibrated_f1": float(
                payloads["stage239"]["primary_f1"]["mean_difference"]
            ),
            "selected_minus_recalibrated_f1_ci95_low": float(
                payloads["stage239"]["primary_f1"]["ci95_low"]
            ),
            "selected_minus_recalibrated_event_coverage": float(
                payloads["stage239"]["primary_event_coverage"][
                    "mean_difference"
                ]
            ),
            "selected_minus_recalibrated_event_ci95_low": float(
                payloads["stage239"]["primary_event_coverage"]["ci95_low"]
            ),
            "direct_threshold_transfer_f1": float(
                stage238_aggregate["dwd_threshold_transfer"]["f1_mean"]
            ),
            "direct_threshold_transfer_event_coverage": float(
                stage238_aggregate["dwd_threshold_transfer"][
                    "event_coverage_mean"
                ]
            ),
        },
    ]

    checks = [
        {
            "check": "all_target_comparisons_use_selected_minus_recalibrated",
            "status": (
                "PASS"
                if all(
                    row["comparison"].startswith(
                        (
                            "swiss_valid_selected_minus_"
                            "swiss_valid_recalibrated"
                        )
                    )
                    or row["comparison"].startswith(
                        (
                            "noaa_valid_selected_minus_"
                            "noaa_valid_recalibrated"
                        )
                    )
                    or row["comparison"].startswith(
                        (
                            "knmi_valid_selected_minus_"
                            "knmi_valid_recalibrated"
                        )
                    )
                    or row["comparison"].startswith(
                        (
                            "eccc_valid_selected_minus_"
                            "eccc_valid_recalibrated"
                        )
                    )
                    for row in target_rows
                )
                else "FAIL"
            ),
            "value": [row["comparison"] for row in target_rows],
        },
        {
            "check": "all_selected_differences_positive",
            "status": (
                "PASS"
                if all(
                    row["selected_minus_recalibrated_f1"] > 0.0
                    and row["selected_minus_recalibrated_event_coverage"]
                    > 0.0
                    for row in target_rows
                )
                else "FAIL"
            ),
            "value": {
                row["target_source"]: {
                    "f1": row["selected_minus_recalibrated_f1"],
                    "event_coverage": row[
                        "selected_minus_recalibrated_event_coverage"
                    ],
                }
                for row in target_rows
            },
        },
        {
            "check": "direct_dwd_threshold_transfer_failure_retained",
            "status": (
                "PASS"
                if all(
                    row["direct_threshold_transfer_f1"] == 0.0
                    and row["direct_threshold_transfer_event_coverage"]
                    == 0.0
                    for row in target_rows
                )
                else "FAIL"
            ),
            "value": {
                row["target_source"]: {
                    "f1": row["direct_threshold_transfer_f1"],
                    "event_coverage": row[
                        "direct_threshold_transfer_event_coverage"
                    ],
                }
                for row in target_rows
            },
        },
        {
            "check": "cross_provider_synthesis_uses_corrected_direction",
            "status": (
                "PASS"
                if int(
                    payloads["stage230"][
                        "target_valid_selected_f1_advantage_count"
                    ]
                )
                == 4
                and int(
                    payloads["stage230"][
                        "target_valid_selected_event_advantage_count"
                    ]
                )
                == 4
                else "FAIL"
            ),
            "value": {
                "f1_advantage_count": int(
                    payloads["stage230"][
                        "target_valid_selected_f1_advantage_count"
                    ]
                ),
                "event_advantage_count": int(
                    payloads["stage230"][
                        "target_valid_selected_event_advantage_count"
                    ]
                ),
            },
        },
        {
            "check": "stuck_concentration_boundary_retained",
            "status": (
                "PASS"
                if any(
                    "stuck" in str(item)
                    and (
                        "移除" in str(item)
                        or "负" in str(item)
                        or "集中" in str(item)
                    )
                    for item in payloads["stage232"].get("limitations", [])
                )
                else "FAIL"
            ),
            "value": payloads["stage232"].get("limitations", []),
        },
    ]

    failed = [item for item in checks if item["status"] == "FAIL"]
    payload = {
        "status": (
            "PASS_PIPELINE_REVIEW_REQUIRED"
            if not failed
            else "FAIL"
        ),
        "protocol": (
            "阶段233只读取阶段219-239已冻结产物，记录旧valid参考臂未注入攻击"
            "导致的结论失效，以及修正valid/test分离攻击后方向反转。旧方向"
            "“目标域valid重校准迁移优于本地选择”不得再用于论文、表格、图"
            "或后续汇总；修正方向为“目标域valid本地选择优于重校准迁移”。"
            "本阶段不训练、不重选、不重校准、不使用目标域test标签做任何选"
            "择；所有攻击均为透明受控注入，不是真实攻击日志。"
        ),
        "old_claim_status": "INVALIDATED",
        "old_claim": (
            "DWD特征/窗口/权重迁移到目标域后，经目标域valid重校准阈值，"
            "优于目标域valid本地选择。"
        ),
        "corrected_claim": (
            "修正valid参考臂攻击注入后，目标域valid本地选择优于DWD特征/"
            "窗口/权重迁移加目标域valid重校准阈值。DWD绝对阈值直接迁移"
            "仍完全失败。"
        ),
        "affected_stages": sorted(SOURCES),
        "source_sha256": {
            name: sha256_file(path) for name, path in SOURCES.items()
        },
        "source_script_sha256": {
            name: sha256_file(path) for name, path in SOURCE_SCRIPTS.items()
        },
        "target_results": target_rows,
        "checks": checks,
        "failed_checks": failed,
        "limitations": [
            "只有MeteoSwiss、NOAA ISD、KNMI和ECCC四个迁移目标，不能写成多机构普适领先。",
            "NOAA的事件覆盖优势不显著；KNMI的F1 BH q=0.0625，接近但不低于0.05。",
            "跨机构优势主要集中在stuck攻击类型；移除stuck后其余类型方向转为负。",
            "所有攻击均为透明受控注入，不是真实攻击日志。",
        ],
    }

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(target_rows).to_csv(
        TABLES / f"{PREFIX}_targets.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(checks).to_csv(
        TABLES / f"{PREFIX}_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段233 valid攻击修正审计",
        "",
        f"- 状态：**{payload['status']}**",
        "- 旧结论：**INVALIDATED**",
        "",
        "## 修正后目标域结果",
        "",
        "| 目标域 | 站点数 | 主比较 | F1差值 | 事件覆盖差值 |",
        "|---|---:|---|---:|---:|",
    ]
    for row in target_rows:
        lines.append(
            f"| {row['target_source']} | {row['station_count']} | "
            f"`{row['comparison']}` | "
            f"{row['selected_minus_recalibrated_f1']:+.6f} | "
            f"{row['selected_minus_recalibrated_event_coverage']:+.6f} |"
        )
    lines += [
        "",
        "## 纠错结论",
        "",
        f"- 旧结论状态：`{payload['old_claim_status']}`",
        f"- 旧结论：{payload['old_claim']}",
        f"- 修正结论：{payload['corrected_claim']}",
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
        "",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": payload["status"],
                "old_claim_status": payload["old_claim_status"],
                "targets": [
                    {
                        "target_source": row["target_source"],
                        "f1": row["selected_minus_recalibrated_f1"],
                        "event_coverage": row[
                            "selected_minus_recalibrated_event_coverage"
                        ],
                    }
                    for row in target_rows
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
