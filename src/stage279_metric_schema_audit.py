from __future__ import annotations

"""Stage 279: audit the reporting schema of the frozen primary tables.

This stage is read-only. It checks whether every primary result table exposes
the fields required for fair comparison and honest uncertainty reporting. It
does not rename, backfill, impute, retrain, recalibrate, or alter any result.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage279_metric_schema_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


REQUIRED_GROUPS = {
    "identity": {
        "source": ["source", "provider", "stage"],
        "method": ["method", "candidate"],
        "seed": ["seed"],
    },
    "classification": {
        "f1": ["f1", "test_f1", "f1_mean", "provider_equal_f1"],
        "precision": [
            "precision",
            "test_precision",
            "precision_mean",
            "provider_equal_precision",
        ],
        "recall": [
            "recall",
            "test_recall",
            "recall_mean",
            "provider_equal_recall",
        ],
        "far": [
            "false_alarm_rate",
            "test_false_alarm_rate",
            "far",
            "far_mean",
            "provider_equal_false_alarm_rate",
        ],
    },
    "event": {
        "event_coverage": [
            "event_coverage",
            "event_coverage_mean",
            "event_coverage_difference",
            "provider_equal_event_coverage",
        ],
        "event_denominator": [
            "events",
            "event_count",
            "event_truth_count",
            "event_denominator",
            "attack_event_count",
            "event_denominator_definition",
        ],
    },
    "ranking": {
        "auprc": [
            "average_precision",
            "test_average_precision",
            "auprc",
            "auprc_mean",
            "provider_equal_average_precision",
        ],
        "row_recall": [
            "row_recall",
            "row_recall_mean",
            "provider_equal_row_recall",
        ],
    },
}

OPTIONAL_GROUPS = {
    "low_far_operating_point": {
        "threshold": ["threshold", "alarm_budget", "budget"],
    },
    "uncertainty": {
        "ci_low": [
            "ci95_low",
            "f1_ci_low",
            "event_ci95_low",
            "ci_low",
        ],
        "ci_high": [
            "ci95_high",
            "f1_ci_high",
            "event_ci95_high",
            "ci_high",
        ],
    },
}

SUPPORTING_TABLES = {
    "stage263_comparisons": (
        "stage263_unified_statistical_report_comparisons.csv",
        {
            "uncertainty": {
                "f1_ci_low": ["f1_ci95_low"],
                "f1_ci_high": ["f1_ci95_high"],
                "event_coverage_ci_low": ["event_ci95_low"],
                "event_coverage_ci_high": ["event_ci95_high"],
            },
        },
    ),
    "stage263_event_definition": (
        "stage263_unified_statistical_report_event_definition_audit.csv",
        {
            "event_length": {
                "mean_rows_per_event": ["mean_rows_per_event"],
                "min_rows_per_event": ["min_rows_per_event"],
                "max_rows_per_event": ["max_rows_per_event"],
                "single_row_event_types": ["single_row_event_types"],
            },
        },
    ),
}

TABLES_TO_AUDIT = {
    "stage248_cross_provider_fair_baseline_audit": (
        "stage248_cross_provider_fair_baseline_audit_evaluation.csv",
        "cross_provider_evaluation",
        "required",
    ),
    "stage248_station_detail": (
        "stage248_cross_provider_fair_baseline_audit_station_detail.csv",
        "station_detail",
        "station_counts",
    ),
    "stage263_pooled": (
        "stage263_unified_statistical_report_pooled.csv",
        "pooled_result",
        "required",
    ),
    "stage263_station_equal": (
        "stage263_unified_statistical_report_station_equal.csv",
        "station_equal_result",
        "required",
    ),
    "stage263_provider_equal": (
        "stage263_unified_statistical_report_provider_equal.csv",
        "provider_equal_aggregate",
        "provider_equal",
    ),
    "stage263_event_definition": (
        "stage263_unified_statistical_report_event_definition_audit.csv",
        "event_definition",
        "event_only",
    ),
    "stage259_fmi_evaluation": (
        "stage259_fmi_locked_transfer_evaluation.csv",
        "cross_provider_evaluation",
        "required",
    ),
    "stage267_smhi_evaluation": (
        "stage267_smhi_locked_transfer_evaluation.csv",
        "cross_provider_evaluation",
        "required",
    ),
}

EVENT_ONLY_GROUPS = {
    "event": {
        "event_denominator": REQUIRED_GROUPS["event"][
            "event_denominator"
        ],
    },
    "definition": {
        "event_denominator_definition": [
            "event_denominator_definition",
        ],
        "event_coverage_definition": [
            "event_coverage_definition",
        ],
    },
}

PROVIDER_EQUAL_GROUPS = {
    "identity": {
        "stage": ["stage"],
        "method": ["method"],
        "provider_count": ["provider_count"],
    },
    "classification": {
        "f1": ["provider_equal_f1"],
        "precision": ["provider_equal_precision"],
        "recall": ["provider_equal_recall"],
        "far": ["provider_equal_false_alarm_rate"],
    },
    "event": {
        "event_coverage": ["provider_equal_event_coverage"],
    },
    "ranking": {
        "auprc": ["provider_equal_average_precision"],
        "row_recall": ["provider_equal_row_recall"],
    },
}


def _resolve_group(
    columns: list[str],
    aliases: list[str],
) -> str | None:
    lookup = {str(column).lower(): str(column) for column in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for artifact, (filename, groups) in SUPPORTING_TABLES.items():
        path = TABLES / filename
        if not path.is_file():
            missing_rows.append(
                {
                    "artifact": artifact,
                    "relative_path": f"tables/{filename}",
                    "table_role": "supporting_table",
                    "requirement_mode": "supporting",
                    "group": "artifact",
                    "required_field": "file_exists",
                    "resolved_field": "",
                    "status": "FAIL_MISSING_ARTIFACT",
                }
            )
            continue
        columns = list(pd.read_csv(path, nrows=1).columns)
        for group, aliases in groups.items():
            for required_field, field_aliases in aliases.items():
                resolved = _resolve_group(columns, field_aliases)
                row = {
                    "artifact": artifact,
                    "relative_path": f"tables/{filename}",
                    "table_role": "supporting_table",
                    "requirement_mode": "supporting",
                    "group": group,
                    "required_field": required_field,
                    "resolved_field": resolved or "",
                    "status": (
                        "PASS_FIELD_PRESENT"
                        if resolved is not None
                        else "FAIL_FIELD_MISSING"
                    ),
                }
                rows.append(row)
                if resolved is None:
                    missing_rows.append(row)
    for artifact, (filename, table_role, mode) in TABLES_TO_AUDIT.items():
        path = TABLES / filename
        if not path.is_file():
            missing_rows.append(
                {
                    "artifact": artifact,
                    "relative_path": f"tables/{filename}",
                    "group": "artifact",
                    "required_field": "file_exists",
                    "resolved_field": "",
                    "status": "FAIL_MISSING_ARTIFACT",
                }
            )
            continue
        frame = pd.read_csv(path, nrows=200000)
        columns = list(frame.columns)
        if mode == "event_only":
            groups = EVENT_ONLY_GROUPS
        elif mode == "station_counts":
            groups = {
                "identity": {
                    "source": ["source", "provider"],
                    "method": ["method", "candidate"],
                    "seed": ["seed"],
                    "station": ["station_id", "station"],
                },
                "counts": {
                    "true_positive": ["tp", "true_positive"],
                    "false_positive": ["fp", "false_positive"],
                    "false_negative": ["fn", "false_negative"],
                    "true_negative": ["tn", "true_negative"],
                    "attack_rows": ["attack_rows"],
                    "normal_rows": ["normal_rows"],
                },
                "event": {
                    "event_denominator": REQUIRED_GROUPS["event"][
                        "event_denominator"
                    ],
                },
            }
        elif mode == "provider_equal":
            groups = PROVIDER_EQUAL_GROUPS
        else:
            groups = REQUIRED_GROUPS
        for group, aliases in groups.items():
            for required_field, field_aliases in aliases.items():
                resolved = _resolve_group(columns, field_aliases)
                status = (
                    "PASS_FIELD_PRESENT"
                    if resolved is not None
                    else "FAIL_FIELD_MISSING"
                )
                row = {
                    "artifact": artifact,
                    "relative_path": f"tables/{filename}",
                    "table_role": table_role,
                    "requirement_mode": mode,
                    "group": group,
                    "required_field": required_field,
                    "resolved_field": resolved or "",
                    "status": status,
                }
                rows.append(row)
                if resolved is None and mode != "event_only":
                    missing_rows.append(row)

        if mode == "event_only":
            continue
        for group, aliases in OPTIONAL_GROUPS.items():
            for required_field, field_aliases in aliases.items():
                resolved = _resolve_group(columns, field_aliases)
                optional_status = (
                    "OPTIONAL_FIELD_PRESENT"
                    if resolved is not None
                    else "OPTIONAL_FIELD_NOT_IN_THIS_TABLE"
                )
                audit_optional = {
                    "artifact": artifact,
                    "relative_path": f"tables/{filename}",
                    "table_role": table_role,
                    "requirement_mode": mode,
                    "group": group,
                    "required_field": required_field,
                    "resolved_field": resolved or "",
                    "status": optional_status,
                }
                rows.append(audit_optional)
    audit = pd.DataFrame(rows)
    if audit.empty:
        raise RuntimeError("metric schema audit produced no rows")
    audit.to_csv(
        TABLES / f"{PREFIX}_fields.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing = pd.DataFrame(missing_rows)
    if not missing.empty:
        missing.to_csv(
            TABLES / f"{PREFIX}_missing.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        (TABLES / f"{PREFIX}_missing.csv").write_text(
            "artifact,relative_path,group,required_field,resolved_field,status\n",
            encoding="utf-8",
        )

    required_count = int(
        audit["status"].isin(
            [
                "PASS_FIELD_PRESENT",
                "FAIL_FIELD_MISSING",
            ]
        ).sum()
    )
    present_count = int(audit["status"].eq("PASS_FIELD_PRESENT").sum())
    missing_count = int(len(missing))
    payload = {
        "status": (
            "PASS_METRIC_SCHEMA_AUDIT"
            if missing_count == 0
            else "REVIEW_REQUIRED_METRIC_SCHEMA"
        ),
        "protocol": (
            "Stage279 reads frozen primary result CSVs and checks required "
            "identity, classification, event, ranking, and uncertainty "
            "fields. It does not modify or synthesize any result."
        ),
        "artifact_count": int(
            len(TABLES_TO_AUDIT) + len(SUPPORTING_TABLES)
        ),
        "required_field_check_count": required_count,
        "present_field_check_count": present_count,
        "missing_field_check_count": missing_count,
        "optional_field_check_count": int(
            audit["status"].eq("OPTIONAL_FIELD_PRESENT").sum()
            + audit["status"].eq(
                "OPTIONAL_FIELD_NOT_IN_THIS_TABLE"
            ).sum()
        ),
        "optional_field_present_count": int(
            audit["status"].eq("OPTIONAL_FIELD_PRESENT").sum()
        ),
        "missing_by_artifact": {
            str(key): int(value)
            for key, value in (
                missing.groupby("artifact").size().to_dict().items()
                if not missing.empty
                else {}
            )
        },
        "input_sha256": {
            **{
                f"supporting::{artifact}": sha256_file(
                    TABLES / filename
                )
                for artifact, (filename, _groups) in sorted(
                    SUPPORTING_TABLES.items()
                )
                if (TABLES / filename).is_file()
            },
            **{
                f"required::{artifact}": sha256_file(
                    TABLES / filename
                )
                for artifact, (filename, _role, _mode) in sorted(
                    TABLES_TO_AUDIT.items()
                )
                if (TABLES / filename).is_file()
            },
        },
        "limitations": [
            "A missing field is reported as missing; it is not backfilled.",
            "Aliases allow historical naming variants, but they do not prove "
            "metric equivalence across different definitions.",
            "This audit does not establish a performance advantage.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage 279 指标模式审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 审计表数：`{payload['artifact_count']}`",
        f"- 必需字段检查：`{payload['required_field_check_count']}`",
        f"- 已存在：`{payload['present_field_check_count']}`",
        f"- 缺失：`{payload['missing_field_check_count']}`",
        f"- 可选字段检查：`{payload['optional_field_check_count']}`",
        f"- 可选字段实际存在：`{payload['optional_field_present_count']}`",
        "",
        "## 规则",
        "",
        "- 缺失字段如实报告，不自动回填或推断。",
        "- 别名只处理历史命名差异，不证明不同指标定义等价。",
        "- 本阶段不训练、不校准阈值、不改变任何性能结果。",
        "",
    ]
    if not missing.empty:
        report += ["## 缺失字段", ""]
        for row in missing.to_dict("records"):
            report.append(
                f"- `{row['artifact']}`：`{row['group']}.{row['required_field']}`"
            )
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "artifacts": payload["artifact_count"],
                "required": required_count,
                "present": present_count,
                "missing": missing_count,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
