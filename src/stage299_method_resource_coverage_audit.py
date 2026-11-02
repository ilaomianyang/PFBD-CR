from __future__ import annotations

"""Stage 299: read-only method-level resource-evidence coverage audit.

This stage does not train a model, recalibrate a threshold, generate an
attack, read valid/test labels, or replace a frozen result. It reconciles
the resource fields already present in Stage269, Stage290, Stage288,
Stage255, and Stage272 and records exactly which fields are measured,
which are only structural provenance, and which remain unavailable.

The audit intentionally keeps evidence layers separate. In particular, the
same-contract baseline resource budget is not merged with the independent
public SWaT/USAD benchmark or the authorized testbed path.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage299_method_resource_coverage_audit"
PROTOCOL_ID = "TMQC-METHOD-RESOURCE-COVERAGE-20260921-v1.0"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


def _load_json(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(name: str) -> pd.DataFrame:
    path = TABLES / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _is_available(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().upper() not in {"", "UNAVAILABLE", "NAN", "NONE"}
    try:
        return bool(pd.notna(value))
    except TypeError:
        return True


def _status(value: object) -> str:
    if _is_available(value):
        return "MEASURED"
    return "UNAVAILABLE"


def _method_identity(row: dict) -> tuple[str, str, str]:
    return (
        str(row["method"]),
        str(row.get("family", "")),
        str(row.get("evidence_layer", "")),
    )


def _stage269_rows(stage269: dict) -> dict[str, dict]:
    return {
        str(row["method"]): row
        for row in stage269.get("methods", [])
        if isinstance(row, dict) and "method" in row
    }


def _stage290_rows(stage290: dict) -> dict[str, dict]:
    return {
        str(row["method"]): row
        for row in stage290.get("methods", [])
        if isinstance(row, dict) and "method" in row
    }


def _stage288_rows(stage288: dict) -> dict[str, dict]:
    methods = stage288.get("methods", [])
    if not isinstance(methods, list) or not methods:
        return {}
    return {
        str(row["method"]): row
        for row in methods
        if isinstance(row, dict) and "method" in row
    }


def _evidence_class(method: str, layer: str) -> str:
    if method.startswith("trust_"):
        return "proposed_method"
    if layer == "independent_public_benchmark":
        return "independent_public_benchmark"
    if layer == "authorized_testbed":
        return "authorized_testbed"
    return "same_contract_baseline"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stage269 = _load_json("stage269_deployment_complexity_audit")
    stage290 = _load_json("stage290_same_contract_baseline_resource_budget")
    stage288 = _load_json("stage288_swat_usad_resource_audit")
    stage255 = _load_json("stage255_online_causal_testbed")
    stage272 = _load_json("stage272_final_parameter_table")
    stage272_table = _load_csv("stage272_final_parameter_table")
    stage301 = _load_json("stage301_trust_resource_budget")

    s269 = _stage269_rows(stage269)
    s290 = _stage290_rows(stage290)
    s288 = _stage288_rows(stage288)
    if not s269:
        raise RuntimeError("Stage269 has no method rows")
    if not s290:
        raise RuntimeError("Stage290 has no method rows")

    # Stage269 is the structural inventory. Stage290 is the authoritative
    # same-contract measured budget for the 11 frozen baseline methods.
    # Stage301 supplies the separately measured deterministic Trust budget.
    trust_measurement = stage301.get("measurement", {})
    trust_available = (
        str(stage301.get("status")) == "PASS_TRUST_RESOURCE_BUDGET_COMPLETE"
        and bool(stage301.get("trust_resource_fields_complete"))
        and isinstance(trust_measurement, dict)
    )
    method_rows: list[dict[str, object]] = []
    for method, base in sorted(s269.items()):
        measured = s290.get(method, {})
        evidence_class = _evidence_class(
            method,
            str(base.get("evidence_layer", "")),
        )
        if method.startswith("trust_") and trust_available:
            score_latency = trust_measurement.get("score_latency")
            parameter_count = trust_measurement.get(
                "configuration_parameter_count"
            )
            fit_seconds = trust_measurement.get("component_fit_seconds")
            peak_memory_mb = trust_measurement.get("peak_working_set_mb")
            score_latency_ms = (
                score_latency.get("mean_ms")
                if isinstance(score_latency, dict)
                else None
            )
            fit_rows = trust_measurement.get("fit_rows")
            failure_state = trust_measurement.get("failure_state")
            source_stage = "stage301_trust_resource_budget"
            evidence_scope = str(
                trust_measurement.get("evidence_scope", "")
            )
            same_contract = False
        elif measured:
            parameter_count = measured.get("parameter_count")
            fit_seconds = measured.get("fit_seconds")
            peak_memory_mb = measured.get("peak_working_set_mb")
            score_latency = measured.get("score_latency")
            if isinstance(score_latency, dict):
                score_latency_ms = score_latency.get("mean_ms")
            else:
                score_latency_ms = None
            fit_rows = measured.get("fit_rows")
            failure_state = measured.get("failure_state")
            source_stage = "stage290_same_contract_baseline_resource_budget"
            evidence_scope = measured.get(
                "evidence_scope",
                "fit_and_score_train_normal_matrix_only_no_valid_no_test",
            )
            same_contract = evidence_class == "same_contract_baseline"
        else:
            parameter_count = base.get("parameter_count")
            fit_seconds = base.get("fit_seconds")
            peak_memory_mb = base.get("peak_memory_mb")
            score_latency_ms = base.get("score_latency_ms")
            fit_rows = base.get("fit_rows")
            failure_state = "completed" if method == "usad_author_code" else None
            source_stage = (
                "stage288_swat_usad_resource_audit"
                if method == "usad_author_code"
                else "stage269_deployment_complexity_audit"
            )
            evidence_scope = str(base.get("evidence_status", ""))
            same_contract = evidence_class == "same_contract_baseline"

        structure_available = _is_available(base.get("model_structure"))
        method_rows.append(
            {
                "method": method,
                "display_name": base.get("display_name", method),
                "family": base.get("family", ""),
                "evidence_layer": base.get("evidence_layer", ""),
                "evidence_class": evidence_class,
                "same_contract_as_trust": bool(same_contract),
                "source_stage": source_stage,
                "evidence_scope": evidence_scope,
                "fit_rows": fit_rows,
                "fit_rows_status": _status(fit_rows),
                "parameter_count": parameter_count,
                "parameter_count_status": _status(parameter_count),
                "parameter_count_evidence": (
                    "MEASURED"
                    if _is_available(parameter_count)
                    else (
                        "STRUCTURE_ONLY"
                        if structure_available
                        else "UNAVAILABLE"
                    )
                ),
                "model_structure": base.get("model_structure", ""),
                "fit_seconds": fit_seconds,
                "fit_seconds_status": _status(fit_seconds),
                "peak_memory_mb": peak_memory_mb,
                "peak_memory_mb_status": _status(peak_memory_mb),
                "score_latency_ms": score_latency_ms,
                "score_latency_ms_status": _status(score_latency_ms),
                "throughput_fps": (
                    trust_measurement.get(
                        "single_station_throughput_rows_per_second"
                    )
                    if method.startswith("trust_") and trust_available
                    else base.get("throughput_fps")
                ),
                "throughput_fps_status": _status(
                    trust_measurement.get(
                        "single_station_throughput_rows_per_second"
                    )
                    if method.startswith("trust_") and trust_available
                    else base.get("throughput_fps")
                ),
                "online_causal": base.get("online_causal"),
                "gpu_required": base.get("gpu_required"),
                "station_recalibration_required": base.get(
                    "station_recalibration_required"
                ),
                "failure_state": failure_state,
                "failure_state_status": _status(failure_state),
            }
        )

    frame = pd.DataFrame(method_rows)
    if frame["method"].duplicated().any():
        raise RuntimeError("duplicate method rows in Stage299")

    same_contract = frame[
        frame["evidence_class"].eq("same_contract_baseline")
    ]
    same_contract_count = int(len(same_contract))
    same_contract_complete = bool(
        same_contract_count > 0
        and same_contract["parameter_count_status"].ne("UNAVAILABLE").all()
        and same_contract["fit_seconds_status"].ne("UNAVAILABLE").all()
        and same_contract["peak_memory_mb_status"].ne("UNAVAILABLE").all()
        and same_contract["score_latency_ms_status"].ne("UNAVAILABLE").all()
        and same_contract["failure_state_status"].ne("UNAVAILABLE").all()
    )
    all_methods_complete = bool(
        frame["parameter_count_status"].ne("UNAVAILABLE").all()
        and frame["fit_seconds_status"].ne("UNAVAILABLE").all()
        and frame["peak_memory_mb_status"].ne("UNAVAILABLE").all()
        and frame["score_latency_ms_status"].ne("UNAVAILABLE").all()
    )

    field_names = [
        "fit_rows",
        "parameter_count",
        "fit_seconds",
        "peak_memory_mb",
        "score_latency_ms",
        "throughput_fps",
        "failure_state",
    ]
    coverage_rows = []
    for field in field_names:
        status_column = f"{field}_status"
        if status_column not in frame.columns:
            continue
        for layer, subset in [
            ("all_methods", frame),
            ("same_contract_baseline", same_contract),
            (
                "proposed_method",
                frame[frame["evidence_class"].eq("proposed_method")],
            ),
            (
                "independent_public_benchmark",
                frame[
                    frame["evidence_layer"].eq(
                        "independent_public_benchmark"
                    )
                ],
            ),
            (
                "authorized_testbed",
                frame[frame["evidence_layer"].eq("authorized_testbed")],
            ),
        ]:
            if subset.empty:
                continue
            coverage_rows.append(
                {
                    "evidence_layer": layer,
                    "field": field,
                    "method_count": int(len(subset)),
                    "measured_or_structural_count": int(
                        subset[status_column].eq("MEASURED").sum()
                    ),
                    "measured_count": int(
                        subset[status_column].eq("MEASURED").sum()
                    ),
                    "structural_count": 0,
                    "unavailable_count": int(
                        subset[status_column].eq("UNAVAILABLE").sum()
                    ),
                }
            )
    coverage = pd.DataFrame(coverage_rows)

    stage272_method_names = set(
        stage272_table.loc[
            stage272_table["parameter_scope"].eq("method"),
            "parameter",
        ]
        .dropna()
        .astype(str)
    )
    missing_from_stage272 = sorted(set(frame["method"]) - stage272_method_names)

    payload = {
        "status": "PASS_METHOD_RESOURCE_COVERAGE_AUDIT",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "Stage299 is read-only. It reconciles method-level resource "
            "fields already present in Stage269, Stage290, Stage288, "
            "Stage255, Stage272, and Stage300. It does not train, tune, "
            "read valid/test labels, or merge evidence across contracts."
        ),
        "method_count": int(len(frame)),
        "same_contract_method_count": same_contract_count,
        "same_contract_resource_budget_complete": same_contract_complete,
        "all_method_resource_evidence_complete": all_methods_complete,
        "same_contract_complete_scope": (
            "11 frozen same-contract baselines on the MeteoSwiss "
            "train-normal feature matrix; this excludes Trust candidates, "
            "USAD, and the authorized testbed path."
        ),
        "trust_parameter_count_status": (
            "AVAILABLE"
            if frame.loc[
                frame["method"].eq("trust_valid_f1"),
                "parameter_count_status",
            ].eq("MEASURED").all()
            else "UNAVAILABLE"
        ),
        "trust_resource_fields_complete": bool(
            frame.loc[
                frame["method"].isin(
                    ["trust_valid_f1", "trust_nonstuck_event_macro"]
                )
            ][
                [
                    "parameter_count_status",
                    "fit_seconds_status",
                    "peak_memory_mb_status",
                    "score_latency_ms_status",
                ]
            ]
            .ne("UNAVAILABLE")
            .all()
            .all()
        ),
        "usad_resource_fields_complete": bool(
            frame.loc[
                frame["method"].eq("usad_author_code"),
                [
                    "parameter_count_status",
                    "fit_seconds_status",
                    "peak_memory_mb_status",
                    "score_latency_ms_status",
                ],
            ]
            .ne("UNAVAILABLE")
            .all()
            .all()
        ),
        "testbed_resource_fields_complete": bool(
            frame.loc[
                frame["method"].eq("testbed_causal_detector"),
                [
                    "parameter_count_status",
                    "fit_seconds_status",
                    "peak_memory_mb_status",
                    "score_latency_ms_status",
                ],
            ]
            .ne("UNAVAILABLE")
            .all()
            .all()
        ),
        "missing_from_stage272_methods": missing_from_stage272,
        "coverage": coverage.to_dict("records"),
        "methods": frame.to_dict("records"),
        "input_sha256": {
            "stage269_deployment_complexity_audit": sha256_file(
                TABLES / "stage269_deployment_complexity_audit.json"
            ),
            "stage290_same_contract_baseline_resource_budget": sha256_file(
                TABLES
                / "stage290_same_contract_baseline_resource_budget.json"
            ),
            "stage288_swat_usad_resource_audit": sha256_file(
                TABLES / "stage288_swat_usad_resource_audit.json"
            ),
            "stage255_online_causal_testbed": sha256_file(
                TABLES / "stage255_online_causal_testbed.json"
            ),
            "stage272_final_parameter_table": sha256_file(
                TABLES / "stage272_final_parameter_table.json"
            ),
            "stage301_trust_resource_budget": sha256_file(
                TABLES / "stage301_trust_resource_budget.json"
            ),
        },
        "limitations": [
            "Stage290's complete budget applies only to its 11 same-contract "
            "baselines on the frozen MeteoSwiss train-normal matrix.",
            "Trust configuration complexity is counted, not neural "
            "parameters; Stage301 measures fit/memory/latency on the frozen "
            "MeteoSwiss train-normal matrix only.",
            "USAD is an independent public SWaT benchmark with a different "
            "endpoint and must not be merged into the same-contract table.",
            "The authorized testbed path is runtime evidence, not a "
            "same-contract trained baseline.",
            "No unavailable value is estimated or imputed.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    frame.to_csv(
        TABLES / f"{PREFIX}_methods.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage.to_csv(
        TABLES / f"{PREFIX}_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = [
        "# Stage 299 方法资源证据覆盖审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 方法数：`{payload['method_count']}`",
        f"- 同契约基线数：`{payload['same_contract_method_count']}`",
        f"- 同契约资源预算完整：`{payload['same_contract_resource_budget_complete']}`",
        f"- 全部方法资源字段完整：`{payload['all_method_resource_evidence_complete']}`",
        f"- Trust 参数数量状态：`{payload['trust_parameter_count_status']}`",
        f"- Trust 资源字段完整：`{payload['trust_resource_fields_complete']}`",
        f"- USAD 资源字段完整：`{payload['usad_resource_fields_complete']}`",
        f"- 测试床资源字段完整：`{payload['testbed_resource_fields_complete']}`",
        "",
        "## 覆盖矩阵",
        "",
        "| 证据层 | 字段 | 方法数 | 实测或结构 | 实测 | 结构 | 缺失 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in coverage.to_dict("records"):
        report.append(
            f"| {row['evidence_layer']} | {row['field']} | "
            f"{row['method_count']} | {row['measured_or_structural_count']} | "
            f"{row['measured_count']} | {row['structural_count']} | "
            f"{row['unavailable_count']} |"
        )
    report += [
        "",
        "## 边界",
        "",
        "- 本阶段只对账已有字段，不训练、不调阈值、不读取 valid/test。",
        "- Stage290 的完整性只适用于 11 个同契约基线，不适用于 Trust、USAD 或测试床。",
        "- 模型结构文字不能当作参数量；缺失值保持 `UNAVAILABLE`。",
        "- 不同合同、不同端点和运行时测试床证据不合并计算。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "method_count": payload["method_count"],
                "same_contract_resource_budget_complete": payload[
                    "same_contract_resource_budget_complete"
                ],
                "all_method_resource_evidence_complete": payload[
                    "all_method_resource_evidence_complete"
                ],
                "trust_resource_fields_complete": payload[
                    "trust_resource_fields_complete"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
