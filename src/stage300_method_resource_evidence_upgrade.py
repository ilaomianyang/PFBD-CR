from __future__ import annotations

"""Stage 300: reconcile measurable resource fields without inventing values.

This stage is read-only with respect to all predictions. It converts the
frozen TrustMeteo-QC selection table into a deterministic configuration count,
reuses the already measured authorized-testbed throughput/latency from
Stage255, and leaves fields that were never measured as UNAVAILABLE.

It does not train, calibrate thresholds, inspect valid/test labels, alter any
prediction, or promote an exploratory result.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage300_method_resource_evidence_upgrade"
PROTOCOL_ID = "TMQC-METHOD-RESOURCE-EVIDENCE-UPGRADE-20260921-v1.0"

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
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _status(value: object) -> str:
    if value is None:
        return "UNAVAILABLE"
    if isinstance(value, float) and pd.isna(value):
        return "UNAVAILABLE"
    if isinstance(value, str) and not value.strip():
        return "UNAVAILABLE"
    return "MEASURED"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    selection = _load_csv(
        "stage248_cross_provider_fair_baseline_audit_selection.csv"
    )
    stage248 = _load_json(
        "stage248_cross_provider_fair_baseline_audit"
    )
    stage255 = _load_json("stage255_online_causal_testbed")
    stage269 = _load_json("stage269_deployment_complexity_audit")
    stage290 = _load_json(
        "stage290_same_contract_baseline_resource_budget"
    )
    stage299 = _load_json("stage299_method_resource_coverage_audit")
    stage301 = _load_json("stage301_trust_resource_budget")

    required_selection = {
        "source",
        "seed",
        "method",
        "rule",
        "feature",
        "window",
        "weight",
        "threshold",
    }
    missing_selection = sorted(required_selection - set(selection.columns))
    if missing_selection:
        raise RuntimeError(
            f"selection table missing columns: {missing_selection}"
        )

    trust_methods = [
        "trust_valid_f1",
        "trust_nonstuck_event_macro",
    ]
    trust_selection = selection.loc[selection["method"].isin(trust_methods)]
    if trust_selection.empty:
        raise RuntimeError("no Trust selection rows found")

    # A Trust candidate is a deterministic rule: three fixed statistical
    # components plus one valid-selected feature/weight/threshold configuration.
    # We count only frozen configuration fields, never neural parameters.
    selection_rows = int(len(trust_selection))
    selection_parameter_fields = [
        "window",
        "weight",
        "threshold",
    ]
    selection_parameter_count = int(
        selection_rows * len(selection_parameter_fields)
    )
    fixed_component_count = 3
    fixed_component_weight_count = 3
    trust_configuration_parameter_count = int(
        selection_parameter_count
        + fixed_component_count
        + fixed_component_weight_count
    )

    fit_rows_by_source = {
        str(row["source"]): int(row["train_normal_rows"])
        for row in stage248.get("fit_rows", [])
    }
    if len(fit_rows_by_source) != 4:
        raise RuntimeError(
            "expected four frozen train-normal sources in Stage248"
        )

    testbed_throughput = float(
        stage255["throughput_mean_frames_per_second"]
    )
    testbed_latency_ms = float(1000.0 / testbed_throughput)
    trust_measurement = stage301.get("measurement", {})
    trust_resource_fields_complete = bool(
        str(stage301.get("status")) == "PASS_TRUST_RESOURCE_BUDGET_COMPLETE"
        and stage301.get("trust_resource_fields_complete")
        and isinstance(trust_measurement, dict)
    )
    if not trust_resource_fields_complete:
        raise RuntimeError("Stage301 Trust resource measurement did not pass")

    rows: list[dict[str, object]] = []
    for method in trust_methods:
        current = trust_selection.loc[trust_selection["method"].eq(method)]
        if len(current) != 20:
            raise RuntimeError(
                f"expected 20 frozen selections for {method}, got {len(current)}"
            )
        rows.append(
            {
                "method": method,
                "evidence_layer": "proposed_method",
                "source_stage": (
                    "stage248_cross_provider_fair_baseline_audit;"
                    "stage300_method_resource_evidence_upgrade;"
                    "stage301_trust_resource_budget"
                ),
                "resource_scope": (
                    "frozen_selection_configuration_and_isolated_train_only_measurement"
                ),
                "selection_configuration_rows": int(len(current)),
                "selection_parameter_fields": (
                    "window;weight;threshold"
                ),
                "selection_parameter_count": int(
                    len(current) * len(selection_parameter_fields)
                ),
                "fixed_component_count": fixed_component_count,
                "fixed_component_weight_count": (
                    fixed_component_weight_count
                ),
                "configuration_parameter_count": int(
                    len(current) * len(selection_parameter_fields)
                    + fixed_component_count
                    + fixed_component_weight_count
                ),
                "fit_rows_total": int(sum(fit_rows_by_source.values())),
                "fit_rows_by_source": fit_rows_by_source,
                "fit_seconds": trust_measurement.get(
                    "component_fit_seconds"
                ),
                "fit_seconds_scope": (
                    "MeteoSwiss train-normal component-fit replay in "
                    "Stage301; not a four-provider aggregate."
                ),
                "peak_memory_mb": trust_measurement.get(
                    "peak_working_set_mb"
                ),
                "peak_memory_mb_scope": (
                    "Isolated Stage301 child process on the MeteoSwiss "
                    "train-normal matrix."
                ),
                "score_latency_ms": (
                    trust_measurement.get("score_latency") or {}
                ).get("mean_ms"),
                "score_latency_ms_scope": (
                    "Mean across the 10 frozen Trust rule configurations; "
                    "train-normal end-to-end scoring replay."
                ),
                "throughput_fps": trust_measurement.get(
                    "single_station_throughput_rows_per_second"
                ),
                "throughput_scope": (
                    "Single-station train-normal replay throughput in rows "
                    "per second."
                ),
                "parameter_count_evidence": "CONFIGURATION_COUNT",
                "parameter_count_claim": (
                    "Deterministic configuration complexity, not neural "
                    "network parameter count."
                ),
            }
        )

    testbed = stage269.get("methods", [])
    testbed_rows = [
        row for row in testbed if str(row.get("method")) == "testbed_causal_detector"
    ]
    if len(testbed_rows) != 1:
        raise RuntimeError("expected one testbed_causal_detector row")
    testbed_row = testbed_rows[0]
    testbed_output = {
        "method": "testbed_causal_detector",
        "evidence_layer": "authorized_testbed",
        "source_stage": (
            "stage255_online_causal_testbed;"
            "stage269_deployment_complexity_audit;"
            "stage300_method_resource_evidence_upgrade"
        ),
        "resource_scope": "authorized_loopback_testbed_runtime_path",
        "parameter_count": "UNAVAILABLE",
        "parameter_count_reason": (
            "The testbed path is a deterministic runtime detector/recovery "
            "path; no neural parameter artifact exists."
        ),
        "fit_rows": "UNAVAILABLE",
        "fit_rows_reason": (
            "The testbed path is evaluated from frozen runtime logs; no "
            "separate fit row count was retained."
        ),
        "fit_seconds": "UNAVAILABLE",
        "fit_seconds_reason": (
            "No isolated fit timing was retained for the testbed path."
        ),
        "peak_memory_mb": "UNAVAILABLE",
        "peak_memory_mb_reason": (
            "No isolated peak working-set measurement was retained for the "
            "testbed path."
        ),
        "score_latency_ms": testbed_latency_ms,
        "score_latency_ms_source": (
            "Derived as 1000 / measured mean throughput from Stage255."
        ),
        "throughput_fps": testbed_throughput,
        "throughput_fps_source": "Stage255 measured mean throughput.",
        "failure_state": "completed",
    }

    # Confirm the upstream stages remain internally consistent.
    if not bool(stage299.get("same_contract_resource_budget_complete")):
        raise RuntimeError(
            "Stage299 same-contract resource budget unexpectedly incomplete"
        )
    if (
        str(stage290.get("status"))
        != "PASS_SAME_CONTRACT_RESOURCE_BUDGET_COMPLETE"
        or int(stage290.get("failed_method_count", -1)) != 0
        or int(stage290.get("completed_method_count", -1))
        != int(stage290.get("method_count", -2))
    ):
        raise RuntimeError(
            "Stage290 same-contract resource budget did not complete"
        )

    payload = {
        "status": "PASS_METHOD_RESOURCE_EVIDENCE_UPGRADE",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "Stage300 is read-only. It counts frozen Trust configuration "
            "fields from Stage248, reuses measured Stage255 testbed "
            "throughput/latency, and leaves unmeasured fields UNAVAILABLE. "
            "It does not train, tune, read valid/test labels, or alter "
            "predictions."
        ),
        "trust_selection_rows": selection_rows,
        "trust_selection_parameter_count": selection_parameter_count,
        "trust_configuration_parameter_count": (
            trust_configuration_parameter_count
        ),
        "trust_resource_fields_complete": trust_resource_fields_complete,
        "trust_resource_completion_boundary": (
            "Configuration complexity, isolated fit time, peak working set, "
            "end-to-end score latency, and single-station throughput are "
            "measured on MeteoSwiss train-normal rows only."
        ),
        "testbed_resource_fields_complete": False,
        "testbed_resource_completion_boundary": (
            "Throughput and derived latency are measured; parameter count, "
            "fit rows, fit time, and peak memory remain UNAVAILABLE."
        ),
        "all_method_resource_evidence_complete": False,
        "trust_rows": rows,
        "testbed_row": testbed_output,
        "input_sha256": {
            name: sha256_file(TABLES / name)
            for name in [
                "stage248_cross_provider_fair_baseline_audit.json",
                "stage248_cross_provider_fair_baseline_audit_selection.csv",
                "stage255_online_causal_testbed.json",
                "stage269_deployment_complexity_audit.json",
                "stage290_same_contract_baseline_resource_budget.json",
                "stage299_method_resource_coverage_audit.json",
                "stage301_trust_resource_budget.json",
            ]
        },
        "limitations": [
            "Configuration count is not a neural parameter count.",
            "Derived latency is the reciprocal of measured mean throughput, not a per-frame latency distribution.",
            "Stage301 Trust fit/memory/latency is measured on one frozen "
            "provider's train-normal matrix and is not a four-provider "
            "aggregate.",
            "No unmeasured resource field is estimated or imputed.",
            "The testbed is authorized and isolated; it is not a production environment.",
            "This stage does not close any P0 claim gate.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }

    json_path = TABLES / f"{PREFIX}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(
        TABLES / f"{PREFIX}_trust.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([testbed_output]).to_csv(
        TABLES / f"{PREFIX}_testbed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = [
        "# Stage 300 方法资源证据补全审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- Trust 冻结选择行：`{selection_rows}`",
        f"- Trust 选择参数字段计数：`{selection_parameter_count}`",
        (
            "- Trust 配置复杂度参数计数："
            f"`{trust_configuration_parameter_count}`"
        ),
        f"- Trust 资源字段完整：`{trust_resource_fields_complete}`",
        "- 测试床资源字段完整：`False`",
        "- 全部方法资源字段完整：`False`",
        "",
        "## 实测或可计数内容",
        "",
        (
            "- Trust 配置：4 个来源 × 5 个种子 × 2 个候选 = "
            f"`{selection_rows}` 行冻结选择配置。"
        ),
        (
            "- 每条配置含 `window`、`weight`、`threshold` 三个字段，"
            f"合计 `{selection_parameter_count}` 个配置参数；"
            "另计 3 个固定统计组件和 3 个固定融合权重。"
        ),
        (
            "- Stage301 在 MeteoSwiss 冻结 train-normal 矩阵上实测："
            f"拟合行数 `{trust_measurement.get('fit_rows')}`、"
            f"分量拟合秒 `{trust_measurement.get('component_fit_seconds')}`、"
            f"峰值工作集 `{trust_measurement.get('peak_working_set_mb')}` MB、"
            "全量冻结规则评分均值 "
            f"`{(trust_measurement.get('score_latency') or {}).get('mean_ms')}` ms、"
            "单站点吞吐 "
            f"`{trust_measurement.get('single_station_throughput_rows_per_second')}` "
            "行/秒。"
        ),
        (
            "- 授权测试床平均吞吐："
            f"`{testbed_throughput}` 帧/秒；"
            "由吞吐倒数得到平均帧延迟 "
            f"`{testbed_latency_ms}` ms。"
        ),
        "",
        "## 仍然缺失的字段",
        "",
        "- 测试床：`parameter_count`、`fit_rows`、`fit_seconds`、`peak_memory_mb` 仍为 `UNAVAILABLE`。",
        "- 上述字段没有被估算、插补或替换为结构文字。",
        "",
        "## 边界",
        "",
        "- 配置参数量不是神经网络参数量。",
        "- 吞吐倒数得到的平均延迟不是逐帧延迟分布。",
        "- 本阶段不训练、不调阈值、不读取 valid/test，也不关闭任何 P0 声称门。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": payload["status"],
                "trust_selection_rows": selection_rows,
                "trust_configuration_parameter_count": (
                    trust_configuration_parameter_count
                ),
                "trust_resource_fields_complete": (
                    trust_resource_fields_complete
                ),
                "testbed_resource_fields_complete": False,
                "all_method_resource_evidence_complete": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
