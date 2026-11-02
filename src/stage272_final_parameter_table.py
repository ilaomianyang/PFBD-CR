from __future__ import annotations

"""Stage 272: consolidated final parameter and evidence-table inventory.

The table is read-only. It extracts frozen parameters, method inventories,
fit budgets, seeds, and evidence-layer labels from existing JSON artifacts.
Unknown fields remain ``UNAVAILABLE``; nothing is estimated.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage272_final_parameter_table"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


def _load(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _row(
    source_stage: str,
    evidence_layer: str,
    parameter_scope: str,
    parameter: str,
    value: object,
    unit: str,
    source_path: str,
    note: str,
) -> dict[str, object]:
    if value is None:
        value = "UNAVAILABLE"
    return {
        "source_stage": source_stage,
        "evidence_layer": evidence_layer,
        "parameter_scope": parameter_scope,
        "parameter": parameter,
        "value": value,
        "unit": unit,
        "source_path": source_path,
        "note": note,
    }


def _join(value: object) -> str:
    if value is None:
        return "UNAVAILABLE"
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stage246 = _load("stage246_frozen_submission_package")
    stage248 = _load("stage248_cross_provider_fair_baseline_audit")
    stage254 = _load("stage254_ocsvm_training_scale_sensitivity")
    stage256 = _load("stage256_lightweight_strong_baselines")
    stage262 = _load("stage262_security_domain_baselines")
    stage264 = _load("stage264_smhi_protocol_preregistration")
    stage269 = _load("stage269_deployment_complexity_audit")

    rows: list[dict[str, object]] = []
    for name, payload, layer, path in [
        (
            "stage246_frozen_submission_package",
            stage246,
            "frozen_confirmatory",
            "stage246_frozen_submission_package.json",
        ),
        (
            "stage248_cross_provider_fair_baseline_audit",
            stage248,
            "frozen_confirmatory_cross_provider",
            "stage248_cross_provider_fair_baseline_audit.json",
        ),
        (
            "stage254_ocsvm_training_scale_sensitivity",
            stage254,
            "sensitivity",
            "stage254_ocsvm_training_scale_sensitivity.json",
        ),
        (
            "stage256_lightweight_strong_baselines",
            stage256,
            "exploratory_supplement",
            "stage256_lightweight_strong_baselines.json",
        ),
        (
            "stage262_security_domain_baselines",
            stage262,
            "exploratory_supplement",
            "stage262_security_domain_baselines.json",
        ),
        (
            "stage264_smhi_protocol_preregistration",
            stage264,
            "external_locked_preregistration",
            "stage264_smhi_protocol_preregistration.json",
        ),
        (
            "stage269_deployment_complexity_audit",
            stage269,
            "deployment_audit",
            "stage269_deployment_complexity_audit.json",
        ),
    ]:
        for parameter, unit, note in [
            ("status", "text", "artifact status"),
            ("primary_far", "fraction", "primary low-FAR operating point"),
            ("primary_endpoint", "text", "primary endpoint"),
            ("provider_count", "count", "providers in this artifact"),
            ("station_count", "count", "stations in this artifact"),
            ("seed_count", "count", "seeds in this artifact"),
            ("method_count", "count", "methods in this artifact"),
            ("methods", "list", "method inventory"),
            ("ocsvm_fit_cap", "rows", "One-Class SVM fit cap"),
            ("fit_limits", "rows", "requested fit sizes"),
            ("fit_seed", "seed", "fit seed"),
            ("knn_fit_cap", "rows", "kNN core-set cap"),
            ("attack_parameters", "text", "frozen attack parameters"),
            ("folds", "list", "folds"),
            ("seeds", "list", "seeds"),
        ]:
            if parameter in payload:
                rows.append(
                    _row(
                        source_stage=name,
                        evidence_layer=layer,
                        parameter_scope=(
                            "artifact"
                            if parameter not in {"methods", "folds", "seeds"}
                            else "inventory"
                        ),
                        parameter=parameter,
                        value=_join(payload.get(parameter)),
                        unit=unit,
                        source_path=path,
                        note=note,
                    )
                )

    method_frame = pd.DataFrame(
        stage269.get("methods", []),
    )
    if method_frame.empty:
        raise RuntimeError("Stage269 method inventory is empty")
    for record in method_frame.to_dict("records"):
        rows.append(
            _row(
                source_stage="stage269_deployment_complexity_audit",
                evidence_layer=str(record.get("evidence_layer", "unknown")),
                parameter_scope="method",
                parameter=str(record.get("method", "unknown")),
                value=_join(
                    {
                        "family": record.get("family"),
                        "model_structure": record.get("model_structure"),
                        "parameter_count": record.get("parameter_count"),
                        "fit_rows": record.get("fit_rows"),
                        "fit_seconds": record.get("fit_seconds"),
                        "peak_memory_mb": record.get("peak_memory_mb"),
                        "score_latency_ms": record.get("score_latency_ms"),
                        "online_causal": record.get("online_causal"),
                        "station_recalibration_required": record.get(
                            "station_recalibration_required"
                        ),
                        "gpu_required": record.get("gpu_required"),
                    }
                ),
                unit="method_descriptor",
                source_path="stage269_deployment_complexity_audit_methods.csv",
                note="Frozen method descriptor; missing fields remain UNAVAILABLE.",
            )
        )

    parameters = pd.DataFrame(rows)
    if parameters.empty:
        raise RuntimeError("parameter table is empty")
    if parameters.duplicated(
        ["source_stage", "evidence_layer", "parameter_scope", "parameter"]
    ).any():
        raise RuntimeError("duplicate parameter rows")
    parameters.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_FINAL_PARAMETER_TABLE",
        "protocol": (
            "Stage272 reads frozen JSON/CSV artifacts and records parameters "
            "without estimation. Missing values remain UNAVAILABLE. It does "
            "not train, recalibrate, or change any prediction."
        ),
        "parameter_rows": int(len(parameters)),
        "source_stage_count": int(parameters["source_stage"].nunique()),
        "method_rows": int(
            parameters["parameter_scope"].eq("method").sum()
        ),
        "input_sha256": {
            path: sha256_file(TABLES / path)
            for path in [
                "stage246_frozen_submission_package.json",
                "stage248_cross_provider_fair_baseline_audit.json",
                "stage254_ocsvm_training_scale_sensitivity.json",
                "stage256_lightweight_strong_baselines.json",
                "stage262_security_domain_baselines.json",
                "stage264_smhi_protocol_preregistration.json",
                "stage269_deployment_complexity_audit.json",
                "stage269_deployment_complexity_audit_methods.csv",
            ]
        },
        "limitations": [
            "UNAVAILABLE values are not estimates.",
            "The table combines artifacts with different evidence layers; "
            "the evidence_layer column must be retained in any paper table.",
            "Method-level parameter counts, memory, and training time are "
            "not available for every method.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report = [
        "# 阶段272 最终参数表",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 参数行数：`{payload['parameter_rows']}`",
        f"- 来源阶段数：`{payload['source_stage_count']}`",
        f"- 方法级行数：`{payload['method_rows']}`",
        "",
        "## 边界",
        "",
        "- 缺失参数保持 `UNAVAILABLE`，不做估计。",
        "- 不同证据层必须保留 `evidence_layer` 列，不能混写。",
        "- 该表不训练、不校阈值、不改变任何预测。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "parameter_rows": payload["parameter_rows"],
                "method_rows": payload["method_rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
