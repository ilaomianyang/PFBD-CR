from __future__ import annotations

"""Stage 270: unified attack-factor evidence table.

This stage is read-only. It consolidates the already-frozen intensity,
duration, position, and event-length artifacts into one long table with
explicit protocol-layer labels. It never pools incomparable protocols and
never changes a split, threshold, prediction, or endpoint.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage270_attack_factor_unified_table"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


FACTOR_SOURCES = {
    "intensity": {
        "local_stage151": {
            "path": "stage151_noaa_attack_intensity_sensitivity.json",
            "aggregate_key": "aggregate",
            "factor_column": "amplitude",
            "method": "TrustMeteo-QC frozen expanded",
            "protocol_layer": "local_multiyear_candidate_sensitivity",
            "evaluation_scope": "four_fold_five_seed",
        },
        "noaa_stage169": {
            "path": "stage169_noaa_controlled_attack_intensity_full_baseline.json",
            "aggregate_key": "summary",
            "factor_column": "amplitude",
            "method": "method",
            "protocol_layer": "noaa_controlled_full_baseline",
            "evaluation_scope": "four_fold_five_seed",
        },
    },
    "duration": {
        "local_stage118": {
            "path": "stage118_attack_duration_sensitivity.json",
            "aggregate_key": "duration_aggregate",
            "factor_column": "duration",
            "method": "TrustMeteo-QC frozen expanded",
            "protocol_layer": "local_9c_candidate_sensitivity",
            "evaluation_scope": "five_seed",
        },
        "noaa_stage170": {
            "path": "stage170_noaa_controlled_attack_duration_full_baseline.json",
            "aggregate_key": "summary",
            "factor_column": "duration",
            "method": "method",
            "protocol_layer": "noaa_controlled_full_baseline",
            "evaluation_scope": "four_fold_five_seed",
        },
    },
    "position": {
        "local_stage119": {
            "path": "stage119_attack_position_sensitivity.json",
            "aggregate_key": "shift_aggregate",
            "factor_column": "shift_rows",
            "method": "TrustMeteo-QC frozen expanded",
            "protocol_layer": "local_9c_candidate_sensitivity",
            "evaluation_scope": "five_seed",
        },
        "noaa_stage171": {
            "path": "stage171_noaa_controlled_attack_position_full_baseline.json",
            "aggregate_key": "summary",
            "factor_column": "shift_rows",
            "method": "method",
            "protocol_layer": "noaa_controlled_full_baseline",
            "evaluation_scope": "four_fold_five_seed",
        },
    },
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_metric_frame(
    frame: pd.DataFrame,
    factor_name: str,
    factor_value_column: str,
    source_stage: str,
    protocol_layer: str,
    evaluation_scope: str,
    method_column: str,
) -> pd.DataFrame:
    required = {
        factor_value_column,
        "test_f1_mean",
        "test_recall_mean",
        "test_precision_mean",
        "test_auprc_mean",
        "test_false_alarm_rate_mean",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{source_stage} missing columns: {sorted(missing)}"
        )

    normalized = pd.DataFrame(
        {
            "factor": factor_name,
            "source_stage": source_stage,
            "protocol_layer": protocol_layer,
            "evaluation_scope": evaluation_scope,
            "method": (
                frame[method_column].astype(str)
                if method_column in frame.columns
                else "TrustMeteo-QC frozen expanded"
            ),
            "factor_value": frame[factor_value_column],
            "test_f1_mean": frame["test_f1_mean"],
            "test_recall_mean": frame["test_recall_mean"],
            "test_precision_mean": frame["test_precision_mean"],
            "test_auprc_mean": frame["test_auprc_mean"],
            "test_false_alarm_rate_mean": frame[
                "test_false_alarm_rate_mean"
            ],
        }
    )
    for column in (
        "test_f1_mean",
        "test_recall_mean",
        "test_precision_mean",
        "test_auprc_mean",
        "test_false_alarm_rate_mean",
    ):
        values = pd.to_numeric(normalized[column], errors="coerce")
        if values.isna().any():
            raise RuntimeError(f"{source_stage} contains non-finite {column}")
        if not values.between(0.0, 1.0).all():
            raise RuntimeError(f"{source_stage} {column} outside [0, 1]")
    if normalized["factor_value"].isna().any():
        raise RuntimeError(f"{source_stage} contains missing factor values")
    return normalized


def _event_length_table() -> pd.DataFrame:
    audit = pd.read_csv(
        TABLES / "stage268_provider_scope_audit_event_audit.csv"
    )
    required = {
        "stage",
        "layer",
        "attack_type",
        "events",
        "attack_rows",
        "rows_per_event",
        "single_row_event",
    }
    missing = required.difference(audit.columns)
    if missing:
        raise RuntimeError(
            f"event-length audit missing columns: {sorted(missing)}"
        )
    if not np.isfinite(audit["rows_per_event"]).all():
        raise RuntimeError("event-length audit contains non-finite lengths")
    if audit["rows_per_event"].le(0).any():
        raise RuntimeError("event-length audit contains non-positive lengths")
    out = audit[
        [
            "stage",
            "layer",
            "attack_type",
            "events",
            "attack_rows",
            "rows_per_event",
            "single_row_event",
        ]
    ].copy()
    out.insert(0, "factor", "event_length")
    out.insert(1, "source_stage", out["stage"])
    out.insert(2, "protocol_layer", out["layer"])
    out["method"] = "UNAVAILABLE"
    out["test_f1_mean"] = np.nan
    out["test_recall_mean"] = np.nan
    out["test_precision_mean"] = np.nan
    out["test_auprc_mean"] = np.nan
    out["test_false_alarm_rate_mean"] = np.nan
    return out


def _paired_rows(
    factor_name: str,
    payload: dict,
    path: Path,
    factor_column: str,
    protocol_layer: str,
) -> pd.DataFrame:
    if "paired" not in payload:
        return pd.DataFrame()
    frame = payload["paired"]
    if not frame:
        return pd.DataFrame()
    rows = pd.DataFrame(frame)
    if factor_column not in rows.columns:
        return pd.DataFrame()
    rows.insert(0, "factor", factor_name)
    rows.insert(1, "source_stage", path.name.removesuffix(".json"))
    rows.insert(2, "protocol_layer", protocol_layer)
    rows.insert(3, "factor_value", rows[factor_column])
    rows["fold_seed_cells"] = rows.get("fold_seed_cells", np.nan)
    rows["mean_fold_diff"] = rows.get("mean_fold_diff", np.nan)
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    metric_frames: list[pd.DataFrame] = []
    paired_frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    source_manifest: list[dict[str, object]] = []

    for factor_name, sources in FACTOR_SOURCES.items():
        for source_name, info in sources.items():
            path = TABLES / str(info["path"])
            payload = _load_json(path)
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
            aggregate = payload.get(str(info["aggregate_key"]))
            if not aggregate:
                raise RuntimeError(
                    f"{source_name} has no aggregate rows"
                )
            normalized = _normalize_metric_frame(
                pd.DataFrame(aggregate),
                factor_name=factor_name,
                factor_value_column=str(info["factor_column"]),
                source_stage=source_name,
                protocol_layer=str(info["protocol_layer"]),
                evaluation_scope=str(info["evaluation_scope"]),
                method_column=str(info["method"]),
            )
            metric_frames.append(normalized)
            paired = _paired_rows(
                factor_name,
                payload,
                path,
                str(info["factor_column"]),
                str(info["protocol_layer"]),
            )
            if not paired.empty:
                paired_frames.append(paired)
            source_manifest.append(
                {
                    "factor": factor_name,
                    "source_stage": source_name,
                    "path": str(path.relative_to(ROOT)),
                    "protocol_layer": info["protocol_layer"],
                    "evaluation_scope": info["evaluation_scope"],
                    "rows": int(len(normalized)),
                    "methods": sorted(
                        normalized["method"].unique().tolist()
                    ),
                }
            )

    factor_table = pd.concat(metric_frames, ignore_index=True)
    event_table = _event_length_table()
    unified = pd.concat(
        [factor_table, event_table],
        ignore_index=True,
        sort=False,
    )
    paired_table = (
        pd.concat(paired_frames, ignore_index=True, sort=False)
        if paired_frames
        else pd.DataFrame()
    )

    method_summary = (
        factor_table.groupby(
            [
                "factor",
                "protocol_layer",
                "evaluation_scope",
                "method",
            ],
            as_index=False,
        )
        .agg(
            factor_levels=("factor_value", "nunique"),
            rows=("factor_value", "size"),
            f1_mean=("test_f1_mean", "mean"),
            f1_min=("test_f1_mean", "min"),
            f1_max=("test_f1_mean", "max"),
            far_mean=("test_false_alarm_rate_mean", "mean"),
            far_min=("test_false_alarm_rate_mean", "min"),
            far_max=("test_false_alarm_rate_mean", "max"),
            auprc_mean=("test_auprc_mean", "mean"),
        )
        .sort_values(
            ["factor", "protocol_layer", "method"],
        )
    )
    event_summary = (
        event_table.groupby(
            ["stage", "layer", "attack_type"],
            as_index=False,
        )
        .agg(
            events=("events", "sum"),
            attack_rows=("attack_rows", "sum"),
            rows_per_event_min=("rows_per_event", "min"),
            rows_per_event_max=("rows_per_event", "max"),
            single_row_event_types=("single_row_event", "sum"),
        )
        .sort_values(["stage", "attack_type"])
    )

    if factor_table["protocol_layer"].nunique() < 2:
        raise RuntimeError("protocol layers were incorrectly pooled")
    if event_table["single_row_event"].any():
        raise RuntimeError("single-row event is present; interpretation "
                           "must be revised")

    payload = {
        "status": "PASS_ATTACK_FACTOR_UNIFIED_EVIDENCE_TABLE",
        "protocol": (
            "Stage270 only reads frozen intensity, duration, position, and "
            "event-length artifacts. It normalizes column names while "
            "preserving dataset, protocol layer, evaluation scope, and "
            "source stage. Rows from different protocols are never averaged "
            "together."
        ),
        "metric_factor_count": int(factor_table["factor"].nunique()),
        "factor_count": int(unified["factor"].nunique()),
        "source_stage_count": int(len(source_manifest)),
        "protocol_layer_count": int(
            factor_table["protocol_layer"].nunique()
        ),
        "method_count": int(factor_table["method"].nunique()),
        "metric_rows": int(len(factor_table)),
        "event_length_rows": int(len(event_table)),
        "unified_rows": int(len(unified)),
        "paired_rows": int(len(paired_table)),
        "event_definition": (
            "Each (station_id, attack_type, split, seed) attack block is one "
            "event; event coverage is the fraction of events with at least "
            "one detected row."
        ),
        "event_denominator_note": (
            "Event length is reported from Stage268. Event coverage must be "
            "reported together with row recall, precision, FAR, F1, AUPRC, "
            "and event count."
        ),
        "source_manifest": source_manifest,
        "limitations": [
            "Different protocol layers are kept separate and are not "
            "pooled into a single average.",
            "The local duration and position experiments use five fixed "
            "seeds, while the NOAA full-baseline experiments use four "
            "rolling folds and five seeds.",
            "Attack factors are transparent controlled injections, not "
            "real attack logs.",
            "Intensity does not affect time_shift or stuck amplitude; those "
            "types must be interpreted by mechanism.",
            "The available factors do not cover all possible attack "
            "strengths, durations, or positions.",
        ],
        "input_sha256": hashes,
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    unified.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    method_summary.to_csv(
        TABLES / f"{PREFIX}_method_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_summary.to_csv(
        TABLES / f"{PREFIX}_event_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not paired_table.empty:
        paired_table.to_csv(
            TABLES / f"{PREFIX}_paired.csv",
            index=False,
            encoding="utf-8-sig",
        )

    report = [
        "# 阶段270 攻击因子统一证据表",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 统一表因子数：`{payload['factor_count']}`",
        f"- 带性能指标的因子数：`{payload['metric_factor_count']}`",
        f"- 来源阶段数：`{payload['source_stage_count']}`",
        f"- 协议层级数：`{payload['protocol_layer_count']}`",
        f"- 指标行数：`{payload['metric_rows']}`",
        f"- 事件长度行数：`{payload['event_length_rows']}`",
        f"- 合并长表行数：`{payload['unified_rows']}`",
        f"- 配对差值行数：`{payload['paired_rows']}`",
        "",
        "## 统一规则",
        "",
        "- 每行保留 `factor`、`source_stage`、`protocol_layer`、`evaluation_scope` 和 `method`。",
        "- 不同协议层级只统一字段，不直接求平均，也不互相替代。",
        "- 事件长度来自 Stage268；事件覆盖必须与行召回、precision、FAR、F1、AUPRC 和事件数同时报告。",
        "",
        "## 边界",
        "",
        "- 本阶段只读冻结产物，不训练、不校阈值、不改变任何预测。",
        "- 攻击均为透明受控注入，不是真实攻击日志。",
        "- `time_shift` 和 `stuck` 的机制不随数值强度变化，必须单独解释。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "factor_count": payload["factor_count"],
                "metric_factor_count": payload["metric_factor_count"],
                "metric_rows": payload["metric_rows"],
                "event_length_rows": payload["event_length_rows"],
                "unified_rows": payload["unified_rows"],
                "protocol_layer_count": payload["protocol_layer_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
