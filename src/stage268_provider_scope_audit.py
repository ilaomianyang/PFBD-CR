from __future__ import annotations

"""Stage 268: provider-scope and event-denominator audit.

This stage is read-only with respect to all frozen evaluation artifacts.  It
does not train a model, recalibrate a threshold, or modify any test
prediction.  Its purpose is to prevent stage-record counts from being
reported as independent-provider counts and to keep the SMHI locked
external test separate from the four-provider frozen main result.
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
PREFIX = "stage268_provider_scope_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


LAYERS = {
    "stage248_fair_baseline": {
        "prefix": "stage248_cross_provider_fair_baseline_audit",
        "layer": "frozen_main_confirmatory",
        "independent_provider_count": 4,
        "included_in_cross_provider_equal_weight": True,
        "external_locked_single_provider": False,
    },
    "stage256_lightweight": {
        "prefix": "stage256_lightweight_strong_baselines",
        "layer": "exploratory_supplement",
        "independent_provider_count": 4,
        "included_in_cross_provider_equal_weight": True,
        "external_locked_single_provider": False,
    },
    "stage262_security_domain": {
        "prefix": "stage262_security_domain_baselines",
        "layer": "exploratory_supplement",
        "independent_provider_count": 4,
        "included_in_cross_provider_equal_weight": True,
        "external_locked_single_provider": False,
    },
    "stage259_fmi_locked_external": {
        "prefix": "stage259_fmi_locked_transfer",
        "layer": "external_locked_single_provider",
        "independent_provider_count": 1,
        "included_in_cross_provider_equal_weight": False,
        "external_locked_single_provider": True,
    },
    "stage267_smhi_locked_external": {
        "prefix": "stage267_smhi_locked_transfer",
        "layer": "external_locked_single_provider",
        "independent_provider_count": 1,
        "included_in_cross_provider_equal_weight": False,
        "external_locked_single_provider": True,
    },
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _event_audit(
    stage: str,
    info: dict[str, object],
) -> pd.DataFrame:
    prefix = str(info["prefix"])
    type_detail = _load_csv(TABLES / f"{prefix}_type_detail.csv")
    required = {
        "source",
        "method",
        "attack_type",
        "events",
        "attack_rows",
        "row_recall",
        "event_coverage",
    }
    missing = required.difference(type_detail.columns)
    if missing:
        raise RuntimeError(
            f"{stage} type detail missing columns: {sorted(missing)}"
        )
    rows: list[dict[str, object]] = []
    for (source, method, attack_type), group in type_detail.groupby(
        ["source", "method", "attack_type"],
        sort=True,
    ):
        events = int(group["events"].sum())
        attack_rows = int(group["attack_rows"].sum())
        if events <= 0:
            continue
        if attack_rows <= 0:
            raise RuntimeError(
                f"{stage}/{source}/{method}/{attack_type} has no attack rows"
            )
        if attack_rows % events != 0:
            raise RuntimeError(
                f"{stage}/{source}/{method}/{attack_type} event length "
                "is not integral"
            )
        rows.append(
            {
                "stage": stage,
                "layer": str(info["layer"]),
                "source": str(source),
                "method": str(method),
                "attack_type": str(attack_type),
                "events": events,
                "attack_rows": attack_rows,
                "rows_per_event": attack_rows / events,
                "single_row_event": bool(attack_rows == events),
                "row_recall_mean": float(group["row_recall"].mean()),
                "event_coverage_mean": float(
                    group["event_coverage"].mean()
                ),
                "event_denominator_definition": (
                    "each (station_id, attack_type, split, seed) block"
                ),
                "event_coverage_definition": (
                    "an event is detected if any row in the event is "
                    "predicted as an attack"
                ),
            }
        )
    return pd.DataFrame(rows)


def _provider_scope(
    stage_audits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage, info in LAYERS.items():
        prefix = str(info["prefix"])
        evaluation = _load_csv(TABLES / f"{prefix}_evaluation.csv")
        station = _load_csv(TABLES / f"{prefix}_station_detail.csv")
        sources = sorted(evaluation["source"].astype(str).unique().tolist())
        if len(sources) != int(info["independent_provider_count"]):
            raise RuntimeError(
                f"{stage} independent provider count mismatch: {sources}"
            )
        station_count = int(
            station[["source", "station_id"]].drop_duplicates().shape[0]
        )
        event_rows = stage_audits[stage]
        rows.append(
            {
                "stage": stage,
                "layer": str(info["layer"]),
                "independent_provider_count": int(
                    info["independent_provider_count"]
                ),
                "source_count_in_stage": int(evaluation["source"].nunique()),
                "sources": ";".join(sources),
                "station_count": station_count,
                "method_count": int(evaluation["method"].nunique()),
                "seed_count": int(evaluation["seed"].nunique()),
                "included_in_cross_provider_equal_weight": bool(
                    info["included_in_cross_provider_equal_weight"]
                ),
                "external_locked_single_provider": bool(
                    info["external_locked_single_provider"]
                ),
                "event_rows": int(len(event_rows)),
                "events": int(event_rows["events"].sum()),
                "attack_rows": int(event_rows["attack_rows"].sum()),
                "single_row_event_types": int(
                    event_rows["single_row_event"].sum()
                ),
                "scope_note": (
                    "Included in the multi-provider equal-weight summary."
                    if info["included_in_cross_provider_equal_weight"]
                    else "Evaluated as an isolated locked external provider; "
                    "not merged into the multi-provider equal-weight summary."
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stage_audits = {
        stage: _event_audit(stage, info)
        for stage, info in LAYERS.items()
    }
    event_audit = pd.concat(
        stage_audits.values(),
        ignore_index=True,
    )
    provider_scope = _provider_scope(stage_audits)
    event_summary = (
        event_audit.groupby(
            ["stage", "layer"],
            as_index=False,
        )
        .agg(
            source_count=("source", "nunique"),
            method_count=("method", "nunique"),
            attack_type_count=("attack_type", "nunique"),
            event_rows=("events", "size"),
            events=("events", "sum"),
            attack_rows=("attack_rows", "sum"),
            mean_rows_per_event=("rows_per_event", "mean"),
            min_rows_per_event=("rows_per_event", "min"),
            max_rows_per_event=("rows_per_event", "max"),
            single_row_event_types=("single_row_event", "sum"),
        )
        .sort_values(["layer", "stage"])
    )
    frozen_main = provider_scope.loc[
        provider_scope["stage"].eq("stage248_fair_baseline")
    ].iloc[0]
    fmi = provider_scope.loc[
        provider_scope["stage"].eq("stage259_fmi_locked_external")
    ].iloc[0]
    smhi = provider_scope.loc[
        provider_scope["stage"].eq("stage267_smhi_locked_external")
    ].iloc[0]

    input_hashes: dict[str, str] = {}
    for info in LAYERS.values():
        prefix = str(info["prefix"])
        for suffix in (
            "evaluation.csv",
            "station_detail.csv",
            "type_detail.csv",
        ):
            path = TABLES / f"{prefix}_{suffix}"
            input_hashes[str(path.relative_to(ROOT))] = sha256_file(path)

    payload = {
        "status": "PASS_PROVIDER_SCOPE_AND_EVENT_DENOMINATOR_AUDIT",
        "protocol": (
            "Stage268 only reads frozen evaluation, station-detail, and "
            "type-detail artifacts. It does not train, recalibrate, or "
            "change any prediction. It separates the four-provider frozen "
            "main result, exploratory multi-provider supplements, and the "
            "single-provider FMI/SMHI locked external tests.",
        ),
        "frozen_main_independent_provider_count": int(
            frozen_main["independent_provider_count"]
        ),
        "frozen_main_sources": str(frozen_main["sources"]),
        "frozen_main_station_count": int(frozen_main["station_count"]),
        "fmi_external_independent_provider_count": int(
            fmi["independent_provider_count"]
        ),
        "fmi_external_sources": str(fmi["sources"]),
        "fmi_external_station_count": int(fmi["station_count"]),
        "smhi_external_independent_provider_count": int(
            smhi["independent_provider_count"]
        ),
        "smhi_external_sources": str(smhi["sources"]),
        "smhi_external_station_count": int(smhi["station_count"]),
        "combined_named_independent_provider_count": int(
            frozen_main["independent_provider_count"]
            + fmi["independent_provider_count"]
            + smhi["independent_provider_count"]
        ),
        "stage_record_count_is_not_provider_count": True,
        "cross_provider_equal_weight_rule": (
            "Only the four frozen main providers are used for the frozen "
            "cross-provider equal-weight result. FMI and SMHI are isolated "
            "single-provider locked external tests and are never counted as "
            "additional providers inside that equal-weight estimate."
        ),
        "provider_scope": provider_scope.to_dict("records"),
        "event_summary": event_summary.to_dict("records"),
        "event_definition": (
            "Each (station_id, attack_type, split, seed) attack block is one "
            "event; event coverage is the fraction of events with at least "
            "one detected row."
        ),
        "event_denominator_warning": (
            "Event coverage must be reported with row recall, precision, FAR, "
            "F1, AUPRC, event count, and event-length distribution. It must "
            "not be reported alone."
        ),
        "single_row_event_count": int(
            event_audit["single_row_event"].sum()
        ),
        "input_sha256": input_hashes,
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    if payload["combined_named_independent_provider_count"] != 6:
        raise RuntimeError("unexpected combined provider count")
    if not event_audit["single_row_event"].eq(False).all():
        raise RuntimeError("single-row event detected; audit interpretation "
                           "must be revised")

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    provider_scope.to_csv(
        TABLES / f"{PREFIX}_provider_scope.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_audit.to_csv(
        TABLES / f"{PREFIX}_event_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_summary.to_csv(
        TABLES / f"{PREFIX}_event_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = [
        "# 阶段268 独立机构口径与事件分母审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 冻结主结果独立机构数：`{payload['frozen_main_independent_provider_count']}`",
        f"- 冻结主结果机构：`{payload['frozen_main_sources']}`",
        f"- FMI：`1` 个单机构锁定外部验证，机构 `{payload['fmi_external_sources']}`",
        f"- SMHI：`1` 个单机构锁定外部验证，机构 `{payload['smhi_external_sources']}`",
        f"- 合并命名独立机构数：`{payload['combined_named_independent_provider_count']}`",
        "- Stage263 的 `provider_count=12` 是阶段记录数，不是 12 个独立机构。",
        "- FMI 和 SMHI 不并入四机构等权主估计。",
        f"- 单行事件类型数：`{payload['single_row_event_count']}`",
        "",
        "## 边界",
        "",
        "- 本阶段只读冻结产物，不训练、不校阈值、不改变任何预测。",
        "- 事件覆盖必须与行召回、precision、FAR、F1、AUPRC 和事件长度同时报告。",
        "- 六个命名独立机构仍不足以证明机构无关或站点无关。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "frozen_main_provider_count": payload[
                    "frozen_main_independent_provider_count"
                ],
                "combined_named_provider_count": payload[
                    "combined_named_independent_provider_count"
                ],
                "event_rows": int(len(event_audit)),
                "single_row_event_count": payload[
                    "single_row_event_count"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
