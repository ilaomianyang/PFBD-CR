from __future__ import annotations

"""Stage 313: cross-locked-test win/loss and non-stuck stability audit.

This stage is deliberately read-only.  It unifies the three locked-transfer
comparison ledgers that already exist:

* Stage297: NOAA Japan ISD network (same provider as the frozen NOAA main
  result).
* Stage307: MET Norway network (supplementary provider, not part of the
  frozen four-provider main result).
* Stage312: disjoint MET Norway network (same provider as Stage307, different
  station network).

It also aggregates the available MET Norway attack-type detail tables for the
two Trust candidates and the five non-stuck attack types.  Japan has no
attack-type detail table in the frozen outputs, so that gap is recorded
explicitly rather than filled with a synthetic value.

The output is boundary evidence only.  It cannot create a new provider, close
P0-1/P0-2/P0-3/P0-4, or support a universal-superiority claim.
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
PREFIX = "stage313_cross_locked_test_win_loss_matrix"
PROTOCOL_ID = "TMQC-CROSS-LOCKED-TEST-WIN-LOSS-20260922-v1.0"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


NONSTUCK_TYPES = [
    "spike",
    "offset",
    "drift",
    "time_shift",
    "gaussian_noise",
]
TRUST_METHODS = [
    "trust_valid_f1",
    "trust_nonstuck_event_macro",
]
NONINFERIORITY_MARGIN = -0.02

SOURCES = [
    {
        "source_stage": "stage297_japan_isd_locked_transfer",
        "comparison_file": (
            "stage297_japan_isd_locked_transfer_comparisons.csv"
        ),
        "provider_equal_file": (
            "stage297_japan_isd_locked_transfer_provider_equal.csv"
        ),
        "type_detail_file": None,
        "locked_test": "NOAA_JAPAN_ISD_LOCKED_TEST",
        "protocol_layer": "same_provider_noaa_network",
        "provider_independence": (
            "same_as_frozen_main_noaa_provider"
        ),
        "provider_counted_as_new": False,
    },
    {
        "source_stage": "stage307_met_norway_locked_transfer",
        "comparison_file": (
            "stage307_met_norway_locked_transfer_comparisons.csv"
        ),
        "provider_equal_file": (
            "stage307_met_norway_locked_transfer_provider_equal.csv"
        ),
        "type_detail_file": (
            "stage307_met_norway_locked_transfer_type_detail.csv"
        ),
        "locked_test": "MET_NORWAY_LOCKED_TEST",
        "protocol_layer": "supplementary_provider_locked_transfer",
        "provider_independence": (
            "separate_provider_not_in_frozen_main_four"
        ),
        "provider_counted_as_new": False,
    },
    {
        "source_stage": "stage312_met_norway_disjoint_locked_transfer",
        "comparison_file": (
            "stage312_met_norway_disjoint_locked_transfer_comparisons.csv"
        ),
        "provider_equal_file": (
            "stage312_met_norway_disjoint_locked_transfer_provider_equal.csv"
        ),
        "type_detail_file": (
            "stage312_met_norway_disjoint_locked_transfer_type_detail.csv"
        ),
        "locked_test": "MET_NORWAY_DISJOINT_LOCKED_TEST",
        "protocol_layer": "same_provider_disjoint_network",
        "provider_independence": (
            "same_as_supplementary_met_norway_provider"
        ),
        "provider_counted_as_new": False,
    },
]

COMMON_COMPARISON_COLUMNS = [
    "candidate",
    "baseline",
    "station_count",
    "paired_rows",
    "iterations",
    "f1_difference",
    "f1_ci95_low",
    "f1_ci95_high",
    "event_coverage_difference",
    "event_ci95_low",
    "event_ci95_high",
    "f1_bootstrap_two_sided_p",
    "event_coverage_bootstrap_two_sided_p",
    "f1_bootstrap_bh_q",
    "event_coverage_bootstrap_bh_q",
]


def _read_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_json(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _direction(value: float) -> str:
    if float(value) > 0.0:
        return "trust_higher"
    if float(value) < 0.0:
        return "baseline_higher"
    return "tie"


def _normalize_comparisons(source: dict) -> pd.DataFrame:
    frame = _read_csv(str(source["comparison_file"]))
    missing = sorted(set(COMMON_COMPARISON_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{source['source_stage']} missing comparison columns: {missing}"
        )
    frame = frame[COMMON_COMPARISON_COLUMNS].copy()
    frame.insert(0, "source_stage", str(source["source_stage"]))
    frame.insert(1, "locked_test", str(source["locked_test"]))
    frame.insert(2, "protocol_layer", str(source["protocol_layer"]))
    frame.insert(
        3,
        "provider_independence",
        str(source["provider_independence"]),
    )
    frame.insert(
        4,
        "provider_counted_as_new",
        bool(source["provider_counted_as_new"]),
    )
    for column in [
        "f1_difference",
        "f1_ci95_low",
        "f1_ci95_high",
        "event_coverage_difference",
        "event_ci95_low",
        "event_ci95_high",
        "f1_bootstrap_bh_q",
        "event_coverage_bootstrap_bh_q",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["f1_direction"] = frame["f1_difference"].map(_direction)
    frame["event_direction"] = frame[
        "event_coverage_difference"
    ].map(_direction)
    frame["f1_advantage_supported"] = (
        frame["f1_ci95_low"].gt(0.0)
        & frame["f1_bootstrap_bh_q"].le(0.05)
    )
    frame["event_advantage_supported"] = (
        frame["event_ci95_low"].gt(0.0)
        & frame["event_coverage_bootstrap_bh_q"].le(0.05)
    )
    frame["f1_noninferior_supported"] = frame["f1_ci95_low"].ge(
        NONINFERIORITY_MARGIN
    )
    frame["event_noninferior_supported"] = frame["event_ci95_low"].ge(
        NONINFERIORITY_MARGIN
    )
    frame["both_endpoints_noninferior"] = (
        frame["f1_noninferior_supported"]
        & frame["event_noninferior_supported"]
    )
    # The pre-registered primary endpoint is event coverage.  A comparison
    # passes the normalized primary gate when event coverage is either
    # significantly higher or within the frozen non-inferiority margin.
    frame["normalized_primary_pass"] = (
        frame["event_advantage_supported"]
        | frame["event_noninferior_supported"]
    )
    frame["normalized_primary_direction"] = np.where(
        frame["event_advantage_supported"],
        "trust_superior",
        np.where(
            frame["event_noninferior_supported"],
            "trust_noninferior",
            "baseline_superior",
        ),
    )
    frame["claim_ceiling"] = (
        "LOCKED_TEST_BOUNDARY_EVIDENCE_ONLY"
        if not bool(source["provider_counted_as_new"])
        else "UNEXPECTED_NEW_PROVIDER"
    )
    return frame


def _nonstuck_rows(source: dict) -> pd.DataFrame:
    if not source.get("type_detail_file"):
        return pd.DataFrame()
    frame = _read_csv(str(source["type_detail_file"]))
    required = {
        "source",
        "seed",
        "method",
        "attack_type",
        "attack_rows",
        "detected_rows",
        "row_recall",
        "events",
        "event_coverage",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{source['source_stage']} missing type-detail columns: {missing}"
        )
    frame = frame[
        frame["method"].isin(TRUST_METHODS)
        & frame["attack_type"].isin(NONSTUCK_TYPES)
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    for column in [
        "attack_rows",
        "detected_rows",
        "row_recall",
        "events",
        "event_coverage",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["detected_events"] = frame["events"] * frame["event_coverage"]
    rows: list[dict[str, object]] = []
    for (method, attack_type), group in frame.groupby(
        ["method", "attack_type"], sort=True
    ):
        events = float(group["events"].sum())
        attack_rows = float(group["attack_rows"].sum())
        detected_rows = float(group["detected_rows"].sum())
        detected_events = float(group["detected_events"].sum())
        rows.append(
            {
                "source_stage": str(source["source_stage"]),
                "locked_test": str(source["locked_test"]),
                "protocol_layer": str(source["protocol_layer"]),
                "method": method,
                "attack_type": attack_type,
                "seed_count": int(group["seed"].nunique()),
                "events": events,
                "attack_rows": attack_rows,
                "detected_rows": detected_rows,
                "detected_events": detected_events,
                "row_recall": (
                    detected_rows / attack_rows if attack_rows else 0.0
                ),
                "event_coverage": (
                    detected_events / events if events else 0.0
                ),
                "min_seed_event_coverage": float(
                    group["event_coverage"].min()
                ),
                "min_seed_row_recall": float(
                    group["row_recall"].min()
                ),
            }
        )
    return pd.DataFrame(rows)


def _cross_network_nonstuck(
    nonstuck: pd.DataFrame,
) -> pd.DataFrame:
    if nonstuck.empty:
        return pd.DataFrame()
    stage307 = nonstuck[
        nonstuck["source_stage"].eq(
            "stage307_met_norway_locked_transfer"
        )
    ]
    stage312 = nonstuck[
        nonstuck["source_stage"].eq(
            "stage312_met_norway_disjoint_locked_transfer"
        )
    ]
    rows: list[dict[str, object]] = []
    for method in TRUST_METHODS:
        for attack_type in NONSTUCK_TYPES:
            left = stage307[
                stage307["method"].eq(method)
                & stage307["attack_type"].eq(attack_type)
            ]
            right = stage312[
                stage312["method"].eq(method)
                & stage312["attack_type"].eq(attack_type)
            ]
            if left.empty or right.empty:
                continue
            left_row = left.iloc[0]
            right_row = right.iloc[0]
            rows.append(
                {
                    "method": method,
                    "attack_type": attack_type,
                    "stage307_event_coverage": float(
                        left_row["event_coverage"]
                    ),
                    "stage307_row_recall": float(left_row["row_recall"]),
                    "stage307_min_seed_event_coverage": float(
                        left_row["min_seed_event_coverage"]
                    ),
                    "stage312_event_coverage": float(
                        right_row["event_coverage"]
                    ),
                    "stage312_row_recall": float(
                        right_row["row_recall"]
                    ),
                    "stage312_min_seed_event_coverage": float(
                        right_row["min_seed_event_coverage"]
                    ),
                    "event_direction_agreement": (
                        _direction(
                            float(left_row["event_coverage"])
                            - float(right_row["event_coverage"])
                        )
                        == "tie"
                    ),
                    "both_networks_event_coverage_positive": bool(
                        float(left_row["event_coverage"]) > 0.0
                        and float(right_row["event_coverage"]) > 0.0
                    ),
                    "both_networks_row_recall_positive": bool(
                        float(left_row["row_recall"]) > 0.0
                        and float(right_row["row_recall"]) > 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _cross_test_direction(
    unified: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, baseline), group in unified.groupby(
        ["candidate", "baseline"], sort=True
    ):
        f1_trust = int(group["f1_direction"].eq("trust_higher").sum())
        f1_baseline = int(
            group["f1_direction"].eq("baseline_higher").sum()
        )
        event_trust = int(
            group["event_direction"].eq("trust_higher").sum()
        )
        event_baseline = int(
            group["event_direction"].eq("baseline_higher").sum()
        )
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "locked_test_count": int(
                    group["source_stage"].nunique()
                ),
                "f1_trust_higher_count": f1_trust,
                "f1_baseline_higher_count": f1_baseline,
                "f1_tie_count": int(
                    group["f1_direction"].eq("tie").sum()
                ),
                "event_trust_higher_count": event_trust,
                "event_baseline_higher_count": event_baseline,
                "event_tie_count": int(
                    group["event_direction"].eq("tie").sum()
                ),
                "f1_direction_stable": bool(
                    f1_trust == 0 or f1_baseline == 0
                ),
                "event_direction_stable": bool(
                    event_trust == 0 or event_baseline == 0
                ),
                "primary_pass_count": int(
                    group["normalized_primary_pass"].sum()
                ),
                "all_tests_primary_pass": bool(
                    group["normalized_primary_pass"].all()
                ),
                "f1_difference_min": float(
                    group["f1_difference"].min()
                ),
                "f1_difference_max": float(
                    group["f1_difference"].max()
                ),
                "event_difference_min": float(
                    group["event_coverage_difference"].min()
                ),
                "event_difference_max": float(
                    group["event_coverage_difference"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    frames: list[pd.DataFrame] = []
    provider_frames: list[pd.DataFrame] = []
    nonstuck_frames: list[pd.DataFrame] = []
    source_meta: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}

    for source in SOURCES:
        stage = _read_json(str(source["source_stage"]))
        normalized = _normalize_comparisons(source)
        frames.append(normalized)

        provider = _read_csv(str(source["provider_equal_file"]))
        provider.insert(0, "source_stage", str(source["source_stage"]))
        provider.insert(1, "locked_test", str(source["locked_test"]))
        provider.insert(2, "protocol_layer", str(source["protocol_layer"]))
        provider.insert(
            3,
            "provider_independence",
            str(source["provider_independence"]),
        )
        provider_frames.append(provider)

        type_detail_available = bool(source.get("type_detail_file"))
        if type_detail_available:
            nonstuck_frames.append(_nonstuck_rows(source))

        input_hashes[str(source["comparison_file"])] = sha256_file(
            TABLES / str(source["comparison_file"])
        )
        input_hashes[str(source["provider_equal_file"])] = sha256_file(
            TABLES / str(source["provider_equal_file"])
        )
        if type_detail_available:
            input_hashes[str(source["type_detail_file"])] = sha256_file(
                TABLES / str(source["type_detail_file"])
            )
        source_meta.append(
            {
                "source_stage": source["source_stage"],
                "status": stage.get("status"),
                "locked_test": source["locked_test"],
                "protocol_layer": source["protocol_layer"],
                "provider_independence": source[
                    "provider_independence"
                ],
                "provider_counted_as_new": bool(
                    source["provider_counted_as_new"]
                ),
                "station_count": stage.get("station_count"),
                "seed_count": stage.get("seed_count"),
                "method_count": stage.get("method_count"),
                "comparison_rows": int(len(normalized)),
                "normalized_primary_pass_count": int(
                    normalized["normalized_primary_pass"].sum()
                ),
                "type_detail_available": type_detail_available,
            }
        )

    unified = pd.concat(frames, ignore_index=True)
    provider_unified = pd.concat(provider_frames, ignore_index=True)
    nonstuck = (
        pd.concat(nonstuck_frames, ignore_index=True)
        if nonstuck_frames
        else pd.DataFrame()
    )
    if unified.duplicated(
        ["source_stage", "candidate", "baseline"]
    ).any():
        raise RuntimeError("duplicate source/candidate/baseline rows")

    direction = _cross_test_direction(unified)
    cross_network = _cross_network_nonstuck(nonstuck)
    ledger = (
        unified.groupby(
            [
                "source_stage",
                "locked_test",
                "protocol_layer",
                "provider_independence",
                "candidate",
                "baseline",
            ],
            as_index=False,
        )
        .agg(
            comparison_rows=("baseline", "size"),
            f1_advantage_supported=("f1_advantage_supported", "sum"),
            event_advantage_supported=(
                "event_advantage_supported",
                "sum",
            ),
            f1_noninferior_supported=(
                "f1_noninferior_supported",
                "sum",
            ),
            event_noninferior_supported=(
                "event_noninferior_supported",
                "sum",
            ),
            normalized_primary_pass=(
                "normalized_primary_pass",
                "sum",
            ),
            f1_difference_mean=("f1_difference", "mean"),
            event_coverage_difference_mean=(
                "event_coverage_difference",
                "mean",
            ),
        )
        .sort_values(
            ["source_stage", "candidate", "baseline"]
        )
    )

    nonstuck_zero_event_cells = 0
    nonstuck_zero_row_cells = 0
    if not nonstuck.empty:
        nonstuck_zero_event_cells = int(
            nonstuck["event_coverage"].le(0.0).sum()
        )
        nonstuck_zero_row_cells = int(
            nonstuck["row_recall"].le(0.0).sum()
        )

    payload = {
        "status": "PASS_CROSS_LOCKED_TEST_WIN_LOSS_AND_NONSTUCK_AUDIT",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "Stage313 only reads the frozen Stage297/307/312 comparison and "
            "type-detail tables. It does not train, recalibrate, alter "
            "predictions, or create a new locked test. It separates same-"
            "provider networks from the supplementary MET Norway provider "
            "and refuses to close P0-1 through P0-4."
        ),
        "source_count": len(SOURCES),
        "locked_test_count": int(unified["source_stage"].nunique()),
        "comparison_row_count": int(len(unified)),
        "candidate_count": int(unified["candidate"].nunique()),
        "baseline_count": int(unified["baseline"].nunique()),
        "normalized_primary_pass_count": int(
            unified["normalized_primary_pass"].sum()
        ),
        "normalized_primary_fail_count": int(
            (~unified["normalized_primary_pass"]).sum()
        ),
        "all_locked_tests_primary_pass": bool(
            unified.groupby("source_stage")[
                "normalized_primary_pass"
            ].all().all()
        ),
        "all_baselines_supported_on_every_locked_test": bool(
            direction["all_tests_primary_pass"].all()
        ),
        "f1_direction_stable_baseline_count": int(
            direction["f1_direction_stable"].sum()
        ),
        "event_direction_stable_baseline_count": int(
            direction["event_direction_stable"].sum()
        ),
        "baseline_count_with_all_locked_tests": int(len(direction)),
        "nonstuck_type_row_count": int(len(nonstuck)),
        "cross_network_nonstuck_row_count": int(len(cross_network)),
        "nonstuck_zero_event_cell_count": nonstuck_zero_event_cells,
        "nonstuck_zero_row_cell_count": nonstuck_zero_row_cells,
        "japan_nonstuck_type_detail_available": False,
        "met_norway_nonstuck_type_detail_available": bool(
            not nonstuck.empty
        ),
        "same_provider_networks_counted_as_new_provider": False,
        "new_provider_added": False,
        "p0_1_status_after": "OPEN",
        "p0_2_status_after": "OPEN",
        "p0_3_status_after": "OPEN",
        "p0_4_status_after": "OPEN",
        "claim_ceiling": "CROSS_LOCKED_TEST_BOUNDARY_EVIDENCE_ONLY",
        "sources": source_meta,
        "input_sha256": input_hashes,
        "limitations": [
            "The Japan locked transfer has comparison-level results but no "
            "attack-type detail table in the frozen outputs.",
            "The MET Norway networks are same-provider networks and cannot "
            "close provider-independence claims.",
            "The disjoint MET Norway network has adverse One-Class SVM and "
            "reconstruction-baseline results that must be retained.",
            "All attacks are transparent controlled injections, not real "
            "production incidents.",
            "This stage does not add a new provider, rerun a model, or "
            "change any threshold.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    unified.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ledger.to_csv(
        TABLES / f"{PREFIX}_ledger.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provider_unified.to_csv(
        TABLES / f"{PREFIX}_provider_equal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    direction.to_csv(
        TABLES / f"{PREFIX}_direction_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nonstuck.to_csv(
        TABLES / f"{PREFIX}_nonstuck_type.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cross_network.to_csv(
        TABLES / f"{PREFIX}_nonstuck_cross_network.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = [
        "# Stage 313 跨锁定测试胜负与非 stuck 稳定性审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 锁定测试数：`{payload['locked_test_count']}`",
        f"- 比较行：`{payload['comparison_row_count']}`",
        f"- 归一化主终点通过：`{payload['normalized_primary_pass_count']}`",
        f"- 归一化主终点未通过：`{payload['normalized_primary_fail_count']}`",
        f"- 全部锁定测试主终点通过：`{payload['all_locked_tests_primary_pass']}`",
        f"- 全部基线在全部锁定测试通过：`{payload['all_baselines_supported_on_every_locked_test']}`",
        f"- 非 stuck 类型行：`{payload['nonstuck_type_row_count']}`",
        f"- 非 stuck 跨网络行：`{payload['cross_network_nonstuck_row_count']}`",
        f"- 非 stuck 零事件覆盖单元：`{payload['nonstuck_zero_event_cell_count']}`",
        f"- 非 stuck 零行召回单元：`{payload['nonstuck_zero_row_cell_count']}`",
        f"- 日本 type detail 可用：`{payload['japan_nonstuck_type_detail_available']}`",
        f"- P0-1：`{payload['p0_1_status_after']}`",
        f"- P0-2：`{payload['p0_2_status_after']}`",
        f"- P0-3：`{payload['p0_3_status_after']}`",
        f"- P0-4：`{payload['p0_4_status_after']}`",
        "",
        "## 协议层",
        "",
        "| 来源 | 锁定测试 | 协议层 | provider 独立性 | 比较行 | 主终点通过 | type detail |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in source_meta:
        report.append(
            f"| {row['source_stage']} | {row['locked_test']} | "
            f"{row['protocol_layer']} | {row['provider_independence']} | "
            f"{row['comparison_rows']} | "
            f"{row['normalized_primary_pass_count']} | "
            f"{row['type_detail_available']} |"
        )
    report += [
        "",
        "## 边界",
        "",
        "- Stage297 仍属于 NOAA provider，只是日本站点网络。",
        "- Stage307 是补充 provider，不属于冻结四机构主结果。",
        "- Stage312 与 Stage307 属于同一 MET Norway provider，只是站网不相交。",
        "- 本阶段不新增 provider、不重训、不调阈值、不生成新性能结果。",
        "- 非 stuck 的零覆盖单元和不利方向必须保留，不能据此升级主声称。",
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": payload["status"],
                "locked_tests": payload["locked_test_count"],
                "comparison_rows": payload["comparison_row_count"],
                "normalized_primary_pass": payload[
                    "normalized_primary_pass_count"
                ],
                "normalized_primary_fail": payload[
                    "normalized_primary_fail_count"
                ],
                "nonstuck_rows": payload["nonstuck_type_row_count"],
                "nonstuck_zero_event_cells": payload[
                    "nonstuck_zero_event_cell_count"
                ],
                "p0_1": payload["p0_1_status_after"],
                "p0_2": payload["p0_2_status_after"],
                "p0_3": payload["p0_3_status_after"],
                "p0_4": payload["p0_4_status_after"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
