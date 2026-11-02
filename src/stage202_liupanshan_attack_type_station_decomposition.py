from __future__ import annotations

"""Stage 202: attack-type and station-cluster decomposition.

Stage 199 found a negative average cross-station transfer result and
Stage 200 showed that a threshold shift was not the main explanation.
This stage asks whether the average hides a stable gain on a particular
attack type or station subset.

The analysis is post-hoc and evaluation-only:

* the Stage 199 ``valid_f1`` selections and thresholds are reused;
* no candidate, threshold, feature, station, attack type, or event is
  selected from held-out test labels;
* for each FAR and canonical attack type, the five seed results are
  averaged within each held-out station;
* an exact station-level sign-flip test is used for the paired difference
  (protected minus baseline);
* Benjamini-Hochberg correction is applied across the 24 FAR-by-type
  primary hypotheses.

The primary endpoint is event coverage.  Row recall is reported as a
secondary endpoint.  A positive result is retained only if the primary
endpoint survives both BH correction and the station-cluster sign-flip
test; otherwise the negative Stage 199 result remains authoritative.

Liupanshan controlled events in the frozen test detail are currently
single-row events.  Event coverage and row recall therefore coincide
exactly in this dataset.  This is a limitation of the event construction,
not evidence that the two endpoints are redundant in general.

The Liupanshan attacks are transparent controlled injections, not real
attack logs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage150_noaa_attack_type_multiple_comparison import (  # noqa: E402
    bh_adjust,
    exact_sign_flip,
)
from stage186_noaa_targeted_nonstuck_feature_response import (  # noqa: E402
    ATTACK_TYPES,
)
from stage193_noaa_persistent_repeat_protection import (  # noqa: E402
    build_run_lengths,
    build_rolling_scores,
    gate_thresholds,
    valid_far_threshold,
)
from stage195_noaa_seed_randomized_placement import (  # noqa: E402
    FAR_GRID,
    PRIMARY_FAR,
    clean_records,
)
from stage199_liupanshan_cross_station_protection_transfer import (  # noqa: E402
    SEEDS,
    local_score_frame,
    prepare_context,
)

STAGE199_PREFIX = (
    "stage199_liupanshan_cross_station_protection_transfer"
)
PREFIX = "stage202_liupanshan_attack_type_station_decomposition"
PRIMARY_RULE = "valid_f1"
PRIMARY_ENDPOINT = "event_coverage_diff"
SECONDARY_ENDPOINT = "row_recall_diff"
STATION_ALPHA = 0.05


def reconstruct_selection_predictions(
    context: dict[str, object],
    selection: dict[str, object],
) -> dict[str, object]:
    """Reconstruct the frozen Stage 199 scores and predictions.

    This duplicates the small score-reconstruction block from Stage 200
    deliberately: Stage 202 needs both valid and held-out test scores,
    while Stage 200 only needs the test-side score.  The selection and
    threshold rules are identical to Stage 199.
    """

    frame = context["frame"]
    source_valid = context["source_valid"]
    held_test = context["held_test"]
    y_valid = frame.loc[source_valid, "is_attack"].astype(bool).to_numpy()
    base_valid = frame.loc[source_valid, "base_score"].to_numpy(dtype=float)
    base_test = frame.loc[held_test, "base_score"].to_numpy(dtype=float)
    gate_valid = frame.loc[source_valid, "repeat_ratio"].to_numpy(dtype=float)
    thresholds = gate_thresholds(gate_valid, ~y_valid)
    runs = build_run_lengths(frame, thresholds)
    rolling = build_rolling_scores(frame)

    max_far = float(selection["max_far"])
    candidate_weight = float(selection["candidate_weight"])
    gate_quantile = float(selection["gate_quantile"])
    run_length = int(selection["run_length"])
    key = (int(selection["window"]), str(selection["aggregation"]))
    protected_valid = (
        runs[gate_quantile][source_valid] >= run_length
        if gate_quantile > 0.0
        else np.zeros(len(base_valid), dtype=bool)
    )
    protected_test = (
        runs[gate_quantile][held_test] >= run_length
        if gate_quantile > 0.0
        else np.zeros(len(base_test), dtype=bool)
    )

    valid_score = base_valid.copy()
    test_score = base_test.copy()
    if candidate_weight > 0.0:
        valid_score = np.where(
            protected_valid,
            base_valid,
            (1.0 - candidate_weight) * base_valid
            + candidate_weight * rolling[key][source_valid],
        )
        test_score = np.where(
            protected_test,
            base_test,
            (1.0 - candidate_weight) * base_test
            + candidate_weight * rolling[key][held_test],
        )

    baseline_threshold = valid_far_threshold(
        base_valid, ~y_valid, max_far
    )
    protected_threshold = valid_far_threshold(
        valid_score, ~y_valid, max_far
    )
    return {
        "baseline_pred": base_test >= baseline_threshold,
        "protected_pred": test_score >= protected_threshold,
        "baseline_threshold": float(baseline_threshold),
        "protected_threshold": float(protected_threshold),
        "protected_rows": int(protected_test.sum()),
        "test_rows": int(len(base_test)),
    }


def attack_type_metrics(
    test_frame: pd.DataFrame,
    baseline_pred: np.ndarray,
    protected_pred: np.ndarray,
) -> pd.DataFrame:
    """Return one row per canonical attack type present in the test split."""

    is_attack = test_frame["is_attack"].astype(bool).to_numpy()
    attack_type = test_frame["attack_type"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for current_type in ATTACK_TYPES:
        mask = is_attack & (attack_type == current_type)
        if not mask.any():
            continue
        event_ids = test_frame.loc[mask, "event_id"].astype(str)
        if event_ids.eq("").any():
            raise ValueError(f"empty event id for attack type {current_type}")
        event_count = int(event_ids.nunique())
        protected_any = (
            pd.DataFrame(
                {
                    "event_id": event_ids.to_numpy(),
                    "selected": protected_pred[mask],
                }
            )
            .groupby("event_id", sort=False)["selected"]
            .any()
        )
        baseline_any = (
            pd.DataFrame(
                {
                    "event_id": event_ids.to_numpy(),
                    "selected": baseline_pred[mask],
                }
            )
            .groupby("event_id", sort=False)["selected"]
            .any()
        )
        protected_event_coverage = float(protected_any.mean())
        baseline_event_coverage = float(baseline_any.mean())
        protected_recall = float(protected_pred[mask].mean())
        baseline_recall = float(baseline_pred[mask].mean())
        rows.append(
            {
                "attack_type": current_type,
                "attack_rows": int(mask.sum()),
                "events": event_count,
                "protected_event_coverage": protected_event_coverage,
                "baseline_event_coverage": baseline_event_coverage,
                "event_coverage_diff": (
                    protected_event_coverage - baseline_event_coverage
                ),
                "protected_row_recall": protected_recall,
                "baseline_row_recall": baseline_recall,
                "row_recall_diff": protected_recall - baseline_recall,
            }
        )
    return pd.DataFrame(rows)


def summarize_station_effects(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate seeds within stations and test across station clusters."""

    station_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for (max_far, attack_type), group in detail.groupby(
        ["max_far", "attack_type"], sort=True
    ):
        station = (
            group.groupby("held_out_station", sort=True)
            .agg(
                seed_count=("seed", "nunique"),
                attack_rows_mean=("attack_rows", "mean"),
                events_mean=("events", "mean"),
                protected_event_coverage_mean=(
                    "protected_event_coverage",
                    "mean",
                ),
                baseline_event_coverage_mean=(
                    "baseline_event_coverage",
                    "mean",
                ),
                event_coverage_diff_mean=("event_coverage_diff", "mean"),
                protected_row_recall_mean=("protected_row_recall", "mean"),
                baseline_row_recall_mean=("baseline_row_recall", "mean"),
                row_recall_diff_mean=("row_recall_diff", "mean"),
            )
            .reset_index()
        )
        station["max_far"] = float(max_far)
        station["attack_type"] = str(attack_type)
        station_rows.extend(station.to_dict(orient="records"))

        event_values = station["event_coverage_diff_mean"].to_numpy(
            dtype=float
        )
        recall_values = station["row_recall_diff_mean"].to_numpy(dtype=float)
        event_p, event_abs = exact_sign_flip(event_values)
        recall_p, recall_abs = exact_sign_flip(recall_values)
        test_rows.append(
            {
                "max_far": float(max_far),
                "attack_type": str(attack_type),
                "stations": int(len(station)),
                "station_names": "|".join(
                    station["held_out_station"].astype(str).tolist()
                ),
                "event_coverage_diff_mean": float(event_values.mean()),
                "event_coverage_diff_std": float(
                    np.std(event_values, ddof=1)
                ),
                "event_coverage_positive_stations": int(
                    (event_values > 0).sum()
                ),
                "event_coverage_negative_stations": int(
                    (event_values < 0).sum()
                ),
                "event_coverage_sign_flip_p": float(event_p),
                "event_coverage_observed_abs": float(event_abs),
                "row_recall_diff_mean": float(recall_values.mean()),
                "row_recall_diff_std": float(
                    np.std(recall_values, ddof=1)
                ),
                "row_recall_positive_stations": int(
                    (recall_values > 0).sum()
                ),
                "row_recall_negative_stations": int(
                    (recall_values < 0).sum()
                ),
                "row_recall_sign_flip_p": float(recall_p),
                "row_recall_observed_abs": float(recall_abs),
            }
        )

    station_frame = pd.DataFrame(station_rows)
    test_frame = pd.DataFrame(test_rows)
    if test_frame.empty:
        raise ValueError("no station-level tests were produced")
    test_frame["event_coverage_bh_q"] = bh_adjust(
        test_frame["event_coverage_sign_flip_p"].tolist()
    )
    test_frame["row_recall_bh_q"] = bh_adjust(
        test_frame["row_recall_sign_flip_p"].tolist()
    )
    test_frame["event_coverage_stable_positive"] = (
        test_frame["event_coverage_diff_mean"].gt(0.0)
        & test_frame["event_coverage_bh_q"].lt(STATION_ALPHA)
    )
    test_frame["row_recall_stable_positive"] = (
        test_frame["row_recall_diff_mean"].gt(0.0)
        & test_frame["row_recall_bh_q"].lt(STATION_ALPHA)
    )
    return station_frame, test_frame


def write_outputs(
    detail: pd.DataFrame,
    station: pd.DataFrame,
    tests: pd.DataFrame,
) -> int:
    detail.to_csv(
        TABLES / f"{PREFIX}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station.to_csv(
        TABLES / f"{PREFIX}_station.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tests.to_csv(
        TABLES / f"{PREFIX}_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary = tests[
        tests["max_far"].eq(float(PRIMARY_FAR))
    ].sort_values("attack_type")
    primary_event = primary[
        primary["event_coverage_stable_positive"]
    ]
    primary_recall = primary[
        primary["row_recall_stable_positive"]
    ]
    if not primary_event.empty or not primary_recall.empty:
        status = "CANDIDATE_TYPE_LEVEL_EFFECT_REQUIRES_REPLICATION"
    else:
        status = "NEGATIVE_RESULT_RETAINED"

    payload = {
        "status": status,
        "protocol": (
            "阶段202只复用阶段199在valid_f1规则下冻结的保护配置和"
            "固定FAR阈值，做事后攻击类型与站点聚类分解。"
            "对每个攻击构造种子和留出站点，在留出站点test上分别计算"
            "保护与无门控基线的逐攻击类型事件覆盖和行召回；"
            "每个FAR×攻击类型先在站点内平均五个固定种子，再以站点为"
            "聚类执行精确sign-flip检验；对四个FAR×六个标准攻击类型的"
            "24个主假设执行Benjamini-Hochberg校正。"
            "不重新选择候选、阈值、特征、站点、攻击类型或事件；"
            "留出站点test只评价；不根据test选择；不修改冻结主模型。"
            "事件覆盖是主终点，行召回是次终点。"
            "攻击为透明受控注入，不是真实攻击日志。"
        ),
        "source_stage": STAGE199_PREFIX,
        "seeds": SEEDS,
        "stations": sorted(
            detail["held_out_station"].astype(str).unique().tolist()
        ),
        "attack_types": ATTACK_TYPES,
        "far_grid": [float(value) for value in FAR_GRID],
        "primary_far": float(PRIMARY_FAR),
        "primary_rule": PRIMARY_RULE,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoint": SECONDARY_ENDPOINT,
        "station_alpha": STATION_ALPHA,
        "detail_rows": int(len(detail)),
        "station_rows": int(len(station)),
        "test_rows": int(len(tests)),
        "primary_event_stable_positive_cells": int(
            len(primary_event)
        ),
        "primary_row_recall_stable_positive_cells": int(
            len(primary_recall)
        ),
        "tests": clean_records(tests),
        "station_effects": clean_records(station),
        "limitations": [
            "六盘山只有单一时段、单一地区的受控攻击数据。",
            "五个攻击构造种子不等于五个独立地区或年份。",
            "冻结test明细中的受控攻击事件当前均为单行事件，"
            "因此事件覆盖与行召回在本数据上数值完全一致；"
            "这是事件构造限制，不代表两个终点在一般情况下等价。",
            "攻击类型缺失的站点不进入该类型站点级检验；"
            "站点数量为5–7，精确sign-flip功效有限。",
            "即使出现类型级正信号，也只能作为需独立复现的诊断假设，"
            "不能替换冻结主模型。",
            "攻击为透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段202 六盘山攻击类型与站点聚类分解",
        "",
        "复用阶段199的valid_f1冻结配置；事件覆盖为主终点，行召回为次终点；"
        "站点内先平均五个固定种子，再做站点级精确sign-flip和BH校正。",
        "",
        "| FAR | 攻击类型 | 站点数 | 事件覆盖差 | 正向站点 | sign-flip p | BH q | 行召回差 | 稳定正事件覆盖 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tests.sort_values(
        ["max_far", "attack_type"]
    ).to_dict(orient="records"):
        lines.append(
            f"| {row['max_far']:.3f} | {row['attack_type']} | "
            f"{int(row['stations'])} | "
            f"{row['event_coverage_diff_mean']:+.6f} | "
            f"{int(row['event_coverage_positive_stations'])}/"
            f"{int(row['stations'])} | "
            f"{row['event_coverage_sign_flip_p']:.6f} | "
            f"{row['event_coverage_bh_q']:.6f} | "
            f"{row['row_recall_diff_mean']:+.6f} | "
            f"{'是' if row['event_coverage_stable_positive'] else '否'} |"
        )
    lines += [
        "",
        f"状态：`{status}`。",
        "",
        "1%FAR下按攻击类型的逐站结果：",
        "",
    ]
    for row in primary.to_dict(orient="records"):
        lines.append(
            f"- {row['attack_type']}：事件覆盖差="
            f"{row['event_coverage_diff_mean']:+.6f}，"
            f"站点级p={row['event_coverage_sign_flip_p']:.6f}，"
            f"BH q={row['event_coverage_bh_q']:.6f}，"
            f"行召回差={row['row_recall_diff_mean']:+.6f}。"
        )
    lines += [
        "",
        "解释边界：如果没有任何类型在BH校正后稳定为正，则阶段199的"
        "平均负结果仍然成立；如果出现正信号，也只能作为需独立外部"
        "复现的诊断假设，不能替换冻结主模型。",
        "",
        "六盘山冻结test中的受控攻击事件当前均为单行事件，因此事件覆盖"
        "与行召回完全一致；这是事件构造限制，不是两个终点在一般情况下"
        "等价。",
        "",
        "该结果是受控攻击数据上的站点聚类分解，不是真实攻击日志验证。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "detail_rows": int(len(detail)),
                "station_rows": int(len(station)),
                "test_rows": int(len(tests)),
                "primary_event_stable_positive_cells": int(
                    len(primary_event)
                ),
                "primary_row_recall_stable_positive_cells": int(
                    len(primary_recall)
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args(argv)

    detail_path = TABLES / f"{PREFIX}_detail.csv"
    if args.summarize_existing:
        detail = pd.read_csv(detail_path)
        station, tests = summarize_station_effects(detail)
        return write_outputs(detail, station, tests)

    selection_path = TABLES / f"{STAGE199_PREFIX}_selections.csv"
    if not selection_path.is_file():
        raise FileNotFoundError(
            "stage202 requires stage199 selections; run stage199 first"
        )
    selections = pd.read_csv(selection_path)
    selections = selections[
        selections["selection_rule"].eq(PRIMARY_RULE)
        & selections["max_far"].isin([float(value) for value in FAR_GRID])
    ].copy()
    if selections.empty:
        raise ValueError("no Stage 199 valid_f1 selections matched FAR grid")

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        seed_selections = selections[
            selections["seed"].eq(int(seed))
        ].copy()
        for station in sorted(
            seed_selections["held_out_station"].astype(str).unique().tolist()
        ):
            current = seed_selections[
                seed_selections["held_out_station"].astype(str).eq(station)
            ].copy()
            if current.empty:
                raise ValueError(
                    f"missing Stage 199 selection for {seed}/{station}"
                )
            frame = local_score_frame(int(seed), station)
            context = prepare_context(frame, station, int(seed))
            test_frame = context["frame"].loc[context["held_test"]].reset_index(
                drop=True
            )
            for selection in current.to_dict(orient="records"):
                predictions = reconstruct_selection_predictions(
                    context, selection
                )
                type_metrics = attack_type_metrics(
                    test_frame,
                    predictions["baseline_pred"],
                    predictions["protected_pred"],
                )
                for metric_row in type_metrics.to_dict(orient="records"):
                    rows.append(
                        {
                            "seed": int(seed),
                            "held_out_station": station,
                            "selection_rule": PRIMARY_RULE,
                            "max_far": float(selection["max_far"]),
                            "gate_quantile": float(
                                selection["gate_quantile"]
                            ),
                            "run_length": int(selection["run_length"]),
                            "window": int(selection["window"]),
                            "aggregation": str(selection["aggregation"]),
                            "candidate_weight": float(
                                selection["candidate_weight"]
                            ),
                            "baseline_threshold": float(
                                predictions["baseline_threshold"]
                            ),
                            "protected_threshold": float(
                                predictions["protected_threshold"]
                            ),
                            "protected_rows": int(
                                predictions["protected_rows"]
                            ),
                            "test_rows": int(predictions["test_rows"]),
                            **metric_row,
                        }
                    )
            print(
                json.dumps(
                    {
                        "seed": int(seed),
                        "held_out_station": station,
                        "status": "completed",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("stage202 produced no attack-type rows")
    station_effects, tests = summarize_station_effects(detail)
    return write_outputs(detail, station_effects, tests)


if __name__ == "__main__":
    raise SystemExit(main())
