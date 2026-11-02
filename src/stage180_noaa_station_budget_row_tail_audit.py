from __future__ import annotations

"""Stage 180: row-level audit of fixed station alarm budgets.

Stage 179 showed that the main q=0.995 loss for the size/uniform station
budgets occurs inside stations rather than across stations.  This stage
replays the frozen expanded risk exactly as stage 178 does, but keeps only
the pre-fixed low-FAR operating points q=0.995 and q=0.999 and writes the
selected test rows.

The audit records, for every fold, seed, q, station, and allocation rule:

* how many attack rows of each attack type are inside the fixed budget;
* whether the selected rows are the station's highest-risk rows;
* the risk percentile of attack rows that fall outside the budget;
* event-level coverage for the controlled attack events.

No model, alpha, window, q, or allocation rule is selected from test.  The
oracle-style row counts are diagnostics only and cannot be deployed.
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
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"
AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
FROZEN = TABLES / "stage141_noaa_frozen_external_validation_selected.csv"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
Q_GRID = [0.995, 0.999]
ALLOCATION_RULES = ["size", "uniform", "risk"]
METHOD = "TrustMeteo-QC frozen expanded"
ATTACK_TYPES = [
    "spike",
    "offset",
    "drift",
    "time_shift",
    "stuck",
    "gaussian_noise",
]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage141_noaa_frozen_external_validation import add_components  # noqa: E402
from stage151_noaa_attack_intensity_sensitivity import build_fold_scaled  # noqa: E402
from stage178_noaa_station_calibrated_alarm_budget import (  # noqa: E402
    station_budget_mask,
    top_k_mask,
)
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage09b_robust_fusion import smooth_risk  # noqa: E402


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def clean_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        cleaned: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, float) and not np.isfinite(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def event_summary(
    rows: pd.DataFrame,
    q: float,
    allocation_rule: str,
    seed: int,
    fold: str,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    attacks = rows[rows["is_attack"]].copy()
    if attacks.empty:
        return out
    attacks["event_key"] = (
        fold
        + "|"
        + attacks["station_key"].astype(str)
        + "|"
        + attacks["attack_type"].astype(str)
        + "|"
        + str(seed)
    )
    for event_id, group in attacks.groupby("event_key", sort=True):
        attack_type = str(group["attack_type"].iloc[0])
        selected = group["selected"].astype(bool)
        covered = int(bool(selected.any()))
        out.append(
            {
                "fold": fold,
                "seed": int(seed),
                "q": float(q),
                "allocation_rule": allocation_rule,
                "event_id": event_id,
                "attack_type": attack_type,
                "station_id": str(group["station_key"].iloc[0]),
                "event_rows": int(len(group)),
                "selected_rows": int(selected.sum()),
                "event_recall": float(selected.mean()),
                "event_covered": covered,
                "risk_percentile_min": float(group["risk_percentile"].min()),
                "risk_percentile_median": float(group["risk_percentile"].median()),
                "risk_percentile_max": float(group["risk_percentile"].max()),
            }
        )
    return out


def build_rows() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {
        (int(row["year"]), str(row["station_id"]))
        for row in audit["detail"]
        if row.get("eligible_for_extended_validation")
    }
    years = sorted(int(x) for x in source["year"].unique())
    stations = {
        station
        for station in source["station_id"].astype(str).unique()
        if all((year, station) in eligible for year in years)
    }
    frozen = (
        pd.read_csv(FROZEN)
        .loc[lambda x: x["method"].eq(METHOD)]
        .set_index(["fold", "seed"])
    )
    row_frames: list[pd.DataFrame] = []
    station_frames: list[pd.DataFrame] = []
    station_type_frames: list[pd.DataFrame] = []
    event_rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            alpha = float(frozen.loc[(fold, seed), "alpha_base"])
            window = int(frozen.loc[(fold, seed), "window"])
            frame = build_fold_scaled(
                source, seed, train_years, valid_year, test_year, stations, 1.0
            )
            scored, _ = score(frame)
            scored = add_components(scored)
            train = scored["split"].eq("train").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            normal_train = train & (
                ~scored["combined_is_attack"].astype(bool).to_numpy()
            )
            base = scored["base_fused_risk"].to_numpy(dtype=float)
            structural = scored["structural_risk"].to_numpy(dtype=float)
            raw_risk = alpha * base + (1.0 - alpha) * structural
            risk = np.full(len(scored), np.nan, dtype=float)
            for split_name in ["train", "valid", "test"]:
                split_mask = scored["split"].eq(split_name).to_numpy()
                risk[split_mask] = smooth_risk(
                    scored.loc[split_mask], raw_risk[split_mask], window
                )
            test_frame = scored.loc[test].reset_index(drop=True).copy()
            test_risk = risk[test]
            test_stations = test_frame["station_key"].astype(str).to_numpy()
            y_test = test_frame["combined_is_attack"].astype(bool).to_numpy()
            attack_types = (
                test_frame["combined_attack_type"].astype(str).to_numpy()
            )
            event_ids = test_frame["combined_event_id"].astype(str).to_numpy()
            train_values = risk[normal_train & np.isfinite(risk)]
            if train_values.size == 0:
                raise RuntimeError(f"no finite train-normal risk for {fold}/{seed}")
            for q in Q_GRID:
                threshold = float(np.nanquantile(train_values, q))
                budget = int(round(float(len(test_risk)) * (1.0 - q)))
                budget = max(0, min(int(len(test_risk)), budget))
                global_pred = top_k_mask(test_risk, budget)
                if int(global_pred.sum()) != budget:
                    raise RuntimeError("global budget mismatch")
                for rule in ALLOCATION_RULES:
                    pred = station_budget_mask(
                        test_stations, test_risk, budget, rule=rule
                    )
                    if int(pred.sum()) != budget:
                        raise RuntimeError(
                            f"station budget mismatch for {fold}/{seed}/{q}/{rule}"
                        )
                    test_frame["selected"] = pred
                    test_frame["risk"] = test_risk
                    test_frame["risk_percentile"] = (
                        pd.Series(test_risk)
                        .rank(pct=True, method="average")
                        .to_numpy()
                    )
                    test_frame["is_attack"] = y_test
                    test_frame["attack_type"] = attack_types
                    test_frame["event_id"] = event_ids
                    test_frame["fold"] = fold
                    test_frame["seed"] = int(seed)
                    test_frame["q"] = float(q)
                    test_frame["allocation_rule"] = rule
                    test_frame["global_selected"] = global_pred
                    selected_rows = test_frame[
                        [
                            "fold",
                            "seed",
                            "q",
                            "allocation_rule",
                            "station_key",
                            "timestamp_utc",
                            "risk",
                            "risk_percentile",
                            "selected",
                            "global_selected",
                            "is_attack",
                            "attack_type",
                            "event_id",
                        ]
                    ].copy()
                    selected_rows["station_id"] = test_frame["station_key"]
                    row_frames.append(selected_rows)
                    for station in sorted(set(test_stations.tolist())):
                        station_mask = test_stations == station
                        station_attack = y_test[station_mask]
                        station_pred = pred[station_mask]
                        for attack_type in ATTACK_TYPES:
                            type_mask = (
                                station_mask
                                & (attack_types == attack_type)
                            )
                            if not bool(type_mask.any()):
                                continue
                            selected_type_mask = type_mask & pred
                            unselected_type_mask = type_mask & ~pred
                            station_type_frames.append(
                                pd.DataFrame(
                                    [
                                        {
                                            "fold": fold,
                                            "seed": int(seed),
                                            "q": float(q),
                                            "allocation_rule": rule,
                                            "station_id": station,
                                            "attack_type": attack_type,
                                            "attack_rows": int(
                                                type_mask.sum()
                                            ),
                                            "selected_attack_rows": int(
                                                selected_type_mask.sum()
                                            ),
                                            "attack_recall": float(
                                                selected_type_mask.sum()
                                                / type_mask.sum()
                                            ),
                                            "mean_risk_percentile_selected": float(
                                                test_frame.loc[
                                                    selected_type_mask,
                                                    "risk_percentile",
                                                ].mean()
                                            )
                                            if bool(
                                                selected_type_mask.any()
                                            )
                                            else None,
                                            "mean_risk_percentile_unselected": float(
                                                test_frame.loc[
                                                    unselected_type_mask,
                                                    "risk_percentile",
                                                ].mean()
                                            )
                                            if bool(
                                                unselected_type_mask.any()
                                            )
                                            else None,
                                        }
                                    ]
                                )
                            )
                        station_frames.append(
                            pd.DataFrame(
                                [
                                    {
                                        "fold": fold,
                                        "seed": int(seed),
                                        "q": float(q),
                                        "allocation_rule": rule,
                                        "station_id": station,
                                        "test_rows": int(station_mask.sum()),
                                        "attack_rows": int(
                                            station_attack.sum()
                                        ),
                                        "selected_rows": int(
                                            station_pred.sum()
                                        ),
                                        "selected_attack_rows": int(
                                            (station_pred & station_attack).sum()
                                        ),
                                        "selected_normal_rows": int(
                                            (
                                                station_pred
                                                & ~station_attack
                                            ).sum()
                                        ),
                                    }
                                ]
                            )
                        )
                    event_rows.extend(
                        event_summary(
                            test_frame, q, rule, seed, fold
                        )
                    )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "q_values": Q_GRID,
                    "status": "completed",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    row_detail = pd.concat(row_frames, ignore_index=True)
    station_detail = pd.concat(station_frames, ignore_index=True)
    station_type_detail = pd.concat(
        station_type_frames, ignore_index=True
    )
    event_detail = pd.DataFrame(event_rows)
    return row_detail, station_detail, station_type_detail, event_detail


def write_summaries(
    row_detail: pd.DataFrame,
    station_detail: pd.DataFrame,
    station_type_detail: pd.DataFrame,
    event_detail: pd.DataFrame,
) -> int:
    prefix = "stage180_noaa_station_budget_row_tail_audit"
    row_detail.to_parquet(
        TABLES / f"{prefix}_row_detail.parquet", index=False
    )
    station_detail.to_csv(
        TABLES / f"{prefix}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_type_detail.to_csv(
        TABLES / f"{prefix}_station_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_detail.to_csv(
        TABLES / f"{prefix}_event_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        row_detail.groupby(
            ["fold", "seed", "q", "allocation_rule"], sort=True
        )
        .agg(
            test_rows=("selected", "size"),
            selected_rows=("selected", "sum"),
            global_selected_rows=("global_selected", "sum"),
            attack_rows=("is_attack", "sum"),
            selected_attack_rows=(
                "selected",
                lambda x: int(
                    (
                        x
                        & row_detail.loc[x.index, "is_attack"]
                    ).sum()
                ),
            ),
            selected_normal_rows=(
                "selected",
                lambda x: int(
                    (
                        x
                        & ~row_detail.loc[x.index, "is_attack"]
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )
    summary["attack_recall"] = (
        summary["selected_attack_rows"] / summary["attack_rows"]
    )
    summary["selected_normal_rate"] = (
        summary["selected_normal_rows"]
        / (summary["test_rows"] - summary["attack_rows"])
    )
    summary.to_csv(
        TABLES / f"{prefix}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_aggregate = (
        event_detail.groupby(["q", "allocation_rule", "attack_type"], sort=True)
        .agg(
            events=("event_id", "nunique"),
            covered_events=("event_covered", "sum"),
            event_rows=("event_rows", "sum"),
            selected_rows=("selected_rows", "sum"),
            mean_event_recall=("event_recall", "mean"),
        )
        .reset_index()
    )
    event_aggregate["event_coverage"] = (
        event_aggregate["covered_events"] / event_aggregate["events"]
    )
    event_aggregate.to_csv(
        TABLES / f"{prefix}_event_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段178冻结扩展评分设置和同一受控注入；只保留预先固定的"
            "q=0.995和q=0.999低误报运行点，保存逐行test风险、风险分位、"
            "三种站点预算选择、攻击类型和事件ID。模型、alpha、window、q和"
            "分配规则均不根据test选择。四折五种子结果全部保留。"
            "oracle式计数仅作诊断，不部署。"
            "NOAA攻击是透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [
            NOAA.name,
            AUDIT.name,
            FROZEN.name,
        ],
        "method": METHOD,
        "q_grid": Q_GRID,
        "allocation_rules": ALLOCATION_RULES,
        "attack_types": ATTACK_TYPES,
        "row_detail_rows": int(len(row_detail)),
        "station_detail_rows": int(len(station_detail)),
        "station_type_detail_rows": int(len(station_type_detail)),
        "event_detail_rows": int(len(event_detail)),
        "event_aggregate_rows": int(len(event_aggregate)),
        "summary_rows": int(len(summary)),
        "summary": clean_records(summary),
        "event_aggregate": clean_records(event_aggregate),
        "limitations": [
            "q=0.995和0.999是预先固定的低误报运行点，不是按test选择的最优点。",
            "oracle式计数和风险分位只用于解释漏检位置，不能部署。",
            "逐行证据仍来自透明受控注入，不代表真实攻击日志。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段180 NOAA站点预算逐行尾部审计",
        "",
        "固定q=0.995/0.999，保存冻结扩展风险、三种站点预算选择和攻击类型；不按test调参。",
        "",
        "| q | 分配规则 | 站点预算TP | 站点预算Recall | 全局预算Recall | 选中正常行率 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['q']:.4f} | {row['allocation_rule']} | "
            f"{int(row['selected_attack_rows'])} | "
            f"{row['attack_recall']:.6f} | "
            f"{row['global_selected_rows'] / row['attack_rows']:.6f} | "
            f"{row['selected_normal_rate']:.6f} |"
        )
    lines += [
        "",
        f"逐行={len(row_detail)}；站点={len(station_detail)}；"
        f"站点×类型={len(station_type_detail)}；"
        f"事件={len(event_detail)}；事件聚合={len(event_aggregate)}。",
        "风险分位和逐行结果只用于诊断固定预算下的漏检位置。",
        "NOAA攻击是透明受控注入，不是真实攻击日志。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "row_detail_rows": len(row_detail),
                "station_detail_rows": len(station_detail),
                "station_type_detail_rows": len(station_type_detail),
                "event_detail_rows": len(event_detail),
                "event_aggregate_rows": len(event_aggregate),
            },
            ensure_ascii=False,
        )
    )
    return 0


def rebuild_event_detail(row_detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["fold", "seed", "q", "allocation_rule"]
    for key, group in row_detail.groupby(keys, sort=True):
        fold, seed, q, rule = key
        rows.extend(event_summary(group, float(q), str(rule), int(seed), str(fold)))
    return pd.DataFrame(rows)


def rebuild_station_details(
    row_detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "fold",
        "seed",
        "q",
        "allocation_rule",
        "station_id",
    ]
    rows = row_detail.copy()
    rows["selected"] = rows["selected"].astype(bool)
    rows["is_attack"] = rows["is_attack"].astype(bool)
    station_detail = (
        rows.groupby(keys, sort=True)
        .agg(
            test_rows=("selected", "size"),
            attack_rows=("is_attack", "sum"),
            selected_rows=("selected", "sum"),
            selected_attack_rows=(
                "selected",
                lambda x: int(
                    (
                        x
                        & rows.loc[x.index, "is_attack"]
                    ).sum()
                ),
            ),
            selected_normal_rows=(
                "selected",
                lambda x: int(
                    (
                        x
                        & ~rows.loc[x.index, "is_attack"]
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )
    attacks = rows[rows["is_attack"]].copy()
    station_type_detail = (
        attacks.groupby([*keys, "attack_type"], sort=True)
        .agg(
            attack_rows=("selected", "size"),
            selected_attack_rows=("selected", "sum"),
            mean_risk_percentile_selected=(
                "risk_percentile",
                lambda x: float(x[attacks.loc[x.index, "selected"]].mean())
                if bool(attacks.loc[x.index, "selected"].any())
                else None,
            ),
            mean_risk_percentile_unselected=(
                "risk_percentile",
                lambda x: float(x[~attacks.loc[x.index, "selected"]].mean())
                if bool((~attacks.loc[x.index, "selected"]).any())
                else None,
            ),
        )
        .reset_index()
    )
    station_type_detail["attack_recall"] = (
        station_type_detail["selected_attack_rows"]
        / station_type_detail["attack_rows"]
    )
    return station_detail, station_type_detail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild summaries from persisted row and event details.",
    )
    args = parser.parse_args(argv)
    prefix = "stage180_noaa_station_budget_row_tail_audit"
    if args.summarize_existing:
        row_detail = pd.read_parquet(TABLES / f"{prefix}_row_detail.parquet")
        station_detail, station_type_detail = rebuild_station_details(
            row_detail
        )
        event_detail = rebuild_event_detail(row_detail)
        return write_summaries(
            row_detail,
            station_detail,
            station_type_detail,
            event_detail,
        )
    row_detail, station_detail, station_type_detail, event_detail = build_rows()
    return write_summaries(
        row_detail,
        station_detail,
        station_type_detail,
        event_detail,
    )


if __name__ == "__main__":
    raise SystemExit(main())
