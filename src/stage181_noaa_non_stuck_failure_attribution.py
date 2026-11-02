from __future__ import annotations

"""Stage 181: component-level attribution for non-stuck low-FAR misses.

Stage 180 showed that the pre-fixed low-FAR station budgets are dominated by
stuck events.  This stage replays the same frozen expanded candidate and
controlled injections, but writes the component risks and structural features
that are already computed by the frozen pipeline.

No model, alpha, window, q, threshold, or allocation rule is selected from
test.  The component fields are diagnostics only; they are not used to choose
a new operating point.

不重新训练，不根据test选择模型、alpha、window、q、阈值或分配规则；
五种子仅用于不确定性描述。
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
COMPONENTS = [
    "base_fused_risk",
    "structural_risk",
    "stuck_run",
    "drift_delta",
    "lag_residual",
    "spatial_residual",
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
from stage09b_robust_fusion import STRUCTURAL_BASE, smooth_risk, structural_features  # noqa: E402


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


def add_structural_columns(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    out = scored.copy()
    parts = []
    for split_name in ["train", "valid", "test"]:
        mask = out["split"].eq(split_name)
        part = structural_features(out.loc[mask], STRUCTURAL_BASE).set_axis(
            out.index[mask]
        )
        parts.append(part)
    features = pd.concat(parts).reindex(out.index).fillna(0.0)
    for column in STRUCTURAL_BASE:
        out[column] = features[column].to_numpy(dtype=float)
    return out


def add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["global_risk_percentile"] = (
        out["risk"].rank(pct=True, method="average").to_numpy(dtype=float)
    )
    out["station_risk_percentile"] = (
        out.groupby("station_key", sort=False)["risk"]
        .rank(pct=True, method="average")
        .to_numpy(dtype=float)
    )
    out["station_rank"] = (
        out.groupby("station_key", sort=False)["risk"]
        .rank(method="first", ascending=False)
        .to_numpy(dtype=float)
    )
    out["station_rows"] = (
        out.groupby("station_key", sort=False)["risk"]
        .transform("size")
        .to_numpy(dtype=float)
    )
    return out


def build_row_detail() -> pd.DataFrame:
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
    frames: list[pd.DataFrame] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            alpha = float(frozen.loc[(fold, seed), "alpha_base"])
            window = int(frozen.loc[(fold, seed), "window"])
            frame = build_fold_scaled(
                source,
                seed,
                train_years,
                valid_year,
                test_year,
                stations,
                1.0,
            )
            scored, _ = score(frame)
            scored = add_components(scored)
            scored = add_structural_columns(scored)
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
            test_frame["risk"] = test_risk
            test_frame["is_attack"] = test_frame[
                "combined_is_attack"
            ].astype(bool)
            test_frame["attack_type"] = test_frame[
                "combined_attack_type"
            ].astype(str)
            test_frame["event_id"] = test_frame["combined_event_id"].astype(str)
            test_frame["fold"] = fold
            test_frame["seed"] = int(seed)
            test_stations = test_frame["station_key"].astype(str).to_numpy()
            y_test = test_frame["is_attack"].to_numpy(dtype=bool)
            train_values = risk[normal_train & np.isfinite(risk)]
            if train_values.size == 0:
                raise RuntimeError(f"no finite train-normal risk for {fold}/{seed}")
            for q in Q_GRID:
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
                    current = test_frame.copy()
                    current["q"] = float(q)
                    current["allocation_rule"] = rule
                    current["selected"] = pred
                    current["global_selected"] = global_pred
                    current = add_ranks(current)
                    keep = [
                        "fold",
                        "seed",
                        "q",
                        "allocation_rule",
                        "station_key",
                        "timestamp_utc",
                        "risk",
                        "is_attack",
                        "attack_type",
                        "event_id",
                        "selected",
                        "global_selected",
                        "global_risk_percentile",
                        "station_risk_percentile",
                        "station_rank",
                        "station_rows",
                        *COMPONENTS,
                    ]
                    frames.append(current[keep].copy())
        print(
            json.dumps(
                {"fold": fold, "q_values": Q_GRID, "status": "completed"},
                ensure_ascii=False,
            ),
            flush=True,
        )
    return pd.concat(frames, ignore_index=True)


def summarize(
    row_detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attacks = row_detail[row_detail["is_attack"]].copy()
    normals = row_detail[~row_detail["is_attack"]].copy()
    keys = ["fold", "seed", "q", "allocation_rule", "attack_type"]
    rows: list[dict[str, object]] = []
    for key, group in attacks.groupby(keys, sort=True):
        fold, seed, q, rule, attack_type = key
        selected = group["selected"].astype(bool)
        global_selected = group["global_selected"].astype(bool)
        normal_group = normals[
            normals["fold"].eq(fold)
            & normals["seed"].eq(seed)
            & normals["q"].eq(q)
            & normals["allocation_rule"].eq(rule)
        ]
        row: dict[str, object] = {
            "fold": fold,
            "seed": int(seed),
            "q": float(q),
            "allocation_rule": rule,
            "attack_type": attack_type,
            "attack_rows": int(len(group)),
            "selected_attack_rows": int(selected.sum()),
            "global_selected_attack_rows": int(global_selected.sum()),
            "attack_recall": float(selected.mean()),
            "global_attack_recall": float(global_selected.mean()),
            "event_rows": int(group["event_id"].nunique()),
            "event_coverage": float(
                group.groupby("event_id")["selected"].any().mean()
            ),
        }
        for component in COMPONENTS:
            attack_values = pd.to_numeric(group[component], errors="coerce")
            normal_values = pd.to_numeric(normal_group[component], errors="coerce")
            row[f"{component}_attack_mean"] = float(attack_values.mean())
            row[f"{component}_normal_mean"] = float(normal_values.mean())
            row[f"{component}_normal_p95"] = float(normal_values.quantile(0.95))
            row[f"{component}_attack_above_normal_p95_rate"] = float(
                (attack_values > row[f"{component}_normal_p95"]).mean()
            )
        rows.append(row)
    detail = pd.DataFrame(rows)
    aggregate = (
        detail.groupby(["q", "allocation_rule", "attack_type"], sort=True)
        .agg(
            attack_rows=("attack_rows", "sum"),
            selected_attack_rows=("selected_attack_rows", "sum"),
            global_selected_attack_rows=("global_selected_attack_rows", "sum"),
            event_rows=("event_rows", "sum"),
            mean_attack_recall=("attack_recall", "mean"),
            mean_global_attack_recall=("global_attack_recall", "mean"),
            mean_event_coverage=("event_coverage", "mean"),
        )
        .reset_index()
    )
    aggregate["pooled_attack_recall"] = (
        aggregate["selected_attack_rows"] / aggregate["attack_rows"]
    )
    aggregate["pooled_global_attack_recall"] = (
        aggregate["global_selected_attack_rows"] / aggregate["attack_rows"]
    )
    component_rows: list[dict[str, object]] = []
    for component in COMPONENTS:
        for (q, attack_type), group in detail.groupby(
            ["q", "attack_type"], sort=True
        ):
            component_rows.append(
                {
                    "q": float(q),
                    "attack_type": attack_type,
                    "component": component,
                    "attack_mean": float(
                        group[f"{component}_attack_mean"].mean()
                    ),
                    "normal_mean": float(
                        group[f"{component}_normal_mean"].mean()
                    ),
                    "normal_p95": float(
                        group[f"{component}_normal_p95"].mean()
                    ),
                    "attack_above_normal_p95_rate": float(
                        group[f"{component}_attack_above_normal_p95_rate"].mean()
                    ),
                }
            )
    components = pd.DataFrame(component_rows)
    return detail, aggregate, components


def write_outputs(
    row_detail: pd.DataFrame,
    detail: pd.DataFrame,
    aggregate: pd.DataFrame,
    components: pd.DataFrame,
) -> int:
    prefix = "stage181_noaa_non_stuck_failure_attribution"
    row_detail.to_parquet(TABLES / f"{prefix}_row_detail.parquet", index=False)
    detail.to_csv(
        TABLES / f"{prefix}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        TABLES / f"{prefix}_type_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    components.to_csv(
        TABLES / f"{prefix}_component_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "复用阶段178冻结扩展评分设置和同一透明受控注入；"
            "只保存基础融合分量、结构分量、四个结构特征、"
            "全局/站内风险分位和三种预先固定站点预算选择；"
            "模型、alpha、window、q、阈值和分配规则均不根据test选择；"
            "不重新训练；分量统计仅用于诊断，不部署；"
            "五种子仅用于不确定性描述；"
            "NOAA攻击为透明受控注入，不是真实攻击日志。"
        ),
        "source_artifacts": [NOAA.name, AUDIT.name, FROZEN.name],
        "method": METHOD,
        "q_grid": Q_GRID,
        "allocation_rules": ALLOCATION_RULES,
        "attack_types": ATTACK_TYPES,
        "components": COMPONENTS,
        "row_detail_rows": int(len(row_detail)),
        "type_detail_rows": int(len(detail)),
        "type_aggregate_rows": int(len(aggregate)),
        "component_detail_rows": int(len(components)),
        "type_aggregate": clean_records(aggregate),
        "component_detail": clean_records(components),
        "limitations": [
            "分量诊断不改变冻结模型和运行点，不能据此在test上重新选择模型。",
            "攻击仍为透明受控注入，不代表真实攻击日志。",
            "结构特征来自当前冻结定义，缺失新特征不等于不存在可部署信号。",
        ],
    }
    (TABLES / f"{prefix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段181 NOAA非stuck低误报失败归因",
        "",
        "复用阶段178冻结扩展评分设置和同一透明受控注入；只保存分量风险、结构特征和站内排名，不重新选择模型、q或分配规则。",
        "",
        "| q | 分配规则 | 攻击类型 | 行Recall | 全局Recall | 事件覆盖 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['q']:.4f} | {row['allocation_rule']} | "
            f"{row['attack_type']} | {row['pooled_attack_recall']:.6f} | "
            f"{row['pooled_global_attack_recall']:.6f} | "
            f"{row['mean_event_coverage']:.6f} |"
        )
    lines += [
        "",
        f"逐行={len(row_detail)}；类型明细={len(detail)}；类型聚合={len(aggregate)}；分量明细={len(components)}。",
        "所有攻击均为透明受控注入，不能描述为真实攻击日志。",
    ]
    (REPORTS / f"{prefix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "row_detail_rows": len(row_detail),
                "type_detail_rows": len(detail),
                "type_aggregate_rows": len(aggregate),
                "component_detail_rows": len(components),
            },
            ensure_ascii=False,
        )
    )
    return 0


def rebuild_existing() -> int:
    prefix = "stage181_noaa_non_stuck_failure_attribution"
    row_detail = pd.read_parquet(TABLES / f"{prefix}_row_detail.parquet")
    detail, aggregate, components = summarize(row_detail)
    return write_outputs(row_detail, detail, aggregate, components)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild summaries from persisted row detail.",
    )
    args = parser.parse_args(argv)
    if args.summarize_existing:
        return rebuild_existing()
    row_detail = build_row_detail()
    detail, aggregate, components = summarize(row_detail)
    return write_outputs(row_detail, detail, aggregate, components)


if __name__ == "__main__":
    raise SystemExit(main())
