from __future__ import annotations

"""Read-only cross-holdout failure decomposition.

This stage does not fit a detector, choose a threshold, or inspect a test
result before a protocol freeze.  It only joins already-frozen Stage267,
Stage367, and Stage372 summaries to identify stable failure boundaries.
The output is diagnostic evidence, not confirmatory performance evidence.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

LEGACY_TYPE = TABLES / "stage267_smhi_locked_transfer_type_aggregate.csv"
STATION_TYPE = TABLES / "stage367_smhi_station_holdout_locked_transfer_type_aggregate.csv"
STATION_COMPARISONS = TABLES / "stage367_smhi_station_holdout_locked_transfer_comparisons.csv"
FACTOR_SCENARIO = TABLES / "stage372_smhi_factor_holdout_locked_evaluation_scenario_summary.csv"
FACTOR_SUMMARY = TABLES / "stage372_smhi_factor_holdout_locked_evaluation_summary.csv"

OUT_JSON = TABLES / "stage374_cross_holdout_failure_decomposition.json"
OUT_TYPE = TABLES / "stage374_cross_holdout_common_attack_failure.csv"
OUT_FACTOR = TABLES / "stage374_cross_holdout_factor_failure.csv"
OUT_REPORT = REPORTS / "stage374_cross_holdout_failure_decomposition.md"

TRUST_TRANSFER = "trust_valid_f1"
TRUST_FACTOR = "trust_shape_union"
BASELINES_TRANSFER = [
    "robust_max_deviation",
    "isolation_forest",
    "one_class_svm",
    "pca_reconstruction",
    "mlp_autoencoder_reconstruction",
    "knn_distance",
    "arima_innovation",
    "lightweight_autoencoder",
]
BASELINES_FACTOR = [
    "robust_max_deviation",
    "isolation_forest",
    "one_class_svm",
    "pca_reconstruction",
    "mlp_autoencoder_reconstruction",
    "knn_distance",
    "ewma_deviation",
    "arima_innovation",
    "gru_predictor",
    "lightweight_autoencoder",
    "ics_cps_spc_proxy",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _read_type(path: Path, holdout: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"method", "attack_type", "event_coverage", "row_recall"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["holdout"] = holdout
    frame["attack_type"] = frame["attack_type"].astype(str)
    return frame[["holdout", "method", "attack_type", "event_coverage", "row_recall"]]


def main() -> int:
    for path in [
        LEGACY_TYPE,
        STATION_TYPE,
        STATION_COMPARISONS,
        FACTOR_SCENARIO,
        FACTOR_SUMMARY,
    ]:
        _require(path)

    legacy = _read_type(LEGACY_TYPE, "SMHI_legacy_stage267")
    station = _read_type(STATION_TYPE, "SMHI_new_station_stage367")
    transfer = pd.concat([legacy, station], ignore_index=True)

    # Common attack types only: no re-labeling or imputation is performed.
    common_types = sorted(
        set(legacy["attack_type"].unique()).intersection(station["attack_type"].unique())
    )
    common = transfer[transfer.attack_type.isin(common_types)]
    trust = common[common.method.eq(TRUST_TRANSFER)].rename(
        columns={
            "event_coverage": "trust_event_coverage",
            "row_recall": "trust_row_recall",
        }
    )
    trust = trust.drop(columns=["method"])
    failure_rows: list[pd.DataFrame] = []
    for baseline in BASELINES_TRANSFER:
        base = common[common.method.eq(baseline)].rename(
            columns={
                "event_coverage": "baseline_event_coverage",
                "row_recall": "baseline_row_recall",
            }
        )
        base = base.drop(columns=["method"])
        merged = trust.merge(base, on=["holdout", "attack_type"], how="inner")
        if merged.empty:
            continue
        merged["baseline"] = baseline
        merged["event_difference"] = (
            merged["trust_event_coverage"] - merged["baseline_event_coverage"]
        )
        merged["row_recall_difference"] = (
            merged["trust_row_recall"] - merged["baseline_row_recall"]
        )
        merged["event_failure"] = merged["event_difference"] < 0
        merged["row_recall_failure"] = merged["row_recall_difference"] < 0
        failure_rows.append(merged)
    common_failure = pd.concat(failure_rows, ignore_index=True) if failure_rows else pd.DataFrame()
    common_failure.to_csv(OUT_TYPE, index=False)

    # Stage372 is a factor stress test. Preserve each scenario label exactly.
    factor = pd.read_csv(FACTOR_SCENARIO)
    required_factor = {"method", "scenario", "event_coverage", "row_recall"}
    missing_factor = required_factor.difference(factor.columns)
    if missing_factor:
        raise ValueError(f"{FACTOR_SCENARIO.name} missing columns: {sorted(missing_factor)}")
    factor = factor.copy()
    trust_factor = factor[factor.method.eq(TRUST_FACTOR)].rename(
        columns={
            "event_coverage": "trust_event_coverage",
            "row_recall": "trust_row_recall",
        }
    )[["scenario", "trust_event_coverage", "trust_row_recall"]]
    factor_rows: list[pd.DataFrame] = []
    for baseline in BASELINES_FACTOR:
        base = factor[factor.method.eq(baseline)].rename(
            columns={
                "event_coverage": "baseline_event_coverage",
                "row_recall": "baseline_row_recall",
            }
        )[["scenario", "baseline_event_coverage", "baseline_row_recall"]]
        merged = trust_factor.merge(base, on="scenario", how="inner")
        if merged.empty:
            continue
        merged["baseline"] = baseline
        merged["event_difference"] = (
            merged["trust_event_coverage"] - merged["baseline_event_coverage"]
        )
        merged["row_recall_difference"] = (
            merged["trust_row_recall"] - merged["baseline_row_recall"]
        )
        merged["event_failure"] = merged["event_difference"] < 0
        merged["row_recall_failure"] = merged["row_recall_difference"] < 0
        factor_rows.append(merged)
    factor_failure = pd.concat(factor_rows, ignore_index=True) if factor_rows else pd.DataFrame()
    factor_failure.to_csv(OUT_FACTOR, index=False)

    # Summaries are deliberately descriptive; no p-value or promotion decision
    # is recomputed from these already-viewed test outputs.
    stable_event_failures = []
    if not common_failure.empty:
        pivot = common_failure.pivot_table(
            index=["attack_type", "baseline"],
            columns="holdout",
            values="event_difference",
            aggfunc="mean",
        ).reset_index()
        holdout_cols = [c for c in pivot.columns if c != "attack_type" and c != "baseline"]
        if len(holdout_cols) == 2:
            stable = pivot[(pivot[holdout_cols[0]] < 0) & (pivot[holdout_cols[1]] < 0)]
            stable_event_failures = stable.to_dict(orient="records")

    factor_failure_summary = []
    if not factor_failure.empty:
        for baseline, group in factor_failure.groupby("baseline"):
            factor_failure_summary.append(
                {
                    "baseline": baseline,
                    "scenario_count": int(group["scenario"].nunique()),
                    "event_failure_count": int(group["event_failure"].sum()),
                    "event_failure_share": float(group["event_failure"].mean()),
                    "mean_event_difference": float(group["event_difference"].mean()),
                    "unseen_event_difference": (
                        float(
                            group[
                                group["scenario"].astype(str).str.contains(
                                    "unseen", case=False, regex=False
                                )
                            ]["event_difference"].mean()
                        )
                        if group["scenario"].astype(str)
                        .str.contains("unseen", case=False, regex=False)
                        .any()
                        else None
                    ),
                }
            )

    summary = {
        "status": "READ_ONLY_CROSS_HOLDOUT_FAILURE_DECOMPOSITION_COMPLETED",
        "inputs": {
            "legacy_type_sha256": sha256(LEGACY_TYPE),
            "station_type_sha256": sha256(STATION_TYPE),
            "station_comparisons_sha256": sha256(STATION_COMPARISONS),
            "factor_scenario_sha256": sha256(FACTOR_SCENARIO),
            "factor_summary_sha256": sha256(FACTOR_SUMMARY),
        },
        "transfer_holdouts": ["SMHI_legacy_stage267", "SMHI_new_station_stage367"],
        "common_attack_types": common_types,
        "common_transfer_rows": int(len(common_failure)),
        "stable_event_failures_across_both_transfer_holdouts": stable_event_failures,
        "factor_rows": int(len(factor_failure)),
        "factor_failure_summary": factor_failure_summary,
        "conclusions": [
            "该阶段只读汇总已经查看过的 Stage267、Stage367、Stage372 结果，不是新的确认性性能证据。",
            "两批 SMHI 站点的结果若同时对同一基线和攻击类型为负，只能作为稳定失败边界，不能通过重新调参消除。",
            "Stage372 的因子测试属于探索性压力证据，不能替换 V6 或升级全面领先主张。",
            "所有攻击仍是透明受控模拟注入；结果不能写成真实生产事故召回率。",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stage374 跨 holdout 失败分解",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 共同攻击类型：`{', '.join(common_types)}`",
        f"- 共同迁移失败比较行：`{len(common_failure)}`",
        f"- 因子压力测试比较行：`{len(factor_failure)}`",
        "",
        "## 两批 SMHI 站点均为负的共同边界",
        "",
    ]
    if stable_event_failures:
        lines.append(pd.DataFrame(stable_event_failures).to_markdown(index=False))
    else:
        lines.append("没有找到同时在两批 holdout 上为负的共同比较；不代表全面领先。")
    lines.extend(["", "## Stage372 因子压力测试失败比例", ""])
    if factor_failure_summary:
        lines.append(pd.DataFrame(factor_failure_summary).to_markdown(index=False))
    else:
        lines.append("没有可汇总的因子比较。")
    lines.extend(
        [
            "",
            "## 科学边界",
            "",
            *[f"- {item}" for item in summary["conclusions"]],
            "",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "common_attack_types": common_types,
                "stable_failure_count": len(stable_event_failures),
                "factor_rows": len(factor_failure),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
