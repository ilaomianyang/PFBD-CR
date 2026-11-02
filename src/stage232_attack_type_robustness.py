from __future__ import annotations

"""Stage 232: attack-type decomposition of the cross-provider direction.

This stage is read-only with respect to the frozen transfer results.  It
does not train a detector, recalibrate a threshold, or select a provider,
station, seed, attack type, operating point, or metric.  It reconstructs
the seed-level attack-type metrics from the frozen per-seed type details
and asks whether the Stage230/231 positive direction is uniform across
attack types or concentrated in one pre-declared type.

The available frozen type-level metrics are row recall and event coverage.
Precision and F1 are not reconstructed here because the type details do
not persist the corresponding normal-row false-positive counts.  All
attacks are transparent controlled injections, not real attack logs.
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
PREFIX = "stage232_attack_type_robustness"
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20261002

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage218_swiss_negative_result_uncertainty import (  # noqa: E402
    exact_sign_flip_two_sided,
    sha256_file,
)
from stage223_noaa_transfer_uncertainty import clean_records  # noqa: E402

SOURCE_SPECS = [
    {
        "source": "MeteoSwiss",
        "stage": 219,
        "detail": TABLES
        / "stage219_swiss_dwd_protection_transfer_type_detail.csv",
        "aggregate": TABLES
        / "stage219_swiss_dwd_protection_transfer_type_aggregate.csv",
        "baseline_arm": "swiss_valid_recalibrated",
        "candidate_arm": "swiss_valid_selected",
    },
    {
        "source": "NOAA ISD",
        "stage": 222,
        "detail": TABLES
        / "stage222_noaa_dwd_protection_transfer_type_detail.csv",
        "aggregate": TABLES
        / "stage222_noaa_dwd_protection_transfer_type_aggregate.csv",
        "baseline_arm": "noaa_valid_recalibrated",
        "candidate_arm": "noaa_valid_selected",
    },
    {
        "source": "KNMI",
        "stage": 228,
        "detail": TABLES
        / "stage228_knmi_dwd_protection_transfer_type_detail.csv",
        "aggregate": TABLES
        / "stage228_knmi_dwd_protection_transfer_type_aggregate.csv",
        "baseline_arm": "knmi_valid_recalibrated",
        "candidate_arm": "knmi_valid_selected",
    },
    {
        "source": "ECCC",
        "stage": 238,
        "detail": TABLES
        / "stage238_eccc_dwd_protection_transfer_type_detail.csv",
        "aggregate": TABLES
        / "stage238_eccc_dwd_protection_transfer_type_aggregate.csv",
        "baseline_arm": "eccc_valid_recalibrated",
        "candidate_arm": "eccc_valid_selected",
    },
]


def type_metric(detail: pd.DataFrame, arm: str) -> pd.DataFrame:
    rows = detail[detail["arm"].eq(arm)].copy()
    if rows.empty:
        raise RuntimeError(f"missing type rows for arm={arm}")
    aggregated = (
        rows.groupby("attack_type", sort=True, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            attack_rows=("attack_rows", "sum"),
            detected_rows=("detected_rows", "sum"),
            row_recall=("row_recall", "mean"),
            event_coverage=("event_coverage", "mean"),
        )
        .sort_values("attack_type")
    )
    return aggregated


def reconstruct_provider(spec: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.read_csv(Path(spec["detail"]))
    aggregate = pd.read_csv(Path(spec["aggregate"]))
    baseline = type_metric(detail, str(spec["baseline_arm"]))
    candidate = type_metric(detail, str(spec["candidate_arm"]))
    aligned = baseline.merge(
        candidate,
        on="attack_type",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    aligned["row_recall_difference"] = (
        aligned["row_recall_candidate"] - aligned["row_recall_baseline"]
    )
    aligned["event_coverage_difference"] = (
        aligned["event_coverage_candidate"]
        - aligned["event_coverage_baseline"]
    )
    for arm in (str(spec["baseline_arm"]), str(spec["candidate_arm"])):
        frozen = aggregate[aggregate["arm"].eq(arm)].set_index("attack_type")
        reconstructed = (
            baseline if arm == str(spec["baseline_arm"]) else candidate
        ).set_index("attack_type")
        if set(frozen.index) != set(reconstructed.index):
            raise RuntimeError(f"{spec['source']} attack-type set changed")
        for metric in ("row_recall", "event_coverage"):
            if not np.allclose(
                frozen.loc[reconstructed.index, metric].to_numpy(dtype=float),
                reconstructed[metric].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"{spec['source']} {arm} {metric} reconstruction changed"
                )
        for metric in ("attack_rows", "detected_rows"):
            if not np.array_equal(
                frozen.loc[reconstructed.index, metric].to_numpy(dtype=int),
                reconstructed[metric].to_numpy(dtype=int),
            ):
                raise RuntimeError(
                    f"{spec['source']} {arm} {metric} reconstruction changed"
                )

    seed_baseline = detail[detail["arm"].eq(str(spec["baseline_arm"]))][
        ["seed", "attack_type", "row_recall", "event_coverage"]
    ].rename(
        columns={
            "row_recall": "row_recall_baseline",
            "event_coverage": "event_coverage_baseline",
        }
    )
    seed_candidate = detail[detail["arm"].eq(str(spec["candidate_arm"]))][
        ["seed", "attack_type", "row_recall", "event_coverage"]
    ].rename(
        columns={
            "row_recall": "row_recall_candidate",
            "event_coverage": "event_coverage_candidate",
        }
    )
    seed_aligned = seed_baseline.merge(
        seed_candidate,
        on=["seed", "attack_type"],
        validate="one_to_one",
    )
    seed_aligned["row_recall_difference"] = (
        seed_aligned["row_recall_candidate"]
        - seed_aligned["row_recall_baseline"]
    )
    seed_aligned["event_coverage_difference"] = (
        seed_aligned["event_coverage_candidate"]
        - seed_aligned["event_coverage_baseline"]
    )
    seed_aligned.insert(0, "source", str(spec["source"]))
    aligned.insert(0, "source", str(spec["source"]))
    aligned.insert(1, "stage", int(spec["stage"]))
    return aligned, seed_aligned


def hierarchical_bootstrap_by_type(
    seed_rows: pd.DataFrame,
    attack_type: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    rows = seed_rows[seed_rows["attack_type"].eq(attack_type)]
    providers = sorted(rows["source"].unique())
    by_provider = {
        provider: rows.loc[
            rows["source"].eq(provider),
            ["row_recall_difference", "event_coverage_difference"],
        ].to_numpy(dtype=float)
        for provider in providers
    }
    rng = np.random.default_rng(int(seed))
    provider_draws = rng.integers(
        0,
        len(providers),
        size=(int(iterations), len(providers)),
    )
    row_boot = np.empty(int(iterations), dtype=float)
    event_boot = np.empty(int(iterations), dtype=float)
    for index in range(int(iterations)):
        row_provider_means: list[float] = []
        event_provider_means: list[float] = []
        for provider_index in provider_draws[index]:
            values = by_provider[providers[int(provider_index)]]
            seed_draws = rng.integers(0, len(values), size=len(values))
            sampled = values[seed_draws]
            row_provider_means.append(float(sampled[:, 0].mean()))
            event_provider_means.append(float(sampled[:, 1].mean()))
        row_boot[index] = float(np.mean(row_provider_means))
        event_boot[index] = float(np.mean(event_provider_means))
    return {
        "attack_type": attack_type,
        "iterations": int(iterations),
        "provider_count": int(len(providers)),
        "seed_rows": int(len(rows)),
        "row_recall_difference_mean": float(
            rows.groupby("source")["row_recall_difference"].mean().mean()
        ),
        "row_recall_ci95_low": float(np.quantile(row_boot, 0.025)),
        "row_recall_ci95_high": float(np.quantile(row_boot, 0.975)),
        "event_coverage_difference_mean": float(
            rows.groupby("source")["event_coverage_difference"].mean().mean()
        ),
        "event_coverage_ci95_low": float(np.quantile(event_boot, 0.025)),
        "event_coverage_ci95_high": float(np.quantile(event_boot, 0.975)),
    }


def main() -> int:
    required = [path for spec in SOURCE_SPECS for path in (spec["detail"], spec["aggregate"])]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing frozen artifacts for Stage232: " + ", ".join(missing)
        )

    provider_rows: list[pd.DataFrame] = []
    seed_rows_list: list[pd.DataFrame] = []
    for spec in SOURCE_SPECS:
        aligned, seed_aligned = reconstruct_provider(spec)
        provider_rows.append(aligned)
        seed_rows_list.append(seed_aligned)
    provider_type = pd.concat(provider_rows, ignore_index=True)
    seed_type = pd.concat(seed_rows_list, ignore_index=True)
    attack_types = sorted(provider_type["attack_type"].unique())
    if len(attack_types) != 6:
        raise RuntimeError(f"expected six attack types, found {len(attack_types)}")

    provider_equal_rows: list[dict[str, object]] = []
    for attack_type in attack_types:
        rows = provider_type[provider_type["attack_type"].eq(attack_type)]
        row_values = rows["row_recall_difference"].to_numpy(dtype=float)
        event_values = rows["event_coverage_difference"].to_numpy(dtype=float)
        provider_equal_rows.append(
            {
                "attack_type": attack_type,
                "provider_count": int(len(rows)),
                "row_recall_difference": float(row_values.mean()),
                "row_recall_positive_providers": int((row_values > 0).sum()),
                "row_recall_negative_providers": int((row_values < 0).sum()),
                "row_recall_zero_providers": int((row_values == 0).sum()),
                "row_recall_sign_flip_p": float(
                    exact_sign_flip_two_sided(row_values)
                ),
                "event_coverage_difference": float(event_values.mean()),
                "event_coverage_positive_providers": int(
                    (event_values > 0).sum()
                ),
                "event_coverage_negative_providers": int(
                    (event_values < 0).sum()
                ),
                "event_coverage_zero_providers": int(
                    (event_values == 0).sum()
                ),
                "event_coverage_sign_flip_p": float(
                    exact_sign_flip_two_sided(event_values)
                ),
            }
        )
    provider_equal = pd.DataFrame(provider_equal_rows)

    leave_one_provider_rows: list[dict[str, object]] = []
    for removed in sorted(provider_type["source"].unique()):
        retained = provider_type[~provider_type["source"].eq(removed)]
        for attack_type in attack_types:
            rows = retained[retained["attack_type"].eq(attack_type)]
            leave_one_provider_rows.append(
                {
                    "removed_source": removed,
                    "retained_sources": ",".join(
                        sorted(retained["source"].unique())
                    ),
                    "attack_type": attack_type,
                    "provider_count": int(len(rows)),
                    "row_recall_difference": float(
                        rows["row_recall_difference"].mean()
                    ),
                    "event_coverage_difference": float(
                        rows["event_coverage_difference"].mean()
                    ),
                    "row_recall_direction_positive": bool(
                        float(rows["row_recall_difference"].mean()) > 0.0
                    ),
                    "event_coverage_direction_positive": bool(
                        float(rows["event_coverage_difference"].mean()) > 0.0
                    ),
                }
            )
    leave_one_provider = pd.DataFrame(leave_one_provider_rows)

    leave_one_type_rows: list[dict[str, object]] = []
    for removed in attack_types:
        retained = provider_equal[
            ~provider_equal["attack_type"].eq(removed)
        ]
        leave_one_type_rows.append(
            {
                "removed_attack_type": removed,
                "retained_attack_types": ",".join(
                    [value for value in attack_types if value != removed]
                ),
                "row_recall_difference": float(
                    retained["row_recall_difference"].mean()
                ),
                "event_coverage_difference": float(
                    retained["event_coverage_difference"].mean()
                ),
                "row_recall_direction_positive": bool(
                    float(retained["row_recall_difference"].mean()) > 0.0
                ),
                "event_coverage_direction_positive": bool(
                    float(retained["event_coverage_difference"].mean()) > 0.0
                ),
            }
        )
    leave_one_type = pd.DataFrame(leave_one_type_rows)

    bootstrap_rows = [
        hierarchical_bootstrap_by_type(
            seed_type,
            attack_type,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, attack_type in enumerate(attack_types)
    ]
    bootstrap = pd.DataFrame(bootstrap_rows)

    stuck_equal = provider_equal.set_index("attack_type").loc["stuck"]
    non_stuck = provider_equal[~provider_equal["attack_type"].eq("stuck")]
    stuck_leave_one_provider = leave_one_provider[
        leave_one_provider["attack_type"].eq("stuck")
    ]
    stuck_bootstrap = bootstrap.set_index("attack_type").loc["stuck"]
    remove_stuck = leave_one_type[
        leave_one_type["removed_attack_type"].eq("stuck")
    ].iloc[0]
    dominant_row = bool(
        float(stuck_equal["row_recall_difference"])
        > float(non_stuck["row_recall_difference"].max())
    )
    dominant_event = bool(
        float(stuck_equal["event_coverage_difference"])
        > float(non_stuck["event_coverage_difference"].max())
    )

    checks: list[dict[str, object]] = [
        {
            "check": "source_count",
            "status": "PASS" if len(SOURCE_SPECS) == 4 else "FAIL",
            "value": len(SOURCE_SPECS),
        },
        {
            "check": "attack_type_count",
            "status": "PASS" if len(attack_types) == 6 else "FAIL",
            "value": attack_types,
        },
        {
            "check": "frozen_type_reconstruction_exact",
            "status": "PASS",
            "value": True,
        },
        {
            "check": "provider_equal_overall_positive",
            "status": "PASS"
            if float(provider_equal["row_recall_difference"].mean()) > 0.0
            and float(provider_equal["event_coverage_difference"].mean()) > 0.0
            else "FAIL",
            "value": {
                "row_recall": float(
                    provider_equal["row_recall_difference"].mean()
                ),
                "event_coverage": float(
                    provider_equal["event_coverage_difference"].mean()
                ),
            },
        },
        {
            "check": "stuck_direction_positive",
            "status": "PASS"
            if float(stuck_equal["row_recall_difference"]) > 0.0
            and float(stuck_equal["event_coverage_difference"]) > 0.0
            else "FAIL",
            "value": {
                "row_recall": float(stuck_equal["row_recall_difference"]),
                "event_coverage": float(
                    stuck_equal["event_coverage_difference"]
                ),
            },
        },
        {
            "check": "stuck_dominates_non_stuck_types",
            "status": "PASS" if dominant_row and dominant_event else "FAIL",
            "value": {
                "row_recall_stuck": float(
                    stuck_equal["row_recall_difference"]
                ),
                "row_recall_non_stuck_max": float(
                    non_stuck["row_recall_difference"].max()
                ),
                "event_coverage_stuck": float(
                    stuck_equal["event_coverage_difference"]
                ),
                "event_coverage_non_stuck_max": float(
                    non_stuck["event_coverage_difference"].max()
                ),
            },
        },
        {
            "check": "leave_one_attack_type_out_stuck_required",
            "status": "PASS"
            if float(remove_stuck["row_recall_difference"]) <= 0.0
            and float(remove_stuck["event_coverage_difference"]) <= 0.0
            else "FAIL",
            "value": remove_stuck.to_dict(),
        },
        {
            "check": "leave_one_provider_out_stuck_positive",
            "status": "PASS"
            if len(stuck_leave_one_provider) == 4
            and bool(
                stuck_leave_one_provider[
                    "row_recall_direction_positive"
                ].all()
            )
            and bool(
                stuck_leave_one_provider[
                    "event_coverage_direction_positive"
                ].all()
            )
            else "FAIL",
            "value": stuck_leave_one_provider[
                [
                    "removed_source",
                    "row_recall_difference",
                    "event_coverage_difference",
                ]
            ].to_dict("records"),
        },
        {
            "check": "hierarchical_bootstrap_stuck_positive",
            "status": "PASS"
            if float(stuck_bootstrap["row_recall_ci95_low"]) > 0.0
            and float(stuck_bootstrap["event_coverage_ci95_low"]) > 0.0
            else "FAIL",
            "value": stuck_bootstrap.to_dict(),
        },
        {
            "check": "provider_sign_flip_power_boundary_retained",
            "status": "PASS"
            if float(stuck_equal["row_recall_sign_flip_p"]) >= 0.125
            and float(stuck_equal["event_coverage_sign_flip_p"]) >= 0.125
            else "FAIL",
            "value": {
                "row_recall_p": float(stuck_equal["row_recall_sign_flip_p"]),
                "event_coverage_p": float(
                    stuck_equal["event_coverage_sign_flip_p"]
                ),
                "minimum_two_sided_p": 0.125,
            },
        },
        {
            "check": "controlled_injection_boundary_retained",
            "status": "PASS",
            "value": True,
        },
    ]
    failed_checks = [row for row in checks if row["status"] == "FAIL"]
    status = (
        "PASS_PIPELINE_REVIEW_REQUIRED"
        if not failed_checks
        else "ATTACK_TYPE_ROBUSTNESS_REVIEW_REQUIRED"
    )

    payload = {
        "status": status,
        "protocol": (
            "阶段232只读取阶段219、222和228已冻结的逐种子攻击类型明细和"
            "类型汇总，以及阶段238的ECCC逐种子攻击类型明细和类型汇总，"
            "重建目标域valid本地选择减目标域valid重校准的"
            "逐类型行召回和事件覆盖差值。先在机构内对同类型种子取均值，"
            "再对MeteoSwiss、NOAA ISD、KNMI和ECCC四个机构等权取均值，"
            "并执行留一机构、留一攻击类型和固定种子的两级层级bootstrap"
            "（先重采样机构，再在机构内重采样种子）。"
            "不重新训练、不重新选择特征、窗口、权重、阈值、站点、种子、"
            "攻击类型或运行点，不使用目标域test标签做任何选择。"
            "类型级冻结明细不包含正常行误报计数，因此本阶段只报告"
            "行召回和事件覆盖，不伪造类型级Precision或F1。"
            "机构级精确sign-flip检验只有4个机构，最小双侧p=0.125，"
            "因此仅作描述性报告。所有攻击都是透明受控注入，"
            "不是真实攻击日志。"
        ),
        "source_count": len(SOURCE_SPECS),
        "attack_type_count": len(attack_types),
        "provider_type_rows": int(len(provider_type)),
        "provider_equal_rows": clean_records(
            provider_equal.to_dict("records")
        ),
        "leave_one_provider_out_rows": clean_records(
            leave_one_provider.to_dict("records")
        ),
        "leave_one_attack_type_out_rows": clean_records(
            leave_one_type.to_dict("records")
        ),
        "hierarchical_bootstrap_rows": clean_records(
            bootstrap.to_dict("records")
        ),
        "checks": checks,
        "failed_checks": failed_checks,
        "limitations": [
            "只有MeteoSwiss、NOAA ISD、KNMI和ECCC四个迁移目标，机构级推断功效仍然不足。",
            "机构级精确sign-flip检验只有4个观测，最小双侧p=0.125，只能作低功效描述性检查。",
            "本地选择优势若主要集中在stuck攻击类型；去掉stuck后，行召回和事件覆盖的跨机构等权方向将不再为正。",
            "冻结的类型级明细不包含正常行误报计数，本阶段不能构造类型级Precision或F1。",
            "MeteoSwiss为11站、NOAA ISD为11站、KNMI为6站、ECCC为14站；类型级结果仍受站点集合和攻击块限制。",
            "所有攻击都是透明受控注入，不是真实攻击日志，不能解释为真实入侵率。",
        ],
        "artifact_sha256": {
            f"{spec['source']}_detail": sha256_file(Path(spec["detail"]))
            for spec in SOURCE_SPECS
        }
        | {
            f"{spec['source']}_aggregate": sha256_file(
                Path(spec["aggregate"])
            )
            for spec in SOURCE_SPECS
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    provider_type.to_csv(
        TABLES / f"{PREFIX}_sources.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provider_equal.to_csv(
        TABLES / f"{PREFIX}_provider_equal_by_type.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one_provider.to_csv(
        TABLES / f"{PREFIX}_leave_one_provider_out_by_type.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one_type.to_csv(
        TABLES / f"{PREFIX}_leave_one_attack_type_out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap.to_csv(
        TABLES / f"{PREFIX}_hierarchical_bootstrap_by_type.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(checks).to_csv(
        TABLES / f"{PREFIX}_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段232 跨机构本地选择优势攻击类型稳健性审计",
        "",
        f"- 状态：**{status}**",
        f"- 机构数：`{len(SOURCE_SPECS)}`；攻击类型数：`{len(attack_types)}`",
        "- 汇总规则：机构内同类型种子先取均值，再对机构等权取均值。",
        f"- 两级bootstrap：`{BOOTSTRAP_ITERATIONS}` 次；seed `{BOOTSTRAP_SEED}`",
        "",
        "## 机构等权类型级结果",
        "",
        "| 攻击类型 | 本地−重校准行召回差值 | 正/负/零机构 | 事件覆盖差值 | 正/负/零机构 |",
        "|---|---:|---|---:|---|",
    ]
    for row in provider_equal.to_dict("records"):
        lines.append(
            f"| {row['attack_type']} | "
            f"{float(row['row_recall_difference']):+.6f} | "
            f"{int(row['row_recall_positive_providers'])}/"
            f"{int(row['row_recall_negative_providers'])}/"
            f"{int(row['row_recall_zero_providers'])} | "
            f"{float(row['event_coverage_difference']):+.6f} | "
            f"{int(row['event_coverage_positive_providers'])}/"
            f"{int(row['event_coverage_negative_providers'])}/"
            f"{int(row['event_coverage_zero_providers'])} |"
        )
    lines += [
        "",
        "## 留一攻击类型",
        "",
        "| 移除类型 | 行召回差值 | 事件覆盖差值 | 方向均为正 |",
        "|---|---:|---:|---|",
    ]
    for row in leave_one_type.to_dict("records"):
        lines.append(
            f"| {row['removed_attack_type']} | "
            f"{float(row['row_recall_difference']):+.6f} | "
            f"{float(row['event_coverage_difference']):+.6f} | "
            f"{bool(row['row_recall_direction_positive'] and row['event_coverage_direction_positive'])} |"
        )
    lines += [
        "",
        "## 两级Bootstrap",
        "",
        "| 攻击类型 | 行召回均值 | 行召回95%区间 | 事件覆盖均值 | 事件覆盖95%区间 |",
        "|---|---:|---|---:|---|",
    ]
    for row in bootstrap.to_dict("records"):
        lines.append(
            f"| {row['attack_type']} | "
            f"{float(row['row_recall_difference_mean']):+.6f} | "
            f"[{float(row['row_recall_ci95_low']):+.6f}, "
            f"{float(row['row_recall_ci95_high']):+.6f}] | "
            f"{float(row['event_coverage_difference_mean']):+.6f} | "
            f"[{float(row['event_coverage_ci95_low']):+.6f}, "
            f"{float(row['event_coverage_ci95_high']):+.6f}] |"
        )
    lines += [
        "",
        "## 审计检查",
        "",
        "| 检查 | 状态 | 值 |",
        "|---|---|---|",
    ]
    for row in checks:
        lines.append(
            f"| {row['check']} | {row['status']} | {row['value']} |"
        )
    lines += [
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
        "",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "attack_type_count": len(attack_types),
                "provider_equal_rows": clean_records(
                    provider_equal.to_dict("records")
                ),
                "leave_one_attack_type_out_rows": clean_records(
                    leave_one_type.to_dict("records")
                ),
                "stuck_bootstrap": stuck_bootstrap.to_dict(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
