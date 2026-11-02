from __future__ import annotations

"""Stage 231: robustness audit of the local-selection advantage.

This stage is read-only with respect to every frozen source artifact.  It
does not train a detector, recalibrate a threshold, or select a provider,
station, seed, attack type, operating point, or metric.  It asks whether
the Stage230 local-selection advantage is an artifact of unequal station
counts or of a single provider by:

1. rebuilding station-level F1 and event-coverage differences from the
   frozen per-station transfer details;
2. averaging stations within each provider and then providers equally;
3. repeating the provider-equal summary with each provider removed; and
4. running a two-level hierarchical bootstrap that resamples providers
   and then stations within the selected providers.

Four transfer targets now exist, so the provider-level exact sign-flip
test has a minimum two-sided p-value of 0.125 and remains descriptive
only.  All attacks are transparent controlled injections, not real
attack logs.
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
PREFIX = "stage231_cross_provider_robustness"
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20261001

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage218_swiss_negative_result_uncertainty import (  # noqa: E402
    exact_sign_flip_two_sided,
    sha256_file,
)
from stage223_noaa_transfer_uncertainty import (  # noqa: E402
    clean_records,
    station_metric,
)

STAGE219_STATION = (
    TABLES / "stage219_swiss_dwd_protection_transfer_station_detail.csv"
)
STAGE219_EVENT = (
    TABLES / "stage219_swiss_dwd_protection_transfer_station_event_detail.csv"
)
STAGE220 = TABLES / "stage220_swiss_dwd_transfer_uncertainty.json"
STAGE220_PAIRED = (
    TABLES / "stage220_swiss_dwd_transfer_uncertainty_paired_station.csv"
)
STAGE222_STATION = (
    TABLES / "stage222_noaa_dwd_protection_transfer_station_detail.csv"
)
STAGE222_EVENT = (
    TABLES / "stage222_noaa_dwd_protection_transfer_station_event_detail.csv"
)
STAGE223 = TABLES / "stage223_noaa_transfer_uncertainty.json"
STAGE223_PAIRED = TABLES / "stage223_noaa_transfer_uncertainty_paired_station.csv"
STAGE228_STATION = (
    TABLES / "stage228_knmi_dwd_protection_transfer_station_detail.csv"
)
STAGE228_EVENT = (
    TABLES / "stage228_knmi_dwd_protection_transfer_station_event_detail.csv"
)
STAGE229 = TABLES / "stage229_knmi_transfer_uncertainty.json"
STAGE229_PAIRED = TABLES / "stage229_knmi_transfer_uncertainty_paired_station.csv"
STAGE238_STATION = (
    TABLES / "stage238_eccc_dwd_protection_transfer_station_detail.csv"
)
STAGE238_EVENT = (
    TABLES
    / "stage238_eccc_dwd_protection_transfer_station_event_detail.csv"
)
STAGE239 = TABLES / "stage239_eccc_transfer_uncertainty.json"
STAGE239_PAIRED = TABLES / "stage239_eccc_transfer_uncertainty_paired_station.csv"

SOURCE_SPECS = [
    {
        "source": "MeteoSwiss",
        "stage": 220,
        "station_path": STAGE219_STATION,
        "event_path": STAGE219_EVENT,
        "paired_path": STAGE220_PAIRED,
        "uncertainty_path": STAGE220,
        "baseline_arm": "swiss_valid_recalibrated",
        "candidate_arm": "swiss_valid_selected",
        "station_column": "station_id",
        "station_count": 11,
        "f1_reference": "stage220_swiss_dwd_transfer_uncertainty",
    },
    {
        "source": "NOAA ISD",
        "stage": 223,
        "station_path": STAGE222_STATION,
        "event_path": STAGE222_EVENT,
        "paired_path": STAGE223_PAIRED,
        "uncertainty_path": STAGE223,
        "baseline_arm": "noaa_valid_recalibrated",
        "candidate_arm": "noaa_valid_selected",
        "station_column": "station_id",
        "station_count": 11,
        "f1_reference": "stage223_noaa_transfer_uncertainty",
    },
    {
        "source": "KNMI",
        "stage": 229,
        "station_path": STAGE228_STATION,
        "event_path": STAGE228_EVENT,
        "paired_path": STAGE229_PAIRED,
        "uncertainty_path": STAGE229,
        "baseline_arm": "knmi_valid_recalibrated",
        "candidate_arm": "knmi_valid_selected",
        "station_column": "station_id",
        "station_count": 6,
        "f1_reference": "stage229_knmi_transfer_uncertainty",
    },
    {
        "source": "ECCC",
        "stage": 239,
        "station_path": STAGE238_STATION,
        "event_path": STAGE238_EVENT,
        "paired_path": STAGE239_PAIRED,
        "uncertainty_path": STAGE239,
        "baseline_arm": "eccc_valid_recalibrated",
        "candidate_arm": "eccc_valid_selected",
        "station_column": "station_id",
        "station_count": 14,
        "f1_reference": "stage239_eccc_transfer_uncertainty",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def provider_station_effects(spec: dict[str, object]) -> pd.DataFrame:
    """Rebuild station-level paired F1 and event-coverage differences."""
    station_path = Path(spec["station_path"])
    event_path = Path(spec["event_path"])
    station_detail = pd.read_csv(station_path, dtype={"station_id": str})
    event_detail = pd.read_csv(event_path, dtype={"station_id": str})

    baseline_metric = station_metric(
        station_detail,
        str(spec["baseline_arm"]),
    )
    candidate_metric = station_metric(
        station_detail,
        str(spec["candidate_arm"]),
    )
    aligned = baseline_metric.merge(
        candidate_metric,
        on="station_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    aligned["f1_difference"] = (
        aligned["f1_candidate"] - aligned["f1_baseline"]
    )

    event_baseline = (
        event_detail[event_detail["arm"].eq(str(spec["baseline_arm"]))]
        .groupby("station_id", sort=True, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            events=("events", "sum"),
            detected_events=("detected_events", "sum"),
        )
    )
    event_candidate = (
        event_detail[event_detail["arm"].eq(str(spec["candidate_arm"]))]
        .groupby("station_id", sort=True, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            events=("events", "sum"),
            detected_events=("detected_events", "sum"),
        )
    )
    event_aligned = event_baseline.merge(
        event_candidate,
        on="station_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    event_aligned["event_coverage_baseline"] = np.divide(
        event_aligned["detected_events_baseline"],
        event_aligned["events_baseline"],
        out=np.zeros(len(event_aligned), dtype=float),
        where=event_aligned["events_baseline"].to_numpy() > 0,
    )
    event_aligned["event_coverage_candidate"] = np.divide(
        event_aligned["detected_events_candidate"],
        event_aligned["events_candidate"],
        out=np.zeros(len(event_aligned), dtype=float),
        where=event_aligned["events_candidate"].to_numpy() > 0,
    )
    event_aligned["event_difference"] = (
        event_aligned["event_coverage_candidate"]
        - event_aligned["event_coverage_baseline"]
    )

    combined = aligned.merge(
        event_aligned[
            ["station_id", "event_coverage_baseline", "event_coverage_candidate", "event_difference"]
        ],
        on="station_id",
        validate="one_to_one",
    )
    if len(combined) != int(spec["station_count"]):
        raise RuntimeError(
            f"{spec['source']} station count changed: {len(combined)}"
        )
    combined.insert(0, "source", str(spec["source"]))
    combined.insert(1, "stage", int(spec["stage"]))
    return combined


def assert_reference_reconstruction(
    effects: pd.DataFrame,
    reference: dict,
    label: str,
) -> None:
    f1_mean = float(effects["f1_difference"].mean())
    event_mean = float(effects["event_difference"].mean())
    if not np.isclose(
        f1_mean,
        float(reference["primary_f1"]["mean_difference"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{label} F1 reconstruction changed")
    if not np.isclose(
        event_mean,
        float(reference["primary_event_coverage"]["mean_difference"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{label} event reconstruction changed")
    if int(reference["station_count"]) != len(effects):
        raise RuntimeError(f"{label} station count changed")


def provider_summary(
    effects: pd.DataFrame,
    provider_rows: list[str] | None = None,
) -> dict[str, float]:
    selected = (
        effects
        if provider_rows is None
        else effects[effects["source"].isin(provider_rows)]
    )
    grouped = (
        selected.groupby("source", sort=True, as_index=False)
        .agg(
            stations=("station_id", "nunique"),
            f1_difference=("f1_difference", "mean"),
            event_difference=("event_difference", "mean"),
        )
    )
    return {
        "provider_count": int(len(grouped)),
        "station_count": int(grouped["stations"].sum()),
        "f1_difference": float(grouped["f1_difference"].mean()),
        "event_difference": float(grouped["event_difference"].mean()),
    }


def hierarchical_bootstrap(
    effects: pd.DataFrame,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    providers = sorted(effects["source"].unique())
    by_provider = {
        provider: effects.loc[
            effects["source"].eq(provider),
            ["f1_difference", "event_difference"],
        ].to_numpy(dtype=float)
        for provider in providers
    }
    rng = np.random.default_rng(int(seed))
    provider_draws = rng.integers(
        0,
        len(providers),
        size=(int(iterations), len(providers)),
    )
    f1_boot = np.empty(int(iterations), dtype=float)
    event_boot = np.empty(int(iterations), dtype=float)
    for index in range(int(iterations)):
        provider_means_f1: list[float] = []
        provider_means_event: list[float] = []
        for provider_index in provider_draws[index]:
            values = by_provider[providers[int(provider_index)]]
            station_draws = rng.integers(0, len(values), size=len(values))
            sampled = values[station_draws]
            provider_means_f1.append(float(sampled[:, 0].mean()))
            provider_means_event.append(float(sampled[:, 1].mean()))
        f1_boot[index] = float(np.mean(provider_means_f1))
        event_boot[index] = float(np.mean(provider_means_event))
    return {
        "iterations": int(iterations),
        "provider_count": int(len(providers)),
        "station_count": int(len(effects)),
        "f1_mean": float(effects.groupby("source")["f1_difference"].mean().mean()),
        "f1_ci95_low": float(np.quantile(f1_boot, 0.025)),
        "f1_ci95_high": float(np.quantile(f1_boot, 0.975)),
        "event_mean": float(
            effects.groupby("source")["event_difference"].mean().mean()
        ),
        "event_ci95_low": float(np.quantile(event_boot, 0.025)),
        "event_ci95_high": float(np.quantile(event_boot, 0.975)),
    }


def main() -> int:
    required = [
        STAGE219_STATION,
        STAGE219_EVENT,
        STAGE220,
        STAGE220_PAIRED,
        STAGE222_STATION,
        STAGE222_EVENT,
        STAGE223,
        STAGE223_PAIRED,
        STAGE228_STATION,
        STAGE228_EVENT,
        STAGE229,
        STAGE229_PAIRED,
        STAGE238_STATION,
        STAGE238_EVENT,
        STAGE239,
        STAGE239_PAIRED,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing frozen artifacts for Stage231: " + ", ".join(missing)
        )

    all_effects: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    for spec in SOURCE_SPECS:
        effects = provider_station_effects(spec)
        reference = load_json(Path(spec["uncertainty_path"]))
        assert_reference_reconstruction(
            effects,
            reference,
            str(spec["source"]),
        )
        all_effects.append(effects)
        source_rows.append(
            {
                "source": str(spec["source"]),
                "stage": int(spec["stage"]),
                "station_count": int(len(effects)),
                "baseline_arm": str(spec["baseline_arm"]),
                "candidate_arm": str(spec["candidate_arm"]),
                "f1_difference": float(effects["f1_difference"].mean()),
                "f1_positive_stations": int(
                    (effects["f1_difference"] > 0).sum()
                ),
                "f1_negative_stations": int(
                    (effects["f1_difference"] < 0).sum()
                ),
                "f1_zero_stations": int(
                    (effects["f1_difference"] == 0).sum()
                ),
                "event_difference": float(effects["event_difference"].mean()),
                "event_positive_stations": int(
                    (effects["event_difference"] > 0).sum()
                ),
                "event_negative_stations": int(
                    (effects["event_difference"] < 0).sum()
                ),
                "event_zero_stations": int(
                    (effects["event_difference"] == 0).sum()
                ),
                "reference_f1_mean_difference": float(
                    reference["primary_f1"]["mean_difference"]
                ),
                "reference_event_mean_difference": float(
                    reference["primary_event_coverage"]["mean_difference"]
                ),
                "reconstruction_exact": True,
            }
        )

    effects = pd.concat(all_effects, ignore_index=True)
    provider_equal = provider_summary(effects)
    leave_one_out_rows: list[dict[str, object]] = []
    providers = sorted(effects["source"].unique())
    for removed in providers:
        retained = [provider for provider in providers if provider != removed]
        summary = provider_summary(effects, retained)
        leave_one_out_rows.append(
            {
                "removed_source": removed,
                "retained_sources": ",".join(retained),
                "provider_count": int(summary["provider_count"]),
                "station_count": int(summary["station_count"]),
                "f1_difference": float(summary["f1_difference"]),
                "event_difference": float(summary["event_difference"]),
                "f1_direction_positive": bool(
                    float(summary["f1_difference"]) > 0.0
                ),
                "event_direction_positive": bool(
                    float(summary["event_difference"]) > 0.0
                ),
            }
        )

    provider_f1 = (
        effects.groupby("source", sort=True, as_index=False)["f1_difference"]
        .mean()["f1_difference"]
        .to_numpy(dtype=float)
    )
    provider_event = (
        effects.groupby("source", sort=True, as_index=False)["event_difference"]
        .mean()["event_difference"]
        .to_numpy(dtype=float)
    )
    provider_sign_flip = {
        "provider_count": int(len(providers)),
        "f1_mean": float(provider_f1.mean()),
        "f1_sign_flip_p": float(exact_sign_flip_two_sided(provider_f1)),
        "event_mean": float(provider_event.mean()),
        "event_sign_flip_p": float(exact_sign_flip_two_sided(provider_event)),
        "minimum_two_sided_p": float(2.0 ** (-(len(providers) - 1))),
        "interpretation": (
            "Only four provider-level observations are available, so the "
            "exact two-sided sign-flip test cannot reach p<0.125 and is "
            "descriptive only."
        ),
    }
    hierarchical = hierarchical_bootstrap(
        effects,
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
    )
    station_rows = (
        effects[
            [
                "source",
                "station_id",
                "f1_difference",
                "event_difference",
            ]
        ]
        .sort_values(["source", "station_id"], kind="stable")
        .reset_index(drop=True)
    )
    leave_one_out = pd.DataFrame(leave_one_out_rows)

    checks: list[dict[str, object]] = [
        {
            "check": "source_count",
            "status": "PASS" if len(source_rows) == 4 else "FAIL",
            "value": len(source_rows),
        },
        {
            "check": "station_counts",
            "status": "PASS"
            if [int(row["station_count"]) for row in source_rows]
            == [11, 11, 6, 14]
            else "FAIL",
            "value": [int(row["station_count"]) for row in source_rows],
        },
        {
            "check": "frozen_reconstruction_exact",
            "status": "PASS"
            if all(bool(row["reconstruction_exact"]) for row in source_rows)
            else "FAIL",
            "value": True,
        },
        {
            "check": "all_provider_directions_positive",
            "status": "PASS"
            if all(float(row["f1_difference"]) > 0.0 for row in source_rows)
            and all(float(row["event_difference"]) > 0.0 for row in source_rows)
            else "FAIL",
            "value": {
                "f1": [float(row["f1_difference"]) for row in source_rows],
                "event": [
                    float(row["event_difference"]) for row in source_rows
                ],
            },
        },
        {
            "check": "provider_equal_direction_positive",
            "status": "PASS"
            if float(provider_equal["f1_difference"]) > 0.0
            and float(provider_equal["event_difference"]) > 0.0
            else "FAIL",
            "value": provider_equal,
        },
        {
            "check": "leave_one_provider_out_direction_positive",
            "status": "PASS"
            if bool(leave_one_out["f1_direction_positive"].all())
            and bool(leave_one_out["event_direction_positive"].all())
            else "FAIL",
            "value": leave_one_out[
                ["removed_source", "f1_difference", "event_difference"]
            ].to_dict("records"),
        },
        {
            "check": "not_driven_by_single_provider",
            "status": "PASS"
            if len(leave_one_out) == 4
            and bool(leave_one_out["f1_direction_positive"].all())
            and bool(leave_one_out["event_direction_positive"].all())
            else "FAIL",
            "value": True,
        },
        {
            "check": "provider_sign_flip_power_boundary_retained",
            "status": "PASS"
            if float(provider_sign_flip["f1_sign_flip_p"]) >= 0.125
            and float(provider_sign_flip["event_sign_flip_p"]) >= 0.125
            else "FAIL",
            "value": {
                "f1_p": provider_sign_flip["f1_sign_flip_p"],
                "event_p": provider_sign_flip["event_sign_flip_p"],
                "minimum_two_sided_p": provider_sign_flip[
                    "minimum_two_sided_p"
                ],
            },
        },
        {
            "check": "hierarchical_bootstrap_direction_positive",
            "status": "PASS"
            if float(hierarchical["f1_mean"]) > 0.0
            and float(hierarchical["event_mean"]) > 0.0
            else "FAIL",
            "value": {
                "f1_mean": hierarchical["f1_mean"],
                "event_mean": hierarchical["event_mean"],
            },
        },
        {
            "check": "provider_count_boundary_retained",
            "status": "PASS" if len(providers) == 4 else "FAIL",
            "value": len(providers),
        },
    ]
    failed_checks = [row for row in checks if row["status"] == "FAIL"]
    status = (
        "PASS_PIPELINE_REVIEW_REQUIRED"
        if not failed_checks
        else "CROSS_PROVIDER_ROBUSTNESS_REVIEW_REQUIRED"
    )

    payload = {
        "status": status,
        "protocol": (
            "阶段231只读取阶段219/220、222/223、228/229和238/239已冻结的逐站点"
            "test明细，重建每个迁移目标在目标域valid本地选择减目标域"
            "valid重校准的站点级F1和事件覆盖差值。先在机构内对站点取均值，"
            "再对MeteoSwiss、NOAA ISD、KNMI和ECCC四个机构等权取均值，"
            "避免11/11/6/14站点数不平衡。随后执行四个留一机构汇总和固定种子的"
            "两级层级bootstrap（先重采样机构，再在机构内重采样站点）。"
            "不重新训练、不重新选择特征、窗口、权重、阈值、站点、种子、"
            "攻击类型或运行点，不使用目标域test标签做任何选择。"
            "机构级精确sign-flip检验只有4个观测，最小双侧p=0.125，"
            "因此仅作描述性报告，不构成显著性声明。"
            "所有攻击都是透明受控注入，不是真实攻击日志。"
        ),
        "provider_equal_summary": provider_equal,
        "leave_one_provider_out": leave_one_out_rows,
        "provider_level_sign_flip": provider_sign_flip,
        "hierarchical_bootstrap": hierarchical,
        "source_count": len(source_rows),
        "station_count": int(len(effects)),
        "source_rows": source_rows,
        "station_effects": clean_records(
            station_rows.to_dict("records")
        ),
        "checks": checks,
        "failed_checks": failed_checks,
        "limitations": [
            "只有MeteoSwiss、NOAA ISD、KNMI和ECCC四个跨机构迁移目标，机构级推断功效仍然不足。",
            "机构级精确sign-flip检验只有4个观测，最小双侧p=0.125；本阶段只作低功效描述性稳健性检查，不能写成多机构普适领先。",
            "MeteoSwiss为11站、NOAA ISD为11站、KNMI为6站、ECCC为14站；机构等权和两级bootstrap避免站数拉动，但不改变站点集合本身的限制。",
            "NOAA的F1方向很小且不显著，KNMI的F1精确检验也不显著；方向一致不等于每个机构或指标都显著。",
            "所有攻击都是透明受控注入，不是真实攻击日志，不能解释为真实入侵率。",
            "本阶段只检验Stage230本地选择优势的跨机构稳健性，不替代原始JSON、CSV、Parquet和逐行明细。",
        ],
        "artifact_sha256": {
            "stage219_station": sha256_file(STAGE219_STATION),
            "stage219_event": sha256_file(STAGE219_EVENT),
            "stage220": sha256_file(STAGE220),
            "stage220_paired": sha256_file(STAGE220_PAIRED),
            "stage222_station": sha256_file(STAGE222_STATION),
            "stage222_event": sha256_file(STAGE222_EVENT),
            "stage223": sha256_file(STAGE223),
            "stage223_paired": sha256_file(STAGE223_PAIRED),
            "stage228_station": sha256_file(STAGE228_STATION),
            "stage228_event": sha256_file(STAGE228_EVENT),
            "stage229": sha256_file(STAGE229),
            "stage229_paired": sha256_file(STAGE229_PAIRED),
            "stage238_station": sha256_file(STAGE238_STATION),
            "stage238_event": sha256_file(STAGE238_EVENT),
            "stage239": sha256_file(STAGE239),
            "stage239_paired": sha256_file(STAGE239_PAIRED),
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(source_rows).to_csv(
        TABLES / f"{PREFIX}_sources.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one_out.to_csv(
        TABLES / f"{PREFIX}_leave_one_provider_out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([hierarchical]).to_csv(
        TABLES / f"{PREFIX}_hierarchical_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_rows.to_csv(
        TABLES / f"{PREFIX}_station_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(checks).to_csv(
        TABLES / f"{PREFIX}_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段231 跨机构本地选择优势稳健性审计",
        "",
        f"- 状态：**{status}**",
        f"- 机构数：`{provider_equal['provider_count']}`；站点数：`{provider_equal['station_count']}`",
        "- 汇总规则：机构内站点先取均值，再对机构等权取均值。",
        f"- 两级bootstrap：`{BOOTSTRAP_ITERATIONS}` 次；seed `{BOOTSTRAP_SEED}`",
        "",
        "## 来源级重建",
        "",
        "| 来源 | 阶段 | 站点 | 本地−重校准F1差值 | 正/负/零站点 | 事件覆盖差值 | 正/负/零站点 |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for row in source_rows:
        lines.append(
            f"| {row['source']} | {int(row['stage'])} | "
            f"{int(row['station_count'])} | "
            f"{float(row['f1_difference']):+.6f} | "
            f"{int(row['f1_positive_stations'])}/"
            f"{int(row['f1_negative_stations'])}/"
            f"{int(row['f1_zero_stations'])} | "
            f"{float(row['event_difference']):+.6f} | "
            f"{int(row['event_positive_stations'])}/"
            f"{int(row['event_negative_stations'])}/"
            f"{int(row['event_zero_stations'])} |"
        )
    lines += [
        "",
        "## 机构等权与留一机构",
        "",
        f"- 机构等权F1差值：`{float(provider_equal['f1_difference']):+.6f}`",
        f"- 机构等权事件覆盖差值：`{float(provider_equal['event_difference']):+.6f}`",
        "",
        "| 移除机构 | 保留机构 | 站点 | F1差值 | 事件覆盖差值 | 方向均为正 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in leave_one_out_rows:
        lines.append(
            f"| {row['removed_source']} | {row['retained_sources']} | "
            f"{int(row['station_count'])} | "
            f"{float(row['f1_difference']):+.6f} | "
            f"{float(row['event_difference']):+.6f} | "
            f"{bool(row['f1_direction_positive'] and row['event_direction_positive'])} |"
        )
    lines += [
        "",
        "## 两级层级Bootstrap",
        "",
        f"- F1：均值 `{float(hierarchical['f1_mean']):+.6f}`，95%区间 "
        f"`[{float(hierarchical['f1_ci95_low']):+.6f}, "
        f"{float(hierarchical['f1_ci95_high']):+.6f}]`",
        f"- 事件覆盖：均值 `{float(hierarchical['event_mean']):+.6f}`，95%区间 "
        f"`[{float(hierarchical['event_ci95_low']):+.6f}, "
        f"{float(hierarchical['event_ci95_high']):+.6f}]`",
        "",
        "## 机构级精确检验",
        "",
        f"- F1：均值 `{float(provider_sign_flip['f1_mean']):+.6f}`，"
        f"精确双侧p=`{float(provider_sign_flip['f1_sign_flip_p']):.6f}`",
        f"- 事件覆盖：均值 `{float(provider_sign_flip['event_mean']):+.6f}`，"
        f"精确双侧p=`{float(provider_sign_flip['event_sign_flip_p']):.6f}`",
        f"- 最小可能双侧p：`{float(provider_sign_flip['minimum_two_sided_p']):.6f}`",
        f"- 解释：{provider_sign_flip['interpretation']}",
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
                "provider_equal_summary": provider_equal,
                "leave_one_provider_out": leave_one_out_rows,
                "provider_level_sign_flip": provider_sign_flip,
                "hierarchical_bootstrap": hierarchical,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
