from __future__ import annotations

"""Stage 223: station-clustered uncertainty for the NOAA transfer.

This stage is read-only with respect to Stage222.  It does not fit
detectors, recalibrate thresholds, or select features.  It quantifies
whether the NOAA-valid local-selection advantage over the DWD
feature/window/weight transfer with NOAA-valid recalibration is stable
when stations are the resampling unit.

The primary comparison is:

    noaa_valid_selected - noaa_valid_recalibrated

All attacks are transparent controlled injections, not real attack logs.
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

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage218_swiss_negative_result_uncertainty import (  # noqa: E402
    benjamini_hochberg,
    cluster_bootstrap_mean,
    exact_sign_flip_two_sided,
    sha256_file,
)

STAGE222_JSON = TABLES / "stage222_noaa_dwd_protection_transfer.json"
STAGE222_STATION = (
    TABLES / "stage222_noaa_dwd_protection_transfer_station_detail.csv"
)
STAGE222_STATION_EVENT = (
    TABLES / "stage222_noaa_dwd_protection_transfer_station_event_detail.csv"
)
PREFIX = "stage223_noaa_transfer_uncertainty"
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260922


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for record in records:
        cleaned: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(
                float(value)
            ):
                cleaned[key] = None
            elif isinstance(value, np.integer):
                cleaned[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        output.append(cleaned)
    return output


def station_metric(
    station_detail: pd.DataFrame,
    arm: str,
) -> pd.DataFrame:
    rows = station_detail[station_detail["arm"].eq(arm)].copy()
    if rows.empty:
        raise RuntimeError(f"missing station rows for arm={arm}")
    aggregated = (
        rows.groupby("station_id", sort=True, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tn=("tn", "sum"),
        )
        .sort_values("station_id")
    )
    precision = np.divide(
        aggregated["tp"],
        aggregated["tp"] + aggregated["fp"],
        out=np.zeros(len(aggregated), dtype=float),
        where=(aggregated["tp"] + aggregated["fp"]).to_numpy() > 0,
    )
    recall = np.divide(
        aggregated["tp"],
        aggregated["tp"] + aggregated["fn"],
        out=np.zeros(len(aggregated), dtype=float),
        where=(aggregated["tp"] + aggregated["fn"]).to_numpy() > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(aggregated), dtype=float),
        where=(precision + recall) > 0,
    )
    false_alarm_rate = np.divide(
        aggregated["fp"],
        aggregated["fp"] + aggregated["tn"],
        out=np.zeros(len(aggregated), dtype=float),
        where=(aggregated["fp"] + aggregated["tn"]).to_numpy() > 0,
    )
    aggregated["precision"] = precision
    aggregated["recall"] = recall
    aggregated["f1"] = f1
    aggregated["false_alarm_rate"] = false_alarm_rate
    return aggregated


def paired_interval(
    station_ids: list[str],
    candidate: np.ndarray,
    baseline: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("candidate and baseline must be aligned 1-D arrays")
    if len(station_ids) != candidate.size:
        raise ValueError("station_ids and metric arrays must have equal length")
    differences = candidate - baseline
    values_by_station = {
        str(station_id): float(value)
        for station_id, value in zip(station_ids, differences)
    }
    bootstrap = cluster_bootstrap_mean(
        values_by_station,
        iterations=iterations,
        seed=seed,
    )
    p_value = exact_sign_flip_two_sided(differences)
    return {
        "stations": int(differences.size),
        "mean_difference": float(differences.mean()),
        "std_difference": float(differences.std(ddof=0)),
        "ci95_low": float(bootstrap["ci95_low"]),
        "ci95_high": float(bootstrap["ci95_high"]),
        "positive_stations": int((differences > 0).sum()),
        "negative_stations": int((differences < 0).sum()),
        "zero_stations": int((differences == 0).sum()),
        "sign_flip_p": float(p_value),
    }


def main() -> int:
    stage222 = load_json(STAGE222_JSON)
    station_detail = pd.read_csv(
        STAGE222_STATION,
        dtype={"station_id": str},
    )
    station_event_detail = pd.read_csv(
        STAGE222_STATION_EVENT,
        dtype={"station_id": str},
    )
    required_arms = {
        "dwd_threshold_transfer",
        "noaa_valid_recalibrated",
        "noaa_valid_selected",
    }
    if set(station_detail["arm"]) != required_arms:
        raise RuntimeError("Stage222 station arm set changed")
    if set(station_event_detail["arm"]) != required_arms:
        raise RuntimeError("Stage222 station-event arm set changed")
    primary_far = float(stage222["primary_far"])
    if not np.isclose(primary_far, 0.01):
        raise RuntimeError("Stage222 primary FAR changed")

    baseline = station_metric(
        station_detail,
        "noaa_valid_recalibrated",
    )
    candidate = station_metric(
        station_detail,
        "noaa_valid_selected",
    )
    aligned = baseline.merge(
        candidate,
        on="station_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(aligned) != 11:
        raise RuntimeError(
            f"expected 11 NOAA stations, found {len(aligned)}"
        )

    metric_rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for index, metric in enumerate(
        ["f1", "recall", "precision", "false_alarm_rate"]
    ):
        result = paired_interval(
            aligned["station_id"].astype(str).tolist(),
            aligned[f"{metric}_candidate"].to_numpy(dtype=float),
            aligned[f"{metric}_baseline"].to_numpy(dtype=float),
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED + index,
        )
        p_values.append(float(result["sign_flip_p"]))
        metric_rows.append({"metric": metric, **result})
    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(metric_rows, q_values):
        row["bh_q"] = float(q_value)
    metric_table = pd.DataFrame(metric_rows)

    event_baseline = (
        station_event_detail[
            station_event_detail["arm"].eq("noaa_valid_recalibrated")
        ]
        .groupby("station_id", sort=True, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            events=("events", "sum"),
            detected_events=("detected_events", "sum"),
        )
    )
    event_candidate = (
        station_event_detail[
            station_event_detail["arm"].eq("noaa_valid_selected")
        ]
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
    event_result = paired_interval(
        event_aligned["station_id"].astype(str).tolist(),
        event_aligned["event_coverage_candidate"].to_numpy(dtype=float),
        event_aligned["event_coverage_baseline"].to_numpy(dtype=float),
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED + 100,
    )
    event_result["metric"] = "event_coverage"
    # Event coverage is the single pre-registered primary endpoint here.
    # It is not part of the four-metric BH family above, so its q value is
    # the exact sign-flip p value rather than a multiple-comparison claim.
    event_result["bh_q"] = float(exact_sign_flip_two_sided(
        event_aligned["event_coverage_candidate"].to_numpy(dtype=float)
        - event_aligned["event_coverage_baseline"].to_numpy(dtype=float)
    ))

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段223只读取阶段222冻结的NOAA第三来源三臂逐站点test"
            "明细和逐种子事件指标，按11个NOAA站点做10000次站点簇"
            "配对bootstrap，并对F1、Recall、Precision、FAR做精确"
            "sign-flip检验和BH校正。不重新训练、不重新选择特征、"
            "窗口、权重或阈值，不使用NOAA test标签做任何选择。"
            "事件覆盖是单一预注册主终点，其bh_q字段直接记录精确"
            "sign-flip p值，不构成BH多重校正家族。"
            "主比较为noaa_valid_selected减noaa_valid_recalibrated。"
        ),
        "primary_far": primary_far,
        "comparison": "noaa_valid_selected_minus_noaa_valid_recalibrated",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "station_count": int(len(aligned)),
        "stage222_json_sha256": sha256_file(STAGE222_JSON),
        "stage222_station_sha256": sha256_file(STAGE222_STATION),
        "stage222_station_event_sha256": sha256_file(STAGE222_STATION_EVENT),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "primary_f1": clean_records([
            next(row for row in metric_rows if row["metric"] == "f1")
        ])[0],
        "primary_event_coverage": clean_records([event_result])[0],
        "metric_uncertainty": clean_records(metric_table.to_dict("records")),
        "limitations": [
            "只有11个NOAA站点，站点簇区间只描述当前站点集合的不确定性。",
            "修正valid攻击注入后，目标域valid本地选择在NOAA上相对DWD特征/窗口/权重配合NOAA重校准为正，但仍不能扩展为所有机构普适领先。",
            "DWD绝对阈值直接迁移在NOAA上仍为保留负结果。",
            "事件覆盖是单一预注册主终点，其bh_q字段直接记录精确sign-flip p值，不构成BH多重校正家族。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    metric_table.to_csv(
        TABLES / f"{PREFIX}_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_aligned.to_csv(
        TABLES / f"{PREFIX}_paired_station.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段223 NOAA第三来源迁移站点簇不确定性",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 主FAR：`{primary_far}`；站点：`{len(aligned)}`",
        "- 主比较：`noaa_valid_selected - noaa_valid_recalibrated`",
        f"- Bootstrap：`{BOOTSTRAP_ITERATIONS}` 次；seed `{BOOTSTRAP_SEED}`",
        "",
        "## 站点簇配对结果",
        "",
        "| 指标 | 差值均值 | 95%区间 | 正/负站点 | Sign-flip p | BH q |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in metric_table.to_dict("records"):
        lines.append(
            f"| {row['metric']} | {row['mean_difference']:+.6f} | "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | "
            f"{int(row['positive_stations'])}/"
            f"{int(row['negative_stations'])} | "
            f"{row['sign_flip_p']:.6f} | {row['bh_q']:.6f} |"
        )
    lines += [
        "",
        "## 主终点事件覆盖",
        "",
        f"- 差值：`{event_result['mean_difference']:+.6f}`",
        f"- 95%区间：`[{event_result['ci95_low']:+.6f}, "
        f"{event_result['ci95_high']:+.6f}]`",
        f"- 正/负站点：`{event_result['positive_stations']}/"
        f"{event_result['negative_stations']}`",
        f"- Sign-flip p：`{event_result['sign_flip_p']:.6f}`",
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
                "status": payload["status"],
                "primary_f1": payload["primary_f1"],
                "primary_event_coverage": payload["primary_event_coverage"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
