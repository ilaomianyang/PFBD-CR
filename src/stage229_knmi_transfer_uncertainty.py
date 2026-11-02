from __future__ import annotations

"""Stage 229: station-clustered uncertainty for the KNMI transfer.

This stage is read-only with respect to Stage228.  It does not fit
detectors, recalibrate thresholds, or select features.  It quantifies
whether the KNMI-valid local-selection advantage over the DWD
feature/window/weight transfer with KNMI-valid recalibration is stable
when stations are the resampling unit.

The primary comparison is:

    knmi_valid_selected - knmi_valid_recalibrated

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
    exact_sign_flip_two_sided,
    sha256_file,
)
from stage223_noaa_transfer_uncertainty import (  # noqa: E402
    clean_records,
    paired_interval,
    station_metric,
)

STAGE228_JSON = TABLES / "stage228_knmi_dwd_protection_transfer.json"
STAGE228_STATION = (
    TABLES / "stage228_knmi_dwd_protection_transfer_station_detail.csv"
)
STAGE228_STATION_EVENT = (
    TABLES
    / "stage228_knmi_dwd_protection_transfer_station_event_detail.csv"
)
PREFIX = "stage229_knmi_transfer_uncertainty"
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260923


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    stage228 = load_json(STAGE228_JSON)
    station_detail = pd.read_csv(
        STAGE228_STATION,
        dtype={"station_id": str},
    )
    station_event_detail = pd.read_csv(
        STAGE228_STATION_EVENT,
        dtype={"station_id": str},
    )
    required_arms = {
        "dwd_threshold_transfer",
        "knmi_valid_recalibrated",
        "knmi_valid_selected",
    }
    if set(station_detail["arm"]) != required_arms:
        raise RuntimeError("Stage228 station arm set changed")
    if set(station_event_detail["arm"]) != required_arms:
        raise RuntimeError("Stage228 station-event arm set changed")
    primary_far = float(stage228["primary_far"])
    if not np.isclose(primary_far, 0.01):
        raise RuntimeError("Stage228 primary FAR changed")

    baseline = station_metric(
        station_detail,
        "knmi_valid_recalibrated",
    )
    candidate = station_metric(
        station_detail,
        "knmi_valid_selected",
    )
    aligned = baseline.merge(
        candidate,
        on="station_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(aligned) != 6:
        raise RuntimeError(
            f"expected 6 KNMI stations, found {len(aligned)}"
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
            station_event_detail["arm"].eq("knmi_valid_recalibrated")
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
            station_event_detail["arm"].eq("knmi_valid_selected")
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
    event_result["bh_q"] = float(
        exact_sign_flip_two_sided(
            event_aligned["event_coverage_candidate"].to_numpy(dtype=float)
            - event_aligned["event_coverage_baseline"].to_numpy(dtype=float)
        )
    )

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段229只读取阶段228冻结的KNMI第三来源三臂逐站点test"
            "明细和逐种子事件指标，按6个KNMI站点做10000次站点簇"
            "配对bootstrap，并对F1、Recall、Precision、FAR做精确"
            "sign-flip检验和BH校正。不重新训练、不重新选择特征、"
            "窗口、权重或阈值，不使用KNMI test标签做任何选择。"
            "事件覆盖是单一预注册主终点，其bh_q字段直接记录精确"
            "sign-flip p值，不构成BH多重校正家族。"
            "主比较为knmi_valid_selected减knmi_valid_recalibrated。"
        ),
        "primary_far": primary_far,
        "comparison": "knmi_valid_selected_minus_knmi_valid_recalibrated",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "station_count": int(len(aligned)),
        "stage228_json_sha256": sha256_file(STAGE228_JSON),
        "stage228_station_sha256": sha256_file(STAGE228_STATION),
        "stage228_station_event_sha256": sha256_file(
            STAGE228_STATION_EVENT
        ),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "primary_f1": clean_records([
            next(row for row in metric_rows if row["metric"] == "f1")
        ])[0],
        "primary_event_coverage": clean_records([event_result])[0],
        "metric_uncertainty": clean_records(metric_table.to_dict("records")),
        "limitations": [
            "只有6个KNMI可用站点，站点簇区间只描述当前站点集合的不确定性。",
            "KNMI主变量可用的站点数少于原始冻结站点数，这是下载后可用性更正，不是性能筛选。",
            "DWD绝对阈值直接迁移在KNMI上仍为保留负结果；修正valid攻击注入后，目标域valid本地选择相对DWD特征/窗口/权重配合KNMI重校准为正。",
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
        "# 阶段229 KNMI第三来源迁移站点簇不确定性",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 主FAR：`{primary_far}`；站点：`{len(aligned)}`",
        "- 主比较：`knmi_valid_selected - knmi_valid_recalibrated`",
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
