from __future__ import annotations

"""Stage 220: station-clustered uncertainty for the Stage219 transfer.

This stage is read-only with respect to Stage219.  It does not fit
detectors, recalibrate thresholds, or select features.  It quantifies
whether the target-domain valid local-selection advantage over the
DWD feature/window/weight transfer with Swiss-valid recalibration is
stable when stations are the resampling unit.

The primary comparison is:

    swiss_valid_selected - swiss_valid_recalibrated

because both arms use the same Swiss valid/test boundary and differ only
in whether the frozen feature/window/weight transferred from DWD is used
with a Swiss-valid-recalibrated threshold or whether the local valid
selection is used.  The exact sign-flip test is reported alongside the
bootstrap interval.

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

STAGE219_JSON = TABLES / "stage219_swiss_dwd_protection_transfer.json"
STAGE219_STATION = (
    TABLES / "stage219_swiss_dwd_protection_transfer_station_detail.csv"
)
STAGE219_EVENTS = (
    TABLES / "stage219_swiss_dwd_protection_transfer_events.csv"
)
STAGE219_EVALUATION = (
    TABLES / "stage219_swiss_dwd_protection_transfer_evaluation.csv"
)
STAGE219_STATION_EVENT = (
    TABLES / "stage219_swiss_dwd_protection_transfer_station_event_detail.csv"
)
PREFIX = "stage220_swiss_dwd_transfer_uncertainty"
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260920


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


def paired_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("paired bootstrap arrays must be one-dimensional")
    if candidate.size == 0 or not (
        np.isfinite(candidate).all() and np.isfinite(baseline).all()
    ):
        raise ValueError("paired bootstrap arrays must be finite and non-empty")
    differences = candidate - baseline
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(
        0,
        differences.size,
        size=(int(iterations), differences.size),
    )
    boot = differences[draws].mean(axis=1)
    return {
        "stations": int(differences.size),
        "mean_difference": float(differences.mean()),
        "std_difference": float(differences.std(ddof=0)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_mean": float(boot.mean()),
        "bootstrap_std": float(boot.std(ddof=0)),
        "positive_probability": float(np.mean(boot > 0.0)),
        "negative_probability": float(np.mean(boot < 0.0)),
        "positive_stations": int(np.sum(differences > 0.0)),
        "negative_stations": int(np.sum(differences < 0.0)),
        "zero_stations": int(np.sum(differences == 0.0)),
    }


def main() -> int:
    stage219 = load_json(STAGE219_JSON)
    station_detail = pd.read_csv(STAGE219_STATION)
    events = pd.read_csv(STAGE219_EVENTS)
    evaluation = pd.read_csv(STAGE219_EVALUATION)
    station_event_detail = pd.read_csv(STAGE219_STATION_EVENT)
    required_arms = {
        "dwd_threshold_transfer",
        "swiss_valid_recalibrated",
        "swiss_valid_selected",
    }
    if set(station_detail["arm"]) != required_arms:
        raise RuntimeError("Stage219 station detail arm set changed")
    if set(events["arm"]) != required_arms:
        raise RuntimeError("Stage219 event arm set changed")
    if set(evaluation["arm"]) != required_arms:
        raise RuntimeError("Stage219 evaluation arm set changed")

    primary_far = float(stage219["primary_far"])
    if not np.isclose(primary_far, 0.01):
        raise RuntimeError("Stage219 primary FAR changed")

    station_frames: dict[str, pd.DataFrame] = {}
    for arm in sorted(required_arms):
        station_frames[arm] = station_metric(station_detail, arm)

    baseline = station_frames["swiss_valid_recalibrated"]
    candidate = station_frames["swiss_valid_selected"]
    aligned = baseline.merge(
        candidate,
        on="station_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(aligned) != 11:
        raise RuntimeError(
            f"expected 11 Swiss stations, found {len(aligned)}"
        )

    metrics_to_test = [
        "f1",
        "recall",
        "precision",
        "false_alarm_rate",
    ]
    metric_rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for index, metric in enumerate(metrics_to_test):
        bootstrap = paired_bootstrap(
            aligned[f"{metric}_candidate"].to_numpy(dtype=float),
            aligned[f"{metric}_baseline"].to_numpy(dtype=float),
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED + index,
        )
        differences = (
            aligned[f"{metric}_candidate"].to_numpy(dtype=float)
            - aligned[f"{metric}_baseline"].to_numpy(dtype=float)
        )
        p_value = exact_sign_flip_two_sided(differences)
        p_values.append(float(p_value))
        metric_rows.append(
            {
                "metric": metric,
                **bootstrap,
                "sign_flip_p": float(p_value),
            }
        )
    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(metric_rows, q_values):
        row["bh_q"] = float(q_value)
    metric_table = pd.DataFrame(metric_rows)

    event_baseline = (
        station_event_detail[
            station_event_detail["arm"].eq("swiss_valid_recalibrated")
        ]
        .groupby("station_id", sort=True, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            events=("events", "sum"),
            detected_events=("detected_events", "sum"),
        )
        .sort_values("station_id")
    )
    event_candidate = (
        station_event_detail[
            station_event_detail["arm"].eq("swiss_valid_selected")
        ]
        .groupby("station_id", sort=True, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            events=("events", "sum"),
            detected_events=("detected_events", "sum"),
        )
        .sort_values("station_id")
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
    event_bootstrap = paired_bootstrap(
        event_aligned["event_coverage_candidate"].to_numpy(dtype=float),
        event_aligned["event_coverage_baseline"].to_numpy(dtype=float),
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED + 200,
    )
    event_differences = (
        event_aligned["event_coverage_candidate"].to_numpy(dtype=float)
        - event_aligned["event_coverage_baseline"].to_numpy(dtype=float)
    )
    event_bootstrap["metric"] = "event_coverage"
    event_bootstrap["sign_flip_p"] = exact_sign_flip_two_sided(
        event_differences
    )
    event_bootstrap["bh_q"] = float(
        benjamini_hochberg([float(event_bootstrap["sign_flip_p"])])[0]
    )

    event_summary = (
        events.groupby("arm", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            event_coverage_mean=("event_coverage", "mean"),
            event_coverage_std=("event_coverage", "std"),
            row_recall_mean=("row_recall", "mean"),
        )
        .sort_values("arm")
    )
    event_rows: list[dict[str, object]] = []
    station_event = (
        station_detail.groupby(
            ["arm", "station_id"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            attack_rows=("attack_rows", "sum"),
        )
        .sort_values(["arm", "station_id"])
    )
    event_bootstrap_rows: list[dict[str, object]] = []
    for arm in sorted(required_arms):
        arm_events = events[events["arm"].eq(arm)]
        if len(arm_events) != len(stage219["seeds"]):
            raise RuntimeError(f"unexpected event seed count for {arm}")
        values = dict(
            zip(
                station_event.loc[
                    station_event["arm"].eq(arm), "station_id"
                ].astype(str),
                station_frames[arm]["f1"].astype(float),
            )
        )
        bootstrap = cluster_bootstrap_mean(
            values,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED + 100,
        )
        event_bootstrap_rows.append(
            {
                "arm": arm,
                "station_f1_mean": float(bootstrap["mean"]),
                "station_f1_std": float(bootstrap["std"]),
                "station_f1_ci95_low": float(bootstrap["ci95_low"]),
                "station_f1_ci95_high": float(bootstrap["ci95_high"]),
            }
        )
    station_bootstrap = pd.DataFrame(event_bootstrap_rows)

    primary = metric_table[metric_table["metric"].eq("f1")].iloc[0]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段220只读取阶段219冻结的三臂逐站点test明细和逐种子事件"
            "指标，按11个MeteoSwiss站点做10000次站点簇配对bootstrap，"
            "并对F1、Recall、Precision、FAR做精确sign-flip检验和BH"
            "校正。不重新训练、不重新选择特征、窗口、权重或阈值，"
            "不使用瑞士test标签做任何选择。主比较为"
            "swiss_valid_selected减swiss_valid_recalibrated。"
        ),
        "primary_far": primary_far,
        "comparison": (
            "swiss_valid_selected_minus_swiss_valid_recalibrated"
        ),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "station_count": int(len(aligned)),
        "stage219_json_sha256": sha256_file(STAGE219_JSON),
        "stage219_station_sha256": sha256_file(STAGE219_STATION),
        "stage219_events_sha256": sha256_file(STAGE219_EVENTS),
        "stage219_evaluation_sha256": sha256_file(STAGE219_EVALUATION),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "primary_f1": clean_records([primary.to_dict()])[0],
        "primary_event_coverage": clean_records(
            [event_bootstrap]
        )[0],
        "metric_uncertainty": clean_records(
            metric_table.to_dict("records")
        ),
        "station_bootstrap": clean_records(
            station_bootstrap.to_dict("records")
        ),
        "event_aggregate": clean_records(
            event_summary.to_dict("records")
        ),
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
    station_bootstrap.to_csv(
        TABLES / f"{PREFIX}_station_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aligned.to_csv(
        TABLES / f"{PREFIX}_paired_station.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段220 DWD保护候选跨机构迁移站点簇不确定性",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 主FAR：`{primary_far}`；站点：`{payload['station_count']}`",
        "- 主比较：`swiss_valid_selected - swiss_valid_recalibrated`",
        f"- Bootstrap：`{BOOTSTRAP_ITERATIONS}` 次；seed `{BOOTSTRAP_SEED}`",
        "",
        "## 站点簇配对结果",
        "",
        "| 指标 | 差值均值 | 95%区间 | 正站点 | 负站点 | Sign-flip p | BH q |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in metric_table.to_dict("records"):
        lines.append(
            f"| {row['metric']} | {row['mean_difference']:+.6f} | "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | "
            f"{int(row['positive_stations'])} | "
            f"{int(row['negative_stations'])} | "
            f"{row['sign_flip_p']:.6f} | {row['bh_q']:.6f} |"
        )
    lines += [
        "",
        "## 主终点事件覆盖",
        "",
        f"- 差值均值：`{event_bootstrap['mean_difference']:+.6f}`",
        f"- 站点簇95%区间：`[{event_bootstrap['ci95_low']:+.6f}, "
        f"{event_bootstrap['ci95_high']:+.6f}]`",
        f"- 正向站点：`{event_bootstrap['positive_stations']}`；"
        f"负向站点：`{event_bootstrap['negative_stations']}`；"
        f"零差站点：`{event_bootstrap['zero_stations']}`",
        f"- Sign-flip p：`{event_bootstrap['sign_flip_p']:.6f}`；"
        f"BH q：`{event_bootstrap['bh_q']:.6f}`",
    ]
    lines += [
        "",
        "## 限制",
        "",
        "- 站点数只有11个，站点簇bootstrap只能描述当前MeteoSwiss站点集合的不确定性。",
        "- DWD绝对阈值直接迁移完全失效，是跨机构分数尺度错配的负结果；修正valid攻击注入后，本阶段评价的是目标域valid本地选择相对DWD特征/窗口/权重配合瑞士valid重校准的优势。",
        "- 所有攻击都是透明受控注入，不是真实攻击日志；结果不能解释为真实入侵率。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "primary_f1": payload["primary_f1"],
                "primary_event_coverage": payload[
                    "primary_event_coverage"
                ],
                "metric_uncertainty": payload["metric_uncertainty"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
