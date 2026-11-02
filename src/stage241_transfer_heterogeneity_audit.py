from __future__ import annotations

"""Stage 241: heterogeneity audit of the frozen transfer advantage.

Stage 231/232/239 established that target-domain valid local selection
outperformed target-domain valid recalibration in four providers, but
also showed that the advantage is concentrated in ``stuck``.  This stage
is a read-only follow-up.  It does not train, select, recalibrate, or
modify any frozen artifact.

The audit answers three reviewer-facing questions:

1. Is the provider-equal advantage carried by a small number of target
   stations or by a broad station-level majority?
2. Does removing any single target provider reverse the provider-equal
   direction?
3. Is there any frozen evidence for a non-``stuck`` advantage, or is the
   positive direction confined to ``stuck``?

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
PREFIX = "stage241_transfer_heterogeneity_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage218_swiss_negative_result_uncertainty import (  # noqa: E402
    sha256_file,
)


SOURCE_SPECS = [
    {
        "source": "MeteoSwiss",
        "station_path": (
            TABLES
            / "stage219_swiss_dwd_protection_transfer_station_detail.csv"
        ),
        "type_path": (
            TABLES / "stage219_swiss_dwd_protection_transfer_type_detail.csv"
        ),
        "baseline_arm": "swiss_valid_recalibrated",
        "candidate_arm": "swiss_valid_selected",
        "station_count": 11,
    },
    {
        "source": "NOAA ISD",
        "station_path": (
            TABLES
            / "stage222_noaa_dwd_protection_transfer_station_detail.csv"
        ),
        "type_path": (
            TABLES / "stage222_noaa_dwd_protection_transfer_type_detail.csv"
        ),
        "baseline_arm": "noaa_valid_recalibrated",
        "candidate_arm": "noaa_valid_selected",
        "station_count": 11,
    },
    {
        "source": "KNMI",
        "station_path": (
            TABLES
            / "stage228_knmi_dwd_protection_transfer_station_detail.csv"
        ),
        "type_path": (
            TABLES / "stage228_knmi_dwd_protection_transfer_type_detail.csv"
        ),
        "baseline_arm": "knmi_valid_recalibrated",
        "candidate_arm": "knmi_valid_selected",
        "station_count": 6,
    },
    {
        "source": "ECCC",
        "station_path": (
            TABLES
            / "stage238_eccc_dwd_protection_transfer_station_detail.csv"
        ),
        "type_path": (
            TABLES / "stage238_eccc_dwd_protection_transfer_type_detail.csv"
        ),
        "baseline_arm": "eccc_valid_recalibrated",
        "candidate_arm": "eccc_valid_selected",
        "station_count": 14,
    },
]


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


def station_metrics(detail: pd.DataFrame, arm: str) -> pd.DataFrame:
    selected = detail[detail["arm"].eq(arm)]
    if selected.empty:
        raise RuntimeError(f"missing station arm: {arm}")
    grouped = (
        selected.groupby("station_id", sort=True, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tn=("tn", "sum"),
        )
    )
    denominator = (
        2.0 * grouped["tp"] + grouped["fp"] + grouped["fn"]
    ).to_numpy(dtype=float)
    grouped["f1"] = np.divide(
        2.0 * grouped["tp"].to_numpy(dtype=float),
        denominator,
        out=np.zeros(len(grouped), dtype=float),
        where=denominator > 0.0,
    )
    return grouped[["station_id", "seeds", "tp", "fp", "fn", "tn", "f1"]]


def station_effects(spec: dict[str, object]) -> pd.DataFrame:
    detail = pd.read_csv(
        Path(spec["station_path"]),
        dtype={"station_id": str},
    )
    baseline = station_metrics(detail, str(spec["baseline_arm"]))
    candidate = station_metrics(detail, str(spec["candidate_arm"]))
    aligned = baseline.merge(
        candidate,
        on="station_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(aligned) != int(spec["station_count"]):
        raise RuntimeError(
            f"{spec['source']} station count changed: {len(aligned)}"
        )
    aligned.insert(0, "source", str(spec["source"]))
    aligned["f1_difference"] = (
        aligned["f1_candidate"] - aligned["f1_baseline"]
    )
    return aligned


def type_effects(spec: dict[str, object]) -> pd.DataFrame:
    detail = pd.read_csv(Path(spec["type_path"]))
    baseline = (
        detail[detail["arm"].eq(str(spec["baseline_arm"]))]
        .groupby("attack_type", sort=True, as_index=False)
        .agg(
            row_recall_baseline=("row_recall", "mean"),
            event_coverage_baseline=("event_coverage", "mean"),
        )
    )
    candidate = (
        detail[detail["arm"].eq(str(spec["candidate_arm"]))]
        .groupby("attack_type", sort=True, as_index=False)
        .agg(
            row_recall_candidate=("row_recall", "mean"),
            event_coverage_candidate=("event_coverage", "mean"),
        )
    )
    aligned = baseline.merge(
        candidate,
        on="attack_type",
        validate="one_to_one",
    )
    aligned.insert(0, "source", str(spec["source"]))
    aligned["row_recall_difference"] = (
        aligned["row_recall_candidate"] - aligned["row_recall_baseline"]
    )
    aligned["event_coverage_difference"] = (
        aligned["event_coverage_candidate"]
        - aligned["event_coverage_baseline"]
    )
    return aligned


def provider_equal(
    effects: pd.DataFrame,
    metric: str,
) -> float:
    return float(
        effects.groupby("source", sort=True)[metric].mean().mean()
    )


def main() -> int:
    required: list[Path] = []
    for spec in SOURCE_SPECS:
        required.extend(
            [Path(spec["station_path"]), Path(spec["type_path"])]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing frozen artifacts for Stage241: " + ", ".join(missing)
        )

    station_frames: list[pd.DataFrame] = []
    type_frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    for spec in SOURCE_SPECS:
        stations = station_effects(spec)
        types = type_effects(spec)
        station_frames.append(stations)
        type_frames.append(types)
        source_rows.append(
            {
                "source": str(spec["source"]),
                "station_count": int(len(stations)),
                "f1_difference": float(stations["f1_difference"].mean()),
                "positive_stations": int(
                    (stations["f1_difference"] > 0.0).sum()
                ),
                "negative_stations": int(
                    (stations["f1_difference"] < 0.0).sum()
                ),
                "zero_stations": int(
                    (stations["f1_difference"] == 0.0).sum()
                ),
                "max_station_share_of_positive_sum": float(
                    stations.loc[
                        stations["f1_difference"] > 0.0,
                        "f1_difference",
                    ].max()
                    / stations.loc[
                        stations["f1_difference"] > 0.0,
                        "f1_difference",
                    ].sum()
                )
                if stations["f1_difference"].gt(0.0).any()
                else None,
            }
        )

    stations = pd.concat(station_frames, ignore_index=True)
    types = pd.concat(type_frames, ignore_index=True)
    sources = sorted(stations["source"].unique().tolist())

    leave_one_target_rows: list[dict[str, object]] = []
    for removed in sources:
        retained = stations[~stations["source"].eq(removed)]
        leave_one_target_rows.append(
            {
                "removed_target": removed,
                "retained_targets": ",".join(
                    source for source in sources if source != removed
                ),
                "target_count": int(retained["source"].nunique()),
                "station_count": int(len(retained)),
                "f1_difference": provider_equal(
                    retained,
                    "f1_difference",
                ),
                "direction_positive": bool(
                    provider_equal(retained, "f1_difference") > 0.0
                ),
            }
        )
    leave_one_target = pd.DataFrame(leave_one_target_rows)

    nonstuck_types = [
        "spike",
        "offset",
        "drift",
        "time_shift",
        "gaussian_noise",
    ]
    type_rows: list[dict[str, object]] = []
    for attack_type in sorted(types["attack_type"].unique().tolist()):
        subset = types[types["attack_type"].eq(attack_type)]
        type_rows.append(
            {
                "attack_type": attack_type,
                "is_stuck": bool(attack_type == "stuck"),
                "provider_count": int(subset["source"].nunique()),
                "row_recall_difference": float(
                    subset["row_recall_difference"].mean()
                ),
                "event_coverage_difference": float(
                    subset["event_coverage_difference"].mean()
                ),
                "row_recall_positive_providers": int(
                    (subset["row_recall_difference"] > 0.0).sum()
                ),
                "event_coverage_positive_providers": int(
                    (subset["event_coverage_difference"] > 0.0).sum()
                ),
            }
        )
    type_summary = pd.DataFrame(type_rows)
    nonstuck = type_summary[
        type_summary["attack_type"].isin(nonstuck_types)
    ]
    provider_equal_f1 = provider_equal(stations, "f1_difference")
    nonstuck_row_recall = float(
        nonstuck["row_recall_difference"].mean()
    )
    nonstuck_event_coverage = float(
        nonstuck["event_coverage_difference"].mean()
    )

    checks: list[dict[str, object]] = [
        {
            "check": "target_count",
            "status": "PASS" if len(sources) == 4 else "FAIL",
            "value": len(sources),
        },
        {
            "check": "station_count",
            "status": "PASS" if len(stations) == 42 else "FAIL",
            "value": len(stations),
        },
        {
            "check": "provider_equal_f1_reconstruction",
            "status": "PASS"
            if np.isclose(
                provider_equal_f1,
                0.04529343767356737,
                rtol=0.0,
                atol=1e-12,
            )
            else "FAIL",
            "value": provider_equal_f1,
        },
        {
            "check": "all_stations_direction_positive",
            "status": "PASS"
            if bool((stations["f1_difference"] > 0.0).all())
            else "FAIL",
            "value": int((stations["f1_difference"] > 0.0).sum()),
        },
        {
            "check": "leave_one_target_direction_positive",
            "status": "PASS"
            if bool(leave_one_target["direction_positive"].all())
            else "FAIL",
            "value": leave_one_target[
                ["removed_target", "f1_difference"]
            ].to_dict("records"),
        },
        {
            "check": "stuck_direction_positive",
            "status": "PASS"
            if float(
                type_summary.loc[
                    type_summary["attack_type"].eq("stuck"),
                    "event_coverage_difference",
                ].iloc[0]
            )
            > 0.0
            else "FAIL",
            "value": float(
                type_summary.loc[
                    type_summary["attack_type"].eq("stuck"),
                    "event_coverage_difference",
                ].iloc[0]
            ),
        },
        {
            "check": "nonstuck_direction_negative",
            "status": "PASS"
            if nonstuck_row_recall < 0.0 and nonstuck_event_coverage < 0.0
            else "FAIL",
            "value": {
                "row_recall_difference": nonstuck_row_recall,
                "event_coverage_difference": nonstuck_event_coverage,
            },
        },
    ]
    failed = [check for check in checks if check["status"] == "FAIL"]
    payload = {
        "status": (
            "PASS_PIPELINE_REVIEW_REQUIRED"
            if not failed
            else "FAIL"
        ),
        "protocol": (
            "Read-only heterogeneity audit of the four frozen transfer "
            "targets. No training, selection, recalibration, or test-label "
            "use for tuning. All attacks are transparent controlled "
            "injections."
        ),
        "target_count": len(sources),
        "station_count": int(len(stations)),
        "provider_equal_f1_difference": provider_equal_f1,
        "nonstuck_row_recall_difference": nonstuck_row_recall,
        "nonstuck_event_coverage_difference": nonstuck_event_coverage,
        "source_rows": clean_records(source_rows),
        "leave_one_target_rows": clean_records(
            leave_one_target.to_dict("records")
        ),
        "type_rows": clean_records(type_summary.to_dict("records")),
        "checks": checks,
        "artifacts": {
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    }

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stations.to_csv(
        TABLES / f"{PREFIX}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_summary.to_csv(
        TABLES / f"{PREFIX}_type_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one_target.to_csv(
        TABLES / f"{PREFIX}_leave_one_target_out.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段241 迁移异质性审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 目标机构数：`{payload['target_count']}`；站点数：`{payload['station_count']}`",
        f"- 机构等权 F1 差值：`{provider_equal_f1:+.6f}`",
        "",
        "## 站点方向",
        "",
        "| 目标机构 | 站点数 | F1差值 | 正向 | 负向 | 零 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in source_rows:
        lines.append(
            f"| {row['source']} | {row['station_count']} | "
            f"{row['f1_difference']:+.6f} | "
            f"{row['positive_stations']} | "
            f"{row['negative_stations']} | "
            f"{row['zero_stations']} |"
        )
    lines += [
        "",
        "## 留一目标机构",
        "",
        "| 移除目标 | 保留目标 | 机构等权F1差值 | 方向为正 |",
        "|---|---|---:|---|",
    ]
    for row in leave_one_target_rows:
        lines.append(
            f"| {row['removed_target']} | {row['retained_targets']} | "
            f"{row['f1_difference']:+.6f} | "
            f"{'是' if row['direction_positive'] else '否'} |"
        )
    lines += [
        "",
        "## 攻击类型",
        "",
        "| 攻击类型 | 行召回差值 | 事件覆盖差值 | 行召回正向机构 | 事件覆盖正向机构 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in type_summary.to_dict("records"):
        lines.append(
            f"| {row['attack_type']} | "
            f"{row['row_recall_difference']:+.6f} | "
            f"{row['event_coverage_difference']:+.6f} | "
            f"{row['row_recall_positive_providers']}/"
            f"{row['provider_count']} | "
            f"{row['event_coverage_positive_providers']}/"
            f"{row['provider_count']} |"
        )
    lines += [
        "",
        "## 结论边界",
        "",
        "- 四个目标的站点级 F1 方向均为正，说明当前方向不是单个站点驱动。",
        "- 移除任一目标机构后，机构等权 F1 方向仍为正。",
        "- 非 `stuck` 的行召回和事件覆盖方向仍为负，不能写成所有攻击类型普适占优。",
        "- 只有四个目标机构，机构级精确检验功效有限；所有攻击均为透明受控注入，不是真实攻击日志。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "target_count": payload["target_count"],
                "station_count": payload["station_count"],
                "provider_equal_f1_difference": provider_equal_f1,
                "nonstuck_row_recall_difference": nonstuck_row_recall,
                "nonstuck_event_coverage_difference": nonstuck_event_coverage,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
