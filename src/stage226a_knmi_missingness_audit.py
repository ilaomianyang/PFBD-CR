from __future__ import annotations

"""Stage 226a: audit KNMI missingness before any performance evaluation.

Stage 226 showed a large missing share in the frozen 12-station KNMI
table.  This stage localizes that missingness by station, split, and
year.  The retain/exclude decision uses train and validation rows only;
test availability is reported separately and never enters the decision.
It does not train a model or evaluate performance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
INPUT = TABLES / "stage226_knmi_normalized.parquet"
PREFIX = "stage226a_knmi_missingness_audit"
DECISION_SPLITS = ("train", "valid")
TEST_SPLIT = "test"
AVAILABILITY_THRESHOLD = 0.95


def availability_summary(
    frame: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    summary = (
        frame.groupby("station_id", sort=True)
        .agg(
            **{
                f"{name}_rows": ("observation_present", "size"),
                f"{name}_observation_present": (
                    "observation_present",
                    "sum",
                ),
                f"{name}_temperature_present": (
                    "temperature_c",
                    lambda values: int(values.notna().sum()),
                ),
                f"{name}_humidity_present": (
                    "humidity_pct",
                    lambda values: int(values.notna().sum()),
                ),
            }
        )
        .reset_index()
    )
    for variable in ("observation", "temperature", "humidity"):
        summary[f"{name}_{variable}_present_pct"] = (
            summary[f"{name}_{variable}_present"]
            / summary[f"{name}_rows"]
        )
    return summary


def main() -> int:
    frame = pd.read_parquet(INPUT)
    frame["year"] = frame["timestamp_utc"].dt.year
    rows: list[dict[str, object]] = []
    for (station_id, split), group in frame.groupby(
        ["station_id", "split"], sort=True
    ):
        rows.append(
            {
                "station_id": station_id,
                "split": split,
                "rows": int(len(group)),
                "observation_present": int(
                    group["observation_present"].sum()
                ),
                "observation_present_pct": float(
                    group["observation_present"].mean()
                ),
                "temperature_present": int(
                    group["temperature_c"].notna().sum()
                ),
                "temperature_present_pct": float(
                    group["temperature_c"].notna().mean()
                ),
                "humidity_present": int(
                    group["humidity_pct"].notna().sum()
                ),
                "humidity_present_pct": float(
                    group["humidity_pct"].notna().mean()
                ),
                "timestamp_min": str(group["timestamp_utc"].min()),
                "timestamp_max": str(group["timestamp_utc"].max()),
            }
        )
    split_audit = pd.DataFrame(rows)

    year_audit = (
        frame.groupby(["station_id", "year"], sort=True)
        .agg(
            rows=("observation_present", "size"),
            observation_present=("observation_present", "sum"),
            temperature_present=("temperature_c", lambda values: int(values.notna().sum())),
            humidity_present=("humidity_pct", lambda values: int(values.notna().sum())),
        )
        .reset_index()
    )
    year_audit["observation_present_pct"] = (
        year_audit["observation_present"] / year_audit["rows"]
    )
    year_audit["temperature_present_pct"] = (
        year_audit["temperature_present"] / year_audit["rows"]
    )
    year_audit["humidity_present_pct"] = (
        year_audit["humidity_present"] / year_audit["rows"]
    )

    station_summary = (
        frame.groupby("station_id", sort=True)
        .agg(
            rows=("observation_present", "size"),
            observation_present=("observation_present", "sum"),
            temperature_present=("temperature_c", lambda values: int(values.notna().sum())),
            humidity_present=("humidity_pct", lambda values: int(values.notna().sum())),
            first_timestamp=("timestamp_utc", "min"),
            last_timestamp=("timestamp_utc", "max"),
        )
        .reset_index()
    )
    station_summary["observation_present_pct"] = (
        station_summary["observation_present"] / station_summary["rows"]
    )
    station_summary["temperature_present_pct"] = (
        station_summary["temperature_present"] / station_summary["rows"]
    )
    station_summary["humidity_present_pct"] = (
        station_summary["humidity_present"] / station_summary["rows"]
    )
    decision_frame = frame[frame["split"].isin(DECISION_SPLITS)]
    test_frame = frame[frame["split"].eq(TEST_SPLIT)]
    if decision_frame.empty or test_frame.empty:
        raise RuntimeError("Stage226 must contain train/valid and test rows")
    decision_summary = availability_summary(decision_frame, "decision")
    test_summary = availability_summary(test_frame, "test")
    station_summary = station_summary.merge(
        decision_summary,
        on="station_id",
        how="left",
        validate="one_to_one",
    ).merge(
        test_summary,
        on="station_id",
        how="left",
        validate="one_to_one",
    )
    station_summary["first_observation"] = (
        frame.loc[frame["observation_present"]]
        .groupby("station_id")["timestamp_utc"]
        .min()
        .reindex(station_summary["station_id"])
        .to_numpy()
    )
    station_summary["last_observation"] = (
        frame.loc[frame["observation_present"]]
        .groupby("station_id")["timestamp_utc"]
        .max()
        .reindex(station_summary["station_id"])
        .to_numpy()
    )
    for frame_to_clean in (station_summary, year_audit, split_audit):
        for column in frame_to_clean.select_dtypes(
            include=["float", "float64"]
        ).columns:
            frame_to_clean[column] = frame_to_clean[column].fillna(0.0)
    for column in [
        "observation_present_pct",
        "temperature_present_pct",
        "humidity_present_pct",
    ]:
        station_summary[column] = station_summary[column].fillna(0.0)
    for column in [
        column
        for column in station_summary.columns
        if column.endswith("_present_pct")
    ]:
        station_summary[column] = station_summary[column].fillna(0.0)
    for column in [
        "observation_present_pct",
        "temperature_present_pct",
        "humidity_present_pct",
    ]:
        year_audit[column] = year_audit[column].fillna(0.0)
    for column in [
        "first_timestamp",
        "last_timestamp",
        "first_observation",
        "last_observation",
    ]:
        station_summary[column] = station_summary[column].map(
            lambda value: "" if pd.isna(value) else str(value)
        )

    split_audit.to_csv(
        TABLES / f"{PREFIX}_split.csv",
        index=False,
        encoding="utf-8-sig",
    )
    year_audit.to_csv(
        TABLES / f"{PREFIX}_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_summary.to_csv(
        TABLES / f"{PREFIX}_station.csv",
        index=False,
        encoding="utf-8-sig",
    )
    retained_station_ids = (
        station_summary.loc[
            station_summary[
                "decision_temperature_present_pct"
            ].ge(AVAILABILITY_THRESHOLD)
            & station_summary[
                "decision_humidity_present_pct"
            ].ge(AVAILABILITY_THRESHOLD),
            "station_id",
        ]
        .astype(int)
        .tolist()
    )
    excluded_station_ids = (
        station_summary.loc[
            station_summary[
                "decision_temperature_present_pct"
            ].lt(AVAILABILITY_THRESHOLD)
            | station_summary[
                "decision_humidity_present_pct"
            ].lt(AVAILABILITY_THRESHOLD),
            "station_id",
        ]
        .astype(int)
        .tolist()
    )
    station_summary["decision"] = np.where(
        station_summary["station_id"]
        .astype(int)
        .isin(retained_station_ids),
        "retain",
        "exclude_missing_primary_variables",
    )
    payload = {
        "status": "MISSINGNESS_LOCALIZED",
        "protocol": (
            "阶段226a只定位KNMI冻结12站数据中的缺失分布，"
            "按站点、切分和年份审计观测可用率。站点保留或排除只依据"
            "2021-2023 train与2024 valid主变量可用率是否均不低于"
            f"{AVAILABILITY_THRESHOLD:.2f}；2025 test可用率仅作事后"
            "数据质量报告，不参与站点、特征、窗口、权重、阈值或"
            "攻击位置选择。不训练模型、不选择阈值、不计算性能。"
        ),
        "decision_splits": list(DECISION_SPLITS),
        "test_split": TEST_SPLIT,
        "availability_threshold": AVAILABILITY_THRESHOLD,
        "station_count": int(frame["station_id"].nunique()),
        "rows": int(len(frame)),
        "observation_present_rows": int(
            frame["observation_present"].sum()
        ),
        "temperature_present_rows": int(
            frame["temperature_c"].notna().sum()
        ),
        "humidity_present_rows": int(
            frame["humidity_pct"].notna().sum()
        ),
        "retained_station_ids": retained_station_ids,
        "excluded_station_ids": excluded_station_ids,
        "retained_station_count": len(retained_station_ids),
        "excluded_station_count": len(excluded_station_ids),
        "decision_rows": int(len(decision_frame)),
        "test_rows": int(len(test_frame)),
        "station_summary": station_summary.to_dict(orient="records"),
        "split_summary": split_audit.to_dict(orient="records"),
        "limitations": [
            "这是缺失审计，不是性能结果。",
            "站点排除只使用train+valid可用率，不使用test可用率或test性能。",
            "这是下载后、性能评价前的可用性协议更正，不是性能筛选。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段226a KNMI缺失定位审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 站点：`{payload['station_count']}`；总行数：`{payload['rows']}`",
        f"- 决定依据：`{DECISION_SPLITS}`；可用率阈值：`{AVAILABILITY_THRESHOLD:.2f}`；test只报告不参与决定",
        f"- 有观测行：`{payload['observation_present_rows']}`；温度可用：`{payload['temperature_present_rows']}`；湿度可用：`{payload['humidity_present_rows']}`",
        "",
        "## 站点汇总",
        "",
        "| 站点 | 总行 | train+valid温度率 | train+valid湿度率 | test温度率 | test湿度率 | 决定 | 首次观测 | 最后观测 |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in station_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['station_id']} | {row['rows']} | "
            f"{row['decision_temperature_present_pct']:.4f} | "
            f"{row['decision_humidity_present_pct']:.4f} | "
            f"{row['test_temperature_present_pct']:.4f} | "
            f"{row['test_humidity_present_pct']:.4f} | "
            f"{row['decision']} | "
            f"{row['first_observation']} | {row['last_observation']} |"
        )
    lines += [
        "",
        "站点保留或排除只依据train+valid主变量可用率；test可用率仅作"
        "数据质量披露，不参与任何选择，也不评价检测性能。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "stations": payload["station_count"],
                "rows": payload["rows"],
                "observation_present_rows": payload[
                    "observation_present_rows"
                ],
                "temperature_present_rows": payload[
                    "temperature_present_rows"
                ],
                "humidity_present_rows": payload[
                    "humidity_present_rows"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
