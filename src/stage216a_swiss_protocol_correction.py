from __future__ import annotations

"""Stage 216A: record the MeteoSwiss availability correction.

Stage 214 froze 12 MeteoSwiss stations before downloading observations.
Stage 215 downloaded all 12 official files, and Stage 216 normalized
them.  The post-download audit found that station ``aeg`` contains no
temperature, humidity, dew-point, or pressure values in 2021-2025.

This stage records that availability correction before any detector
performance is computed.  It does not change the years, split, attack
types, attack lengths, seeds, FAR grid, primary endpoint, clustering
unit, or multiplicity rule.  It does not train a detector, select a
threshold, or compute performance.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL = TABLES / "stage214_swiss_protocol_preregistration.json"
NORMALIZED = TABLES / "stage216_swiss_normalized.json"
STATION_AUDIT = TABLES / "stage216_swiss_normalized_station_audit.csv"
PREFIX = "stage216a_swiss_protocol_correction"


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    normalized = json.loads(NORMALIZED.read_text(encoding="utf-8"))
    audit = pd.read_csv(STATION_AUDIT, dtype={"station_id": str})
    if normalized["status"] != "PASS_PIPELINE_REVIEW_REQUIRED":
        raise RuntimeError("Stage216 normalization did not pass")

    excluded = audit[
        audit["missing_observation_rows"].eq(audit["rows"])
    ].copy()
    retained = audit[
        ~audit["station_id"].isin(excluded["station_id"])
    ].copy()
    if excluded["station_id"].tolist() != ["aeg"]:
        raise RuntimeError(
            "expected only station aeg to be fully unavailable, found "
            f"{excluded['station_id'].tolist()}"
        )
    if len(retained) != 11:
        raise RuntimeError(f"expected 11 retained stations, found {len(retained)}")

    payload = {
        "status": "PROTOCOL_AVAILABILITY_CORRECTED_BEFORE_PERFORMANCE",
        "protocol": (
            "阶段216A在性能评价前记录MeteoSwiss下载后的可用性修正："
            "阶段214预先冻结的12个站点中，aeg在2021-2025的所有小时"
            "温度、湿度、露点和气压均为空，无法形成任何正常训练段或"
            "攻击评价行，因此从第二来源性能评价中排除；其余11个站点"
            "保留。年份、train/valid/test划分、攻击类型、攻击长度、"
            "种子、FAR网格、主终点、聚类单位和多重校正保持不变。"
            "本阶段不训练模型、不选择阈值、不计算性能结果；"
            "排除aeg只依据完整协议年份全空，不依据任何test性能，"
            "test仍只用于后续评价。"
        ),
        "original_protocol_status": protocol["status"],
        "original_station_count": int(protocol["station_count"]),
        "original_station_ids": protocol["station_ids"],
        "excluded_stations": excluded.to_dict(orient="records"),
        "retained_station_count": int(len(retained)),
        "retained_station_ids": retained["station_id"].tolist(),
        "exclusion_rule": (
            "Exclude a pre-registered station only when every protocol-year "
            "hour has no temperature, humidity, dew-point, or pressure "
            "observation; record the exact post-download audit evidence."
        ),
        "frozen_items_unchanged": [
            "provider",
            "dataset_id",
            "years",
            "train_valid_test_split",
            "primary_variables",
            "auxiliary_variables",
            "missing_rule",
            "attack_types",
            "attack_length",
            "attack_blocks",
            "seeds",
            "far_grid",
            "primary_far",
            "primary_endpoint",
            "secondary_endpoint",
            "cluster_unit",
            "multiplicity",
            "selection_boundary",
            "attack_boundary",
        ],
        "performance_status": "NOT_EVALUATED",
        "limitations": [
            "这是下载后、性能评价前的可用性修正，不是性能结果。",
            "排除aeg依据的是完整协议年份全空，不依据任何test性能。",
            "MeteoSwiss没有与DWD QN_9等价的逐行质量码，不能伪造质量码。",
            "所有后续攻击仍必须是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段216A MeteoSwiss可用性修正记录",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 原冻结站点数：`{payload['original_station_count']}`",
        f"- 排除站点：`{', '.join(payload['retained_station_ids'] and excluded['station_id'].tolist())}`",
        f"- 保留站点数：`{payload['retained_station_count']}`",
        f"- 保留站点：`{', '.join(payload['retained_station_ids'])}`",
        "",
        "排除规则：只有在完整协议年份内所有小时均无温度、湿度、露点或气压观测时排除，并保留下载后审计证据。",
        "",
        "## 未改变项",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["frozen_items_unchanged"])
    lines += [
        "",
        "本阶段不训练模型、不选择阈值、不计算性能结果；test仍只用于后续评价。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "original_station_count": payload["original_station_count"],
                "excluded_station_ids": excluded["station_id"].tolist(),
                "retained_station_count": payload["retained_station_count"],
                "retained_station_ids": payload["retained_station_ids"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
