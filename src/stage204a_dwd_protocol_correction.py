from __future__ import annotations

"""Stage 204A: record the DWD schema correction before performance work.

The Stage-204 protocol correctly froze the provider, stations, years,
splits, transparent attacks, FAR grid, endpoint, clustering unit, and
multiplicity rule.  It also recorded two DWD product schema details from
memory rather than from the downloaded archive metadata:

* humidity was labelled ``TF_TU`` in product ``TF``;
* one quality field, ``QN_9``, was used for both products.

The official archives show that the required relative-humidity series is
``RF_TU`` in the ``TU`` archive, whose quality field is ``QN_9``.  The
separate ``TF`` product contains wet-bulb temperature and relative
humidity as ``TF_STD`` and ``RF_STD`` with quality field ``QN_8``.

This stage records the correction explicitly.  It does not alter the
station list, years, split, attack protocol, FAR grid, endpoints,
clustering, multiplicity rule, or any model result.  It also does not
train a detector, select a threshold, or compute performance.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL = TABLES / "stage204_dwd_protocol_preregistration.json"
DOWNLOAD = TABLES / "stage205_dwd_official_download_detail.csv"
PREFIX = "stage204a_dwd_protocol_schema_correction"


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    detail = pd.read_csv(DOWNLOAD, dtype={"station_id": str})
    tu = detail[detail["product"].eq("TU")].copy()
    tf = detail[detail["product"].eq("TF")].copy()

    tu_ok = (
        len(tu) == 12
        and tu["status"].isin(["downloaded", "already_present"]).all()
        and tu["audit_required_columns_ok"].fillna(False).astype(bool).all()
        and tu["audit_error"].isna().all()
    )
    tf_aux_ok = (
        len(tf) == 12
        and tf["status"].isin(["downloaded", "already_present"]).all()
        and tf["audit_error"].isna().all()
    )
    if not (tu_ok and tf_aux_ok):
        raise RuntimeError("Stage205 archives are not ready for correction")

    payload = {
        "status": "PROTOCOL_SCHEMA_CORRECTED_BEFORE_PERFORMANCE",
        "protocol": (
            "阶段204A在性能评价前，根据DWD官方归档元数据更正"
            "阶段204中的产品字段记录：主湿度变量使用TU产品中的RF_TU，"
            "质量码QN_9；TF产品中的RF_STD和QN_8仅作为辅助归档保留，"
            "不进入主分析。站点、年份、train/valid/test划分、透明攻击、"
            "种子、FAR、主终点、聚类单位和多重校正均保持不变。本阶段"
            "不训练模型、不选择阈值、不计算性能结果。"
        ),
        "original_protocol_status": protocol["status"],
        "original_humidity_field": protocol["variables"]["TF_TU"],
        "corrected_primary_humidity": {
            "product": "TU",
            "field": "RF_TU",
            "description": "relative humidity",
            "unit": "%",
            "missing_code": 9990.0,
            "quality_field": "QN_9",
            "accepted_minimum": 3,
        },
        "auxiliary_tf_product": {
            "product": "TF",
            "field": "RF_STD",
            "description": "relative humidity from the moisture product",
            "unit": "%",
            "missing_code": 9990.0,
            "quality_field": "QN_8",
            "use_in_primary_analysis": False,
        },
        "frozen_items_unchanged": [
            "provider",
            "station_ids",
            "years",
            "train_valid_test_split",
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
        "tu_archives_ok": int(len(tu)),
        "tf_archives_retained_as_auxiliary": int(len(tf)),
        "performance_status": "NOT_EVALUATED",
        "limitations": [
            "本阶段是下载后、性能评价前的字段更正记录，不是性能结果。",
            "TF归档只保留为可追溯辅助证据，不能与TU中的RF_TU混用。",
            "所有DWD攻击仍须是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段204A DWD产品字段更正记录",
        "",
        f"- 状态：**{payload['status']}**",
        "- 原协议：阶段204在下载前冻结了来源、站点、年份、划分、攻击、FAR、终点和多重校正。",
        "- 更正项：主湿度变量由记录错误的 `TF_TU/TF` 更正为官方 `TU` 归档中的 `RF_TU`。",
        "- 质量码：`TU/RF_TU` 使用 `QN_9`，接受阈值仍为 `QN_9 >= 3`。",
        "- 辅助归档：`TF` 产品包含 `RF_STD/QN_8`，保留但不进入主分析。",
        "",
        "## 未改变项",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["frozen_items_unchanged"])
    lines += [
        "",
        "本阶段不训练模型、不选择阈值、不计算性能结果；test 仍只用于后续评价。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "corrected_field": "TU.RF_TU",
                "quality_field": "QN_9",
                "tf_archives_retained_as_auxiliary": payload[
                    "tf_archives_retained_as_auxiliary"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
