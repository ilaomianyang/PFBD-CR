from __future__ import annotations

"""Stage 203: independent external evidence feasibility audit.

Stage 199--202 closed the Liupanshan cross-station transfer question
with a retained negative result.  The remaining evidence gap is not
another internal split: it is whether the frozen detector can be tested
on data that is independent at the source/sensor-network level, with a
transparent attack protocol and no target-domain tuning.

This stage is an audit, not a new performance experiment.  It:

* inventories the external sources already present in the project;
* labels each source by independence level and attack-label status;
* records the minimum requirements for a genuinely independent test;
* records candidate official data sources that are not yet validated;
* emits a pre-declared protocol that must be frozen before any download
  or target-domain evaluation.

No model, feature, threshold, station, year, attack type, or rule is
selected here.  No new data are downloaded.  The result is an evidence
gap statement plus a reproducible next-step contract.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage203_independent_external_evidence_feasibility"

NOAA_EXTENDED = TABLES / "stage82_noaa_extended_normalized.parquet"
NOAA_EXTENDED_JSON = TABLES / "stage82_noaa_extended_ingest.json"
NOAA_2022 = TABLES / "stage06_noaa_isd_normalized.parquet"
ERA5 = TABLES / "stage06_era5_station_reference.parquet"
LOCAL_FEATURE = Path(
    r"F:\数据集\Analysis\parquet\ml_feature_table.parquet"
)
EXTERNAL_REGISTRY = REPORTS / "external_data_registry.md"

INDEPENDENCE_LEVELS = {
    "same_source_internal": (
        "同一原始来源的内部年份或站点划分；只能支持内部泛化，"
        "不能作为跨来源独立验证。"
    ),
    "same_source_cross_network": (
        "同一机构或同一数据产品族的不同网络/传感器；"
        "独立性强于内部划分，但仍需说明共享处理链和共享偏差。"
    ),
    "different_source_auxiliary": (
        "不同来源的辅助背景场；不能替代现场传感器真值，"
        "也不能单独提供攻击标签。"
    ),
    "different_source_sensor_network": (
        "不同机构、不同传感器网络和不同处理链的现场观测；"
        "才能作为真正意义上的跨来源外部验证主体。"
    ),
}

MINIMUM_REQUIREMENTS = [
    {
        "requirement": "source_independence",
        "description": (
            "验证数据必须来自与六盘山主实验和NOAA ISD扩展实验不同的"
            "机构或传感器网络；共享同一原始文件的再分析产品不算独立。"
        ),
    },
    {
        "requirement": "transparent_attack_protocol",
        "description": (
            "攻击事件必须按预先冻结的透明注入规则生成，保留事件ID、"
            "攻击类型、强度、位置和种子；不得把自然异常或质量标记"
            "冒充攻击真值。"
        ),
    },
    {
        "requirement": "no_target_domain_tuning",
        "description": (
            "冻结模型、特征、保护配置和阈值选择协议；目标域只允许"
            "用无标签校准或预声明的固定FAR阈值，不得用目标test标签"
            "选择任何组件。"
        ),
    },
    {
        "requirement": "operating_point_and_uncertainty",
        "description": (
            "预先声明主FAR运行点、主终点和聚类单位；报告站点/事件级"
            "不确定性、多重比较校正和所有负结果。"
        ),
    },
    {
        "requirement": "reproducible_access",
        "description": (
            "记录官方URL、访问日期、文件校验值、字段定义、时间分辨率、"
            "缺失规则和质量标记，使第三方可以复现数据接入。"
        ),
    },
]

CANDIDATE_SOURCES = [
    {
        "candidate": "MeteoSwiss / SwissMetNet",
        "provider": "MeteoSwiss",
        "source_family": "national surface station network",
        "official_access": "https://www.meteoswiss.admin.ch/services-and-publications/service/open-data.html",
        "independence": "different_source_sensor_network",
        "status": "NOT_DOWNLOADED_NOT_VALIDATED",
        "notes": (
            "官方开放数据，独立于NOAA和六盘山；需确认站点覆盖、"
            "许可、时间分辨率、质量标记和是否可构造透明攻击协议。"
        ),
    },
    {
        "candidate": "KNMI Data Platform",
        "provider": "Royal Netherlands Meteorological Institute",
        "source_family": "national surface station network",
        "official_access": "https://dataplatform.knmi.nl/",
        "independence": "different_source_sensor_network",
        "status": "NOT_DOWNLOADED_NOT_VALIDATED",
        "notes": (
            "官方开放数据，独立于NOAA和六盘山；需确认API许可、"
            "站点元数据、缺失编码和可复现下载路径。"
        ),
    },
    {
        "candidate": "DWD Climate Data Center",
        "provider": "Deutscher Wetterdienst",
        "source_family": "national surface station network",
        "official_access": "https://www.dwd.de/EN/climate_environment/cdc/cdc_node.html",
        "independence": "different_source_sensor_network",
        "status": "NOT_DOWNLOADED_NOT_VALIDATED",
        "notes": (
            "官方开放数据，独立于NOAA和六盘山；需确认站点版本、"
            "质量等级、缺失编码和下载许可。"
        ),
    },
    {
        "candidate": "ECMWF ERA5",
        "provider": "Copernicus Climate Data Store",
        "source_family": "reanalysis grid",
        "official_access": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        "independence": "different_source_auxiliary",
        "status": "DOWNLOADED_AUXILIARY_ONLY",
        "notes": (
            "已作为区域背景场使用；再分析产品不是独立现场传感器，"
            "不能作为跨来源外部验证主体。"
        ),
    },
]


def safe_count(path: Path, column: str | None = None) -> int:
    if not path.is_file():
        return 0
    frame = pd.read_parquet(path, columns=[column] if column else None)
    if column is None:
        return int(len(frame))
    return int(frame[column].nunique())


def build_inventory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if LOCAL_FEATURE.is_file():
        local = pd.read_parquet(
            LOCAL_FEATURE,
            columns=["station_key", "timestamp_utc"],
        )
        rows.append(
            {
                "source_id": "liupanshan_local",
                "provider": "local field observation",
                "product": "Liupanshan multi-station observations",
                "independence_level": "same_source_internal",
                "rows": int(len(local)),
                "stations": int(local["station_key"].nunique()),
                "attack_labels": "transparent_controlled_injection",
                "usable_as_independent_external_validation": False,
                "reason": (
                    "主实验数据源；只能用于内部站点/时间留出，"
                    "不能验证跨来源独立性。"
                ),
            }
        )
    else:
        rows.append(
            {
                "source_id": "liupanshan_local",
                "provider": "local field observation",
                "product": "Liupanshan multi-station observations",
                "independence_level": "same_source_internal",
                "rows": 0,
                "stations": 0,
                "attack_labels": "unavailable",
                "usable_as_independent_external_validation": False,
                "reason": "feature table not found",
            }
        )

    rows.append(
        {
            "source_id": "noaa_isd_extended",
            "provider": "NOAA/NCEI",
            "product": "Integrated Surface Database Global Hourly",
            "independence_level": "same_source_internal",
            "rows": safe_count(NOAA_EXTENDED),
            "stations": safe_count(NOAA_EXTENDED, "station_id"),
            "attack_labels": "transparent_controlled_injection",
            "usable_as_independent_external_validation": False,
            "reason": (
                "当前最强外部协议仍来自NOAA ISD；2020-2025年份和"
                "11/12站点是同一来源内部留出，不是跨机构独立验证。"
            ),
        }
    )
    rows.append(
        {
            "source_id": "noaa_isd_2022_baseline",
            "provider": "NOAA/NCEI",
            "product": "Integrated Surface Database Global Hourly",
            "independence_level": "same_source_internal",
            "rows": safe_count(NOAA_2022),
            "stations": safe_count(NOAA_2022, "station_id"),
            "attack_labels": "none_natural_observation_only",
            "usable_as_independent_external_validation": False,
            "reason": (
                "2022无攻击自然观测基线；可用于报警率与漂移审计，"
                "不能计算攻击召回率。"
            ),
        }
    )
    rows.append(
        {
            "source_id": "era5_liupanshan",
            "provider": "Copernicus/ECMWF",
            "product": "ERA5 reanalysis",
            "independence_level": "different_source_auxiliary",
            "rows": safe_count(ERA5),
            "stations": safe_count(ERA5, "station_key"),
            "attack_labels": "none_auxiliary_background",
            "usable_as_independent_external_validation": False,
            "reason": (
                "不同来源的再分析背景场，不是独立现场传感器；"
                "只能辅助解释区域天气，不能替代现场外部验证。"
            ),
        }
    )
    return pd.DataFrame(rows)


def write_outputs(inventory: pd.DataFrame) -> int:
    inventory.to_csv(
        TABLES / f"{PREFIX}_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    requirements = pd.DataFrame(MINIMUM_REQUIREMENTS)
    requirements.to_csv(
        TABLES / f"{PREFIX}_requirements.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidates = pd.DataFrame(CANDIDATE_SOURCES)
    candidates.to_csv(
        TABLES / f"{PREFIX}_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    available = int(
        inventory[
            "usable_as_independent_external_validation"
        ].sum()
    )
    status = (
        "EVIDENCE_GAP_RETAINED"
        if available == 0
        else "INDEPENDENT_EXTERNAL_CANDIDATE_AVAILABLE"
    )
    payload = {
        "status": status,
        "protocol": (
            "阶段203只审计独立外部证据可行性，不训练模型、不选择阈值、"
            "不下载新数据、不计算新的性能结果，也不读取目标域test标签"
            "或根据test选择任何组件。现有来源按来源独立性、"
            "攻击标签状态和是否可作为跨来源外部验证主体分类。"
            "只有不同机构、不同传感器网络和不同处理链的现场观测，"
            "配合预先冻结的透明攻击协议和无目标域调参，才算真正独立"
            "外部验证。NOAA ISD内部年份/站点留出、六盘山内部站点留出"
            "和ERA5辅助背景均不满足该定义。"
        ),
        "source_registry": str(EXTERNAL_REGISTRY),
        "inventory_rows": int(len(inventory)),
        "requirement_rows": int(len(requirements)),
        "candidate_rows": int(len(candidates)),
        "usable_independent_external_sources": available,
        "independent_external_validation_completed": False,
        "independence_levels": INDEPENDENCE_LEVELS,
        "minimum_requirements": MINIMUM_REQUIREMENTS,
        "candidate_sources": CANDIDATE_SOURCES,
        "inventory": inventory.to_dict(orient="records"),
        "next_actions": [
            "在下载任何新数据前冻结候选来源、年份、站点、变量、"
            "攻击协议、主FAR、主终点和统计校正方案。",
            "优先验证一个不同机构的现场站点网络；ERA5只作为背景，"
            "不作为独立真值。",
            "下载后先做许可、字段、缺失编码、质量标记、时间覆盖和"
            "站点元数据审计，再进入模型评价。",
            "如果新来源无法提供可复现访问或透明攻击协议，保留缺口"
            "并明确写入论文限制，不能用同一来源内部划分替代。",
        ],
        "limitations": [
            "阶段203是可行性审计，不产生新的模型性能声明。",
            "候选来源尚未下载和验证，不能写成已完成外部验证。",
            "NOAA ISD内部年份/站点留出和ERA5辅助背景不满足"
            "跨机构、跨传感器网络的独立验证定义。",
            "攻击仍须是透明受控注入，除非来源本身提供可审计的真实"
            "故障或攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段203 独立外部证据可行性审计",
        "",
        f"状态：`{status}`",
        "",
        "本阶段不下载、不训练、不调参，只判断现有来源能否满足"
        "跨来源独立验证。",
        "",
        "## 现有来源",
        "",
        "| 来源 | 机构/产品 | 独立性等级 | 行数 | 站点 | 攻击标签 | 可作为独立外部验证 |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in inventory.to_dict(orient="records"):
        lines.append(
            f"| {row['source_id']} | {row['provider']} / "
            f"{row['product']} | {row['independence_level']} | "
            f"{int(row['rows'])} | {int(row['stations'])} | "
            f"{row['attack_labels']} | "
            f"{'是' if row['usable_as_independent_external_validation'] else '否'} |"
        )
    lines += [
        "",
        "## 最低要求",
        "",
    ]
    for item in MINIMUM_REQUIREMENTS:
        lines.append(
            f"- `{item['requirement']}`：{item['description']}"
        )
    lines += [
        "",
        "## 候选官方来源",
        "",
        "| 候选 | 机构 | 官方入口 | 独立性等级 | 当前状态 |",
        "|---|---|---|---|---|",
    ]
    for row in candidates.to_dict(orient="records"):
        lines.append(
            f"| {row['candidate']} | {row['provider']} | "
            f"{row['official_access']} | {row['independence']} | "
            f"{row['status']} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "当前仓库内没有满足跨机构、跨传感器网络定义的独立外部验证"
        "数据。NOAA ISD 2020-2025 的 12 站、六盘山 7 站和 ERA5"
        "背景都不能替代真正的跨来源现场验证。因此，独立外部验证"
        "必须明确标记为尚未完成，不能由内部留出结果替代。",
        "",
        "下一步若继续增强证据，应先在下载前冻结候选来源、站点、年份、"
        "变量、攻击协议、主FAR、主终点和统计校正方案；下载后先做"
        "许可、字段、缺失编码、质量标记、时间覆盖和站点元数据审计。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "inventory_rows": int(len(inventory)),
                "requirement_rows": int(len(requirements)),
                "candidate_rows": int(len(candidates)),
                "usable_independent_external_sources": available,
                "independent_external_validation_completed": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize_existing:
        inventory = pd.read_csv(TABLES / f"{PREFIX}_inventory.csv")
        return write_outputs(inventory)
    return write_outputs(build_inventory())


if __name__ == "__main__":
    raise SystemExit(main())
