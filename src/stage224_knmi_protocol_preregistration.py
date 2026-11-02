from __future__ import annotations

"""Stage 224: pre-register the third cross-provider external validation.

The DWD evidence chain and the MeteoSwiss transfer are already frozen.
This stage freezes a third independent national surface-station network,
KNMI, before any KNMI hourly observation values are inspected for model
selection or performance.  Only the official KNMI hourly-data catalogue
and the official CDN file naming convention are used to define the
download contract.

No detector is trained, no threshold is selected, no performance is
computed, and no target-domain test labels are read.  The result is a
download-before-performance protocol that later stages must follow.
"""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage224_knmi_protocol_preregistration"

PROVIDER = "Royal Netherlands Meteorological Institute (KNMI)"
DATASET = "KNMI hourly meteorological observations"
CATALOGUE_URL = (
    "https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens"
)
CDN_BASE = (
    "https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/"
    "uurgegevens"
)
STATION_METADATA_URL = (
    "https://dataplatform.knmi.nl/dataset/preview/"
    "waarneemstations-csv-1-0"
)
LICENSE = "Creative Commons Attribution 4.0 (CC BY 4.0)"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
USER_AGENT = "TrustMeteo-QC/0.1 (reproducible research; KNMI OpenData)"

STATION_COUNT = 12
STATION_GRID_DEGREES = 0.75
GRID_ORIGIN_LATITUDE = 50.5
GRID_ORIGIN_LONGITUDE = 3.0

YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_YEARS = [2021, 2022, 2023]
VALID_YEAR = 2024
TEST_YEAR = 2025

PRIMARY_VARIABLES = {
    "temperature_c": {
        "source_field": "T",
        "description": "air temperature at 1.5 m",
        "unit": "degC",
        "scale": 0.1,
    },
    "humidity_pct": {
        "source_field": "U",
        "description": "relative humidity at 1.5 m",
        "unit": "%",
        "scale": 1.0,
    },
}
AUXILIARY_VARIABLES = {
    "dew_point_c": {
        "source_field": "TD",
        "unit": "degC",
        "scale": 0.1,
    },
    "pressure_hpa": {
        "source_field": "P",
        "unit": "hPa",
        "scale": 0.1,
    },
}
MISSING_RULE = (
    "Empty KNMI fields are converted to missing values; no sentinel "
    "substitution and no imputation are allowed."
)

ATTACK_TYPES = [
    "spike",
    "offset",
    "drift",
    "time_shift",
    "stuck",
    "gaussian_noise",
]
ATTACK_LENGTH = {
    "spike": 8,
    "offset": 8,
    "drift": 12,
    "time_shift": 8,
    "stuck": 8,
    "gaussian_noise": 8,
}
ATTACK_BLOCKS = 6
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FAR_GRID = [0.005, 0.01, 0.02, 0.05]
PRIMARY_FAR = 0.01
PRIMARY_ENDPOINT = "event_coverage"
SECONDARY_ENDPOINT = "row_recall"
CLUSTER_UNIT = "station"
MULTIPLICITY = "Benjamini-Hochberg over attack types within the primary FAR"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8", errors="replace")


def build_protocol() -> dict[str, object]:
    catalogue = fetch_text(CATALOGUE_URL)
    if "uurgeg_" not in catalogue or "cdn.knmi.nl" not in catalogue:
        raise RuntimeError("KNMI hourly catalogue contract changed")
    ckan_url = (
        "https://dataplatform.knmi.nl/api/3/action/"
        "package_show?id=waarneemstations-csv-1-0"
    )
    ckan = json.loads(fetch_text(ckan_url))
    result = ckan.get("result", {})
    if (
        ckan.get("success") is not True
        or result.get("license_title")
        != "Creative Commons Attribution 4.0"
        or result.get("license_url") != LICENSE_URL
    ):
        raise RuntimeError("KNMI station metadata license changed")
    return {
        "status": "PROTOCOL_FROZEN_BEFORE_KNMI_PERFORMANCE",
        "protocol": (
            "阶段224在任何KNMI小时观测值用于模型选择或性能评价之前冻结"
            "第三跨机构现场站点协议。来源为KNMI官方逐小时气象观测目录"
            "和官方CDN；年份固定为2021-2025，2021-2023为train、"
            "2024为valid、2025为test。站点先从KNMI小时文件覆盖的STN"
            "集合中按0.75度经纬网格每格取站号最小者，再按站号取前12个，"
            "避免按任何性能结果选站。主变量为T和U，辅助变量为TD和P；"
            "空字段为缺失，不做插补。主FAR=0.01，主终点为event_coverage，"
            "次终点为row_recall，聚类单位为station。攻击类型、块数、"
            "长度和种子与DWD/MeteoSwiss协议一致。test只评价，不根据test"
            "选择特征、窗口、权重、阈值、站点或攻击位置。所有攻击都是"
            "透明受控注入，不是真实攻击日志。"
        ),
        "provider": PROVIDER,
        "dataset": DATASET,
        "catalogue_url": CATALOGUE_URL,
        "cdn_base": CDN_BASE,
        "station_metadata_url": STATION_METADATA_URL,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "station_count": STATION_COUNT,
        "station_grid_degrees": STATION_GRID_DEGREES,
        "grid_origin_latitude": GRID_ORIGIN_LATITUDE,
        "grid_origin_longitude": GRID_ORIGIN_LONGITUDE,
        "years": YEARS,
        "split": {
            "train": TRAIN_YEARS,
            "valid": [VALID_YEAR],
            "test": [TEST_YEAR],
        },
        "primary_variables": PRIMARY_VARIABLES,
        "auxiliary_variables": AUXILIARY_VARIABLES,
        "missing_rule": MISSING_RULE,
        "timezone": "UTC",
        "timezone_evidence": (
            "KNMI hourly files define HH as UT (universal time); the "
            "hourly division convention is documented in the official "
            "hourly-data file header."
        ),
        "attack_types": ATTACK_TYPES,
        "attack_length": ATTACK_LENGTH,
        "attack_blocks": ATTACK_BLOCKS,
        "seeds": SEEDS,
        "far_grid": FAR_GRID,
        "primary_far": PRIMARY_FAR,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoint": SECONDARY_ENDPOINT,
        "cluster_unit": CLUSTER_UNIT,
        "multiplicity": MULTIPLICITY,
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "catalogue_sha256": hashlib.sha256(
            catalogue.encode("utf-8")
        ).hexdigest(),
        "station_metadata_status": (
            "OFFICIAL_CKAN_METADATA_VERIFIED"
        ),
        "station_metadata_note": (
            "The official KNMI Data Platform CKAN record confirms CC BY "
            "4.0. The station-coordinate file itself may require the "
            "platform download flow; station selection therefore uses "
            "only the official hourly-file STN set and the pre-declared "
            "grid rule, with no credential bypass."
        ),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "这是下载前预注册，不是性能结果。",
            "KNMI站点选择只依据官方站点编号和预声明网格，不依据test性能。",
            "KNMI小时序列官方说明可能因站址迁移和观测方式变化而不均质，不能用于趋势分析。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
        ],
    }


def main() -> int:
    payload = build_protocol()
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "provider": PROVIDER,
                "dataset": DATASET,
                "catalogue_url": CATALOGUE_URL,
                "license": LICENSE,
                "station_count": STATION_COUNT,
                "years": ",".join(str(value) for value in YEARS),
                "primary_far": PRIMARY_FAR,
                "primary_endpoint": PRIMARY_ENDPOINT,
            }
        ]
    ).to_csv(
        TABLES / f"{PREFIX}_contract.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段224 KNMI第三独立来源预注册",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 来源：`{PROVIDER}`",
        f"- 数据集：`{DATASET}`",
        f"- 年份：`{YEARS}`；切分：`2021-2023/2024/2025`",
        f"- 主FAR：`{PRIMARY_FAR}`；主终点：`{PRIMARY_ENDPOINT}`；聚类：`{CLUSTER_UNIT}`",
        f"- 站点：`{STATION_COUNT}`；下载状态：`{payload['download_status']}`；性能状态：`{payload['performance_status']}`",
        "",
        "## 冻结边界",
        "",
        "- 站点只按官方STN和预声明经纬网格选择，不按任何test性能选择。",
        "- 主变量T/U，辅助变量TD/P；空字段为缺失，不插补。",
        "- test只评价，不选择特征、窗口、权重、阈值、站点或攻击位置。",
        "- 所有攻击都是透明受控注入，不是真实攻击日志。",
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
                "provider": payload["provider"],
                "station_count": payload["station_count"],
                "download_status": payload["download_status"],
                "performance_status": payload["performance_status"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
