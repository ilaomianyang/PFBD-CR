from __future__ import annotations

"""Stage 257: freeze the fifth-provider FMI protocol before download.

This stage reads only official FMI station metadata.  It does not download,
inspect, or evaluate any observation values.  The station grid, years,
variables, missingness rule, transparent attack protocol, seeds, primary
FAR, primary endpoint, and promotion boundary are frozen here so that a
later FMI download and locked transfer cannot use test feedback to select
the protocol.
"""

import hashlib
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage257_fmi_protocol_preregistration"

STATION_QUERY = (
    "https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0"
    "&request=GetFeature&storedquery_id=fmi::ef::stations"
)
PROTOCOL_ID = "TMQC-FMI-EXTERNAL-20260919-v1"
STATION_COUNT = 12
GRID_DEGREES = 0.75
YEARS = [2021, 2022, 2023, 2024, 2025]
VALID_YEAR = 2024
TEST_YEAR = 2025
SEEDS = [3407, 6257, 7919, 104729, 20260715]
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
PRIMARY_FAR = 0.01
USER_AGENT = "TrustMeteo-QC protocol preregistration"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_station_metadata() -> tuple[bytes, list[dict[str, object]]]:
    request = urllib.request.Request(
        STATION_QUERY,
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    root = ET.fromstring(payload)
    namespace = {
        "wfs": "http://www.opengis.net/wfs/2.0",
        "ef": "http://inspire.ec.europa.eu/schemas/ef/4.0",
        "gml": "http://www.opengis.net/gml/3.2",
    }
    rows: list[dict[str, object]] = []
    for member in root.findall("wfs:member", namespace):
        facility = member.find(
            "ef:EnvironmentalMonitoringFacility",
            namespace,
        )
        if facility is None:
            continue
        identifier = facility.find(
            "gml:identifier",
            namespace,
        )
        name = facility.find(
            "gml:name",
            namespace,
        )
        point = facility.find(
            ".//gml:Point/gml:pos",
            namespace,
        )
        mobile = facility.find("ef:mobile", namespace)
        if identifier is None or name is None or point is None:
            continue
        station_id = (identifier.text or "").strip()
        station_name = (name.text or "").strip()
        coordinates = (point.text or "").split()
        if not station_id or not station_name or len(coordinates) != 2:
            continue
        latitude, longitude = (float(value) for value in coordinates)
        rows.append(
            {
                "station_id": station_id,
                "station_name": station_name,
                "latitude": latitude,
                "longitude": longitude,
                "mobile": (
                    (mobile.text or "").strip().lower() == "true"
                    if mobile is not None
                    else False
                ),
            }
        )
    if len(rows) < STATION_COUNT:
        raise RuntimeError("FMI returned fewer stations than expected")
    return payload, rows


def select_stations(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = frame[
        frame["latitude"].between(59.0, 70.5)
        & frame["longitude"].between(19.0, 32.0)
        & ~frame["mobile"]
    ].copy()
    frame["spatial_cell_lat"] = (
        (frame["latitude"] - 59.0) // GRID_DEGREES
    ).astype(int)
    frame["spatial_cell_lon"] = (
        (frame["longitude"] - 19.0) // GRID_DEGREES
    ).astype(int)
    frame = frame.sort_values(
        [
            "spatial_cell_lat",
            "spatial_cell_lon",
            "station_id",
        ],
        kind="stable",
    )
    frame = frame.drop_duplicates(
        subset=["spatial_cell_lat", "spatial_cell_lon"],
        keep="first",
    )
    if len(frame) < STATION_COUNT:
        raise RuntimeError("FMI spatial grid produced too few stations")
    selected = frame.head(STATION_COUNT).copy()
    selected["station_id"] = selected["station_id"].astype(str)
    return selected.sort_values("station_id").reset_index(drop=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raw_metadata, rows = fetch_station_metadata()
    selected = select_stations(rows)
    station_rows = selected[
        [
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "spatial_cell_lat",
            "spatial_cell_lon",
        ]
    ].copy()

    metadata_path = TABLES / f"{PREFIX}_station_metadata.xml"
    metadata_path.write_bytes(raw_metadata)
    station_path = TABLES / f"{PREFIX}_stations.csv"
    station_rows.to_csv(
        station_path,
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "status": "PROTOCOL_FROZEN_BEFORE_DOWNLOAD",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "阶段257在下载任何FMI观测值之前冻结第五个独立机构外部"
            "验证协议。只读取FMI官方INSPIRE站点元数据，不读取、不"
            "下载、不评价任何逐小时观测值。固定站点空间分箱、年份、"
            "变量、缺失规则、透明受控攻击、随机种子、FAR和主终点。"
            "test只用于后续锁定评价，不用于选择站点、特征、阈值、"
            "权重或攻击参数。攻击不是真实攻击日志。"
        ),
        "provider": "Finnish Meteorological Institute (FMI)",
        "dataset": "FMI Open Data hourly weather observations",
        "station_query": STATION_QUERY,
        "license": "Creative Commons Attribution 4.0",
        "independent_source": True,
        "source_independence": (
            "FMI is a different national provider and sensor network from "
            "DWD, MeteoSwiss, NOAA ISD, KNMI, ECCC, and the Liupanshan "
            "local observations."
        ),
        "station_metadata_sha256": sha256_bytes(raw_metadata),
        "station_count": int(len(station_rows)),
        "station_selection_rule": (
            "Retain non-mobile FMI stations inside latitude 59.0-70.5 "
            "and longitude 19.0-32.0. Form 0.75-degree cells from "
            "(latitude-59.0, longitude-19.0). Sort by cell latitude, "
            "cell longitude, and station_id; keep the first station per "
            "cell; take the first 12 station IDs."
        ),
        "station_ids": station_rows["station_id"].tolist(),
        "station_names": station_rows["station_name"].tolist(),
        "years": YEARS,
        "valid_year": VALID_YEAR,
        "test_year": TEST_YEAR,
        "split": {
            "train": [2021, 2022, 2023],
            "valid": [2024],
            "test": [2025],
        },
        "primary_variables": {
            "temperature_c": {
                "source_field": "TA_PT1H_AVG",
                "unit": "degC",
            },
            "humidity_pct": {
                "source_field": "RH_PT1H_AVG",
                "unit": "%",
            },
            "wind_speed_ms": {
                "source_field": "WS_PT1H_AVG",
                "unit": "m/s",
            },
            "pressure_hpa": {
                "source_field": "PA_PT1H_AVG",
                "unit": "hPa",
            },
        },
        "missing_rule": (
            "Missing WFS parameters or missing values are retained as "
            "missing; no sentinel substitution and no imputation."
        ),
        "coverage_rule": (
            "A station is retained only if the primary temperature "
            "coverage is at least 0.70 in every split year and the "
            "station has at least one non-missing value in each split."
        ),
        "attack_types": ATTACK_TYPES,
        "attack_length": ATTACK_LENGTH,
        "attack_blocks": 6,
        "seeds": SEEDS,
        "primary_far": PRIMARY_FAR,
        "primary_endpoint": "event_coverage",
        "secondary_endpoint": "row_recall",
        "cluster_unit": "station",
        "multiplicity": (
            "Benjamini-Hochberg over attack types within the primary FAR"
        ),
        "selection_boundary": (
            "Detector, features, fusion weights, protection rule, "
            "threshold, attack type, station, and year are frozen before "
            "the FMI test evaluation. Test labels are used only for "
            "evaluation. No target-domain test tuning is permitted."
        ),
        "attack_boundary": (
            "All attacks are transparent controlled injections generated "
            "by the frozen protocol. They are not real attack logs."
        ),
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "promotion_policy": {
            "current": "BOUNDED",
            "after_locked_transfer": "PARTIALLY_CLOSED",
            "resolved_requires": (
                "Independent production incident or fault logs with "
                "auditable labels and governance."
            ),
        },
        "forbidden_claims": [
            "FMI transfer proves provider-independent or station-independent performance.",
            "FMI controlled injections are real production attacks.",
            "FMI transfer can overwrite the frozen Stage246 endpoint.",
            "FMI test labels may be used to tune any model or threshold.",
        ],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol_path = TABLES / f"{PREFIX}.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段257 FMI第五机构预注册",
        "",
        f"- 状态：**{protocol['status']}**",
        f"- 协议ID：`{PROTOCOL_ID}`",
        f"- 机构：`{protocol['provider']}`",
        f"- 站点数：`{protocol['station_count']}`",
        f"- 年份：`{', '.join(str(value) for value in YEARS)}`",
        f"- 主FAR：`{PRIMARY_FAR}`",
        f"- 主终点：`{protocol['primary_endpoint']}`",
        "",
        "## 冻结边界",
        "",
        "- 本阶段只读取官方站点元数据，不下载任何观测值。",
        "- test只用于后续锁定评价，不用于任何调参。",
        "- 所有攻击均为透明受控注入，不是真实攻击日志。",
        "- FMI结果最多把机构依赖风险降为 `PARTIALLY_CLOSED`，不能升级为机构无关。",
        "",
        "## 站点",
        "",
        *[
            f"- `{row.station_id}` {row.station_name} "
            f"({row.latitude:.4f}, {row.longitude:.4f})"
            for row in station_rows.itertuples()
        ],
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "protocol_id": PROTOCOL_ID,
                "station_count": protocol["station_count"],
                "station_ids": protocol["station_ids"],
                "metadata_sha256": protocol["station_metadata_sha256"],
                "protocol_sha256": sha256_file(protocol_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
