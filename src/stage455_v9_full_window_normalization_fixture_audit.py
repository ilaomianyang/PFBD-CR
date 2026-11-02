from __future__ import annotations

"""Stage 455: full-window provider normalization fixture audit.

Four provider-shaped synthetic raw inputs are parsed and passed through the
same ``normalize_official_frame`` helper used by the production V9 template.
No official file, historical parquet, attack label, or performance table is
read.  The audit checks the complete 744-hour grid, missing-value retention,
window filtering, duplicate rejection, station-set rejection, and grid
alignment rejection.
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
ADAPTERS = SRC / "v9_official_adapters.py"
TEMPLATE = SRC / "stage425_v9_2026_official_window_download_template.py"
OUT_JSON = ROOT / "tables" / "stage455_v9_full_window_normalization_fixture_audit.json"
OUT_REPORT = ROOT / "reports" / "stage455_v9_full_window_normalization_fixture_audit.md"
FIXTURE_ROOT = ROOT / ".adapter_fixture_tmp" / "stage455_full_window"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage425_v9_compatibility import EXPECTED_GRID, EXPECTED_STATIONS  # noqa: E402
from stage425_v9_2026_official_window_download_template import (  # noqa: E402
    normalize_official_frame,
)
from v9_official_adapters import parse_provider  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_rows(station_id: str, provider_index: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for hour, timestamp in enumerate(EXPECTED_GRID):
        value = 10.0 + provider_index + 0.01 * hour
        rows.append(
            {
                "station_id": station_id,
                "timestamp": timestamp,
                "value": value,
            }
        )
    return rows


def write_fixtures() -> dict[str, list[Path]]:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    paths: dict[str, list[Path]] = {
        provider: [] for provider in EXPECTED_STATIONS
    }
    all_rows: dict[str, list[dict[str, object]]] = {
        provider: [] for provider in EXPECTED_STATIONS
    }
    for provider_index, provider in enumerate(
        ("MeteoSwiss", "NOAA ISD", "KNMI", "ECCC")
    ):
        for station_id in sorted(EXPECTED_STATIONS[provider]):
            all_rows[provider].extend(base_rows(station_id, provider_index))

    # One missing temperature per provider exercises observation_present.
    missing_keys = {
        "MeteoSwiss": ("abo", EXPECTED_GRID[0]),
        "NOAA ISD": ("52533199999", EXPECTED_GRID[1]),
        "KNMI": ("209", EXPECTED_GRID[2]),
        "ECCC": ("52", EXPECTED_GRID[3]),
    }
    for provider, (station_id, timestamp) in missing_keys.items():
        for row in all_rows[provider]:
            if row["station_id"] == station_id and row["timestamp"] == timestamp:
                row["value"] = np.nan
                break

    swiss = FIXTURE_ROOT / "meteoswiss_2026_10.csv"
    pd.DataFrame(
        {
            "station_abbr": [row["station_id"] for row in all_rows["MeteoSwiss"]],
            "reference_timestamp": [
                row["timestamp"].strftime("%d.%m.%Y %H:%M")
                for row in all_rows["MeteoSwiss"]
            ],
            "tre200h0": [row["value"] for row in all_rows["MeteoSwiss"]],
        }
    ).to_csv(swiss, sep=";", index=False, encoding="utf-8")
    paths["MeteoSwiss"].append(swiss)

    for row_station in sorted(EXPECTED_STATIONS["NOAA ISD"]):
        path = FIXTURE_ROOT / f"noaa_{row_station}.csv"
        rows = [
            row for row in all_rows["NOAA ISD"] if row["station_id"] == row_station
        ]
        pd.DataFrame(
            {
                "STATION": [row["station_id"] for row in rows],
                "LATITUDE": [45.0] * len(rows),
                "LONGITUDE": [90.0] * len(rows),
                "DATE": [
                    row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S")
                    for row in rows
                ],
                "TMP": [
                    "" if pd.isna(row["value"]) else f"{float(row['value']) * 10:.1f},1"
                    for row in rows
                ],
            }
        ).to_csv(path, index=False, encoding="utf-8")
        paths["NOAA ISD"].append(path)

    for row_station in sorted(EXPECTED_STATIONS["KNMI"]):
        path = FIXTURE_ROOT / f"knmi_{row_station}.zip"
        rows = [
            row for row in all_rows["KNMI"] if row["station_id"] == row_station
        ]
        lines = ["# STN,YYYYMMDD,HH,T"]
        for row in rows:
            timestamp = row["timestamp"]
            temp = "" if pd.isna(row["value"]) else f"{float(row['value']) * 10:.1f}"
            lines.append(
                f"{row['station_id']},{timestamp.strftime('%Y%m%d')},"
                f"{timestamp.hour + 1},{temp}"
            )
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"uurgeg_{row_station}.txt", "\n".join(lines) + "\n")
        paths["KNMI"].append(path)

    eccc = FIXTURE_ROOT / "eccc_2026_10.csv"
    pd.DataFrame(
        {
            "STN_ID": [row["station_id"] for row in all_rows["ECCC"]],
            "UTC_DATE": [
                row["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ")
                for row in all_rows["ECCC"]
            ],
            "TEMP": [row["value"] for row in all_rows["ECCC"]],
        }
    ).to_csv(eccc, index=False, encoding="utf-8")
    paths["ECCC"].append(eccc)
    return paths


def must_raise(callback) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def main() -> int:
    paths = write_fixtures()
    checks: list[dict[str, object]] = []
    normalized: dict[str, pd.DataFrame] = {}

    def add(name: str, passed: bool, value: object, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "value": value,
                "detail": detail,
            }
        )

    for provider in EXPECTED_STATIONS:
        frame, meta = parse_provider(provider, paths[provider])
        result, raw_rows, actual_stations = normalize_official_frame(provider, frame)
        normalized[provider] = result
        expected_rows = len(EXPECTED_STATIONS[provider]) * len(EXPECTED_GRID)
        complete = (
            len(result) == expected_rows
            and set(result["station_id"].astype(str)) == EXPECTED_STATIONS[provider]
            and set(pd.to_datetime(result["timestamp_utc"], utc=True)) == set(EXPECTED_GRID)
            and not result.duplicated(["station_id", "timestamp_utc"]).any()
            and set(result["split"].astype(str)) == {"test"}
            and list(result.columns) == [
                "timestamp_utc",
                "station_id",
                "temperature_c",
                "observation_present",
                "split",
            ]
        )
        missing_retained = bool((~result["observation_present"]).sum() == 1)
        add(
            f"{provider}_complete_744_hour_grid",
            complete,
            {
                "adapter": meta.get("parser_version"),
                "raw_rows": raw_rows,
                "normalized_rows": int(len(result)),
                "expected_rows": expected_rows,
            },
            "适配器输出必须统一到冻结站点集合和 744 小时 UTC 网格。",
        )
        add(
            f"{provider}_missing_value_preserved",
            missing_retained,
            {"missing_rows": int((~result["observation_present"]).sum())},
            "官方缺测只能保留为缺测，不能在规范化阶段被填补。",
        )

    add(
        "all_four_provider_normalizers_pass",
        all(row["status"] == "PASS" for row in checks),
        {"provider_count": 4},
        "四个 provider 的生产规范化核心必须共享同一完整网格契约。",
    )

    base = normalized["ECCC"].copy()
    outside = base.iloc[[0]].copy()
    outside["timestamp_utc"] = pd.Timestamp("2026-09-30T23:00:00Z")
    outside_result, _, _ = normalize_official_frame(
        "ECCC", pd.concat([base, outside], ignore_index=True)
    )
    add(
        "outside_window_rows_are_filtered",
        len(outside_result) == len(base)
        and outside_result["timestamp_utc"].min() >= pd.Timestamp(
            "2026-10-01T00:00:00Z"
        ),
        {"rows_after_filter": int(len(outside_result))},
        "窗口外数据只能被过滤，不能进入 V9 test 网格。",
    )

    duplicate = base.iloc[[0]].copy()
    add(
        "duplicate_station_timestamp_is_rejected",
        must_raise(
            lambda: normalize_official_frame(
                "ECCC", pd.concat([base, duplicate], ignore_index=True)
            )
        ),
        {"fixture": "duplicate first station/timestamp"},
        "同一站点和时间戳重复时必须硬失败。",
    )

    missing_station = base[
        ~base["station_id"].astype(str).eq(sorted(EXPECTED_STATIONS["ECCC"])[0])
    ].copy()
    add(
        "missing_station_is_rejected",
        must_raise(lambda: normalize_official_frame("ECCC", missing_station)),
        {"removed_station": sorted(EXPECTED_STATIONS["ECCC"])[0]},
        "冻结站点集合不完整时必须硬失败。",
    )

    misaligned = base.iloc[[0]].copy()
    misaligned["timestamp_utc"] = pd.Timestamp("2026-10-01T00:30:00Z")
    misaligned = pd.concat([base.iloc[1:], misaligned], ignore_index=True)
    add(
        "misaligned_timestamp_is_rejected",
        must_raise(lambda: normalize_official_frame("ECCC", misaligned)),
        {"timestamp": "2026-10-01T00:30:00Z"},
        "非整点 UTC 时间戳不能进入冻结 hourly grid。",
    )

    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_FULL_WINDOW_NORMALIZATION_FIXTURE_AUDIT"
        if not failures
        else "FAIL_V9_FULL_WINDOW_NORMALIZATION_FIXTURE_AUDIT"
    )
    payload = {
        "stage": 455,
        "status": status,
        "protocol_id": json.loads(PROTOCOL.read_text(encoding="utf-8"))[
            "protocol_id"
        ],
        "checks": checks,
        "failed_checks": failures,
        "synthetic_fixture_only": True,
        "official_observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "provider_count": 4,
        "expected_grid_rows_per_station": len(EXPECTED_GRID),
        "claim_boundary": [
            "夹具只验证四 provider 解析和 V9 规范化契约，不是官方观测数据。",
            "通过不代表 V9 性能结果，也不代表模拟攻击全面领先。",
        ],
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "adapters": sha256(ADAPTERS),
            "normalization_template": sha256(TEMPLATE),
        },
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage455 V9 完整窗口规范化夹具审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 四个 provider 均使用 744 小时、冻结站点集合的完整合成原始文件。",
                "- 验证缺测保留、窗口外过滤、重复时间戳拒绝、缺站拒绝和非整点拒绝。",
                "- 未读取官方观测值、旧 test、攻击标签或性能表。",
                "",
                "## 失败检查",
                "",
                *([f"- `{name}`" for name in failures] or ["- 无。"]),
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": 455,
                "status": status,
                "failed_checks": failures,
                "synthetic_fixture_only": True,
                "performance_evaluated": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
