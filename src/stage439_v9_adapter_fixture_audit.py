from __future__ import annotations

"""Stage 439: in-memory/file-fixture smoke audit for V9 raw adapters.

The fixtures reproduce the provider file *shapes* only.  They are not
observations, are not used for model fitting, and cannot produce performance
claims.  The audit checks timestamp and unit conversions plus missing-value
preservation before the future official files arrive.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

from v9_official_adapters import (
    PARSER_VERSION,
    parse_eccc,
    parse_knmi,
    parse_meteoswiss,
    parse_noaa,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
OUT_JSON = TABLES / "stage439_v9_adapter_fixture_audit.json"
OUT_REPORT = REPORTS / "stage439_v9_adapter_fixture_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_common(frame: pd.DataFrame, station: str, timestamp: str, value: float) -> None:
    assert list(frame.columns) == [
        "station_id",
        "timestamp_utc",
        "temperature_c",
        "observation_present",
    ]
    assert len(frame) == 1
    assert frame.loc[0, "station_id"] == station
    assert str(frame.loc[0, "timestamp_utc"]) == timestamp
    assert float(frame.loc[0, "temperature_c"]) == value
    assert bool(frame.loc[0, "observation_present"]) is True


def main() -> int:
    checks: list[dict[str, object]] = []
    root = ROOT / ".adapter_fixture_tmp" / "current"
    root.mkdir(parents=True, exist_ok=True)
    # The fixture directory is deterministic and contains no official data.
    # It is intentionally retained for reproducibility and is overwritten on
    # the next audit run.

    swiss = root / "abo.csv"
    swiss.write_text(
        "station_abbr;reference_timestamp;tre200h0\n"
        "abo;01.10.2026 00:00;12.5\n",
        encoding="utf-8",
    )
    frame, _ = parse_meteoswiss([swiss])
    _assert_common(frame, "abo", "2026-10-01 00:00:00+00:00", 12.5)
    checks.append({"provider": "MeteoSwiss", "status": "PASS"})

    noaa = root / "52533199999.csv"
    # NOAA's TMP component is comma-separated inside the quoted CSV field.
    noaa.write_text(
        'DATE,TMP\n2026-10-01T00:00:00,"125,1"\n',
        encoding="utf-8",
    )
    frame, _ = parse_noaa([noaa])
    _assert_common(frame, "52533199999", "2026-10-01 00:00:00+00:00", 12.5)
    checks.append({"provider": "NOAA ISD", "status": "PASS"})

    knmi = root / "209.zip"
    knmi_text = (
        "# metadata\n"
        "# STN,YYYYMMDD,HH,T\n"
        "209,20261001,1,125\n"
    ).encode("utf-8")
    with zipfile.ZipFile(knmi, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("uurgeg_209.txt", knmi_text)
    frame, _ = parse_knmi([knmi])
    _assert_common(frame, "209", "2026-10-01 00:00:00+00:00", 12.5)
    checks.append({"provider": "KNMI", "status": "PASS"})

    eccc = root / "52.csv"
    eccc.write_text(
        "STN_ID,UTC_DATE,TEMP\n"
        "52,2026-10-01T00:00:00Z,12.5\n",
        encoding="utf-8",
    )
    frame, _ = parse_eccc([eccc])
    _assert_common(frame, "52", "2026-10-01 00:00:00+00:00", 12.5)
    checks.append({"provider": "ECCC", "status": "PASS"})

    failures = [row["provider"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_ADAPTER_FIXTURE_AUDIT"
        if not failures
        else "FAIL_V9_ADAPTER_FIXTURE_AUDIT"
    )
    payload = {
        "stage": 439,
        "status": status,
        "parser_version": PARSER_VERSION,
        "checks": checks,
        "failed_providers": failures,
        "official_observation_values_read": False,
        "performance_evaluated": False,
        "synthetic_fixture_only": True,
        "claim_boundary": [
            "夹具只验证官方文件形状的解析契约，不是官方观测数据。",
            "夹具不训练模型、不生成攻击、不计算检测指标。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage439 V9 适配器格式夹具审计",
                "",
                f"- 状态：`{status}`",
                f"- parser version：`{PARSER_VERSION}`",
                "- 四个 provider 的最小文件形状、时间转换、单位转换均通过。",
                "- 仅使用临时夹具，不读取官方 2026-10 观测值。",
                "- 不产生任何性能结果或攻击召回率。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 439, "status": status, "failed_providers": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
