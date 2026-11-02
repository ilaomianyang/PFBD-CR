from __future__ import annotations

"""Freeze new September 2026 temporal windows before observation access."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

START = "2026-09-01T00:00:00Z"
END = "2026-09-20T23:00:00Z"
SEEDS = [3407, 6257, 7919, 104729, 20260715]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenarios() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for intensity, magnitude in (("weak", 0.8), ("strong", 4.0)):
        for duration, active_length in (("short", 3), ("long", 12)):
            for length_name, length in (("short", 8), ("long", 24)):
                rows.append(
                    {
                        "name": f"offset_{intensity}_{duration}_{length_name}",
                        "base_type": "offset",
                        "magnitude": magnitude,
                        "length": length,
                        "active_length": active_length,
                        "intensity": intensity,
                        "duration": duration,
                        "event_length": length_name,
                    }
                )
    rows.append(
        {
            "name": "coordinated_drift_unseen",
            "base_type": "drift",
            "start": 0.0,
            "end": 4.0,
            "length": 24,
            "active_length": 12,
            "station_count": 2,
            "coordinated": True,
            "unseen_combination": True,
        }
    )
    return rows


def load_ids(path: Path, key: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(value) for value in payload[key]]


def build(
    *,
    protocol_id: str,
    provider: str,
    source_protocol: Path,
    source_audit: Path,
    station_ids: list[str],
    source_station_sha256: str,
) -> dict[str, object]:
    return {
        "status": "PROTOCOL_FROZEN_BEFORE_SEPTEMBER_2026_OBSERVATION_ACCESS",
        "protocol_id": protocol_id,
        "provider": provider,
        "scope": "new September 2026 temporal factor window",
        "station_ids": station_ids,
        "station_count": len(station_ids),
        "years": [2021, 2022, 2023, 2024, 2026],
        "split": {"train": [2021, 2022, 2023], "valid": [2024], "test": [2026]},
        "excluded_years": [2025],
        "test_period_start_utc": START,
        "test_period_end_utc": END,
        "test_period_is_partial_month": True,
        "primary_variable": "temperature_c",
        "coverage_rule": "At least 0.70 of expected UTC hourly slots in every split.",
        "attack_factor_protocol": {
            "attack_types": [
                "spike",
                "offset",
                "drift",
                "time_shift",
                "stuck",
                "gaussian_noise",
            ],
            "intensity": ["weak", "strong"],
            "duration": ["short", "long"],
            "event_length": ["short", "long"],
            "unseen_combination": "coordinated_drift_unseen",
            "scenarios": scenarios(),
            "seeds": SEEDS,
            "primary_far": 0.01,
            "primary_endpoint": "event_coverage",
        },
        "selection_boundary": (
            "Provider, station IDs, years, split, variable, factor grid, "
            "methods, thresholds, seeds, test period and statistics were "
            "frozen before September 2026 observation access. Test labels "
            "are evaluation-only."
        ),
        "provider_boundary": "Existing provider temporal holdout; not a new institution.",
        "attack_boundary": "All attacks are controlled simulations, not production incidents.",
        "exploratory": True,
        "observation_access": False,
        "performance_access": False,
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "source_protocol_sha256": sha256(source_protocol),
        "source_audit_sha256": sha256(source_audit),
        "source_station_sha256": source_station_sha256,
        "frozen_at_utc": "2026-09-24T00:00:00Z",
    }


def main() -> int:
    fmi_source = TABLES / "stage390_fmi_2026_factor_protocol.json"
    fmi_audit = TABLES / "stage392_fmi_2026_factor_download.json"
    met_source = TABLES / "stage384_met_norway_2026_factor_protocol.json"
    met_audit = TABLES / "stage386_met_norway_2026_factor_download.json"
    fmi_ids = load_ids(fmi_source, "station_ids")
    met_ids = load_ids(met_source, "station_ids")
    fmi_payload = build(
        protocol_id="TMQC-FMI-SEPTEMBER-2026-TEMPORAL-FACTOR-20260924-v1",
        provider="Finnish Meteorological Institute (FMI)",
        source_protocol=fmi_source,
        source_audit=fmi_audit,
        station_ids=fmi_ids,
        source_station_sha256=sha256(TABLES / "stage390_fmi_2026_factor_protocol.json"),
    )
    met_payload = build(
        protocol_id="TMQC-MET-NORWAY-SEPTEMBER-2026-TEMPORAL-FACTOR-20260924-v1",
        provider="Norwegian Meteorological Institute (MET Norway)",
        source_protocol=met_source,
        source_audit=met_audit,
        station_ids=met_ids,
        source_station_sha256=sha256(TABLES / "stage384_met_norway_2026_factor_protocol_stations.csv"),
    )
    outputs = [
        ("stage397_fmi_sep2026_factor_protocol", fmi_payload),
        ("stage397_met_norway_sep2026_factor_protocol", met_payload),
    ]
    for prefix, payload in outputs:
        output = TABLES / f"{prefix}.json"
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (REPORTS / f"{prefix}.md").write_text(
            "\n".join(
                [
                    f"# {prefix} 协议冻结",
                    "",
                    f"- 状态：`{payload['status']}`",
                    f"- provider：`{payload['provider']}`；站点：`{payload['station_count']}`。",
                    f"- 测试窗口：`{START}` 至 `{END}`；2025 明确排除。",
                    "- 协议冻结时未访问 2026 年 9 月观测值。",
                    "- 强度、持续时间、事件长度和 coordinated drift 未见组合均已冻结。",
                    "- 该窗口是既有 provider 的新时间窗，不是新增独立机构。",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "PROTOCOLS_FROZEN_BEFORE_SEPTEMBER_2026_OBSERVATION_ACCESS",
                "fmi_station_count": len(fmi_ids),
                "met_norway_station_count": len(met_ids),
                "test_start": START,
                "test_end": END,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
