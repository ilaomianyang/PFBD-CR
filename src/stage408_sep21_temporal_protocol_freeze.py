from __future__ import annotations

"""Freeze a new, previously unaccessed 2026-09-21--23 temporal holdout."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

START = "2026-09-21T00:00:00Z"
END = "2026-09-23T23:00:00Z"
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


def load_ids(path: Path) -> list[str]:
    return [str(v) for v in json.loads(path.read_text(encoding="utf-8"))["station_ids"]]


def build(
    *,
    protocol_id: str,
    provider: str,
    station_ids: list[str],
    source_protocol: Path,
    source_station: Path,
    source_audit: Path,
) -> dict[str, object]:
    return {
        "status": "PROTOCOL_FROZEN_BEFORE_SEPTEMBER_21_2026_OBSERVATION_ACCESS",
        "protocol_id": protocol_id,
        "provider": provider,
        "scope": "new 2026-09-21--23 temporal factor window",
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
            "Provider, stations, years, split, variable, factor grid, "
            "methods, thresholds, seeds, test period and statistics were "
            "frozen before 2026-09-21 observation access. Test labels are "
            "evaluation-only."
        ),
        "provider_boundary": "Existing provider temporal holdout; not a new institution.",
        "attack_boundary": "All attacks are controlled simulations, not production incidents.",
        "exploratory": True,
        "observation_access": False,
        "performance_access": False,
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "source_protocol_sha256": sha256(source_protocol),
        "source_station_sha256": sha256(source_station),
        "source_audit_sha256": sha256(source_audit),
        "frozen_at_utc": "2026-09-24T00:00:00Z",
    }


def main() -> int:
    jobs = [
        (
            "stage408_fmi_sep21_factor_protocol",
            build(
                protocol_id="TMQC-FMI-SEP21-23-2026-TEMPORAL-FACTOR-20260924-v1",
                provider="Finnish Meteorological Institute (FMI)",
                station_ids=load_ids(TABLES / "stage397_fmi_sep2026_factor_protocol.json"),
                source_protocol=TABLES / "stage397_fmi_sep2026_factor_protocol.json",
                source_station=TABLES / "stage390_fmi_2026_factor_protocol.json",
                source_audit=TABLES / "stage401_fmi_sep2026_factor_download.json",
            ),
        ),
        (
            "stage408_met_norway_sep21_factor_protocol",
            build(
                protocol_id="TMQC-MET-NORWAY-SEP21-23-2026-TEMPORAL-FACTOR-20260924-v1",
                provider="Norwegian Meteorological Institute (MET Norway)",
                station_ids=load_ids(TABLES / "stage397_met_norway_sep2026_factor_protocol.json"),
                source_protocol=TABLES / "stage397_met_norway_sep2026_factor_protocol.json",
                source_station=TABLES / "stage384_met_norway_2026_factor_protocol_stations.csv",
                source_audit=TABLES / "stage403_met_norway_sep2026_factor_download.json",
            ),
        ),
    ]
    for prefix, payload in jobs:
        output = TABLES / f"{prefix}.json"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (REPORTS / f"{prefix}.md").write_text(
            "\n".join(
                [
                    f"# {prefix} 协议冻结",
                    "",
                    f"- 状态：`{payload['status']}`",
                    f"- provider：`{payload['provider']}`；站点：`{payload['station_count']}`。",
                    f"- 测试窗口：`{START}` 至 `{END}`。",
                    "- 冻结发生在本时间窗观测访问前；该窗口属于既有 provider 的探索性时间稳健性检查。",
                    "- 强度、持续时间、事件长度和 coordinated drift 未见组合均已冻结。",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "PROTOCOLS_FROZEN_BEFORE_SEP21_2026_OBSERVATION_ACCESS"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
