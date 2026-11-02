from __future__ import annotations

"""Freeze an exploratory FMI 2026 temporal factor protocol before access."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE_PROTOCOL = TABLES / "stage257_fmi_protocol_preregistration.json"
SOURCE_AUDIT = TABLES / "stage258_fmi_official_download.json"
SOURCE_NORMALIZED = TABLES / "stage258_fmi_official_download_normalized.parquet"
OUT_PREFIX = "stage390_fmi_2026_factor_protocol"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenario_grid() -> list[dict[str, object]]:
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


def main() -> int:
    source = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "PASS_PIPELINE_REVIEW_REQUIRED":
        raise RuntimeError("Stage258 source audit is not the expected completed artifact")
    # The retained set is fixed from the already-audited historical coverage
    # only; no FMI 2026 observation value is read here.
    station_ids = sorted(str(value) for value in audit["retained_station_ids"])
    if len(station_ids) != 11:
        raise RuntimeError("unexpected retained FMI station count")
    scenarios = scenario_grid()
    payload = {
        "status": "PROTOCOL_FROZEN_BEFORE_2026_OBSERVATION_ACCESS",
        "protocol_id": "TMQC-FMI-2026-TEMPORAL-FACTOR-20260924-v1",
        "provider": source["provider"],
        "dataset": source["dataset"],
        "scope": "exploratory FMI 2026 temporal factor window",
        "station_ids": station_ids,
        "station_count": len(station_ids),
        "years": [2021, 2022, 2023, 2024, 2026],
        "split": {"train": [2021, 2022, 2023], "valid": [2024], "test": [2026]},
        "excluded_year": 2025,
        "test_period_start_utc": "2026-01-01T00:00:00Z",
        "test_period_end_utc": "2026-08-31T23:59:59Z",
        "test_period_is_partial_year": True,
        "primary_variable": "temperature_c",
        "quality_rule": source["missing_rule"],
        "coverage_rule": source["coverage_rule"],
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
            "scenarios": scenarios,
            "seeds": [3407, 6257, 7919, 104729, 20260715],
            "primary_far": 0.01,
            "primary_endpoint": "event_coverage",
        },
        "selection_boundary": (
            "Station IDs, years, split, variables, factor grid, methods, "
            "thresholds, seeds and statistics are frozen before FMI 2026 "
            "observation access. Test labels are evaluation-only."
        ),
        "provider_boundary": (
            "FMI is an existing provider; this is a new temporal window, "
            "not a new institution."
        ),
        "attack_boundary": "All attacks are controlled simulations, not production incidents.",
        "exploratory": True,
        "observation_access": False,
        "performance_access": False,
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "source_stage257_protocol_sha256": sha256(SOURCE_PROTOCOL),
        "source_stage258_audit_sha256": sha256(SOURCE_AUDIT),
        "source_stage258_normalized_sha256": sha256(SOURCE_NORMALIZED),
        "frozen_at_utc": "2026-09-24T00:00:00Z",
    }
    out_json = TABLES / f"{OUT_PREFIX}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPORTS / f"{OUT_PREFIX}.md").write_text(
        "\n".join(
            [
                "# Stage390 FMI 2026 因子协议冻结",
                "",
                f"- 状态：`{payload['status']}`",
                f"- 站点：`{payload['station_count']}`；训练：2021-2023；验证：2024；测试：2026-01-01 至 2026-08-31。",
                "- 2025 明确排除；协议冻结时未访问 FMI 2026 观测值。",
                "- 强度、持续时间、事件长度和 coordinated drift 未见组合均已冻结。",
                "- 本阶段属于新时间窗探索性证据，不替换 FMI 2025 或四 provider 主结果。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "station_count": len(station_ids), "scenario_count": len(scenarios)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
