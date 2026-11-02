from __future__ import annotations

"""Freeze a prospective NOAA ISD 2026 partial-year holdout.

No 2026 observations are read here.  The station set comes only from the
already-frozen Stage294 Japan network, while 2025 is deliberately excluded
from the new train/valid/test split.  This package is independent of the
previous test results and cannot overwrite any historical result.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
PACKAGE = ROOT / "confirmatory_package" / "noaa_2026_temporal"
PACKAGE.mkdir(parents=True, exist_ok=True)

SOURCE_PROTOCOL = TABLES / "stage294_japan_isd_protocol_preregistration.json"
OUT_PROTOCOL = PACKAGE / "protocol_freeze.json"
OUT_TABLE = TABLES / "stage375_noaa_2026_protocol.json"
OUT_REPORT = ROOT / "reports" / "stage375_noaa_2026_protocol.md"

STATIONS = [
    "47911099999",
    "47942599999",
    "47991099999",
    "47843099999",
    "47766099999",
    "47672099999",
    "47790099999",
    "47624099999",
    "47585099999",
    "47574099999",
    "47425099999",
    "47420099999",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    source = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    if source["status"] != "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD":
        raise RuntimeError("Stage294 source protocol is not frozen")
    if source["station_ids"] != STATIONS:
        raise RuntimeError("Stage294 station list changed")

    protocol = {
        "status": "PROTOCOL_FROZEN_BEFORE_2026_OBSERVATION_ACCESS",
        "protocol_id": "TMQC-NOAA-JAPAN-2026-TEMPORAL-20260924-v1.0",
        "package_id": "noaa_2026_temporal_v1",
        "provider": "NOAA National Centers for Environmental Information (NCEI)",
        "dataset": "Integrated Surface Database (ISD), hourly station observations",
        "dataset_root": "https://www.ncei.noaa.gov/data/global-hourly/access",
        "network_id": "NOAA_ISD_JAPAN_MAINLAND_V1",
        "station_ids": STATIONS,
        "station_count": len(STATIONS),
        "years": [2021, 2022, 2023, 2024, 2026],
        "split": {
            "train": [2021, 2022, 2023],
            "valid": [2024],
            "test": [2026],
        },
        "excluded_year": 2025,
        "test_period_start_utc": "2026-01-01T00:00:00Z",
        "test_period_end_utc": "2026-08-31T23:59:59Z",
        "test_period_is_partial_year": True,
        "primary_variable": "temperature_c",
        "auxiliary_variables": [
            "dew_point_c",
            "wind_speed_ms",
            "pressure_hpa",
        ],
        "coverage_rule": (
            "Retain a station only if temperature coverage is at least 0.70 "
            "in every split year, with at least one non-missing value in each."
        ),
        "quality_rule": (
            "Use only official ISD per-field quality flag 1; all other "
            "flags and missing sentinels remain missing."
        ),
        "download_contract": (
            "Only https://www.ncei.noaa.gov/data/global-hourly/access/2026/"
            "<station_id>.csv; record URL, retrieval time, bytes, response "
            "headers and SHA256. No mirrors or derived observations."
        ),
        "attack_types": [
            "spike",
            "offset",
            "drift",
            "time_shift",
            "stuck",
            "gaussian_noise",
        ],
        "attack_parameters": source["attack_parameters"],
        "seeds": [3407, 6257, 7919, 104729, 20260715],
        "primary_far": 0.01,
        "primary_endpoint": "event_coverage",
        "secondary_endpoints": [
            "f1",
            "precision",
            "recall",
            "row_recall",
            "false_alarm_rate",
            "average_precision",
        ],
        "methods": [
            "trust_valid_f1",
            "trust_nonstuck_event_macro",
            "robust_max_deviation",
            "isolation_forest",
            "one_class_svm",
            "pca_reconstruction",
            "mlp_autoencoder_reconstruction",
            "knn_distance",
            "ewma_deviation",
            "arima_innovation",
            "gru_predictor",
            "lightweight_autoencoder",
            "ics_cps_spc_proxy",
        ],
        "selection_boundary": (
            "All methods, features, weights, thresholds, attack parameters, "
            "and statistics are frozen before 2026 observations are read. "
            "The 2026 test is evaluation-only."
        ),
        "provider_boundary": (
            "This is a new time window on the same NOAA provider and Japan "
            "station network, not a new provider or an independence proof."
        ),
        "attack_boundary": (
            "All attack labels are transparent controlled simulations, not "
            "real production attacks."
        ),
        "source_stage294_sha256": sha256(SOURCE_PROTOCOL),
        "performance_status": "NOT_EVALUATED",
        "download_status": "NOT_DOWNLOADED",
        "observations_accessed_before_freeze": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    serialized = json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    OUT_PROTOCOL.write_text(serialized, encoding="utf-8")
    OUT_TABLE.write_text(serialized, encoding="utf-8")
    report = [
        "# Stage375 NOAA ISD 2026 时间窗协议冻结",
        "",
        "- 状态：`PROTOCOL_FROZEN_BEFORE_2026_OBSERVATION_ACCESS`",
        "- provider：NOAA NCEI；网络：既有日本站网，不是新 provider。",
        "- 站点：12 个，来源为 Stage294 已冻结站点清单。",
        "- 切分：2021–2023 train，2024 valid，2026-01-01 至 2026-08-31 test。",
        "- 2025 年明确排除，不参与本包训练、验证或测试。",
        "- 方法、基线、攻击类型、种子、FAR=0.01、主终点和统计边界已冻结。",
        "- 本脚本未读取任何 2026 观测值；下载和覆盖率审计属于下一步。",
        "- 所有攻击仍是透明受控模拟，不能解释为真实生产攻击。",
    ]
    OUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "station_count": protocol["station_count"],
                "test_period": [
                    protocol["test_period_start_utc"],
                    protocol["test_period_end_utc"],
                ],
                "observations_accessed_before_freeze": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
