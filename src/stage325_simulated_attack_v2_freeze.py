from __future__ import annotations
"""Freeze simulated-attack protocol V2 before any new evaluation."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v2"
PROTOCOL = PACKAGE / "protocol_freeze.json"

PROVIDERS = {
    "MeteoSwiss": {
        "normalized": "tables/stage216_swiss_normalized.parquet",
        "protocol": "tables/stage214_swiss_protocol_preregistration.json",
        "split": "2021-2023 train / 2024 valid / 2025 test",
    },
    "NOAA ISD": {
        "normalized": "tables/stage82_noaa_extended_normalized.parquet",
        "protocol": "tables/stage222_noaa_transfer_protocol_preregistration.json",
        "split": "2020-2022 train / 2023 valid / 2024 test",
    },
    "KNMI": {
        "normalized": "tables/stage226_knmi_normalized.parquet",
        "protocol": "tables/stage224_knmi_protocol_preregistration.json",
        "split": "2021-2023 train / 2024 valid / 2025 test",
    },
    "ECCC": {
        "normalized": "tables/stage236_eccc_normalized.parquet",
        "protocol": "tables/stage234_eccc_protocol_preregistration.json",
        "split": "2021-2023 train / 2024 valid / 2025 test",
    },
}

SEEDS = [3407, 6257, 7919, 104729, 20260715]

ATTACK_SCENARIOS = [
    {"name": "low_spike", "base_type": "spike", "magnitude": 5.0, "length": 4},
    {"name": "low_offset", "base_type": "offset", "magnitude": 2.5, "length": 12},
    {"name": "slow_drift", "base_type": "drift", "start": 0.0, "end": 3.0, "length": 24},
    {"name": "short_lag", "base_type": "time_shift", "lag_rows": 2, "length": 6},
    {"name": "delay", "base_type": "time_shift", "lag_rows": 4, "length": 12},
    {"name": "long_stuck", "base_type": "stuck", "length": 16},
    {"name": "drop_replay", "base_type": "stuck", "length": 8, "description": "丢包后旧值保持不变的链路重放"},
    {"name": "low_noise", "base_type": "gaussian_noise", "sigma": 1.0, "length": 12},
    {"name": "coordinated_offset", "base_type": "offset", "magnitude": 2.5, "length": 8, "station_count": 2, "coordinated": True},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(Path(__file__))
    provider_payload = {}
    for name, spec in PROVIDERS.items():
        normalized = ROOT / spec["normalized"].replace("/", "\\")
        protocol = ROOT / spec["protocol"].replace("/", "\\")
        if not normalized.exists() or not protocol.exists():
            raise FileNotFoundError(f"missing frozen input for {name}")
        provider_payload[name] = {**spec, "normalized_sha256": sha256(normalized), "protocol_sha256": sha256(protocol)}
    payload = {
        "status": "SIMULATION_PROTOCOL_V2_FROZEN_BEFORE_PERFORMANCE_EVALUATION",
        "protocol_id": "TMQC-SIMULATED-ATTACK-V2-20260922",
        "frozen_at_utc": "2026-09-22T00:00:00Z",
        "purpose": "真实气象观测背景上的预注册模拟攻击确认性评价；不代表真实生产攻击日志。",
        "provider_count": 4,
        "providers": provider_payload,
        "seeds": SEEDS,
        "primary_far": 0.01,
        "primary_endpoint": "event_coverage",
        "secondary_endpoints": ["f1", "precision", "row_recall", "auprc", "false_alarm_rate"],
        "bootstrap": {"unit": "station", "iterations": 10000, "seed": 20260923},
        "multiplicity": "Benjamini-Hochberg over provider x attack scenario comparisons",
        "attack_scenarios": ATTACK_SCENARIOS,
        "valid_attack_seed_offset": 930001,
        "test_attack_seed_offset": 940001,
        "attack_generation_rules": {
            "train": "normal observations only; no attack injection",
            "valid": "scenario blocks injected with the frozen V2 generator; valid labels may select only the frozen candidate rule and FAR threshold",
            "test": "scenario blocks injected with an independent V2 seed stream; test labels are evaluation-only",
            "delay": "value replayed from a fixed four-row lag; it is a delayed measurement, not a timestamp edit",
            "drop_replay": "a dropped packet is represented by an old-value hold while observation_present remains true; this avoids giving every method a free missingness label",
            "coordinated": "two stations receive the same attack start and scenario, selected deterministically from the frozen station order",
            "unseen_boundary": "delay/drop-replay are new attack families relative to V1; V1 parameters are not reused for selection",
        },
        "method_freeze": {
            "trust_candidates": ["trust_valid_f1", "trust_nonstuck_event_macro"],
            "baselines": ["robust_max_deviation", "isolation_forest", "one_class_svm_20k", "one_class_svm_50k", "one_class_svm_100k", "pca_reconstruction", "mlp_autoencoder_reconstruction", "knn_distance", "ewma_deviation", "arima_innovation", "gru_predictor", "lightweight_autoencoder", "ics_cps_spc_proxy"],
            "threshold_rule": "valid normal rows only at FAR=0.01",
            "selection_rule": "no test labels, no test threshold selection, no attack-specific test tuning",
        },
        "claim_boundary": [
            "This package can support only simulated/controlled attack claims.",
            "Any baseline win or attack-type failure must remain in the final report.",
            "The package reuses the four frozen provider observations; it does not establish institution independence.",
        ],
        "source_code_sha256": source_hash,
    }
    PROTOCOL.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "protocol": str(PROTOCOL), "scenario_count": len(ATTACK_SCENARIOS), "source_code_sha256": source_hash}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
