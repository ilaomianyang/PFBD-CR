from __future__ import annotations

"""Freeze an explicitly exploratory ARIMA-aware union variant.

This protocol is created after Stage395 was inspected.  It therefore cannot
produce confirmatory evidence and must never replace the frozen main result.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    payload = {
        "status": "EXPLORATORY_PROTOCOL_FROZEN_AFTER_STAGE395_VIEW",
        "protocol_id": "TMQC-STAGE396-ARIMA-AWARE-UNION-20260925-v1",
        "exploratory": True,
        "confirmatory_promotion": False,
        "reason": "Stage395 showed stable temporal-window weakness against arima_innovation on weak/long offset factors.",
        "candidate": "trust_arima_max_single_threshold",
        "candidate_definition": {
            "score": "max(trust_valid_f1_z, ewma_deviation_z, arima_innovation_z)",
            "threshold": "one threshold calibrated on valid normal rows at FAR=0.01",
            "test_access": "test labels are evaluation-only",
        },
        "inputs": [
            "stage390_fmi_2026_factor_protocol.json",
            "stage391_fmi_2026_factor_contract.json",
            "stage392_fmi_2026_factor_download_normalized.parquet",
            "stage384_met_norway_2026_factor_protocol.json",
            "stage385_met_norway_2026_factor_contract.json",
            "stage386_met_norway_2026_factor_download_normalized.parquet",
        ],
        "fixed_methods": [
            "trust_valid_f1",
            "trust_nonstuck_event_macro",
            "trust_shape_union",
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
        "primary_far": 0.01,
        "primary_endpoint": "event_coverage",
        "seeds": [3407, 6257, 7919, 104729, 20260715],
        "scenario_count": 9,
        "baselines": 11,
        "acceptance": {
            "integrity": "5 seeds × 14 methods × 9 scenarios for each partial temporal window",
            "far_max": 0.03,
            "diagnostic_goal": "improve or match ARIMA event coverage without losing Trust F1 and without changing frozen results",
            "no_claim": "even if thresholds pass, result remains exploratory because Stage395 inputs were already viewed",
        },
        "stop_rules": [
            "Do not change any frozen Stage246/V6/V8 result.",
            "Do not select a candidate using test labels.",
            "If the candidate is identical to or worse than arima_innovation on both windows, stop this route.",
            "If FAR exceeds 0.03 or any shape is incomplete, stop and retain failure.",
        ],
    }
    output = TABLES / "stage396_adaptive_arima_union_protocol.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload["sha256"] = sha256(output)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "sha256": payload["sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
