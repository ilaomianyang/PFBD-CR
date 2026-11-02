from __future__ import annotations

"""Freeze evaluation contracts for the new 2026-09-21--23 holdout."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

METHODS = [
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
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(prefix: str, protocol_path: Path, source_contract: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = json.loads(source_contract.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("PROTOCOL_FROZEN_BEFORE_SEPTEMBER_21_2026"):
        raise RuntimeError(f"{prefix}: protocol not frozen")
    return {
        "status": "EVALUATION_CONTRACT_FROZEN_BEFORE_SEP21_2026_PERFORMANCE",
        "contract_id": f"TMQC-{prefix.upper()}-EVAL-20260924-v1",
        "protocol_id": protocol["protocol_id"],
        "provider": protocol["provider"],
        "station_count": protocol["station_count"],
        "seed_count": 5,
        "method_count": len(METHODS),
        "primary_far": 0.01,
        "primary_endpoint": "event_coverage",
        "non_inferiority_margin": -0.02,
        "bootstrap": {
            "iterations": 10000,
            "seed": 20260920,
            "hierarchy": "station within provider",
            "resampling_unit": "station",
        },
        "factor_protocol": protocol["attack_factor_protocol"],
        "methods": METHODS,
        "method_contract": source.get("method_contract", {}),
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "observations_accessed_before_freeze": False,
        "exploratory": True,
        "claim_boundary": [
            "Existing-provider temporal holdout; not a new institution.",
            "All attacks are controlled simulations, not production incidents.",
            "This exploratory window cannot replace the frozen main result.",
        ],
        "source_protocol_sha256": sha256(protocol_path),
        "source_contract_sha256": sha256(source_contract),
        "frozen_at_utc": "2026-09-24T00:00:00Z",
    }


def main() -> int:
    jobs = [
        (
            "fmi_sep21",
            TABLES / "stage408_fmi_sep21_factor_protocol.json",
            TABLES / "stage398_fmi_sep2026_factor_contract.json",
        ),
        (
            "met_norway_sep21",
            TABLES / "stage408_met_norway_sep21_factor_protocol.json",
            TABLES / "stage398_met_norway_sep2026_factor_contract.json",
        ),
    ]
    for name, protocol, source in jobs:
        payload = build(name, protocol, source)
        output = TABLES / f"stage409_{name}_factor_contract.json"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (REPORTS / f"stage409_{name}_factor_contract.md").write_text(
            "\n".join(
                [
                    f"# {name} 评价契约冻结",
                    "",
                    f"- 状态：`{payload['status']}`",
                    f"- 方法：`{payload['method_count']}`；种子：`{payload['seed_count']}`；FAR：`{payload['primary_far']}`。",
                    "- 协议和统计契约在 2026-09-21--23 观测访问前冻结。",
                    "- 结果只能作为探索性时间稳健性证据。",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "CONTRACTS_FROZEN_BEFORE_SEP21_2026_PERFORMANCE"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
