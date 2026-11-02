from __future__ import annotations

"""Freeze the FMI 2026 factor evaluation contract before download."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL = TABLES / "stage390_fmi_2026_factor_protocol.json"
SOURCE_CONTRACT = TABLES / "stage385_met_norway_2026_factor_contract.json"
OUT_PREFIX = "stage391_fmi_2026_factor_contract"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_2026_OBSERVATION_ACCESS":
        raise RuntimeError("Stage390 protocol is not frozen")
    contract = {
        "status": "EVALUATION_CONTRACT_FROZEN_BEFORE_PERFORMANCE",
        "contract_id": "TMQC-FMI-2026-TEMPORAL-FACTOR-EVAL-20260924-v1",
        "protocol_id": protocol["protocol_id"],
        "provider": "Finnish Meteorological Institute (FMI)",
        "station_count": protocol["station_count"],
        "seed_count": 5,
        "method_count": 14,
        "primary_far": 0.01,
        "primary_endpoint": "event_coverage",
        "non_inferiority_margin": -0.02,
        "bootstrap": {
            "iterations": 10000,
            "seed": 20260920,
            "hierarchy": "station within FMI",
            "resampling_unit": "station",
        },
        "factor_protocol": protocol["attack_factor_protocol"],
        "methods": [
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
        "method_contract": source.get("method_contract", {}),
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "observations_accessed_before_freeze": False,
        "exploratory": True,
        "claim_boundary": [
            "This is a new temporal window on FMI, not a new provider.",
            "The 2025 FMI test remains untouched and is not used for tuning.",
            "All attacks are controlled simulations, not production incidents.",
            "A partial download or failed statistical gate cannot support comprehensive leadership.",
        ],
        "source_protocol_sha256": sha256(PROTOCOL),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "frozen_at_utc": "2026-09-24T00:00:00Z",
    }
    output = TABLES / f"{OUT_PREFIX}.json"
    output.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPORTS / f"{OUT_PREFIX}.md").write_text(
        "\n".join(
            [
                "# Stage391 FMI 2026 因子评价契约冻结",
                "",
                f"- 状态：`{contract['status']}`",
                "- 14 个方法、5 个种子、FAR=0.01、event coverage 主终点。",
                "- 契约冻结发生在 FMI 2026 下载和性能评价前。",
                "- 结果即使完整，也属于新时间窗探索性证据。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": contract["status"], "contract_id": contract["contract_id"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
