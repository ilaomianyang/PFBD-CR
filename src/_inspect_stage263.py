from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
PREFIX = "stage263_unified_statistical_report"


def main() -> None:
    payload = json.loads(
        (TABLES / f"{PREFIX}.json").read_text(encoding="utf-8")
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    for suffix in [
        "pooled",
        "station_equal",
        "provider_equal",
        "station_direction",
        "attack_type",
        "power_mde",
        "event_definition_audit",
        "low_far_and_f1_optimum",
        "comparisons",
        "leave_one_provider_out",
        "bootstrap_structure",
        "pooled_cross_source",
    ]:
        frame = pd.read_csv(TABLES / f"{PREFIX}_{suffix}.csv")
        print(f"\n{suffix}: {frame.shape}")
        print(list(frame.columns))
        print(frame.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
