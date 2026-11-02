from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"


def describe(name: str, path: Path) -> None:
    frame = pd.read_csv(path)
    print(f"\n=== {name} ===")
    print("shape", frame.shape)
    print("columns", list(frame.columns))
    print(frame.head(3).to_string(index=False))


def main() -> None:
    for prefix in [
        "stage248_cross_provider_fair_baseline_audit",
        "stage256_lightweight_strong_baselines",
        "stage262_security_domain_baselines",
    ]:
        for suffix in [
            "evaluation",
            "station_detail",
            "type_detail",
            "type_aggregate",
            "evaluation_aggregate",
            "provider_equal",
            "comparisons",
            "leave_one_provider_out",
        ]:
            path = TABLES / f"{prefix}_{suffix}.csv"
            if path.exists():
                describe(f"{prefix}_{suffix}", path)


if __name__ == "__main__":
    main()
