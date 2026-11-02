from __future__ import annotations

"""Stage 429: metadata-only schema contract audit for V9.

No parquet rows are read.  The audit checks that frozen historical sources
expose the columns required by the V9 train/valid + future-test merge, and
records the required schema for the future provider-specific adapters.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
TABLES = ROOT / "tables"
OUT_JSON = TABLES / "stage429_v9_schema_contract_audit.json"
OUT_REPORT = ROOT / "reports" / "stage429_v9_schema_contract_audit.md"

COMMON_COLUMNS = {
    "station_id",
    "timestamp_utc",
    "temperature_c",
}
FUTURE_ADAPTER_COLUMNS = {
    "station_id",
    "timestamp_utc",
    "temperature_c",
    "observation_present",
}


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    providers: list[dict[str, object]] = []
    all_old_pass = True
    for provider, meta in protocol["providers"].items():
        path = ROOT / str(meta["normalized"])
        row: dict[str, object] = {
            "provider": provider,
            "path": str(path),
            "file_present": path.exists(),
            "observation_values_read": False,
            "future_test_schema_status": "NOT_READY",
            "required_future_adapter_columns": sorted(FUTURE_ADAPTER_COLUMNS),
        }
        required = set(COMMON_COLUMNS)
        if provider != "NOAA ISD":
            required |= {"observation_present", "split"}
        row["required_historical_columns"] = sorted(required)
        if not path.exists():
            row["old_schema_status"] = "FAIL_MISSING"
            row["columns"] = []
            all_old_pass = False
            providers.append(row)
            continue
        parquet = pq.ParquetFile(path)
        names = set(str(name) for name in parquet.schema_arrow.names)
        missing = sorted(required - names)
        row.update(
            {
                "old_schema_status": "PASS" if not missing else "FAIL",
                "columns": sorted(names),
                "missing_columns": missing,
                "num_rows_metadata": int(parquet.metadata.num_rows),
                "row_groups_metadata": int(parquet.metadata.num_row_groups),
            }
        )
        if missing:
            all_old_pass = False
        providers.append(row)
    status = (
        "PASS_V9_SCHEMA_CONTRACT_AUDIT"
        if all_old_pass
        else "FAIL_V9_SCHEMA_CONTRACT_AUDIT"
    )
    payload = {
        "stage": 429,
        "status": status,
        "protocol_id": protocol["protocol_id"],
        "observation_values_read": False,
        "parquet_rows_read": False,
        "performance_evaluated": False,
        "old_train_valid_schema": "PASS" if all_old_pass else "FAIL",
        "future_test_schema": "NOT_READY",
        "required_common_columns": sorted(COMMON_COLUMNS),
        "providers": providers,
        "claim_boundary": [
            "本阶段只读取 parquet 元数据，不读取观测行。",
            "新官方 test 文件未到位前，不能运行 V9 性能评价。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage429 V9 schema contract audit",
                "",
                f"- 状态：`{status}`",
                f"- provider：`{len(providers)}`",
                "- 仅读取 parquet schema/metadata，未读取观测值。",
                "- 新官方 test schema 当前为 `NOT_READY`。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": 429,
                "status": status,
                "provider_count": len(providers),
                "observation_values_read": False,
                "future_test_schema": "NOT_READY",
            },
            ensure_ascii=False,
        )
    )
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
