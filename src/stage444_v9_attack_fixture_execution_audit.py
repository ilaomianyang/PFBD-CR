from __future__ import annotations

"""Stage 444: execute the V9 attack generator on synthetic fixtures only."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
GENERATOR = SRC / "stage320_simulated_attack_confirmatory_evaluation.py"
OUT_JSON = ROOT / "tables" / "stage444_v9_attack_fixture_execution_audit.json"
OUT_REPORT = ROOT / "reports" / "stage444_v9_attack_fixture_execution_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_time = pd.Timestamp("2020-01-01T00:00:00Z")
    for split, start in (("train", 0), ("valid", 1000), ("test", 2000)):
        for station_index, station_id in enumerate(("s1", "s2", "s3")):
            for hour in range(400):
                timestamp = base_time + pd.Timedelta(hours=start + hour)
                value = (
                    12.0
                    + station_index
                    + 0.02 * hour
                    + 0.5 * np.sin(hour / 12.0)
                )
                rows.append(
                    {
                        "station_id": station_id,
                        "timestamp_utc": timestamp,
                        "temperature_c": value,
                        "observation_present": True,
                        "split": split,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    module = importlib.import_module("stage320_simulated_attack_confirmatory_evaluation")
    compat = importlib.import_module("stage425_v9_compatibility")
    module._scenario_base_type = compat.scenario_type
    frame = synthetic_frame()
    scenarios = {
        str(row["name"]): row for row in protocol["attack_scenarios"]
    }
    seed = 3407
    valid_blocks = module.make_simulated_blocks(
        frame, protocol, seed, "valid", int(protocol["valid_attack_seed_offset"])
    )
    test_blocks = module.make_simulated_blocks(
        frame, protocol, seed, "test", int(protocol["test_attack_seed_offset"])
    )
    valid_attacked = module.inject_simulated_attack(frame, valid_blocks, scenarios)
    test_attacked = module.inject_simulated_attack(frame, test_blocks, scenarios)

    checks: list[dict[str, object]] = []
    checks.append(
        {
            "check": "all_frozen_scenarios_present",
            "status": "PASS"
            if set(valid_blocks["scenario"]) == set(scenarios)
            and set(test_blocks["scenario"]) == set(scenarios)
            else "FAIL",
        }
    )
    expected_lengths = {
        str(name): int(spec["length"]) for name, spec in scenarios.items()
    }
    length_ok = True
    for block in (valid_blocks, test_blocks):
        for (station_id, scenario), part in block.groupby(
            ["station_id", "scenario"], sort=True
        ):
            if scenario == "coordinated_drift_unseen" and station_id not in {"s1", "s2"}:
                length_ok = False
            if scenario != "coordinated_drift_unseen" and len(part) != expected_lengths[scenario]:
                length_ok = False
            if scenario == "coordinated_drift_unseen" and len(part) != expected_lengths[scenario]:
                length_ok = False
    checks.append(
        {
            "check": "event_lengths_match_protocol",
            "status": "PASS" if length_ok else "FAIL",
        }
    )
    coord_valid = valid_blocks[valid_blocks["scenario"].eq("coordinated_drift_unseen")]
    coord_test = test_blocks[test_blocks["scenario"].eq("coordinated_drift_unseen")]
    coord_ok = (
        set(coord_valid["station_id"]) == {"s1", "s2"}
        and set(coord_test["station_id"]) == {"s1", "s2"}
        and set(
            coord_valid[coord_valid["station_id"].eq("s1")]["timestamp_utc"]
        )
        == set(coord_valid[coord_valid["station_id"].eq("s2")]["timestamp_utc"])
        and set(
            coord_test[coord_test["station_id"].eq("s1")]["timestamp_utc"]
        )
        == set(coord_test[coord_test["station_id"].eq("s2")]["timestamp_utc"])
    )
    checks.append(
        {
            "check": "coordinated_unseen_has_common_two_station_times",
            "status": "PASS" if coord_ok else "FAIL",
        }
    )
    blocks_unique = (
        not valid_blocks.duplicated(["station_id", "row_position"]).any()
        and not test_blocks.duplicated(["station_id", "row_position"]).any()
    )
    checks.append(
        {
            "check": "within_split_blocks_do_not_overlap",
            "status": "PASS" if blocks_unique else "FAIL",
        }
    )
    lag_rows_ok = True
    for block in (valid_blocks, test_blocks):
        for scenario, lag in (("short_lag", 1), ("long_delay", 8)):
            part = block[block["scenario"].eq(scenario)]
            lag_rows_ok &= bool(
                (part["row_position"] - part["source_row_position"] >= lag).all()
            )
    checks.append(
        {
            "check": "time_shift_source_precedes_target",
            "status": "PASS" if lag_rows_ok else "FAIL",
        }
    )
    train_clean = (
        valid_attacked.loc[valid_attacked["split"].eq("train"), "is_attack"].sum()
        == 0
        and test_attacked.loc[test_attacked["split"].eq("train"), "is_attack"].sum()
        == 0
    )
    checks.append(
        {
            "check": "train_has_no_attack_labels",
            "status": "PASS" if train_clean else "FAIL",
        }
    )
    valid_keys = set(
        zip(
            valid_blocks["station_id"],
            valid_blocks["scenario"],
            valid_blocks["row_position"],
        )
    )
    test_keys = set(
        zip(
            test_blocks["station_id"],
            test_blocks["scenario"],
            test_blocks["row_position"],
        )
    )
    checks.append(
        {
            "check": "valid_and_test_seed_streams_change_positions",
            "status": "PASS" if valid_keys != test_keys else "FAIL",
        }
    )
    # Verify one deterministic factor in the injected values: strong spike is
    # exactly 8 C above the clean fixture at every selected row.
    strong = test_blocks[test_blocks["scenario"].eq("strong_spike_long")]
    clean_lookup = frame.set_index(["station_id", "timestamp_utc"])["temperature_c"]
    injected = test_attacked.set_index(["station_id", "timestamp_utc"])["temperature_c"]
    spike_diffs = []
    for row in strong.itertuples(index=False):
        key = (row.station_id, row.timestamp_utc)
        spike_diffs.append(float(injected.loc[key] - clean_lookup.loc[key]))
    checks.append(
        {
            "check": "strong_spike_magnitude_matches_protocol",
            "status": "PASS"
            if spike_diffs and np.allclose(spike_diffs, 8.0)
            else "FAIL",
        }
    )
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_ATTACK_FIXTURE_EXECUTION_AUDIT"
        if not failures
        else "FAIL_V9_ATTACK_FIXTURE_EXECUTION_AUDIT"
    )
    payload = {
        "stage": 444,
        "status": status,
        "checks": checks,
        "failed_checks": failures,
        "synthetic_fixture_only": True,
        "official_observation_values_read": False,
        "old_test_rows_read": False,
        "performance_evaluated": False,
        "claim_boundary": [
            "本阶段只在人工构造夹具上执行攻击生成器，不使用官方观测值。",
            "夹具通过不代表任何检测性能或全面领先结论。",
        ],
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "generator": sha256(GENERATOR),
        },
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage444 V9 攻击生成器夹具执行审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 使用三站点人工构造序列，仅验证攻击生成器执行契约。",
                "- 未读取官方观测值、旧 test 或性能表。",
                "- 不产生检测性能结果。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 444, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
