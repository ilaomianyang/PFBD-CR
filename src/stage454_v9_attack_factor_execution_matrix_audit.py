from __future__ import annotations

"""Stage 454: execute the frozen V9 attack-factor matrix on memory fixtures.

This is a generator-contract audit, not a performance experiment.  It uses
only a deterministic in-memory frame and the frozen protocol.  It verifies
that intensity, duration, event length, lag, transformation values, labels,
the unseen coordinated combination, and independent seed streams are really
implemented by the V9 attack generator.
"""

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
OUT_JSON = ROOT / "tables" / "stage454_v9_attack_factor_execution_matrix_audit.json"
OUT_REPORT = ROOT / "reports" / "stage454_v9_attack_factor_execution_matrix_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_time = pd.Timestamp("2020-01-01T00:00:00Z")
    for split, start in (("train", 0), ("valid", 2000), ("test", 4000)):
        for station_index, station_id in enumerate(("s1", "s2", "s3")):
            for hour in range(600):
                timestamp = base_time + pd.Timedelta(hours=start + hour)
                value = (
                    12.0
                    + station_index
                    + 0.03 * hour
                    + 0.7 * np.sin(hour / 11.0)
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
    scenarios = {str(row["name"]): row for row in protocol["attack_scenarios"]}
    seed = 3407
    valid_offset = int(protocol["valid_attack_seed_offset"])
    test_offset = int(protocol["test_attack_seed_offset"])
    valid_blocks = module.make_simulated_blocks(frame, protocol, seed, "valid", valid_offset)
    test_blocks = module.make_simulated_blocks(frame, protocol, seed, "test", test_offset)
    valid_attacked = module.inject_simulated_attack(frame, valid_blocks, scenarios)
    test_attacked = module.inject_simulated_attack(frame, test_blocks, scenarios)
    valid_attacked_repeat = module.inject_simulated_attack(frame, valid_blocks, scenarios)

    clean_values = pd.to_numeric(frame["temperature_c"], errors="coerce").to_numpy(dtype=float)
    clean_lookup = frame.set_index(["station_id", "timestamp_utc"])["temperature_c"]
    attacked_values = pd.to_numeric(
        test_attacked["temperature_c"], errors="coerce"
    ).to_numpy(dtype=float)
    repeated_values = pd.to_numeric(
        valid_attacked_repeat["temperature_c"], errors="coerce"
    ).to_numpy(dtype=float)
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, value: object, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "value": value,
                "detail": detail,
            }
        )

    expected_names = set(scenarios)
    for split_name, blocks in (("valid", valid_blocks), ("test", test_blocks)):
        add(
            f"{split_name}_all_frozen_scenarios_present",
            set(blocks["scenario"].astype(str)) == expected_names,
            {
                "observed": sorted(set(blocks["scenario"].astype(str))),
                "expected_count": len(expected_names),
            },
            "每个 split 必须实际生成全部 13 个冻结场景。",
        )

    expected_lengths = {
        name: int(spec["length"]) for name, spec in scenarios.items()
    }
    length_failures: list[dict[str, object]] = []
    for split_name, blocks in (("valid", valid_blocks), ("test", test_blocks)):
        for (station_id, scenario), part in blocks.groupby(
            ["station_id", "scenario"], sort=True
        ):
            expected = expected_lengths[str(scenario)]
            if len(part) != expected:
                length_failures.append(
                    {
                        "split": split_name,
                        "station_id": str(station_id),
                        "scenario": str(scenario),
                        "actual": int(len(part)),
                        "expected": expected,
                    }
                )
    add(
        "all_event_lengths_match_protocol",
        not length_failures,
        {"failure_count": len(length_failures), "examples": length_failures[:5]},
        "强弱、持续时间和事件长度必须落实为冻结 length。",
    )

    overlap_failures = []
    for split_name, blocks in (("valid", valid_blocks), ("test", test_blocks)):
        duplicate = blocks.duplicated(["station_id", "row_position"])
        if duplicate.any():
            overlap_failures.append(
                {"split": split_name, "duplicate_rows": int(duplicate.sum())}
            )
    add(
        "scenario_blocks_do_not_overlap",
        not overlap_failures,
        overlap_failures,
        "同一站点同一 split 的不同场景不能共享攻击行。",
    )

    factor_missing = [
        name
        for name, spec in scenarios.items()
        if not all(
            key in spec
            for key in ("factor_intensity", "factor_duration", "factor_length")
        )
    ]
    factor_cells = {
        (
            str(spec["factor_intensity"]),
            str(spec["factor_duration"]),
            str(spec["factor_length"]),
        )
        for spec in scenarios.values()
    }
    add(
        "factor_metadata_complete_and_nonempty",
        not factor_missing and factor_cells == {("weak", "short", "short"), ("strong", "long", "long")},
        {
            "missing": factor_missing,
            "factor_cells": sorted(factor_cells),
        },
        "协议中的强度、持续时间和事件长度因子必须完整且只使用冻结单元。",
    )

    transformation_failures: list[dict[str, object]] = []
    clean_test = frame[frame["split"].eq("test")].copy()
    attacked_test = test_attacked[test_attacked["split"].eq("test")].copy()
    attacked_test_lookup = attacked_test.set_index(["station_id", "timestamp_utc"])[
        "temperature_c"
    ]
    for scenario_name, spec in scenarios.items():
        scenario_part = test_blocks[test_blocks["scenario"].eq(scenario_name)]
        for station_id, part in scenario_part.groupby("station_id", sort=True):
            for row in part.itertuples(index=False):
                key = (str(row.station_id), row.timestamp_utc)
                clean = float(clean_lookup.loc[key])
                attacked = float(attacked_test_lookup.loc[key])
                diff = attacked - clean
                base_type = str(spec["base_type"])
                if base_type in {"spike", "offset"}:
                    ok = np.isclose(diff, float(spec["magnitude"]))
                elif base_type == "drift":
                    positions = part["row_position"].to_numpy(dtype=int)
                    local_index = int(
                        np.where(positions == int(row.row_position))[0][0]
                    )
                    expected = np.linspace(
                        float(spec["start"]), float(spec["end"]), len(positions)
                    )[local_index]
                    ok = np.isclose(diff, expected)
                elif base_type == "time_shift":
                    source_key = (
                        str(row.station_id),
                        frame.loc[int(row.source_row_position), "timestamp_utc"],
                    )
                    ok = np.isclose(attacked, float(clean_lookup.loc[source_key]))
                elif base_type == "stuck":
                    first = part.iloc[0]
                    first_key = (
                        str(first["station_id"]),
                        first["timestamp_utc"],
                    )
                    ok = np.isclose(attacked, float(clean_lookup.loc[first_key]))
                else:
                    # Noise is stochastic by design; the deterministic seed
                    # check below is the stronger contract for this family.
                    ok = True
                if not ok:
                    transformation_failures.append(
                        {
                            "scenario": scenario_name,
                            "station_id": str(station_id),
                            "row_position": int(row.row_position),
                            "base_type": base_type,
                        }
                    )
    add(
        "scenario_value_transformations_match_protocol",
        not transformation_failures,
        {
            "failure_count": len(transformation_failures),
            "examples": transformation_failures[:5],
        },
        "spike/offset/drift/time-shift/stuck 的数值变换必须与冻结参数一致。",
    )

    noise_deltas = []
    for scenario_name in ("weak_noise_short", "strong_noise_long"):
        part = test_blocks[test_blocks["scenario"].eq(scenario_name)]
        for row in part.itertuples(index=False):
            key = (str(row.station_id), row.timestamp_utc)
            noise_deltas.append(
                float(attacked_test_lookup.loc[key]) - float(clean_lookup.loc[key])
            )
    add(
        "noise_is_nonconstant_and_injected",
        len(noise_deltas) > 0
        and np.isfinite(noise_deltas).all()
        and float(np.std(noise_deltas)) > 0.0
        and float(np.max(np.abs(noise_deltas))) > 0.0,
        {
            "count": len(noise_deltas),
            "std": float(np.std(noise_deltas)) if noise_deltas else 0.0,
        },
        "噪声场景必须产生非恒定扰动，不能退化成平滑或零扰动。",
    )

    add(
        "noise_and_full_injection_are_reproducible",
        np.array_equal(
            pd.to_numeric(valid_attacked["temperature_c"], errors="coerce").to_numpy(
                dtype=float
            ),
            repeated_values,
        ),
        {"same_seed_repeated_values": True},
        "相同 seed、相同 blocks 的注入值必须可复现。",
    )

    coord_checks: list[bool] = []
    for blocks in (valid_blocks, test_blocks):
        coord = blocks[blocks["scenario"].eq("coordinated_drift_unseen")]
        coord_checks.extend(
            [
                set(coord["station_id"].astype(str)) == {"s1", "s2"},
                len(
                    set(
                        coord[coord["station_id"].eq("s1")]["timestamp_utc"]
                    )
                    ^ set(
                        coord[coord["station_id"].eq("s2")]["timestamp_utc"]
                    )
                )
                == 0,
                len(coord) == 2 * int(scenarios["coordinated_drift_unseen"]["length"]),
            ]
        )
    add(
        "unseen_coordinated_combination_is_two_station_common_time",
        all(coord_checks),
        {"checks": len(coord_checks), "passed": int(sum(coord_checks))},
        "未见组合必须只作用于两站点并共享完整事件时间。",
    )

    attack_type_map = {
        name: str(spec["base_type"]) for name, spec in scenarios.items()
    }
    label_failures = []
    for row in test_blocks.itertuples(index=False):
        attacked_row = test_attacked.iloc[int(row.row_position)]
        if (
            not bool(attacked_row["is_attack"])
            or str(attacked_row["attack_scenario"]) != str(row.scenario)
            or str(attacked_row["attack_type"]) != attack_type_map[str(row.scenario)]
            or not str(attacked_row["attack_event_id"])
        ):
            label_failures.append(
                {
                    "scenario": str(row.scenario),
                    "station_id": str(row.station_id),
                    "row_position": int(row.row_position),
                }
            )
            if len(label_failures) >= 5:
                break
    train_attack_count = int(
        valid_attacked.loc[valid_attacked["split"].eq("train"), "is_attack"].sum()
    ) + int(test_attacked.loc[test_attacked["split"].eq("train"), "is_attack"].sum())
    add(
        "attack_labels_match_blocks_and_train_is_clean",
        not label_failures and train_attack_count == 0,
        {
            "label_failure_count": len(label_failures),
            "train_attack_count": train_attack_count,
        },
        "标签必须与 blocks 一一对应，train 必须保持无攻击。",
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
    add(
        "independent_valid_test_seed_stream_changes_positions",
        valid_keys != test_keys and valid_offset != test_offset,
        {
            "valid_attack_seed_offset": valid_offset,
            "test_attack_seed_offset": test_offset,
            "same_position_key_set": valid_keys == test_keys,
        },
        "valid/test 必须使用不同攻击随机流，不能复用位置。",
    )

    expected_attack_rows = int(
        sum(
            int(spec["length"])
            * (int(spec.get("station_count", 3)) if spec.get("coordinated") else 3)
            for spec in scenarios.values()
        )
    )
    add(
        "attack_row_count_matches_frozen_scenario_grid",
        len(valid_blocks) == expected_attack_rows
        and len(test_blocks) == expected_attack_rows,
        {
            "valid_rows": int(len(valid_blocks)),
            "test_rows": int(len(test_blocks)),
            "expected_rows": expected_attack_rows,
        },
        "场景×站点×事件长度总行数必须与冻结协议一致。",
    )

    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_ATTACK_FACTOR_EXECUTION_MATRIX_AUDIT"
        if not failures
        else "FAIL_V9_ATTACK_FACTOR_EXECUTION_MATRIX_AUDIT"
    )
    payload = {
        "stage": 454,
        "status": status,
        "protocol_id": protocol.get("protocol_id"),
        "checks": checks,
        "failed_checks": failures,
        "synthetic_fixture_only": True,
        "official_observation_values_read": False,
        "old_test_rows_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "claim_boundary": [
            "本阶段只在人工构造内存序列上验证攻击生成器执行契约。",
            "通过不代表官方观测数据已下载，也不代表检测性能或全面领先。",
            "所有攻击仍是透明受控模拟攻击，不是真实生产攻击日志。",
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
                "# Stage454 V9 攻击因子执行矩阵审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 使用三站点、三 split 的人工内存序列，执行全部 13 个冻结场景。",
                "- 核验强度、持续时间、事件长度、数值变换、标签、独立 seed 和未见协调组合。",
                "- 未读取官方观测值、旧 test 或性能表。",
                "",
                "## 失败检查",
                "",
                *([f"- `{name}`" for name in failures] or ["- 无。"]),
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": 454,
                "status": status,
                "failed_checks": failures,
                "synthetic_fixture_only": True,
                "performance_evaluated": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
