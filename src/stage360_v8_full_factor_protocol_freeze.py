from __future__ import annotations

"""Freeze an orthogonal attack-factor matrix before evaluation.

V8 separates three factors in the simulator contract:
intensity (weak/strong), active attack duration (short/long), and the
surrounding event-window length (short/long).  The eight combinations are
applied to a common offset attack family, plus one unseen coordinated-drift
combination.  This is a stress-test package, not a new provider or a claim
of real production attacks.
"""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
V6 = ROOT / "confirmatory_package" / "simulated_attack_v6" / "protocol_freeze.json"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v8"
PROTOCOL = PACKAGE / "protocol_freeze.json"
REPORT = ROOT / "reports" / "stage360_v8_full_factor_protocol_freeze.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out = copy.deepcopy(json.loads(V6.read_text(encoding="utf-8")))
    scenarios = []
    for intensity, magnitude in (("weak", 0.8), ("strong", 4.0)):
        for duration, active_length in (("short", 3), ("long", 12)):
            for event_length_name, event_length in (("short", 8), ("long", 24)):
                scenarios.append(
                    {
                        "name": f"offset_{intensity}_{duration}_{event_length_name}",
                        "base_type": "offset",
                        "magnitude": magnitude,
                        "length": event_length,
                        "active_length": active_length,
                        "factor_intensity": intensity,
                        "factor_duration": duration,
                        "factor_length": event_length_name,
                        "factorial": True,
                    }
                )
    scenarios.append(
        {
            "name": "coordinated_drift_unseen",
            "base_type": "drift",
            "start": 0.0,
            "end": 4.0,
            "length": 24,
            "active_length": 12,
            "station_count": 2,
            "coordinated": True,
            "factor_intensity": "strong",
            "factor_duration": "long",
            "factor_length": "long",
            "unseen_combination": True,
            "factorial": False,
        }
    )
    out["status"] = "SIMULATION_PROTOCOL_V8_FULL_FACTOR_FROZEN"
    out["protocol_id"] = "TMQC-SIMULATED-ATTACK-V8-FULL-FACTOR-20260924"
    out["frozen_at_utc"] = "2026-09-24T00:00:00Z"
    out["purpose"] = (
        "在四个冻结 provider 的真实观测背景上，独立分离攻击强度、有效攻击持续时间、"
        "事件窗口长度并检验未见 coordinated drift 组合；所有攻击均为透明模拟注入。"
    )
    out["valid_attack_seed_offset"] = 1050001
    out["test_attack_seed_offset"] = 1060001
    out["primary_far"] = 0.0065
    out["attack_scenarios"] = scenarios
    out["factor_matrix"] = {
        "factorial_family": "offset",
        "intensity": ["weak", "strong"],
        "active_duration": ["short=3 rows", "long=12 rows"],
        "event_window_length": ["short=8 rows", "long=24 rows"],
        "factorial_cells": 8,
        "unseen_combination": "coordinated_drift_unseen",
        "inactive_context_rule": "event window remains in the time series, but only active_length rows receive attack labels and perturbation",
    }
    out["claim_boundary"] = [
        "This package supports only simulated/controlled attack claims.",
        "V8 uses the same four provider/time-window inputs as V6 with a new attack random stream.",
        "Because the underlying V6 test windows were previously observed, V8 is exploratory and cannot replace the V6 confirmatory result.",
        "All baseline wins, failures, and factor-specific reversals must remain reportable.",
    ]
    out["source_protocol_sha256"] = sha256(V6)
    out["source_code_sha256"] = sha256(SRC / "stage360_v8_full_factor_protocol_freeze.py")
    PACKAGE.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "\n".join(
            [
                "# Stage360 V8 全因子攻击压力测试协议冻结",
                "",
                "- 状态：`SIMULATION_PROTOCOL_V8_FULL_FACTOR_FROZEN`",
                f"- 协议：`{out['protocol_id']}`",
                "- 因子：强度（弱/强）×有效持续时间（短/长）×事件窗口长度（短/长），共 8 个全因子单元。",
                "- 另加一个未见 coordinated drift 组合。",
                "- 四个 provider 和原始时间窗沿用 V6；V8 使用新的攻击随机流。",
                "- 由于底层 V6 test 窗口曾被查看，本轮只能作为探索性压力测试，不能替换 V6 确认性结果。",
                "- 所有攻击均为透明模拟注入，不是真实生产攻击日志。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": out["status"], "protocol_id": out["protocol_id"], "path": str(PROTOCOL)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
