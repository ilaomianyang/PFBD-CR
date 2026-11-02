from __future__ import annotations

"""Freeze a new attack-factor stress-test before evaluating its outputs.

The V7 package reuses the four frozen provider/time-window inputs and the
frozen V6 detector/baseline contract, but introduces a new attack-factor
matrix (intensity, duration, event length, and an unseen coordinated drift
combination).  It is not allowed to overwrite V6 outputs.
"""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
V6 = ROOT / "confirmatory_package" / "simulated_attack_v6" / "protocol_freeze.json"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v7"
PROTOCOL = PACKAGE / "protocol_freeze.json"
REPORT = ROOT / "reports" / "stage357_v7_factor_protocol_freeze.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out = copy.deepcopy(json.loads(V6.read_text(encoding="utf-8")))
    out["status"] = "SIMULATION_PROTOCOL_V7_FACTOR_STRESS_TEST_FROZEN"
    out["protocol_id"] = "TMQC-SIMULATED-ATTACK-V7-FACTOR-20260924"
    out["frozen_at_utc"] = "2026-09-24T00:00:00Z"
    out["purpose"] = (
        "在四个冻结 provider 的真实观测背景上，对攻击强度、持续时间、事件长度和未见组合进行"
        "预先冻结的模拟攻击压力测试；不代表真实生产攻击日志。"
    )
    out["valid_attack_seed_offset"] = 1030001
    out["test_attack_seed_offset"] = 1040001
    out["primary_far"] = 0.0065
    out["attack_scenarios"] = [
        {"name": "weak_spike_short", "base_type": "spike", "magnitude": 2.0, "length": 3,
         "factor_intensity": "weak", "factor_duration": "short", "factor_length": "short"},
        {"name": "strong_spike_long", "base_type": "spike", "magnitude": 8.0, "length": 20,
         "factor_intensity": "strong", "factor_duration": "long", "factor_length": "long"},
        {"name": "weak_offset_short", "base_type": "offset", "magnitude": 0.8, "length": 6,
         "factor_intensity": "weak", "factor_duration": "short", "factor_length": "short"},
        {"name": "strong_offset_long", "base_type": "offset", "magnitude": 4.0, "length": 24,
         "factor_intensity": "strong", "factor_duration": "long", "factor_length": "long"},
        {"name": "short_drift", "base_type": "drift", "start": 0.0, "end": 1.5, "length": 12,
         "factor_intensity": "weak", "factor_duration": "short", "factor_length": "short"},
        {"name": "long_drift", "base_type": "drift", "start": 0.0, "end": 6.0, "length": 48,
         "factor_intensity": "strong", "factor_duration": "long", "factor_length": "long"},
        {"name": "short_lag", "base_type": "time_shift", "lag_rows": 1, "length": 4,
         "factor_intensity": "weak", "factor_duration": "short", "factor_length": "short"},
        {"name": "long_delay", "base_type": "time_shift", "lag_rows": 8, "length": 24,
         "factor_intensity": "strong", "factor_duration": "long", "factor_length": "long"},
        {"name": "short_stuck", "base_type": "stuck", "length": 6,
         "factor_intensity": "weak", "factor_duration": "short", "factor_length": "short"},
        {"name": "long_stuck", "base_type": "stuck", "length": 32,
         "factor_intensity": "strong", "factor_duration": "long", "factor_length": "long"},
        {"name": "weak_noise_short", "base_type": "gaussian_noise", "sigma": 0.4, "length": 6,
         "factor_intensity": "weak", "factor_duration": "short", "factor_length": "short"},
        {"name": "strong_noise_long", "base_type": "gaussian_noise", "sigma": 2.5, "length": 24,
         "factor_intensity": "strong", "factor_duration": "long", "factor_length": "long"},
        {
            "name": "coordinated_drift",
            "base_type": "drift",
            "start": 0.0,
            "end": 4.0,
            "length": 24,
            "station_count": 2,
            "coordinated": True,
            "factor_intensity": "strong",
            "factor_duration": "long",
            "factor_length": "long",
            "unseen_combination": True,
        },
    ]
    out["factor_matrix"] = {
        "intensity": ["weak", "strong"],
        "duration": ["short", "long"],
        "event_length": ["short", "long"],
        "unseen_combination": "coordinated_drift (coordinated drift was absent from V6)",
    }
    out["claim_boundary"] = [
        "This package supports only simulated/controlled attack claims.",
        "V7 reuses the four frozen provider/time-window inputs and tests a new attack-factor stream.",
        "Because the underlying V6 test windows were previously observed, V7 is exploratory with respect to "
        "model selection and cannot replace the V6 confirmatory result.",
        "All baseline wins and failures must remain reportable.",
    ]
    out["source_protocol_sha256"] = sha256(V6)
    out["source_code_sha256"] = sha256(SRC / "stage357_v7_factor_protocol_freeze.py")
    PACKAGE.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "\n".join(
            [
                "# Stage357 V7 攻击因子压力测试协议冻结",
                "",
                "- 状态：`SIMULATION_PROTOCOL_V7_FACTOR_STRESS_TEST_FROZEN`",
                f"- 协议：`{out['protocol_id']}`",
                "- 范围：强度、持续时间、事件长度与未见 coordinated drift 组合。",
                "- 四个 provider 和原始时间窗沿用 V6；新攻击随机流和协议已在评价前冻结。",
                "- 由于底层 V6 test 窗口曾被查看，本轮结果只能作为探索性压力测试，不能替换 V6 确认性结果。",
                "- 所有攻击均为透明模拟注入，不是真实生产攻击日志。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": out["status"], "protocol_id": out["protocol_id"], "path": str(PROTOCOL)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
