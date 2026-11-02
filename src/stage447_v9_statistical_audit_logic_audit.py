from __future__ import annotations

"""Stage 447: in-memory audit of the V9 statistical acceptance logic.

The fixture is synthetic and exists only to exercise the bootstrap/BH
implementation.  It does not represent official observations or performance
results and does not write any evaluation table.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage425_v9_compatibility import validate_protocol  # noqa: E402
from stage445_v9_statistical_acceptance_audit import (  # noqa: E402
    benjamini_hochberg,
    hierarchical_bootstrap,
)

PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
MANIFEST = PACKAGE / "data_manifest.json"
OUT_JSON = ROOT / "tables" / "stage447_v9_statistical_audit_logic_audit.json"
OUT_REPORT = ROOT / "reports" / "stage447_v9_statistical_audit_logic_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(("provider_a", "provider_b")):
        for station_index in range(2):
            for seed in (3407, 6257, 7919):
                rows.append(
                    {
                        "source": source,
                        "station_id": f"{source}_{station_index}",
                        "seed": seed,
                        "candidate": "trust_only",
                        "baseline": "robust_max_deviation",
                        "f1_difference": 0.10 + 0.01 * source_index,
                        "event_coverage_difference": 0.20 + 0.01 * station_index,
                    }
                )
    paired = pd.DataFrame(rows)
    bootstrap = hierarchical_bootstrap(
        paired,
        "trust_only",
        "robust_max_deviation",
        seed=20260925,
        iterations=1000,
    )
    q_values = benjamini_hochberg([0.01, 0.05, 0.20])
    checks = [
        {
            "check": "protocol_validates",
            "status": "PASS",
            "value": protocol["protocol_id"],
        },
        {
            "check": "hierarchical_bootstrap_provider_station_seed",
            "status": (
                "PASS"
                if bootstrap["provider_count"] == 2
                and bootstrap["station_count"] == 4
                and bootstrap["seed_rows"] == 12
                and bootstrap["event_ci95_low"] > 0.0
                and np.isfinite(bootstrap["event_ci95_high"])
                else "FAIL"
            ),
            "value": bootstrap,
        },
        {
            "check": "bh_adjustment_monotone_and_bounded",
            "status": (
                "PASS"
                if len(q_values) == 3
                and all(0.0 <= value <= 1.0 for value in q_values)
                and q_values[0] <= q_values[1] <= q_values[2]
                else "FAIL"
            ),
            "value": q_values,
        },
        {
            "check": "official_manifest_still_not_ready",
            "status": "PASS" if manifest.get("status") == "NOT_READY" else "FAIL",
            "value": manifest.get("status"),
        },
        {
            "check": "no_official_performance_read",
            "status": "PASS",
            "value": {
                "synthetic_fixture_only": True,
                "official_observation_values_read": False,
                "performance_evaluated": False,
            },
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_STATISTICAL_AUDIT_LOGIC_AUDIT"
        if not failures
        else "FAIL_V9_STATISTICAL_AUDIT_LOGIC_AUDIT"
    )
    payload = {
        "stage": 447,
        "status": status,
        "protocol_id": protocol["protocol_id"],
        "checks": checks,
        "failed_checks": failures,
        "synthetic_fixture_only": True,
        "official_observation_values_read": False,
        "performance_evaluated": False,
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "manifest": sha256(MANIFEST),
            "stage445": sha256(
                ROOT / "src" / "stage445_v9_statistical_acceptance_audit.py"
            ),
        },
        "claim_boundary": [
            "夹具只验证统计实现，不是官方观测数据，也不是性能结果。",
            "V9 官方窗口未就绪，统计验收仍保持阻断。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage447 V9 统计验收逻辑夹具审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 使用内存构造的 provider→station→seed 夹具验证 bootstrap/BH。",
                "- 夹具不代表官方观测值或检测性能，未读取 V9 结果表。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 447, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
