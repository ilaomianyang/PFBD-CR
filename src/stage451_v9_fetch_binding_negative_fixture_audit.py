from __future__ import annotations

"""Stage 451: negative fixtures for the V9 fetch-record binding gate.

The fixtures contain no observations.  They only verify that the
normalization boundary accepts an exactly bound file and rejects a stale
path, a modified file, and a non-completed fetch record.
"""

import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage425_v9_2026_official_window_download_template import (  # noqa: E402
    validate_fetch_binding,
)

OUT_JSON = ROOT / "tables" / "stage451_v9_fetch_binding_negative_fixture_audit.json"
OUT_REPORT = ROOT / "reports" / "stage451_v9_fetch_binding_negative_fixture_audit.md"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FixturePath:
    """Minimal Path-like object; keeps the fixture entirely in memory."""

    def __init__(self, name: str, payload: bytes):
        self.name = name
        self.payload = payload

    def resolve(self) -> "FixturePath":
        return self

    def stat(self) -> SimpleNamespace:
        return SimpleNamespace(st_size=len(self.payload))

    def open(self, *args, **kwargs):
        return io.BytesIO(self.payload)

    def read_bytes(self) -> bytes:
        return self.payload

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FixturePath) and self.name == other.name


def must_raise(callback) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def main() -> int:
    checks: list[dict[str, object]] = []
    good = FixturePath("meteoswiss_2026_10.csv", b"fixture-only")
    stale = FixturePath("stale_2025.csv", b"fixture-only")
    record_row = {
        "provider": "MeteoSwiss",
        "path": str(good.name),
        "bytes": good.stat().st_size,
        "sha256": file_hash(good),
    }
    rows = {good.resolve(): record_row}
    validate_fetch_binding(
        "MeteoSwiss",
        [good],
        "PASS_V9_OFFICIAL_FETCH_COMPLETED",
        rows,
    )
    checks.append(
        {
            "check": "exact_binding_accepts",
            "status": "PASS",
            "value": True,
        }
    )
    checks.append(
        {
            "check": "stale_or_unlisted_path_rejected",
            "status": "PASS"
            if must_raise(
                lambda: validate_fetch_binding(
                    "MeteoSwiss",
                    [stale],
                    "PASS_V9_OFFICIAL_FETCH_COMPLETED",
                    rows,
                )
            )
            else "FAIL",
            "value": stale.name,
        }
    )
    good.payload = b"modified-fixture"
    checks.append(
        {
            "check": "modified_bytes_or_hash_rejected",
            "status": "PASS"
            if must_raise(
                lambda: validate_fetch_binding(
                    "MeteoSwiss",
                    [good],
                    "PASS_V9_OFFICIAL_FETCH_COMPLETED",
                    rows,
                )
            )
            else "FAIL",
            "value": "byte-count/SHA256 mismatch",
        }
    )
    checks.append(
        {
            "check": "non_completed_fetch_rejected",
            "status": "PASS"
            if must_raise(
                lambda: validate_fetch_binding(
                    "MeteoSwiss",
                    [good],
                    "BLOCKED_V9_FETCH_BEFORE_WINDOW_CLOSE",
                    rows,
                )
            )
            else "FAIL",
            "value": "BLOCKED_V9_FETCH_BEFORE_WINDOW_CLOSE",
        }
    )

    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_FETCH_BINDING_NEGATIVE_FIXTURE_AUDIT"
        if not failures
        else "FAIL_V9_FETCH_BINDING_NEGATIVE_FIXTURE_AUDIT"
    )
    payload = {
        "stage": 451,
        "status": status,
        "checks": checks,
        "failed_checks": failures,
        "synthetic_fixture_only": True,
        "official_observation_values_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "historical_substitution": False,
        "claim_boundary": [
            "夹具只验证采集记录绑定门禁，不代表官方数据或性能结果。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage451 V9 采集绑定负向夹具审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 使用无观测值临时文件验证正确绑定通过、路径/哈希/状态错误拒绝。",
                "- 夹具不产生性能结论。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 451, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
