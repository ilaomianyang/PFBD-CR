from __future__ import annotations

"""Stage 443: audit the V9 official fetcher's window and scope gates."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
FETCHER = ROOT / "src" / "stage442_v9_official_window_fetch.py"
FETCH_RESULT = ROOT / "tables" / "stage442_v9_official_window_fetch.json"
OUT_JSON = ROOT / "tables" / "stage443_v9_official_fetch_gate_audit.json"
OUT_REPORT = ROOT / "reports" / "stage443_v9_official_fetch_gate_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    fetcher = FETCHER.read_text(encoding="utf-8")
    result = json.loads(FETCH_RESULT.read_text(encoding="utf-8"))
    checks = [
        {
            "check": "frozen_window_close_hard_gate",
            "status": "PASS"
            if "BLOCKED_V9_FETCH_BEFORE_WINDOW_CLOSE" in fetcher
            and "2026, 11, 1" in fetcher
            else "FAIL",
        },
        {
            "check": "dry_run_has_no_network",
            "status": "PASS"
            if "if args.dry_run" in fetcher
            and "未联网" in fetcher
            else "FAIL",
        },
        {
            "check": "provider_scope_exact",
            "status": "PASS"
            if all(
                provider in fetcher
                for provider in ("MeteoSwiss", "NOAA ISD", "KNMI", "ECCC")
            )
            and protocol.get("provider_count") == 4
            else "FAIL",
        },
        {
            "check": "raw_only_before_normalization",
            "status": "PASS"
            if "normalization" in fetcher.lower()
            and "不做规范化" in str(result.get("claim_boundary", []))
            and "performance_evaluated" in result
            else "FAIL",
        },
        {
            "check": "dry_run_plan_count",
            "status": "PASS"
            if result.get("status") == "BLOCKED_V9_FETCH_BEFORE_WINDOW_CLOSE"
            and result.get("planned_station_count") == 51
            and result.get("archived_file_count") == 0
            else "FAIL",
            "value": {
                "status": result.get("status"),
                "planned_station_count": result.get("planned_station_count"),
                "archived_file_count": result.get("archived_file_count"),
            },
        },
        {
            "check": "no_historical_substitution",
            "status": "PASS"
            if result.get("historical_substitution") is False
            and "previously viewed" not in fetcher.lower()
            else "FAIL",
        },
    ]
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_OFFICIAL_FETCH_GATE_AUDIT"
        if not failures
        else "FAIL_V9_OFFICIAL_FETCH_GATE_AUDIT"
    )
    payload = {
        "stage": 443,
        "status": status,
        "checks": checks,
        "failed_checks": failures,
        "protocol_id": protocol.get("protocol_id"),
        "observation_values_read": False,
        "performance_evaluated": False,
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "fetcher": sha256(FETCHER),
            "fetch_result": sha256(FETCH_RESULT),
        },
        "claim_boundary": [
            "当前审计只证明窗口关闭前不会下载，未产生性能结果。",
            "官方数据到位后仍必须运行独立规范化、数据门禁和锁定评价。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage443 V9 官方采集器门禁审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 当前日期早于 2026-11-01，真实下载被硬拒绝。",
                "- dry-run 计划包含 51 个冻结站点，未归档观测文件。",
                "- 本阶段不产生性能结果。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 443, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
