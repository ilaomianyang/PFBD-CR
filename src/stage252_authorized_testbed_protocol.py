from __future__ import annotations

"""Stage 252: pre-register an authorized, isolated attack testbed.

This stage freezes the testbed threat model and evidence contract before any
testbed result is produced.  It does not train a detector, choose a threshold,
or evaluate the frozen project endpoints.  The testbed uses real local TCP
sockets and a real wire-format protocol so that attack traffic and clean
traffic are produced at runtime rather than by editing a static table.

The contract is deliberately explicit about scope: an authorized testbed is
stronger than numeric injection because the protocol exchange and packet
timing are real, but it is still not a production incident log and cannot
support a claim of global real-world superiority.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage252_authorized_testbed_protocol"

PROTOCOL_ID = "TMQC-AUTH-TCP-20260919-v1"
TRANSPORT = "localhost TCP, loopback only"
MESSAGE_FORMAT = (
    "UTF-8 newline-delimited JSON telemetry frames with station_id, "
    "timestamp_utc, sequence, temperature_c, humidity_pct, wind_mps, "
    "pressure_hpa, nonce, and payload_sha256"
)
CLEAN_WINDOW_SECONDS = 30
ATTACK_WINDOW_SECONDS = 10
REPLAY_WINDOW_SECONDS = 10
STATION_IDS = ["TB01", "TB02", "TB03", "TB04"]
ATTACK_TYPES = [
    "replay",
    "tamper",
    "freeze",
    "delay",
    "drop",
    "coordinated_offset",
]
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FAR_POINT = 0.01
PRIMARY_ENDPOINT = "detection and recovery under authorized testbed attacks"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    protocol = {
        "status": "PROTOCOL_FROZEN_BEFORE_TESTBED_EXECUTION",
        "protocol_id": PROTOCOL_ID,
        "scope": (
            "Authorized, isolated, local-only testbed. The testbed executes "
            "real runtime protocol attacks on loopback sockets owned by the "
            "project. It is not a production system and is not represented "
            "as a real-world incident log."
        ),
        "transport": TRANSPORT,
        "message_format": MESSAGE_FORMAT,
        "network_boundary": (
            "All sockets bind to 127.0.0.1 only; no external host, public "
            "network, or third-party service is contacted."
        ),
        "authorization_boundary": (
            "The testbed is created and attacked by the project on its own "
            "loopback interface. No external target is contacted. The paper "
            "must state that attack execution is real but the environment is "
            "controlled and authorized."
        ),
        "attack_types": ATTACK_TYPES,
        "seeds": SEEDS,
        "clean_window_seconds": CLEAN_WINDOW_SECONDS,
        "attack_window_seconds": ATTACK_WINDOW_SECONDS,
        "replay_window_seconds": REPLAY_WINDOW_SECONDS,
        "station_ids": STATION_IDS,
        "primary_far": FAR_POINT,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "evaluation_boundary": (
            "Testbed evaluation is separate from the frozen project test "
            "set. It may improve the external-validity boundary, but it "
            "cannot overwrite the frozen Stage246 endpoint or promote a new "
            "variant into the existing confirmatory claim."
        ),
        "promotion_policy": {
            "current": "BOUNDED",
            "after_successful_testbed_run": "PARTIALLY_CLOSED",
            "resolved_requires": (
                "Independent production incident or fault logs with "
                "auditable labels and governance."
            ),
        },
        "evidence_required": [
            "frozen protocol file and SHA256",
            "config snapshot and code SHA256",
            "raw wire capture or byte-log with monotonic timestamps",
            "attacker command log and clean-run command log",
            "per-frame receive log and per-attack run summary",
            "detector output on clean and attacked windows",
            "recovery output and safety-boundary metrics",
            "final manifest with no post-hoc threshold changes",
        ],
        "forbidden_claims": [
            "This testbed proves global real-world attack superiority.",
            "This testbed replaces independent production incident logs.",
            "Testbed attacks may be described as third-party attacks.",
            "Testbed results may be merged into the frozen main endpoint without a new lock.",
        ],
        "planned_execution": {
            "clean_runner": "src/stage252_testbed_runner.py --mode clean",
            "attack_runner": "src/stage252_testbed_runner.py --mode attack",
            "analysis": "src/stage252_testbed_analysis.py",
        },
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
    }

    protocol_path = TABLES / f"{PREFIX}.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "protocol_id": PROTOCOL_ID,
                "transport": TRANSPORT,
                "network_boundary": protocol["network_boundary"],
                "attack_types": ",".join(ATTACK_TYPES),
                "seeds": ",".join(str(value) for value in SEEDS),
                "primary_far": FAR_POINT,
                "current_status": "BOUNDED",
                "successful_run_status": "PARTIALLY_CLOSED",
                "resolved_status": "NOT_AVAILABLE_WITHOUT_PRODUCTION_LOGS",
            }
        ]
    ).to_csv(
        TABLES / f"{PREFIX}_contract.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段252 授权真实测试床预注册",
        "",
        f"- 状态：**{protocol['status']}**",
        f"- 协议ID：`{PROTOCOL_ID}`",
        f"- 传输：`{TRANSPORT}`",
        f"- 报文：`{MESSAGE_FORMAT}`",
        f"- 主FAR：`{FAR_POINT}`",
        f"- 攻击类型：`{', '.join(ATTACK_TYPES)}`",
        f"- 随机种子：`{', '.join(str(value) for value in SEEDS)}`",
        "",
        "## 真实边界",
        "",
        "- 攻击在本项目自己的 loopback TCP 连接上运行时执行，使用真实报文和真实时间戳。",
        "- 测试床完全隔离，不接触外部主机、公网或第三方服务。",
        "- 真实的是攻击执行、协议交换和日志；受控的是环境和来源。",
        "- 测试床结果只能把外部效度风险降为 `PARTIALLY_CLOSED`，不能写成生产环境真实攻击。",
        "",
        "## 必须保存的证据",
        "",
        *[f"- {item}" for item in protocol["evidence_required"]],
        "",
        "## 禁止声称",
        "",
        *[f"- {item}" for item in protocol["forbidden_claims"]],
        "",
        "## 当前状态",
        "",
        "- 当前仍为 `BOUNDED`。",
        "- 完成测试床执行并通过证据审计后，最多降为 `PARTIALLY_CLOSED`。",
        "- 没有独立生产日志时，不能升级为 `RESOLVED`。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "protocol_id": PROTOCOL_ID,
                "dry_run": args.dry_run,
                "protocol_sha256": sha256_file(protocol_path),
                "attack_type_count": len(ATTACK_TYPES),
                "seed_count": len(SEEDS),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
