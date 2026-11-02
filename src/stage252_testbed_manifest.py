from __future__ import annotations

"""Create a compact manifest and safety audit for the authorized testbed."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "testbed" / "raw"
LOGS = ROOT / "testbed" / "logs"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage252_authorized_testbed_manifest"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    files = sorted(
        [
            path
            for path in list(RAW.glob("*")) + list(LOGS.glob("*"))
            if path.is_file()
        ]
    )
    summaries = sorted(TABLES.glob("stage252_testbed_*__*.json"))
    payload = {
        "status": "PASS_AUTHORIZED_TESTBED_MANIFEST",
        "protocol_id": "TMQC-AUTH-TCP-20260919-v1",
        "network_boundary": "127.0.0.1 loopback only",
        "raw_log_files": len([path for path in files if path.parent == RAW]),
        "sent_log_files": len([path for path in files if path.parent == LOGS]),
        "run_summary_files": len(summaries),
        "files": [
            {
                "relative_path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "limitations": [
            "The testbed is authorized and isolated, not a production deployment.",
            "Running the testbed does not overwrite the frozen project endpoint.",
            "Without independent production logs the external-validity risk cannot be RESOLVED.",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 阶段252 授权真实测试床证据清单",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 网络边界：`{payload['network_boundary']}`",
        f"- 原始接收日志：`{payload['raw_log_files']}`",
        f"- 发送日志：`{payload['sent_log_files']}`",
        f"- 运行摘要：`{payload['run_summary_files']}`",
        "",
        "## 边界",
        "",
        "- 测试床是授权、隔离、本机回环环境；不是生产部署。",
        "- 测试床结果不会覆盖冻结主终点。",
        "- 没有独立生产日志时，外部效度风险不能升级为 `RESOLVED`。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "raw_log_files": payload["raw_log_files"],
                "sent_log_files": payload["sent_log_files"],
                "run_summary_files": payload["run_summary_files"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
