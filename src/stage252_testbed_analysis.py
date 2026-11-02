from __future__ import annotations

"""Aggregate authorized testbed runs into a compact review artifact.

This analysis reads only the raw receive logs and run summaries produced by
stage252_testbed_runner.py.  It does not train a detector, select a threshold,
or merge testbed observations into the frozen project test set.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "testbed" / "raw"
LOGS = ROOT / "testbed" / "logs"
PREFIX = "stage252_authorized_testbed_analysis"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_summaries() -> list[Path]:
    return sorted(TABLES.glob("stage252_testbed_*__*.json"))


def summarize(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    received_path = ROOT / str(payload.get("received_path", ""))
    sent_path = ROOT / str(payload.get("sent_path", ""))
    received = read_jsonl(received_path)
    sent = read_jsonl(sent_path)
    sequence_counts: dict[int, int] = {}
    raw_hashes: set[str] = set()
    for row in received:
        sequence = int(row["sequence"])
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
        raw_hashes.add(str(row["raw_sha256"]))
    duplicate_sequences = sorted(
        sequence for sequence, count in sequence_counts.items() if count > 1
    )
    sent_sequences = {int(row["sequence"]) for row in sent}
    missing_sequences = sorted(sent_sequences - set(sequence_counts))
    return {
        "run_id": payload["run_id"],
        "mode": payload["mode"],
        "attack_type": payload["attack_type"],
        "seed": int(payload["seed"]),
        "sent_frames": len(sent),
        "received_frames": len(received),
        "unique_received_sequences": len(sequence_counts),
        "duplicate_sequence_count": len(duplicate_sequences),
        "missing_sequence_count": len(missing_sequences),
        "unique_raw_hash_count": len(raw_hashes),
        "complete": bool(payload.get("complete", False)),
        "receive_log": str(received_path.relative_to(ROOT)),
        "sent_log": str(sent_path.relative_to(ROOT)),
        "send_error": payload.get("send_error", []),
        "receive_error": payload.get("receive_error", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    summaries = [summarize(path) for path in run_summaries()]
    if not summaries:
        raise RuntimeError("no stage252 testbed run summaries found")
    frame = pd.DataFrame(summaries).sort_values(
        ["mode", "attack_type", "seed"]
    ).reset_index(drop=True)
    failures = frame.loc[
        ~frame["complete"]
        | frame["send_error"].astype(bool)
        | frame["receive_error"].astype(bool),
        "run_id",
    ].tolist()
    status = (
        "PASS_AUTHORIZED_TESTBED_ANALYSIS"
        if not failures
        else "REVIEW_REQUIRED"
    )
    if failures and not args.allow_partial:
        raise SystemExit(
            json.dumps(
                {
                    "status": status,
                    "failures": failures,
                    "hint": "use --allow-partial to write a partial audit",
                },
                ensure_ascii=False,
            )
        )
    payload = {
        "status": status,
        "protocol": (
            "聚合授权loopback TCP测试床的真实运行结果；只读取运行时"
            "发送/接收日志，不训练检测器、不选择阈值，不修改冻结项目test。"
        ),
        "scope": (
            "攻击执行、协议交换和接收时间均为真实运行时事件；环境为本机"
            "隔离测试床，不是生产环境事故日志。"
        ),
        "run_count": len(frame),
        "clean_run_count": int(frame["mode"].eq("clean").sum()),
        "attack_run_count": int(frame["mode"].eq("attack").sum()),
        "attack_type_count": int(frame.loc[frame["mode"].eq("attack"), "attack_type"].nunique()),
        "seed_count": int(frame["seed"].nunique()),
        "failure_count": len(failures),
        "failures": failures,
        "summary_rows": frame.to_dict(orient="records"),
        "claim_ceiling": (
            "完成并通过本审计后，测试床证据最多将真实攻击外部效度风险"
            "降至PARTIALLY_CLOSED；没有独立生产日志时不能升级为RESOLVED。"
        ),
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in run_summaries()
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    frame.to_csv(
        TABLES / f"{PREFIX}_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段252 授权真实测试床聚合分析",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 运行数：`{payload['run_count']}`；干净运行：`{payload['clean_run_count']}`；攻击运行：`{payload['attack_run_count']}`",
        f"- 攻击类型：`{payload['attack_type_count']}`；种子：`{payload['seed_count']}`",
        f"- 失败运行：`{payload['failure_count']}`",
        "",
        "## 运行结果",
        "",
        "| 运行 | 模式 | 攻击 | 种子 | 发送 | 接收 | 唯一序列 | 重复序列 | 丢失序列 | 完整 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            f"| {row['run_id']} | {row['mode']} | {row['attack_type']} | "
            f"{row['seed']} | {row['sent_frames']} | {row['received_frames']} | "
            f"{row['unique_received_sequences']} | {row['duplicate_sequence_count']} | "
            f"{row['missing_sequence_count']} | {row['complete']} |"
        )
    lines += [
        "",
        "## 声称上限",
        "",
        f"- {payload['claim_ceiling']}",
        "- 测试床证据不能替换Stage246冻结主终点，也不能把探索性变体升级为确认性结论。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "run_count": payload["run_count"],
                "attack_type_count": payload["attack_type_count"],
                "seed_count": payload["seed_count"],
                "failure_count": payload["failure_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
