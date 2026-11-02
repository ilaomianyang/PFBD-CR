from __future__ import annotations

"""Stage 419: resource-only audit of the authorized clean testbed path.

Only clean loopback receive logs are read.  Attack logs, attack labels,
project test sets, and thresholds are not accessed.  The result measures
process-local runtime cost; it is not a new detection-performance result.
"""

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage419_testbed_resource_only_audit"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
REPEATS = 3

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage255_online_causal_testbed as stage255  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_working_set_mb() -> float | None:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        ):
            return None
        return float(counters.PeakWorkingSetSize / (1024.0 * 1024.0))
    import resource

    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage / (1024.0 * 1024.0) if sys.platform == "darwin" else usage / 1024.0


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def clean_path(seed: int) -> Path:
    return ROOT / "testbed" / "raw" / f"clean__clean__seed{seed}_received.jsonl"


def measure_seed(seed: int, repeats: int) -> dict[str, Any]:
    path = clean_path(seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    timings: list[float] = []
    frames = 0
    alerts = 0
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        received = stage255.parse_received_logs("clean", seed)
        detected, _ = stage255.online_detect(received, "clean")
        stage255.causal_recovery(detected, stage255.RECOVERY_SIGMA_K)
        timings.append((time.perf_counter() - started) * 1000.0)
        frames = len(received)
        alerts = sum(bool(row["alert"]) for row in detected)
    return {
        "status": "PASS",
        "seed": int(seed),
        "input_path": str(path.relative_to(ROOT)),
        "input_sha256": sha256_file(path),
        "frames": int(frames),
        "alerts": int(alerts),
        "batch_elapsed_ms": {
            "samples": len(timings),
            "mean_ms": float(statistics.fmean(timings)),
            "p50_ms": float(np.quantile(timings, 0.50)),
            "p95_ms": float(np.quantile(timings, 0.95)),
            "max_ms": float(max(timings)),
        },
        "peak_working_set_mb": peak_working_set_mb(),
        "fit_seconds": "NOT_APPLICABLE",
        "parameter_count": "NOT_APPLICABLE",
        "attack_logs_read": False,
        "test_labels_read": False,
        "evidence_scope": (
            "clean loopback receive log; parse + causal detector + recovery "
            "replay only; no attack or project test data"
        ),
    }


def child(seed: int, repeats: int) -> int:
    try:
        result = measure_seed(seed, repeats)
    except Exception as error:  # noqa: BLE001
        result = {
            "status": "FAILED",
            "seed": int(seed),
            "error_type": type(error).__name__,
            "error": str(error),
            "attack_logs_read": False,
            "test_labels_read": False,
        }
    print(json.dumps(json_safe(result), ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


def run_child(seed: int, repeats: int) -> dict[str, Any]:
    command = [
        sys.executable,
        "-B",
        "-u",
        str(Path(__file__).resolve()),
        "--internal-seed",
        str(seed),
        "--repeats",
        str(repeats),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=120,
    )
    payload: dict[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if line.strip().startswith("{"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if payload is None:
        payload = {
            "status": "FAILED",
            "seed": int(seed),
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-2000:],
            "attack_logs_read": False,
            "test_labels_read": False,
        }
    if completed.returncode != 0:
        payload["returncode"] = int(completed.returncode)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-seed", type=int)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.internal_seed is not None:
        return child(int(args.internal_seed), max(1, int(args.repeats)))

    runs = [run_child(seed, max(1, int(args.repeats))) for seed in SEEDS]
    failures = [row for row in runs if row.get("status") != "PASS"]
    payload = {
        "status": "PASS_TESTBED_RESOURCE_ONLY_AUDIT" if not failures else "PARTIAL_TESTBED_RESOURCE_ONLY_AUDIT",
        "protocol": (
            "只使用授权 loopback 测试床的 clean 接收日志，在独立子进程测量"
            "解析、严格因果检测和恢复的运行耗时与峰值工作集；不读取攻击日志、"
            "攻击标签或项目 test，不训练、不调阈值、不改变 Stage255 结果。"
        ),
        "seed_count": len(SEEDS),
        "repeat_count": max(1, int(args.repeats)),
        "run_count": len(runs),
        "completed_count": len(runs) - len(failures),
        "failure_count": len(failures),
        "runs": runs,
        "attack_logs_read": False,
        "test_labels_read": False,
        "performance_status": "NOT_EVALUATED",
        "fit_seconds": "NOT_APPLICABLE",
        "parameter_count": "NOT_APPLICABLE",
        "claim_boundary": [
            "这是授权测试床运行资源证据，不是主检测性能结果。",
            "测试床不是生产环境，不能支持真实事故召回率。",
            "批处理重放耗时不能直接替代生产在线吞吐分布。",
            "失败状态原样保留，不用估算值填补。",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    output = TABLES / f"{PREFIX}.json"
    output.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage419 授权测试床资源只读审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- clean 种子：`{payload['seed_count']}`；每个种子重复：`{payload['repeat_count']}`",
        "- 读取范围：仅 clean loopback 接收日志；攻击日志、攻击标签和项目 test 均未读取。",
        "- 拟合时间/参数量：`NOT_APPLICABLE`；该路径是确定性运行时检测和恢复，不含独立训练阶段。",
        "",
        "| 种子 | 帧数 | 批处理均值(ms) | P95(ms) | 峰值工作集(MB) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        interval = row.get("batch_elapsed_ms", {})
        lines.append(
            f"| {row.get('seed')} | {row.get('frames', 'UNAVAILABLE')} | "
            f"{interval.get('mean_ms', 'UNAVAILABLE')} | "
            f"{interval.get('p95_ms', 'UNAVAILABLE')} | "
            f"{row.get('peak_working_set_mb', 'UNAVAILABLE')} |"
        )
    lines += [
        "",
        "## 边界",
        "",
        "- 本阶段只补充部署资源证据，不替换 Stage255 在线测试床性能或 Stage248 主检测结果。",
        "- 所有结果来自授权隔离 loopback 测试床，不是生产事故日志。",
        "- 不得把批处理重放耗时写成生产端到端延迟。",
        "- 失败状态保留，不用估算值填补。",
        "",
        f"- JSON：`{output}`",
        f"- 报告：`{REPORTS / f'{PREFIX}.md'}`",
    ]
    (REPORTS / f"{PREFIX}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "seed_count": payload["seed_count"],
                "completed_count": payload["completed_count"],
                "failure_count": payload["failure_count"],
                "attack_logs_read": False,
                "test_labels_read": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
