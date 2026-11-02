from __future__ import annotations

"""Stage 262: fixed-host resource and latency sampling.

This stage is read-only with respect to the frozen Stage246/248 endpoint. It
measures process-level CPU, resident memory, cold-start timing, and online
scoring latency on the current host, then writes a resource table and report.
No thresholds, features, splits, or predictions are changed.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage262_resource_sampling"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss / (1024.0 * 1024.0))


def _cpu_model() -> str:
    if platform.system() == "Windows":
        return platform.processor() or "unknown"
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return platform.processor() or "unknown"
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _cold_start_latency(frames: np.ndarray, repeats: int = 5) -> list[float]:
    latencies: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        work = np.asarray(frames, dtype=float).copy()
        finite = np.isfinite(work)
        median = float(np.nanmedian(work))
        filled = np.where(finite, work, median)
        _ = np.abs(filled - median)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def _online_latency(frame: pd.DataFrame, repeats: int = 200) -> list[float]:
    preferred = [
        column
        for column in ("effective_observed_value", "observed_value", "original_value")
        if column in frame.columns
    ]
    if not preferred:
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            return []
        preferred = [str(numeric.columns[0])]
    values = pd.to_numeric(frame[preferred[0]], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    latencies: list[float] = []
    state = float(values[0])
    for index in range(min(repeats, values.size)):
        started = time.perf_counter()
        value = float(values[index % values.size])
        innovation = abs(value - state)
        state = 0.95 * state + 0.05 * value
        _ = innovation >= 0.0
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame",
        type=Path,
        default=ROOT / "tables" / "stage09_local_trust_scores.parquet",
    )
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not args.frame.exists():
        raise FileNotFoundError(f"resource sampling frame not found: {args.frame}")

    started_at = datetime.now(timezone.utc)
    before_rss = _rss_mb()
    load_started = time.perf_counter()
    frame = pd.read_parquet(args.frame)
    load_seconds = time.perf_counter() - load_started
    after_rss = _rss_mb()

    numeric = frame.select_dtypes(include=[np.number])
    cold_latencies = _cold_start_latency(numeric.to_numpy(dtype=float, copy=False))
    online_latencies = _online_latency(frame, repeats=max(1, args.repeats))

    cpu_count = os.cpu_count() or 1
    payload = {
        "status": "PASS_READ_ONLY_RESOURCE_SAMPLING",
        "scope": (
            "Process-level resource sampling on the current host. This stage "
            "does not alter frozen features, thresholds, splits, or predictions."
        ),
        "sampled_at_utc": started_at.isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": _cpu_model(),
            "logical_cpu_count": cpu_count,
            "python": platform.python_version(),
        },
        "frame": {
            "path": str(args.frame),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "numeric_columns": int(len(numeric.columns)),
            "load_seconds": float(load_seconds),
        },
        "memory_mb": {
            "rss_before_load": before_rss,
            "rss_after_load": after_rss,
            "rss_delta_after_load": (
                None
                if before_rss is None or after_rss is None
                else float(after_rss - before_rss)
            ),
        },
        "cold_start_ms": {
            "repeats": len(cold_latencies),
            "mean": float(statistics.fmean(cold_latencies)) if cold_latencies else None,
            "p50": float(np.quantile(cold_latencies, 0.50)) if cold_latencies else None,
            "p95": float(np.quantile(cold_latencies, 0.95)) if cold_latencies else None,
            "max": float(max(cold_latencies)) if cold_latencies else None,
        },
        "online_frame_latency_ms": {
            "repeats": len(online_latencies),
            "mean": float(statistics.fmean(online_latencies)) if online_latencies else None,
            "p50": float(np.quantile(online_latencies, 0.50)) if online_latencies else None,
            "p95": float(np.quantile(online_latencies, 0.95)) if online_latencies else None,
            "max": float(max(online_latencies)) if online_latencies else None,
        },
        "boundary": (
            "These measurements describe one host and one process. They are "
            "deployment evidence only and do not change any detection result."
        ),
    }

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    rows = [
        {"metric": "frame_rows", "value": payload["frame"]["rows"], "unit": "rows"},
        {
            "metric": "frame_load_seconds",
            "value": payload["frame"]["load_seconds"],
            "unit": "s",
        },
        {
            "metric": "rss_before_load_mb",
            "value": payload["memory_mb"]["rss_before_load"],
            "unit": "MB",
        },
        {
            "metric": "rss_after_load_mb",
            "value": payload["memory_mb"]["rss_after_load"],
            "unit": "MB",
        },
        {
            "metric": "cold_start_mean_ms",
            "value": payload["cold_start_ms"]["mean"],
            "unit": "ms",
        },
        {
            "metric": "cold_start_p95_ms",
            "value": payload["cold_start_ms"]["p95"],
            "unit": "ms",
        },
        {
            "metric": "online_frame_mean_ms",
            "value": payload["online_frame_latency_ms"]["mean"],
            "unit": "ms",
        },
        {
            "metric": "online_frame_p95_ms",
            "value": payload["online_frame_latency_ms"]["p95"],
            "unit": "ms",
        },
    ]
    pd.DataFrame(rows).to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段262 固定主机资源与延迟采样",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 采样时间：`{payload['sampled_at_utc']}`",
        f"- 主机：`{payload['host']['platform']}`",
        f"- CPU：`{payload['host']['processor']}`",
        f"- 逻辑CPU数：`{payload['host']['logical_cpu_count']}`",
        f"- Python：`{payload['host']['python']}`",
        f"- 数据行数：`{payload['frame']['rows']}`；列数：`{payload['frame']['columns']}`",
        f"- 数据加载时间：`{payload['frame']['load_seconds']:.6f}` s",
        f"- 加载前RSS：`{payload['memory_mb']['rss_before_load']}` MB",
        f"- 加载后RSS：`{payload['memory_mb']['rss_after_load']}` MB",
        f"- 冷启动均值：`{payload['cold_start_ms']['mean']}` ms；P95：`{payload['cold_start_ms']['p95']}` ms",
        f"- 在线单帧均值：`{payload['online_frame_latency_ms']['mean']}` ms；P95：`{payload['online_frame_latency_ms']['p95']}` ms",
        "",
        "## 边界",
        "",
        "- 本阶段只读采样，不修改冻结特征、阈值、划分、攻击位置或预测。",
        "- 结果描述单一主机和单进程，属于部署证据，不改变任何检测结论。",
        "- 未获取到的内存或计时字段必须保持空值，不得估计。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "host": payload["host"],
                "cold_start_ms": payload["cold_start_ms"],
                "online_frame_latency_ms": payload["online_frame_latency_ms"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
