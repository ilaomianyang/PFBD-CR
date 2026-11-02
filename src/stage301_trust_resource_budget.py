from __future__ import annotations

"""Stage 301: isolated train-only resource budget for the frozen Trust rule.

This stage measures the deterministic proposed detector without changing any
frozen result. It reconstructs only the frozen train-normal feature matrix,
then measures:

* detector-component fitting time;
* candidate-feature construction time;
* peak working set of the isolated child process;
* frozen-rule scoring latency and throughput;
* single-station scoring latency and throughput;
* frozen configuration size and component/model size.

It never prepares attack cases, reads valid/test labels, calibrates a
threshold, computes a detection metric, or replaces a frozen prediction.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage301_trust_resource_budget"
PROTOCOL_ID = "TMQC-TRUST-RESOURCE-BUDGET-20260921-v1.0"
DEFAULT_SOURCE = "MeteoSwiss"
DEFAULT_REPEATS = 5
DEFAULT_TIMEOUT_SECONDS = 900
EVIDENCE_SCOPE = "fit_and_score_train_normal_matrix_only_no_valid_no_test"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
import stage254_ocsvm_training_scale_sensitivity as s254  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    detector_frame,
    fit_detectors,
    score_components,
    sha256_file,
)
from stage217_swiss_controlled_attack_evaluation import fuse_scores  # noqa: E402
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_features,
)


def _load_json(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _peak_working_set_mb() -> float | None:
    if os.name == "nt":
        import ctypes

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
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def _reset_peak_working_set() -> bool:
    if os.name != "nt":
        return False
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessWorkingSetSize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    kernel32.SetProcessWorkingSetSize.restype = ctypes.c_int
    return bool(
        kernel32.SetProcessWorkingSetSize(
            kernel32.GetCurrentProcess(),
            ctypes.c_size_t(-1).value,
            ctypes.c_size_t(-1).value,
        )
    )


def _timed_call(function, repeats: int) -> dict[str, object]:
    milliseconds: list[float] = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        function()
        milliseconds.append((time.perf_counter() - started) * 1000.0)
    return {
        "samples": len(milliseconds),
        "mean_ms": float(statistics.fmean(milliseconds)),
        "p50_ms": float(np.quantile(milliseconds, 0.50)),
        "p95_ms": float(np.quantile(milliseconds, 0.95)),
        "max_ms": float(max(milliseconds)),
    }


def _trust_selection(source: str) -> list[dict[str, object]]:
    selection = _load_csv(
        "stage248_cross_provider_fair_baseline_audit_selection.csv"
    )
    current = selection.loc[selection["source"].astype(str).eq(source)]
    output: list[dict[str, object]] = []
    for row in current.to_dict("records"):
        output.append(
            {
                "source": str(row["source"]),
                "seed": int(row["seed"]),
                "method": str(row["method"]),
                "rule": str(row["rule"]),
                "feature": str(row["feature"]),
                "window": int(row["window"]),
                "weight": float(row["weight"]),
                "threshold": float(row["threshold"]),
            }
        )
    return output


def _trust_configuration_count() -> int:
    selection = _load_csv(
        "stage248_cross_provider_fair_baseline_audit_selection.csv"
    )
    methods = {"trust_valid_f1", "trust_nonstuck_event_macro"}
    frozen = selection.loc[selection["method"].isin(methods)]
    if len(frozen) != 40:
        raise RuntimeError(
            f"expected 40 frozen Trust rows, got {len(frozen)}"
        )
    if frozen["source"].nunique() != 4:
        raise RuntimeError("expected four frozen Trust providers")
    return int(len(frozen) * 3 + 3 + 3)


def _prepare_train_data(
    source: str,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    protocol = _load_json(Path(s248.PROTOCOLS[source]).stem)
    frame, _ = s254.cached_frame(source, protocol)
    frame = frame.loc[
        frame["split"].eq("train") & frame["observation_present"]
    ].copy()
    base = detector_frame(frame)
    train_normal = base["observation_present"].to_numpy(dtype=bool)
    detectors = fit_detectors(base, protocol)
    components = score_components(base, detectors)
    base_fused, _ = fuse_scores(base, components)
    base_z = s248.robust_standardize(
        base_fused,
        pd.Series(train_normal, index=base.index),
    ).to_numpy(dtype=float)
    features = candidate_features(base, train_normal)
    matrix, feature_names = s248.feature_matrix(
        components,
        features,
        train_normal,
    )
    return (
        protocol,
        base,
        np.asarray(matrix, dtype=float),
        base_z,
        feature_names,
    )


def _score_base(
    base: pd.DataFrame,
    fitted: dict[str, object],
) -> np.ndarray:
    components = score_components(base, fitted)
    fused, _ = fuse_scores(base, components)
    train_normal = base["observation_present"].to_numpy(dtype=bool)
    return s248.robust_standardize(
        fused,
        pd.Series(train_normal, index=base.index),
    ).to_numpy(dtype=float)


def _score_rule(
    base_z: np.ndarray,
    matrix: np.ndarray,
    feature_index: int,
    weight: float,
) -> np.ndarray:
    return (
        (1.0 - float(weight)) * base_z
        + float(weight) * np.asarray(matrix[:, feature_index], dtype=float)
    )


def _measure(source: str, repeats: int) -> dict[str, object]:
    started = time.perf_counter()
    protocol, base, matrix, base_z, feature_names = _prepare_train_data(source)
    preparation_seconds = time.perf_counter() - started

    if matrix.shape[1] != 16:
        raise RuntimeError(
            f"expected 16 frozen Trust features, got {matrix.shape[1]}"
        )
    train_normal = base["observation_present"].to_numpy(dtype=bool)
    if not train_normal.any():
        raise RuntimeError("no train-normal rows")
    if len(base) != int(train_normal.sum()):
        raise RuntimeError(
            "train-only frame contains rows without observations"
        )

    selected_rows = _trust_selection(source)
    method_names = ("trust_valid_f1", "trust_nonstuck_event_macro")
    for method in method_names:
        count = sum(
            int(row["method"] == method) for row in selected_rows
        )
        if count != 5:
            raise RuntimeError(
                f"expected 5 frozen {method} rows for {source}"
            )
    if len(selected_rows) != 10:
        raise RuntimeError(
            f"expected 10 frozen Trust rows for {source}, "
            f"got {len(selected_rows)}"
        )

    base_scores = np.asarray(base_z, dtype=float)
    if not np.isfinite(base_scores).all():
        raise RuntimeError("non-finite Trust base scores")

    station_ids = base["station_id"].astype(str).to_numpy()
    single_station_id = sorted(pd.unique(station_ids))[0]
    single_station_mask = station_ids == single_station_id
    if not single_station_mask.any():
        raise RuntimeError("single-station subset is empty")

    _reset_peak_working_set()
    fit_started = time.perf_counter()
    fitted_components = fit_detectors(base, protocol)
    fit_seconds = time.perf_counter() - fit_started
    if not fitted_components:
        raise RuntimeError("Trust component fit returned no detectors")

    component_latency = _timed_call(
        lambda: _score_base(base, fitted_components),
        repeats,
    )
    frozen_base_z = np.asarray(
        _score_base(base, fitted_components),
        dtype=float,
    )
    if not np.isfinite(frozen_base_z).all():
        raise RuntimeError("non-finite frozen Trust base scores")
    per_selection_rows: list[dict[str, object]] = []
    for row in selected_rows:
        feature_name = (
            f"{row['feature']}_w{int(row['window'])}"
        )
        if feature_name not in feature_names:
            raise RuntimeError(
                f"frozen feature {feature_name} is absent from the matrix"
            )
        feature_index = int(feature_names.index(feature_name))
        weight = float(row["weight"])
        latency = _timed_call(
            lambda: _score_rule(
                frozen_base_z,
                matrix,
                feature_index,
                weight,
            ),
            repeats,
        )
        station_latency = _timed_call(
            lambda: _score_rule(
                frozen_base_z[single_station_mask],
                np.asarray(matrix[single_station_mask], dtype=float),
                feature_index,
                weight,
            ),
            repeats,
        )
        per_selection_rows.append(
            {
                **row,
                "feature_name": feature_name,
                "end_to_end_latency": latency,
                "single_station_latency": station_latency,
            }
        )
    peak_working_set_mb = _peak_working_set_mb()
    end_to_end_means = [
        float(row["end_to_end_latency"]["mean_ms"])
        for row in per_selection_rows
    ]
    station_means = [
        float(row["single_station_latency"]["mean_ms"])
        for row in per_selection_rows
    ]
    mean_ms = float(statistics.fmean(end_to_end_means))
    station_mean_ms = float(statistics.fmean(station_means))
    throughput_fps = (
        float(len(base_z) * 1000.0 / mean_ms) if mean_ms > 0.0 else None
    )
    single_station_throughput_fps = (
        float(int(single_station_mask.sum()) * 1000.0 / station_mean_ms)
        if station_mean_ms > 0.0
        else None
    )

    return {
        "status": "PASS",
        "source": source,
        "method": "trust_frozen_rule",
        "fit_rows": int(train_normal.sum()),
        "rows_scored": int(len(base_z)),
        "feature_count": int(matrix.shape[1]),
        "component_count": 3,
        "component_weights": {
            "Hampel MAD": 1.0 / 3.0,
            "Kalman Innovation": 1.0 / 3.0,
            "Spatial Consistency": 1.0 / 3.0,
        },
        "configuration_parameter_count": _trust_configuration_count(),
        "configuration_parameter_scope": (
            "frozen selection rows x (window, weight, threshold) plus "
            "three fixed components and three fixed component weights; "
            "not a neural-network parameter count"
        ),
        "selection_rows": int(len(selected_rows)),
        "selection": per_selection_rows,
        "preparation_seconds": float(preparation_seconds),
        "component_fit_seconds": float(fit_seconds),
        "component_score_latency": component_latency,
        "peak_working_set_mb": peak_working_set_mb,
        "score_latency": {
            "samples": int(len(end_to_end_means)),
            "mean_ms": mean_ms,
            "p50_ms": float(np.quantile(end_to_end_means, 0.50)),
            "p95_ms": float(np.quantile(end_to_end_means, 0.95)),
            "max_ms": float(max(end_to_end_means)),
        },
        "score_throughput_rows_per_second": throughput_fps,
        "single_station_id": single_station_id,
        "single_station_rows": int(single_station_mask.sum()),
        "single_station_score_latency": {
            "samples": int(len(station_means)),
            "mean_ms": station_mean_ms,
            "p50_ms": float(np.quantile(station_means, 0.50)),
            "p95_ms": float(np.quantile(station_means, 0.95)),
            "max_ms": float(max(station_means)),
        },
        "single_station_throughput_rows_per_second": (
            single_station_throughput_fps
        ),
        "online_causal": True,
        "gpu_required": False,
        "station_recalibration_required": (
            "Yes: target-valid calibration is part of the frozen transfer rule."
        ),
        "failure_state": "completed",
        "evidence_scope": EVIDENCE_SCOPE,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _child(source: str, repeats: int) -> int:
    try:
        result = _measure(source, repeats)
    except Exception as error:  # noqa: BLE001
        result = {
            "status": "FAILED",
            "source": source,
            "method": "trust_frozen_rule",
            "failure_state": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "evidence_scope": EVIDENCE_SCOPE,
        }
    print(
        json.dumps(_json_safe(result), ensure_ascii=False, allow_nan=False),
        flush=True,
    )
    return 0 if result["status"] == "PASS" else 1


def _run_child(
    source: str,
    repeats: int,
    timeout_seconds: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-B",
        "-u",
        str(Path(__file__).resolve()),
        "--internal-source",
        source,
        "--repeats",
        str(max(1, int(repeats))),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(1, int(timeout_seconds)),
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "FAILED",
            "source": source,
            "method": "trust_frozen_rule",
            "failure_state": "timeout",
            "timeout_seconds": int(timeout_seconds),
            "evidence_scope": EVIDENCE_SCOPE,
        }
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
    if payload is None:
        payload = {
            "status": "FAILED",
            "source": source,
            "method": "trust_frozen_rule",
            "failure_state": "missing_json_result",
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-2000:],
            "evidence_scope": EVIDENCE_SCOPE,
        }
    if completed.returncode != 0:
        payload["returncode"] = int(completed.returncode)
        payload["stderr_tail"] = completed.stderr[-2000:]
    return payload


def _platform() -> dict[str, object]:
    import platform

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--internal-source",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.internal_source:
        return _child(
            str(args.internal_source),
            max(1, int(args.repeats)),
        )
    if args.source not in s248.PROTOCOLS:
        parser.error(f"source must be one of {sorted(s248.PROTOCOLS)}")

    print(
        json.dumps(
            {
                "event": "trust_resource_measurement_started",
                "source": str(args.source),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    result = _run_child(
        str(args.source),
        max(1, int(args.repeats)),
        max(1, int(args.timeout_seconds)),
    )
    print(
        json.dumps(
            {
                "event": "trust_resource_measurement_completed",
                "status": result["status"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    passed = result["status"] == "PASS"
    payload = {
        "status": (
            "PASS_TRUST_RESOURCE_BUDGET_COMPLETE"
            if passed
            else "FAILED_TRUST_RESOURCE_BUDGET"
        ),
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "Stage301 reconstructs only the frozen train-normal feature "
            "matrix and measures the deterministic Trust rule in an isolated "
            "child process. It does not prepare attacks, read valid/test "
            "labels, calibrate a threshold, compute a detection metric, or "
            "replace a frozen prediction."
        ),
        "source": str(args.source),
        "repeats": int(max(1, int(args.repeats))),
        "method_count": 1,
        "completed_method_count": int(passed),
        "failed_method_count": int(not passed),
        "host": _platform(),
        "measurement": result,
        "trust_resource_fields_complete": bool(passed),
        "resource_completion_boundary": (
            "Frozen-rule fit, score latency, throughput, peak working set, "
            "configuration count, causal status, GPU requirement, and "
            "station-recalibration requirement are measured on train-normal "
            "rows only. This is not a detection-performance result."
        ),
        "input_sha256": {
            str(Path(s248.PROTOCOLS[str(args.source)]).relative_to(ROOT)): (
                sha256_file(Path(s248.PROTOCOLS[str(args.source)]))
            ),
            str(s254.NORMALIZED[str(args.source)].relative_to(ROOT)): (
                sha256_file(s254.NORMALIZED[str(args.source)])
            ),
            str(s254.AUDITS[str(args.source)].relative_to(ROOT)): (
                sha256_file(s254.AUDITS[str(args.source)])
            ),
            "tables/stage248_cross_provider_fair_baseline_audit_selection.csv": (
                sha256_file(
                    TABLES
                    / "stage248_cross_provider_fair_baseline_audit_selection.csv"
                )
            ),
            "tables/stage269_deployment_complexity_audit.json": (
                sha256_file(
                    TABLES / "stage269_deployment_complexity_audit.json"
                )
            ),
        },
        "limitations": [
            "Resource evidence only; it does not change or replace any detection result.",
            "Only train-normal rows are used for fitting and score-latency measurement; valid/test labels are not read.",
            "Peak working set is measured per isolated child process on the current host.",
            "Configuration count is not a neural-network parameter count.",
            "The target-valid calibration step is not re-run in this resource-only stage.",
            "Failure states are retained rather than imputed.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame([result]).to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = [
        "# Stage 301 Trust 资源预算",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 数据源：`{payload['source']}`",
        f"- 方法数：`{payload['method_count']}`",
        f"- 完成：`{payload['completed_method_count']}`",
        f"- 失败：`{payload['failed_method_count']}`",
        "",
        "## 实测",
        "",
        (
            f"- 拟合行数：`{result.get('fit_rows', 'UNAVAILABLE')}`；"
            f"评分行数：`{result.get('rows_scored', 'UNAVAILABLE')}`。"
        ),
        (
            f"- 分量拟合秒：`{result.get('component_fit_seconds', 'UNAVAILABLE')}`；"
            "分量评分延迟均值ms："
            f"`{(result.get('component_score_latency') or {}).get('mean_ms', 'UNAVAILABLE')}`。"
        ),
        (
            f"- 峰值工作集MB：`{result.get('peak_working_set_mb', 'UNAVAILABLE')}`。"
        ),
        (
            f"- 全量冻结规则评分均值ms："
            f"`{(result.get('score_latency') or {}).get('mean_ms', 'UNAVAILABLE')}`；"
            f"吞吐行/秒：`{result.get('score_throughput_rows_per_second', 'UNAVAILABLE')}`。"
        ),
        (
            f"- 单站点 `{result.get('single_station_id', 'UNAVAILABLE')}` "
            f"评分均值ms："
            f"`{(result.get('single_station_score_latency') or {}).get('mean_ms', 'UNAVAILABLE')}`；"
            f"吞吐行/秒："
            f"`{result.get('single_station_throughput_rows_per_second', 'UNAVAILABLE')}`。"
        ),
        (
            f"- 配置复杂度参数计数："
            f"`{result.get('configuration_parameter_count', 'UNAVAILABLE')}`；"
            "不是神经网络参数量。"
        ),
        "",
        "## 边界",
        "",
        "- 本阶段只测资源，不产生或替换任何检测性能结果。",
        "- 只使用 train-normal 行拟合和评分延迟；不读取 valid/test 标签。",
        "- 目标域 valid 重校准步骤未在本阶段重跑。",
        "- 失败状态保留，不用估算值填补。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "source": payload["source"],
                "completed_method_count": payload[
                    "completed_method_count"
                ],
                "failed_method_count": payload["failed_method_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
