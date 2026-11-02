from __future__ import annotations

"""Stage 290: resource budgets for the frozen same-contract baseline set.

This stage is resource-only.  It reconstructs the frozen train-normal
feature matrix for one frozen provider, fits each final same-contract
baseline with its frozen configuration in an isolated child process, and
records fit time, peak working set, parameter/model size, score latency,
and failure state.

It never prepares attack cases, reads valid/test labels, computes a
detection metric, changes a threshold, or replaces a frozen result.
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
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage290_same_contract_baseline_resource_budget"
PROTOCOL_ID = "TMQC-SAME-CONTRACT-BASELINE-RESOURCE-20260921-v1.0"
DEFAULT_SOURCE = "MeteoSwiss"
DEFAULT_REPEATS = 5
DEFAULT_TIMEOUT_SECONDS = 900

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
import stage254_ocsvm_training_scale_sensitivity as s254  # noqa: E402
import stage256_lightweight_strong_baselines as s256  # noqa: E402
import stage262_security_domain_baselines as s262  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    detector_frame,
    fit_detectors,
    score_components,
    sha256_file,
)
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_features,
)


def _load(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def _prepare_train_matrix(source: str) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    protocol = _load(Path(s248.PROTOCOLS[source]).stem)
    frame, _ = s254.cached_frame(source, protocol)
    base = detector_frame(frame)
    train_normal = (
        base["split"].eq("train") & base["observation_present"]
    ).to_numpy(dtype=bool)
    components = score_components(base, fit_detectors(base, protocol))
    features = candidate_features(base, train_normal)
    matrix, _ = s248.feature_matrix(components, features, train_normal)
    return protocol, base, matrix, train_normal


def _method_catalog() -> list[dict[str, object]]:
    return [
        {
            "method": "robust_max_deviation",
            "family": "statistical",
            "implementation": "deterministic_max_positive_robust_deviation",
            "fit_rows": "all_train_normal",
        },
        {
            "method": "isolation_forest",
            "family": "classical_unsupervised",
            "implementation": "sklearn.ensemble.IsolationForest",
            "fit_rows": "all_train_normal",
        },
        {
            "method": "one_class_svm",
            "family": "classical_unsupervised_security",
            "implementation": "StandardScaler + sklearn.svm.OneClassSVM",
            "fit_rows": "frozen_stage248_subsample",
        },
        {
            "method": "pca_reconstruction",
            "family": "reconstruction",
            "implementation": "StandardScaler + sklearn.decomposition.PCA",
            "fit_rows": "frozen_stage256_subsample",
        },
        {
            "method": "mlp_autoencoder_reconstruction",
            "family": "reconstruction",
            "implementation": "StandardScaler + sklearn.neural_network.MLPRegressor",
            "fit_rows": "frozen_stage256_subsample",
        },
        {
            "method": "knn_distance",
            "family": "distance_reconstruction",
            "implementation": "StandardScaler + sklearn.neighbors.NearestNeighbors",
            "fit_rows": "frozen_stage256_core_set",
        },
        {
            "method": "ewma_deviation",
            "family": "statistical_process_control",
            "implementation": "station-wise EWMA deviation",
            "fit_rows": "all_train_normal",
        },
        {
            "method": "arima_innovation",
            "family": "time_series_prediction",
            "implementation": "station-wise AR(4) innovation proxy",
            "fit_rows": "all_train_normal",
        },
        {
            "method": "gru_predictor",
            "family": "sequence_prediction_proxy",
            "implementation": "fixed random gated recurrent reservoir with linear readout",
            "fit_rows": "all_train_normal",
        },
        {
            "method": "lightweight_autoencoder",
            "family": "reconstruction",
            "implementation": "StandardScaler + PCA autoencoder",
            "fit_rows": "frozen_stage262_subsample",
        },
        {
            "method": "ics_cps_spc_proxy",
            "family": "classical_security_control",
            "implementation": "station-wise robust SPC deviation proxy",
            "fit_rows": "all_train_normal",
        },
    ]


def _parameter_count(method: str, matrix: np.ndarray, fitted: object) -> int | str:
    feature_count = int(matrix.shape[1])
    if method == "robust_max_deviation":
        return 2 * feature_count
    if method == "isolation_forest":
        return int(fitted.n_estimators)
    if method == "one_class_svm":
        return int(np.asarray(fitted[-1].support_).size)
    if method == "pca_reconstruction":
        pca = fitted.named_steps["pca"]
        return int(pca.components_.size + pca.mean_.size)
    if method == "mlp_autoencoder_reconstruction":
        mlp = fitted.named_steps["mlpregressor"]
        return int(
            sum(
                np.asarray(value).size
                for value in mlp.coefs_
            )
            + sum(np.asarray(value).size for value in mlp.intercepts_)
        )
    if method == "knn_distance":
        return int(fitted.named_steps["nearestneighbors"]._fit_X.size)
    if method == "ewma_deviation":
        return 1
    if method == "arima_innovation":
        return int(s262.AR_ORDER)
    if method == "gru_predictor":
        return int(
            s262.GRU_HIDDEN * (1 + s262.GRU_HIDDEN) * 3
            + s262.GRU_HIDDEN * 3
            + s262.GRU_HIDDEN
            + 1
        )
    if method == "lightweight_autoencoder":
        pca = fitted["ae"].named_steps["pca"]
        return int(pca.components_.size + pca.mean_.size)
    if method == "ics_cps_spc_proxy":
        return 2
    return "UNAVAILABLE"


def _fit_and_measure(method: str, source: str, repeats: int) -> dict[str, object]:
    _, base, matrix, train_normal = _prepare_train_matrix(source)
    fit_indices = s248.stratified_fit_indices(
        base,
        train_normal,
        s248.MAX_OCSVM_FIT_ROWS,
        s248.BASELINE_FIT_SEED,
    )
    if method == "one_class_svm":
        fit_x = np.asarray(matrix[fit_indices], dtype=float)
    elif method in {
        "pca_reconstruction",
        "mlp_autoencoder_reconstruction",
        "lightweight_autoencoder",
    }:
        fit_x = np.asarray(
            matrix[
                s248.stratified_fit_indices(
                    base,
                    train_normal,
                    s256.MAX_FIT_ROWS,
                    s256.MODEL_SEED,
                )
            ],
            dtype=float,
        )
    elif method == "knn_distance":
        fit_x = np.asarray(
            matrix[
                s248.stratified_fit_indices(
                    base,
                    train_normal,
                    s256.KNN_MAX_FIT_ROWS,
                    s256.MODEL_SEED,
                )
            ],
            dtype=float,
        )
    else:
        fit_x = np.asarray(matrix[train_normal], dtype=float)

    _reset_peak_working_set()
    started = time.perf_counter()
    if method == "robust_max_deviation":
        fitted: object = {
            "kind": "maximum_positive_robust_deviation",
        }
        score = lambda: np.max(matrix, axis=1)
    elif method == "isolation_forest":
        fitted = IsolationForest(
            n_estimators=300,
            random_state=s248.BASELINE_FIT_SEED,
            n_jobs=1,
            contamination="auto",
        ).fit(fit_x)
        score = lambda: fitted.score_samples(matrix)
    elif method == "one_class_svm":
        fitted = make_pipeline(
            StandardScaler(),
            OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=0.05,
                cache_size=1000,
                max_iter=-1,
            ),
        ).fit(fit_x)
        score = lambda: fitted.decision_function(matrix)
    elif method == "pca_reconstruction":
        fitted = make_pipeline(
            StandardScaler(),
            PCA(
                n_components=min(s256.PCA_COMPONENTS, fit_x.shape[1]),
                svd_solver="randomized",
                random_state=s256.MODEL_SEED,
            ),
        ).fit(fit_x)

        def score() -> np.ndarray:
            scaled = fitted.named_steps["standardscaler"].transform(matrix)
            reconstructed = fitted.named_steps["pca"].inverse_transform(
                fitted.named_steps["pca"].transform(scaled)
            )
            return np.mean((scaled - reconstructed) ** 2, axis=1)
    elif method == "mlp_autoencoder_reconstruction":
        fitted = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=s256.MLP_HIDDEN_LAYERS,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=s256.MLP_MAX_ITER,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=s256.MODEL_SEED,
            ),
        ).fit(fit_x, fit_x)

        def score() -> np.ndarray:
            scaled = fitted.named_steps["standardscaler"].transform(matrix)
            reconstructed = fitted.named_steps["mlpregressor"].predict(scaled)
            return np.mean((scaled - reconstructed) ** 2, axis=1)
    elif method == "knn_distance":
        fitted = make_pipeline(
            StandardScaler(),
            NearestNeighbors(
                n_neighbors=min(s256.KNN_NEIGHBOURS, len(fit_x)),
                n_jobs=-1,
            ),
        ).fit(fit_x)

        def score() -> np.ndarray:
            scaled = fitted.named_steps["standardscaler"].transform(matrix)
            return np.mean(
                fitted.named_steps["nearestneighbors"].kneighbors(
                    scaled,
                    return_distance=True,
                )[0],
                axis=1,
            )
    elif method == "ewma_deviation":
        fitted, _ = s262.fit_security_baselines(base, matrix, train_normal)
        score = lambda: s262.score_security_baselines(fitted, base, matrix)[method]
    elif method == "arima_innovation":
        fitted, _ = s262.fit_security_baselines(base, matrix, train_normal)
        score = lambda: s262.score_security_baselines(fitted, base, matrix)[method]
    elif method == "gru_predictor":
        fitted, _ = s262.fit_security_baselines(base, matrix, train_normal)
        score = lambda: s262.score_security_baselines(fitted, base, matrix)[method]
    elif method == "lightweight_autoencoder":
        fitted, _ = s262.fit_security_baselines(base, matrix, train_normal)
        score = lambda: s262.score_security_baselines(fitted, base, matrix)[method]
    elif method == "ics_cps_spc_proxy":
        fitted, _ = s262.fit_security_baselines(base, matrix, train_normal)
        score = lambda: s262.score_security_baselines(fitted, base, matrix)[method]
    else:
        raise ValueError(f"unknown method: {method}")
    fit_seconds = time.perf_counter() - started
    peak_working_set_mb = _peak_working_set_mb()
    latency = _timed_call(score, repeats)
    return {
        "status": "PASS",
        "source": source,
        "method": method,
        "fit_rows": int(len(fit_x)),
        "train_normal_rows": int(train_normal.sum()),
        "feature_count": int(matrix.shape[1]),
        "parameter_count": _parameter_count(method, matrix, fitted),
        "fit_seconds": float(fit_seconds),
        "peak_working_set_mb": peak_working_set_mb,
        "score_latency": latency,
        "failure_state": "completed",
        "evidence_scope": "fit_and_score_train_normal_matrix_only_no_valid_no_test",
    }


def _child(method: str, source: str, repeats: int) -> int:
    try:
        result = _fit_and_measure(method, source, repeats)
    except Exception as error:  # noqa: BLE001
        result = {
            "status": "FAILED",
            "source": source,
            "method": method,
            "failure_state": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "evidence_scope": "fit_and_score_train_normal_matrix_only_no_valid_no_test",
        }
    print(
        json.dumps(_json_safe(result), ensure_ascii=False, allow_nan=False),
        flush=True,
    )
    return 0 if result["status"] == "PASS" else 1


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


def _run_child(
    method: str,
    source: str,
    repeats: int,
    timeout_seconds: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-B",
        "-u",
        str(Path(__file__).resolve()),
        "--internal-method",
        method,
        "--source",
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
            "method": method,
            "failure_state": "timeout",
            "timeout_seconds": int(timeout_seconds),
            "evidence_scope": "fit_and_score_train_normal_matrix_only_no_valid_no_test",
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
            "method": method,
            "failure_state": "missing_json_result",
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-2000:],
            "evidence_scope": "fit_and_score_train_normal_matrix_only_no_valid_no_test",
        }
    if completed.returncode != 0:
        payload["returncode"] = int(completed.returncode)
        payload["stderr_tail"] = completed.stderr[-2000:]
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--internal-method",
        choices=[row["method"] for row in _method_catalog()],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.internal_method:
        return _child(
            str(args.internal_method),
            str(args.source),
            max(1, int(args.repeats)),
        )
    if args.source not in s248.PROTOCOLS:
        parser.error(f"source must be one of {sorted(s248.PROTOCOLS)}")
    catalog = _method_catalog()
    rows = []
    for entry in catalog:
        method = str(entry["method"])
        print(
            json.dumps(
                {"event": "resource_measurement_started", "method": method},
                ensure_ascii=False,
            ),
            flush=True,
        )
        measured = _run_child(
            method,
            str(args.source),
            max(1, int(args.repeats)),
            max(1, int(args.timeout_seconds)),
        )
        rows.append({**entry, **measured})
        print(
            json.dumps(
                {
                    "event": "resource_measurement_completed",
                    "method": method,
                    "status": measured["status"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    failures = frame[frame["status"].ne("PASS")]
    payload = {
        "status": (
            "PASS_SAME_CONTRACT_RESOURCE_BUDGET_COMPLETE"
            if failures.empty
            else "PARTIAL_SAME_CONTRACT_RESOURCE_BUDGET_WITH_FAILURES"
        ),
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "Stage290 reconstructs the frozen train-normal feature matrix and "
            "fits each final same-contract baseline in an isolated child "
            "process. It does not prepare attack cases, read valid/test "
            "labels, compute detection metrics, or replace frozen results."
        ),
        "source": str(args.source),
        "repeats": int(max(1, int(args.repeats))),
        "method_count": int(len(frame)),
        "completed_method_count": int(frame["status"].eq("PASS").sum()),
        "failed_method_count": int(len(failures)),
        "all_methods_have_failure_state": bool(
            frame["failure_state"].notna().all()
        ),
        "methods": _json_safe(frame.to_dict("records")),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                Path(s248.PROTOCOLS[str(args.source)]),
                s254.NORMALIZED[str(args.source)],
                s254.AUDITS[str(args.source)],
                TABLES / "stage248_cross_provider_fair_baseline_audit.json",
                TABLES / "stage256_lightweight_strong_baselines.json",
                TABLES / "stage262_security_domain_baselines.json",
            ]
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "limitations": [
            "Resource evidence only; it does not change or replace any detection result.",
            "Only train-normal rows are used for fitting and score-latency measurement; valid/test labels are not read.",
            "Peak working set is measured per isolated child process on the current host.",
            "The one_class_svm row uses the frozen Stage248 fit budget; the Stage274 full-scale resource audit remains separate.",
            "Failure states are retained rather than imputed.",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    frame.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage 290 同契约最终基线资源预算",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 数据源：`{payload['source']}`",
        f"- 方法数：`{payload['method_count']}`",
        f"- 完成：`{payload['completed_method_count']}`",
        f"- 失败：`{payload['failed_method_count']}`",
        "",
        "| 方法 | 拟合行数 | 参数量/模型规模 | 拟合秒 | 峰值工作集MB | 评分均值ms | 失败状态 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.to_dict("records"):
        latency = row.get("score_latency")
        latency_mean = (
            latency.get("mean_ms")
            if isinstance(latency, dict)
            else None
        )
        lines.append(
            f"| {row['method']} | {row.get('fit_rows', 'UNAVAILABLE')} | "
            f"{row.get('parameter_count', 'UNAVAILABLE')} | "
            f"{row.get('fit_seconds', 'UNAVAILABLE')} | "
            f"{row.get('peak_working_set_mb', 'UNAVAILABLE')} | "
            f"{latency_mean if latency_mean is not None else 'UNAVAILABLE'} | "
            f"{row.get('failure_state', 'UNAVAILABLE')} |"
        )
    lines += [
        "",
        "## 边界",
        "",
        "- 本阶段只测资源，不产生或替换任何检测性能结果。",
        "- 只使用 train-normal 行拟合和评分延迟；不读取 valid/test 标签。",
        "- 失败状态保留，不用估算值填补。",
        "- One-Class SVM 此处使用冻结 Stage248 预算；Stage274 的全量预算证据保持独立。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "method_count": payload["method_count"],
                "completed_method_count": payload["completed_method_count"],
                "failed_method_count": payload["failed_method_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if failures.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
