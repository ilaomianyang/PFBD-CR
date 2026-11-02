from __future__ import annotations

"""Stage 269: deployment-complexity and resource-evidence audit.

This stage is read-only with respect to every frozen result and prediction.
It consolidates the model structures, fit budgets, online-causality status,
recalibration requirements, and measured resource evidence that already
exist across Stage248, Stage254, Stage255, Stage256, Stage261, and Stage262.

The script also performs a bounded reconstruction benchmark on one frozen
provider frame.  The benchmark refits the deterministic unscaled baselines
on the exact train-normal subset used by the frozen audit and measures fit
time, scoring latency, and process peak working set.  It does not change a
threshold, split, prediction, or paper endpoint.  Fields that cannot be
measured from frozen artifacts are explicitly ``UNAVAILABLE``.
"""

import argparse
import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage269_deployment_complexity_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
import stage256_lightweight_strong_baselines as s256  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    detector_frame,
    fit_detectors,
    score_components,
    sha256_file,
)
from stage217_swiss_controlled_attack_evaluation import fuse_scores  # noqa: E402
from stage240_nonstuck_balanced_selection import make_frame  # noqa: E402
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_features,
)


def _load_json(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(name: str) -> pd.DataFrame:
    path = TABLES / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_existing_benchmark() -> tuple[dict, dict] | None:
    """Reuse only the previous Stage269 measurement when it is self-consistent."""

    path = TABLES / f"{PREFIX}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmark = payload["benchmark"]
        reconstruction = payload["reconstruction_benchmark"]
        if not isinstance(benchmark, dict) or not isinstance(
            reconstruction, dict
        ):
            return None
        return benchmark, reconstruction
    except (KeyError, TypeError, ValueError):
        return None


def _peak_working_set_bytes() -> int | None:
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
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return None
        return int(counters.PeakWorkingSetSize)
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage * 1024)


def _reset_peak_working_set() -> bool:
    if os.name != "nt":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessWorkingSetSize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    kernel32.SetProcessWorkingSetSize.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    return bool(
        kernel32.SetProcessWorkingSetSize(
            process,
            ctypes.c_size_t(-1).value,
            ctypes.c_size_t(-1).value,
        )
    )


def _interval(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }
    return {
        "samples": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(max(values)),
    }


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        clean: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, float)) and not np.isfinite(value):
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records


def _benchmark_unscaled_baselines(
    source: str,
    repeats: int,
) -> dict[str, object]:
    protocol = json.loads(s248.PROTOCOLS[source].read_text(encoding="utf-8"))
    raw = make_frame(source, protocol)
    base = detector_frame(raw)
    detectors = fit_detectors(base, protocol)
    train_normal = (
        base["split"].eq("train") & base["observation_present"]
    ).to_numpy(dtype=bool)
    components = score_components(base, detectors)
    fused, _ = fuse_scores(base, components)
    base_z = s248.robust_standardize(fused, train_normal).to_numpy(dtype=float)
    features = candidate_features(base, train_normal)
    matrix, feature_names = s248.feature_matrix(
        components,
        features,
        train_normal,
    )
    fit_indices = s248.stratified_fit_indices(
        base,
        train_normal,
        s248.MAX_OCSVM_FIT_ROWS,
        s248.BASELINE_FIT_SEED,
    )
    fit_x = np.asarray(matrix[fit_indices], dtype=float)
    score_x = np.asarray(matrix, dtype=float)
    station_ids = base["station_id"].astype(str).to_numpy()
    single_station_id = sorted(pd.unique(station_ids))[0]
    single_station_x = np.asarray(
        matrix[station_ids == single_station_id],
        dtype=float,
    )
    if single_station_x.shape[0] == 0:
        raise RuntimeError("single-station benchmark subset is empty")

    _reset_peak_working_set()
    started = time.perf_counter()
    isolation = IsolationForest(
        n_estimators=300,
        random_state=s248.BASELINE_FIT_SEED,
        n_jobs=1,
        contamination="auto",
    ).fit(fit_x)
    fit_seconds = time.perf_counter() - started
    peak_bytes = _peak_working_set_bytes()

    scoring_ms: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        _ = isolation.score_samples(score_x)
        scoring_ms.append((time.perf_counter() - started) * 1000.0)
    single_station_ms: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        _ = isolation.score_samples(single_station_x)
        single_station_ms.append(
            (time.perf_counter() - started) * 1000.0
        )

    return {
        "source": source,
        "protocol_path": str(s248.PROTOCOLS[source]),
        "rows_total": int(len(base)),
        "train_normal_rows": int(train_normal.sum()),
        "fit_rows": int(train_normal.sum()),
        "bounded_fit_rows": int(len(fit_indices)),
        "feature_count": int(matrix.shape[1]),
        "feature_names": feature_names,
        "scored_rows": int(score_x.shape[0]),
        "fit_seconds": float(fit_seconds),
        "peak_working_set_mb": (
            None if peak_bytes is None else float(peak_bytes / (1024.0 * 1024.0))
        ),
        "score_batch_ms": _interval(scoring_ms),
        "single_station": {
            "station_id": str(single_station_id),
            "rows": int(single_station_x.shape[0]),
            "score_batch_ms": _interval(single_station_ms),
            "throughput_rows_per_second": (
                float(single_station_x.shape[0])
                / max(
                    statistics.fmean(single_station_ms) / 1000.0,
                    1e-12,
                )
            ),
        },
        "model_parameters": {
            "n_estimators": 300,
            "max_samples": "auto",
            "random_state": s248.BASELINE_FIT_SEED,
            "n_jobs": 1,
        },
        "benchmark_boundary": (
            "One frozen provider. The reported fit_rows are the full frozen "
            "train-normal rows used by the confirmatory Stage248 result; "
            "bounded_fit_rows is the deterministic measurement replay used "
            "only to obtain deployment evidence. It does not replace or "
            "alter the frozen Stage248 result."
        ),
        "trust_baseline_z_available": bool(np.isfinite(base_z).all()),
    }


def _fit_with_peak(
    estimator: object,
    matrix: np.ndarray,
    *,
    fit_y: np.ndarray | None = None,
) -> tuple[object, float, float | None]:
    _reset_peak_working_set()
    started = time.perf_counter()
    fitted = (
        estimator.fit(matrix, fit_y)
        if fit_y is not None
        else estimator.fit(matrix)
    )
    seconds = time.perf_counter() - started
    peak_bytes = _peak_working_set_bytes()
    return (
        fitted,
        float(seconds),
        None
        if peak_bytes is None
        else float(peak_bytes / (1024.0 * 1024.0)),
    )


def _score_interval(
    scorer: object,
    matrix: np.ndarray,
    repeats: int,
) -> dict[str, float | int | None]:
    values: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        _ = scorer(matrix)
        values.append((time.perf_counter() - started) * 1000.0)
    return _interval(values)


def _measure_in_subprocess(
    estimator_name: str,
    source: str,
    repeats: int,
    timeout_seconds: int = 300,
) -> dict[str, object] | None:
    """Run one benchmark in a fresh child process.

    This isolates peak working set from prior estimators. The child returns
    JSON on stdout; any failure falls back to the in-process measurement for
    that estimator, which is explicitly disclosed in the payload.
    """

    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--internal-benchmark",
        estimator_name,
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
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "status": "FAILED",
            "failure": "TIMEOUT",
            "timeout_seconds": max(1, int(timeout_seconds)),
            "stderr_tail": str(stderr)[-2000:],
        }
    if completed.returncode != 0:
        return {
            "status": "FAILED",
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-2000:],
        }
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {
        "status": "FAILED",
        "returncode": 0,
        "stderr_tail": "missing JSON result",
    }


def _benchmark_reconstruction_baselines(
    source: str,
    repeats: int,
    isolated: bool = True,
    timeout_seconds: int = 300,
) -> dict[str, object]:
    """Measure Stage256 reconstruction/distance deployment cost.

    The model configurations and frozen train-normal feature matrix are
    reused exactly.  Peak working set is measured in this process and is
    intentionally interpreted as a benchmark-local upper bound.
    """

    protocol = json.loads(
        s248.PROTOCOLS[source].read_text(encoding="utf-8")
    )
    raw = make_frame(source, protocol)
    base = detector_frame(raw)
    detectors = fit_detectors(base, protocol)
    train_normal = (
        base["split"].eq("train") & base["observation_present"]
    ).to_numpy(dtype=bool)
    components = score_components(base, detectors)
    features = candidate_features(base, train_normal)
    matrix, _ = s248.feature_matrix(
        components,
        features,
        train_normal,
    )
    station_ids = base["station_id"].astype(str).to_numpy()
    single_station_id = sorted(pd.unique(station_ids))[0]
    single_station_x = np.asarray(
        matrix[station_ids == single_station_id],
        dtype=float,
    )
    if single_station_x.shape[0] == 0:
        raise RuntimeError("single-station benchmark subset is empty")

    fit_indices = s248.stratified_fit_indices(
        base,
        train_normal,
        s256.MAX_FIT_ROWS,
        s256.MODEL_SEED,
    )
    knn_indices = s248.stratified_fit_indices(
        base,
        train_normal,
        s256.KNN_MAX_FIT_ROWS,
        s256.MODEL_SEED,
    )
    fit_x = np.asarray(matrix[fit_indices], dtype=float)
    knn_x = np.asarray(matrix[knn_indices], dtype=float)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Early stopping.*",
            category=UserWarning,
        )
        pca, pca_seconds, pca_peak = _fit_with_peak(
            make_pipeline(
                StandardScaler(),
                PCA(
                    n_components=min(
                        s256.PCA_COMPONENTS,
                        fit_x.shape[1],
                    ),
                    svd_solver="randomized",
                    random_state=s256.MODEL_SEED,
                ),
            ),
            fit_x,
        )
        mlp, mlp_seconds, mlp_peak = _fit_with_peak(
            make_pipeline(
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
            ),
            fit_x,
            fit_y=fit_x,
        )
        knn, knn_seconds, knn_peak = _fit_with_peak(
            make_pipeline(
                StandardScaler(),
                NearestNeighbors(
                    n_neighbors=min(
                        s256.KNN_NEIGHBOURS,
                        len(knn_x),
                    ),
                    n_jobs=-1,
                ),
            ),
            knn_x,
        )

    def pca_scores(values: np.ndarray) -> np.ndarray:
        scaled = pca.named_steps["standardscaler"].transform(values)
        reconstructed = pca.named_steps["pca"].inverse_transform(
            pca.named_steps["pca"].transform(scaled)
        )
        return np.mean((scaled - reconstructed) ** 2, axis=1)

    def mlp_scores(values: np.ndarray) -> np.ndarray:
        scaled = mlp.named_steps["standardscaler"].transform(values)
        reconstructed = mlp.named_steps["mlpregressor"].predict(scaled)
        return np.mean((scaled - reconstructed) ** 2, axis=1)

    def knn_scores(values: np.ndarray) -> np.ndarray:
        scaled = knn.named_steps["standardscaler"].transform(values)
        distances = knn.named_steps["nearestneighbors"].kneighbors(
            scaled,
            return_distance=True,
        )[0]
        return np.mean(distances, axis=1)

    result = {
        "source": source,
        "train_normal_rows": int(train_normal.sum()),
        "fit_rows": int(len(fit_indices)),
        "knn_fit_rows": int(len(knn_indices)),
        "feature_count": int(matrix.shape[1]),
        "pca": {
            "fit_seconds": pca_seconds,
            "peak_working_set_mb": pca_peak,
            "score_batch_ms": _score_interval(
                pca_scores,
                matrix,
                repeats,
            ),
            "single_station_rows": int(single_station_x.shape[0]),
            "single_station_score_batch_ms": _score_interval(
                pca_scores,
                single_station_x,
                repeats,
            ),
        },
        "mlp_autoencoder": {
            "fit_seconds": mlp_seconds,
            "peak_working_set_mb": mlp_peak,
            "iterations": int(
                mlp.named_steps["mlpregressor"].n_iter_
            ),
            "score_batch_ms": _score_interval(
                mlp_scores,
                matrix,
                repeats,
            ),
            "single_station_rows": int(single_station_x.shape[0]),
            "single_station_score_batch_ms": _score_interval(
                mlp_scores,
                single_station_x,
                repeats,
            ),
        },
        "knn": {
            "fit_seconds": knn_seconds,
            "peak_working_set_mb": knn_peak,
            "score_batch_ms": _score_interval(
                knn_scores,
                matrix,
                repeats,
            ),
            "single_station_rows": int(single_station_x.shape[0]),
            "single_station_score_batch_ms": _score_interval(
                knn_scores,
                single_station_x,
                repeats,
            ),
        },
        "boundary": (
            "Measurements use the frozen Stage256 configurations and "
            "train-normal feature matrix. They are deployment evidence "
            "only and do not replace a frozen detection result."
        ),
    }
    if isolated:
        for child_name, key in [
            ("pca_reconstruction", "pca"),
            ("mlp_autoencoder_reconstruction", "mlp_autoencoder"),
            ("knn_distance", "knn"),
        ]:
            child = _measure_in_subprocess(
                child_name,
                source,
                repeats,
                timeout_seconds=max(1, int(timeout_seconds)),
            )
            if child is not None and child.get("status") == "PASS":
                result[key].update(
                    {
                        "isolated_subprocess": True,
                        "fit_seconds": child["fit_seconds"],
                        "peak_working_set_mb": child[
                            "peak_working_set_mb"
                        ],
                        "score_batch_ms": child["score_batch_ms"],
                        "single_station_score_batch_ms": child[
                            "single_station_score_batch_ms"
                        ],
                        "single_station_rows": child[
                            "single_station_rows"
                        ],
                    }
                )
            else:
                result[key]["isolated_subprocess"] = False
                result[key]["isolation_failure"] = (
                    None if child is None else child
                )
    return result


def _internal_benchmark(
    estimator_name: str,
    source: str,
    repeats: int,
) -> int:
    benchmark = _benchmark_reconstruction_baselines(
        source,
        repeats,
        isolated=False,
    )
    mapping = {
        "pca_reconstruction": "pca",
        "mlp_autoencoder_reconstruction": "mlp_autoencoder",
        "knn_distance": "knn",
    }
    key = mapping[estimator_name]
    current = benchmark[key]
    print(
        json.dumps(
            {
                "status": "PASS",
                "estimator": estimator_name,
                "source": source,
                "fit_seconds": current["fit_seconds"],
                "peak_working_set_mb": current["peak_working_set_mb"],
                "score_batch_ms": current["score_batch_ms"],
                "single_station_rows": current["single_station_rows"],
                "single_station_score_batch_ms": current[
                    "single_station_score_batch_ms"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def _method_rows(
    stage248: dict,
    stage254: dict,
    stage274_resource: dict,
    stage256: dict,
    stage262: dict,
    stage288: dict,
    stage255: dict,
    stage261: dict,
    benchmark: dict[str, object],
    reconstruction_benchmark: dict[str, object],
) -> pd.DataFrame:
    provider_rows_248 = {
        str(row["source"]): row for row in stage248["fit_rows"]
    }
    provider_rows_256 = {
        str(row["source"]): row for row in stage256["fit_rows"]
    }
    provider_rows_262 = {
        str(row["source"]): row for row in stage262["fit_rows"]
    }
    ocs_fit = {
        int(row["requested_rows"]): row
        for row in stage254["fit_rows"]
        if float(row["peak_rss_mb"]) > 0.0
    }
    full_ocsvm_fits = {
        str(row["source"]): row
        for row in stage274_resource["fit_rows"]
    }
    full_eccc = full_ocsvm_fits.get("ECCC")
    if full_eccc is None:
        raise RuntimeError("Stage274 resource audit is missing ECCC")
    benchmark_fit_rows = int(benchmark["bounded_fit_rows"])
    benchmark_fit_seconds = float(benchmark["fit_seconds"])
    benchmark_peak = benchmark["peak_working_set_mb"]
    reconstruction_fit_rows = int(
        reconstruction_benchmark["fit_rows"]
    )
    reconstruction_knn_fit_rows = int(
        reconstruction_benchmark["knn_fit_rows"]
    )
    reconstruction_train_rows = int(
        reconstruction_benchmark["train_normal_rows"]
    )
    pca_benchmark = reconstruction_benchmark["pca"]
    mlp_benchmark = reconstruction_benchmark["mlp_autoencoder"]
    knn_benchmark = reconstruction_benchmark["knn"]
    usad_model = stage288["model"]
    usad_training = stage288["training"]
    usad_inference = stage288["valid_normal_inference"]

    rows: list[dict[str, object]] = []

    def add(
        method: str,
        display_name: str,
        family: str,
        layer: str,
        structure: str,
        parameter_count: int | str,
        fit_rows: int | str,
        train_normal_rows: int | str,
        fit_seconds: float | str,
        peak_memory_mb: float | str,
        latency_ms: float | str,
        throughput_fps: float | str,
        online_causal: bool,
        station_recalibration: str,
        gpu_required: bool,
        evidence: str,
    ) -> None:
        rows.append(
            {
                "method": method,
                "display_name": display_name,
                "family": family,
                "evidence_layer": layer,
                "model_structure": structure,
                "parameter_count": parameter_count,
                "fit_rows": fit_rows,
                "train_normal_rows": train_normal_rows,
                "fit_seconds": fit_seconds,
                "peak_memory_mb": peak_memory_mb,
                "score_latency_ms": latency_ms,
                "throughput_fps": throughput_fps,
                "single_station_throughput_rows_per_second": None,
                "online_causal": bool(online_causal),
                "station_recalibration_required": station_recalibration,
                "gpu_required": bool(gpu_required),
                "evidence_status": evidence,
            }
        )

    add(
        "trust_valid_f1",
        "TrustMeteo-QC valid-F1 candidate",
        "proposed_rule",
        "frozen_main_confirmatory",
        "Hampel-MAD + Kalman innovation + spatial consistency fused with "
        "one valid-selected robust-deviation feature and weight",
        "UNAVAILABLE",
        int(provider_rows_248["MeteoSwiss"]["train_normal_rows"]),
        int(provider_rows_248["MeteoSwiss"]["train_normal_rows"]),
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        True,
        "Yes: target-valid calibration is part of the frozen transfer rule.",
        False,
        "structure and fit row provenance; latency and memory unavailable",
    )
    add(
        "trust_nonstuck_event_macro",
        "TrustMeteo-QC non-stuck event-macro candidate",
        "proposed_rule",
        "frozen_main_confirmatory",
        "same three-component fused score with a different valid-selected "
        "feature/weight rule",
        "UNAVAILABLE",
        int(provider_rows_248["MeteoSwiss"]["train_normal_rows"]),
        int(provider_rows_248["MeteoSwiss"]["train_normal_rows"]),
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        True,
        "Yes: target-valid calibration is part of the frozen transfer rule.",
        False,
        "structure and fit row provenance; latency and memory unavailable",
    )
    add(
        "robust_max_deviation",
        "Robust max deviation",
        "statistical",
        "frozen_main_confirmatory",
        "maximum positive robust-standardized feature deviation",
        0,
        int(provider_rows_248["MeteoSwiss"]["train_normal_rows"]),
        int(provider_rows_248["MeteoSwiss"]["train_normal_rows"]),
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        True,
        "No separate learned model; only the valid FAR threshold is calibrated.",
        False,
        "deterministic statistic; no fitted-parameter artifact",
    )
    add(
        "isolation_forest",
        "Isolation Forest",
        "classical_unsupervised",
        "frozen_main_confirmatory",
        "300 isolation trees fitted on 16 robust-standardized features",
        300,
        int(provider_rows_248["MeteoSwiss"]["train_normal_rows"]),
        int(provider_rows_248["MeteoSwiss"]["train_normal_rows"]),
        benchmark_fit_seconds,
        benchmark_peak,
        benchmark["score_batch_ms"]["mean_ms"],
        "UNAVAILABLE",
        True,
        "No; valid-normal FAR threshold is calibration, not model refitting.",
        False,
        "measured bounded one-provider reconstruction benchmark",
    )
    add(
        "one_class_svm",
        "RBF One-Class SVM",
        "classical_unsupervised_security",
        "frozen_main_confirmatory",
        "StandardScaler + RBF One-Class SVM (nu=0.05, gamma=scale, "
        "cache_size=1000)",
        "UNAVAILABLE",
        int(full_eccc["effective_train_rows"]),
        int(full_eccc["train_normal_rows"]),
        float(full_eccc["fit_seconds"]),
        float(full_eccc["peak_rss_mb"]),
        "UNAVAILABLE",
        "UNAVAILABLE",
        True,
        "No; valid-normal FAR threshold is calibration, not model refitting.",
        False,
        (
            "fit time from Stage274 full-train-normal ECCC fit; peak working "
            "set is a shared-process cumulative value attributable only to "
            "ECCC at the observed maximum; latency unavailable. Stage254 "
            "20k primary-cap fit time remains "
            f"{float(ocs_fit[20000]['fit_seconds']) if 20000 in ocs_fit else 'UNAVAILABLE'}"
        ),
    )
    add(
        "ewma_deviation",
        "EWMA deviation",
        "statistical_process_control",
        "exploratory_supplement",
        "per-station exponentially weighted moving absolute residual "
        "(alpha=0.20)",
        1,
        int(provider_rows_262["MeteoSwiss"]["fit_rows"]),
        int(provider_rows_262["MeteoSwiss"]["train_normal_rows"]),
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        True,
        "Yes: per-station reference statistics are fitted on train-normal rows.",
        False,
        "structure and fit-budget provenance; measured latency unavailable",
    )
    add(
        "arima_innovation",
        "AR(4) innovation",
        "time_series_prediction",
        "exploratory_supplement",
        "per-station AR(4) least-squares prediction and absolute innovation",
        4,
        int(provider_rows_262["MeteoSwiss"]["fit_rows"]),
        int(provider_rows_262["MeteoSwiss"]["train_normal_rows"]),
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        True,
        "Yes: per-station AR coefficients are fitted on train-normal rows.",
        False,
        "structure and fit-budget provenance; measured latency unavailable",
    )
    add(
        "gru_predictor",
        "Lightweight GRU reservoir predictor",
        "sequence_prediction_proxy",
        "exploratory_supplement",
        "fixed random gated recurrent reservoir with 8 hidden units and a "
        "per-station linear train-normal readout",
        8,
        int(provider_rows_262["MeteoSwiss"]["fit_rows"]),
        int(provider_rows_262["MeteoSwiss"]["train_normal_rows"]),
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        True,
        "Yes: per-station readout coefficients are fitted on train-normal rows.",
        False,
        "transparent proxy; no measured latency or full GRU claim",
    )
    add(
        "lightweight_autoencoder",
        "Lightweight PCA autoencoder",
        "reconstruction",
        "exploratory_supplement",
        "StandardScaler + 12-component PCA bottleneck and linear reconstruction",
        12,
        int(provider_rows_262["MeteoSwiss"]["fit_rows"]),
        int(provider_rows_262["MeteoSwiss"]["train_normal_rows"]),
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        True,
        "No separate station refit; a global train-normal projection is used.",
        False,
        "structure and fit-budget provenance; measured latency unavailable",
    )
    add(
        "ics_cps_spc_proxy",
        "ICS/CPS SPC proxy",
        "classical_security_control",
        "exploratory_supplement",
        "per-station median/MAD standardized deviation",
        2,
        int(provider_rows_262["MeteoSwiss"]["fit_rows"]),
        int(provider_rows_262["MeteoSwiss"]["train_normal_rows"]),
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        True,
        "Yes: per-station center and scale are fitted on train-normal rows.",
        False,
        "transparent SPC proxy, not a paper-level ICS/CPS reproduction",
    )
    add(
        "usad_author_code",
        "USAD author code on public SWaT",
        "public_ics_cps_anomaly_detection",
        "independent_public_benchmark",
        (
            "fixed-commit author three-layer encoder with two sigmoid "
            "decoders, 12-row windows, and 1200 latent units"
        ),
        int(usad_model["parameter_count"]),
        int(usad_training["train_windows"]),
        int(usad_training["train_windows"]),
        float(usad_training["wall_seconds"]),
        float(usad_training["peak_working_set_mb"]),
        float(usad_inference["per_row_ms"]),
        float(usad_inference["throughput_rows_per_second"]),
        False,
        (
            "No per-station recalibration; the author-linked global "
            "train-normal scaler is used."
        ),
        True,
        (
            "Stage288 resource-only run on the author-linked SWaT normal "
            "file. It does not read the attack test file, replace Stage286 "
            "metrics, or form a same-contract TrustMeteo-QC comparison. "
            "The independent public SWaT endpoint is different from the "
            "TrustMeteo-QC endpoint."
        ),
    )
    add(
        "pca_reconstruction",
        "PCA reconstruction",
        "reconstruction",
        "exploratory_supplement",
        "StandardScaler + 12-component PCA reconstruction error",
        12,
        reconstruction_fit_rows,
        reconstruction_train_rows,
        pca_benchmark["fit_seconds"],
        pca_benchmark["peak_working_set_mb"],
        pca_benchmark["score_batch_ms"]["mean_ms"],
        "UNAVAILABLE",
        True,
        "No separate station refit; a global train-normal projection is used.",
        False,
        "measured bounded one-provider reconstruction benchmark",
    )
    add(
        "mlp_autoencoder_reconstruction",
        "MLP autoencoder reconstruction",
        "reconstruction",
        "exploratory_supplement",
        "StandardScaler + MLPRegressor 64-16-64 autoencoder",
        "UNAVAILABLE",
        reconstruction_fit_rows,
        reconstruction_train_rows,
        mlp_benchmark["fit_seconds"],
        mlp_benchmark["peak_working_set_mb"],
        mlp_benchmark["score_batch_ms"]["mean_ms"],
        "UNAVAILABLE",
        True,
        "No separate station refit; a global train-normal projection is used.",
        False,
        "measured bounded one-provider reconstruction benchmark",
    )
    add(
        "knn_distance",
        "kNN distance",
        "distance_reconstruction",
        "exploratory_supplement",
        "StandardScaler + 5-neighbour distance on a 10k train-normal core set",
        0,
        reconstruction_knn_fit_rows,
        reconstruction_train_rows,
        knn_benchmark["fit_seconds"],
        knn_benchmark["peak_working_set_mb"],
        knn_benchmark["score_batch_ms"]["mean_ms"],
        "UNAVAILABLE",
        True,
        "No separate station refit; a global train-normal core set is used.",
        False,
        "measured bounded one-provider reconstruction benchmark",
    )
    add(
        "testbed_causal_detector",
        "Authorized testbed causal detector",
        "deployment_path",
        "authorized_testbed",
        "strict online causal detector and recovery path on loopback TCP frames",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        "UNAVAILABLE",
        float(stage255["throughput_mean_frames_per_second"]),
        True,
        "UNAVAILABLE",
        False,
        "measured throughput from Stage255; latency and memory unavailable",
    )

    frame = pd.DataFrame(rows)
    frame.loc[
        frame["method"].eq("testbed_causal_detector"),
        "throughput_fps",
    ] = float(stage255["throughput_mean_frames_per_second"])
    frame.loc[
        frame["method"].eq("testbed_causal_detector"),
        "score_latency_ms",
    ] = 1000.0 / float(stage255["throughput_mean_frames_per_second"])
    frame.loc[
        frame["method"].eq("isolation_forest"),
        "single_station_throughput_rows_per_second",
    ] = float(benchmark["single_station"]["throughput_rows_per_second"])
    for method, benchmark_key in [
        ("pca_reconstruction", "pca"),
        ("mlp_autoencoder_reconstruction", "mlp_autoencoder"),
        ("knn_distance", "knn"),
    ]:
        current = reconstruction_benchmark[benchmark_key]
        station_interval = current["single_station_score_batch_ms"]
        frame.loc[
            frame["method"].eq(method),
            "single_station_throughput_rows_per_second",
        ] = (
            float(current["single_station_rows"])
            / max(float(station_interval["mean_ms"]) / 1000.0, 1e-12)
        )

    stage261_penalty = float(stage261["causal_penalty_mae_mean"])
    frame.loc[frame.index[-1], "causal_offline_penalty_mae"] = stage261_penalty
    if "causal_offline_penalty_mae" not in frame.columns:
        frame["causal_offline_penalty_mae"] = np.nan
    frame["causal_offline_penalty_mae"] = frame[
        "causal_offline_penalty_mae"
    ].fillna(stage261_penalty)
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="MeteoSwiss")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="timeout for each isolated deployment benchmark child process",
    )
    parser.add_argument(
        "--remeasure",
        action="store_true",
        help=(
            "rerun the expensive bounded deployment benchmark instead of "
            "reusing the prior measured benchmark"
        ),
    )
    parser.add_argument(
        "--internal-benchmark",
        choices=[
            "pca_reconstruction",
            "mlp_autoencoder_reconstruction",
            "knn_distance",
        ],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.internal_benchmark:
        return _internal_benchmark(
            str(args.internal_benchmark),
            str(args.source),
            max(1, int(args.repeats)),
        )

    stage248 = _load_json("stage248_cross_provider_fair_baseline_audit")
    stage254 = _load_json("stage254_ocsvm_training_scale_sensitivity")
    stage274_resource = _load_json(
        "stage274_ocsvm_full_scale_sensitivity_resource_audit"
    )
    stage256 = _load_json("stage256_lightweight_strong_baselines")
    stage262 = _load_json("stage262_security_domain_baselines")
    stage288 = _load_json("stage288_swat_usad_resource_audit")
    stage255 = _load_json("stage255_online_causal_testbed")
    stage261 = _load_json("stage261_causal_offline_recovery_contrast")
    resource_sampling = _load_json("stage262_resource_sampling")

    existing = None if args.remeasure else _load_existing_benchmark()
    if existing is None:
        benchmark = _benchmark_unscaled_baselines(
            str(args.source),
            max(1, int(args.repeats)),
        )
        reconstruction_benchmark = _benchmark_reconstruction_baselines(
            str(args.source),
            max(1, int(args.repeats)),
            timeout_seconds=max(1, int(args.timeout_seconds)),
        )
        benchmark["measurement_mode"] = "fresh"
        reconstruction_benchmark["measurement_mode"] = "fresh"
    else:
        benchmark, reconstruction_benchmark = existing
        benchmark["measurement_mode"] = "reused"
        reconstruction_benchmark["measurement_mode"] = "reused"
    methods = _method_rows(
        stage248,
        stage254,
        stage274_resource,
        stage256,
        stage262,
        stage288,
        stage255,
        stage261,
        benchmark,
        reconstruction_benchmark,
    )

    method_columns = [
        "method",
        "display_name",
        "family",
        "evidence_layer",
        "model_structure",
        "parameter_count",
        "fit_rows",
        "train_normal_rows",
        "fit_seconds",
        "peak_memory_mb",
        "score_latency_ms",
        "throughput_fps",
        "single_station_throughput_rows_per_second",
        "online_causal",
        "station_recalibration_required",
        "gpu_required",
        "causal_offline_penalty_mae",
        "evidence_status",
    ]
    methods = methods[method_columns]

    frozen = methods[
        methods["evidence_layer"].eq("frozen_main_confirmatory")
    ]
    measured_memory = methods["peak_memory_mb"].map(
        lambda value: isinstance(value, (int, float)) and np.isfinite(value)
    )
    measured_latency = methods["score_latency_ms"].map(
        lambda value: isinstance(value, (int, float)) and np.isfinite(value)
    )
    host = resource_sampling["host"]
    payload = {
        "status": "PASS_DEPLOYMENT_COMPLEXITY_EVIDENCE_AUDIT",
        "protocol": (
            "Stage269 reads frozen Stage248/254/255/256/261/262 artifacts and "
            "performs bounded deterministic refits of Isolation Forest, PCA, "
            "MLP autoencoder, and kNN using frozen Stage256 configurations "
            "to measure deployment cost. It does not change any split, "
            "threshold, prediction, endpoint, or claim."
        ),
        "method_count": int(len(methods)),
        "frozen_main_method_count": int(len(frozen)),
        "measured_memory_method_count": int(measured_memory.sum()),
        "measured_latency_method_count": int(measured_latency.sum()),
        "unavailable_memory_method_count": int((~measured_memory).sum()),
        "unavailable_latency_method_count": int((~measured_latency).sum()),
        "measured_subset_boundary": (
            "Isolation Forest, PCA, MLP autoencoder, and kNN are benchmark-"
            "refit in Stage269 using frozen configurations. PCA, MLP "
            "autoencoder, and kNN are re-measured in separate child "
            "processes so their peak working sets are attributable to the "
            "individual estimator process. Isolation Forest remains a "
            "bounded in-process measurement because its input construction "
            "is part of the frozen audit replay. "
            "The scaled One-Class SVM resource evidence comes from Stage274's "
            "separate full-train-normal ECCC fit-only replay; its peak RSS is "
            "a shared-process cumulative peak and cannot be attributed to "
            "individual providers. Fields without measured evidence remain "
            "UNAVAILABLE."
        ),
        "host": {
            "platform": host["platform"],
            "processor": host["processor"],
            "logical_cpu_count": int(host["logical_cpu_count"]),
            "python": host["python"],
        },
        "benchmark": benchmark,
        "reconstruction_benchmark": reconstruction_benchmark,
        "online_causal_boundary": {
            "stage255_scope": stage255["scope"],
            "stage261_scope": stage261["scope"],
            "online_equals_offline_any": bool(
                stage261["online_equals_offline_any"]
            ),
            "causal_penalty_mae_mean": float(
                stage261["causal_penalty_mae_mean"]
            ),
        },
        "resource_sampling_reference": {
            "path": "tables/stage262_resource_sampling.json",
            "cold_start_mean_ms": resource_sampling["cold_start_ms"]["mean"],
            "online_frame_mean_ms": resource_sampling[
                "online_frame_latency_ms"
            ]["mean"],
            "rss_after_load_mb": resource_sampling["memory_mb"][
                "rss_after_load"
            ],
        },
        "methods": _json_safe_records(methods),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                TABLES / "stage248_cross_provider_fair_baseline_audit.json",
                TABLES / "stage254_ocsvm_training_scale_sensitivity.json",
                TABLES
                / "stage274_ocsvm_full_scale_sensitivity_resource_audit.json",
                TABLES / "stage255_online_causal_testbed.json",
                TABLES / "stage256_lightweight_strong_baselines.json",
                TABLES / "stage261_causal_offline_recovery_contrast.json",
                TABLES / "stage262_security_domain_baselines.json",
                TABLES / "stage262_resource_sampling.json",
                TABLES / "stage288_swat_usad_resource_audit.json",
            ]
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "limitations": [
            "Stage269 is deployment evidence only and does not change the "
            "frozen main detection result.",
            "Parameter counts and model sizes that cannot be obtained from "
            "frozen artifacts remain UNAVAILABLE.",
            "The bounded Isolation Forest benchmark uses one target provider "
            "and does not represent four-provider total training time.",
            "PCA, MLP autoencoder, and kNN are bounded one-provider "
            "reconstruction benchmarks using their frozen Stage256 settings; "
            "their peak memory values are process-local upper bounds.",
            "The USAD resource row is an independent public SWaT "
            "benchmark with a different endpoint from TrustMeteo-QC; it "
            "must not be merged into the same-contract main comparison.",
            "The Stage274 One-Class SVM peak RSS is a shared-process "
            "cumulative peak and cannot be interpreted as an independently "
            "provider-attributable memory requirement.",
            "Testbed throughput is from an authorized isolated loopback "
            "testbed and is not production throughput.",
            "Online causal recovery is not equivalent to offline "
            "bidirectional interpolation.",
            "The USAD resource run is an independent public SWaT benchmark "
            "with a different endpoint from TrustMeteo-QC and does not "
            "upgrade the frozen main comparison.",
        ],
    }
    if payload["frozen_main_method_count"] != len(
        stage248["methods"]
    ):
        raise RuntimeError("frozen main method inventory mismatch")
    if payload["online_causal_boundary"]["online_equals_offline_any"]:
        raise RuntimeError("online/offline boundary unexpectedly violated")
    gpu_methods = set(
        methods.loc[methods["gpu_required"].astype(bool), "method"]
    )
    if gpu_methods != {"usad_author_code"}:
        raise RuntimeError(
            f"unexpected GPU-required method inventory: {sorted(gpu_methods)}"
        )

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    methods.to_csv(
        TABLES / f"{PREFIX}_methods.csv",
        index=False,
        encoding="utf-8-sig",
    )
    benchmark_rows = [
        {
            "source": benchmark["source"],
            "rows_total": benchmark["rows_total"],
            "train_normal_rows": benchmark["train_normal_rows"],
            "fit_rows": benchmark["bounded_fit_rows"],
            "bounded_fit_rows": benchmark["bounded_fit_rows"],
            "confirmatory_fit_rows": benchmark["fit_rows"],
            "feature_count": benchmark["feature_count"],
            "fit_seconds": benchmark["fit_seconds"],
            "peak_working_set_mb": benchmark["peak_working_set_mb"],
            "score_batch_mean_ms": benchmark["score_batch_ms"]["mean_ms"],
            "score_batch_p95_ms": benchmark["score_batch_ms"]["p95_ms"],
            "task": "isolation_forest",
            "model_label": "Isolation Forest",
        }
    ]
    for benchmark_key, model_label in [
        ("pca", "PCA reconstruction"),
        ("mlp_autoencoder", "MLP autoencoder reconstruction"),
        ("knn", "kNN distance"),
    ]:
        current = reconstruction_benchmark[benchmark_key]
        benchmark_rows.append(
            {
                "source": reconstruction_benchmark["source"],
                "rows_total": benchmark["rows_total"],
                "train_normal_rows": reconstruction_benchmark[
                    "train_normal_rows"
                ],
                "fit_rows": (
                    reconstruction_benchmark["knn_fit_rows"]
                    if benchmark_key == "knn"
                    else reconstruction_benchmark["fit_rows"]
                ),
                "bounded_fit_rows": (
                    reconstruction_benchmark["knn_fit_rows"]
                    if benchmark_key == "knn"
                    else reconstruction_benchmark["fit_rows"]
                ),
                "confirmatory_fit_rows": reconstruction_benchmark[
                    "train_normal_rows"
                ],
                "feature_count": reconstruction_benchmark["feature_count"],
                "fit_seconds": current["fit_seconds"],
                "peak_working_set_mb": current["peak_working_set_mb"],
                "score_batch_mean_ms": current["score_batch_ms"]["mean_ms"],
                "score_batch_p95_ms": current["score_batch_ms"]["p95_ms"],
                "task": benchmark_key,
                "model_label": model_label,
            }
        )
    pd.DataFrame(benchmark_rows).to_csv(
        TABLES / f"{PREFIX}_benchmark.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = [
        "# 阶段269 部署复杂度与资源证据审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 方法数：`{payload['method_count']}`",
        f"- 冻结主结果方法数：`{payload['frozen_main_method_count']}`",
        f"- 有实测内存方法数：`{payload['measured_memory_method_count']}`",
        f"- 有实测延迟方法数：`{payload['measured_latency_method_count']}`",
        f"- 内存仍不可得方法数：`{payload['unavailable_memory_method_count']}`",
        f"- 延迟仍不可得方法数：`{payload['unavailable_latency_method_count']}`",
        "",
        "## 有界重建基准",
        "",
        f"- 数据源：`{benchmark['source']}`",
        f"- 总行数：`{benchmark['rows_total']}`",
        f"- train-normal行数：`{benchmark['train_normal_rows']}`",
        f"- 确认性拟合行数：`{benchmark['fit_rows']}`",
        f"- 有界测量拟合行数：`{benchmark['bounded_fit_rows']}`",
        f"- 特征数：`{benchmark['feature_count']}`",
        f"- Isolation Forest拟合秒数：`{benchmark['fit_seconds']}`",
        f"- 峰值工作集：`{benchmark['peak_working_set_mb']}` MB",
        f"- 全矩阵评分批均值：`{benchmark['score_batch_ms']['mean_ms']}` ms",
        f"- 单站点评分站数：`1`；站点：`{benchmark['single_station']['station_id']}`；行数：`{benchmark['single_station']['rows']}`",
        f"- 单站点评分吞吐：`{benchmark['single_station']['throughput_rows_per_second']}` 行/秒",
        "",
        "## 轻量重建基线资源实测",
        "",
        "| 方法 | train-normal行数 | 实际fit行数 | fit秒数 | 峰值工作集MB | 全矩阵评分均值ms | 单站吞吐行/秒 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in benchmark_rows[1:]:
        method_key = row["task"]
        current = reconstruction_benchmark[method_key]
        station_interval = current["single_station_score_batch_ms"]
        throughput = float(current["single_station_rows"]) / max(
            float(station_interval["mean_ms"]) / 1000.0,
            1e-12,
        )
        report.append(
            f"| {row['model_label']} | {row['train_normal_rows']} | "
            f"{row['fit_rows']} | {row['fit_seconds']:.6f} | "
            f"{row['peak_working_set_mb']:.6f} | "
            f"{row['score_batch_mean_ms']:.6f} | {throughput:.6f} |"
        )
    report += [
        "",
        "",
        "## 边界",
        "",
        "- 本阶段只读冻结产物；额外拟合仅用于部署复杂度证据，不改变任何预测或阈值。",
        "- Isolation Forest使用冻结配置重建，峰值内存是本进程上限；PCA、MLP自编码器和kNN在独立子进程中重建，其峰值内存可分别归因。",
        "- 没有冻结依据的参数量、内存和延迟保持 `UNAVAILABLE`。",
        "- 单机构基准不代表四机构总训练时间。",
        "- 授权测试床吞吐不是生产吞吐。",
        "- 在线因果恢复不等同于离线双向插值。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "method_count": payload["method_count"],
                "measured_memory_method_count": payload[
                    "measured_memory_method_count"
                ],
                "measured_latency_method_count": payload[
                    "measured_latency_method_count"
                ],
                "benchmark_fit_seconds": benchmark["fit_seconds"],
                "benchmark_peak_working_set_mb": benchmark[
                    "peak_working_set_mb"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
