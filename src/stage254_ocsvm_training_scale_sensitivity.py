from __future__ import annotations

"""Stage 254: pre-registered One-Class SVM training-scale sensitivity.

Stage248 deliberately capped the RBF One-Class SVM fit at 20,000
train-normal rows for compute reasons.  This stage asks whether that cap
materially changes the complementary conclusion.  It reuses the frozen
provider protocols, normalized tables, split boundaries, transparent attack
generators, detector fitting, feature construction and primary FAR=0.01
endpoint from Stage248, while changing only the One-Class SVM fit size.

The scale grid and endpoint are frozen before results are inspected.  Test
labels are evaluation-only.  This stage does not modify the frozen Stage246
result or promote a new variant into the confirmatory claim set.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage254_ocsvm_training_scale_sensitivity"
CACHE = ROOT / "testbed" / "stage254_cache"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    ATTACK_PARAMETERS,
    COMPONENTS,
    detector_frame,
    fit_detectors,
    inject_attack,
    metrics,
    score_components,
    sha256_file,
)
from stage208_dwd_persistent_repeat_protection import (  # noqa: E402
    VALID_INJECTION_SEED_OFFSET,
    choose_split_attack_blocks,
    event_metrics,
    far_threshold,
    type_metrics,
)
from stage217_swiss_controlled_attack_evaluation import (  # noqa: E402
    fuse_scores,
    robust_standardize,
)
from stage219_swiss_dwd_protection_transfer import clean_records  # noqa: E402
from stage240_nonstuck_balanced_selection import (  # noqa: E402
    AUDITS,
    NORMALIZED,
    PROTOCOLS,
    make_frame,
)
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_features,
)
from stage248_cross_provider_fair_baseline_audit import (  # noqa: E402
    BASELINE_FIT_SEED,
    MAX_OCSVM_FIT_ROWS,
    feature_matrix,
    stratified_fit_indices,
)

# Frozen before execution.  The absolute grid answers whether the original
# 20k cap created a discontinuity that changes the complementarity claim.
FIT_ROWS = [20000, 50000, 100000]
FIT_SEED = BASELINE_FIT_SEED
PRIMARY_ENDPOINT = {
    "metric": "event_coverage",
    "candidate": "one_class_svm",
    "far": 0.01,
    "selection_rule": "report_all_pre_registered_scale_limits",
}
CLAIM_CEILING = "PARTIALLY_CLOSED"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_slug(source: str) -> str:
    return source.lower().replace(" ", "_").replace("-", "_")


def artifact_fingerprint(source: str) -> str:
    digest = hashlib.sha256()
    for path in [
        Path(__file__).resolve(),
        PROTOCOLS[source],
        NORMALIZED[source],
        AUDITS[source],
    ]:
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def cached_frame(
    source: str,
    protocol: dict,
) -> tuple[pd.DataFrame, bool]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"frame__{cache_slug(source)}.joblib"
    key = artifact_fingerprint(source)
    if path.is_file():
        payload = joblib.load(path)
        if payload.get("key") == key:
            return payload["frame"], True
    frame = make_frame(source, protocol)
    joblib.dump(
        {"key": key, "frame": frame},
        path,
        compress=3,
    )
    return frame, False


def cached_cases(
    source: str,
    frame: pd.DataFrame,
    protocol: dict,
    detectors: dict,
    train_normal: np.ndarray,
) -> tuple[dict[int, dict[str, object]], bool]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"cases__{cache_slug(source)}.joblib"
    key = artifact_fingerprint(source)
    if path.is_file():
        payload = joblib.load(path)
        if payload.get("key") == key:
            return payload["cases"], True
    cases = prepare_attack_cases(
        frame,
        protocol,
        detectors,
        train_normal,
        source,
    )
    joblib.dump(
        {"key": key, "cases": cases},
        path,
        compress=3,
    )
    return cases, False


def peak_rss_mb() -> float:
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
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

            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = ctypes.c_void_p
            get_process_memory_info = (
                ctypes.windll.psapi.GetProcessMemoryInfo
            )
            get_process_memory_info.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ProcessMemoryCounters),
                ctypes.c_ulong,
            ]
            get_process_memory_info.restype = ctypes.c_int
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = get_current_process()
            ok = get_process_memory_info(
                handle,
                ctypes.byref(counters),
                counters.cb,
            )
            if not ok:
                raise ctypes.WinError()
            return float(counters.PeakWorkingSetSize / (1024.0 * 1024.0))
        except Exception:
            try:
                import ctypes

                access = process_query_limited_information
                get_current_process_id = (
                    ctypes.windll.kernel32.GetCurrentProcessId
                )
                open_process = ctypes.windll.kernel32.OpenProcess
                open_process.argtypes = [
                    ctypes.c_ulong,
                    ctypes.c_int,
                    ctypes.c_ulong,
                ]
                open_process.restype = ctypes.c_void_p
                close_handle = ctypes.windll.kernel32.CloseHandle
                close_handle.argtypes = [ctypes.c_void_p]
                close_handle.restype = ctypes.c_int

                class ProcessMemoryCountersFallback(ctypes.Structure):
                    _fields_ = ProcessMemoryCounters._fields_

                get_process_memory_info = (
                    ctypes.windll.psapi.GetProcessMemoryInfo
                )
                get_process_memory_info.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ProcessMemoryCountersFallback),
                    ctypes.c_ulong,
                ]
                get_process_memory_info.restype = ctypes.c_int
                handle = open_process(
                    access,
                    0,
                    int(get_current_process_id()),
                )
                if not handle:
                    return float("nan")
                try:
                    counters = ProcessMemoryCountersFallback()
                    counters.cb = ctypes.sizeof(counters)
                    ok = get_process_memory_info(
                        handle,
                        ctypes.byref(counters),
                        counters.cb,
                    )
                    if not ok:
                        return float("nan")
                    return float(
                        counters.PeakWorkingSetSize / (1024.0 * 1024.0)
                    )
                finally:
                    close_handle(handle)
            except Exception:
                return float("nan")
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(usage / 1024.0)
    except Exception:
        return float("nan")


def effective_limits(train_rows: int) -> list[int]:
    values: list[int] = []
    for requested in FIT_ROWS:
        values.append(int(min(int(requested), int(train_rows))))
    return list(dict.fromkeys(values))


def fit_ocsvm(
    matrix: np.ndarray,
    frame: pd.DataFrame,
    train_normal: np.ndarray,
    limit: int,
) -> tuple[object, dict[str, object]]:
    indices = stratified_fit_indices(
        frame,
        train_normal,
        int(limit),
        int(FIT_SEED),
    )
    started = time.perf_counter()
    model = make_pipeline(
        StandardScaler(),
        OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05,
            cache_size=1000,
            max_iter=-1,
        ),
    ).fit(matrix[indices])
    elapsed = time.perf_counter() - started
    support_vectors = int(np.asarray(model[-1].support_).size)
    return model, {
        "fit_rows": int(indices.size),
        "fit_seconds": float(elapsed),
        "peak_rss_mb": float(peak_rss_mb()),
        "support_vectors": support_vectors,
        "support_vector_ratio": float(
            support_vectors / max(1, indices.size)
        ),
    }


def matrix_for(
    components: pd.DataFrame,
    features: dict[tuple[str, int], np.ndarray],
    train_normal: np.ndarray,
) -> np.ndarray:
    matrix, _ = feature_matrix(components, features, train_normal)
    return matrix


def score_in_chunks(
    model: object,
    matrix: np.ndarray,
    chunk_size: int = 2000,
) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=float)
    scores = np.empty(matrix.shape[0], dtype=float)
    for start in range(0, matrix.shape[0], int(chunk_size)):
        end = min(matrix.shape[0], start + int(chunk_size))
        scores[start:end] = -model.decision_function(
            matrix[start:end]
        ).ravel()
    return scores


def prepare_attack_cases(
    base: pd.DataFrame,
    protocol: dict,
    detectors: dict,
    train_normal: np.ndarray,
    source: str | None = None,
) -> dict[int, dict[str, object]]:
    cases: dict[int, dict[str, object]] = {}
    for raw_seed in protocol["seeds"]:
        seed = int(raw_seed)
        seed_path = (
            CACHE / f"case__{cache_slug(source)}__seed{seed}.joblib"
            if source is not None
            else None
        )
        if seed_path is not None and seed_path.is_file():
            payload = joblib.load(seed_path)
            if payload.get("key") == artifact_fingerprint(source):
                cases[seed] = payload["case"]
                print(
                    json.dumps(
                        {
                            "source": source,
                            "seed": int(seed),
                            "event": "case_cache_hit",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
        valid_blocks = choose_split_attack_blocks(
            base,
            protocol,
            seed,
            "valid",
        )
        test_blocks = choose_split_attack_blocks(
            base,
            protocol,
            seed,
            "test",
        )
        valid_attacked = inject_attack(
            base,
            valid_blocks,
            seed + VALID_INJECTION_SEED_OFFSET,
        )
        test_attacked = inject_attack(base, test_blocks, seed)
        for attacked, split in (
            (valid_attacked, "valid"),
            (test_attacked, "test"),
        ):
            attacked["attack_event_id"] = np.where(
                attacked["is_attack"],
                attacked["station_id"].astype(str)
                + "__"
                + attacked["attack_type"].astype(str)
                + "__"
                + split
                + "__"
                + str(seed),
                "",
            )
        valid_components = score_components(valid_attacked, detectors)
        test_components = score_components(test_attacked, detectors)
        valid_fused, _ = fuse_scores(valid_attacked, valid_components)
        test_fused, _ = fuse_scores(test_attacked, test_components)
        valid_base_z = robust_standardize(
            valid_fused,
            train_normal,
        ).to_numpy(dtype=float)
        test_base_z = robust_standardize(
            test_fused,
            train_normal,
        ).to_numpy(dtype=float)
        valid_features = candidate_features(valid_attacked, train_normal)
        test_features = candidate_features(test_attacked, train_normal)
        valid_matrix = matrix_for(
            valid_components,
            valid_features,
            train_normal,
        )
        test_matrix = matrix_for(
            test_components,
            test_features,
            train_normal,
        )
        valid_mask = valid_attacked["split"].eq("valid").to_numpy(
            dtype=bool
        )
        test_mask = test_attacked["split"].eq("test").to_numpy(dtype=bool)
        valid_frame = valid_attacked.loc[valid_mask].reset_index(drop=True)
        test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
        valid_normal = (
            valid_frame["observation_present"]
            & ~valid_frame["is_attack"]
        ).to_numpy(dtype=bool)
        cases[seed] = {
            "valid_matrix": valid_matrix[valid_mask],
            "test_matrix": test_matrix[test_mask],
            "valid_frame": valid_frame,
            "test_frame": test_frame,
            "valid_normal": valid_normal,
        }
        if seed_path is not None:
            joblib.dump(
                {
                    "key": artifact_fingerprint(source),
                    "case": cases[seed],
                },
                seed_path,
                compress=3,
            )
            print(
                json.dumps(
                    {
                        "source": source,
                        "seed": int(seed),
                        "event": "case_cached",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return cases


def evaluate_case(
    model: object,
    case: dict[str, object],
    source: str,
    seed: int,
    requested_rows: int,
    fit_rows: int,
    threshold_far: float,
    chunk_size: int = 2000,
) -> dict[str, object]:
    valid_matrix = np.asarray(case["valid_matrix"], dtype=float)
    test_matrix = np.asarray(case["test_matrix"], dtype=float)
    valid_frame = case["valid_frame"]
    test_frame = case["test_frame"]
    valid_normal = np.asarray(case["valid_normal"], dtype=bool)
    valid_score = score_in_chunks(
        model,
        valid_matrix[valid_normal],
        int(chunk_size),
    )
    test_score = score_in_chunks(
        model,
        test_matrix,
        int(chunk_size),
    )
    threshold = far_threshold(
        valid_score,
        np.ones(valid_score.size, dtype=bool),
        float(threshold_far),
    )
    predicted = (
        test_frame["observation_present"].to_numpy(dtype=bool)
        & (test_score >= threshold)
    )
    row_metrics = metrics(
        test_frame["is_attack"].astype(bool).to_numpy(),
        predicted,
        test_score,
    )
    event_values = event_metrics(test_frame, predicted)
    type_values = type_metrics(test_frame, predicted, test_score)
    nonstuck = type_values.get("nonstuck", {})
    stuck = type_values.get("stuck", {})
    return {
        "source": source,
        "seed": int(seed),
        "requested_rows": int(requested_rows),
        "fit_rows": int(fit_rows),
        "fit_seed": int(FIT_SEED),
        "threshold": float(threshold),
        **row_metrics,
        **event_values,
        "nonstuck_event_coverage": float(
            nonstuck.get("event_coverage", np.nan)
        ),
        "stuck_event_coverage": float(
            stuck.get("event_coverage", np.nan)
        ),
    }


def write_summary(
    provider_rows: list[dict[str, object]],
    fit_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    providers: list[str],
    dry_run: bool = False,
) -> int:
    fits = pd.DataFrame(fit_rows)
    if dry_run:
        payload = {
            "status": "DRY_RUN_STAGE254_OCSVM_TRAINING_SCALE_SENSITIVITY",
            "protocol": (
                "仅验证冻结Stage248 One-Class SVM训练规模档位，"
                "不评价test，不修改冻结主结果。"
            ),
            "fit_limits": [int(value) for value in FIT_ROWS],
            "fit_seed": int(FIT_SEED),
            "provider_rows": provider_rows,
            "fit_rows": clean_records(fits.to_dict("records")),
            "fit_row_count": int(len(fits)),
            "primary_endpoint": PRIMARY_ENDPOINT,
            "claim_ceiling": CLAIM_CEILING,
        }
    else:
        detail = pd.DataFrame(detail_rows)
        if detail.empty:
            raise RuntimeError("stage254 produced no evaluation rows")
        aggregate = (
            detail.groupby(
                ["source", "requested_rows", "fit_rows"],
                as_index=False,
            )
            .agg(
                seed_count=("seed", "nunique"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
                recall_mean=("recall", "mean"),
                false_alarm_rate_mean=("false_alarm_rate", "mean"),
                event_coverage_mean=("event_coverage", "mean"),
                event_coverage_std=("event_coverage", "std"),
                nonstuck_event_coverage_mean=(
                    "nonstuck_event_coverage",
                    "mean",
                ),
                stuck_event_coverage_mean=(
                    "stuck_event_coverage",
                    "mean",
                ),
            )
            .sort_values(["source", "requested_rows"])
        )
        provider_equal = (
            aggregate.groupby(
                "requested_rows",
                as_index=False,
            )
            .agg(
                provider_count=("source", "nunique"),
                fit_rows_min=("fit_rows", "min"),
                fit_rows_max=("fit_rows", "max"),
                f1_mean=("f1_mean", "mean"),
                event_coverage_mean=("event_coverage_mean", "mean"),
                event_coverage_min=("event_coverage_mean", "min"),
                event_coverage_max=("event_coverage_mean", "max"),
                nonstuck_event_coverage_mean=(
                    "nonstuck_event_coverage_mean",
                    "mean",
                ),
                stuck_event_coverage_mean=(
                    "stuck_event_coverage_mean",
                    "mean",
                ),
            )
            .sort_values("requested_rows")
        )
        payload = {
            "status": "PASS_OCSVM_TRAINING_SCALE_SENSITIVITY",
            "protocol": (
                "阶段254复用Stage248冻结provider、split、透明攻击、"
                "检测器拟合、特征构造、valid正常样本FAR=0.01阈值和"
                "test只评价协议；只把One-Class SVM训练规模改为预注册"
                "绝对档位20k、50k、100k，并使用同一固定拟合种子。"
                "所有档位必须报告，不能用test选择档位。"
            ),
            "primary_endpoint": PRIMARY_ENDPOINT,
            "claim_ceiling": CLAIM_CEILING,
            "fit_limits": [int(value) for value in FIT_ROWS],
            "fit_seed": int(FIT_SEED),
            "provider_count": int(len(provider_rows)),
            "provider_rows": provider_rows,
            "fit_rows": clean_records(fits.to_dict("records")),
            "fit_row_count": int(len(fits)),
            "detail_rows": int(len(detail)),
            "aggregate_rows": int(len(aggregate)),
            "provider_equal_rows": clean_records(
                provider_equal.to_dict("records")
            ),
            "attack_parameters": ATTACK_PARAMETERS,
            "components": COMPONENTS,
            "protocol_sha256": {
                source: sha256_file(path)
                for source, path in PROTOCOLS.items()
                if source in providers
            },
            "normalized_sha256": {
                source: sha256_file(path)
                for source, path in NORMALIZED.items()
                if source in providers
            },
            "audit_sha256": {
                source: sha256_file(path)
                for source, path in AUDITS.items()
                if source in providers
            },
            "code_sha256": sha256_file(Path(__file__).resolve()),
            "limitations": [
                "该阶段只重新拟合One-Class SVM，不替换Stage246冻结主结果。",
                "20k、50k、100k是预注册主曲线，不能外推为所有预算结论。",
                "所有攻击仍为透明受控注入，不是生产攻击日志。",
                "若某个规模档位优势消失，必须保留该结果并写明互补边界。",
            ],
        }
        detail.to_csv(
            TABLES / f"{PREFIX}_detail.csv",
            index=False,
            encoding="utf-8-sig",
        )
        aggregate.to_csv(
            TABLES / f"{PREFIX}_aggregate.csv",
            index=False,
            encoding="utf-8-sig",
        )
        provider_equal.to_csv(
            TABLES / f"{PREFIX}_provider_equal.csv",
            index=False,
            encoding="utf-8-sig",
        )
        fits.to_csv(
            TABLES / f"{PREFIX}_fits.csv",
            index=False,
            encoding="utf-8-sig",
        )
        lines = [
            "# 阶段254 One-Class SVM训练规模敏感性",
            "",
            f"- 状态：**{payload['status']}**",
            f"- 预注册训练行数：`{', '.join(str(v) for v in FIT_ROWS)}`",
            f"- 固定拟合种子：`{FIT_SEED}`",
            f"- 详情行：`{payload['detail_rows']}`；拟合行：`{payload['fit_row_count']}`",
            "",
            "## 机构等权结果",
            "",
            "| 请求行数 | 实际拟合行数范围 | provider数 | F1均值 | 事件覆盖均值 | 事件覆盖最小 | 事件覆盖最大 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in provider_equal.to_dict("records"):
            fit_range = (
                f"{int(row['fit_rows_min'])}"
                if int(row["fit_rows_min"]) == int(row["fit_rows_max"])
                else (
                    f"{int(row['fit_rows_min'])}-"
                    f"{int(row['fit_rows_max'])}"
                )
            )
            lines.append(
                f"| {int(row['requested_rows'])} | {fit_range} | "
                f"{int(row['provider_count'])} | {row['f1_mean']:.6f} | "
                f"{row['event_coverage_mean']:.6f} | "
                f"{row['event_coverage_min']:.6f} | "
                f"{row['event_coverage_max']:.6f} |"
            )
        lines += [
            "",
            "## 边界",
            "",
            "- 该曲线只检验20k预算截断是否改变One-Class SVM互补结论，不是新的确认性主终点。",
            "- 所有预注册档位必须同时报告；不得只保留对Trust有利或不利的档位。",
            "- 结果不改变冻结Stage246主表。",
        ]
        (REPORTS / f"{PREFIX}.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "provider_rows": len(provider_rows),
                "fit_rows": len(fit_rows),
                "detail_rows": int(payload.get("detail_rows", 0)),
                "aggregate_rows": int(payload.get("aggregate_rows", 0)),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-checkpoints", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=2000)
    args = parser.parse_args(argv)
    providers = args.provider or list(PROTOCOLS)
    if not providers or any(value not in PROTOCOLS for value in providers):
        parser.error("provider must be one of the frozen provider names")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    detail_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    provider_rows: list[dict[str, object]] = []
    if args.from_checkpoints:
        for source in providers:
            checkpoint_slug = (
                source.lower().replace(" ", "_").replace("-", "_")
            )
            checkpoint_path = TABLES / (
                f"{PREFIX}__{checkpoint_slug}.json"
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"missing Stage254 checkpoint: {checkpoint_path}"
                )
            checkpoint = load_json(checkpoint_path)
            provider_rows.extend(checkpoint["provider_rows"])
            fit_rows.extend(checkpoint["fit_rows"])
            detail_rows.extend(checkpoint["detail_rows"])
        return write_summary(
            provider_rows,
            fit_rows,
            detail_rows,
            providers,
            bool(args.dry_run),
        )

    case_cache: OrderedDict[str, dict[int, dict[str, object]]] = (
        OrderedDict()
    )
    for source in providers:
        provider_started = time.perf_counter()
        protocol = load_json(PROTOCOLS[source])
        frame, frame_cached = cached_frame(source, protocol)
        base = detector_frame(frame)
        detectors = fit_detectors(base, protocol)
        train_normal = (
            base["split"].eq("train") & base["observation_present"]
        ).to_numpy(dtype=bool)
        base_components = score_components(base, detectors)
        base_fused, _ = fuse_scores(base, base_components)
        base_features = candidate_features(base, train_normal)
        base_matrix = matrix_for(
            base_components,
            base_features,
            train_normal,
        )
        print(
            json.dumps(
                {
                    "source": source,
                    "event": "base_prepared",
                    "rows": int(len(base)),
                    "train_normal_rows": int(train_normal.sum()),
                    "frame_cache_hit": bool(frame_cached),
                    "elapsed_seconds": round(
                        time.perf_counter() - provider_started,
                        3,
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        train_rows = int(train_normal.sum())
        limits = effective_limits(train_rows)
        if args.dry_run:
            cases = {}
        elif source in case_cache:
            cases = case_cache[source]
        else:
            cases, cases_cached = cached_cases(
                source,
                base,
                protocol,
                detectors,
                train_normal,
            )
            case_cache[source] = cases
            print(
                json.dumps(
                    {
                        "source": source,
                        "event": "attack_cases_prepared",
                        "case_count": len(cases),
                        "cache_hit": bool(cases_cached),
                        "elapsed_seconds": round(
                            time.perf_counter() - provider_started,
                            3,
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        provider_fit_count = 0
        for requested_rows in limits:
            model, meta = fit_ocsvm(
                base_matrix,
                base,
                train_normal,
                requested_rows,
            )
            fit_row = {
                "source": source,
                "requested_rows": int(requested_rows),
                "effective_train_rows": int(meta["fit_rows"]),
                "train_normal_rows": int(train_rows),
                "fit_seed": int(FIT_SEED),
                "fit_seconds": float(meta["fit_seconds"]),
                "peak_rss_mb": float(meta["peak_rss_mb"]),
                "support_vectors": int(meta["support_vectors"]),
                "support_vector_ratio": float(
                    meta["support_vector_ratio"]
                ),
                "is_frozen_20000_cap": bool(
                    int(meta["fit_rows"])
                    == min(MAX_OCSVM_FIT_ROWS, train_rows)
                ),
            }
            fit_rows.append(fit_row)
            provider_fit_count += 1
            if not args.dry_run:
                for seed, case in cases.items():
                    detail_rows.append(
                        evaluate_case(
                            model,
                            case,
                            source,
                            int(seed),
                            int(requested_rows),
                            int(meta["fit_rows"]),
                            float(protocol["primary_far"]),
                            int(args.chunk_size),
                        )
                    )
            print(
                json.dumps(
                    {
                        "source": source,
                        "requested_rows": int(requested_rows),
                        "event": "fit_completed",
                        "effective_train_rows": int(meta["fit_rows"]),
                        "fit_seconds": float(meta["fit_seconds"]),
                        "elapsed_seconds": round(
                            time.perf_counter() - provider_started,
                            3,
                        ),
                        "status": "completed",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        provider_rows.append(
            {
                "source": source,
                "train_normal_rows": int(train_rows),
                "fit_count": int(provider_fit_count),
                "requested_limits": [int(value) for value in limits],
            }
        )
        checkpoint = {
            "status": "PROVIDER_CHECKPOINT_STAGE254",
            "source": source,
            "fit_limits": [int(value) for value in FIT_ROWS],
            "fit_seed": int(FIT_SEED),
            "provider_rows": clean_records([provider_rows[-1]]),
            "fit_rows": clean_records(
                [row for row in fit_rows if row["source"] == source]
            ),
            "detail_rows": clean_records(
                [row for row in detail_rows if row["source"] == source]
            ),
            "primary_endpoint": PRIMARY_ENDPOINT,
            "claim_ceiling": CLAIM_CEILING,
        }
        checkpoint_slug = (
            source.lower().replace(" ", "_").replace("-", "_")
        )
        (TABLES / f"{PREFIX}__{checkpoint_slug}.json").write_text(
            json.dumps(
                checkpoint,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    return write_summary(
        provider_rows,
        fit_rows,
        detail_rows,
        providers,
        bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
