from __future__ import annotations

"""Stage 274: exploratory full-scale One-Class SVM sensitivity.

Stage254 froze 20k, 50k, and 100k fit budgets and showed that the
One-Class SVM complementarity conclusion does not change at those budgets.
This stage answers the remaining resource question by fitting the same frozen
RBF One-Class SVM on every train-normal row for each provider.

Important boundary: this stage is exploratory.  The old test has already
been used for Stage254 reporting, so no result here is promoted to the
confirmatory main table or used to select a model, feature, threshold, or
attack parameter.  Every provider is fitted on train-normal rows only;
the valid-normal FAR=0.01 threshold and test evaluation remain unchanged.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
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
PREFIX = "stage274_ocsvm_full_scale_sensitivity"
FULL_LIMIT = "ALL_TRAIN_NORMAL"
FIT_SEED = 20260919
CLAIM_CEILING = "EXPLORATORY_ONLY"
MODEL_CACHE = ROOT / "testbed" / "stage274_models"
FIT_CACHE_MB = max(
    1000,
    int(os.environ.get("TMQC_OCSVM_CACHE_MB", "1000")),
)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage254_ocsvm_training_scale_sensitivity as stage254  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    ATTACK_PARAMETERS,
    COMPONENTS,
    detector_frame,
    fit_detectors,
    score_components,
    sha256_file,
)
from stage217_swiss_controlled_attack_evaluation import (  # noqa: E402
    fuse_scores,
)
from stage219_swiss_dwd_protection_transfer import clean_records  # noqa: E402
from stage240_nonstuck_balanced_selection import (  # noqa: E402
    AUDITS,
    NORMALIZED,
    PROTOCOLS,
)
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_features,
)


def slug(source: str) -> str:
    return source.lower().replace(" ", "_").replace("-", "_")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_fingerprint(source: str) -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        PROTOCOLS[source],
        NORMALIZED[source],
        AUDITS[source],
    ):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(f"|cache_size_mb={FIT_CACHE_MB}".encode("ascii"))
    digest.update(f"|fit_seed={FIT_SEED}".encode("ascii"))
    return digest.hexdigest()


def full_fit_indices(
    frame: pd.DataFrame,
    train_normal: np.ndarray,
) -> np.ndarray:
    """Return every train-normal row in deterministic frame order."""
    positions = np.flatnonzero(train_normal)
    if positions.size == 0:
        raise ValueError("provider has no train-normal rows")
    return positions.astype(int, copy=False)


def fit_full_ocsvm(
    matrix: np.ndarray,
    indices: np.ndarray,
    source: str,
) -> tuple[object, dict[str, object]]:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = MODEL_CACHE / f"{slug(source)}.joblib"
    fingerprint = model_fingerprint(source)
    if cache_path.is_file():
        cached = joblib.load(cache_path)
        if cached.get("fingerprint") == fingerprint:
            print(
                json.dumps(
                    {
                        "source": source,
                        "event": "full_model_cache_hit",
                        "fit_rows": int(indices.size),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return cached["model"], cached["meta"]
    started = time.perf_counter()
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(60.0):
            print(
                json.dumps(
                    {
                        "source": source,
                        "event": "fit_heartbeat",
                        "fit_rows": int(indices.size),
                        "elapsed_seconds": round(
                            time.perf_counter() - started,
                            3,
                        ),
                        "peak_rss_mb": float(stage254.peak_rss_mb()),
                        "cache_size_mb": int(FIT_CACHE_MB),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        daemon=True,
    )
    heartbeat_thread.start()
    model = make_pipeline(
        StandardScaler(),
        OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05,
            cache_size=int(FIT_CACHE_MB),
            max_iter=-1,
        ),
    ).fit(matrix[indices])
    stop_heartbeat.set()
    heartbeat_thread.join(timeout=2.0)
    elapsed = time.perf_counter() - started
    support_vectors = int(np.asarray(model[-1].support_).size)
    meta = {
        "fit_rows": int(indices.size),
        "fit_seconds": float(elapsed),
        "peak_rss_mb": float(stage254.peak_rss_mb()),
        "support_vectors": support_vectors,
        "support_vector_ratio": float(
            support_vectors / max(1, indices.size)
        ),
        "cache_size_mb": int(FIT_CACHE_MB),
    }
    joblib.dump(
        {
            "fingerprint": fingerprint,
            "source": source,
            "meta": meta,
            "model": model,
        },
        cache_path,
        compress=3,
    )
    return model, meta


def fit_row(
    source: str,
    train_rows: int,
    meta: dict[str, object],
) -> dict[str, object]:
    return {
        "source": source,
        "fit_mode": FULL_LIMIT,
        "requested_rows": int(train_rows),
        "effective_train_rows": int(meta["fit_rows"]),
        "train_normal_rows": int(train_rows),
        "fit_seed": int(FIT_SEED),
        "fit_seconds": float(meta["fit_seconds"]),
        "peak_rss_mb": float(meta["peak_rss_mb"]),
        "support_vectors": int(meta["support_vectors"]),
        "support_vector_ratio": float(meta["support_vector_ratio"]),
        "cache_size_mb": int(meta["cache_size_mb"]),
        "is_full_train_normal": True,
    }


def evaluate_provider(
    source: str,
    protocol: dict,
    base: pd.DataFrame,
    base_matrix: np.ndarray,
    train_normal: np.ndarray,
    cases: dict[int, dict[str, object]],
    chunk_size: int,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    indices = full_fit_indices(base, train_normal)
    model, meta = fit_full_ocsvm(base_matrix, indices, source)
    fit_record = fit_row(source, int(indices.size), meta)
    detail: list[dict[str, object]] = []
    for seed, case in cases.items():
        row = stage254.evaluate_case(
            model,
            case,
            source,
            int(seed),
            int(indices.size),
            int(meta["fit_rows"]),
            float(protocol["primary_far"]),
            int(chunk_size),
        )
        row.update(
            {
                "fit_mode": FULL_LIMIT,
                "is_full_train_normal": True,
                "claim_ceiling": CLAIM_CEILING,
            }
        )
        detail.append(row)
    provider = {
        "source": source,
        "fit_mode": FULL_LIMIT,
        "train_normal_rows": int(indices.size),
        "fit_rows": int(indices.size),
        "seed_count": int(len(detail)),
        "fit_seconds": float(meta["fit_seconds"]),
        "peak_rss_mb": float(meta["peak_rss_mb"]),
        "support_vectors": int(meta["support_vectors"]),
        "support_vector_ratio": float(meta["support_vector_ratio"]),
        "cache_size_mb": int(meta["cache_size_mb"]),
    }
    return provider, fit_record, detail


def write_summary(
    provider_rows: list[dict[str, object]],
    fit_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    providers: list[str],
    dry_run: bool,
    write_artifacts: bool = True,
) -> int:
    fits = pd.DataFrame(fit_rows)
    if dry_run:
        payload = {
            "status": "DRY_RUN_STAGE274_OCSVM_FULL_SCALE_SENSITIVITY",
            "protocol": (
                "仅登记全量train-normal拟合规模；不评价test，不改变"
                "Stage246/248/254结果。"
            ),
            "fit_mode": FULL_LIMIT,
            "fit_seed": int(FIT_SEED),
            "provider_rows": provider_rows,
            "fit_rows": clean_records(fits.to_dict("records")),
            "fit_row_count": int(len(fits)),
            "claim_ceiling": CLAIM_CEILING,
        }
    else:
        detail = pd.DataFrame(detail_rows)
        if detail.empty:
            raise RuntimeError("stage274 produced no evaluation rows")
        aggregate = (
            detail.groupby(["source", "fit_mode"], as_index=False)
            .agg(
                seed_count=("seed", "nunique"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
                recall_mean=("recall", "mean"),
                false_alarm_rate_mean=("false_alarm_rate", "mean"),
                event_coverage_mean=("event_coverage", "mean"),
                event_coverage_std=("event_coverage", "std"),
                row_recall_mean=("row_recall", "mean"),
            )
            .sort_values("source")
        )
        provider_equal = pd.DataFrame(
            [
                {
                    "provider_count": int(aggregate["source"].nunique()),
                    "fit_mode": FULL_LIMIT,
                    "f1_mean": float(aggregate["f1_mean"].mean()),
                    "event_coverage_mean": float(
                        aggregate["event_coverage_mean"].mean()
                    ),
                    "event_coverage_min": float(
                        aggregate["event_coverage_mean"].min()
                    ),
                    "event_coverage_max": float(
                        aggregate["event_coverage_mean"].max()
                    ),
                    "row_recall_mean": float(
                        aggregate["row_recall_mean"].mean()
                    ),
                }
            ]
        )
        payload = {
            "status": "PASS_OCSVM_FULL_SCALE_EXPLORATORY",
            "protocol": (
                "阶段274在Stage254之后执行，固定每个机构的One-Class SVM"
                "训练集合为全部train-normal行；复用相同特征、攻击、"
                "valid正常样本FAR=0.01阈值、五个种子和test评价代码。"
                "该阶段是探索性全量预算敏感性，不回写确认性主表，"
                "不用于选择模型、特征、阈值或攻击参数。"
            ),
            "claim_ceiling": CLAIM_CEILING,
            "fit_mode": FULL_LIMIT,
            "fit_seed": int(FIT_SEED),
            "provider_count": int(len(provider_rows)),
            "provider_rows": provider_rows,
            "fit_rows": clean_records(fits.to_dict("records")),
            "fit_row_count": int(len(fits)),
            "detail_rows": int(len(detail)),
            "aggregate_rows": int(len(aggregate)),
            "provider_equal": clean_records(
                provider_equal.to_dict("records")
            ),
            "attack_parameters": ATTACK_PARAMETERS,
            "components": COMPONENTS,
            "protocol_sha256": {
                source: sha256_file(PROTOCOLS[source])
                for source in providers
            },
            "normalized_sha256": {
                source: sha256_file(NORMALIZED[source])
                for source in providers
            },
            "audit_sha256": {
                source: sha256_file(AUDITS[source])
                for source in providers
            },
            "code_sha256": sha256_file(Path(__file__).resolve()),
            "limitations": [
                "旧test已用于Stage254，因此本阶段只能作为探索性证据，不能升级确认性主张。",
                "全量拟合只覆盖train-normal行；valid只用于FAR阈值，test只评价。",
                "所有攻击仍是透明受控注入，不是真实攻击日志。",
                "若全量档位仍呈互补或负向结果，必须保留，不得删除或选择性报告。",
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
            "# 阶段274 One-Class SVM 全量训练探索性敏感性",
            "",
            f"- 状态：**{payload['status']}**",
            "- 训练集合：全部 train-normal 行（逐机构）",
            f"- 固定拟合种子：`{FIT_SEED}`",
            f"- provider数：`{payload['provider_count']}`；详情行：`{payload['detail_rows']}`",
            "",
            "## 机构等权结果",
            "",
            "| 机构数 | F1均值 | 事件覆盖均值 | 事件覆盖最小 | 事件覆盖最大 | 行召回均值 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in provider_equal.to_dict("records"):
            lines.append(
                f"| {int(row['provider_count'])} | "
                f"{row['f1_mean']:.6f} | "
                f"{row['event_coverage_mean']:.6f} | "
                f"{row['event_coverage_min']:.6f} | "
                f"{row['event_coverage_max']:.6f} | "
                f"{row['row_recall_mean']:.6f} |"
            )
        lines += [
            "",
            "## 逐机构拟合规模",
            "",
            "| 机构 | 全量拟合行数 | 拟合秒 | 峰值RSS MB | 支持向量 |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in fits.sort_values("source").to_dict("records"):
            lines.append(
                f"| {row['source']} | {int(row['effective_train_rows'])} | "
                f"{row['fit_seconds']:.3f} | "
                f"{row['peak_rss_mb']:.3f} | "
                f"{int(row['support_vectors'])} |"
            )
        lines += [
            "",
            "## 边界",
            "",
            "- 这是旧 test 上的探索性全量预算敏感性，不是确认性主结果。",
            "- 不替换 Stage246/248，也不改变 Stage254 的预注册 20k/50k/100k 曲线。",
            "- 所有结果必须完整保留，包括全量训练仍未改善互补关系的负结果。",
        ]
        (REPORTS / f"{PREFIX}.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    if write_artifacts:
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
            },
            ensure_ascii=False,
        )
    )
    return 0


def write_resource_summary(
    provider_rows: list[dict[str, object]],
    fit_rows: list[dict[str, object]],
    providers: list[str],
) -> int:
    fits = pd.DataFrame(fit_rows)
    fits["peak_rss_scope"] = "shared_process_cumulative_peak"
    fits["provider_attributable_peak"] = False
    payload = {
        "status": "PASS_OCSVM_FULL_SCALE_RESOURCE_AUDIT",
        "protocol": (
            "仅重放Stage274冻结的One-Class SVM全量train-normal拟合，"
            "记录可复核的PeakWorkingSetSize、拟合时间和支持向量；"
            "不读取valid/test，不执行攻击注入或评价，不回写主结果。"
        ),
        "claim_ceiling": CLAIM_CEILING,
        "fit_mode": FULL_LIMIT,
        "fit_seed": int(FIT_SEED),
        "cache_size_mb": int(FIT_CACHE_MB),
        "provider_count": int(len(provider_rows)),
        "provider_rows": provider_rows,
        "fit_rows": clean_records(fits.to_dict("records")),
        "fit_row_count": int(len(fits)),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "limitations": [
            "资源审计在旧test上没有新增评价，不能升级任何性能主张。",
            "PeakWorkingSetSize是当前Windows主机的进程峰值工作集，不是受控跨平台内存基准。",
            "四个机构在同一进程内顺序拟合，因此前三机构记录的是此前最高水位；只有最高值可归因于触发它的机构，不能把1568.66796875 MB解释为每个机构的独立峰值。",
            "全量拟合只覆盖train-normal行，未接触valid/test标签或攻击评价。",
        ],
    }
    (TABLES / f"{PREFIX}_resource_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    fits.to_csv(
        TABLES / f"{PREFIX}_resource_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "provider_rows": len(provider_rows),
                "fit_rows": len(fit_rows),
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
    parser.add_argument("--resource-only", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=2000)
    args = parser.parse_args(argv)
    providers = args.provider or list(PROTOCOLS)
    if not providers or any(value not in PROTOCOLS for value in providers):
        parser.error("provider must be one of the frozen provider names")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    provider_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    if args.from_checkpoints:
        for source in providers:
            checkpoint_path = TABLES / f"{PREFIX}__{slug(source)}.json"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"missing Stage274 checkpoint: {checkpoint_path}"
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
            write_artifacts=not args.dry_run,
        )

    if args.resource_only:
        resource_providers: list[dict[str, object]] = []
        resource_fits: list[dict[str, object]] = []
        for source in providers:
            protocol = load_json(PROTOCOLS[source])
            frame, _ = stage254.cached_frame(source, protocol)
            base = detector_frame(frame)
            train_normal = (
                base["split"].eq("train") & base["observation_present"]
            ).to_numpy(dtype=bool)
            indices = full_fit_indices(base, train_normal)
            components = score_components(
                base,
                fit_detectors(base, protocol),
            )
            features = candidate_features(base, train_normal)
            matrix = stage254.matrix_for(
                components,
                features,
                train_normal,
            )
            _, meta = fit_full_ocsvm(matrix, indices, source)
            resource_providers.append(
                {
                    "source": source,
                    "fit_mode": FULL_LIMIT,
                    "train_normal_rows": int(indices.size),
                    "fit_rows": int(indices.size),
                    "seed_count": 0,
                    "fit_seconds": float(meta["fit_seconds"]),
                    "peak_rss_mb": float(meta["peak_rss_mb"]),
                    "support_vectors": int(meta["support_vectors"]),
                    "support_vector_ratio": float(
                        meta["support_vector_ratio"]
                    ),
                    "cache_size_mb": int(meta["cache_size_mb"]),
                }
            )
            resource_fits.append(
                {
                    **fit_row(source, int(indices.size), meta),
                    "evidence_scope": "fit_only_no_valid_no_test",
                }
            )
        return write_resource_summary(
            resource_providers,
            resource_fits,
            providers,
        )

    for source in providers:
        provider_started = time.perf_counter()
        protocol = load_json(PROTOCOLS[source])
        frame, frame_cached = stage254.cached_frame(source, protocol)
        base = detector_frame(frame)
        detectors = fit_detectors(base, protocol)
        train_normal = (
            base["split"].eq("train") & base["observation_present"]
        ).to_numpy(dtype=bool)
        base_components = score_components(base, detectors)
        base_fused, _ = fuse_scores(base, base_components)
        base_features = candidate_features(base, train_normal)
        base_matrix = stage254.matrix_for(
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
        if args.dry_run:
            provider = {
                "source": source,
                "fit_mode": FULL_LIMIT,
                "train_normal_rows": int(train_normal.sum()),
                "fit_rows": int(train_normal.sum()),
                "seed_count": 0,
                "fit_seconds": None,
                "peak_rss_mb": None,
                "support_vectors": None,
                "support_vector_ratio": None,
            }
            record = {
                "source": source,
                "fit_mode": FULL_LIMIT,
                "requested_rows": int(train_normal.sum()),
                "effective_train_rows": int(train_normal.sum()),
                "train_normal_rows": int(train_normal.sum()),
                "fit_seed": int(FIT_SEED),
                "fit_seconds": None,
                "peak_rss_mb": None,
                "support_vectors": None,
                "support_vector_ratio": None,
                "is_full_train_normal": True,
            }
            provider_rows.append(provider)
            fit_rows.append(record)
            checkpoint = {
                "status": "PROVIDER_CHECKPOINT_STAGE274_DRY_RUN",
                "source": source,
                "provider_rows": clean_records([provider]),
                "fit_rows": clean_records([record]),
                "detail_rows": [],
                "claim_ceiling": CLAIM_CEILING,
            }
        else:
            cases = stage254.cached_cases(
                source,
                base,
                protocol,
                detectors,
                train_normal,
            )[0]
            provider, record, detail = evaluate_provider(
                source,
                protocol,
                base,
                base_matrix,
                train_normal,
                cases,
                int(args.chunk_size),
            )
            provider_rows.append(provider)
            fit_rows.append(record)
            detail_rows.extend(detail)
            checkpoint = {
                "status": "PROVIDER_CHECKPOINT_STAGE274",
                "source": source,
                "fit_mode": FULL_LIMIT,
                "fit_seed": int(FIT_SEED),
                "provider_rows": clean_records([provider]),
                "fit_rows": clean_records([record]),
                "detail_rows": clean_records(detail),
                "claim_ceiling": CLAIM_CEILING,
            }
        if not args.dry_run:
            (TABLES / f"{PREFIX}__{slug(source)}.json").write_text(
                json.dumps(
                    checkpoint,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "source": source,
                    "event": "provider_completed",
                    "fit_rows": int(
                        record["effective_train_rows"]
                    ),
                    "fit_seconds": record["fit_seconds"],
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

    if args.dry_run:
        return write_summary(
            provider_rows,
            fit_rows,
            detail_rows,
            providers,
            True,
            write_artifacts=False,
        )
    if set(providers) != set(PROTOCOLS):
        print(
            json.dumps(
                {
                    "status": "PROVIDER_CHECKPOINTS_COMPLETE",
                    "provider_rows": len(provider_rows),
                    "fit_rows": len(fit_rows),
                    "detail_rows": len(detail_rows),
                    "summary_written": False,
                    "reason": (
                        "Only provider checkpoints are written for a subset "
                        "run; run --from-checkpoints to rebuild the formal "
                        "four-provider summary."
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0
    return write_summary(
        provider_rows,
        fit_rows,
        detail_rows,
        providers,
        False,
        write_artifacts=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
