from __future__ import annotations

"""Stage 256: exploratory lightweight strong baselines.

This stage adds three unsupervised reconstruction/distance baselines to the
frozen Stage248 evaluation contract:

* ``pca_reconstruction``: PCA reconstruction error.
* ``mlp_autoencoder_reconstruction``: deterministic MLP autoencoder
  reconstruction error.
* ``knn_distance``: mean distance to the nearest train-normal neighbours.

The stage is deliberately isolated from the frozen confirmatory package.
It uses the same providers, splits, transparent controlled attacks, seeds,
and ``FAR=0.01`` endpoint as Stage248, but every new comparison is marked
``exploratory_only`` and ``not_part_of_frozen_main_results``.  No existing
test label is used to choose a feature, window, weight, threshold, or model
configuration.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage256_lightweight_strong_baselines"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
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
from stage240_nonstuck_balanced_selection import (  # noqa: E402
    AUDITS,
    NORMALIZED,
    PROTOCOLS,
    make_frame,
)
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_features,
    select_candidate_stage244,
)

TRUST_METHODS = s248.TRUST_METHODS
EXISTING_BASELINES = s248.BASELINE_METHODS
NEW_BASELINES = [
    "pca_reconstruction",
    "mlp_autoencoder_reconstruction",
    "knn_distance",
]
BASELINES = list(NEW_BASELINES)
METHODS = [*TRUST_METHODS, *BASELINES]
SELECTION_RULE_BY_METHOD = s248.SELECTION_RULE_BY_METHOD

MAX_FIT_ROWS = 100_000
KNN_MAX_FIT_ROWS = 10_000
PCA_COMPONENTS = 12
MLP_HIDDEN_LAYERS = (64, 16, 64)
MLP_MAX_ITER = 200
KNN_NEIGHBOURS = 5
MODEL_SEED = 20260921


def fit_lightweight_baselines(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    train_normal: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    indices = s248.stratified_fit_indices(
        frame,
        train_normal,
        MAX_FIT_ROWS,
        MODEL_SEED,
    )
    train_x = matrix[indices]
    knn_indices = s248.stratified_fit_indices(
        frame,
        train_normal,
        KNN_MAX_FIT_ROWS,
        MODEL_SEED,
    )
    knn_x = matrix[knn_indices]
    pca = make_pipeline(
        StandardScaler(),
        PCA(
            n_components=min(PCA_COMPONENTS, train_x.shape[1]),
            svd_solver="randomized",
            random_state=MODEL_SEED,
        ),
    ).fit(train_x)
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=MLP_MAX_ITER,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=MODEL_SEED,
        ),
    ).fit(train_x, train_x)
    knn = make_pipeline(
        StandardScaler(),
        NearestNeighbors(
            n_neighbors=min(KNN_NEIGHBOURS, len(knn_x)),
            n_jobs=-1,
        ),
    ).fit(knn_x)
    fitted = {
        "pca_reconstruction": pca,
        "mlp_autoencoder_reconstruction": mlp,
        "knn_distance": knn,
    }
    fit_info = {
        "fit_rows": int(len(indices)),
        "train_normal_rows": int(train_normal.sum()),
        "subsampled": bool(len(indices) < int(train_normal.sum())),
        "feature_count": int(matrix.shape[1]),
        "pca_components": int(min(PCA_COMPONENTS, matrix.shape[1])),
        "mlp_hidden_layers": list(MLP_HIDDEN_LAYERS),
        "mlp_iterations": int(mlp.named_steps["mlpregressor"].n_iter_),
        "knn_neighbours": int(min(KNN_NEIGHBOURS, len(knn_x))),
        "knn_fit_rows": int(len(knn_indices)),
        "knn_subsampled": bool(
            int(train_normal.sum()) > KNN_MAX_FIT_ROWS
        ),
        "model_seed": MODEL_SEED,
    }
    return fitted, fit_info


def score_lightweight_baselines(
    fitted: dict[str, object],
    matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    pca = fitted["pca_reconstruction"]
    pca_scaled = pca.named_steps["standardscaler"].transform(matrix)
    pca_reconstructed = pca.named_steps["pca"].inverse_transform(
        pca.named_steps["pca"].transform(pca_scaled)
    )
    pca_error = np.mean((pca_scaled - pca_reconstructed) ** 2, axis=1)

    mlp = fitted["mlp_autoencoder_reconstruction"]
    mlp_scaled = mlp.named_steps["standardscaler"].transform(matrix)
    mlp_reconstructed = mlp.named_steps["mlpregressor"].predict(mlp_scaled)
    mlp_error = np.mean((mlp_scaled - mlp_reconstructed) ** 2, axis=1)

    knn = fitted["knn_distance"]
    knn_scaled = knn.named_steps["standardscaler"].transform(matrix)
    distances = knn.named_steps["nearestneighbors"].kneighbors(
        knn_scaled,
        return_distance=True,
    )[0]
    return {
        "pca_reconstruction": pca_error,
        "mlp_autoencoder_reconstruction": mlp_error,
        "knn_distance": np.mean(distances, axis=1),
    }


def paired_station_table(
    station_metrics: pd.DataFrame,
    baselines: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in baselines:
            trust = station_metrics[
                station_metrics["method"].eq(candidate)
            ]
            reference = station_metrics[
                station_metrics["method"].eq(baseline)
            ]
            aligned = trust.merge(
                reference,
                on=["source", "seed", "station_id"],
                suffixes=("_trust", "_baseline"),
                validate="one_to_one",
            )
            for row in aligned.to_dict("records"):
                rows.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "source": str(row["source"]),
                        "seed": int(row["seed"]),
                        "station_id": str(row["station_id"]),
                        "f1_difference": float(
                            row["f1_trust"] - row["f1_baseline"]
                        ),
                        "event_coverage_difference": float(
                            row["event_coverage_trust"]
                            - row["event_coverage_baseline"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []

    for source, protocol_path in PROTOCOLS.items():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        frame = make_frame(source, protocol)
        base = detector_frame(frame)
        detectors = fit_detectors(base, protocol)
        train_normal = (
            base["split"].eq("train") & base["observation_present"]
        ).to_numpy(dtype=bool)
        base_components = score_components(base, detectors)
        base_fused, _ = fuse_scores(base, base_components)
        base_z = robust_standardize(base_fused, train_normal).to_numpy(
            dtype=float
        )
        base_features = candidate_features(base, train_normal)
        base_matrix, feature_names = s248.feature_matrix(
            base_components,
            base_features,
            train_normal,
        )
        fitted_lightweight, fit_info = fit_lightweight_baselines(
            base,
            base_matrix,
            train_normal,
        )
        primary_far = float(protocol["primary_far"])
        station_ids = sorted(
            frame["station_id"].astype(str).unique().tolist()
        )
        fit_rows.append(
            {
                "source": source,
                "feature_names": feature_names,
                **fit_info,
            }
        )

        for seed in protocol["seeds"]:
            seed = int(seed)
            valid_blocks = choose_split_attack_blocks(
                base, protocol, seed, "valid"
            )
            test_blocks = choose_split_attack_blocks(
                base, protocol, seed, "test"
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
            valid_fused, _ = fuse_scores(valid_attacked, valid_components)
            valid_base_z = robust_standardize(
                valid_fused, train_normal
            ).to_numpy(dtype=float)
            test_components = score_components(test_attacked, detectors)
            test_fused, _ = fuse_scores(test_attacked, test_components)
            test_base_z = robust_standardize(
                test_fused, train_normal
            ).to_numpy(dtype=float)
            valid_features = candidate_features(valid_attacked, train_normal)
            test_features = candidate_features(test_attacked, train_normal)
            valid_matrix, _ = s248.feature_matrix(
                valid_components,
                valid_features,
                train_normal,
            )
            test_matrix, _ = s248.feature_matrix(
                test_components,
                test_features,
                train_normal,
            )
            valid_mask = valid_attacked["split"].eq("valid").to_numpy(
                dtype=bool
            )
            test_mask = test_attacked["split"].eq("test").to_numpy(
                dtype=bool
            )
            valid_frame = valid_attacked.loc[valid_mask].reset_index(
                drop=True
            )
            test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
            if not valid_frame["is_attack"].any():
                raise RuntimeError(f"{source} valid attack injection is empty")
            if not test_frame["is_attack"].any():
                raise RuntimeError(f"{source} test attack injection is empty")
            valid_normal = (
                valid_frame["observation_present"]
                & ~valid_frame["is_attack"]
            ).to_numpy(dtype=bool)
            valid_base = valid_base_z[valid_mask]
            test_base = test_base_z[test_mask]
            valid_lightweight = score_lightweight_baselines(
                fitted_lightweight,
                valid_matrix[valid_mask],
            )
            test_lightweight = score_lightweight_baselines(
                fitted_lightweight,
                test_matrix[test_mask],
            )
            valid_feature_masked = {
                key: values[valid_mask]
                for key, values in valid_features.items()
            }
            test_feature_masked = {
                key: values[test_mask]
                for key, values in test_features.items()
            }

            valid_scores: dict[str, np.ndarray] = {}
            test_scores: dict[str, np.ndarray] = {}
            for method, rule in SELECTION_RULE_BY_METHOD.items():
                selected = select_candidate_stage244(
                    rule,
                    valid_base,
                    valid_feature_masked,
                    valid_frame,
                    valid_normal,
                    primary_far,
                )
                key = (
                    str(selected["feature"]),
                    int(selected["window"]),
                )
                test_scores[method] = (
                    (1.0 - float(selected["weight"])) * test_base
                    + float(selected["weight"]) * test_feature_masked[key]
                )
                valid_scores[method] = (
                    (1.0 - float(selected["weight"])) * valid_base
                    + float(selected["weight"]) * valid_feature_masked[key]
                )
                selection_rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "method": method,
                        "rule": rule,
                        "feature": selected["feature"],
                        "window": int(selected["window"]),
                        "weight": float(selected["weight"]),
                        "threshold": float(selected["threshold"]),
                    }
                )
            for method in NEW_BASELINES:
                valid_scores[method] = valid_lightweight[method]
                test_scores[method] = test_lightweight[method]

            y_test = test_frame["is_attack"].astype(bool).to_numpy()
            station_values = test_frame["station_id"].astype(str).to_numpy()
            for method in METHODS:
                valid_score = np.asarray(valid_scores[method], dtype=float)
                test_score = np.asarray(test_scores[method], dtype=float)
                threshold = far_threshold(
                    valid_score,
                    valid_normal,
                    primary_far,
                )
                predicted = (
                    test_frame["observation_present"].to_numpy(dtype=bool)
                    & (test_score >= threshold)
                )
                row_metrics = metrics(y_test, predicted, test_score)
                event_values = event_metrics(test_frame, predicted)
                type_values = type_metrics(
                    test_frame,
                    predicted,
                    test_score,
                )
                evaluation_rows.append(
                    {
                        "source": source,
                        "seed": seed,
                        "method": method,
                        "threshold": float(threshold),
                        **row_metrics,
                        **event_values,
                    }
                )
                for attack_type, values in type_values.items():
                    type_rows.append(
                        {
                            "source": source,
                            "seed": seed,
                            "method": method,
                            "attack_type": attack_type,
                            **values,
                        }
                    )
                event_frame = test_frame.loc[y_test].copy()
                event_frame["predicted"] = predicted[y_test]
                event_grouped = (
                    event_frame.groupby(
                        ["station_id", "attack_event_id"],
                        sort=True,
                        as_index=False,
                    )["predicted"]
                    .any()
                )
                for station_id in station_ids:
                    mask = station_values == station_id
                    y_station = y_test[mask]
                    pred_station = predicted[mask]
                    station_events = event_grouped[
                        event_grouped["station_id"].eq(station_id)
                    ]
                    station_rows.append(
                        {
                            "source": source,
                            "seed": seed,
                            "method": method,
                            "station_id": station_id,
                            "rows": int(mask.sum()),
                            "attack_rows": int(y_station.sum()),
                            "normal_rows": int((~y_station).sum()),
                            "tp": int((y_station & pred_station).sum()),
                            "fp": int((~y_station & pred_station).sum()),
                            "fn": int((y_station & ~pred_station).sum()),
                            "tn": int((~y_station & ~pred_station).sum()),
                            "events": int(len(station_events)),
                            "detected_events": int(
                                station_events["predicted"].sum()
                            ),
                        }
                    )
        print(
            json.dumps(
                {
                    "source": source,
                    "status": "completed",
                    "seeds": len(protocol["seeds"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    evaluation = pd.DataFrame(evaluation_rows)
    station = pd.DataFrame(station_rows)
    type_detail = pd.DataFrame(type_rows)
    selections = pd.DataFrame(selection_rows)
    evaluation_aggregate = (
        evaluation.groupby(["source", "method"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            false_alarm_rate_mean=("false_alarm_rate", "mean"),
            average_precision_mean=("average_precision", "mean"),
            event_coverage_mean=("event_coverage", "mean"),
            event_coverage_std=("event_coverage", "std"),
            row_recall_mean=("row_recall", "mean"),
        )
        .sort_values(["source", "method"])
    )
    provider_equal = (
        evaluation_aggregate.groupby("method", as_index=False)
        .agg(
            provider_count=("source", "nunique"),
            f1_mean=("f1_mean", "mean"),
            event_coverage_mean=("event_coverage_mean", "mean"),
            row_recall_mean=("row_recall_mean", "mean"),
        )
        .sort_values("method")
    )
    station_metrics = s248.station_metric_table(station)
    paired = paired_station_table(station_metrics, BASELINES)
    comparison_rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in BASELINES:
            comparison_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    **s248.hierarchical_bootstrap(
                        paired,
                        candidate,
                        baseline,
                    ),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["f1_bootstrap_bh_q"] = s248.benjamini_hochberg(
        comparisons["f1_bootstrap_two_sided_p"].tolist()
    )
    comparisons["event_coverage_bootstrap_bh_q"] = s248.benjamini_hochberg(
        comparisons["event_coverage_bootstrap_two_sided_p"].tolist()
    )
    comparisons["f1_primary_supported"] = (
        comparisons["f1_ci95_low"].gt(0.0)
        & comparisons["f1_bootstrap_bh_q"].le(0.05)
    )
    comparisons["event_coverage_primary_supported"] = (
        comparisons["event_ci95_low"].gt(0.0)
        & comparisons["event_coverage_bootstrap_bh_q"].le(0.05)
    )
    leave_one_rows: list[dict[str, object]] = []
    for candidate in TRUST_METHODS:
        for baseline in NEW_BASELINES:
            leave_one_rows.extend(
                s248.leave_one_provider_bootstrap(
                    paired,
                    candidate,
                    baseline,
                )
            )
    leave_one = pd.DataFrame(leave_one_rows)
    type_aggregate = (
        type_detail.groupby(
            ["source", "method", "attack_type"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            attack_rows=("attack_rows", "sum"),
            detected_rows=("detected_rows", "sum"),
            row_recall=("row_recall", "mean"),
            event_coverage=("event_coverage", "mean"),
        )
        .sort_values(["source", "method", "attack_type"])
    )
    payload = {
        "status": "PASS_EXPLORATORY_BASELINE_AUDIT",
        "exploratory_only": True,
        "not_part_of_frozen_main_results": True,
        "protocol": (
            "阶段256在阶段248冻结的四个目标机构、train/valid/test边界、"
            "透明受控攻击和FAR=0.01主终点上补充三个轻量强基线。"
            "PCA、MLP自编码器和kNN都只在train正常行拟合；阈值只由"
            "valid正常行校准；test只评价。所有新结果均标为探索性，"
            "不替换阶段246/248的确认性主结果。"
        ),
        "primary_far": 0.01,
        "primary_endpoint": "event_coverage",
        "provider_count": int(len(PROTOCOLS)),
        "station_count": int(
            station[["source", "station_id"]].drop_duplicates().shape[0]
        ),
        "seed_count": int(evaluation["seed"].nunique()),
        "method_count": int(len(METHODS)),
        "methods": METHODS,
        "new_baseline_methods": NEW_BASELINES,
        "existing_baseline_methods": EXISTING_BASELINES,
        "fit_cap": MAX_FIT_ROWS,
        "knn_fit_cap": KNN_MAX_FIT_ROWS,
        "fit_rows": fit_rows,
        "evaluation_rows": int(len(evaluation)),
        "station_rows": int(len(station)),
        "type_rows": int(len(type_detail)),
        "selection_rows": int(len(selections)),
        "provider_equal": s248.clean_records(
            provider_equal.to_dict("records")
        ),
        "comparisons": s248.clean_records(
            comparisons.to_dict("records")
        ),
        "leave_one_provider_out": s248.clean_records(
            leave_one.to_dict("records")
        ),
        "attack_parameters": ATTACK_PARAMETERS,
        "components": COMPONENTS,
        "protocol_sha256": {
            source: sha256_file(path)
            for source, path in PROTOCOLS.items()
        },
        "normalized_sha256": {
            source: sha256_file(path)
            for source, path in NORMALIZED.items()
        },
        "audit_sha256": {
            source: sha256_file(path)
            for source, path in AUDITS.items()
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "limitations": [
            "这是探索性补充基线审计，不替换阶段246/248的确认性主结果。",
            "所有模型只在train正常行拟合，使用固定100k分层上限。",
            "kNN为控制近邻查询成本使用固定10k train-normal核心集。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
            "如果新基线占优，结果必须原样保留。",
            "四个机构不能支持机构无关或站点无关的普适声称。",
            "在线因果恢复不等同于离线双向插值。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    evaluation.to_csv(
        TABLES / f"{PREFIX}_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station.to_csv(
        TABLES / f"{PREFIX}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_detail.to_csv(
        TABLES / f"{PREFIX}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selections.to_csv(
        TABLES / f"{PREFIX}_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evaluation_aggregate.to_csv(
        TABLES / f"{PREFIX}_evaluation_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provider_equal.to_csv(
        TABLES / f"{PREFIX}_provider_equal.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparisons.to_csv(
        TABLES / f"{PREFIX}_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one.to_csv(
        TABLES / f"{PREFIX}_leave_one_provider_out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_aggregate.to_csv(
        TABLES / f"{PREFIX}_type_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段256 轻量强基线探索性补充",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 探索性：`{payload['exploratory_only']}`",
        f"- 不属于冻结主结果：`{payload['not_part_of_frozen_main_results']}`",
        f"- 机构数：`{payload['provider_count']}`",
        f"- 站点数：`{payload['station_count']}`",
        f"- 种子数：`{payload['seed_count']}`",
        f"- 新增基线：`{', '.join(NEW_BASELINES)}`",
        "",
        "## 边界",
        "",
        "- 新基线只在train正常行拟合，固定100k分层上限。",
        "- 阈值只由valid正常行校准，test只评价。",
        "- 所有攻击均为透明受控注入，不是真实攻击日志。",
        "- 本阶段不能升级为确认性主结果，也不能宣称机构无关。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "provider_count": payload["provider_count"],
                "station_count": payload["station_count"],
                "method_count": payload["method_count"],
                "new_baselines": NEW_BASELINES,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
