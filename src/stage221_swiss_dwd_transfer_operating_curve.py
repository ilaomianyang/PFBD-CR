from __future__ import annotations

"""Stage 221: frozen operating curve for the DWD-to-Swiss transfer.

Stage 219 compares a DWD feature/window/weight transfer whose threshold
is recalibrated on Swiss valid-normal rows with a Swiss-valid local
selection.  The primary result was evaluated at the pre-registered FAR
of 0.01.  This stage asks whether the selected-minus-recalibrated
direction is stable across the frozen grid ``0.005/0.01/0.02/0.05``.

The implementation reuses the exact Stage219 detector fitting, attack
injection, feature construction, and candidate grid.  The only change is
that all frozen FAR values are evaluated instead of only the primary one.
Test labels remain evaluation-only, and the primary operating point stays
at 0.01.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    detector_frame,
    fit_detectors,
    inject_attack,
    metrics,
    sha256_file,
    score_components,
)
from stage208_dwd_persistent_repeat_protection import (  # noqa: E402
    FEATURES,
    VALID_INJECTION_SEED_OFFSET,
    WINDOWS,
    candidate_grid,
    choose_split_attack_blocks,
    event_metrics,
    far_threshold,
    repeat_feature,
    selection_key,
    type_metrics,
)
from stage217_swiss_controlled_attack_evaluation import (  # noqa: E402
    fuse_scores,
    robust_standardize,
)
from stage218_swiss_negative_result_uncertainty import (  # noqa: E402
    benjamini_hochberg,
    cluster_bootstrap_mean,
    exact_sign_flip_two_sided,
)
from stage219_swiss_dwd_protection_transfer import (  # noqa: E402
    clean_records,
    frozen_dwd_selection,
)

PROTOCOL = TABLES / "stage214_swiss_protocol_preregistration.json"
NORMALIZED = TABLES / "stage216_swiss_normalized.parquet"
CORRECTION = TABLES / "stage216a_swiss_protocol_correction.json"
DWD_SELECTION = (
    TABLES / "stage208_dwd_persistent_repeat_protection_selection.csv"
)
STAGE219_JSON = TABLES / "stage219_swiss_dwd_protection_transfer.json"
PREFIX = "stage221_swiss_dwd_transfer_operating_curve"

ARMS = [
    "dwd_threshold_transfer",
    "swiss_valid_recalibrated",
    "swiss_valid_selected",
]
METRICS = [
    "f1",
    "recall",
    "precision",
    "false_alarm_rate",
]
EVENT_METRICS = ["event_coverage"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def station_confusion(
    frame: pd.DataFrame,
    predicted: np.ndarray,
) -> pd.DataFrame:
    predicted = np.asarray(predicted, dtype=bool)
    rows: list[dict[str, object]] = []
    for station_id, part in frame.groupby("station_id", sort=True):
        mask = frame["station_id"].eq(station_id).to_numpy(dtype=bool)
        truth = frame.loc[mask, "is_attack"].astype(bool).to_numpy()
        station_pred = predicted[mask]
        rows.append(
            {
                "station_id": str(station_id),
                "rows": int(mask.sum()),
                "attack_rows": int(truth.sum()),
                "normal_rows": int((~truth).sum()),
                "tp": int((truth & station_pred).sum()),
                "fp": int((~truth & station_pred).sum()),
                "fn": int((truth & ~station_pred).sum()),
                "tn": int((~truth & ~station_pred).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)


def station_event_table(
    frame: pd.DataFrame,
    predicted: np.ndarray,
) -> pd.DataFrame:
    predicted = np.asarray(predicted, dtype=bool)
    rows: list[dict[str, object]] = []
    for station_id, part in frame.groupby("station_id", sort=True):
        mask = frame["station_id"].eq(station_id).to_numpy(dtype=bool)
        attack = frame.loc[mask, "is_attack"].astype(bool).to_numpy()
        station_pred = predicted[mask]
        event_part = frame.loc[
            mask & frame["is_attack"].astype(bool),
            ["attack_event_id"],
        ].copy()
        event_part["predicted"] = station_pred[attack]
        grouped = event_part.groupby("attack_event_id", sort=True)[
            "predicted"
        ].any()
        rows.append(
            {
                "station_id": str(station_id),
                "events": int(len(grouped)),
                "detected_events": int(grouped.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)


def station_metric_table(confusion: pd.DataFrame) -> pd.DataFrame:
    result = confusion.copy()
    precision = np.divide(
        result["tp"],
        result["tp"] + result["fp"],
        out=np.zeros(len(result), dtype=float),
        where=(result["tp"] + result["fp"]).to_numpy() > 0,
    )
    recall = np.divide(
        result["tp"],
        result["tp"] + result["fn"],
        out=np.zeros(len(result), dtype=float),
        where=(result["tp"] + result["fn"]).to_numpy() > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(result), dtype=float),
        where=(precision + recall) > 0,
    )
    false_alarm_rate = np.divide(
        result["fp"],
        result["fp"] + result["tn"],
        out=np.zeros(len(result), dtype=float),
        where=(result["fp"] + result["tn"]).to_numpy() > 0,
    )
    result["precision"] = precision
    result["recall"] = recall
    result["f1"] = f1
    result["false_alarm_rate"] = false_alarm_rate
    return result


def aggregate_confusion(confusion: pd.DataFrame) -> dict[str, float]:
    tp = float(confusion["tp"].sum())
    fp = float(confusion["fp"].sum())
    fn = float(confusion["fn"].sum())
    tn = float(confusion["tn"].sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    far = fp / (fp + tn) if fp + tn > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "false_alarm_rate": float(far),
    }


def event_station_metric(
    station_events: pd.DataFrame,
) -> pd.DataFrame:
    result = station_events.copy()
    result["event_coverage"] = np.divide(
        result["detected_events"],
        result["events"],
        out=np.zeros(len(result), dtype=float),
        where=result["events"].to_numpy() > 0,
    )
    return result


def paired_interval(
    station_ids: list[str],
    candidate: np.ndarray,
    baseline: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("candidate and baseline must be aligned 1-D arrays")
    if len(station_ids) != candidate.size:
        raise ValueError("station_ids and metric arrays must have equal length")
    differences = candidate - baseline
    values_by_station = {
        str(station_id): float(value)
        for station_id, value in zip(station_ids, differences)
    }
    bootstrap = cluster_bootstrap_mean(
        values_by_station,
        iterations=iterations,
        seed=seed,
    )
    p_value = exact_sign_flip_two_sided(differences)
    return {
        "stations": int(differences.size),
        "mean_difference": float(differences.mean()),
        "std_difference": float(differences.std(ddof=0)),
        "ci95_low": float(bootstrap["ci95_low"]),
        "ci95_high": float(bootstrap["ci95_high"]),
        "positive_stations": int((differences > 0).sum()),
        "negative_stations": int((differences < 0).sum()),
        "zero_stations": int((differences == 0).sum()),
        "sign_flip_p": float(p_value),
    }


def main() -> int:
    protocol = load_json(PROTOCOL)
    correction = load_json(CORRECTION)
    retained = set(correction["retained_station_ids"])
    frame = pd.read_parquet(NORMALIZED)
    frame = frame[
        frame["station_id"].astype(str).isin(retained)
    ].copy()
    frame = frame.sort_values(
        ["station_id", "timestamp_utc"],
        kind="stable",
    ).reset_index(drop=True)
    if set(frame["station_id"]) != retained:
        raise RuntimeError("retained station set differs from Stage216A")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise RuntimeError("unexpected Stage216 split labels")
    if frame.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError("Stage216 contains duplicate station timestamps")

    far_grid = [float(value) for value in protocol["far_grid"]]
    primary_far = float(protocol["primary_far"])
    if not np.isclose(primary_far, 0.01):
        raise RuntimeError("Stage214 primary FAR changed")

    selection = pd.read_csv(DWD_SELECTION)
    base = detector_frame(frame)
    detectors = fit_detectors(base, protocol)
    train_normal = (
        base["split"].eq("train") & base["observation_present"]
    )
    components = score_components(base, detectors)
    fused, _ = fuse_scores(base, components)
    base_z = robust_standardize(fused, train_normal).to_numpy(dtype=float)

    feature_z: dict[tuple[str, int], np.ndarray] = {}
    for feature_name in FEATURES:
        for window in WINDOWS:
            feature_z[(feature_name, int(window))] = robust_standardize(
                repeat_feature(base, feature_name, int(window)),
                train_normal,
            ).to_numpy(dtype=float)

    candidates = candidate_grid()
    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    station_event_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    transfer_manifest: list[dict[str, object]] = []

    for seed in protocol["seeds"]:
        seed = int(seed)
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
        valid_fused, _ = fuse_scores(valid_attacked, valid_components)
        valid_base_z = robust_standardize(
            valid_fused,
            train_normal,
        ).to_numpy(dtype=float)
        test_components = score_components(test_attacked, detectors)
        test_fused, _ = fuse_scores(test_attacked, test_components)
        test_base_z = robust_standardize(
            test_fused,
            train_normal,
        ).to_numpy(dtype=float)
        valid_feature_z: dict[tuple[str, int], np.ndarray] = {}
        test_feature_z: dict[tuple[str, int], np.ndarray] = {}
        for feature_name in FEATURES:
            for window in WINDOWS:
                key = (feature_name, int(window))
                valid_feature_z[key] = robust_standardize(
                    repeat_feature(
                        valid_attacked,
                        feature_name,
                        int(window),
                    ),
                    train_normal,
                ).to_numpy(dtype=float)
                test_feature_z[key] = robust_standardize(
                    repeat_feature(
                        test_attacked,
                        feature_name,
                        int(window),
                    ),
                    train_normal,
                ).to_numpy(dtype=float)

        valid_mask = valid_attacked["split"].eq("valid").to_numpy(dtype=bool)
        test_mask = test_attacked["split"].eq("test").to_numpy(dtype=bool)
        valid_frame = valid_attacked.loc[valid_mask].reset_index(drop=True)
        test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
        if not valid_frame["is_attack"].any():
            raise RuntimeError("Swiss valid attack injection is empty")
        if not test_frame["is_attack"].any():
            raise RuntimeError("Swiss test attack injection is empty")
        valid_normal = (
            valid_frame["observation_present"]
            & ~valid_frame["is_attack"]
        ).to_numpy(dtype=bool)
        valid_base = valid_base_z[valid_mask]
        test_base = test_base_z[test_mask]
        frozen = frozen_dwd_selection(selection, seed, primary_far)
        key = (str(frozen["feature"]), int(frozen["window"]))
        valid_transferred = (
            (1.0 - float(frozen["weight"])) * valid_base
            + float(frozen["weight"]) * valid_feature_z[key][valid_mask]
        )
        test_transferred = (
            (1.0 - float(frozen["weight"])) * test_base
            + float(frozen["weight"]) * test_feature_z[key][test_mask]
        )
        transfer_manifest.append(
            {
                "seed": seed,
                "dwd_feature": str(frozen["feature"]),
                "dwd_window": int(frozen["window"]),
                "dwd_weight": float(frozen["weight"]),
                "dwd_threshold": float(frozen["dwd_threshold"]),
                "valid_attack_rows": int(valid_frame["is_attack"].sum()),
                "test_attack_rows": int(test_frame["is_attack"].sum()),
            }
        )

        for far in far_grid:
            far = float(far)
            recalibrated_threshold = far_threshold(
                valid_transferred,
                valid_normal,
                far,
            )

            best_local: dict[str, object] | None = None
            for feature_name, window, weight in candidates:
                valid_score = (
                    (1.0 - weight) * valid_base
                    + weight
                    * valid_feature_z[(feature_name, window)][valid_mask]
                )
                threshold = far_threshold(
                    valid_score,
                    valid_normal,
                    far,
                )
                valid_pred = valid_score >= threshold
                valid_metrics = metrics(
                    valid_frame["is_attack"].to_numpy(dtype=bool),
                    valid_pred,
                    valid_score,
                )
                valid_events = event_metrics(valid_frame, valid_pred)
                valid_types = type_metrics(
                    valid_frame,
                    valid_pred,
                    valid_score,
                )
                valid_stuck = float(
                    valid_types.get("stuck", {}).get("event_coverage", 0.0)
                )
                valid_event_macro = float(
                    np.mean(
                        [
                            float(row["event_coverage"])
                            for row in valid_types.values()
                        ]
                    )
                )
                valid_nonstuck = float(
                    np.mean(
                        [
                            float(row["event_coverage"])
                            for name, row in valid_types.items()
                            if name != "stuck"
                        ]
                    )
                )
                current = {
                    "feature": feature_name,
                    "window": int(window),
                    "weight": float(weight),
                    "threshold": float(threshold),
                    "valid_metrics": valid_metrics,
                    "valid_events": valid_events,
                    "valid_stuck_event": valid_stuck,
                    "valid_event_macro": valid_event_macro,
                    "valid_nonstuck_event": valid_nonstuck,
                }
                current_key = selection_key(
                    "valid_f1",
                    valid_metrics,
                    valid_event_macro,
                    valid_stuck,
                    valid_nonstuck,
                    feature_name,
                    int(window),
                    float(weight),
                )
                if best_local is None or current_key > best_local["key"]:
                    best_local = {**current, "key": current_key}
            if best_local is None:
                raise RuntimeError("no local Swiss candidate was selected")

            arm_predictions = {
                "dwd_threshold_transfer": (
                    test_transferred >= float(frozen["dwd_threshold"])
                ),
                "swiss_valid_recalibrated": (
                    test_transferred >= recalibrated_threshold
                ),
                "swiss_valid_selected": (
                    (
                        (1.0 - float(best_local["weight"])) * test_base
                        + float(best_local["weight"])
                        * test_feature_z[
                            (
                                str(best_local["feature"]),
                                int(best_local["window"]),
                            )
                        ][test_mask]
                    )
                    >= float(best_local["threshold"])
                ),
            }
            test_truth = test_frame["is_attack"].astype(bool).to_numpy()
            for arm, predicted in arm_predictions.items():
                predicted = np.asarray(predicted, dtype=bool)
                score = (
                    test_transferred
                    if arm != "swiss_valid_selected"
                    else (
                        (1.0 - float(best_local["weight"])) * test_base
                        + float(best_local["weight"])
                        * test_feature_z[
                            (
                                str(best_local["feature"]),
                                int(best_local["window"]),
                            )
                        ][test_mask]
                    )
                )
                row_metrics = metrics(test_truth, predicted, score)
                row_events = event_metrics(test_frame, predicted)
                evaluation_rows.append(
                    {
                        "seed": seed,
                        "max_far": far,
                        "is_primary": bool(np.isclose(far, primary_far)),
                        "arm": arm,
                        **row_metrics,
                        **row_events,
                    }
                )
                confusion = station_confusion(test_frame, predicted)
                confusion.insert(0, "arm", arm)
                confusion.insert(0, "max_far", far)
                confusion.insert(0, "seed", seed)
                station_rows.extend(confusion.to_dict(orient="records"))
                station_events = station_event_table(test_frame, predicted)
                station_events.insert(0, "arm", arm)
                station_events.insert(0, "max_far", far)
                station_events.insert(0, "seed", seed)
                station_event_rows.extend(
                    station_events.to_dict(orient="records")
                )
                type_part = type_metrics(test_frame, predicted, score)
                for attack_type, values in type_part.items():
                    type_rows.append(
                        {
                            "seed": seed,
                            "max_far": far,
                            "arm": arm,
                            "attack_type": attack_type,
                            **values,
                        }
                    )
        print(
            json.dumps(
                {"seed": seed, "status": "completed"},
                ensure_ascii=False,
            ),
            flush=True,
        )

    evaluation = pd.DataFrame(evaluation_rows)
    station = pd.DataFrame(station_rows)
    station_events = pd.DataFrame(station_event_rows)
    type_detail = pd.DataFrame(type_rows)
    station_metric = station.groupby(
        ["max_far", "arm", "station_id"],
        as_index=False,
    ).agg(
        seeds=("seed", "nunique"),
        tp=("tp", "sum"),
        fp=("fp", "sum"),
        fn=("fn", "sum"),
        tn=("tn", "sum"),
    )
    station_metric_rows: list[pd.DataFrame] = []
    for (far, arm), group in station_metric.groupby(
        ["max_far", "arm"],
        sort=True,
    ):
        part = station_metric_table(
            group.drop(columns=["max_far", "arm"])
        )
        part.insert(0, "arm", arm)
        part.insert(0, "max_far", float(far))
        station_metric_rows.append(part)
    station_metric = pd.concat(station_metric_rows, ignore_index=True)
    station_event_metric = station_events.groupby(
        ["max_far", "arm", "station_id"],
        as_index=False,
    ).agg(
        seeds=("seed", "nunique"),
        events=("events", "sum"),
        detected_events=("detected_events", "sum"),
    )
    station_event_rows: list[pd.DataFrame] = []
    for (far, arm), group in station_event_metric.groupby(
        ["max_far", "arm"],
        sort=True,
    ):
        part = event_station_metric(
            group.drop(columns=["max_far", "arm"])
        )
        part.insert(0, "arm", arm)
        part.insert(0, "max_far", float(far))
        station_event_rows.append(part)
    station_event_metric = pd.concat(
        station_event_rows,
        ignore_index=True,
    )

    curve_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    p_values: list[float] = []
    p_meta: list[tuple[float, str]] = []
    iterations = 10000
    bootstrap_seed = 20260921
    for far in far_grid:
        far = float(far)
        for metric in METRICS:
            candidate = station_metric.loc[
                station_metric["max_far"].eq(far)
                & station_metric["arm"].eq("swiss_valid_selected"),
                ["station_id", metric],
            ].rename(columns={metric: "candidate"})
            baseline = station_metric.loc[
                station_metric["max_far"].eq(far)
                & station_metric["arm"].eq("swiss_valid_recalibrated"),
                ["station_id", metric],
            ].rename(columns={metric: "baseline"})
            aligned = candidate.merge(
                baseline,
                on="station_id",
                validate="one_to_one",
            )
            result = paired_interval(
                aligned["station_id"].astype(str).tolist(),
                aligned["candidate"].to_numpy(dtype=float),
                aligned["baseline"].to_numpy(dtype=float),
                iterations,
                bootstrap_seed + len(curve_rows),
            )
            curve_rows.append(
                {
                    "max_far": far,
                    "is_primary": bool(np.isclose(far, primary_far)),
                    "metric": metric,
                    "candidate_mean": float(aligned["candidate"].mean()),
                    "baseline_mean": float(aligned["baseline"].mean()),
                    **result,
                }
            )
            p_values.append(float(result["sign_flip_p"]))
            p_meta.append((far, metric))
        for metric in EVENT_METRICS:
            candidate = station_event_metric.loc[
                station_event_metric["max_far"].eq(far)
                & station_event_metric["arm"].eq(
                    "swiss_valid_selected"
                ),
                ["station_id", metric],
            ].rename(columns={metric: "candidate"})
            baseline = station_event_metric.loc[
                station_event_metric["max_far"].eq(far)
                & station_event_metric["arm"].eq("swiss_valid_recalibrated"),
                ["station_id", metric],
            ].rename(columns={metric: "baseline"})
            aligned = candidate.merge(
                baseline,
                on="station_id",
                validate="one_to_one",
            )
            result = paired_interval(
                aligned["station_id"].astype(str).tolist(),
                aligned["candidate"].to_numpy(dtype=float),
                aligned["baseline"].to_numpy(dtype=float),
                iterations,
                bootstrap_seed + len(curve_rows),
            )
            curve_rows.append(
                {
                    "max_far": far,
                    "is_primary": bool(np.isclose(far, primary_far)),
                    "metric": metric,
                    "candidate_mean": float(aligned["candidate"].mean()),
                    "baseline_mean": float(aligned["baseline"].mean()),
                    **result,
                }
            )
            p_values.append(float(result["sign_flip_p"]))
            p_meta.append((far, metric))
    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(curve_rows, q_values):
        row["bh_q"] = float(q_value)

    for far in far_grid:
        far = float(far)
        for arm in ARMS:
            part = evaluation.loc[
                evaluation["max_far"].eq(far)
                & evaluation["arm"].eq(arm)
            ]
            aggregate = aggregate_confusion(
                station.loc[
                    station["max_far"].eq(far)
                    & station["arm"].eq(arm)
                ]
            )
            event_part = station_events.loc[
                station_events["max_far"].eq(far)
                & station_events["arm"].eq(arm)
            ]
            event_aggregate = (
                event_part.groupby("station_id", as_index=False)
                .agg(
                    events=("events", "sum"),
                    detected_events=("detected_events", "sum"),
                )
            )
            event_coverage = np.divide(
                event_aggregate["detected_events"],
                event_aggregate["events"],
                out=np.zeros(len(event_aggregate), dtype=float),
                where=event_aggregate["events"].to_numpy() > 0,
            )
            paired_rows.append(
                {
                    "max_far": far,
                    "is_primary": bool(np.isclose(far, primary_far)),
                    "arm": arm,
                    "seed_f1_mean": float(part["f1"].mean()),
                    "seed_f1_std": float(part["f1"].std(ddof=0)),
                    **aggregate,
                    "station_equal_event_coverage": float(
                        event_coverage.mean()
                    ),
                    "station_event_min": int(event_aggregate["events"].min()),
                    "station_event_max": int(event_aggregate["events"].max()),
                }
            )

    curve = pd.DataFrame(curve_rows)
    paired = pd.DataFrame(paired_rows)
    primary = curve.loc[
        curve["is_primary"]
        & curve["metric"].isin(["f1", "event_coverage"])
    ]
    if len(primary) != 2:
        raise RuntimeError("primary F1/event-coverage rows are incomplete")
    if not (primary["ci95_low"] > 0).all():
        raise RuntimeError("primary transfer interval must remain positive")

    stage219 = load_json(STAGE219_JSON)
    if float(stage219["primary_far"]) != primary_far:
        raise RuntimeError("Stage219 and Stage221 primary FAR differ")

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段221复用阶段219冻结的DWD选特征/窗口/权重、MeteoSwiss"
            "数据切分、透明受控攻击、检测器和候选网格，只把评价从"
            "主FAR=0.01扩展到阶段214预先冻结的0.005/0.01/0.02/0.05"
            "运行曲线。三臂为DWD绝对阈值直接迁移、保留DWD特征并在"
            "瑞士valid正常样本重校准阈值、以及瑞士valid本地选择。"
            "valid和test使用分离的确定性攻击序列，主比较为"
            "swiss_valid_selected减swiss_valid_recalibrated。"
            "test只评价，不重新选择特征、窗口、权重或阈值；主运行点"
            "仍为FAR=0.01，其他点只是敏感性分析，不得事后改主点。"
            "所有攻击都是透明受控注入，不是真实攻击日志。"
        ),
        "source": "MeteoSwiss_SwissMetNet",
        "transfer_source": "Deutscher Wetterdienst (DWD)",
        "primary_far": primary_far,
        "primary_endpoint": "event_coverage",
        "cluster_unit": "station",
        "seeds": protocol["seeds"],
        "far_grid": far_grid,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": bootstrap_seed,
        "normalized_sha256": sha256_file(NORMALIZED),
        "protocol_sha256": sha256_file(PROTOCOL),
        "correction_sha256": sha256_file(CORRECTION),
        "dwd_selection_sha256": sha256_file(DWD_SELECTION),
        "stage219_json_sha256": sha256_file(STAGE219_JSON),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "transfer_manifest": transfer_manifest,
        "evaluation_rows": int(len(evaluation)),
        "station_detail_rows": int(len(station)),
        "station_event_detail_rows": int(len(station_events)),
        "type_detail_rows": int(len(type_detail)),
        "curve": clean_records(curve.to_dict(orient="records")),
        "arm_aggregate": clean_records(paired.to_dict(orient="records")),
        "limitations": [
            "FAR=0.005/0.02/0.05只是冻结网格上的敏感性运行点，不是test选出的最优FAR。",
            "DWD绝对阈值直接迁移在瑞士分数尺度上完全失效，该负结果必须保留。",
            "瑞士valid本地选择只作为目标域参考，不代表无目标域标签部署。",
            "只有11个MeteoSwiss站点，站点簇区间只描述当前站点集合。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
        ],
    }
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
    station_events.to_csv(
        TABLES / f"{PREFIX}_station_event_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_detail.to_csv(
        TABLES / f"{PREFIX}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curve.to_csv(
        TABLES / f"{PREFIX}_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        TABLES / f"{PREFIX}_arm_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段221 DWD到MeteoSwiss迁移冻结运行曲线",
        "",
        f"- 状态：**{payload['status']}**",
        "- 主运行点：`FAR=0.01`（阶段214预先冻结）",
        f"- FAR网格：`{far_grid}`",
        f"- Bootstrap：`{iterations}` 次，种子 `{bootstrap_seed}`",
        "- 主比较：`swiss_valid_selected - swiss_valid_recalibrated`",
        "",
        "## F1与事件覆盖运行曲线",
        "",
        "| FAR | 指标 | 本地选择 | 重校准 | 差值 | 95%区间 | 正/负站点 | BH q |",
        "|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for _, row in curve.loc[
        curve["metric"].isin(["f1", "event_coverage"])
    ].iterrows():
        lines.append(
            f"| {row['max_far']:.3f} | {row['metric']} | "
            f"{row['candidate_mean']:.4f} | {row['baseline_mean']:.4f} | "
            f"{row['mean_difference']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
            f"{int(row['positive_stations'])}/"
            f"{int(row['negative_stations'])} | "
            f"{row['bh_q']:.6f} |"
        )
    lines += [
        "",
        "## 三臂FAR点结果",
        "",
        "| FAR | 方案 | F1 | Recall | Precision | FAR | 站点等权事件覆盖 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in paired.iterrows():
        lines.append(
            f"| {row['max_far']:.3f} | {row['arm']} | "
            f"{row['f1']:.4f} | {row['recall']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['false_alarm_rate']:.4f} | "
            f"{row['station_equal_event_coverage']:.4f} |"
        )
    lines += [
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
        "",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "evaluation_rows": len(evaluation),
                "station_detail_rows": len(station),
                "curve_rows": len(curve),
                "primary": primary[
                    [
                        "metric",
                        "candidate_mean",
                        "baseline_mean",
                        "mean_difference",
                        "ci95_low",
                        "ci95_high",
                        "positive_stations",
                        "negative_stations",
                        "bh_q",
                    ]
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
