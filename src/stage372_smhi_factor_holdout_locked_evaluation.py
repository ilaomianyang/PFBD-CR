from __future__ import annotations

"""Stage 372: exploratory locked factor evaluation on the Stage369 holdout.

The station set and factor matrix are frozen before this run. No test result
is used for routing, threshold, feature, or baseline selection.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
DOWNLOAD_PREFIX = os.environ.get(
    "TMQC_FACTOR_DOWNLOAD_PREFIX",
    "stage370_smhi_factor_holdout_download",
)
PROTOCOL_PATH = TABLES / f"{os.environ.get('TMQC_FACTOR_PROTOCOL_PREFIX', 'stage369_smhi_factor_holdout_protocol')}.json"
CONTRACT_PATH = TABLES / f"{os.environ.get('TMQC_FACTOR_CONTRACT_PREFIX', 'stage371_smhi_factor_holdout_contract')}.json"
PREFIX = os.environ.get(
    "TMQC_FACTOR_EVAL_PREFIX",
    "stage372_smhi_factor_holdout_locked_evaluation",
)
VALID_OFFSET = int(os.environ.get("TMQC_FACTOR_VALID_OFFSET", "1070001"))
TEST_OFFSET = int(os.environ.get("TMQC_FACTOR_TEST_OFFSET", "1080001"))
PRIMARY_FAR = 0.01
SEEDS = [3407, 6257, 7919, 104729, 20260715]
SOURCE_NAME = os.environ.get("TMQC_FACTOR_SOURCE", "SMHI")
RESULT_STATUS = os.environ.get(
    "TMQC_FACTOR_RESULT_STATUS",
    "COMPLETED_SMHI_FACTOR_HOLDOUT_LOCKED_EVALUATION",
)
UNION_BUDGETS = {
    "trust_valid_f1": float(os.environ.get("TMQC_UNION_BUDGET_TRUST", "0.50")),
    "ewma_deviation": float(os.environ.get("TMQC_UNION_BUDGET_EWMA", "0.25")),
    "arima_innovation": float(os.environ.get("TMQC_UNION_BUDGET_ARIMA", "0.25")),
}
UNION_MODE = os.environ.get("TMQC_UNION_MODE", "budgeted_or")
ISOLATED_SCENARIO_REPLAY = (
    os.environ.get("TMQC_FACTOR_ISOLATED_SCENARIOS", "0") == "1"
)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage320_simulated_attack_confirmatory_evaluation as simbase  # noqa: E402
import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
import stage256_lightweight_strong_baselines as s256  # noqa: E402
import stage262_security_domain_baselines as s262  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    detector_frame,
    fit_detectors,
    metrics,
    score_components,
)
from stage208_dwd_persistent_repeat_protection import (  # noqa: E402
    event_metrics,
    far_threshold,
)
from stage217_swiss_controlled_attack_evaluation import (  # noqa: E402
    fuse_scores,
    robust_standardize,
)
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_features,
    select_candidate_stage244,
)

TRUST_METHODS = ["trust_valid_f1", "trust_nonstuck_event_macro"]
BASELINES = [
    *s248.BASELINE_METHODS,
    *s256.NEW_BASELINES,
    *s262.SECURITY_BASELINES,
]
METHODS = [*TRUST_METHODS, "trust_shape_union", *BASELINES]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def internal_type(name: str) -> str:
    if str(name) == "coordinated_drift_unseen":
        return "drift"
    return "offset"


simbase._scenario_base_type = internal_type


def normalize_pair(
    valid: np.ndarray,
    test: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(valid, dtype=float)[normal]
    reference = reference[np.isfinite(reference)]
    if len(reference) == 0:
        return np.zeros_like(valid, dtype=float), np.zeros_like(test, dtype=float)
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    scale = max(1.4826 * mad, 1e-6)
    return (
        np.nan_to_num((np.asarray(valid, dtype=float) - median) / scale),
        np.nan_to_num((np.asarray(test, dtype=float) - median) / scale),
    )


def inject_factor(
    frame: pd.DataFrame,
    blocks: pd.DataFrame,
    scenarios: dict[str, dict],
) -> pd.DataFrame:
    attacked = simbase.inject_simulated_attack(frame, blocks, scenarios)
    clean = pd.to_numeric(
        frame["temperature_c"], errors="coerce"
    ).to_numpy(dtype=float)
    for (_, scenario_name), part in blocks.groupby(
        ["station_id", "scenario"], sort=True
    ):
        spec = scenarios[str(scenario_name)]
        active_length = min(
            int(spec.get("active_length", spec["length"])), len(part)
        )
        positions = part["row_position"].to_numpy(dtype=int)
        inactive = positions[active_length:]
        if len(inactive):
            attacked.loc[inactive, "temperature_c"] = clean[inactive]
            attacked.loc[inactive, "observed_value"] = clean[inactive]
            attacked.loc[inactive, "is_attack"] = False
            attacked.loc[inactive, "attack_type"] = "normal"
            attacked.loc[inactive, "attack_scenario"] = "normal"
            attacked.loc[inactive, "attack_event_id"] = ""
    return attacked


def scenario_rows(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    seed: int,
    method: str,
) -> list[dict[str, object]]:
    attack = frame["is_attack"].astype(bool).to_numpy()
    labels = frame["attack_scenario"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for scenario in sorted(set(labels[attack])):
        mask = attack & labels.eq(scenario) if hasattr(labels, "eq") else attack & (labels == scenario)
        event_frame = frame.loc[mask, ["attack_event_id"]].copy()
        event_frame["predicted"] = predicted[mask]
        grouped = event_frame.groupby("attack_event_id")["predicted"].any()
        rows.append(
            {
                "source": SOURCE_NAME,
                "seed": int(seed),
                "method": method,
                "scenario": scenario,
                "events": int(len(grouped)),
                "event_coverage": float(grouped.mean()) if len(grouped) else 0.0,
                "row_recall": float(predicted[mask].mean()) if mask.any() else 0.0,
            }
        )
    return rows


def station_rows(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    seed: int,
    method: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    attack = frame["is_attack"].astype(bool).to_numpy()
    observed = frame["observation_present"].astype(bool).to_numpy()
    for station_id, part in frame.groupby("station_id", sort=True):
        indices = part.index.to_numpy(dtype=int)
        y = attack[indices] & observed[indices]
        p = predicted[indices] & observed[indices]
        tp = int((y & p).sum())
        fp = int((~y & p).sum())
        fn = int((y & ~p).sum())
        tn = int((~y & ~p & observed[indices]).sum())
        event_frame = part.loc[y, ["attack_event_id"]].copy()
        event_frame["predicted"] = p[y]
        grouped = event_frame.groupby("attack_event_id")["predicted"].any()
        denom = 2 * tp + fp + fn
        rows.append(
            {
                "source": SOURCE_NAME,
                "seed": int(seed),
                "method": method,
                "station_id": str(station_id),
                "rows": int(len(indices)),
                "attack_rows": int(y.sum()),
                "normal_rows": int((~y).sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "f1": float(2 * tp / denom) if denom else 0.0,
                "events": int(len(grouped)),
                "detected_events": int(grouped.sum()),
            }
        )
    return rows


def isolated_seed_results(
    base: pd.DataFrame,
    detectors: dict,
    old_fit: dict,
    light_fit: dict,
    security_fit: dict,
    scenarios: dict[str, dict],
    scenario_list: list[dict],
    seed: int,
    train_normal: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate each factor scenario on an independent replay of the short window.

    The Sep-21--23 window contains only 72 hourly rows.  The frozen factor grid
    contains 8*scenario blocks per station, so a single non-overlapping replay
    cannot be constructed.  Each scenario is therefore replayed independently
    on the same frozen window; valid-normal calibration is pooled across the
    independent scenario replays and test labels are never used for selection.
    """
    valid_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    valid_base_z: list[np.ndarray] = []
    test_base_z: list[np.ndarray] = []
    valid_features: dict[tuple[str, int], list[np.ndarray]] = {}
    test_features: dict[tuple[str, int], list[np.ndarray]] = {}
    valid_baselines: dict[str, list[np.ndarray]] = {}
    test_baselines: dict[str, list[np.ndarray]] = {}
    valid_security: dict[str, list[np.ndarray]] = {}
    test_security: dict[str, list[np.ndarray]] = {}
    scenario_slices: list[tuple[str, int, int, pd.DataFrame]] = []
    test_offset = 0

    for scenario in scenario_list:
        single = [scenario]
        valid_blocks = simbase.make_simulated_blocks(
            base, {"attack_scenarios": single}, seed, "valid", VALID_OFFSET
        )
        test_blocks = simbase.make_simulated_blocks(
            base, {"attack_scenarios": single}, seed, "test", TEST_OFFSET
        )
        valid = inject_factor(base, valid_blocks, scenarios)
        test = inject_factor(base, test_blocks, scenarios)
        vc = score_components(valid, detectors)
        tc = score_components(test, detectors)
        vf, _ = fuse_scores(valid, vc)
        tf, _ = fuse_scores(test, tc)
        vz = robust_standardize(vf, train_normal).to_numpy(dtype=float)
        tz = robust_standardize(tf, train_normal).to_numpy(dtype=float)
        vf_features_full = candidate_features(valid, train_normal)
        tf_features_full = candidate_features(test, train_normal)
        vm = valid["split"].eq("valid").to_numpy(dtype=bool)
        tm = test["split"].eq("test").to_numpy(dtype=bool)
        vf_frame = valid.loc[vm].reset_index(drop=True)
        tf_frame = test.loc[tm].reset_index(drop=True)
        valid_frames.append(vf_frame)
        test_frames.append(tf_frame)
        valid_base_z.append(vz[vm])
        test_base_z.append(tz[tm])

        vmatrix, _ = s248.feature_matrix(vc, vf_features_full, train_normal)
        tmatrix, _ = s248.feature_matrix(tc, tf_features_full, train_normal)
        for key, values in vf_features_full.items():
            valid_features.setdefault(key, []).append(values[vm])
        for key, values in tf_features_full.items():
            test_features.setdefault(key, []).append(values[tm])
        old_valid = s248.score_baselines(old_fit, vmatrix[vm])
        old_test = s248.score_baselines(old_fit, tmatrix[tm])
        light_valid = s256.score_lightweight_baselines(light_fit, vmatrix[vm])
        light_test = s256.score_lightweight_baselines(light_fit, tmatrix[tm])
        for method, values in {**old_valid, **light_valid}.items():
            valid_baselines.setdefault(method, []).append(values)
        for method, values in {**old_test, **light_test}.items():
            test_baselines.setdefault(method, []).append(values)
        sec_valid_full = s262.score_security_baselines(security_fit, valid, vmatrix)
        sec_test_full = s262.score_security_baselines(security_fit, test, tmatrix)
        for method in s262.SECURITY_BASELINES:
            valid_security.setdefault(method, []).append(sec_valid_full[method][vm])
            test_security.setdefault(method, []).append(sec_test_full[method][tm])
        start = test_offset
        test_offset += len(tf_frame)
        scenario_slices.append((str(scenario["name"]), start, test_offset, tf_frame))

    valid_frame = pd.concat(valid_frames, ignore_index=True)
    test_frame = pd.concat(test_frames, ignore_index=True)
    normal = (
        valid_frame["observation_present"] & ~valid_frame["is_attack"]
    ).to_numpy(dtype=bool)
    valid_z: dict[str, np.ndarray] = {}
    test_z: dict[str, np.ndarray] = {}
    valid_base = np.concatenate(valid_base_z)
    test_base = np.concatenate(test_base_z)
    valid_feature_arrays = {
        key: np.concatenate(values) for key, values in valid_features.items()
    }
    test_feature_arrays = {
        key: np.concatenate(values) for key, values in test_features.items()
    }
    for method, rule in {
        "trust_valid_f1": "valid_f1",
        "trust_nonstuck_event_macro": "valid_nonstuck_event_macro",
    }.items():
        selected = select_candidate_stage244(
            rule,
            valid_base,
            {key: value for key, value in valid_feature_arrays.items()},
            valid_frame,
            normal,
            PRIMARY_FAR,
        )
        key = (str(selected["feature"]), int(selected["window"]))
        valid_z[method] = (
            (1 - float(selected["weight"])) * valid_base
            + float(selected["weight"]) * valid_feature_arrays[key]
        )
        test_z[method] = (
            (1 - float(selected["weight"])) * test_base
            + float(selected["weight"]) * test_feature_arrays[key]
        )
    for method, values in valid_baselines.items():
        valid_z[method], test_z[method] = normalize_pair(
            np.concatenate(values), np.concatenate(test_baselines[method]), normal
        )
    for method, values in valid_security.items():
        valid_z[method], test_z[method] = normalize_pair(
            np.concatenate(values), np.concatenate(test_security[method]), normal
        )

    if any(value < 0.0 for value in UNION_BUDGETS.values()):
        raise RuntimeError("union budget shares must be non-negative")
    if not np.isclose(sum(UNION_BUDGETS.values()), 1.0):
        raise RuntimeError("union budget shares must sum to 1")
    if UNION_MODE == "single_threshold_max":
        union_valid_score = np.maximum.reduce(
            [valid_z["trust_valid_f1"], valid_z["ewma_deviation"], valid_z["arima_innovation"]]
        )
        union_test_score = np.maximum.reduce(
            [test_z["trust_valid_f1"], test_z["ewma_deviation"], test_z["arima_innovation"]]
        )
        union_threshold = far_threshold(union_valid_score, normal, PRIMARY_FAR)
        union_valid = union_valid_score >= union_threshold
        union_test = union_test_score >= union_threshold
    elif UNION_MODE == "budgeted_or":
        union_valid = np.zeros(len(valid_frame), dtype=bool)
        union_test = np.zeros(len(test_frame), dtype=bool)
        for method, share in UNION_BUDGETS.items():
            threshold = far_threshold(valid_z[method], normal, share * PRIMARY_FAR)
            union_valid |= valid_z[method] >= threshold
            union_test |= test_z[method] >= threshold
    else:
        raise RuntimeError(f"unknown TMQC_UNION_MODE={UNION_MODE!r}")
    valid_z["trust_shape_union"] = np.maximum.reduce(
        [valid_z["trust_valid_f1"], valid_z["ewma_deviation"], valid_z["arima_innovation"]]
    )
    test_z["trust_shape_union"] = np.maximum.reduce(
        [test_z["trust_valid_f1"], test_z["ewma_deviation"], test_z["arima_innovation"]]
    )

    y = test_frame["is_attack"].astype(bool).to_numpy()
    observed = test_frame["observation_present"].to_numpy(dtype=bool)
    eval_rows: list[dict[str, object]] = []
    scenario_rows_out: list[dict[str, object]] = []
    station_rows_out: list[dict[str, object]] = []
    for method in METHODS:
        if method == "trust_shape_union":
            predicted = observed & union_test
            score = test_z[method]
        else:
            threshold = far_threshold(valid_z[method], normal, PRIMARY_FAR)
            predicted = observed & (test_z[method] >= threshold)
            score = test_z[method]
        eval_rows.append(
            {
                "source": SOURCE_NAME,
                "seed": int(seed),
                "method": method,
                **metrics(y, predicted, score),
                **event_metrics(test_frame, predicted),
            }
        )
        station_rows_out.extend(station_rows(test_frame, predicted, seed, method))
        for scenario_name, start, end, frame_part in scenario_slices:
            part_predicted = predicted[start:end]
            scenario_rows_out.extend(
                scenario_rows(frame_part, part_predicted, seed, method)
            )
    return (
        {"evaluation": eval_rows},
        scenario_rows_out,
        station_rows_out,
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    protocol = load_json(PROTOCOL_PATH)
    contract = load_json(CONTRACT_PATH)
    # Stage384 uses an explicit year-qualified freeze status.  Accept only
    # these two freeze statuses; neither permits performance access before
    # this evaluation run.
    if protocol["status"] not in {
        "PROTOCOL_FROZEN_BEFORE_OBSERVATION_ACCESS",
        "PROTOCOL_FROZEN_BEFORE_2026_OBSERVATION_ACCESS",
        "PROTOCOL_FROZEN_BEFORE_SEPTEMBER_2026_OBSERVATION_ACCESS",
        "PROTOCOL_FROZEN_BEFORE_SEPTEMBER_21_2026_OBSERVATION_ACCESS",
    }:
        raise RuntimeError("factor protocol is not frozen")
    if contract["status"] not in {
        "EVALUATION_CONTRACT_FROZEN_BEFORE_PERFORMANCE",
        "EVALUATION_CONTRACT_FROZEN_BEFORE_SEP21_2026_PERFORMANCE",
    }:
        raise RuntimeError("factor contract is not frozen")
    if contract["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("factor contract was already evaluated")
    frame = pd.read_parquet(
        TABLES / f"{DOWNLOAD_PREFIX}_normalized.parquet"
    )
    frame["station_id"] = frame["station_id"].astype(str)
    retained = set(protocol["station_ids"])
    allow_partial = os.environ.get("TMQC_FACTOR_ALLOW_PARTIAL_STATIONS", "0") == "1"
    if allow_partial:
        download_meta_path = TABLES / f"{DOWNLOAD_PREFIX}.json"
        download_meta = load_json(download_meta_path)
        retained = set(str(value) for value in download_meta["retained_station_ids"])
        if not retained.issubset(set(protocol["station_ids"])):
            raise RuntimeError("partial station set is not a subset of frozen protocol")
    observed_station_ids = set(frame["station_id"])
    if not observed_station_ids.issubset(set(protocol["station_ids"])):
        raise RuntimeError("factor holdout contains stations outside the frozen protocol")
    if allow_partial:
        if not retained.issubset(observed_station_ids):
            raise RuntimeError("partial station audit lists unavailable stations")
        # The downloader keeps an auditable normalized file for all requested
        # stations, including a failed station with zero test coverage.  A
        # partial exploratory evaluation may use only the retained subset from
        # that audit; it must never silently include the failed station.
        frame = frame[frame["station_id"].isin(retained)].reset_index(drop=True)
    elif observed_station_ids != retained:
        raise RuntimeError("factor holdout station set changed")
    base = detector_frame(frame)
    detectors = fit_detectors(base, contract)
    train_normal = (
        base["split"].eq("train") & base["observation_present"]
    ).to_numpy(dtype=bool)
    components = score_components(base, detectors)
    fused, _ = fuse_scores(base, components)
    base_z = robust_standardize(fused, train_normal).to_numpy(dtype=float)
    base_features = candidate_features(base, train_normal)
    matrix, _ = s248.feature_matrix(
        components, base_features, train_normal
    )
    old_fit = s248.fit_baselines(base, matrix, train_normal)
    light_fit, _ = s256.fit_lightweight_baselines(
        base, matrix, train_normal
    )
    security_fit, _ = s262.fit_security_baselines(
        base, matrix, train_normal
    )
    factor_protocol = protocol.get(
        "attack_factor_protocol",
        protocol.get("attack_protocol", {}),
    )
    scenarios = {
        str(item["name"]): item
        for item in factor_protocol.get("scenarios", [])
    }
    if not scenarios:
        scenarios = {}
        for intensity, magnitude in (("weak", 0.8), ("strong", 4.0)):
            for duration, active_length in (("short", 3), ("long", 12)):
                for length_name, length in (("short", 8), ("long", 24)):
                    name = f"offset_{intensity}_{duration}_{length_name}"
                    scenarios[name] = {
                        "name": name,
                        "base_type": "offset",
                        "magnitude": magnitude,
                        "length": length,
                        "active_length": active_length,
                    }
        scenarios["coordinated_drift_unseen"] = {
            "name": "coordinated_drift_unseen",
            "base_type": "drift",
            "start": 0.0,
            "end": 4.0,
            "length": 24,
            "active_length": 12,
            "station_count": 2,
            "coordinated": True,
        }
    # Ensure the factor protocol itself is explicit in the execution artifact.
    scenario_list = list(scenarios.values())
    evaluation_rows: list[dict[str, object]] = []
    scenario_detail: list[dict[str, object]] = []
    station_detail: list[dict[str, object]] = []
    for seed in SEEDS:
        if ISOLATED_SCENARIO_REPLAY:
            seed_result, seed_scenarios, seed_stations = isolated_seed_results(
                base,
                detectors,
                old_fit,
                light_fit,
                security_fit,
                scenarios,
                scenario_list,
                seed,
                train_normal,
            )
            evaluation_rows.extend(seed_result["evaluation"])
            scenario_detail.extend(seed_scenarios)
            station_detail.extend(seed_stations)
            print(
                json.dumps(
                    {
                        "source": SOURCE_NAME,
                        "seed": int(seed),
                        "status": "completed_isolated_scenario_replay",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        valid_blocks = simbase.make_simulated_blocks(
            base, {"attack_scenarios": scenario_list}, seed, "valid", VALID_OFFSET
        )
        test_blocks = simbase.make_simulated_blocks(
            base, {"attack_scenarios": scenario_list}, seed, "test", TEST_OFFSET
        )
        valid = inject_factor(base, valid_blocks, scenarios)
        test = inject_factor(base, test_blocks, scenarios)
        vc = score_components(valid, detectors)
        tc = score_components(test, detectors)
        vf, _ = fuse_scores(valid, vc)
        tf, _ = fuse_scores(test, tc)
        vz = robust_standardize(vf, train_normal).to_numpy(dtype=float)
        tz = robust_standardize(tf, train_normal).to_numpy(dtype=float)
        vf_features = candidate_features(valid, train_normal)
        tf_features = candidate_features(test, train_normal)
        vm = valid["split"].eq("valid").to_numpy(dtype=bool)
        tm = test["split"].eq("test").to_numpy(dtype=bool)
        vf_frame = valid.loc[vm].reset_index(drop=True)
        tf_frame = test.loc[tm].reset_index(drop=True)
        normal = (
            vf_frame["observation_present"] & ~vf_frame["is_attack"]
        ).to_numpy(dtype=bool)
        vmatrix, _ = s248.feature_matrix(vc, vf_features, train_normal)
        tmatrix, _ = s248.feature_matrix(tc, tf_features, train_normal)
        valid_scores: dict[str, np.ndarray] = {}
        test_scores: dict[str, np.ndarray] = {}
        for method, rule in {
            "trust_valid_f1": "valid_f1",
            "trust_nonstuck_event_macro": "valid_nonstuck_event_macro",
        }.items():
            selected = select_candidate_stage244(
                rule,
                vz[vm],
                {key: value[vm] for key, value in vf_features.items()},
                vf_frame,
                normal,
                PRIMARY_FAR,
            )
            key = (str(selected["feature"]), int(selected["window"]))
            valid_scores[method] = (
                (1 - float(selected["weight"])) * vz[vm]
                + float(selected["weight"]) * vf_features[key][vm]
            )
            test_scores[method] = (
                (1 - float(selected["weight"])) * tz[tm]
                + float(selected["weight"]) * tf_features[key][tm]
            )
        valid_scores.update(s248.score_baselines(old_fit, vmatrix[vm]))
        test_scores.update(s248.score_baselines(old_fit, tmatrix[tm]))
        valid_scores.update(
            s256.score_lightweight_baselines(light_fit, vmatrix[vm])
        )
        test_scores.update(
            s256.score_lightweight_baselines(light_fit, tmatrix[tm])
        )
        security_valid_full = s262.score_security_baselines(
            security_fit, valid, vmatrix
        )
        security_test_full = s262.score_security_baselines(
            security_fit, test, tmatrix
        )
        for method in s262.SECURITY_BASELINES:
            valid_scores[method] = security_valid_full[method][vm]
            test_scores[method] = security_test_full[method][tm]
        valid_z: dict[str, np.ndarray] = {}
        test_z: dict[str, np.ndarray] = {}
        for method in valid_scores:
            valid_z[method], test_z[method] = normalize_pair(
                valid_scores[method], test_scores[method], normal
            )
        union_valid = np.zeros(len(normal), dtype=bool)
        union_test = np.zeros(len(tf_frame), dtype=bool)
        if any(value < 0.0 for value in UNION_BUDGETS.values()):
            raise RuntimeError("union budget shares must be non-negative")
        if not np.isclose(sum(UNION_BUDGETS.values()), 1.0):
            raise RuntimeError("union budget shares must sum to 1")
        if UNION_MODE == "single_threshold_max":
            max_valid = np.maximum.reduce(
                [
                    valid_z["trust_valid_f1"],
                    valid_z["ewma_deviation"],
                    valid_z["arima_innovation"],
                ]
            )
            max_test = np.maximum.reduce(
                [
                    test_z["trust_valid_f1"],
                    test_z["ewma_deviation"],
                    test_z["arima_innovation"],
                ]
            )
            threshold = far_threshold(max_valid, normal, PRIMARY_FAR)
            union_valid = max_valid >= threshold
            union_test = max_test >= threshold
        elif UNION_MODE == "budgeted_or":
            for method, share in UNION_BUDGETS.items():
                budget = share * PRIMARY_FAR
                threshold = far_threshold(valid_z[method], normal, budget)
                union_valid |= valid_z[method] >= threshold
                union_test |= test_z[method] >= threshold
        else:
            raise RuntimeError(f"unknown TMQC_UNION_MODE={UNION_MODE!r}")
        valid_z["trust_shape_union"] = np.maximum.reduce(
            [valid_z["trust_valid_f1"], valid_z["ewma_deviation"], valid_z["arima_innovation"]]
        )
        test_z["trust_shape_union"] = np.maximum.reduce(
            [test_z["trust_valid_f1"], test_z["ewma_deviation"], test_z["arima_innovation"]]
        )
        y = tf_frame["is_attack"].astype(bool).to_numpy()
        observed = tf_frame["observation_present"].to_numpy(dtype=bool)
        for method in METHODS:
            if method == "trust_shape_union":
                predicted = observed & union_test
                score = test_z[method]
            else:
                threshold = far_threshold(
                    valid_z[method], normal, PRIMARY_FAR
                )
                predicted = observed & (test_z[method] >= threshold)
                score = test_z[method]
            row = metrics(y, predicted, score)
            event = event_metrics(tf_frame, predicted)
            evaluation_rows.append(
                {
                    "source": SOURCE_NAME,
                    "seed": int(seed),
                    "method": method,
                    **row,
                    **event,
                }
            )
            scenario_detail.extend(
                scenario_rows(tf_frame, predicted, seed, method)
            )
            station_detail.extend(
                station_rows(tf_frame, predicted, seed, method)
            )
        print(
            json.dumps(
                {
                    "source": SOURCE_NAME,
                    "seed": int(seed),
                    "status": "completed",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    evaluation = pd.DataFrame(evaluation_rows)
    scenarios_frame = pd.DataFrame(scenario_detail)
    stations_frame = pd.DataFrame(station_detail)
    evaluation.to_csv(
        TABLES / f"{PREFIX}.csv", index=False, encoding="utf-8-sig"
    )
    scenarios_frame.to_csv(
        TABLES / f"{PREFIX}_scenario.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stations_frame.to_csv(
        TABLES / f"{PREFIX}_station.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        evaluation.groupby(["source", "method"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            f1_mean=("f1", "mean"),
            event_coverage_mean=("event_coverage", "mean"),
            row_recall_mean=("row_recall", "mean"),
            average_precision_mean=("average_precision", "mean"),
            false_alarm_rate_mean=("false_alarm_rate", "mean"),
        )
        .sort_values("event_coverage_mean", ascending=False)
    )
    scenario_summary = (
        scenarios_frame.groupby(["method", "scenario"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            events=("events", "sum"),
            event_coverage=("event_coverage", "mean"),
            row_recall=("row_recall", "mean"),
        )
    )
    summary.to_csv(
        TABLES / f"{PREFIX}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenario_summary.to_csv(
        TABLES / f"{PREFIX}_scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "status": RESULT_STATUS,
        "protocol_id": protocol["protocol_id"],
        "contract_id": contract["contract_id"],
        "exploratory": True,
        "partial_station_subset": bool(allow_partial),
        "station_count": int(frame["station_id"].nunique()),
        "seed_count": int(evaluation["seed"].nunique()),
        "method_count": int(evaluation["method"].nunique()),
        "evaluation_rows": int(len(evaluation)),
        "scenario_rows": int(len(scenarios_frame)),
        "station_rows": int(len(stations_frame)),
        "primary_far": PRIMARY_FAR,
        "primary_endpoint": "event_coverage",
        "union_budget_shares": UNION_BUDGETS,
        "union_mode": UNION_MODE,
        "isolated_scenario_replay": ISOLATED_SCENARIO_REPLAY,
        "scenario_names": sorted(
            scenarios_frame["scenario"].unique().tolist()
        ),
        "summary": summary.to_dict("records"),
        "factor_protocol": factor_protocol,
        "claim_boundary": [
            "Exploratory because initiated after Stage367 results were observed.",
            "All attacks are controlled simulations, not production incidents.",
            "This is an SMHI station boundary, not an independent provider.",
            "Short-window factor evaluation uses independent scenario replays "
            "because the frozen 72-hour test window cannot hold all nine "
            "non-overlapping blocks simultaneously.",
        ],
    }
    output = TABLES / f"{PREFIX}.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = REPORTS / f"{PREFIX}.md"
    report.write_text(
        "\n".join(
            [
                f"# {PREFIX} 因子压力测试锁定评价",
                "",
                f"- 状态：`{payload['status']}`",
                f"- 站点：`{payload['station_count']}`；种子：`{payload['seed_count']}`；方法：`{payload['method_count']}`",
                f"- 场景：`{len(payload['scenario_names'])}`",
                "- 探索性：是（协议启动晚于 Stage367 结果查看）。",
                "- 所有攻击为透明受控模拟，不是真实生产攻击。",
                "",
                summary.to_markdown(index=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "station_count": payload["station_count"],
                "seed_count": payload["seed_count"],
                "method_count": payload["method_count"],
                "scenario_count": len(payload["scenario_names"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
