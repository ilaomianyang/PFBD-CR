from __future__ import annotations

"""Stage 356: exploratory failure-mode feature study.

This stage is deliberately exploratory.  V6 test results had already been
observed before this script was written, so the newly selected long-horizon
feature is not allowed to replace V6 or enter a confirmatory claim.

The study asks a narrow, pre-declared question: can causal long-horizon slope
features improve the two V6 failure scenarios (slow_drift and
coordinated_offset) without exceeding the frozen low-FAR budget?  Candidate
weights are selected from validation only.  The V6 test is evaluated only to
quantify exploratory transfer and is never used for selection.
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
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v6"
PROTOCOL_PATH = PACKAGE / "protocol_freeze.json"
OUT_JSON = TABLES / "stage356_exploratory_failure_mode_features.json"
OUT_DETAIL = TABLES / "stage356_exploratory_failure_mode_features_detail.csv"
OUT_SCENARIO = TABLES / "stage356_exploratory_failure_mode_features_scenario.csv"
OUT_REPORT = REPORTS / "stage356_exploratory_failure_mode_features.md"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage340_v4_route_locked_evaluation as v6  # noqa: E402
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
)
from stage240_nonstuck_balanced_selection import (  # noqa: E402
    PROTOCOLS,
    make_frame,
)
from stage326_simulated_attack_v2_evaluation import (  # noqa: E402
    _internal_type,
)

v6.simbase._scenario_base_type = _internal_type

SEEDS = (3407, 6257, 7919, 104729, 20260715)
VALID_OFFSET = 1010001
TEST_OFFSET = 1020001
PRIMARY_FAR = 0.0065
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
TARGET_SCENARIOS = ("slow_drift", "coordinated_offset")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def long_horizon_feature(frame: pd.DataFrame) -> np.ndarray:
    """Causal long-horizon slope normalized by a local robust increment scale."""
    work = frame[["station_id", "timestamp_utc", "temperature_c"]].copy()
    work["_value"] = pd.to_numeric(work["temperature_c"], errors="coerce")
    work["_timestamp"] = pd.to_datetime(
        work["timestamp_utc"], utc=True, errors="coerce"
    )
    out = pd.Series(0.0, index=frame.index, dtype=float)
    for _, part in work.sort_values(
        ["station_id", "_timestamp"], kind="stable"
    ).groupby("station_id", sort=False):
        values = part["_value"]
        diff = values.diff().abs()
        scale = diff.rolling(24, min_periods=8).median().clip(lower=0.1)
        slope24 = (values - values.shift(24)).abs() / 24.0 / scale
        slope48 = (values - values.shift(48)).abs() / 48.0 / scale
        score = 0.6 * slope24.fillna(0.0) + 0.4 * slope48.fillna(0.0)
        out.loc[part.index] = score.to_numpy(dtype=float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(
        dtype=float
    )


def robust_nonnegative(values: np.ndarray, train_normal: np.ndarray) -> np.ndarray:
    ref = np.asarray(values, dtype=float)[train_normal]
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.zeros(len(values), dtype=float)
    med = float(np.median(ref))
    mad = max(1.4826 * float(np.median(np.abs(ref - med))), 1e-6)
    return np.maximum((np.asarray(values, dtype=float) - med) / mad, 0.0)


def scenario_metric(frame: pd.DataFrame, pred: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    attack = frame["is_attack"].astype(bool).to_numpy()
    for scenario in sorted(frame.loc[attack, "attack_scenario"].unique()):
        mask = attack & frame["attack_scenario"].eq(scenario).to_numpy()
        sub = frame.loc[mask, ["attack_event_id"]].copy()
        sub["predicted"] = np.asarray(pred, dtype=bool)[mask]
        grouped = sub.groupby("attack_event_id", sort=True)["predicted"].any()
        rows.append(
            {
                "scenario": str(scenario),
                "events": int(len(grouped)),
                "event_coverage": float(grouped.mean()) if len(grouped) else 0.0,
                "row_recall": float(np.asarray(pred, dtype=bool)[mask].mean())
                if mask.any()
                else 0.0,
            }
        )
    return rows


def main() -> int:
    frozen = load_json(PROTOCOL_PATH)
    protocol = {
        "status": "EXPLORATORY_PROTOCOL_FROZEN_AFTER_V6_TEST_OBSERVED",
        "protocol_id": "TMQC-EXPLORATORY-FAILURE-MODE-20260923-v1",
        "purpose": (
            "验证长时因果斜率特征对slow_drift和coordinated_offset的"
            "潜在改善；不替换V6确认性结果。"
        ),
        "provider_boundary": [
            "ECCC",
            "KNMI",
            "MeteoSwiss",
            "NOAA ISD",
        ],
        "seed_set": list(SEEDS),
        "primary_far": PRIMARY_FAR,
        "candidate_weights": list(WEIGHTS),
        "selection": (
            "valid-only; maximize the mean event coverage of the two target "
            "scenarios subject to valid F1 >= 95% of trust_only F1"
        ),
        "test_boundary": (
            "V6 test was previously observed; all test outputs are "
            "exploratory and cannot replace V6 confirmatory results."
        ),
        "attack_boundary": "transparent simulated attacks, not production incidents",
        "source_protocol_sha256": sha256_file(PROTOCOL_PATH),
    }
    package_dir = ROOT / "confirmatory_package" / "exploratory_failure_mode_v1"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "protocol_freeze.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    detail: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    for source, protocol_path in PROTOCOLS.items():
        source_protocol = load_json(protocol_path)
        frame = make_frame(source, source_protocol)
        base = detector_frame(frame)
        detectors = fit_detectors(base, source_protocol)
        train_normal = (
            base["split"].eq("train") & base["observation_present"]
        ).to_numpy(dtype=bool)
        components = score_components(base, detectors)
        fused, _ = fuse_scores(base, components)
        base_z = robust_standardize(fused, train_normal).to_numpy(dtype=float)
        long_feature = robust_nonnegative(
            long_horizon_feature(base), train_normal
        )
        for seed in SEEDS:
            valid_blocks = v6.simbase.make_simulated_blocks(
                base, frozen, seed, "valid", VALID_OFFSET
            )
            test_blocks = v6.simbase.make_simulated_blocks(
                base, frozen, seed, "test", TEST_OFFSET
            )
            scenarios = {str(x["name"]): x for x in frozen["attack_scenarios"]}
            valid_attacked = v6.simbase.inject_simulated_attack(
                base, valid_blocks, scenarios
            )
            test_attacked = v6.simbase.inject_simulated_attack(
                base, test_blocks, scenarios
            )
            valid_components = score_components(valid_attacked, detectors)
            test_components = score_components(test_attacked, detectors)
            valid_fused, _ = fuse_scores(valid_attacked, valid_components)
            test_fused, _ = fuse_scores(test_attacked, test_components)
            valid_z = robust_standardize(
                valid_fused, train_normal
            ).to_numpy(dtype=float)
            test_z = robust_standardize(
                test_fused, train_normal
            ).to_numpy(dtype=float)
            valid_mask = valid_attacked["split"].eq("valid").to_numpy(dtype=bool)
            test_mask = test_attacked["split"].eq("test").to_numpy(dtype=bool)
            valid_frame = valid_attacked.loc[valid_mask].reset_index(drop=True)
            test_frame = test_attacked.loc[test_mask].reset_index(drop=True)
            valid_normal = (
                valid_frame["observation_present"]
                & ~valid_frame["is_attack"]
            ).to_numpy(dtype=bool)
            trust_valid = valid_z[valid_mask]
            trust_test = test_z[test_mask]
            long_valid = long_feature[valid_mask]
            long_test = long_feature[test_mask]
            baseline_thr = far_threshold(trust_valid, valid_normal, PRIMARY_FAR)
            trust_valid_pred = (
                valid_frame["observation_present"].to_numpy(dtype=bool)
                & (trust_valid >= baseline_thr)
            )
            trust_test_pred = (
                test_frame["observation_present"].to_numpy(dtype=bool)
                & (trust_test >= baseline_thr)
            )
            trust_valid_f1 = metrics(
                valid_frame["is_attack"].to_numpy(dtype=bool),
                trust_valid_pred,
                trust_valid,
            )["f1"]
            best: dict[str, object] | None = None
            for weight in WEIGHTS:
                score_v = (1.0 - weight) * trust_valid + weight * long_valid
                threshold = far_threshold(score_v, valid_normal, PRIMARY_FAR)
                pred_v = (
                    valid_frame["observation_present"].to_numpy(dtype=bool)
                    & (score_v >= threshold)
                )
                metric_v = metrics(
                    valid_frame["is_attack"].to_numpy(dtype=bool),
                    pred_v,
                    score_v,
                )
                sc_v = scenario_metric(valid_frame, pred_v)
                target_cov = float(
                    np.mean(
                        [
                            row["event_coverage"]
                            for row in sc_v
                            if row["scenario"] in TARGET_SCENARIOS
                        ]
                    )
                )
                eligible = metric_v["f1"] >= 0.95 * trust_valid_f1
                key = (
                    target_cov if eligible else -1.0,
                    metric_v["f1"],
                    -weight,
                )
                if best is None or key > best["key"]:
                    best = {
                        "key": key,
                        "weight": float(weight),
                        "threshold": float(threshold),
                        "valid_f1": float(metric_v["f1"]),
                        "valid_target_event_coverage": target_cov,
                    }
            assert best is not None
            weight = float(best["weight"])
            score_v = (1.0 - weight) * trust_valid + weight * long_valid
            score_t = (1.0 - weight) * trust_test + weight * long_test
            pred_t = (
                test_frame["observation_present"].to_numpy(dtype=bool)
                & (score_t >= float(best["threshold"]))
            )
            metric_t = metrics(
                test_frame["is_attack"].to_numpy(dtype=bool),
                pred_t,
                score_t,
            )
            event_t = event_metrics(test_frame, pred_t)
            detail.append(
                {
                    "source": source,
                    "seed": int(seed),
                    "method": "trust_long_slope_exploratory",
                    "weight": weight,
                    "valid_f1": float(best["valid_f1"]),
                    "valid_target_event_coverage": float(
                        best["valid_target_event_coverage"]
                    ),
                    "exploratory_test": True,
                    **metric_t,
                    **event_t,
                }
            )
            for row in scenario_metric(test_frame, pred_t):
                scenario_rows.append(
                    {
                        "source": source,
                        "seed": int(seed),
                        "method": "trust_long_slope_exploratory",
                        **row,
                    }
                )
            for row in scenario_metric(test_frame, trust_test_pred):
                scenario_rows.append(
                    {
                        "source": source,
                        "seed": int(seed),
                        "method": "trust_only_reference",
                        **row,
                    }
                )
        print(
            json.dumps(
                {"source": source, "status": "completed"},
                ensure_ascii=False,
            ),
            flush=True,
        )

    detail_df = pd.DataFrame(detail)
    scenario_df = pd.DataFrame(scenario_rows)
    detail_df.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
    scenario_df.to_csv(OUT_SCENARIO, index=False, encoding="utf-8-sig")
    target = scenario_df[
        scenario_df["scenario"].isin(TARGET_SCENARIOS)
    ].groupby("method", as_index=False)["event_coverage"].mean()
    payload = {
        **protocol,
        "status": "EXPLORATORY_EVALUATION_COMPLETED",
        "detail_rows": int(len(detail_df)),
        "scenario_rows": int(len(scenario_df)),
        "summary": detail_df.groupby("method", as_index=False)[
            ["f1", "event_coverage", "row_recall", "false_alarm_rate"]
        ].mean().to_dict("records"),
        "target_scenario_summary": target.to_dict("records"),
        "limitations": [
            "V6 test was observed before this candidate was designed.",
            "The result is exploratory and cannot replace V6.",
            "No claim of confirmatory improvement or universal superiority is allowed.",
            "All attacks are transparent simulated injections.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage356 失败模式长时斜率特征探索",
        "",
        "- 状态：`EXPLORATORY_EVALUATION_COMPLETED`",
        "- 目标：slow_drift 与 coordinated_offset 的定向诊断。",
        "- 选择：仅使用 valid；V6 test 已在脚本设计前查看，因此 test 结果只能作为探索性。",
        "",
        "## 探索性摘要",
        "",
        target.to_markdown(index=False),
        "",
        "## 边界",
        "",
        "- 不替换 V6 主结果，不改变 73/99=73.7% 结论。",
        "- 不把探索性结果升级为确认性优势。",
        "- 所有攻击为透明模拟注入，不是真实生产攻击。",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "detail_rows": len(detail_df),
                "scenario_rows": len(scenario_df),
                "target_summary": payload["target_scenario_summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
