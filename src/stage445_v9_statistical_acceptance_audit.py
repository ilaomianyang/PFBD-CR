from __future__ import annotations

"""Stage 445: independent V9 statistical acceptance audit.

Before the official manifest is ready this command records a hard blocked
status and does not open any result table.  Once ready, it evaluates the
pre-registered scenario win-share, factor-cell coverage, FAR drift, BH/CI
non-inferiority, and unseen coordinated-drift checks without changing any
threshold or selecting a new method.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage425_v9_compatibility import require_ready_data_manifest, validate_protocol


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
MANIFEST = PACKAGE / "data_manifest.json"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
EVAL = TABLES / "stage425_v9_locked_evaluation_detail.csv"
SCENARIO = TABLES / "stage425_v9_locked_evaluation_scenario.csv"
STATION = TABLES / "stage425_v9_locked_evaluation_station.csv"
SUMMARY = TABLES / "stage425_v9_locked_evaluation.json"
OUT_JSON = TABLES / "stage445_v9_statistical_acceptance_audit.json"
OUT_REPORT = REPORTS / "stage445_v9_statistical_acceptance_audit.md"

TRUST_CANDIDATES = ("trust_only", "trust_arima", "trust_shape_union")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benjamini_hochberg(values: list[float]) -> list[float]:
    """Return BH-adjusted q values in the original order."""
    p = np.asarray(values, dtype=float)
    if p.size == 0:
        return []
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    for reverse_index in range(len(ranked) - 1, -1, -1):
        rank = reverse_index + 1
        running = min(running, float(ranked[reverse_index]) * len(ranked) / rank)
        adjusted[reverse_index] = min(1.0, running)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output.tolist()


def hierarchical_bootstrap(
    paired: pd.DataFrame,
    candidate: str,
    baseline: str,
    *,
    seed: int,
    iterations: int = 10000,
) -> dict[str, float]:
    """Provider -> station -> seed paired bootstrap for primary comparisons."""
    current = paired[
        paired["candidate"].eq(candidate) & paired["baseline"].eq(baseline)
    ].copy()
    if current.empty:
        raise ValueError(f"no paired station rows for {candidate} vs {baseline}")
    providers = sorted(current["source"].astype(str).unique())
    nested: dict[str, dict[str, np.ndarray]] = {}
    for provider in providers:
        nested[provider] = {}
        provider_rows = current[current["source"].astype(str).eq(provider)]
        for station_id, group in provider_rows.groupby("station_id", sort=True):
            nested[provider][str(station_id)] = group[
                ["f1_difference", "event_coverage_difference"]
            ].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    f1_draws = np.empty(iterations, dtype=float)
    event_draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        provider_f1: list[float] = []
        provider_event: list[float] = []
        for provider_index in rng.integers(0, len(providers), size=len(providers)):
            provider = providers[int(provider_index)]
            stations = sorted(nested[provider])
            station_f1: list[float] = []
            station_event: list[float] = []
            for station_index in rng.integers(
                0, len(stations), size=len(stations)
            ):
                values = nested[provider][stations[int(station_index)]]
                sampled = values[
                    rng.integers(0, len(values), size=len(values))
                ]
                station_f1.append(float(sampled[:, 0].mean()))
                station_event.append(float(sampled[:, 1].mean()))
            provider_f1.append(float(np.mean(station_f1)))
            provider_event.append(float(np.mean(station_event)))
        f1_draws[index] = float(np.mean(provider_f1))
        event_draws[index] = float(np.mean(provider_event))

    station_means = current.groupby(
        ["source", "station_id"], as_index=False
    )[["f1_difference", "event_coverage_difference"]].mean()
    provider_means = station_means.groupby("source", as_index=False)[
        ["f1_difference", "event_coverage_difference"]
    ].mean()

    def two_sided_p(draws: np.ndarray) -> float:
        return float(
            min(
                1.0,
                2.0
                * min(float(np.mean(draws <= 0.0)), float(np.mean(draws >= 0.0))),
            )
        )

    return {
        "provider_count": int(len(providers)),
        "station_count": int(len(station_means)),
        "seed_rows": int(len(current)),
        "iterations": int(iterations),
        "f1_difference": float(provider_means["f1_difference"].mean()),
        "f1_ci95_low": float(np.quantile(f1_draws, 0.025)),
        "f1_ci95_high": float(np.quantile(f1_draws, 0.975)),
        "f1_p": two_sided_p(f1_draws),
        "event_coverage_difference": float(
            provider_means["event_coverage_difference"].mean()
        ),
        "event_ci95_low": float(np.quantile(event_draws, 0.025)),
        "event_ci95_high": float(np.quantile(event_draws, 0.975)),
        "event_p": two_sided_p(event_draws),
    }


def record_blocked(protocol: dict, reason: str) -> int:
    payload = {
        "stage": 445,
        "status": "BLOCKED_V9_STATISTICAL_AUDIT_NOT_READY",
        "protocol_id": protocol.get("protocol_id"),
        "manifest_status": "NOT_READY",
        "checks": [
            {
                "check": "official_manifest_ready",
                "status": "BLOCKED",
                "value": reason,
            }
        ],
        "failed_checks": ["official_manifest_ready"],
        "observation_values_read": False,
        "attack_labels_read": False,
        "old_test_rows_read": False,
        "performance_evaluated": False,
        "claim_boundary": [
            "官方窗口未就绪时不读取任何 V9 结果表。",
            "本状态不是性能失败，而是数据门禁阻断。",
        ],
        "hashes": {"protocol": sha256(PROTOCOL), "manifest": sha256(MANIFEST)},
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage445 V9 统计验收审计",
                "",
                "- 状态：`BLOCKED_V9_STATISTICAL_AUDIT_NOT_READY`",
                "- 未读取评价、场景、站点结果表。",
                f"- 原因：{reason}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": 445,
                "status": "BLOCKED_V9_STATISTICAL_AUDIT_NOT_READY",
            },
            ensure_ascii=False,
        )
    )
    return 2


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "OFFICIAL_WINDOW_READY":
        return record_blocked(
            protocol,
            "data_manifest.status is not OFFICIAL_WINDOW_READY",
        )
    # This is the only path that opens performance outputs.
    require_ready_data_manifest(MANIFEST)
    missing = [str(path) for path in (EVAL, SCENARIO, STATION, SUMMARY) if not path.exists()]
    if missing:
        raise RuntimeError(f"V9 statistical outputs are missing: {missing}")
    evaluation = pd.read_csv(EVAL)
    scenario = pd.read_csv(SCENARIO)
    station = pd.read_csv(STATION)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    baselines = tuple(str(x) for x in protocol["method_freeze"]["baselines"])
    expected_methods = set(TRUST_CANDIDATES) | set(baselines)
    checks: list[dict[str, object]] = []

    checks.append(
        {
            "check": "method_set_complete",
            "status": "PASS"
            if set(evaluation["method"].astype(str)) == expected_methods
            else "FAIL",
            "value": sorted(evaluation["method"].astype(str).unique()),
        }
    )
    checks.append(
        {
            "check": "provider_seed_station_scope",
            "status": "PASS"
            if evaluation["source"].nunique() == 4
            and evaluation["seed"].nunique() == len(protocol["seeds"])
            and station[["source", "station_id"]].drop_duplicates().shape[0] > 0
            else "FAIL",
            "value": {
                "providers": int(evaluation["source"].nunique()),
                "seeds": int(evaluation["seed"].nunique()),
                "stations": int(
                    station[["source", "station_id"]].drop_duplicates().shape[0]
                ),
            },
        }
    )
    primary_far = float(protocol["primary_far"])
    far_drift = (evaluation["false_alarm_rate"] - primary_far).abs()
    checks.append(
        {
            "check": "far_drift_within_003",
            "status": "PASS" if bool((far_drift <= 0.03).all()) else "FAIL",
            "value": float(far_drift.max()),
        }
    )

    scenario_factor_map = {
        str(item["name"]): {
            "factor_intensity": item["factor_intensity"],
            "factor_duration": item["factor_duration"],
            "factor_length": item["factor_length"],
        }
        for item in protocol["attack_scenarios"]
    }
    for field in ("factor_intensity", "factor_duration", "factor_length"):
        if field not in scenario.columns:
            scenario[field] = scenario["scenario"].map(
                lambda name: scenario_factor_map[str(name)][field]
            )

    # Scenario win share is calculated transparently from mean event coverage
    # over source/seed rows; all candidates and baselines are retained.
    scenario_mean = (
        scenario.groupby(["scenario", "method"], as_index=False)["event_coverage"]
        .mean()
    )
    expected_scenarios = {
        str(item["name"]) for item in protocol["attack_scenarios"]
    }
    observed_scenarios = set(scenario["scenario"].astype(str))
    scenario_rows: list[dict[str, object]] = []
    for candidate in TRUST_CANDIDATES:
        for baseline in baselines:
            for scenario_name in sorted(expected_scenarios):
                left = scenario_mean[
                    scenario_mean["scenario"].eq(scenario_name)
                    & scenario_mean["method"].eq(candidate)
                ]["event_coverage"]
                right = scenario_mean[
                    scenario_mean["scenario"].eq(scenario_name)
                    & scenario_mean["method"].eq(baseline)
                ]["event_coverage"]
                if len(left) != 1 or len(right) != 1:
                    continue
                scenario_rows.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "scenario": scenario_name,
                        "candidate_event_coverage": float(left.iloc[0]),
                        "baseline_event_coverage": float(right.iloc[0]),
                        "win": bool(float(left.iloc[0]) >= float(right.iloc[0])),
                        **scenario_factor_map[scenario_name],
                    }
                )
    scenario_compare = pd.DataFrame(scenario_rows)
    win_share = (
        scenario_compare.groupby(["candidate", "baseline"], as_index=False)["win"]
        .mean()
        .rename(columns={"win": "scenario_win_share"})
    )
    expected_scenario_pair_rows = (
        len(TRUST_CANDIDATES) * len(baselines) * len(expected_scenarios)
    )
    scenario_complete = (
        observed_scenarios == expected_scenarios
        and len(scenario_compare) == expected_scenario_pair_rows
    )
    checks.append(
        {
            "check": "scenario_baseline_win_share_ge_070",
            "status": "PASS"
            if scenario_complete
            and not win_share.empty
            and bool((win_share["scenario_win_share"] >= 0.70).all())
            else "FAIL",
            "value": {
                "expected_scenario_pair_rows": expected_scenario_pair_rows,
                "observed_scenario_pair_rows": int(len(scenario_compare)),
                "scenario_names_complete": observed_scenarios == expected_scenarios,
                "win_share": win_share.to_dict("records"),
            },
        }
    )
    expected_factor_cells = {
        (
            str(item["factor_intensity"]),
            str(item["factor_duration"]),
            str(item["factor_length"]),
        )
        for item in protocol["attack_scenarios"]
    }
    factor_share = (
        scenario_compare.groupby(
            ["candidate", "baseline", "factor_intensity", "factor_duration", "factor_length"],
            as_index=False,
        )["win"]
        .mean()
        .rename(columns={"win": "factor_cell_win_share"})
    )
    checks.append(
        {
            "check": "every_factor_cell_reported",
            "status": "PASS"
            if not factor_share.empty
            and set(
                zip(
                    factor_share["factor_intensity"],
                    factor_share["factor_duration"],
                    factor_share["factor_length"],
                )
            )
            == expected_factor_cells
            and len(factor_share)
            == len(TRUST_CANDIDATES) * len(baselines) * len(expected_factor_cells)
            else "FAIL",
            "value": {
                "expected_factor_cells": sorted(expected_factor_cells),
                "reported_rows": int(len(factor_share)),
                "factor_cell_win_share": factor_share.to_dict("records"),
            },
        }
    )

    # Rebuild the primary comparison directly from the locked station table.
    # The V9 evaluator intentionally writes raw detail only; relying on a
    # legacy summary["comparisons"] field would make this gate silently fail.
    station_pairs: list[pd.DataFrame] = []
    for candidate in TRUST_CANDIDATES:
        left = station[station["method"].astype(str).eq(candidate)][
            ["source", "seed", "station_id", "f1", "event_coverage"]
        ].rename(
            columns={
                "f1": "f1_candidate",
                "event_coverage": "event_coverage_candidate",
            }
        )
        for baseline in baselines:
            right = station[station["method"].astype(str).eq(baseline)][
                ["source", "seed", "station_id", "f1", "event_coverage"]
            ].rename(
                columns={
                    "f1": "f1_baseline",
                    "event_coverage": "event_coverage_baseline",
                }
            )
            aligned = left.merge(
                right,
                on=["source", "seed", "station_id"],
                how="inner",
                validate="one_to_one",
            )
            aligned["candidate"] = candidate
            aligned["baseline"] = baseline
            aligned["f1_difference"] = (
                aligned["f1_candidate"] - aligned["f1_baseline"]
            )
            aligned["event_coverage_difference"] = (
                aligned["event_coverage_candidate"]
                - aligned["event_coverage_baseline"]
            )
            station_pairs.append(aligned)
    paired = (
        pd.concat(station_pairs, ignore_index=True)
        if station_pairs
        else pd.DataFrame()
    )
    comparison_rows: list[dict[str, object]] = []
    for index, (candidate, baseline) in enumerate(
        ((
            candidate,
            baseline,
        ) for candidate in TRUST_CANDIDATES for baseline in baselines)
    ):
        result = hierarchical_bootstrap(
            paired,
            candidate,
            baseline,
            seed=20260924 + index,
            iterations=int(protocol["bootstrap"]["iterations"]),
        )
        comparison_rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                **result,
            }
        )
    event_comparisons = pd.DataFrame(comparison_rows)
    event_comparisons["event_bh_q"] = benjamini_hochberg(
        event_comparisons["event_p"].astype(float).tolist()
    )
    event_comparisons["f1_bh_q"] = benjamini_hochberg(
        event_comparisons["f1_p"].astype(float).tolist()
    )
    event_gate = bool(
        len(event_comparisons)
        == len(TRUST_CANDIDATES) * len(baselines)
        and event_comparisons["event_ci95_low"].astype(float).gt(-0.02).all()
        and event_comparisons["event_bh_q"].astype(float).le(0.05).all()
    )
    checks.append(
        {
            "check": "event_ci_bh_noninferiority",
            "status": "PASS" if event_gate else "FAIL",
            "value": {
                "required_comparisons": len(TRUST_CANDIDATES) * len(baselines),
                "reported_comparisons": int(len(event_comparisons)),
                "comparisons": event_comparisons.to_dict("records"),
            },
        }
    )

    # Enforce consistent primary-endpoint direction at pooled, station-equal
    # and provider-equal levels. Negative results remain visible and fail the
    # gate rather than being removed from the package.
    level_rows: list[dict[str, object]] = []
    for candidate in TRUST_CANDIDATES:
        for baseline in baselines:
            pair = paired[
                paired["candidate"].eq(candidate)
                & paired["baseline"].eq(baseline)
            ]
            detail_left = evaluation[
                evaluation["method"].astype(str).eq(candidate)
            ][["source", "seed", "event_coverage"]].rename(
                columns={"event_coverage": "candidate_event_coverage"}
            )
            detail_right = evaluation[
                evaluation["method"].astype(str).eq(baseline)
            ][["source", "seed", "event_coverage"]].rename(
                columns={"event_coverage": "baseline_event_coverage"}
            )
            detail_pair = detail_left.merge(
                detail_right,
                on=["source", "seed"],
                how="inner",
                validate="one_to_one",
            )
            detail_pair["difference"] = (
                detail_pair["candidate_event_coverage"]
                - detail_pair["baseline_event_coverage"]
            )
            station_means = pair.groupby(
                ["source", "station_id"], as_index=False
            )["event_coverage_difference"].mean()
            provider_means = station_means.groupby("source")[
                "event_coverage_difference"
            ].mean()
            level_rows.extend(
                [
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "level": "pooled",
                        "event_coverage_difference": float(
                            detail_pair["difference"].mean()
                        ),
                    },
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "level": "station_equal",
                        "event_coverage_difference": float(
                            station_means["event_coverage_difference"].mean()
                        ),
                    },
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "level": "provider_equal",
                        "event_coverage_difference": float(provider_means.mean()),
                    },
                ]
            )
    level_df = pd.DataFrame(level_rows)
    direction_gate = bool(
        len(level_df) == len(TRUST_CANDIDATES) * len(baselines) * 3
        and level_df["event_coverage_difference"].astype(float).ge(0.0).all()
    )
    checks.append(
        {
            "check": "pooled_station_provider_direction_consistent",
            "status": "PASS" if direction_gate else "FAIL",
            "value": {
                "required_rows": len(TRUST_CANDIDATES) * len(baselines) * 3,
                "reported_rows": int(len(level_df)),
                "levels": level_df.to_dict("records"),
            },
        }
    )

    unseen = scenario[
        scenario["scenario"].astype(str).eq("coordinated_drift_unseen")
    ]
    unseen_rows: list[dict[str, object]] = []
    for candidate in TRUST_CANDIDATES:
        cand = unseen[unseen["method"].eq(candidate)]
        for baseline in baselines:
            base = unseen[unseen["method"].eq(baseline)]
            by_provider = cand.groupby("source")["event_coverage"].mean().to_dict()
            base_by_provider = base.groupby("source")["event_coverage"].mean().to_dict()
            differences = {
                source: float(by_provider[source] - base_by_provider[source])
                for source in by_provider.keys() & base_by_provider.keys()
            }
            unseen_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "candidate_event_coverage": float(cand["event_coverage"].mean())
                    if not cand.empty
                    else 0.0,
                    "provider_differences": differences,
                    "positive": bool(
                        not cand.empty
                        and cand["event_coverage"].mean() > 0.0
                        and all(value >= 0.0 for value in differences.values())
                    ),
                }
            )
    unseen_df = pd.DataFrame(unseen_rows)
    checks.append(
        {
            "check": "unseen_combination_positive_without_provider_reversal",
            "status": "PASS"
            if not unseen_df.empty and bool(unseen_df["positive"].all())
            else "FAIL",
            "value": unseen_df.to_dict("records"),
        }
    )
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_STATISTICAL_ACCEPTANCE_AUDIT"
        if not failures
        else "FAIL_V9_STATISTICAL_ACCEPTANCE_AUDIT"
    )
    payload = {
        "stage": 445,
        "status": status,
        "protocol_id": protocol["protocol_id"],
        "manifest_status": manifest["status"],
        "checks": checks,
        "failed_checks": failures,
        "scenario_win_share": win_share.to_dict("records"),
        "factor_cell_win_share": factor_share.to_dict("records"),
        "event_comparisons": event_comparisons.to_dict("records"),
        "pooled_station_provider_levels": level_df.to_dict("records"),
        "unseen_combination": unseen_df.to_dict("records"),
        "observation_values_read": True,
        "attack_labels_read": True,
        "performance_evaluated": True,
        "claim_boundary": [
            "这是预注册透明模拟攻击下的统计验收，不是真实生产事故召回率。",
            "任一基线胜出、CI 跨界或未见组合反转均保留并阻断全面领先主张。",
        ],
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "manifest": sha256(MANIFEST),
            "evaluation": sha256(EVAL),
            "scenario": sha256(SCENARIO),
            "station": sha256(STATION),
            "summary": sha256(SUMMARY),
        },
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage445 V9 统计验收审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 逐场景、逐因子单元、CI/BH、FAR 漂移和未见组合均单独审计。",
                "- 不通过时保留失败边界，不升级全面领先结论。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 445, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
