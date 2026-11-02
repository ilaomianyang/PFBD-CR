from __future__ import annotations

"""Independent audit for the V7 attack-factor stress test.

This script reads only V7 outputs and the V7 frozen protocol.  It does not
select a method, alter thresholds, or rewrite V6 artifacts.  Its purpose is
to detect missing runs, inconsistent event denominators, FAR violations,
factor-specific reversals, and uncertainty that disappears in pooled means.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v7"
PROTOCOL = PACKAGE / "protocol_freeze.json"
DETAIL = TABLES / "stage358_v7_factor_locked_evaluation_detail.csv"
SCENARIO = TABLES / "stage358_v7_factor_locked_evaluation_scenario.csv"
STATION = TABLES / "stage358_v7_factor_locked_evaluation_station.csv"
OUT_JSON = TABLES / "stage359_v7_factor_audit.json"
OUT_PAIR = TABLES / "stage359_v7_factor_audit_pairwise.csv"
OUT_STATION_PAIR = TABLES / "stage359_v7_factor_audit_station_pairwise.csv"
OUT_FACTOR = TABLES / "stage359_v7_factor_audit_factor_summary.csv"
OUT_REPORT = REPORTS / "stage359_v7_factor_audit.md"

SEEDS = [3407, 6257, 7919, 104729, 20260715]
SOURCES = ["ECCC", "KNMI", "MeteoSwiss", "NOAA ISD"]
CANDIDATES = ["trust_only", "route_knn_arima", "trust_shape_union"]
BASELINES = [
    "robust_max_deviation",
    "isolation_forest",
    "one_class_svm",
    "pca_reconstruction",
    "mlp_autoencoder_reconstruction",
    "knn_distance",
    "ewma_deviation",
    "arima_innovation",
    "gru_predictor",
    "lightweight_autoencoder",
    "ics_cps_spc_proxy",
]
ALL_METHODS = CANDIDATES + BASELINES


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(values[idx])]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.minimum(adjusted, 1.0)
    return out


def hierarchical_bootstrap(
    aligned: pd.DataFrame, metric_left: str, metric_right: str, seed: int, iterations: int = 10000
) -> tuple[float, float, float, float]:
    """Provider -> seed -> station bootstrap for paired station differences."""
    work = aligned.copy()
    work["diff"] = work[metric_left].astype(float) - work[metric_right].astype(float)
    work = work[np.isfinite(work["diff"])]
    by_provider: dict[str, dict[int, np.ndarray]] = {}
    for provider, pg in work.groupby("source", sort=True):
        by_provider[str(provider)] = {}
        for run_seed, sg in pg.groupby("seed", sort=True):
            by_provider[str(provider)][int(run_seed)] = sg["diff"].to_numpy(dtype=float)
    providers = sorted(by_provider)
    if not providers:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for i in range(iterations):
        provider_means = []
        for _ in providers:
            provider = providers[int(rng.integers(0, len(providers)))]
            seed_map = by_provider[provider]
            run_seeds = sorted(seed_map)
            seed_means = []
            for _ in run_seeds:
                chosen_seed = run_seeds[int(rng.integers(0, len(run_seeds)))]
                values = seed_map[chosen_seed]
                seed_means.append(float(np.mean(values[rng.integers(0, len(values), size=len(values))])))
            provider_means.append(float(np.mean(seed_means)))
        draws[i] = float(np.mean(provider_means))
    point = float(work["diff"].mean())
    lo = float(np.quantile(draws, 0.025))
    hi = float(np.quantile(draws, 0.975))
    p = float(min(1.0, 2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0))))
    return point, lo, hi, p


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    detail = pd.read_csv(DETAIL)
    scenario = pd.read_csv(SCENARIO)
    station = pd.read_csv(STATION)
    frozen_scenarios = {str(x["name"]): x for x in protocol["attack_scenarios"]}
    expected_scenarios = sorted(frozen_scenarios)

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, value: object) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "value": value})

    check("protocol_status_frozen", protocol.get("status") == "SIMULATION_PROTOCOL_V7_FACTOR_STRESS_TEST_FROZEN", protocol.get("status"))
    check("protocol_id", protocol.get("protocol_id") == "TMQC-SIMULATED-ATTACK-V7-FACTOR-20260924", protocol.get("protocol_id"))
    check("source_protocol_hash_matches_v6", protocol.get("source_protocol_sha256") == sha256(ROOT / "confirmatory_package" / "simulated_attack_v6" / "protocol_freeze.json"), protocol.get("source_protocol_sha256"))
    check("source_count", sorted(detail.source.astype(str).unique()) == sorted(SOURCES), sorted(detail.source.astype(str).unique()))
    check("seed_count", sorted(detail.seed.astype(int).unique()) == sorted(SEEDS), sorted(detail.seed.astype(int).unique().tolist()))
    check("scenario_count", sorted(scenario.scenario.astype(str).unique()) == expected_scenarios, sorted(scenario.scenario.astype(str).unique()))
    check("detail_row_count", len(detail) == len(SOURCES) * len(SEEDS) * len(ALL_METHODS), len(detail))
    check("scenario_row_count", len(scenario) == len(SOURCES) * len(SEEDS) * len(ALL_METHODS) * len(expected_scenarios), len(scenario))
    check("station_nonempty", len(station) > 0 and station[["source", "seed", "station_id"]].drop_duplicates().shape[0] > 0, len(station))

    run_method = detail.groupby(["source", "seed"], sort=False).method.nunique()
    run_method_summary = {
        "run_count": int(len(run_method)),
        "min_methods_per_run": int(run_method.min()) if len(run_method) else 0,
        "max_methods_per_run": int(run_method.max()) if len(run_method) else 0,
    }
    check("every_run_has_all_methods", bool((run_method == len(ALL_METHODS)).all()) and len(run_method) == 20, run_method_summary)
    scenario_key = scenario.groupby(["source", "seed", "method"], sort=False).scenario.nunique()
    scenario_key_summary = {
        "method_run_count": int(len(scenario_key)),
        "min_scenarios_per_method_run": int(scenario_key.min()) if len(scenario_key) else 0,
        "max_scenarios_per_method_run": int(scenario_key.max()) if len(scenario_key) else 0,
    }
    check("every_method_run_has_all_scenarios", bool((scenario_key == len(expected_scenarios)).all()) and len(scenario_key) == 280, scenario_key_summary)
    check("all_event_denominators_positive", bool((scenario.events.astype(int) > 0).all()), int(scenario.events.min()))
    metric_columns = ["f1", "event_coverage", "row_recall", "false_alarm_rate"]
    check("no_nan_metrics", bool(detail[metric_columns].notna().all().all()), int(detail[metric_columns].isna().sum().sum()))

    selected = detail[detail.selected.astype(bool)]
    selected_far = float(selected.false_alarm_rate.mean()) if len(selected) else float("nan")
    check("selected_run_count", len(selected) == 20, len(selected))
    check("selected_mean_far_le_001", bool(selected_far <= 0.01), selected_far)

    factor_rows = []
    for _, row in scenario.iterrows():
        spec = frozen_scenarios[str(row.scenario)]
        factor_rows.append(
            {
                **row.to_dict(),
                "factor_intensity": spec.get("factor_intensity", ""),
                "factor_duration": spec.get("factor_duration", ""),
                "factor_length": spec.get("factor_length", ""),
                "unseen_combination": bool(spec.get("unseen_combination", False)),
            }
        )
    factor = pd.DataFrame(factor_rows)
    factor_summary = (
        factor.groupby(["method", "factor_intensity", "factor_duration", "factor_length", "unseen_combination"], as_index=False)
        [["event_coverage", "row_recall"]]
        .mean()
    )
    factor_summary.to_csv(OUT_FACTOR, index=False)

    # Recompute scenario-level candidate-vs-baseline wins from the raw scenario table.
    means = factor.groupby(["scenario", "method"], as_index=False).event_coverage.mean()
    left = means[means.method.eq("trust_shape_union")].rename(columns={"method": "candidate", "event_coverage": "candidate_event_coverage"})
    right = means[means.method.isin(BASELINES)].rename(columns={"method": "baseline", "event_coverage": "baseline_event_coverage"})
    pairs = left.merge(right, on="scenario", how="inner")
    pairs["difference"] = pairs.candidate_event_coverage - pairs.baseline_event_coverage
    pairs["win"] = pairs.difference > 0.0
    pairs["factor_intensity"] = pairs.scenario.map(lambda x: frozen_scenarios[str(x)].get("factor_intensity", ""))
    pairs["factor_duration"] = pairs.scenario.map(lambda x: frozen_scenarios[str(x)].get("factor_duration", ""))
    pairs["factor_length"] = pairs.scenario.map(lambda x: frozen_scenarios[str(x)].get("factor_length", ""))
    pairs["unseen_combination"] = pairs.scenario.map(lambda x: bool(frozen_scenarios[str(x)].get("unseen_combination", False)))
    pairs.to_csv(OUT_PAIR, index=False)
    scenario_win_share = float(pairs.win.mean()) if len(pairs) else float("nan")
    scenario_gate = bool(scenario_win_share >= 0.70)
    check("scenario_baseline_win_share_ge_070", scenario_gate, {"wins": int(pairs.win.sum()), "total": int(len(pairs)), "share": scenario_win_share})

    pair_rows = []
    for baseline in BASELINES:
        l = station[station.method.eq("trust_shape_union")][["source", "seed", "station_id", "event_coverage", "f1"]].rename(columns={"event_coverage": "event_left", "f1": "f1_left"})
        r = station[station.method.eq(baseline)][["source", "seed", "station_id", "event_coverage", "f1"]].rename(columns={"event_coverage": "event_right", "f1": "f1_right"})
        aligned = l.merge(r, on=["source", "seed", "station_id"], validate="one_to_one")
        point, lo, hi, p = hierarchical_bootstrap(aligned, "event_left", "event_right", 20260924 + len(pair_rows))
        pair_rows.append(
            {
                "candidate": "trust_shape_union",
                "baseline": baseline,
                "event_coverage_difference": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "p_two_sided": p,
                "station_rows": len(aligned),
                "provider_count": int(aligned.source.nunique()),
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    pair_df["bh_q"] = bh(pair_df.p_two_sided.to_numpy(dtype=float))
    pair_df.to_csv(OUT_STATION_PAIR, index=False)
    all_ci_positive = bool((pair_df.ci95_low > 0.0).all())
    all_bh = bool((pair_df.bh_q < 0.05).all())
    check("all_station_ci_lower_bounds_positive", all_ci_positive, pair_df.ci95_low.tolist())
    check("all_station_bh_q_lt_005", all_bh, pair_df.bh_q.tolist())

    factor_gate = (
        factor_summary[factor_summary.method.eq("trust_shape_union")]
        .groupby(["factor_intensity", "factor_duration", "factor_length"], as_index=False)
        .event_coverage.mean()
    )
    payload = {
        "status": "PASS_V7_FACTOR_AUDIT_WITH_BOUNDARIES" if all(c["status"] == "PASS" for c in checks) else "FAIL_V7_FACTOR_AUDIT",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL),
        "detail_sha256": sha256(DETAIL),
        "scenario_sha256": sha256(SCENARIO),
        "station_sha256": sha256(STATION),
        "checks": checks,
        "scenario_win_share": {"wins": int(pairs.win.sum()), "total": int(len(pairs)), "share": scenario_win_share},
        "selected_summary": {
            "runs": len(selected),
            "mean_f1": float(selected.f1.mean()),
            "mean_event_coverage": float(selected.event_coverage.mean()),
            "mean_far": selected_far,
        },
        "factor_summary": factor_gate.to_dict("records"),
        "station_pair_summary": pair_df.to_dict("records"),
        "limitations": [
            "V7 使用四个已冻结 provider 的既有时间窗；由于 V6 test 已查看，本轮仅为探索性压力测试，不能替换 V6 确认性主结果。",
            "所有攻击均为透明模拟注入，不是真实生产攻击日志。",
            "因子压力测试通过不等于机构无关、站点无关或所有攻击全面领先。",
            "任何基线胜出、因子方向反转或事件覆盖低值均必须保留。",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage359 V7 攻击因子压力测试独立审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{protocol['protocol_id']}`",
        f"- 事件覆盖场景×基线胜出：`{int(pairs.win.sum())}/{len(pairs)}={scenario_win_share:.1%}`",
        f"- 选择运行 F1 / event coverage / FAR：`{payload['selected_summary']['mean_f1']:.6f}` / `{payload['selected_summary']['mean_event_coverage']:.6f}` / `{selected_far:.6f}`",
        "",
        "## 检查结果",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
        "## Trust-shape 因子分层",
        "",
        factor_gate.to_markdown(index=False),
        "",
        "## 站点层级 bootstrap 与 BH",
        "",
        pair_df.to_markdown(index=False),
        "",
        "## 科学边界",
        "",
        *[f"- {x}" for x in payload["limitations"]],
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks), "failed_checks": sum(c["status"] == "FAIL" for c in checks), "scenario_win_share": payload["scenario_win_share"], "selected_summary": payload["selected_summary"], "all_station_ci_positive": all_ci_positive, "all_station_bh": all_bh}, ensure_ascii=False))
    return 0 if payload["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
