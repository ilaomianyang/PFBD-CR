from __future__ import annotations

"""Independent audit of the V8 orthogonal attack-factor evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL = ROOT / "confirmatory_package" / "simulated_attack_v8" / "protocol_freeze.json"
DETAIL = TABLES / "stage361_v8_full_factor_locked_evaluation_detail.csv"
SCENARIO = TABLES / "stage361_v8_full_factor_locked_evaluation_scenario.csv"
STATION = TABLES / "stage361_v8_full_factor_locked_evaluation_station.csv"
OUT_JSON = TABLES / "stage362_v8_full_factor_audit.json"
OUT_FACTOR = TABLES / "stage362_v8_full_factor_summary.csv"
OUT_PAIR = TABLES / "stage362_v8_full_factor_station_pairwise.csv"
OUT_REPORT = REPORTS / "stage362_v8_full_factor_audit.md"

SOURCES = ["ECCC", "KNMI", "MeteoSwiss", "NOAA ISD"]
SEEDS = [3407, 6257, 7919, 104729, 20260715]
CANDIDATE = "trust_shape_union"
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
METHODS = ["trust_only", "route_knn_arima", CANDIDATE] + BASELINES


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx])]
    a = p[order] * len(order) / np.arange(1, len(order) + 1)
    a = np.minimum.accumulate(a[::-1])[::-1]
    out[order] = np.minimum(a, 1.0)
    return out


def bootstrap(aligned: pd.DataFrame, seed: int, iterations: int = 10000):
    x = aligned.copy()
    x["diff"] = x["left"] - x["right"]
    by_provider = {}
    for provider, pg in x.groupby("source", sort=True):
        by_provider[str(provider)] = {
            int(s): g["diff"].to_numpy(float) for s, g in pg.groupby("seed", sort=True)
        }
    providers = sorted(by_provider)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        provider_means = []
        for _ in providers:
            provider = providers[int(rng.integers(0, len(providers)))]
            seed_map = by_provider[provider]
            run_seeds = sorted(seed_map)
            seed_means = []
            for _ in run_seeds:
                s = run_seeds[int(rng.integers(0, len(run_seeds)))]
                vals = seed_map[s]
                seed_means.append(float(np.mean(vals[rng.integers(0, len(vals), size=len(vals))])))
            provider_means.append(float(np.mean(seed_means)))
        draws.append(float(np.mean(provider_means)))
    draws = np.asarray(draws)
    return (
        float(x["diff"].mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
    )


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    detail = pd.read_csv(DETAIL)
    scenario = pd.read_csv(SCENARIO)
    station = pd.read_csv(STATION)
    specs = {str(x["name"]): x for x in protocol["attack_scenarios"]}
    expected_scenarios = sorted(specs)
    checks = []

    def check(name, ok, value):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "value": value})

    check("frozen_status", protocol.get("status") == "SIMULATION_PROTOCOL_V8_FULL_FACTOR_FROZEN", protocol.get("status"))
    check("protocol_id", protocol.get("protocol_id") == "TMQC-SIMULATED-ATTACK-V8-FULL-FACTOR-20260924", protocol.get("protocol_id"))
    check("v6_source_hash", protocol.get("source_protocol_sha256") == sha256(ROOT / "confirmatory_package" / "simulated_attack_v6" / "protocol_freeze.json"), protocol.get("source_protocol_sha256"))
    check("providers", sorted(detail.source.astype(str).unique()) == sorted(SOURCES), sorted(detail.source.astype(str).unique()))
    check("seeds", sorted(detail.seed.astype(int).unique()) == sorted(SEEDS), sorted(detail.seed.astype(int).unique().tolist()))
    check("scenario_names", sorted(scenario.scenario.astype(str).unique()) == expected_scenarios, sorted(scenario.scenario.astype(str).unique()))
    check("detail_rows", len(detail) == 20 * len(METHODS), len(detail))
    check("scenario_rows", len(scenario) == 20 * len(METHODS) * len(expected_scenarios), len(scenario))
    check("station_rows_nonzero", len(station) > 0, len(station))

    run_counts = detail.groupby(["source", "seed"]).method.nunique()
    method_counts = scenario.groupby(["source", "seed", "method"]).scenario.nunique()
    check("all_methods_per_run", len(run_counts) == 20 and bool((run_counts == len(METHODS)).all()), {"runs": len(run_counts), "min": int(run_counts.min()), "max": int(run_counts.max())})
    check("all_scenarios_per_method_run", len(method_counts) == 20 * len(METHODS) and bool((method_counts == len(expected_scenarios)).all()), {"runs": len(method_counts), "min": int(method_counts.min()), "max": int(method_counts.max())})
    check("positive_event_denominators", bool((scenario.events.astype(int) > 0).all()), int(scenario.events.min()))
    metrics = ["f1", "event_coverage", "row_recall", "false_alarm_rate"]
    check("metric_values_finite", bool(np.isfinite(detail[metrics].to_numpy(float)).all()), int(np.isfinite(detail[metrics].to_numpy(float)).sum()))
    selected = detail[detail.selected.astype(bool)]
    selected_far = float(selected.false_alarm_rate.mean()) if len(selected) else float("nan")
    check("selected_runs", len(selected) == 20, len(selected))
    check("selected_far", selected_far <= 0.01, selected_far)

    factor_rows = []
    for _, row in scenario.iterrows():
        spec = specs[str(row.scenario)]
        factor_rows.append({
            **row.to_dict(),
            "factor_intensity": spec.get("factor_intensity", ""),
            "factor_duration": spec.get("factor_duration", ""),
            "factor_length": spec.get("factor_length", ""),
            "factorial": bool(spec.get("factorial", False)),
            "unseen_combination": bool(spec.get("unseen_combination", False)),
        })
    factor = pd.DataFrame(factor_rows)
    factor_summary = factor.groupby(
        ["method", "factor_intensity", "factor_duration", "factor_length", "factorial", "unseen_combination"],
        as_index=False,
    )[["event_coverage", "row_recall"]].mean()
    factor_summary.to_csv(OUT_FACTOR, index=False)

    means = factor.groupby(["scenario", "method"], as_index=False).event_coverage.mean()
    left = means[means.method.eq(CANDIDATE)].rename(columns={"method": "candidate", "event_coverage": "candidate_event_coverage"})
    right = means[means.method.isin(BASELINES)].rename(columns={"method": "baseline", "event_coverage": "baseline_event_coverage"})
    pairs = left.merge(right, on="scenario", how="inner")
    pairs["difference"] = pairs.candidate_event_coverage - pairs.baseline_event_coverage
    pairs["win"] = pairs.difference > 0
    scenario_share = float(pairs.win.mean()) if len(pairs) else float("nan")
    check("scenario_win_share_ge_070", scenario_share >= 0.70, {"wins": int(pairs.win.sum()), "total": len(pairs), "share": scenario_share})

    pair_rows = []
    l = station[station.method.eq(CANDIDATE)][["source", "seed", "station_id", "event_coverage"]].rename(columns={"event_coverage": "left"})
    for baseline in BASELINES:
        r = station[station.method.eq(baseline)][["source", "seed", "station_id", "event_coverage"]].rename(columns={"event_coverage": "right"})
        a = l.merge(r, on=["source", "seed", "station_id"], validate="one_to_one")
        point, lo, hi, p = bootstrap(a, 20260924 + len(pair_rows))
        pair_rows.append({"candidate": CANDIDATE, "baseline": baseline, "difference": point, "ci95_low": lo, "ci95_high": hi, "p_two_sided": p, "station_rows": len(a), "provider_count": int(a.source.nunique())})
    pair_df = pd.DataFrame(pair_rows)
    pair_df["bh_q"] = bh(pair_df.p_two_sided.to_numpy(float))
    pair_df.to_csv(OUT_PAIR, index=False)
    all_ci = bool((pair_df.ci95_low > 0).all())
    all_bh = bool((pair_df.bh_q < 0.05).all())
    check("station_ci_lower_positive", all_ci, pair_df.ci95_low.tolist())
    check("station_bh_q_lt_005", all_bh, pair_df.bh_q.tolist())

    trust_factor = factor_summary[factor_summary.method.eq(CANDIDATE)].to_dict("records")
    payload = {
        "status": "PASS_V8_FULL_FACTOR_AUDIT_WITH_BOUNDARIES" if all(x["status"] == "PASS" for x in checks) else "FAIL_V8_FULL_FACTOR_AUDIT",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL),
        "detail_sha256": sha256(DETAIL),
        "scenario_sha256": sha256(SCENARIO),
        "station_sha256": sha256(STATION),
        "checks": checks,
        "scenario_win_share": {"wins": int(pairs.win.sum()), "total": len(pairs), "share": scenario_share},
        "selected_summary": {"runs": len(selected), "mean_f1": float(selected.f1.mean()), "mean_event_coverage": float(selected.event_coverage.mean()), "mean_far": selected_far},
        "trust_factor_summary": trust_factor,
        "station_pair_summary": pair_df.to_dict("records"),
        "limitations": [
            "V8 使用四个已冻结 provider 的既有时间窗；因 V6 test 已查看，V8 只能作为探索性压力测试，不能替换 V6 确认性结果。",
            "所有攻击均为透明模拟注入，不是真实生产攻击日志。",
            "全因子压力测试通过不等于机构无关、站点无关或所有攻击全面领先。",
            "任何基线胜出、短弱因子低覆盖或未见 coordinated drift 失败均必须保留。",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Stage362 V8 全因子压力测试独立审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{protocol['protocol_id']}`",
        f"- 场景×基线胜出：`{int(pairs.win.sum())}/{len(pairs)}={scenario_share:.1%}`",
        f"- 选择运行 F1 / event coverage / FAR：`{payload['selected_summary']['mean_f1']:.6f}` / `{payload['selected_summary']['mean_event_coverage']:.6f}` / `{selected_far:.6f}`",
        "",
        "## 检查结果",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
        "## Trust 全因子分层",
        "",
        pd.DataFrame(trust_factor).to_markdown(index=False),
        "",
        "## 站点层级 bootstrap 与 BH",
        "",
        pair_df.to_markdown(index=False),
        "",
        "## 边界",
        "",
        *[f"- {x}" for x in payload["limitations"]],
        "",
    ]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks), "failed_checks": sum(x["status"] == "FAIL" for x in checks), "scenario_win_share": payload["scenario_win_share"], "selected_summary": payload["selected_summary"], "all_station_ci_positive": all_ci, "all_station_bh": all_bh}, ensure_ascii=False))
    return 0 if payload["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
