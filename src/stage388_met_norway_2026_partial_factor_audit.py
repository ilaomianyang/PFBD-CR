from __future__ import annotations

"""Independent integrity and statistical audit for the partial MET Norway run.

This audit deliberately treats the 11 retained stations as an exploratory
subset.  It cannot promote the result to a 12-station confirmatory result.
"""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

EVAL_PREFIX = os.environ.get(
    "TMQC_MET_EVAL_PREFIX",
    "stage387_met_norway_2026_partial_factor_evaluation",
)
PROTOCOL_PREFIX = os.environ.get(
    "TMQC_MET_PROTOCOL_PREFIX",
    "stage384_met_norway_2026_factor_protocol",
)
CONTRACT_PREFIX = os.environ.get(
    "TMQC_MET_CONTRACT_PREFIX",
    "stage385_met_norway_2026_factor_contract",
)
DOWNLOAD_PREFIX = os.environ.get(
    "TMQC_MET_DOWNLOAD_PREFIX",
    "stage386_met_norway_2026_factor_download",
)
AUDIT_PREFIX = os.environ.get(
    "TMQC_MET_AUDIT_PREFIX",
    "stage388_met_norway_2026_partial_factor_audit",
)

if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))
import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402

TRUST_CANDIDATES = [
    "trust_valid_f1",
    "trust_nonstuck_event_macro",
    "trust_shape_union",
]
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_table(station: pd.DataFrame) -> pd.DataFrame:
    grouped = s248.station_metric_table(station)
    rows: list[dict[str, object]] = []
    for candidate in TRUST_CANDIDATES:
        for baseline in BASELINES:
            left = grouped[grouped["method"].eq(candidate)]
            right = grouped[grouped["method"].eq(baseline)]
            aligned = left.merge(
                right,
                on=["source", "seed", "station_id"],
                suffixes=("_candidate", "_baseline"),
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
                            row["f1_candidate"] - row["f1_baseline"]
                        ),
                        "event_coverage_difference": float(
                            row["event_coverage_candidate"]
                            - row["event_coverage_baseline"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    result_path = TABLES / f"{EVAL_PREFIX}.json"
    protocol_path = TABLES / f"{PROTOCOL_PREFIX}.json"
    contract_path = TABLES / f"{CONTRACT_PREFIX}.json"
    download_path = TABLES / f"{DOWNLOAD_PREFIX}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    download = json.loads(download_path.read_text(encoding="utf-8"))
    evaluation = pd.read_csv(TABLES / f"{EVAL_PREFIX}.csv")
    scenario = pd.read_csv(TABLES / f"{EVAL_PREFIX}_scenario.csv")
    scenario_summary = pd.read_csv(
        TABLES / f"{EVAL_PREFIX}_scenario_summary.csv"
    )
    summary = pd.read_csv(TABLES / f"{EVAL_PREFIX}_summary.csv")
    station = pd.read_csv(TABLES / f"{EVAL_PREFIX}_station.csv")
    provider_name = str(protocol.get("provider", "provider"))
    if "Finnish Meteorological Institute" in provider_name:
        provider_slug = "FMI"
    elif "MET Norway" in provider_name or "Norwegian Meteorological Institute" in provider_name:
        provider_slug = "MET_NORWAY"
    else:
        provider_slug = (
            provider_name.replace(" ", "_").replace("-", "_").upper()
        )

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, value: object) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "value": value,
            }
        )

    check(
        "protocol_contract_frozen_before_performance",
        protocol["status"]
        in {
            "PROTOCOL_FROZEN_BEFORE_OBSERVATION_ACCESS",
            "PROTOCOL_FROZEN_BEFORE_2026_OBSERVATION_ACCESS",
            "PROTOCOL_FROZEN_BEFORE_SEPTEMBER_2026_OBSERVATION_ACCESS",
            "PROTOCOL_FROZEN_BEFORE_SEPTEMBER_21_2026_OBSERVATION_ACCESS",
        }
        and contract["status"]
        in {
            "EVALUATION_CONTRACT_FROZEN_BEFORE_PERFORMANCE",
            "EVALUATION_CONTRACT_FROZEN_BEFORE_SEP21_2026_PERFORMANCE",
        }
        and contract["performance_status"] == "NOT_EVALUATED",
        {
            "protocol_status": protocol["status"],
            "contract_status": contract["status"],
            "contract_performance_status": contract["performance_status"],
        },
    )
    protocol_station_ids = set(str(value) for value in protocol["station_ids"])
    retained_station_ids = set(
        str(value) for value in download["retained_station_ids"]
    )
    check(
        "partial_download_boundary",
        str(download["status"]).startswith("FAIL_")
        and int(download["station_count"]) == int(protocol["station_count"])
        and int(download["retained_station_count"]) == len(retained_station_ids)
        and int(download["failed_count"]) >= 0
        and retained_station_ids.issubset(protocol_station_ids),
        {
            "station_count": download["station_count"],
            "retained_station_count": download["retained_station_count"],
            "failed_count": download["failed_count"],
            "retained_station_ids": download["retained_station_ids"],
        },
    )
    check(
        "exploratory_and_partial_flags",
        bool(result["exploratory"])
        and bool(result["partial_station_subset"])
        and int(result["station_count"]) == len(retained_station_ids)
        and str(result["status"]).startswith("COMPLETED_")
        and "EXPLORATORY" in str(result["status"]),
        {
            "status": result["status"],
            "exploratory": result["exploratory"],
            "partial_station_subset": result["partial_station_subset"],
            "station_count": result["station_count"],
        },
    )
    check(
        "evaluation_shape",
        len(evaluation)
        == int(result["seed_count"]) * int(result["method_count"])
        and len(station)
        == int(result["station_count"])
        * int(result["seed_count"])
        * int(result["method_count"])
        and evaluation["seed"].nunique() == 5
        and evaluation["method"].nunique() == 14
        and evaluation.groupby(["source", "seed"]).size().eq(14).all(),
        {
            "evaluation_rows": int(len(evaluation)),
            "station_rows": int(len(station)),
            "seed_count": int(evaluation["seed"].nunique()),
            "method_count": int(evaluation["method"].nunique()),
        },
    )
    check(
        "factor_scenario_shape",
        scenario["scenario"].nunique() == 9
        and scenario_summary["scenario"].nunique() == 9
        and scenario.groupby(["seed", "method"]).size().eq(9).all(),
        {
            "scenario_count": int(scenario["scenario"].nunique()),
            "scenario_names": sorted(scenario["scenario"].unique()),
        },
    )
    metric_columns = [
        "f1",
        "event_coverage",
        "row_recall",
        "average_precision",
        "false_alarm_rate",
    ]
    check(
        "finite_metrics",
        bool(evaluation[metric_columns].notna().all().all()),
        int(len(evaluation)),
    )
    check(
        "far_range",
        float(evaluation["false_alarm_rate"].min()) >= 0.0
        and float(evaluation["false_alarm_rate"].max()) <= 0.03,
        {
            "min": float(evaluation["false_alarm_rate"].min()),
            "max": float(evaluation["false_alarm_rate"].max()),
            "primary_far": float(contract["primary_far"]),
            "tolerance": 0.03,
        },
    )

    candidate = "trust_shape_union"
    candidate_values = (
        scenario_summary[scenario_summary["method"].eq(candidate)]
        .set_index("scenario")["event_coverage"]
    )
    win_rows: list[dict[str, object]] = []
    for baseline in BASELINES:
        reference = (
            scenario_summary[scenario_summary["method"].eq(baseline)]
            .set_index("scenario")["event_coverage"]
        )
        for scenario_name in candidate_values.index:
            difference = float(
                candidate_values[scenario_name] - reference[scenario_name]
            )
            win_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "scenario": scenario_name,
                    "candidate_event_coverage": float(
                        candidate_values[scenario_name]
                    ),
                    "baseline_event_coverage": float(
                        reference[scenario_name]
                    ),
                    "difference": difference,
                    "win": bool(difference > 0.0),
                }
            )
    wins = pd.DataFrame(win_rows)
    win_share = float(wins["win"].mean())
    unseen = wins[wins["scenario"].eq("coordinated_drift_unseen")]
    check(
        "scenario_win_share_ge_070",
        win_share >= 0.70,
        {
            "wins": int(wins["win"].sum()),
            "total": int(len(wins)),
            "share": win_share,
        },
    )
    check(
        "unseen_combination_event_coverage_positive",
        float(candidate_values["coordinated_drift_unseen"]) > 0.0,
        float(candidate_values["coordinated_drift_unseen"]),
    )

    paired = paired_table(station)
    comparison_rows: list[dict[str, object]] = []
    for candidate_name in TRUST_CANDIDATES:
        for baseline in BASELINES:
            stats = s248.hierarchical_bootstrap(
                paired,
                candidate_name,
                baseline,
                iterations=10000,
            )
            comparison_rows.append(
                {
                    "candidate": candidate_name,
                    "baseline": baseline,
                    **stats,
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["f1_bh_q"] = s248.benjamini_hochberg(
        comparisons["f1_bootstrap_two_sided_p"].tolist()
    )
    comparisons["event_bh_q"] = s248.benjamini_hochberg(
        comparisons["event_coverage_bootstrap_two_sided_p"].tolist()
    )
    margin = float(contract["non_inferiority_margin"])
    comparisons["event_non_inferior"] = (
        comparisons["event_ci95_low"].gt(margin)
        & comparisons["event_bh_q"].le(0.05)
    )
    comparisons["event_supported"] = (
        comparisons["event_ci95_low"].gt(0.0)
        & comparisons["event_bh_q"].le(0.05)
    )
    comparisons["event_gate_pass"] = (
        comparisons["event_non_inferior"] | comparisons["event_supported"]
    )
    comparison_path = TABLES / f"{AUDIT_PREFIX}_comparisons.csv"
    comparisons.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    primary = comparisons[comparisons["candidate"].eq(candidate)]
    event_gate_count = int(primary["event_gate_pass"].sum())
    check(
        "trust_shape_union_statistical_event_gate",
        event_gate_count == len(BASELINES),
        {"passed": event_gate_count, "total": len(BASELINES)},
    )

    summary_rows: list[dict[str, object]] = []
    for candidate_name, group in comparisons.groupby("candidate", sort=True):
        summary_rows.append(
            {
                "candidate": candidate_name,
                "baseline_count": int(len(group)),
                "event_gate_pass_count": int(group["event_gate_pass"].sum()),
                "all_event_baselines_pass": bool(
                    group["event_gate_pass"].all()
                ),
                "mean_event_difference": float(
                    group["event_coverage_difference"].mean()
                ),
            }
        )
    comparison_summary = pd.DataFrame(summary_rows)
    comparison_summary_path = TABLES / f"{AUDIT_PREFIX}_summary.csv"
    comparison_summary.to_csv(
        comparison_summary_path, index=False, encoding="utf-8-sig"
    )

    integrity_passed = all(
        item["status"] == "PASS"
        for item in checks
        if item["check"] not in {
            "scenario_win_share_ge_070",
            "unseen_combination_event_coverage_positive",
            "trust_shape_union_statistical_event_gate",
        }
    )
    audit = {
        "status": (
            f"PASS_{provider_slug}_2026_PARTIAL_EXPLORATORY_INTEGRITY_ONLY"
            if integrity_passed
            and not (
                win_share >= 0.70
                and float(candidate_values["coordinated_drift_unseen"]) > 0.0
                and event_gate_count == len(BASELINES)
            )
            else f"FAIL_{provider_slug}_2026_PARTIAL_EXPLORATORY_FACTOR_GATE"
        ),
        "integrity_passed": integrity_passed,
        "exploratory": True,
        "partial_station_subset": True,
        "protocol_id": protocol["protocol_id"],
        "contract_id": contract["contract_id"],
        "result_sha256": sha256(result_path),
        "station_sha256": sha256(TABLES / f"{EVAL_PREFIX}_station.csv"),
        "comparison_sha256": sha256(comparison_path),
        "comparison_summary_sha256": sha256(comparison_summary_path),
        "checks": checks,
        "station_count": int(result["station_count"]),
        "seed_count": 5,
        "baseline_count": len(BASELINES),
        "scenario_win_share": {
            "candidate": candidate,
            "wins": int(wins["win"].sum()),
            "total": int(len(wins)),
            "share": win_share,
        },
        "unseen_combination_event_coverage": float(
            candidate_values["coordinated_drift_unseen"]
        ),
        "statistical_event_gate": {
            "candidate": candidate,
            "passed": event_gate_count,
            "total": len(BASELINES),
            "comparisons": primary.to_dict("records"),
        },
        "ranking": summary.sort_values(
            "event_coverage_mean", ascending=False
        ).to_dict("records"),
        "claim_boundary": [
            "This is an exploratory partial temporal window, not a complete confirmatory result.",
            f"{protocol.get('provider', 'The provider')} is an existing provider; this does not add an independent institution.",
            "Stations failing the predeclared coverage rule are retained in the audit and excluded from the partial evaluation.",
            "All attacks are controlled simulations, not production incidents.",
            "A scenario or statistical gate failure prevents any comprehensive-leadership claim.",
        ],
    }
    out_json = TABLES / f"{AUDIT_PREFIX}.json"
    out_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    wins.to_csv(
        TABLES / f"{AUDIT_PREFIX}_scenario_win_loss.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = [
        "# Stage388 部分时间窗因子独立审计",
        "",
        f"- 状态：`{audit['status']}`",
        f"- 完整性门：`{integrity_passed}`；探索性：`{audit['exploratory']}`；部分站点：`{audit['partial_station_subset']}`。",
        f"- 保留站点：`{download['retained_station_count']}/{download['station_count']}`；覆盖率失败站点已在下载审计中原样保留。",
        f"- 场景×基线胜出：`{int(wins['win'].sum())}/{len(wins)}={win_share:.1%}`。",
        f"- 未见组合 coordinated_drift_unseen event coverage：`{float(candidate_values['coordinated_drift_unseen']):.6f}`。",
        f"- trust_shape_union 统计 event coverage 门槛：`{event_gate_count}/{len(BASELINES)}`。",
        "",
        "## 方法汇总",
        "",
        summary.sort_values("event_coverage_mean", ascending=False).to_markdown(
            index=False
        ),
        "",
        "## 统计汇总",
        "",
        comparison_summary.to_markdown(index=False),
        "",
        "## 边界",
        "",
        *[f"- {item}" for item in audit["claim_boundary"]],
    ]
    (REPORTS / f"{AUDIT_PREFIX}.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "integrity_passed": integrity_passed,
                "scenario_win_share": win_share,
                "wins": int(wins["win"].sum()),
                "total": int(len(wins)),
                "unseen_event_coverage": float(
                    candidate_values["coordinated_drift_unseen"]
                ),
                "statistical_event_gate": f"{event_gate_count}/{len(BASELINES)}",
                "failed_checks": [
                    item["check"]
                    for item in checks
                    if item["status"] == "FAIL"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
