from __future__ import annotations

"""Stage 373: independent integrity and factor-gate audit for Stage372."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = os.environ.get(
    "TMQC_FACTOR_EVAL_PREFIX",
    "stage372_smhi_factor_holdout_locked_evaluation",
)
AUDIT_PREFIX = os.environ.get(
    "TMQC_FACTOR_AUDIT_PREFIX",
    "stage373_smhi_factor_holdout_audit",
)
PROTOCOL_PREFIX = os.environ.get(
    "TMQC_FACTOR_PROTOCOL_PREFIX",
    "stage369_smhi_factor_holdout_protocol",
)
CONTRACT_PREFIX = os.environ.get(
    "TMQC_FACTOR_CONTRACT_PREFIX",
    "stage371_smhi_factor_holdout_contract",
)
DOWNLOAD_PREFIX = os.environ.get(
    "TMQC_FACTOR_DOWNLOAD_PREFIX",
    "stage370_smhi_factor_holdout_download",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result_path = TABLES / f"{PREFIX}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = json.loads(
        (
            TABLES / f"{PROTOCOL_PREFIX}.json"
        ).read_text(encoding="utf-8")
    )
    contract = json.loads(
        (
            TABLES / f"{CONTRACT_PREFIX}.json"
        ).read_text(encoding="utf-8")
    )
    download = json.loads(
        (
            TABLES / f"{DOWNLOAD_PREFIX}.json"
        ).read_text(encoding="utf-8")
    )
    evaluation = pd.read_csv(TABLES / f"{PREFIX}.csv")
    scenario = pd.read_csv(TABLES / f"{PREFIX}_scenario.csv")
    scenario_summary = pd.read_csv(
        TABLES / f"{PREFIX}_scenario_summary.csv"
    )
    summary = pd.read_csv(TABLES / f"{PREFIX}_summary.csv")
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
        "protocol_and_contract_frozen",
        protocol["status"] == "PROTOCOL_FROZEN_BEFORE_OBSERVATION_ACCESS"
        and contract["status"] == "EVALUATION_CONTRACT_FROZEN_BEFORE_PERFORMANCE"
        and contract["performance_status"] == "NOT_EVALUATED",
        {
            "protocol": protocol["status"],
            "contract": contract["status"],
            "performance_status": contract["performance_status"],
        },
    )
    check(
        "download_complete",
        download["status"]
        == "PASS_SMHI_STATION_HOLDOUT_DOWNLOAD_AND_COVERAGE_AUDIT"
        and download["failed_count"] == 0
        and download["retained_station_count"] == 18,
        {
            "expected_archives": download["expected_archives"],
            "downloaded_count": download["downloaded_count"],
            "failed_count": download["failed_count"],
            "retained_station_count": download["retained_station_count"],
        },
    )
    check(
        "shape",
        len(evaluation) == 70
        and evaluation["seed"].nunique() == 5
        and evaluation["method"].nunique() == 14
        and evaluation.groupby(["source", "seed"]).size().eq(14).all(),
        {
            "evaluation_rows": int(len(evaluation)),
            "seed_count": int(evaluation["seed"].nunique()),
            "method_count": int(evaluation["method"].nunique()),
        },
    )
    check(
        "factor_scenarios",
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
        and float(evaluation["false_alarm_rate"].max()) <= 0.02,
        {
            "min": float(evaluation["false_alarm_rate"].min()),
            "max": float(evaluation["false_alarm_rate"].max()),
        },
    )
    candidate = "trust_shape_union"
    baselines = [
        method
        for method in summary["method"].tolist()
        if method not in {
            candidate,
            "trust_valid_f1",
            "trust_nonstuck_event_macro",
        }
    ]
    candidate_values = scenario_summary[
        scenario_summary["method"].eq(candidate)
    ].set_index("scenario")["event_coverage"]
    win_rows: list[dict[str, object]] = []
    for baseline in baselines:
        reference = scenario_summary[
            scenario_summary["method"].eq(baseline)
        ].set_index("scenario")["event_coverage"]
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
    weak = wins[wins["scenario"].str.contains("offset_weak")]
    strong = wins[wins["scenario"].str.contains("offset_strong")]
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
        "unseen_combination_event_coverage",
        float(candidate_values["coordinated_drift_unseen"]) > 0.0,
        float(candidate_values["coordinated_drift_unseen"]),
    )
    check(
        "weak_factor_event_coverage",
        float(
            scenario_summary[
                (scenario_summary["method"].eq(candidate))
                & scenario_summary["scenario"].str.contains("offset_weak")
            ]["event_coverage"].mean()
        )
        > 0.0,
        float(
            scenario_summary[
                (scenario_summary["method"].eq(candidate))
                & scenario_summary["scenario"].str.contains("offset_weak")
            ]["event_coverage"].mean()
        ),
    )
    audit = {
        "status": (
            "PASS_SMHI_FACTOR_HOLDOUT_INTEGRITY_ONLY"
            if all(item["status"] == "PASS" for item in checks)
            else "FAIL_SMHI_FACTOR_HOLDOUT_FACTOR_GATE"
        ),
        "protocol_id": protocol["protocol_id"],
        "contract_id": contract["contract_id"],
        "exploratory": True,
        "result_sha256": sha256_file(result_path),
        "download_sha256": sha256_file(
            TABLES / f"{DOWNLOAD_PREFIX}.json"
        ),
        "checks": checks,
        "scenario_win_share": {
            "candidate": candidate,
            "wins": int(wins["win"].sum()),
            "total": int(len(wins)),
            "share": win_share,
        },
        "unseen_combination": unseen.to_dict("records"),
        "weak_factor": weak.to_dict("records"),
        "strong_factor": strong.to_dict("records"),
        "ranking": summary.sort_values(
            "event_coverage_mean", ascending=False
        ).to_dict("records"),
        "claim_boundary": [
            "This is exploratory because the protocol was initiated after Stage367 results were observed.",
            "This is a same-provider SMHI station holdout, not a new institution.",
            "All attacks are controlled simulations, not production incidents.",
            "The factor gate is not passed when the scenario win share is below 70% or unseen coverage is zero.",
        ],
    }
    output = TABLES / f"{AUDIT_PREFIX}.json"
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    win_path = TABLES / f"{AUDIT_PREFIX}_scenario_win_loss.csv"
    wins.to_csv(win_path, index=False, encoding="utf-8-sig")
    report = REPORTS / f"{AUDIT_PREFIX}.md"
    report.write_text(
        "\n".join(
            [
                "# Stage373 SMHI 因子压力测试独立审计",
                "",
                f"- 状态：`{audit['status']}`",
                f"- 探索性：`{audit['exploratory']}`",
                f"- 场景×基线胜出：`{int(wins['win'].sum())}/{len(wins)}={win_share:.1%}`",
                f"- coordinated_drift_unseen event coverage：`{float(candidate_values['coordinated_drift_unseen']):.6f}`",
                "",
                "## 方法汇总",
                "",
                summary.sort_values(
                    "event_coverage_mean", ascending=False
                ).to_markdown(index=False),
                "",
                "## 失败边界",
                "",
                "- 低于 70% 场景×基线门槛时，不得声称全因子全面领先。",
                "- 未见 coordinated drift 覆盖为 0 时，不能声称未见组合泛化。",
                "- One-Class SVM、kNN 和 ARIMA 的强 offset 结果高于 Trust，必须保留。",
                "- 本阶段不能替换 V6 或 Stage367，也不能把模拟攻击写成真实事故。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "win_share": win_share,
                "wins": int(wins["win"].sum()),
                "total": int(len(wins)),
                "unseen_event_coverage": float(
                    candidate_values["coordinated_drift_unseen"]
                ),
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
