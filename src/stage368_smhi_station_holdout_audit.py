from __future__ import annotations

"""Stage 368: independent audit of the Stage367 SMHI station holdout."""

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage367_smhi_station_holdout_locked_transfer"
AUDIT_PREFIX = "stage368_smhi_station_holdout_audit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_frame(frame: pd.DataFrame, columns: list[str]) -> bool:
    return bool(
        frame[columns].apply(pd.to_numeric, errors="coerce").notna().all().all()
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result_path = TABLES / f"{PREFIX}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    download = json.loads(
        (
            TABLES / "stage365_smhi_station_holdout_download.json"
        ).read_text(encoding="utf-8")
    )
    contract = json.loads(
        (
            TABLES / "stage366_smhi_station_holdout_contract_evaluation_contract.json"
        ).read_text(encoding="utf-8")
    )
    evaluation = pd.read_csv(TABLES / f"{PREFIX}_evaluation.csv")
    station = pd.read_csv(TABLES / f"{PREFIX}_station_detail.csv")
    comparisons = pd.read_csv(TABLES / f"{PREFIX}_comparisons.csv")
    type_aggregate = pd.read_csv(TABLES / f"{PREFIX}_type_aggregate.csv")
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, value: object) -> None:
        checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "value": value}
        )

    check(
        "locked_result_status",
        result.get("status") == "COMPLETED_SMHI_LOCKED_TRANSFER",
        result.get("status"),
    )
    check(
        "download_audit_pass",
        download.get("status")
        == "PASS_SMHI_STATION_HOLDOUT_DOWNLOAD_AND_COVERAGE_AUDIT"
        and download.get("failed_count") == 0
        and download.get("retained_station_count") == 18,
        {
            "failed_count": download.get("failed_count"),
            "retained_station_count": download.get("retained_station_count"),
        },
    )
    check(
        "contract_frozen_before_performance",
        contract.get("status") == "EVALUATION_CONTRACT_FROZEN_BEFORE_PERFORMANCE"
        and contract.get("performance_status") == "NOT_EVALUATED",
        {
            "status": contract.get("status"),
            "performance_status": contract.get("performance_status"),
        },
    )
    check(
        "station_count",
        int(result.get("station_count", -1)) == 18
        and station["station_id"].nunique() == 18,
        int(station["station_id"].nunique()),
    )
    check(
        "seed_count",
        int(result.get("seed_count", -1)) == 5
        and evaluation["seed"].nunique() == 5,
        int(evaluation["seed"].nunique()),
    )
    check(
        "method_count",
        int(result.get("method_count", -1)) == 13
        and evaluation["method"].nunique() == 13,
        int(evaluation["method"].nunique()),
    )
    check(
        "evaluation_rows",
        len(evaluation) == 65
        and evaluation.groupby(["source", "seed"]).size().eq(13).all(),
        int(len(evaluation)),
    )
    check(
        "station_rows",
        len(station) == 1170
        and station.groupby(["source", "seed", "method"]).size().eq(18).all(),
        int(len(station)),
    )
    check(
        "type_rows",
        len(type_aggregate) == 78
        and type_aggregate["attack_type"].nunique() == 6,
        {
            "rows": int(len(type_aggregate)),
            "attack_types": sorted(type_aggregate["attack_type"].unique()),
        },
    )
    check(
        "metric_values_finite",
        finite_frame(
            evaluation,
            [
                "precision",
                "recall",
                "f1",
                "false_alarm_rate",
                "average_precision",
                "event_coverage",
                "row_recall",
            ],
        ),
        int(len(evaluation)),
    )
    far_max = float(evaluation["false_alarm_rate"].max())
    check(
        "far_values_reasonable",
        0.0 <= float(evaluation["false_alarm_rate"].min())
        and far_max <= 0.02,
        {
            "min": float(evaluation["false_alarm_rate"].min()),
            "max": far_max,
        },
    )
    check(
        "comparison_rows",
        len(comparisons) == 22
        and set(comparisons["candidate"]) == {
            "trust_valid_f1",
            "trust_nonstuck_event_macro",
        },
        int(len(comparisons)),
    )
    supported = (
        comparisons["event_coverage_primary_supported"].astype(bool)
        | comparisons["event_coverage_non_inferior"].astype(bool)
    )
    check(
        "all_primary_pairs_supported",
        bool(supported.all()),
        {
            "pass": int(supported.sum()),
            "total": int(len(supported)),
        },
    )
    ci_failures = comparisons[
        ~comparisons["event_coverage_primary_supported"].astype(bool)
        & ~comparisons["event_coverage_non_inferior"].astype(bool)
    ][
        [
            "candidate",
            "baseline",
            "event_coverage_difference",
            "event_ci95_low",
            "event_ci95_high",
            "event_coverage_bootstrap_bh_q",
        ]
    ].to_dict("records")
    check(
        "comparison_statistics_finite",
        finite_frame(
            comparisons,
            [
                "event_coverage_difference",
                "event_ci95_low",
                "event_ci95_high",
                "event_coverage_bootstrap_bh_q",
            ],
        ),
        int(len(comparisons)),
    )
    provider_equal = pd.read_csv(TABLES / f"{PREFIX}_provider_equal.csv")
    ranking = (
        provider_equal[
            ["method", "f1_mean", "event_coverage_mean", "false_alarm_rate_mean"]
        ]
        .sort_values("event_coverage_mean", ascending=False)
        .to_dict("records")
    )
    attack_summary = (
        type_aggregate.groupby("method", as_index=False)
        .agg(
            mean_event_coverage=("event_coverage", "mean"),
            mean_row_recall=("row_recall", "mean"),
        )
        .sort_values("mean_event_coverage", ascending=False)
        .to_dict("records")
    )
    audit = {
        "status": "FAIL_SMHI_STATION_HOLDOUT_SCIENTIFIC_GATE"
        if not bool(supported.all())
        else "PASS_SMHI_STATION_HOLDOUT_SCIENTIFIC_GATE",
        "protocol_id": contract["holdout_protocol_id"],
        "contract_id": contract["contract_id"],
        "result_sha256": sha256_file(result_path),
        "download_sha256": sha256_file(
            TABLES / "stage365_smhi_station_holdout_download.json"
        ),
        "evaluation_rows": int(len(evaluation)),
        "station_rows": int(len(station)),
        "type_rows": int(len(type_aggregate)),
        "primary_supported_count": int(supported.sum()),
        "primary_comparison_count": int(len(supported)),
        "all_primary_pairs_supported": bool(supported.all()),
        "comparison_failures": ci_failures,
        "checks": checks,
        "provider_equal_ranking": ranking,
        "attack_type_summary": attack_summary,
        "adverse_results": {
            "trust_valid_f1_vs_arima_event_difference": float(
                comparisons.loc[
                    (comparisons["candidate"] == "trust_valid_f1")
                    & (comparisons["baseline"] == "arima_innovation"),
                    "event_coverage_difference",
                ].iloc[0]
            ),
            "trust_valid_f1_vs_one_class_svm_event_difference": float(
                comparisons.loc[
                    (comparisons["candidate"] == "trust_valid_f1")
                    & (comparisons["baseline"] == "one_class_svm"),
                    "event_coverage_difference",
                ].iloc[0]
            ),
            "trust_nonstuck_vs_arima_event_difference": float(
                comparisons.loc[
                    (comparisons["candidate"] == "trust_nonstuck_event_macro")
                    & (comparisons["baseline"] == "arima_innovation"),
                    "event_coverage_difference",
                ].iloc[0]
            ),
        },
        "claim_boundary": [
            "This is an untouched station holdout within SMHI, not a new independent provider.",
            "All attacks are controlled simulations, not real production incidents.",
            "The 8/22 primary pair pass count does not support comprehensive superiority.",
            "ARIMA, One-Class SVM, kNN and reconstruction baselines remain adverse or stronger on this holdout.",
        ],
    }
    output = TABLES / f"{AUDIT_PREFIX}.json"
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = REPORTS / f"{AUDIT_PREFIX}.md"
    report.write_text(
        "\n".join(
            [
                "# Stage368 SMHI 未使用站点锁定评价独立审计",
                "",
                f"- 状态：`{audit['status']}`",
                f"- 主比较通过：`{audit['primary_supported_count']}/{audit['primary_comparison_count']}`",
                f"- 全部基线通过：`{audit['all_primary_pairs_supported']}`",
                "",
                "## 主要排序",
                "",
                "| 方法 | F1 | event coverage | FAR |",
                "|---|---:|---:|---:|",
            ]
            + [
                f"| {row['method']} | {float(row['f1_mean']):.6f} | "
                f"{float(row['event_coverage_mean']):.6f} | "
                f"{float(row['false_alarm_rate_mean']):.6f} |"
                for row in ranking
            ]
            + [
                "",
                "## 失败比较",
                "",
                *[
                    f"- `{row['candidate']} vs {row['baseline']}`："
                    f" event difference={float(row['event_coverage_difference']):.6f},"
                    f" CI=[{float(row['event_ci95_low']):.6f},"
                    f" {float(row['event_ci95_high']):.6f}],"
                    f" BH q={float(row['event_coverage_bootstrap_bh_q']):.6g}"
                    for row in ci_failures
                ],
                "",
                "## 边界",
                "",
                "- 新站点结果只能作为 SMHI 内部 station-holdout 证据，不能证明机构无关。",
                "- 所有攻击都是透明受控模拟，不是真实生产攻击。",
                "- ARIMA、One-Class SVM、kNN 和重建基线的胜出必须原样保留。",
                "- Stage367 不能替换 V6 主结果，也不能把新变体升级为全面领先。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "primary_supported_count": audit["primary_supported_count"],
                "primary_comparison_count": audit["primary_comparison_count"],
                "all_primary_pairs_supported": audit["all_primary_pairs_supported"],
                "failed_checks": [
                    item["check"] for item in checks if item["status"] == "FAIL"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
