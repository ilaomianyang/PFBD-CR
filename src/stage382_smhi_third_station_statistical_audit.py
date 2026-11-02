from __future__ import annotations

"""Station-clustered bootstrap and BH audit for Stage380."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
EVAL_PREFIX = os.environ.get(
    "TMQC_STAT_EVAL_PREFIX",
    "stage380_smhi_third_station_locked_evaluation",
)
CONTRACT = TABLES / f"{os.environ.get('TMQC_STAT_CONTRACT_PREFIX', 'stage379_smhi_third_station_contract')}.json"
RESULT = TABLES / f"{EVAL_PREFIX}.json"
STATION = TABLES / f"{EVAL_PREFIX}_station.csv"
OUT_PREFIX = os.environ.get(
    "TMQC_STAT_OUT_PREFIX",
    "stage382_smhi_third_station_statistical_audit",
)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
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
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    station = pd.read_csv(STATION)
    if contract["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("contract status changed")
    if result["status"] != "COMPLETED_SMHI_FACTOR_HOLDOUT_LOCKED_EVALUATION":
        raise RuntimeError("Stage380 result status changed")
    if len(station) != 18 * 5 * 14:
        raise RuntimeError("unexpected Stage380 station row count")
    paired = paired_table(station)
    rows: list[dict[str, object]] = []
    for candidate in TRUST_CANDIDATES:
        for baseline in BASELINES:
            stats = s248.hierarchical_bootstrap(
                paired,
                candidate,
                baseline,
                iterations=10000,
            )
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    **stats,
                }
            )
    comparisons = pd.DataFrame(rows)
    comparisons["f1_bh_q"] = s248.benjamini_hochberg(
        comparisons["f1_bootstrap_two_sided_p"].tolist()
    )
    comparisons["event_bh_q"] = s248.benjamini_hochberg(
        comparisons["event_coverage_bootstrap_two_sided_p"].tolist()
    )
    margin = float(contract["non_inferiority_margin"])
    comparisons["f1_supported"] = (
        comparisons["f1_ci95_low"].gt(0.0)
        & comparisons["f1_bh_q"].le(0.05)
    )
    comparisons["event_supported"] = (
        comparisons["event_ci95_low"].gt(0.0)
        & comparisons["event_bh_q"].le(0.05)
    )
    comparisons["event_non_inferior"] = (
        comparisons["event_ci95_low"].gt(margin)
        & comparisons["event_bh_q"].le(0.05)
    )
    comparisons["event_gate_pass"] = (
        comparisons["event_supported"] | comparisons["event_non_inferior"]
    )
    comparison_path = TABLES / f"{OUT_PREFIX}_comparisons.csv"
    comparisons.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    summary_rows = []
    for candidate, group in comparisons.groupby("candidate", sort=True):
        summary_rows.append(
            {
                "candidate": candidate,
                "baseline_count": int(len(group)),
                "f1_supported_count": int(group["f1_supported"].sum()),
                "event_supported_count": int(group["event_supported"].sum()),
                "event_non_inferior_count": int(group["event_non_inferior"].sum()),
                "event_gate_pass_count": int(group["event_gate_pass"].sum()),
                "all_event_baselines_pass": bool(group["event_gate_pass"].all()),
                "mean_event_difference": float(
                    group["event_coverage_difference"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = TABLES / f"{OUT_PREFIX}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    primary = summary[summary["candidate"].eq("trust_shape_union")].iloc[0]
    payload = {
        "status": (
            "PASS_SMHI_THIRD_STATION_STATISTICAL_INTEGRITY_ONLY"
            if bool(primary["all_event_baselines_pass"])
            else "FAIL_SMHI_THIRD_STATION_STATISTICAL_GATE"
        ),
        "exploratory": True,
        "contract_id": contract["contract_id"],
        "result_sha256": sha256(RESULT),
        "station_sha256": sha256(STATION),
        "comparison_sha256": sha256(comparison_path),
        "summary_sha256": sha256(summary_path),
        "station_count": 18,
        "seed_count": 5,
        "baseline_count": len(BASELINES),
        "non_inferiority_margin": margin,
        "comparisons": int(len(comparisons)),
        "candidate_summary": summary.to_dict("records"),
        "trust_shape_union_event_gate_pass_count": int(
            primary["event_gate_pass_count"]
        ),
        "trust_shape_union_event_gate_total": len(BASELINES),
        "claim_boundary": [
            "该统计审计只读取 Stage380 已完成结果，不能升级为确认性主结果。",
            "若任一主要基线稳定胜出，不能声称全面领先。",
            "该批是 SMHI 同一 provider 的站点 holdout，不是新增机构。",
            "攻击均为透明受控模拟，不是真实生产事故。",
        ],
    }
    out_json = TABLES / f"{OUT_PREFIX}.json"
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage382 SMHI 第三批站点统计审计",
        "",
        f"- 状态：`{payload['status']}`",
        "- bootstrap：站点聚类、10000 次；BH 校正；主终点 event coverage。",
        f"- Trust shape union：`{int(primary['event_gate_pass_count'])}/{len(BASELINES)}` 个基线通过 event coverage 优势或非劣效门槛。",
        "",
        summary.to_markdown(index=False),
        "",
        "## 边界",
        "",
        *[f"- {item}" for item in payload["claim_boundary"]],
    ]
    (REPORTS / f"{OUT_PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "trust_shape_union_event_gate_pass_count": int(
                    primary["event_gate_pass_count"]
                ),
                "baseline_count": len(BASELINES),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
