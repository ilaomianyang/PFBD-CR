from __future__ import annotations

"""Stage 298: read-only failure decomposition of the Japan transfer.

This stage does not retrain a method, calibrate a threshold, generate an
attack, or inspect any new test. It reads the frozen Stage297 comparison and
evaluation tables and records exactly why the pre-registered transfer gate
did not pass. In particular, it preserves the adverse ARIMA/GRU event
coverage results and the fact that station-level and attack-type-level
Stage297 details were not archived by the original run.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage298_japan_transfer_failure_decomposition"
SOURCE_PREFIX = "stage297_japan_isd_locked_transfer"
PROTOCOL_ID = "TMQC-JAPAN-TRANSFER-FAILURE-DECOMPOSITION-20260921-v1.0"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


def _load_json(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(name: str) -> pd.DataFrame:
    path = TABLES / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _direction(value: float) -> str:
    if float(value) > 0.0:
        return "trust_higher"
    if float(value) < 0.0:
        return "baseline_higher"
    return "tie"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    source = _load_json(SOURCE_PREFIX)
    comparisons = _load_csv(f"{SOURCE_PREFIX}_comparisons")
    evaluation = _load_csv(f"{SOURCE_PREFIX}_evaluation")
    provider_equal = _load_csv(f"{SOURCE_PREFIX}_provider_equal")

    required = {
        "candidate",
        "baseline",
        "f1_difference",
        "f1_ci95_low",
        "f1_ci95_high",
        "event_coverage_difference",
        "event_ci95_low",
        "event_ci95_high",
        "f1_bootstrap_bh_q",
        "event_coverage_bootstrap_bh_q",
        "event_coverage_non_inferior",
        "event_coverage_primary_supported",
    }
    missing = sorted(required - set(comparisons.columns))
    if missing:
        raise RuntimeError(f"Stage297 comparison columns missing: {missing}")
    if len(comparisons) != int(source["primary_comparison_count"]):
        raise RuntimeError("Stage297 primary comparison count changed")

    primary_far = float(source["primary_far"])
    primary = evaluation[evaluation["max_far"].eq(primary_far)].copy()
    if primary.empty:
        raise RuntimeError("Stage297 primary-FAR evaluation is empty")
    if int(primary["seed"].nunique()) != int(source["seed_count"]):
        raise RuntimeError("Stage297 primary seed count changed")
    if int(primary["method"].nunique()) != int(source["method_count"]):
        raise RuntimeError("Stage297 primary method count changed")

    comparison_rows: list[dict[str, object]] = []
    for row in comparisons.to_dict("records"):
        advantage = bool(row["event_coverage_primary_supported"])
        non_inferior = bool(row["event_coverage_non_inferior"])
        passed = bool(advantage or non_inferior)
        if advantage:
            reason = "pre_registered_event_coverage_advantage"
        elif non_inferior:
            reason = "pre_registered_event_coverage_non_inferiority"
        elif float(row["event_ci95_low"]) <= 0.0:
            reason = "event_coverage_interval_includes_or_below_zero"
        else:
            reason = "event_coverage_advantage_not_significant_after_bh"
        comparison_rows.append(
            {
                "candidate": str(row["candidate"]),
                "baseline": str(row["baseline"]),
                "f1_difference": float(row["f1_difference"]),
                "f1_ci95_low": float(row["f1_ci95_low"]),
                "f1_ci95_high": float(row["f1_ci95_high"]),
                "f1_direction": _direction(float(row["f1_difference"])),
                "f1_bootstrap_bh_q": float(row["f1_bootstrap_bh_q"]),
                "event_coverage_difference": float(
                    row["event_coverage_difference"]
                ),
                "event_ci95_low": float(row["event_ci95_low"]),
                "event_ci95_high": float(row["event_ci95_high"]),
                "event_direction": _direction(
                    float(row["event_coverage_difference"])
                ),
                "event_coverage_bootstrap_bh_q": float(
                    row["event_coverage_bootstrap_bh_q"]
                ),
                "event_coverage_advantage_supported": advantage,
                "event_coverage_non_inferior": non_inferior,
                "primary_pass": passed,
                "failure_reason": reason,
            }
        )
    detail = pd.DataFrame(comparison_rows).sort_values(
        ["primary_pass", "candidate", "baseline"],
        ascending=[True, True, True],
    )
    failed = detail[~detail["primary_pass"]].copy()

    method_summary = (
        primary.groupby("method", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            f1_mean=("f1", "mean"),
            event_coverage_mean=("event_coverage", "mean"),
            event_coverage_min=("event_coverage", "min"),
            event_coverage_max=("event_coverage", "max"),
            row_recall_mean=("row_recall", "mean"),
            false_alarm_rate_mean=("false_alarm_rate", "mean"),
        )
        .sort_values(
            ["event_coverage_mean", "method"],
            ascending=[False, True],
        )
    )
    seed_summary = (
        primary.groupby(["method", "seed"], as_index=False)
        .agg(
            f1=("f1", "mean"),
            event_coverage=("event_coverage", "mean"),
            row_recall=("row_recall", "mean"),
            false_alarm_rate=("false_alarm_rate", "mean"),
        )
        .sort_values(["method", "seed"])
    )
    provider_rows = provider_equal[
        provider_equal["max_far"].eq(primary_far)
    ].copy()
    if provider_rows.empty:
        raise RuntimeError("Stage297 primary provider-equal rows are empty")

    seed_directions: list[dict[str, object]] = []
    for row in failed.to_dict("records"):
        candidate = str(row["candidate"])
        baseline = str(row["baseline"])
        current = seed_summary[seed_summary["method"].eq(candidate)]
        reference = seed_summary[seed_summary["method"].eq(baseline)]
        aligned = current.merge(
            reference,
            on="seed",
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        for item in aligned.to_dict("records"):
            seed_directions.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "seed": int(item["seed"]),
                    "f1_difference": float(
                        item["f1_candidate"] - item["f1_baseline"]
                    ),
                    "event_coverage_difference": float(
                        item["event_coverage_candidate"]
                        - item["event_coverage_baseline"]
                    ),
                    "row_recall_difference": float(
                        item["row_recall_candidate"]
                        - item["row_recall_baseline"]
                    ),
                }
            )
    seed_detail = pd.DataFrame(seed_directions)

    payload = {
        "status": "PASS_READ_ONLY_JAPAN_TRANSFER_FAILURE_DECOMPOSITION",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "Stage298 only reads the frozen Stage297 comparison, evaluation "
            "and provider-equal tables. It does not retrain, recalibrate, "
            "generate attacks, inspect new test labels, or create a new "
            "confirmatory result."
        ),
        "source_stage": SOURCE_PREFIX,
        "source_transfer_gate_status": source["transfer_gate_status"],
        "source_primary_far": primary_far,
        "source_primary_endpoint": source["primary_endpoint"],
        "source_provider": source["provider"],
        "source_provider_count": int(source["provider_count"]),
        "source_station_count": int(source["station_count"]),
        "source_seed_count": int(source["seed_count"]),
        "source_method_count": int(source["method_count"]),
        "comparison_count": int(len(detail)),
        "passed_comparison_count": int(detail["primary_pass"].sum()),
        "failed_comparison_count": int((~detail["primary_pass"]).sum()),
        "source_primary_pass_count": int(source["primary_pass_count"]),
        "source_primary_comparison_count": int(
            source["primary_comparison_count"]
        ),
        "all_baselines_supported": bool(
            source["all_baselines_supported"]
        ),
        "failed_comparisons": detail[~detail["primary_pass"]].to_dict(
            "records"
        ),
        "failed_seed_directions": seed_detail.to_dict("records"),
        "method_summary": method_summary.to_dict("records"),
        "provider_equal": provider_rows.to_dict("records"),
        "station_level_detail_available": False,
        "attack_type_level_detail_available": False,
        "station_level_recovery_requires_stage297_rerun": True,
        "attack_type_level_recovery_requires_stage297_rerun": True,
        "claim_ceiling": (
            "The Japan locked transfer does not pass the pre-registered "
            "all-baseline gate. The adverse ARIMA and GRU event-coverage "
            "results must be retained. Station-level and attack-type-level "
            "failure attribution cannot be recovered from the archived "
            "Stage297 outputs without a rerun, which would not be a new "
            "confirmatory test."
        ),
        "input_sha256": {
            SOURCE_PREFIX: sha256_file(
                TABLES / f"{SOURCE_PREFIX}.json"
            ),
            f"{SOURCE_PREFIX}_comparisons": sha256_file(
                TABLES / f"{SOURCE_PREFIX}_comparisons.csv"
            ),
            f"{SOURCE_PREFIX}_evaluation": sha256_file(
                TABLES / f"{SOURCE_PREFIX}_evaluation.csv"
            ),
            f"{SOURCE_PREFIX}_provider_equal": sha256_file(
                TABLES / f"{SOURCE_PREFIX}_provider_equal.csv"
            ),
        },
        "limitations": [
            "This is a read-only decomposition, not a new experiment.",
            "The Japan network is under the same NOAA provider.",
            "All attacks are transparent controlled injections.",
            "The 2025 test period is partial and ends on 2025-08-24.",
            "No station-level or attack-type-level Stage297 detail was "
            "archived; that gap must be reported rather than filled by "
            "selective reanalysis.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    if int(payload["failed_comparison_count"]) != (
        int(source["primary_comparison_count"])
        - int(source["primary_pass_count"])
    ):
        raise RuntimeError("Stage297 primary pass count is inconsistent")

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    detail.to_csv(
        TABLES / f"{PREFIX}_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    failed.to_csv(
        TABLES / f"{PREFIX}_failed_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seed_detail.to_csv(
        TABLES / f"{PREFIX}_failed_seed_directions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    method_summary.to_csv(
        TABLES / f"{PREFIX}_method_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = [
        "# Stage 298 日本锁定迁移失败分解",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 源迁移门槛：`{payload['source_transfer_gate_status']}`",
        f"- 主终点：`{payload['source_primary_endpoint']}`，主 FAR：`{payload['source_primary_far']}`",
        f"- 比较通过：`{payload['passed_comparison_count']}/{payload['comparison_count']}`",
        f"- 比较未通过：`{payload['failed_comparison_count']}/{payload['comparison_count']}`",
        f"- 站点级明细可用：`{payload['station_level_detail_available']}`",
        f"- 攻击类型级明细可用：`{payload['attack_type_level_detail_available']}`",
        "",
        "## 未通过比较",
        "",
        "| 候选 | 基线 | 事件覆盖差 | 95% CI | BH q | F1差 | 失败原因 |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for row in failed.to_dict("records"):
        report.append(
            f"| {row['candidate']} | {row['baseline']} | "
            f"{row['event_coverage_difference']:+.6f} | "
            f"[{row['event_ci95_low']:+.6f}, "
            f"{row['event_ci95_high']:+.6f}] | "
            f"{row['event_coverage_bootstrap_bh_q']:.6f} | "
            f"{row['f1_difference']:+.6f} | "
            f"{row['failure_reason']} |"
        )
    report += [
        "",
        "## 边界",
        "",
        "- 本阶段只读取 Stage297 冻结结果，不训练、不调阈值、不生成攻击。",
        "- 日本网络仍属 NOAA，不能关闭 provider 独立性问题。",
        "- Stage297 没有归档站点级和攻击类型级明细；该缺口不能靠选择性重分析补齐。",
        "- ARIMA 的显著负向结果和 GRU 的方向负向结果必须保留。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "comparison_count": payload["comparison_count"],
                "passed_comparison_count": payload[
                    "passed_comparison_count"
                ],
                "failed_comparison_count": payload[
                    "failed_comparison_count"
                ],
                "source_transfer_gate_status": payload[
                    "source_transfer_gate_status"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
