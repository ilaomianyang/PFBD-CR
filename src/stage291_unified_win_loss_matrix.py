from __future__ import annotations

"""Stage 291: unify all existing baseline win/loss evidence.

This is a read-only synthesis of the frozen Stage248 comparison and the
exploratory Stage256/262 comparisons. It does not train, calibrate, inspect
labels, alter predictions, or promote an exploratory result. Its purpose is
to make the complete win/loss ledger visible in one place while preserving
the evidence layer and the fact that it is not a new locked-test rerun.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage291_unified_win_loss_matrix"
PROTOCOL_ID = "TMQC-UNIFIED-WIN-LOSS-20260921-v1.0"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


SOURCES = [
    {
        "stage": "stage248_cross_provider_fair_baseline_audit",
        "evidence_layer": "frozen_same_contract_main_candidate",
        "exploratory_only": False,
        "comparison_file": (
            "stage248_cross_provider_fair_baseline_audit_comparisons.csv"
        ),
        "provider_equal_file": (
            "stage248_cross_provider_fair_baseline_audit_provider_equal.csv"
        ),
        "baseline_family": "classical_or_confirmatory",
    },
    {
        "stage": "stage256_lightweight_strong_baselines",
        "evidence_layer": "exploratory_reconstruction_candidate",
        "exploratory_only": True,
        "comparison_file": (
            "stage256_lightweight_strong_baselines_comparisons.csv"
        ),
        "provider_equal_file": (
            "stage256_lightweight_strong_baselines_provider_equal.csv"
        ),
        "baseline_family": "lightweight_reconstruction",
    },
    {
        "stage": "stage262_security_domain_baselines",
        "evidence_layer": "exploratory_security_or_series_candidate",
        "exploratory_only": True,
        "comparison_file": (
            "stage262_security_domain_baselines_comparisons.csv"
        ),
        "provider_equal_file": (
            "stage262_security_domain_baselines_provider_equal.csv"
        ),
        "baseline_family": "security_or_time_series",
    },
]


def _load_json(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    source_frames: list[pd.DataFrame] = []
    provider_frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    source_meta: list[dict[str, object]] = []

    for source in SOURCES:
        stage_name = str(source["stage"])
        stage = _load_json(stage_name)
        comparisons = _load_csv(str(source["comparison_file"]))
        provider_equal = _load_csv(str(source["provider_equal_file"]))
        if comparisons.empty:
            raise RuntimeError(f"empty comparisons: {stage_name}")
        required = {
            "candidate",
            "baseline",
            "f1_difference",
            "event_coverage_difference",
            "f1_ci95_low",
            "event_ci95_low",
            "f1_bootstrap_bh_q",
            "event_coverage_bootstrap_bh_q",
            "f1_primary_supported",
            "event_coverage_primary_supported",
        }
        missing = sorted(required - set(comparisons.columns))
        if missing:
            raise RuntimeError(
                f"{stage_name} comparison columns missing: {missing}"
            )
        frame = comparisons.copy()
        frame.insert(0, "source_stage", stage_name)
        frame.insert(
            1,
            "evidence_layer",
            str(source["evidence_layer"]),
        )
        frame.insert(
            2,
            "exploratory_only",
            bool(source["exploratory_only"]),
        )
        frame.insert(
            3,
            "baseline_family",
            str(source["baseline_family"]),
        )
        frame["f1_direction"] = frame["f1_difference"].map(
            lambda value: (
                "trust_higher"
                if float(value) > 0.0
                else "baseline_higher"
                if float(value) < 0.0
                else "tie"
            )
        )
        frame["event_direction"] = frame[
            "event_coverage_difference"
        ].map(
            lambda value: (
                "trust_higher"
                if float(value) > 0.0
                else "baseline_higher"
                if float(value) < 0.0
                else "tie"
            )
        )
        frame["f1_advantage_supported"] = (
            frame["f1_ci95_low"].astype(float).gt(0.0)
            & frame["f1_bootstrap_bh_q"].astype(float).le(0.05)
        )
        frame["event_advantage_supported"] = (
            frame["event_ci95_low"].astype(float).gt(0.0)
            & frame[
                "event_coverage_bootstrap_bh_q"
            ].astype(float).le(0.05)
        )
        frame["both_primary_advantages_supported"] = (
            frame["f1_advantage_supported"]
            & frame["event_advantage_supported"]
        )
        frame["claim_ceiling"] = (
            "FROZEN_CANDIDATE_LAYER"
            if not bool(source["exploratory_only"])
            else "EXPLORATORY_ONLY_UNTIL_NEW_LOCKED_TEST"
        )
        source_frames.append(frame)

        provider = provider_equal.copy()
        provider.insert(0, "source_stage", stage_name)
        provider.insert(
            1,
            "evidence_layer",
            str(source["evidence_layer"]),
        )
        provider.insert(
            2,
            "exploratory_only",
            bool(source["exploratory_only"]),
        )
        provider_frames.append(provider)

        source_hashes[stage_name] = sha256_file(
            TABLES / f"{stage_name}.json"
        )
        source_hashes[str(source["comparison_file"])] = sha256_file(
            TABLES / str(source["comparison_file"])
        )
        source_hashes[str(source["provider_equal_file"])] = sha256_file(
            TABLES / str(source["provider_equal_file"])
        )
        source_meta.append(
            {
                "source_stage": stage_name,
                "status": stage.get("status"),
                "primary_far": stage.get("primary_far"),
                "primary_endpoint": stage.get("primary_endpoint"),
                "provider_count": stage.get("provider_count"),
                "station_count": stage.get("station_count"),
                "seed_count": stage.get("seed_count"),
                "comparison_rows": int(len(comparisons)),
                "exploratory_only": bool(source["exploratory_only"]),
                "evidence_layer": str(source["evidence_layer"]),
            }
        )

    unified = pd.concat(source_frames, ignore_index=True)
    provider_unified = pd.concat(provider_frames, ignore_index=True)
    if unified.duplicated(["source_stage", "candidate", "baseline"]).any():
        raise RuntimeError("duplicate source/candidate/baseline rows")

    frozen = unified.loc[~unified["exploratory_only"].astype(bool)]
    exploratory = unified.loc[unified["exploratory_only"].astype(bool)]
    if frozen.empty:
        raise RuntimeError("no frozen comparison rows")
    if exploratory.empty:
        raise RuntimeError("no exploratory comparison rows")

    ledger = (
        unified.groupby(
            [
                "candidate",
                "baseline",
                "evidence_layer",
                "exploratory_only",
            ],
            as_index=False,
        )
        .agg(
            comparison_rows=("baseline", "size"),
            f1_advantage_supported=(
                "f1_advantage_supported",
                "sum",
            ),
            event_advantage_supported=(
                "event_advantage_supported",
                "sum",
            ),
            both_primary_advantages_supported=(
                "both_primary_advantages_supported",
                "sum",
            ),
            f1_difference_mean=("f1_difference", "mean"),
            event_coverage_difference_mean=(
                "event_coverage_difference",
                "mean",
            ),
        )
        .sort_values(
            [
                "candidate",
                "evidence_layer",
                "baseline",
            ]
        )
    )

    payload = {
        "status": "PASS_UNIFIED_WIN_LOSS_LEDGER_WITH_EVIDENCE_BOUNDARIES",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "Stage291 only reads existing frozen and exploratory "
            "comparison tables. It does not train, calibrate, inspect "
            "labels, alter predictions, or create a new locked test. It "
            "keeps frozen and exploratory layers separate."
        ),
        "source_count": len(SOURCES),
        "comparison_row_count": int(len(unified)),
        "frozen_comparison_row_count": int(len(frozen)),
        "exploratory_comparison_row_count": int(len(exploratory)),
        "baseline_count": int(
            unified["baseline"].nunique()
        ),
        "candidate_count": int(
            unified["candidate"].nunique()
        ),
        "frozen_baseline_count": int(
            frozen["baseline"].nunique()
        ),
        "exploratory_baseline_count": int(
            exploratory["baseline"].nunique()
        ),
        "all_rows_are_same_contract": True,
        "new_locked_test_rerun": False,
        "existing_evidence_win_loss_ledger_available": True,
        "same_contract_win_loss_matrix_available": False,
        "same_contract_win_loss_matrix_is_new_locked_test": False,
        "p0_3_status": "IN_PROGRESS",
        "claim_ceiling": (
            "The complete existing same-contract ledger is visible, but "
            "it is not a new locked-test rerun. Exploratory rows remain "
            "exploratory and no universal-superiority claim is allowed."
        ),
        "sources": source_meta,
        "input_sha256": source_hashes,
        "limitations": [
            "Stage248 is a frozen same-contract candidate audit, not a "
            "new untouched test.",
            "Stage256 and Stage262 rows remain exploratory until they are "
            "rerun on a new locked test.",
            "A positive or negative comparison here does not create a "
            "new confirmatory result.",
            "One-Class SVM still does not pass the F1 and event-coverage "
            "non-inferiority gates.",
            "All attacks are transparent controlled injections, not real "
            "production attacks.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    unified.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ledger.to_csv(
        TABLES / f"{PREFIX}_ledger.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provider_unified.to_csv(
        TABLES / f"{PREFIX}_provider_equal.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = [
        "# Stage 291 统一基线胜负矩阵",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 来源阶段：`{payload['source_count']}`",
        f"- 比较行：`{payload['comparison_row_count']}`",
        f"- 冻结层比较行：`{payload['frozen_comparison_row_count']}`",
        f"- 探索层比较行：`{payload['exploratory_comparison_row_count']}`",
        f"- 是否已有现有证据统一胜负账本：`{payload['existing_evidence_win_loss_ledger_available']}`",
        f"- 是否已有统一新锁定 test 胜负矩阵：`{payload['same_contract_win_loss_matrix_available']}`",
        f"- 是否是新的锁定 test 重跑：`{payload['same_contract_win_loss_matrix_is_new_locked_test']}`",
        f"- P0-3 状态：`{payload['p0_3_status']}`",
        "",
        "## 边界",
        "",
        "- 本阶段只汇总已有比较，不训练、不调阈值、不读取标签。",
        "- Stage248 冻结层与 Stage256/262 探索层严格分开。",
        "- 现有证据的统一胜负账本可用，但它不是新的锁定 test，不能升级为确认性结论。",
        "- `same_contract_win_loss_matrix_available=false`：尚未在新锁定 test 上统一重跑所有最终基线。",
        "- 所有受控攻击不是真实生产攻击日志。",
        "",
        "## 方法级账本",
        "",
        "| 候选 | 基线 | 证据层 | 探索性 | 比较行 | F1优势支持 | 事件覆盖优势支持 | 双主终点优势支持 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in ledger.to_dict("records"):
        report.append(
            f"| {row['candidate']} | {row['baseline']} | "
            f"{row['evidence_layer']} | {row['exploratory_only']} | "
            f"{int(row['comparison_rows'])} | "
            f"{int(row['f1_advantage_supported'])} | "
            f"{int(row['event_advantage_supported'])} | "
            f"{int(row['both_primary_advantages_supported'])} |"
        )
    report += [
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
        "",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": payload["status"],
                "sources": payload["source_count"],
                "comparison_rows": payload["comparison_row_count"],
                "frozen_rows": payload["frozen_comparison_row_count"],
                "exploratory_rows": payload[
                    "exploratory_comparison_row_count"
                ],
                "baselines": payload["baseline_count"],
                "p0_3_status": payload["p0_3_status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
