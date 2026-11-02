from __future__ import annotations

"""Stage 260: reviewer-risk closure audit after FMI and new baselines.

This stage does not train or evaluate a detector.  It reads the frozen
Stage246 package, the Stage248/254/255 risk evidence, the new Stage256
lightweight baselines, and the Stage259 FMI locked transfer.  It maps each
reviewer risk to its strongest current evidence and to an explicit claim
ceiling.

The audit is intentionally conservative: new evidence can move a risk from
``BOUNDED`` to ``PARTIALLY_CLOSED`` only when the supporting experiment
exists.  Production-incident, provider-independence, online-equals-offline,
and confirmatory-test-reuse claims remain forbidden unless their required
independent evidence exists.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage260_reviewer_risk_closure_audit"


def load_json(name: str) -> dict:
    return json.loads(
        (TABLES / f"{name}.json").read_text(encoding="utf-8")
    )


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stage246 = load_json("stage246_frozen_submission_package")
    stage248 = load_json("stage248_cross_provider_fair_baseline_audit")
    stage250 = load_json("stage250_jisa_risk_closure_audit")
    stage253 = load_json("stage253_residual_risk_optimization_plan")
    stage254 = load_json("stage254_ocsvm_training_scale_sensitivity")
    stage255 = load_json("stage255_online_causal_testbed")
    stage256 = load_json("stage256_lightweight_strong_baselines")
    stage259 = load_json("stage259_fmi_locked_transfer")
    stage261 = load_json("stage261_causal_offline_recovery_contrast")

    stage256_equal = pd.read_csv(
        TABLES / "stage256_lightweight_strong_baselines_provider_equal.csv"
    )
    stage259_equal = pd.read_csv(
        TABLES / "stage259_fmi_locked_transfer_provider_equal.csv"
    )
    stage259_comparisons = pd.read_csv(
        TABLES / "stage259_fmi_locked_transfer_comparisons.csv"
    )

    stage259_trust_f1 = float(
        stage259_equal.loc[
            stage259_equal["method"].eq("trust_valid_f1"),
            "f1_mean",
        ].iloc[0]
    )
    stage259_trust_event = float(
        stage259_equal.loc[
            stage259_equal["method"].eq("trust_valid_f1"),
            "event_coverage_mean",
        ].iloc[0]
    )
    stage259_best_f1_method = str(
        stage259_equal.sort_values("f1_mean", ascending=False).iloc[0][
            "method"
        ]
    )
    stage259_best_f1 = float(
        stage259_equal["f1_mean"].max()
    )
    stage259_best_event_method = str(
        stage259_equal.sort_values(
            "event_coverage_mean",
            ascending=False,
        ).iloc[0]["method"]
    )
    stage259_best_event = float(
        stage259_equal["event_coverage_mean"].max()
    )

    risks: list[dict[str, object]] = [
        {
            "risk_id": "R1_real_production_attack_superiority",
            "reviewer_question": (
                "Can the paper claim broad superiority on real production "
                "attacks?"
            ),
            "status": "BOUNDED",
            "evidence": [
                "stage252_authorized_testbed_protocol",
                "stage255_online_causal_testbed",
                "stage246_frozen_submission_package",
            ],
            "what_improved": (
                "Authorized loopback TCP testbed and online causal "
                "evaluation now exist; attack execution, packet timing, "
                "raw send/receive logs, and latency are real."
            ),
            "remaining_gap": (
                "No independent production incident or production fault "
                "log with auditable labels is available."
            ),
            "allowed_claim": (
                "Real runtime attacks under a transparent controlled "
                "testbed; not production attack recall."
            ),
            "forbidden_claim": (
                "Do not claim production-wide or real-incident attack "
                "superiority."
            ),
        },
        {
            "risk_id": "R2_provider_and_station_independence",
            "reviewer_question": (
                "Can the paper claim provider-independent or "
                "station-independent performance?"
            ),
            "status": "PARTIALLY_CLOSED",
            "evidence": [
                "stage248_cross_provider_fair_baseline_audit",
                "stage257_fmi_protocol_preregistration",
                "stage258_fmi_official_download",
                "stage259_fmi_locked_transfer",
            ],
            "what_improved": (
                "A fifth independent official provider (FMI) was "
                "pre-registered before download, downloaded with 60/60 "
                "successful station-year archives, coverage-audited, and "
                "evaluated under the same locked attack/FAR protocol."
            ),
            "remaining_gap": (
                "FMI has 11 retained stations, and Trust did not lead "
                "every FMI baseline; provider and station independence "
                "cannot be claimed."
            ),
            "allowed_claim": (
                "Five-provider external evaluation with a locked fifth "
                "provider test and an honest baseline comparison."
            ),
            "forbidden_claim": (
                "Do not claim provider independence, station "
                "independence, or universal superiority."
            ),
        },
        {
            "risk_id": "R3_online_causal_vs_offline_bidirectional",
            "reviewer_question": (
                "Is online causal recovery equivalent to offline "
                "bidirectional interpolation?"
            ),
            "status": "BOUNDED",
            "evidence": [
                "stage133_causal_temporal_recovery",
                "stage134_causal_recovery_optimization",
                "stage255_online_causal_testbed",
                "stage261_causal_offline_recovery_contrast",
            ],
            "what_improved": (
                "Stage255 evaluates strictly causal frame-level detection, "
                "arrival timing, recovery latency, throughput, and clean "
                "safety on the authorized testbed. Stage261 quantifies the "
                "causal penalty against an offline bidirectional "
                "interpolation reference on the same runtime attacks."
            ),
            "remaining_gap": (
                "Online causal recovery is not mathematically or "
                "empirically equivalent to offline bidirectional "
                "interpolation; Stage261 measures a nonzero causal penalty."
            ),
            "allowed_claim": (
                "Report online causal latency and safety trade-offs "
                "separately from offline interpolation."
            ),
            "forbidden_claim": (
                "Do not say online recovery equals offline bidirectional "
                "interpolation."
            ),
        },
        {
            "risk_id": "R4_confirmatory_test_reuse",
            "reviewer_question": (
                "Were already-explored test data reused for confirmatory "
                "tuning?"
            ),
            "status": "BOUNDED",
            "evidence": [
                "stage246_frozen_submission_package",
                "stage250_jisa_risk_closure_audit",
                "stage259_fmi_locked_transfer",
            ],
            "what_improved": (
                "Stage256 and Stage259 are isolated from the frozen "
                "Stage246 endpoint.  Stage259 uses a new provider and a "
                "protocol frozen before download; no old test is used to "
                "tune a new variant."
            ),
            "remaining_gap": (
                "Any new variant that is selected using old test feedback "
                "cannot be promoted into the confirmatory result."
            ),
            "allowed_claim": (
                "Frozen main result plus separate exploratory/locked "
                "external analyses."
            ),
            "forbidden_claim": (
                "Do not call a post-hoc variant confirmatory on reused "
                "test data."
            ),
        },
        {
            "risk_id": "R5_controlled_attack_and_weak_label_recall",
            "reviewer_question": (
                "Can controlled injections, weak labels, or testbed "
                "results be reported as real-incident recall?"
            ),
            "status": "BOUNDED",
            "evidence": [
                "stage246_frozen_submission_package",
                "stage252_authorized_testbed_protocol",
                "stage255_online_causal_testbed",
                "stage259_fmi_locked_transfer",
            ],
            "what_improved": (
                "The evidence now separates numeric controlled injection, "
                "authorized runtime testbed attacks, weak labels, and "
                "official quality labels in separate result tables."
            ),
            "remaining_gap": (
                "None of these sources is an independent production "
                "incident log with governed labels."
            ),
            "allowed_claim": (
                "Controlled attack recall, testbed event coverage, and "
                "weak-label association as separate evidence classes."
            ),
            "forbidden_claim": (
                "Do not write controlled or testbed results as real "
                "incident recall."
            ),
        },
        {
            "risk_id": "R6_lightweight_strong_baselines",
            "reviewer_question": (
                "Are there lightweight strong baselines beyond the "
                "original three?"
            ),
            "status": "PARTIALLY_CLOSED",
            "evidence": [
                "stage256_lightweight_strong_baselines",
                "stage259_fmi_locked_transfer",
            ],
            "what_improved": (
                "PCA reconstruction, MLP autoencoder reconstruction, and "
                "kNN distance were added under the same train-normal fit, "
                "valid-normal FAR calibration, and test-only evaluation "
                "contract."
            ),
            "remaining_gap": (
                "The new baselines are exploratory and some are stronger "
                "than Trust on FMI; they cannot be used to claim broad "
                "superiority."
            ),
            "allowed_claim": (
                "Report them as exploratory strong baselines and retain "
                "all wins and losses."
            ),
            "forbidden_claim": (
                "Do not merge them into the frozen confirmatory main "
                "result or hide their FMI wins."
            ),
        },
        {
            "risk_id": "R7_one_class_svm_complementarity",
            "reviewer_question": (
                "Does Trust fully dominate One-Class SVM?"
            ),
            "status": "BOUNDED",
            "evidence": [
                "stage248_cross_provider_fair_baseline_audit",
                "stage254_ocsvm_training_scale_sensitivity",
                "stage259_fmi_locked_transfer",
            ],
            "what_improved": (
                "Training-scale sensitivity was extended to 20k/50k/100k, "
                "and FMI provides a new independent target-domain check."
            ),
            "remaining_gap": (
                "One-Class SVM remains complementary and is stronger on "
                "some FMI F1/event-coverage comparisons."
            ),
            "allowed_claim": (
                "Report complementarity and all One-Class SVM wins."
            ),
            "forbidden_claim": (
                "Do not claim comprehensive superiority over One-Class "
                "SVM."
            ),
        },
    ]

    payload = {
        "status": "PASS_REVIEWER_RISK_CLOSURE_AUDIT",
        "protocol": (
            "阶段260只读取冻结主结果、阶段248/254/255风险证据、阶段256"
            "轻量强基线和阶段259 FMI第五机构锁定迁移，不训练、不选择"
            "阈值、不重新评价旧test。每个风险只允许在其证据范围内升级"
            "声称；生产事故、机构无关、在线等同离线、确认性test复用"
            "和受控攻击等同真实召回率均保持禁止。"
        ),
        "risk_count": int(len(risks)),
        "resolved_count": int(
            sum(risk["status"] == "RESOLVED" for risk in risks)
        ),
        "partially_closed_count": int(
            sum(
                risk["status"] == "PARTIALLY_CLOSED"
                for risk in risks
            )
        ),
        "bounded_count": int(
            sum(risk["status"] == "BOUNDED" for risk in risks)
        ),
        "risks": risks,
        "fmi_locked_transfer_summary": {
            "provider": stage259["provider"],
            "station_count": stage259["station_count"],
            "seed_count": stage259["seed_count"],
            "method_count": stage259["method_count"],
            "trust_valid_f1": stage259_trust_f1,
            "trust_valid_event_coverage": stage259_trust_event,
            "best_f1_method": stage259_best_f1_method,
            "best_f1": stage259_best_f1,
            "best_event_method": stage259_best_event_method,
            "best_event_coverage": stage259_best_event,
            "trust_f1_rank": int(
                (
                    stage259_equal["f1_mean"]
                    > stage259_trust_f1
                ).sum()
                + 1
            ),
            "trust_event_rank": int(
                (
                    stage259_equal["event_coverage_mean"]
                    > stage259_trust_event
                ).sum()
                + 1
            ),
            "comparisons": stage259["comparisons"],
        },
        "lightweight_baseline_summary": {
            "methods": stage256["new_baseline_methods"],
            "provider_equal": stage256["provider_equal"],
        },
        "causal_offline_contrast_summary": {
            "run_count": stage261["run_count"],
            "online_mae_mean": stage261["online_mae_mean"],
            "offline_mae_mean": stage261["offline_mae_mean"],
            "causal_penalty_mae_mean": stage261[
                "causal_penalty_mae_mean"
            ],
            "online_rmse_mean": stage261["online_rmse_mean"],
            "offline_rmse_mean": stage261["offline_rmse_mean"],
            "causal_penalty_rmse_mean": stage261[
                "causal_penalty_rmse_mean"
            ],
            "online_equals_offline_any": stage261[
                "online_equals_offline_any"
            ],
            "claim_ceiling": stage261["claim_ceiling"],
        },
        "input_hashes": {
            "stage246_frozen_submission_package.json": sha256_file(
                TABLES / "stage246_frozen_submission_package.json"
            ),
            "stage248_cross_provider_fair_baseline_audit.json": sha256_file(
                TABLES
                / "stage248_cross_provider_fair_baseline_audit.json"
            ),
            "stage254_ocsvm_training_scale_sensitivity.json": sha256_file(
                TABLES
                / "stage254_ocsvm_training_scale_sensitivity.json"
            ),
            "stage255_online_causal_testbed.json": sha256_file(
                TABLES / "stage255_online_causal_testbed.json"
            ),
            "stage256_lightweight_strong_baselines.json": sha256_file(
                TABLES
                / "stage256_lightweight_strong_baselines.json"
            ),
            "stage259_fmi_locked_transfer.json": sha256_file(
                TABLES / "stage259_fmi_locked_transfer.json"
            ),
            "stage261_causal_offline_recovery_contrast.json": sha256_file(
                TABLES
                / "stage261_causal_offline_recovery_contrast.json"
            ),
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(risks).to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段260 模拟审稿风险闭合审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 风险总数：`{payload['risk_count']}`",
        f"- 已解决：`{payload['resolved_count']}`",
        f"- 部分闭合：`{payload['partially_closed_count']}`",
        f"- 仍有边界：`{payload['bounded_count']}`",
        "",
        "## FMI 第五机构锁定结果",
        "",
        f"- Trust valid F1：`{stage259_trust_f1:.6f}`",
        f"- Trust valid 事件覆盖：`{stage259_trust_event:.6f}`",
        f"- 最佳 F1 方法：`{stage259_best_f1_method}` = "
        f"`{stage259_best_f1:.6f}`",
        f"- 最佳事件覆盖方法：`{stage259_best_event_method}` = "
        f"`{stage259_best_event:.6f}`",
        f"- Trust F1 排名：`{payload['fmi_locked_transfer_summary']['trust_f1_rank']}`",
        f"- Trust 事件覆盖排名：`{payload['fmi_locked_transfer_summary']['trust_event_rank']}`",
        "",
        "## 风险处置",
        "",
        *[
            f"- `{risk['risk_id']}`：`{risk['status']}`；"
            f"允许声称：{risk['allowed_claim']}；"
            f"禁止声称：{risk['forbidden_claim']}"
            for risk in risks
        ],
        "",
        "## 不能解决的边界",
        "",
        "- 没有独立生产事故/故障日志，不能证明真实生产环境攻击普适领先。",
        "- FMI 是第五个机构，但 Trust 未在 FMI 上全面领先，不能证明机构无关或站点无关。",
        "- 在线因果恢复不等于离线双向插值。",
        "- 已探索过的旧 test 不能再用于确认性调参。",
        "- 受控注入、弱标签或测试床结果不能写成真实事故召回率。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "risk_count": payload["risk_count"],
                "resolved_count": payload["resolved_count"],
                "partially_closed_count": payload[
                    "partially_closed_count"
                ],
                "bounded_count": payload["bounded_count"],
                "fmi_trust_f1_rank": payload[
                    "fmi_locked_transfer_summary"
                ]["trust_f1_rank"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
