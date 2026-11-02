from __future__ import annotations

"""Stage 314: machine-readable claim ceiling and submission scope audit.

This stage is deliberately read-only. It freezes the author's decision to stop
adding providers, maps the remaining P0 blockers to permitted and forbidden
paper wording, and prevents engineering or reproduction checks from being
promoted into scientific superiority claims.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage314_claim_ceiling_and_submission_scope_audit"
PROTOCOL_ID = "TMQC-CLAIM-CEILING-AND-SUBMISSION-SCOPE-20260922-v1.0"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


FROZEN_MAIN_PROVIDERS = [
    "ECCC",
    "KNMI",
    "MeteoSwiss",
    "NOAA ISD",
]
P0_IDS = [
    "P0-1_external_provider_gate",
    "P0-2_ocsvm_non_inferiority",
    "P0-3_public_ics_cps_baseline",
    "P0-4_nonstuck_stability",
    "P0-5_real_production_labels",
    "P0-6_new_confirmatory_test",
]


def _load(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _check(
    checks: list[dict[str, object]],
    check_id: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stage281 = _load("stage281_submission_blocker_register")
    stage246 = _load("stage246_frozen_submission_package")
    stage250 = _load("stage250_jisa_risk_closure_audit")
    stage251 = _load("stage251_jisa_stable_submission_objective")
    stage263 = _load("stage263_unified_statistical_report")
    stage293 = _load("stage293_confirmatory_package_freeze_audit")
    stage313 = _load("stage313_cross_locked_test_win_loss_matrix")

    p0_rows = [
        row
        for row in stage281.get("rows", [])
        if str(row.get("blocker_id", "")).startswith("P0-")
    ]
    p0_ids = [str(row.get("blocker_id", "")) for row in p0_rows]
    p0_status = {
        str(row.get("blocker_id", "")): str(row.get("status", ""))
        for row in p0_rows
    }

    checks: list[dict[str, object]] = []
    _check(
        checks,
        "stage281_six_p0_open",
        len(p0_rows) == 6
        and stage281.get("p0_open_count") == 6
        and all(p0_status.get(item) == "OPEN" for item in P0_IDS),
        (
            f"P0 rows={len(p0_rows)}, p0_open_count="
            f"{stage281.get('p0_open_count')}, statuses={p0_status}"
        ),
    )
    _check(
        checks,
        "universal_and_production_claims_closed",
        stage281.get("universal_superiority_allowed") is False
        and stage281.get("production_superiority_allowed") is False,
        (
            "universal_superiority_allowed="
            f"{stage281.get('universal_superiority_allowed')}; "
            "production_superiority_allowed="
            f"{stage281.get('production_superiority_allowed')}"
        ),
    )
    _check(
        checks,
        "four_provider_scope_frozen",
        stage246.get("provider_count") == 4
        and len(FROZEN_MAIN_PROVIDERS) == 4
        and stage313.get("new_provider_added") is False
        and stage313.get(
            "same_provider_networks_counted_as_new_provider"
        )
        is False,
        (
            f"frozen main providers={FROZEN_MAIN_PROVIDERS}; "
            f"stage246 provider_count={stage246.get('provider_count')}; "
            f"stage313 new_provider_added={stage313.get('new_provider_added')}; "
            "same_provider_networks_counted_as_new_provider="
            f"{stage313.get('same_provider_networks_counted_as_new_provider')}"
        ),
    )
    _check(
        checks,
        "stage313_adverse_results_preserved",
        stage313.get("comparison_row_count") == 74
        and stage313.get("normalized_primary_pass_count") == 32
        and stage313.get("normalized_primary_fail_count") == 42
        and stage313.get("all_baselines_supported_on_every_locked_test")
        is False
        and stage313.get("nonstuck_zero_event_cell_count") == 1
        and stage313.get("nonstuck_zero_row_cell_count") == 1,
        (
            "comparison_rows="
            f"{stage313.get('comparison_row_count')}; pass="
            f"{stage313.get('normalized_primary_pass_count')}; fail="
            f"{stage313.get('normalized_primary_fail_count')}; "
            "all_baselines_supported_on_every_locked_test="
            f"{stage313.get('all_baselines_supported_on_every_locked_test')}; "
            "zero_event_cells="
            f"{stage313.get('nonstuck_zero_event_cell_count')}; "
            "zero_row_cells="
            f"{stage313.get('nonstuck_zero_row_cell_count')}"
        ),
    )
    _check(
        checks,
        "risk_closure_remains_bounded",
        stage250.get("risk_count") == 7
        and stage250.get("resolved_count") == 0
        and stage250.get("bounded_count") == 7
        and stage251.get("risk_ceiling_count") == 7
        and stage251.get("resolved_count") == 0
        and stage251.get("bounded_count") == 7,
        (
            f"stage250 risk_count={stage250.get('risk_count')}, "
            f"resolved={stage250.get('resolved_count')}, "
            f"bounded={stage250.get('bounded_count')}; "
            f"stage251 risk_ceiling_count={stage251.get('risk_ceiling_count')}, "
            f"resolved={stage251.get('resolved_count')}, "
            f"bounded={stage251.get('bounded_count')}"
        ),
    )
    _check(
        checks,
        "stage263_provider_count_not_institution_count",
        stage263.get("provider_count") == 12
        and stage263.get("stage_count") == 3
        and "机构" in str(stage263.get("bootstrap_hierarchy", "")),
        (
            "stage263 provider_count=12 is a stage-record count, not 12 "
            "independent institutions; frozen main provider count is 4."
        ),
    )
    _check(
        checks,
        "confirmatory_package_present_but_not_new_provider",
        stage293.get("package_present") is True
        and stage293.get("new_provider_selected") is False
        and stage293.get("is_a_new_confirmatory_test_package") is False
        and stage293.get("analysis_frozen") is True,
        (
            f"package_present={stage293.get('package_present')}; "
            f"new_provider_selected={stage293.get('new_provider_selected')}; "
            "is_a_new_confirmatory_test_package="
            f"{stage293.get('is_a_new_confirmatory_test_package')}; "
            f"analysis_frozen={stage293.get('analysis_frozen')}"
        ),
    )
    _check(
        checks,
        "engineering_gate_is_not_scientific_claim",
        stage281.get("clean_environment_reproduction_allowed") is True
        and stage281.get("universal_superiority_allowed") is False,
        (
            "clean_environment_reproduction_allowed=true only supports "
            "engineering reproducibility; it does not close P0 or support "
            "performance, production, or universal-superiority claims."
        ),
    )

    failed_checks = [
        row for row in checks if not bool(row.get("passed"))
    ]
    if failed_checks:
        raise RuntimeError(
            "Stage314 claim-ceiling checks failed: "
            + json.dumps(failed_checks, ensure_ascii=False)
        )

    allowed_claims = [
        {
            "claim_id": "A1",
            "topic": "冻结公平基线优势",
            "allowed_wording": (
                "在冻结四机构、FAR=0.01 和透明受控注入协议下，"
                "Trust 相对 Robust max-deviation 与 Isolation Forest "
                "的 F1 优势区间为正。"
            ),
            "forbidden_wording": (
                "不能扩展为对所有公平基线全面领先。"
            ),
            "evidence_stages": (
                "stage246_frozen_submission_package;"
                "stage250_jisa_risk_closure_audit;"
                "stage251_jisa_stable_submission_objective"
            ),
            "condition": (
                "仅限冻结主终点、冻结候选、四机构和预定义低误报运行点。"
            ),
        },
        {
            "claim_id": "A2",
            "topic": "One-Class SVM 互补性",
            "allowed_wording": (
                "One-Class SVM 与 Trust 在 F1 和事件覆盖上呈互补关系，"
                "其训练预算和不利占优结果已披露。"
            ),
            "forbidden_wording": (
                "不能写成 Trust 对所有 One-Class SVM 预算或所有基线"
                "全面领先。"
            ),
            "evidence_stages": (
                "stage281_submission_blocker_register;"
                "stage313_cross_locked_test_win_loss_matrix"
            ),
            "condition": (
                "必须同时报告日本、MET Norway 和不相交 MET Norway "
                "站网上的不利结果。"
            ),
        },
        {
            "claim_id": "A3",
            "topic": "透明受控攻击有效性",
            "allowed_wording": (
                "方法在预定义透明攻击模型、冻结阈值和低误报运行点下"
                "有效。"
            ),
            "forbidden_wording": (
                "不能把受控注入写成真实生产攻击或真实事故召回率。"
            ),
            "evidence_stages": (
                "stage246_frozen_submission_package;"
                "stage250_jisa_risk_closure_audit"
            ),
            "condition": (
                "受控攻击、测试床攻击和自然弱标签必须分开呈现。"
            ),
        },
        {
            "claim_id": "A4",
            "topic": "授权测试床证据",
            "allowed_wording": (
                "授权 loopback 测试床中的运行时重放、篡改、冻结、延迟、"
                "丢包和协同偏移已完成并保存日志。"
            ),
            "forbidden_wording": (
                "不能把测试床攻击写成生产事故召回率。"
            ),
            "evidence_stages": "stage251_jisa_stable_submission_objective",
            "condition": (
                "测试床证据只作为独立外部效度证据，不替换冻结主结果。"
            ),
        },
        {
            "claim_id": "A5",
            "topic": "因果恢复安全权衡",
            "allowed_wording": (
                "安全约束恢复报告了风险权衡、现场安全边界和相对离线"
                "双向插值的因果代价。"
            ),
            "forbidden_wording": (
                "不能声称在线因果恢复等同离线双向恢复。"
            ),
            "evidence_stages": (
                "stage246_frozen_submission_package;"
                "stage250_jisa_risk_closure_audit"
            ),
            "condition": (
                "在线因果评价与离线双向评价必须分别报告。"
            ),
        },
        {
            "claim_id": "A6",
            "topic": "非 stuck 边界扩展",
            "allowed_wording": (
                "非 stuck 事件覆盖在扩展分析中保持方向性证据，但主优势"
                "仍集中在 stuck/重复值/冻结状态。"
            ),
            "forbidden_wording": (
                "不能宣称非 stuck F1 稳定显著优于基线或优势不由单一"
                "机构驱动。"
            ),
            "evidence_stages": (
                "stage281_submission_blocker_register;"
                "stage313_cross_locked_test_win_loss_matrix"
            ),
            "condition": (
                "必须保留区间跨零、移除 ECCC 后 F1 转负和 time_shift "
                "零覆盖单元。"
            ),
        },
        {
            "claim_id": "A7",
            "topic": "四机构与外部验证边界",
            "allowed_wording": (
                "报告 ECCC、KNMI、MeteoSwiss、NOAA ISD 四机构冻结主结果，"
                "并将 FMI/SMHI 作为单机构锁定外部验证。"
            ),
            "forbidden_wording": (
                "不能声称机构无关、站点无关或对所有未观测机构普适。"
            ),
            "evidence_stages": (
                "stage246_frozen_submission_package;"
                "stage263_unified_statistical_report;"
                "stage281_submission_blocker_register"
            ),
            "condition": (
                "Stage263 的 provider_count=12 只能解释为跨阶段记录数，"
                "不是 12 个独立机构。"
            ),
        },
        {
            "claim_id": "A8",
            "topic": "干净环境复现",
            "allowed_wording": (
                "冻结输出可在隔离环境中完成安装、重建和回归复现。"
            ),
            "forbidden_wording": (
                "不能把干净环境复现写成性能领先或生产普适性证明。"
            ),
            "evidence_stages": "stage281_submission_blocker_register",
            "condition": (
                "该结论只属于工程可复现性，不关闭 P0-1 至 P0-6。"
            ),
        },
    ]

    forbidden_claims = [
        {
            "claim_id": "F1",
            "forbidden_wording": "对所有基线和所有攻击类型全面领先。",
            "reason": (
                "Stage313 归一化主终点仅 32/74 通过，42/74 未通过。"
            ),
            "evidence_stages": "stage313_cross_locked_test_win_loss_matrix",
        },
        {
            "claim_id": "F2",
            "forbidden_wording": "机构无关。",
            "reason": "冻结主结果只有四个独立 provider，不能外推到未观测机构。",
            "evidence_stages": "stage281_submission_blocker_register",
        },
        {
            "claim_id": "F3",
            "forbidden_wording": "站点无关。",
            "reason": (
                "同 provider 的不同站网仍是同一机构；不相交 MET Norway "
                "站网结果并不支持站点无关。"
            ),
            "evidence_stages": "stage313_cross_locked_test_win_loss_matrix",
        },
        {
            "claim_id": "F4",
            "forbidden_wording": "真实生产事故召回率或生产环境普适领先。",
            "reason": "没有经授权、可审计的真实生产事故日志。",
            "evidence_stages": "stage281_submission_blocker_register",
        },
        {
            "claim_id": "F5",
            "forbidden_wording": "在线因果恢复等同离线双向恢复。",
            "reason": "冻结证据明确保留非零因果代价。",
            "evidence_stages": "stage250_jisa_risk_closure_audit",
        },
        {
            "claim_id": "F6",
            "forbidden_wording": "把受控注入、测试床或弱标签写成真实事故召回率。",
            "reason": "三类证据的标签层级和外部效度不同，必须分开报告。",
            "evidence_stages": "stage250_jisa_risk_closure_audit",
        },
        {
            "claim_id": "F7",
            "forbidden_wording": "把同一 provider 的不同站网包装成新的独立机构。",
            "reason": (
                "Stage313 明确 new_provider_added=false，"
                "same_provider_networks_counted_as_new_provider=false。"
            ),
            "evidence_stages": "stage313_cross_locked_test_win_loss_matrix",
        },
        {
            "claim_id": "F8",
            "forbidden_wording": "把探索性新变体写成确认性主结果。",
            "reason": "旧 test 反馈产生的新变体必须保持探索性。",
            "evidence_stages": "stage250_jisa_risk_closure_audit",
        },
        {
            "claim_id": "F9",
            "forbidden_wording": "用工程 PASS 或哈希完整证明科学领先。",
            "reason": "工程门不关闭任何 P0，也不证明性能。",
            "evidence_stages": "stage281_submission_blocker_register",
        },
        {
            "claim_id": "F10",
            "forbidden_wording": "把干净环境复现写成性能、生产或全面领先证明。",
            "reason": "复现通过只支持冻结输出的工程可重建性。",
            "evidence_stages": "stage281_submission_blocker_register",
        },
    ]

    scope_action = {
        "P0-1_external_provider_gate": (
            "STOP_PROVIDER_EXPANSION; UNIVERSAL_CLAIM_REMAINS_UNAVAILABLE"
        ),
        "P0-2_ocsvm_non_inferiority": (
            "REPORT_COMPLEMENTARITY_AND_ALL_ADVERSE_BUDGET_RESULTS"
        ),
        "P0-3_public_ics_cps_baseline": (
            "REPORT_EXISTING_MIXED_RESULTS; NO_CONFIRMATORY_PROMOTION"
        ),
        "P0-4_nonstuck_stability": (
            "LIMIT_PRIMARY_CONTRIBUTION_TO_STUCK_AND_BOUNDED_NONSTUCK"
        ),
        "P0-5_real_production_labels": "NO_PRODUCTION_CLAIM",
        "P0-6_new_confirmatory_test": (
            "NO_NEW_CONFIRMATORY_CLAIM; NEW_VARIANTS_EXPLORATORY"
        ),
    }
    p0_frame = pd.DataFrame(
        [
            {
                "blocker_id": row.get("blocker_id"),
                "category": row.get("category"),
                "status": row.get("status"),
                "evidence": row.get("evidence"),
                "claim_ceiling": row.get("claim_ceiling"),
                "required_action": row.get("required_action"),
                "current_scope_action": scope_action.get(
                    str(row.get("blocker_id")), "UNMAPPED"
                ),
            }
            for row in p0_rows
        ]
    ).sort_values("blocker_id")
    if p0_frame["current_scope_action"].eq("UNMAPPED").any():
        raise RuntimeError("unmapped P0 blocker in Stage314 scope action")

    claim_matrix = pd.DataFrame(
        [
            {
                **row,
                "claim_class": "ALLOWED",
                "status": "ALLOWED_WITH_CONDITIONS",
            }
            for row in allowed_claims
        ]
        + [
            {
                **row,
                "claim_class": "FORBIDDEN",
                "status": "FORBIDDEN",
            }
            for row in forbidden_claims
        ]
    )
    wording_map = pd.DataFrame(
        [
            {
                "topic": "主性能声称",
                "allowed_wording": allowed_claims[0]["allowed_wording"],
                "forbidden_wording": forbidden_claims[0]["forbidden_wording"],
                "condition": allowed_claims[0]["condition"],
                "evidence_stages": allowed_claims[0]["evidence_stages"],
            },
            {
                "topic": "基线比较",
                "allowed_wording": allowed_claims[1]["allowed_wording"],
                "forbidden_wording": forbidden_claims[0]["forbidden_wording"],
                "condition": allowed_claims[1]["condition"],
                "evidence_stages": allowed_claims[1]["evidence_stages"],
            },
            {
                "topic": "外部效度",
                "allowed_wording": allowed_claims[6]["allowed_wording"],
                "forbidden_wording": (
                    f"{forbidden_claims[1]['forbidden_wording']}；"
                    f"{forbidden_claims[2]['forbidden_wording']}"
                ),
                "condition": allowed_claims[6]["condition"],
                "evidence_stages": allowed_claims[6]["evidence_stages"],
            },
            {
                "topic": "非 stuck 扩展",
                "allowed_wording": allowed_claims[5]["allowed_wording"],
                "forbidden_wording": allowed_claims[5]["forbidden_wording"],
                "condition": allowed_claims[5]["condition"],
                "evidence_stages": allowed_claims[5]["evidence_stages"],
            },
            {
                "topic": "恢复与因果性",
                "allowed_wording": allowed_claims[4]["allowed_wording"],
                "forbidden_wording": forbidden_claims[4]["forbidden_wording"],
                "condition": allowed_claims[4]["condition"],
                "evidence_stages": allowed_claims[4]["evidence_stages"],
            },
            {
                "topic": "真实世界证据",
                "allowed_wording": allowed_claims[2]["allowed_wording"],
                "forbidden_wording": (
                    f"{forbidden_claims[3]['forbidden_wording']}；"
                    f"{forbidden_claims[5]['forbidden_wording']}"
                ),
                "condition": allowed_claims[2]["condition"],
                "evidence_stages": allowed_claims[2]["evidence_stages"],
            },
            {
                "topic": "可复现性",
                "allowed_wording": allowed_claims[7]["allowed_wording"],
                "forbidden_wording": forbidden_claims[9]["forbidden_wording"],
                "condition": allowed_claims[7]["condition"],
                "evidence_stages": allowed_claims[7]["evidence_stages"],
            },
            {
                "topic": "provider 范围",
                "allowed_wording": (
                    "外部验证范围固定为四个冻结主 provider；FMI/SMHI "
                    "保持单机构锁定外部验证。"
                ),
                "forbidden_wording": forbidden_claims[6]["forbidden_wording"],
                "condition": "不再新增 provider，也不把同 provider 站网计为新机构。",
                "evidence_stages": "stage313_cross_locked_test_win_loss_matrix",
            },
        ]
    )

    input_names = [
        "stage281_submission_blocker_register",
        "stage246_frozen_submission_package",
        "stage250_jisa_risk_closure_audit",
        "stage251_jisa_stable_submission_objective",
        "stage263_unified_statistical_report",
        "stage293_confirmatory_package_freeze_audit",
        "stage313_cross_locked_test_win_loss_matrix",
    ]
    input_sha256 = {
        name: sha256_file(TABLES / f"{name}.json")
        for name in input_names
    }
    payload = {
        "status": "PASS_CLAIM_CEILING_AND_SUBMISSION_SCOPE_AUDIT",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "Stage314 only reads frozen audit summaries. It does not train, "
            "tune, alter predictions, inspect test labels, or create a new "
            "provider. It freezes the author's four-provider scope decision "
            "and maps remaining P0 blockers to permitted and forbidden "
            "paper wording."
        ),
        "provider_scope": {
            "frozen_main_providers": FROZEN_MAIN_PROVIDERS,
            "frozen_main_provider_count": 4,
            "provider_expansion_stopped_by_author_decision": True,
            "new_provider_added": False,
            "same_provider_networks_counted_as_new_provider": False,
            "external_validation_boundary": (
                "FMI/SMHI remain single-provider locked external "
                "validations; MET Norway and Japan ISD remain station-"
                "network robustness evidence only."
            ),
        },
        "p0_blocker_count": int(len(p0_frame)),
        "p0_open_count": int(p0_frame["status"].eq("OPEN").sum()),
        "allowed_claim_count": len(allowed_claims),
        "forbidden_claim_count": len(forbidden_claims),
        "wording_map_count": int(len(wording_map)),
        "universal_superiority_allowed": False,
        "production_superiority_allowed": False,
        "institution_agnostic_claim_allowed": False,
        "station_agnostic_claim_allowed": False,
        "online_equals_offline_claim_allowed": False,
        "engineering_pass_is_scientific_claim": False,
        "clean_environment_reproduction_is_performance_claim": False,
        "stage313_pass_count": stage313.get(
            "normalized_primary_pass_count"
        ),
        "stage313_fail_count": stage313.get(
            "normalized_primary_fail_count"
        ),
        "stage313_nonstuck_zero_event_cell_count": stage313.get(
            "nonstuck_zero_event_cell_count"
        ),
        "stage313_nonstuck_zero_row_cell_count": stage313.get(
            "nonstuck_zero_row_cell_count"
        ),
        "claim_ceiling": (
            "bounded, conditional claims only; no universal, institution-"
            "agnostic, station-agnostic, production, or online-equals-offline "
            "claim"
        ),
        "checks": checks,
        "failed_checks": failed_checks,
        "p0_blockers": p0_frame.to_dict("records"),
        "allowed_claims": allowed_claims,
        "forbidden_claims": forbidden_claims,
        "wording_map": wording_map.to_dict("records"),
        "input_sha256": input_sha256,
        "limitations": [
            "Stage314 is a claim-boundary audit, not new performance evidence.",
            "It does not close P0-1 through P0-6.",
            "The author decision to stop adding providers does not create "
            "independent-provider evidence.",
            "Stage263 provider_count=12 is a cross-stage record count, not "
            "12 independent institutions.",
            "Engineering reproducibility, hash integrity, and PASS statuses "
            "do not establish scientific superiority.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }

    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    claim_matrix.to_csv(
        TABLES / f"{PREFIX}_claim_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    p0_frame.to_csv(
        TABLES / f"{PREFIX}_p0_blockers.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wording_map.to_csv(
        TABLES / f"{PREFIX}_wording_map.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(forbidden_claims).to_csv(
        TABLES / f"{PREFIX}_forbidden_claims.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = [
        "# Stage 314 声称上限与投稿范围固化审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 冻结主 provider：`{', '.join(FROZEN_MAIN_PROVIDERS)}`",
        "- provider 扩展：`STOPPED_BY_AUTHOR_DECISION`",
        f"- P0 阻断项：`{payload['p0_blocker_count']}`，全部 OPEN：`{payload['p0_open_count'] == 6}`",
        f"- 允许的有条件声称：`{payload['allowed_claim_count']}`",
        f"- 禁止声称：`{payload['forbidden_claim_count']}`",
        f"- Stage313 通过/未通过：`{payload['stage313_pass_count']}/{payload['stage313_fail_count']}`",
        f"- Stage313 非 stuck 零事件覆盖单元：`{payload['stage313_nonstuck_zero_event_cell_count']}`",
        f"- Stage313 非 stuck 零行召回单元：`{payload['stage313_nonstuck_zero_row_cell_count']}`",
        f"- 全面领先允许：`{payload['universal_superiority_allowed']}`",
        f"- 生产环境领先允许：`{payload['production_superiority_allowed']}`",
        "",
        "## 论文措辞映射",
        "",
        "| 主题 | 允许写法 | 禁止写法 | 条件 |",
        "|---|---|---|---|",
    ]
    for row in wording_map.to_dict("records"):
        report.append(
            f"| {row['topic']} | {row['allowed_wording']} | "
            f"{row['forbidden_wording']} | {row['condition']} |"
        )
    report += [
        "",
        "## P0 阻断项",
        "",
        "| 阻断项 | 状态 | 当前范围动作 | 声称上限 |",
        "|---|---|---|---|",
    ]
    for row in p0_frame.to_dict("records"):
        report.append(
            f"| {row['blocker_id']} | {row['status']} | "
            f"{row['current_scope_action']} | {row['claim_ceiling']} |"
        )
    report += [
        "",
        "## 证据边界",
        "",
        "- Stage314 只固化声称边界，不训练、不调参、不新增 provider，也不产生新的性能结果。",
        "- 四个主 provider 固定为 ECCC、KNMI、MeteoSwiss、NOAA ISD；不再新增 provider。",
        "- FMI/SMHI 保持单机构锁定外部验证；MET Norway 和日本 ISD 只作同 provider 站网稳健性证据。",
        "- Stage263 的 `provider_count=12` 是跨阶段记录数，不是 12 个独立机构。",
        "- 工程 PASS、哈希完整和干净环境复现均不能升级为性能、生产或全面领先声称。",
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": payload["status"],
                "provider_count": 4,
                "p0_open": payload["p0_open_count"],
                "allowed_claims": payload["allowed_claim_count"],
                "forbidden_claims": payload["forbidden_claim_count"],
                "stage313_pass": payload["stage313_pass_count"],
                "stage313_fail": payload["stage313_fail_count"],
                "universal_superiority_allowed": payload[
                    "universal_superiority_allowed"
                ],
                "production_superiority_allowed": payload[
                    "production_superiority_allowed"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
