from __future__ import annotations

"""Stage 213: close the DWD external-evidence audit against Stage 203.

Stage 203 recorded five minimum requirements for a genuinely independent
external validation.  Stage 204--212 later froze a DWD protocol, downloaded
official DWD archives, normalized them, evaluated transparent controlled
attacks, replicated with new attack seeds, and quantified station-clustered
uncertainty.

This stage does not run a new performance experiment.  It reads the frozen
artifacts and audits whether each Stage-203 requirement is now supported,
partially supported, or still missing.  The result must not be read as a
multi-provider generalization claim: only one cross-provider source (DWD)
has been evaluated.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage213_dwd_external_evidence_closure"

STAGE203 = TABLES / "stage203_independent_external_evidence_feasibility.json"
STAGE204 = TABLES / "stage204_dwd_protocol_preregistration.json"
STAGE204A = TABLES / "stage204a_dwd_protocol_schema_correction.json"
STAGE205 = TABLES / "stage205_dwd_official_download.json"
STAGE206 = TABLES / "stage206_dwd_normalized.json"
STAGE207 = TABLES / "stage207_dwd_controlled_attack.json"
STAGE208 = TABLES / "stage208_dwd_persistent_repeat_protection.json"
STAGE209 = TABLES / "stage209_dwd_station_cluster_uncertainty.json"
STAGE210 = TABLES / "stage210_dwd_seed_randomized_replication.json"
STAGE211 = TABLES / "stage211_dwd_replication_station_uncertainty.json"
STAGE212 = TABLES / "stage212_dwd_operating_curve.json"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def has_text(payload: dict[str, object], *needles: str) -> bool:
    text = json.dumps(payload, ensure_ascii=False)
    return all(needle in text for needle in needles)


def main() -> int:
    required_paths = [
        STAGE203,
        STAGE204,
        STAGE204A,
        STAGE205,
        STAGE206,
        STAGE207,
        STAGE208,
        STAGE209,
        STAGE210,
        STAGE211,
        STAGE212,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing frozen artifacts for Stage213: " + ", ".join(missing)
        )

    stage203 = load_json(STAGE203)
    stage204 = load_json(STAGE204)
    stage204a = load_json(STAGE204A)
    stage205 = load_json(STAGE205)
    stage206 = load_json(STAGE206)
    stage207 = load_json(STAGE207)
    stage208 = load_json(STAGE208)
    stage209 = load_json(STAGE209)
    stage210 = load_json(STAGE210)
    stage211 = load_json(STAGE211)
    stage212 = load_json(STAGE212)

    station_ids = [str(value) for value in stage204["station_ids"]]
    primary_far = float(stage204["primary_far"])
    stage208_primary = [
        row
        for row in stage208["primary_paired"]
        if row["selection_rule"] == "valid_f1"
        and row["metric"] == "test_f1"
    ]
    stage210_f1 = next(
        row for row in stage210["paired"] if row["metric"] == "test_f1"
    )
    stage209_f1 = next(
        row for row in stage209["bootstrap"] if row["metric"] == "f1"
    )
    stage211_f1 = next(
        row for row in stage211["bootstrap"] if row["metric"] == "f1"
    )
    stage212_primary = [
        row
        for row in stage212["curve"]
        if row["metric"] == "f1" and row["is_primary"]
    ]

    checks: list[dict[str, object]] = []

    source_independence = (
        bool(stage204.get("independent_source"))
        and str(stage204.get("provider", "")).startswith("Deutscher Wetterdienst")
        and "different provider and sensor network" in str(stage204.get("source_independence", ""))
        and len(station_ids) == 12
        and int(stage206.get("station_count", 0)) == 12
        and int(stage205.get("failed_count", -1)) == 0
    )
    checks.append(
        {
            "requirement": "source_independence",
            "status": "SUPPORTED_SINGLE_PROVIDER" if source_independence else "MISSING",
            "evidence": [
                "stage204: DWD is a different provider and sensor network",
                "stage205: official DWD OpenData archives, 0 failed downloads",
                "stage206: 12 frozen DWD stations normalized",
            ],
            "scope_limit": (
                "Only one cross-provider source (DWD) is evaluated; this "
                "supports cross-provider evidence, not multi-provider generalization."
            ),
        }
    )

    attack_protocol = (
        has_text(stage204, "透明受控攻击", "不是真实攻击日志")
        and has_text(stage207, "不根据test", "不是真实攻击日志")
        and int(stage207.get("evaluation_rows", 0)) == 20
        and int(stage207.get("event_rows", 0)) == 20
        and int(stage208.get("candidate_count", 0)) == 42
        and int(stage208.get("evaluation_rows", 0)) == 80
        and int(stage210.get("evaluation_rows", 0)) == 80
    )
    checks.append(
        {
            "requirement": "transparent_attack_protocol",
            "status": "SUPPORTED" if attack_protocol else "MISSING",
            "evidence": [
                "stage204: attack types, lengths, blocks, seeds, and FAR frozen before download",
                "stage207: 20 evaluation rows and 20 event rows under the frozen baseline",
                "stage208: 42 pre-declared protection candidates",
                "stage210: independent attack-block seed replication",
            ],
            "scope_limit": (
                "All attacks are transparent controlled injections, not real "
                "attack logs. The result does not estimate a real intrusion rate."
            ),
        }
    )

    no_target_tuning = (
        has_text(stage204, "test只用于后续评价", "不根据test选择任何组件")
        and has_text(stage207, "valid正常段", "test只评价", "不根据test")
        and has_text(stage208, "valid_f1", "test只评价", "不根据test")
        and has_text(stage210, "valid_f1", "test只评价", "不根据test")
        and has_text(stage212, "不重新选择", "不重新校准", "不根据test选择FAR")
    )
    checks.append(
        {
            "requirement": "no_target_domain_tuning",
            "status": "SUPPORTED" if no_target_tuning else "MISSING",
            "evidence": [
                "stage204: test is evaluation-only and no target-domain test tuning is permitted",
                "stage207: train-normal fitting, valid-normal calibration, test-only evaluation",
                "stage208/210: valid_f1 selection, with test evaluation only",
                "stage212: frozen FAR grid, no reselection or recalibration",
            ],
            "scope_limit": (
                "The audit checks the frozen protocol and artifact text; it "
                "does not replace line-by-line review of every code path."
            ),
        }
    )

    operating_point = (
        abs(primary_far - 0.01) < 1e-12
        and str(stage204.get("primary_endpoint", "")) == "event_coverage"
        and str(stage204.get("cluster_unit", "")) == "station"
        and "Benjamini-Hochberg" in str(stage204.get("multiplicity", ""))
        and int(stage209.get("iterations", 0)) == 5000
        and int(stage211.get("iterations", 0)) == 5000
        and int(stage212.get("iterations", 0)) == 5000
        and len(stage212_primary) == 2
        and all(float(row["difference_ci95"][0]) > 0 for row in stage212_primary)
    )
    checks.append(
        {
            "requirement": "operating_point_and_uncertainty",
            "status": "SUPPORTED" if operating_point else "MISSING",
            "evidence": [
                "stage204: primary FAR=0.01, primary endpoint=event_coverage, cluster unit=station",
                "stage209/211: 5000 station-clustered paired bootstrap iterations",
                "stage212: frozen FAR curve, both primary F1 intervals above zero",
            ],
            "scope_limit": (
                "Event coverage is not uniformly positive at every sensitivity "
                "point; the primary point remains FAR=0.01."
            ),
        }
    )

    reproducible_access = (
        str(stage204.get("official_root", "")).startswith("https://opendata.dwd.de/")
        and str(stage205.get("protocol_status", "")) == "PROTOCOL_FROZEN_BEFORE_DOWNLOAD"
        and int(stage205.get("expected_archives", 0)) == 24
        and int(stage205.get("required_columns_ok_count", 0)) == 24
        and int(stage205.get("archive_audit_failure_count", -1)) == 0
        and int(stage206.get("duplicate_station_timestamp_rows", -1)) == 0
        and int(stage206.get("rows", 0)) == 525888
        and has_text(stage206, "QN_9", "2021-2023", "2024", "2025")
    )
    checks.append(
        {
            "requirement": "reproducible_access",
            "status": "SUPPORTED" if reproducible_access else "MISSING",
            "evidence": [
                "stage204: official DWD URLs, dataset ID, version, license note, and station rule frozen",
                "stage205: 24/24 archives, field audit and archive audit recorded",
                "stage206: 525888 normalized rows with explicit missing and QN_9 quality fields",
            ],
            "scope_limit": (
                "The audit records access and schema reproducibility; it does "
                "not assert that the upstream DWD files can never change."
            ),
        }
    )

    supported = [
        item for item in checks if str(item["status"]).startswith("SUPPORTED")
    ]
    unsupported = [
        item for item in checks if not str(item["status"]).startswith("SUPPORTED")
    ]
    if unsupported:
        status = "EXTERNAL_EVIDENCE_CLOSURE_INCOMPLETE"
    else:
        status = "EXTERNAL_EVIDENCE_CLOSURE_SINGLE_PROVIDER"

    payload = {
        "status": status,
        "protocol": (
            "阶段213只读取阶段203的最低要求和阶段204-212已冻结的DWD协议、"
            "下载、标准化、评价、复现和不确定性产物；逐项核验来源独立性、"
            "透明受控攻击、无目标域调参、主运行点与不确定性、可复现访问。"
            "本阶段不训练、不重新选择、不重新校准阈值，不下载新数据，"
            "不计算新的性能结果；所有攻击都是透明受控注入，不是真实攻击日志。"
            "当前只完成一个跨机构来源DWD，不能扩展为多机构泛化声明。"
        ),
        "source_provider": "Deutscher Wetterdienst (DWD)",
        "single_provider_only": True,
        "requirements_total": len(checks),
        "requirements_supported": len(supported),
        "requirements_unsupported": len(unsupported),
        "checks": checks,
        "primary_far": primary_far,
        "primary_endpoint": stage204["primary_endpoint"],
        "cluster_unit": stage204["cluster_unit"],
        "station_count": len(station_ids),
        "stage208_f1_pair": stage208_primary,
        "stage209_f1_bootstrap": stage209_f1,
        "stage210_f1_pair": stage210_f1,
        "stage211_f1_bootstrap": stage211_f1,
        "stage212_primary_f1": stage212_primary,
        "limitations": [
            "只有一个跨机构来源DWD；该闭环支持跨来源证据，不支持多机构泛化。",
            "所有攻击都是透明受控注入，不是真实攻击日志，不能解释为真实入侵率。",
            "DWD质量码QN_9不是攻击标签，低质量行被排除而不是当作正事件。",
            "事件覆盖在FAR=0.005等敏感性点不占优，主运行点必须保持FAR=0.01。",
            "本审计核验冻结协议和产物一致性，不替代人工逐行代码复核。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(checks).to_csv(
        TABLES / f"{PREFIX}_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段213 DWD跨机构外部证据闭环审计",
        "",
        f"- 状态：**{status}**",
        f"- 跨机构来源：DWD",
        f"- 最低要求支持：`{len(supported)}/{len(checks)}`",
        f"- 主运行点：`FAR={primary_far}`",
        f"- 主终点：`{stage204['primary_endpoint']}`；聚类单位：`{stage204['cluster_unit']}`",
        "",
        "## 逐项审计",
        "",
        "| 要求 | 状态 | 证据 | 边界 |",
        "|---|---|---|---|",
    ]
    for item in checks:
        lines.append(
            f"| {item['requirement']} | {item['status']} | "
            f"{'; '.join(item['evidence'])} | {item['scope_limit']} |"
        )
    lines += [
        "",
        "## 关键结果",
        "",
        f"- 阶段208五种子F1配对结果行数：`{len(stage208_primary)}`",
        f"- 阶段209 F1差值：`{float(stage209_f1['difference_mean']):+.6f}`，95%区间 `[{float(stage209_f1['difference_ci95'][0]):+.6f}, {float(stage209_f1['difference_ci95'][1]):+.6f}]`",
        f"- 阶段210 F1差值：`{float(stage210_f1['mean_diff']):+.6f}`，正向 `{int(stage210_f1['positive_pairs'])}/{int(stage210_f1['pairs'])}`",
        f"- 阶段211 F1差值：`{float(stage211_f1['difference_mean']):+.6f}`，95%区间 `[{float(stage211_f1['difference_ci95'][0]):+.6f}, {float(stage211_f1['difference_ci95'][1]):+.6f}]`",
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in payload["limitations"]],
        "",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "requirements_total": len(checks),
                "requirements_supported": len(supported),
                "requirements_unsupported": len(unsupported),
                "single_provider_only": True,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if not unsupported else 1


if __name__ == "__main__":
    raise SystemExit(main())
