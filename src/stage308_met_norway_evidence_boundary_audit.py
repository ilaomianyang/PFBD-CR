from __future__ import annotations

"""Stage 308: audit the MET Norway transfer evidence boundary.

This stage is deliberately read-only.  It inspects the frozen Stage304
protocol, Stage305 contract, Stage306 download audit, the current Stage307
transfer result, and the preserved first-pass Stage307 result.  It does not
download observations, train models, calibrate thresholds, inject attacks,
or compute new detection metrics.

The first Stage307 statistical implementation did not filter the paired
table by candidate and baseline before station-cluster bootstrap.  That
error produced an invalid 26/26 result.  Stage307 was corrected after the
first test result was visible.  The corrected run therefore remains
exploratory and cannot be promoted as a pre-registered confirmatory test.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage308_met_norway_evidence_boundary_audit"
DATA_PROTOCOL_PREFIX = "stage304_met_norway_protocol_preregistration"
CONTRACT_PREFIX = "stage305_met_norway_locked_evaluation_contract"
DOWNLOAD_PREFIX = "stage306_met_norway_official_download"
TRANSFER_PREFIX = "stage307_met_norway_locked_transfer"
INVALID_TRANSFER = (
    TABLES / "stage307_met_norway_locked_transfer_invalid_v1.json"
)

EXPECTED_TRANSFER_JSON_SHA256 = (
    "9baf538f439ca7bdfb2a83008e28ff4e88bd1f2727a062ed9209fdde205d97c9"
)
EXPECTED_INVALID_JSON_SHA256 = (
    "9a20aa1793c0a441806fe56e955d4586a9fcf03921f4e247218d9c2a4598b652"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bool_value(value: object) -> bool:
    return bool(value)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    protocol_path = TABLES / f"{DATA_PROTOCOL_PREFIX}.json"
    contract_path = TABLES / f"{CONTRACT_PREFIX}.json"
    download_path = TABLES / f"{DOWNLOAD_PREFIX}.json"
    transfer_path = TABLES / f"{TRANSFER_PREFIX}.json"
    for path in (
        protocol_path,
        contract_path,
        download_path,
        transfer_path,
        INVALID_TRANSFER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    protocol = read_json(protocol_path)
    contract = read_json(contract_path)
    download = read_json(download_path)
    transfer = read_json(transfer_path)
    invalid = read_json(INVALID_TRANSFER)

    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD":
        raise RuntimeError("MET Norway protocol is not frozen")
    if contract["status"] != (
        "EVALUATION_CONTRACT_FROZEN_BEFORE_FULL_DOWNLOAD"
    ):
        raise RuntimeError("MET Norway contract is not frozen")
    if download["status"] != (
        "PASS_MET_NORWAY_DOWNLOAD_AND_COVERAGE_AUDIT"
    ):
        raise RuntimeError("MET Norway download audit did not pass")
    if transfer["status"] != "COMPLETED_MET_NORWAY_LOCKED_TRANSFER":
        raise RuntimeError("MET Norway transfer status changed")
    if invalid["status"] != "COMPLETED_MET_NORWAY_LOCKED_TRANSFER":
        raise RuntimeError("preserved invalid transfer status changed")

    invalid_pass = int(invalid["primary_pass_count"])
    invalid_total = int(invalid["primary_comparison_count"])
    corrected_pass = int(transfer["primary_pass_count"])
    corrected_total = int(transfer["primary_comparison_count"])
    if not (
        invalid_pass == invalid_total
        and bool(invalid["all_baselines_supported"])
    ):
        raise RuntimeError(
            "preserved invalid transfer is not the known 26/26 result"
        )
    if corrected_pass >= corrected_total:
        raise RuntimeError(
            "corrected MET Norway transfer no longer contains adverse results"
        )
    if bool(transfer["all_baselines_supported"]):
        raise RuntimeError("corrected MET Norway transfer unexpectedly passed")

    comparison_columns = [
        "candidate",
        "baseline",
        "f1_difference",
        "f1_ci95_low",
        "f1_ci95_high",
        "event_coverage_difference",
        "event_ci95_low",
        "event_ci95_high",
        "f1_non_inferior",
        "event_coverage_non_inferior",
        "event_coverage_primary_supported",
        "f1_primary_supported",
    ]
    comparisons = pd.read_csv(
        TABLES / f"{TRANSFER_PREFIX}_comparisons.csv"
    )
    missing_columns = [
        column for column in comparison_columns if column not in comparisons
    ]
    if missing_columns:
        raise RuntimeError(
            f"missing comparison columns: {missing_columns}"
        )
    if len(comparisons) != corrected_total:
        raise RuntimeError("comparison row count changed")

    adverse = comparisons.loc[
        ~(
            comparisons["event_coverage_primary_supported"]
            | comparisons["event_coverage_non_inferior"]
        )
    ].copy()
    adverse = adverse.sort_values(
        ["event_coverage_difference", "candidate", "baseline"],
        kind="stable",
    )
    adverse.to_csv(
        TABLES / f"{PREFIX}_adverse_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    provider_equal = pd.read_csv(
        TABLES / f"{TRANSFER_PREFIX}_provider_equal.csv"
    )
    trust_valid = provider_equal.loc[
        provider_equal["method"].eq("trust_valid_f1")
    ].iloc[0]
    trust_nonstuck = provider_equal.loc[
        provider_equal["method"].eq("trust_nonstuck_event_macro")
    ].iloc[0]
    best_f1 = provider_equal.sort_values(
        ["f1_mean", "method"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    best_event = provider_equal.sort_values(
        ["event_coverage_mean", "method"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]

    invalid_hash = sha256_file(INVALID_TRANSFER)
    transfer_hash = sha256_file(transfer_path)
    protocol_hash = sha256_file(protocol_path)
    contract_hash = sha256_file(contract_path)
    download_hash = sha256_file(download_path)
    comparisons_hash = sha256_file(
        TABLES / f"{TRANSFER_PREFIX}_comparisons.csv"
    )
    provider_equal_hash = sha256_file(
        TABLES / f"{TRANSFER_PREFIX}_provider_equal.csv"
    )
    code_hash = sha256_file(Path(__file__).resolve())

    checks = [
        {
            "check": "protocol_frozen_before_download",
            "passed": bool_value(
                protocol["status"]
                == "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD"
                and protocol.get(
                    "strict_no_observation_access_before_freeze"
                )
                is True
                and protocol.get("download_status") == "NOT_DOWNLOADED"
                and protocol.get("performance_status") == "NOT_EVALUATED"
            ),
        },
        {
            "check": "contract_frozen_before_download",
            "passed": bool_value(
                contract["status"]
                == "EVALUATION_CONTRACT_FROZEN_BEFORE_FULL_DOWNLOAD"
                and contract.get("download_status") == "NOT_DOWNLOADED"
                and contract.get("performance_status") == "NOT_EVALUATED"
            ),
        },
        {
            "check": "official_download_audit_complete",
            "passed": bool_value(
                download["status"]
                == "PASS_MET_NORWAY_DOWNLOAD_AND_COVERAGE_AUDIT"
                and int(download["failed_count"]) == 0
                and int(download["retained_station_count"]) == 12
            ),
        },
        {
            "check": "first_pass_false_positive_preserved",
            "passed": bool_value(
                invalid_pass == invalid_total
                and bool(invalid["all_baselines_supported"])
                and invalid_hash == EXPECTED_INVALID_JSON_SHA256
            ),
        },
        {
            "check": "corrected_transfer_retains_adverse_results",
            "passed": bool_value(
                corrected_pass < corrected_total
                and not bool(transfer["all_baselines_supported"])
                and len(adverse) > 0
            ),
        },
        {
            "check": "corrected_transfer_hash_is_frozen",
            "passed": bool_value(
                transfer_hash == EXPECTED_TRANSFER_JSON_SHA256
            ),
        },
        {
            "check": "corrected_transfer_is_not_confirmatory",
            "passed": bool_value(
                transfer["transfer_gate_status"] == "NOT_PASSED"
                and len(adverse) > 0
            ),
        },
        {
            "check": "independent_provider_boundary_retained",
            "passed": bool_value(
                int(transfer["provider_count"]) == 1
                and int(transfer["station_count"]) == 12
            ),
        },
    ]
    passed_checks = int(sum(row["passed"] for row in checks))

    payload = {
        "status": "PASS_READ_ONLY_MET_NORWAY_EVIDENCE_BOUNDARY_AUDIT",
        "protocol_id": protocol["protocol_id"],
        "contract_id": contract["contract_id"],
        "provider": protocol["provider"],
        "network_id": protocol["network_id"],
        "station_count": int(transfer["station_count"]),
        "seed_count": int(transfer["seed_count"]),
        "method_count": int(transfer["method_count"]),
        "primary_far": float(transfer["primary_far"]),
        "primary_endpoint": transfer["primary_endpoint"],
        "first_pass": {
            "artifact": str(INVALID_TRANSFER.relative_to(ROOT)),
            "sha256": invalid_hash,
            "primary_pass_count": invalid_pass,
            "primary_comparison_count": invalid_total,
            "all_baselines_supported": bool(
                invalid["all_baselines_supported"]
            ),
            "interpretation": (
                "INVALID_DUE_TO_UNFILTERED_PAIRED_BOOTSTRAP; "
                "must never be promoted"
            ),
        },
        "corrected_pass": {
            "artifact": str(transfer_path.relative_to(ROOT)),
            "sha256": transfer_hash,
            "primary_pass_count": corrected_pass,
            "primary_comparison_count": corrected_total,
            "all_baselines_supported": bool(
                transfer["all_baselines_supported"]
            ),
            "transfer_gate_status": transfer["transfer_gate_status"],
            "adverse_comparison_count": int(len(adverse)),
            "interpretation": (
                "EXPLORATORY_AND_ADVERSE; code was corrected after the "
                "first test result became visible, so it cannot be used "
                "as a frozen confirmatory result"
            ),
        },
        "trust_valid_f1": {
            "f1_mean": float(trust_valid["f1_mean"]),
            "event_coverage_mean": float(
                trust_valid["event_coverage_mean"]
            ),
            "false_alarm_rate_mean": float(
                trust_valid["false_alarm_rate_mean"]
            ),
        },
        "trust_nonstuck_event_macro": {
            "f1_mean": float(trust_nonstuck["f1_mean"]),
            "event_coverage_mean": float(
                trust_nonstuck["event_coverage_mean"]
            ),
            "false_alarm_rate_mean": float(
                trust_nonstuck["false_alarm_rate_mean"]
            ),
        },
        "best_f1_method": {
            "method": str(best_f1["method"]),
            "f1_mean": float(best_f1["f1_mean"]),
            "event_coverage_mean": float(
                best_f1["event_coverage_mean"]
            ),
        },
        "best_event_coverage_method": {
            "method": str(best_event["method"]),
            "f1_mean": float(best_event["f1_mean"]),
            "event_coverage_mean": float(
                best_event["event_coverage_mean"]
            ),
        },
        "checks": checks,
        "passed_check_count": passed_checks,
        "check_count": int(len(checks)),
        "evidence_boundary": {
            "independent_provider": True,
            "independent_station_network": False,
            "confirmatory_result_allowed": False,
            "real_production_attack_evidence": False,
            "may_close_p0_1": False,
            "may_close_p0_2": False,
            "may_close_p0_3": False,
            "may_close_p0_4": False,
            "may_close_p0_5": False,
            "may_close_p0_6": False,
            "allowed_claim": (
                "bounded exploratory independent-provider transfer evidence; "
                "strong baselines win on several primary comparisons and "
                "the corrected analysis is not a frozen confirmatory test"
            ),
            "forbidden_claims": [
                "MET Norway controlled injections are real production attacks.",
                "One independent provider proves station independence.",
                "The corrected post-result analysis is a confirmatory test.",
                "The invalid 26/26 result may be reported as a finding.",
                "MET Norway results may overwrite Stage246, FMI, SMHI, or Japan.",
            ],
        },
        "hashes": {
            "protocol_sha256": protocol_hash,
            "contract_sha256": contract_hash,
            "download_sha256": download_hash,
            "transfer_sha256": transfer_hash,
            "invalid_transfer_sha256": invalid_hash,
            "comparisons_sha256": comparisons_hash,
            "provider_equal_sha256": provider_equal_hash,
            "code_sha256": code_hash,
        },
        "limitations": [
            "只读审计，不重新下载、不训练、不校准阈值、不注入攻击、不计算新指标。",
            "第一次26/26结果因配对bootstrap未按candidate/baseline过滤而无效，必须永久保留。",
            "修正后的8/26结果是在首次接触test后修正代码得到的，只能标为探索性。",
            "MET Norway是独立机构，但仍只有一个provider网络，不能证明站点无关。",
            "所有攻击均为透明受控注入，不是真实生产事故或攻击日志。",
            "本阶段不关闭任何P0，也不支持全面领先、真实攻击召回率或生产普适性声称。",
        ],
    }
    output_path = TABLES / f"{PREFIX}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(checks).to_csv(
        TABLES / f"{PREFIX}_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report_lines = [
        "# 阶段308 MET Norway证据边界审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 独立机构：`{payload['provider']}`",
        f"- 网络：`{payload['network_id']}`",
        f"- 站点数：`{payload['station_count']}`",
        f"- 种子数：`{payload['seed_count']}`",
        f"- 方法数：`{payload['method_count']}`",
        f"- 检查通过：`{passed_checks}/{len(checks)}`",
        "",
        "## 关键事实",
        "",
        f"- 第一次Stage307无效结果：`{invalid_pass}/{invalid_total}`，"
        "原因是配对bootstrap未过滤candidate/baseline。",
        f"- 修正后结果：`{corrected_pass}/{corrected_total}`，"
        f"`all_baselines_supported={payload['corrected_pass']['all_baselines_supported']}`。",
        f"- 修正后不利比较数：`{len(adverse)}`。",
        "- 修正发生在首次接触test之后，因此修正版只能作为探索性和不利结果证据。",
        "",
        "## 声称边界",
        "",
        "- 本阶段不允许关闭P0-1、P0-2、P0-3、P0-4、P0-5或P0-6。",
        "- 不允许把MET Norway受控注入写成真实攻击召回率。",
        "- 不允许把无效26/26结果当作发现或论文主结果。",
        "- 不允许把一个独立provider写成机构无关或站点无关。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "passed_checks": passed_checks,
                "check_count": len(checks),
                "invalid_pass_count": invalid_pass,
                "corrected_pass_count": corrected_pass,
                "adverse_comparison_count": int(len(adverse)),
                "json_sha256": sha256_file(output_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
