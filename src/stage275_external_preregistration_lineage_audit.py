from __future__ import annotations

"""Stage 275: audit the FMI/SMHI external preregistration lineage.

This stage is strictly read-only.  It does not download data, train a model,
recalibrate a threshold, or inspect any new test observation.  Its purpose is
to make the externally auditable parts of the preregistration record explicit
and to prevent a partial time chain from being described as a strict
zero-observation preregistration.
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage275_external_preregistration_lineage_audit"


INPUT_FILES = [
    "stage257_fmi_protocol_preregistration.json",
    "stage258_fmi_official_download.json",
    "stage258_fmi_official_download_detail.csv",
    "stage259_fmi_locked_transfer.json",
    "stage264_smhi_protocol_preregistration.json",
    "stage265_smhi_locked_evaluation_contract.json",
    "stage266_smhi_official_download.json",
    "stage266_smhi_official_download_detail.csv",
    "stage267_smhi_locked_transfer.json",
]


def _load_json(name: str) -> dict:
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_utc(value: object) -> datetime:
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp is not timezone-aware: {value}")
    return parsed


def _detail_timestamp(
    frame: pd.DataFrame,
    column: str,
) -> str | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna().astype(str)
    if values.empty:
        return None
    parsed = sorted(_parse_utc(value) for value in values.unique())
    if len(parsed) != 1:
        raise RuntimeError(
            f"expected one unique {column} value, found {len(parsed)}"
        )
    return parsed[0].isoformat()


def _provider_rows() -> list[dict[str, object]]:
    fmi_protocol = _load_json("stage257_fmi_protocol_preregistration.json")
    fmi_download = _load_json("stage258_fmi_official_download.json")
    fmi_transfer = _load_json("stage259_fmi_locked_transfer.json")
    smhi_protocol = _load_json(
        "stage264_smhi_protocol_preregistration.json"
    )
    smhi_contract = _load_json(
        "stage265_smhi_locked_evaluation_contract.json"
    )
    smhi_download = _load_json("stage266_smhi_official_download.json")
    smhi_transfer = _load_json("stage267_smhi_locked_transfer.json")
    fmi_detail = pd.read_csv(
        TABLES / "stage258_fmi_official_download_detail.csv"
    )
    smhi_detail = pd.read_csv(
        TABLES / "stage266_smhi_official_download_detail.csv"
    )

    fmi_freeze = _parse_utc(fmi_protocol["frozen_at_utc"])
    fmi_download_time = _detail_timestamp(fmi_detail, "downloaded_at_utc")
    smhi_freeze = _parse_utc(smhi_protocol["frozen_at_utc"])
    smhi_contract_time = _parse_utc(smhi_contract["frozen_at_utc"])
    smhi_download_time = _detail_timestamp(
        smhi_detail,
        "downloaded_at_utc",
    )
    if smhi_download_time is None:
        raise RuntimeError("SMHI download detail lacks downloaded_at_utc")
    smhi_download_at = _parse_utc(smhi_download_time)
    if not (smhi_freeze < smhi_contract_time < smhi_download_at):
        raise RuntimeError("SMHI freeze/contract/download chronology failed")
    if fmi_download_time is not None:
        if not (fmi_freeze < _parse_utc(fmi_download_time)):
            raise RuntimeError("FMI freeze/download chronology failed")

    fmi_comparisons = pd.DataFrame(fmi_transfer["comparisons"])
    smhi_comparisons = pd.DataFrame(smhi_transfer["comparisons"])
    fmi_gate_passed = bool(
        fmi_comparisons["event_coverage_primary_supported"].all()
    )
    smhi_gate_passed = bool(
        smhi_transfer["all_baselines_supported"]
    )
    if smhi_gate_passed:
        raise RuntimeError(
            "Stage275 expected the disclosed adverse SMHI transfer result"
        )

    common = {
        "attack_protocol": "transparent_controlled_injection",
        "real_production_incident_labels": False,
        "promotion_status": "BOUNDED_OR_PARTIAL_ONLY",
        "test_used_for_tuning": False,
    }
    rows = [
        {
            "provider": "FMI",
            "country": "Finland",
            "independent_source": bool(fmi_protocol["independent_source"]),
            "protocol_id": str(fmi_protocol["protocol_id"]),
            "freeze_status": str(fmi_protocol["status"]),
            "data_status": str(fmi_download["status"]),
            "evaluation_status": str(fmi_transfer["status"]),
            "freeze_time_utc": fmi_freeze.isoformat(),
            "download_time_utc": fmi_download_time,
            "evaluation_time_utc": None,
            "freeze_before_full_download": True,
            "strict_zero_observation_access_before_freeze": True,
            "time_chain_status": (
                "PARTIAL_INTERNAL_DOWNLOAD_TIMESTAMP_MISSING"
                if fmi_download_time is None
                else "COMPLETE_INTERNAL_TIMESTAMPS"
            ),
            "station_count": int(fmi_download["retained_station_count"]),
            "method_count": int(fmi_transfer["method_count"]),
            "primary_far": float(fmi_transfer["primary_far"]),
            "primary_endpoint": str(fmi_transfer["primary_endpoint"]),
            "target_test_tuning": False,
            "external_locked_test_only": True,
            "performance_gate_passed": fmi_gate_passed,
            **common,
        },
        {
            "provider": "SMHI",
            "country": "Sweden",
            "independent_source": bool(
                smhi_protocol["independent_source"]
            ),
            "protocol_id": str(smhi_protocol["protocol_id"]),
            "freeze_status": str(smhi_protocol["status"]),
            "data_status": str(smhi_download["status"]),
            "evaluation_status": str(smhi_transfer["status"]),
            "freeze_time_utc": smhi_freeze.isoformat(),
            "download_time_utc": smhi_download_time,
            "evaluation_time_utc": None,
            "freeze_before_full_download": True,
            "strict_zero_observation_access_before_freeze": bool(
                smhi_protocol["strict_no_observation_access_before_freeze"]
            ),
            "time_chain_status": (
                "COMPLETE_INTERNAL_TIMESTAMPS_WITH_DISCLOSED_"
                "PRE_FREEZE_HEADER_PROBE"
            ),
            "station_count": int(smhi_download["retained_station_count"]),
            "method_count": int(smhi_transfer["method_count"]),
            "primary_far": float(smhi_transfer["primary_far"]),
            "primary_endpoint": str(smhi_transfer["primary_endpoint"]),
            "target_test_tuning": False,
            "external_locked_test_only": True,
            "performance_gate_passed": smhi_gate_passed,
            **common,
        },
    ]
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    missing = [name for name in INPUT_FILES if not (TABLES / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage275 inputs: {missing}")

    rows = _provider_rows()
    detail = pd.DataFrame(rows).sort_values("provider").reset_index(drop=True)
    fmi = detail.loc[detail["provider"].eq("FMI")].iloc[0]
    smhi = detail.loc[detail["provider"].eq("SMHI")].iloc[0]
    strict_zero = bool(
        detail["strict_zero_observation_access_before_freeze"].all()
    )
    time_chain_complete = bool(
        detail["time_chain_status"].eq(
            "COMPLETE_INTERNAL_TIMESTAMPS"
        ).all()
    )
    payload = {
        "status": "PASS_EXTERNAL_PREREGISTRATION_LINEAGE_AUDIT",
        "protocol": (
            "Stage275 reads only the frozen FMI and SMHI protocol, "
            "download, contract, and locked-transfer artifacts. It does "
            "not download data, train, recalibrate, inspect new test "
            "observations, or change any prediction."
        ),
        "provider_count": int(detail["provider"].nunique()),
        "independent_provider_count": int(detail["independent_source"].sum()),
        "frozen_before_full_download_count": int(
            detail["freeze_before_full_download"].sum()
        ),
        "strict_zero_observation_access_before_freeze": strict_zero,
        "complete_internal_time_chain_count": int(
            detail["time_chain_status"].eq(
                "COMPLETE_INTERNAL_TIMESTAMPS"
            ).sum()
        ),
        "time_chain_complete_for_both_providers": time_chain_complete,
        "is_a_new_confirmatory_test_package": False,
        "can_promote_provider_independence": False,
        "can_promote_real_production_claim": False,
        "fmi": {
            "freeze_before_full_download": bool(
                fmi["freeze_before_full_download"]
            ),
            "strict_zero_observation_access_before_freeze": bool(
                fmi["strict_zero_observation_access_before_freeze"]
            ),
            "internal_download_timestamp_present": False,
            "time_chain_status": str(fmi["time_chain_status"]),
        },
        "smhi": {
            "freeze_before_full_download": bool(
                smhi["freeze_before_full_download"]
            ),
            "strict_zero_observation_access_before_freeze": bool(
                smhi["strict_zero_observation_access_before_freeze"]
            ),
            "internal_download_timestamp_present": True,
            "time_chain_status": str(smhi["time_chain_status"]),
            "pre_freeze_reconnaissance": "DISCLOSED_20KB_HEADER_ONLY",
        },
        "claims": {
            "protocol_frozen_before_full_download": True,
            "strict_zero_observation_preregistration_for_both": False,
            "target_test_used_only_for_evaluation": bool(
                detail["external_locked_test_only"].all()
                and detail["target_test_tuning"].eq(False).all()
            ),
            "provider_independence_proven": False,
            "全面领先统计门槛已通过": False,
            "real_production_attack_recall_proven": False,
        },
        "provider_lineage": [
            {
                key: (None if pd.isna(value) else value)
                for key, value in row.items()
            }
            for row in detail.to_dict("records")
        ],
        "input_sha256": {
            name: _sha256(TABLES / name) for name in INPUT_FILES
        },
        "code_sha256": _sha256(Path(__file__).resolve()),
        "limitations": [
            "FMI has a pre-download freeze timestamp, but the download "
            "detail table has no internal UTC download-time column.",
            "SMHI has complete internal freeze/download timestamps, but a "
            "20 KB observation-header probe occurred before freeze.",
            "These two locked external tests do not by themselves prove "
            "provider independence or station independence.",
            "The controlled injections remain protocol-generated attacks, "
            "not real production incident labels.",
            "The SMHI locked-transfer gate is not passed, so the external "
            "performance claim remains bounded.",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    detail.to_csv(
        TABLES / f"{PREFIX}_providers.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = [
        "# 阶段275 外部预注册血缘审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 外部机构数：`{payload['provider_count']}`",
        "- 下载前冻结协议：FMI 与 SMHI 均为是",
        "- SMHI 内部时间链：冻结、评价契约、下载时间均存在",
        "- FMI 内部时间链：只有协议冻结时间；下载明细缺少下载 UTC 时间字段",
        "- SMHI 限制：冻结前有 `20 KB` 头部探测，因此不是严格零观测访问",
        "",
        "## 边界",
        "",
        "- 本阶段只读既有产物，不下载、不训练、不校阈值、不改变预测。",
        "- 该审计不能替代全新的确认性测试包。",
        "- 该审计不能证明机构无关、站点无关或真实生产攻击召回率。",
        "- 只要新锁定性能门槛未通过，全面领先主张仍不能成立。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "provider_count": payload["provider_count"],
                "strict_zero_observation_access_before_freeze": (
                    payload["strict_zero_observation_access_before_freeze"]
                ),
                "time_chain_complete_for_both_providers": (
                    payload["time_chain_complete_for_both_providers"]
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
