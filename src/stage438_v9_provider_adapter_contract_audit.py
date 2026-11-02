from __future__ import annotations

"""Stage 438: audit the V9 provider-specific raw-data adapter contract.

This is a source-contract audit.  It does not download data, open official
observation bodies, fit a detector, generate attacks, or evaluate a test set.
"""

import hashlib
import json
from pathlib import Path

from v9_official_adapters import COMMON_COLUMNS, PARSER_VERSION, PROVIDER_CONTRACTS


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
TEMPLATE = SRC / "stage425_v9_2026_official_window_download_template.py"
ADAPTERS = SRC / "v9_official_adapters.py"
OUT_JSON = ROOT / "tables" / "stage438_v9_provider_adapter_contract_audit.json"
OUT_REPORT = ROOT / "reports" / "stage438_v9_provider_adapter_contract_audit.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol = json.loads(
        (PACKAGE / "protocol_freeze.json").read_text(encoding="utf-8")
    )
    adapter_text = ADAPTERS.read_text(encoding="utf-8")
    template_text = TEMPLATE.read_text(encoding="utf-8")
    expected = {"MeteoSwiss", "NOAA ISD", "KNMI", "ECCC"}
    checks = [
        {
            "check": "provider_scope_exact",
            "status": "PASS"
            if set(PROVIDER_CONTRACTS) == expected
            and set(protocol.get("providers", {})) == expected
            else "FAIL",
            "value": sorted(PROVIDER_CONTRACTS),
        },
        {
            "check": "common_output_schema",
            "status": "PASS"
            if set(COMMON_COLUMNS)
            == {
                "station_id",
                "timestamp_utc",
                "temperature_c",
                "observation_present",
            }
            else "FAIL",
            "value": list(COMMON_COLUMNS),
        },
        {
            "check": "provider_parser_functions",
            "status": "PASS"
            if all(
                f"def parse_{name}" in adapter_text
                for name in ("meteoswiss", "noaa", "knmi", "eccc")
            )
            else "FAIL",
        },
        {
            "check": "raw_adapter_is_used_by_template",
            "status": "PASS"
            if "from v9_official_adapters import" in template_text
            and "parse_provider(provider, inputs)" in template_text
            else "FAIL",
        },
        {
            "check": "all_raw_members_hash_bound",
            "status": "PASS"
            if '"raw_sha256s"' in template_text
            and "raw member SHA256 does not match manifest" in (
                (SRC / "stage425_v9_compatibility.py").read_text(
                    encoding="utf-8"
                )
            )
            else "FAIL",
        },
        {
            "check": "no_historical_substitution",
            "status": "PASS"
            if "historical_normalized_path" not in adapter_text
            and "normalized_path" not in adapter_text
            else "FAIL",
        },
        {
            "check": "parser_version_declared",
            "status": "PASS" if PARSER_VERSION else "FAIL",
            "value": PARSER_VERSION,
        },
    ]
    # The explicit provider rules are part of the audit record, so a reviewer
    # can see the raw-to-common field contract without opening observation data.
    checks.append(
        {
            "check": "provider_field_contracts_complete",
            "status": "PASS"
            if all(
                all(
                    key in PROVIDER_CONTRACTS[provider]
                    for key in (
                        "format",
                        "station_field",
                        "timestamp_field",
                        "temperature_field",
                        "timestamp_rule",
                        "temperature_rule",
                    )
                )
                for provider in expected
            )
            else "FAIL",
        }
    )
    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_PROVIDER_ADAPTER_CONTRACT_AUDIT"
        if not failures
        else "FAIL_V9_PROVIDER_ADAPTER_CONTRACT_AUDIT"
    )
    payload = {
        "stage": 438,
        "status": status,
        "parser_version": PARSER_VERSION,
        "provider_contracts": PROVIDER_CONTRACTS,
        "checks": checks,
        "failed_checks": failures,
        "observation_values_read": False,
        "http_response_bodies_read": False,
        "attack_labels_read": False,
        "performance_evaluated": False,
        "hashes": {
            "adapter": sha256(ADAPTERS),
            "template": sha256(TEMPLATE),
            "compatibility": sha256(SRC / "stage425_v9_compatibility.py"),
        },
        "claim_boundary": [
            "本阶段只审计 provider-specific 原始格式适配契约，不产生性能结果。",
            "官方 2026-10 数据缺失时仍必须保持 V9 NOT_READY。",
            "适配层不插补观测、不选择站点、不校准阈值，也不读取攻击标签。",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage438 V9 provider-specific 适配契约审计",
                "",
                f"- 状态：`{status}`",
                f"- parser version：`{PARSER_VERSION}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 仅审计源代码和冻结协议元数据，未下载或读取官方观测正文。",
                "- 四个 provider 均使用独立解析规则，输出统一中间格式。",
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"stage": 438, "status": status, "failed_checks": failures},
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
