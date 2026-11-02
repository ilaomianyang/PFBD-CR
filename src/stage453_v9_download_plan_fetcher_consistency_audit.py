from __future__ import annotations

"""Stage 453: compare the V9 dry-run plan with the formal fetcher.

This audit is metadata-only. It imports the two local planning functions
without making a request and checks that every frozen station has identical
provider, URL, UTC window, raw directory and file-naming contracts in the
dry-run and formal-fetch paths. ECCC pagination is checked separately
because the formal fetcher expands one station into multiple raw pages.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
DRY_RUN_JSON = ROOT / "tables" / "stage428_v9_official_download_dry_run.json"
DRY_RUN_CSV = ROOT / "tables" / "stage428_v9_official_download_dry_run.csv"
FETCHER = ROOT / "src" / "stage442_v9_official_window_fetch.py"
DRY_RUNNER = ROOT / "src" / "stage428_v9_official_download_dry_run.py"
OUT_JSON = ROOT / "tables" / "stage453_v9_download_plan_fetcher_consistency_audit.json"
OUT_REPORT = ROOT / "reports" / "stage453_v9_download_plan_fetcher_consistency_audit.md"

START = "2026-10-01T00:00:00Z"
END = "2026-10-31T23:00:00Z"
EXPECTED_PROVIDERS = ("MeteoSwiss", "NOAA ISD", "KNMI", "ECCC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def naming_contract(provider: str, station_id: str) -> str:
    if provider == "ECCC":
        return f"{station_id}_offset_*.csv"
    suffix = "zip" if provider == "KNMI" else "csv"
    return f"{station_id}_2026_10.{suffix}"


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    dry_payload = json.loads(DRY_RUN_JSON.read_text(encoding="utf-8"))
    dry = pd.read_csv(DRY_RUN_CSV, dtype=str)
    dry_module = load_module("stage428_v9_download_plan", DRY_RUNNER)
    fetch_module = load_module("stage442_v9_window_fetch", FETCHER)
    formal = pd.DataFrame(fetch_module.plans(protocol))

    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, value: object, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "value": value,
                "detail": detail,
            }
        )

    add(
        "dry_run_artifact_is_metadata_only",
        dry_payload.get("status") == "PASS_V9_OFFICIAL_DOWNLOAD_DRY_RUN"
        and dry_payload.get("request_made") is False
        and dry_payload.get("response_body_read") is False
        and dry_payload.get("observation_values_read") is False,
        {
            "status": dry_payload.get("status"),
            "request_made": dry_payload.get("request_made"),
            "response_body_read": dry_payload.get("response_body_read"),
        },
        "比较只能使用下载计划，不得读取响应正文或观测值。",
    )
    add(
        "formal_plan_has_frozen_scope",
        set(formal["provider"]) == set(EXPECTED_PROVIDERS)
        and len(formal) == 51
        and set(formal["window_start_utc"]) == {START}
        and set(formal["window_end_utc"]) == {END},
        {
            "provider_count": int(formal["provider"].nunique()),
            "station_plan_rows": int(len(formal)),
            "window_start": sorted(formal["window_start_utc"].unique().tolist()),
            "window_end": sorted(formal["window_end_utc"].unique().tolist()),
        },
        "正式采集计划必须保持冻结四 provider、51 个站点和同一 UTC 窗口。",
    )
    add(
        "dry_run_plan_has_frozen_scope",
        set(dry["provider"]) == set(EXPECTED_PROVIDERS)
        and len(dry) == 51
        and set(dry["window_start_utc"]) == {START}
        and set(dry["window_end_utc"]) == {END},
        {
            "provider_count": int(dry["provider"].nunique()),
            "station_plan_rows": int(len(dry)),
            "window_start": sorted(dry["window_start_utc"].unique().tolist()),
            "window_end": sorted(dry["window_end_utc"].unique().tolist()),
        },
        "dry-run 计划必须保持冻结四 provider、51 个站点和同一 UTC 窗口。",
    )

    dry_index = dry.set_index(["provider", "station_id"])
    formal_index = formal.set_index(["provider", "station_id"])
    keys_match = set(dry_index.index) == set(formal_index.index)
    add(
        "provider_station_sets_match",
        keys_match,
        {
            "dry_only": sorted(set(dry_index.index) - set(formal_index.index)),
            "formal_only": sorted(set(formal_index.index) - set(dry_index.index)),
        },
        "每个冻结 provider/station 组合必须在两个入口中恰好出现一次。",
    )

    url_mismatches: list[dict[str, str]] = []
    window_mismatches: list[dict[str, str]] = []
    directory_mismatches: list[dict[str, str]] = []
    naming_mismatches: list[dict[str, str]] = []
    if keys_match:
        for key in sorted(formal_index.index):
            provider, station_id = key
            formal_row = formal_index.loc[key]
            dry_row = dry_index.loc[key]
            expected_dry_url = dry_module.build_url(provider, station_id)
            expected_dir = str(
                PACKAGE
                / "raw_official_2026_10"
                / provider.replace(" ", "_").lower()
            )
            expected_name = naming_contract(provider, station_id)
            if str(formal_row["url"]) != str(dry_row["url"]) or str(
                dry_row["url"]
            ) != expected_dry_url:
                url_mismatches.append(
                    {
                        "provider": provider,
                        "station_id": station_id,
                        "dry_run": str(dry_row["url"]),
                        "formal": str(formal_row["url"]),
                        "expected": expected_dry_url,
                    }
                )
            if (
                str(formal_row["window_start_utc"]) != START
                or str(formal_row["window_end_utc"]) != END
                or str(dry_row["window_start_utc"]) != START
                or str(dry_row["window_end_utc"]) != END
            ):
                window_mismatches.append(
                    {"provider": provider, "station_id": station_id}
                )
            if str(dry_row["expected_raw_directory"]) != expected_dir:
                directory_mismatches.append(
                    {
                        "provider": provider,
                        "station_id": station_id,
                        "actual": str(dry_row["expected_raw_directory"]),
                        "expected": expected_dir,
                    }
                )
            if str(dry_row["raw_naming_contract"]) != expected_name:
                naming_mismatches.append(
                    {
                        "provider": provider,
                        "station_id": station_id,
                        "actual": str(dry_row["raw_naming_contract"]),
                        "expected": expected_name,
                    }
                )
    add(
        "provider_station_url_contract_matches",
        keys_match and not url_mismatches,
        {"mismatch_count": len(url_mismatches), "examples": url_mismatches[:5]},
        "dry-run 与正式采集器必须逐站点使用相同 URL。",
    )
    add(
        "utc_window_contract_matches",
        keys_match and not window_mismatches,
        {"mismatch_count": len(window_mismatches), "examples": window_mismatches[:5]},
        "dry-run 与正式采集器必须逐站点使用相同 UTC 边界。",
    )
    add(
        "raw_directory_contract_matches",
        not directory_mismatches,
        {"mismatch_count": len(directory_mismatches), "examples": directory_mismatches[:5]},
        "dry-run 记录的目录必须与正式采集器的归档目录一致。",
    )
    add(
        "raw_naming_contract_matches",
        not naming_mismatches,
        {"mismatch_count": len(naming_mismatches), "examples": naming_mismatches[:5]},
        "MeteoSwiss/NOAA/KNMI 文件名及 ECCC 分页文件名必须一致。",
    )

    fetcher_text = FETCHER.read_text(encoding="utf-8")
    add(
        "eccc_pagination_contract_matches",
        all(
            token in fetcher_text
            for token in (
                '"limit": 10000',
                '"offset": offset',
                "if len(rows) < 10000",
                "offset += 10000",
                "_offset_{offset:08d}.csv",
            )
        )
        and bool(
            (
                dry.loc[dry["provider"] == "ECCC", "raw_naming_contract"]
                .astype(str)
                .str.endswith("_offset_*.csv")
            ).all()
        ),
        {
            "eccc_station_count": int((dry["provider"] == "ECCC").sum()),
            "page_size": 10000,
            "naming": "_offset_{offset:08d}.csv",
        },
        "ECCC 正式采集器的分页大小、停止条件、offset 增量和归档命名必须与计划一致。",
    )

    failures = [row["check"] for row in checks if row["status"] != "PASS"]
    status = (
        "PASS_V9_DOWNLOAD_PLAN_FETCHER_CONSISTENCY_AUDIT"
        if not failures
        else "FAIL_V9_DOWNLOAD_PLAN_FETCHER_CONSISTENCY_AUDIT"
    )
    payload = {
        "stage": 453,
        "status": status,
        "protocol_id": protocol.get("protocol_id"),
        "checks": checks,
        "failed_checks": failures,
        "provider_count": 4,
        "station_count": 51,
        "observation_values_read": False,
        "http_response_bodies_read": False,
        "performance_evaluated": False,
        "synthetic_fixture_only": False,
        "claim_boundary": [
            "本阶段只比较本地下载计划和采集器元数据，不发起网络请求。",
            "通过不代表官方数据已下载，也不代表任何攻击性能或全面领先结论。",
        ],
        "hashes": {
            "protocol": sha256(PROTOCOL),
            "dry_run_script": sha256(DRY_RUNNER),
            "dry_run_json": sha256(DRY_RUN_JSON),
            "fetcher": sha256(FETCHER),
        },
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_REPORT.write_text(
        "\n".join(
            [
                "# Stage453 V9 下载计划与正式采集器一致性审计",
                "",
                f"- 状态：`{status}`",
                f"- 检查数：`{len(checks)}`；失败数：`{len(failures)}`",
                "- 逐站点比较 provider、站点、URL、UTC 窗口、raw 目录和命名契约。",
                "- 单独检查 ECCC 分页大小、停止条件、offset 和分页文件命名。",
                "- 未发起网络请求，未读取 HTTP 响应正文、观测值、攻击标签或性能表。",
                "",
                "## 失败检查",
                "",
                *([f"- `{name}`" for name in failures] or ["- 无。"]),
                "",
                f"- JSON：`{OUT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": 453,
                "status": status,
                "failed_checks": failures,
                "observation_values_read": False,
                "performance_evaluated": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
