from __future__ import annotations

"""Stage 309: audit additional public ICS/CPS sources before any download.

This stage is deliberately read-only. It does not download datasets, inspect
labels, train a detector, calibrate a threshold, or compute performance. Its
purpose is to distinguish four things that are often conflated:

1. a public benchmark from a controlled testbed,
2. an independently labelled operational dataset,
3. an authorized production incident log,
4. a synthetic or replayed dataset that cannot support production claims.

The audit can therefore narrow the evidence gap and block unsafe promotion,
but it cannot by itself close P0-3 or P0-5.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage309_public_ics_cps_source_feasibility"
PROTOCOL_ID = "TMQC-PUBLIC-ICS-CPS-SOURCE-FEASIBILITY-20260921-v1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # The source facts below are deliberately conservative. A reachable
    # landing page or a repository mirror is not treated as proof that a
    # dataset is downloadable, licensed for the intended use, independently
    # labelled, or equivalent to a production incident log.
    rows = [
        {
            "candidate_id": "swat",
            "dataset": "SWaT",
            "domain": "water treatment testbed",
            "official_entry": (
                "https://itrust.sutd.edu.sg/itrust-labs_datasets/"
                "datasets/"
            ),
            "primary_reference": (
                "https://itrust.sutd.edu.sg/itrust-labs_datasets/"
                "datasets/"
            ),
            "data_type": "controlled_testbed",
            "independent_labels": "YES",
            "event_level_labels": "YES",
            "production_incident_log": False,
            "official_machine_access": (
                "REQUEST_OR_AUTHOR_LINK: the official landing page describes "
                "the testbed datasets; the local USAD evaluation used the "
                "author-linked CSV mirror and is retained as a separate "
                "provenance chain."
            ),
            "license_status": (
                "REFERENCE_ONLY: the official page does not establish a "
                "standalone redistribution licence for the CSV mirror. "
                "Redistribution must not be assumed."
            ),
            "same_contract_ready": False,
            "promotion_status": "BLOCKED_DIFFERENT_CONTRACT_AND_REDISTRIBUTION",
            "blocking_reason": (
                "SWaT is an authorized testbed benchmark, not a production "
                "incident log. It is also not the TrustMeteo-QC weather "
                "station-hour contract, so a USAD/SWaT result cannot be "
                "merged into the same-contract win/loss matrix."
            ),
        },
        {
            "candidate_id": "wadi",
            "dataset": "WADI",
            "domain": "water distribution testbed",
            "official_entry": (
                "https://itrust.sutd.edu.sg/itrust-labs_datasets/"
                "datasets/"
            ),
            "primary_reference": (
                "https://itrust.sutd.edu.sg/itrust-labs_datasets/"
                "datasets/"
            ),
            "data_type": "controlled_testbed",
            "independent_labels": "YES",
            "event_level_labels": "YES",
            "production_incident_log": False,
            "official_machine_access": (
                "REQUEST_OR_MANUAL_DOWNLOAD: the official landing page "
                "describes the testbed datasets; a stable unauthenticated "
                "machine-readable archive was not established."
            ),
            "license_status": (
                "REFERENCE_ONLY: no standalone redistribution licence was "
                "verified for this audit."
            ),
            "same_contract_ready": False,
            "promotion_status": "BLOCKED_DIFFERENT_CONTRACT_AND_ACCESS",
            "blocking_reason": (
                "WADI can support a separate ICS testbed benchmark, but it "
                "cannot support production-incident recall or a same-contract "
                "comparison with the weather station network."
            ),
        },
        {
            "candidate_id": "batadal",
            "dataset": "BATADAL",
            "domain": "water distribution network",
            "official_entry": (
                "https://www.batadal.net/data.html"
            ),
            "primary_reference": (
                "https://www.batadal.net/data.html"
            ),
            "data_type": "controlled_simulation_or_testbed",
            "independent_labels": "YES",
            "event_level_labels": "YES",
            "production_incident_log": False,
            "official_machine_access": (
                "PUBLIC_DOWNLOAD_PAGE: the dataset page exposes historical "
                "files, but this audit did not download or inspect labels."
            ),
            "license_status": (
                "REFERENCE_ONLY: the page must be checked for attribution and "
                "redistribution terms before any release packaging."
            ),
            "same_contract_ready": False,
            "promotion_status": "BLOCKED_DIFFERENT_CONTRACT",
            "blocking_reason": (
                "BATADAL is a separate water-network benchmark and does not "
                "provide production incident labels for the target claim."
            ),
        },
        {
            "candidate_id": "hai",
            "dataset": "HAI",
            "domain": "industrial control testbed",
            "official_entry": (
                "https://github.com/icsdataset/hai"
            ),
            "primary_reference": (
                "https://github.com/icsdataset/hai"
            ),
            "data_type": "controlled_testbed",
            "independent_labels": "YES",
            "event_level_labels": "YES",
            "production_incident_log": False,
            "official_machine_access": (
                "PUBLIC_REPOSITORY: a public repository is available, but "
                "this audit did not download the data or inspect labels."
            ),
            "license_status": (
                "REFERENCE_ONLY: the repository licence and dataset terms "
                "must be recorded before reuse."
            ),
            "same_contract_ready": False,
            "promotion_status": "BLOCKED_DIFFERENT_CONTRACT",
            "blocking_reason": (
                "HAI is a separate ICS testbed benchmark; it cannot be used "
                "as a production incident log or as a weather-domain "
                "same-contract baseline without a separately frozen protocol."
            ),
        },
        {
            "candidate_id": "smd",
            "dataset": "SMD",
            "domain": "server machine dataset",
            "official_entry": (
                "https://github.com/NetManAIOps/OmniAnomaly"
            ),
            "primary_reference": (
                "https://github.com/NetManAIOps/OmniAnomaly"
            ),
            "data_type": "operational_system_benchmark",
            "independent_labels": "YES",
            "event_level_labels": "YES",
            "production_incident_log": False,
            "official_machine_access": (
                "PUBLIC_REPOSITORY_REFERENCE: the benchmark is referenced by "
                "the OmniAnomaly repository; this audit did not download it."
            ),
            "license_status": (
                "REFERENCE_ONLY: dataset and repository terms must be checked."
            ),
            "same_contract_ready": False,
            "promotion_status": "BLOCKED_DOMAIN_AND_CONTRACT",
            "blocking_reason": (
                "SMD is a server-system anomaly benchmark, not an ICS/CPS "
                "production incident log or a weather station-hour contract."
            ),
        },
        {
            "candidate_id": "smap_msl",
            "dataset": "SMAP/MSL",
            "domain": "spacecraft telemetry",
            "official_entry": (
                "https://github.com/khundman/telemanom"
            ),
            "primary_reference": (
                "https://github.com/khundman/telemanom"
            ),
            "data_type": "operational_telemetry_benchmark",
            "independent_labels": "YES",
            "event_level_labels": "YES",
            "production_incident_log": False,
            "official_machine_access": (
                "PUBLIC_REPOSITORY_REFERENCE: the benchmark is referenced by "
                "Telemanom; this audit did not download it."
            ),
            "license_status": (
                "REFERENCE_ONLY: dataset and repository terms must be checked."
            ),
            "same_contract_ready": False,
            "promotion_status": "BLOCKED_DOMAIN_AND_CONTRACT",
            "blocking_reason": (
                "SMAP/MSL is spacecraft telemetry, not a weather or ICS "
                "station network and not a production incident log for the "
                "target claim."
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    if frame["candidate_id"].duplicated().any():
        raise RuntimeError("duplicate candidate id")
    if frame["same_contract_ready"].any():
        raise RuntimeError(
            "Stage309 must not promote a source without a frozen "
            "same-contract protocol"
        )
    if frame["production_incident_log"].any():
        raise RuntimeError(
            "Stage309 must not label a testbed or benchmark as a "
            "production incident log"
        )

    blocked = frame[
        frame["promotion_status"].str.startswith("BLOCKED")
    ]
    testbed_count = int(
        frame["data_type"].str.contains("testbed", case=False).sum()
    )
    production_log_count = int(frame["production_incident_log"].sum())
    payload = {
        "status": "PASS_READ_ONLY_PUBLIC_ICS_CPS_FEASIBILITY_AUDIT",
        "protocol_id": PROTOCOL_ID,
        "protocol": (
            "阶段309只审计公开ICS/CPS候选数据源的官方入口、数据性质、"
            "标签层级、许可边界、机器访问和能否进入同契约评价；不下载"
            "数据、不读取标签、不训练模型、不校准阈值、不计算性能。"
        ),
        "audit_date_utc": "2026-09-21",
        "candidate_count": int(len(frame)),
        "blocked_candidate_count": int(len(blocked)),
        "testbed_candidate_count": testbed_count,
        "production_incident_log_count": production_log_count,
        "same_contract_ready_count": int(frame["same_contract_ready"].sum()),
        "promotable_candidate_count": 0,
        "p0_3_status_after": "OPEN",
        "p0_5_status_after": "OPEN",
        "claim_ceiling": "NO_NEW_PERFORMANCE_OR_PRODUCTION_CLAIM",
        "conclusion": (
            "公开ICS/CPS候选可以支持独立测试床基准或互补性分析，但"
            "本轮没有发现同时满足同契约天气站点评价和真实生产事故"
            "标签要求的新数据源；P0-3与P0-5保持OPEN。"
        ),
        "candidates": rows,
        "limitations": [
            "A public landing page is not proof of licence, machine access, "
            "or reproducibility.",
            "A controlled testbed is not a production incident log.",
            "A different-domain anomaly benchmark cannot be merged into the "
            "weather same-contract win/loss matrix.",
            "This audit does not download data, inspect labels, train a "
            "model, or produce a performance result.",
        ],
        "input_sha256": {
            "stage287_public_ics_cps_upgrade": sha256_file(
                TABLES / "stage287_swat_usad_evidence_upgrade_audit.json"
            ),
            "stage277_baseline_formality": sha256_file(
                TABLES / "stage277_baseline_formality_matrix.json"
            ),
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage 309 公开 ICS/CPS 新数据源可行性审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 候选数：`{payload['candidate_count']}`",
        f"- 阻断候选：`{payload['blocked_candidate_count']}`",
        f"- 测试床候选：`{payload['testbed_candidate_count']}`",
        f"- 真实生产事故日志：`{payload['production_incident_log_count']}`",
        f"- 可直接进入同契约评价：`{payload['same_contract_ready_count']}`",
        f"- P0-3：`{payload['p0_3_status_after']}`",
        f"- P0-5：`{payload['p0_5_status_after']}`",
        f"- 声称上限：`{payload['claim_ceiling']}`",
        "",
        "## 候选结论",
        "",
        "| 数据集 | 数据性质 | 独立标签 | 生产事故日志 | 同契约可用 | 状态 |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['dataset']} | {row['data_type']} | "
            f"{row['independent_labels']} | "
            f"{row['production_incident_log']} | "
            f"{row['same_contract_ready']} | "
            f"{row['promotion_status']} |"
        )
    report += [
        "",
        "## 边界",
        "",
        *[f"- {item}" for item in payload["limitations"]],
        "",
        payload["conclusion"],
        "",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "candidate_count": payload["candidate_count"],
                "blocked_candidate_count": payload[
                    "blocked_candidate_count"
                ],
                "production_incident_log_count": payload[
                    "production_incident_log_count"
                ],
                "same_contract_ready_count": payload[
                    "same_contract_ready_count"
                ],
                "p0_3_status_after": payload["p0_3_status_after"],
                "p0_5_status_after": payload["p0_5_status_after"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
