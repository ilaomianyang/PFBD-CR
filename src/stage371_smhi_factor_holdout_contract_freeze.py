from __future__ import annotations

"""Freeze the method/metric contract for the Stage369 factor holdout."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL_PREFIX = "stage369_smhi_factor_holdout_protocol"
DOWNLOAD_PREFIX = "stage370_smhi_factor_holdout_download"
SOURCE_CONTRACT = TABLES / "stage265_smhi_locked_evaluation_contract.json"
PREFIX = "stage371_smhi_factor_holdout_contract"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    protocol_path = TABLES / f"{PROTOCOL_PREFIX}.json"
    download_path = TABLES / f"{DOWNLOAD_PREFIX}.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    download = json.loads(download_path.read_text(encoding="utf-8"))
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_OBSERVATION_ACCESS":
        raise RuntimeError("Stage369 protocol is not frozen")
    if download["status"] != (
        "PASS_SMHI_STATION_HOLDOUT_DOWNLOAD_AND_COVERAGE_AUDIT"
    ):
        raise RuntimeError("Stage370 download did not pass")
    if download["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("Stage370 was already evaluated")
    stations = pd.read_csv(
        TABLES / f"{PROTOCOL_PREFIX}_stations.csv",
        dtype={"station_id": str},
    )
    if stations["station_id"].astype(str).tolist() != protocol["station_ids"]:
        raise RuntimeError("Stage369 station table changed")
    source = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    contract = dict(source)
    contract.update(
        {
            "status": "EVALUATION_CONTRACT_FROZEN_BEFORE_PERFORMANCE",
            "contract_id": "TMQC-SMHI-FACTOR-HOLDOUT-EVAL-20260924-v1",
            "data_protocol_id": protocol["protocol_id"],
            "data_protocol_sha256": sha256_file(protocol_path),
            "stations_sha256": protocol["stations_sha256"],
            "download_status": "DOWNLOADED_AFTER_PROTOCOL_FREEZE",
            "performance_status": "NOT_EVALUATED",
            "provider_level": "single_provider_SMHI_factor_holdout",
            "primary_far": 0.01,
            "primary_endpoint": "event_coverage",
            "observation_access": (
                "Stage369 station IDs and factor protocol were frozen before "
                "observation access; Stage370 completed the official download. "
                "No factor-holdout performance result has been read."
            ),
            "source_contract_id": source["contract_id"],
            "source_contract_sha256": sha256_file(SOURCE_CONTRACT),
            "holdout_protocol_id": protocol["protocol_id"],
            "holdout_download_id": download["protocol_id"],
            "factor_protocol": protocol["attack_factor_protocol"],
            "exploratory_boundary": protocol["exploratory_boundary"],
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    output = TABLES / f"{PREFIX}.json"
    write_json(output, contract)
    summary = {
        "status": "PASS_SMHI_FACTOR_HOLDOUT_CONTRACT_FROZEN",
        "protocol_id": protocol["protocol_id"],
        "contract_id": contract["contract_id"],
        "station_count": int(len(stations)),
        "method_count": int(len(contract["methods"])),
        "seed_count": 5,
        "primary_far": 0.01,
        "primary_endpoint": "event_coverage",
        "performance_status": "NOT_EVALUATED",
        "exploratory": True,
        "contract_sha256": sha256_file(output),
    }
    summary_path = TABLES / f"{PREFIX}.json"
    # Keep a compact summary beside the full contract with a distinct suffix.
    summary_path = TABLES / f"{PREFIX}_summary.json"
    write_json(summary_path, summary)
    report = REPORTS / f"{PREFIX}.md"
    report.write_text(
        "\n".join(
            [
                "# Stage371 SMHI 因子压力测试评价契约冻结",
                "",
                f"- 状态：`{summary['status']}`",
                f"- 协议：`{summary['protocol_id']}`",
                f"- 评价契约：`{summary['contract_id']}`",
                f"- 站点：`{summary['station_count']}`；方法：`{summary['method_count']}`；种子：`{summary['seed_count']}`",
                "- 主 FAR：`0.01`；主终点：`event_coverage`。",
                "- 该阶段在 Stage367 结果查看后启动，结果只能标记为探索性。",
                "- 性能结果尚未读取；不得在 test 上调参或删除基线。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
