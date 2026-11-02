from __future__ import annotations

"""Freeze the evaluation contract for the third SMHI station holdout."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL_PREFIX = "stage377_smhi_third_station_holdout_protocol"
DOWNLOAD_PREFIX = "stage378_smhi_third_station_holdout_download"
SOURCE_CONTRACT = TABLES / "stage265_smhi_locked_evaluation_contract.json"
PREFIX = "stage379_smhi_third_station_contract"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    protocol_path = TABLES / f"{PROTOCOL_PREFIX}.json"
    download_path = TABLES / f"{DOWNLOAD_PREFIX}.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    download = json.loads(download_path.read_text(encoding="utf-8"))
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_OBSERVATION_ACCESS":
        raise RuntimeError("third station protocol is not frozen")
    if download["status"] != "PASS_SMHI_STATION_HOLDOUT_DOWNLOAD_AND_COVERAGE_AUDIT":
        raise RuntimeError("third station download did not pass")
    if download["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("third station download already evaluated")
    stations = pd.read_csv(
        TABLES / f"{PROTOCOL_PREFIX}_stations.csv",
        dtype={"station_id": str},
    )
    if stations["station_id"].astype(str).tolist() != protocol["station_ids"]:
        raise RuntimeError("third station list changed")
    source = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    contract = dict(source)
    contract.update(
        {
            "status": "EVALUATION_CONTRACT_FROZEN_BEFORE_PERFORMANCE",
            "contract_id": "TMQC-SMHI-THIRD-STATION-EVAL-20260924-v1",
            "data_protocol_id": protocol["protocol_id"],
            "data_protocol_sha256": sha256_file(protocol_path),
            "stations_sha256": protocol["stations_sha256"],
            "download_status": "DOWNLOADED_AFTER_PROTOCOL_FREEZE",
            "performance_status": "NOT_EVALUATED",
            "provider_level": "single_provider_SMHI_third_station_holdout",
            "primary_far": 0.01,
            "primary_endpoint": "event_coverage",
            "observation_access": (
                "Stage377 station IDs and attack factors were frozen before "
                "observation access; Stage378 completed official downloads. "
                "No third-holdout performance result has been read."
            ),
            "source_contract_id": source["contract_id"],
            "source_contract_sha256": sha256_file(SOURCE_CONTRACT),
            "holdout_protocol_id": protocol["protocol_id"],
            "holdout_download_id": download["protocol_id"],
            "factor_protocol": protocol["attack_protocol"],
            "exploratory_boundary": protocol["exploratory_boundary"],
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    output = TABLES / f"{PREFIX}.json"
    output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "PASS_SMHI_THIRD_STATION_CONTRACT_FROZEN",
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
    (TABLES / f"{PREFIX}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(
            [
                "# Stage379 SMHI 第三批站点评价契约冻结",
                "",
                f"- 状态：`{summary['status']}`",
                f"- 协议：`{summary['protocol_id']}`",
                f"- 契约：`{summary['contract_id']}`",
                f"- 站点：`{summary['station_count']}`；方法：`{summary['method_count']}`；种子：`{summary['seed_count']}`",
                "- 主 FAR：`0.01`；主终点：`event_coverage`。",
                "- 结果仅为探索性，不能替换 V6 或历史确认性主结果。",
                "- 不得在第三批 test 上调参、选方法或删除基线。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
