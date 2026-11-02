from __future__ import annotations

"""Stage 366: freeze the evaluation contract for the untouched SMHI holdout.

The station set was frozen before observation access in Stage364 and audited
in Stage365. This stage creates immutable, holdout-specific copies of the
existing method and metric contracts before any holdout performance result is
computed. It does not fit, score, tune, or inspect test labels.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL_PREFIX = "stage364_smhi_station_holdout_protocol"
DOWNLOAD_PREFIX = "stage365_smhi_station_holdout_download"
SOURCE_DATA_PROTOCOL = TABLES / "stage264_smhi_protocol_preregistration.json"
SOURCE_CONTRACT = TABLES / "stage265_smhi_locked_evaluation_contract.json"
PREFIX = "stage366_smhi_station_holdout_contract"


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
    protocol = json.loads(
        (TABLES / f"{PROTOCOL_PREFIX}.json").read_text(encoding="utf-8")
    )
    download = json.loads(
        (TABLES / f"{DOWNLOAD_PREFIX}.json").read_text(encoding="utf-8")
    )
    if protocol["status"] != "PROTOCOL_FROZEN_BEFORE_OBSERVATION_ACCESS":
        raise RuntimeError("Stage364 protocol is not frozen")
    if download["status"] != (
        "PASS_SMHI_STATION_HOLDOUT_DOWNLOAD_AND_COVERAGE_AUDIT"
    ):
        raise RuntimeError("Stage365 download audit did not pass")
    if download["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("Stage365 performance was already evaluated")
    station_table = pd.read_csv(
        TABLES / f"{PROTOCOL_PREFIX}_stations.csv",
        dtype={"station_id": str},
    )
    if station_table["station_id"].astype(str).tolist() != protocol["station_ids"]:
        raise RuntimeError("Stage364 station table changed")
    source_protocol = json.loads(
        SOURCE_DATA_PROTOCOL.read_text(encoding="utf-8")
    )
    source_contract = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))

    data_protocol = dict(source_protocol)
    data_protocol.update(
        {
            "status": "PROTOCOL_FROZEN_BEFORE_OBSERVATION_ACCESS",
            "protocol_id": "TMQC-SMHI-STATION-HOLDOUT-DATA-20260924-v1",
            "station_ids": protocol["station_ids"],
            "station_names": protocol["station_names"],
            "station_count": protocol["station_count"],
            "eligible_station_count": protocol[
                "eligible_station_count_after_exclusion"
            ],
            "station_selection_rule": (
                "Use the Stage364 frozen metadata-only holdout: exclude all "
                "18 Stage267 station IDs, then select six evenly spaced "
                "stations from each fixed latitude band among the remaining "
                "four-parameter metadata intersection."
            ),
            "stations_sha256": sha256_file(
                TABLES / f"{PROTOCOL_PREFIX}_stations.csv"
            ),
            "observation_access": (
                "Stage364 froze station IDs before observation access; "
                "Stage365 downloaded and audited only these IDs."
            ),
            "download_status": "DOWNLOADED_AFTER_PROTOCOL_FREEZE",
            "performance_status": "NOT_EVALUATED",
            "source_protocol_id": source_protocol["protocol_id"],
            "source_protocol_sha256": sha256_file(SOURCE_DATA_PROTOCOL),
            "holdout_protocol_sha256": sha256_file(
                TABLES / f"{PROTOCOL_PREFIX}.json"
            ),
        }
    )
    data_protocol_path = TABLES / f"{PREFIX}_data_protocol.json"
    write_json(data_protocol_path, data_protocol)

    contract = dict(source_contract)
    contract.update(
        {
            "status": "EVALUATION_CONTRACT_FROZEN_BEFORE_PERFORMANCE",
            "contract_id": "TMQC-SMHI-STATION-HOLDOUT-EVAL-20260924-v1",
            "data_protocol_id": data_protocol["protocol_id"],
            "data_protocol_sha256": sha256_file(data_protocol_path),
            "stations_sha256": data_protocol["stations_sha256"],
            "download_status": "DOWNLOADED_AFTER_PROTOCOL_FREEZE",
            "performance_status": "NOT_EVALUATED",
            "provider_level": "single_provider_SMHI_station_holdout",
            "observation_access": (
                "The Stage364 station IDs were frozen before observation "
                "access. Stage365 completed the official download and "
                "coverage audit. No holdout performance result has been read."
            ),
            "source_contract_id": source_contract["contract_id"],
            "source_contract_sha256": sha256_file(SOURCE_CONTRACT),
            "holdout_protocol_id": protocol["protocol_id"],
            "holdout_download_id": download["protocol_id"],
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "claim_boundary": (
                "This contract evaluates an untouched station holdout within "
                "SMHI. It is not a new independent provider, and all attacks "
                "remain controlled simulations rather than real incidents."
            ),
        }
    )
    contract_path = TABLES / f"{PREFIX}_evaluation_contract.json"
    write_json(contract_path, contract)

    summary = {
        "status": "PASS_SMHI_STATION_HOLDOUT_CONTRACT_FROZEN",
        "protocol_id": protocol["protocol_id"],
        "data_protocol_id": data_protocol["protocol_id"],
        "contract_id": contract["contract_id"],
        "station_count": int(len(station_table)),
        "method_count": int(len(contract["methods"])),
        "seed_count": int(len(contract["bootstrap"]["seed"] if isinstance(contract["bootstrap"].get("seed"), list) else [3407, 6257, 7919, 104729, 20260715])),
        "primary_far": contract["primary_far"],
        "primary_endpoint": contract["primary_endpoint"],
        "performance_status": "NOT_EVALUATED",
        "observation_values_selected_after_freeze": True,
        "test_results_read": False,
        "data_protocol_sha256": sha256_file(data_protocol_path),
        "evaluation_contract_sha256": sha256_file(contract_path),
    }
    # Keep the seed count explicit because the historical contract stores the
    # seed list in the data protocol rather than under bootstrap.
    summary["seed_count"] = int(len(data_protocol["seeds"]))
    output = TABLES / f"{PREFIX}.json"
    write_json(output, summary)
    report = REPORTS / f"{PREFIX}.md"
    report.write_text(
        "\n".join(
            [
                "# Stage366 SMHI 未使用站点评价契约冻结",
                "",
                f"- 状态：`{summary['status']}`",
                f"- 数据协议：`{summary['data_protocol_id']}`",
                f"- 评价契约：`{summary['contract_id']}`",
                f"- 站点数：`{summary['station_count']}`",
                f"- 方法数：`{summary['method_count']}`",
                f"- 种子数：`{summary['seed_count']}`",
                f"- 主 FAR：`{summary['primary_far']}`",
                f"- 主终点：`{summary['primary_endpoint']}`",
                "- 性能结果访问：否。",
                "",
                "## 冻结边界",
                "",
                "- 13 个方法、5 个种子、FAR=0.01、事件覆盖主终点和统计规则沿用既有契约。",
                "- 不在 SMHI holdout test 上调参、改特征、改权重、改阈值或删减基线。",
                "- 该 holdout 属于 SMHI 内部新站点边界，不是新增独立机构。",
                "- 受控攻击仍是透明模拟，不能写成真实攻击或生产事故。",
                "",
                "## 产物",
                "",
                f"- `{data_protocol_path}`",
                f"- `{contract_path}`",
                f"- `{output}`",
                f"- `{report}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
