from __future__ import annotations

"""Stage 422: inventory of locally available independent-lock candidates.

Filename and aggregate-artifact metadata only.  Observation values, attack
labels, old test rows, and thresholds are not read.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "raw"
OUTPUT = TABLES / "stage422_unseen_lock_candidate_inventory.json"
REPORT = REPORTS / "stage422_unseen_lock_candidate_inventory.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    fmi_protocol = json.loads(
        (TABLES / "stage257_fmi_protocol_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    fmi_download = json.loads(
        (TABLES / "stage258_fmi_official_download.json").read_text(
            encoding="utf-8"
        )
    )
    noaa_2026 = json.loads(
        (TABLES / "stage416_noaa_2026_download_retry.json").read_text(
            encoding="utf-8"
        )
    )
    smhi = json.loads(
        (TABLES / "stage267_smhi_locked_transfer.json").read_text(
            encoding="utf-8"
        )
    )
    met_disjoint = json.loads(
        (TABLES / "stage312_met_norway_disjoint_locked_transfer.json").read_text(
            encoding="utf-8"
        )
    )

    fmi_ids = list(fmi_protocol["station_ids"])
    fmi_raw_ids = sorted(
        {
            path.name.split("_", 1)[0]
            for path in (RAW / "fmi").glob("*.xml")
            if "_" in path.name
        }
    )
    candidates: list[dict[str, Any]] = [
        {
            "candidate": "FMI",
            "candidate_type": "existing_provider_unretained_or_unused_station",
            "protocol_station_count": len(fmi_ids),
            "local_filename_station_count": len(fmi_raw_ids),
            "retained_station_count": fmi_download.get("retained_station_count"),
            "unretained_station_ids": sorted(
                set(fmi_ids) - set(fmi_download.get("retained_station_ids", []))
            ),
            "performance_status": "LOCKED_TRANSFER_ALREADY_EVALUATED",
            "eligible_for_new_confirmatory_lock": False,
            "reason": "冻结站点均已进入既有下载/覆盖率审计；未保留站点无满足覆盖率的证据。",
        },
        {
            "candidate": "MET_Norway_disjoint",
            "candidate_type": "disjoint_station_network",
            "performance_status": "LOCKED_TRANSFER_ALREADY_EVALUATED",
            "eligible_for_new_confirmatory_lock": False,
            "reason": "Stage312 已有 disjoint 锁定迁移结果。",
        },
        {
            "candidate": "SMHI_holdout",
            "candidate_type": "external_provider_holdout",
            "performance_status": "LOCKED_TRANSFER_ALREADY_EVALUATED",
            "eligible_for_new_confirmatory_lock": False,
            "reason": "Stage267 已完成锁定迁移，且迁移门槛未通过。",
        },
        {
            "candidate": "NOAA_ISD_2026",
            "candidate_type": "future_official_window",
            "performance_status": noaa_2026.get("performance_status"),
            "eligible_for_new_confirmatory_lock": False,
            "reason": "官方归档下载失败，normalized_rows=0；不得用旧年份替代。",
        },
    ]
    result = {
        "status": "PASS_NO_UNSEEN_LOCAL_LOCK_CANDIDATE",
        "stage": 422,
        "observation_values_read": False,
        "attack_labels_read": False,
        "old_test_rows_read": False,
        "candidate_count": len(candidates),
        "eligible_candidate_count": sum(
            bool(item["eligible_for_new_confirmatory_lock"]) for item in candidates
        ),
        "candidates": candidates,
        "input_hashes": {
            "stage257": sha256_file(TABLES / "stage257_fmi_protocol_preregistration.json"),
            "stage258": sha256_file(TABLES / "stage258_fmi_official_download.json"),
            "stage267": sha256_file(TABLES / "stage267_smhi_locked_transfer.json"),
            "stage312": sha256_file(TABLES / "stage312_met_norway_disjoint_locked_transfer.json"),
            "stage416": sha256_file(TABLES / "stage416_noaa_2026_download_retry.json"),
        },
        "rule": (
            "只读协议、聚合审计和文件名；不读取观测值、攻击标签或旧test明细，"
            "不训练、不调参、不生成性能结果。"
        ),
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage422 本地未查看锁定候选盘点",
        "",
        f"- 状态：`{result['status']}`",
        "- 读取范围：协议、聚合审计和文件名；未读取观测值、攻击标签或旧 test 明细。",
        f"- 候选数：`{result['candidate_count']}`；可直接进入新确认性锁定：`{result['eligible_candidate_count']}`。",
        "",
        "| 候选 | 性能状态 | 可进入新确认性锁定 | 结论 |",
        "|---|---|---:|---|",
    ]
    for item in candidates:
        lines.append(
            f"| {item['candidate']} | {item['performance_status']} | "
            f"{item['eligible_for_new_confirmatory_lock']} | {item['reason']} |"
        )
    lines += [
        "",
        "## 结论边界",
        "",
        "- 本地现有候选中没有可以在不复用旧 test 的前提下直接升级为新确认性评价的数据包。",
        "- 该结论不是算法性能失败，也不允许用旧年份、旧窗口或旧 test 复制替代。",
        "- 下一次可推进性能目标的前提是获得新的官方时间窗或新的独立锁定数据，并在访问前冻结协议。",
        f"- JSON：`{OUTPUT}`",
        f"- 报告：`{REPORT}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_count": result["candidate_count"],
                "eligible_candidate_count": result["eligible_candidate_count"],
                "output": str(OUTPUT),
                "report": str(REPORT),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
