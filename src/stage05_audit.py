from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"F:\数据集\Analysis\experiments\pc_mstf_ap_v4\parquet\v4_anomaly_injection_table.parquet")
EXTENDED = ROOT / "tables" / "stage05_controlled_attack_extensions.parquet"
EVENTS = ROOT / "tables" / "stage05_controlled_attack_events.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    source_hash_before = sha256(SOURCE)
    source = pd.read_parquet(SOURCE)
    extended = pd.read_parquet(EXTENDED)
    events = pd.read_csv(EVENTS)
    source_hash_after = sha256(SOURCE)
    attack_rows = extended["extension_is_attack"].astype(bool)
    labeled_events = set(extended.loc[attack_rows, "extension_event_id"].dropna().astype(str)) - {""}
    manifest_events = set(events["event_id"].astype(str))
    changed = (
        pd.to_numeric(extended.loc[attack_rows, "extension_observed_value"], errors="coerce")
        - pd.to_numeric(extended.loc[attack_rows, "original_value"], errors="coerce")
    )
    checks = {
        "source_hash_unchanged": source_hash_before == source_hash_after,
        "row_count_unchanged": len(source) == len(extended),
        "no_extension_attack_in_train": not bool((attack_rows & extended["split"].eq("train")).any()),
        "event_manifest_matches_labels": labeled_events == manifest_events,
        "all_attack_rows_have_event_id": bool(extended.loc[attack_rows, "extension_event_id"].astype(str).ne("").all()),
        "attack_values_changed_or_missing": bool((changed.abs().gt(0) | changed.isna()).all()),
        "only_expected_extension_types": set(extended.loc[attack_rows, "extension_attack_type"].unique()) == {"replay", "coordinated_offset", "mixed_missing_attack"},
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_sha256": source_hash_after,
        "source_rows": int(len(source)),
        "extended_rows": int(len(extended)),
        "extension_attack_rows": int(attack_rows.sum()),
        "event_count": int(len(events)),
        "labeling": "受控攻击扩展，不代表真实网络入侵",
    }
    out_json = ROOT / "tables" / "stage05_audit.json"
    out_md = ROOT / "reports" / "stage05_audit.md"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段5受控攻击扩展审计", "", f"- 状态：**{result['status']}**", f"- 事件数：`{len(events)}`", f"- 扩展攻击行数：`{int(attack_rows.sum())}`", ""]
    lines += [f"- {'通过' if value else '未通过'}：`{key}`" for key, value in checks.items()]
    lines += ["", "所有扩展均为真实气象观测上的受控攻击实验，不代表真实网络入侵事件。"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
