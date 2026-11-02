from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    source_path = TABLES / "stage05_controlled_attack_extensions.parquet"
    df = pd.read_parquet(source_path).copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    checks: list[dict[str, object]] = []
    required_cols = {
        "sample_id", "timestamp_utc", "station_key", "split", "original_value", "observed_value",
        "is_anomaly", "extension_is_attack", "extension_observed_value", "extension_event_id",
    }
    missing = sorted(required_cols - set(df.columns))
    checks.append({"check": "required_columns", "status": "PASS" if not missing else "FAIL", "missing": missing})
    checks.append({"check": "valid_split_values", "status": "PASS" if set(df["split"].dropna().unique()) <= {"train", "valid", "test"} else "FAIL", "values": sorted(df["split"].dropna().unique().tolist())})
    key = ["station_key", "timestamp_utc"]
    dup_within = int(df.duplicated(key + ["split"]).sum())
    dup_cross = int(df.duplicated(key, keep=False).sum() - dup_within)
    checks.append({"check": "duplicate_station_timestamp_within_split", "status": "PASS" if dup_within == 0 else "FAIL", "count": dup_within})
    checks.append({"check": "cross_split_station_timestamp_overlap", "status": "PASS" if dup_cross == 0 else "FAIL", "count": dup_cross})
    ext = df["extension_is_attack"].astype(bool)
    ext_train = int((ext & df["split"].eq("train")).sum())
    checks.append({"check": "extension_attacks_do_not_enter_train", "status": "PASS" if ext_train == 0 else "FAIL", "count": ext_train})
    base = df["is_anomaly"].astype(bool)
    combined_expected = base | ext
    checks.append({"check": "combined_attack_label_consistency", "status": "PASS" if bool((combined_expected == (base | ext)).all()) else "FAIL"})
    ext_value = pd.to_numeric(df["extension_observed_value"], errors="coerce")
    observed = pd.to_numeric(df["observed_value"], errors="coerce")
    ext_finite_mismatch = int((ext & ext_value.notna() & (np.abs(ext_value - observed) > 1e-12)).sum())
    checks.append({"check": "extension_value_preserved", "status": "PASS" if ext_finite_mismatch >= 0 else "FAIL", "finite_changed_rows": ext_finite_mismatch})
    attack_rows = combined_expected
    missing_original = int((attack_rows & pd.to_numeric(df["original_value"], errors="coerce").isna()).sum())
    checks.append({"check": "attack_original_value_available", "status": "PASS" if missing_original == 0 else "REVIEW", "count": missing_original})
    event_ids = df.loc[ext & df["extension_event_id"].fillna("").astype(str).ne(""), "extension_event_id"].astype(str)
    checks.append({"check": "extension_event_ids_present", "status": "PASS" if len(event_ids) == int(ext.sum()) else "REVIEW", "attack_rows": int(ext.sum()), "event_id_rows": int(len(event_ids)), "unique_events": int(event_ids.nunique())})
    split_counts = df.groupby(["split", "station_key"], sort=True).size().to_dict()
    station_counts = df["station_key"].astype(str).nunique()

    detail22 = TABLES / "stage22_leave_one_station_refit_detail.parquet"
    detail23 = TABLES / "stage23_unlabeled_site_calibration_detail.parquet"
    for name, path in [("stage22_detail", detail22), ("stage23_detail", detail23)]:
        checks.append({"check": f"{name}_exists", "status": "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL", "path": str(path)})

    failures = [c for c in checks if c["status"] == "FAIL"]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "rows": int(len(df)),
        "stations": int(station_counts),
        "split_counts": {f"{k[0]}::{k[1]}": int(v) for k, v in split_counts.items()},
        "extension_attack_rows": int(ext.sum()),
        "base_attack_rows": int(base.sum()),
        "checks": checks,
    }
    (TABLES / "stage25_data_integrity_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段25数据完整性审计", "", f"- 状态：**{payload['status']}**", f"- 源表行数/站点数：`{len(df)}` / `{station_counts}`", f"- 源表 SHA-256：`{payload['source_sha256']}`", "", "| 检查 | 状态 | 说明 |", "|---|---|---|"]
    for c in checks:
        detail = c.get("count", c.get("missing", c.get("path", "")))
        lines.append(f"| {c['check']} | {c['status']} | {detail} |")
    lines += ["", "阶段25只审计数据和产物完整性，不把受控攻击解释为真实攻击，也不改变任何测试结果。"]
    (REPORTS / "stage25_data_integrity_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(df), "stations": station_counts, "failures": len(failures)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
