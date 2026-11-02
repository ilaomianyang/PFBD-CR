from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


FILES = {
    "stage09b_core": "stage09b_test_recovery.parquet",
    "stage09d_low_far": "stage09d_test_recovery.parquet",
    "stage09b_ablation_nostruct": "stage09b_test_recovery_ablation_nostruct.parquet",
    "stage09b_ablation_noera5": "stage09b_test_recovery_ablation_noera5.parquet",
    "stage09b_empirical": "stage09b_test_recovery_empirical.parquet",
    "stage09b_intensity0p5": "stage09b_test_recovery_intensity0p5.parquet",
    "stage09b_intensity1p0": "stage09b_test_recovery_intensity1p0.parquet",
    "stage09b_intensity1p5": "stage09b_test_recovery_intensity1p5.parquet",
    "stage09b_intensity2p0": "stage09b_test_recovery_intensity2p0.parquet",
    "stage22_strict_refit": "stage22_leave_one_station_refit_detail.parquet",
    "stage23_unlabeled_calibration": "stage23_unlabeled_site_calibration_detail.parquet",
}


def main() -> int:
    rows = []
    failures = []
    for name, filename in FILES.items():
        path = TABLES / filename
        if not path.exists():
            failures.append(f"missing:{filename}")
            continue
        df = pd.read_parquet(path)
        observed_col = "effective_observed_value" if "effective_observed_value" in df else "observed_value"
        if not {"predicted_attack", "recovered_value", observed_col}.issubset(df.columns):
            failures.append(f"columns:{filename}")
            continue
        pred = df["predicted_attack"].astype(bool).to_numpy()
        observed = pd.to_numeric(df[observed_col], errors="coerce").to_numpy(dtype=float)
        recovered = pd.to_numeric(df["recovered_value"], errors="coerce").to_numpy(dtype=float)
        changed = np.isfinite(observed) & np.isfinite(recovered) & (np.abs(recovered - observed) > 1e-9)
        unflagged_changed = int((changed & ~pred).sum())
        # A normal row may be changed only when the detector raised an alert;
        # this is the false-positive recovery cost, distinct from silent edits.
        normal_rows_changed_by_alert = int((changed & (~df.get("combined_is_attack", pd.Series(False, index=df.index)).astype(bool).to_numpy()) & pred).sum()) if "combined_is_attack" in df else int((changed & ~pred).sum())
        rows.append({"artifact": name, "file": filename, "rows": int(len(df)), "predicted_rows": int(pred.sum()), "changed_rows": int(changed.sum()), "unflagged_changed_rows": unflagged_changed, "unflagged_unchanged_pass": unflagged_changed == 0, "normal_rows_changed_by_alert": normal_rows_changed_by_alert})
        if unflagged_changed:
            failures.append(f"unflagged_changed:{filename}:{unflagged_changed}")
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "stage30_recovery_integrity_audit.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS" if not failures else "FAIL", "protocol": "只有predicted_attack=True的行允许使用时空/ERA5候选；未报警行必须原样透传。误报行属于报警动作，可单独统计但不作为攻击恢复收益。", "results": rows, "failures": failures}
    (TABLES / "stage30_recovery_integrity_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段30恢复完整性审计", "", f"- 状态：**{payload['status']}**", "- 规则：只有预测异常行允许恢复；未报警行必须原样透传。", "", "| 产物 | 行数 | 报警行 | 改写行 | 未报警改写行 | 状态 |", "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['artifact']} | {r['rows']} | {r['predicted_rows']} | {r['changed_rows']} | {r['unflagged_changed_rows']} | {'PASS' if r['unflagged_unchanged_pass'] else 'FAIL'} |")
    lines += ["", "误报行可能被恢复动作改写，这是报警策略的代价；本审计只禁止未报警行被静默改写。"]
    (REPORTS / "stage30_recovery_integrity_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "artifacts": len(rows), "failures": len(failures)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
