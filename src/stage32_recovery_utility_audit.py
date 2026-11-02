from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


ARTIFACTS = {
    "9B核心F1运行点": TABLES / "stage09b_test_recovery.parquet",
    "9D低误报运行点": TABLES / "stage09d_test_recovery.parquet",
    "22严格留一站点": TABLES / "stage22_leave_one_station_refit_detail.parquet",
    "23无标签站点校准": TABLES / "stage23_unlabeled_site_calibration_detail.parquet",
}


def summarize(name: str, frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    df = frame.copy()
    obs_col = "effective_observed_value" if "effective_observed_value" in df else "observed_value"
    true = pd.to_numeric(df["original_value"], errors="coerce").to_numpy(dtype=float)
    observed = pd.to_numeric(df[obs_col], errors="coerce").to_numpy(dtype=float)
    recovered = pd.to_numeric(df["recovered_value"], errors="coerce").to_numpy(dtype=float)
    pred = df["predicted_attack"].astype(bool).to_numpy()
    label = df["combined_is_attack"].astype(bool).to_numpy()
    finite = np.isfinite(true) & np.isfinite(observed) & np.isfinite(recovered)
    attack_detected = finite & label & pred
    attack_missed = finite & label & ~pred
    normal_alert = finite & ~label & pred
    before_err = np.abs(true - observed)
    after_err = np.abs(true - recovered)
    improvement = before_err - after_err
    details = []
    for station, idx_station in df.groupby("held_out_station" if "held_out_station" in df else "station_key", sort=True).groups.items():
        idx = np.asarray(idx_station, dtype=int)
        ad = attack_detected[idx]
        am = attack_missed[idx]
        nf = normal_alert[idx]
        details.append({
            "artifact": name,
            "station": station,
            "detected_attack_rows": int(ad.sum()),
            "detected_improved_rows": int((improvement[idx][ad] > 1e-9).sum()),
            "detected_worsened_rows": int((improvement[idx][ad] < -1e-9).sum()),
            "detected_recovery_improvement_rate": float(np.mean(improvement[idx][ad] > 1e-9)) if ad.any() else float("nan"),
            "detected_recovery_worsening_rate": float(np.mean(improvement[idx][ad] < -1e-9)) if ad.any() else float("nan"),
            "detected_mae_before": float(np.mean(before_err[idx][ad])) if ad.any() else float("nan"),
            "detected_mae_after": float(np.mean(after_err[idx][ad])) if ad.any() else float("nan"),
            "missed_attack_rows": int(am.sum()),
            "missed_mae_before": float(np.mean(before_err[idx][am])) if am.any() else float("nan"),
            "missed_mae_after": float(np.mean(after_err[idx][am])) if am.any() else float("nan"),
            "false_positive_alert_rows": int(nf.sum()),
            "false_positive_changed_rows": int((np.abs(recovered[idx][nf] - observed[idx][nf]) > 1e-9).sum()),
            "false_positive_mean_abs_change": float(np.mean(np.abs(recovered[idx][nf] - observed[idx][nf]))) if nf.any() else 0.0,
            "false_positive_harm_rate": float(np.mean(after_err[idx][nf] > before_err[idx][nf] + 1e-9)) if nf.any() else 0.0,
        })
    y = label & finite
    overall = {
        "artifact": name,
        "rows": int(len(df)),
        "detected_attack_rows": int(attack_detected.sum()),
        "detected_recovery_improvement_rate": float(np.mean(improvement[attack_detected] > 1e-9)) if attack_detected.any() else float("nan"),
        "detected_recovery_worsening_rate": float(np.mean(improvement[attack_detected] < -1e-9)) if attack_detected.any() else float("nan"),
        "detected_mae_before": float(np.mean(before_err[attack_detected])) if attack_detected.any() else float("nan"),
        "detected_mae_after": float(np.mean(after_err[attack_detected])) if attack_detected.any() else float("nan"),
        "missed_attack_rows": int(attack_missed.sum()),
        "false_positive_alert_rows": int(normal_alert.sum()),
        "false_positive_changed_rows": int((np.abs(recovered[normal_alert] - observed[normal_alert]) > 1e-9).sum()),
        "false_positive_mean_abs_change": float(np.mean(np.abs(recovered[normal_alert] - observed[normal_alert]))) if normal_alert.any() else 0.0,
        "false_positive_harm_rate": float(np.mean(after_err[normal_alert] > before_err[normal_alert] + 1e-9)) if normal_alert.any() else 0.0,
    }
    return overall, pd.DataFrame(details)


def main() -> int:
    overall_rows, detail_frames = [], []
    for name, path in ARTIFACTS.items():
        if not path.exists():
            continue
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            overall_rows.append({"artifact": name, "rows": payload.get("test", {}).get("rows", 0), "detected_attack_rows": payload.get("test", {}).get("attack_rows", 0), "detected_recovery_improvement_rate": float("nan"), "detected_recovery_worsening_rate": float("nan"), "detected_mae_before": payload.get("test_recovery", {}).get("mae_before", payload.get("test_recovery", {}).get("mae_before", float("nan"))), "detected_mae_after": payload.get("test_recovery", {}).get("mae_after", float("nan")), "missed_attack_rows": float("nan"), "false_positive_alert_rows": float("nan"), "false_positive_changed_rows": float("nan"), "false_positive_mean_abs_change": float("nan"), "false_positive_harm_rate": float("nan"), "note": "aggregate-only JSON"})
            continue
        frame = pd.read_parquet(path)
        overall, detail = summarize(name, frame)
        overall_rows.append(overall)
        if not detail.empty:
            detail_frames.append(detail)
    overall = pd.DataFrame(overall_rows)
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    overall.to_csv(TABLES / "stage32_recovery_utility_overall.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(TABLES / "stage32_recovery_utility_by_station.csv", index=False, encoding="utf-8-sig")
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "只对锁定test逐行检查：恢复仅作用于报警行；检测到的真实攻击分别统计改善/恶化；误报行单独统计改写和伤害率；不重新调参。", "overall": overall.to_dict(orient="records"), "by_station": detail.to_dict(orient="records")}
    (TABLES / "stage32_recovery_utility_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段32恢复处置效用审计", "", "该审计区分检测到的真实攻击、漏报攻击和误报触发的恢复动作；不重新调参。", "", "| 产物 | 检测攻击行 | 检测后MAE前→后 | 检测后改善率 | 检测后恶化率 | 漏报攻击行 | 误报改写行 | 误报平均改变量 | 误报伤害率 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in overall.iterrows():
        lines.append(f"| {r['artifact']} | {int(r['detected_attack_rows']) if pd.notna(r['detected_attack_rows']) else 'NA'} | {r['detected_mae_before']:.4f}→{r['detected_mae_after']:.4f} | {r['detected_recovery_improvement_rate']:.4f} | {r['detected_recovery_worsening_rate']:.4f} | {int(r['missed_attack_rows']) if pd.notna(r['missed_attack_rows']) else 'NA'} | {int(r['false_positive_changed_rows']) if pd.notna(r['false_positive_changed_rows']) else 'NA'} | {r['false_positive_mean_abs_change']:.4f} | {r['false_positive_harm_rate']:.4f} |")
    lines += ["", "按站点细分见 `stage32_recovery_utility_by_station.csv`。误报触发的恢复动作不应被计入真实攻击恢复收益。"]
    (REPORTS / "stage32_recovery_utility_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "artifacts": len(overall), "station_rows": len(detail)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
