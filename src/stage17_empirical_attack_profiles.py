from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from stage05_attack_extensions import select_segments, stable_seed  # noqa: E402


def main() -> int:
    base_path = Path(r"F:\数据集\Analysis\experiments\pc_mstf_ap_v4\parquet\v4_anomaly_injection_table.parquet")
    noaa_path = ROOT / "tables" / "stage06_noaa_isd_normalized.parquet"
    source = pd.read_parquet(base_path).sort_values(["split", "timestamp_utc", "station_key"], kind="stable").reset_index(drop=True)
    noaa = pd.read_parquet(noaa_path).copy()
    noaa["timestamp_utc"] = pd.to_datetime(noaa["timestamp_utc"], utc=True, errors="coerce")
    temp = pd.to_numeric(noaa["temperature_c"], errors="coerce").dropna()
    diff = temp.groupby(noaa.loc[temp.index, "station_id"]).diff().dropna()
    abs_diff = diff.abs().to_numpy(dtype=float)
    signed_diff = diff.to_numpy(dtype=float)
    abs_diff = abs_diff[np.isfinite(abs_diff)]
    signed_diff = signed_diff[np.isfinite(signed_diff)]
    if len(abs_diff) < 20:
        raise ValueError("NOAA empirical profile has too few finite differences")
    q50, q90, q95 = np.quantile(abs_diff, [0.50, 0.90, 0.95])
    rng = np.random.default_rng(stable_seed(20260911, "empirical_profiles"))
    out = source.copy()
    out["extension_attack_type"] = "normal"
    out["extension_is_attack"] = False
    out["extension_delta"] = 0.0
    out["extension_event_id"] = ""
    out["extension_observed_value"] = pd.to_numeric(out["observed_value"], errors="coerce")
    events = []
    for split in ["valid", "test"]:
        part = out[(out["split"] == split) & out["original_value"].notna() & ~out["extension_is_attack"]]
        # Three disjoint, externally calibrated profiles per split.
        profiles = [("external_calibrated_noise", "noise", q50), ("external_calibrated_offset", "offset", q90), ("external_calibrated_drift", "drift", q95)]
        used = set()
        for event_no, (name, kind, scale) in enumerate(profiles):
            segments = select_segments(part, 1, 12, rng)
            if not segments:
                continue
            seg = segments[0]
            if any(int(i) in used for i in seg):
                continue
            used.update(map(int, seg))
            sign = float(rng.choice([-1.0, 1.0]))
            original = out.loc[seg, "original_value"].to_numpy(dtype=float)
            if kind == "noise":
                draw = rng.choice(signed_diff, size=len(seg), replace=True)
                delta = np.clip(draw, -2.5, 2.5)
            elif kind == "offset":
                delta = np.full(len(seg), sign * max(float(scale), 0.5), dtype=float)
            else:
                delta = sign * np.linspace(max(float(q50), 0.25), max(float(scale), 0.75), len(seg))
            event_id = f"{split}_{name}_{event_no:03d}"
            out.loc[seg, "extension_observed_value"] = original + delta
            out.loc[seg, "extension_attack_type"] = name
            out.loc[seg, "extension_is_attack"] = True
            out.loc[seg, "extension_delta"] = delta
            out.loc[seg, "extension_event_id"] = event_id
            events.append({"event_id": event_id, "split": split, "attack_type": name, "rows": len(seg), "profile_scale_c": float(scale), "source": "NOAA ISD empirical temperature-difference distribution", "seed": 20260911})
    out_dir = ROOT / "tables"
    input_path = out_dir / "stage17_empirical_attack_profiles.parquet"
    out.to_parquet(input_path, index=False)
    (out_dir / "stage17_empirical_attack_events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [sys.executable, str(SRC / "stage09b_robust_fusion.py"), "--local", str(input_path), "--tag", "empirical"]
    result = subprocess.run(cmd, cwd=ROOT.parents[1], capture_output=True, text=True)
    result_path = out_dir / "stage09b_robust_fusion_empirical.json"
    if result.returncode != 0 or not result_path.exists():
        raise RuntimeError(f"stage09b empirical run failed: {result.stdout[-1000:]}\n{result.stderr[-1000:]}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["empirical_profile_quantiles_c"] = {"q50_abs_diff": float(q50), "q90_abs_diff": float(q90), "q95_abs_diff": float(q95)}
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    report = ROOT / "reports" / "stage17_empirical_attack_profiles.md"
    lines = ["# 阶段17外部经验分布攻击", "", "攻击幅度和噪声从 NOAA ISD 真实温度差分分布估计，再注入本地 valid/test；这些仍是受控攻击，不是真实入侵日志。", "", f"- NOAA |ΔT| 分位数：Q50=`{q50:.4f}`，Q90=`{q90:.4f}`，Q95=`{q95:.4f}` °C", f"- test F1/Recall/AUPRC：`{payload['local_test']['f1']:.4f}` / `{payload['local_test']['recall']:.4f}` / `{payload['local_test']['average_precision']:.4f}`", f"- test正常样本误伤率：`{payload['test_false_alarm_rate_on_clean_rows']:.4f}`", f"- 恢复后 MAE/RMSE：`{payload['test_recovery']['mae_after']:.4f}` / `{payload['test_recovery']['rmse_after']:.4f}`", "", "该实验用于检验固定人工幅度之外的经验分布敏感性，不能替代真实攻击数据。"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_PIPELINE_REVIEW_REQUIRED", "report": str(report), "test": payload["local_test"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
