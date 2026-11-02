from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PYTHON = Path(sys.executable)


WINDOWS = {
    "early": ("2022-07-14T00:00:00Z", "2022-07-19T06:20:00Z"),
    "mid": ("2022-07-16T00:00:00Z", "2022-07-21T00:00:00Z"),
    "late": ("2022-07-18T00:00:00Z", "2022-07-23T00:00:00Z"),
}


def remap_splits(source: pd.DataFrame, train_end: str, valid_end: str) -> pd.DataFrame:
    out = source.copy()
    ts = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    t0 = pd.Timestamp(train_end)
    t1 = pd.Timestamp(valid_end)
    out["split"] = "test"
    out.loc[ts < t0, "split"] = "train"
    out.loc[(ts >= t0) & (ts < t1), "split"] = "valid"
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="阶段27：滚动时间留出验证")
    parser.add_argument("--local", type=Path, default=TABLES / "stage05_controlled_attack_extensions.parquet")
    parser.add_argument("--era5", type=Path, default=TABLES / "stage06_era5_station_reference.parquet")
    args = parser.parse_args()
    source = pd.read_parquet(args.local)
    rows = []
    manifests = []
    for name, (train_end, valid_end) in WINDOWS.items():
        remapped = remap_splits(source, train_end, valid_end)
        input_path = TABLES / f"stage27_temporal_input_{name}.parquet"
        remapped.to_parquet(input_path, index=False)
        counts = remapped.groupby(["split", "station_key"], sort=True).size().to_dict()
        extension_train = int((remapped["extension_is_attack"].astype(bool) & remapped["split"].eq("train")).sum())
        if extension_train:
            raise RuntimeError(f"window {name} is invalid: {extension_train} extension attack rows entered train")
        manifests.append({"window": name, "train_end": train_end, "valid_end": valid_end, "input": str(input_path), "rows": int(len(remapped)), "split_counts": {f"{k[0]}::{k[1]}": int(v) for k, v in counts.items()}, "extension_attack_rows_in_train": extension_train})
        tag = f"temporal_{name}"
        cmd = [str(PYTHON), "-u", str(ROOT / "src" / "stage09b_robust_fusion.py"), "--local", str(input_path), "--era5", str(args.era5), "--tag", tag]
        completed = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"window {name} failed:\n{completed.stdout}\n{completed.stderr}")
        payload_path = TABLES / f"stage09b_robust_fusion_{tag}.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        m = payload["local_test"]
        rows.append({
            "window": name,
            "train_end": train_end,
            "valid_end": valid_end,
            "train_rows": int((remapped["split"] == "train").sum()),
            "valid_rows": int((remapped["split"] == "valid").sum()),
            "test_rows": int((remapped["split"] == "test").sum()),
            "test_attack_rows": m["attack_rows"],
            "test_f1": m["f1"],
            "test_recall": m["recall"],
            "test_precision": m["precision"],
            "test_auprc": m["average_precision"],
            "test_false_alarm_rate": payload["test_false_alarm_rate_on_clean_rows"],
            "recovery_mae_before": payload["test_recovery"]["mae_before"],
            "recovery_mae_after": payload["test_recovery"]["mae_after"],
            "threshold": payload["setting"]["threshold"],
        })
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "stage27_temporal_rolling_holdout.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "固定时间截止点滚动重划分；每个窗口仅用train拟合、valid选择设置和阈值、test只评价。扩展攻击若进入train则窗口作废。",
        "windows": manifests,
        "results": rows,
        "summary": {"mean_test_f1": float(result["test_f1"].mean()), "std_test_f1": float(result["test_f1"].std(ddof=1)), "mean_test_recall": float(result["test_recall"].mean()), "mean_test_auprc": float(result["test_auprc"].mean()), "mean_test_false_alarm_rate": float(result["test_false_alarm_rate"].mean()), "mean_recovery_mae_after": float(result["recovery_mae_after"].mean())},
    }
    (TABLES / "stage27_temporal_rolling_holdout.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段27滚动时间留出验证", "", "每个窗口重新按固定时间截止点划分 train/valid/test；train拟合、valid选设置、test只评价。", "", "| 窗口 | train结束 | valid结束 | test F1 | Recall | Precision | AUPRC | 误伤率 | 恢复MAE前→后 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in result.iterrows():
        lines.append(f"| {r['window']} | {r['train_end']} | {r['valid_end']} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_precision']:.4f} | {r['test_auprc']:.4f} | {r['test_false_alarm_rate']:.4f} | {r['recovery_mae_before']:.4f}→{r['recovery_mae_after']:.4f} |")
    s = payload["summary"]
    lines += ["", f"窗口均值±标准差 F1：`{s['mean_test_f1']:.4f} ± {s['std_test_f1']:.4f}`；平均 Recall/AUPRC/误伤率：`{s['mean_test_recall']:.4f}` / `{s['mean_test_auprc']:.4f}` / `{s['mean_test_false_alarm_rate']:.4f}`", "", "窗口清单与是否有扩展攻击进入train见 `stage27_temporal_rolling_holdout.json`。"]
    (REPORTS / "stage27_temporal_rolling_holdout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
