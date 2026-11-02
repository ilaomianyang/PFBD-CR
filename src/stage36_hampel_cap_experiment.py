from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage09b_robust_fusion import best_robust_setting, smooth_risk
from stage09_trust_pipeline import _best_threshold, _metrics
from trust_fusion import TrustFusion

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SCORES = TABLES / "stage09b_local_robust_scores.parquet"


def capped_transform(fusion: TrustFusion, frame: pd.DataFrame, extreme: pd.Series, cap: float) -> np.ndarray:
    total = np.zeros(len(frame), dtype=float)
    for name in fusion.components:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        med, scale = fusion.location_scale[name]
        z = np.nan_to_num(np.maximum(values - med, 0.0) / scale, nan=0.0, posinf=1e6)
        if name == "Hampel MAD" and np.isfinite(cap):
            z = np.minimum(z, float(cap))
        total += fusion.weights.get(name, 0.0) * z
    missing = frame.get("missing_observation", pd.Series(False, index=frame.index)).astype(bool).to_numpy()
    total[missing] += 1.0
    if extreme is not None:
        total = np.where(extreme.to_numpy(dtype=bool), 0.75 * total, total)
    return total


def main() -> int:
    df = pd.read_parquet(SCORES).copy()
    train = df["split"].eq("train")
    valid = df["split"].eq("valid")
    test = df["split"].eq("test")
    yv = df.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    yt = df.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    fusion = TrustFusion(include_era5=True).fit(df.loc[train], df.loc[valid], df.loc[valid, "combined_is_attack"])
    extreme = df["era5_extreme_background"].astype(bool)
    rows = []
    candidates = [float("inf"), 8.0, 6.0, 4.0, 3.0, 2.0, 1.5]
    for cap in candidates:
        base = capped_transform(fusion, df, extreme, cap)
        setting = best_robust_setting(df.loc[valid], base[valid.to_numpy()], df.loc[valid, "structural_risk"].to_numpy(dtype=float))
        raw = setting["alpha_base"] * base + setting["alpha_structural"] * df["structural_risk"].to_numpy(dtype=float)
        rv = smooth_risk(df.loc[valid], raw[valid.to_numpy()], int(setting["window"]))
        rt = smooth_risk(df.loc[test], raw[test.to_numpy()], int(setting["window"]))
        threshold = float(setting["threshold"])
        mv = _metrics(yv, rv >= threshold, rv)
        mt = _metrics(yt, rt >= threshold, rt)
        test_frame = df.loc[test].copy()
        test_frame["pred"] = rt >= threshold
        test_frame["candidate_risk"] = rt
        clean = test_frame[~test_frame["combined_is_attack"].astype(bool)]
        liu = test_frame[test_frame["station_key"].astype(str).eq("liupanshan_meteo")]
        liu_clean = liu[~liu["combined_is_attack"].astype(bool)]
        rows.append({
            "hampel_cap": None if not np.isfinite(cap) else cap,
            "valid_f1": mv["f1"],
            "valid_false_alarm_rate": float((rv[~yv] >= threshold).mean()),
            "test_f1": mt["f1"],
            "test_recall": mt["recall"],
            "test_precision": mt["precision"],
            "test_auprc": mt["average_precision"],
            "test_false_alarm_rate": float((rt[~yt] >= threshold).mean()),
            "liupanshan_test_f1": _metrics(liu["combined_is_attack"].astype(bool), liu["pred"], liu["candidate_risk"])["f1"] if len(liu) else float("nan"),
            "liupanshan_false_alarm_rate": float(liu_clean["pred"].mean()) if len(liu_clean) else float("nan"),
            "setting": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in setting.items()},
        })
    result = pd.DataFrame([{k: v for k, v in row.items() if k != "setting"} for row in rows])
    selected = max(rows, key=lambda row: (row["valid_f1"], -row["valid_false_alarm_rate"], -(row["hampel_cap"] or 1e9)))
    payload = {
        "status": "CANDIDATE_GAIN_REQUIRES_MULTISEED" if selected["test_f1"] > rows[0]["test_f1"] and selected["test_auprc"] >= rows[0]["test_auprc"] else "NEGATIVE_RESULT_RETAINED",
        "protocol": "只在valid选择Hampel MAD分量上限、融合设置和阈值；locked test只评价；不改写stage09B主结果。",
        "candidates": rows,
        "selected_by_valid": selected,
        "reference_uncapped": rows[0],
    }
    result.to_csv(TABLES / "stage36_hampel_cap_candidates.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage36_hampel_cap_experiment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段36 Hampel MAD分量上限候选实验", "", "Hampel上限只由valid选择；test只评价。无增益时保留负结果，不替换9B主模型。", "", "| Hampel上限 | valid F1 | test F1 | test Recall | test Precision | test AUPRC | test FAR | liupanshan FAR |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        cap = "∞" if row["hampel_cap"] is None else f"{row['hampel_cap']:.1f}"
        lines.append(f"| {cap} | {row['valid_f1']:.4f} | {row['test_f1']:.4f} | {row['test_recall']:.4f} | {row['test_precision']:.4f} | {row['test_auprc']:.4f} | {row['test_false_alarm_rate']:.4f} | {row['liupanshan_false_alarm_rate']:.4f} |")
    lines += ["", f"- 状态：**{payload['status']}**", "- 只有在多种子和跨站点复核后，候选增益才可进入主模型讨论。"]
    (REPORTS / "stage36_hampel_cap_experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "candidate_count": len(rows), "selected_cap": selected["hampel_cap"], "selected_test_f1": selected["test_f1"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
