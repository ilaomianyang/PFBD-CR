from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, best_robust_setting, structural_features, structural_risk_from_norm, smooth_risk  # noqa: E402
from stage09_trust_pipeline import _metrics  # noqa: E402
from trust_fusion import TrustFusion, robust_location_scale  # noqa: E402

TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
INPUT = TABLES / "stage38_noaa_controlled_scores.parquet"
COMPONENTS = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency"]


def modified_base(fusion: TrustFusion, frame: pd.DataFrame, cap: float) -> np.ndarray:
    total = np.zeros(len(frame), dtype=float)
    for name in fusion.components:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        med, scale = fusion.location_scale[name]
        z = np.nan_to_num(np.maximum(values - med, 0.0) / scale, nan=0.0, posinf=1e6)
        if name == "Physical Rules" and np.isfinite(cap):
            z = np.minimum(z, float(cap))
        total += fusion.weights.get(name, 0.0) * z
    return total


def main() -> int:
    df = pd.read_parquet(INPUT).copy()
    train, valid, test = [df["split"].eq(s) for s in ["train", "valid", "test"]]
    normal_train = train & (~df["combined_is_attack"].astype(bool))
    parts = []
    for s in ["train", "valid", "test"]:
        m = df["split"].eq(s)
        parts.append(structural_features(df.loc[m], STRUCTURAL_BASE).set_axis(df.index[m]))
    features = pd.concat(parts).reindex(df.index).fillna(0.0)
    features["station_key"] = df["station_key"].astype(str)
    scaler = AdaptiveScaler(STRUCTURAL_BASE).fit(features.loc[train], normal_train.loc[train])
    structural = structural_risk_from_norm(scaler.transform(features), enhanced=False)
    fusion = TrustFusion(include_era5=False).fit(df.loc[train], df.loc[valid], df.loc[valid, "combined_is_attack"])
    yv = df.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    yt = df.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    rows = []
    for cap in [float("inf"), 10.0, 5.0, 3.0, 2.0, 1.0, 0.0]:
        base = modified_base(fusion, df, cap)
        setting = best_robust_setting(df.loc[valid], base[valid.to_numpy()], structural[valid.to_numpy()])
        raw = setting["alpha_base"] * base + setting["alpha_structural"] * structural
        rv = smooth_risk(df.loc[valid], raw[valid.to_numpy()], int(setting["window"]))
        rt = smooth_risk(df.loc[test], raw[test.to_numpy()], int(setting["window"]))
        threshold = float(setting["threshold"])
        mv = _metrics(yv, rv >= threshold, rv)
        mt = _metrics(yt, rt >= threshold, rt)
        rows.append({"physical_rule_cap": None if not np.isfinite(cap) else cap, "valid_f1": mv["f1"], "test_f1": mt["f1"], "test_recall": mt["recall"], "test_precision": mt["precision"], "test_auprc": mt["average_precision"], "test_false_alarm_rate": float((rt[~yt] >= threshold).mean()), "setting": setting})
    selected = max(rows, key=lambda x: (x["valid_f1"], -x["test_false_alarm_rate"]))
    reference = rows[0]
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_MULTISEED" if selected["test_f1"] > reference["test_f1"] and selected["test_auprc"] >= reference["test_auprc"] else "NEGATIVE_RESULT_RETAINED", "protocol": "NOAA受控攻击外部测试；物理规则分量上限只在valid选择；test只评价。", "candidates": rows, "selected_by_valid": selected, "reference": reference}
    pd.DataFrame([{k: v for k, v in r.items() if k != "setting"} for r in rows]).to_csv(TABLES / "stage39_noaa_physical_rule_cap.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage39_noaa_physical_rule_cap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段39 NOAA物理规则分量上限实验", "", "只在valid选择物理规则分量上限；locked test评价。", "", "| 物理规则上限 | valid F1 | test F1 | test Recall | test Precision | test AUPRC | test FAR |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        cap = "∞" if r["physical_rule_cap"] is None else f"{r['physical_rule_cap']:.1f}"
        lines.append(f"| {cap} | {r['valid_f1']:.4f} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_precision']:.4f} | {r['test_auprc']:.4f} | {r['test_false_alarm_rate']:.4f} |")
    lines += ["", f"- 状态：**{payload['status']}**", "- 若无稳定增益，季节域偏移作为外部限制保留，不替换主模型。"]
    (REPORTS / "stage39_noaa_physical_rule_cap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected_cap": selected["physical_rule_cap"], "selected_test_f1": selected["test_f1"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
