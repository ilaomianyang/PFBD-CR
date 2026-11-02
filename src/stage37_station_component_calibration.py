from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage09b_robust_fusion import best_robust_setting, smooth_risk
from stage09_trust_pipeline import _metrics
from trust_fusion import TrustFusion, robust_location_scale

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SCORES = TABLES / "stage09b_local_robust_scores.parquet"
COMPONENTS = ["Physical Rules", "Hampel MAD", "Kalman Innovation", "CUSUM", "Spatial Consistency", "ERA5 Background"]


def station_normalize(df: pd.DataFrame, columns: list[str], station_specific: bool) -> pd.DataFrame:
    out = df.copy()
    train_normal = df["split"].eq("train") & (~df["combined_is_attack"].astype(bool))
    global_stats = {c: robust_location_scale(df.loc[train_normal, c], floor=0.25) for c in columns}
    station_stats: dict[str, dict[str, tuple[float, float]]] = {}
    if station_specific:
        for station, g in df.loc[train_normal].groupby(df.loc[train_normal, "station_key"].astype(str), sort=False):
            if len(g) < 20:
                continue
            station_stats[str(station)] = {c: robust_location_scale(g[c], floor=0.25) for c in columns}
    for c in columns:
        values = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        result = np.zeros(len(df), dtype=float)
        stations = df["station_key"].astype(str).to_numpy()
        for i, station in enumerate(stations):
            med, scale = station_stats.get(station, {}).get(c, global_stats[c])
            result[i] = max(values[i] - med, 0.0) / scale if np.isfinite(values[i]) else 0.0
        out[c] = np.nan_to_num(result, nan=0.0, posinf=1e6)
    return out


def evaluate_variant(df: pd.DataFrame, mode: str) -> tuple[dict[str, object], pd.DataFrame]:
    cols = ["Hampel MAD"] if mode == "hampel_only" else COMPONENTS
    normalized = station_normalize(df, cols, station_specific=True)
    train, valid, test = [df["split"].eq(x) for x in ["train", "valid", "test"]]
    fusion = TrustFusion(include_era5=True).fit(normalized.loc[train], normalized.loc[valid], normalized.loc[valid, "combined_is_attack"])
    base = fusion.transform(normalized, extreme_background=df["era5_extreme_background"].astype(bool))
    setting = best_robust_setting(df.loc[valid], base[valid.to_numpy()], df.loc[valid, "structural_risk"].to_numpy(dtype=float))
    raw = setting["alpha_base"] * base + setting["alpha_structural"] * df["structural_risk"].to_numpy(dtype=float)
    rv = smooth_risk(df.loc[valid], raw[valid.to_numpy()], int(setting["window"]))
    rt = smooth_risk(df.loc[test], raw[test.to_numpy()], int(setting["window"]))
    threshold = float(setting["threshold"])
    yv = df.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    yt = df.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    mv = _metrics(yv, rv >= threshold, rv)
    mt = _metrics(yt, rt >= threshold, rt)
    test_frame = df.loc[test, ["station_key", "combined_is_attack"]].copy()
    test_frame["risk"] = rt
    test_frame["pred"] = rt >= threshold
    rows = []
    for station, g in test_frame.groupby(test_frame["station_key"].astype(str), sort=True):
        y = g["combined_is_attack"].astype(bool).to_numpy()
        p = g["pred"].astype(bool).to_numpy()
        m = _metrics(y, p, g["risk"].to_numpy(dtype=float))
        rows.append({"mode": mode, "station_key": station, "test_f1": m["f1"], "test_recall": m["recall"], "test_precision": m["precision"], "test_false_alarm_rate": float(p[~y].mean()) if (~y).any() else float("nan")})
    summary = {
        "mode": mode,
        "valid": mv,
        "test": mt,
        "test_false_alarm_rate": float((rt[~yt] >= threshold).mean()),
        "setting": setting,
        "fusion_weights": fusion.weights,
        "station_stats_count": int(normalized["station_key"].astype(str).nunique()),
    }
    return summary, pd.DataFrame(rows)


def main() -> int:
    df = pd.read_parquet(SCORES).copy()
    ref_json = json.loads((TABLES / "stage09b_robust_fusion.json").read_text(encoding="utf-8"))
    # Use the published 9B values verbatim as the locked reference.
    reference = {"mode": "global_reference", "valid": ref_json["local_valid"], "test": ref_json["local_test"], "test_false_alarm_rate": ref_json["test_false_alarm_rate_on_clean_rows"]}
    results = [reference]
    station_frames = []
    for mode in ["hampel_only", "all_components"]:
        summary, stations = evaluate_variant(df, mode)
        results.append(summary)
        station_frames.append(stations)
    result_rows = []
    for x in results:
        result_rows.append({"mode": x["mode"], "valid_f1": x["valid"].get("f1", float("nan")), "test_f1": x["test"]["f1"], "test_recall": x["test"]["recall"], "test_precision": x["test"]["precision"], "test_auprc": x["test"]["average_precision"], "test_false_alarm_rate": x["test_false_alarm_rate"]})
    result = pd.DataFrame(result_rows)
    selected = max([x for x in results if x["mode"] != "global_reference"], key=lambda x: (x["valid"]["f1"], -x["valid"].get("false_alarm_rate", 1.0)))
    ref_f1 = float(reference["test"]["f1"])
    payload = {
        "status": "CANDIDATE_GAIN_REQUIRES_MULTISEED" if selected["test"]["f1"] > ref_f1 and selected["test"]["average_precision"] >= reference["test"]["average_precision"] else "NEGATIVE_RESULT_RETAINED",
        "protocol": "检测器分量尺度只由train正常样本估计；valid选择融合设置；locked test只评价；9B主结果不替换。",
        "results": results,
        "selected_by_valid": selected,
        "reference_9b": reference,
    }
    result.to_csv(TABLES / "stage37_station_component_calibration.csv", index=False, encoding="utf-8-sig")
    pd.concat(station_frames, ignore_index=True).to_csv(TABLES / "stage37_station_component_calibration_by_station.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage37_station_component_calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段37站点自适应检测器分量尺度实验", "", "只用train正常样本估计站点尺度；valid选择；locked test评价。", "", "| 模式 | valid F1 | test F1 | test Recall | test Precision | test AUPRC | test FAR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in result.iterrows():
        lines.append(f"| {r['mode']} | {r['valid_f1']:.4f} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_precision']:.4f} | {r['test_auprc']:.4f} | {r['test_false_alarm_rate']:.4f} |")
    lines += ["", f"- 状态：**{payload['status']}**", "- 站点分解见 `tables/stage37_station_component_calibration_by_station.csv`。"]
    (REPORTS / "stage37_station_component_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected_mode": selected["mode"], "selected_test_f1": selected["test"]["f1"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
