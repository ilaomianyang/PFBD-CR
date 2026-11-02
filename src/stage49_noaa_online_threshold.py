from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402
from stage09_trust_pipeline import _metrics  # noqa: E402

NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
QUANTILES = [0.99, 0.995, 0.999]
WINDOWS = [72, 240, 720]


def causal_predictions(frame: pd.DataFrame, quantile: float, window: int) -> np.ndarray:
    out = np.zeros(len(frame), dtype=bool)
    ordered = frame.assign(_row=np.arange(len(frame))).sort_values(["station_key", "timestamp_utc", "_row"], kind="stable")
    for _, group in ordered.groupby("station_key", sort=False):
        history = deque(maxlen=int(window))
        for idx, row in group.iterrows():
            risk = float(row["robust_fused_risk"])
            threshold = float(np.quantile(np.asarray(history, dtype=float), quantile)) if history else float("inf")
            out[int(idx)] = bool(np.isfinite(risk) and risk >= threshold)
            if np.isfinite(risk):
                history.append(risk)
    return out


def evaluate_seed(source: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    frame = build_dataset(source, seed)
    scored, fixed = score(frame)
    scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
    train = scored["split"].eq("train")
    valid = scored["split"].eq("valid")
    test = scored["split"].eq("test")
    yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    rows = []
    for q in QUANTILES:
        for window in WINDOWS:
            pred = causal_predictions(scored, q, window)
            pv = pred[valid.to_numpy()]
            pt = pred[test.to_numpy()]
            mv = _metrics(yv, pv, scored.loc[valid, "robust_fused_risk"].to_numpy(dtype=float))
            mt = _metrics(yt, pt, scored.loc[test, "robust_fused_risk"].to_numpy(dtype=float))
            rows.append({"seed": seed, "quantile": q, "window_rows": window, "valid_f1": mv["f1"], "valid_far": float(pv[~yv].mean()), "test_f1": mt["f1"], "test_recall": mt["recall"], "test_precision": mt["precision"], "test_auprc": mt["average_precision"], "test_false_alarm_rate": float(pt[~yt].mean()), "fixed_test_f1": fixed["test"]["f1"], "fixed_test_auprc": fixed["test"]["average_precision"], "fixed_test_false_alarm_rate": fixed["test_false_alarm_rate"]})
    return rows


def main() -> int:
    source = pd.read_parquet(NOAA)
    rows = []
    for seed in SEEDS:
        rows.extend(evaluate_seed(source, seed))
    detail = pd.DataFrame(rows)
    selected = []
    for seed, group in detail.groupby("seed", sort=True):
        selected.append(group.sort_values(["valid_f1", "valid_far"], ascending=[False, True]).iloc[0].to_dict())
    selected_df = pd.DataFrame(selected)
    fixed_f1 = float(selected_df["fixed_test_f1"].mean())
    online_f1 = float(selected_df["test_f1"].mean())
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_CROSS_SEASON_REVIEW" if online_f1 > fixed_f1 else "NEGATIVE_RESULT_RETAINED", "protocol": "NOAA 12站点受控外部测试；阈值使用每站点历史风险的因果滚动分位数，不使用当前行或任何攻击标签；q和窗口仅由valid选择；test只评价。", "seeds": SEEDS, "quantiles": QUANTILES, "windows": WINDOWS, "detail": rows, "selected_by_valid": selected, "comparison": {"online_test_f1_mean": online_f1, "fixed_test_f1_mean": fixed_f1}}
    detail.to_csv(TABLES / "stage49_noaa_online_threshold_detail.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage49_noaa_online_threshold_selected.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage49_noaa_online_threshold.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段49 NOAA因果滚动阈值实验", "", "阈值只使用每个站点过去的风险分布，q和窗口由valid选择，test只评价。", "", "| seed | q | 窗口行数 | test F1 | Recall | Precision | AUPRC | FAR | 固定阈值F1 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in selected:
        lines.append(f"| {int(r['seed'])} | {r['quantile']:.3f} | {int(r['window_rows'])} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_precision']:.4f} | {r['test_auprc']:.4f} | {r['test_false_alarm_rate']:.4f} | {r['fixed_test_f1']:.4f} |")
    lines += ["", f"- 状态：**{payload['status']}**", f"- 在线阈值test F1均值：`{online_f1:.4f}`；固定阈值：`{fixed_f1:.4f}`。", "- 该实验是无标签部署策略候选，不改变9B主模型结果。"]
    (REPORTS / "stage49_noaa_online_threshold.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "online_test_f1_mean": online_f1, "fixed_test_f1_mean": fixed_f1}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
