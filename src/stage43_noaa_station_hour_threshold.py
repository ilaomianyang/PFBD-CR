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

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402

TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEED = 20260912


def thresholds(train: pd.DataFrame, q: float) -> tuple[pd.Series, pd.Series, float]:
    normal = train[~train["combined_is_attack"].astype(bool)]
    station = normal.groupby(normal["station_key"].astype(str))["robust_fused_risk"].quantile(q)
    hour = normal.groupby(normal["timestamp_utc"].dt.hour)["robust_fused_risk"].quantile(q)
    global_t = float(normal["robust_fused_risk"].quantile(q))
    return station, hour, global_t


def apply(frame: pd.DataFrame, station: pd.Series, hour: pd.Series, global_t: float, lam: float) -> np.ndarray:
    s = frame["station_key"].astype(str).map(station).fillna(global_t).to_numpy(dtype=float)
    h = frame["timestamp_utc"].dt.hour.map(hour).fillna(global_t).to_numpy(dtype=float)
    return frame["robust_fused_risk"].to_numpy(dtype=float) >= ((1.0 - lam) * s + lam * h)


def main() -> int:
    source = pd.read_parquet(NOAA)
    frame = build_dataset(source, SEED)
    scored, reference = score(frame)
    scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
    train, valid, test = [scored["split"].eq(s) for s in ["train", "valid", "test"]]
    yv = scored.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    rows = []
    for q in [0.95, 0.975, 0.99, 0.995]:
        station, hour, global_t = thresholds(scored.loc[train], q)
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pv = apply(scored.loc[valid], station, hour, global_t, lam)
            pt = apply(scored.loc[test], station, hour, global_t, lam)
            mv = _metrics(yv, pv, scored.loc[valid, "robust_fused_risk"])
            mt = _metrics(yt, pt, scored.loc[test, "robust_fused_risk"])
            rows.append({"q": q, "hour_weight": lam, "valid_f1": mv["f1"], "valid_far": float(pv[~yv].mean()), "test_f1": mt["f1"], "test_recall": mt["recall"], "test_precision": mt["precision"], "test_auprc": mt["average_precision"], "test_false_alarm_rate": float(pt[~yt].mean())})
    candidate = max(rows, key=lambda x: (x["valid_f1"], -x["valid_far"]))
    payload = {"status": "CANDIDATE_GAIN_REQUIRES_MULTISEED" if candidate["test_f1"] > reference["test"]["f1"] and candidate["test_auprc"] >= reference["test"]["average_precision"] else "NEGATIVE_RESULT_RETAINED", "protocol": "NOAA受控外部测试；阈值仅由train正常样本按站点×UTC小时估计，valid选择q与收缩比例，test只评价。", "reference": {"test": reference["test"], "test_false_alarm_rate": reference["test_false_alarm_rate"]}, "candidates": rows, "selected_by_valid": candidate}
    pd.DataFrame(rows).to_csv(TABLES / "stage43_noaa_station_hour_threshold.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage43_noaa_station_hour_threshold.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段43 NOAA站点×小时阈值候选", "", "只用train正常样本估计阈值，valid选择q和小时收缩比例，test只评价。", "", "| q | 小时权重 | valid F1 | test F1 | Recall | Precision | AUPRC | FAR |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['q']:.3f} | {r['hour_weight']:.2f} | {r['valid_f1']:.4f} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_precision']:.4f} | {r['test_auprc']:.4f} | {r['test_false_alarm_rate']:.4f} |")
    lines += ["", f"- 状态：**{payload['status']}**", f"- 选中候选：q={candidate['q']:.3f}，小时权重={candidate['hour_weight']:.2f}。"]
    (REPORTS / "stage43_noaa_station_hour_threshold.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected": [candidate["q"], candidate["hour_weight"]], "test_f1": candidate["test_f1"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
