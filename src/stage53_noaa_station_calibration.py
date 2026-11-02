from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage38_noaa_controlled_benchmark import build_dataset, score  # noqa: E402


def station_stats(frame: pd.DataFrame, values: np.ndarray, groups: pd.Series, normal_train: np.ndarray) -> dict[str, tuple[float, float, np.ndarray]]:
    stats: dict[str, tuple[float, float, np.ndarray]] = {}
    for key, idx in groups.groupby(groups, sort=True).groups.items():
        mask = normal_train & groups.astype(str).eq(str(key)).to_numpy()
        sample = values[mask]
        sample = sample[np.isfinite(sample)]
        if len(sample) < 8:
            continue
        med = float(np.median(sample))
        mad = float(max(1.4826 * np.median(np.abs(sample - med)), 0.05))
        stats[str(key)] = (med, mad, np.sort(sample))
    return stats


def calibrate(frame: pd.DataFrame, risk: np.ndarray, mode: str) -> np.ndarray:
    train = frame["split"].eq("train").to_numpy()
    normal_train = train & (~frame["combined_is_attack"].astype(bool).to_numpy())
    stations = frame["station_key"].astype(str)
    hours = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce").dt.hour.fillna(-1).astype(int)
    out = np.full(len(frame), np.nan, dtype=float)
    station = station_stats(frame, risk, stations, normal_train)
    if mode == "raw":
        return risk.astype(float)
    if mode == "station_mad":
        for key, (med, mad, _) in station.items():
            mask = stations.eq(key).to_numpy()
            out[mask] = np.maximum((risk[mask] - med) / mad, 0.0)
        return out
    if mode == "station_quantile":
        for key, (_, _, ordered) in station.items():
            mask = stations.eq(key).to_numpy()
            out[mask] = np.searchsorted(ordered, risk[mask], side="right") / float(len(ordered))
        return out
    if mode != "station_hour_mad":
        raise ValueError(f"unknown calibration mode: {mode}")
    # Hourly calibration uses only normal train rows and falls back to station
    # statistics when a station-hour cell has too few observations.
    cells: dict[tuple[str, int], tuple[float, float]] = {}
    for (station_key, hour), idx in frame.loc[normal_train].groupby([stations[normal_train], hours[normal_train]], sort=True).groups.items():
        sample = risk[np.asarray(idx, dtype=int)]
        sample = sample[np.isfinite(sample)]
        if len(sample) >= 8:
            med = float(np.median(sample))
            mad = float(max(1.4826 * np.median(np.abs(sample - med)), 0.05))
            cells[(str(station_key), int(hour))] = (med, mad)
    for i, (station_key, hour) in enumerate(zip(stations, hours)):
        key = (str(station_key), int(hour))
        med, mad, _ = station.get(str(station_key), (0.0, 1.0, np.array([], dtype=float)))
        if key in cells:
            med, mad = cells[key]
        out[i] = max((risk[i] - med) / mad, 0.0) if np.isfinite(risk[i]) else np.nan
    return out


def metrics(y: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    normal = ~y
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "auprc": float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
        "far": float(pred[normal].mean()) if normal.any() else float("nan"),
    }


def select_threshold(y_valid: np.ndarray, valid_scores: np.ndarray, operating_point: str) -> tuple[float, dict[str, float]]:
    finite = np.isfinite(valid_scores)
    y_valid = y_valid[finite]
    valid_scores = valid_scores[finite]
    candidates = np.unique(np.quantile(valid_scores, np.linspace(0.01, 0.999, 401)))
    choices: list[tuple[float, float, float]] = []
    for threshold in candidates:
        pred = valid_scores >= threshold
        far = float(pred[~y_valid].mean()) if (~y_valid).any() else 1.0
        f1 = float(f1_score(y_valid, pred, zero_division=0))
        choices.append((f1, -far, float(threshold)))
    if operating_point == "f1":
        best = max(choices)
    elif operating_point == "far1":
        feasible = [x for x in choices if -x[1] <= 0.01]
        best = max(feasible, key=lambda x: (x[0], x[1])) if feasible else max(choices, key=lambda x: (x[1], x[0]))
    else:
        raise ValueError(f"unknown operating point: {operating_point}")
    threshold = best[2]
    valid_pred = valid_scores >= threshold
    return threshold, metrics(y_valid, valid_scores, valid_pred)


def main() -> int:
    source = pd.read_parquet(NOAA)
    rows: list[dict[str, object]] = []
    modes = ["raw", "station_mad", "station_quantile", "station_hour_mad"]
    for seed in SEEDS:
        frame = build_dataset(source, seed)
        scored, _ = score(frame)
        risk = scored["robust_fused_risk"].to_numpy(dtype=float)
        y = scored["combined_is_attack"].astype(bool).to_numpy()
        valid = scored["split"].eq("valid").to_numpy()
        test = scored["split"].eq("test").to_numpy()
        for mode in modes:
            calibrated = calibrate(scored, risk, mode)
            for operating_point in ["f1", "far1"]:
                threshold, valid_metrics = select_threshold(y[valid], calibrated[valid], operating_point)
                test_scores = calibrated[test]
                test_pred = np.isfinite(test_scores) & (test_scores >= threshold)
                test_metrics = metrics(y[test], np.nan_to_num(test_scores, nan=-np.inf), test_pred)
                rows.append({
                    "seed": seed,
                    "mode": mode,
                    "operating_point": operating_point,
                    "threshold": threshold,
                    "valid_f1": valid_metrics["f1"],
                    "valid_far": valid_metrics["far"],
                    "test_f1": test_metrics["f1"],
                    "test_recall": test_metrics["recall"],
                    "test_precision": test_metrics["precision"],
                    "test_auprc": test_metrics["auprc"],
                    "test_far": test_metrics["far"],
                })
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby(["mode", "operating_point"], sort=False)[["valid_f1", "valid_far", "test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["mode", "operating_point"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[2:]]
    best = aggregate.sort_values(["operating_point", "test_f1_mean"], ascending=[True, False]).groupby("operating_point", as_index=False).head(1)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "NOAA扩展12站点；站点校准只使用每个站点train正常样本；阈值只在valid选择；test只评价；不修改9B主模型或锁定测试集。",
        "seeds": SEEDS,
        "modes": modes,
        "detail": rows,
        "aggregate": aggregate.to_dict(orient="records"),
        "best_by_operating_point": best.to_dict(orient="records"),
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "stage53_noaa_station_calibration_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage53_noaa_station_calibration.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage53_noaa_station_calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段53 NOAA站点校准实验",
        "",
        "站点校准只使用train正常样本，阈值只在valid选择，test只评价；该实验不改变9B主模型。",
        "",
        "| 校准模式 | 运行点 | test F1 | Recall | Precision | AUPRC | FAR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['mode']} | {row['operating_point']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} |")
    lines += ["", "该结果用于判断站点异质性校准是否值得进入后续多季节验证；任何改进都必须在独立季节和新增站点上复核。"]
    (REPORTS / "stage53_noaa_station_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "best": payload["best_by_operating_point"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
