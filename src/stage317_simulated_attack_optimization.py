from __future__ import annotations

"""Stage 317: exploratory simulation-first detector optimization.

This stage does not modify any frozen result.  It evaluates a pre-declared,
small expert-fusion grid on the DWD controlled-attack data.  The candidate is
designed to reduce cancellation in the equal-weight signed fusion used by
older stages: every channel is converted to non-negative evidence, then the
maximum component evidence, causal change evidence, and short persistence are
combined.  Valid labels are used only for selecting among the frozen grid and
for FAR calibration; test labels are evaluated only after selection.

Because the historical DWD test has already been explored, all outputs from
this stage are exploratory and cannot replace Stage246 or enter a confirmatory
main table without a new protocol/data/analysis freeze.
"""

import hashlib
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    detector_frame,
    fit_detectors,
    inject_attack,
    score_components,
)
from stage208_dwd_persistent_repeat_protection import (  # noqa: E402
    choose_split_attack_blocks,
    event_metrics,
    far_threshold,
)

NORMALIZED = TABLES / "stage206_dwd_normalized.parquet"
PROTOCOL = TABLES / "stage204_dwd_protocol_preregistration.json"
PREFIX = "stage317_simulated_attack_optimization"
FAR = 0.01

# Frozen before reading any test labels.  The grid is intentionally small and
# contains no dataset-specific or attack-specific hand tuning.
GRID = [
    {"mean": 0.50, "max": 0.25, "change": 0.15, "persist": 0.10, "window": 1},
    {"mean": 0.35, "max": 0.35, "change": 0.20, "persist": 0.10, "window": 1},
    {"mean": 0.25, "max": 0.35, "change": 0.25, "persist": 0.15, "window": 3},
    {"mean": 0.20, "max": 0.30, "change": 0.30, "persist": 0.20, "window": 3},
    {"mean": 0.15, "max": 0.35, "change": 0.30, "persist": 0.20, "window": 6},
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def positive_z(values: pd.Series, reference: pd.Series) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    if r.size == 0:
        return np.zeros(len(x), dtype=float)
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    scale = max(1.4826 * mad, float(np.quantile(np.abs(r - med), 0.99)), 1e-6)
    return np.nan_to_num(np.maximum((x - med) / scale, 0.0), nan=0.0, posinf=1e6)


def causal_change_evidence(frame: pd.DataFrame, train_normal: pd.Series) -> np.ndarray:
    """Causal station-local change evidence; no future values are used."""
    ordered = frame[["station_id", "timestamp_utc", "temperature_c"]].copy()
    ordered["_v"] = pd.to_numeric(ordered["temperature_c"], errors="coerce")
    ordered = ordered.sort_values(["station_id", "timestamp_utc"], kind="stable")
    out = pd.Series(0.0, index=frame.index, dtype=float)
    for _, group in ordered.groupby("station_id", sort=False):
        v = group["_v"]
        diff1 = (v - v.shift(1)).abs()
        diff2 = (v - 2.0 * v.shift(1) + v.shift(2)).abs()
        lag4 = (v - v.shift(4)).abs() / 4.0
        slope = (v - v.shift(6)).abs() / 6.0
        local = pd.concat([diff1, diff2, lag4, slope], axis=1).max(axis=1)
        ref = local.loc[group.index.intersection(train_normal[train_normal].index)]
        out.loc[group.index] = positive_z(local, ref)
    return out.to_numpy(dtype=float)


def persistence(evidence: np.ndarray, frame: pd.DataFrame, window: int) -> np.ndarray:
    temp = frame[["station_id", "timestamp_utc"]].copy()
    temp["_e"] = evidence
    temp = temp.sort_values(["station_id", "timestamp_utc"], kind="stable")
    if window <= 1:
        return temp["_e"].reindex(frame.index).fillna(0.0).to_numpy(dtype=float)
    rolled = temp.groupby("station_id", sort=False)["_e"].rolling(window, min_periods=1).max()
    rolled.index = rolled.index.droplevel(0)
    return rolled.reindex(frame.index).fillna(0.0).to_numpy(dtype=float)


def build_channels(base: pd.DataFrame, components: pd.DataFrame) -> dict[str, np.ndarray]:
    train_normal = base["split"].eq("train") & base["observation_present"]
    z = pd.DataFrame(index=base.index)
    for col in components.columns:
        z[col] = positive_z(components[col], components.loc[train_normal, col])
    matrix = z.to_numpy(dtype=float)
    mean_e = np.nanmean(matrix, axis=1)
    max_e = np.nanmax(matrix, axis=1)
    change_e = causal_change_evidence(base, train_normal)
    return {"mean": mean_e, "max": max_e, "change": change_e}


def score_candidate(channels: dict[str, np.ndarray], frame: pd.DataFrame, setting: dict) -> np.ndarray:
    raw = (
        float(setting["mean"]) * channels["mean"]
        + float(setting["max"]) * channels["max"]
        + float(setting["change"]) * channels["change"]
    )
    return (
        (1.0 - float(setting["persist"])) * raw
        + float(setting["persist"]) * persistence(raw, frame, int(setting["window"]))
    )


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "far": float(np.mean(pred[~y])) if (~y).any() else 0.0,
        "auprc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else 0.0,
    }


def objective(frame: pd.DataFrame, score: np.ndarray, threshold: float) -> tuple[float, float, float]:
    y = frame["is_attack"].to_numpy(dtype=bool)
    m = metrics(y, score, threshold)
    nonstuck = y & frame["attack_type"].astype(str).ne("stuck").to_numpy()
    nonstuck_frame = frame.loc[nonstuck].copy()
    nonstuck_event = 0.0
    if nonstuck.any():
        nonstuck_event = float(event_metrics(nonstuck_frame, (score >= threshold)[nonstuck]).get("event_coverage", 0.0))
    # Prefer non-stuck coverage while retaining overall F1 and FAR feasibility.
    value = 0.45 * m["f1"] + 0.35 * nonstuck_event + 0.20 * m["auprc"]
    return value, m["f1"], nonstuck_event


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frame = pd.read_parquet(NORMALIZED).copy()
    base = detector_frame(frame)
    detectors = fit_detectors(base, protocol)
    base_components = score_components(base, detectors)
    channels = build_channels(base, base_components)

    rows = []
    selected_rows = []
    for seed in protocol["seeds"]:
        valid_blocks = choose_split_attack_blocks(frame, protocol, int(seed), "valid")
        test_blocks = choose_split_attack_blocks(frame, protocol, int(seed), "test")
        valid = inject_attack(frame, valid_blocks, int(seed))
        test = inject_attack(frame, test_blocks, int(seed))
        # Recompute channels on each attacked frame; detector fitting remains train-normal only.
        valid_channels = build_channels(detector_frame(valid), score_components(detector_frame(valid), detectors))
        test_channels = build_channels(detector_frame(test), score_components(detector_frame(test), detectors))
        y_valid = valid["is_attack"].to_numpy(dtype=bool)
        y_test = test["is_attack"].to_numpy(dtype=bool)
        normal_valid = valid["split"].eq("valid") & ~valid["is_attack"]
        best = None
        for setting in GRID:
            valid_score = score_candidate(valid_channels, valid, setting)
            threshold = far_threshold(pd.Series(valid_score), normal_valid, FAR)
            value, f1, nonstuck_event = objective(valid, valid_score, threshold)
            cand = (value, f1, -threshold)
            if best is None or cand > best[0]:
                best = (cand, dict(setting), float(threshold), value, f1, nonstuck_event)
        _, setting, threshold, value, f1, nonstuck_event = best
        test_score = score_candidate(test_channels, test, setting)
        test_metrics = metrics(y_test, test_score, threshold)
        test_event = event_metrics(test, test_score >= threshold)
        rows.append({"seed": int(seed), "split": "test", "threshold": threshold, "selection_value": value, **setting, **test_metrics, "event_coverage": float(test_event.get("event_coverage", 0.0)), "event_row_recall": float(test_event.get("row_recall", 0.0))})
        selected_rows.append({"seed": int(seed), "threshold": threshold, "valid_selection_value": value, "valid_f1": f1, "valid_nonstuck_event": nonstuck_event, **setting})

    detail = pd.DataFrame(rows)
    selected = pd.DataFrame(selected_rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "stage": "stage317_simulated_attack_optimization",
        "protocol": "真实DWD观测背景上的透明模拟攻击；train-normal拟合，valid攻击选择有限冻结网格和FAR=0.01阈值，test只评价；历史test已被探索，因此本阶段仅为探索性，不替换Stage246。",
        "objective": "0.45 overall F1 + 0.35 non-stuck event coverage + 0.20 AUPRC on valid; FAR fixed at 0.01",
        "grid": GRID,
        "seed_count": int(len(detail)),
        "mean_test": {k: float(detail[k].mean()) for k in ["f1", "precision", "recall", "far", "auprc", "event_coverage", "event_row_recall"]},
        "limitations": [
            "历史DWD test已被探索，所有本阶段结果是探索性。",
            "模拟攻击不是真实生产攻击日志。",
            "本阶段不关闭P0-2/P0-3/P0-4/P0-6，也不证明全面领先。",
        ],
        "input_sha256": {"protocol": sha256_file(PROTOCOL), "normalized": sha256_file(NORMALIZED)},
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{PREFIX}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    detail.to_csv(TABLES / f"{PREFIX}_test_detail.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(TABLES / f"{PREFIX}_valid_selection.csv", index=False, encoding="utf-8-sig")
    report = [
        "# Stage317 模拟攻击算法优化（探索性）", "",
        "- 目标：减少多个异常证据平均时的抵消，加入非负证据、最大证据、因果变化和短期持续性。",
        "- 数据：DWD 真实观测背景 + 透明模拟攻击。",
        "- 主 FAR：0.01；valid 选择冻结网格，test 只评价。",
        "- 重要边界：历史 test 已被探索，本阶段不能升级为确认性主结果。", "",
        "## Test 五种子均值", "",
        json.dumps(payload["mean_test"], ensure_ascii=False, indent=2), "",
        "## 结果解释", "",
        "该候选仅表示可进一步锁定的算法方向；若后续新模拟攻击锁定测试中仍有任一主要基线稳定领先，则停止全面领先声称并报告互补性。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "mean_test": payload["mean_test"], "report": str(REPORTS / f"{PREFIX}.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
