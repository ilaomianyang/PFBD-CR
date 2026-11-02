from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
NOAA = TABLES / "stage06_noaa_isd_normalized.parquet"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import COMPLETE_TYPES, score  # noqa: E402
from stage50_noaa_rolling_seasonal_review import FOLDS, build_fold  # noqa: E402
from stage61_noaa_hierarchical_score_calibration import calibrated_score, fit_hierarchical  # noqa: E402


def main() -> int:
    source = pd.read_parquet(NOAA)
    prior = json.loads((TABLES / "stage62_noaa_hierarchical_rolling_review.json").read_text(encoding="utf-8"))
    selected = {(str(row["fold"]), int(row["seed"])): row for row in prior["selected_by_valid"]}
    rows: list[dict[str, object]] = []
    for fold, bounds in FOLDS.items():
        for seed in SEEDS:
            frame = build_fold(source, seed, bounds)
            scored, _ = score(frame)
            scored["timestamp_utc"] = pd.to_datetime(scored["timestamp_utc"], utc=True, errors="coerce")
            key = (fold, seed)
            if key not in selected:
                raise KeyError(f"missing stage62 selection for {key}")
            choice = selected[key]
            calibration = fit_hierarchical(scored.loc[scored["split"].eq("train")], float(choice["shrink_k"]))
            calibrated = calibrated_score(scored, calibration, float(choice["hour_weight"]))
            test = scored["split"].eq("test").to_numpy()
            threshold = float(choice["threshold"])
            pred = calibrated >= threshold
            fixed = scored["predicted_attack"].astype(bool).to_numpy()
            y = scored["combined_is_attack"].astype(bool).to_numpy()
            normal = test & ~y
            for attack_type in COMPLETE_TYPES:
                mask = test & scored["combined_attack_type"].eq(attack_type).to_numpy()
                if not mask.any():
                    continue
                rows.append({
                    "fold": fold,
                    "seed": seed,
                    "attack_type": attack_type,
                    "rows": int(mask.sum()),
                    "hierarchical_detected_rows": int(pred[mask].sum()),
                    "hierarchical_recall": float(pred[mask].mean()),
                    "fixed_detected_rows": int(fixed[mask].sum()),
                    "fixed_recall": float(fixed[mask].mean()),
                    "recall_gain_vs_fixed": float(pred[mask].mean() - fixed[mask].mean()),
                    "hierarchical_far": float(pred[normal].mean()) if normal.any() else float("nan"),
                    "fixed_far": float(fixed[normal].mean()) if normal.any() else float("nan"),
                })
            print(json.dumps({"fold": fold, "seed": seed, "rows": len(rows)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("attack_type", sort=False)[["hierarchical_recall", "fixed_recall", "recall_gain_vs_fixed", "hierarchical_far", "fixed_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["attack_type"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "使用阶段62每折每种子由valid选出的层级校准参数；不重新使用test标签；仅统计test攻击类型Recall和正常FAR。",
        "folds": FOLDS,
        "seeds": SEEDS,
        "detail": rows,
        "aggregate_by_attack_type": aggregate.to_dict(orient="records"),
        "interpretation": "该表是诊断性分层证据，不能替代总体F1/AUPRC，也不能删除弱攻击类型。",
    }
    detail.to_csv(TABLES / "stage63_noaa_hierarchical_attack_type_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage63_noaa_hierarchical_attack_type_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage63_noaa_hierarchical_attack_type.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段63 NOAA层级校准攻击类型分层", "", "使用阶段62由valid选出的参数，不重新使用test标签；按攻击类型统计测试Recall，并报告同一测试集正常FAR。", "", "| 攻击类型 | 层级Recall | 固定Recall | Recall差值 | 层级FAR | 固定FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['attack_type']} | {row['hierarchical_recall_mean']:.4f}±{row['hierarchical_recall_std']:.4f} | {row['fixed_recall_mean']:.4f}±{row['fixed_recall_std']:.4f} | {row['recall_gain_vs_fixed_mean']:.4f} | {row['hierarchical_far_mean']:.4f} | {row['fixed_far_mean']:.4f} |")
    lines += ["", "这是诊断性分层结果，必须和总体F1、AUPRC、FAR一起报告；不得通过删除低Recall攻击类型改善结论。"]
    (REPORTS / "stage63_noaa_hierarchical_attack_type.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "attack_types": len(aggregate), "rows": len(detail)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
