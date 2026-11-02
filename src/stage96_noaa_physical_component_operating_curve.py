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
NOAA = TABLES / "stage82_noaa_extended_normalized.parquet"
AUDIT = TABLES / "stage83_noaa_extended_quality_audit.json"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
METHODS = ["legacy", "component_zmax"]
Q_GRID = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _metrics  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from stage91_noaa_physical_component_repair import component_physical_scores  # noqa: E402
from stage92_noaa_physical_component_bootstrap import output_for_method  # noqa: E402


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    rows: list[dict[str, object]] = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        fold = f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, _ = score(frame)
            physical, _ = component_physical_scores(frame)
            train = scored["split"].eq("train").to_numpy()
            test = scored["split"].eq("test").to_numpy()
            yt = scored.loc[test, "combined_is_attack"].astype(bool).to_numpy()
            for method in METHODS:
                _, _, risk = output_for_method(scored, physical[method], method)
                finite_train = train & np.isfinite(risk)
                for q in Q_GRID:
                    threshold = float(np.nanquantile(risk[finite_train], q))
                    pred = risk[test] >= threshold
                    metrics = _metrics(yt, pred, risk[test])
                    far = float(pred[~yt].mean()) if (~yt).any() else float("nan")
                    rows.append({"fold": fold, "seed": seed, "method": method, "q": q, "threshold": threshold, "test_f1": float(metrics["f1"]), "test_recall": float(metrics["recall"]), "test_precision": float(metrics["precision"]), "test_auprc": float(metrics["average_precision"]), "test_far": far})
            print(json.dumps({"fold": fold, "seed": seed, "methods": len(METHODS), "q_values": len(Q_GRID)}, ensure_ascii=False), flush=True)
        print(json.dumps({"fold_completed": fold, "seeds": len(SEEDS)}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby(["method", "q"], sort=False)[["test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["method", "q"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[2:]]
    payload = {"status": "PASS_PIPELINE_REVIEW_REQUIRED", "protocol": "NOAA官方2020–2025扩展表；11站点；四折累计历史训练→前一年验证→目标年测试；5种子；legacy与component_zmax的融合alpha/window由每折valid选择，阈值仅取训练正常风险分位数q网格，不使用test标签。", "stations": stations, "folds": [f"{'-'.join(map(str, a))}->{b}->{c}" for a, b, c in FOLDS], "seeds": SEEDS, "q_grid": Q_GRID, "detail": rows, "aggregate": aggregate.to_dict(orient="records"), "limitations": ["运行曲线用于展示Recall/FAR权衡，不自动选择主模型。", "攻击为透明受控注入，不是真实攻击日志。", "component_zmax的低误报候选不能据此宣称主检测能力提升。"]}
    detail.to_csv(TABLES / "stage96_noaa_physical_component_operating_curve_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage96_noaa_physical_component_operating_curve.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage96_noaa_physical_component_operating_curve.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段96 NOAA物理组件运行曲线", "", "阈值仅由训练正常风险分位数确定，展示legacy与component_zmax的Recall/FAR权衡。", "", "| 方法 | q | test F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['q']:.3f} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_far_mean']:.4f}±{row['test_far_std']:.4f} |")
    lines += ["", "该曲线只用于确定可解释的安全运行点；不能用来挑选有利test结果后替换9B。"]
    (REPORTS / "stage96_noaa_physical_component_operating_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(detail), "q_values": len(Q_GRID)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
