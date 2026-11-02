from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
DEFAULT_DATA = Path(r"F:\数据集\Analysis\experiments\pc_mstf_ap_v4\parquet\v4_anomaly_injection_table.parquet")
CONFIRMATION_SEEDS = [123457, 234569, 345679, 456791, 567899]
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from stage05_attack_extensions import build_extensions  # noqa: E402
from stage139_expanded_fusion_grid import ALPHAS, WINDOWS, bootstrap_diff, evaluate_candidate  # noqa: E402


def ensure_default_run(seed: int, source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    input_path = TABLES / f"stage140_input_seed{seed}.parquet"
    tag = f"stage140_seed{seed}"
    score_path = TABLES / f"stage09b_local_robust_scores_{tag}.parquet"
    result_path = TABLES / f"stage09b_robust_fusion_{tag}.json"
    if not input_path.is_file():
        extended, _ = build_extensions(source, seed)
        extended.to_parquet(input_path, index=False)
    if not score_path.is_file() or not result_path.is_file():
        command = [sys.executable, str(SRC / "stage09b_robust_fusion.py"), "--local", str(input_path), "--noaa", str(TABLES / "stage06_noaa_isd_normalized.parquet"), "--era5", str(TABLES / "stage06_era5_station_reference.parquet"), "--tag", tag]
        completed = subprocess.run(command, cwd=ROOT.parents[1], capture_output=True, text=True)
        if completed.returncode != 0 or not score_path.is_file() or not result_path.is_file():
            raise RuntimeError(f"stage140 default run failed for seed {seed}:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")
    return pd.read_parquet(score_path), json.loads(result_path.read_text(encoding="utf-8"))


def main() -> int:
    source = pd.read_parquet(DEFAULT_DATA)
    candidate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    default_rows: list[dict[str, object]] = []
    for seed in CONFIRMATION_SEEDS:
        frame, default = ensure_default_run(seed, source)
        candidates: list[dict[str, object]] = []
        for alpha in ALPHAS:
            for window in WINDOWS:
                row = evaluate_candidate(frame, alpha, window)
                row["seed"] = seed
                candidate_rows.append(row)
                candidates.append(row)
        best = max(candidates, key=lambda row: (row["valid_f1"], -row["valid_false_alarm_rate"], -row["threshold"], row["window"], row["alpha_base"]))
        selected_rows.append({**best, "selection_policy": "stage139_frozen_expanded_grid_max_valid_f1"})
        default_rows.append({"seed": seed, "test_f1": float(default["local_test"]["f1"]), "test_recall": float(default["local_test"]["recall"]), "test_precision": float(default["local_test"]["precision"]), "test_auprc": float(default["local_test"]["average_precision"]), "test_false_alarm_rate": float(default["test_false_alarm_rate_on_clean_rows"])})
        print(json.dumps({"seed": seed, "status": "completed", "selected_alpha": best["alpha_base"], "selected_window": best["window"], "expanded_f1": best["test_f1"], "default_f1": default["local_test"]["f1"]}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(candidate_rows)
    selected = pd.DataFrame(selected_rows)
    default_df = pd.DataFrame(default_rows)
    paired = selected.merge(default_df, on="seed", suffixes=("_expanded", "_default"))
    for metric in METRICS:
        paired[f"diff_{metric}"] = paired[f"{metric}_expanded"] - paired[f"{metric}_default"]
    aggregate = pd.DataFrame([{
        "variant": "frozen_expanded_grid",
        "seed_count": len(selected),
        **{f"{metric}_mean": float(selected[metric].mean()) for metric in METRICS},
        **{f"{metric}_std": float(selected[metric].std(ddof=1)) for metric in METRICS},
    }, {
        "variant": "default_reference",
        "seed_count": len(default_df),
        **{f"{metric}_mean": float(default_df[metric].mean()) for metric in METRICS},
        **{f"{metric}_std": float(default_df[metric].std(ddof=1)) for metric in METRICS},
    }])
    metric_rows = []
    for index, metric in enumerate(METRICS):
        differences = paired[f"diff_{metric}"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_diff(differences, seed=140140 + index)
        metric_rows.append({"metric": metric, "expanded_minus_default_mean": float(differences.mean()), "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high, "expanded_count_better": int((differences > 0).sum()), "expanded_count_worse": int((differences < 0).sum())})
    metric_paired = pd.DataFrame(metric_rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段140在阶段139结果完成后冻结相同alpha/window网格和valid选择规则，并对五个此前未使用的确认种子仅运行一次；每个种子只用valid选择，locked test只评价，不根据test重新选择或扩展网格。",
        "seeds": CONFIRMATION_SEEDS,
        "alpha_grid": ALPHAS,
        "window_grid": WINDOWS,
        "candidate_count": len(detail),
        "selected_count": len(selected),
        "aggregate_count": len(aggregate),
        "paired_count": len(metric_paired),
        "aggregate": aggregate.to_dict(orient="records"),
        "paired": metric_paired.to_dict(orient="records"),
        "limitations": [
            "确认种子仍来自同一气象数据源和同一受控攻击生成器，不是外部真实攻击。",
            "五种子确认只能检验攻击构造随机性，不能证明跨地区或跨年份普适。",
            "即使F1提高，也必须同时报告FAR变化并在外部数据上继续复核。",
        ],
    }
    detail.to_csv(TABLES / "stage140_expanded_grid_independent_confirmation_detail.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(TABLES / "stage140_expanded_grid_independent_confirmation_selected.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / "stage140_expanded_grid_independent_confirmation_paired.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage140_expanded_grid_independent_confirmation_aggregate.csv", index=False, encoding="utf-8-sig")
    metric_paired.to_csv(TABLES / "stage140_expanded_grid_independent_confirmation_metric_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage140_expanded_grid_independent_confirmation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段140扩展网格第三组种子独立确认", "", "网格和选择规则在确认种子运行前冻结；每个种子只用valid选择。", "", "| 方案 | F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['variant']} | {row['test_f1_mean']:.4f} | {row['test_recall_mean']:.4f} | {row['test_precision_mean']:.4f} | {row['test_auprc_mean']:.4f} | {row['test_false_alarm_rate_mean']:.4f} |")
    lines += ["", "成对bootstrap见metric_paired表；确认成功也不等于外部真实攻击验证。"]
    (REPORTS / "stage140_expanded_grid_independent_confirmation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "candidate_count": len(detail), "selected_count": len(selected), "paired": metric_rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
