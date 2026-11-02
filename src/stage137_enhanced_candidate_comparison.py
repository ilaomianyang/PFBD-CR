from __future__ import annotations

import itertools
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
SEEDS = [271828, 314159, 161803, 8675309, 424242]
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate", "event_recall", "recovery_mae_after"]


def exact_permutation_p(old: np.ndarray, new: np.ndarray) -> float:
    pooled = np.concatenate([old, new])
    n_old = len(old)
    observed = float(new.mean() - old.mean())
    diffs = []
    for chosen in itertools.combinations(range(len(pooled)), n_old):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(chosen)] = True
        diffs.append(float(pooled[~mask].mean() - pooled[mask].mean()))
    return float(np.mean(np.abs(np.asarray(diffs)) >= abs(observed) - 1e-15))


def bootstrap_diff(old: np.ndarray, new: np.ndarray, seed: int, repeats: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    old_idx = rng.integers(0, len(old), size=(repeats, len(old)))
    new_idx = rng.integers(0, len(new), size=(repeats, len(new)))
    values = new[new_idx].mean(axis=1) - old[old_idx].mean(axis=1)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def run_variant(seed: int, enhanced: bool) -> dict[str, object]:
    input_path = TABLES / f"stage135_input_seed{seed}.parquet"
    tag = f"stage137_{'enhanced' if enhanced else 'default'}_seed{seed}"
    result_path = TABLES / f"stage09b_robust_fusion_{tag}.json"
    attack_path = TABLES / f"stage09b_attack_type_metrics_{tag}.csv"
    if not result_path.is_file() or not attack_path.is_file():
        command = [
            sys.executable,
            str(SRC / "stage09b_robust_fusion.py"),
            "--local",
            str(input_path),
            "--noaa",
            str(TABLES / "stage06_noaa_isd_normalized.parquet"),
            "--era5",
            str(TABLES / "stage06_era5_station_reference.parquet"),
            "--tag",
            tag,
        ]
        if enhanced:
            command.append("--enhanced")
        completed = subprocess.run(command, cwd=ROOT.parents[1], capture_output=True, text=True)
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(f"stage137 {tag} failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    test = payload["local_test"]
    return {
        "seed": seed,
        "test_f1": float(test["f1"]),
        "test_recall": float(test["recall"]),
        "test_precision": float(test["precision"]),
        "test_auprc": float(test["average_precision"]),
        "test_false_alarm_rate": float(payload["test_false_alarm_rate_on_clean_rows"]),
        "event_recall": float(payload["test_event_metrics"]["event_recall"]),
        "recovery_mae_after": float(payload["test_recovery"]["mae_after"]),
        "valid_f1": float(payload["setting"]["valid_f1"]),
        "valid_false_alarm_rate": float(payload["setting"]["valid_false_alarm_rate"]),
        "alpha_base": float(payload["setting"]["alpha_base"]),
        "alpha_structural": float(payload["setting"]["alpha_structural"]),
        "window": int(payload["setting"]["window"]),
        "threshold": float(payload["setting"]["threshold"]),
    }


def main() -> int:
    rows: list[dict[str, object]] = []
    attack_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        default = run_variant(seed, enhanced=False)
        enhanced = run_variant(seed, enhanced=True)
        row: dict[str, object] = {"seed": seed}
        for metric in METRICS:
            row[f"default_{metric}"] = default[metric]
            row[f"enhanced_{metric}"] = enhanced[metric]
            row[f"diff_{metric}"] = float(enhanced[metric]) - float(default[metric])
        row.update({"default_valid_f1": default["valid_f1"], "enhanced_valid_f1": enhanced["valid_f1"], "default_valid_far": default["valid_false_alarm_rate"], "enhanced_valid_far": enhanced["valid_false_alarm_rate"], "default_alpha_base": default["alpha_base"], "enhanced_alpha_base": enhanced["alpha_base"], "default_window": default["window"], "enhanced_window": enhanced["window"]})
        rows.append(row)
        for label, enabled in [("default", False), ("enhanced", True)]:
            tag = f"stage137_{'enhanced' if enabled else 'default'}_seed{seed}"
            frame = pd.read_csv(TABLES / f"stage09b_attack_type_metrics_{tag}.csv")
            frame.insert(0, "seed", seed)
            frame.insert(1, "variant", label)
            attack_frames.append(frame)
        print(json.dumps({"seed": seed, "status": "completed", "default_f1": default["test_f1"], "enhanced_f1": enhanced["test_f1"]}, ensure_ascii=False), flush=True)
    detail = pd.DataFrame(rows)
    attack_detail = pd.concat(attack_frames, ignore_index=True)
    attack_numeric = ["f1", "recall", "precision", "average_precision"]
    attack_aggregate = attack_detail.groupby(["variant", "split", "attack_type"], sort=True).agg(**{f"{col}_mean": (col, "mean") for col in attack_numeric}).reset_index()
    comparisons = []
    for index, metric in enumerate(METRICS):
        values = detail[f"diff_{metric}"].to_numpy(dtype=float)
        enhanced_values = detail[f"enhanced_{metric}"].to_numpy(dtype=float)
        default_values = detail[f"default_{metric}"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_diff(default_values, enhanced_values, seed=137137 + index)
        comparisons.append({"metric": metric, "default_mean": float(default_values.mean()), "enhanced_mean": float(enhanced_values.mean()), "observed_diff": float(values.mean()), "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high, "exact_permutation_p": exact_permutation_p(default_values, enhanced_values)})
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段137在阶段135五个新种子上比较预先存在的default结构与--enhanced结构化特征候选；两种方案各自在自身valid选择设置和阈值，locked test只做成对比较，不根据test重新调参。若候选未形成稳定增益则保留负结果。",
        "seed_count": len(SEEDS),
        "seeds": SEEDS,
        "result_count": len(detail),
        "attack_detail_rows": len(attack_detail),
        "attack_aggregate_rows": len(attack_aggregate),
        "comparisons": comparisons,
        "limitations": [
            "阶段137只比较一个预先存在的增强特征候选，不证明该候选是全局最优。",
            "五个新种子仍来自同一气象数据源和同一受控攻击生成器。",
            "成对test比较用于候选审计，不用于回写阶段9B主模型或锁定测试。",
        ],
    }
    detail.to_csv(TABLES / "stage137_enhanced_candidate_comparison_detail.csv", index=False, encoding="utf-8-sig")
    attack_detail.to_csv(TABLES / "stage137_enhanced_candidate_comparison_attack_detail.csv", index=False, encoding="utf-8-sig")
    attack_aggregate.to_csv(TABLES / "stage137_enhanced_candidate_comparison_attack_aggregate.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage137_enhanced_candidate_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段137 default 与 enhanced 候选比较", "", "两种方案均只由各自 valid 选择；test 只做成对评价。", "", "| 指标 | default | enhanced | 差值 | bootstrap 95% CI | 精确置换 p |", "|---|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['metric']} | {row['default_mean']:.4f} | {row['enhanced_mean']:.4f} | {row['observed_diff']:+.4f} | [{row['bootstrap_ci95_low']:+.4f}, {row['bootstrap_ci95_high']:+.4f}] | {row['exact_permutation_p']:.4f} |")
    lines += ["", "攻击类型分层明细见阶段137 CSV；若增强候选只改善单一类型或牺牲整体指标，不替换主模型。"]
    (REPORTS / "stage137_enhanced_candidate_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "result_count": len(detail), "attack_detail_rows": len(attack_detail), "attack_aggregate_rows": len(attack_aggregate), "comparisons": comparisons}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
