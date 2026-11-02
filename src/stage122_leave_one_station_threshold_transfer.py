from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
Q_GRID = [0.90, 0.925, 0.95, 0.975, 0.99, 0.995]


def _metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    y = frame["combined_is_attack"].astype(bool).to_numpy()
    score = pd.to_numeric(frame["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
    pred = score >= float(threshold)
    clean = ~y
    tp = int((pred & y).sum())
    fp = int((pred & clean).sum())
    fn = int((~pred & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "far": float(pred[clean].mean()) if clean.any() else float("nan"),
    }


def _macro_f1(frame: pd.DataFrame, threshold: float, stations: list[str]) -> float:
    values = []
    for station in stations:
        part = frame.loc[frame["station_key"].astype(str).eq(station)]
        if len(part):
            values.append(_metrics(part, threshold)["f1"])
    return float(np.mean(values)) if values else 0.0


def _best_source_valid_threshold(frame: pd.DataFrame, stations: list[str], fallback: float) -> float:
    scores = pd.to_numeric(frame["robust_fused_risk"], errors="coerce").to_numpy(dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return float(fallback)
    # Match the canonical stage-09 threshold protocol: fixed quantile grid,
    # avoiding a data-size-dependent exhaustive scan of every score value.
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    best_threshold = float(candidates[0])
    best_key = (-1.0, -1.0, -best_threshold)
    for threshold in candidates:
        metrics = _metrics(frame, float(threshold))
        far = metrics["far"] if np.isfinite(metrics["far"]) else 1.0
        key = (_macro_f1(frame, float(threshold), stations), -far, -float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _quantile(values: pd.Series, q: float, fallback: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.quantile(q)) if len(clean) else float(fallback)


def main() -> int:
    candidate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        score_path = TABLES / f"stage09b_local_robust_scores_seed{seed}.parquet"
        result_path = TABLES / f"stage09b_robust_fusion_seed{seed}.json"
        if not score_path.is_file() or not result_path.is_file():
            raise FileNotFoundError(f"missing stage9C artifact for seed={seed}")
        frame = pd.read_parquet(score_path).copy()
        frame["station_key"] = frame["station_key"].astype(str)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        baseline_threshold = float(result["setting"]["threshold"])
        stations = sorted(frame["station_key"].dropna().unique().tolist())
        heldout_seed_rows: list[dict[str, float | str | int]] = []

        for heldout in stations:
            source_stations = [station for station in stations if station != heldout]
            source_train = frame.loc[frame["split"].eq("train") & frame["station_key"].isin(source_stations)].copy()
            source_valid = frame.loc[frame["split"].eq("valid") & frame["station_key"].isin(source_stations)].copy()
            target_train = frame.loc[frame["split"].eq("train") & frame["station_key"].eq(heldout)].copy()
            target_test = frame.loc[frame["split"].eq("test") & frame["station_key"].eq(heldout)].copy()
            source_normal = source_train.loc[~source_train["combined_is_attack"].astype(bool)]
            target_normal = target_train.loc[~target_train["combined_is_attack"].astype(bool)]
            fallback = float(pd.to_numeric(source_normal["robust_fused_risk"], errors="coerce").median()) if len(source_normal) else 0.0

            candidates: list[dict[str, float]] = []
            for q in Q_GRID:
                global_threshold = _quantile(source_normal["robust_fused_risk"], q, fallback)
                source_valid_macro_f1 = _macro_f1(source_valid, global_threshold, source_stations)
                source_valid_metrics = _metrics(source_valid, global_threshold)
                row = {
                    "seed": seed,
                    "heldout_station": heldout,
                    "q": q,
                    "global_threshold": global_threshold,
                    "source_valid_macro_f1": source_valid_macro_f1,
                    "source_valid_f1": source_valid_metrics["f1"],
                    "source_valid_recall": source_valid_metrics["recall"],
                    "source_valid_far": source_valid_metrics["far"],
                }
                candidate_rows.append(row)
                candidates.append(row)
            best = max(candidates, key=lambda row: (row["source_valid_macro_f1"], -row["source_valid_far"], row["q"]))
            q = float(best["q"])
            global_threshold = float(best["global_threshold"])
            target_threshold = _quantile(target_normal["robust_fused_risk"], q, global_threshold)
            baseline = _metrics(target_test, baseline_threshold)
            source_valid_threshold = _best_source_valid_threshold(source_valid, source_stations, global_threshold)
            source_baseline = _metrics(target_test, source_valid_threshold)
            strict = _metrics(target_test, global_threshold)
            adapted = _metrics(target_test, target_threshold)
            selected = {
                "seed": seed,
                "heldout_station": heldout,
                "source_station_count": len(source_stations),
                "source_train_normal_rows": len(source_normal),
                "target_train_normal_rows": len(target_normal),
                "target_test_rows": len(target_test),
                "selected_q": q,
                "source_valid_macro_f1": float(best["source_valid_macro_f1"]),
                "global_threshold": global_threshold,
                "target_adapted_threshold": target_threshold,
                "baseline_threshold": baseline_threshold,
                "source_valid_baseline_threshold": source_valid_threshold,
                "baseline_f1": baseline["f1"],
                "baseline_recall": baseline["recall"],
                "baseline_far": baseline["far"],
                "source_baseline_f1": source_baseline["f1"],
                "source_baseline_recall": source_baseline["recall"],
                "source_baseline_far": source_baseline["far"],
                "strict_f1": strict["f1"],
                "strict_recall": strict["recall"],
                "strict_far": strict["far"],
                "adapted_f1": adapted["f1"],
                "adapted_recall": adapted["recall"],
                "adapted_far": adapted["far"],
            }
            selected_rows.append(selected)
            station_rows.append(selected)
            heldout_seed_rows.append(selected)

        seed_rows.append({
            "seed": seed,
            "stations": len(stations),
            "baseline_f1": float(np.mean([row["baseline_f1"] for row in heldout_seed_rows])),
            "source_baseline_f1": float(np.mean([row["source_baseline_f1"] for row in heldout_seed_rows])),
            "strict_f1": float(np.mean([row["strict_f1"] for row in heldout_seed_rows])),
            "adapted_f1": float(np.mean([row["adapted_f1"] for row in heldout_seed_rows])),
            "baseline_recall": float(np.mean([row["baseline_recall"] for row in heldout_seed_rows])),
            "source_baseline_recall": float(np.mean([row["source_baseline_recall"] for row in heldout_seed_rows])),
            "strict_recall": float(np.mean([row["strict_recall"] for row in heldout_seed_rows])),
            "adapted_recall": float(np.mean([row["adapted_recall"] for row in heldout_seed_rows])),
            "baseline_far": float(np.mean([row["baseline_far"] for row in heldout_seed_rows])),
            "source_baseline_far": float(np.mean([row["source_baseline_far"] for row in heldout_seed_rows])),
            "strict_far": float(np.mean([row["strict_far"] for row in heldout_seed_rows])),
            "adapted_far": float(np.mean([row["adapted_far"] for row in heldout_seed_rows])),
        })

    selected_df = pd.DataFrame(selected_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    station_df = pd.DataFrame(station_rows)
    seed_df = pd.DataFrame(seed_rows)
    station_summary = station_df.groupby("heldout_station", as_index=False).agg(
        seeds=("seed", "nunique"),
        baseline_f1_mean=("baseline_f1", "mean"), source_baseline_f1_mean=("source_baseline_f1", "mean"), strict_f1_mean=("strict_f1", "mean"), adapted_f1_mean=("adapted_f1", "mean"),
        baseline_recall_mean=("baseline_recall", "mean"), source_baseline_recall_mean=("source_baseline_recall", "mean"), strict_recall_mean=("strict_recall", "mean"), adapted_recall_mean=("adapted_recall", "mean"),
        baseline_far_mean=("baseline_far", "mean"), source_baseline_far_mean=("source_baseline_far", "mean"), strict_far_mean=("strict_far", "mean"), adapted_far_mean=("adapted_far", "mean"),
    )
    comparison = {
        "baseline": {metric: float(selected_df[f"baseline_{metric}"].mean()) for metric in ["f1", "recall", "far"]},
        "source_valid_baseline": {metric: float(selected_df[f"source_baseline_{metric}"].mean()) for metric in ["f1", "recall", "far"]},
        "strict_leave_one_station": {metric: float(selected_df[f"strict_{metric}"].mean()) for metric in ["f1", "recall", "far"]},
        "target_history_adapted": {metric: float(selected_df[f"adapted_{metric}"].mean()) for metric in ["f1", "recall", "far"]},
    }
    gains = {
        "strict_f1_diff": comparison["strict_leave_one_station"]["f1"] - comparison["baseline"]["f1"],
        "strict_recall_diff": comparison["strict_leave_one_station"]["recall"] - comparison["baseline"]["recall"],
        "strict_far_diff": comparison["strict_leave_one_station"]["far"] - comparison["baseline"]["far"],
        "adapted_f1_diff": comparison["target_history_adapted"]["f1"] - comparison["baseline"]["f1"],
        "adapted_recall_diff": comparison["target_history_adapted"]["recall"] - comparison["baseline"]["recall"],
        "adapted_far_diff": comparison["target_history_adapted"]["far"] - comparison["baseline"]["far"],
    }
    gains_vs_source = {
        "strict_f1_diff": comparison["strict_leave_one_station"]["f1"] - comparison["source_valid_baseline"]["f1"],
        "strict_recall_diff": comparison["strict_leave_one_station"]["recall"] - comparison["source_valid_baseline"]["recall"],
        "strict_far_diff": comparison["strict_leave_one_station"]["far"] - comparison["source_valid_baseline"]["far"],
        "adapted_f1_diff": comparison["target_history_adapted"]["f1"] - comparison["source_valid_baseline"]["f1"],
        "adapted_recall_diff": comparison["target_history_adapted"]["recall"] - comparison["source_valid_baseline"]["recall"],
        "adapted_far_diff": comparison["target_history_adapted"]["far"] - comparison["source_valid_baseline"]["far"],
    }
    status = "CANDIDATE_TRANSFER_REQUIRES_REVIEW" if gains["strict_f1_diff"] > 0 or gains["adapted_f1_diff"] > 0 or gains_vs_source["strict_f1_diff"] > 0 or gains_vs_source["adapted_f1_diff"] > 0 else "NEGATIVE_RESULT_RETAINED"
    payload = {
        "status": status,
        "protocol": "对每个预先固定种子逐一留出一个站点；q仅由其余站点的train正常样本估计、由其余站点valid按站点宏F1选择；严格迁移只把源站点全局阈值用于留出站点test，目标站点test不参与选择。另报告允许使用目标站点train正常历史的无标签站点适配，但该适配与严格冷启动分开，不改变检测器、权重或攻击构造。",
        "seeds": SEEDS,
        "stations": sorted(selected_df["heldout_station"].unique().tolist()),
        "q_grid": Q_GRID,
        "candidate_count": len(candidate_df),
        "selected_count": len(selected_df),
        "comparison": comparison,
        "gains_vs_baseline": gains,
        "gains_vs_source_valid_baseline": gains_vs_source,
        "candidate_detail": selected_rows,
        "station_summary": station_summary.to_dict(orient="records"),
        "seed_summary": seed_rows,
        "limitations": [
            "阶段122只重评估阈值迁移，阶段9B检测器仍由全部本地站点拟合；因此不是完整的跨站点模型冷启动重训。",
            "严格迁移不使用留出站点train；目标历史适配明确使用留出站点train正常样本，不能与冷启动结果混写。",
            "攻击标签仍为阶段5受控注入，结果不等同于真实网络攻击泛化。",
        ],
    }
    candidate_df.to_csv(TABLES / "stage122_leave_one_station_threshold_transfer_candidates.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLES / "stage122_leave_one_station_threshold_transfer_selected.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(TABLES / "stage122_leave_one_station_threshold_transfer_station_summary.csv", index=False, encoding="utf-8-sig")
    seed_df.to_csv(TABLES / "stage122_leave_one_station_threshold_transfer_seed_summary.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage122_leave_one_station_threshold_transfer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# 阶段122留一站点阈值迁移验证", "",
        "q只由源站点train/valid选择；严格迁移不使用留出站点train，目标历史适配单独报告。", "",
        f"- 状态：**{status}**",
        f"- 源站点valid-only基线F1：`{comparison['source_valid_baseline']['f1']:.4f}`（避免使用留出站点valid标签）",
        f"- 严格留一站点F1：`{comparison['strict_leave_one_station']['f1']:.4f}`（全站点基线差值 `{gains['strict_f1_diff']:+.4f}`；源站点基线差值 `{gains_vs_source['strict_f1_diff']:+.4f}`）",
        f"- 严格留一站点Recall：`{comparison['strict_leave_one_station']['recall']:.4f}`（基线 `{comparison['baseline']['recall']:.4f}`，差值 `{gains['strict_recall_diff']:+.4f}`）",
        f"- 严格留一站点FAR：`{comparison['strict_leave_one_station']['far']:.4f}`（基线 `{comparison['baseline']['far']:.4f}`，差值 `{gains['strict_far_diff']:+.4f}`）",
        f"- 目标历史适配F1：`{comparison['target_history_adapted']['f1']:.4f}`（基线差值 `{gains['adapted_f1_diff']:+.4f}`）",
        "",
        "| 留出站点 | 种子数 | 全站点基线F1 | 源站点基线F1 | 严格迁移F1 | 历史适配F1 | 源站点基线Recall | 严格迁移Recall | 历史适配Recall | 严格迁移FAR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in station_summary.to_dict(orient="records"):
        lines.append(f"| {row['heldout_station']} | {row['seeds']} | {row['baseline_f1_mean']:.4f} | {row['source_baseline_f1_mean']:.4f} | {row['strict_f1_mean']:.4f} | {row['adapted_f1_mean']:.4f} | {row['source_baseline_recall_mean']:.4f} | {row['strict_recall_mean']:.4f} | {row['adapted_recall_mean']:.4f} | {row['strict_far_mean']:.4f} |")
    lines += ["", "阶段122不自动替换阶段9B或阶段121设置；完整跨站点模型冷启动仍需重新拟合检测器并保持同一攻击/划分协议。"]
    (REPORTS / "stage122_leave_one_station_threshold_transfer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "candidates": len(candidate_df), "selected": len(selected_df), "stations": len(stations)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
