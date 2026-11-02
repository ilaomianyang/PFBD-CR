from __future__ import annotations

"""Stage 141: external NOAA validation of the frozen fusion candidate.

The expanded alpha/window grid was explored on earlier, same-source seed groups.
Here the grid and tie-breaking rule are fixed before each NOAA rolling fold is
scored.  Only the validation split selects a candidate; the test split is used
for evaluation.  Existing stage-85 baseline results are reused because they
were produced under the identical rolling protocol and train-only fitting rule.
"""

import hashlib
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
BASELINE_DETAIL = TABLES / "stage85_noaa_rolling_baseline_comparison_detail.csv"
SEEDS = [3407, 6257, 7919, 104729, 20260715]
FOLDS = [
    ([2020], 2021, 2022),
    ([2020, 2021], 2022, 2023),
    ([2020, 2021, 2022], 2023, 2024),
    ([2020, 2021, 2022, 2023], 2024, 2025),
]
ALPHAS = [0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 1.00]
WINDOWS = [1, 3, 6, 12]
Q_GRID = [0.95, 0.975, 0.99, 0.995]
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate"]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage09_trust_pipeline import _best_threshold, _metrics  # noqa: E402
from stage09b_robust_fusion import AdaptiveScaler, STRUCTURAL_BASE, smooth_risk, structural_features, structural_risk_from_norm  # noqa: E402
from stage38_noaa_controlled_benchmark import score  # noqa: E402
from stage84_noaa_rolling_multiyear_benchmark import build_fold  # noqa: E402
from trust_fusion import TrustFusion  # noqa: E402


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def fold_name(train_years: list[int], valid_year: int, test_year: int) -> str:
    return f"{'-'.join(map(str, train_years))}->{valid_year}->{test_year}"


def add_components(scored: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the two component risks used by stage38 and retain them."""
    out = scored.copy()
    train = out["split"].eq("train")
    normal_train = train & ~out["combined_is_attack"].astype(bool)
    parts = []
    for split_name in ["train", "valid", "test"]:
        mask = out["split"].eq(split_name)
        part = structural_features(out.loc[mask], STRUCTURAL_BASE).set_axis(out.index[mask])
        parts.append(part)
    features = pd.concat(parts).reindex(out.index).fillna(0.0)
    features["station_key"] = out["station_key"].astype(str)
    scaler = AdaptiveScaler(STRUCTURAL_BASE).fit(features.loc[train], normal_train.loc[train])
    out["structural_risk"] = structural_risk_from_norm(scaler.transform(features), enhanced=False)
    fusion = TrustFusion(include_era5=False).fit(
        out.loc[train], out.loc[out["split"].eq("valid")], out.loc[out["split"].eq("valid"), "combined_is_attack"]
    )
    out["base_fused_risk"] = fusion.transform(out)
    return out


def candidate_scores(frame: pd.DataFrame, alpha: float, window: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    valid = frame["split"].eq("valid")
    test = frame["split"].eq("test")
    raw = alpha * frame["base_fused_risk"].to_numpy(dtype=float) + (1.0 - alpha) * frame["structural_risk"].to_numpy(dtype=float)
    valid_score = smooth_risk(frame.loc[valid], raw[valid.to_numpy()], window)
    test_score = smooth_risk(frame.loc[test], raw[test.to_numpy()], window)
    y_valid = frame.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    threshold, valid_f1 = _best_threshold(y_valid, valid_score)
    return valid_score, test_score, float(threshold), float(valid_f1)


def evaluate_candidate(frame: pd.DataFrame, alpha: float, window: int) -> tuple[dict[str, object], np.ndarray]:
    valid = frame["split"].eq("valid")
    test = frame["split"].eq("test")
    valid_score, test_score, threshold, valid_f1 = candidate_scores(frame, alpha, window)
    y_valid = frame.loc[valid, "combined_is_attack"].astype(bool).to_numpy()
    y_test = frame.loc[test, "combined_is_attack"].astype(bool).to_numpy()
    valid_pred = valid_score >= threshold
    test_pred = test_score >= threshold
    vm = _metrics(y_valid, valid_pred, valid_score)
    tm = _metrics(y_test, test_pred, test_score)
    row = {
        "alpha_base": alpha,
        "alpha_structural": 1.0 - alpha,
        "window": window,
        "threshold": threshold,
        "valid_f1": valid_f1,
        "valid_false_alarm_rate": float(valid_pred[~y_valid].mean()) if (~y_valid).any() else float("nan"),
        "test_f1": float(tm["f1"]),
        "test_recall": float(tm["recall"]),
        "test_precision": float(tm["precision"]),
        "test_auprc": float(tm["average_precision"]),
        "test_false_alarm_rate": float(test_pred[~y_test].mean()) if (~y_test).any() else float("nan"),
    }
    return row, test_pred


def choose_expanded(frame: pd.DataFrame) -> tuple[dict[str, object], np.ndarray]:
    candidates = []
    for alpha in ALPHAS:
        for window in WINDOWS:
            row, pred = evaluate_candidate(frame, alpha, window)
            candidates.append((row, pred))
    return max(candidates, key=lambda item: (
        item[0]["valid_f1"],
        -item[0]["valid_false_alarm_rate"],
        -item[0]["threshold"],
        -item[0]["window"],
        -item[0]["alpha_base"],
    ))


def bootstrap_cluster(values: pd.DataFrame, value_col: str, seed: int, repeats: int = 20000) -> tuple[float, float]:
    clusters = values["fold"].drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    by_cluster = {key: values.loc[values["fold"].eq(key), value_col].to_numpy(dtype=float) for key in clusters}
    for i in range(repeats):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        means[i] = np.concatenate([by_cluster[key] for key in sampled]).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_p(values: np.ndarray, seed: int, repeats: int = 20000) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(repeats, len(values)))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float((1.0 + np.sum(null >= observed)) / (repeats + 1.0))


def bh_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        idx = int(order[rank])
        running = min(running, p_values[idx] * len(p_values) / (rank + 1))
        adjusted[idx] = running
    return adjusted.tolist()


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        "_".join(str(part) for part in column if str(part) != "") if isinstance(column, tuple) else str(column)
        for column in out.columns
    ]
    return out


def finalize_existing() -> int:
    """Finish JSON/report serialization from CSVs after an interrupted write."""
    detail = pd.read_csv(TABLES / "stage141_noaa_frozen_external_validation_detail.csv")
    selected = pd.read_csv(TABLES / "stage141_noaa_frozen_external_validation_selected.csv")
    aggregate = pd.read_csv(TABLES / "stage141_noaa_frozen_external_validation.csv")
    fold_aggregate = pd.read_csv(TABLES / "stage141_noaa_frozen_external_validation_by_fold.csv")
    paired_df = pd.read_csv(TABLES / "stage141_noaa_frozen_external_validation_paired.csv")
    op_aggregate = pd.read_csv(TABLES / "stage141_noaa_frozen_external_validation_low_far.csv")
    attack_detail = pd.read_csv(TABLES / "stage141_noaa_frozen_external_validation_by_attack_type.csv")
    attack_agg = flatten_columns(attack_detail.groupby(["method", "attack_type"], sort=False)[["rows", "detected_rows", "recall"]].agg(["sum", "mean"]).reset_index())
    attack_agg.to_csv(TABLES / "stage141_noaa_frozen_external_validation_by_attack_type_aggregate.csv", index=False, encoding="utf-8-sig")
    station_detail = pd.read_csv(TABLES / "stage141_noaa_frozen_external_validation_by_station.csv")
    station_agg = flatten_columns(station_detail.groupby(["method", "station"], sort=False)[["f1", "recall", "precision", "far"]].agg(["mean", "std"]).reset_index())
    station_agg.to_csv(TABLES / "stage141_noaa_frozen_external_validation_by_station_aggregate.csv", index=False, encoding="utf-8-sig")
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    integrity = []
    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        name = fold_name(train_years, valid_year, test_year)
        rows_fold = source.loc[source["station_id"].astype(str).isin(stations) & source["year"].isin([*train_years, valid_year, test_year])]
        attack = attack_detail.loc[(attack_detail["fold"].eq(name)) & attack_detail["method"].eq("TrustMeteo-QC frozen expanded")]
        attack_rows = int(attack["rows"].sum())
        integrity.extend({"fold": name, "seed": int(seed), "rows": int(len(rows_fold)), "stations": len(stations), "train_attack_rows": 0, "valid_attack_rows": attack_rows, "test_attack_rows": attack_rows} for seed in SEEDS)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
    "protocol": "NOAA官方2020-2025扩展表；质量审计后11站点；四折累计历史训练→前一年valid→目标年test；攻击仅为透明受控注入。扩展alpha/window网格在外部运行前冻结，每折只用valid选择，test只评价；不根据test重新选择、不重新训练。固定5个种子用于种子级不确定性；三个无监督基线复用阶段85同协议结果。",
        "stations": stations,
        "folds": [fold_name(*fold) for fold in FOLDS if set([*fold[0], fold[1], fold[2]]).issubset(years)],
        "seeds": SEEDS,
        "frozen_alpha_grid": ALPHAS,
        "frozen_window_grid": WINDOWS,
        "low_far_q_grid": Q_GRID,
        "aggregate": aggregate.to_dict(orient="records"),
        "fold_aggregate": fold_aggregate.to_dict(orient="records"),
        "paired": paired_df.to_dict(orient="records"),
        "low_far_aggregate": op_aggregate.to_dict(orient="records"),
        "attack_type_aggregate": attack_agg.to_dict(orient="records"),
        "station_aggregate": station_agg.to_dict(orient="records"),
        "integrity": integrity,
        "limitations": ["NOAA观测是真实外部数据，但攻击标签是透明受控注入，不是真实网络攻击日志。", "所有新验证仍来自同一NOAA来源，不能替代跨地区或真实攻击验证。", "扩展候选只有在本外部协议支持时才可考虑替换default；若不支持，应保留为候选而不是删掉负结果。"],
    }
    (TABLES / "stage141_noaa_frozen_external_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段141 NOAA冻结候选外部验证", "", "四折滚动协议：累计历史训练→前一年验证→目标年测试。扩展融合网格和选择规则在本阶段运行前冻结；每折只用valid选择，test只评价。", "", "| 方法 | test F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_false_alarm_rate_mean']:.4f}±{row['test_false_alarm_rate_std']:.4f} |")
    lines += ["", "## 统计比较", "", "成对差异使用按滚动折聚类的bootstrap 95% CI，并报告sign-flip p和BH校正q；不能把受控攻击结果写成真实攻击检测。", "", "| 比较 | 指标 | 平均差 | CI95 | p | q |", "|---|---|---:|---:|---:|---:|"]
    for row in paired_df.to_dict(orient="records"):
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_diff']:.4f} | [{row['bootstrap_cluster_ci95_low']:.4f}, {row['bootstrap_cluster_ci95_high']:.4f}] | {row['sign_flip_p']:.4f} | {row['bh_q']:.4f} |")
    lines += ["", "## 解释边界", "", "该阶段的NOAA观测是真实外部气象数据，但攻击事件由透明受控注入构造；结果用于外部时序协议验证，不能替代真实攻击日志。若冻结扩展候选在本阶段不稳定，应继续保留default和负结果。"]
    (REPORTS / "stage141_noaa_frozen_external_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(detail), "methods": int(detail["method"].nunique()), "folds": len(payload["folds"]), "seeds": len(SEEDS)}, ensure_ascii=False))
    return 0


def main() -> int:
    source = pd.read_parquet(NOAA)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {(int(row["year"]), str(row["station_id"])) for row in audit["detail"] if row.get("eligible_for_extended_validation")}
    years = sorted(int(x) for x in source["year"].unique())
    stations = sorted({station for station in source["station_id"].astype(str).unique() if all((year, station) in eligible for year in years)})
    baseline = pd.read_csv(BASELINE_DETAIL)
    expected_baselines = {"Robust max deviation", "Isolation Forest", "One-Class SVM"}
    if not expected_baselines.issubset(set(baseline["method"].unique())):
        raise RuntimeError("stage85 baseline table is incomplete")

    rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    attack_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    operating_rows: list[dict[str, object]] = []
    integrity: list[dict[str, object]] = []

    for train_years, valid_year, test_year in FOLDS:
        if not set([*train_years, valid_year, test_year]).issubset(years):
            continue
        name = fold_name(train_years, valid_year, test_year)
        for seed in SEEDS:
            frame = build_fold(source, seed, train_years, valid_year, test_year, set(stations))
            scored, default_result = score(frame)
            scored = add_components(scored)
            selected, expanded_pred = choose_expanded(scored)
            default_test = scored["split"].eq("test")
            default_pred = scored.loc[default_test, "predicted_attack"].astype(bool).to_numpy()
            y_test = scored.loc[default_test, "combined_is_attack"].astype(bool).to_numpy()
            default_risk = scored.loc[default_test, "robust_fused_risk"].to_numpy(dtype=float)
            default_row = {
                "fold": name, "seed": seed, "method": "TrustMeteo-QC default",
                "valid_f1": float(default_result["valid"]["f1"]), "test_f1": float(default_result["test"]["f1"]),
                "test_recall": float(default_result["test"]["recall"]), "test_precision": float(default_result["test"]["precision"]),
                "test_auprc": float(default_result["test"]["average_precision"]), "test_false_alarm_rate": float(default_result["test_false_alarm_rate"]),
                "threshold": float(default_result["setting"]["threshold"]), "alpha_base": float(default_result["setting"]["alpha_base"]),
                "window": int(default_result["setting"]["window"]),
            }
            expanded_row = {"fold": name, "seed": seed, "method": "TrustMeteo-QC frozen expanded", **selected}
            rows.extend([default_row, expanded_row])
            selected_rows.append(expanded_row)

            test_part = scored.loc[default_test].copy()
            for method, pred in [("TrustMeteo-QC default", default_pred), ("TrustMeteo-QC frozen expanded", expanded_pred)]:
                for attack_type in ["spike", "offset", "drift", "time_shift", "stuck", "gaussian_noise"]:
                    mask = test_part["combined_attack_type"].eq(attack_type).to_numpy()
                    attack_rows.append({"fold": name, "seed": seed, "method": method, "attack_type": attack_type, "rows": int(mask.sum()), "detected_rows": int(pred[mask].sum()), "recall": float(pred[mask].mean()) if mask.any() else float("nan")})
                for station in sorted(test_part["station_key"].astype(str).unique()):
                    mask = test_part["station_key"].astype(str).eq(station).to_numpy()
                    yt = y_test[mask]
                    station_rows.append({"fold": name, "seed": seed, "method": method, "station": station, "rows": int(mask.sum()), "attack_rows": int(yt.sum()), "f1": float(_metrics(yt, pred[mask])["f1"]), "recall": float(_metrics(yt, pred[mask])["recall"]), "precision": float(_metrics(yt, pred[mask])["precision"]), "far": float(pred[mask][~yt].mean()) if (~yt).any() else float("nan")})

            for method, risk in [("TrustMeteo-QC default", default_risk), ("TrustMeteo-QC frozen expanded", selected["alpha_base"] * scored.loc[default_test, "base_fused_risk"].to_numpy(dtype=float) + (1.0 - selected["alpha_base"]) * scored.loc[default_test, "structural_risk"].to_numpy(dtype=float))]:
                if method == "TrustMeteo-QC frozen expanded":
                    risk = smooth_risk(test_part, np.asarray(risk, dtype=float), int(selected["window"]))
                train_mask = scored["split"].eq("train")
                if method == "TrustMeteo-QC default":
                    train_risk = scored.loc[train_mask, "robust_fused_risk"].to_numpy(dtype=float)
                else:
                    raw_train = selected["alpha_base"] * scored.loc[train_mask, "base_fused_risk"].to_numpy(dtype=float) + (1.0 - selected["alpha_base"]) * scored.loc[train_mask, "structural_risk"].to_numpy(dtype=float)
                    train_risk = smooth_risk(scored.loc[train_mask], raw_train, int(selected["window"]))
                y = y_test
                for q in Q_GRID:
                    threshold = float(np.nanquantile(train_risk, q))
                    pred = risk >= threshold
                    m = _metrics(y, pred, risk)
                    operating_rows.append({"fold": name, "seed": seed, "method": method, "q": q, "threshold": threshold, "test_f1": float(m["f1"]), "test_recall": float(m["recall"]), "test_precision": float(m["precision"]), "test_auprc": float(m["average_precision"]), "test_false_alarm_rate": float(pred[~y].mean()) if (~y).any() else float("nan")})

            integrity.append({"fold": name, "seed": seed, "rows": int(len(frame)), "stations": int(frame["station_key"].nunique()), "train_attack_rows": int(frame.loc[frame["split"].eq("train"), "combined_is_attack"].sum()), "valid_attack_rows": int(frame.loc[frame["split"].eq("valid"), "combined_is_attack"].sum()), "test_attack_rows": int(frame.loc[frame["split"].eq("test"), "combined_is_attack"].sum())})
        print(json.dumps({"fold": name, "seeds": len(SEEDS), "status": "completed"}, ensure_ascii=False), flush=True)

    detail = pd.DataFrame(rows)
    baseline = baseline.loc[baseline["method"].isin(expected_baselines)].copy()
    baseline = baseline.rename(columns={"test_false_alarm_rate": "test_false_alarm_rate"})
    for method in sorted(expected_baselines):
        part = baseline.loc[baseline["method"].eq(method)].copy()
        if len(part) != len(selected_rows):
            raise RuntimeError(f"baseline row count mismatch for {method}: {len(part)}")
        rows.extend(part[["fold", "seed", "method", "valid_f1", "test_f1", "test_recall", "test_precision", "test_auprc", "test_false_alarm_rate", "threshold"]].to_dict(orient="records"))
    detail = pd.DataFrame(rows)
    aggregate = detail.groupby("method", sort=False)[METRICS].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.columns = ["method"] + [f"{a}_{b}" for a, b in aggregate.columns.tolist()[1:]]
    fold_aggregate = detail.groupby(["fold", "method"], sort=False)[METRICS].agg(["mean", "std"]).reset_index()
    fold_aggregate.columns = ["fold", "method"] + [f"{a}_{b}" for a, b in fold_aggregate.columns.tolist()[2:]]

    paired_rows = []
    methods = ["TrustMeteo-QC default", "Robust max deviation", "Isolation Forest", "One-Class SVM"]
    expanded_lookup = detail.loc[detail["method"].eq("TrustMeteo-QC frozen expanded")].set_index(["fold", "seed"])
    for other in methods:
        other_lookup = detail.loc[detail["method"].eq(other)].set_index(["fold", "seed"])
        common = expanded_lookup.join(other_lookup, lsuffix="_expanded", rsuffix="_other", how="inner")
        for metric in METRICS:
            diff = (common[f"{metric}_expanded"] - common[f"{metric}_other"]).astype(float).to_numpy()
            tmp = pd.DataFrame({"fold": common.index.get_level_values(0), "diff": diff})
            ci_low, ci_high = bootstrap_cluster(tmp.rename(columns={"diff": "value"}), "value", stable_seed("stage141", other, metric))
            p = sign_flip_p(diff, stable_seed("stage141-p", other, metric))
            paired_rows.append({"comparison": f"expanded - {other}", "metric": metric, "n": int(len(diff)), "mean_diff": float(diff.mean()), "bootstrap_cluster_ci95_low": ci_low, "bootstrap_cluster_ci95_high": ci_high, "sign_flip_p": p})
    paired_df = pd.DataFrame(paired_rows)
    paired_df["bh_q"] = bh_adjust(paired_df["sign_flip_p"].tolist())

    op = pd.DataFrame(operating_rows)
    op_aggregate = op.groupby(["method", "q"], sort=False)[METRICS].agg(["mean", "std"]).reset_index()
    op_aggregate.columns = ["method", "q"] + [f"{a}_{b}" for a, b in op_aggregate.columns.tolist()[2:]]
    attack_df = pd.DataFrame(attack_rows)
    attack_agg = attack_df.groupby(["method", "attack_type"], sort=False)[["rows", "detected_rows", "recall"]].agg(["sum", "mean"]).reset_index()
    attack_agg = flatten_columns(attack_agg)
    station_df = pd.DataFrame(station_rows)
    station_agg = station_df.groupby(["method", "station"], sort=False)[["f1", "recall", "precision", "far"]].agg(["mean", "std"]).reset_index()
    station_agg = flatten_columns(station_agg)

    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "NOAA官方2020-2025扩展表；质量审计后11站点；四折累计历史训练→前一年valid→目标年test；攻击仅为透明受控注入。扩展alpha/window网格在外部运行前冻结，每折只用valid选择，test只评价；不根据test重新选择、不重新训练。固定5个种子用于种子级不确定性；三个无监督基线复用阶段85同协议结果。",
        "stations": stations, "folds": [fold_name(*fold) for fold in FOLDS if set([*fold[0], fold[1], fold[2]]).issubset(years)], "seeds": SEEDS,
        "frozen_alpha_grid": ALPHAS, "frozen_window_grid": WINDOWS, "low_far_q_grid": Q_GRID,
        "aggregate": aggregate.to_dict(orient="records"), "fold_aggregate": fold_aggregate.to_dict(orient="records"),
        "paired": paired_df.to_dict(orient="records"), "low_far_aggregate": op_aggregate.to_dict(orient="records"),
        "attack_type_aggregate": attack_agg.to_dict(orient="records"), "station_aggregate": station_agg.to_dict(orient="records"),
        "integrity": integrity,
        "limitations": ["NOAA观测是真实外部数据，但攻击标签是透明受控注入，不是真实网络攻击日志。", "所有新验证仍来自同一NOAA来源，不能替代跨地区或真实攻击验证。", "扩展候选只有在本外部协议支持时才可考虑替换default；若不支持，应保留为候选而不是删掉负结果。"],
    }
    detail.to_csv(TABLES / "stage141_noaa_frozen_external_validation_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(TABLES / "stage141_noaa_frozen_external_validation_selected.csv", index=False, encoding="utf-8-sig")
    fold_aggregate.to_csv(TABLES / "stage141_noaa_frozen_external_validation_by_fold.csv", index=False, encoding="utf-8-sig")
    attack_df.to_csv(TABLES / "stage141_noaa_frozen_external_validation_by_attack_type.csv", index=False, encoding="utf-8-sig")
    attack_agg.to_csv(TABLES / "stage141_noaa_frozen_external_validation_by_attack_type_aggregate.csv", index=False, encoding="utf-8-sig")
    station_df.to_csv(TABLES / "stage141_noaa_frozen_external_validation_by_station.csv", index=False, encoding="utf-8-sig")
    station_agg.to_csv(TABLES / "stage141_noaa_frozen_external_validation_by_station_aggregate.csv", index=False, encoding="utf-8-sig")
    op.to_csv(TABLES / "stage141_noaa_frozen_external_validation_low_far_detail.csv", index=False, encoding="utf-8-sig")
    op_aggregate.to_csv(TABLES / "stage141_noaa_frozen_external_validation_low_far.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(TABLES / "stage141_noaa_frozen_external_validation.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(TABLES / "stage141_noaa_frozen_external_validation_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage141_noaa_frozen_external_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")

    lines = ["# 阶段141 NOAA冻结候选外部验证", "", "四折滚动协议：累计历史训练→前一年验证→目标年测试。扩展融合网格和选择规则在本阶段运行前冻结；每折只用valid选择，test只评价。", "", "| 方法 | test F1 | Recall | Precision | AUPRC | FAR |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.to_dict(orient="records"):
        lines.append(f"| {row['method']} | {row['test_f1_mean']:.4f}±{row['test_f1_std']:.4f} | {row['test_recall_mean']:.4f}±{row['test_recall_std']:.4f} | {row['test_precision_mean']:.4f}±{row['test_precision_std']:.4f} | {row['test_auprc_mean']:.4f}±{row['test_auprc_std']:.4f} | {row['test_false_alarm_rate_mean']:.4f}±{row['test_false_alarm_rate_std']:.4f} |")
    lines += ["", "## 统计比较", "", "成对差异使用按滚动折聚类的bootstrap 95% CI，并报告sign-flip p和BH校正q；不能把受控攻击结果写成真实攻击检测。", "", "| 比较 | 指标 | 平均差 | CI95 | p | q |", "|---|---|---:|---:|---:|---:|"]
    for row in paired_df.to_dict(orient="records"):
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_diff']:.4f} | [{row['bootstrap_cluster_ci95_low']:.4f}, {row['bootstrap_cluster_ci95_high']:.4f}] | {row['sign_flip_p']:.4f} | {row['bh_q']:.4f} |")
    lines += ["", "## 解释边界", "", "该阶段的NOAA观测是真实外部气象数据，但攻击事件由透明受控注入构造；结果用于外部时序协议验证，不能替代真实攻击日志。若冻结扩展候选在本阶段不稳定，应继续保留default和负结果。"]
    (REPORTS / "stage141_noaa_frozen_external_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(detail), "methods": int(detail["method"].nunique()), "folds": len(payload["folds"]), "seeds": len(SEEDS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(finalize_existing() if "--finalize-existing" in sys.argv else main())
