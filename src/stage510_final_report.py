"""Build the final Stage503 deep-baseline report from the measured artifacts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "tables"
R = ROOT / "reports"
LABELS = {
    "lstm_ae": "LSTM-AE",
    "usad": "USAD",
    "timesnet": "TimesNet",
    "anomaly_transformer": "Anomaly Transformer",
    "trust_only": "Trust only",
    "route_knn_arima": "Route kNN+ARIMA",
    "trust_shape_union": "Trust+shape union",
    "mlp_autoencoder_reconstruction": "MLP AE (64-16-64)",
    "pca_reconstruction": "PCA AE",
    "isolation_forest": "Isolation Forest",
}


def f4(x: float) -> str:
    return f"{float(x):.4f}"


def main() -> int:
    deep = pd.read_csv(T / "stage503_deep_baseline_summary.csv")
    frozen = pd.read_csv(T / "stage349_v6_locked_evaluation_detail.csv")
    totals = pd.read_csv(T / "stage505_deep_totals.csv")
    h2h = pd.read_csv(T / "stage505_union_vs_deep.csv")
    res = pd.read_csv(T / "stage506_resource_claims.csv")
    boot = pd.read_csv(T / "stage507_deep_bootstrap.csv")

    macro = frozen.groupby("method", sort=False)[
        ["precision", "row_recall", "f1", "event_coverage", "average_precision",
         "false_alarm_rate", "roc_auc"]
    ].mean()
    for method in ["trust_shape_union", "mlp_autoencoder_reconstruction", "pca_reconstruction"]:
        assert method in macro.index, method
    union = macro.loc["trust_shape_union"]
    mlp = macro.loc["mlp_autoencoder_reconstruction"]
    pca = macro.loc["pca_reconstruction"]
    union_far = float(union["false_alarm_rate"])
    deep_far_lo = float(deep["false_alarm_rate"].min())
    deep_far_hi = float(deep["false_alarm_rate"].max())
    knmi_union_far = float(
        frozen[(frozen["method"].eq("trust_shape_union")) & (frozen["source"].eq("KNMI"))][
            "false_alarm_rate"
        ].mean()
    )

    lines = [
        "# 主流深度异常检测基线补跑：最终报告（Stage503–Stage507）",
        "",
        "> 状态：**EXPLORATORY_POST_V6**。V6 测试集已在 Stage349 检视；本补充只用于回应“Table 4 缺少主流深度基线”的审稿批评，不替代 V6 主证据，不得标为确认性。",
        "",
        "## 1. 实验口径",
        "",
        "- 4 个提供方：ECCC、KNMI、MeteoSwiss、NOAA ISD；5 seeds；9 个 V6 场景；共 20 个 provider-seed 运行/模型、720 个模型×场景×运行单元。",
        "- 同一 16 维特征空间（3 个 Trust 组件 + 13 个形态特征）；滑动窗口 24；每机构干净 train 行抽样 20,000 窗训练；模型 seed 固定 `20260922`。",
        "- 目标 FAR `α=0.0065`；阈值只在 **valid normal 行** 上取 `1-α` 分位数，原样用于 test；test 不使用任何标签调阈值；不做 point adjustment。",
        "- 模型：LSTM-AE（h64/z16，denoise 0.1，30 ep）、USAD（latent 8，官方 min-max/Sigmoid，10 ep）、TimesNet（d32，top-3 周期，20 ep）、Anomaly Transformer（d64，4 heads，1 layer，Gaussian prior，20 ep）。",
        "- 这些是 **CPU 可运行的缩减规模实现**，不是原论文的完整模型；结论只对本实现/本特征空间/本硬件成立。",
        "- 超参与 epoch 是在 MeteoSwiss 的 prelim 运行上选定的，未做嵌套交叉验证；这是本探索性补充的额外限制，不能当作确认性调参。",
        "",
        "## 2. 主结果（provider-seed 等权；与 Table 4 同口径）",
        "",
        "| 方法 | 精确率 | 行召回 | F1 | 事件覆盖（主终点） | AP | FAR | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["trust_shape_union", "mlp_autoencoder_reconstruction", "pca_reconstruction"]
    for method in order:
        row = macro.loc[method]
        lines.append(
            f"| {LABELS[method]} | {f4(row['precision'])} | {f4(row['row_recall'])} | {f4(row['f1'])} | "
            f"{f4(row['event_coverage'])} | {f4(row['average_precision'])} | "
            f"{100*row['false_alarm_rate']:.3f}% | {f4(row['roc_auc'])} |"
        )
    for _, row in deep.iterrows():
        lines.append(
            f"| {LABELS[row['method']]} | {f4(row['precision'])} | {f4(row['row_recall'])} | {f4(row['f1'])} | "
            f"{f4(row['event_coverage'])} | {f4(row['average_precision'])} | "
            f"{100*row['false_alarm_rate']:.3f}% | {f4(row['roc_auc'])} |"
        )
    lines += [
        "",
        "### 2.1 必须正面承认的权衡",
        "",
        f"- 事件覆盖（论文主终点）：union `{f4(union['event_coverage'])}`，四个深度基线为 "
        + "、".join(f"{LABELS[r['method']]} `{f4(r['event_coverage'])}`" for _, r in deep.iterrows())
        + "；union 最高（深度最高值仍低约 0.077）。",
        f"- F1：union `{f4(union['f1'])}`，深度基线为 "
        + "、".join(f"{LABELS[r['method']]} `{f4(r['f1'])}`" for _, r in deep.iterrows())
        + f"；**只有 TimesNet（`{f4(float(deep.loc[deep['method'].eq('timesnet'), 'f1'].iloc[0]))}`）高于 union**，USAD/AT/LSTM-AE 均低于 union。",
        f"- AP：union `{f4(union['average_precision'])}`，深度基线为 "
        + "、".join(f"{LABELS[r['method']]} `{f4(r['average_precision'])}`" for _, r in deep.iterrows())
        + "；**USAD/TimesNet/AT 的 AP 高于 union，LSTM-AE 低于 union**。",
        f"- FAR：union 宏平均 `{100*union_far:.3f}%`，深度基线 `{100*deep_far_lo:.3f}%–{100*deep_far_hi:.3f}%`；**深度基线 FAR 不高于 union，union 并没有“误报更少”的优势**。KNMI 上 union FAR 为 `{100*knmi_union_far:.3f}%`，而深度约 0.65%，该机构的覆盖差不能直接当作同 FAR 下的优势。",
        "- 因此正确表述是：**在统一低 FAR 上限下，union 的事件覆盖更高；但行级 F1（对 TimesNet）、AP（对 USAD/TimesNet/AT）和 FAR 并不是 union 的优势项**。不能写成 union 在所有指标上全面领先，也不能声称深度基线误报更多。",
        "",
        "### 2.2 分机构核对（防止宏平均掩盖问题）",
        "",
        "| 机构 | union 覆盖 | union F1 | union FAR | 深度最高覆盖（模型） | 深度最高 F1（模型） | 深度 FAR 范围 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    prov_union = frozen[frozen["method"].eq("trust_shape_union")].groupby("source")[
        ["event_coverage", "f1", "false_alarm_rate"]
    ].mean()
    prov_deep = pd.read_csv(T / "stage503_deep_baseline_detail.csv")
    prov_deep = prov_deep[prov_deep["split"].eq("test")].groupby(["source", "method"])[
        ["event_coverage", "f1", "false_alarm_rate"]
    ].mean()
    for source in ["MeteoSwiss", "KNMI", "ECCC", "NOAA ISD"]:
        u = prov_union.loc[source]
        sub = prov_deep.loc[source]
        best_cov_model = sub["event_coverage"].idxmax()
        best_f1_model = sub["f1"].idxmax()
        lines.append(
            f"| {source} | {f4(u['event_coverage'])} | {f4(u['f1'])} | {100*u['false_alarm_rate']:.3f}% | "
            f"{f4(sub.loc[best_cov_model, 'event_coverage'])}（{LABELS[best_cov_model]}） | "
            f"{f4(sub.loc[best_f1_model, 'f1'])}（{LABELS[best_f1_model]}） | "
            f"{100*sub['false_alarm_rate'].min():.3f}%–{100*sub['false_alarm_rate'].max():.3f}% |"
        )
    lines += [
        "",
        "- **MeteoSwiss**：union 覆盖 0.409 vs 深度最高 0.362，FAR 接近（0.69% vs 0.60–0.70%），union 明确更好。",
        "- **ECCC**：union 覆盖 0.346 vs LSTM-AE 0.344（基本打平）；union F1 0.263 vs USAD 0.263（打平），AP 低于 USAD/TimesNet。",
        "- **NOAA ISD**：union 覆盖 0.253 vs TimesNet 0.247（差 0.006，基本打平），但 TimesNet F1 0.229 vs union 0.160，FAR 同为约 0.7%；该机构上深度模型更实用。",
        "- **KNMI**：union FAR 1.675%，是目标 0.65% 的 2.6 倍；深度 FAR 约 0.65–0.74%。因此 KNMI 的覆盖差（0.416 vs 0.308–0.360）**不是同 FAR 比较**，不能用来支持 union 优势。",
        "- **结论**：宏平均上 union 覆盖仍最高，但分机构看是“MeteoSwiss 明确胜、ECCC/NOAA 打平、KNMI 被 FAR 违规混淆”。论文若要用这组数，必须补 matched-FAR 分析或明确承认 KNMI 的 FAR 失守，不能只报宏平均。",
        "",
        "## 3. 场景×基线胜率（深度模型按事件覆盖 vs 11 个非 Trust 基线）",
        "",
        "| 模型 | 胜/99 | 总胜率 | ≥70% 场景数 | 平均覆盖 | 平均 F1 | 平均 AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in totals.iterrows():
        lines.append(
            f"| {LABELS[row['model']]} | {int(row['wins_total'])}/99 | "
            f"{100*row['win_share_total']:.1f}% | {int(row['scenarios_ge_70pct'])}/9 | "
            f"{f4(row['mean_event_coverage'])} | {f4(row['mean_f1'])} | {f4(row['mean_average_precision'])} |"
        )
    lines += [
        "",
        "> 读法警告：LSTM-AE 胜 67/99 靠的是“报警更宽”——覆盖 0.243 但 F1 仅 0.069、精确率 0.098；与 union 头对头只赢 4/9 个场景。**事件覆盖必须和精确率/F1 一起读**，否则“覆盖高”可以靠滥报获得。",
        "",
        "## 4. union 与深度基线逐场景头对头（事件覆盖）",
        "",
        "| 深度模型 | union 胜场景 | 深度胜场景 | 平局 | 平均覆盖差（union−深度） |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in h2h.iterrows():
        lines.append(
            f"| {LABELS[row['model']]} | {int(row['union_wins'])}/9 | {int(row['deep_wins'])}/9 | "
            f"{int(row['ties'])} | {f4(row['mean_union_minus_model'])} |"
        )
    lines += [
        "",
        "## 5. 单步时延与内存（对 Table 2 声称的实测核对）",
        "",
        "> batch=1、16 维、窗口 24、PyTorch 单线程、warmup 200、计时 2000 步；进程峰值含 Python/PyTorch 运行时；硬件 i5-14600KF（非论文的 i7-12700H）。",
        "",
        "| 模型 | 均值 ms | P95 ms | 参数 MB | 进程峰值 MB | 论文时延声称 | 论文内存声称 | 时延复现 | 进程峰值超声称 |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for _, row in res.iterrows():
        lat = "—" if pd.isna(row["paper_latency_claim_ms"]) else f">{row['paper_latency_claim_ms']:.0f} ms"
        mem = "—" if pd.isna(row["paper_memory_claim_mb"]) else f">{row['paper_memory_claim_mb']:.0f} MB"
        lines.append(
            f"| {LABELS[row['model']]} | {row['latency_mean_ms']:.3f} | {row['latency_p95_ms']:.3f} | "
            f"{row['param_mb']:.4f} | {row['peak_wset_mb']:.1f} | {lat} | {mem} | "
            f"{'是' if row['latency_claim_reproduced'] else '否'} | "
            f"{'是' if row['memory_claim_reproduced_process_peak'] else '否'} |"
        )
    lines += [
        "",
        "### 5.1 对 Table 2 的处理建议",
        "",
        "- 本实现下，LSTM-AE/AT 的单步时延都是 **毫秒级**，没有复现 `>100 ms`/`>200 ms`；AT 进程峰值也没有达到 `>500 MB`。",
        "- LSTM-AE 进程峰值确实超过 `>200 MB`，但参数只有不到 1 MB，峰值主要来自 Python/PyTorch 运行时；把它写成“模型内存 >200 MB”会被审稿人追问。",
        "- 建议二选一：① 把这两行改为“文献报告量级（模型/硬件相关）”并补引用；② 删除绝对阈值，改为“本系统 <0.1 ms/2.92 ms 与 83.5 MB，深度基线需 GPU/更大内存”的定性表述。",
        "- 不要用本实现的小模型去宣称原论文的大模型也 <1 ms；也不要用原论文的数字去否认本实现的测量。",
        "",
        "## 6. 探索性 95% 区间（provider-seed 重抽样 4000 次）",
        "",
        "| 模型 | F1 [95%CI] | 事件覆盖 [95%CI] | AP [95%CI] |",
        "|---|---:|---:|---:|",
    ]
    for model in deep["method"]:
        def cell(metric: str) -> str:
            sub = boot[(boot["model"].eq(model)) & (boot["metric"].eq(metric))]
            if sub.empty:
                return "—"
            r = sub.iloc[0]
            return f"{f4(r['mean'])} [{f4(r['ci95_lo'])}, {f4(r['ci95_hi'])}]"
        lines.append(f"| {LABELS[model]} | {cell('f1')} | {cell('event_coverage')} | {cell('average_precision')} |")
    lines += [
        "",
        "> 这不是 V6 的 provider-first 分层 bootstrap；只用于说明 20 个 provider-seed 运行的抽样波动，不能替代确认性区间。",
        "",
        "## 7. 结论与可用表述",
        "",
        "1. 四个主流深度基线已在 **完全相同的 16 维特征空间、同一 valid-only 阈值协议、同一 9 场景/5 seeds** 下补跑；Table 4 可以从 14 行扩到 18 行。",
        "2. **主终点（事件覆盖）union 最高**：`{:.4f}` vs 深度 `{:.4f}–{:.4f}`；但 union FAR 为 `{:.3f}%`（KNMI `{:.3f}%`），不是严格低 FAR，分机构比较必须用 matched-FAR。".format(
            float(union["event_coverage"]),
            float(deep["event_coverage"].min()),
            float(deep["event_coverage"].max()),
            100*union_far,
            100*knmi_union_far,
        ),
        "3. **行级指标不是 union 全面领先**：TimesNet F1 `{:.4f}` 高于 union `{:.4f}`；USAD/TimesNet/AT 的 AP（`{:.4f}/{:.4f}/{:.4f}`）也高于 union `{:.4f}`。论文必须写成“覆盖—行级精度权衡”，不能宣称全面领先。".format(
            float(deep.loc[deep["method"].eq("timesnet"), "f1"].iloc[0]),
            float(union["f1"]),
            float(deep.loc[deep["method"].eq("usad"), "average_precision"].iloc[0]),
            float(deep.loc[deep["method"].eq("timesnet"), "average_precision"].iloc[0]),
            float(deep.loc[deep["method"].eq("anomaly_transformer"), "average_precision"].iloc[0]),
            float(union["average_precision"]),
        ),
        "4. **FAR 不能作为 union 的优势**：union 宏平均 `{:.3f}%`，深度 `{:.3f}%–{:.3f}%`；KNMI 上 union `{:.3f}%`。深度基线并没有更多误报，反而普遍更低。".format(
            100*union_far, 100*deep_far_lo, 100*deep_far_hi, 100*knmi_union_far
        ),
        "5. LSTM-AE 在本特征空间下 F1 仅 `{:.4f}`、覆盖 `{:.4f}`；覆盖不低但精确率 `{:.4f}`、F1 极低，说明“覆盖高”可以靠滥报获得，必须与 F1/精确率联读。这是缩减规模实现，不能引申为原论文方法无效。".format(
            float(deep.loc[deep["method"].eq("lstm_ae"), "f1"].iloc[0]),
            float(deep.loc[deep["method"].eq("lstm_ae"), "event_coverage"].iloc[0]),
            float(deep.loc[deep["method"].eq("lstm_ae"), "precision"].iloc[0]),
        ),
        "6. Table 2 的 `>100 ms`/`>200 ms`/`>500 MB` 在本实现/本硬件下没有实测支持；必须限定条件或改引文献。",
        "7. 以上均为 **V6 已检视后的探索性补充**；不能写成确认性结论，也不能支持生产攻击召回、跨机构普适或在线=离线等价。",
        "",
        "## 8. 产物",
        "",
        "- `tables/stage503_deep_baseline_detail.csv`（160 行：4 模型×4 机构×5 seeds×2 splits）",
        "- `tables/stage503_deep_baseline_scenario.csv`（720 行）",
        "- `tables/stage503_deep_baseline_summary.csv`（4 行，provider-seed 宏平均）",
        "- `tables/stage503_main_results_combined.csv`（18 行，V6 14 方法 + 4 深度）",
        "- `tables/stage505_deep_scenario_wins.csv` / `stage505_deep_totals.csv` / `stage505_union_vs_deep.csv`",
        "- `tables/stage506_resource_claims.csv`；`tables/stage503_deep_baseline_resources.csv`",
        "- `tables/stage507_deep_bootstrap.csv`",
        "- `submission_final_20261009/final_experiment_evidence.xlsx` 的工作表 `探索503_*`",
        "",
    ]
    (R / "stage510_deep_baselines_final.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", R / "stage510_deep_baselines_final.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
