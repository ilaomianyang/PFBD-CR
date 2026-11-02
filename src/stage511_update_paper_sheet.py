"""Append the Stage503 deep-baseline evidence to the hand-curated
`论文直用数据` sheet of the submission workbook.

The sheet is the author's direct-copy source for the manuscript.  It must stay
usable, so this script:

* preserves every existing row (V6 sections 1-7) exactly;
* appends one clearly labelled exploratory section 8 with
  Table-4 extension (18 methods), per-provider caveats, scenario wins,
  union-vs-deep head-to-head, resource claims, and bootstrap intervals;
* runs through Excel COM so the 95 comments / VML drawings survive;
* is idempotent: an existing section 8 is removed before it is rewritten.

The section is exploratory (EXPLORATORY_POST_V6): the V6 test set was already
inspected in Stage349, so it must never be relabelled as confirmatory.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import win32com.client as win32

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
DEFAULT_WB = ROOT / "submission_final_20261009" / "final_experiment_evidence.xlsx"
SHEET = "论文直用数据"
MARKER = "八、Table 4 扩表"

DEEP_ORDER = ["lstm_ae", "usad", "timesnet", "anomaly_transformer"]
LABELS = {
    "lstm_ae": "LSTM-AE",
    "usad": "USAD",
    "timesnet": "TimesNet",
    "anomaly_transformer": "Anomaly Transformer",
}
PROVIDERS = ["MeteoSwiss", "KNMI", "ECCC", "NOAA ISD"]

# Excel COM colours are BGR integers.
C_TITLE_FONT = 0x45351E      # RGB 1E3545
C_TITLE_FILL = 0xEDE6DC      # RGB DCE6ED
C_HEADER_FILL = 0x5C4424     # RGB 24445C
C_BODY_FONT = 0x332B20       # RGB 202B33
C_SUB_FILL = 0xF7F3EE        # RGB EEF3F7
C_NOTE_FILL = 0xCCF2FF       # RGB FFF2CC


def label(model: str) -> str:
    return LABELS.get(model, model)


def _num(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def load_data() -> dict:
    deep = pd.read_csv(TABLES / "stage503_deep_baseline_summary.csv").set_index("method")
    deep_detail = pd.read_csv(TABLES / "stage503_deep_baseline_detail.csv")
    deep_test = deep_detail[deep_detail["split"].eq("test")].copy()
    v6 = pd.read_csv(TABLES / "stage349_v6_locked_evaluation_detail.csv")
    union = v6[v6["method"].eq("trust_shape_union")]

    per_provider = []
    expected = {
        "MeteoSwiss": (0.4089, 0.3622, "LSTM-AE", 0.2632, "USAD", 0.00602, 0.00698),
        "KNMI": (0.4160, 0.3600, "Anomaly Transformer", 0.2631, "USAD", 0.00662, 0.00739),
        "ECCC": (0.3456, 0.3439, "LSTM-AE", 0.2632, "USAD", 0.00538, 0.00719),
        "NOAA ISD": (0.2533, 0.2467, "TimesNet", 0.2290, "TimesNet", 0.00631, 0.00720),
    }
    for provider in PROVIDERS:
        u = union[union["source"].eq(provider)]
        d = deep_test[deep_test["source"].eq(provider)]
        # Compare the union mean with each deep model's mean over the five
        # attack seeds.  Taking the best individual seed would cherry-pick a
        # lucky run and inflate the deep baseline.
        g = d.groupby("method")[["event_coverage", "f1", "false_alarm_rate"]].mean()
        best_cov_model = g["event_coverage"].idxmax()
        best_f1_model = g["f1"].idxmax()
        item = {
            "provider": provider,
            "union_coverage": _num(u["event_coverage"].mean()),
            "union_f1": _num(u["f1"].mean()),
            "union_far": _num(u["false_alarm_rate"].mean(), 6),
            "deep_best_coverage": _num(g.loc[best_cov_model, "event_coverage"]),
            "deep_best_coverage_model": label(best_cov_model),
            "deep_best_f1": _num(g.loc[best_f1_model, "f1"]),
            "deep_best_f1_model": label(best_f1_model),
            "deep_far_lo": _num(g["false_alarm_rate"].min(), 6),
            "deep_far_hi": _num(g["false_alarm_rate"].max(), 6),
        }
        exp = expected[provider]
        assert abs(item["union_coverage"] - exp[0]) < 5e-5, (provider, item, exp)
        assert abs(item["deep_best_coverage"] - exp[1]) < 5e-5, (provider, item, exp)
        assert item["deep_best_coverage_model"] == exp[2], (provider, item, exp)
        assert abs(item["deep_best_f1"] - exp[3]) < 5e-5, (provider, item, exp)
        assert item["deep_best_f1_model"] == exp[4], (provider, item, exp)
        assert abs(item["deep_far_lo"] - exp[5]) < 5e-5, (provider, item, exp)
        assert abs(item["deep_far_hi"] - exp[6]) < 5e-5, (provider, item, exp)
        per_provider.append(item)

    wins = pd.read_csv(TABLES / "stage505_deep_totals.csv").set_index("model")
    h2h = pd.read_csv(TABLES / "stage505_union_vs_deep.csv").set_index("model")
    resources = pd.read_csv(TABLES / "stage506_resource_claims.csv").set_index("model")
    boot = pd.read_csv(TABLES / "stage507_deep_bootstrap.csv")
    boot = boot[boot["metric"].isin(["f1", "event_coverage", "average_precision"])]

    return {
        "deep": deep,
        "per_provider": per_provider,
        "wins": wins,
        "h2h": h2h,
        "resources": resources,
        "boot": boot,
    }


def set_title(ws, row: int, text: str, ncols: int = 10) -> None:
    for c in range(1, ncols + 1):
        cell = ws.Cells(row, c)
        cell.Font.Name = "Arial"
        cell.Font.Size = 11
        cell.Font.Bold = True
        cell.Font.Color = C_TITLE_FONT
        cell.Interior.Color = C_TITLE_FILL
        cell.HorizontalAlignment = -4131  # xlLeft
        cell.VerticalAlignment = -4108   # xlCenter
    ws.Cells(row, 1).Value = text
    ws.Rows(row).RowHeight = 18


def set_subtitle(ws, row: int, text: str, ncols: int = 10) -> None:
    for c in range(1, ncols + 1):
        cell = ws.Cells(row, c)
        cell.Font.Name = "Arial"
        cell.Font.Size = 10
        cell.Font.Bold = True
        cell.Font.Color = C_TITLE_FONT
        cell.Interior.Color = C_SUB_FILL
        cell.VerticalAlignment = -4108
    ws.Cells(row, 1).Value = text
    ws.Rows(row).RowHeight = 16


def write_table(
    ws,
    row: int,
    headers: list[str],
    rows: list[list],
    *,
    formats: dict[int, str] | None = None,
    widths: dict[str, float] | None = None,
) -> int:
    formats = formats or {}
    for c, header in enumerate(headers, 1):
        cell = ws.Cells(row, c)
        cell.Value = header
        cell.Font.Name = "Arial"
        cell.Font.Size = 10
        cell.Font.Bold = True
        cell.Font.Color = 0xFFFFFF
        cell.Interior.Color = C_HEADER_FILL
        cell.HorizontalAlignment = -4108
        cell.VerticalAlignment = -4108
        cell.WrapText = True
        cell.Borders.LineStyle = 1
        cell.Borders.Weight = 2
    ws.Rows(row).RowHeight = 32
    for i, values in enumerate(rows, 1):
        for c, value in enumerate(values, 1):
            cell = ws.Cells(row + i, c)
            cell.Value = value
            cell.Font.Name = "Arial"
            cell.Font.Size = 10
            cell.Font.Color = C_BODY_FONT
            cell.VerticalAlignment = -4108
            cell.Borders.LineStyle = 1
            cell.Borders.Weight = 2
            if c in formats:
                cell.NumberFormat = formats[c]
    if widths:
        for col, width in widths.items():
            ws.Columns(col).ColumnWidth = width
    return row + 1 + len(rows)


def append_section(ws, start: int, data: dict) -> int:
    deep = data["deep"]
    row = start

    set_title(ws, row, "八、Table 4 扩表：V6 14 方法 + 深度基线 4 方法（深度为探索性补充，不并入 V6 主表）")
    row += 2

    # 8.1 Table 4 extension
    set_subtitle(ws, row, "8.1 主结果（provider-seed 等权；与第一节同口径，深度为 20 次运行）")
    row += 1
    headers = ["方法", "精确率", "逐行召回率", "F1", "事件覆盖率", "AP", "误报率", "ROC-AUC", "证据状态"]
    rows: list[list] = []
    r = 3
    while ws.Cells(r, 1).Value not in (None, ""):
        vals = [ws.Cells(r, c).Value for c in range(1, 9)]
        rows.append(vals + ["V6冻结（Stage349）"])
        r += 1
    assert len(rows) == 14, f"expected 14 V6 methods, got {len(rows)}"
    for model in DEEP_ORDER:
        d = deep.loc[model]
        rows.append([
            model, float(d["precision"]), float(d["recall"]), float(d["f1"]),
            float(d["event_coverage"]), float(d["average_precision"]),
            float(d["false_alarm_rate"]), float(d["roc_auc"]),
            "探索性补充（本表新增）",
        ])
    row = write_table(
        ws, row, headers, rows,
        formats={**{c: "0.0000" for c in range(2, 9)}, 7: "0.000%"},
        widths={"A": 38, "B": 14, "C": 14, "D": 12, "E": 16, "F": 12, "G": 12, "H": 12, "I": 26},
    )
    ws.Rows(row - len(rows) - 1).RowHeight = 42
    row += 1

    # 8.2 per-provider caveat
    set_subtitle(ws, row, "8.2 分机构（必须与宏平均一起报；KNMI union FAR 失守，不是同 FAR 比较）")
    row += 1
    headers = ["提供方", "union 覆盖", "union F1", "union FAR", "深度最高均值覆盖（模型）", "深度最高均值 F1（模型）", "深度均值 FAR 范围"]
    rows = []
    for p in data["per_provider"]:
        rows.append([
            p["provider"], p["union_coverage"], p["union_f1"], p["union_far"],
            f"{p['deep_best_coverage']:.4f}（{p['deep_best_coverage_model']}）",
            f"{p['deep_best_f1']:.4f}（{p['deep_best_f1_model']}）",
            f"{p['deep_far_lo']*100:.3f}%–{p['deep_far_hi']*100:.3f}%",
        ])
    row = write_table(
        ws, row, headers, rows,
        formats={2: "0.0000", 3: "0.0000", 4: "0.000%"},
        widths={"A": 30, "B": 11, "C": 11, "D": 10, "E": 32, "F": 21, "G": 18},
    )
    ws.Rows(row - len(rows) - 1).RowHeight = 42
    row += 1

    # 8.3 scenario wins
    set_subtitle(ws, row, "8.3 场景×11 基线胜率（事件覆盖；LSTM-AE 的胜率来自报警更宽，必须联读 F1）")
    row += 1
    headers = ["模型", "胜场/99", "胜率", "覆盖≥70%场景数", "平均覆盖", "平均 F1", "平均 AP"]
    rows = []
    for model in DEEP_ORDER:
        w = data["wins"].loc[model]
        rows.append([
            label(model), int(w["wins_total"]), _num(w["win_share_total"], 4),
            int(w["scenarios_ge_70pct"]), _num(w["mean_event_coverage"]),
            _num(w["mean_f1"]), _num(w["mean_average_precision"]),
        ])
    row = write_table(
        ws, row, headers, rows,
        formats={2: "0", 3: "0.0%", 4: "0", 5: "0.0000", 6: "0.0000", 7: "0.0000"},
        widths={"A": 30, "B": 13, "C": 12, "D": 22, "E": 14, "F": 12, "G": 12},
    )
    ws.Rows(row - len(rows) - 1).RowHeight = 42
    row += 1

    # 8.4 head-to-head vs union
    set_subtitle(ws, row, "8.4 union 与深度基线头对头（9 个场景的事件覆盖；正差值=union 更高）")
    row += 1
    headers = ["模型", "union 胜场景", "深度胜场景", "平局", "平均覆盖差（union−深度）"]
    rows = []
    for model in DEEP_ORDER:
        h = data["h2h"].loc[model]
        rows.append([
            label(model), int(h["union_wins"]), int(h["deep_wins"]), int(h["ties"]),
            _num(h["mean_union_minus_model"]),
        ])
    row = write_table(
        ws, row, headers, rows,
        formats={2: "0", 3: "0", 4: "0", 5: "+0.0000;-0.0000;0.0000"},
        widths={"A": 30, "B": 16, "C": 14, "D": 9, "E": 26},
    )
    ws.Rows(row - len(rows) - 1).RowHeight = 36
    row += 1

    # 8.5 resources / Table 2
    set_subtitle(ws, row, "8.5 单步资源与 Table 2 声称核对（i5-14600KF，单线程，batch=1，窗口 24）")
    row += 1
    headers = ["模型", "单步均值 ms", "P95 ms", "参数 MB", "进程峰值 MB",
               "论文时延声称", "论文内存声称", "时延是否复现", "内存是否复现（进程峰值）"]
    rows = []
    for model in DEEP_ORDER:
        d = data["resources"].loc[model]
        claim_ms = "—" if pd.isna(d["paper_latency_claim_ms"]) else f">{d['paper_latency_claim_ms']:.0f} ms"
        claim_mb = "—" if pd.isna(d["paper_memory_claim_mb"]) else f">{d['paper_memory_claim_mb']:.0f} MB"
        rows.append([
            label(model), _num(d["latency_mean_ms"], 3), _num(d["latency_p95_ms"], 3),
            _num(d["param_mb"]), _num(d["peak_wset_mb"], 1), claim_ms, claim_mb,
            "是" if bool(d["latency_claim_reproduced"]) else "否",
            "是" if bool(d["memory_claim_reproduced_process_peak"]) else "否",
        ])
    row = write_table(
        ws, row, headers, rows,
        formats={2: "0.000", 3: "0.000", 4: "0.0000", 5: "0.0"},
        widths={"A": 30, "B": 13, "C": 10, "D": 10, "E": 13, "F": 14, "G": 14, "H": 12, "I": 24},
    )
    ws.Rows(row - len(rows) - 1).RowHeight = 42
    row += 1

    # 8.6 exploratory bootstrap
    set_subtitle(ws, row, "8.6 探索性 95% 区间（provider-seed 重抽样 4000 次；不替代确认性区间）")
    row += 1
    headers = ["模型", "指标", "均值", "95% CI 下限", "95% CI 上限"]
    rows = []
    boot = data["boot"]
    for model in DEEP_ORDER:
        d = boot[boot["model"].eq(model)]
        for metric, metric_label in [("f1", "F1"), ("event_coverage", "事件覆盖率"), ("average_precision", "AP")]:
            m = d[d["metric"].eq(metric)].iloc[0]
            rows.append([label(model), metric_label, _num(m["mean"]), _num(m["ci95_lo"]), _num(m["ci95_hi"])])
    row = write_table(
        ws, row, headers, rows,
        formats={3: "0.0000", 4: "0.0000", 5: "0.0000"},
        widths={"A": 30, "B": 14, "C": 12, "D": 14, "E": 14},
    )
    ws.Rows(row - len(rows) - 1).RowHeight = 36
    row += 1

    # note
    note = (
        "注：深度基线为 EXPLORATORY_POST_V6（V6 test 已在 Stage349 检视，不能写成确认性）；"
        "模型只训练一次（seed 20260922），5 个 seed 是攻击生成种子，不是 5 次独立模型初始化；"
        "超参在 MeteoSwiss prelim 上选定，未做嵌套 CV；LSTM-AE 为 h64/z16/denoise 0.1 的降噪变体，不是原论文完整规模。"
        "KNMI union FAR=1.675%，是目标 0.65% 的 2.6 倍，不能把该机构覆盖差当作同 FAR 优势。"
        "Table 2 的 >100 ms/>200 ms/>500 MB 在本实现/本硬件下未复现；只有 LSTM-AE 进程峰值 >200 MB 成立，但参数仅 0.168 MB。"
    )
    ws.Range(ws.Cells(row, 1), ws.Cells(row, 9)).Merge()
    cell = ws.Cells(row, 1)
    cell.Value = note
    cell.Font.Name = "Arial"
    cell.Font.Size = 9
    cell.Font.Color = C_BODY_FONT
    cell.Interior.Color = C_NOTE_FILL
    cell.WrapText = True
    cell.VerticalAlignment = -4108
    ws.Rows(row).RowHeight = 92

    # One final width pass: several tables share columns, and the last table
    # would otherwise shrink the provider names used by 8.2.  Keep A wide
    # enough for `mlp_autoencoder_reconstruction`, and E wide enough for
    # `0.3600（Anomaly Transformer）`.
    for col, width in {
        "A": 38, "B": 14, "C": 14, "D": 20, "E": 30,
        "F": 22, "G": 18, "H": 18, "I": 26,
    }.items():
        ws.Columns(col).ColumnWidth = width
    return row


def main() -> int:
    wb_path = DEFAULT_WB
    backup = wb_path.with_name(wb_path.stem + f".backup_{datetime.now():%Y%m%d_%H%M%S}.xlsx")
    backup.write_bytes(wb_path.read_bytes())
    print(f"backup: {backup}")

    data = load_data()
    app = win32.DispatchEx("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    app.ScreenUpdating = False
    app.EnableEvents = False
    wb = None
    try:
        wb = app.Workbooks.Open(str(wb_path), UpdateLinks=0, ReadOnly=False)
        try:
            app.Calculation = -4135
        except Exception:
            pass
        ws = wb.Worksheets(SHEET)
        last = ws.UsedRange.Rows.Count
        start = None
        for r in range(1, last + 1):
            value = ws.Cells(r, 1).Value
            if isinstance(value, str) and value.startswith(MARKER):
                start = r
                break
        if start is not None:
            ws.Rows(f"{start}:{max(last, start + 120)}").Delete()
            last = ws.UsedRange.Rows.Count
        start = max(last + 2, 86)
        end = append_section(ws, start, data)
        wb.Save()
        print(f"updated {SHEET}: rows {start}-{end}, sheets={wb.Worksheets.Count}")
    finally:
        if wb is not None:
            try:
                wb.Close(SaveChanges=False)
            except Exception:
                pass
        try:
            app.Calculation = -4105
        except Exception:
            pass
        app.Quit()
        del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
