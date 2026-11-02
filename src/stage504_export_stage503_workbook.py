"""Append Stage503 deep-baseline results to the frozen evidence workbook.

The workbook already contains 95 sheets of V6 evidence.  This exporter only
adds new ``探索503_*`` sheets and updates the sheet index; it never rewrites
the frozen V6 sheets.  Excel COM is used so comments, VML and the existing
formatting survive unchanged.
"""
from __future__ import annotations

import argparse
import math
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
DEFAULT_WB = ROOT / "submission_final_20261009" / "final_experiment_evidence.xlsx"

HEADERS = {
    "detail": [
        "source（来源）", "seed（随机种子）", "split（切分）", "method（方法）",
        "threshold（阈值）", "rows（行数）", "attack_rows（攻击样本行数）",
        "precision（精确率）", "recall（召回率）", "f1（F1分数）",
        "false_alarm_rate（误报率）", "roc_auc（ROC曲线下面积）",
        "average_precision（平均精确率）", "events（事件数）",
        "detected_events（检出事件数）", "event_coverage（事件覆盖率）",
        "row_recall（行 召回率）", "valid_only（仅验证集）",
    ],
    "summary": [
        "method（方法）", "precision（精确率）", "recall（召回率）", "f1（F1分数）",
        "false_alarm_rate（误报率）", "roc_auc（ROC曲线下面积）",
        "average_precision（平均精确率）", "event_coverage（事件覆盖率）",
        "row_recall（行 召回率）", "runs（运行数）", "family（类别）",
    ],
    "scenario": [
        "source（来源）", "seed（随机种子）", "method（方法）", "scenario（场景）",
        "rows（行数）", "attack_rows（攻击样本行数）", "tp（真正例）",
        "fp（假正例）", "fn（假负例）", "tn（真负例）", "precision（精确率）",
        "recall（召回率）", "row_recall（行 召回率）", "f1（F1分数）",
        "false_alarm_rate（误报率）", "roc_auc（ROC曲线下面积）",
        "average_precision（平均精确率）", "events（事件数）",
        "detected_events（检出事件数）", "event_coverage（事件覆盖率）",
        "mean_score_attack（攻击行均分）",
    ],
    "resources": [
        "method（方法）", "family（类别）", "implementation（实现）",
        "parameter_count（参数数量）", "param_mb（参数MB）",
        "latency_mean_ms（单步均值毫秒）", "latency_p50_ms（单步P50毫秒）",
        "latency_p95_ms（单步P95毫秒）", "latency_max_ms（单步最大毫秒）",
        "peak_wset_mb（进程峰值工作集MB）", "rss_before_model_mb（模型前RSS MB）",
        "rss_after_model_mb（模型后RSS MB）", "threads（线程数）",
        "length（窗口长度）", "features（特征数）", "processor（处理器）",
        "platform（平台）", "status（状态）",
    ],
    "bootstrap": [
        "model（模型）", "label（标签）", "metric（指标）", "mean（均值）",
        "ci95_lo（95%CI下限）", "ci95_hi（95%CI上限）", "runs（运行数）",
        "method（方法）", "status（状态）",
    ],
    "combined": [
        "source（来源）", "method（方法）", "precision（精确率）",
        "recall（召回率）", "f1（F1分数）", "false_alarm_rate（误报率）",
        "roc_auc（ROC曲线下面积）", "average_precision（平均精确率）",
        "event_coverage（事件覆盖率）", "row_recall（行 召回率）", "runs（运行数）",
    ],
}

DETAIL_COLS = [
    "source", "seed", "split", "method", "threshold", "rows", "attack_rows",
    "precision", "recall", "f1", "false_alarm_rate", "roc_auc",
    "average_precision", "events", "detected_events", "event_coverage",
    "row_recall", "valid_only",
]
SUMMARY_COLS = [
    "method", "precision", "recall", "f1", "false_alarm_rate", "roc_auc",
    "average_precision", "event_coverage", "row_recall", "runs", "family",
]
SCENARIO_COLS = [
    "source", "seed", "method", "scenario", "rows", "attack_rows", "tp", "fp",
    "fn", "tn", "precision", "recall", "row_recall", "f1", "false_alarm_rate",
    "roc_auc", "average_precision", "events", "detected_events",
    "event_coverage", "mean_score_attack",
]
RESOURCE_COLS = [
    "method", "family", "implementation", "parameter_count", "param_mb",
    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_max_ms",
    "peak_wset_mb", "rss_before_model_mb", "rss_after_model_mb", "threads",
    "length", "features", "processor", "platform", "status",
]
BOOTSTRAP_COLS = [
    "model", "label", "metric", "mean", "ci95_lo", "ci95_hi", "runs", "method", "status",
]
COMBINED_COLS = [
    "source", "method", "precision", "recall", "f1", "false_alarm_rate",
    "roc_auc", "average_precision", "event_coverage", "row_recall", "runs",
]


def clean(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def read_table(path: Path | None, cols: list[str]):
    if path is None or not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame.reindex(columns=cols)
    return [[clean(v) for v in row] for row in frame.itertuples(index=False, name=None)]


def metric_columns(headers: list[str]) -> list[int]:
    hits = []
    for i, name in enumerate(headers, start=1):
        if any(key in name for key in ("precision", "recall", "f1", "false_alarm",
                                       "roc_auc", "average_precision", "event_coverage",
                                       "threshold", "mean_score")):
            hits.append(i)
    return hits


def write_sheet(wb, name, headers, rows, widths=None):
    try:
        wb.Worksheets(name).Delete()
    except Exception:
        pass
    sheet = wb.Worksheets.Add(After=wb.Worksheets(wb.Worksheets.Count))
    sheet.Name = name
    data = [headers] + rows
    last_row, last_col = len(data), len(headers)
    target = sheet.Range(sheet.Cells(1, 1), sheet.Cells(last_row, last_col))
    target.Value = tuple(tuple(r) for r in data)
    header = sheet.Range(sheet.Cells(1, 1), sheet.Cells(1, last_col))
    header.Font.Bold = True
    header.Interior.Color = 15921906
    header.HorizontalAlignment = -4108
    header.VerticalAlignment = -4108
    header.WrapText = True
    sheet.Rows(1).RowHeight = 30
    sheet.Range(sheet.Cells(1, 1), sheet.Cells(last_row, last_col)).AutoFilter()
    for i in metric_columns(headers):
        sheet.Range(sheet.Cells(2, i), sheet.Cells(last_row, i)).NumberFormat = "0.000000"
    if widths:
        for i, width in enumerate(widths, start=1):
            sheet.Columns(i).ColumnWidth = width
    try:
        sheet.Activate()
        sheet.Range("A2").Select()
        wb.Application.ActiveWindow.FreezePanes = True
    except Exception:
        pass
    return sheet


def frozen_summary(wb):
    """Read the 14 frozen V6 method rows from the workbook itself."""
    sheet = wb.Worksheets("V6方法指标汇总")
    values = sheet.Range(sheet.Cells(2, 1), sheet.Cells(sheet.UsedRange.Rows.Count, 9)).Value
    if values is None:
        return []
    if not isinstance(values, tuple):
        values = ((values,),)
    rows = []
    for row in values:
        if not row or row[0] in (None, ""):
            continue
        rows.append([
            "V6_frozen（Stage349）", row[0], row[1], row[2], row[3], row[4],
            row[5], row[6], row[7], row[8], 20,
        ])
    return rows


def update_index(wb, new_rows):
    sheet = wb.Worksheets("工作表索引")
    last = sheet.Cells(sheet.Rows.Count, 2).End(-4162).Row  # xlUp
    # Remove previous Stage503 rows so reruns stay idempotent.
    for r in range(last, 1, -1):
        name = sheet.Cells(r, 2).Value
        if isinstance(name, str) and name.startswith("探索503_"):
            sheet.Rows(r).Delete()
            last -= 1
    if new_rows:
        sheet.Range(sheet.Cells(last + 1, 1), sheet.Cells(last + len(new_rows), 6)).Value = tuple(
            tuple(r) for r in new_rows
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WB)
    parser.add_argument("--detail", type=Path, default=TABLES / "stage503_deep_baseline_detail.csv")
    parser.add_argument("--summary", type=Path, default=TABLES / "stage503_deep_baseline_summary.csv")
    parser.add_argument("--scenario", type=Path, default=TABLES / "stage503_deep_baseline_scenario.csv")
    parser.add_argument("--resources", type=Path, default=TABLES / "stage503_deep_baseline_resources.csv")
    parser.add_argument("--combined", type=Path, default=TABLES / "stage503_main_results_combined.csv")
    parser.add_argument("--bootstrap", type=Path, default=TABLES / "stage507_deep_bootstrap.csv")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if not args.workbook.exists():
        raise SystemExit(f"workbook not found: {args.workbook}")
    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = args.workbook.with_name(f"{args.workbook.stem}.backup_{stamp}{args.workbook.suffix}")

        shutil.copy2(args.workbook, backup)
        print(f"backup: {backup}")

    detail = read_table(args.detail, DETAIL_COLS)
    summary = read_table(args.summary, SUMMARY_COLS)
    scenario = read_table(args.scenario, SCENARIO_COLS)
    resources = read_table(args.resources, RESOURCE_COLS)
    combined_extra = read_table(args.combined, COMBINED_COLS)
    bootstrap = read_table(args.bootstrap, BOOTSTRAP_COLS)

    import win32com.client as win32

    app = win32.DispatchEx("Excel.Application")
    app.Visible = False
    app.DisplayAlerts = False
    app.ScreenUpdating = False
    app.EnableEvents = False
    app.AskToUpdateLinks = False
    wb = None
    try:
        wb = app.Workbooks.Open(str(args.workbook.resolve()), UpdateLinks=0, ReadOnly=False)
        try:
            app.Calculation = -4135  # xlCalculationManual
        except Exception:
            pass
        frozen = frozen_summary(wb)
        if combined_extra:
            deep = {"lstm_ae", "usad", "timesnet", "anomaly_transformer"}
            combined = []
            for row in combined_extra:
                source = row[0] or ("Stage503_deep" if row[1] in deep else "V6_frozen（Stage349）")
                recall = row[3] if row[3] != "" else row[9]
                combined.append([source, row[1], row[2], recall, row[4], row[5],
                                 row[6], row[7], row[8], row[9], 20])
        else:
            combined = frozen

        write_sheet(wb, "探索503_深度基线_明细", HEADERS["detail"], detail,
                    widths=[12, 10, 8, 18] + [15] * 14)
        write_sheet(wb, "探索503_深度基线_汇总", HEADERS["summary"], summary,
                    widths=[20] + [15] * 9 + [8, 20])
        write_sheet(wb, "探索503_深度基线_场景", HEADERS["scenario"], scenario,
                    widths=[12, 10, 18, 18, 9, 12] + [9] * 4 + [14] * 11)
        if resources:
            write_sheet(wb, "探索503_深度基线_资源", HEADERS["resources"], resources,
                        widths=[18, 18, 34] + [13] * 15)
        write_sheet(wb, "探索503_深度基线_合并汇总", HEADERS["combined"], combined,
                    widths=[20, 20] + [15] * 8 + [8])
        if bootstrap:
            write_sheet(wb, "探索503_深度基线_置信区间", HEADERS["bootstrap"], bootstrap,
                        widths=[18, 20, 18, 13, 15, 15, 9, 28, 24])

        notes = [
            ["阶段", "Stage503 深度异常检测基线（探索性补充）"],
            ["状态", "EXPLORATORY_POST_V6：V6 测试集已在 Stage349 检视，此补充不替代 V6 主证据"],
            ["协议", "V6 protocol_freeze.json；9 场景；5 seeds；α=0.0065；valid normal 行定阈值"],
            ["特征空间", "16 维：3 个 trust 分量 + 13 个候选特征（窗口 2/3/6）"],
            ["模型", "LSTM-AE（hidden 64/latent 16, denoise 0.1, 30 epoch）；USAD（latent 8, min-max 输入, 10 epoch）；TimesNet（d_model 32, top-3 period, 20 epoch）；Anomaly Transformer（d 64, 4 heads, 1 层, 20 epoch）"],
            ["训练", "每提供方训练一次；train+observed 滑窗 20,000 个，窗口长度 24；模型种子 20260922"],
            ["阈值", "仅在 valid normal 行上取 1-α 分位数；不使用测试标签；不做 point-adjustment"],
            ["延迟口径", "batch=1；torch 单线程；warmup 200；计时 2000 步；单位 ms"],
            ["内存口径", "进程峰值工作集（含 Python/PyTorch 运行时）；参数 MB 单独列出，不能把进程峰值等同于模型权重"],
            ["硬件", "Intel Core i5-14600KF；31.79 GB RAM；Windows；非论文声称的 i7-12700H"],
            ["结果范围", "支持“预注册透明模拟攻击 + 低 FAR 约束下的场景×基线多数优势”；不支持生产攻击召回、跨机构普适或在线=离线等价"],
            ["生成脚本", "src/stage503_deep_anomaly_baselines.py；src/stage504_export_stage503_workbook.py"],
        ]
        write_sheet(wb, "探索503_深度基线_说明", ["项目（项目）", "内容（内容）"], notes,
                    widths=[16, 110])

        index_rows = [
            ["stage503_deep_baseline_detail", "探索503_深度基线_明细", len(detail), len(HEADERS["detail"]),
             "探索性补充（V6 已检视）", "4 模型 × 4 提供方 × 5 seeds × valid/test 的逐次指标"],
            ["stage503_deep_baseline_summary", "探索503_深度基线_汇总", len(summary), len(HEADERS["summary"]),
             "探索性补充（V6 已检视）", "provider-seed 等权汇总的深度基线指标"],
            ["stage503_deep_baseline_scenario", "探索503_深度基线_场景", len(scenario), len(HEADERS["scenario"]),
             "探索性补充（V6 已检视）", "逐场景混淆矩阵与事件覆盖"],
            ["stage503_deep_baseline_resources", "探索503_深度基线_资源", len(resources), len(HEADERS["resources"]),
             "探索性补充（V6 已检视）", "单步时延与进程峰值内存（实测硬件）"],
            ["stage503_main_results_combined", "探索503_深度基线_合并汇总", len(combined), len(HEADERS["combined"]),
             "探索性补充（V6 已检视）", "V6 冻结 14 方法 + 4 个深度基线的合并主结果"],
            ["stage507_deep_bootstrap", "探索503_深度基线_置信区间", len(bootstrap), len(HEADERS["bootstrap"]),
             "探索性补充（V6 已检视）", "provider-seed 重抽样 4000 次的探索性 95% 区间"],
            ["stage503_deep_baseline_notes", "探索503_深度基线_说明", len(notes), 2,
             "探索性补充（V6 已检视）", "协议、特征、训练、阈值、硬件与主张边界"],
        ]
        update_index(wb, index_rows)
        wb.Save()
        print(f"saved: {args.workbook}")
        for name in [s.Name for s in wb.Worksheets]:
            if name.startswith("探索503_"):
                ws = wb.Worksheets(name)
                print(f"  {name}: {ws.UsedRange.Rows.Count} rows x {ws.UsedRange.Columns.Count} cols")
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


if __name__ == "__main__":
    main()
