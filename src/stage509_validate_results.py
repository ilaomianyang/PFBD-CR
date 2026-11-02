"""Validate the Stage503/505/506/507 artifacts before they enter the workbook."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "tables"
WB = ROOT / "submission_final_20261009" / "final_experiment_evidence.xlsx"
MODELS = {"lstm_ae", "usad", "timesnet", "anomaly_transformer"}


def main() -> int:
    detail = pd.read_csv(T / "stage503_deep_baseline_detail.csv")
    scen = pd.read_csv(T / "stage503_deep_baseline_scenario.csv")
    summary = pd.read_csv(T / "stage503_deep_baseline_summary.csv")
    combined = pd.read_csv(T / "stage503_main_results_combined.csv")
    resources = pd.read_csv(T / "stage503_deep_baseline_resources.csv")
    boot = pd.read_csv(T / "stage507_deep_bootstrap.csv")

    assert len(detail) == 160, len(detail)
    assert len(scen) == 720, len(scen)
    assert len(summary) == 4, len(summary)
    assert len(combined) == 18, len(combined)
    assert len(resources) == 4, len(resources)
    assert len(boot) == 4 * 8, len(boot)

    test = detail[detail["split"].eq("test")]
    assert len(test) == 80, len(test)
    assert not test.duplicated(["source", "seed", "method"]).any()
    assert set(test["method"]) == MODELS
    assert test["source"].nunique() == 4
    assert test["seed"].nunique() == 5
    assert test["event_coverage"].between(0, 1).all()
    assert (test["false_alarm_rate"] < 0.02).all(), test[["method", "false_alarm_rate"]]
    assert (test["average_precision"].notna()).all()

    assert combined["method"].nunique() == 18
    assert MODELS.issubset(set(combined["method"]))
    assert (resources["peak_wset_mb"] > 0).all()
    assert (resources["latency_mean_ms"] > 0).all()

    wb = openpyxl.load_workbook(WB, data_only=True)
    assert len(wb.sheetnames) == 102, len(wb.sheetnames)
    ws = wb["论文直用数据"]
    assert ws.max_row == 152, ws.max_row
    assert str(ws["A86"].value).startswith("八、Table 4 扩表")
    assert ws["A89"].value == "方法" and ws["I89"].value == "证据状态"
    assert ws["A90"].value == "arima_innovation"
    assert ws["A103"].value == "trust_shape_union"
    assert abs(ws["E103"].value - 0.3559590643274853) < 1e-12
    assert abs(ws["D103"].value - 0.2202677796800844) < 1e-12
    assert ws["A104"].value == "lstm_ae"
    assert ws["A107"].value == "anomaly_transformer"
    assert abs(ws["E104"].value - 0.242520) < 5e-5
    assert ws["A109"].value.startswith("8.2 分机构")
    assert ws["A111"].value == "MeteoSwiss"
    assert ws["A114"].value == "NOAA ISD"
    assert ws["A116"].value.startswith("8.3 场景")
    assert ws["A118"].value == "LSTM-AE" and ws["B118"].value == 67
    assert ws["A123"].value.startswith("8.4 union")
    assert ws["B126"].value == 9  # USAD: union wins all nine scenarios
    assert ws["A130"].value.startswith("8.5 单步资源")
    assert ws["I132"].value == "是"  # only LSTM process peak exceeds the 200 MB claim
    assert ws["A137"].value.startswith("8.6 探索性")
    assert ws["A152"].value.startswith("注：深度基线为 EXPLORATORY_POST_V6")
    assert any(str(rng) == "A152:I152" for rng in ws.merged_cells.ranges)
    wb.close()

    print("detail", detail.shape, "test", test.shape, "scenario", scen.shape)
    print("summary\n", summary.to_string(index=False))
    print("resources\n", resources[["model", "param_mb", "latency_mean_ms", "latency_p95_ms", "peak_wset_mb"]].to_string(index=False))
    print("bootstrap f1/coverage\n", boot[boot["metric"].isin(["f1", "event_coverage"])].to_string(index=False))
    print("VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
