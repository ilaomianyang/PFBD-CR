from __future__ import annotations
"""Recompute Stage334 candidate win shares with the Stage328 audit denominator."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
IN_SCENARIO = TABLES / "stage334_exploratory_budgeted_ensemble_search_scenario.csv"
BASE_SCENARIO = TABLES / "stage326_simulated_attack_v2_scenario.csv"
OUT_JSON = TABLES / "stage335_corrected_ensemble_audit.json"
OUT_REPORT = REPORTS / "stage335_corrected_ensemble_audit.md"
BASELINES = ["robust_max_deviation", "isolation_forest", "one_class_svm", "pca_reconstruction", "mlp_autoencoder_reconstruction", "knn_distance", "ewma_deviation", "arima_innovation", "gru_predictor", "lightweight_autoencoder", "ics_cps_spc_proxy"]

def main() -> int:
    candidate = pd.read_csv(IN_SCENARIO)
    baseline = pd.read_csv(BASE_SCENARIO)
    candidate = candidate.groupby(["candidate", "scenario"], as_index=False)["event_coverage"].mean()
    baseline = baseline[baseline["method"].isin(BASELINES)].groupby(["method", "scenario"], as_index=False)["event_coverage"].mean()
    rows = []
    for name in sorted(candidate["candidate"].unique()):
        left = candidate[candidate["candidate"].eq(name)].rename(columns={"event_coverage": "candidate_event_coverage"})
        merged = left.merge(baseline.rename(columns={"event_coverage": "baseline_event_coverage"}), on="scenario", how="inner")
        merged["win"] = merged["candidate_event_coverage"] > merged["baseline_event_coverage"]
        share = float(merged["win"].mean()) if len(merged) else float("nan")
        rows.append({"candidate": name, "scenario_baseline_win_share_stage328_definition": share, "beaten_pairs": int(merged["win"].sum()), "total_pairs": int(len(merged))})
    summary = pd.DataFrame(rows).sort_values("scenario_baseline_win_share_stage328_definition", ascending=False)
    matrix = candidate.merge(baseline.rename(columns={"event_coverage": "baseline_event_coverage"}), on="scenario", how="inner")
    matrix["win"] = matrix["event_coverage"] > matrix["baseline_event_coverage"]
    payload = {"status": "EXPLORATORY_POST_HOC_NOT_CONFIRMATORY", "definition": "per-scenario mean event coverage, then candidate > baseline, matching Stage328", "summary": summary.to_dict("records"), "limitations": ["V2 test 已被查看；本审计只校正口径，不升级确认性结论。", "超过70%仍需新独立锁定窗口复核。"]}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_REPORT.write_text("\n".join(["# Stage335 预算式融合口径校正审计", "", f"- 状态：`{payload['status']}`", f"- 口径：{payload['definition']}", "", summary.to_markdown(index=False), "", "- 该结果只用于探索性诊断。", ""]), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
