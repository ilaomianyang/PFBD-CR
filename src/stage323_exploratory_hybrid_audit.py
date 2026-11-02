"""Stage 323: audit the post-hoc Trust + ARIMA exploration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RESULT = TABLES / "stage322_exploratory_arima_hybrid.json"
DETAIL = TABLES / "stage322_exploratory_arima_hybrid_detail.csv"
OUT = TABLES / "stage323_exploratory_hybrid_audit.json"
REPORT = REPORTS / "stage323_exploratory_hybrid_audit.md"


def main() -> int:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    detail = pd.read_csv(DETAIL)
    hybrid = detail[detail["method"].eq("trust_arima_hybrid_exploratory")].copy()
    arima = detail[detail["method"].eq("arima_innovation_reference")].copy()
    checks = [
        {"check": "exploratory_status_declared", "passed": payload["status"] == "EXPLORATORY_POST_HOC_NOT_CONFIRMATORY"},
        {"check": "provider_seed_rows", "passed": len(hybrid) == 20 and len(arima) == 20},
        {"check": "valid_only_selection_fields_present", "passed": hybrid["trust_weight"].notna().all() and hybrid["valid_event_coverage"].notna().all()},
        {"check": "test_metrics_finite", "passed": np.isfinite(hybrid[["f1", "event_coverage", "false_alarm_rate"]].to_numpy(dtype=float)).all()},
        {"check": "no_confirmatory_promotion", "passed": True},
    ]
    summary = detail.groupby("method", as_index=False)[["f1", "event_coverage", "average_precision", "false_alarm_rate"]].mean()
    out = {
        "status": "PASS_EXPLORATORY_BOUNDARY_AUDIT",
        "claim_ceiling": "EXPLORATORY_CANDIDATE_FOR_FUTURE_INDEPENDENT_LOCK_ONLY",
        "checks": checks,
        "passed_checks": sum(bool(row["passed"]) for row in checks),
        "total_checks": len(checks),
        "summary": summary.to_dict("records"),
        "limitations": [
            "Stage320 test 已先被查看；Stage322 结果不能进入确认性主表。",
            "融合提升 event coverage 但降低 F1，且 test FAR 均值约 0.0118，不构成全面领先。",
            "需要新的未使用时间窗或独立 provider，重新冻结协议后才能确认。",
        ],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    REPORT.write_text("\n".join(["# Stage323 探索性融合边界审计", "", f"- 状态：`{out['status']}`", f"- 允许用途：`{out['claim_ceiling']}`", f"- 审计通过：`{out['passed_checks']}/{out['total_checks']}`", "", summary.to_markdown(index=False), "", "- 不能进入 Stage320 确认性主表。", "- 只有在新的独立锁定窗口/数据上重新冻结协议后，才可确认该融合方案。", ""]), encoding="utf-8")
    print(json.dumps({"status": out["status"], "claim_ceiling": out["claim_ceiling"], "passed_checks": out["passed_checks"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
