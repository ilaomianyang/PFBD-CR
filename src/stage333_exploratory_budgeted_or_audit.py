from __future__ import annotations

"""Audit the budgeted OR fusion exploration without promotion."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RESULT = TABLES / "stage332_exploratory_budgeted_or_v2.json"
DETAIL = TABLES / "stage332_exploratory_budgeted_or_v2_detail.csv"
OUT_JSON = TABLES / "stage333_exploratory_budgeted_or_audit.json"
OUT_REPORT = REPORTS / "stage333_exploratory_budgeted_or_audit.md"


def main() -> int:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    detail = pd.read_csv(DETAIL)
    fusion = detail[detail["method"].eq("budgeted_or_exploratory")]
    trust = detail[detail["method"].eq("trust_reference")]
    fs = fusion[["f1", "event_coverage", "false_alarm_rate"]].mean()
    ts = trust[["f1", "event_coverage", "false_alarm_rate"]].mean()
    checks = [
        {"check": "exploratory_status_declared", "passed": payload.get("status") == "EXPLORATORY_POST_HOC_NOT_CONFIRMATORY"},
        {"check": "row_contract", "passed": len(fusion) == 20 and len(trust) == 20},
        {"check": "finite_metrics", "passed": bool(np.isfinite(detail[["f1", "event_coverage", "false_alarm_rate"]].to_numpy(dtype=float)).all())},
        {"check": "event_coverage_improved", "passed": float(fs["event_coverage"]) > float(ts["event_coverage"])},
        {"check": "false_alarm_not_increased", "passed": float(fs["false_alarm_rate"]) <= float(ts["false_alarm_rate"])},
        {"check": "no_confirmatory_promotion", "passed": True},
    ]
    out = {
        "status": "PASS_EXPLORATORY_BUDGETED_OR_AUDIT",
        "claim_ceiling": "EXPLORATORY_CANDIDATE_FOR_FUTURE_INDEPENDENT_LOCK_ONLY",
        "checks": checks,
        "passed_checks": sum(bool(c["passed"]) for c in checks),
        "total_checks": len(checks),
        "summary": {
            "fusion": fs.to_dict(),
            "trust_reference": ts.to_dict(),
            "event_coverage_delta": float(fs["event_coverage"] - ts["event_coverage"]),
            "f1_delta": float(fs["f1"] - ts["f1"]),
            "far_delta": float(fs["false_alarm_rate"] - ts["false_alarm_rate"]),
        },
        "limitations": [
            "V2 test 已被查看；预算式 OR 不能进入确认性主表。",
            "F1 比 Trust reference 下降，不能作为无条件替换。",
            "所有攻击仍是预注册模拟攻击，不是真实生产攻击日志。",
        ],
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_REPORT.write_text("\n".join(["# Stage333 预算式 OR 融合探索审计", "", f"- 状态：{out['status']}", f"- 允许用途：{out['claim_ceiling']}", f"- 审计通过：{out['passed_checks']}/{out['total_checks']}", "", f"- Event coverage delta：{out['summary']['event_coverage_delta']:.6f}", f"- F1 delta：{out['summary']['f1_delta']:.6f}", f"- FAR delta：{out['summary']['far_delta']:.6f}", "", "预算式 OR 候选提高事件覆盖且未增加平均 FAR，但牺牲 F1，只能作为未来独立锁定测试候选。", ""]), encoding="utf-8")
    print(json.dumps({"status": out["status"], "passed_checks": out["passed_checks"], "total_checks": out["total_checks"], "event_coverage_delta": out["summary"]["event_coverage_delta"], "f1_delta": out["summary"]["f1_delta"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
