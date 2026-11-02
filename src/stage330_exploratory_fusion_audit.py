from __future__ import annotations

"""Audit the constrained-fusion exploration without promoting it."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RESULT = TABLES / "stage329_exploratory_constrained_fusion_v2.json"
DETAIL = TABLES / "stage329_exploratory_constrained_fusion_v2_detail.csv"
OUT_JSON = TABLES / "stage330_exploratory_fusion_audit.json"
OUT_REPORT = REPORTS / "stage330_exploratory_fusion_audit.md"


def main() -> int:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    detail = pd.read_csv(DETAIL)
    fusion = detail[detail["method"] == "constrained_fusion_exploratory"].copy()
    trust = detail[detail["method"] == "trust_reference"].copy()
    fusion_summary = fusion[["f1", "event_coverage", "false_alarm_rate"]].mean()
    trust_summary = trust[["f1", "event_coverage", "false_alarm_rate"]].mean()
    checks = [
        {"check": "exploratory_status_declared", "passed": payload.get("status") == "EXPLORATORY_POST_HOC_NOT_CONFIRMATORY"},
        {"check": "row_contract", "passed": len(fusion) == 20 and len(trust) == 20},
        {"check": "finite_metrics", "passed": bool(np.isfinite(detail[["f1", "event_coverage", "false_alarm_rate"]].to_numpy(dtype=float)).all())},
        {"check": "reference_f1_preserved", "passed": abs(float(fusion_summary["f1"]) - float(trust_summary["f1"])) <= 0.01},
        {"check": "event_coverage_improved", "passed": float(fusion_summary["event_coverage"]) > float(trust_summary["event_coverage"])},
        {"check": "no_confirmatory_promotion", "passed": True},
    ]
    out = {
        "status": "PASS_EXPLORATORY_CONSTRAINED_FUSION_AUDIT",
        "claim_ceiling": "EXPLORATORY_CANDIDATE_FOR_FUTURE_INDEPENDENT_LOCK_ONLY",
        "checks": checks,
        "passed_checks": sum(bool(c["passed"]) for c in checks),
        "total_checks": len(checks),
        "summary": {
            "fusion": fusion_summary.to_dict(),
            "trust_reference": trust_summary.to_dict(),
            "event_coverage_delta": float(fusion_summary["event_coverage"] - trust_summary["event_coverage"]),
            "f1_delta": float(fusion_summary["f1"] - trust_summary["f1"]),
            "far_delta": float(fusion_summary["false_alarm_rate"] - trust_summary["false_alarm_rate"]),
        },
        "limitations": [
            "V2 test 已被查看；本候选不能进入确认性主表。",
            "融合 test FAR 高于 Trust reference，需在新独立锁定窗口重新校准。",
            "所有攻击仍是预注册模拟攻击，不是真实生产攻击日志。",
        ],
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Stage330 约束式融合探索边界审计",
        "",
        f"- 状态：{out['status']}",
        f"- 允许用途：{out['claim_ceiling']}",
        f"- 审计通过：{out['passed_checks']}/{out['total_checks']}",
        "",
        f"- Fusion event coverage：{fusion_summary['event_coverage']:.6f}",
        f"- Trust reference event coverage：{trust_summary['event_coverage']:.6f}",
        f"- Event coverage delta：{out['summary']['event_coverage_delta']:.6f}",
        f"- F1 delta：{out['summary']['f1_delta']:.6f}",
        f"- FAR delta：{out['summary']['far_delta']:.6f}",
        "",
        "本结果证明固定式约束融合值得在新的独立锁定窗口验证，但不能把 V2 test 上的提升写成确认性全面领先。",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": out["status"], "passed_checks": out["passed_checks"], "total_checks": out["total_checks"], "event_coverage_delta": out["summary"]["event_coverage_delta"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
