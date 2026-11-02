from __future__ import annotations

"""Stage 466: audit the algorithm-optimization 70% goal.

This stage is read-only. It consumes the existing V6 scenario audit and the
station-level bootstrap/BH audit. It does not refit a detector, inspect raw
observations, change thresholds, or promote a post-hoc V6 candidate to a new
confirmatory result.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
OUT = TABLES / "stage466_algorithm_optimization_70pct_goal_audit.json"
REPORT = REPORTS / "stage466_algorithm_optimization_70pct_goal_audit.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    stage350_path = TABLES / "stage350_v6_locked_audit.json"
    stage351_path = TABLES / "stage351_v6_station_bh_audit.json"
    stage350 = json.loads(stage350_path.read_text(encoding="utf-8"))
    stage351 = json.loads(stage351_path.read_text(encoding="utf-8"))

    best = stage350["best_candidate"]
    selected = stage350["selected_summary"]
    gate350 = stage350["gate"]
    gate351 = stage351["gate"]

    checks = [
        {
            "check": "scenario_baseline_share_ge_70pct",
            "passed": float(best["scenario_baseline_win_share"]) >= 0.70,
            "value": {
                "beaten_pairs": int(best["beaten_pairs"]),
                "total_pairs": int(best["total_pairs"]),
                "share": float(best["scenario_baseline_win_share"]),
            },
        },
        {
            "check": "selected_far_within_001",
            "passed": bool(gate350["selected_mean_far_le_0.01"]),
            "value": float(selected["false_alarm_rate"]),
        },
        {
            "check": "provider_seed_ci_gate",
            "passed": bool(gate350["best_candidate_all_provider_seed_ci_low_positive"]),
        },
        {
            "check": "station_bootstrap_ci_gate",
            "passed": bool(gate351["best_candidate_all_station_ci_low_positive"]),
        },
        {
            "check": "station_level_bh_gate",
            "passed": bool(gate351["best_candidate_all_bh_q_lt_0.05"]),
        },
        {
            "check": "posthoc_status_preserved",
            "passed": True,
            "value": "V6 test 已查看；该算法候选只能作为探索性优化结果。",
        },
    ]
    result = {
        "stage": 466,
        "status": (
            "PASS_ALGORITHM_OPTIMIZATION_70PCT_EXPLORATORY"
            if all(row["passed"] for row in checks)
            else "FAIL_ALGORITHM_OPTIMIZATION_70PCT"
        ),
        "candidate": best["candidate"],
        "scenario_baseline_win_share": float(best["scenario_baseline_win_share"]),
        "beaten_pairs": int(best["beaten_pairs"]),
        "total_pairs": int(best["total_pairs"]),
        "selected_summary": selected,
        "checks": checks,
        "confirmatory_promotion": False,
        "claim_ceiling": (
            "V6 旧 test 上的探索性算法优化达到 70% 门槛；"
            "不得升级为新的确认性主结果或未见测试结论。"
        ),
        "input_sha256": {
            str(stage350_path): sha256_file(stage350_path),
            str(stage351_path): sha256_file(stage351_path),
        },
        "code_sha256": sha256_file(Path(__file__)),
        "rule": (
            "只读取已有 V6 审计产物；不训练、不调参、不改变分母、"
            "不删除不利场景、不读取原始观测值。"
        ),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage466 算法优化 70% 目标审计",
        "",
        f"- 状态：`{result['status']}`",
        f"- 候选：`{result['candidate']}`",
        f"- 场景×基线胜出：`{result['beaten_pairs']}/{result['total_pairs']} = {result['scenario_baseline_win_share']:.1%}`",
        f"- 选择运行 F1 / event coverage / FAR：`{selected['f1']:.6f}` / "
        f"`{selected['event_coverage']:.6f}` / `{selected['false_alarm_rate']:.6f}`",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | `{row['passed']}` |")
    lines += [
        "",
        "## 证据边界",
        "",
        "- 场景×基线 70% 目标在现有 V6 结果上达到，且站点级 bootstrap、BH 和 FAR 门槛通过。",
        "- V6 test 已被查看，`trust_shape_union` 属于事后探索性算法候选。",
        "- 本审计不把该结果升级为新的确认性主结果，不支持未见测试或真实生产攻击结论。",
        f"- 输入：`{stage350_path}`、`{stage351_path}`",
        f"- JSON：`{OUT}`",
        f"- 报告：`{REPORT}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "status": result["status"],
                "candidate": result["candidate"],
                "scenario_baseline_win_share": result["scenario_baseline_win_share"],
                "beaten_pairs": result["beaten_pairs"],
                "total_pairs": result["total_pairs"],
                "confirmatory_promotion": result["confirmatory_promotion"],
                "output": str(OUT),
                "report": str(REPORT),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
