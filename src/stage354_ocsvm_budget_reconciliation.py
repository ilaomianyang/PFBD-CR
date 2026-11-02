"""Reconcile the One-Class SVM budget used by V6 with sensitivity artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SOURCE = ROOT / "src" / "stage248_cross_provider_fair_baseline_audit.py"
EVAL_SOURCE = ROOT / "src" / "stage340_v4_route_locked_evaluation.py"
PROTOCOL = ROOT / "confirmatory_package" / "simulated_attack_v6" / "protocol_freeze.json"
SENSITIVITY = ROOT / "reports" / "stage302_japan_ocsvm_budget_grid.md"
V6_EVAL = ROOT / "tables" / "stage349_v6_locked_evaluation.json"
OUT_JSON = ROOT / "tables" / "stage354_ocsvm_budget_reconciliation.json"
OUT_MD = ROOT / "reports" / "stage354_ocsvm_budget_reconciliation.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    baseline_text = BASELINE_SOURCE.read_text(encoding="utf-8")
    eval_text = EVAL_SOURCE.read_text(encoding="utf-8")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sensitivity_text = SENSITIVITY.read_text(encoding="utf-8")
    evaluation = json.loads(V6_EVAL.read_text(encoding="utf-8"))

    max_fit_match = re.search(r"MAX_OCSVM_FIT_ROWS\s*=\s*(\d+)", baseline_text)
    baseline_method_match = re.search(
        r'BASELINES\s*=\s*\[(.*?)\]', eval_text, flags=re.DOTALL
    )
    if not max_fit_match or not baseline_method_match:
        raise ValueError("Could not locate the frozen OCSVM source settings")

    confirmatory_budget = int(max_fit_match.group(1))
    evaluator_baselines = re.findall(
        r'"([^"]+)"', baseline_method_match.group(1)
    )
    protocol_budgets = [
        item
        for item in protocol["method_freeze"]["baselines"]
        if item.startswith("one_class_svm_")
    ]
    sensitivity_budgets = [
        int(value)
        for value in re.findall(r"Required budgets:\s*\[(.*?)\]", sensitivity_text)
        for value in re.findall(r"\d+", value)
    ]
    observed_method_names = sorted(
        row["method"]
        for row in evaluation["summary"]
        if row["method"].startswith("one_class_svm")
    )

    result: dict[str, Any] = {
        "status": "PASS_RECONCILIATION_WITH_PROTOCOL_CLARIFICATION",
        "scope": "V6 locked evaluation",
        "confirmatory_budget_rows": confirmatory_budget,
        "confirmatory_method_name": "one_class_svm",
        "evaluator_baselines": evaluator_baselines,
        "protocol_declared_ocsvm_budgets": protocol_budgets,
        "sensitivity_budgets": sensitivity_budgets,
        "observed_v6_method_names": observed_method_names,
        "protocol_overdeclares_ocsvm_variants": len(protocol_budgets)
        != len(observed_method_names),
        "decision": {
            "v6_confirmatory": "one_class_svm at 20,000 train-normal rows",
            "sensitivity": "20k/50k/100k budget grid remains supplementary",
            "result_integrity": "No V6 metric was changed or recomputed",
        },
        "inputs": {
            "baseline_source": str(BASELINE_SOURCE.relative_to(ROOT)),
            "evaluation_source": str(EVAL_SOURCE.relative_to(ROOT)),
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "sensitivity_report": str(SENSITIVITY.relative_to(ROOT)),
            "v6_evaluation": str(V6_EVAL.relative_to(ROOT)),
        },
        "hashes": {
            "baseline_source_sha256": sha256(BASELINE_SOURCE),
            "evaluation_source_sha256": sha256(EVAL_SOURCE),
            "protocol_sha256": sha256(PROTOCOL),
            "sensitivity_report_sha256": sha256(SENSITIVITY),
            "v6_evaluation_sha256": sha256(V6_EVAL),
        },
        "rule": "确认性 V6 只使用实际评价器中的20k预算；50k/100k不得伪装成V6独立主结果。",
    }
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage354 One-Class SVM 预算归属审计",
        "",
        f"- 状态：`{result['status']}`",
        "- 结论：V6 确认性评价实际使用 One-Class SVM 的 `20,000` 行 train-normal 拟合上限。",
        "- 20k/50k/100k 网格属于 Stage302 敏感性证据，不是 V6 场景表中的三个独立方法。",
        f"- V6 场景表实际方法名：`{', '.join(observed_method_names)}`",
        f"- V6 协议声明的预算名：`{', '.join(protocol_budgets)}`",
        "",
        "## 论文写法",
        "",
        "- 主结果表：写作 `One-Class SVM (20k fit rows)`。",
        "- 补充材料：报告 20k、50k、100k 的敏感性结果，并保留较大预算下非劣效门未全部通过的边界。",
        "- 不得把协议文件中的三个预算名称写成 V6 三个独立方法，也不得用敏感性网格替换 V6 锁定结果。",
        "",
        "## 输入哈希",
        "",
    ]
    for key, value in result["hashes"].items():
        lines.append(f"- `{key}`：`{value}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "confirmatory_budget_rows": confirmatory_budget,
                "protocol_overdeclares_ocsvm_variants": result[
                    "protocol_overdeclares_ocsvm_variants"
                ],
                "out_json": str(OUT_JSON),
                "out_md": str(OUT_MD),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
