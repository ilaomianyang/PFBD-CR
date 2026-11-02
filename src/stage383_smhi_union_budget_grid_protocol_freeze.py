from __future__ import annotations

"""Freeze an exploratory union-budget grid for the already-viewed SMHI holdout."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
EVALUATOR = ROOT / "src" / "stage372_smhi_factor_holdout_locked_evaluation.py"
OUT_PREFIX = "stage383_smhi_union_budget_grid_protocol"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    grids = [
        {"name": "current", "trust_valid_f1": 0.50, "ewma_deviation": 0.25, "arima_innovation": 0.25},
        {"name": "arima_half", "trust_valid_f1": 0.25, "ewma_deviation": 0.25, "arima_innovation": 0.50},
        {"name": "arima_dominant", "trust_valid_f1": 0.25, "ewma_deviation": 0.00, "arima_innovation": 0.75},
        {"name": "trust_arima", "trust_valid_f1": 0.50, "ewma_deviation": 0.00, "arima_innovation": 0.50},
    ]
    for item in grids:
        shares = [item["trust_valid_f1"], item["ewma_deviation"], item["arima_innovation"]]
        if abs(sum(shares) - 1.0) > 1e-12:
            raise RuntimeError(f"invalid budget grid item: {item}")
    payload = {
        "status": "EXPLORATORY_PROTOCOL_FROZEN_AFTER_HOLDOUT_VIEW",
        "protocol_id": "TMQC-SMHI-UNION-BUDGET-GRID-20260924-v1",
        "base_protocol": "TMQC-SMHI-THIRD-STATION-EVAL-20260924-v1",
        "exploratory": True,
        "reason": "The SMHI holdout outcome was already observed; this grid cannot produce confirmatory evidence.",
        "primary_far": 0.01,
        "endpoint": "event_coverage",
        "grid": grids,
        "evaluator_sha256": sha256(EVALUATOR),
        "claim_boundary": [
            "Reuse of the Stage380/Stage381 SMHI holdout is exploratory only.",
            "No test result may be used to claim confirmatory superiority.",
            "All attacks remain controlled simulations, not production incidents.",
        ],
    }
    output = TABLES / f"{OUT_PREFIX}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPORTS / f"{OUT_PREFIX}.md").write_text(
        "\n".join(
            [
                "# Stage383 SMHI union budget grid protocol",
                "",
                f"- 状态：`{payload['status']}`",
                "- 这是在已查看 SMHI holdout 结果之后进行的探索性网格，不可升级为确认性结论。",
                "- 主终点：`event_coverage`；主 FAR：`0.01`。",
                "",
                "| name | Trust | EWMA | ARIMA |",
                "|---|---:|---:|---:|",
                *[
                    f"| {item['name']} | {item['trust_valid_f1']:.2f} | {item['ewma_deviation']:.2f} | {item['arima_innovation']:.2f} |"
                    for item in grids
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "grid_count": len(grids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
