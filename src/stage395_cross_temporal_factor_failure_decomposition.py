from __future__ import annotations

"""Read-only cross-temporal factor failure decomposition.

This stage compares already generated exploratory factor audits from MET
Norway 2026 and FMI 2026.  It does not fit models, tune thresholds, or
promote either temporal window to confirmatory evidence.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

INPUTS = {
    "MET_NORWAY_2026": {
        "scenario": TABLES
        / "stage388_met_norway_2026_partial_factor_audit_scenario_win_loss.csv",
        "audit": TABLES / "stage388_met_norway_2026_partial_factor_audit.json",
    },
    "FMI_2026_PARTIAL": {
        "scenario": TABLES
        / "stage394_fmi_2026_partial_factor_audit_scenario_win_loss.csv",
        "audit": TABLES / "stage394_fmi_2026_partial_factor_audit.json",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    frames: list[pd.DataFrame] = []
    source_meta: dict[str, dict[str, object]] = {}
    for source, paths in INPUTS.items():
        frame = pd.read_csv(paths["scenario"])
        frame = frame[frame["candidate"].eq("trust_shape_union")].copy()
        frame["temporal_source"] = source
        frames.append(frame)
        audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
        source_meta[source] = {
            "audit_status": audit["status"],
            "scenario_wins": audit["scenario_win_share"]["wins"],
            "scenario_total": audit["scenario_win_share"]["total"],
            "unseen_event_coverage": audit["unseen_combination_event_coverage"],
            "statistical_event_gate": audit["statistical_event_gate"],
            "scenario_sha256": sha256(paths["scenario"]),
            "audit_sha256": sha256(paths["audit"]),
        }

    combined = pd.concat(frames, ignore_index=True)
    # Keep the explicit pivot separate so missing/duplicate rows cannot be
    # silently interpreted as a direction.
    pivot = combined.pivot_table(
        index=["baseline", "scenario"],
        columns="temporal_source",
        values="difference",
        aggfunc="first",
    ).reset_index()
    required_sources = {"MET_NORWAY_2026", "FMI_2026_PARTIAL"}
    if not required_sources.issubset(pivot.columns):
        raise ValueError("cross-temporal factor inputs are incomplete")
    pivot = pivot.dropna(subset=sorted(required_sources)).copy()
    pivot["stable_positive"] = (
        (pivot["MET_NORWAY_2026"] > 0.0)
        & (pivot["FMI_2026_PARTIAL"] > 0.0)
    )
    pivot["stable_negative"] = (
        (pivot["MET_NORWAY_2026"] < 0.0)
        & (pivot["FMI_2026_PARTIAL"] < 0.0)
    )
    pivot["direction_flip"] = ~(
        pivot["stable_positive"] | pivot["stable_negative"]
    )
    pivot["absolute_difference_gap"] = (
        pivot["FMI_2026_PARTIAL"] - pivot["MET_NORWAY_2026"]
    ).abs()
    source_count = len(required_sources)
    expected_rows = 11 * 9
    if len(pivot) != expected_rows:
        raise ValueError(f"expected {expected_rows} paired rows, got {len(pivot)}")

    baseline_summary = (
        pivot.groupby("baseline", as_index=False)
        .agg(
            scenario_count=("scenario", "count"),
            stable_positive_count=("stable_positive", "sum"),
            stable_negative_count=("stable_negative", "sum"),
            direction_flip_count=("direction_flip", "sum"),
            mean_met_difference=("MET_NORWAY_2026", "mean"),
            mean_fmi_difference=("FMI_2026_PARTIAL", "mean"),
        )
        .sort_values(
            ["stable_negative_count", "stable_positive_count", "baseline"],
            ascending=[False, True, True],
        )
    )
    scenario_summary = (
        pivot.groupby("scenario", as_index=False)
        .agg(
            baseline_count=("baseline", "count"),
            stable_positive_count=("stable_positive", "sum"),
            stable_negative_count=("stable_negative", "sum"),
            direction_flip_count=("direction_flip", "sum"),
            mean_met_difference=("MET_NORWAY_2026", "mean"),
            mean_fmi_difference=("FMI_2026_PARTIAL", "mean"),
        )
        .sort_values("stable_negative_count", ascending=False)
    )

    out_json = TABLES / "stage395_cross_temporal_factor_failure_decomposition.json"
    out_pairs = TABLES / "stage395_cross_temporal_factor_failure_decomposition.csv"
    out_baseline = (
        TABLES / "stage395_cross_temporal_factor_failure_by_baseline.csv"
    )
    out_scenario = (
        TABLES / "stage395_cross_temporal_factor_failure_by_scenario.csv"
    )
    pivot.to_csv(out_pairs, index=False, encoding="utf-8-sig")
    baseline_summary.to_csv(out_baseline, index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(out_scenario, index=False, encoding="utf-8-sig")
    payload = {
        "status": "READ_ONLY_CROSS_TEMPORAL_FACTOR_DECOMPOSITION_COMPLETED",
        "exploratory": True,
        "confirmatory_promotion": False,
        "inputs": source_meta,
        "candidate": "trust_shape_union",
        "source_count": source_count,
        "paired_rows": len(pivot),
        "expected_paired_rows": expected_rows,
        "stable_positive_rows": int(pivot["stable_positive"].sum()),
        "stable_negative_rows": int(pivot["stable_negative"].sum()),
        "direction_flip_rows": int(pivot["direction_flip"].sum()),
        "stable_negative_examples": pivot.loc[
            pivot["stable_negative"],
            ["baseline", "scenario", "MET_NORWAY_2026", "FMI_2026_PARTIAL"],
        ].to_dict("records"),
        "claim_boundary": [
            "This is a read-only comparison of two already viewed exploratory temporal windows.",
            "It does not create a new independent test set and cannot support confirmatory claims.",
            "All attacks are controlled simulations, not production incidents.",
            "Stable failures are retained as algorithmic limitations, not removed by changing denominators.",
        ],
        "output_sha256": {
            "paired": sha256(out_pairs),
            "baseline": sha256(out_baseline),
            "scenario": sha256(out_scenario),
        },
    }
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage395 跨时间窗因子失败分解",
        "",
        f"- 状态：`{payload['status']}`",
        "- 范围：只读比较 MET Norway 2026 与 FMI 2026 已生成的探索性结果。",
        f"- 配对比较：`{len(pivot)}/{expected_rows}` 条（11 个基线 × 9 个场景）。",
        f"- 两个时间窗均为正：`{payload['stable_positive_rows']}` 条。",
        f"- 两个时间窗均为负：`{payload['stable_negative_rows']}` 条。",
        f"- 方向翻转：`{payload['direction_flip_rows']}` 条。",
        "",
        "## 按基线",
        "",
        baseline_summary.to_markdown(index=False),
        "",
        "## 按场景",
        "",
        scenario_summary.to_markdown(index=False),
        "",
        "## 边界",
        "",
        *[f"- {item}" for item in payload["claim_boundary"]],
    ]
    report_path = REPORTS / "stage395_cross_temporal_factor_failure_decomposition.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    payload["output_sha256"]["json"] = sha256(out_json)
    payload["output_sha256"]["report"] = sha256(report_path)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "paired_rows": len(pivot),
                "stable_positive_rows": payload["stable_positive_rows"],
                "stable_negative_rows": payload["stable_negative_rows"],
                "direction_flip_rows": payload["direction_flip_rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
