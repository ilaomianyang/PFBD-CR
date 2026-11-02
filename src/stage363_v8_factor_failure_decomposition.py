from __future__ import annotations

"""Read-only decomposition of V8 factor failures.

No thresholds, methods, labels, or test rows are changed.  The output
quantifies where Trust loses to each baseline by factor cell, provider, and
scenario, and records the evidence boundary for the exploratory V8 package.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL = ROOT / "confirmatory_package" / "simulated_attack_v8" / "protocol_freeze.json"
SCENARIO = TABLES / "stage361_v8_full_factor_locked_evaluation_scenario.csv"
OUT_JSON = TABLES / "stage363_v8_factor_failure_decomposition.json"
OUT_CELL = TABLES / "stage363_v8_factor_failure_cells.csv"
OUT_PROVIDER = TABLES / "stage363_v8_factor_failure_provider.csv"
OUT_REPORT = REPORTS / "stage363_v8_factor_failure_decomposition.md"

BASELINES = [
    "robust_max_deviation",
    "isolation_forest",
    "one_class_svm",
    "pca_reconstruction",
    "mlp_autoencoder_reconstruction",
    "knn_distance",
    "ewma_deviation",
    "arima_innovation",
    "gru_predictor",
    "lightweight_autoencoder",
    "ics_cps_spc_proxy",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    scenario = pd.read_csv(SCENARIO)
    specs = {str(x["name"]): x for x in protocol["attack_scenarios"]}
    rows = []
    for _, row in scenario.iterrows():
        spec = specs[str(row.scenario)]
        rows.append(
            {
                **row.to_dict(),
                "factor_intensity": spec.get("factor_intensity", ""),
                "factor_duration": spec.get("factor_duration", ""),
                "factor_length": spec.get("factor_length", ""),
                "factorial": bool(spec.get("factorial", False)),
                "unseen_combination": bool(spec.get("unseen_combination", False)),
            }
        )
    data = pd.DataFrame(rows)
    means = data.groupby(["source", "scenario", "method"], as_index=False)[
        ["event_coverage", "row_recall"]
    ].mean()
    t = means[means.method.eq("trust_shape_union")].rename(
        columns={
            "method": "candidate",
            "event_coverage": "candidate_event_coverage",
            "row_recall": "candidate_row_recall",
        }
    )
    cells = []
    for baseline in BASELINES:
        b = means[means.method.eq(baseline)].rename(
            columns={
                "method": "baseline",
                "event_coverage": "baseline_event_coverage",
                "row_recall": "baseline_row_recall",
            }
        )
        m = t.merge(b, on=["source", "scenario"], how="inner")
        m["baseline"] = baseline
        m["event_difference"] = m.candidate_event_coverage - m.baseline_event_coverage
        m["event_win"] = m.event_difference > 0
        m["factor_intensity"] = m.scenario.map(lambda x: specs[str(x)].get("factor_intensity", ""))
        m["factor_duration"] = m.scenario.map(lambda x: specs[str(x)].get("factor_duration", ""))
        m["factor_length"] = m.scenario.map(lambda x: specs[str(x)].get("factor_length", ""))
        m["factorial"] = m.scenario.map(lambda x: bool(specs[str(x)].get("factorial", False)))
        m["unseen_combination"] = m.scenario.map(lambda x: bool(specs[str(x)].get("unseen_combination", False)))
        cells.append(m)
    cell_df = pd.concat(cells, ignore_index=True)
    cell_df.to_csv(OUT_CELL, index=False)
    provider_df = (
        cell_df.groupby(["baseline", "source"], as_index=False)
        .agg(
            event_difference_mean=("event_difference", "mean"),
            event_win_share=("event_win", "mean"),
        )
    )
    provider_df.to_csv(OUT_PROVIDER, index=False)
    weak = cell_df[cell_df.factor_intensity.eq("weak")]
    strong = cell_df[cell_df.factor_intensity.eq("strong") & cell_df.factorial]
    unseen = cell_df[cell_df.unseen_combination]
    summary = {
        "status": "READ_ONLY_FAILURE_DECOMPOSITION_COMPLETED",
        "protocol_id": protocol["protocol_id"],
        "scenario_sha256": sha256(SCENARIO),
        "scenario_rows": len(scenario),
        "cell_rows": len(cell_df),
        "weak_candidate_event_coverage": float(
            data[(data.method.eq("trust_shape_union")) & data.scenario.map(lambda x: specs[str(x)].get("factor_intensity") == "weak")].event_coverage.mean()
        ),
        "strong_factorial_candidate_event_coverage": float(
            data[(data.method.eq("trust_shape_union")) & data.scenario.map(lambda x: specs[str(x)].get("factor_intensity") == "strong") & data.scenario.map(lambda x: bool(specs[str(x)].get("factorial", False)))].event_coverage.mean()
        ),
        "unseen_candidate_event_coverage": float(
            data[(data.method.eq("trust_shape_union")) & data.scenario.map(lambda x: bool(specs[str(x)].get("unseen_combination", False)))].event_coverage.mean()
        ),
        "baseline_summary": {
            str(b): {
                "weak_event_difference": float(weak[weak.baseline.eq(b)].event_difference.mean()),
                "strong_factorial_event_difference": float(strong[strong.baseline.eq(b)].event_difference.mean()),
                "unseen_event_difference": float(unseen[unseen.baseline.eq(b)].event_difference.mean()),
            }
            for b in BASELINES
        },
        "limitations": [
            "这是对已完成 V8 输出的只读分解，不是新的确认性性能证据。",
            "V8 底层 test 时间窗此前已查看，结果不能替换 V6 主结果。",
            "所有攻击均为透明模拟注入，不是真实生产攻击日志。",
            "不得删除 ARIMA、kNN、弱攻击或 coordinated drift 的不利结果。",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Stage363 V8 因子失败分解",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 协议：`{protocol['protocol_id']}`",
        f"- Trust 弱强度单元平均 event coverage：`{summary['weak_candidate_event_coverage']:.6f}`",
        f"- Trust 强度单元平均 event coverage：`{summary['strong_factorial_candidate_event_coverage']:.6f}`",
        f"- Trust 未见 coordinated drift event coverage：`{summary['unseen_candidate_event_coverage']:.6f}`",
        "",
        "## 基线差值分解",
        "",
        pd.DataFrame(
            [
                {
                    "baseline": b,
                    **v,
                }
                for b, v in summary["baseline_summary"].items()
            ]
        ).to_markdown(index=False),
        "",
        "## 边界",
        "",
        *[f"- {x}" for x in summary["limitations"]],
        "",
    ]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "cell_rows": len(cell_df), "weak_event_coverage": summary["weak_candidate_event_coverage"], "unseen_event_coverage": summary["unseen_candidate_event_coverage"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
