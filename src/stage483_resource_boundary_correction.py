"""Stage 483: correct resource-table comparison boundaries.

The existing Trust resource measurement times the fusion rule after feature
construction.  The baseline resource table times model fitting/scoring on a
precomputed feature matrix, and some security methods share a fitting call.
This stage labels those boundaries explicitly instead of presenting them as
one common end-to-end benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

BASELINE = TABLES / "final_baseline_resources.csv"
TRUST = TABLES / "final_trust_resources.csv"
TESTBED = TABLES / "final_testbed_resources.csv"
OUT_BASELINE = TABLES / "final_baseline_resources_corrected.csv"
OUT_TRUST = TABLES / "final_trust_resources_corrected.csv"
OUT_TESTBED = TABLES / "final_testbed_resources_corrected.csv"
OUT_JSON = REPORTS / "final_resource_boundary_correction.json"
OUT_MD = REPORTS / "final_resource_boundary_correction.md"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASELINE)
    trust = pd.read_csv(TRUST)
    testbed = pd.read_csv(TESTBED)

    baseline["latency_scope"] = (
        "model_fit_or_score_on_precomputed_train_normal_feature_matrix"
    )
    baseline["comparison_boundary"] = (
        "Not a full raw-stream end-to-end measurement. Some methods share "
        "a security-family fit call; do not compare this column directly "
        "with the Trust fusion-layer latency."
    )
    trust["latency_scope"] = (
        "fusion_rule_after_precomputed_base_z_and_feature_matrix"
    )
    trust["comparison_boundary"] = (
        "The reported latency is not full feature extraction, model fitting, "
        "threshold calibration, or recovery."
    )
    trust["observed_fit_plus_score_seconds"] = (
        pd.to_numeric(trust["fit_seconds"], errors="coerce")
        if "fit_seconds" in trust
        else pd.NA
    )
    if "mean_end_to_end_latency_ms" in trust:
        trust["observed_fit_plus_score_seconds"] = (
            pd.to_numeric(
                trust.get("fit_seconds", pd.Series([pd.NA] * len(trust))),
                errors="coerce",
            )
            + pd.to_numeric(
                trust["mean_end_to_end_latency_ms"],
                errors="coerce",
            )
            / 1000.0
        )
    testbed["latency_scope"] = (
        "authorized_loopback_runtime_not_v6_algorithm_end_to_end"
    )
    testbed["comparison_boundary"] = (
        "Runtime evidence for a separate streaming testbed; it does not "
        "measure the frozen V6 batch pipeline."
    )

    baseline.to_csv(OUT_BASELINE, index=False, encoding="utf-8-sig")
    trust.to_csv(OUT_TRUST, index=False, encoding="utf-8-sig")
    testbed.to_csv(OUT_TESTBED, index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_RESOURCE_BOUNDARY_CORRECTION",
        "baseline_rows": int(len(baseline)),
        "trust_rows": int(len(trust)),
        "testbed_rows": int(len(testbed)),
        "trust_latency_scope": str(trust["latency_scope"].iloc[0])
        if len(trust)
        else None,
        "baseline_latency_scope": str(
            baseline["latency_scope"].iloc[0]
        )
        if len(baseline)
        else None,
        "rules": [
            "The old Trust end-to-end label is withdrawn.",
            "Baseline and Trust resource columns are not directly comparable.",
            "Testbed throughput is separate runtime evidence.",
            "Missing measurements remain missing rather than imputed.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Resource-boundary correction",
        "",
        f"- Status: `{payload['status']}`",
        f"- Trust latency scope: `{payload['trust_latency_scope']}`",
        f"- Baseline latency scope: `{payload['baseline_latency_scope']}`",
        "",
        "## Corrected files",
        "",
        f"- `{OUT_BASELINE.relative_to(FINAL)}`",
        f"- `{OUT_TRUST.relative_to(FINAL)}`",
        f"- `{OUT_TESTBED.relative_to(FINAL)}`",
        "",
        "## Rule",
        "",
        "- The previous `mean_end_to_end_latency_ms` label is not used for "
        "cross-method efficiency claims.",
        "- Trust timing covers the fusion rule after precomputed features.",
        "- Baseline timing covers model fit/score on the precomputed matrix "
        "and may include shared family fitting.",
        "- Testbed timing is separate streaming runtime evidence.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "baseline_rows": payload["baseline_rows"],
                "trust_rows": payload["trust_rows"],
                "testbed_rows": payload["testbed_rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
