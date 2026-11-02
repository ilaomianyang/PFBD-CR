"""Stage 481: audit information sets for V6 and the streaming testbed.

The V6 Trust candidates use a centered Hampel window through the frozen
component pipeline.  This stage records that fact and verifies the separate
Stage 255 streaming detector with a future-perturbation prefix test.  It does
not relabel the frozen V6 performance as causal or change any prediction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors import HampelMADDetector  # noqa: E402
from stage255_online_causal_testbed import online_detect  # noqa: E402

FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"
BOUNDARY = TABLES / "final_method_implementation_boundary.csv"
OUT_BOUNDARY = TABLES / "final_method_implementation_boundary_corrected.csv"
OUT_JSON = REPORTS / "final_v6_information_set_audit.json"
OUT_MD = REPORTS / "final_v6_information_set_audit.md"


def hampel_prefix_test() -> dict[str, object]:
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2020-01-01",
                periods=15,
                freq="h",
                tz="UTC",
            ),
            "station_key": ["probe"] * 15,
            "observed_value": np.zeros(15, dtype=float),
        }
    )
    detector = HampelMADDetector().fit(frame)
    original = frame.copy()
    original.loc[7, "observed_value"] = 5.0
    changed_future = original.copy()
    changed_future.loc[8:10, "observed_value"] = 100.0
    original_scores = detector.score(original)
    changed_scores = detector.score(changed_future)
    prefix_equal = bool(
        np.allclose(original_scores[:8], changed_scores[:8])
    )
    return {
        "test": "future_perturbation_prefix_invariance",
        "current_and_past_inputs_equal": True,
        "future_values_changed": True,
        "score_at_index_7_original": float(original_scores[7]),
        "score_at_index_7_changed_future": float(changed_scores[7]),
        "prefix_invariance_pass": prefix_equal,
        "interpretation": (
            "V6 Hampel component reads future values because its rolling "
            "median uses center=True."
        ),
    }


def streaming_prefix_test() -> dict[str, object]:
    base = [
        {
            "station_id": "s1",
            "received_monotonic": float(index),
            "sequence": index,
            "temperature_c": 10.0 + 0.1 * index,
            "raw_sha256": f"row-{index}",
        }
        for index in range(12)
    ]
    changed = [dict(row) for row in base]
    for index in range(8, 12):
        changed[index]["temperature_c"] = 100.0
    original_output, _ = online_detect(base, "spike")
    changed_output, _ = online_detect(changed, "spike")
    prefix_equal = bool(
        original_output[:8] == changed_output[:8]
    )
    return {
        "test": "stage255_streaming_future_perturbation_prefix_invariance",
        "prefix_invariance_pass": prefix_equal,
        "interpretation": (
            "Stage255 streaming detector consumes only the current frame, "
            "earlier frames, and arrival time in this prefix test."
        ),
    }


def corrected_boundary() -> pd.DataFrame:
    frame = pd.read_csv(BOUNDARY)
    frame["v6_information_set"] = "not_audited_in_stage481"
    frame["strictly_causal"] = pd.Series(
        [None] * len(frame),
        dtype=object,
        index=frame.index,
    )
    frame["online_evidence"] = "not_audited_in_stage481"
    trust = frame["method"].isin(
        ["trust_only", "route_knn_arima", "trust_shape_union"]
    )
    frame.loc[trust, "v6_information_set"] = (
        "batch_centered_hampel_window_plus_causal_features"
    )
    frame.loc[trust, "strictly_causal"] = False
    frame.loc[trust, "online_evidence"] = (
        "No: V6 batch component is not the Stage255 streaming detector."
    )
    extra = pd.DataFrame(
        [
            {
                "method": "stage255_streaming_testbed_detector",
                "role": "separate runtime evidence",
                "implementation_status": "strict streaming rule",
                "causal_online": True,
                "training_boundary": (
                    "online arrival-time and past-frame state only"
                ),
                "notes": (
                    "Separate from V6; not a reproduction of the frozen "
                    "batch V6 detector."
                ),
                "v6_information_set": "not_applicable",
                "strictly_causal": True,
                "online_evidence": (
                    "Yes for the streaming prefix test; not a V6 result."
                ),
            }
        ]
    )
    return pd.concat([frame, extra], ignore_index=True, sort=False)


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    hampel = hampel_prefix_test()
    streaming = streaming_prefix_test()
    boundary = corrected_boundary()
    boundary.to_csv(OUT_BOUNDARY, index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS_V6_INFORMATION_SET_AUDIT",
        "v6_hampel": hampel,
        "stage255_streaming": streaming,
        "v6_strictly_causal_claim_allowed": False,
        "stage255_is_separate_evidence": True,
        "corrected_boundary": str(OUT_BOUNDARY.relative_to(ROOT)),
        "rules": [
            "V6 performance is retained as a batch/centered-window result.",
            "Stage255 is a separate streaming runtime analysis.",
            "No V6 prediction or threshold was changed.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# V6 information-set audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- V6 strictly causal claim allowed: "
        f"`{payload['v6_strictly_causal_claim_allowed']}`",
        f"- Stage255 separate streaming evidence: "
        f"`{payload['stage255_is_separate_evidence']}`",
        "",
        "## Hampel future perturbation",
        "",
        f"- Original score at index 7: `{hampel['score_at_index_7_original']}`",
        f"- Changed-future score at index 7: "
        f"`{hampel['score_at_index_7_changed_future']}`",
        f"- Prefix invariance pass: `{hampel['prefix_invariance_pass']}`",
        "",
        "## Stage255 streaming prefix test",
        "",
        f"- Prefix invariance pass: `{streaming['prefix_invariance_pass']}`",
        "",
        "## Boundary",
        "",
        "- V6 remains a batch/centered-window evaluation.",
        "- Stage255 streaming evidence must be reported separately.",
        "- The corrected method-boundary table is "
        "`final_method_implementation_boundary_corrected.csv`.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "hampel_prefix_invariance": hampel[
                    "prefix_invariance_pass"
                ],
                "streaming_prefix_invariance": streaming[
                    "prefix_invariance_pass"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
