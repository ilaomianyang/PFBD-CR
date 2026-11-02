"""Stage 479: scenario-level hierarchical bootstrap for V6.

The earlier scenario CI treated 20 provider-seed rows as independent.  This
stage resamples providers with replacement and then seeds within each sampled
provider.  Count-based metrics are recomputed from TP/FP/FN/TN; macro metrics
are reported separately.  The script reads saved scenario metrics only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"

SCENARIO = TABLES / "final_v6_scenario_metrics.csv"
OUT_CSV = TABLES / "final_v6_scenario_ci_corrected.csv"
OUT_JSON = REPORTS / "final_v6_scenario_ci_correction.json"
OUT_MD = REPORTS / "final_v6_scenario_ci_correction.md"

COUNT_METRICS = [
    "precision",
    "recall",
    "f1",
    "false_alarm_rate",
    "event_coverage",
]
MACRO_METRICS = [
    "precision",
    "recall",
    "f1",
    "false_alarm_rate",
    "event_coverage",
    "average_precision",
    "roc_auc",
]
ITERATIONS = 10000
SEED = 20261009


def count_metrics(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (
        float(2 * tp / (2 * tp + fp + fn))
        if (2 * tp + fp + fn)
        else 0.0
    )
    far = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
    }


def hierarchical_draws(
    group: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    providers = sorted(group["source"].astype(str).unique())
    provider_arrays = {
        provider: group.index[
            group["source"].astype(str).eq(provider)
        ].to_numpy(dtype=int)
        for provider in providers
    }
    rng = np.random.default_rng(seed)
    count_draws = {
        metric: np.empty(iterations, dtype=float) for metric in COUNT_METRICS
    }
    macro_draws = {
        metric: np.empty(iterations, dtype=float) for metric in MACRO_METRICS
    }
    tp = group["tp"].to_numpy(dtype=float)
    fp = group["fp"].to_numpy(dtype=float)
    fn = group["fn"].to_numpy(dtype=float)
    tn = group["tn"].to_numpy(dtype=float)
    detected = group["detected_events"].to_numpy(dtype=float)
    events = group["events"].to_numpy(dtype=float)
    for iteration in range(iterations):
        sampled_providers = rng.choice(
            providers,
            size=len(providers),
            replace=True,
        )
        selected: list[int] = []
        for provider in sampled_providers:
            rows = provider_arrays[str(provider)]
            selected.extend(
                rng.choice(rows, size=len(rows), replace=True).tolist()
            )
        idx = np.asarray(selected, dtype=int)
        values = count_metrics(
            tp[idx].sum(),
            fp[idx].sum(),
            fn[idx].sum(),
            tn[idx].sum(),
        )
        values["event_coverage"] = (
            float(detected[idx].sum() / events[idx].sum())
            if events[idx].sum()
            else 0.0
        )
        for metric in COUNT_METRICS:
            count_draws[metric][iteration] = values[metric]
        for metric in MACRO_METRICS:
            macro_draws[metric][iteration] = float(
                group[metric].to_numpy(dtype=float)[idx].mean()
            )
    return count_draws, macro_draws


def point_count_metrics(group: pd.DataFrame) -> dict[str, float]:
    values = count_metrics(
        group["tp"].sum(),
        group["fp"].sum(),
        group["fn"].sum(),
        group["tn"].sum(),
    )
    values["event_coverage"] = (
        float(group["detected_events"].sum() / group["events"].sum())
        if group["events"].sum()
        else 0.0
    )
    return values


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SCENARIO)
    rows: list[dict[str, object]] = []
    for (scenario, method), group in frame.groupby(
        ["scenario", "method"], sort=True
    ):
        group = group.reset_index(drop=True)
        point_count = point_count_metrics(group)
        count_draws, macro_draws = hierarchical_draws(
            group,
            ITERATIONS,
            SEED + len(rows),
        )
        for metric in COUNT_METRICS:
            draws = count_draws[metric]
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "metric": metric,
                    "estimand": "count_pooled",
                    "point_estimate": float(point_count[metric]),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "iterations": ITERATIONS,
                    "resampling": (
                        "providers_with_replacement_then_seeds_within_"
                        "provider_with_replacement"
                    ),
                    "provider_count": int(group["source"].nunique()),
                    "seed_count": int(group["seed"].nunique()),
                    "note": (
                        "Count metric recomputed from TP/FP/FN/TN; "
                        "conditioned on the four observed providers."
                    ),
                }
            )
        for metric in MACRO_METRICS:
            draws = macro_draws[metric]
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "metric": metric,
                    "estimand": "provider_seed_macro",
                    "point_estimate": float(group[metric].mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "iterations": ITERATIONS,
                    "resampling": (
                        "providers_with_replacement_then_seeds_within_"
                        "provider_with_replacement"
                    ),
                    "provider_count": int(group["source"].nunique()),
                    "seed_count": int(group["seed"].nunique()),
                    "note": (
                        "Macro metric is the mean of provider-seed rows; "
                        "this CI does not make the four providers a random "
                        "population."
                    ),
                }
            )
    corrected = pd.DataFrame(rows)
    corrected.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    union = corrected[
        corrected["method"].eq("trust_shape_union")
        & corrected["scenario"].eq("long_stuck")
        & corrected["metric"].isin(["f1", "event_coverage"])
    ]
    payload = {
        "status": "PASS_V6_HIERARCHICAL_SCENARIO_CI_CORRECTION",
        "source": str(SCENARIO.relative_to(ROOT)),
        "scenario_count": int(frame["scenario"].nunique()),
        "method_count": int(frame["method"].nunique()),
        "iterations": ITERATIONS,
        "resampling_unit": (
            "provider cluster then seed within provider; both with replacement"
        ),
        "estimands": [
            "count_pooled",
            "provider_seed_macro",
        ],
        "long_stuck_union_examples": union.to_dict("records"),
        "rules": [
            "The four observed providers are treated as fixed; bootstrap "
            "quantifies conditional sampling uncertainty.",
            "Repeated seeds are not treated as independent providers.",
            "Count CIs use the same count estimator as the point estimate.",
            "AP/ROC are reported as macro metrics only.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# V6 hierarchical scenario CI correction",
        "",
        f"- Status: `{payload['status']}`",
        f"- Iterations: `{ITERATIONS}`",
        f"- Resampling: `{payload['resampling_unit']}`",
        "",
        "## Long-stuck union examples",
        "",
        union.to_markdown(index=False),
        "",
        "## Rules",
        "",
        "- Count-pooled point estimates and count-pooled CIs use the same "
        "TP/FP/FN/TN estimator.",
        "- Provider-seed macro intervals are separate and are not population "
        "inference over institutions.",
        "- AP/ROC are not reconstructed from counts.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": len(corrected),
                "scenarios": payload["scenario_count"],
                "methods": payload["method_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
