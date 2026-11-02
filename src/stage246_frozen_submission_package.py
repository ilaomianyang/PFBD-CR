from __future__ import annotations

"""Stage 246: frozen paper tables, figures, and submission manifest.

This stage is intentionally read-only with respect to the experiments.  It
reads the already frozen JSON/CSV artifacts, emits paper-facing tables and
figures, and records SHA256 hashes for every input and generated deliverable.
It performs no training, threshold selection, feature selection, or
re-evaluation of any test set.

All attack experiments included here use transparent controlled injection.
None of the included artifacts is a real attack log.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
PREFIX = "stage246_frozen_submission_package"

FROZEN_INPUTS = {
    "stage207": TABLES / "stage207_dwd_controlled_attack.json",
    "stage208": TABLES / "stage208_dwd_persistent_repeat_protection.json",
    "stage209": TABLES / "stage209_dwd_station_cluster_uncertainty.json",
    "stage210": TABLES / "stage210_dwd_seed_randomized_replication.json",
    "stage211": TABLES / "stage211_dwd_replication_station_uncertainty.json",
    "stage212": TABLES / "stage212_dwd_operating_curve.json",
    "stage219": TABLES / "stage219_swiss_dwd_protection_transfer.json",
    "stage222": TABLES / "stage222_noaa_dwd_protection_transfer.json",
    "stage228": TABLES / "stage228_knmi_dwd_protection_transfer.json",
    "stage230": TABLES / "stage230_cross_provider_evidence_synthesis.json",
    "stage231": TABLES / "stage231_cross_provider_robustness.json",
    "stage232": TABLES / "stage232_attack_type_robustness.json",
    "stage233": TABLES / "stage233_valid_attack_correction_audit.json",
    "stage238": TABLES / "stage238_eccc_dwd_protection_transfer.json",
    "stage240": TABLES / "stage240_nonstuck_balanced_selection.json",
    "stage241": TABLES / "stage241_transfer_heterogeneity_audit.json",
    "stage242": TABLES / "stage242_multisource_transfer_matrix.json",
    "stage244": TABLES / "stage244_cross_provider_nonstuck_candidate_features.json",
    "stage245": TABLES / "stage245_cross_provider_nonstuck_seed_replication.json",
    "stage247": TABLES / "stage247_provider_balanced_nonstuck_audit.json",
    "stage248": TABLES / "stage248_cross_provider_fair_baseline_audit.json",
    "stage107": TABLES / "stage107_noaa_natural_svm_validation.json",
    "stage109": TABLES / "stage109_noaa_quality_weak_label_bootstrap.json",
    "stage110": TABLES / "stage110_noaa_quality_hierarchical_bootstrap.json",
    "stage116": TABLES / "stage116_noaa_quality_label_permutation.json",
    "stage142": TABLES / "stage142_noaa_frozen_natural_operational_validation.json",
    "stage153": TABLES / "stage153_noaa_natural_station_operational_comparison.json",
    "stage157": TABLES / "stage157_noaa_natural_two_way_cluster_bootstrap.json",
    "stage130": TABLES / "stage130_noaa_safety_constrained_recovery.json",
    "stage133": TABLES / "stage133_causal_temporal_recovery.json",
    "stage134": TABLES / "stage134_causal_recovery_optimization.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(old, new)
    return text


def write_latex(frame: pd.DataFrame, path: Path, caption: str) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        r"\begin{tabular}{" + "l" + "r" * (len(frame.columns) - 1) + "}",
        r"\hline",
        " & ".join(latex_escape(column) for column in frame.columns) + r" \\",
        r"\hline",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    missing = [name for name, path in FROZEN_INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")

    s207 = load_json(FROZEN_INPUTS["stage207"])
    s208 = load_json(FROZEN_INPUTS["stage208"])
    s210 = load_json(FROZEN_INPUTS["stage210"])
    s212 = load_json(FROZEN_INPUTS["stage212"])
    s230 = load_json(FROZEN_INPUTS["stage230"])
    s231 = load_json(FROZEN_INPUTS["stage231"])
    s232 = load_json(FROZEN_INPUTS["stage232"])
    s240 = load_json(FROZEN_INPUTS["stage240"])
    s242 = load_json(FROZEN_INPUTS["stage242"])
    s244 = load_json(FROZEN_INPUTS["stage244"])
    s245 = load_json(FROZEN_INPUTS["stage245"])
    s247 = load_json(FROZEN_INPUTS["stage247"])
    s248 = load_json(FROZEN_INPUTS["stage248"])
    s107 = load_json(FROZEN_INPUTS["stage107"])
    s109 = load_json(FROZEN_INPUTS["stage109"])
    s110 = load_json(FROZEN_INPUTS["stage110"])
    s116 = load_json(FROZEN_INPUTS["stage116"])
    s142 = load_json(FROZEN_INPUTS["stage142"])
    s153 = load_json(FROZEN_INPUTS["stage153"])
    s157 = load_json(FROZEN_INPUTS["stage157"])
    s133 = load_json(FROZEN_INPUTS["stage133"])
    s134 = load_json(FROZEN_INPUTS["stage134"])

    required_failure = [
        name for name, payload in (
            ("stage207", s207),
            ("stage208", s208),
            ("stage210", s210),
            ("stage212", s212),
            ("stage230", s230),
            ("stage231", s231),
            ("stage232", s232),
            ("stage240", s240),
            ("stage242", s242),
            ("stage244", s244),
            ("stage245", s245),
            ("stage247", s247),
            ("stage248", s248),
            ("stage107", s107),
            ("stage109", s109),
            ("stage110", s110),
            ("stage116", s116),
            ("stage142", s142),
            ("stage153", s153),
            ("stage157", s157),
            ("stage133", s133),
            ("stage134", s134),
        )
        if str(payload.get("status")) not in {
            "PASS",
            "PASS_PIPELINE_REVIEW_REQUIRED",
            "NEGATIVE_RESULT_RETAINED",
            "CANDIDATE_TRANSFER_REQUIRES_REVIEW",
            "ATTACK_TYPE_ROBUSTNESS_REVIEW_REQUIRED",
            "BOUNDARY_EVIDENCE_RETAINED",
        }
    ]
    if required_failure:
        raise ValueError(f"unexpected frozen status: {required_failure}")

    primary_rows = [
        row for row in s208["primary_aggregate"]
        if int(row["seeds"]) == 5
    ]
    if not primary_rows:
        raise ValueError("Stage208 has no primary aggregate rows")
    dwd_primary = next(
        (
            row
            for row in primary_rows
            if row.get("selection_rule") == "valid_f1"
            and float(row.get("max_far", -1.0)) == 0.01
        ),
        None,
    )
    if dwd_primary is None:
        raise ValueError("Stage208 primary valid_f1/FAR=0.01 row is missing")
    dwd_f1 = float(dwd_primary["test_f1"])
    dwd_f1_baseline = float(dwd_primary["baseline_test_f1"])
    dwd_event = float(dwd_primary["test_event_coverage"])
    dwd_event_baseline = float(dwd_primary["baseline_test_event_coverage"])
    dwd_far = float(dwd_primary["test_false_alarm_rate"])
    dwd_row_recall = float(dwd_primary["test_row_recall"])

    transfer_rows = []
    for row in s230.get("transfer_rows", []):
        transfer_rows.append(
            {
                "target_provider": row.get("target_source", ""),
                "stations": row.get("station_count", ""),
                "direct_threshold_transfer_f1": row.get(
                    "direct_threshold_transfer_f1", ""
                ),
                "selected_valid_f1": row.get(
                    "target_valid_selected_f1", ""
                ),
                "recalibrated_f1": row.get(
                    "target_valid_recalibrated_f1", ""
                ),
                "selected_minus_recalibrated_f1": row.get(
                    "selected_minus_recalibrated_f1", ""
                ),
                "selected_minus_recalibrated_event": row.get(
                    "selected_minus_recalibrated_event_coverage", ""
                ),
            }
        )
    transfer_table = pd.DataFrame(transfer_rows).sort_values(
        "target_provider"
    )

    attack_type_bootstrap = {
        row.get("attack_type"): row
        for row in s232.get("hierarchical_bootstrap_rows", [])
    }
    attack_type_rows = []
    for row in s232.get("provider_equal_rows", []):
        bootstrap = attack_type_bootstrap.get(row.get("attack_type"), {})
        attack_type_rows.append(
            {
                "attack_type": row.get("attack_type", ""),
                "row_recall_difference": row.get(
                    "row_recall_difference", ""
                ),
                "event_coverage_difference": row.get(
                    "event_coverage_difference", ""
                ),
                "row_recall_ci95_low": bootstrap.get(
                    "row_recall_ci95_low", ""
                ),
                "row_recall_ci95_high": bootstrap.get(
                    "row_recall_ci95_high", ""
                ),
                "event_coverage_ci95_low": bootstrap.get(
                    "event_coverage_ci95_low", ""
                ),
                "event_coverage_ci95_high": bootstrap.get(
                    "event_coverage_ci95_high", ""
                ),
            }
        )
    attack_type = pd.DataFrame(attack_type_rows)

    nonstuck_rows = []
    for stage_name, payload in (("stage244", s244), ("stage245", s245)):
        for row in payload.get("nonstuck_type_equal_rows", []):
            nonstuck_rows.append(
                {
                    "stage": stage_name,
                    "attack_type": row.get("attack_type", ""),
                    "row_recall_difference": row.get(
                        "row_recall_difference", ""
                    ),
                    "event_coverage_difference": row.get(
                        "event_coverage_difference", ""
                    ),
                }
            )
    nonstuck = pd.DataFrame(nonstuck_rows)

    summary_rows = [
        {
            "evidence": "DWD controlled benchmark",
            "primary_far": 0.01,
            "f1": dwd_f1,
            "event_coverage": dwd_event,
            "row_recall": dwd_row_recall,
            "false_alarm_rate": dwd_far,
            "baseline_f1": dwd_f1_baseline,
            "baseline_event_coverage": dwd_event_baseline,
        },
        {
            "evidence": "Cross-provider provider-equal transfer",
            "primary_far": 0.01,
            "f1": s231["provider_equal_summary"]["f1_difference"],
            "event_coverage": s231["provider_equal_summary"][
                "event_difference"
            ],
            "row_recall": "",
            "false_alarm_rate": "",
            "baseline_f1": "",
            "baseline_event_coverage": "",
        },
        {
            "evidence": "Non-stuck balanced selection",
            "primary_far": 0.01,
            "f1": s240["hierarchical_bootstrap"]["f1_difference"],
            "event_coverage": s240["hierarchical_bootstrap"][
                "event_coverage_difference"
            ],
            "row_recall": "",
            "false_alarm_rate": "",
            "baseline_f1": "",
            "baseline_event_coverage": "",
        },
        {
            "evidence": "Non-stuck expanded features",
            "primary_far": 0.01,
            "f1": s244["hierarchical_bootstrap"]["f1_difference"],
            "event_coverage": s244["hierarchical_bootstrap"][
                "event_coverage_difference"
            ],
            "row_recall": "",
            "false_alarm_rate": "",
            "baseline_f1": "",
            "baseline_event_coverage": "",
        },
        {
            "evidence": "Non-stuck independent-seed replication",
            "primary_far": 0.01,
            "f1": s245["hierarchical_bootstrap"]["f1_difference"],
            "event_coverage": s245["hierarchical_bootstrap"][
                "event_coverage_difference"
            ],
            "row_recall": "",
            "false_alarm_rate": "",
            "baseline_f1": "",
            "baseline_event_coverage": "",
        },
    ]
    summary = pd.DataFrame(summary_rows)

    recovery_rows = []
    for row in s134.get("aggregate", []):
        recovery_rows.append(
            {
                "recovery_group": row.get("group", ""),
                "test_utility": row.get("test_utility_mean", ""),
                "attack_mae_after": row.get(
                    "test_attack_mae_after_mean", ""
                ),
                "attack_improvement_rate": row.get(
                    "test_attack_improvement_rate_mean", ""
                ),
                "clean_harm_rate": row.get("test_clean_harm_rate_mean", ""),
            }
        )
    recovery = pd.DataFrame(recovery_rows)

    provider_balanced_rows = []
    for row in s247.get("provider_equal_rows", []):
        provider_balanced_rows.append(
            {
                "stage": row.get("stage", ""),
                "scope": "provider_equal",
                "removed_provider": "",
                "f1_difference": row.get("f1_difference", ""),
                "event_coverage_difference": row.get(
                    "event_coverage_difference", ""
                ),
                "f1_direction_positive": "",
                "event_direction_positive": "",
            }
        )
    for row in s247.get("leave_one_provider_out_rows", []):
        provider_balanced_rows.append(
            {
                "stage": row.get("stage", ""),
                "scope": "leave_one_provider_out",
                "removed_provider": row.get("removed_source", ""),
                "f1_difference": row.get("f1_difference", ""),
                "event_coverage_difference": row.get(
                    "event_coverage_difference", ""
                ),
                "f1_direction_positive": row.get(
                    "f1_direction_positive", ""
                ),
                "event_direction_positive": row.get(
                    "event_direction_positive", ""
                ),
            }
        )
    provider_balanced = pd.DataFrame(provider_balanced_rows)

    fair_baseline_win = {
        (str(row.get("candidate")), str(row.get("baseline"))): row
        for row in s248.get("paired_win_rows", [])
    }
    fair_baseline_rows = []
    for row in s248.get("comparison_rows", []):
        candidate = str(row.get("candidate"))
        baseline = str(row.get("baseline"))
        win = fair_baseline_win.get((candidate, baseline), {})
        fair_baseline_rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "f1_difference": row.get("f1_difference", ""),
                "f1_ci95_low": row.get("f1_ci95_low", ""),
                "f1_ci95_high": row.get("f1_ci95_high", ""),
                "event_coverage_difference": row.get(
                    "event_coverage_difference", ""
                ),
                "event_ci95_low": row.get("event_ci95_low", ""),
                "event_ci95_high": row.get("event_ci95_high", ""),
                "f1_win_rate": win.get("f1_win_rate", ""),
                "event_win_rate": win.get("event_win_rate", ""),
            }
        )
    fair_baseline = pd.DataFrame(fair_baseline_rows)
    natural_rows = [
        {
            "evidence": "NOAA natural operational projection",
            "source_stage": "stage142",
            "scope": "natural observations; no injected attack",
            "primary_value": s142.get("status", ""),
            "supported_interpretation": (
                "alarm-rate projection on natural observations"
            ),
        },
        {
            "evidence": "NOAA weak-label association",
            "source_stage": "stage109",
            "scope": "official quality flags as weak labels",
            "primary_value": s109.get("status", ""),
            "supported_interpretation": (
                "association between alarms and weak quality flags"
            ),
        },
        {
            "evidence": "NOAA year-station hierarchical weak-label bootstrap",
            "source_stage": "stage110",
            "scope": "year-level then station-level resampling",
            "primary_value": s110.get("status", ""),
            "supported_interpretation": (
                "stability of the weak-label association"
            ),
        },
        {
            "evidence": "NOAA weak-label conditional permutation",
            "source_stage": "stage116",
            "scope": "within year-station conditioned label permutation",
            "primary_value": s116.get("status", ""),
            "supported_interpretation": (
                "statistical association; not attack ground truth"
            ),
        },
        {
            "evidence": "NOAA station natural operational comparison",
            "source_stage": "stage153",
            "scope": "station-cluster bootstrap on natural alarms",
            "primary_value": s153.get("status", ""),
            "supported_interpretation": (
                "natural operational safety across stations"
            ),
        },
        {
            "evidence": "NOAA two-way natural cluster bootstrap",
            "source_stage": "stage157",
            "scope": "two-way cluster bootstrap; stage155 and stage156",
            "primary_value": s157.get("status", ""),
            "supported_interpretation": (
                "natural-regime robustness, not attack recall"
            ),
        },
    ]
    natural = pd.DataFrame(natural_rows)

    summary.to_csv(
        TABLES / f"{PREFIX}_table1_primary_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    transfer_table.to_csv(
        TABLES / f"{PREFIX}_table2_cross_provider_transfer.csv",
        index=False,
        encoding="utf-8-sig",
    )
    attack_type.to_csv(
        TABLES / f"{PREFIX}_table3_attack_type.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nonstuck.to_csv(
        TABLES / f"{PREFIX}_table4_nonstuck_replication.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recovery.to_csv(
        TABLES / f"{PREFIX}_table5_recovery_safety.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provider_balanced.to_csv(
        TABLES / f"{PREFIX}_table6_provider_balanced_nonstuck.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fair_baseline.to_csv(
        TABLES / f"{PREFIX}_table7_fair_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    natural.to_csv(
        TABLES / f"{PREFIX}_table8_natural_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_latex(
        summary,
        TABLES / f"{PREFIX}_table1_primary_summary.tex",
        "Primary frozen results at the predeclared FAR=0.01 endpoint.",
    )
    write_latex(
        transfer_table,
        TABLES / f"{PREFIX}_table2_cross_provider_transfer.tex",
        "Cross-provider transfer after target-domain valid selection.",
    )
    write_latex(
        attack_type,
        TABLES / f"{PREFIX}_table3_attack_type.tex",
        "Provider-equal attack-type differences and uncertainty.",
    )
    write_latex(
        nonstuck,
        TABLES / f"{PREFIX}_table4_nonstuck_replication.tex",
        "Non-stuck feature expansion and independent-seed replication.",
    )
    write_latex(
        recovery,
        TABLES / f"{PREFIX}_table5_recovery_safety.tex",
        "Causal recovery utility and clean-row harm boundary.",
    )
    write_latex(
        provider_balanced,
        TABLES / f"{PREFIX}_table6_provider_balanced_nonstuck.tex",
        "Provider-equal and leave-one-provider-out non-stuck boundary.",
    )
    write_latex(
        fair_baseline,
        TABLES / f"{PREFIX}_table7_fair_baseline.tex",
        "Cross-provider fair low-FAR baseline comparisons.",
    )
    write_latex(
        natural,
        TABLES / f"{PREFIX}_table8_natural_validation.tex",
        "Natural and semi-real validation boundaries.",
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    transfer_plot = transfer_table.dropna(
        subset=["selected_minus_recalibrated_f1"]
    ).copy()
    x = np.arange(len(transfer_plot))
    f1_values = pd.to_numeric(
        transfer_plot["selected_minus_recalibrated_f1"], errors="coerce"
    ).to_numpy(dtype=float)
    event_values = pd.to_numeric(
        transfer_plot["selected_minus_recalibrated_event"], errors="coerce"
    ).to_numpy(dtype=float)
    ax.bar(x - 0.18, f1_values, width=0.36, label="F1")
    ax.bar(x + 0.18, event_values, width=0.36, label="Event coverage")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(transfer_plot["target_provider"], rotation=20, ha="right")
    ax.set_ylabel("Difference")
    ax.set_title("Target-domain valid selection minus recalibrated transfer")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{PREFIX}_figure1_transfer.{suffix}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    attack_plot = attack_type.copy()
    attack_plot["row_recall_difference"] = pd.to_numeric(
        attack_plot["row_recall_difference"], errors="coerce"
    )
    attack_plot["event_coverage_difference"] = pd.to_numeric(
        attack_plot["event_coverage_difference"], errors="coerce"
    )
    x = np.arange(len(attack_plot))
    ax.bar(
        x - 0.18,
        attack_plot["row_recall_difference"],
        width=0.36,
        label="Row recall",
    )
    ax.bar(
        x + 0.18,
        attack_plot["event_coverage_difference"],
        width=0.36,
        label="Event coverage",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(attack_plot["attack_type"], rotation=20, ha="right")
    ax.set_ylabel("Difference")
    ax.set_title("Frozen attack-type robustness")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{PREFIX}_figure2_attack_type.{suffix}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, stage_name in zip(axes, ("stage244", "stage245"), strict=True):
        subset = nonstuck[nonstuck["stage"].eq(stage_name)].copy()
        subset["row_recall_difference"] = pd.to_numeric(
            subset["row_recall_difference"], errors="coerce"
        )
        subset["event_coverage_difference"] = pd.to_numeric(
            subset["event_coverage_difference"], errors="coerce"
        )
        x = np.arange(len(subset))
        ax.bar(
            x - 0.18,
            subset["row_recall_difference"],
            width=0.36,
            label="Row recall",
        )
        ax.bar(
            x + 0.18,
            subset["event_coverage_difference"],
            width=0.36,
            label="Event coverage",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            subset["attack_type"], rotation=35, ha="right", fontsize=7
        )
        ax.set_title(stage_name)
        ax.set_ylabel("Difference")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Non-stuck attack-type directions")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{PREFIX}_figure3_nonstuck.{suffix}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    recovery_plot = recovery.copy()
    for column in (
        "test_utility",
        "attack_mae_after",
        "attack_improvement_rate",
        "clean_harm_rate",
    ):
        recovery_plot[column] = pd.to_numeric(
            recovery_plot[column], errors="coerce"
        )
    x = np.arange(len(recovery_plot))
    ax.bar(
        x - 0.2,
        recovery_plot["attack_improvement_rate"],
        width=0.4,
        label="Attack improvement rate",
    )
    ax.bar(
        x + 0.2,
        recovery_plot["clean_harm_rate"],
        width=0.4,
        label="Clean-row harm rate",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        recovery_plot["recovery_group"], rotation=20, ha="right"
    )
    ax.set_ylabel("Rate")
    ax.set_title("Strictly causal recovery and clean-row safety")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{PREFIX}_figure4_recovery.{suffix}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    leave_plot = provider_balanced[
        provider_balanced["scope"].eq("leave_one_provider_out")
    ].copy()
    for ax, stage_name in zip(axes, ("stage244", "stage245"), strict=True):
        subset = leave_plot[leave_plot["stage"].eq(stage_name)].copy()
        subset["f1_difference"] = pd.to_numeric(
            subset["f1_difference"], errors="coerce"
        )
        subset["event_coverage_difference"] = pd.to_numeric(
            subset["event_coverage_difference"], errors="coerce"
        )
        x = np.arange(len(subset))
        ax.bar(
            x - 0.18,
            subset["f1_difference"],
            width=0.36,
            label="F1",
        )
        ax.bar(
            x + 0.18,
            subset["event_coverage_difference"],
            width=0.36,
            label="Event coverage",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            subset["removed_provider"], rotation=25, ha="right"
        )
        ax.set_title(stage_name)
        ax.set_ylabel("Difference")
    axes[0].legend(frameon=False)
    fig.suptitle("Non-stuck provider-balanced leave-one-out boundary")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"{PREFIX}_figure5_provider_balanced_nonstuck.{suffix}"
        )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fair_plot = fair_baseline.copy()
    fair_plot["label"] = (
        fair_plot["candidate"].str.replace("trust_", "", regex=False)
        + " vs "
        + fair_plot["baseline"]
    )
    fair_plot = fair_plot.set_index("label")
    y = np.arange(len(fair_plot))
    f1_values = pd.to_numeric(
        fair_plot["f1_difference"],
        errors="coerce",
    ).to_numpy(dtype=float)
    event_values = pd.to_numeric(
        fair_plot["event_coverage_difference"],
        errors="coerce",
    ).to_numpy(dtype=float)
    f1_low = pd.to_numeric(
        fair_plot["f1_ci95_low"],
        errors="coerce",
    ).to_numpy(dtype=float)
    f1_high = pd.to_numeric(
        fair_plot["f1_ci95_high"],
        errors="coerce",
    ).to_numpy(dtype=float)
    event_low = pd.to_numeric(
        fair_plot["event_ci95_low"],
        errors="coerce",
    ).to_numpy(dtype=float)
    event_high = pd.to_numeric(
        fair_plot["event_ci95_high"],
        errors="coerce",
    ).to_numpy(dtype=float)
    ax.errorbar(
        f1_values,
        y - 0.12,
        xerr=np.vstack(
            [
                f1_values - f1_low,
                f1_high - f1_values,
            ]
        ),
        fmt="o",
        label="F1 difference",
    )
    ax.errorbar(
        event_values,
        y + 0.12,
        xerr=np.vstack(
            [
                event_values - event_low,
                event_high - event_values,
            ]
        ),
        fmt="s",
        label="Event coverage difference",
    )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(fair_plot.index, fontsize=7)
    ax.set_xlabel("Provider-equal difference with 95% CI")
    ax.set_title("Fair low-FAR comparison against unsupervised baselines")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{PREFIX}_figure6_fair_baseline.{suffix}")
    plt.close(fig)

    frozen_hashes = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in FROZEN_INPUTS.items()
    }
    generated_paths = sorted(
        [
            *TABLES.glob(f"{PREFIX}_table*.csv"),
            *TABLES.glob(f"{PREFIX}_table*.tex"),
            *FIGURES.glob(f"{PREFIX}_figure*.png"),
            *FIGURES.glob(f"{PREFIX}_figure*.pdf"),
        ]
    )
    generated_hashes = {
        path.name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in generated_paths
    }

    checks = [
        {
            "check": "frozen_inputs_present",
            "status": "PASS" if not missing else "FAIL",
            "value": len(FROZEN_INPUTS),
        },
        {
            "check": "primary_endpoint_far_001",
            "status": "PASS"
            if float(s208.get("primary_far", 0.0)) == 0.01
            else "FAIL",
            "value": s208.get("primary_far"),
        },
        {
            "check": "four_provider_boundary",
            "status": "PASS" if int(s231.get("source_count", 0)) == 4
            else "FAIL",
            "value": s231.get("source_count"),
        },
        {
            "check": "nonstuck_stage244_and_stage245_present",
            "status": "PASS"
            if s244.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED"
            and s245.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED"
            else "FAIL",
        },
        {
            "check": "stage245_direction_replicated_but_interval_crosses_zero",
            "status": "PASS"
            if s245["direction_consistency"]["f1_direction_consistent"] is True
            and s245["direction_consistency"][
                "event_coverage_direction_consistent"
            ]
            is True
            and float(s245["hierarchical_bootstrap"]["f1_ci95_low"]) < 0.0
            and float(s245["hierarchical_bootstrap"]["event_ci95_low"]) < 0.0
            else "FAIL",
        },
        {
            "check": "provider_balanced_nonstuck_boundary_retained",
            "status": "PASS"
            if s247.get("status") == "BOUNDARY_EVIDENCE_RETAINED"
            and all(
                float(row.get("f1_difference", 0.0)) > 0.0
                and float(row.get("event_coverage_difference", 0.0)) > 0.0
                for row in s247.get("provider_equal_rows", [])
            )
            and all(
                bool(row.get("event_direction_positive"))
                for row in s247.get("leave_one_provider_out_rows", [])
            )
            and any(
                str(row.get("removed_source")) == "ECCC"
                and float(row.get("f1_difference", 0.0)) < 0.0
                for row in s247.get("leave_one_provider_out_rows", [])
            )
            else "FAIL",
        },
          {
              "check": "fair_baseline_limitations_retained",
              "status": "PASS"
              if s248.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED"
              and int(s248.get("provider_count", 0)) == 4
              and int(s248.get("method_count", 0)) == 5
              and all(
                  float(row.get("f1_difference", 0.0)) > 0.0
                  and float(row.get("f1_ci95_low", -1.0)) > 0.0
                  and float(row.get("f1_bootstrap_bh_q", 1.0)) <= 0.05
                  and bool(row.get("f1_primary_supported"))
                  for row in s248.get("comparison_rows", [])
                  if row.get("candidate") == "trust_valid_f1"
                  and row.get("baseline")
                  in {"robust_max_deviation", "isolation_forest"}
              )
              and all(
                  float(row.get("event_coverage_difference", 0.0)) > 0.0
                  and float(row.get("event_ci95_low", -1.0)) > 0.0
                  and float(
                      row.get("event_coverage_bootstrap_bh_q", 1.0)
                  )
                  <= 0.05
                  and bool(row.get("event_coverage_primary_supported"))
                  for row in s248.get("comparison_rows", [])
                  if row.get("candidate")
                  == "trust_nonstuck_event_macro"
                  and row.get("baseline")
                  in {"robust_max_deviation", "isolation_forest"}
              )
              and all(
                  bool(row.get("f1_direction_positive"))
                  for row in s248.get("leave_one_provider_rows", [])
                  if row.get("candidate") == "trust_valid_f1"
                  and row.get("baseline")
                  in {"robust_max_deviation", "isolation_forest"}
              )
              and all(
                  bool(row.get("event_direction_positive"))
                  for row in s248.get("leave_one_provider_rows", [])
                  if row.get("candidate")
                  == "trust_nonstuck_event_macro"
                  and row.get("baseline")
                  in {"robust_max_deviation", "isolation_forest"}
              )
            and any(
                row.get("candidate") == "trust_valid_f1"
                and row.get("baseline") == "one_class_svm"
                and float(row.get("f1_ci95_low", 1.0)) < 0.0
                for row in s248.get("comparison_rows", [])
            )
            and any(
                row.get("candidate") == "trust_valid_f1"
                and row.get("baseline") == "one_class_svm"
                and float(row.get("event_coverage_difference", 0.0)) < 0.0
                for row in s248.get("comparison_rows", [])
            )
              else "FAIL",
          },
          {
              "check": "natural_validation_boundary_retained",
              "status": "PASS"
              if s107.get("status")
              in {"PASS", "PASS_PIPELINE_REVIEW_REQUIRED"}
              and s109.get("status")
              in {"PASS", "PASS_PIPELINE_REVIEW_REQUIRED"}
              and s110.get("status")
              in {"PASS", "PASS_PIPELINE_REVIEW_REQUIRED"}
              and s116.get("status")
              in {"PASS", "PASS_PIPELINE_REVIEW_REQUIRED"}
              and s142.get("status")
              in {"PASS", "PASS_PIPELINE_REVIEW_REQUIRED"}
              and s153.get("status")
              in {"PASS", "PASS_PIPELINE_REVIEW_REQUIRED"}
              and s157.get("status")
              in {"PASS", "PASS_PIPELINE_REVIEW_REQUIRED"}
              else "FAIL",
              "value": {
                  "stage107": s107.get("status"),
                  "stage109": s109.get("status"),
                  "stage110": s110.get("status"),
                  "stage116": s116.get("status"),
                  "stage142": s142.get("status"),
                  "stage153": s153.get("status"),
                  "stage157": s157.get("status"),
              },
          },
        {
            "check": "causal_recovery_boundary_present",
            "status": "PASS"
            if float(s133.get("offline_temporal_attack_mae", 0.0))
            < float(s133.get("causal_temporal_attack_mae", 1.0))
            and float(
                next(
                    row["test_clean_harm_rate_mean"]
                    for row in s134["aggregate"]
                    if row["group"] == "safety_constrained_selected_by_valid"
                )
            )
            == 0.0
            else "FAIL",
        },
        {
            "check": "controlled_attack_boundary_retained",
            "status": "PASS",
            "value": "所有输入攻击均为透明受控注入，不是真实攻击日志。",
        },
    ]
    failed = [row for row in checks if row["status"] == "FAIL"]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED" if not failed else "FAIL",
        "protocol": (
            "阶段246只读取已经冻结的检测、迁移、非stuck复现和因果恢复结果，"
            "生成论文主表、主图和SHA256投稿包清单；不训练、不重新选择特征/"
            "窗口/权重/阈值、不重新评价任何test集。所有攻击实验均为透明"
            "受控注入，不是真实攻击日志。"
        ),
        "primary_far": 0.01,
        "provider_count": 4,
        "frozen_input_count": len(FROZEN_INPUTS),
        "generated_deliverable_count": len(generated_paths),
        "frozen_inputs": frozen_hashes,
        "generated_deliverables": generated_hashes,
        "checks": checks,
        "failed_checks": failed,
        "key_results": {
            "dwd_f1": dwd_f1,
            "dwd_event_coverage": dwd_event,
            "dwd_row_recall": dwd_row_recall,
            "cross_provider_f1_difference": s231[
                "provider_equal_summary"
            ]["f1_difference"],
            "cross_provider_event_difference": s231[
                "provider_equal_summary"
            ]["event_difference"],
            "stage244_f1_difference": s244["hierarchical_bootstrap"][
                "f1_difference"
            ],
            "stage244_f1_ci95_low": s244["hierarchical_bootstrap"][
                "f1_ci95_low"
            ],
            "stage245_f1_difference": s245["hierarchical_bootstrap"][
                "f1_difference"
            ],
            "stage245_f1_ci95_low": s245["hierarchical_bootstrap"][
                "f1_ci95_low"
            ],
            "stage245_event_difference": s245["hierarchical_bootstrap"][
                "event_coverage_difference"
            ],
            "stage245_event_ci95_low": s245["hierarchical_bootstrap"][
                "event_ci95_low"
            ],
            "provider_balanced_stage245_event_difference": next(
                row["event_coverage_difference"]
                for row in s247["provider_equal_rows"]
                if row["stage"] == "stage245"
            ),
            "provider_balanced_stage245_remove_eccc_f1_difference": next(
                row["f1_difference"]
                for row in s247["leave_one_provider_out_rows"]
                if row["stage"] == "stage245" and row["removed_source"] == "ECCC"
            ),
            "fair_baseline_trust_valid_f1_vs_robust_f1_difference": next(
                row["f1_difference"]
                for row in s248["comparison_rows"]
                if row["candidate"] == "trust_valid_f1"
                and row["baseline"] == "robust_max_deviation"
            ),
            "fair_baseline_trust_valid_f1_vs_isolation_f1_difference": next(
                row["f1_difference"]
                for row in s248["comparison_rows"]
                if row["candidate"] == "trust_valid_f1"
                and row["baseline"] == "isolation_forest"
            ),
            "fair_baseline_trust_nonstuck_vs_robust_event_difference": next(
                row["event_coverage_difference"]
                for row in s248["comparison_rows"]
                if row["candidate"] == "trust_nonstuck_event_macro"
                and row["baseline"] == "robust_max_deviation"
            ),
            "fair_baseline_trust_nonstuck_vs_isolation_event_difference": next(
                row["event_coverage_difference"]
                for row in s248["comparison_rows"]
                if row["candidate"] == "trust_nonstuck_event_macro"
                and row["baseline"] == "isolation_forest"
            ),
            "causal_recovery_utility": next(
                row["test_utility_mean"]
                for row in s134["aggregate"]
                if row["group"] == "safety_constrained_selected_by_valid"
            ),
            "causal_recovery_attack_mae": next(
                row["test_attack_mae_after_mean"]
                for row in s134["aggregate"]
                if row["group"] == "safety_constrained_selected_by_valid"
            ),
        },
        "limitations": [
            "所有攻击标签均为透明受控注入，不是真实攻击日志。",
            "非stuck扩展与独立种子复现仅方向一致，provider等权bootstrap区间仍跨零。",
            "机构等权非stuck审计中，事件覆盖在留一机构后方向仍为正，但F1在移除ECCC后转负，不能宣称F1优势不由单一机构驱动。",
            "优势主要集中在stuck攻击类型，不能宣称全部攻击类型普遍领先。",
            "公平基线审计中Trust的F1对Robust max deviation和Isolation Forest稳定为正，但One-Class SVM在F1与事件覆盖上呈互补关系，不能宣称对所有基线全面领先。",
            "严格因果恢复弱于离线双向插值，当前结果不能写成在线等同离线。",
            "只有四个跨机构目标，机构级精确检验功效有限。",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段246 冻结投稿包",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 主运行点：FAR=`{payload['primary_far']}`",
        f"- 冻结输入：`{payload['frozen_input_count']}`；生成交付物：`{payload['generated_deliverable_count']}`",
        "",
        "## 主表",
        "",
        "- `stage246_frozen_submission_package_table1_primary_summary.csv`",
        "- `stage246_frozen_submission_package_table2_cross_provider_transfer.csv`",
        "- `stage246_frozen_submission_package_table3_attack_type.csv`",
        "- `stage246_frozen_submission_package_table4_nonstuck_replication.csv`",
        "- `stage246_frozen_submission_package_table5_recovery_safety.csv`",
        "- `stage246_frozen_submission_package_table6_provider_balanced_nonstuck.csv`",
        "- `stage246_frozen_submission_package_table7_fair_baseline.csv`",
        "- `stage246_frozen_submission_package_table8_natural_validation.csv`",
        "",
        "## 主图",
        "",
        "- `figures/stage246_frozen_submission_package_figure1_transfer.png/pdf`",
        "- `figures/stage246_frozen_submission_package_figure2_attack_type.png/pdf`",
        "- `figures/stage246_frozen_submission_package_figure3_nonstuck.png/pdf`",
        "- `figures/stage246_frozen_submission_package_figure4_recovery.png/pdf`",
        "- `figures/stage246_frozen_submission_package_figure5_provider_balanced_nonstuck.png/pdf`",
        "- `figures/stage246_frozen_submission_package_figure6_fair_baseline.png/pdf`",
        "",
        "## 关键结论",
        "",
        f"- DWD受控基准：F1 `{dwd_f1:.6f}`、事件覆盖 `{dwd_event:.6f}`、行召回 `{dwd_row_recall:.6f}`、FAR `{dwd_far:.6f}`。",
        f"- 四机构等权迁移：F1 差值 `{s231['provider_equal_summary']['f1_difference']:+.6f}`、事件覆盖差值 `{s231['provider_equal_summary']['event_difference']:+.6f}`。",
        f"- 非stuck扩展：F1 差值 `{s244['hierarchical_bootstrap']['f1_difference']:+.6f}`，95% CI `[{s244['hierarchical_bootstrap']['f1_ci95_low']:+.6f}, {s244['hierarchical_bootstrap']['f1_ci95_high']:+.6f}]`。",
        f"- 非stuck独立种子复现：F1 差值 `{s245['hierarchical_bootstrap']['f1_difference']:+.6f}`，95% CI `[{s245['hierarchical_bootstrap']['f1_ci95_low']:+.6f}, {s245['hierarchical_bootstrap']['f1_ci95_high']:+.6f}]`。",
        f"- 非stuck机构等权留一审计：事件覆盖留一方向全正；stage245移除ECCC后F1差值 `{next(row['f1_difference'] for row in s247['leave_one_provider_out_rows'] if row['stage'] == 'stage245' and row['removed_source'] == 'ECCC'):+.6f}`。",
        f"- 跨机构公平基线：Trust(valid_f1)对Robust max deviation的F1差值 `{next(row['f1_difference'] for row in s248['comparison_rows'] if row['candidate'] == 'trust_valid_f1' and row['baseline'] == 'robust_max_deviation'):+.6f}`，对Isolation Forest `{next(row['f1_difference'] for row in s248['comparison_rows'] if row['candidate'] == 'trust_valid_f1' and row['baseline'] == 'isolation_forest'):+.6f}`；对One-Class SVM只呈互补，不能宣称全面领先。",
        "",
        "## 边界",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["limitations"])
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": payload["status"],
                "frozen_input_count": payload["frozen_input_count"],
                "generated_deliverable_count": payload[
                    "generated_deliverable_count"
                ],
                "failures": len(failed),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
