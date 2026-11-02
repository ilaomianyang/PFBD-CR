"""Create the final submission figures from frozen package tables.

Figure contract
---------------
Core conclusion: V6 supports majority scenario-by-baseline leadership under a
controlled simulated-attack protocol, while preserving external-transfer and
online-recovery boundaries.
Evidence chain: scenario-level performance -> aggregation levels -> external
transfer -> online/offline recovery.
Archetype: quantitative grid.
Backend: Python/matplotlib only.
Export: editable SVG/PDF plus 600-dpi PNG/TIFF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
FIGURES = FINAL / "figures"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
    }
)

COLORS = {
    "trust": "#3B6FB6",
    "baseline": "#9AA6B2",
    "warning": "#C46B4A",
    "positive": "#4E8C6A",
    "neutral": "#6F7782",
}


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = FIGURES / name
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure1_scenario() -> None:
    scenario = pd.read_csv(TABLES / "final_v6_scenario_aggregate.csv")
    trust = scenario[scenario["method"].eq("trust_shape_union")].copy()
    trust = trust.sort_values("event_coverage", ascending=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.2),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    y = np.arange(len(trust))
    axes[0].barh(
        y,
        trust["event_coverage"],
        color=COLORS["trust"],
        alpha=0.9,
        height=0.65,
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(trust["scenario"])
    axes[0].set_xlabel("Event coverage")
    axes[0].set_title("A  Scenario-level event coverage")
    axes[0].set_xlim(0, 1.0)
    for yi, value in zip(y, trust["event_coverage"]):
        axes[0].text(
            min(value + 0.02, 0.96),
            yi,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=6,
        )

    wins = pd.read_csv(TABLES / "final_v6_scenario_wins.csv")
    wins = wins.set_index("scenario").loc[trust["scenario"]].reset_index()
    axes[1].barh(
        y,
        wins["win_share"],
        color=[
            COLORS["positive"] if value >= 0.70 else COLORS["warning"]
            for value in wins["win_share"]
        ],
        height=0.65,
    )
    axes[1].axvline(0.70, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Scenario-by-baseline win share")
    axes[1].set_title("B  Majority threshold")
    axes[1].set_xlim(0, 1.05)
    for yi, value, wins_count in zip(
        y, wins["win_share"], wins["wins"]
    ):
        axes[1].text(
            min(value + 0.02, 1.0),
            yi,
            f"{int(wins_count)}/11",
            va="center",
            ha="left",
            fontsize=6,
        )
    fig.suptitle(
        "V6 scenario performance: majority leadership with retained failure boundaries",
        x=0.02,
        ha="left",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "figure1_v6_scenario_performance")


def figure2_aggregation() -> None:
    data = pd.read_csv(TABLES / "final_v6_aggregation_levels.csv")
    methods = [
        "trust_shape_union",
        "isolation_forest",
        "one_class_svm",
        "pca_reconstruction",
        "knn_distance",
        "robust_max_deviation",
    ]
    labels = [
        "Trust union",
        "Isolation Forest",
        "One-Class SVM",
        "PCA",
        "kNN",
        "Robust dev.",
    ]
    levels = ["pooled", "provider_equal", "station_equal"]
    level_labels = ["Pooled", "Provider-equal", "Station-equal"]
    x = np.arange(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for idx, (level, label) in enumerate(zip(levels, level_labels)):
        values = []
        for method in methods:
            row = data[
                data["aggregation"].eq(level) & data["method"].eq(method)
            ]
            values.append(float(row["event_coverage"].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            label=label,
            color=[
                COLORS["trust"],
                COLORS["positive"],
                COLORS["baseline"],
            ][idx],
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Event coverage")
    ax.set_title("Aggregation sensitivity: Trust union versus selected baselines")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    save(fig, "figure2_v6_aggregation_levels")


def figure3_external_transfer() -> None:
    transfer = pd.read_csv(TABLES / "final_external_transfer.csv")
    transfer = transfer[
        transfer["trust_f1"].notna() & transfer["best_baseline_f1"].notna()
    ].copy()
    x = np.arange(len(transfer))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(
        x - width / 2,
        transfer["trust_f1"],
        width,
        color=COLORS["trust"],
        label="Trust",
    )
    ax.bar(
        x + width / 2,
        transfer["best_baseline_f1"],
        width,
        color=COLORS["warning"],
        label="Best baseline",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(transfer["source"])
    ax.set_ylabel("F1")
    ax.set_title("External transfer does not support universal superiority")
    ax.legend()
    for idx, row in transfer.iterrows():
        ax.text(
            idx,
            max(row["trust_f1"], row["best_baseline_f1"]) + 0.012,
            row["best_baseline"],
            ha="center",
            va="bottom",
            fontsize=6,
        )
    fig.tight_layout()
    save(fig, "figure3_external_transfer_boundary")


def figure4_recovery() -> None:
    recovery = pd.read_csv(TABLES / "final_online_offline_recovery.csv")
    recovery = recovery.sort_values("causal_penalty_mae", ascending=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = np.arange(len(recovery))
    ax.barh(
        y,
        recovery["causal_penalty_mae"],
        color=[
            COLORS["positive"] if value <= 0 else COLORS["warning"]
            for value in recovery["causal_penalty_mae"]
        ],
        height=0.65,
    )
    ax.axvline(0, color=COLORS["neutral"], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(recovery["attack_type"])
    ax.set_xlabel("Online minus offline MAE (causal penalty)")
    ax.set_title("Online causal recovery is not equivalent to offline interpolation")
    fig.tight_layout()
    save(fig, "figure4_online_offline_recovery")


def main() -> int:
    figure1_scenario()
    figure2_aggregation()
    figure3_external_transfer()
    figure4_recovery()
    print(
        {
            "status": "PASS_FINAL_FIGURES",
            "figures_dir": str(FIGURES),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
