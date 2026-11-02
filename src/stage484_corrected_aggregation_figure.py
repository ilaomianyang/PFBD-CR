"""Stage 484: figure from corrected V6 aggregation estimands."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
OUT = FINAL / "figures_corrected"

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


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / name
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    frame = pd.read_csv(
        TABLES / "final_v6_aggregation_levels_corrected.csv"
    )
    methods = [
        "trust_shape_union",
        "trust_only",
        "isolation_forest",
        "one_class_svm",
        "pca_reconstruction",
        "knn_distance",
    ]
    labels = [
        "Trust union",
        "Trust only",
        "Isolation Forest",
        "One-Class SVM",
        "PCA",
        "kNN",
    ]
    levels = [
        "pooled_counts",
        "provider_equal_counts",
        "station_equal_counts",
    ]
    level_labels = [
        "Count-pooled",
        "Provider-equal",
        "Station-equal",
    ]
    colors = ["#3B6FB6", "#4E8C6A", "#C46B4A"]
    x = np.arange(len(methods))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for axis, metric, title in (
        (axes[0], "f1", "A  F1 by corrected estimand"),
        (axes[1], "event_coverage", "B  Event coverage by corrected estimand"),
    ):
        for index, (level, label) in enumerate(zip(levels, level_labels)):
            values = []
            for method in methods:
                row = frame[
                    frame["aggregation"].eq(level)
                    & frame["method"].eq(method)
                ]
                values.append(
                    float(row[metric].iloc[0]) if not row.empty else np.nan
                )
            axis.bar(
                x + (index - 1) * width,
                values,
                width=width,
                color=colors[index],
                label=label,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylim(0.0, max(0.45, float(frame[metric].max()) * 1.15))
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title(title)
        axis.legend(loc="upper right", ncol=1)
    fig.suptitle(
        "Corrected V6 aggregation: count-pooled and equal-weight summaries",
        x=0.02,
        ha="left",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "figure2_v6_aggregation_levels_corrected")
    print(
        {
            "status": "PASS_CORRECTED_AGGREGATION_FIGURE",
            "output_dir": str(OUT),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
