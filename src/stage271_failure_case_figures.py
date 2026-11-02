from __future__ import annotations

"""Stage 271: figure and case listing for failure, false-alarm, and miss analysis.

The stage is read-only. It uses event-level failure decomposition for NOAA
controlled injections and cross-provider negative-case decomposition for
provider transfer. It does not invent row-level cases where only aggregate
evidence exists and it does not train or recalibrate any model.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
PREFIX = "stage271_failure_case_figures"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


def _load_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _save_figure(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _noaa_failure_by_type() -> tuple[pd.DataFrame, Path]:
    frame = _load_csv(
        "stage189_noaa_event_level_failure_decomposition_type_aggregate.csv"
    )
    frame = frame.loc[
        frame["q"].eq(0.995) & frame["allocation_rule"].eq("risk")
    ].copy()
    if frame.empty:
        raise RuntimeError("NOAA failure type table is empty")
    order = [
        "spike",
        "offset",
        "drift",
        "time_shift",
        "gaussian_noise",
        "stuck",
    ]
    frame["attack_type"] = pd.Categorical(
        frame["attack_type"],
        categories=order,
        ordered=True,
    )
    frame = frame.sort_values("attack_type")
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    y = np.arange(len(frame))
    ax.barh(
        y,
        frame["event_coverage"],
        color="#2f6f9f",
        label="event coverage",
    )
    ax.barh(
        y,
        frame["ranking_failure_rate"],
        left=frame["event_coverage"],
        color="#d9a441",
        label="ranking failure",
    )
    ax.barh(
        y,
        frame["station_budget_suppression_rate"],
        left=(
            frame["event_coverage"] + frame["ranking_failure_rate"]
        ),
        color="#b55252",
        label="station-budget suppression",
    )
    ax.set_yticks(y, frame["attack_type"].astype(str))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of events")
    ax.set_title("NOAA low-FAR miss decomposition by attack type")
    ax.legend(frameon=False, ncols=3, loc="lower right")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    return frame, _save_figure(
        fig,
        "stage271_noaa_failure_type_stack.png",
    )


def _noaa_failure_by_length() -> tuple[pd.DataFrame, Path]:
    frame = _load_csv(
        "stage189_noaa_event_level_failure_decomposition_length_aggregate.csv"
    )
    frame = frame.loc[
        frame["q"].eq(0.995) & frame["allocation_rule"].eq("risk")
    ].copy()
    if frame.empty:
        raise RuntimeError("NOAA failure length table is empty")
    frame = frame.sort_values("event_rows")
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(frame))
    width = 0.38
    ax.bar(
        x - width / 2,
        frame["event_coverage"],
        width=width,
        color="#2f6f9f",
        label="event coverage",
    )
    ax.bar(
        x + width / 2,
        frame["ranking_failure_rate"],
        width=width,
        color="#d9a441",
        label="ranking failure",
    )
    ax.set_xticks(x, frame["event_rows"].astype(int).astype(str))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Event length (rows)")
    ax.set_ylabel("Fraction of events")
    ax.set_title("NOAA low-FAR miss decomposition by event length")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    return frame, _save_figure(
        fig,
        "stage271_noaa_failure_length.png",
    )


def _noaa_failure_by_risk() -> tuple[pd.DataFrame, Path]:
    frame = _load_csv(
        "stage189_noaa_event_level_failure_decomposition_risk_aggregate.csv"
    )
    frame = frame.loc[
        frame["q"].eq(0.995) & frame["allocation_rule"].eq("risk")
    ].copy()
    if frame.empty:
        raise RuntimeError("NOAA failure risk table is empty")
    order = [
        "[0.000,0.500)",
        "[0.500,0.800)",
        "[0.800,0.900)",
        "[0.900,0.950)",
        "[0.950,0.980)",
        "[0.980,0.990)",
        "[0.990,0.995)",
        "[0.995,0.999)",
        "[0.999,1.000]",
    ]
    frame["risk_bin"] = pd.Categorical(
        frame["risk_bin"],
        categories=order,
        ordered=True,
    )
    frame = frame.sort_values("risk_bin")
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    x = np.arange(len(frame))
    ax.plot(
        x,
        frame["event_coverage"],
        marker="o",
        color="#2f6f9f",
        label="event coverage",
    )
    ax.plot(
        x,
        frame["ranking_failure_rate"],
        marker="s",
        color="#d9a441",
        label="ranking failure",
    )
    ax.set_xticks(
        x,
        frame["risk_bin"].astype(str),
        rotation=45,
        ha="right",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Fraction of events")
    ax.set_xlabel("Event maximum risk percentile")
    ax.set_title("NOAA low-FAR miss decomposition by risk percentile")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    return frame, _save_figure(
        fig,
        "stage271_noaa_failure_risk.png",
    )


def _provider_negative_case() -> tuple[pd.DataFrame, Path]:
    station = _load_csv(
        "stage243_knmi_negative_case_decomposition_station_detail.csv"
    )
    metric = _load_csv(
        "stage243_knmi_negative_case_decomposition_metric_detail.csv"
    )
    if station.empty or metric.empty:
        raise RuntimeError("provider negative-case table is empty")
    station = station.copy()
    station["cases"] = (
        station["tp_target"]
        + station["fp_target"]
        + station["fn_target"]
        + station["tn_target"]
    )
    station = station.sort_values(
        ["target", "source", "f1_difference"],
        ascending=[True, True, True],
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.8),
        gridspec_kw={"width_ratios": [1.3, 1.0]},
    )
    labels = (
        station["target"].astype(str)
        + " <- "
        + station["source"].astype(str)
        + " / "
        + station["station_id"].astype(str)
    )
    axes[0].barh(
        np.arange(len(station)),
        station["f1_difference"],
        color=np.where(
            station["f1_difference"] >= 0,
            "#4b8b6a",
            "#b55252",
        ),
    )
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_yticks(np.arange(len(station)), labels)
    axes[0].set_xlabel("F1(target) - F1(source)")
    axes[0].set_title("Provider transfer station-level differences")
    axes[0].grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    axes[0].set_axisbelow(True)

    metric = metric.copy()
    metric_labels = (
        metric["metric"].astype(str)
        + "\\n"
        + metric["target"].astype(str)
        + " <- "
        + metric["source"].astype(str)
    )
    axes[1].barh(
        np.arange(len(metric)),
        metric["mean_target_minus_source"],
        color="#7a7a7a",
    )
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(metric)), metric_labels)
    axes[1].set_xlabel("Mean metric difference")
    axes[1].set_title("Provider transfer metric differences")
    axes[1].grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    axes[1].set_axisbelow(True)
    path = _save_figure(fig, "stage271_provider_negative_cases.png")
    station["case_scope"] = "station_level_transfer_difference"
    return station, path


def _case_listing(
    noaa_type: pd.DataFrame,
    noaa_length: pd.DataFrame,
    noaa_risk: pd.DataFrame,
    provider: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in noaa_type.to_dict("records"):
        rows.append(
            {
                "evidence_layer": "controlled_injection",
                "source_stage": "stage189_noaa_event_level_failure_decomposition",
                "case_scope": "attack_type",
                "case_key": str(record["attack_type"]),
                "events": int(record["events"]),
                "event_coverage": float(record["event_coverage"]),
                "ranking_failure_rate": float(
                    record["ranking_failure_rate"]
                ),
                "station_budget_suppression_rate": float(
                    record["station_budget_suppression_rate"]
                ),
                "failure_class": "mixed_event_level",
                "interpretation": (
                    "Fraction of events not covered by the frozen low-FAR "
                    "operating point, separated into ranking and budget "
                    "suppression classes."
                ),
            }
        )
    for record in noaa_length.to_dict("records"):
        rows.append(
            {
                "evidence_layer": "controlled_injection",
                "source_stage": "stage189_noaa_event_level_failure_decomposition",
                "case_scope": "event_length",
                "case_key": f"{int(record['event_rows'])} rows",
                "events": int(record["events"]),
                "event_coverage": float(record["event_coverage"]),
                "ranking_failure_rate": float(
                    record["ranking_failure_rate"]
                ),
                "station_budget_suppression_rate": float(
                    record["station_budget_suppression_rate"]
                ),
                "failure_class": "mixed_event_level",
                "interpretation": (
                    "Event-length stratification; short events are not "
                    "reported as a separate performance estimate."
                ),
            }
        )
    for record in noaa_risk.to_dict("records"):
        rows.append(
            {
                "evidence_layer": "controlled_injection",
                "source_stage": "stage189_noaa_event_level_failure_decomposition",
                "case_scope": "risk_percentile_bin",
                "case_key": str(record["risk_bin"]),
                "events": int(record["events"]),
                "event_coverage": float(record["event_coverage"]),
                "ranking_failure_rate": float(
                    record["ranking_failure_rate"]
                ),
                "station_budget_suppression_rate": float(
                    record["station_budget_suppression_rate"]
                ),
                "failure_class": "mixed_event_level",
                "interpretation": (
                    "Risk-bin diagnostic; not an additional confirmatory "
                    "operating point."
                ),
            }
        )
    for record in provider.to_dict("records"):
        rows.append(
            {
                "evidence_layer": "cross_provider_transfer",
                "source_stage": "stage243_knmi_negative_case_decomposition",
                "case_scope": "target_source_station",
                "case_key": (
                    f"{record['target']}<-{record['source']}/"
                    f"{record['station_id']}"
                ),
                "events": np.nan,
                "event_coverage": np.nan,
                "ranking_failure_rate": np.nan,
                "station_budget_suppression_rate": np.nan,
                "failure_class": (
                    "negative_transfer"
                    if float(record["f1_difference"]) < 0
                    else "nonnegative_transfer"
                ),
                "interpretation": (
                    "Station-level F1 difference from the negative-case "
                    "decomposition; all provider-transfer outcomes are "
                    "retained."
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _style()
    noaa_type, type_figure = _noaa_failure_by_type()
    noaa_length, length_figure = _noaa_failure_by_length()
    noaa_risk, risk_figure = _noaa_failure_by_risk()
    provider, provider_figure = _provider_negative_case()
    cases = _case_listing(
        noaa_type,
        noaa_length,
        noaa_risk,
        provider,
    )

    hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [
            TABLES
            / "stage189_noaa_event_level_failure_decomposition_type_aggregate.csv",
            TABLES
            / "stage189_noaa_event_level_failure_decomposition_length_aggregate.csv",
            TABLES
            / "stage189_noaa_event_level_failure_decomposition_risk_aggregate.csv",
            TABLES
            / "stage243_knmi_negative_case_decomposition_station_detail.csv",
            TABLES
            / "stage243_knmi_negative_case_decomposition_metric_detail.csv",
        ]
    }
    payload = {
        "status": "PASS_FAILURE_CASE_FIGURE_AND_LISTING_AUDIT",
        "protocol": (
            "Stage271 only reads frozen event-level failure attribution and "
            "provider negative-case decomposition artifacts. It produces "
            "diagnostic figures and a case listing; it does not train, "
            "recalibrate, or alter any prediction."
        ),
        "case_rows": int(len(cases)),
        "controlled_injection_rows": int(
            cases["evidence_layer"].eq("controlled_injection").sum()
        ),
        "cross_provider_rows": int(
            cases["evidence_layer"].eq("cross_provider_transfer").sum()
        ),
        "figures": [
            str(path.relative_to(ROOT))
            for path in [
                type_figure,
                length_figure,
                risk_figure,
                provider_figure,
            ]
        ],
        "source_artifact_note": (
            "NOAA failure classes are event-level and controlled-injection "
            "diagnostics. Provider negative cases are station-level "
            "differences from the frozen cross-provider transfer analysis."
        ),
        "limitations": [
            "The NOAA case listing is event-level; it is not row-level "
            "prediction evidence.",
            "The provider negative-case listing is station-level and does "
            "not identify individual alarm rows.",
            "Controlled injections and provider transfer results are not "
            "real production incident logs.",
            "The figures are diagnostic and do not replace the frozen "
            "primary endpoint.",
        ],
        "input_sha256": hashes,
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    cases.to_csv(
        TABLES / f"{PREFIX}_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = [
        "# 阶段271 失败、误报与漏报案例图",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 案例行数：`{payload['case_rows']}`",
        f"- 受控注入事件级行数：`{payload['controlled_injection_rows']}`",
        f"- 跨机构迁移站点级行数：`{payload['cross_provider_rows']}`",
        "",
        "## 图件",
        "",
        "- `figures/stage271_noaa_failure_type_stack.png`",
        "- `figures/stage271_noaa_failure_length.png`",
        "- `figures/stage271_noaa_failure_risk.png`",
        "- `figures/stage271_provider_negative_cases.png`",
        "",
        "## 边界",
        "",
        "- NOAA 图为事件级诊断，不是行级预测明细。",
        "- 跨机构图为站点级差异，不定位单条告警。",
        "- 受控注入和机构迁移结果不是真实生产事故日志。",
        "- 诊断图不替换冻结主终点。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "case_rows": payload["case_rows"],
                "figure_count": len(payload["figures"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
