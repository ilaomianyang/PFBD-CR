from __future__ import annotations

"""Read-only cross-window decomposition for the factor holdouts.

This stage never refits a detector, changes a threshold, or reads raw test
values.  It only joins the already audited scenario win/loss tables from four
temporal windows and records stable failures and direction reversals.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

WINDOWS = [
    ("FMI", "sep2026", "stage406_fmi_sep2026_factor_audit"),
    ("MET_NORWAY", "sep2026", "stage407_met_norway_sep2026_factor_audit"),
    ("FMI", "sep21", "stage414_fmi_sep21_factor_audit"),
    ("MET_NORWAY", "sep21", "stage415_met_norway_sep21_factor_audit"),
]
CANDIDATE = "trust_shape_union"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    frames: list[pd.DataFrame] = []
    input_hashes: dict[str, str] = {}
    audit_payloads: list[dict] = []
    for source, window, audit_prefix in WINDOWS:
        audit_path = TABLES / f"{audit_prefix}.json"
        win_path = TABLES / f"{audit_prefix}_scenario_win_loss.csv"
        audit_payloads.append(json.loads(audit_path.read_text(encoding="utf-8")))
        input_hashes[str(audit_path)] = sha256(audit_path)
        input_hashes[str(win_path)] = sha256(win_path)
        frame = pd.read_csv(win_path)
        frame = frame[frame["candidate"].eq(CANDIDATE)].copy()
        frame["source"] = source
        frame["window"] = window
        frames.append(frame)

    detail = pd.concat(frames, ignore_index=True)
    detail["positive"] = detail["difference"] > 1e-12
    detail["negative"] = detail["difference"] < -1e-12
    detail["zero"] = ~detail["positive"] & ~detail["negative"]

    by_scenario = (
        detail.groupby(["baseline", "scenario"], as_index=False)
        .agg(
            windows=("window", "nunique"),
            positive_windows=("positive", "sum"),
            negative_windows=("negative", "sum"),
            zero_windows=("zero", "sum"),
            mean_difference=("difference", "mean"),
            min_difference=("difference", "min"),
            max_difference=("difference", "max"),
        )
        .sort_values(["scenario", "baseline"])
    )
    by_baseline = (
        detail.groupby("baseline", as_index=False)
        .agg(
            windows=("window", "nunique"),
            scenario_rows=("scenario", "size"),
            positive_rows=("positive", "sum"),
            negative_rows=("negative", "sum"),
            zero_rows=("zero", "sum"),
            mean_difference=("difference", "mean"),
        )
        .sort_values("mean_difference", ascending=False)
    )
    unseen = detail[detail["scenario"].eq("coordinated_drift_unseen")].copy()
    unseen = unseen[
        [
            "source",
            "window",
            "baseline",
            "candidate_event_coverage",
            "baseline_event_coverage",
            "difference",
            "win",
        ]
    ].sort_values(["baseline", "source", "window"])

    stable_nonpositive = by_scenario[
        (by_scenario["negative_windows"] + by_scenario["zero_windows"])
        == by_scenario["windows"]
    ]
    direction_flip = by_scenario[
        (by_scenario["positive_windows"] > 0)
        & (by_scenario["negative_windows"] > 0)
    ]
    audit_summary = {
        "status": "PASS_CROSS_TEMPORAL_FACTOR_FAILURE_DECOMPOSITION",
        "candidate": CANDIDATE,
        "window_count": len(WINDOWS),
        "windows": [
            {"source": source, "window": window, "audit_prefix": prefix}
            for source, window, prefix in WINDOWS
        ],
        "baseline_count": int(detail["baseline"].nunique()),
        "scenario_count": int(detail["scenario"].nunique()),
        "detail_rows": int(len(detail)),
        "stable_nonpositive_cells": int(len(stable_nonpositive)),
        "direction_flip_cells": int(len(direction_flip)),
        "unseen_rows": int(len(unseen)),
        "unseen_positive_rows": int((unseen["difference"] > 1e-12).sum()),
        "unseen_zero_rows": int((unseen["difference"].abs() <= 1e-12).sum()),
        "unseen_negative_rows": int((unseen["difference"] < -1e-12).sum()),
        "all_attacks_controlled_simulations": True,
        "exploratory_only": True,
        "claim_boundary": [
            "Read-only synthesis of already audited temporal windows.",
            "The windows were added after earlier results were observed and are exploratory.",
            "This stage cannot promote any result to confirmatory evidence.",
            "No threshold, method, baseline, or test result was changed.",
        ],
        "stable_nonpositive": stable_nonpositive.to_dict("records"),
        "direction_flips": direction_flip.to_dict("records"),
        "input_sha256": input_hashes,
        "audit_statuses": [payload["status"] for payload in audit_payloads],
    }

    by_scenario.to_csv(
        TABLES / "stage417_cross_temporal_factor_failure_by_scenario.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_baseline.to_csv(
        TABLES / "stage417_cross_temporal_factor_failure_by_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unseen.to_csv(
        TABLES / "stage417_cross_temporal_factor_unseen.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / "stage417_cross_temporal_factor_failure_decomposition.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage417 跨时间窗因子失败分解",
        "",
        "- 状态：`PASS_CROSS_TEMPORAL_FACTOR_FAILURE_DECOMPOSITION`",
        "- 类型：只读、探索性证据整合；未重新拟合、调阈值或读取原始 test 值。",
        f"- 时间窗：`{len(WINDOWS)}`；基线：`{audit_summary['baseline_count']}`；场景：`{audit_summary['scenario_count']}`。",
        f"- 稳定非正比较单元：`{audit_summary['stable_nonpositive_cells']}`。",
        f"- 方向翻转比较单元：`{audit_summary['direction_flip_cells']}`。",
        f"- 未见 coordinated_drift_unseen 行：正向 `{audit_summary['unseen_positive_rows']}`；零 `{audit_summary['unseen_zero_rows']}`；负 `{audit_summary['unseen_negative_rows']}`。",
        "",
        "## 未见组合",
        "",
        unseen.to_markdown(index=False),
        "",
        "## 解释边界",
        "",
        "- 所有攻击均为真实观测背景上的透明受控模拟，不是真实生产攻击。",
        "- 该分解不能把探索性窗口升级为确认性主结果。",
        "- 稳定负向或方向翻转结果必须保留，不能用窗口选择规避。",
    ]
    (REPORTS / "stage417_cross_temporal_factor_failure_decomposition.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": audit_summary["status"],
                "detail_rows": audit_summary["detail_rows"],
                "stable_nonpositive_cells": audit_summary["stable_nonpositive_cells"],
                "direction_flip_cells": audit_summary["direction_flip_cells"],
                "unseen_positive_rows": audit_summary["unseen_positive_rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
