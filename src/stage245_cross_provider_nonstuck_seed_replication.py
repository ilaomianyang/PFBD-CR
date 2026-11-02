from __future__ import annotations

"""Stage 245: independent-seed replication of Stage 244.

Stage 244 extended the causal feature family and found positive point
estimates for four non-stuck attack types, but its provider-equal intervals
crossed zero and one provider drove the F1 estimate.  This stage predeclares
five seeds disjoint from the frozen Stage 244 seeds and replays the same
protocol, data, candidate grid, fitting, valid selection, and test-only
evaluation.  It does not modify or replace Stage 244 or any frozen main
result.  All attacks are transparent controlled injections, not real attack
logs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage245_cross_provider_nonstuck_seed_replication"
REPLICATION_SEEDS = [41011, 73189, 96001, 131071, 196613]

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    ATTACK_PARAMETERS,
    COMPONENTS,
    sha256_file,
)
from stage219_swiss_dwd_protection_transfer import clean_records  # noqa: E402
from stage240_nonstuck_balanced_selection import (  # noqa: E402
    NORMALIZED,
    PROTOCOLS,
)
from stage219_swiss_dwd_protection_transfer import (  # noqa: E402
    frozen_dwd_selection,
)
from stage244_cross_provider_nonstuck_candidate_features import (  # noqa: E402
    candidate_grid_stage244,
    evaluate_source,
    hierarchical_bootstrap,
    paired_station_table,
    station_metric_table,
)


def main() -> int:
    selection = pd.read_csv(
        TABLES / "stage208_dwd_persistent_repeat_protection_selection.csv"
    )
    evaluation_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    seed_audit: list[dict[str, object]] = []

    for source, protocol_path in PROTOCOLS.items():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        original_seeds = [int(value) for value in protocol["seeds"]]
        replication_protocol = dict(protocol)
        replication_protocol["seeds"] = list(REPLICATION_SEEDS)
        overlap = sorted(set(original_seeds) & set(REPLICATION_SEEDS))
        if overlap:
            raise RuntimeError(
                f"{source} replication seeds overlap frozen seeds: {overlap}"
            )
        seed_audit.append(
            {
                "source": source,
                "frozen_seeds": ",".join(str(value) for value in original_seeds),
                "replication_seeds": ",".join(
                    str(value) for value in REPLICATION_SEEDS
                ),
                "overlap_count": len(overlap),
            }
        )
        if len(original_seeds) != len(REPLICATION_SEEDS):
            raise RuntimeError(
                f"{source} frozen and replication seed counts differ"
            )
        frozen_by_seed = {
            replication_seed: frozen_dwd_selection(
                selection,
                frozen_seed,
                float(protocol["primary_far"]),
            )
            for frozen_seed, replication_seed in zip(
                original_seeds,
                REPLICATION_SEEDS,
                strict=True,
            )
        }
        source_eval, source_station, source_type, source_selection = (
            evaluate_source(
                source,
                replication_protocol,
                selection,
                frozen_selection_by_seed=frozen_by_seed,
            )
        )
        evaluation_rows.extend(source_eval)
        station_rows.extend(source_station)
        type_rows.extend(source_type)
        selection_rows.extend(source_selection)
        print(
            json.dumps(
                {
                    "source": source,
                    "status": "completed",
                    "rows": len(source_eval),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    evaluation = pd.DataFrame(evaluation_rows)
    station = pd.DataFrame(station_rows)
    type_detail = pd.DataFrame(type_rows)
    selections = pd.DataFrame(selection_rows)
    evaluation_aggregate = (
        evaluation.groupby(["source", "arm", "selection_rule"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            false_alarm_rate_mean=("false_alarm_rate", "mean"),
            average_precision_mean=("average_precision", "mean"),
            event_coverage_mean=("event_coverage", "mean"),
            event_coverage_std=("event_coverage", "std"),
            row_recall_mean=("row_recall", "mean"),
        )
        .sort_values(["source", "arm"])
    )
    station_metric = station_metric_table(station)
    paired, provider, leave_one = paired_station_table(station_metric)
    bootstrap = hierarchical_bootstrap(paired)

    type_aggregate = (
        type_detail.groupby(["source", "arm", "attack_type"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            attack_rows=("attack_rows", "sum"),
            detected_rows=("detected_rows", "sum"),
            row_recall=("row_recall", "mean"),
            event_coverage=("event_coverage", "mean"),
        )
        .sort_values(["source", "arm", "attack_type"])
    )
    nonstuck_arm = {
        source: f"{source.lower().replace(' ', '_')}_valid_nonstuck_event_macro"
        for source in PROTOCOLS
    }
    recalibrated_arm = {
        "MeteoSwiss": "swiss_valid_recalibrated",
        "NOAA ISD": "noaa_valid_recalibrated",
        "KNMI": "knmi_valid_recalibrated",
        "ECCC": "eccc_valid_recalibrated",
    }
    nonstuck_rows: list[dict[str, object]] = []
    for source in PROTOCOLS:
        baseline = type_aggregate[
            type_aggregate["source"].eq(source)
            & type_aggregate["arm"].eq(recalibrated_arm[source])
        ].set_index("attack_type")
        candidate = type_aggregate[
            type_aggregate["source"].eq(source)
            & type_aggregate["arm"].eq(nonstuck_arm[source])
        ].set_index("attack_type")
        for attack_type in sorted(candidate.index):
            if attack_type == "stuck":
                continue
            nonstuck_rows.append(
                {
                    "source": source,
                    "attack_type": attack_type,
                    "row_recall_difference": float(
                        candidate.loc[attack_type, "row_recall"]
                        - baseline.loc[attack_type, "row_recall"]
                    ),
                    "event_coverage_difference": float(
                        candidate.loc[attack_type, "event_coverage"]
                        - baseline.loc[attack_type, "event_coverage"]
                    ),
                }
            )
    nonstuck_type = pd.DataFrame(nonstuck_rows)
    nonstuck_type_equal = (
        nonstuck_type.groupby("attack_type", as_index=False)
        .agg(
            provider_count=("source", "nunique"),
            row_recall_difference=("row_recall_difference", "mean"),
            event_coverage_difference=("event_coverage_difference", "mean"),
        )
    )
    frozen = json.loads(
        (TABLES / "stage244_cross_provider_nonstuck_candidate_features.json")
        .read_text(encoding="utf-8")
    )
    frozen_bootstrap = frozen["hierarchical_bootstrap"]
    direction = {
        "frozen_f1_difference": float(frozen_bootstrap["f1_difference"]),
        "replication_f1_difference": float(bootstrap["f1_difference"]),
        "frozen_event_coverage_difference": float(
            frozen_bootstrap["event_coverage_difference"]
        ),
        "replication_event_coverage_difference": float(
            bootstrap["event_coverage_difference"]
        ),
        "f1_direction_consistent": bool(
            np.sign(frozen_bootstrap["f1_difference"])
            == np.sign(bootstrap["f1_difference"])
        ),
        "event_coverage_direction_consistent": bool(
            np.sign(frozen_bootstrap["event_coverage_difference"])
            == np.sign(bootstrap["event_coverage_difference"])
        ),
    }
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": (
            "阶段245预先声明五个不与阶段244冻结种子重叠的新攻击块种子，"
            "复用阶段244的四个目标机构、标准化数据、train/valid/test边界、"
            "透明受控攻击生成器、检测器拟合、91个候选特征、valid选择规则"
            "和test只评价边界；不修改或替换阶段244及冻结主结果。"
            "所有攻击都是透明受控注入，不是真实攻击日志。"
        ),
        "replication_seeds": REPLICATION_SEEDS,
        "provider_count": int(len(PROTOCOLS)),
        "station_count": int(
            station[["source", "station_id"]].drop_duplicates().shape[0]
        ),
        "candidate_count": int(len(candidate_grid_stage244())),
        "evaluation_rows": int(len(evaluation)),
        "station_rows": int(len(station)),
        "type_rows": int(len(type_detail)),
        "selection_rows": int(len(selections)),
        "seed_audit_rows": clean_records(seed_audit),
        "direction_consistency": direction,
        "hierarchical_bootstrap": bootstrap,
        "provider_station_rows": clean_records(provider.to_dict("records")),
        "leave_one_provider_out_rows": clean_records(
            leave_one.to_dict("records")
        ),
        "nonstuck_type_equal_rows": clean_records(
            nonstuck_type_equal.to_dict("records")
        ),
        "attack_parameters": ATTACK_PARAMETERS,
        "components": COMPONENTS,
        "protocol_sha256": {
            source: sha256_file(path)
            for source, path in PROTOCOLS.items()
        },
        "normalized_sha256": {
            source: sha256_file(path) for source, path in NORMALIZED.items()
        },
        "stage244_json_sha256": sha256_file(
            TABLES / "stage244_cross_provider_nonstuck_candidate_features.json"
        ),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "limitations": [
            "这是阶段244的独立种子复现，不替换冻结主结果。",
            "只有四个外部目标机构，攻击位置变化不等于新机构或真实攻击数据。",
            "所有攻击都是透明受控注入，不是真实攻击日志。",
        ],
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    evaluation.to_csv(
        TABLES / f"{PREFIX}_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station.to_csv(
        TABLES / f"{PREFIX}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    type_detail.to_csv(
        TABLES / f"{PREFIX}_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selections.to_csv(
        TABLES / f"{PREFIX}_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evaluation_aggregate.to_csv(
        TABLES / f"{PREFIX}_evaluation_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provider.to_csv(
        TABLES / f"{PREFIX}_provider_station.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leave_one.to_csv(
        TABLES / f"{PREFIX}_leave_one_provider_out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nonstuck_type.to_csv(
        TABLES / f"{PREFIX}_nonstuck_type_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nonstuck_type_equal.to_csv(
        TABLES / f"{PREFIX}_nonstuck_type_equal.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 阶段245 跨机构非stuck独立种子复现",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 复现种子：`{', '.join(str(value) for value in REPLICATION_SEEDS)}`",
        f"- 机构数：`{payload['provider_count']}`；站点数：`{payload['station_count']}`；候选数：`{payload['candidate_count']}`",
        "",
        "## 方向一致性",
        "",
        "| 指标 | 阶段244 | 阶段245 | 方向一致 |",
        "|---|---:|---:|---|",
        f"| F1 | {direction['frozen_f1_difference']:+.6f} | {direction['replication_f1_difference']:+.6f} | {direction['f1_direction_consistent']} |",
        f"| 事件覆盖 | {direction['frozen_event_coverage_difference']:+.6f} | {direction['replication_event_coverage_difference']:+.6f} | {direction['event_coverage_direction_consistent']} |",
        "",
        "## 复现层级bootstrap",
        "",
        f"- F1：`{bootstrap['f1_difference']:+.6f}`，95% CI `[{bootstrap['f1_ci95_low']:+.6f}, {bootstrap['f1_ci95_high']:+.6f}]`",
        f"- 事件覆盖：`{bootstrap['event_coverage_difference']:+.6f}`，95% CI `[{bootstrap['event_ci95_low']:+.6f}, {bootstrap['event_ci95_high']:+.6f}]`",
        "",
        "## 非stuck攻击类型",
        "",
        "| 攻击类型 | 行召回差值 | 事件覆盖差值 |",
        "|---|---:|---:|",
    ]
    for row in nonstuck_type_equal.to_dict("records"):
        lines.append(
            f"| {row['attack_type']} | {row['row_recall_difference']:+.6f} | "
            f"{row['event_coverage_difference']:+.6f} |"
        )
    lines += [
        "",
        "## 限制",
        "",
        "- 这是独立种子复现，不替换阶段244或冻结主结果。",
        "- 攻击块位置变化不等于新机构或真实攻击日志。",
        "- 只有四个外部目标机构。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "direction_consistency": direction,
                "hierarchical_bootstrap": bootstrap,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
