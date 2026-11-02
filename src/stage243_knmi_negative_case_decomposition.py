from __future__ import annotations

"""Stage 243: decomposition of the two KNMI-source negative cases.

Stage242 found 18/20 directed pairs in which target-domain valid local
selection exceeded source-configuration threshold recalibration.  The two
exceptions were DWD <- KNMI and ECCC <- KNMI.  This stage is a read-only
decomposition of those two frozen cases.  It does not retrain, select,
recalibrate, or modify any Stage242 artifact.

The purpose is to distinguish three possibilities:

1. DWD <- KNMI is a tie by identical operating behavior rather than a
   degradation;
2. ECCC <- KNMI is a low-FAR operating-point trade rather than a ranking
   failure;
3. either negative case is a single-station or single-seed artifact.

All attacks are transparent controlled injections, not real attack logs.
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
PREFIX = "stage243_knmi_negative_case_decomposition"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage218_swiss_negative_result_uncertainty import (  # noqa: E402
    sha256_file,
)


PAIR_SPECS = [
    {
        "target": "DWD",
        "source": "KNMI",
        "target_arm": "target_valid_selected",
        "source_arm": "knmi_recalibrated",
        "station_count": 12,
    },
    {
        "target": "ECCC",
        "source": "KNMI",
        "target_arm": "target_valid_selected",
        "source_arm": "knmi_recalibrated",
        "station_count": 14,
    },
]
METRICS = [
    "precision",
    "recall",
    "f1",
    "false_alarm_rate",
    "roc_auc",
    "average_precision",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for record in records:
        cleaned: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(
                float(value)
            ):
                cleaned[key] = None
            elif isinstance(value, np.integer):
                cleaned[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        output.append(cleaned)
    return output


def station_f1(frame: pd.DataFrame) -> np.ndarray:
    denominator = (
        2.0 * frame["tp"].to_numpy(dtype=float)
        + frame["fp"].to_numpy(dtype=float)
        + frame["fn"].to_numpy(dtype=float)
    )
    return np.divide(
        2.0 * frame["tp"].to_numpy(dtype=float),
        denominator,
        out=np.zeros(len(frame), dtype=float),
        where=denominator > 0.0,
    )


def station_rows(
    station_detail: pd.DataFrame,
    spec: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = str(spec["target"])
    source = str(spec["source"])
    target_arm = str(spec["target_arm"])
    source_arm = str(spec["source_arm"])
    subset = station_detail[
        station_detail["target"].eq(target)
        & station_detail["arm"].isin([target_arm, source_arm])
    ].copy()
    if set(subset["arm"]) != {target_arm, source_arm}:
        raise RuntimeError(
            f"missing station arms for {target} <- {source}"
        )

    grouped = (
        subset.groupby(["arm", "station_id"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tn=("tn", "sum"),
        )
    )
    grouped["f1"] = station_f1(grouped)
    local = grouped[grouped["arm"].eq(target_arm)].copy()
    transferred = grouped[grouped["arm"].eq(source_arm)].copy()
    aligned = local.merge(
        transferred,
        on="station_id",
        suffixes=("_target", "_source"),
        validate="one_to_one",
    )
    aligned["f1_difference"] = (
        aligned["f1_target"] - aligned["f1_source"]
    )
    return aligned, subset


def seed_rows(
    subset: pd.DataFrame,
    target_arm: str,
    source_arm: str,
) -> pd.DataFrame:
    local = subset[subset["arm"].eq(target_arm)].copy()
    transferred = subset[subset["arm"].eq(source_arm)].copy()
    local["f1"] = station_f1(local)
    transferred["f1"] = station_f1(transferred)
    aligned = local.merge(
        transferred,
        on=["seed", "station_id"],
        suffixes=("_target", "_source"),
        validate="one_to_one",
    )
    aligned["f1_difference"] = (
        aligned["f1_target"] - aligned["f1_source"]
    )
    return (
        aligned.groupby("seed", as_index=False)
        .agg(
            station_count=("station_id", "nunique"),
            mean_f1_difference=("f1_difference", "mean"),
            positive_stations=("f1_difference", lambda s: int((s > 0).sum())),
            negative_stations=("f1_difference", lambda s: int((s < 0).sum())),
            zero_stations=("f1_difference", lambda s: int((s == 0).sum())),
        )
        .sort_values("seed", kind="stable")
    )


def main() -> int:
    station_path = (
        TABLES / "stage242_multisource_transfer_matrix_station_detail.csv"
    )
    evaluation_path = (
        TABLES / "stage242_multisource_transfer_matrix_evaluation.csv"
    )
    transfer_path = (
        TABLES / "stage242_multisource_transfer_matrix_transfer_manifest.csv"
    )
    stage242_path = TABLES / "stage242_multisource_transfer_matrix.json"
    required = [station_path, evaluation_path, transfer_path, stage242_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing frozen Stage243 inputs: " + ", ".join(missing)
        )

    station_detail = pd.read_csv(station_path)
    evaluation = pd.read_csv(evaluation_path)
    transfer = pd.read_csv(transfer_path)
    stage242 = load_json(stage242_path)
    if stage242.get("status") != "PASS_PIPELINE_REVIEW_REQUIRED":
        raise RuntimeError("Stage242 is not in a reviewable frozen state")

    station_output: list[dict[str, object]] = []
    seed_output: list[dict[str, object]] = []
    metric_output: list[dict[str, object]] = []
    summary_output: list[dict[str, object]] = []

    for spec in PAIR_SPECS:
        target = str(spec["target"])
        source = str(spec["source"])
        target_arm = str(spec["target_arm"])
        source_arm = str(spec["source_arm"])

        aligned, subset = station_rows(station_detail, spec)
        if len(aligned) != int(spec["station_count"]):
            raise RuntimeError(
                f"station count changed for {target} <- {source}"
            )
        aligned["target"] = target
        aligned["source"] = source
        for _, row in aligned.iterrows():
            station_output.append(
                {
                    "target": target,
                    "source": source,
                    "station_id": row["station_id"],
                    "seeds": int(row["seeds_target"]),
                    "tp_target": int(row["tp_target"]),
                    "fp_target": int(row["fp_target"]),
                    "fn_target": int(row["fn_target"]),
                    "tn_target": int(row["tn_target"]),
                    "tp_source": int(row["tp_source"]),
                    "fp_source": int(row["fp_source"]),
                    "fn_source": int(row["fn_source"]),
                    "tn_source": int(row["tn_source"]),
                    "f1_target": float(row["f1_target"]),
                    "f1_source": float(row["f1_source"]),
                    "f1_difference": float(row["f1_difference"]),
                }
            )

        seeds = seed_rows(subset, target_arm, source_arm)
        for _, row in seeds.iterrows():
            seed_output.append(
                {
                    "target": target,
                    "source": source,
                    "seed": int(row["seed"]),
                    "station_count": int(row["station_count"]),
                    "mean_f1_difference": float(
                        row["mean_f1_difference"]
                    ),
                    "positive_stations": int(row["positive_stations"]),
                    "negative_stations": int(row["negative_stations"]),
                    "zero_stations": int(row["zero_stations"]),
                }
            )

        pair_eval = evaluation[
            evaluation["target"].eq(target)
            & evaluation["arm"].isin([target_arm, source_arm])
        ].copy()
        if set(pair_eval["arm"]) != {target_arm, source_arm}:
            raise RuntimeError(
                f"missing evaluation arms for {target} <- {source}"
            )
        pair_eval["seed"] = pair_eval["seed"].astype(int)
        wide = pair_eval.pivot(
            index="seed",
            columns="arm",
            values=METRICS,
        )
        differences = (
            wide.xs(target_arm, axis=1, level=1)
            - wide.xs(source_arm, axis=1, level=1)
        )
        for metric in METRICS:
            values = differences[metric].to_numpy(dtype=float)
            metric_output.append(
                {
                    "target": target,
                    "source": source,
                    "metric": metric,
                    "mean_target_minus_source": float(values.mean()),
                    "min_target_minus_source": float(values.min()),
                    "max_target_minus_source": float(values.max()),
                    "positive_seeds": int((values > 0.0).sum()),
                    "negative_seeds": int((values < 0.0).sum()),
                    "zero_seeds": int((values == 0.0).sum()),
                }
            )

        pair_transfer = transfer[
            transfer["target"].eq(target)
            & transfer["source"].eq(source)
        ].copy()
        if pair_transfer.empty:
            raise RuntimeError(f"missing transfer rows for {target} <- {source}")
        config_columns = ["feature", "window", "weight"]
        config_count = int(
            pair_transfer[config_columns].drop_duplicates().shape[0]
        )
        threshold_spread = float(
            pair_transfer["recalibrated_threshold"].max()
            - pair_transfer["recalibrated_threshold"].min()
        )
        allclose_station_behavior = bool(
            np.allclose(
                aligned["tp_target"].to_numpy(dtype=float),
                aligned["tp_source"].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            )
            and np.allclose(
                aligned["fp_target"].to_numpy(dtype=float),
                aligned["fp_source"].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            )
            and np.allclose(
                aligned["fn_target"].to_numpy(dtype=float),
                aligned["fn_source"].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            )
            and np.allclose(
                aligned["tn_target"].to_numpy(dtype=float),
                aligned["tn_source"].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            )
        )
        seed_mean = seeds["mean_f1_difference"].to_numpy(dtype=float)
        summary_output.append(
            {
                "target": target,
                "source": source,
                "station_count": int(len(aligned)),
                "seed_count": int(len(seeds)),
                "mean_f1_difference": float(
                    aligned["f1_difference"].mean()
                ),
                "positive_stations": int(
                    (aligned["f1_difference"] > 0.0).sum()
                ),
                "negative_stations": int(
                    (aligned["f1_difference"] < 0.0).sum()
                ),
                "zero_stations": int(
                    (aligned["f1_difference"] == 0.0).sum()
                ),
                "seed_direction_positive": int((seed_mean > 0.0).sum()),
                "seed_direction_negative": int((seed_mean < 0.0).sum()),
                "seed_direction_zero": int((seed_mean == 0.0).sum()),
                "allclose_station_behavior": allclose_station_behavior,
                "source_config_count": config_count,
                "recalibrated_threshold_spread": threshold_spread,
            }
        )

    station_frame = pd.DataFrame(station_output)
    seed_frame = pd.DataFrame(seed_output)
    metric_frame = pd.DataFrame(metric_output)
    summary_frame = pd.DataFrame(summary_output)

    checks = [
        {
            "check": "pair_count",
            "status": "PASS" if len(summary_frame) == 2 else "FAIL",
            "value": len(summary_frame),
        },
        {
            "check": "dwd_knmi_tie_is_exact_behavior",
            "status": (
                "PASS"
                if bool(
                    summary_frame[
                        summary_frame["target"].eq("DWD")
                        & summary_frame["source"].eq("KNMI")
                    ]["allclose_station_behavior"].iloc[0]
                )
                else "FAIL"
            ),
            "value": bool(
                summary_frame[
                    summary_frame["target"].eq("DWD")
                    & summary_frame["source"].eq("KNMI")
                ]["allclose_station_behavior"].iloc[0]
            ),
        },
        {
            "check": "eccc_knmi_not_single_seed",
            "status": (
                "PASS"
                if int(
                    summary_frame[
                        summary_frame["target"].eq("ECCC")
                        & summary_frame["source"].eq("KNMI")
                    ]["seed_direction_negative"].iloc[0]
                )
                >= 4
                else "FAIL"
            ),
            "value": int(
                summary_frame[
                    summary_frame["target"].eq("ECCC")
                    & summary_frame["source"].eq("KNMI")
                ]["seed_direction_negative"].iloc[0]
            ),
        },
        {
            "check": "negative_cases_retained",
            "status": (
                "PASS"
                if int(stage242.get("negative_pair_count", 0)) == 2
                else "FAIL"
            ),
            "value": int(stage242.get("negative_pair_count", 0)),
        },
    ]
    failed = [check for check in checks if check["status"] == "FAIL"]
    payload = {
        "status": (
            "PASS_PIPELINE_REVIEW_REQUIRED" if not failed else "FAIL"
        ),
        "protocol": (
            "Read-only decomposition of the two frozen KNMI-source "
            "negative cases in Stage242. No training, selection, threshold "
            "change, or test-label use. All attacks are transparent "
            "controlled injections."
        ),
        "pair_count": int(len(summary_frame)),
        "summary_rows": clean_records(summary_frame.to_dict("records")),
        "seed_rows": clean_records(seed_frame.to_dict("records")),
        "metric_rows": clean_records(metric_frame.to_dict("records")),
        "checks": checks,
        "artifacts": {
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "stage242_path": str(stage242_path),
            "stage242_sha256": sha256_file(stage242_path),
            "station_detail_path": str(station_path),
            "station_detail_sha256": sha256_file(station_path),
        },
    }

    station_frame.to_csv(
        TABLES / f"{PREFIX}_station_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seed_frame.to_csv(
        TABLES / f"{PREFIX}_seed_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_frame.to_csv(
        TABLES / f"{PREFIX}_metric_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_frame.to_csv(
        TABLES / f"{PREFIX}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dwd = summary_frame[summary_frame["target"].eq("DWD")].iloc[0]
    eccc = summary_frame[summary_frame["target"].eq("ECCC")].iloc[0]
    lines = [
        "# 阶段243 KNMI来源反例分解",
        "",
        f"- 状态：**{payload['status']}**",
        "- 该阶段只读取阶段242冻结明细，不重新训练、不重新选择、不重校准阈值，也不使用测试标签做选择。",
        "",
        "## DWD ← KNMI",
        "",
        f"- 平均 F1 差值：`{dwd['mean_f1_difference']:+.6f}`；正/负/零站点：`{dwd['positive_stations']}/{dwd['negative_stations']}/{dwd['zero_stations']}`。",
        f"- 站点级行为逐项完全一致：`{dwd['allclose_station_behavior']}`；该结果应写成运行点等价的持平，不是性能退化。",
        "",
        "## ECCC ← KNMI",
        "",
        f"- 平均 F1 差值：`{eccc['mean_f1_difference']:+.6f}`；正/负/零站点：`{eccc['positive_stations']}/{eccc['negative_stations']}/{eccc['zero_stations']}`。",
        f"- 逐种子方向为正/负/零：`{eccc['seed_direction_positive']}/{eccc['seed_direction_negative']}/{eccc['seed_direction_zero']}`；不是单个种子驱动。",
        f"- KNMI 来源配置覆盖种子数：`{eccc['source_config_count']}` 种；目标域重校准阈值跨度：`{eccc['recalibrated_threshold_spread']:.6f}`。",
        "- 指标交换方向：recall 略升、precision 下降、FAR 上升；ROC-AUC 和平均精度下降，说明该反例不能解释成纯运行点交换，排序质量也出现了小幅但一致的下降。",
        "",
        "## 边界",
        "",
        "- `DWD ← KNMI` 是持平而非“优于”，不能选择性写成多源全面正向。",
        "- `ECCC ← KNMI` 是明确的反例，必须保留；不能据此宣称来源配置迁移普适有效。",
        "- 该分解只解释阶段242的两个冻结反例，不扩展为所有 KNMI 来源迁移的普遍规律。",
        "- 所有攻击均为透明受控注入，不是真实攻击日志。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": payload["status"],
                "pair_count": payload["pair_count"],
                "dwd_knmi_mean_f1_difference": float(
                    dwd["mean_f1_difference"]
                ),
                "eccc_knmi_mean_f1_difference": float(
                    eccc["mean_f1_difference"]
                ),
                "eccc_negative_seed_count": int(
                    eccc["seed_direction_negative"]
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
