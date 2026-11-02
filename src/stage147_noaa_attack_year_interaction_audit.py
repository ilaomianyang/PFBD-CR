from __future__ import annotations

"""Stage 147: attack-type by target-year interaction audit."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
SOURCE = TABLES / "stage141_noaa_frozen_external_validation_by_attack_type.csv"
METHOD_A = "TrustMeteo-QC frozen expanded"
METHOD_B = "TrustMeteo-QC default"


def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_year(values: pd.DataFrame, column: str, seed: int, repeats: int = 10000) -> tuple[float, float]:
    years = values["target_year"].to_numpy(dtype=int)
    lookup = values.set_index("target_year")[column].to_dict()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(years, size=(repeats, len(years)), replace=True)
    matrix = np.asarray([[lookup[int(year)] for year in row] for row in sampled], dtype=float)
    means = matrix.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    source = pd.read_csv(SOURCE)
    required = {METHOD_A, METHOD_B}
    if not required.issubset(set(source["method"].unique())):
        raise ValueError("stage141 attack-type detail is missing a required method")
    source["target_year"] = source["fold"].astype(str).str.extract(r"(\d{4})$")[0].astype(int)
    source = source[source["method"].isin(required)].copy()
    attacks = sorted(source["attack_type"].astype(str).unique().tolist())
    years = sorted(source["target_year"].unique().tolist())
    grouped = source.groupby(["target_year", "attack_type", "method"], as_index=False)["recall"].mean()
    wide = grouped.pivot_table(index=["target_year", "attack_type"], columns="method", values="recall", aggfunc="first").reset_index()
    wide.columns = ["_".join(str(part) for part in column if str(part) != "") if isinstance(column, tuple) else str(column) for column in wide.columns]
    wide = wide.rename(columns={METHOD_A: "expanded_recall", METHOD_B: "default_recall"})
    wide["expanded_minus_default"] = wide["expanded_recall"] - wide["default_recall"]
    detail = wide.sort_values(["target_year", "attack_type"]).reset_index(drop=True)
    paired_rows: list[dict[str, object]] = []
    for comparison, column in [("expanded - default", "expanded_minus_default")]:
        for attack_type in attacks:
            subset = detail.loc[detail["attack_type"].astype(str).eq(attack_type), ["target_year", column]].rename(columns={column: "value"})
            low, high = bootstrap_year(subset, "value", stable_seed("stage147", comparison, attack_type))
            paired_rows.append({"comparison": comparison, "attack_type": attack_type, "target_year_count": len(subset), "mean_diff": float(subset["value"].mean()), "year_bootstrap_ci95_low": low, "year_bootstrap_ci95_high": high, "positive_years": int((subset["value"] > 0).sum()), "negative_years": int((subset["value"] < 0).sum()), "zero_or_lower_recall_years": int((detail.loc[detail["attack_type"].astype(str).eq(attack_type), "expanded_recall"] <= 0).sum())})
    paired = pd.DataFrame(paired_rows)
    zero_cells = detail.loc[detail["expanded_recall"] <= 0, ["target_year", "attack_type", "expanded_recall"]].to_dict(orient="records")
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段141冻结NOAA攻击类型明细按目标年份和攻击类型分层；不重新训练、不重新选择候选、不根据test结果调参；每个攻击类型使用四个目标年份做年度bootstrap。攻击为透明受控注入，不是真实攻击日志。",
        "source_artifact": SOURCE.name,
        "target_years": years,
        "attack_types": attacks,
        "methods": [METHOD_A, METHOD_B],
        "detail": detail.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "zero_or_lower_recall_cells": zero_cells,
        "limitations": ["攻击类型×年度单元仍来自阶段141同一NOAA来源和受控注入生成器。", "每个攻击类型只有四个年度簇，bootstrap区间只能描述当前年度集合。", "阶段141没有One-Class SVM的攻击类型明细，因此本阶段不能做SVM逐类型比较。", "逐类型Recall下降或零召回必须保留，不能被总体F1掩盖，也不能宣称真实攻击能力。"],
    }
    prefix = "stage147_noaa_attack_year_interaction_audit"
    detail.to_csv(TABLES / f"{prefix}_detail.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / f"{prefix}_paired.csv", index=False, encoding="utf-8-sig")
    (TABLES / f"{prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段147 NOAA攻击类型×目标年度交互审计", "", "复用阶段141冻结攻击类型结果，按四个目标年度检查每类攻击的年度稳定性，不重新调参。", "", "| 比较 | 攻击类型 | 平均Recall差 | CI95 | 正向年份 | 负向年份 |", "|---|---|---:|---:|---:|---:|"]
    for row in paired.to_dict(orient="records"):
        lines.append(f"| {row['comparison']} | {row['attack_type']} | {row['mean_diff']:.4f} | [{row['year_bootstrap_ci95_low']:.4f}, {row['year_bootstrap_ci95_high']:.4f}] | {row['positive_years']} | {row['negative_years']} |")
    lines += ["", f"扩展候选Recall≤0的年度×攻击类型单元：`{len(zero_cells)}`", "", "该审计用于揭示类型级脆弱性和年度漂移，不支持全面领先或真实攻击检测声明。"]
    (REPORTS / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "target_years": len(years), "attack_types": len(attacks), "detail": len(detail), "paired": len(paired), "zero_cells": len(zero_cells)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
