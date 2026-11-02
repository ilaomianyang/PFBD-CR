from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
INPUT = TABLES / "stage105_noaa_attack_type_valid_selected_detail.csv"
ATTACK_TYPES = ["drift", "gaussian_noise", "offset", "spike", "stuck", "time_shift"]
REFERENCES = ["svm_only", "trust_component_only"]
BOOTSTRAPS = 5000


def sample_indices(fold_indices: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    sampled_folds = rng.integers(0, len(fold_indices), size=len(fold_indices))
    chunks: list[np.ndarray] = []
    for fold_pos in sampled_folds:
        block = fold_indices[int(fold_pos)]
        chunks.append(block[rng.integers(0, len(block), size=len(block))])
    return np.concatenate(chunks)


def main() -> int:
    detail = pd.read_csv(INPUT)
    required = {"fold", "seed", "method", "attack_type", "recall"}
    if not required.issubset(detail.columns):
        raise ValueError(f"stage105 detail missing columns: {sorted(required.difference(detail.columns))}")
    methods = {"valid_selected", "svm_only", "trust_component_only"}
    if set(detail["method"].unique()) != methods:
        raise ValueError(f"unexpected stage105 methods: {sorted(detail['method'].unique())}")
    if set(detail["attack_type"].unique()) != set(ATTACK_TYPES):
        raise ValueError(f"unexpected attack types: {sorted(detail['attack_type'].unique())}")

    rng = np.random.default_rng(20260922)
    result_rows: list[dict[str, object]] = []
    for attack_type in ATTACK_TYPES:
        subset = detail.loc[detail["attack_type"].eq(attack_type), ["fold", "seed", "method", "recall"]]
        wide = subset.pivot_table(index=["fold", "seed"], columns="method", values="recall")
        if wide.isna().any().any() or len(wide) != 20:
            raise ValueError(f"incomplete paired attack table for {attack_type}: rows={len(wide)}")
        fold_values = wide.index.get_level_values("fold").to_numpy()
        fold_indices = [np.flatnonzero(fold_values == fold) for fold in wide.index.get_level_values("fold").unique()]
        for reference in REFERENCES:
            diff = wide["valid_selected"].to_numpy(dtype=float) - wide[reference].to_numpy(dtype=float)
            boot = np.asarray([float(diff[sample_indices(fold_indices, rng)].mean()) for _ in range(BOOTSTRAPS)])
            result_rows.append({
                "attack_type": attack_type,
                "candidate": "valid_selected",
                "reference": reference,
                "candidate_recall_mean": float(wide["valid_selected"].mean()),
                "reference_recall_mean": float(wide[reference].mean()),
                "observed_diff": float(diff.mean()),
                "bootstrap_diff_mean": float(boot.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "prob_candidate_better": float((boot > 0).mean()),
                "blocks": int(len(wide)),
                "bootstraps": BOOTSTRAPS,
            })

    out = pd.DataFrame(result_rows)
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED",
        "protocol": "阶段105六类攻击Recall的折×种子层级配对bootstrap；每次先按四个滚动折有放回抽样，再按抽中折内五个种子有放回抽样；不重新选择权重。",
        "source": str(INPUT),
        "candidate": "valid_selected",
        "references": REFERENCES,
        "attack_types": ATTACK_TYPES,
        "blocks": 20,
        "bootstraps": BOOTSTRAPS,
        "results": out.to_dict(orient="records"),
        "limitations": [
            "bootstrap单位为折×种子，不是逐站点簇；站点簇总体证据见阶段100。",
            "攻击类型为透明受控注入，不是真实攻击日志。",
            "该分析只针对攻击行Recall；综合F1、Precision、AUPRC、FAR见阶段105总体表。",
        ],
    }
    out.to_csv(TABLES / "stage106_noaa_attack_type_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage106_noaa_attack_type_paired_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    lines = ["# 阶段106 NOAA逐攻击类型Recall配对bootstrap", "", "对阶段105的六类攻击Recall执行折×种子层级配对bootstrap；差值定义为valid_selected减参考。", "", "| 攻击类型 | 参考 | selected Recall | 参考 Recall | 差值 | 95% CI | selected更优概率 |", "|---|---|---:|---:|---:|---|---:|"]
    for row in result_rows:
        lines.append(f"| {row['attack_type']} | {row['reference']} | {row['candidate_recall_mean']:.4f} | {row['reference_recall_mean']:.4f} | {row['observed_diff']:+.4f} | [{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}] | {row['prob_candidate_better']:.4f} |")
    lines += ["", "该表用于攻击类型诊断，不能替代综合性能，也不能据此宣称对真实攻击普适。"]
    (REPORTS / "stage106_noaa_attack_type_paired_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(out), "attack_types": len(ATTACK_TYPES)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
