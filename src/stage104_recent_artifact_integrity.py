from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
METRICS = ["test_f1", "test_recall", "test_precision", "test_auprc", "test_far"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_unique(frame: pd.DataFrame, keys: list[str], label: str, checks: list[dict[str, object]]) -> None:
    duplicates = int(frame.duplicated(keys).sum())
    checks.append({"check": f"{label}_unique_keys", "status": "PASS" if duplicates == 0 else "FAIL", "duplicates": duplicates, "keys": keys})


def check_finite(frame: pd.DataFrame, columns: list[str], label: str, checks: list[dict[str, object]]) -> None:
    missing = {column: int(frame[column].isna().sum()) for column in columns if column in frame.columns and frame[column].isna().any()}
    checks.append({"check": f"{label}_metric_completeness", "status": "PASS" if not missing else "FAIL", "missing": missing})


def main() -> int:
    checks: list[dict[str, object]] = []
    paths = {
        "source": TABLES / "stage82_noaa_extended_normalized.parquet",
        "stage98_detail": TABLES / "stage98_noaa_low_far_baseline_comparison_detail.csv",
        "stage99_json": TABLES / "stage99_noaa_baseline_paired_bootstrap.json",
        "stage100_station": TABLES / "stage100_noaa_low_far_station_counts.parquet",
        "stage101_detail": TABLES / "stage101_noaa_trust_svm_complementarity_detail.csv",
        "stage102_all": TABLES / "stage102_noaa_trust_svm_valid_selection_all_candidates.csv",
        "stage102_selected": TABLES / "stage102_noaa_trust_svm_valid_selection_selected.csv",
        "stage103_json": TABLES / "stage103_noaa_valid_selection_stability.json",
        "stage105_json": TABLES / "stage105_noaa_attack_type_valid_selected.json",
        "stage105_type_detail": TABLES / "stage105_noaa_attack_type_valid_selected_detail.csv",
        "stage105_overall_detail": TABLES / "stage105_noaa_attack_type_valid_selected_overall_detail.csv",
        "stage106_json": TABLES / "stage106_noaa_attack_type_paired_bootstrap.json",
        "stage107_json": TABLES / "stage107_noaa_natural_svm_validation.json",
        "stage107_detail": TABLES / "stage107_noaa_natural_svm_validation_detail.csv",
        "stage107_station_detail": TABLES / "stage107_noaa_natural_svm_validation_station_detail.csv",
        "stage108_json": TABLES / "stage108_noaa_natural_alarm_station_bootstrap.json",
        "stage109_json": TABLES / "stage109_noaa_quality_weak_label_bootstrap.json",
        "stage109_detail": TABLES / "stage109_noaa_quality_weak_label_bootstrap.csv",
        "stage110_json": TABLES / "stage110_noaa_quality_hierarchical_bootstrap.json",
        "stage110_detail": TABLES / "stage110_noaa_quality_hierarchical_bootstrap.csv",
        "stage111_json": TABLES / "stage111_noaa_quality_count_sensitivity.json",
        "stage111_detail": TABLES / "stage111_noaa_quality_count_sensitivity.csv",
    }
    for label, path in paths.items():
        checks.append({"check": f"{label}_exists", "status": "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL", "path": str(path), "bytes": path.stat().st_size if path.exists() else 0})
    if any(item["status"] == "FAIL" for item in checks):
        payload = {"status": "FAIL", "checks": checks}
        (TABLES / "stage104_recent_artifact_integrity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(json.dumps(payload, ensure_ascii=False))

    source = pd.read_parquet(paths["source"])
    stage98 = pd.read_csv(paths["stage98_detail"])
    stage100 = pd.read_parquet(paths["stage100_station"])
    stage101 = pd.read_csv(paths["stage101_detail"])
    stage102_all = pd.read_csv(paths["stage102_all"])
    stage102_selected = pd.read_csv(paths["stage102_selected"])
    stage105_type = pd.read_csv(paths["stage105_type_detail"])
    stage105_overall = pd.read_csv(paths["stage105_overall_detail"])
    stage99 = json.loads(paths["stage99_json"].read_text(encoding="utf-8"))
    stage103 = json.loads(paths["stage103_json"].read_text(encoding="utf-8"))
    stage105 = json.loads(paths["stage105_json"].read_text(encoding="utf-8"))
    stage106 = json.loads(paths["stage106_json"].read_text(encoding="utf-8"))
    stage107 = json.loads(paths["stage107_json"].read_text(encoding="utf-8"))
    stage108 = json.loads(paths["stage108_json"].read_text(encoding="utf-8"))
    stage107_detail = pd.read_csv(paths["stage107_detail"])
    stage107_station_detail = pd.read_csv(paths["stage107_station_detail"])
    stage109 = json.loads(paths["stage109_json"].read_text(encoding="utf-8"))
    stage109_detail = pd.read_csv(paths["stage109_detail"])
    stage110 = json.loads(paths["stage110_json"].read_text(encoding="utf-8"))
    stage110_detail = pd.read_csv(paths["stage110_detail"])
    stage111 = json.loads(paths["stage111_json"].read_text(encoding="utf-8"))
    stage111_detail = pd.read_csv(paths["stage111_detail"])

    checks.extend([
        {"check": "source_has_year_station_timestamp", "status": "PASS" if {"year", "station_id", "timestamp_utc"}.issubset(source.columns) else "FAIL", "rows": int(len(source))},
        {"check": "stage98_expected_rows", "status": "PASS" if len(stage98) == 500 else "FAIL", "rows": int(len(stage98)), "expected": 500},
        {"check": "stage99_expected_results", "status": "PASS" if len(stage99.get("results", [])) == 150 else "FAIL", "rows": len(stage99.get("results", [])), "expected": 150},
        {"check": "stage100_expected_station_rows", "status": "PASS" if len(stage100) == 1100 else "FAIL", "rows": int(len(stage100)), "expected": 1100},
        {"check": "stage101_expected_rows", "status": "PASS" if len(stage101) == 240 else "FAIL", "rows": int(len(stage101)), "expected": 240},
        {"check": "stage102_all_expected_rows", "status": "PASS" if len(stage102_all) == 240 else "FAIL", "rows": int(len(stage102_all)), "expected": 240},
        {"check": "stage102_selected_expected_rows", "status": "PASS" if len(stage102_selected) == 20 else "FAIL", "rows": int(len(stage102_selected)), "expected": 20},
        {"check": "stage103_expected_blocks", "status": "PASS" if stage103.get("blocks") == 20 else "FAIL", "blocks": stage103.get("blocks"), "expected": 20},
        {"check": "stage105_expected_type_rows", "status": "PASS" if len(stage105_type) == 360 else "FAIL", "rows": int(len(stage105_type)), "expected": 360},
        {"check": "stage105_expected_overall_rows", "status": "PASS" if len(stage105_overall) == 60 else "FAIL", "rows": int(len(stage105_overall)), "expected": 60},
        {"check": "stage105_expected_attack_types", "status": "PASS" if sorted(stage105_type["attack_type"].unique().tolist()) == ["drift", "gaussian_noise", "offset", "spike", "stuck", "time_shift"] else "FAIL", "values": sorted(stage105_type["attack_type"].unique().tolist())},
        {"check": "stage105_status", "status": "PASS" if stage105.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED" else "FAIL", "value": stage105.get("status")},
        {"check": "stage106_expected_results", "status": "PASS" if len(stage106.get("results", [])) == 12 else "FAIL", "rows": len(stage106.get("results", [])), "expected": 12},
        {"check": "stage106_status", "status": "PASS" if stage106.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED" else "FAIL", "value": stage106.get("status")},
        {"check": "stage107_expected_detail_rows", "status": "PASS" if len(stage107_detail) == 45 else "FAIL", "rows": int(len(stage107_detail)), "expected": 45},
        {"check": "stage107_expected_station_rows", "status": "PASS" if len(stage107_station_detail) == 495 else "FAIL", "rows": int(len(stage107_station_detail)), "expected": 495},
        {"check": "stage107_status", "status": "PASS" if stage107.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED" else "FAIL", "value": stage107.get("status")},
        {"check": "stage108_expected_results", "status": "PASS" if len(stage108.get("results", [])) == 30 else "FAIL", "rows": len(stage108.get("results", [])), "expected": 30},
        {"check": "stage108_status", "status": "PASS" if stage108.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED" else "FAIL", "value": stage108.get("status")},
        {"check": "stage109_expected_results", "status": "PASS" if len(stage109.get("results", [])) == 45 and len(stage109_detail) == 45 else "FAIL", "json_rows": len(stage109.get("results", [])), "csv_rows": int(len(stage109_detail)), "expected": 45},
        {"check": "stage109_status", "status": "PASS" if stage109.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED" else "FAIL", "value": stage109.get("status")},
        {"check": "stage110_expected_results", "status": "PASS" if len(stage110.get("results", [])) == 6 and len(stage110_detail) == 6 else "FAIL", "json_rows": len(stage110.get("results", [])), "csv_rows": int(len(stage110_detail)), "expected": 6},
        {"check": "stage110_status", "status": "PASS" if stage110.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED" else "FAIL", "value": stage110.get("status")},
        {"check": "stage111_expected_results", "status": "PASS" if len(stage111.get("results", [])) == 45 and len(stage111_detail) == 45 else "FAIL", "json_rows": len(stage111.get("results", [])), "csv_rows": int(len(stage111_detail)), "expected": 45},
        {"check": "stage111_status", "status": "PASS" if stage111.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED" else "FAIL", "value": stage111.get("status")},
        {"check": "stage98_status", "status": "PASS" if stage99.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED" else "FAIL", "value": stage99.get("status")},
        {"check": "stage103_status", "status": "PASS" if stage103.get("status") == "PASS_PIPELINE_REVIEW_REQUIRED" else "FAIL", "value": stage103.get("status")},
    ])
    check_unique(stage98, ["fold", "seed", "method", "q"], "stage98", checks)
    check_unique(stage100, ["fold", "seed", "station_id", "method", "q"], "stage100", checks)
    check_unique(stage101, ["fold", "seed", "method"], "stage101", checks)
    check_unique(stage102_all, ["fold", "seed", "method"], "stage102_all", checks)
    check_unique(stage102_selected, ["fold", "seed"], "stage102_selected", checks)
    check_unique(stage105_type, ["fold", "seed", "method", "attack_type"], "stage105_type", checks)
    check_unique(stage105_overall, ["fold", "seed", "method"], "stage105_overall", checks)
    check_finite(stage98, METRICS, "stage98", checks)
    check_finite(stage101, METRICS, "stage101", checks)
    check_finite(stage102_all, [f"valid_{metric[5:]}" for metric in METRICS] + METRICS, "stage102_all", checks)
    check_finite(stage105_overall, METRICS, "stage105_overall", checks)
    check_unique(stage107_detail, ["target_year", "method", "q"], "stage107_detail", checks)
    check_unique(stage107_station_detail, ["target_year", "method", "q", "station_key"], "stage107_station_detail", checks)
    check_unique(stage109_detail, ["target_year", "method", "q"], "stage109_detail", checks)
    check_finite(stage109_detail, ["contrast_mean", "ci95_low", "ci95_high", "prob_contrast_positive"], "stage109_detail", checks)
    check_unique(stage110_detail, ["q", "candidate", "reference"], "stage110_detail", checks)
    check_finite(stage110_detail, ["observed_diff_in_contrast", "ci95_low", "ci95_high", "prob_candidate_more_quality_aligned"], "stage110_detail", checks)
    check_unique(stage111_detail, ["q", "method", "min_flagged_rows"], "stage111_detail", checks)
    check_finite(stage111_detail, ["station_equal_contrast", "station_equal_ci95_low", "station_equal_ci95_high", "row_weighted_contrast"], "stage111_detail", checks)
    checks.append({"check": "stage101_stage102_candidate_keys_match", "status": "PASS" if set(zip(stage101["fold"], stage101["seed"], stage101["method"])) == set(zip(stage102_all["fold"], stage102_all["seed"], stage102_all["method"])) else "FAIL"})
    checks.append({"check": "stage98_q_grid_complete", "status": "PASS" if sorted(stage98["q"].unique().tolist()) == [0.9, 0.95, 0.975, 0.99, 0.995] else "FAIL", "values": sorted(stage98["q"].unique().tolist())})
    checks.append({"check": "stage100_single_q", "status": "PASS" if stage100["q"].nunique() == 1 and float(stage100["q"].iloc[0]) == 0.975 else "FAIL", "values": sorted(stage100["q"].unique().tolist())})

    failed = [item for item in checks if item["status"] == "FAIL"]
    payload = {
        "status": "PASS_PIPELINE_REVIEW_REQUIRED" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "source_sha256": sha256(paths["source"]),
        "artifacts": {label: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for label, path in paths.items()},
        "limitations": ["完整性审计不能证明性能优越性；性能结论必须回到阶段98–111原始结果。", "阶段109–111的NOAA质量标记仅为弱标签，不是攻击真值。"],
    }
    (TABLES / "stage104_recent_artifact_integrity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 阶段104 最近实验产物完整性审计", "", f"- 状态：**{payload['status']}**", f"- 源表SHA256：`{payload['source_sha256']}`", "", "| 检查 | 状态 |", "|---|---|"]
    lines.extend(f"| {item['check']} | {item['status']} |" for item in checks)
    lines += ["", "该审计只验证阶段98–111文件的存在性、行数、唯一键、指标完整性和候选集合一致性，不替代性能和统计结论；阶段109–111质量标记仅为弱标签。"]
    (REPORTS / "stage104_recent_artifact_integrity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(checks), "failures": len(failed)}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
