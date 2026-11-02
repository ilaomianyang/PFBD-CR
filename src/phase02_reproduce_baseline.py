from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "config" / "protocol_v0.1.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path):
    import pandas as pd
    return pd.read_csv(path)


def row_matches(row: Any, filters: dict[str, Any]) -> bool:
    return all(str(row.get(k)) == str(v) for k, v in filters.items())


def check_metric_table(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"file": str(path), "exists": path.exists(), "passed": False}
    if not path.exists():
        result["error"] = "missing_file"
        return result
    try:
        df = load_csv(path)
    except Exception as exc:
        result["error"] = f"read_error: {exc}"
        return result
    result["columns"] = list(df.columns)
    filters = spec.get("row_filters", {})
    matches = df[[row_matches(row, filters) for _, row in df.iterrows()]]
    result["matching_rows"] = int(len(matches))
    if len(matches) != 1:
        result["error"] = "expected_exactly_one_matching_row"
        return result
    row = matches.iloc[0]
    tolerance = float(spec.get("tolerance", 1e-9))
    comparisons = {}
    passed = True
    for metric, expected in spec.get("metrics", {}).items():
        if metric not in df.columns:
            comparisons[metric] = {"expected": expected, "actual": None, "passed": False, "error": "missing_column"}
            passed = False
            continue
        actual = float(row[metric])
        ok = abs(actual - float(expected)) <= tolerance
        comparisons[metric] = {"expected": float(expected), "actual": actual, "absolute_error": abs(actual - float(expected)), "passed": ok}
        passed = passed and ok
    result["comparisons"] = comparisons
    result["passed"] = passed
    return result


def inspect_feature_table(path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"file": str(path), "exists": path.exists(), "passed": False}
    if not path.exists():
        result["error"] = "missing_file"
        return result
    try:
        import pandas as pd
        df = pd.read_parquet(path)
    except Exception as exc:
        result["error"] = f"read_error: {exc}"
        return result
    required = [policy["required_column"], *policy["duplicate_keys"]]
    missing = [c for c in required if c not in df.columns]
    result.update({"rows": int(len(df)), "columns": int(len(df.columns)), "missing_required_columns": missing})
    if missing:
        result["error"] = "missing_required_columns"
        return result
    split_values = sorted(df[policy["required_column"]].dropna().astype(str).unique().tolist())
    result["split_values"] = split_values
    result["split_counts"] = {str(k): int(v) for k, v in df.groupby(policy["required_column"], dropna=False).size().items()}
    result["split_values_valid"] = set(split_values).issubset(set(policy["allowed_values"])) and set(policy["allowed_values"]).issubset(set(split_values))
    key_cols = [policy["required_column"], *policy["duplicate_keys"]]
    duplicate_count = int(df.duplicated(key_cols).sum())
    result["duplicate_key_count"] = duplicate_count
    result["no_duplicate_keys"] = duplicate_count == 0
    result["passed"] = not missing and result["split_values_valid"] and result["no_duplicate_keys"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="复现并审计 TrustMeteo-QC 阶段2的 PC-MSTF-AP-v4 基线")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data = protocol["data"]
    baseline_dir = Path(data["baseline_tables_dir"])
    output_json = ROOT / "tables" / "phase02_baseline_reproduction.json"
    output_report = ROOT / "reports" / "phase02_baseline_reproduction.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    files_to_hash = {
        "protocol": protocol_path,
        "baseline_config": Path(data["baseline_config"]),
        "baseline_report": Path(data["baseline_report"]),
        "feature_table": Path(data["feature_table"]),
        "anomaly_metrics": baseline_dir / "v4_anomaly_detection_metrics.csv",
        "reconstruction_metrics": baseline_dir / "v4_anomaly_reconstruction_metrics.csv",
        "trust_calibration": baseline_dir / "v4_trust_calibration.csv",
        "leakage_check": baseline_dir / "v4_leakage_check.csv",
    }
    hashes = {}
    for name, path in files_to_hash.items():
        hashes[name] = {"path": str(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None}

    feature = inspect_feature_table(Path(data["feature_table"]), protocol["split_policy"])
    expected = protocol["phase2_expected_baseline"]
    detector = check_metric_table(baseline_dir / expected["detector"]["file"], expected["detector"])
    reconstruction = check_metric_table(baseline_dir / expected["reconstruction"]["file"], expected["reconstruction"])
    trust = check_metric_table(baseline_dir / expected["trust"]["file"], expected["trust"])
    leakage_path = baseline_dir / "v4_leakage_check.csv"
    leakage_exists = leakage_path.exists()
    leakage_passed = False
    leakage_rows = []
    if leakage_exists:
        try:
            df = load_csv(leakage_path)
            leakage_rows = df.to_dict(orient="records")
            leakage_passed = bool(len(df) > 0 and df["passed"].astype(bool).all())
        except Exception:
            leakage_passed = False

    checks = {
        "required_files_exist": all(x["exists"] for x in hashes.values()),
        "feature_table_structure": feature["passed"],
        "baseline_detector_metrics_match": detector["passed"],
        "baseline_reconstruction_metrics_match": reconstruction["passed"],
        "baseline_trust_metrics_match": trust["passed"],
        "leakage_report_present_and_passed": leakage_exists and leakage_passed,
    }
    overall = all(checks.values())
    result = {
        "protocol_id": protocol["protocol_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if overall else "FAIL",
        "checks": checks,
        "hashes": hashes,
        "feature_table": feature,
        "detector": detector,
        "reconstruction": reconstruction,
        "trust": trust,
        "leakage": {"exists": leakage_exists, "passed": leakage_passed, "rows": leakage_rows},
        "scope": "仅复现既有基线；没有训练模型，没有修改测试集，没有进行阈值调参",
    }
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段 0 到阶段 2 基线复现报告",
        "",
        f"- 协议：`{protocol['protocol_id']}`",
        f"- 生成时间（UTC）：`{result['generated_at_utc']}`",
        f"- 总状态：**{result['status']}**",
        "",
        "## 核对结果",
    ]
    for key, value in checks.items():
        lines.append(f"- {'通过' if value else '未通过'}：`{key}`")
    lines += [
        "",
        "## 范围边界",
        "本报告只核对已有 `pc_mstf_ap_v4` 输出与冻结协议是否一致。受控攻击标签不代表真实网络入侵事件；阶段 2 也没有训练新模型或修改测试集。",
        "",
        "## 后续门槛",
        "只有总状态为 `PASS`，才进入检测器实现和新的攻击矩阵实验。若为 `FAIL`，先修复路径、数据结构、指标或泄漏审计问题。",
    ]
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output_json": str(output_json), "output_report": str(output_report)}, ensure_ascii=False))
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
