from __future__ import annotations

"""Dedicated V9 locked-evaluation entrypoint.

It is intentionally unusable until a separate official-window data manifest
is produced by the V9 download/audit stage.  This prevents the legacy V6
normalized files from being silently evaluated as the October 2026 test.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage340_v4_route_locked_evaluation as base  # noqa: E402
from stage425_v9_compatibility import (  # noqa: E402
    load_v9_provider_frame,
    require_ready_data_manifest,
    scenario_type,
    validate_protocol,
)

V9_PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
V9_PROTOCOL = V9_PACKAGE / "protocol_freeze.json"
V9_MANIFEST = V9_PACKAGE / "data_manifest.json"
V9_DETAIL = ROOT / "tables" / "stage425_v9_locked_evaluation_detail.csv"
V9_SCENARIO = ROOT / "tables" / "stage425_v9_locked_evaluation_scenario.csv"
V9_STATION = ROOT / "tables" / "stage425_v9_locked_evaluation_station.csv"
V9_JSON = ROOT / "tables" / "stage425_v9_locked_evaluation.json"
V9_REPORT = ROOT / "reports" / "stage425_v9_locked_evaluation.md"


def _normalize_v9_outputs(protocol: dict) -> None:
    """Rename legacy route output and attach frozen factor metadata."""
    method_renames = {
        "route_knn_arima": "trust_arima",
        # The frozen V9 contract names the 20k OCSVM budget explicitly.
        "one_class_svm": "one_class_svm_20k",
    }
    if V9_DETAIL.exists():
        detail = pd.read_csv(V9_DETAIL)
        detail["method"] = detail["method"].replace(method_renames)
        detail.to_csv(V9_DETAIL, index=False)
    if V9_STATION.exists():
        station = pd.read_csv(V9_STATION)
        station["method"] = station["method"].replace(method_renames)
        station.to_csv(V9_STATION, index=False)
    if V9_SCENARIO.exists():
        scenario = pd.read_csv(V9_SCENARIO)
        scenario["method"] = scenario["method"].replace(method_renames)
        specs = {
            str(item["name"]): item
            for item in protocol.get("attack_scenarios", [])
        }
        for field in (
            "base_type",
            "factor_intensity",
            "factor_duration",
            "factor_length",
            "unseen_combination",
            "coordinated",
        ):
            scenario[field] = scenario["scenario"].map(
                lambda name: specs.get(str(name), {}).get(field)
            )
        scenario.to_csv(V9_SCENARIO, index=False)
    if V9_JSON.exists():
        payload = json.loads(V9_JSON.read_text(encoding="utf-8"))
        for row in payload.get("summary", []):
            if row.get("method") in method_renames:
                row["method"] = method_renames[row["method"]]
        for row in payload.get("comparisons", []):
            for key in ("candidate", "baseline", "method"):
                if row.get(key) in method_renames:
                    row[key] = method_renames[row[key]]
        counts = payload.get("selection_counts", {})
        for old_name, new_name in method_renames.items():
            if old_name in counts:
                counts[new_name] = counts.pop(old_name)
        payload["method_namespace"] = [
            "trust_only",
            "trust_arima",
            "trust_shape_union",
            *protocol["method_freeze"]["baselines"],
        ]
        V9_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if V9_REPORT.exists():
        text = V9_REPORT.read_text(encoding="utf-8")
        text = text.replace("route_knn_arima", "trust_arima")
        text = text.replace("one_class_svm", "one_class_svm_20k")
        V9_REPORT.write_text(text, encoding="utf-8")


def main() -> int:
    protocol = json.loads(V9_PROTOCOL.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    manifest = require_ready_data_manifest(V9_MANIFEST)
    # The base evaluator is reused only after the dedicated data gate passes.
    # Its output namespace is isolated from every V6/V7/V8 artifact.
    base.PROTOCOL_PATH = V9_PROTOCOL
    base.OUT_JSON = ROOT / "tables" / "stage425_v9_locked_evaluation.json"
    base.OUT_DETAIL = ROOT / "tables" / "stage425_v9_locked_evaluation_detail.csv"
    base.OUT_SCENARIO = ROOT / "tables" / "stage425_v9_locked_evaluation_scenario.csv"
    base.OUT_STATION = ROOT / "tables" / "stage425_v9_locked_evaluation_station.csv"
    base.OUT_REPORT = ROOT / "reports" / "stage425_v9_locked_evaluation.md"
    base.VALID_OFFSET = int(protocol["valid_attack_seed_offset"])
    base.TEST_OFFSET = int(protocol["test_attack_seed_offset"])
    base.PRIMARY_FAR = float(protocol["primary_far"])
    base.simbase._scenario_base_type = scenario_type
    base.make_frame = lambda source, _protocol: load_v9_provider_frame(
        source, manifest, protocol
    )
    base.PROTOCOLS = {
        provider: ROOT / str(meta["protocol"])
        for provider, meta in protocol["providers"].items()
    }
    exit_code = base.main()
    if exit_code != 0:
        return exit_code
    _normalize_v9_outputs(protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
