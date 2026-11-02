from __future__ import annotations

"""Shared V9 compatibility checks.

The frozen V9 protocol uses descriptive scenario names that were not present
in the older V2/V6 wrappers.  This module is the only approved mapping for
the V9 runner.  It also refuses to operate on a pending or historical data
manifest, so an accidental call cannot turn an old test window into a new
confirmatory result.
"""

import json
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V9_PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
V9_RAW_DIR = V9_PACKAGE / "raw_official_2026_10"
V9_NORMALIZED_DIR = V9_PACKAGE / "normalized_official_2026_10"
START = pd.Timestamp("2026-10-01T00:00:00Z")
END = pd.Timestamp("2026-10-31T23:00:00Z")
EXPECTED_GRID = pd.date_range(START, END, freq="h", tz="UTC")
EXPECTED_STATIONS = {
    "MeteoSwiss": {
        "abo", "aeg", "aig", "alt", "and", "arh",
        "att", "bas", "beh", "bez", "bin", "cdf",
    },
    "NOAA ISD": {
        "52533199999", "52983099999", "52996099999",
        "53705099999", "53915099999", "53923099999",
        "56096099999", "57025099999", "57036099999",
        "57127099999", "57131099999",
    },
    "KNMI": {
        "209", "215", "229", "251", "258", "277",
        "278", "285", "286", "290", "308", "315",
    },
    "ECCC": {
        "52", "1556", "1635", "1723", "1839", "2855",
        "3518", "3920", "5237", "6205", "6354", "6545",
        "6561", "27533", "30266", "52118",
    },
}

V9_SCENARIO_TYPE = {
    "weak_spike_short": "spike",
    "strong_spike_long": "spike",
    "weak_offset_short": "offset",
    "strong_offset_long": "offset",
    "short_drift": "drift",
    "long_drift": "drift",
    "short_lag": "time_shift",
    "long_delay": "time_shift",
    "short_stuck": "stuck",
    "long_stuck": "stuck",
    "weak_noise_short": "gaussian_noise",
    "strong_noise_long": "gaussian_noise",
    "coordinated_drift_unseen": "drift",
}

V9_SCENARIOS = set(V9_SCENARIO_TYPE)


def scenario_type(name: str) -> str:
    try:
        return V9_SCENARIO_TYPE[str(name)]
    except KeyError as exc:
        raise KeyError(f"unsupported V9 scenario: {name!r}") from exc


def validate_protocol(protocol: dict) -> None:
    names = {str(item.get("name")) for item in protocol.get("attack_scenarios", [])}
    if names != V9_SCENARIOS or len(names) != 13:
        raise RuntimeError("V9 scenario set changed or is incomplete")
    for item in protocol["attack_scenarios"]:
        name = str(item["name"])
        if str(item.get("base_type")) != V9_SCENARIO_TYPE[name]:
            raise RuntimeError(f"{name}: base_type mapping mismatch")
        if int(item["length"]) <= 0:
            raise RuntimeError(f"{name}: invalid length")
    if protocol["attack_scenarios"][6]["name"] != "short_lag":
        # Order is not used for selection, but the protocol must be explicit.
        raise RuntimeError("unexpected V9 scenario order")
    by_name = {str(item["name"]): item for item in protocol["attack_scenarios"]}
    if by_name["short_lag"].get("lag_rows") != 1:
        raise RuntimeError("short_lag must use lag_rows=1")
    if by_name["long_delay"].get("lag_rows") != 8:
        raise RuntimeError("long_delay must use lag_rows=8")
    coordinated = by_name["coordinated_drift_unseen"]
    if not (
        coordinated.get("coordinated") is True
        and coordinated.get("unseen_combination") is True
        and int(coordinated.get("station_count", 0)) == 2
    ):
        raise RuntimeError("coordinated_drift_unseen contract is incomplete")


def require_ready_data_manifest(path: Path) -> dict:
    """Require a dedicated new-window manifest before any evaluation."""
    if not path.exists():
        raise RuntimeError(f"V9 data manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest_claim_metadata(manifest)
    if manifest.get("status") != "OFFICIAL_WINDOW_READY":
        raise RuntimeError(
            "V9 official window is not ready; old normalized files must not be substituted"
        )
    if manifest.get("window", {}).get("test_start_utc") != "2026-10-01T00:00:00Z":
        raise RuntimeError("V9 manifest test start does not match frozen window")
    if manifest.get("window", {}).get("test_end_utc") != "2026-10-31T23:00:00Z":
        raise RuntimeError("V9 manifest test end does not match frozen window")
    providers = manifest.get("providers", {})
    if set(providers) != {"MeteoSwiss", "NOAA ISD", "KNMI", "ECCC"}:
        raise RuntimeError("V9 manifest provider scope changed")
    for provider, meta in providers.items():
        expected = EXPECTED_STATIONS[provider]
        if str(meta.get("status")) != "PASS":
            raise RuntimeError(f"{provider}: provider manifest row is not PASS")
        actual = {str(item) for item in meta.get("actual_station_ids", [])}
        if actual != expected:
            raise RuntimeError(
                f"{provider}: manifest station set does not match frozen station set"
            )
        expected_rows = len(expected) * len(EXPECTED_GRID)
        if int(meta.get("rows", -1)) != expected_rows:
            raise RuntimeError(
                f"{provider}: normalized test row count does not match the "
                "complete frozen hourly grid"
            )
        timestamp_min = str(EXPECTED_GRID.min())
        timestamp_max = str(EXPECTED_GRID.max())
        if str(meta.get("timestamp_min")) != timestamp_min:
            raise RuntimeError(f"{provider}: test timestamp_min is outside the frozen window")
        if str(meta.get("timestamp_max")) != timestamp_max:
            raise RuntimeError(f"{provider}: test timestamp_max is outside the frozen window")
        raw_path = _resolve_manifest_path(meta["raw_path"])
        normalized_path = _resolve_manifest_path(meta["normalized_path"])
        test_normalized_path = _resolve_manifest_path(meta["test_normalized_path"])
        _require_under(raw_path, V9_RAW_DIR, f"{provider}: raw_path")
        _require_under(normalized_path, V9_NORMALIZED_DIR, f"{provider}: normalized_path")
        _require_under(
            test_normalized_path,
            V9_NORMALIZED_DIR,
            f"{provider}: test_normalized_path",
        )
        for key, candidate in (
            ("raw_path", raw_path),
            ("normalized_path", normalized_path),
            ("test_normalized_path", test_normalized_path),
        ):
            if not candidate.exists():
                raise RuntimeError(
                    f"{provider}: manifest path missing for {key}: {candidate}"
                )
        if normalized_path != test_normalized_path:
            raise RuntimeError(
                f"{provider}: normalized_path and test_normalized_path differ"
            )
        if _sha256(raw_path).lower() != str(meta["raw_sha256"]).lower():
            raise RuntimeError(f"{provider}: raw file SHA256 does not match manifest")
        raw_members = meta.get("raw_sha256s")
        if raw_members is not None:
            if not isinstance(raw_members, list) or not raw_members:
                raise RuntimeError(f"{provider}: raw_sha256s must be a non-empty list")
            listed_paths = []
            for item in raw_members:
                if not isinstance(item, dict):
                    raise RuntimeError(f"{provider}: invalid raw_sha256s entry")
                member_path = _resolve_manifest_path(item.get("path", ""))
                _require_under(member_path, V9_RAW_DIR, f"{provider}: raw member")
                if not member_path.exists():
                    raise RuntimeError(
                        f"{provider}: raw member path missing: {member_path}"
                    )
                if _sha256(member_path).lower() != str(
                    item.get("sha256", "")
                ).lower():
                    raise RuntimeError(
                        f"{provider}: raw member SHA256 does not match manifest"
                    )
                listed_paths.append(str(member_path.resolve()))
            declared_paths = [
                str(_resolve_manifest_path(value).resolve())
                for value in meta.get("raw_paths", [])
            ]
            if declared_paths and sorted(declared_paths) != sorted(listed_paths):
                raise RuntimeError(
                    f"{provider}: raw_paths and raw_sha256s are inconsistent"
                )
            if str(raw_path.resolve()) not in listed_paths:
                raise RuntimeError(
                    f"{provider}: raw_path is absent from raw_sha256s"
                )
        if _sha256(normalized_path).lower() != str(
            meta["normalized_sha256"]
        ).lower():
            raise RuntimeError(
                f"{provider}: normalized test file SHA256 does not match manifest"
            )
        if not meta.get("raw_sha256") or not meta.get("normalized_sha256"):
            raise RuntimeError(f"{provider}: raw/normalized SHA256 is missing")
    return manifest


def validate_manifest_claim_metadata(manifest: dict) -> None:
    """Validate the non-negotiable claim boundary before file access."""
    if manifest.get("status") != "OFFICIAL_WINDOW_READY":
        raise RuntimeError(
            "V9 official window is not ready; old normalized files must not be substituted"
        )
    if manifest.get("failures") != []:
        raise RuntimeError("V9 ready manifest contains provider failures")
    if manifest.get("observation_values_read") is not True:
        raise RuntimeError("V9 ready manifest must declare observation_values_read=true")
    if manifest.get("historical_substitution") is not False:
        raise RuntimeError("V9 manifest cannot use historical substitution")


def _resolve_manifest_path(value: object) -> Path:
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else ROOT / candidate


def _require_under(candidate: Path, root: Path, label: str) -> None:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"{label} escapes the dedicated V9 directory: {candidate}"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v9_provider_frame(
    provider: str, manifest: dict, frozen_protocol: dict
) -> pd.DataFrame:
    """Load frozen train/valid rows plus the new official-window test rows.

    The historical source is read with a split predicate so its previous test
    rows are excluded at the parquet reader boundary.  The new file must
    contain only the frozen October 2026 window and is assigned ``split=test``.
    """
    meta = manifest["providers"][provider]
    historical = Path(str(meta["historical_normalized_path"]))
    test_path = Path(str(meta["test_normalized_path"]))
    if not historical.is_absolute():
        historical = ROOT / historical
    if not test_path.is_absolute():
        test_path = ROOT / test_path
    if not historical.exists() or not test_path.exists():
        raise RuntimeError(f"{provider}: V9 data files are missing")
    provider_protocol_path = Path(
        str(frozen_protocol["providers"][provider]["protocol"])
    )
    if not provider_protocol_path.is_absolute():
        provider_protocol_path = ROOT / provider_protocol_path
    provider_protocol = json.loads(
        provider_protocol_path.read_text(encoding="utf-8")
    )
    if provider == "NOAA ISD":
        train_years = [
            int(year) for year in provider_protocol["split"]["train"]
        ]
        valid_years = [
            int(year) for year in provider_protocol["split"]["valid"]
        ]
        train_valid = pd.read_parquet(
            historical,
            filters=[("year", "in", train_years + valid_years)],
        )
        split_by_year = {
            **{year: "train" for year in train_years},
            **{year: "valid" for year in valid_years},
        }
        train_valid["split"] = (
            train_valid["year"].astype(int).map(split_by_year)
        )
        train_valid["observation_present"] = pd.to_numeric(
            train_valid["temperature_c"], errors="coerce"
        ).notna()
    else:
        train_valid = pd.read_parquet(
            historical,
            filters=[("split", "in", ["train", "valid"])],
        )
    test = pd.read_parquet(test_path)
    if "split" in test.columns and set(test["split"].astype(str)) - {"test"}:
        raise RuntimeError(f"{provider}: new test file contains non-test split rows")
    test = test.copy()
    test["split"] = "test"
    expected_stations = EXPECTED_STATIONS[provider]
    actual_stations = set(test["station_id"].astype(str))
    if actual_stations != expected_stations:
        raise RuntimeError(f"{provider}: test parquet station set mismatch")
    expected_rows = len(expected_stations) * len(EXPECTED_GRID)
    if len(test) != expected_rows:
        raise RuntimeError(
            f"{provider}: test parquet row count {len(test)} != {expected_rows}"
        )
    test_timestamps = pd.to_datetime(
        test["timestamp_utc"], utc=True, errors="coerce"
    )
    if test_timestamps.isna().any() or not test_timestamps.isin(EXPECTED_GRID).all():
        raise RuntimeError(f"{provider}: test parquet timestamps are outside the grid")
    if test.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError(f"{provider}: test parquet has duplicate station/timestamp")
    combined = pd.concat([train_valid, test], ignore_index=True, sort=False)
    combined["station_id"] = combined["station_id"].astype(str)
    combined["timestamp_utc"] = pd.to_datetime(
        combined["timestamp_utc"], utc=True, errors="coerce"
    )
    if combined["timestamp_utc"].isna().any():
        raise RuntimeError(f"{provider}: invalid timestamp_utc after merge")
    if combined.duplicated(["station_id", "timestamp_utc"]).any():
        raise RuntimeError(f"{provider}: duplicate station/timestamp after merge")
    if set(combined["split"].astype(str)) != {"train", "valid", "test"}:
        raise RuntimeError(f"{provider}: merged split labels are incomplete")
    return combined.sort_values(
        ["station_id", "timestamp_utc"], kind="stable"
    ).reset_index(drop=True)
