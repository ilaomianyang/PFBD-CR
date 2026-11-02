from __future__ import annotations

"""Strict official-window data-gate template for V9.

This stage does not substitute historical files.  It accepts provider-specific
raw files only when they are supplied by the official source, filters the
frozen UTC window with both boundaries, and writes a manifest consumed by the
V9 evaluator.  Provider adapters must be filled with the official parser
already used for that provider; an absent adapter is a hard failure.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from stage425_v9_compatibility import EXPECTED_GRID, EXPECTED_STATIONS
from v9_official_adapters import PROVIDER_CONTRACTS, parse_provider


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
RAW = PACKAGE / "raw_official_2026_10"
NORMALIZED = PACKAGE / "normalized_official_2026_10"
PROTOCOL = PACKAGE / "protocol_freeze.json"
MANIFEST = PACKAGE / "data_manifest.json"
FETCH_RECORD = ROOT / "tables" / "stage442_v9_official_window_fetch.json"
START = pd.Timestamp("2026-10-01T00:00:00Z")
END = pd.Timestamp("2026-10-31T23:00:00Z")
PROVIDERS = ("MeteoSwiss", "NOAA ISD", "KNMI", "ECCC")
SOURCE_PLAN = {
    "MeteoSwiss": "stage427 MeteoSwiss official STAC hourly asset",
    "NOAA ISD": "stage427 NOAA/NCEI Global Hourly 2026 station CSV",
    "KNMI": "stage427 KNMI official hourly archive",
    "ECCC": "stage427 ECCC GeoMet climate-hourly API",
}


def raw_inputs(provider: str) -> list[Path]:
    """Find only the frozen provider's supplied official raw files.

    The downloader may supply one consolidated CSV or one file per frozen
    station.  No historical normalized file is considered a valid input.
    """
    slug = provider.replace(" ", "_").lower()
    candidates: list[Path] = []
    consolidated = RAW / f"{slug}_2026_10.csv"
    if consolidated.is_file():
        candidates.append(consolidated)
    provider_dir = RAW / slug
    if provider_dir.is_dir():
        candidates.extend(path for path in provider_dir.rglob("*") if path.is_file())
    candidates.extend(
        path
        for path in RAW.iterdir()
        if path.is_file() and path.name.lower().startswith(slug)
    )
    # Preserve a deterministic order and avoid duplicate paths.
    unique = {path.resolve(): path for path in candidates}
    return sorted(unique.values(), key=lambda path: str(path).lower())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_fetch_record() -> tuple[dict, dict[Path, dict[str, object]]]:
    """Load the formal Stage442 record and bind every raw file to it."""
    if not FETCH_RECORD.exists():
        return {}, {}
    record = json.loads(FETCH_RECORD.read_text(encoding="utf-8"))
    rows = record.get("rows", [])
    by_path: dict[Path, dict[str, object]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict) or not row.get("path"):
                continue
            by_path[resolved(str(row["path"])).resolve()] = row
    return record, by_path


def normalize_official_frame(
    provider: str, frame: pd.DataFrame
) -> tuple[pd.DataFrame, int, set[str]]:
    """Apply the frozen V9 window, station and complete-grid contract."""
    timestamps = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="coerce"
    )
    if timestamps.isna().any():
        raise ValueError("invalid timestamp_utc")
    keep = timestamps.between(START, END, inclusive="both")
    selected = frame.loc[keep].copy()
    if selected.empty:
        raise ValueError("no rows in frozen October 2026 window")
    if "station_id" not in selected.columns:
        raise ValueError("raw adapter must provide station_id")
    if "temperature_c" not in selected.columns:
        raise ValueError("raw adapter must provide temperature_c")
    selected["timestamp_utc"] = timestamps.loc[keep].to_numpy()
    if not selected["timestamp_utc"].isin(EXPECTED_GRID).all():
        raise ValueError("official timestamps are not aligned to the hourly grid")
    selected["station_id"] = selected["station_id"].astype(str)
    expected_stations = EXPECTED_STATIONS[provider]
    actual_stations = set(selected["station_id"].dropna().astype(str))
    if actual_stations != expected_stations:
        raise ValueError(
            f"station set mismatch: expected {sorted(expected_stations)}, "
            f"got {sorted(actual_stations)}"
        )
    if selected.duplicated(["station_id", "timestamp_utc"]).any():
        raise ValueError("duplicate station/timestamp rows in official window")

    # Missing official observations remain missing; they are never imputed.
    normalized_parts = []
    raw_rows = int(len(selected))
    for station_id in sorted(expected_stations):
        part = selected.loc[selected["station_id"].eq(station_id)].set_index(
            "timestamp_utc"
        )
        part = part.reindex(EXPECTED_GRID)
        part["station_id"] = station_id
        part.index.name = "timestamp_utc"
        normalized_parts.append(part.reset_index())
    selected = pd.concat(normalized_parts, ignore_index=True, sort=False)
    selected["observation_present"] = selected["temperature_c"].notna()
    selected["split"] = "test"
    return selected, raw_rows, actual_stations


def validate_fetch_binding(
    provider: str,
    inputs: list[Path],
    fetch_status: object,
    fetch_rows_by_path: dict[Path, dict[str, object]],
) -> None:
    """Reject raw files not produced by the completed Stage442 fetch."""
    if fetch_status != "PASS_V9_OFFICIAL_FETCH_COMPLETED":
        raise ValueError(
            "Stage442 official fetch record is not completed: "
            f"{fetch_status or 'MISSING'}"
        )
    expected_paths = {
        path
        for path, row in fetch_rows_by_path.items()
        if str(row.get("provider")) == provider
    }
    input_paths = {path.resolve() for path in inputs}
    if input_paths != expected_paths:
        raise ValueError(
            f"{provider}: raw files do not exactly match Stage442 "
            "official fetch record"
        )
    if not inputs:
        raise ValueError(f"{provider}: missing official raw file(s) under {RAW}")
    for path in inputs:
        record_row = fetch_rows_by_path.get(path.resolve())
        if record_row is None:
            raise ValueError("raw file is not bound to Stage442 record")
        if int(record_row.get("bytes", -1)) != path.stat().st_size:
            raise ValueError("raw file byte count differs from Stage442 record")
        if str(record_row.get("sha256", "")).lower() != sha256(path).lower():
            raise ValueError("raw file SHA256 differs from Stage442 record")


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "CONFIRMATORY_PROTOCOL_FROZEN_PENDING_NEW_OFFICIAL_WINDOW":
        raise RuntimeError("V9 protocol is not the frozen pending protocol")
    RAW.mkdir(parents=True, exist_ok=True)
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    provider_rows: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    fetch_record, fetch_rows_by_path = load_fetch_record()
    fetch_status = fetch_record.get("status")
    fetch_ready = fetch_status == "PASS_V9_OFFICIAL_FETCH_COMPLETED"
    if not fetch_ready:
        failures.append(
            "Stage442 official fetch record is not completed: "
            f"{fetch_status or 'MISSING'}"
        )
    if fetch_ready:
        for provider in PROVIDERS:
            inputs = raw_inputs(provider)
            normalized = NORMALIZED / f"{provider.replace(' ', '_').lower()}_2026_10.parquet"
            try:
                validate_fetch_binding(
                    provider, inputs, fetch_status, fetch_rows_by_path
                )
                frame, adapter_meta = parse_provider(provider, inputs)
                selected, raw_rows, actual_stations = normalize_official_frame(
                    provider, frame
                )
                expected_stations = EXPECTED_STATIONS[provider]
                selected.to_parquet(normalized, index=False)
                raw_hashes = [
                    {
                        "path": str(path),
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in inputs
                ]
                provider_rows[provider] = {
                    # Keep the singular fields for compatibility when one
                    # file is supplied, while also binding every raw file.
                    "raw_path": str(inputs[0]),
                    "raw_sha256": raw_hashes[0]["sha256"],
                    "raw_paths": [item["path"] for item in raw_hashes],
                    "raw_sha256s": raw_hashes,
                    "source_plan": SOURCE_PLAN[provider],
                    "adapter": adapter_meta,
                    "expected_station_ids": sorted(expected_stations),
                    "actual_station_ids": sorted(actual_stations),
                    "historical_normalized_path": str(
                        protocol["providers"][provider]["normalized"]
                    ),
                    "normalized_path": str(normalized),
                    "test_normalized_path": str(normalized),
                    "normalized_sha256": sha256(normalized),
                    "raw_rows": raw_rows,
                    "rows": int(len(selected)),
                    "expected_rows": int(
                        len(expected_stations) * len(EXPECTED_GRID)
                    ),
                    "missing_timestamp_rows": int(
                        len(expected_stations) * len(EXPECTED_GRID) - raw_rows
                    ),
                    "timestamp_min": str(selected["timestamp_utc"].min()),
                    "timestamp_max": str(selected["timestamp_utc"].max()),
                    "status": "PASS",
                }
            except Exception as exc:
                failures.append(f"{provider}: {type(exc).__name__}: {exc}")
    status = "OFFICIAL_WINDOW_READY" if not failures and len(provider_rows) == 4 else "NOT_READY"
    manifest = {
        "status": status,
        "window": {
            "test_start_utc": START.isoformat().replace("+00:00", "Z"),
            "test_end_utc": END.isoformat().replace("+00:00", "Z"),
        },
        "providers": provider_rows,
        "failures": failures,
        "observation_values_read": bool(provider_rows),
        "historical_substitution": False,
        "claim_boundary": [
            "Only official 2026-10 observations may enter this manifest.",
            "No old normalized file or 2025 window may substitute for a missing provider.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "providers": len(provider_rows), "failures": failures}, ensure_ascii=False))
    return 0 if status == "OFFICIAL_WINDOW_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
