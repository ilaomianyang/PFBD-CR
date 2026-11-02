from __future__ import annotations

"""Provider-specific parsers for the frozen V9 official-window input.

The adapters intentionally perform only schema normalization.  They do not
impute missing observations, select stations, choose thresholds, or inspect
attack labels.  Each parser returns the common columns required by the V9
data gate and a small provenance record describing the source fields.
"""

import io
import gzip
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COMMON_COLUMNS = (
    "station_id",
    "timestamp_utc",
    "temperature_c",
    "observation_present",
)

PARSER_VERSION = "v9-official-adapters-20260925-v1"

PROVIDER_CONTRACTS = {
    "MeteoSwiss": {
        "accepted_suffixes": (".csv",),
        "format": "semicolon-delimited CSV",
        "station_field": "station_abbr",
        "timestamp_field": "reference_timestamp",
        "temperature_field": "tre200h0",
        "timestamp_rule": "%d.%m.%Y %H:%M interpreted as UTC to match legacy Stage216 normalization",
        "temperature_rule": "numeric tre200h0; source missing values remain NaN",
    },
    "NOAA ISD": {
        "accepted_suffixes": (".csv", ".csv.gz", ".gz"),
        "format": "NOAA Global Hourly CSV, optionally gzip-compressed",
        "station_field": "station_id (adapter argument) or USAF-WBAN",
        "timestamp_field": "DATE",
        "temperature_field": "TMP component 0 divided by 10",
        "timestamp_rule": "DATE parsed as UTC",
        "temperature_rule": "TMP missing/sentinel components become NaN; quality codes are retained only as provenance",
    },
    "KNMI": {
        "accepted_suffixes": (".zip", ".txt", ".csv"),
        "format": "KNMI hourly ZIP/TXT/CSV",
        "station_field": "STN",
        "timestamp_field": "YYYYMMDD + HH",
        "temperature_field": "T divided by 10",
        "timestamp_rule": "YYYYMMDD + (HH - 1) hours, UTC",
        "temperature_rule": "numeric T / 10; source missing values remain NaN",
    },
    "ECCC": {
        "accepted_suffixes": (".csv",),
        "format": "ECCC GeoMet climate-hourly CSV",
        "station_field": "STN_ID",
        "timestamp_field": "UTC_DATE",
        "temperature_field": "TEMP",
        "timestamp_rule": "UTC_DATE parsed as UTC",
        "temperature_rule": "numeric TEMP; source missing values remain NaN",
    },
}


def _read_csv(path: Path, *, sep: str = ",") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig")


def _parse_isd_component(value: object) -> float:
    text = "" if value is None else str(value).strip()
    if not text:
        return np.nan
    component = text.split(",", 1)[0].strip()
    try:
        number = float(component)
    except ValueError:
        return np.nan
    if not np.isfinite(number) or number >= 9990:
        return np.nan
    return number / 10.0


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": pd.Series(dtype="string"),
            "timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "temperature_c": pd.Series(dtype="float64"),
            "observation_present": pd.Series(dtype="bool"),
        }
    )


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(COMMON_COLUMNS) - {"observation_present"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"adapter output missing columns: {missing}")
    out = frame[list(required)].copy()
    out["station_id"] = out["station_id"].astype(str).str.strip()
    out["timestamp_utc"] = pd.to_datetime(
        out["timestamp_utc"], errors="coerce", utc=True
    )
    out["temperature_c"] = pd.to_numeric(
        out["temperature_c"], errors="coerce"
    )
    out = out.dropna(subset=["station_id", "timestamp_utc"]).copy()
    out["observation_present"] = out["temperature_c"].notna()
    return out[
        ["station_id", "timestamp_utc", "temperature_c", "observation_present"]
    ].sort_values(["station_id", "timestamp_utc"], kind="stable").reset_index(
        drop=True
    )


def parse_meteoswiss(paths: Iterable[Path]) -> tuple[pd.DataFrame, dict]:
    paths = list(paths)
    frames: list[pd.DataFrame] = []
    for path in paths:
        raw = _read_csv(path, sep=";")
        required = {"station_abbr", "reference_timestamp", "tre200h0"}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"{path.name} missing MeteoSwiss fields: {missing}")
        frames.append(
            pd.DataFrame(
                {
                    "station_id": raw["station_abbr"].astype(str).str.lower().str.strip(),
                    "timestamp_utc": pd.to_datetime(
                        raw["reference_timestamp"].astype(str).str.strip(),
                        format="%d.%m.%Y %H:%M",
                        errors="coerce",
                        utc=True,
                    ),
                    "temperature_c": pd.to_numeric(
                        raw["tre200h0"], errors="coerce"
                    ),
                }
            )
        )
    return (
        _finalize(pd.concat(frames, ignore_index=True))
        if frames
        else _empty_frame(),
        {
            "provider": "MeteoSwiss",
            "parser_version": PARSER_VERSION,
            "input_count": len(paths),
            "raw_fields": ["station_abbr", "reference_timestamp", "tre200h0"],
            "normalized_fields": list(COMMON_COLUMNS),
        },
    )


def parse_noaa(paths: Iterable[Path]) -> tuple[pd.DataFrame, dict]:
    paths = list(paths)
    frames: list[pd.DataFrame] = []
    for path in paths:
        payload = path.read_bytes()
        if path.suffix.lower() == ".gz" or payload.startswith(b"\x1f\x8b"):
            payload = gzip.decompress(payload)
        raw = pd.read_csv(io.BytesIO(payload), dtype=str)
        if "DATE" not in raw.columns or "TMP" not in raw.columns:
            raise ValueError(f"{path.name} missing NOAA DATE/TMP fields")
        station_id = path.stem.replace(".csv", "")
        if {"STATION", "LATITUDE", "LONGITUDE"}.issubset(raw.columns):
            station_values = raw["STATION"].astype(str).str.strip()
            station_id = station_values.dropna().iloc[0] if not station_values.dropna().empty else station_id
        frames.append(
            pd.DataFrame(
                {
                    "station_id": str(station_id).strip(),
                    "timestamp_utc": pd.to_datetime(
                        raw["DATE"], errors="coerce", utc=True
                    ),
                    "temperature_c": raw["TMP"].map(_parse_isd_component),
                }
            )
        )
    return (
        _finalize(pd.concat(frames, ignore_index=True))
        if frames
        else _empty_frame(),
        {
            "provider": "NOAA ISD",
            "parser_version": PARSER_VERSION,
            "input_count": len(paths) if isinstance(paths, list) else None,
            "raw_fields": ["DATE", "TMP"],
            "normalized_fields": list(COMMON_COLUMNS),
        },
    )


def _knmi_text_frames(path: Path) -> list[pd.DataFrame]:
    payload = path.read_bytes()
    members: list[tuple[str, bytes]] = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".txt"):
                    members.append((name, archive.read(name)))
    else:
        members.append((path.name, payload))
    result: list[pd.DataFrame] = []
    for name, member in members:
        text = member.decode("utf-8", errors="replace")
        lines = text.splitlines()
        header_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.lstrip("# ").startswith("STN,")
            ),
            None,
        )
        if header_index is None:
            raise ValueError(f"{path.name}/{name} has no KNMI STN header")
        csv_text = "\n".join(
            [lines[header_index].lstrip("# ").strip()]
            + lines[header_index + 1 :]
        )
        frame = pd.read_csv(io.StringIO(csv_text), dtype=str)
        frame.columns = [str(column).strip() for column in frame.columns]
        required = {"STN", "YYYYMMDD", "HH", "T"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name}/{name} missing KNMI fields: {missing}")
        result.append(frame)
    return result


def parse_knmi(paths: Iterable[Path]) -> tuple[pd.DataFrame, dict]:
    paths = list(paths)
    frames: list[pd.DataFrame] = []
    for path in paths:
        for raw in _knmi_text_frames(path):
            date = pd.to_datetime(
                raw["YYYYMMDD"].astype(str).str.strip(),
                format="%Y%m%d",
                errors="coerce",
                utc=True,
            )
            hour = pd.to_numeric(raw["HH"], errors="coerce")
            frames.append(
                pd.DataFrame(
                    {
                        "station_id": raw["STN"].astype(str).str.strip(),
                        "timestamp_utc": date + pd.to_timedelta(
                            hour - 1, unit="h"
                        ),
                        "temperature_c": pd.to_numeric(
                            raw["T"], errors="coerce"
                        )
                        / 10.0,
                    }
                )
            )
    return (
        _finalize(pd.concat(frames, ignore_index=True))
        if frames
        else _empty_frame(),
        {
            "provider": "KNMI",
            "parser_version": PARSER_VERSION,
            "input_count": len(paths) if isinstance(paths, list) else None,
            "raw_fields": ["STN", "YYYYMMDD", "HH", "T"],
            "normalized_fields": list(COMMON_COLUMNS),
        },
    )


def parse_eccc(paths: Iterable[Path]) -> tuple[pd.DataFrame, dict]:
    paths = list(paths)
    frames: list[pd.DataFrame] = []
    for path in paths:
        raw = _read_csv(path)
        required = {"STN_ID", "UTC_DATE", "TEMP"}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"{path.name} missing ECCC fields: {missing}")
        frames.append(
            pd.DataFrame(
                {
                    "station_id": raw["STN_ID"].astype(str).str.strip(),
                    "timestamp_utc": pd.to_datetime(
                        raw["UTC_DATE"], errors="coerce", utc=True
                    ),
                    "temperature_c": pd.to_numeric(
                        raw["TEMP"], errors="coerce"
                    ),
                }
            )
        )
    return (
        _finalize(pd.concat(frames, ignore_index=True))
        if frames
        else _empty_frame(),
        {
            "provider": "ECCC",
            "parser_version": PARSER_VERSION,
            "input_count": len(paths) if isinstance(paths, list) else None,
            "raw_fields": ["STN_ID", "UTC_DATE", "TEMP"],
            "normalized_fields": list(COMMON_COLUMNS),
        },
    )


PARSERS = {
    "MeteoSwiss": parse_meteoswiss,
    "NOAA ISD": parse_noaa,
    "KNMI": parse_knmi,
    "ECCC": parse_eccc,
}


def parse_provider(provider: str, paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    if provider not in PARSERS:
        raise KeyError(f"unsupported V9 provider: {provider}")
    if not paths:
        raise ValueError(f"{provider}: no raw inputs supplied")
    return PARSERS[provider](paths)
