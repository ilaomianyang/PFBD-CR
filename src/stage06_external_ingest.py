from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
NOAA_DIR = Path(r"F:\数据集\Analysis\dataset\external\noaa_isd")
ERA5_DIR = Path(r"F:\数据集\Analysis\dataset\external\era5")
FEATURE_TABLE = Path(r"F:\数据集\Analysis\parquet\ml_feature_table.parquet")


def parse_noaa_component(value: object, index: int, scale: float = 1.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        text = str(value).split(",")[index]
        number = float(text)
        if number >= 9990:
            return np.nan
        return number / scale
    except (IndexError, TypeError, ValueError):
        return np.nan


def ingest_noaa() -> pd.DataFrame:
    rows = []
    for path in sorted(NOAA_DIR.glob("*99999.csv")):
        raw = pd.read_csv(path, dtype=str)
        if raw.empty:
            continue
        out = pd.DataFrame({
            "source": "NOAA_ISD",
            "station_id": raw["STATION"].astype(str),
            "station_name": raw["NAME"].astype(str),
            "timestamp_utc": pd.to_datetime(raw["DATE"], errors="coerce", utc=True),
            "latitude": pd.to_numeric(raw["LATITUDE"], errors="coerce"),
            "longitude": pd.to_numeric(raw["LONGITUDE"], errors="coerce"),
            "elevation_m": pd.to_numeric(raw["ELEVATION"], errors="coerce"),
            "temperature_c": raw["TMP"].map(lambda x: parse_noaa_component(x, 0, 10.0)),
            "temperature_quality": raw["TMP"].map(lambda x: str(x).split(",")[1] if "," in str(x) else ""),
            "dewpoint_c": raw["DEW"].map(lambda x: parse_noaa_component(x, 0, 10.0)),
            "pressure_hpa": raw["SLP"].map(lambda x: parse_noaa_component(x, 0, 10.0)),
            "wind_direction_deg": raw["WND"].map(lambda x: parse_noaa_component(x, 0, 1.0)),
            "wind_speed_ms": raw["WND"].map(lambda x: parse_noaa_component(x, 3, 10.0)),
            "visibility_m": raw["VIS"].map(lambda x: parse_noaa_component(x, 0, 10.0)),
            "report_type": raw["REPORT_TYPE"].astype(str),
            "quality_control": raw["QUALITY_CONTROL"].astype(str),
        })
        out.loc[~out["wind_direction_deg"].between(0, 360), "wind_direction_deg"] = np.nan
        rows.append(out)
    if not rows:
        raise FileNotFoundError(f"NOAA files not found in {NOAA_DIR}")
    out = pd.concat(rows, ignore_index=True).dropna(subset=["timestamp_utc"])
    return out.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.asin(math.sqrt(min(1.0, a)))


def ingest_era5() -> pd.DataFrame:
    files = sorted(ERA5_DIR.glob("era5_liupanshan_202207*.grib"))
    files = [p for p in files if p.name != "era5_liupanshan_20220701.grib"]
    if not files:
        raise FileNotFoundError(f"ERA5 chunk files not found in {ERA5_DIR}")
    datasets = [xr.open_dataset(path, engine="cfgrib") for path in files]
    ds = xr.concat(datasets, dim="time").sortby("time")
    ds = ds.sel(time=~ds.get_index("time").duplicated())
    stations = pd.read_parquet(FEATURE_TABLE, columns=["station_key", "station_lat", "station_lon", "station_alt_m"]).drop_duplicates().sort_values("station_key")
    rows = []
    for _, station in stations.iterrows():
        lat = float(station["station_lat"])
        lon = float(station["station_lon"])
        nearest_lat = float(ds.latitude.sel(latitude=lat, method="nearest"))
        nearest_lon = float(ds.longitude.sel(longitude=lon, method="nearest"))
        sub = ds.sel(latitude=nearest_lat, longitude=nearest_lon)
        frame = pd.DataFrame({
            "source": "ERA5",
            "station_key": station["station_key"],
            "timestamp_utc": pd.to_datetime(sub["time"].values, utc=True),
            "station_lat": lat,
            "station_lon": lon,
            "station_alt_m": float(station["station_alt_m"]),
            "era5_grid_lat": nearest_lat,
            "era5_grid_lon": nearest_lon,
            "grid_distance_km": haversine_km(lat, lon, nearest_lat, nearest_lon),
            "era5_t2m_c": sub["t2m"].values.astype(float) - 273.15,
            "era5_u10_ms": sub["u10"].values.astype(float),
            "era5_v10_ms": sub["v10"].values.astype(float),
            "era5_wind_speed_ms": np.hypot(sub["u10"].values.astype(float), sub["v10"].values.astype(float)),
            "era5_surface_pressure_hpa": sub["sp"].values.astype(float) / 100.0,
        })
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["station_key", "timestamp_utc"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段6：统一接入 NOAA ISD 与 ERA5")
    parser.add_argument("--noaa-only", action="store_true")
    parser.add_argument("--era5-only", action="store_true")
    args = parser.parse_args()
    out_dir = ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": "PASS", "sources": {}}
    if not args.era5_only:
        noaa = ingest_noaa()
        noaa_path = out_dir / "stage06_noaa_isd_normalized.parquet"
        noaa.to_parquet(noaa_path, index=False)
        payload["sources"]["NOAA_ISD"] = {"path": str(noaa_path), "rows": int(len(noaa)), "stations": int(noaa["station_id"].nunique()), "date_min": str(noaa["timestamp_utc"].min()), "date_max": str(noaa["timestamp_utc"].max())}
    if not args.noaa_only:
        era5 = ingest_era5()
        era5_path = out_dir / "stage06_era5_station_reference.parquet"
        era5.to_parquet(era5_path, index=False)
        payload["sources"]["ERA5"] = {"path": str(era5_path), "rows": int(len(era5)), "stations": int(era5["station_key"].nunique()), "date_min": str(era5["timestamp_utc"].min()), "date_max": str(era5["timestamp_utc"].max()), "variables": ["era5_t2m_c", "era5_u10_ms", "era5_v10_ms", "era5_surface_pressure_hpa"]}
    meta = out_dir / "stage06_external_ingest.json"
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
