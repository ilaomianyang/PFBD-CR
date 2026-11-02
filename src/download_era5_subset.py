from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="通过 Copernicus CDS API 下载六盘山附近 ERA5 小区域样本")
    parser.add_argument("--output", type=Path, default=Path(r"F:\数据集\Analysis\dataset\external\era5\era5_liupanshan_202207.grib"))
    parser.add_argument("--year", default="2022")
    parser.add_argument("--month", default="07")
    parser.add_argument("--day", nargs="+", default=[f"{d:02d}" for d in range(1, 32)])
    parser.add_argument("--area", default="36.75/104.75/34.25/107.75", help="北/西/南/东")
    args = parser.parse_args()
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("未安装 cdsapi，请先运行: python -m pip install cdsapi>=0.7.7") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client_kwargs = {}
    if os.environ.get("CDSAPI_URL"):
        client_kwargs["url"] = os.environ["CDSAPI_URL"]
    if os.environ.get("CDSAPI_KEY"):
        client_kwargs["key"] = os.environ["CDSAPI_KEY"]
    client = cdsapi.Client(**client_kwargs)
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
        ],
        "year": [str(args.year)],
        "month": [str(args.month).zfill(2)],
        "day": [str(x).zfill(2) for x in args.day],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": args.area,
    }
    client.retrieve("reanalysis-era5-single-levels", request, str(args.output))
    print(f"已下载: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
