from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parents[1] / "dataset" / "external" / "noaa_isd"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
HISTORY = DATA_DIR / "isd-history.csv"
REFERENCE_LAT = 35.25
REFERENCE_LON = 106.25


def distance_km(lat: pd.Series, lon: pd.Series) -> pd.Series:
    p1 = np.radians(REFERENCE_LAT)
    p2 = np.radians(lat.astype(float))
    dl = np.radians(lon.astype(float) - REFERENCE_LON)
    dp = p2 - p1
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def main() -> int:
    history = pd.read_csv(HISTORY, dtype={"USAF": str, "WBAN": str, "BEGIN": str, "END": str})
    history["begin_date"] = pd.to_datetime(history["BEGIN"], format="%Y%m%d", errors="coerce")
    history["end_date"] = pd.to_datetime(history["END"], format="%Y%m%d", errors="coerce")
    history["distance_km"] = distance_km(history["LAT"], history["LON"])
    selected = history[
        history["CTRY"].eq("CH")
        & history["begin_date"].le(pd.Timestamp("2022-12-31"))
        & history["end_date"].ge(pd.Timestamp("2022-01-01"))
        & history["distance_km"].le(300.0)
        & history["LAT"].notna()
        & history["LON"].notna()
    ].copy()
    selected["station_id"] = selected["USAF"].astype(str).str.zfill(6) + selected["WBAN"].astype(str).str.zfill(5)
    existing = {p.stem for p in DATA_DIR.glob("*.csv") if p.name != "isd-history.csv" and p.name != "nearby_liupanshan_stations.csv"}
    selected["already_downloaded"] = selected["station_id"].isin(existing)
    selected["official_2022_url"] = "https://www.ncei.noaa.gov/data/global-hourly/access/2022/" + selected["station_id"] + ".csv"
    selected = selected.sort_values(["already_downloaded", "distance_km"], ascending=[False, True])
    columns = ["station_id", "USAF", "WBAN", "STATION NAME", "ICAO", "LAT", "LON", "ELEV(M)", "BEGIN", "END", "begin_date", "end_date", "distance_km", "already_downloaded", "official_2022_url"]
    selected = selected[columns].reset_index(drop=True)
    candidates = selected[~selected["already_downloaded"]].copy()
    payload = {"status": "PASS", "protocol": "仅使用本地NOAA ISD官方站点清单筛选；2022覆盖、距离≤300 km、国家代码CH；不下载、不修改已有数据。", "reference": {"lat": REFERENCE_LAT, "lon": REFERENCE_LON}, "candidate_count": int(len(candidates)), "selected_count": int(len(selected)), "already_downloaded_count": int(selected["already_downloaded"].sum()), "candidates": candidates.to_dict(orient="records")}
    selected.to_csv(TABLES / "stage51_noaa_station_inventory.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(TABLES / "stage51_noaa_station_candidates.csv", index=False, encoding="utf-8-sig")
    (TABLES / "stage51_noaa_station_inventory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True, default=str), encoding="utf-8")
    lines = ["# 阶段51 NOAA扩展站点官方清单", "", "仅根据本地 `isd-history.csv` 筛选，不下载、不修改已有数据。", f"- 参考点：`{REFERENCE_LAT}N, {REFERENCE_LON}E`；距离上限：`300 km`", f"- 符合2022年覆盖站点：`{len(selected)}`；已下载：`{int(selected['already_downloaded'].sum())}`；待扩展候选：`{len(candidates)}`", "", "| 站点ID | 名称 | 距离km | 海拔m | 覆盖起始 | 覆盖结束 | 已下载 | 官方2022地址 |", "|---|---|---:|---:|---|---|---|---|"]
    for _, row in selected.head(30).iterrows():
        lines.append(f"| {row['station_id']} | {row['STATION NAME']} | {row['distance_km']:.1f} | {row['ELEV(M)']} | {row['BEGIN']} | {row['END']} | {'是' if row['already_downloaded'] else '否'} | {row['official_2022_url']} |")
    lines += ["", "扩展候选必须先从官方地址下载并经过字段、时间覆盖和质量标记审计，才能进入后续受控实验。"]
    (REPORTS / "stage51_noaa_station_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected_count": len(selected), "candidate_count": len(candidates)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
