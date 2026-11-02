from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(r"F:\数据集\Analysis\experiments\pc_mstf_ap_v4\parquet\v4_anomaly_injection_table.parquet")


def stable_seed(seed: int, *parts: object) -> int:
    text = "|".join([str(seed), *map(str, parts)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def select_segments(frame: pd.DataFrame, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
    groups = {s: g.sort_values("timestamp_utc").index.to_numpy() for s, g in frame.groupby("station_key", sort=True)}
    stations = sorted(groups)
    out: list[np.ndarray] = []
    used: set[int] = set()
    for _ in range(max(20, n * 30)):
        if len(out) >= n:
            break
        station = str(rng.choice(stations))
        idx = groups[station]
        if len(idx) <= length:
            continue
        start = int(rng.integers(0, len(idx) - length + 1))
        seg = idx[start:start + length]
        if any(int(x) in used for x in seg):
            continue
        out.append(seg)
        used.update(map(int, seg))
    return out


def build_extensions(df: pd.DataFrame, seed: int, strength: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    strength = float(strength)
    if strength <= 0:
        raise ValueError("strength must be positive")
    out = df.copy().sort_values(["split", "timestamp_utc", "station_key"]).reset_index(drop=True)
    out["extension_attack_type"] = "normal"
    out["extension_is_attack"] = False
    out["extension_delta"] = 0.0
    out["extension_event_id"] = ""
    out["extension_observed_value"] = out["observed_value"].astype(float)
    events: list[dict[str, object]] = []

    # Keep training data clean: extensions are generated only in valid/test.
    for split in ["valid", "test"]:
        part = out[out["split"].eq(split) & out["original_value"].notna()]
        rng = np.random.default_rng(stable_seed(seed, split))
        stations = sorted(part["station_key"].unique().tolist())
        if not stations:
            continue

        # Replay: replace a contiguous segment by an earlier value sequence from the same station.
        for event_no, seg in enumerate(select_segments(part, 2, 12, rng)):
            station = str(out.loc[seg[0], "station_key"])
            station_rows = part[part["station_key"].eq(station)].sort_values("timestamp_utc")
            pos = station_rows.index.get_loc(seg[0])
            lag = 24
            source_pos = max(0, pos - lag)
            source_idx = station_rows.index[source_pos:source_pos + len(seg)]
            if len(source_idx) != len(seg):
                continue
            values = out.loc[source_idx, "original_value"].to_numpy(dtype=float)
            event_id = f"{split}_replay_{event_no:03d}"
            out.loc[seg, "extension_observed_value"] = values
            out.loc[seg, "extension_attack_type"] = "replay"
            out.loc[seg, "extension_is_attack"] = True
            out.loc[seg, "extension_delta"] = values - out.loc[seg, "original_value"].to_numpy(dtype=float)
            out.loc[seg, "extension_event_id"] = event_id
            events.append({"event_id": event_id, "split": split, "attack_type": "replay", "stations": [station], "rows": int(len(seg)), "seed": int(seed)})

        # Coordinated offset: same sign and magnitude across two stations at the same timestamps.
        common_times = part.groupby("timestamp_utc")["station_key"].nunique()
        candidate_times = common_times[common_times >= 2].index.to_numpy()
        if len(candidate_times):
            for event_no in range(2):
                ts = pd.Timestamp(rng.choice(candidate_times))
                selected = rng.choice(stations, size=min(2, len(stations)), replace=False).tolist()
                window_times = part[(part["timestamp_utc"] >= ts) & (part["timestamp_utc"] < ts + pd.Timedelta(hours=6)) & part["station_key"].isin(selected)].index.to_numpy()
                if len(window_times) < 2:
                    continue
                delta = float(rng.choice([-1.0, 1.0]) * rng.uniform(2.5, 4.5) * strength)
                event_id = f"{split}_coordinated_offset_{event_no:03d}"
                out.loc[window_times, "extension_observed_value"] = out.loc[window_times, "original_value"].to_numpy(dtype=float) + delta
                out.loc[window_times, "extension_attack_type"] = "coordinated_offset"
                out.loc[window_times, "extension_is_attack"] = True
                out.loc[window_times, "extension_delta"] = delta
                out.loc[window_times, "extension_event_id"] = event_id
                events.append({"event_id": event_id, "split": split, "attack_type": "coordinated_offset", "stations": selected, "rows": int(len(window_times)), "delta": delta, "seed": int(seed)})

        # Mixed missing + attack: mark a short segment as unavailable after a small offset.
        for event_no, seg in enumerate(select_segments(part, 1, 8, rng)):
            event_id = f"{split}_mixed_missing_attack_{event_no:03d}"
            delta = float(rng.choice([-1.0, 1.0]) * 3.0 * strength)
            vals = out.loc[seg, "original_value"].to_numpy(dtype=float) + delta
            vals[::2] = np.nan
            out.loc[seg, "extension_observed_value"] = vals
            out.loc[seg, "extension_attack_type"] = "mixed_missing_attack"
            out.loc[seg, "extension_is_attack"] = True
            out.loc[seg, "extension_delta"] = np.where(np.isfinite(vals), delta, np.nan)
            out.loc[seg, "extension_event_id"] = event_id
            events.append({"event_id": event_id, "split": split, "attack_type": "mixed_missing_attack", "stations": [str(out.loc[seg[0], "station_key"])], "rows": int(len(seg)), "delta": delta, "seed": int(seed)})

    return out, pd.DataFrame(events)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 TrustMeteo-QC 阶段5受控攻击扩展表")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    source = pd.read_parquet(args.data)
    extended, events = build_extensions(source, args.seed, args.strength)
    out_dir = ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_data = out_dir / "stage05_controlled_attack_extensions.parquet"
    out_events = out_dir / "stage05_controlled_attack_events.csv"
    out_meta = out_dir / "stage05_controlled_attack_extensions.json"
    extended.to_parquet(out_data, index=False)
    events.to_csv(out_events, index=False, encoding="utf-8-sig")
    payload = {"status": "PASS", "source": str(args.data), "output": str(out_data), "events": str(out_events), "seed": args.seed, "labeling": "受控攻击扩展，不代表真实网络入侵", "event_count": int(len(events)), "attack_types": sorted(events["attack_type"].unique().tolist()) if len(events) else []}
    out_meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
