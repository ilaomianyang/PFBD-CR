"""Collect Stage503 single-step latency / memory probes into one CSV."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
MODELS = ["lstm_ae", "usad", "timesnet", "anomaly_transformer"]


def main() -> int:
    rows = []
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    for model in MODELS:
        cmd = [
            sys.executable,
            str(ROOT / "src" / "stage503_deep_anomaly_baselines.py"),
            "--resource-only",
            "--model",
            model,
            "--length",
            "24",
        ]
        print(f"[probe] {model}", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-4000:])
            raise SystemExit(f"probe failed for {model}: {proc.returncode}")
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESOURCE_JSON "):
                payload = json.loads(line[len("RESOURCE_JSON ") :])
        if payload is None:
            raise SystemExit(f"no RESOURCE_JSON for {model}:\n{proc.stdout[-2000:]}")
        rows.append(payload)
        print(
            f"  mean={payload['latency_mean_ms']:.3f} ms p95={payload['latency_p95_ms']:.3f} "
            f"peak={payload['peak_wset_mb']:.1f} MB params={payload['params']}",
            flush=True,
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "stage503_resources_raw.csv", index=False, encoding="utf-8-sig")
    print(out[["method", "params", "param_mb", "latency_mean_ms", "latency_p95_ms", "peak_wset_mb"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
