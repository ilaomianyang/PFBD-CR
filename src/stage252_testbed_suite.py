from __future__ import annotations

"""Run the complete authorized testbed suite.

The suite creates one clean run and one run for each frozen attack type and
seed by invoking the same real loopback TCP runner used for individual runs.
It does not train a detector or evaluate the frozen project endpoint.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "stage252_testbed_runner.py"
PREFIX = "stage252_authorized_testbed_suite"
ATTACK_TYPES = [
    "replay",
    "tamper",
    "freeze",
    "delay",
    "drop",
    "coordinated_offset",
]
SEEDS = [3407, 6257, 7919, 104729, 20260715]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--interval", type=float, default=0.01)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--attack-type", action="append")
    args = parser.parse_args(argv)
    seeds = args.seed or SEEDS
    attacks = args.attack_type or ATTACK_TYPES
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    started = time.time()
    results: list[dict[str, object]] = []
    for seed in seeds:
        command = [
            sys.executable,
            "-u",
            str(RUNNER),
            "--mode",
            "clean",
            "--seed",
            str(seed),
            "--frames",
            str(args.frames),
            "--interval",
            str(args.interval),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        results.append(
            {
                "seed": seed,
                "mode": "clean",
                "attack_type": "clean",
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
        for attack_type in attacks:
            command = [
                sys.executable,
                "-u",
                str(RUNNER),
                "--mode",
                "attack",
                "--attack-type",
                attack_type,
                "--seed",
                str(seed),
                "--frames",
                str(args.frames),
                "--interval",
                str(args.interval),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT.parent,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            results.append(
                {
                    "seed": seed,
                    "mode": "attack",
                    "attack_type": attack_type,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                }
            )
    failures = [row for row in results if int(row["returncode"]) != 0]
    payload = {
        "status": "PASS_AUTHORIZED_TESTBED_SUITE"
        if not failures
        else "REVIEW_REQUIRED",
        "run_count": len(results),
        "clean_run_count": sum(row["mode"] == "clean" for row in results),
        "attack_run_count": sum(row["mode"] == "attack" for row in results),
        "seed_count": len(seeds),
        "attack_type_count": len(attacks),
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 3),
        "results": results,
    }
    output = ROOT / "tables" / f"{PREFIX}.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
