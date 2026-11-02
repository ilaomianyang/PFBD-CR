from __future__ import annotations

"""Run a real, authorized, loopback-only telemetry testbed.

The runner creates a local TCP sink and one or more local TCP senders.  The
sender emits newline-delimited JSON telemetry frames.  Attack modes are
executed at runtime by the sender or by an in-path relay; the receiver logs
the bytes and monotonic receipt times that it actually observes.  No value
table is edited after the fact.

The testbed is intentionally small and dependency-free.  It is a real
protocol experiment, not a production incident log.
"""

import argparse
import hashlib
import json
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "testbed" / "raw"
LOGS = ROOT / "testbed" / "logs"
TABLES = ROOT / "tables"

STATION_IDS = ["TB01", "TB02", "TB03", "TB04"]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 25252
ATTACK_TYPES = [
    "replay",
    "tamper",
    "freeze",
    "delay",
    "drop",
    "coordinated_offset",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_frame(frame: dict[str, Any]) -> bytes:
    body = json.dumps(frame, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return body + b"\n"


def base_frame(
    station_id: str,
    sequence: int,
    wall_time: float,
    seed: int,
) -> dict[str, Any]:
    phase = (sequence + seed % 1000) / 7.0
    temperature = 12.0 + 4.0 * ((sequence % 97) / 96.0)
    humidity = 55.0 + 10.0 * ((sequence % 53) / 52.0)
    wind = 1.5 + 1.5 * abs((sequence % 31) / 31.0)
    pressure = 900.0 + 20.0 * ((sequence % 41) / 40.0)
    frame = {
        "station_id": station_id,
        "timestamp_utc": datetime.fromtimestamp(
            wall_time, tz=timezone.utc
        ).isoformat(),
        "sequence": int(sequence),
        "temperature_c": round(temperature + 0.2 * (1.0 + (sequence % 3)), 6),
        "humidity_pct": round(humidity, 6),
        "wind_mps": round(wind, 6),
        "pressure_hpa": round(pressure, 6),
        "nonce": f"{station_id}-{sequence}-{int(time.time_ns())}",
        "phase_marker": round(phase, 6),
    }
    frame["payload_sha256"] = sha256_bytes(canonical_frame(frame)[:-1])
    return frame


def clean_stream(
    station_id: str,
    count: int,
    seed: int,
    interval: float,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for sequence in range(count):
        frames.append(
            base_frame(
                station_id=station_id,
                sequence=sequence,
                wall_time=time.time() + sequence * interval,
                seed=seed,
            )
        )
    return frames


def apply_attack(
    frames: list[dict[str, Any]],
    attack_type: str,
    seed: int,
) -> list[dict[str, Any]]:
    if not frames:
        return []
    if attack_type == "replay":
        split = max(1, len(frames) // 3)
        return frames[:split] + frames[:split] + frames[split:]
    if attack_type == "tamper":
        output = [dict(frame) for frame in frames]
        for index in range(len(output) // 4, len(output) // 2):
            output[index]["temperature_c"] = round(
                float(output[index]["temperature_c"]) + 8.0, 6
            )
        return output
    if attack_type == "freeze":
        output = [dict(frame) for frame in frames]
        if len(output) > 4:
            frozen = dict(output[len(output) // 3])
            for index in range(len(output) // 3, 2 * len(output) // 3):
                copy = dict(frozen)
                copy["sequence"] = output[index]["sequence"]
                copy["timestamp_utc"] = output[index]["timestamp_utc"]
                output[index] = copy
        return output
    if attack_type == "delay":
        output = [dict(frame) for frame in frames]
        for frame in output[len(output) // 3 : 2 * len(output) // 3]:
            frame["delay_seconds"] = 0.2
        return output
    if attack_type == "drop":
        return [
            frame
            for index, frame in enumerate(frames)
            if not (len(frames) // 3 <= index < 2 * len(frames) // 3)
        ]
    if attack_type == "coordinated_offset":
        output = [dict(frame) for frame in frames]
        for index in range(len(output) // 4, 3 * len(output) // 4):
            output[index]["temperature_c"] = round(
                float(output[index]["temperature_c"]) + 4.0, 6
            )
            output[index]["pressure_hpa"] = round(
                float(output[index]["pressure_hpa"]) + 3.0, 6
            )
        return output
    raise ValueError(f"unknown attack type: {attack_type}")


@dataclass
class ReceivedFrame:
    station_id: str
    sequence: int
    received_monotonic: float
    raw_sha256: str
    payload: dict[str, Any]


def receive_frames(
    host: str,
    port: int,
    expected_frames: int,
    timeout: float,
    output_path: Path,
) -> list[ReceivedFrame]:
    received: list[ReceivedFrame] = []
    buffer = bytearray()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(8)
        server.settimeout(timeout)
        connection, address = server.accept()
        connection.settimeout(timeout)
        with connection:
            while len(received) < expected_frames:
                try:
                    chunk = connection.recv(65536)
                except socket.timeout:
                    break
                if not chunk:
                    break
                buffer.extend(chunk)
                while b"\n" in buffer:
                    raw, _, remainder = buffer.partition(b"\n")
                    buffer = bytearray(remainder)
                    if not raw:
                        continue
                    payload = json.loads(raw.decode("utf-8"))
                    received.append(
                        ReceivedFrame(
                            station_id=str(payload["station_id"]),
                            sequence=int(payload["sequence"]),
                            received_monotonic=time.monotonic(),
                            raw_sha256=sha256_bytes(raw),
                            payload=payload,
                        )
                    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for frame in received:
            record = {
                "station_id": frame.station_id,
                "sequence": frame.sequence,
                "received_monotonic": frame.received_monotonic,
                "raw_sha256": frame.raw_sha256,
                "payload": frame.payload,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return received


def send_frames(
    host: str,
    port: int,
    frames: list[dict[str, Any]],
    interval: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with socket.create_connection((host, port), timeout=10.0) as client:
        with output_path.open("w", encoding="utf-8") as handle:
            for frame in frames:
                delay = float(frame.get("delay_seconds", 0.0))
                if delay:
                    time.sleep(delay)
                raw = canonical_frame(frame)
                client.sendall(raw)
                handle.write(
                    json.dumps(
                        {
                            "sent_monotonic": time.monotonic(),
                            "sequence": frame["sequence"],
                            "raw_sha256": sha256_bytes(raw.rstrip(b"\n")),
                            "payload": frame,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                time.sleep(interval)


def run_one(
    mode: str,
    attack_type: str,
    seed: int,
    host: str,
    port: int,
    frame_count: int,
    interval: float,
) -> dict[str, Any]:
    if mode == "clean":
        stream = clean_stream(
            station_id=STATION_IDS[0],
            count=frame_count,
            seed=seed,
            interval=interval,
        )
    else:
        stream = apply_attack(
            clean_stream(
                station_id=STATION_IDS[0],
                count=frame_count,
                seed=seed,
                interval=interval,
            ),
            attack_type=attack_type,
            seed=seed,
        )
    run_id = f"{mode}__{attack_type}__seed{seed}"
    receive_path = RAW / f"{run_id}_received.jsonl"
    send_path = LOGS / f"{run_id}_sent.jsonl"
    received: list[ReceivedFrame] = []
    receive_error: list[str] = []

    def receiver() -> None:
        try:
            received.extend(
                receive_frames(
                    host=host,
                    port=port,
                    expected_frames=len(stream),
                    timeout=max(10.0, interval * frame_count + 5.0),
                    output_path=receive_path,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            receive_error.append(repr(exc))

    thread = threading.Thread(target=receiver, daemon=True)
    thread.start()
    time.sleep(0.2)
    send_error: list[str] = []
    try:
        send_frames(
            host=host,
            port=port,
            frames=stream,
            interval=interval,
            output_path=send_path,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        send_error.append(repr(exc))
    thread.join(timeout=max(15.0, interval * frame_count + 10.0))

    sequence_counts: dict[str, int] = {}
    for frame in received:
        key = str(frame.sequence)
        sequence_counts[key] = sequence_counts.get(key, 0) + 1
    duplicate_sequences = sorted(
        int(sequence)
        for sequence, count in sequence_counts.items()
        if count > 1
    )
    missing_sequences = sorted(
        set(int(frame["sequence"]) for frame in stream)
        - set(int(sequence) for sequence in sequence_counts)
    )
    summary = {
        "run_id": run_id,
        "mode": mode,
        "attack_type": attack_type,
        "seed": seed,
        "host": host,
        "port": port,
        "sent_frames": len(stream),
        "received_frames": len(received),
        "duplicate_sequence_count": len(duplicate_sequences),
        "missing_sequence_count": len(missing_sequences),
        "duplicate_sequences": duplicate_sequences[:20],
        "missing_sequences": missing_sequences[:20],
        "received_path": str(receive_path.relative_to(ROOT)),
        "sent_path": str(send_path.relative_to(ROOT)),
        "send_error": send_error,
        "receive_error": receive_error,
        "complete": not send_error
        and not receive_error
        and len(received) == len(stream),
        "started_at_utc": utc_now(),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["clean", "attack"], required=True)
    parser.add_argument("--attack-type", choices=ATTACK_TYPES)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--interval", type=float, default=0.01)
    args = parser.parse_args()
    attack_type = args.attack_type or "clean"
    if args.mode == "attack" and not args.attack_type:
        parser.error("--attack-type is required in attack mode")
    summary = run_one(
        mode=args.mode,
        attack_type=attack_type,
        seed=args.seed,
        host=args.host,
        port=args.port,
        frame_count=args.frames,
        interval=args.interval,
    )
    summary_path = TABLES / f"stage252_testbed_{summary['run_id']}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
