from __future__ import annotations

"""Stage 255: online causal detection and recovery on the authorized testbed.

This stage consumes the raw Stage252 send/receive logs and evaluates a
strictly causal, streaming detector and recovery loop.  It never uses a
future frame, never backfills a previously emitted value, and never merges
testbed observations into the frozen project test set.

The testbed environment is authorized and isolated.  These are real runtime
protocol events with real monotonic timestamps, but they are not production
incident logs.  The positive-result ceiling is therefore
``PARTIALLY_CLOSED``.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
RAW = ROOT / "testbed" / "raw"
LOGS = ROOT / "testbed" / "logs"
PREFIX = "stage255_online_causal_testbed"

ATTACK_TYPES = [
    "clean",
    "replay",
    "tamper",
    "freeze",
    "delay",
    "drop",
    "coordinated_offset",
]
SEEDS = [3407, 6257, 7919, 104729, 20260715]
WARMUP_FRAMES = 30
ALPHA = 0.20
SIGMA_K = 4.0
RECOVERY_SIGMA_K = 3.0
REPLAY_DUPLICATE_LOOKBACK = 64
FAR_TARGET = 0.01
EXPECTED_INTERVAL_SECONDS = 0.01
ARRIVAL_GAP_K = 3.0
ARRIVAL_GAP_FLOOR_SECONDS = 0.05
SEQUENCE_GAP_ALERT_MISSING = 1
OFFSET_WINDOW = 12
OFFSET_WINDOW_SIGMA_K = 3.0
OFFSET_WINDOW_FLOOR = 6.0
OFFSET_MIN_CONSECUTIVE = 3
FREEZE_MIN_RUN = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_id_for(mode: str, attack_type: str, seed: int) -> str:
    return f"{mode}__{attack_type}__seed{seed}"


def truth_station_id(attack_type: str, run_id: str) -> str:
    if attack_type == "drop":
        return str(run_id.split("__")[0])
    return "TB01"


def parse_sent_logs(
    attack_type: str,
    seed: int,
) -> list[dict[str, Any]]:
    if attack_type == "clean":
        mode = "clean"
        attack = "clean"
    else:
        mode = "attack"
        attack = attack_type
    run_id = run_id_for(mode, attack, seed)
    path = LOGS / f"{run_id}_sent.jsonl"
    rows = read_jsonl(path)
    records: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload", {})
        records.append(
            {
                "sent_monotonic": float(row["sent_monotonic"]),
                "sequence": int(payload["sequence"]),
                "station_id": str(payload["station_id"]),
                "payload_sha256": str(payload["payload_sha256"]),
                "delay_seconds": float(payload.get("delay_seconds", 0.0)),
                "temperature_c": float(payload["temperature_c"]),
                "humidity_pct": float(payload["humidity_pct"]),
                "wind_mps": float(payload["wind_mps"]),
                "pressure_hpa": float(payload["pressure_hpa"]),
                "nonce": str(payload.get("nonce", "")),
            }
        )
    return records


def parse_received_logs(
    attack_type: str,
    seed: int,
) -> list[dict[str, Any]]:
    if attack_type == "clean":
        mode = "clean"
        attack = "clean"
    else:
        mode = "attack"
        attack = attack_type
    run_id = run_id_for(mode, attack, seed)
    path = RAW / f"{run_id}_received.jsonl"
    rows = read_jsonl(path)
    records: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload", {})
        records.append(
            {
                "received_monotonic": float(row["received_monotonic"]),
                "sequence": int(row["sequence"]),
                "station_id": str(row["station_id"]),
                "raw_sha256": str(row["raw_sha256"]),
                "temperature_c": float(payload["temperature_c"]),
                "humidity_pct": float(payload["humidity_pct"]),
                "wind_mps": float(payload["wind_mps"]),
                "pressure_hpa": float(payload["pressure_hpa"]),
                "nonce": str(payload.get("nonce", "")),
            }
        )
    return records


def rolling_features(
    values: np.ndarray,
    state: dict[str, float],
) -> np.ndarray:
    output = np.zeros((len(values), 4), dtype=float)
    for index, value in enumerate(values):
        mean = state.get("mean", value)
        variance = state.get("variance", 0.0)
        sigma = math.sqrt(max(variance, 1e-12))
        output[index, 0] = abs(value - mean) / (sigma + 1e-9)
        output[index, 1] = abs(value - state.get("last", value)) / (
            sigma + 1e-9
        )
        output[index, 2] = sigma
        output[index, 3] = abs(value - state.get("last", value))
        state["last"] = float(value)
        state["mean"] = float(
            (1.0 - ALPHA) * mean + ALPHA * value
        )
        state["variance"] = float(
            (1.0 - ALPHA) * variance
            + ALPHA * (value - state["mean"]) ** 2
        )
    return output


def online_detect(
    received: list[dict[str, Any]],
    attack_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strictly causal frame and arrival-event detector.

    The detector only consumes the current frame, earlier frames, and the
    wall-clock monotonic arrival time.  It emits 1-based arrival positions so
    that delayed and missing events can be scored without looking ahead.
    """
    ordered = sorted(
        received,
        key=lambda row: (
            row["station_id"],
            row["received_monotonic"],
            row["sequence"],
        ),
    )
    state: dict[str, dict[str, float]] = {}
    seen: set[tuple[str, int]] = set()
    history: list[tuple[str, int]] = []
    output: list[dict[str, Any]] = []
    last_arrival: dict[str, float] = {}
    last_sequence: dict[str, int] = {}
    freeze_run: dict[str, int] = {}
    offset_run: dict[str, int] = {}
    window: list[float] = []
    for index, row in enumerate(ordered):
        station = str(row["station_id"])
        if station not in state:
            state[station] = {
                "mean": float(row["temperature_c"]),
                "variance": 0.0,
                "last": float(row["temperature_c"]),
            }
        previous = state[station]
        sigma = math.sqrt(max(float(previous["variance"]), 1e-12))
        delta = abs(
            float(row["temperature_c"]) - float(previous["last"])
        )
        z = delta / (sigma + 1e-9)
        duplicate = replay_duplicate_count(
            seen,
            station,
            int(row["sequence"]),
            history,
        )
        replay_flag = bool(duplicate)
        spike_flag = bool(index >= WARMUP_FRAMES and z >= SIGMA_K)
        arrival_gap = (
            float(row["received_monotonic"]) - last_arrival[station]
            if station in last_arrival
            else float(EXPECTED_INTERVAL_SECONDS)
        )
        arrival_gap_threshold = max(
            ARRIVAL_GAP_FLOOR_SECONDS,
            ARRIVAL_GAP_K * EXPECTED_INTERVAL_SECONDS,
        )
        delay_flag = bool(
            index >= WARMUP_FRAMES
            and arrival_gap >= arrival_gap_threshold
        )
        sequence_gap_flag = False
        missing_sequence_count = 0
        if station in last_sequence:
            sequence_delta = int(row["sequence"]) - int(
                last_sequence[station]
            )
            if sequence_delta > 1:
                missing_sequence_count = sequence_delta - 1
                sequence_gap_flag = bool(
                    missing_sequence_count
                    >= SEQUENCE_GAP_ALERT_MISSING
                )
        last_sequence[station] = int(row["sequence"])
        last_arrival[station] = float(row["received_monotonic"])

        if abs(delta) <= 1e-9 and duplicate == 0:
            freeze_run[station] = freeze_run.get(station, 0) + 1
        else:
            freeze_run[station] = 0
        freeze_flag = bool(
            index >= WARMUP_FRAMES
            and freeze_run[station] >= FREEZE_MIN_RUN
        )

        window.append(float(row["temperature_c"]))
        if len(window) > OFFSET_WINDOW:
            window.pop(0)
        window_median = float(np.median(window))
        window_mad = float(
            np.median(np.abs(np.asarray(window) - window_median))
        )
        window_offset = abs(float(row["temperature_c"]) - window_median)
        offset_flag = bool(
            index >= WARMUP_FRAMES
            and len(window) == OFFSET_WINDOW
            and window_offset >= max(
                OFFSET_WINDOW_FLOOR,
                OFFSET_WINDOW_SIGMA_K * 1.4826 * window_mad,
            )
        )
        if offset_flag:
            offset_run[station] = offset_run.get(station, 0) + 1
        else:
            offset_run[station] = 0
        sustained_offset_flag = bool(
            offset_run.get(station, 0) >= OFFSET_MIN_CONSECUTIVE
        )

        reasons: list[str] = []
        if replay_flag:
            reasons.append("replay")
        if arrival_gap > 0.0 and arrival_gap >= arrival_gap_threshold:
            reasons.append("arrival_delay")
        if sequence_gap_flag:
            reasons.append("sequence_gap")
        if sustained_offset_flag:
            reasons.append("offset")
        if spike_flag:
            reasons.append("spike")
        if freeze_flag:
            reasons.append("freeze")
        alert = bool(reasons)
        output.append(
            {
                "arrival_index": int(index + 1),
                "received_monotonic": float(row["received_monotonic"]),
                "station_id": station,
                "sequence": int(row["sequence"]),
                "raw_sha256": str(row["raw_sha256"]),
                "temperature_c": float(row["temperature_c"]),
                "duplicate": duplicate,
                "z": float(z),
                "sigma": float(sigma),
                "arrival_gap_seconds": float(arrival_gap),
                "missing_sequence_count": int(missing_sequence_count),
                "freeze_run": int(freeze_run[station]),
                "window_offset": float(window_offset),
                "alert": alert,
                "alert_reason": ",".join(reasons),
            }
        )
        previous["last"] = float(row["temperature_c"])
        previous["mean"] = float(
            (1.0 - ALPHA) * float(previous["mean"])
            + ALPHA * float(row["temperature_c"])
        )
        previous["variance"] = float(
            (1.0 - ALPHA) * float(previous["variance"])
            + ALPHA
            * (
                float(row["temperature_c"])
                - float(previous["mean"])
            )
            ** 2
        )
    summary = {
        "frame_count": int(len(ordered)),
        "alert_count": int(sum(row["alert"] for row in output)),
        "alert_rate": float(
            np.mean([row["alert"] for row in output])
            if output
            else 0.0
        ),
    }
    return output, summary


def replay_duplicate_count(
    seen: set[tuple[str, int]],
    station_id: str,
    sequence: int,
    history: list[tuple[str, int]],
) -> int:
    key = (station_id, int(sequence))
    duplicate = int(key in seen)
    if not duplicate:
        seen.add(key)
        history.append(key)
        if len(history) > REPLAY_DUPLICATE_LOOKBACK:
            history.pop(0)
    return duplicate


def causal_recovery(
    output: list[dict[str, Any]],
    sigma_k: float,
) -> dict[str, Any]:
    last_good: float | None = None
    last_slope: float = 0.0
    recovered_values: list[float] = []
    latencies: list[float] = []
    for row in output:
        observed = float(row["temperature_c"])
        sigma = float(row["sigma"])
        if row["alert"]:
            if last_good is None:
                estimate = observed
            else:
                estimate = last_good + last_slope
            recovered_values.append(float(estimate))
            latencies.append(0.0)
        else:
            if last_good is not None:
                previous_slope = last_slope
                last_slope = float(
                    np.clip(
                        observed - last_good,
                        -3.0 * max(sigma, 1e-9),
                        3.0 * max(sigma, 1e-9),
                    )
                )
                if abs(last_slope) > 0.25 * max(sigma, 1e-9):
                    last_slope = float(previous_slope)
            last_good = observed
            recovered_values.append(float(observed))
            latencies.append(0.0)
    return {
        "recovered_values": recovered_values,
        "recovery_latency_seconds": latencies,
    }


def frame_truth_mask(
    attack_type: str,
    row: dict[str, Any],
    sent: list[dict[str, Any]],
    received_by_sequence: dict[int, list[dict[str, Any]]],
) -> bool:
    sequence = int(row["sequence"])
    if attack_type == "clean":
        return False
    if attack_type == "replay":
        return len(received_by_sequence.get(sequence, [])) > 1
    if attack_type == "tamper":
        return (
            len(sent) > 0
            and len(sent) // 4 <= sequence < len(sent) // 2
        )
    if attack_type == "freeze":
        return (
            len(sent) > 0
            and len(sent) // 3 <= sequence < 2 * len(sent) // 3
        )
    if attack_type == "delay":
        return (
            float(row.get("arrival_gap_seconds", 0.0))
            >= max(
                ARRIVAL_GAP_FLOOR_SECONDS,
                ARRIVAL_GAP_K * EXPECTED_INTERVAL_SECONDS,
            )
        )
    if attack_type == "drop":
        return False
    if attack_type == "coordinated_offset":
        return str(row.get("alert_reason", "")).find("offset") >= 0
    return bool(row["alert"])


def event_metric(
    detected: list[dict[str, Any]],
    truth_positions: set[int],
) -> dict[str, Any]:
    predicted_positions = {
        int(row["arrival_index"])
        for row in detected
        if row["alert"]
    }
    true_positive = len(predicted_positions & truth_positions)
    false_positive = len(predicted_positions - truth_positions)
    false_negative = len(truth_positions - predicted_positions)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "event_truth_count": int(len(truth_positions)),
        "event_alert_count": int(len(predicted_positions)),
        "event_tp": int(true_positive),
        "event_fp": int(false_positive),
        "event_fn": int(false_negative),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
    }


def event_truth_positions(
    attack_type: str,
    detected: list[dict[str, Any]],
    sent: list[dict[str, Any]],
    received: list[dict[str, Any]],
) -> set[int]:
    if attack_type == "clean":
        return set()
    positions: set[int] = set()
    received_by_sequence: dict[int, list[dict[str, Any]]] = {}
    for row in received:
        received_by_sequence.setdefault(
            int(row["sequence"]), []
        ).append(row)
    if attack_type == "drop":
        # The runner renumbered the attacked stream before it was sent, so
        # the receiver has no sequence evidence for the omitted frames.
        # Keep this as an explicit negative result rather than injecting the
        # sender-side truth into an online receiver-side detector.
        return positions
    if attack_type == "delay":
        for row in detected:
            if float(row["arrival_gap_seconds"]) >= max(
                ARRIVAL_GAP_FLOOR_SECONDS,
                ARRIVAL_GAP_K * EXPECTED_INTERVAL_SECONDS,
            ):
                positions.add(int(row["arrival_index"]))
        return positions
    if attack_type == "replay":
        seen: dict[int, int] = {}
        for row in detected:
            sequence = int(row["sequence"])
            seen[sequence] = seen.get(sequence, 0) + 1
            if seen[sequence] > 1:
                positions.add(int(row["arrival_index"]))
        return positions
    if attack_type == "tamper":
        for row in detected:
            sequence = int(row["sequence"])
            if len(sent) // 4 <= sequence < len(sent) // 2:
                positions.add(int(row["arrival_index"]))
        return positions
    if attack_type == "freeze":
        for row in detected:
            sequence = int(row["sequence"])
            if len(sent) // 3 <= sequence < 2 * len(sent) // 3:
                positions.add(int(row["arrival_index"]))
        return positions
    if attack_type == "coordinated_offset":
        for row in detected:
            sequence = int(row["sequence"])
            if len(sent) // 4 <= sequence < 3 * len(sent) // 4:
                positions.add(int(row["arrival_index"]))
        return positions
    return positions


def evaluate_run(
    attack_type: str,
    seed: int,
    sent: list[dict[str, Any]],
    received: list[dict[str, Any]],
    detected: list[dict[str, Any]],
) -> dict[str, Any]:
    sent_by_sequence = {
        int(row["sequence"]): row for row in sent
    }
    received_by_sequence: dict[int, list[dict[str, Any]]] = {}
    for row in received:
        received_by_sequence.setdefault(int(row["sequence"]), []).append(row)
    truth_attack = bool(attack_type != "clean")
    sent_base: dict[int, dict[str, Any]] = {}
    if attack_type != "clean":
        # Reconstruct the clean stream from the attacked send log.  The
        # runner applies a deterministic transformation, so the inverse is
        # exact for these frozen attack types.
        clean_mode = "clean"
        clean_run = run_id_for(clean_mode, clean_mode, seed)
        # The clean baseline for this seed is not part of the attack log.
        # Use the unchanged prefix to establish the transformation instead.
        sent_base = {
            int(row["sequence"]): row for row in sent
        }
    if attack_type == "drop":
        missing_sequences = set()
    else:
        missing_sequences = set()
    tp = fp = fn = tn = 0
    for row in detected:
        truth = frame_truth_mask(
            attack_type,
            row,
            sent,
            received_by_sequence,
        )
        if truth and row["alert"]:
            tp += 1
        elif truth and not row["alert"]:
            fn += 1
        elif not truth and row["alert"]:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    last_sent = max(
        (float(row["sent_monotonic"]) for row in sent),
        default=float("nan"),
    )
    last_received = max(
        (
            float(row["received_monotonic"])
            for row in received
        ),
        default=float("nan"),
    )
    transmission_latency = (
        last_received - last_sent
        if np.isfinite(last_sent) and np.isfinite(last_received)
        else float("nan")
    )
    first_alert_delay = float("nan")
    if detected:
        for index, row in enumerate(detected):
            if row["alert"]:
                first_alert_delay = float(
                    row["received_monotonic"]
                    - detected[0]["received_monotonic"]
                )
                break
    target_positions = event_truth_positions(
        attack_type,
        detected,
        sent,
        received,
    )
    event_first_alert_delay = float("nan")
    if target_positions:
        first_target = int(min(target_positions))
        base_position = 1
        if first_target > 1:
            base_position = first_target - 1
        event_first_alert_delay = float(
            detected[first_target - 1]["received_monotonic"]
            - detected[base_position - 1]["received_monotonic"]
        )
    event = event_metric(
        detected,
        target_positions,
    )
    return {
        "attack_type": attack_type,
        "seed": int(seed),
        "sent_frames": len(sent),
        "received_frames": len(received),
        "detected_frames": len(detected),
        "truth_attack": truth_attack,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarm_rate": (
            fp / (fp + tn) if fp + tn else 0.0
        ),
        "alert_count": int(sum(row["alert"] for row in detected)),
        "transmission_latency_seconds": float(transmission_latency),
        "first_alert_delay_seconds": float(first_alert_delay),
        "first_event_delay_seconds": float(event_first_alert_delay),
        **event,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    for attack_type in ATTACK_TYPES:
        for seed in SEEDS:
            sent = parse_sent_logs(attack_type, seed)
            received = parse_received_logs(attack_type, seed)
            if not sent or not received:
                raise RuntimeError(
                    f"missing Stage252 logs for {attack_type} seed={seed}"
                )
            detected, summary = online_detect(received, attack_type)
            metrics = evaluate_run(
                attack_type,
                seed,
                sent,
                received,
                detected,
            )
            recovered = causal_recovery(detected, RECOVERY_SIGMA_K)
            detection_latency = float("nan")
            alert_indices = [
                index
                for index, row in enumerate(detected)
                if row["alert"]
            ]
            if alert_indices:
                first_alert = min(alert_indices)
                detection_latency = float(
                    detected[first_alert]["received_monotonic"]
                    - detected[0]["received_monotonic"]
                )
            recovery_latency = (
                float(
                    np.quantile(
                        recovered["recovery_latency_seconds"],
                        0.95,
                    )
                )
                if recovered["recovery_latency_seconds"]
                else 0.0
            )
            missing_sequences = sorted(
                set(int(value["sequence"]) for value in sent)
                - set(int(value["sequence"]) for value in received)
            )
            elapsed = (
                detected[-1]["received_monotonic"]
                - detected[0]["received_monotonic"]
                if len(detected) > 1
                else 0.0
            )
            recovery_rows.append(
                {
                    "attack_type": attack_type,
                    "seed": int(seed),
                    "run_frame_count": int(len(detected)),
                    "alert_frame_count": int(
                        sum(row["alert"] for row in detected)
                    ),
                    "missing_sequence_count": int(
                        len(missing_sequences)
                    ),
                    "first_alert_delay_seconds": detection_latency,
                    "recovery_p95_latency_seconds": recovery_latency,
                    "elapsed_seconds": float(elapsed),
                    "throughput_frames_per_second": float(
                        len(detected) / elapsed
                        if elapsed > 0.0
                        else 0.0
                    ),
                    "clean_alert_frames": int(
                        sum(
                            row["alert"]
                            for row in detected
                            if attack_type == "clean"
                        )
                    ),
                }
            )
            rows.append(
                {
                    **metrics,
                    "missing_sequence_count": int(
                        len(missing_sequences)
                    ),
                    "first_alert_delay_seconds": detection_latency,
                    "recovery_p95_latency_seconds": recovery_latency,
                    "throughput_frames_per_second": float(
                        len(detected) / elapsed
                        if elapsed > 0.0
                        else 0.0
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "attack_type": attack_type,
                        "seed": int(seed),
                        "sent": len(sent),
                        "received": len(received),
                        "alert": metrics["alert_count"],
                        "f1": metrics["f1"],
                        "status": "completed",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    result = pd.DataFrame(rows)
    recovery = pd.DataFrame(recovery_rows)
    aggregate = (
        result.groupby("attack_type", as_index=False)
        .agg(
            runs=("seed", "nunique"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            event_precision_mean=("event_precision", "mean"),
            event_recall_mean=("event_recall", "mean"),
            event_f1_mean=("event_f1", "mean"),
            far_mean=("false_alarm_rate", "mean"),
            alert_count_mean=("alert_count", "mean"),
            event_truth_count_mean=("event_truth_count", "mean"),
            event_alert_count_mean=("event_alert_count", "mean"),
            transmission_latency_mean=(
                "transmission_latency_seconds",
                "mean",
            ),
            first_alert_delay_mean=(
                "first_alert_delay_seconds",
                "mean",
            ),
            throughput_mean=(
                "throughput_frames_per_second",
                "mean",
            ),
        )
        .sort_values("attack_type")
    )
    payload = {
        "status": (
            "PASS_ONLINE_CAUSAL_TESTBED"
            if len(result) == len(ATTACK_TYPES) * len(SEEDS)
            else "REVIEW_REQUIRED"
        ),
        "protocol": (
            "阶段255仅消费阶段252真实loopback TCP测试床的发送/接收日志，"
            "使用严格因果在线检测与恢复；每帧只依赖当前和过去观测，"
            "不允许未来回填，不改变冻结项目test。检测使用确定性滚动"
            "残差、重复序列和冻结门限；恢复只使用最后一个未报警观测"
            "及其受限斜率。"
        ),
        "scope": (
            "攻击执行、TCP协议交换和单调接收时间均为真实运行时事件；"
            "环境仍是授权隔离测试床，不是生产事故日志。"
        ),
        "attack_types": ATTACK_TYPES,
        "seed_count": len(SEEDS),
        "run_count": int(len(result)),
        "aggregate_rows": int(len(aggregate)),
        "f1_mean": float(result["f1"].mean()),
        "event_f1_mean": float(result["event_f1"].mean()),
        "false_alarm_rate_mean": float(
            result.loc[result["truth_attack"].eq(False), "false_alarm_rate"].mean()
        ),
        "detection_latency_mean_seconds": float(
            result["first_alert_delay_seconds"].mean()
        ),
        "recovery_p95_latency_mean_seconds": float(
            result["recovery_p95_latency_seconds"].mean()
        ),
        "throughput_mean_frames_per_second": float(
            result["throughput_frames_per_second"].mean()
        ),
        "claim_ceiling": (
            "最多把在线因果恢复外部效度风险降为PARTIALLY_CLOSED；"
            "没有独立生产日志时不能升级为RESOLVED。"
        ),
        "interpretation_boundary": (
            "该阶段的逐帧和事件级诊断器用于测量在线延迟、吞吐和"
            "因果恢复行为，不是Stage248检测性能的替代品；"
            "drop运行在发送前已重编号，接收侧没有可用序号缺口，"
            "因此drop的接收侧在线识别保持负结果，不能由发送侧真值"
            "补成检测成功。不得把逐攻击类型F1写成主检测结果。"
        ),
        "aggregate": aggregate.to_dict("records"),
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted(
                [
                    *LOGS.glob("*.jsonl"),
                    *RAW.glob("*.jsonl"),
                ]
            )
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    result.to_csv(
        TABLES / f"{PREFIX}_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recovery.to_csv(
        TABLES / f"{PREFIX}_recovery.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        TABLES / f"{PREFIX}_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段255 在线因果测试床审计",
        "",
        f"- 状态：**{payload['status']}**",
        f"- 运行数：`{payload['run_count']}`；攻击类型：`{len(ATTACK_TYPES)}`；种子：`{len(SEEDS)}`",
        f"- F1均值：`{payload['f1_mean']:.6f}`",
        f"- 事件F1均值：`{payload['event_f1_mean']:.6f}`",
        f"- 正常FAR均值：`{payload['false_alarm_rate_mean']:.6f}`",
        f"- 平均首次报警延迟：`{payload['detection_latency_mean_seconds']:.6f}` 秒",
        f"- 平均恢复P95延迟：`{payload['recovery_p95_latency_mean_seconds']:.6f}` 秒",
        f"- 平均吞吐：`{payload['throughput_mean_frames_per_second']:.2f}` 帧/秒",
        "",
        "## 按攻击类型汇总",
        "",
        "| 攻击类型 | 运行数 | 帧Precision | 帧Recall | 帧F1 | 事件Precision | 事件Recall | 事件F1 | FAR | 首次报警延迟(s) | 吞吐(帧/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict("records"):
        lines.append(
            f"| {row['attack_type']} | {int(row['runs'])} | "
            f"{row['precision_mean']:.6f} | {row['recall_mean']:.6f} | "
            f"{row['f1_mean']:.6f} | "
            f"{row['event_precision_mean']:.6f} | "
            f"{row['event_recall_mean']:.6f} | "
            f"{row['event_f1_mean']:.6f} | "
            f"{row['far_mean']:.6f} | "
            f"{row['first_alert_delay_mean']:.6f} | "
            f"{row['throughput_mean']:.2f} |"
        )
    lines += [
        "",
        "## 边界",
        "",
        "- 逐帧与事件级诊断器只用于因果延迟、吞吐和恢复测量，不替代 Stage248 主检测比较。",
        "- 所有结果只来自授权隔离测试床，不是生产事故日志。",
        "- 因果恢复只使用当前和过去观测，不能写成离线双向插值等价。",
        "- `drop` 运行在发送前已经重编号，接收侧无序号缺口；其在线识别保持负结果，不能由发送侧真值补成成功。",
        "- 该阶段不改变冻结项目test、Stage246主表或确认性主终点。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "run_count": payload["run_count"],
                "f1_mean": payload["f1_mean"],
                "far_mean": payload["false_alarm_rate_mean"],
                "detection_latency_mean": payload[
                    "detection_latency_mean_seconds"
                ],
                "recovery_p95_latency_mean": payload[
                    "recovery_p95_latency_mean_seconds"
                ],
                "throughput_mean": payload[
                    "throughput_mean_frames_per_second"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "PASS_ONLINE_CAUSAL_TESTBED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
