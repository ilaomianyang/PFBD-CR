from __future__ import annotations

"""Stage 288: resource-only audit for the public SWaT USAD baseline.

This stage does not read the SWaT attack test file, does not change the
Stage285 protocol, and does not replace the Stage286 performance result.
It reruns the author USAD training recipe on train-normal windows only and
measures model size, training wall time, peak system/GPU memory, and
valid-normal inference latency. The run is a deployment-cost measurement,
not a new detection-performance result.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PROTOCOL = TABLES / "stage285_swat_usad_protocol_preregistration.json"
PREFIX = "stage288_swat_usad_resource_audit"
RESOURCE_SEED = 20260921

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage269_deployment_complexity_audit as s269  # noqa: E402
import stage286_swat_usad_author_evaluation as s286  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import sha256_file  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(values: np.ndarray, flat_size: int, batch_size: int):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(values)
            .float()
            .view((values.shape[0], flat_size))
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def _measure_inference(
    author,
    model,
    loader,
    repeats: int,
    rows: int,
) -> dict[str, float | int]:
    timings = []
    for _ in range(max(1, repeats)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        _ = author.testing(model, loader)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    mean_seconds = float(np.mean(timings))
    return {
        "repeats": int(max(1, repeats)),
        "rows": int(rows),
        "mean_seconds": mean_seconds,
        "min_seconds": float(np.min(timings)),
        "max_seconds": float(np.max(timings)),
        "mean_ms": mean_seconds * 1000.0,
        "per_row_ms": mean_seconds * 1000.0 / max(1, rows),
        "throughput_rows_per_second": float(rows / max(mean_seconds, 1e-12)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latency-repeats", type=int, default=5)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_PUBLIC_SWAT_USAD_PROTOCOL":
        raise RuntimeError("Stage285 protocol is not frozen")
    s286._verify_frozen_inputs(protocol)
    author = s286._load_author_module()
    _set_seed(RESOURCE_SEED)

    normal, _ = s286._author_feature_frame(
        ROOT / protocol["dataset_files"]["normal"]
    )
    scaler = MinMaxScaler().fit(normal.values)
    normal_scaled = scaler.transform(normal.values)

    length = int(protocol["window"]["length"])
    normal_windows = s286._windows(normal_scaled, length)
    split = int(np.floor(0.8 * normal_windows.shape[0]))
    train_windows = normal_windows[:split]
    valid_windows = normal_windows[split:]
    flat_size = int(normal_windows.shape[1] * normal_windows.shape[2])
    latent_size = int(normal_windows.shape[1] * 100)
    batch_size = int(protocol["model"]["batch_size"])
    epochs = int(protocol["model"]["epochs"])

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = author.UsadModel(flat_size, latent_size)
    model = author.to_device(model, device)
    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    trainable_parameter_count = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )

    train_loader = _loader(train_windows, flat_size, batch_size)
    valid_loader = _loader(valid_windows, flat_size, batch_size)

    reset_ok = s269._reset_peak_working_set()
    started = time.perf_counter()
    history = author.training(epochs, model, train_loader, valid_loader)
    training_seconds = time.perf_counter() - started
    peak_bytes = s269._peak_working_set_bytes()
    inference = _measure_inference(
        author,
        model,
        valid_loader,
        int(args.latency_repeats),
        int(valid_windows.shape[0]),
    )
    cuda_peak_allocated_mb = None
    cuda_peak_reserved_mb = None
    if device.type == "cuda":
        cuda_peak_allocated_mb = float(
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        )
        cuda_peak_reserved_mb = float(
            torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
        )

    payload = {
        "status": "PASS_SWAT_USAD_RESOURCE_AUDIT",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_commit": protocol["source_commit"],
        "resource_seed": RESOURCE_SEED,
        "device": str(device),
        "data_boundary": (
            "Only the author-linked normal file is read. The attack test "
            "file and test labels are not read in this stage."
        ),
        "model": {
            "implementation": "vendor/usad.py",
            "flat_input_size": flat_size,
            "latent_size": latent_size,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "train_windows": int(train_windows.shape[0]),
            "valid_windows": int(valid_windows.shape[0]),
            "wall_seconds": float(training_seconds),
            "wall_minutes": float(training_seconds / 60.0),
            "peak_working_set_mb": (
                None
                if peak_bytes is None
                else float(peak_bytes / (1024.0 * 1024.0))
            ),
            "peak_working_set_reset_succeeded": bool(reset_ok),
            "cuda_peak_allocated_mb": cuda_peak_allocated_mb,
            "cuda_peak_reserved_mb": cuda_peak_reserved_mb,
            "history_last": history[-1] if history else None,
            "failure_state": "completed",
        },
        "valid_normal_inference": inference,
        "claim_ceiling": (
            "Resource evidence only. This run does not replace Stage286's "
            "frozen detection metrics and must not be reported as a new "
            "performance result."
        ),
        "input_sha256": {
            "protocol": sha256_file(PROTOCOL),
            "normal": sha256_file(
                ROOT / protocol["dataset_files"]["normal"]
            ),
            "usad_py": sha256_file(ROOT / "vendor" / "usad.py"),
            "utils_py": sha256_file(ROOT / "vendor" / "utils.py"),
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "method": "usad_author_code",
                "display_name": "USAD author code on public SWaT",
                "family": "public_ics_cps_anomaly_detection",
                "evidence_layer": "independent_public_benchmark",
                "model_structure": (
                    "author three-layer encoder with two sigmoid decoders"
                ),
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "fit_rows": int(train_windows.shape[0]),
                "epochs": epochs,
                "batch_size": batch_size,
                "fit_seconds": float(training_seconds),
                "peak_memory_mb": (
                    None
                    if peak_bytes is None
                    else float(peak_bytes / (1024.0 * 1024.0))
                ),
                "cuda_peak_allocated_mb": cuda_peak_allocated_mb,
                "cuda_peak_reserved_mb": cuda_peak_reserved_mb,
                "inference_mean_ms": inference["mean_ms"],
                "inference_per_row_ms": inference["per_row_ms"],
                "throughput_rows_per_second": (
                    inference["throughput_rows_per_second"]
                ),
                "device": str(device),
                "failure_state": "completed",
                "same_endpoint_as_trust": False,
                "evidence_status": (
                    "measured resource-only run; independent public SWaT "
                    "benchmark with a different endpoint from TrustMeteo-QC"
                ),
            }
        ]
    ).to_csv(
        TABLES / f"{PREFIX}_methods.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# Stage 288 公开 SWaT USAD 资源审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 协议：`{payload['protocol_id']}`",
        f"- 设备：`{payload['device']}`",
        f"- 参数量：`{parameter_count}`",
        f"- 训练窗口：`{train_windows.shape[0]}`；epoch：`{epochs}`；batch：`{batch_size}`",
        f"- 训练墙钟时间：`{training_seconds:.3f}` 秒",
        f"- 系统峰值工作集：`{payload['training']['peak_working_set_mb']}` MB",
        f"- CUDA 峰值已分配：`{cuda_peak_allocated_mb}` MB",
        f"- CUDA 峰值保留：`{cuda_peak_reserved_mb}` MB",
        f"- valid-normal 推理吞吐：`{inference['throughput_rows_per_second']:.3f}` 行/秒",
        "",
        "## 边界",
        "",
        "- 本阶段只读取作者链接 normal 文件，不读取 attack test 文件或 test 标签。",
        "- 本阶段是资源测量，不替换 Stage286 的检测性能结果。",
        "- 该结果仍是公开 SWaT 基准，不是生产事故日志，也不是 TrustMeteo-QC 同契约比较。",
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "parameter_count": parameter_count,
                "training_seconds": training_seconds,
                "peak_working_set_mb": payload["training"][
                    "peak_working_set_mb"
                ],
                "throughput_rows_per_second": inference[
                    "throughput_rows_per_second"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
