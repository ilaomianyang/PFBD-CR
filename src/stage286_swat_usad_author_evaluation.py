from __future__ import annotations

"""Stage 286: paper-faithful USAD evaluation on public SWaT.

The model and training losses come from the fixed-commit author code in
``vendor/usad.py``. The data split, window length, optimizer, epochs, FAR
calibration, and event metrics come from Stage285's frozen protocol.

The test file is read only after the model and threshold are fixed. This is
a public benchmark evaluation, not a production incident-log evaluation.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
VENDOR = ROOT / "vendor"
PROTOCOL = TABLES / "stage285_swat_usad_protocol_preregistration.json"
PREFIX = "stage286_swat_usad_author_evaluation"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import sha256_file  # noqa: E402


def _verify_frozen_inputs(protocol: dict) -> None:
    mismatches = []
    for relative_path, expected in protocol["inputs"].items():
        path = ROOT / relative_path
        actual = {
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        if actual != {
            "bytes": int(expected["bytes"]),
            "sha256": str(expected["sha256"]),
        }:
            mismatches.append(
                {
                    "path": relative_path,
                    "expected": expected,
                    "actual": actual,
                }
            )
    if mismatches:
        raise RuntimeError(
            "frozen input hash mismatch: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


def _author_feature_frame(
    path: Path, *, separator: str | None = None
) -> tuple[pd.DataFrame, np.ndarray | None]:
    read_options = {"dtype": str}
    if separator is not None:
        read_options.update({"sep": separator, "low_memory": False})
    frame = pd.read_csv(path, **read_options)
    labels = None
    if "Normal/Attack" in frame.columns:
        labels = (
            frame["Normal/Attack"]
            .astype(str)
            .str.strip()
            .ne("Normal")
            .to_numpy()
        )
    frame = frame.drop(columns=["Timestamp", "Normal/Attack"])
    for column in frame.columns:
        frame[column] = (
            frame[column]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )
    return frame, labels


def _load_author_module():
    utils_spec = importlib.util.spec_from_file_location(
        "usad_author_utils", VENDOR / "utils.py"
    )
    utils_module = importlib.util.module_from_spec(utils_spec)
    assert utils_spec and utils_spec.loader
    utils_spec.loader.exec_module(utils_module)
    sys.modules["utils"] = utils_module
    usad_spec = importlib.util.spec_from_file_location(
        "usad_author_model", VENDOR / "usad.py"
    )
    usad_module = importlib.util.module_from_spec(usad_spec)
    assert usad_spec and usad_spec.loader
    usad_spec.loader.exec_module(usad_module)
    return usad_module


def _windows(values: np.ndarray, length: int) -> np.ndarray:
    return values[
        np.arange(length)[None, :]
        + np.arange(values.shape[0] - length)[:, None]
    ]


def _event_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    starts = np.flatnonzero(labels & ~np.r_[False, labels[:-1]])
    ends = np.flatnonzero(labels & ~np.r_[labels[1:], False])
    events = []
    for start, end in zip(starts, ends):
        events.append(bool(predicted[start : end + 1].any()))
    positive = int(labels.sum())
    return {
        "events": int(len(events)),
        "detected_events": int(sum(events)),
        "event_coverage": float(np.mean(events)) if events else float("nan"),
        "row_recall": float(predicted[labels].mean()) if positive else float("nan"),
    }


def _metrics(labels: np.ndarray, predicted: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    finite = np.isfinite(scores)
    return {
        "rows": int(len(labels)),
        "positive_rows": int(labels.sum()),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "false_alarm_rate": float(
            predicted[~labels].mean() if (~labels).any() else float("nan")
        ),
        "roc_auc": float(roc_auc_score(labels[finite], scores[finite]))
        if finite.any() and labels[finite].any() and (~labels[finite]).any()
        else float("nan"),
        "auprc": float(
            average_precision_score(labels[finite], scores[finite])
        )
        if finite.any() and labels[finite].any() and (~labels[finite]).any()
        else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_PUBLIC_SWAT_USAD_PROTOCOL":
        raise RuntimeError("Stage285 protocol is not frozen")
    epochs = int(
        protocol["model"]["epochs"] if args.epochs is None else args.epochs
    )
    if epochs != int(protocol["model"]["epochs"]):
        raise RuntimeError("epoch override is not allowed after protocol freeze")

    _verify_frozen_inputs(protocol)
    author = _load_author_module()
    normal, _ = _author_feature_frame(
        ROOT / protocol["dataset_files"]["normal"]
    )
    scaler = MinMaxScaler().fit(normal.values)
    normal_scaled = scaler.transform(normal.values)

    attack, attack_labels = _author_feature_frame(
        ROOT / protocol["dataset_files"]["attack"], separator=";"
    )
    assert attack_labels is not None
    attack_scaled = scaler.transform(attack.values)

    length = int(protocol["window"]["length"])
    normal_windows = _windows(normal_scaled, length)
    attack_windows = _windows(attack_scaled, length)
    attack_window_labels = np.array(
        [
            bool(labels.any())
            for labels in _windows(attack_labels.astype(np.int8), length)
        ],
        dtype=bool,
    )
    split = int(np.floor(0.8 * normal_windows.shape[0]))
    train_windows = normal_windows[:split]
    valid_windows = normal_windows[split:]
    flat_size = int(normal_windows.shape[1] * normal_windows.shape[2])
    latent_size = int(normal_windows.shape[1] * 100)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = author.UsadModel(flat_size, latent_size)
    model = author.to_device(model, device)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(train_windows).float().view(
                (train_windows.shape[0], flat_size)
            )
        ),
        batch_size=int(protocol["model"]["batch_size"]),
        shuffle=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(valid_windows).float().view(
                (valid_windows.shape[0], flat_size)
            )
        ),
        batch_size=int(protocol["model"]["batch_size"]),
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(attack_windows).float().view(
                (attack_windows.shape[0], flat_size)
            )
        ),
        batch_size=int(protocol["model"]["batch_size"]),
        shuffle=False,
    )

    history = author.training(epochs, model, train_loader, valid_loader)
    valid_scores = torch.cat(author.testing(model, valid_loader)).cpu().numpy()
    test_scores = torch.cat(author.testing(model, test_loader)).cpu().numpy()
    threshold = float(
        np.quantile(
            valid_scores[np.isfinite(valid_scores)],
            1.0 - float(protocol["threshold"]["primary_far"]),
        )
    )
    valid_pred = valid_scores >= threshold
    test_pred = test_scores >= threshold
    result = {
        "status": "PASS_PUBLIC_SWAT_USAD_AUTHOR_EVALUATION",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_commit": protocol["source_commit"],
        "device": str(device),
        "epochs": epochs,
        "window_length": length,
        "normal_rows": int(normal_scaled.shape[0]),
        "attack_rows": int(attack_scaled.shape[0]),
        "feature_columns": int(normal_scaled.shape[1]),
        "attack_positive_rows": int(attack_labels.sum()),
        "train_windows": int(train_windows.shape[0]),
        "valid_windows": int(valid_windows.shape[0]),
        "test_windows": int(attack_windows.shape[0]),
        "threshold": threshold,
        "valid_false_alarm_rate": float(valid_pred.mean()),
        "test": {
            **_metrics(attack_window_labels, test_pred, test_scores),
            **_event_metrics(attack_window_labels, test_pred),
        },
        "history_last": history[-1] if history else None,
        "input_sha256": {
            "protocol": sha256_file(PROTOCOL),
            "normal": sha256_file(
                ROOT / protocol["dataset_files"]["normal"]
            ),
            "attack": sha256_file(
                ROOT / protocol["dataset_files"]["attack"]
            ),
            "usad_py": sha256_file(VENDOR / "usad.py"),
            "utils_py": sha256_file(VENDOR / "utils.py"),
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_input_hash_verification": "all_protocol_inputs_matched",
        "evidence_boundary": (
            "Public SWaT benchmark evaluation with author USAD code and "
            "author-linked data; the test-label ROC threshold in the author "
            "notebook is replaced by the frozen valid-normal FAR=0.01 "
            "threshold; not a production incident-log result."
        ),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([result["test"]]).to_csv(
        TABLES / f"{PREFIX}_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {"window_index": np.arange(len(test_scores)), "score": test_scores,
         "label": attack_window_labels, "predicted": test_pred}
    ).to_csv(
        TABLES / f"{PREFIX}_test_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "f1": result["test"]["f1"],
                "event_coverage": result["test"]["event_coverage"],
                "false_alarm_rate": result["test"]["false_alarm_rate"],
                "auprc": result["test"]["auprc"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
