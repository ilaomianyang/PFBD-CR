from __future__ import annotations

"""Stage 285: freeze the public-SWaT USAD evaluation protocol.

This stage is intentionally run before any USAD training or test scoring.
It records the exact author source and data hashes, the fixed split and
window construction, the author loss/training recipe, the FAR calibration
rule, and the event/row evaluation contract.  It does not train a model,
inspect test scores, or choose a threshold from test labels.
"""

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
VENDOR = ROOT / "vendor"
RAW = ROOT / "raw" / "public_ics_cps"
PREFIX = "stage285_swat_usad_protocol_preregistration"

SOURCE_FILES = [
    "tables/stage286_swat_usad_author_evaluation.json",
    "tables/stage286_swat_usad_author_evaluation_test_metrics.csv",
    "tables/stage286_swat_usad_author_evaluation_test_scores.csv",
    "src/stage286_swat_usad_author_evaluation.py",
    "vendor/USAD.ipynb",
    "vendor/usad.py",
    "vendor/utils.py",
    "vendor/LICENSE",
    "raw/public_ics_cps/SWaT_Dataset_Normal_v1.csv",
    "raw/public_ics_cps/SWaT_Dataset_Attack_v0.csv",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    missing = [name for name in SOURCE_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")

    protocol = {
        "status": "FROZEN_PUBLIC_SWAT_USAD_PROTOCOL",
        "protocol_id": "TMQC-PUBLIC-SWAT-USAD-20260921-v1.2",
        "source_commit": "e25af45c8e1c32783aed94fc7e5ab85effef2b1a",
        "source_repository": "https://github.com/manigalati/usad",
        "dataset": "SWaT",
        "dataset_files": {
            "normal": "raw/public_ics_cps/SWaT_Dataset_Normal_v1.csv",
            "attack": "raw/public_ics_cps/SWaT_Dataset_Attack_v0.csv",
        },
        "normalization": {
            "fit_scope": "normal_file_only",
            "method": "MinMaxScaler",
            "drop_columns": ["Timestamp", "Normal/Attack"],
            "numeric_parsing": (
                "convert each feature column to string, replace comma with "
                "dot, then cast to float before scaling"
            ),
        },
        "window": {
            "length": 12,
            "normal_windows": "normal_values[i:i+12]",
            "attack_windows": "attack_values[i:i+12]",
            "window_label_rule": "1 if any row label in window else 0",
        },
        "split": {
            "normal_train_fraction": 0.8,
            "normal_valid_fraction": 0.2,
            "train_rows": "normal windows [: floor(0.8 * n)]",
            "valid_rows": "normal windows [floor(0.8 * n): n]",
            "test_rows": "all attack windows",
            "test_labels": "row labels from Normal/Attack; window label any-positive",
        },
        "model": {
            "implementation": "author code usad.py",
            "latent_size": "window_length * 100",
            "encoder": "author three-layer linear encoder with ReLU",
            "decoders": "author two decoders with sigmoid output",
            "loss": "author training_step loss1/loss2",
            "optimizer": "Adam",
            "batch_size": 7919,
            "epochs": 100,
        },
        "threshold": {
            "source": "valid_normal_windows_only",
            "rule": "empirical quantile at 1 - target_far",
            "primary_far": 0.01,
            "test_labels_used_for_threshold": False,
            "adaptation": (
                "the author notebook selects an ROC equal-error threshold "
                "using test labels; this frozen evaluation replaces that "
                "leaky selection with a valid-normal FAR=0.01 threshold"
            ),
            "score": "author testing score alpha*mean((batch-w1)^2)+beta*mean((batch-w2)^2)",
            "alpha": 0.5,
            "beta": 0.5,
        },
        "events": {
            "definition": "maximal consecutive positive-label rows",
            "event_coverage": "fraction of events with at least one predicted row",
            "row_recall": "fraction of positive-label rows predicted",
            "metrics": [
                "f1",
                "precision",
                "recall",
                "row_recall",
                "event_coverage",
                "false_alarm_rate",
                "auprc",
            ],
        },
        "evidence_boundary": (
            "This is a paper-faithful author-code evaluation on the public "
            "SWaT files linked by the author notebook. It is not a "
            "production incident log and must not be described as real "
            "production attack recall."
        ),
        "inputs": {
            name: {
                "bytes": (ROOT / name).stat().st_size,
                "sha256": sha256_file(ROOT / name),
            }
            for name in SOURCE_FILES
        },
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage 285 公开 SWaT USAD 预注册协议",
        "",
        f"- 状态：`{protocol['status']}`",
        f"- 协议：`{protocol['protocol_id']}`",
        f"- 作者提交：`{protocol['source_commit']}`",
        "- 训练/验证/测试：作者 normal 文件的 80/20 窗口切分；attack 文件仅用于 test 评价。",
        "- 窗口：12；模型：作者 `usad.py`；优化器：Adam；batch：7919；epoch：100。",
        "- 数值解析：作者 notebook 的逗号小数点替换后转 `float`，再做 MinMax 缩放。",
        "- 阈值：仅由 valid-normal 窗口按 `FAR=0.01` 分位数校准；替代作者 notebook 使用 test 标签的 ROC 等误报点选择，避免 test 泄漏。",
        "- 事件：连续正标签行段；同时报告 F1、precision、recall、row recall、event coverage、FAR 和 AUPRC。",
        "",
        "## 输入哈希",
        "",
    ]
    for name, metadata in protocol["inputs"].items():
        lines.append(
            f"- `{name}`：{metadata['bytes']} bytes；SHA256 `{metadata['sha256']}`"
        )
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "protocol_id": protocol["protocol_id"],
                "inputs": len(protocol["inputs"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
