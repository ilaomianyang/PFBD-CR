from __future__ import annotations

"""Stage 261: causal-versus-offline recovery contrast on Stage252 logs.

Stage255 established a strictly causal online detector and recovery loop.
This stage adds the missing reviewer-facing comparison: for the same
attacked runtime streams, compute

* online causal recovery error, which may use only the current and past
  received frames; and
* an offline bidirectional interpolation reference, which may use future
  frames.

The clean stream is used only as evaluation truth.  It is never supplied
to the online causal path.  The purpose is not to make the two methods
look equal; it is to quantify the causal penalty and keep the claim
boundary explicit.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage261_causal_offline_recovery_contrast"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage255_online_causal_testbed as s255  # noqa: E402


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_truth_lookup(
    attack_type: str,
    seed: int,
) -> dict[tuple[str, int], float]:
    clean = s255.parse_received_logs("clean", seed)
    return {
        (str(row["station_id"]), int(row["sequence"])): float(
            row["temperature_c"]
        )
        for row in clean
    }


def causal_estimate(
    detected: list[dict[str, object]],
) -> np.ndarray:
    recovered = s255.causal_recovery(
        detected,
        s255.RECOVERY_SIGMA_K,
    )
    return np.asarray(recovered["recovered_values"], dtype=float)


def offline_interpolation_estimate(
    values: np.ndarray,
    alerts: np.ndarray,
) -> np.ndarray:
    """Bidirectional linear interpolation over contiguous alert runs.

    This is an oracle-style offline reference: it may read values after the
    alert run.  It is not a deployable causal detector and must never be
    reported as online performance.
    """
    output = np.asarray(values, dtype=float).copy()
    index = 0
    count = len(output)
    while index < count:
        if not alerts[index]:
            index += 1
            continue
        start = index
        while index < count and alerts[index]:
            index += 1
        end = index
        left = start - 1
        right = end
        left_value = output[left] if left >= 0 else np.nan
        right_value = output[right] if right < count else np.nan
        if np.isfinite(left_value) and np.isfinite(right_value):
            span = right - left
            for position in range(start, end):
                weight = (position - left) / span
                output[position] = (
                    (1.0 - weight) * left_value
                    + weight * right_value
                )
        elif np.isfinite(left_value):
            output[start:end] = left_value
        elif np.isfinite(right_value):
            output[start:end] = right_value
    return output


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows: list[dict[str, object]] = []
    for attack_type in s255.ATTACK_TYPES:
        if attack_type == "clean":
            continue
        for seed in s255.SEEDS:
            received = s255.parse_received_logs(attack_type, int(seed))
            if not received:
                raise RuntimeError(
                    f"missing Stage252 logs for {attack_type}/{seed}"
                )
            detected, _ = s255.online_detect(received, attack_type)
            ordered = sorted(
                detected,
                key=lambda row: int(row["arrival_index"]),
            )
            values = np.asarray(
                [float(row["temperature_c"]) for row in ordered],
                dtype=float,
            )
            alerts = np.asarray(
                [bool(row["alert"]) for row in ordered],
                dtype=bool,
            )
            truth_map = clean_truth_lookup(attack_type, int(seed))
            truth = np.asarray(
                [
                    truth_map.get(
                        (
                            str(row["station_id"]),
                            int(row["sequence"]),
                        ),
                        np.nan,
                    )
                    for row in ordered
                ],
                dtype=float,
            )
            if not alerts.any():
                rows.append(
                    {
                        "attack_type": attack_type,
                        "seed": int(seed),
                        "frames": int(len(ordered)),
                        "alert_frames": 0,
                        "truth_available_frames": int(
                            np.isfinite(truth).sum()
                        ),
                        "online_mae": None,
                        "offline_mae": None,
                        "causal_penalty_mae": None,
                        "online_rmse": None,
                        "offline_rmse": None,
                        "causal_penalty_rmse": None,
                        "online_equals_offline": False,
                    }
                )
                continue
            online = causal_estimate(ordered)
            offline = offline_interpolation_estimate(values, alerts)
            evaluable = np.isfinite(truth) & alerts
            if not evaluable.any():
                evaluable = np.isfinite(truth)
            online_error = online[evaluable] - truth[evaluable]
            offline_error = offline[evaluable] - truth[evaluable]
            online_mae = float(np.mean(np.abs(online_error)))
            offline_mae = float(np.mean(np.abs(offline_error)))
            online_rmse = float(np.sqrt(np.mean(online_error**2)))
            offline_rmse = float(np.sqrt(np.mean(offline_error**2)))
            rows.append(
                {
                    "attack_type": attack_type,
                    "seed": int(seed),
                    "frames": int(len(ordered)),
                    "alert_frames": int(alerts.sum()),
                    "truth_available_frames": int(
                        np.isfinite(truth).sum()
                    ),
                    "online_mae": online_mae,
                    "offline_mae": offline_mae,
                    "causal_penalty_mae": online_mae - offline_mae,
                    "online_rmse": online_rmse,
                    "offline_rmse": offline_rmse,
                    "causal_penalty_rmse": online_rmse - offline_rmse,
                    "online_equals_offline": bool(
                        np.isclose(online_mae, offline_mae)
                        and np.isclose(online_rmse, offline_rmse)
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "attack_type": attack_type,
                        "seed": int(seed),
                        "alert_frames": int(alerts.sum()),
                        "online_mae": online_mae,
                        "offline_mae": offline_mae,
                        "status": "completed",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    detail = pd.DataFrame(rows)
    aggregate = (
        detail.groupby("attack_type", as_index=False)
        .agg(
            runs=("seed", "nunique"),
            frames=("frames", "sum"),
            alert_frames=("alert_frames", "sum"),
            online_mae=("online_mae", "mean"),
            offline_mae=("offline_mae", "mean"),
            causal_penalty_mae=("causal_penalty_mae", "mean"),
            online_rmse=("online_rmse", "mean"),
            offline_rmse=("offline_rmse", "mean"),
            causal_penalty_rmse=("causal_penalty_rmse", "mean"),
            online_equals_offline=("online_equals_offline", "any"),
        )
        .sort_values("attack_type")
    )
    valid = detail["online_mae"].notna()
    payload = {
        "status": "PASS_CAUSAL_OFFLINE_CONTRAST",
        "protocol": (
            "阶段261复用阶段252原始测试床日志和阶段255严格因果检测"
            "输出，在相同攻击事件上比较在线因果恢复与离线双向线性"
            "插值。clean流只作为评价真值，不进入在线因果路径；离线"
            "对照可使用未来帧，因此只是不可部署的参照上界。"
        ),
        "scope": (
            "在线路径只使用当前和过去观测；离线双向插值使用未来帧。"
            "两者不是同一信息集，不能写成在线等同离线。"
        ),
        "attack_types": [
            value
            for value in s255.ATTACK_TYPES
            if value != "clean"
        ],
        "seed_count": len(s255.SEEDS),
        "run_count": int(len(detail)),
        "evaluable_run_count": int(valid.sum()),
        "online_mae_mean": float(detail.loc[valid, "online_mae"].mean()),
        "offline_mae_mean": float(
            detail.loc[valid, "offline_mae"].mean()
        ),
        "causal_penalty_mae_mean": float(
            detail.loc[valid, "causal_penalty_mae"].mean()
        ),
        "online_rmse_mean": float(
            detail.loc[valid, "online_rmse"].mean()
        ),
        "offline_rmse_mean": float(
            detail.loc[valid, "offline_rmse"].mean()
        ),
        "causal_penalty_rmse_mean": float(
            detail.loc[valid, "causal_penalty_rmse"].mean()
        ),
        "online_equals_offline_any": bool(
            detail["online_equals_offline"].any()
        ),
        "aggregate": aggregate.to_dict("records"),
        "claim_ceiling": "BOUNDED",
        "allowed_claim": (
            "Report the measured causal penalty and online latency/safety "
            "trade-off separately from offline interpolation."
        ),
        "forbidden_claim": (
            "Do not claim online causal recovery equals offline "
            "bidirectional interpolation."
        ),
        "input_hashes": {
            "stage255_online_causal_testbed.json": sha256_file(
                TABLES / "stage255_online_causal_testbed.json"
            ),
            "stage255_online_causal_testbed_runs.csv": sha256_file(
                TABLES / "stage255_online_causal_testbed_runs.csv"
            ),
            "stage255_online_causal_testbed_recovery.csv": sha256_file(
                TABLES / "stage255_online_causal_testbed_recovery.csv"
            ),
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    detail.to_csv(
        TABLES / f"{PREFIX}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        TABLES / f"{PREFIX}_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 阶段261 在线因果与离线双向恢复对照",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 攻击类型：`{', '.join(payload['attack_types'])}`",
        f"- 运行数：`{payload['run_count']}`",
        f"- 可评价运行数：`{payload['evaluable_run_count']}`",
        f"- 在线MAE均值：`{payload['online_mae_mean']:.6f}`",
        f"- 离线MAE均值：`{payload['offline_mae_mean']:.6f}`",
        f"- 因果惩罚MAE均值：`{payload['causal_penalty_mae_mean']:.6f}`",
        f"- 在线RMSE均值：`{payload['online_rmse_mean']:.6f}`",
        f"- 离线RMSE均值：`{payload['offline_rmse_mean']:.6f}`",
        f"- 因果惩罚RMSE均值：`{payload['causal_penalty_rmse_mean']:.6f}`",
        "",
        "## 边界",
        "",
        "- 在线路径只使用当前和过去观测。",
        "- 离线双向插值使用未来帧，只是参照上界。",
        "- 两者信息集不同，不能写成在线等同离线。",
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
                "online_mae_mean": payload["online_mae_mean"],
                "offline_mae_mean": payload["offline_mae_mean"],
                "causal_penalty_mae_mean": payload[
                    "causal_penalty_mae_mean"
                ],
                "online_equals_offline_any": payload[
                    "online_equals_offline_any"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
