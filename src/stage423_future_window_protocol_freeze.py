from __future__ import annotations

"""Stage 423: freeze a future-window confirmatory protocol before data access.

No observation values are downloaded or read.  The package is intentionally
pending the availability of an official, unvisited October 2026 window.
"""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "confirmatory_package" / "simulated_attack_v6" / "protocol_freeze.json"
PACKAGE = ROOT / "confirmatory_package" / "simulated_attack_v9_oct2026"
PROTOCOL = PACKAGE / "protocol_freeze.json"
REPORT = ROOT / "reports" / "stage423_future_window_protocol_freeze.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    base = json.loads(V6.read_text(encoding="utf-8"))
    out = copy.deepcopy(base)
    out.update(
        {
            "status": "CONFIRMATORY_PROTOCOL_FROZEN_PENDING_NEW_OFFICIAL_WINDOW",
            "protocol_id": "TMQC-SIMULATED-ATTACK-V9-OCT2026-20260925-v1.2",
            "supersedes_protocol_id": "TMQC-SIMULATED-ATTACK-V9-OCT2026-20260925-v1.1",
            "revision_reason": (
                "修复冻结 JSON 中执行入口声明仍指向 V6 的元数据错误，并将独立统计验收"
                "纳入续跑顺序；不改变 provider、时间窗、攻击参数、方法集合、阈值、"
                "统计方法或任何性能结果。"
            ),
            "frozen_at_utc": "2026-09-25T00:00:00Z",
            "purpose": (
                "在未访问的官方 2026 年 10 月时间窗上，对攻击强度、持续时间、事件长度和"
                "未见 coordinated drift 组合进行确认性模拟攻击评价；所有攻击均为透明受控注入，"
                "不代表真实生产攻击日志。"
            ),
            "window": {
                "test_start_utc": "2026-10-01T00:00:00Z",
                "test_end_utc": "2026-10-31T23:00:00Z",
                "official_values_accessed_before_freeze": False,
                "window_status": "PENDING_OFFICIAL_ARCHIVE_AND_NETWORK_ACCESS",
                "selection_rule": (
                    "使用四个冻结 provider 的既有站点协议和固定站点顺序；不新增站点，"
                    "不根据 2026 年 10 月观测覆盖率事后选站。"
                ),
            },
            "data_gate": {
                "required_before_evaluation": [
                    "官方端点可访问",
                    "测试窗双边界过滤通过",
                    "每站点时间戳均在 2026-10-01 至 2026-10-31 UTC",
                    "每站点覆盖率审计通过",
                    "原始文件和规范化数据 SHA256 已保存",
                ],
                "if_unavailable": "NOT_EVALUATED; do not substitute 2025 or previously viewed windows",
            },
            "primary_far": 0.0065,
            "primary_endpoint": "event_coverage",
            "secondary_endpoints": [
                "f1",
                "precision",
                "row_recall",
                "auprc",
                "false_alarm_rate",
                "detection_delay",
            ],
            "bootstrap": {
                "unit": "provider -> station -> seed",
                "iterations": 10000,
                "seed": 20260924,
            },
            "multiplicity": "Benjamini-Hochberg over provider x factor cell x baseline comparisons",
            "attack_scenarios": [
                {
                    "name": "weak_spike_short",
                    "base_type": "spike",
                    "magnitude": 2.0,
                    "length": 3,
                    "factor_intensity": "weak",
                    "factor_duration": "short",
                    "factor_length": "short",
                },
                {
                    "name": "strong_spike_long",
                    "base_type": "spike",
                    "magnitude": 8.0,
                    "length": 20,
                    "factor_intensity": "strong",
                    "factor_duration": "long",
                    "factor_length": "long",
                },
                {
                    "name": "weak_offset_short",
                    "base_type": "offset",
                    "magnitude": 0.8,
                    "length": 6,
                    "factor_intensity": "weak",
                    "factor_duration": "short",
                    "factor_length": "short",
                },
                {
                    "name": "strong_offset_long",
                    "base_type": "offset",
                    "magnitude": 4.0,
                    "length": 24,
                    "factor_intensity": "strong",
                    "factor_duration": "long",
                    "factor_length": "long",
                },
                {
                    "name": "short_drift",
                    "base_type": "drift",
                    "start": 0.0,
                    "end": 1.5,
                    "length": 12,
                    "factor_intensity": "weak",
                    "factor_duration": "short",
                    "factor_length": "short",
                },
                {
                    "name": "long_drift",
                    "base_type": "drift",
                    "start": 0.0,
                    "end": 6.0,
                    "length": 48,
                    "factor_intensity": "strong",
                    "factor_duration": "long",
                    "factor_length": "long",
                },
                {
                    "name": "short_lag",
                    "base_type": "time_shift",
                    "lag_rows": 1,
                    "length": 4,
                    "factor_intensity": "weak",
                    "factor_duration": "short",
                    "factor_length": "short",
                },
                {
                    "name": "long_delay",
                    "base_type": "time_shift",
                    "lag_rows": 8,
                    "length": 24,
                    "factor_intensity": "strong",
                    "factor_duration": "long",
                    "factor_length": "long",
                },
                {
                    "name": "short_stuck",
                    "base_type": "stuck",
                    "length": 6,
                    "factor_intensity": "weak",
                    "factor_duration": "short",
                    "factor_length": "short",
                },
                {
                    "name": "long_stuck",
                    "base_type": "stuck",
                    "length": 32,
                    "factor_intensity": "strong",
                    "factor_duration": "long",
                    "factor_length": "long",
                },
                {
                    "name": "weak_noise_short",
                    "base_type": "gaussian_noise",
                    "sigma": 0.4,
                    "length": 6,
                    "factor_intensity": "weak",
                    "factor_duration": "short",
                    "factor_length": "short",
                },
                {
                    "name": "strong_noise_long",
                    "base_type": "gaussian_noise",
                    "sigma": 2.5,
                    "length": 24,
                    "factor_intensity": "strong",
                    "factor_duration": "long",
                    "factor_length": "long",
                },
                {
                    "name": "coordinated_drift_unseen",
                    "base_type": "drift",
                    "start": 0.0,
                    "end": 4.0,
                    "length": 24,
                    "station_count": 2,
                    "coordinated": True,
                    "unseen_combination": True,
                    "factor_intensity": "strong",
                    "factor_duration": "long",
                    "factor_length": "long",
                },
            ],
            "valid_attack_seed_offset": 2030001,
            "test_attack_seed_offset": 2040001,
            "method_freeze": {
                "trust_candidates": ["trust_valid_f1", "trust_nonstuck_event_macro"],
                "baselines": [
                    "robust_max_deviation",
                    "isolation_forest",
                    "one_class_svm_20k",
                    "pca_reconstruction",
                    "mlp_autoencoder_reconstruction",
                    "knn_distance",
                    "ewma_deviation",
                    "arima_innovation",
                    "gru_predictor",
                    "lightweight_autoencoder",
                    "ics_cps_spc_proxy",
                ],
                "ocsvm_sensitivity": ["one_class_svm_50k", "one_class_svm_100k"],
                "threshold_rule": "valid normal rows only at FAR=0.0065",
                "selection_rule": (
                    "valid only; test labels unavailable until final evaluation; "
                    "no attack-specific test tuning"
                ),
            },
            "attack_generation_rules": {
                "train": "normal observations only; no attack injection",
                "valid": (
                    "scenario blocks injected with the frozen generator; valid labels may select only "
                    "the frozen candidate rule and FAR threshold"
                ),
                "test": (
                    "scenario blocks injected with an independent seed stream; test labels are evaluation-only"
                ),
                "delay": (
                    "value replayed from the scenario-specific lag_rows field; it is a delayed measurement, "
                    "not a timestamp edit"
                ),
                "drop_replay": (
                    "if included by a future extension, a dropped packet is represented by an old-value hold "
                    "while observation_present remains true; it is not included in the current 13-scenario set"
                ),
                "coordinated": (
                    "two stations receive the same attack start and scenario, selected deterministically "
                    "from the frozen station order"
                ),
                "unseen_boundary": (
                    "coordinated_drift_unseen is absent from V6 and is not used for model or threshold selection"
                ),
            },
            "acceptance_gates": {
                "scenario_baseline_win_share": ">=0.70 overall and report every factor cell",
                "event_coverage_ci": "BH-adjusted 95% CI lower bound > -0.02 or non-inferiority",
                "far_drift": "absolute test FAR drift <= 0.03",
                "unseen_combination": "positive event coverage and no provider-level reversal",
                "reporting": [
                    "pooled",
                    "station_equal",
                    "provider_equal",
                    "F1",
                    "precision",
                    "row_recall",
                    "event_coverage",
                    "AUPRC",
                    "FAR",
                    "detection_delay",
                ],
            },
            "stop_rules": [
                "If official data are unavailable, status remains NOT_EVALUATED.",
                "Do not replace the window with a previously viewed test or an older year.",
                "If any major baseline stably wins, retain the win and downgrade comprehensive-leadership claims.",
                "If FAR drift exceeds 0.03, do not retune thresholds on the test window.",
            ],
            "claim_boundary": [
                "This package supports only controlled simulated-attack claims.",
                "It is a new future-window protocol and cannot produce results before official data access.",
                "No result may be called a production incident recall.",
            ],
            "evaluation_entrypoint": "src/stage425_v9_locked_evaluation.py",
            "audit_entrypoint": "src/stage426_v9_locked_audit.py",
            "statistical_audit_entrypoint": "src/stage445_v9_statistical_acceptance_audit.py",
            "source_protocol_sha256": sha256(V6),
        }
    )
    out["ensemble_freeze"] = {
        "candidate_family": {
            "trust_only": [["trust_valid_f1", 0.0065]],
            "trust_arima": [
                ["trust_valid_f1", 0.004875],
                ["arima_innovation", 0.001625],
            ],
            "trust_shape_union": [
                ["trust_valid_f1", 0.00325],
                ["ewma_deviation", 0.001625],
                ["arima_innovation", 0.001625],
            ],
        },
        "selection": (
            "valid only; retain at least 95% of trust_valid_f1 validation F1, "
            "then maximize validation event coverage, then F1; test labels forbidden"
        ),
        "primary_far": 0.0065,
        "budget_sum_check": {
            "trust_only": 0.0065,
            "trust_arima": 0.0065,
            "trust_shape_union": 0.0065,
        },
    }
    out["source_code_sha256"] = sha256(Path(__file__).resolve())
    PACKAGE.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "\n".join(
            [
                "# Stage423 2026年10月未来时间窗确认性协议冻结",
                "",
                "- 状态：`CONFIRMATORY_PROTOCOL_FROZEN_PENDING_NEW_OFFICIAL_WINDOW`",
                f"- 协议：`{out['protocol_id']}`",
                f"- supersedes：`{out['supersedes_protocol_id']}`",
                "- 本次修复执行入口元数据并纳入统计验收入口；攻击、方法、阈值、统计规则和数据边界均未改变。",
                "- 测试窗：2026-10-01 至 2026-10-31 UTC。",
                "- 当前不下载、不读取观测值、不生成攻击标签、不计算性能。",
                "- 已冻结攻击强度、持续时间、事件长度、未见 coordinated drift、基线、FAR 和统计规则。",
                "- 官方数据不可用时保持 `NOT_EVALUATED`，不得用旧年份或旧 test 替代。",
                "- 所有攻击均为透明受控模拟，不是真实生产攻击日志。",
                "",
                f"- JSON：`{PROTOCOL}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": out["status"],
                "protocol_id": out["protocol_id"],
                "test_window": out["window"],
                "path": str(PROTOCOL),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
