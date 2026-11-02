from __future__ import annotations

"""Stage 295: freeze the NOAA ISD Japan locked evaluation contract.

Stage294 fixed the new Japanese station network, years, variables, quality
rules, transparent attacks, seeds, primary FAR, and primary endpoint.  This
stage freezes the remaining evaluation choices before any Japanese ISD
observation archive is requested: detector components, candidate features,
all comparison methods and their fit budgets, threshold calibration,
bootstrap/multiplicity rules, and the claim-promotion gate.

No Japanese ISD observation values are downloaded or inspected here.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage295_japan_isd_locked_evaluation_contract"
PROTOCOL_PREFIX = "stage294_japan_isd_protocol_preregistration"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage248_cross_provider_fair_baseline_audit as s248  # noqa: E402
import stage256_lightweight_strong_baselines as s256  # noqa: E402
import stage262_security_domain_baselines as s262  # noqa: E402
from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    COMPONENTS,
    sha256_file,
)

DATA_PROTOCOL_PATH = TABLES / f"{PROTOCOL_PREFIX}.json"
DATA_STATIONS_PATH = TABLES / f"{PROTOCOL_PREFIX}_stations.csv"
DATA_METADATA_PATH = (
    TABLES / f"{PROTOCOL_PREFIX}_metadata_source.json"
)

TRUST_METHODS = list(s248.TRUST_METHODS)
CONFIRMATORY_BASELINES = list(s248.BASELINE_METHODS)
RECONSTRUCTION_BASELINES = list(s256.NEW_BASELINES)
SECURITY_BASELINES = list(s262.SECURITY_BASELINES)
ALL_BASELINES = [
    *CONFIRMATORY_BASELINES,
    *RECONSTRUCTION_BASELINES,
    *SECURITY_BASELINES,
]
METHODS = [*TRUST_METHODS, *ALL_BASELINES]

PRIMARY_FAR = 0.01
FAR_GRID = [0.005, 0.01, 0.02, 0.05]
PRIMARY_ENDPOINT = "event_coverage"
SECONDARY_ENDPOINTS = [
    "f1",
    "precision",
    "recall",
    "row_recall",
    "false_alarm_rate",
    "average_precision",
]
CLUSTER_UNIT = "station"
NETWORK_LEVEL = "single_provider_new_station_network"
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260921
MULTIPLICITY = "Benjamini-Hochberg"
NON_INFERIORITY_MARGIN = -0.02
SUPERIORITY_RULE = "BH-adjusted 95% station-clustered bootstrap CI entirely above 0"
NON_INFERIORITY_RULE = (
    "BH-adjusted 95% station-clustered bootstrap lower bound greater than -0.02"
)

DETECTOR_COMPONENTS = list(COMPONENTS)
TRAIN_RULE = (
    "Fit all train-normal components and models only on Japan train years "
    "2021-2023 after exact metadata, station, split, and ISD quality checks. "
    "No valid or test row may affect fitting."
)
THRESHOLD_RULE = (
    "Calibrate the empirical 1-FAR quantile on the frozen 2024 valid-year "
    "normal rows only, after deterministic valid attack injection. Test "
    "labels are never used for calibration."
)
TRUST_SELECTION_RULE = (
    "Use the pre-existing Stage244/248 selection rules without adding a "
    "Japan-specific choice: trust_valid_f1 selects on valid F1 and "
    "trust_nonstuck_event_macro selects on valid non-stuck event macro."
)

OCSVM_FIT_ROWS = [20000, 50000, 100000]
OCSVM_NU = 0.05
OCSVM_KERNEL = "rbf"
OCSVM_GAMMA = "scale"
OCSVM_CACHE_SIZE = 1000
OCSVM_MAX_ITER = -1
OCSVM_FIT_SEED = int(s248.BASELINE_FIT_SEED)

PCA_COMPONENTS_256 = int(s256.PCA_COMPONENTS)
MLP_HIDDEN_LAYERS = list(s256.MLP_HIDDEN_LAYERS)
MLP_MAX_ITER = int(s256.MLP_MAX_ITER)
KNN_NEIGHBOURS = int(s256.KNN_NEIGHBOURS)
LIGHTWEIGHT_FIT_ROWS = int(s256.MAX_FIT_ROWS)
KNN_FIT_ROWS = int(s256.KNN_MAX_FIT_ROWS)
LIGHTWEIGHT_SEED = int(s256.MODEL_SEED)

EWMA_ALPHA = float(s262.EWMA_ALPHA)
AR_ORDER = int(s262.AR_ORDER)
GRU_HIDDEN = int(s262.GRU_HIDDEN)
AE_COMPONENTS_262 = int(s262.AE_COMPONENTS)
SECURITY_FIT_ROWS = int(s262.MAX_FIT_ROWS)
SECURITY_SEED = int(s262.MODEL_SEED)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data_protocol = load_json(DATA_PROTOCOL_PATH)
    if data_protocol["status"] != "PROTOCOL_FROZEN_BEFORE_FULL_DOWNLOAD":
        raise RuntimeError("Stage294 Japan data protocol is not frozen")
    if data_protocol["download_status"] != "NOT_DOWNLOADED":
        raise RuntimeError("Stage294 Japan download is already consumed")
    if data_protocol["performance_status"] != "NOT_EVALUATED":
        raise RuntimeError("Stage294 Japan performance is already evaluated")
    if float(data_protocol["primary_far"]) != float(PRIMARY_FAR):
        raise RuntimeError("Japan primary FAR changed")
    if data_protocol["primary_endpoint"] != PRIMARY_ENDPOINT:
        raise RuntimeError("Japan primary endpoint changed")
    if data_protocol["test_period_end_utc"] != "2025-08-24T23:59:59Z":
        raise RuntimeError("Japan test-period boundary changed")

    methods = METHODS
    method_contract = {
        "trust_valid_f1": {
            "family": "trust",
            "selection_rule": "valid_f1",
            "target_tuning": True,
            "fit_budget": "pre-existing Stage244 candidate grid",
        },
        "trust_nonstuck_event_macro": {
            "family": "trust",
            "selection_rule": "valid_nonstuck_event_macro",
            "target_tuning": True,
            "fit_budget": "pre-existing Stage244 candidate grid",
        },
        "robust_max_deviation": {
            "family": "existing_baseline",
            "model": "maximum positive robust standardized feature deviation",
            "fit_budget": "all train-normal rows",
        },
        "isolation_forest": {
            "family": "existing_baseline",
            "model": "IsolationForest",
            "n_estimators": 300,
            "contamination": "auto",
            "random_state": int(s248.BASELINE_FIT_SEED),
            "n_jobs": 1,
            "fit_budget": "all train-normal rows",
        },
        "one_class_svm": {
            "family": "existing_baseline",
            "model": "StandardScaler + OneClassSVM",
            "kernel": OCSVM_KERNEL,
            "gamma": OCSVM_GAMMA,
            "nu": OCSVM_NU,
            "cache_size": OCSVM_CACHE_SIZE,
            "max_iter": OCSVM_MAX_ITER,
            "fit_seed": OCSVM_FIT_SEED,
            "reported_fit_budgets": OCSVM_FIT_ROWS,
            "fit_budget_rule": (
                "Report all pre-registered 20k/50k/100k stratified "
                "train-normal budgets; do not choose the budget by test."
            ),
        },
        "pca_reconstruction": {
            "family": "reconstruction_baseline",
            "components": PCA_COMPONENTS_256,
            "random_state": LIGHTWEIGHT_SEED,
            "fit_budget": LIGHTWEIGHT_FIT_ROWS,
        },
        "mlp_autoencoder_reconstruction": {
            "family": "reconstruction_baseline",
            "hidden_layers": MLP_HIDDEN_LAYERS,
            "max_iter": MLP_MAX_ITER,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": LIGHTWEIGHT_SEED,
            "fit_budget": LIGHTWEIGHT_FIT_ROWS,
        },
        "knn_distance": {
            "family": "reconstruction_baseline",
            "neighbours": KNN_NEIGHBOURS,
            "fit_budget": KNN_FIT_ROWS,
            "random_state": LIGHTWEIGHT_SEED,
        },
        "ewma_deviation": {
            "family": "security_baseline",
            "alpha": EWMA_ALPHA,
            "target_ground_truth": False,
        },
        "arima_innovation": {
            "family": "security_baseline",
            "ar_order": AR_ORDER,
            "implementation": "per-station least-squares AR innovation",
        },
        "gru_predictor": {
            "family": "security_baseline",
            "hidden_size": GRU_HIDDEN,
            "implementation": (
                "fixed random gated recurrent reservoir with deterministic "
                "train-normal linear readout"
            ),
            "seed": SECURITY_SEED,
        },
        "lightweight_autoencoder": {
            "family": "security_baseline",
            "components": AE_COMPONENTS_262,
            "random_state": SECURITY_SEED,
            "fit_budget": SECURITY_FIT_ROWS,
        },
        "ics_cps_spc_proxy": {
            "family": "security_baseline",
            "implementation": "transparent per-station SPC proxy",
            "not_original_paper_reproduction": True,
        },
    }
    missing = [method for method in methods if method not in method_contract]
    if missing:
        raise RuntimeError(f"missing Japan method contract: {missing}")

    contract = {
        "status": "EVALUATION_CONTRACT_FROZEN_BEFORE_FULL_DOWNLOAD",
        "contract_id": "TMQC-JAPAN-ISD-EVAL-20260921-v1.0",
        "data_protocol_id": data_protocol["protocol_id"],
        "provider": data_protocol["provider"],
        "data_protocol_sha256": sha256_file(DATA_PROTOCOL_PATH),
        "stations_sha256": sha256_file(DATA_STATIONS_PATH),
        "metadata_source_sha256": sha256_file(DATA_METADATA_PATH),
        "download_status": "NOT_DOWNLOADED",
        "performance_status": "NOT_EVALUATED",
        "observation_access": (
            "No Japanese ISD observation CSV is requested or inspected in "
            "this stage. Stage294 also reports strict_no_observation_access_"
            "before_freeze=true."
        ),
        "methods": methods,
        "method_contract": method_contract,
        "confirmatory_baselines": CONFIRMATORY_BASELINES,
        "reconstruction_baselines": RECONSTRUCTION_BASELINES,
        "security_baselines": SECURITY_BASELINES,
        "detector_components": DETECTOR_COMPONENTS,
        "attack_parameters": data_protocol["attack_parameters"],
        "quality_accepted_code": data_protocol["quality_accepted_code"],
        "quality_rule": data_protocol["quality_rule"],
        "train_rule": TRAIN_RULE,
        "threshold_rule": THRESHOLD_RULE,
        "trust_selection_rule": TRUST_SELECTION_RULE,
        "feature_contract": (
            "Use the existing Stage244 candidate feature set and Stage248 "
            "feature matrix without adding Japan-specific features."
        ),
        "primary_far": PRIMARY_FAR,
        "far_grid": FAR_GRID,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoints": SECONDARY_ENDPOINTS,
        "cluster_unit": CLUSTER_UNIT,
        "network_level": NETWORK_LEVEL,
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "hierarchy": "station within the single NOAA Japan network",
            "resampling_unit": "station",
        },
        "multiplicity": MULTIPLICITY,
        "non_inferiority_margin": NON_INFERIORITY_MARGIN,
        "superiority_rule": SUPERIORITY_RULE,
        "non_inferiority_rule": NON_INFERIORITY_RULE,
        "promotion_gate": {
            "single_network_can_resolve_provider_independence": False,
            "single_network_can_resolve_station_independence": False,
            "maximum_claim": (
                "bounded new-station-network transfer evidence; supplementary "
                "to, not a replacement for, independent-provider and "
                "production-incident evidence"
            ),
            "required_for_Japan_promotion": (
                "All pre-registered baselines must be evaluated on the same "
                "Japan locked test; Trust must meet the superiority or "
                "non-inferiority rule on the primary endpoint and report "
                "F1, precision, recall, row recall, FAR, AUPRC, and event "
                "coverage. This network cannot establish provider "
                "independence or real-attack recall."
            ),
        },
        "forbidden_claims": [
            "Japan ISD tests are real production attacks or incidents.",
            "One new NOAA station network proves provider independence.",
            "One new NOAA station network alone proves station independence.",
            "Japan ISD results may be used to tune the model or any baseline.",
            "A controlled injection may be reported as a real attack recall.",
            "Japan transfer may overwrite Stage246, FMI, or SMHI results.",
        ],
        "promotion_status": {
            "provider_dependence": "NOT_RESOLVED_BY_SAME_PROVIDER_NETWORK",
            "station_network_dependence": (
                "PARTIALLY_TESTABLE_AFTER_LOCKED_TRANSFER"
            ),
            "real_production_incidents": "BOUNDED",
        },
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    contract_path = TABLES / f"{PREFIX}.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# 阶段295 NOAA ISD日本锁定评价契约",
        "",
        f"- 状态：**{contract['status']}**",
        f"- 契约ID：`{contract['contract_id']}`",
        f"- 数据协议：`{contract['data_protocol_id']}`",
        f"- 下载状态：`{contract['download_status']}`",
        f"- 性能状态：`{contract['performance_status']}`",
        f"- 方法数：`{len(methods)}`",
        f"- 主FAR：`{PRIMARY_FAR}`；主终点：`{PRIMARY_ENDPOINT}`",
        f"- 聚类单位：`{CLUSTER_UNIT}`；bootstrap：`{BOOTSTRAP_ITERATIONS}`",
        f"- 非劣效界值：`{NON_INFERIORITY_MARGIN}`",
        "",
        "## 冻结的方法",
        "",
        *[
            f"- `{method}`：{method_contract[method].get('family', '')}"
            for method in methods
        ],
        "",
        "## 冻结边界",
        "",
        "- 所有模型只在日本2021-2023 train正常段拟合。",
        "- 阈值只在2024 valid正常段按冻结FAR校准。",
        "- 20k/50k/100k One-Class SVM预算全部报告，不能用test选预算。",
        "- test仅用于最终评价，不能用于选择模型、特征、权重、阈值或攻击参数。",
        "- 所有攻击均为透明受控注入，不是真实攻击日志。",
        "- 日本网络是同一NOAA provider下的新站点网络，不能证明机构无关。",
        "- 本阶段不替换Stage246、FMI或SMHI的冻结结果。",
        "",
        "## 禁止声称",
        "",
        *[f"- {item}" for item in contract["forbidden_claims"]],
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": contract["status"],
                "contract_id": contract["contract_id"],
                "method_count": len(methods),
                "primary_far": PRIMARY_FAR,
                "primary_endpoint": PRIMARY_ENDPOINT,
                "contract_sha256": sha256_file(contract_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
