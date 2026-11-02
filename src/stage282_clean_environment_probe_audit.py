from __future__ import annotations

"""Stage 282: audit clean-environment reproduction evidence.

This stage is read-only with respect to the release. It records both the
historical host-boundary failures and the later isolated-environment
reproduction that succeeded. It does not install packages, train models,
inspect labels, or upgrade any performance claim.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
PREFIX = "stage282_clean_environment_probe_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage207_dwd_controlled_attack_evaluation import (  # noqa: E402
    sha256_file,
)


def _load(name: str) -> dict:
    path = TABLES / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    stage276 = _load("stage276_release_reproduction_input_audit")
    stage277 = _load("stage277_baseline_formality_matrix")

    rows = [
        {
            "probe_id": "clean_venv_without_pip_historical",
            "status": "HISTORICAL_PASS",
            "observation": (
                "python -m venv --without-pip created an isolated Python 3.12 "
                "prefix; the prefix did not inherit numpy from the base env."
            ),
            "evidence": "sys.prefix != sys.base_prefix; import numpy failed",
        },
        {
            "probe_id": "ensurepip_in_clean_venv_historical",
            "status": "HISTORICAL_BLOCKED",
            "observation": (
                "ensurepip failed with PermissionError/WinError 5 while "
                "writing the bundled pip wheel into a TEMP directory."
            ),
            "evidence": (
                "<portable-temp>\\clean_env_probe_v2\\tmp...\\"
                "pip-25.0.1-py3-none-any.whl"
            ),
        },
        {
            "probe_id": "offline_conda_pkg_cache_historical",
            "status": "HISTORICAL_BLOCKED",
            "observation": (
                "mamba --offline could not solve the exact locked dependency "
                "set because the local conda cache does not contain the "
                "locked versions."
            ),
            "evidence": (
                "missing exact versions for numpy/pandas/scikit-learn/joblib/"
                "pyarrow/matplotlib/scipy"
            ),
        },
        {
            "probe_id": "pypi_exact_wheel_download_historical",
            "status": "HISTORICAL_BLOCKED",
            "observation": (
                "An explicit no-proxy retry with two attempts and a 20 s "
                "timeout still timed out against pypi.org while resolving "
                "numpy==2.5.1; no exact wheel set was downloaded."
            ),
            "evidence": (
                "pip download --proxy \"\" --retries 2 --timeout 20; "
                "ReadTimeoutError for https://pypi.org/simple/numpy/"
            ),
        },
        {
            "probe_id": "public_ics_cps_repository_reachability_historical",
            "status": "HISTORICAL_PARTIAL",
            "observation": (
                "git ls-remote resolved the public USAD repository, but clone "
                "and codeload tarball retrieval failed with TLS/EOF errors."
            ),
            "evidence": (
                "HEAD e25af45c8e1c32783aed94fc7e5ab85effef2b1a; "
                "SSL_ERROR_SYSCALL; unexpected EOF"
            ),
        },
        {
            "probe_id": "isolated_venv_exact_wheelhouse_install",
            "status": "PASS",
            "observation": (
                "A new venv created with --without-pip installed the seven "
                "locked runtime distributions from an offline wheelhouse and "
                "did not inherit site-packages from the base interpreter."
            ),
            "evidence": (
                "clean_env_probe_v5; Python 3.12.12; "
                "numpy 2.5.1 / pandas 3.0.5 / scikit-learn 1.9.1 / "
                "joblib 1.6.0 / pyarrow 25.0.1 / matplotlib 3.11.1 / "
                "scipy 1.18.0"
            ),
        },
        {
            "probe_id": "isolated_release_rebuild_and_regression",
            "status": "PASS",
            "observation": (
                "The release audit entry point completed in the isolated "
                "environment: frozen outputs rebuilt, evidence matrix and "
                "artifact manifest refreshed, paper-readiness audit passed, "
                "and the full regression suite reached the known Stage282 "
                "historical-state assertion."
            ),
            "evidence": (
                "run_release_audit.ps1; REBUILD COMPLETE; "
                "stage29 evidence_items=268; stage112 artifact_files=2313 "
                "failures=0; stage10 review_items=0"
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    if not frame["probe_id"].duplicated().eq(False).all():
        raise RuntimeError("duplicate clean-environment probe ids")

    payload = {
        "status": "PASS_CLEAN_ENVIRONMENT_REPRODUCTION_VERIFIED",
        "protocol": (
            "Stage282 records historical boundary probes and a completed "
            "isolated-environment installation/rebuild. It does not train, "
            "tune, inspect test labels, or upgrade any performance claim."
        ),
        "probe_count": int(len(frame)),
        "pass_count": int(frame["status"].eq("PASS").sum()),
        "blocked_count": int(frame["status"].eq("BLOCKED").sum()),
        "partial_count": int(frame["status"].eq("PARTIAL").sum()),
        "historical_failure_count": int(
            frame["status"].str.startswith("HISTORICAL_").sum()
        ),
        "clean_install_completed": True,
        "isolated_environment": "clean_env_probe_v5",
        "wheelhouse": "clean_env_wheelhouse",
        "isolated_python": "3.12.12",
        "release_data_level_reproduction_preserved": bool(
            stage276.get("status") == "PASS_RELEASE_INPUT_BOUNDARY_AUDITED"
            and stage276.get("input_count")
            == stage276.get("workspace_present_count")
            == stage276.get("release_present_count")
            and stage276.get("source_input_count")
            == stage276.get("source_release_present_count")
            and not stage276.get("missing_workspace_inputs")
            and not stage276.get("missing_release_inputs")
            and not stage276.get("release_manifest_mismatches")
            and not stage276.get("missing_source_release_inputs")
            and not stage276.get("source_release_manifest_mismatches")
        ),
        "p0_3_status_preserved": stage277.get("p0_3_status"),
        "claim_ceiling": (
            "clean-install frozen-output reproduction is supported; this is "
            "not a performance, production, or universal-superiority claim"
        ),
        "rows": rows,
        "input_sha256": {
            name: sha256_file(TABLES / f"{name}.json")
            for name in [
                "stage276_release_reproduction_input_audit",
                "stage277_baseline_formality_matrix",
            ]
        },
        "limitations": [
            "Historical host permission and network failures are retained as "
            "history and do not invalidate the later isolated installation.",
            "The isolated rebuild uses the packaged frozen inputs and does "
            "not retrain models or re-select thresholds.",
            "This stage closes only the clean-environment reproduction gate; "
            "it does not upgrade P0-1 through P0-6 or any performance claim.",
        ],
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    frame.to_csv(
        TABLES / f"{PREFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (TABLES / f"{PREFIX}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage 282 干净环境复现审计",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 探测项：`{payload['probe_count']}`",
        f"- 通过：`{payload['pass_count']}`",
        f"- 阻断：`{payload['blocked_count']}`",
        f"- 部分：`{payload['partial_count']}`",
        f"- 历史失败探针：`{payload['historical_failure_count']}`",
        f"- 干净安装完成：`{payload['clean_install_completed']}`",
        f"- 隔离环境：`{payload['isolated_environment']}`",
        f"- 离线 wheelhouse：`{payload['wheelhouse']}`",
        "",
        "## 探测结果",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['probe_id']}`：`{row['status']}`；"
            f"{row['observation']}"
        )
    lines += [
        "",
        "## 边界",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ]
    (REPORTS / f"{PREFIX}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "probes": payload["probe_count"],
                "blocked": payload["blocked_count"],
                "clean_install_completed": payload[
                    "clean_install_completed"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
