"""Build one queryable evidence file for the final TrustMeteo-QC experiment.

The database is a writing-facing aggregation layer. It contains every CSV
result table and JSON audit artifact from the final submission package, plus a
catalog of excluded binary/raw artifacts with SHA256 and source paths.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission_final_20261009"
TABLES = FINAL / "tables"
REPORTS = FINAL / "reports"
OUT = FINAL / "final_experiment_evidence.sqlite"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_name(path: Path) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem).strip("_").lower()
    return f"csv_{name}"


def json_payload(path: Path) -> str:
    try:
        return json.dumps(
            json.loads(path.read_text(encoding="utf-8")),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (json.JSONDecodeError, UnicodeDecodeError):
        return path.read_text(encoding="utf-8")


def main() -> int:
    OUT.unlink(missing_ok=True)
    connection = sqlite3.connect(OUT)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """
            CREATE TABLE artifact_catalog (
                artifact_name TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                included_in_database INTEGER NOT NULL,
                row_count INTEGER,
                column_count INTEGER,
                sha256 TEXT NOT NULL,
                notes TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE json_artifacts (
                artifact_name TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                json_text TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE report_text (
                artifact_name TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                markdown_text TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE paper_source_map (
                claim_area TEXT PRIMARY KEY,
                authoritative_tables TEXT NOT NULL,
                evidence_class TEXT NOT NULL,
                writing_rule TEXT NOT NULL
            )
            """
        )

        csv_paths = sorted(TABLES.glob("*.csv"))
        json_paths = sorted(set(TABLES.glob("*.json")) | set(REPORTS.glob("*.json")))
        report_paths = sorted(REPORTS.glob("*.md"))

        for path in csv_paths:
            name = table_name(path)
            frame = pd.read_csv(path)
            frame.to_sql(name, connection, if_exists="replace", index=False)
            connection.execute(
                """
                INSERT INTO artifact_catalog VALUES (?, ?, 'csv', 1, ?, ?, ?, ?)
                """,
                (
                    path.name,
                    str(path.relative_to(ROOT)).replace("\\", "/"),
                    int(len(frame)),
                    int(len(frame.columns)),
                    sha256(path),
                    f"query table: {name}",
                ),
            )

        for path in json_paths:
            connection.execute(
                "INSERT INTO json_artifacts VALUES (?, ?, ?, ?)",
                (
                    path.name,
                    str(path.relative_to(ROOT)).replace("\\", "/"),
                    sha256(path),
                    json_payload(path),
                ),
            )
            connection.execute(
                """
                INSERT INTO artifact_catalog VALUES (?, ?, 'json', 1, NULL, NULL, ?, ?)
                """,
                (
                    path.name,
                    str(path.relative_to(ROOT)).replace("\\", "/"),
                    sha256(path),
                    "raw JSON retained in json_artifacts",
                ),
            )

        for path in report_paths:
            connection.execute(
                "INSERT INTO report_text VALUES (?, ?, ?, ?)",
                (
                    path.name,
                    str(path.relative_to(ROOT)).replace("\\", "/"),
                    sha256(path),
                    path.read_text(encoding="utf-8"),
                ),
            )
            connection.execute(
                """
                INSERT INTO artifact_catalog VALUES (?, ?, 'markdown', 1, NULL, NULL, ?, ?)
                """,
                (
                    path.name,
                    str(path.relative_to(ROOT)).replace("\\", "/"),
                    sha256(path),
                    "writing/audit report retained in report_text",
                ),
            )

        binary_paths = sorted(TABLES.glob("*.parquet"))
        binary_paths += sorted(FINAL.rglob("*.pdf"))
        binary_paths += sorted(FINAL.rglob("*.svg"))
        binary_paths += sorted(FINAL.rglob("*.png"))
        for path in binary_paths:
            connection.execute(
                """
                INSERT INTO artifact_catalog VALUES (?, ?, ?, 0, NULL, NULL, ?, ?)
                """,
                (
                    str(path.relative_to(FINAL)).replace("\\", "/"),
                    str(path.relative_to(ROOT)).replace("\\", "/"),
                    path.suffix.lstrip(".").lower(),
                    sha256(path),
                    "kept as external binary artifact; source path and hash retained",
                ),
            )

        source_map = [
            (
                "primary_metrics",
                "csv_final_v6_aggregation_levels_corrected,csv_final_v6_method_summary,csv_final_v6_union_decision_provenance",
                "V6 frozen primary",
                "Use corrected count-pooled and explicitly labeled macro/weighted rows; do not replace with Stage500 exploratory values.",
            ),
            (
                "scenario_and_uncertainty",
                "csv_final_v6_scenario_metrics,csv_final_v6_scenario_ci_corrected,csv_final_v6_station_cluster_ci_corrected,csv_final_v6_event_lengths",
                "V6 frozen primary",
                "Report event coverage, F1, FAR, event counts, and dependency-aware intervals together.",
            ),
            (
                "baseline_and_implementation",
                "csv_final_method_implementation_boundary_corrected,csv_final_baseline_resources_corrected,csv_stage486_formal_lagged_mlp_comparison",
                "primary plus exploratory",
                "Name proxy implementations accurately and keep formal MLP and other post-freeze comparisons exploratory.",
            ),
            (
                "external_and_negative_boundaries",
                "csv_final_external_transfer,csv_final_negative_boundaries,csv_final_evidence_hierarchy",
                "external and limitation evidence",
                "Keep FMI/SMHI mixed transfer, OCSVM complementarity, and failed scenarios visible.",
            ),
            (
                "recovery_and_runtime",
                "csv_final_online_detection,csv_final_online_offline_recovery,csv_final_testbed_resources_corrected,csv_final_trust_resources_corrected",
                "authorized testbed and safety analysis",
                "Do not equate batch V6 detection or offline recovery with strict online causal capability.",
            ),
            (
                "exploratory_closeout",
                "csv_stage500_final_candidate_runs,csv_stage500_final_candidate_scenario,csv_stage500_final_candidate_provider,csv_stage500_final_candidate_bootstrap",
                "exploratory",
                "Use only as sensitivity/ablation context; never substitute for the V6 primary result.",
            ),
        ]
        connection.executemany(
            "INSERT INTO paper_source_map VALUES (?, ?, ?, ?)",
            source_map,
        )

        connection.execute(
            "CREATE INDEX idx_catalog_type ON artifact_catalog(artifact_type)"
        )
        connection.commit()
    finally:
        connection.close()

    with sqlite3.connect(OUT) as check:
        tables = check.execute(
            "SELECT COUNT(*) FROM artifact_catalog WHERE included_in_database=1"
        ).fetchone()[0]
        csv_tables = check.execute(
            "SELECT COUNT(*) FROM artifact_catalog WHERE artifact_type='csv'"
        ).fetchone()[0]
        sources = check.execute(
            "SELECT COUNT(*) FROM paper_source_map"
        ).fetchone()[0]
    print(
        json.dumps(
            {
                "status": "PASS_FINAL_EXPERIMENT_EVIDENCE_DATABASE",
                "output": str(OUT),
                "included_artifacts": int(tables),
                "csv_tables": int(csv_tables),
                "paper_source_map_rows": int(sources),
                "sha256": sha256(OUT),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
