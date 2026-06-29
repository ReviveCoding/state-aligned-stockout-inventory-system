from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from inventory_ai.config import load_config
from inventory_ai.utils.io import sha256_file, write_csv, write_json


def build_sql_marts(root: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Build deterministic SQL marts for an already completed pipeline run.

    The function is shared by the CLI and archive verifier so pipeline-to-SQL
    connectivity can be checked without launching a second numerical Python
    subprocess. Writes are atomic and the SQL stage updates only its own
    provenance section in the run manifest.
    """
    root = Path(root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_config(config_path)
    panel_path = root / "data" / "processed" / cfg.run_name / "training_panel_sample.csv"
    report_dir = root / cfg.output_dir
    inventory_path = report_dir / "inventory_simulation.csv"
    if not panel_path.exists() or not inventory_path.exists():
        raise FileNotFoundError(
            f"run the forecasting pipeline for {config_path} before SQL marts; "
            f"missing panel={not panel_path.exists()} inventory={not inventory_path.exists()}"
        )

    connection = sqlite3.connect(":memory:")
    try:
        pd.read_csv(panel_path).to_sql("training_panel", connection, index=False, if_exists="replace")
        pd.read_csv(inventory_path).to_sql(
            "inventory_simulation", connection, index=False, if_exists="replace"
        )
        for filename in ("build_daily_demand_mart.sql", "build_inventory_kpi_mart.sql"):
            connection.executescript((root / "sql" / filename).read_text(encoding="utf-8"))
        daily = pd.read_sql_query(
            "SELECT * FROM daily_demand_mart ORDER BY series_id, date_idx", connection
        )
        inventory = pd.read_sql_query(
            "SELECT * FROM inventory_kpi_mart ORDER BY model, series_id", connection
        )
    finally:
        connection.close()

    daily_path = report_dir / "sql_daily_demand_mart.csv"
    inventory_mart_path = report_dir / "sql_inventory_kpi_mart.csv"
    write_csv(daily_path, daily)
    write_csv(inventory_mart_path, inventory)

    manifest_path = root / cfg.artifact_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"pipeline run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sql_outputs"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (daily_path, inventory_mart_path)
    }
    write_json(manifest_path, manifest)
    return {
        "run_name": cfg.run_name,
        "daily_rows": int(len(daily)),
        "inventory_rows": int(len(inventory)),
    }
