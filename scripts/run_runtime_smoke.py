from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_ai.config import load_config
from inventory_ai.pipeline import run_pipeline
from inventory_ai.reporting.sql_marts import build_sql_marts


def run_connected_smoke(root: Path, config_path: Path) -> dict[str, Any]:
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_config(config_path)
    result = run_pipeline(cfg, root)
    sql_result = build_sql_marts(root, config_path)
    return {
        "run_name": cfg.run_name,
        "gate_status": result["release"]["gate_status"],
        "sql": sql_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one connected forecasting-to-SQL smoke workflow in a single process."
    )
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    payload = run_connected_smoke(Path(args.root).resolve(), Path(args.config))
    print(json.dumps(payload, sort_keys=True), flush=True)

    # Some constrained Linux/Python numerical stacks can finish all work and
    # emit the final JSON, then wait during native sklearn/BLAS interpreter
    # shutdown. This script is a smoke-test entrypoint rather than a library;
    # artifacts are already fully written and verified after this point. Exit
    # the process directly so archive/runtime verification cannot hang after a
    # successful connected pipeline+SQL run. Library callers should use
    # run_connected_smoke() instead.
    os._exit(0)


if __name__ == "__main__":
    main()
