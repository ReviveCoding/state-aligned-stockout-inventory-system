from __future__ import annotations

import argparse
import json
from pathlib import Path

from inventory_ai import __version__
from inventory_ai.config import load_config


def _resolve_root_and_config(root_arg: str | None, config_arg: str, parser: argparse.ArgumentParser) -> tuple[Path, Path]:
    """Resolve CLI root/config without assuming execution from the source tree.

    Installed console scripts live under ``site-packages``; deriving the project
    root from ``__file__`` makes relative config paths point at the wheel
    location rather than the user's current checkout.  Defaulting to CWD keeps
    README copy/paste commands reproducible while still allowing ``--root`` for
    non-root working directories.
    """
    root = Path(root_arg).expanduser().resolve() if root_arg else Path.cwd().resolve()
    config_path = Path(config_arg).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.exists():
        parser.error(f"configuration file not found: {config_path}")
    return root, config_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run state-aligned stockout-aware forecasting pipeline.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", help="YAML configuration path")
    parser.add_argument("--root", default=None, help="Repository root; inferred from current working directory by default")
    args = parser.parse_args()
    if not args.config:
        parser.error("--config is required unless --version is used")

    root, config_path = _resolve_root_and_config(args.root, args.config, parser)

    # Import the numerical pipeline lazily. `inventory-ai --help` and
    # `inventory-ai --version` remain lightweight in clean wheel environments
    # and do not initialize scikit-learn/BLAS worker pools.
    from inventory_ai.pipeline import run_pipeline

    result = run_pipeline(load_config(config_path), root)
    print(json.dumps({"gate_status": result["release"]["gate_status"], "metrics": result["metrics"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
