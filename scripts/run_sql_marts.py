from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_ai.reporting.sql_marts import build_sql_marts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQL marts for one configured pipeline run.")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    print(build_sql_marts(args.root, args.config))


if __name__ == "__main__":
    main()
