from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_ai.data.spark_features import spark_available
from inventory_ai.models.chronos_adapter import check_chronos_capability


def main() -> None:
    print(json.dumps({"chronos": check_chronos_capability().__dict__, "spark_available": spark_available()}, indent=2))


if __name__ == "__main__":
    main()
