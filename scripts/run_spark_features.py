from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_ai.config import load_config
from inventory_ai.data.spark_features import require_spark, run_spark_feature_parity
from inventory_ai.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Spark time features and verify pandas/Spark parity.")
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_config(config_path)
    require_spark()
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[2]").appName("inventory-ai-feature-parity").getOrCreate()
    try:
        result = run_spark_feature_parity(
            spark,
            root / "data" / "processed" / cfg.run_name / "training_panel_sample.csv",
            root / "data" / "processed" / cfg.run_name / "spark_features.parquet",
        )
    finally:
        spark.stop()
    report = root / cfg.artifact_dir / "spark_feature_parity.json"
    write_json(report, result)
    print(json.dumps({"run_name": cfg.run_name, **result}, indent=2))


if __name__ == "__main__":
    main()
