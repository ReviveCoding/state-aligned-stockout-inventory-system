from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import shutil
import uuid

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

SPARK_PARITY_COLUMNS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28",
    "roll_std_7", "roll_std_14", "roll_std_28",
    "zero_ratio_14", "price_change", "series_age",
]


def spark_available() -> bool:
    try:
        import pyspark  # noqa: F401
        return True
    except ImportError:
        return False


def require_spark() -> None:
    if not spark_available():
        raise RuntimeError("PySpark is optional. Install with `pip install -e .[spark]`.")


def add_time_features_spark(frame: "DataFrame", value_col: str = "recovered_demand_mean") -> "DataFrame":
    """Recompute the causal pandas time-feature contract with Spark windows.

    The implementation intentionally mirrors ``features.basic.add_time_features``
    so a local sample can be used as a parity check before scaling the same
    transformation to partitioned retail data.
    """
    require_spark()
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    order = Window.partitionBy("series_id").orderBy("date_idx")
    out = frame
    for lag in (1, 7, 14, 28):
        out = out.withColumn(f"lag_{lag}", F.coalesce(F.lag(value_col, lag).over(order), F.lit(0.0)))

    for width in (7, 14, 28):
        history = order.rowsBetween(-width, -1)
        count = F.count(F.col(value_col)).over(history)
        out = out.withColumn(
            f"roll_mean_{width}",
            F.when(count >= 2, F.avg(F.col(value_col)).over(history)).otherwise(F.lit(0.0)),
        )
        out = out.withColumn(
            f"roll_std_{width}",
            F.when(count >= 2, F.stddev_samp(F.col(value_col)).over(history)).otherwise(F.lit(0.0)),
        )

    history_14 = order.rowsBetween(-14, -1)
    count_14 = F.count(F.col(value_col)).over(history_14)
    zero_indicator = F.when(F.col(value_col) == 0, F.lit(1.0)).otherwise(F.lit(0.0))
    out = out.withColumn(
        "zero_ratio_14",
        F.when(count_14 >= 2, F.avg(zero_indicator).over(history_14)).otherwise(F.lit(0.0)),
    )
    previous_price = F.lag("price", 1).over(order)
    out = out.withColumn(
        "price_change",
        F.when(previous_price.isNull() | (previous_price == 0), F.lit(0.0))
        .otherwise((F.col("price") - previous_price) / previous_price),
    )
    out = out.withColumn("series_age", (F.row_number().over(order) - 1).cast("double"))
    return out



def validate_feature_parity(
    pandas_values,
    spark_values,
    tolerance: float = 1e-8,
) -> dict[str, float | int]:
    """Validate pandas/Spark feature equality before publishing output."""
    import numpy as np

    if tolerance < 0:
        raise ValueError("Spark parity tolerance must be nonnegative")
    merged = pandas_values.merge(
        spark_values,
        on=["series_id", "date_idx"],
        how="inner",
        suffixes=("_pandas", "_spark"),
        validate="one_to_one",
    )
    if len(merged) != len(pandas_values) or len(merged) != len(spark_values):
        raise ValueError("Spark feature parity key coverage mismatch")
    max_error = 0.0
    for column in SPARK_PARITY_COLUMNS:
        left = merged[f"{column}_pandas"].astype(float).to_numpy()
        right = merged[f"{column}_spark"].astype(float).to_numpy()
        error = float(np.nanmax(np.abs(left - right)))
        if not np.isfinite(error):
            raise ValueError(f"Spark feature parity produced non-finite error: {column}")
        max_error = max(max_error, error)
    if max_error > tolerance:
        raise ValueError(
            f"Spark feature parity exceeded tolerance: max_abs_error={max_error}, tolerance={tolerance}"
        )
    return {
        "rows": int(len(merged)),
        "features": len(SPARK_PARITY_COLUMNS),
        "max_abs_error": max_error,
        "tolerance": float(tolerance),
    }


def publish_directory_atomically(staging: Path, output: Path) -> None:
    """Publish a directory with rollback if the final rename fails."""
    if not staging.is_dir():
        raise FileNotFoundError(staging)
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    had_previous = output.exists()
    if had_previous:
        output.rename(backup)
    try:
        staging.rename(output)
    except Exception:
        if had_previous and backup.exists() and not output.exists():
            backup.rename(output)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def run_spark_feature_parity(
    spark: "SparkSession",
    input_csv: str | Path,
    output_parquet: str | Path,
    value_col: str = "recovered_demand_mean",
    tolerance: float = 1e-8,
) -> dict[str, float | int]:
    """Validate Spark/pandas parity and publish Parquet only after success."""
    require_spark()
    import pandas as pd

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    source = spark.read.option("header", True).option("inferSchema", True).csv(str(input_path))
    transformed = add_time_features_spark(source, value_col=value_col)
    spark_values = (
        transformed.select("series_id", "date_idx", *SPARK_PARITY_COLUMNS)
        .orderBy("series_id", "date_idx")
        .toPandas()
    )
    pandas_values = pd.read_csv(input_path)[["series_id", "date_idx", *SPARK_PARITY_COLUMNS]].sort_values(
        ["series_id", "date_idx"]
    )
    result = validate_feature_parity(pandas_values, spark_values, tolerance=tolerance)

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".{output_path.name}.staging-{uuid.uuid4().hex}")
    try:
        transformed.write.mode("errorifexists").parquet(str(staging))
        publish_directory_atomically(staging, output_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return result
