from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PANEL_COLUMNS = [
    "series_id", "date_idx", "sales", "demand", "stockout", "price",
    "promo", "category", "store", "dow",
]
FORECAST_COLUMNS = ["series_id", "date_idx", "horizon", "q10", "q50", "q90", "model"]


@dataclass(frozen=True)
class ContractResult:
    rows: int
    series: int
    min_date_idx: int
    max_date_idx: int


def validate_panel(df: pd.DataFrame, *, require_truth: bool = True) -> ContractResult:
    missing = sorted(set(PANEL_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"panel missing columns: {missing}")
    if df.empty:
        raise ValueError("panel is empty")
    if df[["series_id", "date_idx", "category", "store"]].isna().any().any():
        raise ValueError("series_id/date_idx/category/store cannot be null")
    date_values = df["date_idx"].astype(float).to_numpy()
    if not np.equal(date_values, np.floor(date_values)).all():
        raise ValueError("date_idx must contain integer day indices")
    if df.duplicated(["series_id", "date_idx"]).any():
        dup = df[df.duplicated(["series_id", "date_idx"], keep=False)].head(5)
        raise ValueError(f"duplicate series/date rows: {dup.to_dict('records')}")
    numeric = ["date_idx", "sales", "demand", "stockout", "price", "promo", "dow"]
    if not np.isfinite(df[numeric].astype(float).to_numpy()).all():
        raise ValueError("panel contains non-finite numeric values")
    if (df[["sales", "demand", "price"]] < 0).any().any():
        raise ValueError("sales, demand, and price must be nonnegative")
    if not set(df["stockout"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("stockout must be binary")
    if not set(df["promo"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("promo must be binary")
    if not df["dow"].between(0, 6).all():
        raise ValueError("dow must be between 0 and 6")
    if require_truth and (df["demand"] + 1e-9 < df["sales"]).any():
        raise ValueError("latent demand cannot be below observed sales")
    counts = df.groupby("series_id")["date_idx"].nunique()
    if counts.min() < 10:
        raise ValueError("every series must contain at least 10 observations")
    bounds = df.groupby("series_id")["date_idx"].agg(["min", "max", "nunique"])
    expected_counts = bounds["max"] - bounds["min"] + 1
    if not bounds["nunique"].eq(expected_counts).all():
        raise ValueError("each series must have contiguous daily date_idx values")
    if bounds[["min", "max"]].drop_duplicates().shape[0] != 1:
        raise ValueError("core panel must be aligned to common min/max date_idx bounds")
    return ContractResult(
        rows=int(len(df)),
        series=int(df["series_id"].nunique()),
        min_date_idx=int(df["date_idx"].min()),
        max_date_idx=int(df["date_idx"].max()),
    )


def validate_forecast(fcst: pd.DataFrame, expected_keys: pd.DataFrame | None = None) -> None:
    missing = sorted(set(FORECAST_COLUMNS) - set(fcst.columns))
    if missing:
        raise ValueError(f"forecast missing columns: {missing}")
    if fcst.empty:
        raise ValueError("forecast is empty")
    if fcst[["model", "series_id", "date_idx", "horizon"]].isna().any().any():
        raise ValueError("forecast identifiers cannot be null")
    horizon = fcst["horizon"].astype(float).to_numpy()
    date_idx = fcst["date_idx"].astype(float).to_numpy()
    if not np.equal(horizon, np.floor(horizon)).all() or (horizon < 1).any():
        raise ValueError("forecast horizon must be a positive integer")
    if not np.equal(date_idx, np.floor(date_idx)).all():
        raise ValueError("forecast date_idx must be integer-valued")
    key = ["model", "series_id", "date_idx"]
    if fcst.duplicated(key).any():
        raise ValueError("forecast contains duplicate model/series/date rows")
    if not np.isfinite(fcst[["q10", "q50", "q90"]].to_numpy()).all():
        raise ValueError("forecast contains non-finite values")
    if (fcst[["q10", "q50", "q90"]] < 0).any().any():
        raise ValueError("forecast contains negative values")
    if ((fcst["q10"] > fcst["q50"]) | (fcst["q50"] > fcst["q90"])).any():
        raise ValueError("forecast quantiles cross")
    for (model, series_id), sub in fcst.groupby(["model", "series_id"], sort=False):
        ordered = sub.sort_values("horizon")
        expected_horizons = np.arange(1, len(ordered) + 1)
        if not np.array_equal(ordered["horizon"].astype(int).to_numpy(), expected_horizons):
            raise ValueError(f"model {model}, series {series_id} has noncontiguous horizons")
        origins = ordered["date_idx"].astype(int) - ordered["horizon"].astype(int)
        if origins.nunique() != 1:
            raise ValueError(f"model {model}, series {series_id} has inconsistent forecast origin")
    if expected_keys is not None:
        expected = set(map(tuple, expected_keys[["series_id", "date_idx"]].drop_duplicates().to_numpy()))
        for model, sub in fcst.groupby("model"):
            actual = set(map(tuple, sub[["series_id", "date_idx"]].to_numpy()))
            if actual != expected:
                missing_keys = list(expected - actual)[:5]
                extra_keys = list(actual - expected)[:5]
                raise ValueError(
                    f"model {model} key coverage mismatch: missing={missing_keys}, extra={extra_keys}"
                )
