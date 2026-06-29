from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, value_col: str = "sales") -> pd.DataFrame:
    """Add strictly past-only lag and rolling features."""
    out = df.sort_values(["series_id", "date_idx"]).copy()
    group = out.groupby("series_id", sort=False, group_keys=False)
    shifted = group[value_col].shift(1)
    for lag in (1, 7, 14, 28):
        out[f"lag_{lag}"] = group[value_col].shift(lag)
    for window in (7, 14, 28):
        roll = shifted.groupby(out["series_id"], sort=False).rolling(window, min_periods=2)
        out[f"roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        out[f"roll_std_{window}"] = roll.std().reset_index(level=0, drop=True)
    out["zero_ratio_14"] = (
        shifted.eq(0)
        .groupby(out["series_id"], sort=False)
        .rolling(14, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    previous_price = group["price"].shift(1)
    out["price_change"] = ((out["price"] - previous_price) / previous_price.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    out["series_age"] = group.cumcount().astype(float)
    numeric = [c for c in out.columns if c.startswith("lag_") or c.startswith("roll_")]
    numeric += ["zero_ratio_14", "price_change", "series_age"]
    out[numeric] = out[numeric].fillna(0.0)
    return out


def build_direct_training_frame(
    featured: pd.DataFrame,
    cutoff: int,
    max_horizon: int,
    target_col: str = "recovered_demand_mean",
    max_rows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create origin-based direct multi-horizon examples without future-target leakage."""
    base = featured[featured["date_idx"] <= cutoff].sort_values(["series_id", "date_idx"]).copy()
    pieces = []
    group = base.groupby("series_id", sort=False)
    for horizon in range(1, max_horizon + 1):
        part = base.copy()
        part["horizon"] = horizon
        part["target"] = group[target_col].shift(-horizon)
        part["future_dow"] = group["dow"].shift(-horizon)
        part["future_price"] = group["price"].shift(-horizon)
        part["future_promo"] = group["promo"].shift(-horizon)
        part = part[part["target"].notna()].copy()
        pieces.append(part)
    training = pd.concat(pieces, ignore_index=True)
    if max_rows and len(training) > max_rows:
        training = training.sample(max_rows, random_state=seed).sort_values(["series_id", "date_idx", "horizon"])
    return training.reset_index(drop=True)


def build_origin_frame(featured: pd.DataFrame, origin: int, horizon: int) -> pd.DataFrame:
    """Build one forecast row per series and horizon using known-future covariates only."""
    origin_rows = featured[featured["date_idx"] == origin].copy()
    if origin_rows["series_id"].nunique() != featured["series_id"].nunique():
        raise ValueError("origin is not available for every series")
    future = featured[(featured["date_idx"] > origin) & (featured["date_idx"] <= origin + horizon)][
        ["series_id", "date_idx", "dow", "price", "promo"]
    ].copy()
    future["horizon"] = future["date_idx"] - origin
    future = future.rename(columns={"dow": "future_dow", "price": "future_price", "promo": "future_promo"})
    grid = origin_rows.assign(_key=1).merge(
        pd.DataFrame({"horizon": range(1, horizon + 1), "_key": 1}), on="_key"
    ).drop(columns="_key")
    grid["date_idx"] = origin + grid["horizon"]
    grid = grid.drop(columns=["dow", "price", "promo"]).merge(
        future,
        on=["series_id", "date_idx", "horizon"],
        how="left",
        validate="one_to_one",
    )
    for col, fallback in [("future_dow", (origin_rows.set_index("series_id")["dow"] + grid["horizon"]) % 7),
                          ("future_price", origin_rows.set_index("series_id")["price"]),
                          ("future_promo", 0)]:
        if col == "future_dow":
            base_map = origin_rows.set_index("series_id")["dow"]
            grid[col] = grid[col].fillna((grid["series_id"].map(base_map) + grid["horizon"]) % 7)
        elif col == "future_price":
            base_map = origin_rows.set_index("series_id")["price"]
            grid[col] = grid[col].fillna(grid["series_id"].map(base_map))
        else:
            grid[col] = grid[col].fillna(fallback)
    return grid.sort_values(["series_id", "horizon"]).reset_index(drop=True)
