from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _select_diverse_active_series(
    sales: pd.DataFrame,
    day_cols: list[str],
    n_series: int,
    series_offset: int = 0,
) -> pd.DataFrame:
    """Select a deterministic, diverse, non-overlapping M5 cohort window."""
    if n_series < 1:
        raise ValueError("n_series must be positive")
    if series_offset < 0:
        raise ValueError("series_offset must be nonnegative")

    required_count = n_series + series_offset
    recent_cols = day_cols[-28:]
    score = sales[recent_cols].sum(axis=1).astype(float)
    nonzero = sales[recent_cols].gt(0).mean(axis=1)
    candidates = sales.assign(_score=score, _nonzero=nonzero)
    candidates = candidates[candidates["_score"] > 0].copy()

    if len(candidates) < required_count:
        candidates = sales.assign(_score=score, _nonzero=nonzero).copy()

    if len(candidates) < required_count:
        raise ValueError(
            f"requested cohort window requires {required_count} series, "
            f"but only {len(candidates)} candidates are available"
        )

    candidates["_volume_bin"] = pd.qcut(
        candidates["_score"].rank(method="first"),
        q=min(4, max(1, len(candidates))),
        labels=False,
        duplicates="drop",
    )

    candidates = candidates.sort_values(
        ["cat_id", "store_id", "_volume_bin", "_score"],
        ascending=[True, True, True, False],
    )

    group_cols = ["cat_id", "store_id", "_volume_bin"]
    picked = candidates.groupby(group_cols, group_keys=False).head(1)

    if len(picked) < required_count:
        remaining = candidates.loc[
            ~candidates.index.isin(picked.index)
        ].sort_values("_score", ascending=False)
        picked = pd.concat([picked, remaining.head(required_count - len(picked))])

    if len(picked) < required_count:
        raise ValueError(
            f"unable to construct deterministic cohort window of {required_count} series"
        )

    cohort = picked.head(required_count).iloc[series_offset : series_offset + n_series].copy()

    if len(cohort) != n_series:
        raise ValueError(
            f"cohort size mismatch: expected {n_series}, received {len(cohort)}"
        )

    return cohort.drop(
        columns=["_score", "_nonzero", "_volume_bin"],
        errors="ignore",
    )


def impute_prices_causally(frame: pd.DataFrame) -> pd.Series:
    """Impute M5 prices without borrowing information from future weeks.

    The ordering is: own-series forward fill, contemporaneous store/week
    median, contemporaneous item/week median, contemporaneous global week
    median, then a neutral positive fallback. No backward fill or full-history
    statistic is used.
    """
    required = {"id", "store_id", "item_id", "wm_yr_wk", "date_idx", "sell_price"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"price imputation frame is missing columns: {missing}")
    ordered = frame.sort_values(["id", "date_idx"]).copy()
    price = ordered.groupby("id", sort=False)["sell_price"].ffill()
    store_week = ordered.groupby(["store_id", "wm_yr_wk"])["sell_price"].transform("median")
    item_week = ordered.groupby(["item_id", "wm_yr_wk"])["sell_price"].transform("median")
    week = ordered.groupby("wm_yr_wk")["sell_price"].transform("median")
    price = price.fillna(store_week).fillna(item_week).fillna(week).fillna(1.0).clip(lower=0.01)
    result = pd.Series(price.to_numpy(), index=ordered.index, dtype=float)
    return result.reindex(frame.index)

def load_m5_sample(
    zip_path: str | Path,
    n_series: int = 30,
    history_days: int = 140,
    series_offset: int = 0,
) -> pd.DataFrame:
    path = Path(zip_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        with archive.open("sales_train_evaluation.csv") as handle:
            header = pd.read_csv(handle, nrows=0).columns.tolist()
        day_cols = [c for c in header if c.startswith("d_")]
        selected_days = day_cols[-history_days:]
        base_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        with archive.open("sales_train_evaluation.csv") as handle:
            sales = pd.read_csv(handle, usecols=base_cols + selected_days)
        sales = _select_diverse_active_series(
            sales,
            selected_days,
            n_series,
            series_offset=series_offset,
        )
        item_store = set(zip(sales["item_id"], sales["store_id"]))
        with archive.open("calendar.csv") as handle:
            calendar = pd.read_csv(
                handle,
                usecols=["d", "wm_yr_wk", "wday", "event_name_1", "snap_CA", "snap_TX", "snap_WI"],
            )
        price_parts = []
        with archive.open("sell_prices.csv") as handle:
            for chunk in pd.read_csv(handle, chunksize=300_000):
                keep = [pair in item_store for pair in zip(chunk["item_id"], chunk["store_id"])]
                if any(keep):
                    price_parts.append(chunk.loc[keep])
        prices = pd.concat(price_parts, ignore_index=True) if price_parts else pd.DataFrame(
            columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]
        )
    long = sales[base_cols + selected_days].melt(
        id_vars=base_cols,
        var_name="d",
        value_name="sales",
    )
    long = long.merge(calendar, on="d", how="left", validate="many_to_one")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one")
    long["date_idx"] = long["d"].str[2:].astype(int)
    min_day = long.groupby("id")["date_idx"].transform("min")
    long["date_idx"] = long["date_idx"] - min_day
    state_snap = np.select(
        [long["state_id"].eq("CA"), long["state_id"].eq("TX"), long["state_id"].eq("WI")],
        [long["snap_CA"], long["snap_TX"], long["snap_WI"]],
        default=0,
    )
    long["promo"] = (long["event_name_1"].notna() | pd.Series(state_snap, index=long.index).fillna(0).eq(1)).astype(int)
    long["price"] = impute_prices_causally(long)
    long["dow"] = (long["wday"].fillna(1).astype(int) - 1).clip(0, 6)
    long["stockout"] = 0
    long["demand"] = long["sales"].astype(float)
    long = long.rename(columns={"id": "series_id", "cat_id": "category", "store_id": "store"})
    columns = ["series_id", "date_idx", "sales", "demand", "stockout", "price", "promo", "category", "store", "dow"]
    return long[columns].sort_values(["series_id", "date_idx"]).reset_index(drop=True)
