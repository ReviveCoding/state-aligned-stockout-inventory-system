from __future__ import annotations

import numpy as np
import pandas as pd

PROBABILITY_COLUMNS = ["drat_accelerating", "drat_stable", "drat_decelerating", "drat_intermittent"]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -40, 40))
    return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)


def _causal_progress(states: pd.Series) -> pd.Series:
    run = []
    previous = None
    count = 0
    for state in states.astype(str):
        if state != previous:
            count = 0
            previous = state
        else:
            count += 1
        run.append(1.0 - np.exp(-count / 21.0))
    return pd.Series(run, index=states.index, dtype=float)


def add_drat(df: pd.DataFrame, value_col: str = "recovered_demand_mean") -> pd.DataFrame:
    """Add causal Demand-Regime-Aligned Time features.

    No feature uses the future maximum date, so appending future rows cannot change
    DRAT values for historical rows.
    """
    out = df.sort_values(["series_id", "date_idx"]).copy()
    if value_col not in out:
        value_col = "sales"
    group = out.groupby("series_id", sort=False, group_keys=False)
    past = group[value_col].shift(1).fillna(0.0)
    short = past.groupby(out["series_id"], sort=False).rolling(7, min_periods=2).mean().reset_index(level=0, drop=True)
    long = past.groupby(out["series_id"], sort=False).rolling(28, min_periods=3).mean().reset_index(level=0, drop=True)
    short = short.fillna(past)
    long = long.fillna(short)
    slope = ((short - long) / (long.abs() + 1.0)).clip(-3, 3)
    zero_ratio = past.eq(0).groupby(out["series_id"], sort=False).rolling(14, min_periods=2).mean().reset_index(level=0, drop=True).fillna(0)
    volatility = past.groupby(out["series_id"], sort=False).rolling(14, min_periods=3).std().reset_index(level=0, drop=True).fillna(0)
    normalized_vol = (volatility / (long.abs() + 1.0)).clip(0, 3)
    logits = np.column_stack(
        [
            5.0 * slope - 0.5 * normalized_vol,
            -4.0 * slope.abs() - 0.6 * normalized_vol + 1.2,
            -5.0 * slope - 0.5 * normalized_vol,
            5.0 * zero_ratio - 1.0,
        ]
    )
    probabilities = _softmax(logits)
    for idx, col in enumerate(PROBABILITY_COLUMNS):
        out[col] = probabilities[:, idx]
    out["drat_entropy"] = (-(probabilities * np.log(probabilities + 1e-12)).sum(axis=1) / np.log(len(PROBABILITY_COLUMNS))).clip(0, 1)
    out["drat_velocity"] = slope
    labels = np.array(["accelerating", "stable", "decelerating", "intermittent"])
    out["drat_state"] = labels[np.argmax(probabilities, axis=1)]
    out["drat_progress"] = out.groupby("series_id", sort=False)["drat_state"].transform(
        lambda values: _causal_progress(values).to_numpy()
    )
    return out


def estimate_ladt(lifecycle_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate lifecycle states from causal demand-regime and level evidence."""
    out = add_drat(lifecycle_df)
    group = out.groupby("series_id", sort=False)
    past_level = group["sales"].shift(1)
    historical_peak = past_level.groupby(out["series_id"], sort=False).cummax().replace(0, np.nan)
    level_ratio = (past_level / historical_peak).fillna(0).clip(0, 1.5)
    prior_level = level_ratio.groupby(out["series_id"], sort=False).shift(1)
    low_memory = (
        prior_level.groupby(out["series_id"], sort=False)
        .rolling(21, min_periods=5)
        .min()
        .reset_index(level=0, drop=True)
        .fillna(1.0)
    )
    age = group.cumcount()
    early = age.lt(14)
    reactivation = (
        low_memory.lt(0.28)
        & level_ratio.gt(0.24)
        & out["drat_accelerating"].gt(0.38)
        & out["drat_velocity"].gt(0.02)
        & ~early
    )
    dormant = level_ratio.lt(0.24) & age.gt(35) & ~reactivation
    decline = (
        level_ratio.between(0.24, 0.88)
        & out["drat_decelerating"].gt(0.34)
        & out["drat_velocity"].lt(-0.015)
        & ~reactivation
    )
    growth = (
        out["drat_accelerating"].gt(0.34)
        & out["drat_velocity"].gt(0.012)
        & ~early
        & ~reactivation
        & ~dormant
    )
    state = np.full(len(out), "mature", dtype=object)
    state[early.to_numpy()] = "intro"
    state[growth.to_numpy()] = "growth"
    state[decline.to_numpy()] = "decline"
    state[dormant.to_numpy()] = "dormant"
    state[reactivation.to_numpy()] = "reactivation"
    out["estimated_lifecycle_state"] = state
    state_base = pd.Series(state, index=out.index).map(
        {"intro": 0.0, "growth": 0.18, "mature": 0.42, "decline": 0.72, "dormant": 0.90, "reactivation": 0.55}
    )
    state_width = pd.Series(state, index=out.index).map(
        {"intro": 0.18, "growth": 0.24, "mature": 0.30, "decline": 0.18, "dormant": 0.10, "reactivation": 0.20}
    )
    out["estimated_ladt"] = (state_base + state_width * out["drat_progress"]).clip(0, 1)
    return out
