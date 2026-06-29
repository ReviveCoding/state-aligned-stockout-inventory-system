from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _stable_uniforms(df: pd.DataFrame, seed: int, stream: int) -> np.ndarray:
    """Deterministic row-level uniforms invariant to row order and future appends."""
    keys = pd.util.hash_pandas_object(
        df[["series_id", "date_idx"]].astype({"series_id": str}), index=False
    ).to_numpy(dtype=np.uint64)
    x = keys ^ np.uint64(seed) ^ np.uint64(stream * 0x9E3779B1)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return ((x >> np.uint64(11)).astype(np.float64)) / float(1 << 53)


def add_controlled_stockouts(df: pd.DataFrame, seed: int = 42, rate: float = 0.12) -> pd.DataFrame:
    """Apply demand-dependent censoring while preserving known latent truth."""
    out = df.sort_values(["series_id", "date_idx"]).copy()
    out["sales"] = out["sales"].astype(float)
    out["demand"] = out["demand"].astype(float)
    original = out["demand"].to_numpy().copy()
    group = out.groupby("series_id", sort=False, group_keys=False)
    past_demand = group["demand"].shift(1)
    causal_reference = (
        past_demand.groupby(out["series_id"], sort=False)
        .rolling(28, min_periods=3)
        .median()
        .reset_index(level=0, drop=True)
    )
    expanding_reference = (
        past_demand.groupby(out["series_id"], sort=False)
        .expanding(min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )
    causal_reference = causal_reference.fillna(expanding_reference).fillna(out["demand"]).clip(lower=0.1)
    surge = out["demand"].gt(1.25 * causal_reference).astype(float)
    probability = (rate + 0.16 * out["promo"].astype(float) + 0.12 * surge).clip(0, 0.75)
    event_uniform = _stable_uniforms(out, seed=seed, stream=1)
    capacity_uniform = _stable_uniforms(out, seed=seed, stream=2)
    mask = (event_uniform < probability.to_numpy()) & (original > 0)
    capacity = original * (0.35 + (0.82 - 0.35) * capacity_uniform)
    out["controlled_true_demand"] = original
    out["controlled_stockout"] = mask.astype(int)
    out.loc[mask, "sales"] = np.minimum(out.loc[mask, "sales"].astype(float), capacity[mask])
    out["stockout"] = np.maximum(out["stockout"].astype(int), mask.astype(int))
    return out


def recover_latent_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only, uncertainty-aware latent-demand recovery baseline."""
    out = df.sort_values(["series_id", "date_idx"]).copy()
    group = out.groupby("series_id", sort=False, group_keys=False)
    past = group["sales"].shift(1)
    seasonal = group["sales"].shift(7)
    rolling_median = past.groupby(out["series_id"], sort=False).rolling(14, min_periods=3).median().reset_index(level=0, drop=True)
    rolling_mean = past.groupby(out["series_id"], sort=False).rolling(28, min_periods=4).mean().reset_index(level=0, drop=True)
    rolling_std = past.groupby(out["series_id"], sort=False).rolling(28, min_periods=4).std().reset_index(level=0, drop=True)
    baseline = pd.concat([seasonal, rolling_median, rolling_mean], axis=1).median(axis=1, skipna=True).fillna(out["sales"])
    promo_multiplier = 1.0 + 0.18 * out["promo"].astype(float)
    candidate = np.maximum(out["sales"].astype(float), baseline.clip(lower=0) * promo_multiplier)
    is_censored = out["stockout"].astype(int).eq(1)
    out["recovered_demand_mean"] = np.where(is_censored, candidate, out["sales"].astype(float))
    gap = (out["recovered_demand_mean"] - out["sales"]).clip(lower=0)
    uncertainty = rolling_std.fillna(0.25 * baseline.abs()).clip(lower=0.5)
    out["recovery_uncertainty"] = np.where(is_censored, uncertainty + 0.35 * gap, 0.05 * out["sales"].clip(lower=1))
    out["recovered_q80"] = np.maximum(out["recovered_demand_mean"], out["recovered_demand_mean"] + 0.84 * out["recovery_uncertainty"])
    out["recovered_q95"] = np.maximum(out["recovered_q80"], out["recovered_demand_mean"] + 1.64 * out["recovery_uncertainty"])
    out["recovery_confidence"] = (1.0 / (1.0 + out["recovery_uncertainty"] / (out["recovered_demand_mean"] + 1.0))).clip(0, 1)
    return out


def sample_recovery_posterior(
    df: pd.DataFrame,
    n_draws: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Draw nonnegative latent-demand trajectories for censored observations.

    Non-stockout rows are intentionally omitted because their posterior is a
    point mass at observed sales in this baseline. Stockout draws are truncated
    at observed sales, preserving the censoring lower-bound contract.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    censored = (
        df[df["stockout"].astype(int).eq(1)]
        .sort_values(["series_id", "date_idx"])
        .reset_index(drop=True)
    )
    if censored.empty:
        return pd.DataFrame(columns=["series_id", "date_idx", "draw", "latent_demand_draw"])
    rows: list[dict] = []
    for row in censored.itertuples(index=False):
        digest = hashlib.sha256(
            f"{seed}|{row.series_id}|{int(row.date_idx)}".encode("utf-8")
        ).digest()
        row_seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(row_seed)
        draws = rng.normal(
            float(row.recovered_demand_mean),
            max(float(row.recovery_uncertainty), 1e-6),
            size=n_draws,
        )
        draws = np.maximum(float(row.sales), draws)
        rows.extend(
            {
                "series_id": row.series_id,
                "date_idx": int(row.date_idx),
                "draw": int(draw),
                "latent_demand_draw": float(value),
            }
            for draw, value in enumerate(draws)
        )
    return pd.DataFrame(rows)


def recovery_diagnostics(df: pd.DataFrame) -> dict[str, float | int | None]:
    mask = df.get("controlled_stockout", pd.Series(0, index=df.index)).astype(int).eq(1)
    if not mask.any() or "controlled_true_demand" not in df:
        return {"n_controlled": 0, "raw_mae": None, "recovered_mae": None, "raw_bias": None, "recovered_bias": None}
    truth = df.loc[mask, "controlled_true_demand"].astype(float)
    raw = df.loc[mask, "sales"].astype(float)
    recovered = df.loc[mask, "recovered_demand_mean"].astype(float)
    raw_mae = float((raw - truth).abs().mean())
    recovered_mae = float((recovered - truth).abs().mean())
    raw_bias = float((raw - truth).mean())
    recovered_bias = float((recovered - truth).mean())
    return {
        "n_controlled": int(mask.sum()),
        "raw_mae": raw_mae,
        "recovered_mae": recovered_mae,
        "raw_bias": raw_bias,
        "recovered_bias": recovered_bias,
        "mae_improvement": float((raw_mae - recovered_mae) / (raw_mae + 1e-9)),
        "absolute_bias_improvement": float((abs(raw_bias) - abs(recovered_bias)) / (abs(raw_bias) + 1e-9)),
        "q80_upper_coverage": float((truth <= df.loc[mask, "recovered_q80"].astype(float)).mean()),
        "q95_upper_coverage": float((truth <= df.loc[mask, "recovered_q95"].astype(float)).mean()),
        "do_no_harm_max": float(
            (df.loc[df["stockout"].astype(int).eq(0), "recovered_demand_mean"]
             - df.loc[df["stockout"].astype(int).eq(0), "sales"]).abs().max()
        ) if df["stockout"].astype(int).eq(0).any() else 0.0,
    }
