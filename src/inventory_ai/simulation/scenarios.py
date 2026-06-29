from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

_NORMAL_Q90 = 1.2815515655446004


def generate_correlated_scenarios(
    forecast: pd.DataFrame,
    n_scenarios: int = 100,
    seed: int = 42,
    temporal_rho: float = 0.65,
) -> pd.DataFrame:
    """Create AR(1)-correlated demand paths preserving q10/q50/q90 anchors.

    For negative and positive Gaussian shocks, separate local scales are inferred
    from the q10-q50 and q50-q90 distances. Therefore z=-1.28155, 0, and
    +1.28155 map exactly to the supplied 10th, 50th, and 90th quantiles before
    the nonnegative demand projection.
    """
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be at least 1")
    if not -0.99 < temporal_rho < 0.99:
        raise ValueError("temporal_rho must be between -0.99 and 0.99")
    rows: list[dict] = []
    for (model, series_id), sub in forecast.groupby(["model", "series_id"], sort=True):
        sub = sub.sort_values("horizon")
        steps = len(sub)
        digest = hashlib.sha256(f"{seed}|{model}|{series_id}".encode("utf-8")).digest()
        group_seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(group_seed)
        shocks = rng.normal(size=(n_scenarios, steps))
        innovation_scale = np.sqrt(1.0 - temporal_rho**2)
        for step in range(1, steps):
            shocks[:, step] = temporal_rho * shocks[:, step - 1] + innovation_scale * shocks[:, step]
        q10 = sub["q10"].to_numpy(dtype=float)
        q50 = sub["q50"].to_numpy(dtype=float)
        q90 = sub["q90"].to_numpy(dtype=float)
        lower_scale = np.maximum(1e-6, (q50 - q10) / _NORMAL_Q90)
        upper_scale = np.maximum(1e-6, (q90 - q50) / _NORMAL_Q90)
        values = q50[None, :] + np.where(shocks < 0, shocks * lower_scale[None, :], shocks * upper_scale[None, :])
        values = np.maximum(0.0, values)
        for scenario in range(n_scenarios):
            rows.extend(
                {
                    "model": model,
                    "series_id": series_id,
                    "scenario": scenario,
                    "date_idx": int(date_idx),
                    "horizon": int(horizon),
                    "demand": float(value),
                    "q10_forecast": float(low),
                    "median_forecast": float(median),
                    "q90_forecast": float(high),
                    "interval_scale": float(max(0.5, high - low)),
                }
                for date_idx, horizon, value, low, median, high in zip(
                    sub["date_idx"], sub["horizon"], values[scenario], q10, q50, q90
                )
            )
    return pd.DataFrame(rows)


def scenario_diagnostics(scenarios: pd.DataFrame) -> dict[str, float | int]:
    if scenarios.empty:
        return {
            "n_scenarios": 0,
            "mean_lag1_correlation": 0.0,
            "negative_values": 0,
            "normalized_quantile_mae": 1.0,
        }
    correlations = []
    for (_, _, scenario), sub in scenarios.groupby(["model", "series_id", "scenario"]):
        ordered = sub.sort_values("horizon")
        values = (ordered["demand"] - ordered["median_forecast"]) / ordered["interval_scale"].clip(lower=0.5)
        if len(values) > 2:
            left = values.to_numpy()[:-1]
            right = values.to_numpy()[1:]
            if left.std() > 1e-12 and right.std() > 1e-12:
                corr = float(np.corrcoef(left, right)[0, 1])
                if np.isfinite(corr):
                    correlations.append(corr)
    empirical = (
        scenarios.groupby(["model", "series_id", "date_idx"])["demand"]
        .quantile([0.1, 0.5, 0.9])
        .rename("empirical")
        .reset_index()
        .rename(columns={"level_3": "quantile"})
    )
    pivot = empirical.pivot(
        index=["model", "series_id", "date_idx"], columns="quantile", values="empirical"
    ).reset_index()
    anchors = scenarios.groupby(["model", "series_id", "date_idx"], as_index=False).first()
    merged = anchors.merge(pivot, on=["model", "series_id", "date_idx"], how="inner")
    errors = []
    for quantile, forecast_col in [(0.1, "q10_forecast"), (0.5, "median_forecast"), (0.9, "q90_forecast")]:
        if quantile in merged.columns:
            errors.append(
                (merged[quantile] - merged[forecast_col]).abs()
                / merged["interval_scale"].clip(lower=1.0)
            )
    normalized_quantile_mae = float(pd.concat(errors, ignore_index=True).mean()) if errors else 1.0
    return {
        "n_scenarios": int(scenarios["scenario"].nunique()),
        "mean_lag1_correlation": float(np.mean(correlations)) if correlations else 0.0,
        "negative_values": int((scenarios["demand"] < 0).sum()),
        "normalized_quantile_mae": normalized_quantile_mae,
    }
