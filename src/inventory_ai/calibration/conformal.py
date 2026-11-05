from __future__ import annotations

import numpy as np
import pandas as pd


def repair_quantiles(forecast: pd.DataFrame) -> pd.DataFrame:
    """Repair interval ordering while preserving the median point forecast.

    Sorting all three quantiles can silently replace q50 with a tail estimate.
    The median is the operational point forecast, so only the interval bounds
    are projected around it.
    """
    repaired_forecast = forecast.copy()
    median_forecast = np.maximum(0.0, repaired_forecast["q50"].to_numpy(dtype=float))
    lower_quantile = np.maximum(0.0, repaired_forecast["q10"].to_numpy(dtype=float))
    upper_quantile = np.maximum(0.0, repaired_forecast["q90"].to_numpy(dtype=float))
    repaired_forecast["q50"] = median_forecast
    repaired_forecast["q10"] = np.minimum(lower_quantile, median_forecast)
    repaired_forecast["q90"] = np.maximum(upper_quantile, median_forecast)
    return repaired_forecast


def _finite_sample_quantile(values: np.ndarray, target_coverage: float) -> float:
    if len(values) == 0:
        return 0.0
    rank = min(len(values) - 1, int(np.ceil((len(values) + 1) * target_coverage)) - 1)
    return float(np.partition(values, rank)[rank])


def fit_conformal_adjustments(
    validation_forecasts: pd.DataFrame,
    validation_truth: pd.DataFrame,
    target_coverage: float = 0.80,
    shrinkage_strength: float = 30.0,
) -> pd.DataFrame:
    """Fit signed conformalized quantile regression adjustments by model and horizon."""
    scored_validation = validation_forecasts.merge(
        validation_truth[["series_id", "date_idx", "demand"]],
        on=["series_id", "date_idx"],
        how="inner",
        validate="many_to_one",
    )
    # Signed CQR scores are negative when an observation lies well inside the
    # base interval. Preserving that sign allows a systematically over-wide
    # interval to shrink instead of making calibration expansion-only.
    scored_validation["nonconformity"] = np.maximum(
        scored_validation["q10"].to_numpy() - scored_validation["demand"].to_numpy(),
        scored_validation["demand"].to_numpy() - scored_validation["q90"].to_numpy(),
    )
    pooled_adjustments = {
        model: _finite_sample_quantile(sub["nonconformity"].to_numpy(), target_coverage)
        for model, sub in scored_validation.groupby("model")
    }
    rows = []
    for (model, horizon), sub in scored_validation.groupby(["model", "horizon"]):
        nonconformity_scores = sub["nonconformity"].to_numpy()
        local_adjustment = _finite_sample_quantile(
            nonconformity_scores, target_coverage
        )
        pooled_adjustment = pooled_adjustments[model]
        local_weight = float(
            len(nonconformity_scores)
            / (len(nonconformity_scores) + max(shrinkage_strength, 0.0))
        )
        adjustment = float(
            local_weight * local_adjustment + (1.0 - local_weight) * pooled_adjustment
        )
        rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "adjustment": adjustment,
                "local_adjustment": local_adjustment,
                "pooled_adjustment": pooled_adjustment,
                "shrinkage_weight": local_weight,
                "n": int(len(nonconformity_scores)),
            }
        )
    return pd.DataFrame(rows)


def apply_conformal_adjustments(
    forecast: pd.DataFrame, adjustments: pd.DataFrame
) -> pd.DataFrame:
    adjusted_forecast = forecast.merge(
        adjustments[["model", "horizon", "adjustment"]],
        on=["model", "horizon"],
        how="left",
    )
    adjusted_forecast["adjustment"] = adjusted_forecast["adjustment"].fillna(0.0)
    adjusted_forecast["q10"] = np.maximum(
        0.0, adjusted_forecast["q10"] - adjusted_forecast["adjustment"]
    )
    adjusted_forecast["q90"] = np.maximum(
        adjusted_forecast["q50"],
        adjusted_forecast["q90"] + adjusted_forecast["adjustment"],
    )
    return repair_quantiles(adjusted_forecast.drop(columns="adjustment"))
