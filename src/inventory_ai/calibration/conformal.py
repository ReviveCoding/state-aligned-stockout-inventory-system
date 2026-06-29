from __future__ import annotations

import numpy as np
import pandas as pd


def repair_quantiles(forecast: pd.DataFrame) -> pd.DataFrame:
    """Repair interval ordering while preserving the median point forecast.

    Sorting all three quantiles can silently replace q50 with a tail estimate.
    The median is the operational point forecast, so only the interval bounds
    are projected around it.
    """
    out = forecast.copy()
    q50 = np.maximum(0.0, out["q50"].to_numpy(dtype=float))
    q10 = np.maximum(0.0, out["q10"].to_numpy(dtype=float))
    q90 = np.maximum(0.0, out["q90"].to_numpy(dtype=float))
    out["q50"] = q50
    out["q10"] = np.minimum(q10, q50)
    out["q90"] = np.maximum(q90, q50)
    return out


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
    merged = validation_forecasts.merge(
        validation_truth[["series_id", "date_idx", "demand"]],
        on=["series_id", "date_idx"],
        how="inner",
        validate="many_to_one",
    )
    # Signed CQR scores are negative when an observation lies well inside the
    # base interval. Preserving that sign allows a systematically over-wide
    # interval to shrink instead of making calibration expansion-only.
    merged["nonconformity"] = np.maximum(
        merged["q10"].to_numpy() - merged["demand"].to_numpy(),
        merged["demand"].to_numpy() - merged["q90"].to_numpy(),
    )
    pooled = {
        model: _finite_sample_quantile(sub["nonconformity"].to_numpy(), target_coverage)
        for model, sub in merged.groupby("model")
    }
    rows = []
    for (model, horizon), sub in merged.groupby(["model", "horizon"]):
        values = sub["nonconformity"].to_numpy()
        local_adjustment = _finite_sample_quantile(values, target_coverage)
        pooled_adjustment = pooled[model]
        weight = float(len(values) / (len(values) + max(shrinkage_strength, 0.0)))
        adjustment = float(weight * local_adjustment + (1.0 - weight) * pooled_adjustment)
        rows.append({
            "model": model,
            "horizon": int(horizon),
            "adjustment": adjustment,
            "local_adjustment": local_adjustment,
            "pooled_adjustment": pooled_adjustment,
            "shrinkage_weight": weight,
            "n": int(len(values)),
        })
    return pd.DataFrame(rows)


def apply_conformal_adjustments(forecast: pd.DataFrame, adjustments: pd.DataFrame) -> pd.DataFrame:
    out = forecast.merge(adjustments[["model", "horizon", "adjustment"]], on=["model", "horizon"], how="left")
    out["adjustment"] = out["adjustment"].fillna(0.0)
    out["q10"] = np.maximum(0.0, out["q10"] - out["adjustment"])
    out["q90"] = np.maximum(out["q50"], out["q90"] + out["adjustment"])
    return repair_quantiles(out.drop(columns="adjustment"))
