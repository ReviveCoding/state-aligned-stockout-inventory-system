from __future__ import annotations

import pandas as pd

from inventory_ai.calibration.conformal import apply_conformal_adjustments, fit_conformal_adjustments
from inventory_ai.censoring.recovery import add_controlled_stockouts, recover_latent_demand
from inventory_ai.contracts import validate_forecast
from inventory_ai.data.synthetic import SyntheticConfig, make_synthetic_retail
from inventory_ai.features.basic import add_time_features
from inventory_ai.lifecycle.drat import add_drat
from inventory_ai.models.baselines import seasonal_naive_forecast, tsb_forecast
from inventory_ai.models.quantile import DirectQuantileForecaster


def _panel():
    frame = make_synthetic_retail(SyntheticConfig(6, 84, 15))
    frame = recover_latent_demand(add_controlled_stockouts(frame, seed=16, rate=0.15))
    return add_drat(add_time_features(frame, value_col="recovered_demand_mean"))


def test_all_forecasters_cover_same_keys():
    panel = _panel()
    origin = 70
    truth = panel[panel["date_idx"].between(71, 77)]
    naive = seasonal_naive_forecast(panel, origin, 7)
    tsb = tsb_forecast(panel, origin, 7)
    model = DirectQuantileForecaster(max_iter=8, random_state=1).fit(panel, origin, 7, 2500)
    advanced = model.predict(panel, origin, 7)
    combined = pd.concat([naive, tsb, advanced], ignore_index=True)
    validate_forecast(combined, truth)
    assert combined.groupby("model").size().nunique() == 1


def test_conformal_adjustment_preserves_quantile_order():
    panel = _panel()
    origin = 63
    truth = panel[panel["date_idx"].between(64, 70)]
    forecast = seasonal_naive_forecast(panel, origin, 7)
    adjustment = fit_conformal_adjustments(forecast, truth)
    calibrated = apply_conformal_adjustments(forecast, adjustment)
    assert (calibrated["q10"] <= calibrated["q50"]).all()
    assert (calibrated["q50"] <= calibrated["q90"]).all()


def test_signed_conformal_score_can_shrink_overwide_intervals():
    from inventory_ai.calibration.conformal import apply_conformal_adjustments, fit_conformal_adjustments

    forecasts = pd.DataFrame({
        "series_id": ["a"] * 5,
        "date_idx": [1, 2, 3, 4, 5],
        "horizon": [1] * 5,
        "q10": [0.0] * 5,
        "q50": [10.0] * 5,
        "q90": [20.0] * 5,
        "model": ["wide"] * 5,
    })
    truth = pd.DataFrame({
        "series_id": ["a"] * 5,
        "date_idx": [1, 2, 3, 4, 5],
        "demand": [9.0, 10.0, 11.0, 10.0, 9.5],
    })
    adjustments = fit_conformal_adjustments(forecasts, truth)
    assert float(adjustments.loc[0, "adjustment"]) < 0.0
    calibrated = apply_conformal_adjustments(forecasts, adjustments)
    assert calibrated["q10"].mean() > forecasts["q10"].mean()
    assert calibrated["q90"].mean() < forecasts["q90"].mean()


def test_quantile_repair_preserves_median_point_forecast():
    from inventory_ai.calibration.conformal import repair_quantiles

    frame = pd.DataFrame({
        "q10": [12.0, 1.0],
        "q50": [10.0, 5.0],
        "q90": [8.0, 4.0],
    })
    repaired = repair_quantiles(frame)
    assert repaired["q50"].tolist() == [10.0, 5.0]
    assert (repaired["q10"] <= repaired["q50"]).all()
    assert (repaired["q50"] <= repaired["q90"]).all()


def test_candidate_selector_uses_robustness_inside_cost_indifference_band():
    from inventory_ai.evaluation.metrics import choose_candidate

    metrics = {
        "seasonal_naive": {"wape": 0.30, "coverage_80": 0.80, "signed_bias": 0.0},
        "tsb": {"wape": 0.28, "coverage_80": 0.80, "signed_bias": 0.0},
        "reliability_router": {"wape": 0.27, "coverage_80": 0.80, "signed_bias": 0.0},
    }
    inventory = {
        "seasonal_naive": {"total_cost": 102.0},
        "tsb": {"total_cost": 100.0},
        "reliability_router": {"total_cost": 104.0},
    }
    slices = {
        "seasonal_naive": {"worst_relative_wape_regression": 0.0},
        "tsb": {"worst_relative_wape_regression": 0.12},
        "reliability_router": {"worst_relative_wape_regression": -0.01},
    }
    selected = choose_candidate(
        metrics,
        inventory,
        slice_summaries=slices,
        inventory_cost_tolerance=0.05,
    )
    assert selected == "reliability_router"


def test_candidate_selector_keeps_safe_baseline_inside_cost_band():
    from inventory_ai.evaluation.metrics import choose_candidate

    metrics = {
        "seasonal_naive": {"wape": 0.30, "coverage_80": 0.80, "signed_bias": 0.0},
        "tsb": {"wape": 0.28, "coverage_80": 0.80, "signed_bias": 0.0},
        "reliability_router": {"wape": 0.27, "coverage_80": 0.80, "signed_bias": 0.0},
    }
    inventory = {
        "seasonal_naive": {"total_cost": 102.0},
        "tsb": {"total_cost": 100.0},
        "reliability_router": {"total_cost": 108.0},
    }
    slices = {
        "seasonal_naive": {"worst_relative_wape_regression": 0.0},
        "tsb": {"worst_relative_wape_regression": 0.12},
        "reliability_router": {"worst_relative_wape_regression": -0.01},
    }
    selected = choose_candidate(
        metrics,
        inventory,
        slice_summaries=slices,
        inventory_cost_tolerance=0.05,
    )
    assert selected == "seasonal_naive"
