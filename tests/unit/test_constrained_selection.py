from __future__ import annotations

import pandas as pd

from inventory_ai.evaluation.metrics import choose_candidate
from inventory_ai.simulation.inventory import best_tuned_policy_by_model


def test_constrained_selection_rejects_low_service_and_overwide_router() -> None:
    metrics = {
        "seasonal_naive": {
            "wape": 0.40,
            "coverage_80": 0.80,
            "signed_bias": 0.0,
            "mean_interval_width": 10.0,
        },
        "reliability_router": {
            "wape": 0.35,
            "coverage_80": 0.80,
            "signed_bias": 0.0,
            "mean_interval_width": 14.0,
        },
        "tsb": {
            "wape": 0.37,
            "coverage_80": 0.80,
            "signed_bias": 0.0,
            "mean_interval_width": 10.5,
        },
    }

    inventory = {
        "seasonal_naive": {
            "total_cost": 100.0,
            "fill_rate": 0.95,
            "cost_win_rate_vs_baseline": 1.0,
        },
        "reliability_router": {
            "total_cost": 70.0,
            "fill_rate": 0.94,
            "cost_win_rate_vs_baseline": 0.95,
        },
        "tsb": {
            "total_cost": 80.0,
            "fill_rate": 0.952,
            "cost_win_rate_vs_baseline": 0.80,
        },
    }

    slices = {
        "seasonal_naive": {"worst_relative_wape_regression": 0.0},
        "reliability_router": {"worst_relative_wape_regression": 0.02},
        "tsb": {"worst_relative_wape_regression": 0.01},
    }

    selected = choose_candidate(
        metrics,
        inventory,
        slice_summaries=slices,
        max_fill_rate_degradation=0.0,
        max_interval_width_ratio=1.35,
    )

    assert selected == "tsb"


def test_policy_selection_uses_independently_tuned_baseline_fill_floor() -> None:
    tuning = pd.DataFrame(
        [
            {
                "model": "seasonal_naive",
                "policy_scale": 0.95,
                "meets_fill_guardrail": True,
                "total_cost": 100.0,
                "fill_rate": 0.950,
                "lost_sales": 10.0,
                "stockout_rate": 0.05,
                "average_inventory": 20.0,
                "order_volatility": 2.0,
            },
            {
                "model": "reliability_router",
                "policy_scale": 0.80,
                "meets_fill_guardrail": True,
                "total_cost": 70.0,
                "fill_rate": 0.949,
                "lost_sales": 20.0,
                "stockout_rate": 0.06,
                "average_inventory": 15.0,
                "order_volatility": 2.0,
            },
            {
                "model": "reliability_router",
                "policy_scale": 0.85,
                "meets_fill_guardrail": True,
                "total_cost": 80.0,
                "fill_rate": 0.951,
                "lost_sales": 15.0,
                "stockout_rate": 0.05,
                "average_inventory": 16.0,
                "order_volatility": 2.0,
            },
        ]
    )

    selected = best_tuned_policy_by_model(
        tuning,
        max_fill_rate_degradation=0.0,
    )

    assert selected["seasonal_naive"]["policy_scale"] == 0.95
    assert selected["reliability_router"]["policy_scale"] == 0.85

def test_constrained_selection_preserves_cost_only_legacy_contract() -> None:
    from inventory_ai.evaluation.metrics import choose_candidate

    metrics = {
        "seasonal_naive": {
            "wape": 0.30,
            "coverage_80": 0.80,
            "signed_bias": 0.0,
        },
        "tsb": {
            "wape": 0.28,
            "coverage_80": 0.80,
            "signed_bias": 0.0,
        },
        "reliability_router": {
            "wape": 0.27,
            "coverage_80": 0.80,
            "signed_bias": 0.0,
        },
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
