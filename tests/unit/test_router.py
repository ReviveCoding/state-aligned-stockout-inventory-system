from __future__ import annotations

import pandas as pd

from inventory_ai.models.router import apply_series_router, learn_series_router


def _forecast(model: str, series: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "series_id": [series] * len(values),
        "date_idx": list(range(1, len(values) + 1)),
        "horizon": list(range(1, len(values) + 1)),
        "q10": [max(0.0, value - 1.0) for value in values],
        "q50": values,
        "q90": [value + 1.0 for value in values],
        "model": [model] * len(values),
    })


def test_router_uses_material_validation_improvement_and_preserves_keys():
    truth = pd.DataFrame({
        "series_id": ["a"] * 14 + ["b"] * 14,
        "date_idx": list(range(1, 15)) * 2,
        "demand": [10.0] * 28,
    })
    forecasts = pd.concat([
        _forecast("seasonal_naive", "a", [14.0] * 14),
        _forecast("quantile_gbm", "a", [10.0] * 14),
        _forecast("seasonal_naive", "b", [10.0] * 14),
        _forecast("quantile_gbm", "b", [10.2] * 14),
    ], ignore_index=True)
    routing = learn_series_router(forecasts, truth, min_relative_improvement=0.05)
    decisions = routing.set_index("series_id")["selected_model"].to_dict()
    assert decisions == {"a": "quantile_gbm", "b": "seasonal_naive"}
    routed = apply_series_router(forecasts, routing)
    assert routed.groupby("series_id")["routed_from_model"].first().to_dict() == decisions
    assert len(routed) == len(truth)


def test_decision_aware_router_respects_wape_guardrail_and_uses_cost():
    import pandas as pd
    from inventory_ai.models.router import learn_decision_aware_router
    from inventory_ai.simulation.inventory import InventoryConfig

    truth = pd.DataFrame({
        "series_id": ["s"] * 4,
        "date_idx": [1, 2, 3, 4],
        "demand": [10.0, 10.0, 10.0, 10.0],
    })
    rows = []
    for model, q50, width in [
        ("seasonal_naive", 10.0, 4.0),
        ("cheap_model", 10.5, 2.0),
        ("bad_model", 15.0, 1.0),
    ]:
        for date_idx in truth["date_idx"]:
            rows.append({
                "series_id": "s", "date_idx": date_idx, "horizon": date_idx,
                "q10": max(0.0, q50 - width), "q50": q50, "q90": q50 + width,
                "model": model,
            })
    forecasts = pd.DataFrame(rows)
    routing = learn_decision_aware_router(
        forecasts,
        truth,
        InventoryConfig(lead_time_days=0, review_period_days=1, initial_inventory_multiplier=1.0),
        {"seasonal_naive": 1.0, "cheap_model": 0.75, "bad_model": 0.5},
        max_relative_wape_regression=0.10,
        min_validation_rows=4,
    )
    assert routing.loc[0, "selected_model"] in {"seasonal_naive", "cheap_model"}
    assert routing.loc[0, "selected_model"] != "bad_model"
    assert routing.loc[0, "routing_reason"] == "lowest_validation_cost_within_wape_guardrail"
