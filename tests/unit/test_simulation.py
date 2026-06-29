from __future__ import annotations

import pandas as pd

from inventory_ai.data.synthetic import SyntheticConfig, make_synthetic_retail
from inventory_ai.models.baselines import seasonal_naive_forecast, tsb_forecast
from inventory_ai.reconciliation.hierarchy import hierarchy_error, reconcile_bottom_up
from inventory_ai.simulation.closed_loop import closed_loop_policy_comparison, closed_loop_safety_sweep
from inventory_ai.simulation.inventory import InventoryConfig, run_open_loop_policy
from inventory_ai.simulation.scenarios import generate_correlated_scenarios, scenario_diagnostics


def test_hierarchy_is_coherent_for_all_quantiles():
    panel = make_synthetic_retail(SyntheticConfig(6, 70, 21))
    forecasts = pd.concat([seasonal_naive_forecast(panel, 62, 7), tsb_forecast(panel, 62, 7)])
    reconciled = reconcile_bottom_up(forecasts, panel)
    assert hierarchy_error(reconciled) <= 1e-10


def test_inventory_uses_identical_truth_and_delayed_arrivals():
    panel = make_synthetic_retail(SyntheticConfig(5, 70, 22))
    truth = panel[panel["date_idx"].between(63, 69)]
    forecasts = pd.concat([seasonal_naive_forecast(panel, 62, 7), tsb_forecast(panel, 62, 7)])
    cfg = InventoryConfig(lead_time_days=2)
    simulation = run_open_loop_policy(forecasts, truth, cfg)
    demand = simulation.groupby("model")["demand"].sum()
    assert demand.max() - demand.min() < 1e-9
    assert (simulation["ending_inventory"] >= 0).all()
    assert (simulation["order_qty"] >= 0).all()
    first_days = simulation.groupby(["model", "series_id"]).head(1)
    assert (first_days["arrivals"] == 0).all()


def test_correlated_scenarios_are_nonnegative_and_temporal():
    panel = make_synthetic_retail(SyntheticConfig(5, 70, 23))
    forecast = seasonal_naive_forecast(panel, 62, 7)
    scenarios = generate_correlated_scenarios(forecast, n_scenarios=30, seed=4)
    diagnostics = scenario_diagnostics(scenarios)
    assert diagnostics["negative_values"] == 0
    assert diagnostics["n_scenarios"] == 30
    assert diagnostics["mean_lag1_correlation"] > -0.2


def test_closed_loop_compares_same_latent_paths_and_sweeps():
    cfg = InventoryConfig(lead_time_days=2)
    frame = closed_loop_policy_comparison(5, 10, cfg, seed=24)
    pivot = frame.pivot_table(index=["series_id", "cycle"], columns="policy", values="latent_demand")
    assert (pivot.iloc[:, 0] == pivot.iloc[:, 1]).all()
    sweep = closed_loop_safety_sweep(5, 10, cfg, seed=24)
    assert sweep["recovery_safety"].nunique() >= 4


def test_inventory_models_share_identical_initial_state():
    panel = make_synthetic_retail(SyntheticConfig(5, 70, 31))
    truth = panel[panel["date_idx"].between(63, 69)]
    forecasts = pd.concat([seasonal_naive_forecast(panel, 62, 7), tsb_forecast(panel, 62, 7)])
    simulation = run_open_loop_policy(forecasts, truth, InventoryConfig(service_level=0.8))
    initial = simulation.groupby(["model", "series_id"])["initial_inventory"].first().unstack("model")
    assert (initial.max(axis=1) - initial.min(axis=1) < 1e-9).all()


def test_service_level_changes_ordering_target_continuously():
    panel = make_synthetic_retail(SyntheticConfig(4, 70, 32))
    truth = panel[panel["date_idx"].between(63, 69)]
    forecast = seasonal_naive_forecast(panel, 62, 7)
    low = run_open_loop_policy(forecast, truth, InventoryConfig(service_level=0.5, initial_inventory_multiplier=0.0))
    high = run_open_loop_policy(forecast, truth, InventoryConfig(service_level=0.8, initial_inventory_multiplier=0.0))
    assert high["order_qty"].sum() > low["order_qty"].sum()


def test_closed_loop_contains_operational_scenarios_and_adaptive_gate_metrics():
    from inventory_ai.simulation.closed_loop import summarize_closed_loop, summarize_closed_loop_slices

    cfg = InventoryConfig(lead_time_days=2, initial_inventory_multiplier=1.0)
    frame = closed_loop_policy_comparison(12, 16, cfg, seed=44, recovery_safety=1.05)
    assert {"stable", "transient_promotion", "persistent_shift", "intermittent"}.issubset(frame["scenario_type"].unique())
    pivot = frame.pivot_table(index=["series_id", "cycle"], columns="policy", values="latent_demand")
    assert (pivot.iloc[:, 0] == pivot.iloc[:, 1]).all()
    summary = summarize_closed_loop(frame)
    assert summary["recovery_state_aware"]["lost_sales"] <= summary["naive_no_recovery"]["lost_sales"] * 1.02
    slices = summarize_closed_loop_slices(frame)
    assert not slices.empty
    assert slices.groupby("policy")["scenario_type"].nunique().min() == 4


def test_scenario_marginals_and_hierarchy_are_reliable():
    from inventory_ai.reconciliation.hierarchy import reconcile_scenarios_bottom_up, scenario_hierarchy_error

    panel = make_synthetic_retail(SyntheticConfig(8, 70, 52))
    forecast = seasonal_naive_forecast(panel, 62, 7)
    scenarios = generate_correlated_scenarios(forecast, n_scenarios=400, seed=52)
    diagnostics = scenario_diagnostics(scenarios)
    assert diagnostics["normalized_quantile_mae"] < 0.20
    reconciled = reconcile_scenarios_bottom_up(scenarios, panel)
    assert scenario_hierarchy_error(reconciled) <= 1e-9


def test_periodic_review_places_orders_only_on_review_epochs():
    panel = make_synthetic_retail(SyntheticConfig(4, 75, 61))
    truth = panel[panel["date_idx"].between(68, 74)]
    forecast = seasonal_naive_forecast(panel, 67, 7)
    simulation = run_open_loop_policy(
        forecast,
        truth,
        InventoryConfig(lead_time_days=1, review_period_days=3, initial_inventory_multiplier=0.0),
    )
    assert (simulation.loc[simulation["is_review_period"].eq(0), "order_qty"] == 0).all()
    # Only the first review epoch has a complete lead+review protection
    # window inside this seven-day evaluation horizon.
    assert simulation.groupby(["model", "series_id"])["is_review_period"].sum().eq(1).all()
    assert (simulation.loc[simulation["decision_eligible"].eq(0), "order_qty"] == 0).all()


def test_replenishment_target_excludes_already_realized_current_period():
    series = "s"
    forecast = pd.DataFrame({
        "series_id": [series] * 4,
        "date_idx": [1, 2, 3, 4],
        "horizon": [1, 2, 3, 4],
        "q10": [10.0, 1.0, 1.0, 1.0],
        "q50": [10.0, 1.0, 1.0, 1.0],
        "q90": [10.0, 1.0, 1.0, 1.0],
        "model": ["m"] * 4,
    })
    truth = pd.DataFrame({"series_id": [series] * 4, "date_idx": [1, 2, 3, 4], "demand": [0.0] * 4})
    simulation = run_open_loop_policy(
        forecast,
        truth,
        InventoryConfig(lead_time_days=1, review_period_days=1, service_level=0.5, initial_inventory_multiplier=0.0),
    )
    first = simulation.iloc[0]
    # The period-1 forecast of 10 is already realized before the decision; only
    # periods 2 and 3 belong to the lead+review protection interval.
    assert first["target_inventory_position"] == 2.0


def test_scenario_generator_rejects_nonpositive_count():
    import pandas as pd
    import pytest
    from inventory_ai.simulation.scenarios import generate_correlated_scenarios

    forecast = pd.DataFrame({
        "model": ["m"], "series_id": ["s"], "date_idx": [1], "horizon": [1],
        "q10": [1.0], "q50": [2.0], "q90": [3.0],
    })
    with pytest.raises(ValueError, match="n_scenarios"):
        generate_correlated_scenarios(forecast, n_scenarios=0)



def test_terminal_orders_are_suppressed_without_full_protection_window():
    series = "s"
    forecast = pd.DataFrame({
        "series_id": [series] * 5,
        "date_idx": [1, 2, 3, 4, 5],
        "horizon": [1, 2, 3, 4, 5],
        "q10": [2.0] * 5,
        "q50": [3.0] * 5,
        "q90": [4.0] * 5,
        "model": ["m"] * 5,
    })
    truth = pd.DataFrame({"series_id": [series] * 5, "date_idx": [1, 2, 3, 4, 5], "demand": [1.0] * 5})
    simulation = run_open_loop_policy(
        forecast,
        truth,
        InventoryConfig(lead_time_days=2, review_period_days=1, initial_inventory_multiplier=0.0),
    )
    # A complete future protection window needs three periods. Only decisions
    # after the first two realized periods can still affect the evaluation.
    assert simulation["decision_eligible"].tolist() == [1, 1, 0, 0, 0]
    assert (simulation.loc[simulation["decision_eligible"].eq(0), "order_qty"] == 0).all()


def test_repeat_stockout_ratio_counts_consecutive_events_not_stockout_rate():
    from inventory_ai.simulation.closed_loop import summarize_closed_loop

    frame = pd.DataFrame({
        "policy": ["p"] * 6,
        "series_id": ["a"] * 4 + ["b"] * 2,
        "cycle": [0, 1, 2, 3, 0, 1],
        "stockout": [1, 1, 0, 1, 0, 1],
        "cost": [1.0] * 6,
        "lost_sales": [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "forecast": [1.0] * 6,
        "latent_demand": [2.0] * 6,
        "ending_inventory": [0.0] * 6,
    })
    summary = summarize_closed_loop(frame)["p"]
    assert summary["stockout_rate"] == 4 / 6
    assert summary["repeat_stockout_events"] == 1
    assert summary["repeat_stockout_ratio"] == 1 / 4  # four adjacent-cycle opportunities
    assert summary["repeat_stockout_share"] == 1 / 4  # four stockout events in this fixture


def test_scenarios_are_invariant_to_forecast_group_order():
    panel = make_synthetic_retail(SyntheticConfig(6, 70, 81))
    forecast = pd.concat([
        seasonal_naive_forecast(panel, 62, 7),
        tsb_forecast(panel, 62, 7),
    ], ignore_index=True)
    first = generate_correlated_scenarios(forecast, n_scenarios=20, seed=81)
    second = generate_correlated_scenarios(
        forecast.sample(frac=1.0, random_state=3), n_scenarios=20, seed=81
    )
    cols = ["model", "series_id", "scenario", "date_idx", "demand"]
    first = first[cols].sort_values(cols[:-1]).reset_index(drop=True)
    second = second[cols].sort_values(cols[:-1]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_policy_scale_summary_reuses_tuning_results():
    from inventory_ai.simulation.inventory import summarize_policy_at_scale

    tuning = pd.DataFrame({
        "model": ["a", "a", "b", "b"],
        "policy_scale": [0.9, 1.0, 0.9, 1.0],
        "total_cost": [9.0, 10.0, 19.0, 20.0],
        "lost_sales": [1.0, 2.0, 3.0, 4.0],
        "fill_rate": [0.9, 0.8, 0.95, 0.85],
        "stockout_rate": [0.1, 0.2, 0.05, 0.15],
        "average_inventory": [2.0, 3.0, 4.0, 5.0],
        "order_volatility": [0.5, 0.6, 0.7, 0.8],
    })
    summary = summarize_policy_at_scale(tuning, 1.0)
    assert summary["a"]["total_cost"] == 10.0
    assert summary["b"]["total_cost"] == 20.0
    assert summary["a"]["policy_scale"] == 1.0


def test_model_isolated_inventory_matches_combined_simulation():
    from inventory_ai.simulation.inventory import common_initial_inventory, run_open_loop_by_model

    panel = make_synthetic_retail(SyntheticConfig(4, 70, 109))
    origin = 62
    forecast = pd.concat([
        seasonal_naive_forecast(panel, origin, 7),
        tsb_forecast(panel, origin, 7),
    ], ignore_index=True)
    truth = panel[panel["date_idx"].between(origin + 1, origin + 7)]
    cfg = InventoryConfig(lead_time_days=2, review_period_days=1)
    scales = {"seasonal_naive": 0.95, "tsb": 0.85}
    initial = common_initial_inventory(forecast, cfg)
    combined = run_open_loop_policy(
        forecast, truth, cfg, scales, initial_inventory_by_series=initial
    ).sort_values(["model", "series_id", "date_idx"]).reset_index(drop=True)
    isolated = run_open_loop_by_model(forecast, truth, cfg, scales)
    columns = [
        "model", "series_id", "date_idx", "order_qty", "ending_inventory",
        "lost_sales", "cost", "policy_scale",
    ]
    pd.testing.assert_frame_equal(combined[columns], isolated[columns])


def test_spark_parity_validation_rejects_mismatch_before_publish():
    import pandas as pd
    import pytest
    from inventory_ai.data.spark_features import SPARK_PARITY_COLUMNS, validate_feature_parity

    base = {"series_id": ["s"], "date_idx": [1]}
    for column in SPARK_PARITY_COLUMNS:
        base[column] = [0.0]
    pandas_values = pd.DataFrame(base)
    spark_values = pandas_values.copy()
    spark_values.loc[0, SPARK_PARITY_COLUMNS[0]] = 0.1
    with pytest.raises(ValueError, match="exceeded tolerance"):
        validate_feature_parity(pandas_values, spark_values, tolerance=1e-8)


def test_atomic_directory_publish_replaces_previous_and_rolls_back(tmp_path):
    import pytest
    from inventory_ai.data.spark_features import publish_directory_atomically

    output = tmp_path / "features.parquet"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    publish_directory_atomically(staging, output)
    assert not staging.exists()
    assert (output / "new.txt").read_text(encoding="utf-8") == "new"
    assert not (output / "old.txt").exists()

    missing_staging = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        publish_directory_atomically(missing_staging, output)
    assert (output / "new.txt").read_text(encoding="utf-8") == "new"
