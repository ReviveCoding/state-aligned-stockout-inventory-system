from __future__ import annotations

import pandas as pd

from inventory_ai.censoring.recovery import add_controlled_stockouts, recover_latent_demand, recovery_diagnostics
from inventory_ai.data.synthetic import SyntheticConfig, make_lifecycle_benchmark, make_synthetic_retail
from inventory_ai.evaluation.metrics import lifecycle_metrics
from inventory_ai.features.basic import add_time_features
from inventory_ai.lifecycle.drat import PROBABILITY_COLUMNS, add_drat, estimate_ladt


def _prepared(seed: int = 3):
    frame = make_synthetic_retail(SyntheticConfig(6, 84, seed))
    frame = add_controlled_stockouts(frame, seed=seed + 1, rate=0.20)
    return recover_latent_demand(frame)


def test_recovery_respects_lower_bound_and_do_no_harm():
    frame = _prepared()
    assert (frame["recovered_demand_mean"] + 1e-9 >= frame["sales"]).all()
    nonstockout = frame["stockout"].eq(0)
    assert (frame.loc[nonstockout, "recovered_demand_mean"] == frame.loc[nonstockout, "sales"]).all()


def test_recovery_improves_controlled_mae():
    diagnostics = recovery_diagnostics(_prepared(7))
    assert diagnostics["n_controlled"] > 0
    assert diagnostics["recovered_mae"] < diagnostics["raw_mae"]
    assert abs(diagnostics["recovered_bias"]) < abs(diagnostics["raw_bias"])


def test_drat_probabilities_sum_to_one():
    frame = add_time_features(_prepared(), value_col="recovered_demand_mean")
    output = add_drat(frame)
    total = output[PROBABILITY_COLUMNS].sum(axis=1)
    assert total.between(0.999999, 1.000001).all()
    assert output["drat_entropy"].between(0, 1).all()
    assert output["drat_progress"].between(0, 1).all()


def test_drat_is_invariant_to_future_rows():
    frame = add_time_features(_prepared(9), value_col="recovered_demand_mean")
    prefix = frame[frame["date_idx"] <= 55].copy()
    full_output = add_drat(frame)
    prefix_output = add_drat(prefix)
    cols = PROBABILITY_COLUMNS + ["drat_entropy", "drat_velocity", "drat_progress"]
    joined = prefix_output[["series_id", "date_idx"] + cols].merge(
        full_output[["series_id", "date_idx"] + cols],
        on=["series_id", "date_idx"],
        suffixes=("_prefix", "_full"),
    )
    for col in cols:
        pd.testing.assert_series_equal(joined[f"{col}_prefix"], joined[f"{col}_full"], check_names=False)


def test_lifecycle_benchmark_is_nontrivial_and_recoverable():
    frame = make_lifecycle_benchmark(SyntheticConfig(12, 112, 11))
    frame = recover_latent_demand(frame)
    frame = add_time_features(frame, value_col="recovered_demand_mean")
    estimated = estimate_ladt(frame)
    metrics = lifecycle_metrics(estimated)
    assert metrics["state_macro_f1"] >= 0.50
    assert metrics["ladt_mae"] <= 0.16
    assert metrics["ladt_spearman"] >= 0.75


def test_recovery_posterior_samples_respect_censoring_lower_bound():
    from inventory_ai.censoring.recovery import sample_recovery_posterior

    frame = _prepared(17)
    samples = sample_recovery_posterior(frame, n_draws=12, seed=17)
    lower = frame.set_index(["series_id", "date_idx"])["sales"]
    joined = samples.join(lower, on=["series_id", "date_idx"])
    assert not samples.empty
    assert (joined["latent_demand_draw"] + 1e-9 >= joined["sales"]).all()
    assert samples.groupby(["series_id", "date_idx"])["draw"].nunique().eq(12).all()


def test_recovery_diagnostics_include_material_improvement_and_coverage():
    diagnostics = recovery_diagnostics(_prepared(19))
    assert diagnostics["mae_improvement"] > 0
    assert diagnostics["absolute_bias_improvement"] > 0
    assert 0 <= diagnostics["q80_upper_coverage"] <= 1
    assert 0 <= diagnostics["q95_upper_coverage"] <= 1


def test_controlled_stockout_injection_is_prefix_and_order_invariant():
    frame = make_synthetic_retail(SyntheticConfig(6, 84, 29))
    prefix = frame[frame["date_idx"] <= 55].copy()
    full = add_controlled_stockouts(frame, seed=91, rate=0.17)
    prefix_only = add_controlled_stockouts(prefix, seed=91, rate=0.17)
    shuffled = add_controlled_stockouts(prefix.sample(frac=1.0, random_state=7), seed=91, rate=0.17)
    cols = ["series_id", "date_idx", "sales", "controlled_stockout"]
    expected = prefix_only[cols].sort_values(["series_id", "date_idx"]).reset_index(drop=True)
    from_full = full[full["date_idx"] <= 55][cols].sort_values(["series_id", "date_idx"]).reset_index(drop=True)
    from_shuffled = shuffled[cols].sort_values(["series_id", "date_idx"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, from_full)
    pd.testing.assert_frame_equal(expected, from_shuffled)


def test_recovery_posterior_is_invariant_to_input_order():
    from inventory_ai.censoring.recovery import sample_recovery_posterior

    frame = _prepared(37)
    first = sample_recovery_posterior(frame, n_draws=8, seed=37)
    second = sample_recovery_posterior(
        frame.sample(frac=1.0, random_state=4), n_draws=8, seed=37
    )
    cols = ["series_id", "date_idx", "draw", "latent_demand_draw"]
    first = first[cols].sort_values(cols[:-1]).reset_index(drop=True)
    second = second[cols].sort_values(cols[:-1]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
