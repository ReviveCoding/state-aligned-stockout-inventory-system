# Release Qualification Report

## Decision

**PASS**

Selected source model: **reliability_router**

Failed or review checks: None

## Forecast evidence

- Candidate WAPE: 0.37977140107095864
- Candidate signed bias: -0.12094583605332117
- Candidate 80% coverage: 0.8035714285714286
- Candidate mean interval width: 20.62578163592314

## Censoring recovery

- Controlled rows: 10414
- Raw MAE: 6.382439905419998
- Recovered MAE: 5.252181834925401
- Raw bias: -6.382439905419998
- Recovered bias: -2.019630824585396

## Inventory decision evidence

- Candidate total cost: 86142.86576086242
- Candidate fill rate: 0.9461257466137805
- Candidate stockout rate: 0.08452380952380953
- Cost win rate versus seasonal naive: 0.8833333333333333

## Lifecycle benchmark

- State macro-F1: 0.6854272284648976
- LADT MAE: 0.08116567009532384
- LADT Spearman correlation: 0.9334469662487592

## Reliability checks

```json
{
  "truth_key_coverage": true,
  "negative_forecasts": true,
  "quantile_crossing": true,
  "hierarchy_coherence": true,
  "scenario_hierarchy_coherence": true,
  "scenario_quantile_fidelity": true,
  "candidate_interval_coverage": true,
  "recovery_mae_improvement": true,
  "recovery_bias_improvement": true,
  "recovery_q95_coverage": true,
  "lifecycle_state_recovery": true,
  "lifecycle_ladt_mae": true,
  "lifecycle_ladt_rank": true,
  "candidate_wape": true,
  "inventory_cost_win_rate": true,
  "inventory_cost_regression": true,
  "inventory_fill_rate": true,
  "candidate_interval_sharpness": true,
  "forecast_worst_slice": true,
  "forecast_slice_win_rate": true,
  "closed_loop_stockout": true,
  "closed_loop_stockout_improvement": true,
  "closed_loop_cost": true,
  "closed_loop_lost_sales": true,
  "closed_loop_worst_slice_cost": true,
  "closed_loop_worst_slice_fill": true,
  "recovery_lower_bound": true,
  "recovery_do_no_harm": true
}
```

## Claim boundaries

- M5 is used for hierarchical retail forecasting mechanics, price/calendar covariates, and controlled censoring experiments. It does not provide true inventory or lifecycle labels.
- Controlled-stockout recovery accuracy is measured only where latent demand is preserved by construction.
- LADT recovery is evaluated only on the controlled variable-duration lifecycle benchmark.
- Inventory savings are offline simulation results under the documented cost and lead-time assumptions.
