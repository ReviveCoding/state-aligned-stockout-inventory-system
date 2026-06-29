# Release Qualification Report

## Decision

**ITERATE**

Selected source model: **reliability_router**

Failed or review checks: inventory_fill_rate, candidate_interval_sharpness

## Forecast evidence

- Candidate WAPE: 0.34909200057126616
- Candidate signed bias: -0.09696859056272739
- Candidate 80% coverage: 0.8416666666666667
- Candidate mean interval width: 26.011296664360465

## Censoring recovery

- Controlled rows: 8938
- Raw MAE: 9.64883267519301
- Recovered MAE: 7.027279616307494
- Raw bias: -9.64883267519301
- Recovered bias: -2.9612900312650683

## Inventory decision evidence

- Candidate total cost: 108037.23232749608
- Candidate fill rate: 0.9438921280812983
- Candidate stockout rate: 0.06398809523809523
- Cost win rate versus seasonal naive: 0.9083333333333333

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
  "inventory_fill_rate": false,
  "candidate_interval_sharpness": false,
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
