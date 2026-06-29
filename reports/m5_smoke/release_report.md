# Release Qualification Report

## Decision

**PASS**

Selected source model: **reliability_router**

Failed or review checks: None

## Forecast evidence

- Candidate WAPE: 0.28550077137722396
- Candidate signed bias: -0.12066441021426573
- Candidate 80% coverage: 0.8303571428571429
- Candidate mean interval width: 16.15050204309757

## Censoring recovery

- Controlled rows: 488
- Raw MAE: 11.661417397326145
- Recovered MAE: 6.9513729485229
- Raw bias: -11.661417397326145
- Recovered bias: -3.573193827863568

## Inventory decision evidence

- Candidate total cost: 8189.124616527259
- Candidate fill rate: 0.9801441778176612
- Candidate stockout rate: 0.041666666666666664
- Cost win rate versus seasonal naive: 0.8333333333333334

## Lifecycle benchmark

- State macro-F1: 0.6658122214133935
- LADT MAE: 0.08665482612221924
- LADT Spearman correlation: 0.9004547714837464

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
