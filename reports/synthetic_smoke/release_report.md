# Release Qualification Report

## Decision

**PASS**

Selected source model: **tsb**

Failed or review checks: None

## Forecast evidence

- Candidate WAPE: 0.34432006877926213
- Candidate signed bias: -0.042336200250311026
- Candidate 80% coverage: 0.9226190476190477
- Candidate mean interval width: 27.657557998940018

## Censoring recovery

- Controlled rows: 272
- Raw MAE: 11.89440576425881
- Recovered MAE: 8.497411778801954
- Raw bias: -11.89440576425881
- Recovered bias: -5.746149762650078

## Inventory decision evidence

- Candidate total cost: 6700.393790443491
- Candidate fill rate: 0.9983150189013918
- Candidate stockout rate: 0.011904761904761904
- Cost win rate versus seasonal naive: 0.8333333333333334

## Lifecycle benchmark

- State macro-F1: 0.6196631917966575
- LADT MAE: 0.09758053578293238
- LADT Spearman correlation: 0.87777558587826

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
