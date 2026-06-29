# Science Design

## Research questions

1. Does demand-state-aligned time improve forecasting or routing beyond calendar age?
2. Does probabilistic recovery reduce the underforecasting caused by controlled stockout censoring?
3. Does the forecast-metric winner also minimize replenishment cost while preserving fill rate?
4. Can validation-only routing reduce worst-slice regressions without test leakage?
5. Does a policy remain reliable after its decisions alter future observed sales?

## Claim matrix

| Track | Data | Valid claim | Invalid claim |
|---|---|---|---|
| M5 | daily sales, prices, events, hierarchy | offline retail forecasting and hierarchy mechanics | true stockout, inventory, or lifecycle recovery |
| Controlled censoring | preserved latent demand plus capped observations | recovery MAE, bias, lower-bound, and posterior coverage | natural lost demand |
| Controlled lifecycle | variable-duration known states | LADT state/progress recovery | real product lifecycle labels |
| Inventory simulator | generated state, orders, costs | forecast-to-decision sensitivity | real retailer savings |
| Closed-loop replay | synthetic repeated planning | feedback and repeated-stockout behavior | online production experiment |

## Temporal experiment design

```text
historical training
-> calibration-only origin
-> sequential validation origins for model/router/policy selection
-> unseen final test origin
```

No test metric is used for model routing, policy scale, conformal fitting, or release candidate selection.

## Core methods

- seasonal naive, TSB, quantile GBM
- DRAT and controlled LADT
- controlled stockout posterior recovery
- sequential conformalized quantile calibration
- decision-aware series router
- correlated quantile-faithful scenarios
- point and sample-level bottom-up reconciliation
- periodic-review order-up-to simulation
- scenario-sliced closed-loop replay

## Core metrics

- Forecast: WAPE, signed bias, pinball loss, coverage, interval width.
- Reliability slices: category, store, promotion, stockout, and demand regime.
- Recovery: MAE, bias, q95 coverage, lower-bound violations, do-no-harm.
- Lifecycle: macro-F1, LADT MAE, LADT rank correlation.
- Hierarchy/scenarios: coherence, quantile fidelity, temporal correlation.
- Decision: total cost, SKU cost-win rate, fill rate, lost sales, stockout rate, order volatility.
- Closed loop: consecutive-stockout event ratio, cumulative cost, lost sales, worst-scenario cost and fill degradation.
