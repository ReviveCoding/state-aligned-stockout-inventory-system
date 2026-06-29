# Assumption Registry

| Assumption | Current default | Risk | Mitigation |
|---|---|---|---|
| Controlled stockouts approximate censoring | demand-dependent, row-keyed synthetic mask | censoring mechanism may differ from a retailer | report separately and claim recovery only where latent demand is retained |
| Lead time | fixed 2 days in smoke config | understates lead-time variability | config-driven sensitivity and external-data extension |
| Review policy | periodic review with complete future protection window | finite horizon can bias terminal orders | suppress decisions without a fully observable protection window |
| Service target | interpolated from q10/q50/q90 at configured service level | sparse quantiles approximate the full distribution | scenario diagnostics and policy-scale validation |
| Holding/shortage/order costs | synthetic configured values | business conclusions depend on ratios | assumption registry and scale/sensitivity evaluation |
| M5 stockouts | no true labels | cannot validate natural lost demand | controlled censoring only; external stockout data required for real-label claims |
| M5 lifecycle | no true labels | state names could be overinterpreted | call real-data output DRAT; validate LADT only on controlled lifecycle data |
| Hierarchy | bottom-up coherence | does not represent all enterprise reconciliation methods | report point/sample coherence and keep MinT as an external extension |
| Closed-loop policy | offline digital twin | does not reproduce retailer operations | frame cost/service results as simulator evidence |
| Chronos/Spark | optional adapters | environment and model downloads may fail | capability checks and CPU/pandas fallback |
