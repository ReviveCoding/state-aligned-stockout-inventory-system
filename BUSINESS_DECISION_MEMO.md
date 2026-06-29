# Business Decision Memo

## Problem

Observed sales can understate latent demand during stockouts. Training directly on censored sales can reinforce under-ordering: low inventory suppresses observed sales, the next forecast falls, and another stockout becomes more likely.

## Proposed framework

Use causal demand-state features, controlled stockout-recovery uncertainty, calibrated probabilistic forecasts, hierarchy-consistent scenarios, and a periodic-review inventory simulator. Select models and policy scales only on validation evidence, then evaluate the frozen decision on an unseen test origin and in closed-loop scenarios.

## Current offline decision

- Synthetic smoke: TSB at policy scale 0.80 passes all release checks.
- M5 small-data: a validation-only decision-aware router at policy scale 0.90 passes all release checks.
- Recovery reduces closed-loop lost sales in both tracks, but also increases total closed-loop cost in the current scenarios. This is treated as a service/cost trade-off, not an unconditional launch win.

## Launch logic for a real pilot

Proceed only after replacing simulator assumptions with retailer inventory, cost, lead-time, and supplier data, and after confirming:

- calibrated service-level uncertainty
- no critical worst-slice regressions
- cost and fill-rate improvement under relevant capacity constraints
- stable repeated-stockout behavior
- observable fallback and rollback paths
- controlled online or shadow-mode evaluation

## Rollback triggers

- sustained interval undercoverage or excessive width
- rising fallback or missing-feature rate
- repeated-stockout amplification
- material fill-rate degradation
- unexpected working-capital or order-volatility growth
- mismatch between simulated and realized cost/service behavior
