# Data Card

## Synthetic track

`configs/smoke.yaml` always creates a deterministic retail panel with known latent demand, promotions, demand regimes, controlled stockouts, and a separate controlled lifecycle benchmark. It is used for CI, contracts, and recovery ground truth.

## M5 track

`configs/m5_smoke.yaml` requires the configured M5 ZIP. The current smoke samples 24 diverse item-store series and 140 days to validate:

- daily demand forecasting
- store/item/category hierarchy
- calendar and event features
- causal price preparation
- rolling temporal validation
- controlled censoring layered over observed M5 sales

M5 does not provide true inventory, supplier lead time, lost demand, stockout labels, or product lifecycle labels. The project makes no such claims.

## Auto-local track

`configs/auto_local.yaml` may use M5 when present and may fall back to synthetic only if the archive is missing and fallback is explicitly enabled. Corrupt archives and schema failures are never hidden.

## FreshRetail extension boundary

FreshRetailNet-50K remains a potential external stockout track, but it is not a core dependency and no result in this repository is labeled as FreshRetail evidence.

## Leakage controls

- no future backward-fill for M5 prices
- past-only rolling and expanding features
- row-keyed controlled-censoring RNG
- disjoint calibration/validation/test origins
- reference and routing decisions learned before the test origin
