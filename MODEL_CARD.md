# Model Card

## Core forecasting ladder

- **Seasonal naive:** business baseline and unconditional operational fallback.
- **TSB:** intermittent-demand specialist.
- **Quantile GBM:** CPU-safe nonlinear challenger with q10/q50/q90 outputs.
- **Decision-aware reliability router:** validation-only, series-level selection among the core models. A challenger must satisfy a forecast guardrail before validation inventory cost can determine its route.

## Optional advanced challenger

The Chronos-2 adapter supports dataframe-based zero-shot forecasting, known-future covariates, and a LoRA fine-tuning path. It is optional and not promoted without the same forecast, uncertainty, slice, inventory, and closed-loop gates applied to the core ladder.

## Calibration and scenarios

- q50-preserving quantile repair
- sequential signed conformalized quantile adjustment
- stable correlated scenario paths
- point and sample-level bottom-up hierarchy reconciliation

## Expanded M5 confirmation evidence

The v0.6.2 selection policy was developed on one deterministic 240-series M5 cohort and confirmed once on a second deterministic 240-series cohort with zero series overlap. On the confirmation cohort, the selected reliability-router policy improved held-out WAPE by 16.56% and reduced simulated inventory cost by 18.17% versus seasonal naive, while preserving fill rate and passing all configured reliability gates.

This evidence demonstrates offline benchmark behavior under the repository's simulated inventory assumptions. It does not establish full-M5 performance, retailer-specific effectiveness, production deployment readiness, or Chronos/GPU performance.

## Intended use

Offline demand-forecasting research, controlled stockout recovery, inventory-policy simulation, model-routing experiments, and release-readiness assessment.

## Not intended use

Direct replenishment deployment without real inventory positions, lead-time distributions, supplier constraints, retailer costs, online monitoring, and controlled production experimentation.

## Promotion criteria

A candidate must pass:

- temporal and truth-key contracts
- nonnegativity and quantile ordering
- interval coverage and sharpness
- hierarchy coherence
- recovery and lifecycle evidence
- aggregate and worst-slice forecasting
- inventory cost, SKU win rate, and fill rate
- closed-loop service/cost stability

A more complex model is not automatically preferred.
