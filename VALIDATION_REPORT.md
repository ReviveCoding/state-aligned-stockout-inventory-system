# Validation and Strengthening Report

## Final status

The CPU-safe core pipeline is release-qualified on deterministic synthetic data and an actual M5 small-data track. Optional Chronos, Spark, and Docker runtimes remain environment-dependent and are not represented as executed production evidence.

## Major strengthening loops

### 1. Forecast-key and inventory-comparison correctness

**Found:** recursive forecasts could evaluate models on inconsistent series keys; models could receive different initial inventories; service level 0.80 was reduced to a single stored quantile.

**Fixed:** direct origin-based multi-horizon forecasting, exact truth-key contracts, common initial state, and interpolation across q10/q50/q90.

### 2. Causal data and feature construction

**Found:** DRAT progress used a future maximum, group indices could misalign, M5 missing-price imputation could use future data, and controlled stockout randomization could change when future rows were appended.

**Fixed:** past-only regime duration, transform-aligned features, causal forward/same-period price imputation, and stable row-keyed randomization.

**Validated:** future-append and row-order invariance tests.

### 3. Censoring and lifecycle evidence

**Found:** structural contracts could pass even if controlled recovery or lifecycle recovery degraded.

**Fixed:** lower-bound, do-no-harm, MAE, signed-bias, q95 posterior coverage, state macro-F1, LADT MAE, and rank-correlation gates.

### 4. Calibration and probabilistic paths

**Found:** conformal residuals overlapped with model selection; nonconformity was expansion-only; quantile repair could change q50; scenario sampling did not preserve quantile anchors or hierarchy.

**Fixed:** calibration-only origin, later validation origins, unseen test origin, sequential signed CQR updates, q50-preserving repair, stable per-series scenario RNG, and sample-level bottom-up reconciliation.

### 5. Inventory timing and closed-loop semantics

**Found:** periodic-review policies could order every day, protection windows could include already-realized demand, terminal periods could place unverifiable orders, and repeat-stockout ratio was previously equivalent to stockout rate.

**Fixed:** review-epoch orders, future-only protection windows, terminal decision suppression, and consecutive-stockout event accounting.

### 6. Decision-aware reliability routing

**Found:** global average accuracy could hide intermittent/store regressions; WAPE-only routing did not necessarily minimize inventory cost; cost-only selection could sacrifice SKU stability.

**Fixed:** validation-only series routing that first applies a forecast guardrail, then minimizes validation inventory cost. Final candidate selection considers validation total cost, SKU cost-win rate, worst-slice regression, WAPE, and fill-rate guardrails. Test data never determines routing or policy scale.

### 7. Release gates

The final gate covers:

- truth-key coverage
- nonnegative forecasts and quantile order
- point and scenario hierarchy coherence
- scenario quantile fidelity
- interval coverage and sharpness
- recovery MAE, bias, q95 coverage, lower bound, and do-no-harm
- lifecycle state and LADT recovery
- candidate WAPE
- inventory cost, SKU win rate, and fill rate
- worst-slice regression and slice win rate
- closed-loop cost, lost sales, repeated stockouts, and worst-scenario behavior

### 8. Repository and release engineering

**Fixed:** canonical source/ZIP mismatch, stage-owned output cleanup, run-specific directories, SQL provenance, manifest SHA-256 and byte checks, idempotent reruns, root-relative configs, wheel single-source versioning, isolated wheel import/CLI, package file-set manifest, and separate extracted-archive static/test/runtime verification.

### 9. Rejected optimization

`HistGradientBoostingRegressor` was tested as a possible efficiency improvement. It stalled in the available Python 3.13 environment during quantile fitting, so the repository retained the slower but stable gradient-boosting backend. No optimization was kept unless it preserved correctness and run reliability.

## Final small-data metrics

| Metric | Synthetic | M5 small-data |
|---|---:|---:|
| Release | PASS | PASS |
| Candidate model | TSB | decision-aware reliability router |
| Policy scale | 0.80 | 0.90 |
| WAPE improvement vs seasonal | 3.23% | 5.06% |
| Candidate 80% coverage | 92.26% | 83.04% |
| Interval-width ratio vs seasonal | 0.980 | 1.320 |
| Simulated total-cost change | -4.04% | -9.40% |
| SKU cost-win rate | 83.33% | 83.33% |
| Fill-rate degradation | 0.04 percentage points | 0.26 percentage points |
| Worst-slice WAPE regression | 15.85% | 5.32% |
| Recovery MAE improvement | 28.56% | 40.39% |
| Recovery absolute-bias improvement | 51.69% | 69.36% |
| Recovery q95 upper coverage | 85.66% | 91.39% |
| LADT state macro-F1 | 0.620 | 0.666 |
| LADT MAE | 0.0976 | 0.0867 |
| LADT Spearman | 0.878 | 0.900 |
| Scenario normalized quantile error | 0.0394 | 0.0402 |
| Point/scenario hierarchy error | 0 / 0 | 0 / 0 |
| Closed-loop lost-sales improvement | 23.56% | 17.02% |
| Closed-loop cost change | +3.18% | +4.66% |

The last row is intentionally retained: recovery reduces lost sales but can increase holding cost. The project treats that result as a service/cost trade-off rather than claiming unconditional policy dominance.

## Test inventory

- 71 tests collected.
- Default CPU-safe suite: 70 passed and one optional M5 integration test skipped.
- The optional M5 integration test passes when `RUN_M5_TESTS=1` and the configured archive is available.

## Verification coverage

- full source tests, including contract, leakage, unit, regression, and integration checks
- optional M5 integration test with the supplied archive
- deterministic synthetic and M5 pipelines
- SQL marts and stage-specific provenance
- source compile and config validation
- wheel build, metadata version, isolated install, import, and CLI
- source ZIP manifest, SHA-256, byte-size, extraction, tests, runtime, SQL, and release-gate verification

## External-runtime limits

- Dockerfile exists, but no Docker daemon was available for an actual image build.
- Chronos-2 adapter and LoRA path exist, but model weights/GPU training were not executed here.
- PySpark implementation exists, but a Spark runtime was not installed here.
- M5 evaluation is a deliberately small 24-series offline run, not the full competition hierarchy.

## 0.6.2 expanded M5 development and confirmation

A 240-series M5 development cohort exposed that cost-first candidate selection could choose a policy that later violated held-out fill-rate and interval-sharpness release gates. The selection logic was revised so that validation-only candidate selection first requires WAPE, worst-slice, service-level, and interval-width feasibility, then minimizes cost only within the feasible set.

The revised logic was tested with targeted tests and the full pytest suite. It was then evaluated once on a deterministic 240-series M5 confirmation cohort using `series_offset: 240`; the cohort disjointness check reported zero overlap with development.

| Confirmation metric | Result |
|---|---:|
| Data source | Actual M5 ZIP, no fallback |
| Rows / series | 53,760 / 240 |
| Selected model | reliability_router |
| Held-out WAPE improvement vs seasonal naive | 16.56% |
| Simulated inventory-cost reduction | 18.17% |
| Series-level cost win rate | 88.33% |
| Fill-rate degradation | -0.103 percentage points |
| Interval-width ratio | 1.078 |
| Worst-slice WAPE regression | -7.05% |
| Forecast-slice win rate | 100% |
| Release gate | PASS |

This is a series-disjoint offline confirmation within the same M5 archive and time window. It is not full-M5, external retail, production, Chronos, or GPU evidence.

## 0.6.1 final hardening loop

The final loop re-opened the source ZIP from a clean extraction and added controls not covered by predictive gates:

- malicious or malformed archive members are rejected before extraction,
- package checksum sidecars and runtime sample byte sizes are verified,
- Docker package installation includes the PEP 639 license file,
- the declared setuptools build backend is new enough to support the selected license metadata,
- failed pipeline reruns restore previous outputs,
- large-panel audit samples retain both earliest and latest periods,
- redundant validation simulation was removed,
- model-isolated simulation was proven equivalent to combined simulation on a controlled fixture,
- optional stage logging localizes long-running operations without changing normal output.

The 0.6.1 patch additionally:

- removes `.coverage`, environment, editor, and notebook-checkpoint residue from release ZIPs,
- rejects duplicate, encrypted, and explicit special-file ZIP entries,
- distinguishes absent Chronos future covariates from dangerous partial future panels,
- requires complete Chronos series-by-horizon prediction coverage,
- accepts both numeric and string Chronos quantile-column encodings,
- validates pandas/Spark parity before writing output,
- publishes Spark Parquet through rollback-capable staging.

These changes improve operational resilience without changing the core forecast or decision algorithms.
