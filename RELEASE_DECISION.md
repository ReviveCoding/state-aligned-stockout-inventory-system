# Release Decision

Current verified decisions:

| Run | Decision | Selected model | Policy scale |
|---|---|---|---:|
| `synthetic_smoke` | PASS | TSB | 0.80 |
| `m5_smoke` | PASS | decision-aware reliability router | 0.90 |

The machine-readable decisions are generated at:

```text
reports/synthetic_smoke/release_gate.json
reports/m5_smoke/release_gate.json
```

## v0.6.2 M5 confirmation decision

| Track | Cohort | Gate | Selected policy | WAPE improvement | Cost reduction | Fill-rate change |
|---|---:|---|---|---:|---:|---:|
| `m5_expanded_v1` | 240 development series | ITERATE | reliability_router | 13.23% | 17.70% | -1.14pp |
| `m5_expanded_v2_confirmation` | 240 series, zero overlap with v1 | PASS | reliability_router | 16.56% | 18.17% | +0.103pp |

The v1 development track was used only to identify the service-level and interval-sharpness selection defect. The v2 confirmation result was produced after the constrained-selection patch and is the release-facing M5 evidence. This remains an offline M5 benchmark result, not production replenishment authorization.

A candidate is promoted only when predictive, probabilistic, recovery, lifecycle, hierarchy, worst-slice, inventory, fill-rate, closed-loop, and engineering checks pass. A PASS is specific to the configured offline experiment and does not authorize real replenishment deployment.
